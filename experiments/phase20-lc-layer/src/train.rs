// Phase 20: LC Circuit Layer
//
// Tests whether frequency-native computation can replace matrix multiplication
// for harmonically-structured data.
//
// The insight: matrix multiplication is structurally blind to frequency.
// A matrix treats every element as an independent grid position — it can't
// know that column pairs encode cos/sin of the same harmonic. This is a
// substrate incompatibility (the transistor/capacitor analogy).
//
// The LC layer operates on harmonic bands (cos/sin pairs) natively:
//   - Resonance: per-band amplitude gain and phase rotation
//   - Coupling: cross-band interaction (like mutual inductance between coils)
//   - Nonlinearity: GELU activation
//
// This is the computational equivalent of an LC circuit:
//   - Resonance = frequency selection (amplify or attenuate per band)
//   - Coupling = mutual inductance (energy transfer between adjacent bands)
//   - Phase rotation = tuning the resonant frequency
//
// Instead of N×M weight matrix (131K params per MLP layer), the LC layer
// uses O(N) params per layer (~148 params). The question: does frequency-native
// structure buy enough expressiveness to compensate for the massive parameter
// reduction?
//
// Modes:
//   frozen_standard — frozen harmonic embeddings, standard MLP (baseline)
//   lc_layer        — frozen harmonic embeddings, LC layer replaces ALL MLPs
//
// Usage:
//   cargo run --release --bin train --features cuda

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{
    layer_norm, linear, linear_no_bias, ops, LayerNorm, Linear, Module, Optimizer, VarBuilder,
    VarMap,
};
use rand::Rng;
use std::fs;
use std::io::Write;
use std::path::Path;

// =============================================================================
// Configuration
// =============================================================================

const N_LAYER: usize = 4;
const N_HEAD: usize = 4;
const N_EMBD: usize = 128;
const N_BANDS: usize = N_EMBD / 2; // 64 harmonic bands
const COUPLING_K: usize = 2; // coupling neighborhood half-width (k=2 → 5-wide window)
const BLOCK_SIZE: usize = 256;
const BATCH_SIZE: usize = 64;
const LEARNING_RATE: f64 = 3e-4;
const MAX_ITERS: usize = 2000;
const EVAL_INTERVAL: usize = 250;
const EVAL_ITERS: usize = 50;

// =============================================================================
// Harmonic Embedding — deterministic phase encoding
// =============================================================================

fn build_harmonic_table(vocab_size: usize, n_embd: usize, device: &Device) -> Result<Tensor> {
    let n_harmonics = n_embd / 2;
    let mut data = vec![0f32; vocab_size * n_embd];

    let scale = 1.0 / (n_harmonics as f32).sqrt();
    for c in 0..vocab_size {
        let theta = (c as f32) * 2.0 * std::f32::consts::PI / (vocab_size as f32);
        for h in 0..n_harmonics {
            let n = (h + 1) as f32;
            let phase = n * theta;
            data[c * n_embd + h * 2] = phase.cos() * scale;
            data[c * n_embd + h * 2 + 1] = phase.sin() * scale;
        }
    }

    Tensor::from_vec(data, (vocab_size, n_embd), device)
}

fn build_positional_table(max_len: usize, n_embd: usize, device: &Device) -> Result<Tensor> {
    let n_harmonics = n_embd / 2;
    let mut data = vec![0f32; max_len * n_embd];

    let scale = 1.0 / (n_harmonics as f32).sqrt();
    for pos in 0..max_len {
        for h in 0..n_harmonics {
            let freq = 1.0 / 10000f32.powf(2.0 * (h as f32) / (n_embd as f32));
            let phase = (pos as f32) * freq;
            data[pos * n_embd + h * 2] = phase.cos() * scale;
            data[pos * n_embd + h * 2 + 1] = phase.sin() * scale;
        }
    }

    Tensor::from_vec(data, (max_len, n_embd), device)
}

fn build_causal_mask(size: usize, device: &Device) -> Result<Tensor> {
    let mut data = vec![f32::NEG_INFINITY; size * size];
    for i in 0..size {
        for j in 0..=i {
            data[i * size + j] = 0.0;
        }
    }
    Tensor::from_vec(data, (1, 1, size, size), device)
}

// =============================================================================
// LC Circuit Layer — the core innovation
// =============================================================================
//
// Replaces a standard MLP (128→512→128, 131K params) with:
//   Resonance (128 params) → Coupling (20 params) → GELU → output
//   Total: 148 params per layer
//
// The layer operates on harmonic bands: input (B*T, 128) is reshaped to
// (B*T, 64, 2) where each band is a (cos, sin) pair. Every operation
// respects the frequency structure.

struct LCLayer {
    // Resonance: per-band gain and phase rotation
    gain: Tensor,  // (N_BANDS,) — amplitude scaling per band
    phase: Tensor, // (N_BANDS,) — phase rotation per band

    // Coupling: cross-band interaction weights
    // Shape: (2*K+1, 2, 2) — for each offset in [-K..K], a 2×2 transform
    coupling_weights: Tensor,

    k: usize,
    n_bands: usize,
}

impl LCLayer {
    fn new(vb: VarBuilder, device: &Device) -> Result<Self> {
        // Resonance: init gain=1 (passthrough), phase=0 (no rotation)
        let gain = vb.pp("resonance").get_with_hints(
            (N_BANDS,),
            "gain",
            candle_nn::Init::Const(1.0),
        )?;
        let phase = vb.pp("resonance").get_with_hints(
            (N_BANDS,),
            "phase",
            candle_nn::Init::Const(0.0),
        )?;

        // Coupling: init to zero — residual design means zero coupling = passthrough
        let coupling_weights = vb.pp("coupling").get_with_hints(
            (2 * COUPLING_K + 1, 2, 2),
            "weight",
            candle_nn::Init::Const(0.0),
        )?;

        let _ = device; // used by caller for consistency

        Ok(Self {
            gain,
            phase,
            coupling_weights,
            k: COUPLING_K,
            n_bands: N_BANDS,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, c) = x.dims3()?;
        let bt = b * t;

        // 1. Reshape to harmonic bands: (B*T, N_BANDS, 2)
        let bands = x.reshape((bt, self.n_bands, 2))?;

        // 2. Resonance: per-band gain and phase rotation
        //    out_cos = gain * (in_cos * cos(δ) - in_sin * sin(δ))
        //    out_sin = gain * (in_cos * sin(δ) + in_sin * cos(δ))
        let cos_p = self.phase.cos()?; // (N_BANDS,)
        let sin_p = self.phase.sin()?;

        let in_cos = bands.i((.., .., 0))?; // (bt, N_BANDS)
        let in_sin = bands.i((.., .., 1))?;

        // Phase rotation
        let rot_cos = (in_cos.broadcast_mul(&cos_p)? - in_sin.broadcast_mul(&sin_p)?)?;
        let rot_sin = (in_cos.broadcast_mul(&sin_p)? + in_sin.broadcast_mul(&cos_p)?)?;

        // Amplitude scaling
        let res_cos = rot_cos.broadcast_mul(&self.gain)?;
        let res_sin = rot_sin.broadcast_mul(&self.gain)?;

        // Stack back: (bt, N_BANDS, 2)
        let resonated = Tensor::stack(&[res_cos, res_sin], 2)?;

        // 3. Coupling: residual cross-band interaction
        //    coupled = resonated + sum_j(coupling_weights[j] @ shifted_bands[j])
        //    Zero init means coupling starts as identity (no cross-band mixing)
        let coupled = self.coupling_residual(&resonated)?;

        // 4. GELU nonlinearity
        let activated = coupled.gelu()?;

        // 5. Reshape back: (B, T, C)
        activated.reshape((b, t, c))
    }

    fn coupling_residual(&self, x: &Tensor) -> Result<Tensor> {
        // x: (bt, n_bands, 2)
        let (bt, n_bands, _two) = x.dims3()?;
        let k = self.k;

        // Zero-pad along band dimension
        let zeros = Tensor::zeros((bt, k, 2), x.dtype(), x.device())?;
        let padded = Tensor::cat(&[&zeros, x, &zeros], 1)?; // (bt, n_bands + 2k, 2)

        // Accumulate coupling contributions
        let mut coupling_sum = Tensor::zeros((bt, n_bands, 2), x.dtype(), x.device())?;

        for j in 0..(2 * k + 1) {
            let slice = padded.narrow(1, j, n_bands)?; // (bt, n_bands, 2)
            let w = self.coupling_weights.i(j)?; // (2, 2)
            // candle matmul doesn't broadcast 3D × 2D — reshape to 2D
            let slice_2d = slice.reshape((bt * n_bands, 2))?;
            let contrib_2d = slice_2d.matmul(&w)?; // (bt*n_bands, 2)
            let contrib = contrib_2d.reshape((bt, n_bands, 2))?;
            coupling_sum = (coupling_sum + contrib)?;
        }

        // Residual: input + coupling
        x + coupling_sum
    }

    /// Report learned parameters for analysis.
    fn report_params(&self) -> Result<()> {
        let gain_vals: Vec<f32> = self.gain.flatten_all()?.to_vec1()?;
        let phase_vals: Vec<f32> = self.phase.flatten_all()?.to_vec1()?;

        // Summary stats
        let gain_min = gain_vals.iter().cloned().fold(f32::INFINITY, f32::min);
        let gain_max = gain_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let gain_avg: f32 = gain_vals.iter().sum::<f32>() / gain_vals.len() as f32;

        let phase_min = phase_vals.iter().cloned().fold(f32::INFINITY, f32::min);
        let phase_max = phase_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let phase_avg: f32 = phase_vals.iter().sum::<f32>() / phase_vals.len() as f32;

        println!(
            "      gain:  avg={gain_avg:.4} min={gain_min:.4} max={gain_max:.4}"
        );
        println!(
            "      phase: avg={phase_avg:.4} min={phase_min:.4} max={phase_max:.4}"
        );

        // Coupling weight magnitudes
        let coupling_vals: Vec<f32> = self.coupling_weights.flatten_all()?.to_vec1()?;
        let coupling_rms: f32 =
            (coupling_vals.iter().map(|v| v * v).sum::<f32>() / coupling_vals.len() as f32).sqrt();
        println!("      coupling RMS={coupling_rms:.6}");

        Ok(())
    }
}

// =============================================================================
// Standard MLP (for baseline comparison)
// =============================================================================

struct MLP {
    c_fc: Linear,
    c_proj: Linear,
}

impl MLP {
    fn new(vb: VarBuilder) -> Result<Self> {
        let c_fc = linear(N_EMBD, 4 * N_EMBD, vb.pp("c_fc"))?;
        let c_proj = linear(4 * N_EMBD, N_EMBD, vb.pp("c_proj"))?;
        Ok(Self { c_fc, c_proj })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.c_fc.forward(x)?;
        let x = x.gelu()?;
        self.c_proj.forward(&x)
    }
}

// =============================================================================
// Attention (standard, shared by both modes)
// =============================================================================

struct CausalSelfAttention {
    c_attn: Linear,
    c_proj: Linear,
    mask: Tensor,
    n_head: usize,
    n_embd: usize,
}

impl CausalSelfAttention {
    fn new(vb: VarBuilder, device: &Device) -> Result<Self> {
        let c_attn = linear(N_EMBD, 3 * N_EMBD, vb.pp("c_attn"))?;
        let c_proj = linear(N_EMBD, N_EMBD, vb.pp("c_proj"))?;
        let mask = build_causal_mask(BLOCK_SIZE, device)?;
        Ok(Self {
            c_attn,
            c_proj,
            mask,
            n_head: N_HEAD,
            n_embd: N_EMBD,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, c) = x.dims3()?;
        let head_dim = c / self.n_head;

        let qkv = self.c_attn.forward(x)?;
        let q = qkv.narrow(D::Minus1, 0, self.n_embd)?;
        let k = qkv.narrow(D::Minus1, self.n_embd, self.n_embd)?;
        let v = qkv.narrow(D::Minus1, 2 * self.n_embd, self.n_embd)?;

        let q = q
            .reshape((b, t, self.n_head, head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b, t, self.n_head, head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b, t, self.n_head, head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let scale = 1.0 / (head_dim as f64).sqrt();
        let k_t = k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        let scores = (q.matmul(&k_t)? * scale)?;

        let mask = self
            .mask
            .i((.., .., ..t, ..t))?
            .broadcast_as(scores.shape())?;
        let att = (scores + mask)?;
        let att = ops::softmax(&att, D::Minus1)?;
        let y = att.matmul(&v)?;
        let y = y.transpose(1, 2)?.contiguous()?.reshape((b, t, c))?;
        self.c_proj.forward(&y)
    }
}

// =============================================================================
// Transformer Block
// =============================================================================

enum FFN {
    Standard(MLP),
    LC(LCLayer),
}

struct Block {
    ln_1: LayerNorm,
    attn: CausalSelfAttention,
    ln_2: LayerNorm,
    ffn: FFN,
}

impl Block {
    fn new(vb: VarBuilder, device: &Device, use_lc: bool) -> Result<Self> {
        let ln_1 = layer_norm(N_EMBD, candle_nn::LayerNormConfig::default(), vb.pp("ln_1"))?;
        let attn = CausalSelfAttention::new(vb.pp("attn"), device)?;
        let ln_2 = layer_norm(N_EMBD, candle_nn::LayerNormConfig::default(), vb.pp("ln_2"))?;

        let ffn = if use_lc {
            FFN::LC(LCLayer::new(vb.pp("lc"), device)?)
        } else {
            FFN::Standard(MLP::new(vb.pp("mlp"))?)
        };

        Ok(Self {
            ln_1,
            attn,
            ln_2,
            ffn,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = (x + self.attn.forward(&self.ln_1.forward(x)?)?)?;
        let ffn_out = match &self.ffn {
            FFN::Standard(mlp) => mlp.forward(&self.ln_2.forward(&x)?)?,
            FFN::LC(lc) => lc.forward(&self.ln_2.forward(&x)?)?,
        };
        let x = (&x + ffn_out)?;
        Ok(x)
    }
}

// =============================================================================
// The Model
// =============================================================================

struct LCGPT {
    wte: Tensor,
    wpe: Tensor,
    blocks: Vec<Block>,
    ln_f: LayerNorm,
    lm_head: Linear,
    use_lc: bool,
}

impl LCGPT {
    fn new(
        vocab_size: usize,
        use_lc: bool,
        varmap: &VarMap,
        device: &Device,
    ) -> Result<Self> {
        let vb = VarBuilder::from_varmap(varmap, DType::F32, device);

        // Always frozen harmonic embeddings
        let wte = build_harmonic_table(vocab_size, N_EMBD, device)?;
        let wpe = build_positional_table(BLOCK_SIZE, N_EMBD, device)?;

        let mut blocks = Vec::new();
        for i in 0..N_LAYER {
            blocks.push(Block::new(vb.pp(format!("blocks.{i}")), device, use_lc)?);
        }

        let ln_f = layer_norm(
            N_EMBD,
            candle_nn::LayerNormConfig::default(),
            vb.pp("ln_f"),
        )?;
        let lm_head = linear_no_bias(N_EMBD, vocab_size, vb.pp("lm_head"))?;

        let n_params: usize = varmap
            .all_vars()
            .iter()
            .map(|v| v.as_tensor().elem_count())
            .sum();
        let mode_str = if use_lc { "lc_layer" } else { "standard" };
        println!("  {mode_str} model: {n_params} trainable parameters");

        if use_lc {
            let lc_params = N_LAYER * (N_BANDS + N_BANDS + (2 * COUPLING_K + 1) * 2 * 2);
            let attn_params = n_params - lc_params;
            println!(
                "  LC params: {} per layer x {} layers = {} total",
                lc_params / N_LAYER,
                N_LAYER,
                lc_params
            );
            println!("  Attention + other params: {attn_params}");
            println!(
                "  Standard MLP equivalent: {} params",
                N_LAYER * (N_EMBD * 4 * N_EMBD + 4 * N_EMBD + 4 * N_EMBD * N_EMBD + N_EMBD)
            );
        }

        Ok(Self {
            wte,
            wpe,
            blocks,
            ln_f,
            lm_head,
            use_lc,
        })
    }

    fn forward(&self, idx: &Tensor, targets: Option<&Tensor>) -> Result<(Tensor, Option<Tensor>)> {
        let (b, t) = idx.dims2()?;

        let idx_flat = idx.flatten_all()?;
        let tok_emb = self.wte.index_select(&idx_flat, 0)?;
        let tok_emb = tok_emb.reshape((b, t, N_EMBD))?;
        let pos_emb = self.wpe.i(0..t)?;
        let x = tok_emb.broadcast_add(&pos_emb)?;

        let mut x = x;
        for block in &self.blocks {
            x = block.forward(&x)?;
        }

        x = self.ln_f.forward(&x)?;
        let logits = self.lm_head.forward(&x)?;

        let loss = match targets {
            Some(targets) => {
                let (b, t, vs) = logits.dims3()?;
                let logits_flat = logits.reshape((b * t, vs))?;
                let targets_flat = targets.reshape((b * t,))?;
                Some(candle_nn::loss::cross_entropy(&logits_flat, &targets_flat)?)
            }
            None => None,
        };

        Ok((logits, loss))
    }

    /// Report LC layer parameters for analysis.
    fn report_lc_params(&self) -> Result<()> {
        if !self.use_lc {
            return Ok(());
        }
        println!("\n  LC Layer Parameters:");
        for (i, block) in self.blocks.iter().enumerate() {
            if let FFN::LC(lc) = &block.ffn {
                println!("    layer {i}:");
                lc.report_params()?;
            }
        }
        Ok(())
    }
}

// =============================================================================
// Attention Entropy Measurement
// =============================================================================

fn measure_attention_entropy(
    model: &LCGPT,
    dataset: &Dataset,
    device: &Device,
) -> Result<Vec<Vec<f32>>> {
    let (x, _) = dataset.get_batch("val", device)?;
    let (b, t) = x.dims2()?;
    let head_dim = N_EMBD / N_HEAD;

    let idx_flat = x.flatten_all()?;
    let tok_emb = model
        .wte
        .index_select(&idx_flat, 0)?
        .reshape((b, t, N_EMBD))?;
    let pos_emb = model.wpe.i(0..t)?;
    let emb = tok_emb.broadcast_add(&pos_emb)?;
    let mut hidden = emb;

    let mut all_entropies = Vec::new();

    for block in &model.blocks {
        let normed = block.ln_1.forward(&hidden)?;
        let attn = &block.attn;

        let qkv = attn.c_attn.forward(&normed)?;
        let q = qkv.narrow(D::Minus1, 0, N_EMBD)?;
        let k = qkv.narrow(D::Minus1, N_EMBD, N_EMBD)?;
        let v = qkv.narrow(D::Minus1, 2 * N_EMBD, N_EMBD)?;

        let q = q
            .reshape((b, t, N_HEAD, head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b, t, N_HEAD, head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b, t, N_HEAD, head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let scale = 1.0 / (head_dim as f64).sqrt();
        let k_t = k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        let scores = (q.matmul(&k_t)? * scale)?;

        let mask_slice = attn
            .mask
            .i((.., .., ..t, ..t))?
            .broadcast_as(scores.shape())?;
        let scores = (scores + mask_slice)?;
        let att_weights = ops::softmax(&scores, D::Minus1)?;

        // Entropy
        let log_att = (att_weights.clone() + 1e-10)?.log()?;
        let neg_plogp = (att_weights.clone() * log_att)?.neg()?;
        let token_entropy = neg_plogp.sum(D::Minus1)?;
        let head_entropy = token_entropy.mean(D::Minus1)?.mean(0)?;
        let entropies: Vec<f32> = head_entropy.to_vec1()?;
        all_entropies.push(entropies);

        // Continue forward pass through the full block
        let y = att_weights.matmul(&v)?;
        let y = y.transpose(1, 2)?.contiguous()?.reshape((b, t, N_EMBD))?;
        let attn_out = attn.c_proj.forward(&y)?;
        hidden = (hidden + attn_out)?;
        let normed2 = block.ln_2.forward(&hidden)?;
        let ffn_out = match &block.ffn {
            FFN::Standard(mlp) => mlp.forward(&normed2)?,
            FFN::LC(lc) => lc.forward(&normed2)?,
        };
        hidden = (hidden + ffn_out)?;
    }

    Ok(all_entropies)
}

// =============================================================================
// Data
// =============================================================================

fn download_shakespeare() -> String {
    let data_dir = "data";
    let filepath = format!("{data_dir}/shakespeare.txt");

    if !Path::new(&filepath).exists() {
        println!("Downloading Shakespeare dataset...");
        fs::create_dir_all(data_dir).unwrap();

        let url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt";
        let output = std::process::Command::new("curl")
            .args(["-sL", url, "-o", &filepath])
            .output()
            .expect("Failed to download. Install curl or download manually.");
        if !output.status.success() {
            panic!("Download failed");
        }
        println!("Done.");
    }

    fs::read_to_string(&filepath).expect("Failed to read shakespeare.txt")
}

#[allow(dead_code)]
struct Dataset {
    train: Vec<u32>,
    val: Vec<u32>,
    vocab_size: usize,
    itos: Vec<char>,
}

impl Dataset {
    fn new(text: &str) -> Self {
        let mut chars: Vec<char> = text
            .chars()
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        chars.sort();
        let vocab_size = chars.len();
        let stoi: std::collections::HashMap<char, u32> = chars
            .iter()
            .enumerate()
            .map(|(i, &c)| (c, i as u32))
            .collect();

        let data: Vec<u32> = text.chars().map(|c| stoi[&c]).collect();

        let n = (data.len() as f64 * 0.9) as usize;
        let train = data[..n].to_vec();
        let val = data[n..].to_vec();

        Dataset {
            train,
            val,
            vocab_size,
            itos: chars,
        }
    }

    fn get_batch(&self, split: &str, device: &Device) -> Result<(Tensor, Tensor)> {
        let data = match split {
            "train" => &self.train,
            "val" => &self.val,
            _ => panic!("Unknown split"),
        };

        let mut rng = rand::rng();
        let max_start = data.len() - BLOCK_SIZE - 1;

        let mut x_data = Vec::with_capacity(BATCH_SIZE * BLOCK_SIZE);
        let mut y_data = Vec::with_capacity(BATCH_SIZE * BLOCK_SIZE);

        for _ in 0..BATCH_SIZE {
            let start = rng.random_range(0..max_start);
            x_data.extend_from_slice(&data[start..start + BLOCK_SIZE]);
            y_data.extend_from_slice(&data[start + 1..start + BLOCK_SIZE + 1]);
        }

        let x = Tensor::from_vec(x_data, (BATCH_SIZE, BLOCK_SIZE), device)?;
        let y = Tensor::from_vec(y_data, (BATCH_SIZE, BLOCK_SIZE), device)?;
        Ok((x, y))
    }
}

// =============================================================================
// Training
// =============================================================================

fn estimate_loss(model: &LCGPT, dataset: &Dataset, device: &Device) -> Result<(f32, f32)> {
    let mut train_loss = 0.0;
    let mut val_loss = 0.0;

    for _ in 0..EVAL_ITERS {
        let (x, y) = dataset.get_batch("train", device)?;
        let (_, loss) = model.forward(&x, Some(&y))?;
        train_loss += loss.unwrap().to_scalar::<f32>()?;
    }

    for _ in 0..EVAL_ITERS {
        let (x, y) = dataset.get_batch("val", device)?;
        let (_, loss) = model.forward(&x, Some(&y))?;
        val_loss += loss.unwrap().to_scalar::<f32>()?;
    }

    Ok((
        train_loss / EVAL_ITERS as f32,
        val_loss / EVAL_ITERS as f32,
    ))
}

struct TrainResult {
    mode_name: String,
    history: Vec<(usize, f32, f32)>,
    entropy: Vec<Vec<f32>>,
    n_params: usize,
}

fn train_mode(
    mode_name: &str,
    use_lc: bool,
    dataset: &Dataset,
    device: &Device,
) -> Result<TrainResult> {
    println!("\n{}", "=".repeat(60));
    println!("  Training: {}", mode_name.to_uppercase());
    println!(
        "  FFN: {}",
        if use_lc {
            "LC circuit layer (resonance + coupling + GELU)"
        } else {
            "standard MLP (128→512→128)"
        }
    );
    println!("{}", "=".repeat(60));

    let varmap = VarMap::new();
    let model = LCGPT::new(dataset.vocab_size, use_lc, &varmap, device)?;

    let n_params: usize = varmap
        .all_vars()
        .iter()
        .map(|v| v.as_tensor().elem_count())
        .sum();

    let mut opt = candle_nn::AdamW::new(
        varmap.all_vars(),
        candle_nn::ParamsAdamW {
            lr: LEARNING_RATE,
            ..Default::default()
        },
    )?;

    let mut history = Vec::new();
    let start = std::time::Instant::now();

    for iter_num in 0..MAX_ITERS {
        if iter_num % EVAL_INTERVAL == 0 || iter_num == MAX_ITERS - 1 {
            let (train_l, val_l) = estimate_loss(&model, dataset, device)?;
            let elapsed = start.elapsed().as_secs_f32();
            println!(
                "  step {:>5} | train loss {:.4} | val loss {:.4} | {:.1}s",
                iter_num, train_l, val_l, elapsed
            );

            // Report LC params at each checkpoint
            if use_lc {
                model.report_lc_params()?;
            }

            history.push((iter_num, train_l, val_l));
        }

        let (x, y) = dataset.get_batch("train", device)?;
        let (_, loss) = model.forward(&x, Some(&y))?;
        let loss = loss.unwrap();
        opt.backward_step(&loss)?;
    }

    let total = start.elapsed().as_secs_f32();
    println!("  Training complete in {:.1}s", total);

    // Measure attention entropy
    println!("  Measuring attention entropy...");
    let entropy = measure_attention_entropy(&model, dataset, device)?;

    // Dump weights
    let dir = format!("weights/{mode_name}");
    fs::create_dir_all(&dir).unwrap();
    let data = varmap.data().lock().unwrap();
    let mut count = 0;
    for (name, var) in data.iter() {
        let tensor = var.as_tensor();
        let shape: Vec<usize> = tensor.shape().dims().to_vec();
        let flat: Vec<f32> = tensor.flatten_all()?.to_vec1()?;
        write_tensor_binary(&format!("{dir}/{name}.bin"), &shape, &flat);
        count += 1;
    }
    drop(data);
    println!("  Dumped {count} weight tensors to {dir}/");

    Ok(TrainResult {
        mode_name: mode_name.to_string(),
        history,
        entropy,
        n_params,
    })
}

fn write_tensor_binary(path: &str, shape: &[usize], values: &[f32]) {
    let mut file = fs::File::create(path)
        .unwrap_or_else(|e| panic!("Failed to create {path}: {e}"));
    let ndims = shape.len() as u32;
    file.write_all(&ndims.to_le_bytes()).unwrap();
    for &d in shape {
        file.write_all(&(d as u32).to_le_bytes()).unwrap();
    }
    for &v in values {
        file.write_all(&v.to_le_bytes()).unwrap();
    }
}

// =============================================================================
// Main
// =============================================================================

fn main() -> Result<()> {
    let device = Device::cuda_if_available(0)?;
    let device_name = if device.is_cuda() { "CUDA" } else { "CPU" };

    println!("{}", "=".repeat(60));
    println!("  Phase 20: LC Circuit Layer");
    println!("  Can frequency-native computation replace matrix multiplication?");
    println!("  Device: {device_name}");
    println!("{}", "=".repeat(60));

    let text = download_shakespeare();
    let dataset = Dataset::new(&text);
    println!(
        "\n  Dataset: {} characters, {} unique",
        text.len(),
        dataset.vocab_size
    );
    println!(
        "  Train: {} | Val: {}",
        dataset.train.len(),
        dataset.val.len()
    );
    println!(
        "  Model: {} layers, {} heads, {} dim, {} bands",
        N_LAYER, N_HEAD, N_EMBD, N_BANDS
    );

    println!("\n  LC layer architecture:");
    println!("    Resonance: per-band gain + phase rotation ({} params)", 2 * N_BANDS);
    println!(
        "    Coupling: {}-wide cross-band interaction ({} params)",
        2 * COUPLING_K + 1,
        (2 * COUPLING_K + 1) * 2 * 2
    );
    println!("    Nonlinearity: GELU");
    let lc_per_layer = 2 * N_BANDS + (2 * COUPLING_K + 1) * 2 * 2;
    println!(
        "    Total: {} params per layer vs ~131K for standard MLP",
        lc_per_layer
    );
    println!(
        "    Reduction: {:.0}x fewer parameters",
        131712.0 / lc_per_layer as f64
    );

    println!("\n  Reference results:");
    println!("    Frozen standard (Phase 17): val ~3.08-3.09");

    // Two modes
    let result_standard = train_mode("frozen_standard", false, &dataset, &device)?;
    let result_lc = train_mode("lc_layer", true, &dataset, &device)?;

    let results = vec![&result_standard, &result_lc];

    // =========================================================================
    // Comparison
    // =========================================================================
    println!("\n{}", "=".repeat(60));
    println!("  COMPARISON: Final Validation Loss");
    println!("{}", "=".repeat(60));
    println!();
    println!(
        "  {:<20} {:>10} {:>12} {:>10} {:>8}",
        "Mode", "Val Loss", "Train Loss", "vs Std", "Params"
    );
    println!(
        "  {:<20} {:>10} {:>12} {:>10} {:>8}",
        "-".repeat(20),
        "-".repeat(10),
        "-".repeat(12),
        "-".repeat(10),
        "-".repeat(8),
    );

    let std_val = result_standard.history.last().unwrap().2;

    for result in &results {
        let (_, train_l, val_l) = result.history.last().unwrap();
        let diff = if result.mode_name == "frozen_standard" {
            "—".to_string()
        } else {
            let pct = (1.0 - val_l / std_val) * 100.0;
            format!("{pct:+.1}%")
        };
        println!(
            "  {:<20} {:>10.4} {:>12.4} {:>10} {:>8}",
            result.mode_name,
            val_l,
            train_l,
            diff,
            format!("{}K", result.n_params / 1000)
        );
    }

    // Parameter efficiency
    println!();
    let std_params = result_standard.n_params;
    let lc_params = result_lc.n_params;
    let lc_val = result_lc.history.last().unwrap().2;
    println!(
        "  Parameter reduction: {} → {} ({:.1}x fewer)",
        std_params,
        lc_params,
        std_params as f64 / lc_params as f64
    );
    println!(
        "  Loss per 1K params:  standard={:.4}, LC={:.4}",
        std_val / (std_params as f32 / 1000.0),
        lc_val / (lc_params as f32 / 1000.0),
    );

    // =========================================================================
    // Convergence Speed
    // =========================================================================
    println!("\n{}", "=".repeat(60));
    println!("  CONVERGENCE: Val Loss at Each Checkpoint");
    println!("{}", "=".repeat(60));
    println!();
    print!("  {:>6}", "Step");
    for result in &results {
        print!("  {:>16}", result.mode_name);
    }
    println!("  {:>10}", "LC gain");
    println!("  {}", "-".repeat(60));

    let std_hist = &result_standard.history;
    let lc_hist = &result_lc.history;
    for i in 0..std_hist.len() {
        let (step, _, std_v) = std_hist[i];
        let (_, _, lc_v) = lc_hist[i];
        let gain = (1.0 - lc_v / std_v) * 100.0;
        println!(
            "  {:>6}  {:>16.4}  {:>16.4}  {:>+9.1}%",
            step, std_v, lc_v, gain
        );
    }

    // =========================================================================
    // Attention Entropy
    // =========================================================================
    println!("\n{}", "=".repeat(60));
    println!("  ATTENTION HEAD ENTROPY (lower = more specialised)");
    println!("{}", "=".repeat(60));

    for result in &results {
        println!("\n  {}:", result.mode_name);
        for (layer, head_entropies) in result.entropy.iter().enumerate() {
            let avg: f32 = head_entropies.iter().sum::<f32>() / head_entropies.len() as f32;
            print!("    layer {layer}:");
            for (h, &e) in head_entropies.iter().enumerate() {
                print!("  h{h}={e:.2}");
            }
            println!("  avg={avg:.2}");
        }
    }

    // =========================================================================
    // Summary
    // =========================================================================
    println!("\n{}", "=".repeat(60));
    println!("  SUMMARY");
    println!("{}", "=".repeat(60));

    let improvement = (1.0 - lc_val / std_val) * 100.0;

    println!();
    if lc_val < std_val {
        println!(
            "  LC layer OUTPERFORMS standard MLP by {:.2}%.",
            improvement
        );
        println!("  Frequency-native computation provides a better inductive bias!");
    } else if (lc_val - std_val).abs() / std_val < 0.05 {
        println!(
            "  LC layer within {:.1}% of standard MLP with {:.0}x fewer FFN params.",
            improvement.abs(),
            std_params as f64 / lc_params as f64
        );
        println!("  Frequency-native computation is viable — structure compensates for capacity.");
    } else {
        println!(
            "  LC layer {:.1}% worse than standard MLP.",
            improvement.abs()
        );
        println!(
            "  The {:.0}x parameter reduction exceeds what frequency structure can compensate.",
            std_params as f64 / lc_params as f64
        );
        println!("  The concept has merit if the gap is smaller than the parameter ratio.");
        let param_ratio = std_params as f64 / lc_params as f64;
        let loss_ratio = lc_val as f64 / std_val as f64;
        println!(
            "  Param ratio: {:.0}x | Loss ratio: {:.2}x | Efficiency: {:.1}x",
            param_ratio,
            loss_ratio,
            param_ratio / loss_ratio
        );
    }

    println!();
    println!("{}", "=".repeat(60));

    Ok(())
}
