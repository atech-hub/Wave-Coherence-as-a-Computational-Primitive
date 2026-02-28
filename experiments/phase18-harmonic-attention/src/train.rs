// Phase 18: Harmonic Attention Heads
//
// Tests whether structuring attention heads by harmonic order improves
// performance. Each head is initialized to attend using a specific harmonic
// order, providing structured relationship detection by construction.
//
// Modes:
//   frozen_standard  — frozen harmonic embeddings, standard attention (Phase 17 rerun)
//   harmonic_heads   — frozen embeddings, Q/K initialized to harmonic orders (trainable)
//   frozen_heads     — frozen embeddings, Q/K frozen at harmonic init (only V trainable)
//
// Based on Phase 17 train.rs with harmonic attention modifications.
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
const BLOCK_SIZE: usize = 256;
const BATCH_SIZE: usize = 64;
const LEARNING_RATE: f64 = 3e-4;
const MAX_ITERS: usize = 2000;
const EVAL_INTERVAL: usize = 250;
const EVAL_ITERS: usize = 50;

/// Harmonic orders assigned to each attention head.
/// Octave-spaced: 1, 2, 4, 8 — each head specialises in a different
/// frequency scale of the harmonic embedding.
const HARMONIC_ORDERS: [usize; N_HEAD] = [1, 2, 4, 8];

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
// Harmonic Q/K Weight Initialization
// =============================================================================

/// Generate a standard normal random f32 via Box-Muller transform.
fn randn(rng: &mut impl Rng) -> f32 {
    let u1: f32 = rng.random_range(0.0001f32..1.0f32);
    let u2: f32 = rng.random_range(0.0f32..1.0f32);
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

/// Build a Q/K weight matrix with harmonic structure.
///
/// Shape: (2 * n_embd, n_embd) = (256, 128) — rows 0..128 are Q, rows 128..256 are K.
///
/// For each head h, the Q/K rows emphasize the embedding dimensions corresponding
/// to harmonic order HARMONIC_ORDERS[h]:
///   - Emphasized dims (cos(n_h*θ), sin(n_h*θ)): normal scale (0.02)
///   - All other dims: 1% of normal scale (0.0002)
///
/// This warm-starts each head to detect relationships at its assigned harmonic order
/// while allowing the model to learn adjustments during training.
fn build_harmonic_qk_init(n_embd: usize, n_head: usize, device: &Device) -> Result<Tensor> {
    let head_dim = n_embd / n_head;
    let qk_rows = 2 * n_embd;
    let mut data = vec![0.0f32; qk_rows * n_embd];
    let mut rng = rand::rng();

    let normal_scale = 0.02f32;
    let suppressed_scale = normal_scale * 0.01;

    for head in 0..n_head {
        let n_h = HARMONIC_ORDERS[head];
        // Emphasized input dims: cos(n_h*θ) at dim 2*(n_h-1), sin(n_h*θ) at dim 2*(n_h-1)+1
        let emph_dim0 = 2 * (n_h - 1);
        let emph_dim1 = 2 * (n_h - 1) + 1;

        // Q block for this head: output rows [head*head_dim .. (head+1)*head_dim]
        for out_row in (head * head_dim)..((head + 1) * head_dim) {
            for in_col in 0..n_embd {
                let scale = if in_col == emph_dim0 || in_col == emph_dim1 {
                    normal_scale
                } else {
                    suppressed_scale
                };
                let val = randn(&mut rng);
                data[out_row * n_embd + in_col] = val * scale;
            }
        }

        // K block for this head: output rows [n_embd + head*head_dim .. n_embd + (head+1)*head_dim]
        for out_row in (n_embd + head * head_dim)..(n_embd + (head + 1) * head_dim) {
            for in_col in 0..n_embd {
                let scale = if in_col == emph_dim0 || in_col == emph_dim1 {
                    normal_scale
                } else {
                    suppressed_scale
                };
                let val = randn(&mut rng);
                data[out_row * n_embd + in_col] = val * scale;
            }
        }
    }

    Tensor::from_vec(data, (qk_rows, n_embd), device)
}

// =============================================================================
// Model Components
// =============================================================================

struct CausalSelfAttention {
    // Standard/warm-start mode: single Linear for Q+K+V
    c_attn: Option<Linear>,
    // Frozen Q/K mode: separate tensors
    qk_weight: Option<Tensor>, // (256, 128) — NOT in VarMap, no gradients
    qk_bias: Option<Tensor>,   // (256,)
    c_v: Option<Linear>,       // trainable V projection (128, 128)
    // Common
    c_proj: Linear,
    mask: Tensor,
    n_head: usize,
    n_embd: usize,
}

impl CausalSelfAttention {
    fn new(vb: VarBuilder, device: &Device, frozen_qk: bool) -> Result<Self> {
        let c_proj = linear(N_EMBD, N_EMBD, vb.pp("c_proj"))?;
        let mask = build_causal_mask(BLOCK_SIZE, device)?;

        if frozen_qk {
            // Q/K are frozen at harmonic init — plain Tensors, not Vars
            let qk_weight = build_harmonic_qk_init(N_EMBD, N_HEAD, device)?;
            let qk_bias = Tensor::zeros(&[2 * N_EMBD], DType::F32, device)?;
            // V projection is trainable
            let c_v = linear(N_EMBD, N_EMBD, vb.pp("c_v"))?;

            Ok(Self {
                c_attn: None,
                qk_weight: Some(qk_weight),
                qk_bias: Some(qk_bias),
                c_v: Some(c_v),
                c_proj,
                mask,
                n_head: N_HEAD,
                n_embd: N_EMBD,
            })
        } else {
            // Standard: single Linear for Q+K+V
            let c_attn = linear(N_EMBD, 3 * N_EMBD, vb.pp("c_attn"))?;

            Ok(Self {
                c_attn: Some(c_attn),
                qk_weight: None,
                qk_bias: None,
                c_v: None,
                c_proj,
                mask,
                n_head: N_HEAD,
                n_embd: N_EMBD,
            })
        }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, c) = x.dims3()?;
        let head_dim = c / self.n_head;

        let (q, k, v) = if let Some(ref c_attn) = self.c_attn {
            let qkv = c_attn.forward(x)?;
            let q = qkv.narrow(D::Minus1, 0, self.n_embd)?;
            let k = qkv.narrow(D::Minus1, self.n_embd, self.n_embd)?;
            let v = qkv.narrow(D::Minus1, 2 * self.n_embd, self.n_embd)?;
            (q, k, v)
        } else {
            // Frozen Q/K mode — reshape to 2D for matmul, then back to 3D
            let qk_w = self.qk_weight.as_ref().unwrap();
            let qk_b = self.qk_bias.as_ref().unwrap();
            let x_2d = x.reshape((b * t, c))?;
            let qk_2d = x_2d.matmul(&qk_w.t()?)?.broadcast_add(qk_b)?;
            let qk = qk_2d.reshape((b, t, 2 * self.n_embd))?;
            let q = qk.narrow(D::Minus1, 0, self.n_embd)?;
            let k = qk.narrow(D::Minus1, self.n_embd, self.n_embd)?;
            let v = self.c_v.as_ref().unwrap().forward(x)?;
            (q, k, v)
        };

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
        let att = (q.matmul(&k_t)? * scale)?;

        let mask = self.mask.i((.., .., ..t, ..t))?.broadcast_as(att.shape())?;
        let att = (att + mask)?;

        let att = ops::softmax(&att, D::Minus1)?;
        let y = att.matmul(&v)?;
        let y = y.transpose(1, 2)?.contiguous()?.reshape((b, t, c))?;
        self.c_proj.forward(&y)
    }
}

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

struct Block {
    ln_1: LayerNorm,
    attn: CausalSelfAttention,
    ln_2: LayerNorm,
    mlp: MLP,
}

impl Block {
    fn new(vb: VarBuilder, device: &Device, frozen_qk: bool) -> Result<Self> {
        let ln_1 = layer_norm(N_EMBD, candle_nn::LayerNormConfig::default(), vb.pp("ln_1"))?;
        let attn = CausalSelfAttention::new(vb.pp("attn"), device, frozen_qk)?;
        let ln_2 = layer_norm(N_EMBD, candle_nn::LayerNormConfig::default(), vb.pp("ln_2"))?;
        let mlp = MLP::new(vb.pp("mlp"))?;
        Ok(Self {
            ln_1,
            attn,
            ln_2,
            mlp,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = (x + self.attn.forward(&self.ln_1.forward(x)?)?)?;
        let x = (&x + self.mlp.forward(&self.ln_2.forward(&x)?)?)?;
        Ok(x)
    }
}

// =============================================================================
// The Model
// =============================================================================

struct HarmonicGPT {
    wte: Tensor,
    wpe: Tensor,
    blocks: Vec<Block>,
    ln_f: LayerNorm,
    lm_head: Linear,
}

impl HarmonicGPT {
    fn new(
        vocab_size: usize,
        embed_mode: &str,
        attn_mode: &str,
        varmap: &VarMap,
        device: &Device,
    ) -> Result<Self> {
        let vb = VarBuilder::from_varmap(varmap, DType::F32, device);

        // Embedding setup (same as Phase 17)
        let wte = match embed_mode {
            "baseline" => vb.pp("wte").get_with_hints(
                (vocab_size, N_EMBD),
                "weight",
                candle_nn::Init::Randn {
                    mean: 0.0,
                    stdev: 0.02,
                },
            )?,
            "harmonic" => {
                let var = vb.pp("wte").get_with_hints(
                    (vocab_size, N_EMBD),
                    "weight",
                    candle_nn::Init::Const(0.0),
                )?;
                {
                    let data = varmap.data().lock().unwrap();
                    if let Some(var) = data.get("wte.weight") {
                        let harmonic = build_harmonic_table(vocab_size, N_EMBD, device)?;
                        var.set(&harmonic)?;
                    }
                }
                var
            }
            "frozen" => build_harmonic_table(vocab_size, N_EMBD, device)?,
            _ => panic!("Unknown embed mode: {embed_mode}"),
        };

        let wpe = match embed_mode {
            "baseline" => vb.pp("wpe").get_with_hints(
                (BLOCK_SIZE, N_EMBD),
                "weight",
                candle_nn::Init::Randn {
                    mean: 0.0,
                    stdev: 0.02,
                },
            )?,
            "harmonic" => {
                let var = vb.pp("wpe").get_with_hints(
                    (BLOCK_SIZE, N_EMBD),
                    "weight",
                    candle_nn::Init::Const(0.0),
                )?;
                {
                    let data = varmap.data().lock().unwrap();
                    if let Some(var) = data.get("wpe.weight") {
                        let table = build_positional_table(BLOCK_SIZE, N_EMBD, device)?;
                        var.set(&table)?;
                    }
                }
                var
            }
            "frozen" => build_positional_table(BLOCK_SIZE, N_EMBD, device)?,
            _ => unreachable!(),
        };

        // Attention mode
        let frozen_qk = attn_mode == "frozen_qk";
        let mut blocks = Vec::new();
        for i in 0..N_LAYER {
            blocks.push(Block::new(
                vb.pp(format!("blocks.{i}")),
                device,
                frozen_qk,
            )?);
        }

        let ln_f = layer_norm(
            N_EMBD,
            candle_nn::LayerNormConfig::default(),
            vb.pp("ln_f"),
        )?;
        let lm_head = linear_no_bias(N_EMBD, vocab_size, vb.pp("lm_head"))?;

        // For harmonic warm-start: overwrite Q/K portions of c_attn weights
        if attn_mode == "harmonic_warmstart" {
            let data = varmap.data().lock().unwrap();
            for i in 0..N_LAYER {
                let key = format!("blocks.{i}.attn.c_attn.weight");
                if let Some(var) = data.get(&key) {
                    let original = var.as_tensor();
                    // Keep V portion (rows 2*N_EMBD .. 3*N_EMBD), replace Q/K
                    let v_portion = original.narrow(0, 2 * N_EMBD, N_EMBD)?.contiguous()?;
                    let qk_init = build_harmonic_qk_init(N_EMBD, N_HEAD, device)?;
                    let new_weight = Tensor::cat(&[&qk_init, &v_portion], 0)?;
                    var.set(&new_weight)?;
                }
            }
        }

        let n_params: usize = varmap
            .all_vars()
            .iter()
            .map(|v| v.as_tensor().elem_count())
            .sum();
        println!(
            "  {embed_mode}/{attn_mode} model: {n_params} trainable parameters"
        );

        Ok(Self {
            wte,
            wpe,
            blocks,
            ln_f,
            lm_head,
        })
    }

    fn forward(&self, idx: &Tensor, targets: Option<&Tensor>) -> Result<(Tensor, Option<Tensor>)> {
        let (b, t) = idx.dims2()?;

        let idx_flat = idx.flatten_all()?;
        let tok_emb = self.wte.index_select(&idx_flat, 0)?;
        let tok_emb = tok_emb.reshape((b, t, N_EMBD))?;
        let pos_emb = self.wpe.i(0..t)?;
        let mut x = tok_emb.broadcast_add(&pos_emb)?;

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
}

// =============================================================================
// Attention Entropy Measurement
// =============================================================================

/// Measure per-head attention entropy for each layer.
/// Returns Vec<Vec<f32>> — [layer][head] entropy values.
/// Lower entropy = more specialised attention pattern.
fn measure_attention_entropy(
    model: &HarmonicGPT,
    dataset: &Dataset,
    device: &Device,
) -> Result<Vec<Vec<f32>>> {
    let (x, _) = dataset.get_batch("val", device)?;
    let (b, t) = x.dims2()?;
    let head_dim = N_EMBD / N_HEAD;

    let idx_flat = x.flatten_all()?;
    let tok_emb = model.wte.index_select(&idx_flat, 0)?.reshape((b, t, N_EMBD))?;
    let pos_emb = model.wpe.i(0..t)?;
    let mut hidden = tok_emb.broadcast_add(&pos_emb)?;

    let mut all_entropies = Vec::new();

    for block in &model.blocks {
        let normed = block.ln_1.forward(&hidden)?;
        let attn = &block.attn;

        // Recompute attention scores to capture the softmax distribution
        let (q, k, v) = if let Some(ref c_attn) = attn.c_attn {
            let qkv = c_attn.forward(&normed)?;
            let q = qkv.narrow(D::Minus1, 0, N_EMBD)?;
            let k = qkv.narrow(D::Minus1, N_EMBD, N_EMBD)?;
            let v = qkv.narrow(D::Minus1, 2 * N_EMBD, N_EMBD)?;
            (q, k, v)
        } else {
            let qk_w = attn.qk_weight.as_ref().unwrap();
            let qk_b = attn.qk_bias.as_ref().unwrap();
            let normed_2d = normed.reshape((b * t, N_EMBD))?;
            let qk_2d = normed_2d.matmul(&qk_w.t()?)?.broadcast_add(qk_b)?;
            let qk = qk_2d.reshape((b, t, 2 * N_EMBD))?;
            let q = qk.narrow(D::Minus1, 0, N_EMBD)?;
            let k = qk.narrow(D::Minus1, N_EMBD, N_EMBD)?;
            let v = attn.c_v.as_ref().unwrap().forward(&normed)?;
            (q, k, v)
        };

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
        let att = (q.matmul(&k_t)? * scale)?;
        let mask_slice = attn.mask.i((.., .., ..t, ..t))?.broadcast_as(att.shape())?;
        let att = (att + mask_slice)?;
        let att_weights = ops::softmax(&att, D::Minus1)?;

        // Entropy: H = -sum(p * log(p)), add epsilon for numerical stability
        let log_att = (att_weights.clone() + 1e-10)?.log()?;
        let neg_plogp = (att_weights.clone() * log_att)?.neg()?; // (b, n_head, t, t)
        // Sum over attended positions, average over query tokens and batch
        let token_entropy = neg_plogp.sum(D::Minus1)?; // (b, n_head, t)
        let head_entropy = token_entropy.mean(D::Minus1)?.mean(0)?; // (n_head,)
        let entropies: Vec<f32> = head_entropy.to_vec1()?;
        all_entropies.push(entropies);

        // Continue forward pass for next layer
        let y = att_weights.matmul(&v)?;
        let y = y.transpose(1, 2)?.contiguous()?.reshape((b, t, N_EMBD))?;
        let attn_out = attn.c_proj.forward(&y)?;
        hidden = (hidden + attn_out)?;
        let normed2 = block.ln_2.forward(&hidden)?;
        let mlp_out = block.mlp.forward(&normed2)?;
        hidden = (hidden + mlp_out)?;
    }

    Ok(all_entropies)
}

// =============================================================================
// Binary Tensor I/O
// =============================================================================

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

fn dump_weights(
    varmap: &VarMap,
    mode: &str,
    vocab_size: usize,
    device: &Device,
) -> Result<()> {
    let dir = format!("weights/{mode}");
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

    let harmonic = build_harmonic_table(vocab_size, N_EMBD, device)?;
    let h_vals: Vec<f32> = harmonic.flatten_all()?.to_vec1()?;
    write_tensor_binary(
        &format!("{dir}/_harmonic_table.bin"),
        &[vocab_size, N_EMBD],
        &h_vals,
    );

    let positional = build_positional_table(BLOCK_SIZE, N_EMBD, device)?;
    let p_vals: Vec<f32> = positional.flatten_all()?.to_vec1()?;
    write_tensor_binary(
        &format!("{dir}/_positional_table.bin"),
        &[BLOCK_SIZE, N_EMBD],
        &p_vals,
    );

    println!("  Dumped {count} weight tensors + reference tables to {dir}/");
    Ok(())
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

fn estimate_loss(
    model: &HarmonicGPT,
    dataset: &Dataset,
    device: &Device,
) -> Result<(f32, f32)> {
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
}

fn train_mode(
    mode_name: &str,
    embed_mode: &str,
    attn_mode: &str,
    dataset: &Dataset,
    device: &Device,
) -> Result<TrainResult> {
    println!("\n{}", "=".repeat(60));
    println!("  Training: {}", mode_name.to_uppercase());
    println!(
        "  Embeddings: {embed_mode} | Attention: {attn_mode}"
    );
    println!("{}", "=".repeat(60));

    let varmap = VarMap::new();
    let model = HarmonicGPT::new(dataset.vocab_size, embed_mode, attn_mode, &varmap, device)?;

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
    dump_weights(&varmap, mode_name, dataset.vocab_size, device)?;

    Ok(TrainResult {
        mode_name: mode_name.to_string(),
        history,
        entropy,
    })
}

// =============================================================================
// Main
// =============================================================================

fn main() -> Result<()> {
    let device = Device::cuda_if_available(0)?;
    let device_name = if device.is_cuda() { "CUDA" } else { "CPU" };

    println!("{}", "=".repeat(60));
    println!("  Phase 18: Harmonic Attention Heads");
    println!("  Does structuring attention by harmonic order help?");
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
        "  Model: {} layers, {} heads, {} dim",
        N_LAYER, N_HEAD, N_EMBD
    );
    println!("  Context: {} characters", BLOCK_SIZE);
    println!(
        "  Harmonic orders per head: {:?}",
        HARMONIC_ORDERS
    );

    // Phase 17 reference results (not re-run, just printed)
    println!("\n  Phase 17 baselines (not re-run):");
    println!("    Baseline (random embed, standard attn):  val 3.1684");
    println!("    Harmonic (trainable embed, standard):    val 3.0899");
    println!("    Frozen   (frozen embed, standard attn):  val 3.0793");

    // Three new modes to test
    let modes: Vec<(&str, &str, &str)> = vec![
        // (mode_name, embed_mode, attn_mode)
        ("frozen_standard", "frozen", "standard"),           // Mode 2 rerun
        ("harmonic_heads", "frozen", "harmonic_warmstart"),  // Mode 3
        ("frozen_heads", "frozen", "frozen_qk"),             // Mode 4
    ];

    let mut results: Vec<TrainResult> = Vec::new();

    for (mode_name, embed_mode, attn_mode) in &modes {
        let result = train_mode(mode_name, embed_mode, attn_mode, &dataset, &device)?;
        results.push(result);
    }

    // =========================================================================
    // Comparison
    // =========================================================================
    println!("\n{}", "=".repeat(60));
    println!("  COMPARISON: Final Validation Loss");
    println!("{}", "=".repeat(60));
    println!();
    println!(
        "  {:<20} {:>10} {:>12} {:>10}",
        "Mode", "Val Loss", "Train Loss", "vs Frozen"
    );
    println!(
        "  {:<20} {:>10} {:>12} {:>10}",
        "-".repeat(20),
        "-".repeat(10),
        "-".repeat(12),
        "-".repeat(10)
    );

    // Phase 17 references
    println!(
        "  {:<20} {:>10.4} {:>12} {:>10}",
        "baseline (P17)", 3.1684, "—", "—"
    );
    println!(
        "  {:<20} {:>10.4} {:>12} {:>10}",
        "harmonic (P17)", 3.0899, "—", "—"
    );

    let frozen_val = results[0].history.last().unwrap().2;

    for result in &results {
        let (_, train_l, val_l) = result.history.last().unwrap();
        let diff = if frozen_val > 0.0 {
            format!("{:+.1}%", (1.0 - val_l / frozen_val) * 100.0)
        } else {
            "—".to_string()
        };
        println!(
            "  {:<20} {:>10.4} {:>12.4} {:>10}",
            result.mode_name, val_l, train_l, diff
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

    let harmonic_val = results[1].history.last().unwrap().2;
    let frozen_heads_val = results[2].history.last().unwrap().2;

    println!();
    if harmonic_val < frozen_val {
        let pct = (1.0 - harmonic_val / frozen_val) * 100.0;
        println!(
            "  Harmonic heads OUTPERFORM standard attention by {:.1}% on val loss.",
            pct
        );
        println!("  Harmonic-initialized Q/K provides better starting point for attention.");
    } else {
        let pct = (harmonic_val / frozen_val - 1.0) * 100.0;
        println!(
            "  Harmonic heads underperform standard attention by {:.1}% on val loss.",
            pct
        );
        println!("  Q/K warm-starting does not help — model prefers unconstrained attention.");
    }

    println!();
    if frozen_heads_val < frozen_val * 1.1 {
        let pct = ((frozen_heads_val / frozen_val) - 1.0) * 100.0;
        println!(
            "  Frozen harmonic heads within {:.1}% of standard attention.",
            pct.abs()
        );
        println!("  Learned relationship detection may not be necessary —");
        println!("  harmonic structure alone can drive useful attention patterns.");
    } else {
        let pct = (frozen_heads_val / frozen_val - 1.0) * 100.0;
        println!(
            "  Frozen harmonic heads {:.1}% worse than standard attention.",
            pct
        );
        println!("  Learned Q/K projections are essential — fixed harmonic");
        println!("  attention is too constrained for this task.");
    }

    // Entropy comparison
    println!();
    let avg_entropy = |r: &TrainResult| -> f32 {
        let all: Vec<f32> = r.entropy.iter().flat_map(|v| v.iter().copied()).collect();
        all.iter().sum::<f32>() / all.len() as f32
    };

    let std_e = avg_entropy(&results[0]);
    let harm_e = avg_entropy(&results[1]);
    let froz_e = avg_entropy(&results[2]);

    println!(
        "  Average attention entropy: standard={:.2}, harmonic_heads={:.2}, frozen_heads={:.2}",
        std_e, harm_e, froz_e
    );
    if harm_e < std_e {
        println!("  Harmonic heads show LOWER entropy — more specialised attention patterns.");
    } else {
        println!("  Harmonic heads show similar or higher entropy — no additional specialisation.");
    }

    println!();
    println!("  Weight matrices dumped to weights/{{frozen_standard,harmonic_heads,frozen_heads}}/");
    println!("  Run 'cargo run --release --bin analyze' for spectral analysis.");
    println!();
    println!("{}", "=".repeat(60));

    Ok(())
}
