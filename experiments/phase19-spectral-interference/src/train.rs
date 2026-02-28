// Phase 19: Spectral Interference Attention
//
// Tests whether harmonic embedding interference can REPLACE learned Q/K
// projections entirely. Instead of learning what to attend to (Q/K),
// use the harmonic embedding's dot-product structure directly as
// attention scores. Each head sees a different frequency band.
//
// Key insight from Phase 18: constraining Q/K weights to harmonic structure
// produced uniform attention and 5.2% worse performance. But that was
// trying to push harmonic structure THROUGH learned projections.
// This experiment skips projections entirely — the embedding IS the
// query and key.
//
// Architecture:
//   Standard (baseline): x → Linear(Q,K,V) → dot-product attention → output
//   Spectral:            emb → partition by freq band → interference → attention
//                        x → Linear(V) → values extracted by interference pattern
//
// The attention pattern is fixed by the harmonic geometry at every layer.
// Only V projections and MLP layers learn. "The geometry tells you who
// to listen to; learning tells you what to hear."
//
// Modes:
//   frozen_standard — frozen harmonic embeddings, standard learned attention
//   spectral        — frozen embeddings, spectral interference attention
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
// Model Components
// =============================================================================

struct CausalSelfAttention {
    // Standard mode: single Linear for Q+K+V
    c_attn: Option<Linear>,
    // Spectral mode: only V is learned (no Q/K at all)
    c_v: Option<Linear>,
    // Common
    c_proj: Linear,
    mask: Tensor,
    n_head: usize,
    n_embd: usize,
}

impl CausalSelfAttention {
    fn new(vb: VarBuilder, device: &Device, spectral: bool) -> Result<Self> {
        let c_proj = linear(N_EMBD, N_EMBD, vb.pp("c_proj"))?;
        let mask = build_causal_mask(BLOCK_SIZE, device)?;

        if spectral {
            // No Q/K parameters at all — attention from embedding interference
            let c_v = linear(N_EMBD, N_EMBD, vb.pp("c_v"))?;

            Ok(Self {
                c_attn: None,
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
                c_v: None,
                c_proj,
                mask,
                n_head: N_HEAD,
                n_embd: N_EMBD,
            })
        }
    }

    /// Forward pass.
    /// `x` — current hidden state (used for V in spectral mode, or Q/K/V in standard)
    /// `emb` — original harmonic embedding (used for attention scores in spectral mode)
    fn forward(&self, x: &Tensor, emb: Option<&Tensor>) -> Result<Tensor> {
        let (b, t, c) = x.dims3()?;
        let head_dim = c / self.n_head;

        let (scores, v) = if let Some(ref c_attn) = self.c_attn {
            // Standard mode — Q, K, V all from x
            let qkv = c_attn.forward(x)?;
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
            let s = (q.matmul(&k_t)? * scale)?;

            (s, v)
        } else {
            // Spectral mode — attention from embedding interference
            // emb is the original harmonic embedding: (b, t, n_embd)
            // Partition into heads by frequency band:
            //   Head 0: dims 0..31   (harmonics 1-16, low frequency)
            //   Head 1: dims 32..63  (harmonics 17-32)
            //   Head 2: dims 64..95  (harmonics 33-48)
            //   Head 3: dims 96..127 (harmonics 49-64, high frequency)
            let emb = emb.unwrap();
            let emb_heads = emb
                .reshape((b, t, self.n_head, head_dim))?
                .transpose(1, 2)?
                .contiguous()?;

            let scale = 1.0 / (head_dim as f64).sqrt();
            let emb_t = emb_heads
                .transpose(D::Minus2, D::Minus1)?
                .contiguous()?;
            let s = (emb_heads.matmul(&emb_t)? * scale)?;

            // V from current hidden state (the only learned part)
            let v = self.c_v.as_ref().unwrap().forward(x)?;
            let v = v
                .reshape((b, t, self.n_head, head_dim))?
                .transpose(1, 2)?
                .contiguous()?;

            (s, v)
        };

        let mask = self.mask.i((.., .., ..t, ..t))?.broadcast_as(scores.shape())?;
        let att = (scores + mask)?;
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
    fn new(vb: VarBuilder, device: &Device, spectral: bool) -> Result<Self> {
        let ln_1 = layer_norm(N_EMBD, candle_nn::LayerNormConfig::default(), vb.pp("ln_1"))?;
        let attn = CausalSelfAttention::new(vb.pp("attn"), device, spectral)?;
        let ln_2 = layer_norm(N_EMBD, candle_nn::LayerNormConfig::default(), vb.pp("ln_2"))?;
        let mlp = MLP::new(vb.pp("mlp"))?;
        Ok(Self {
            ln_1,
            attn,
            ln_2,
            mlp,
        })
    }

    /// Forward pass. `emb` is passed through for spectral attention.
    fn forward(&self, x: &Tensor, emb: Option<&Tensor>) -> Result<Tensor> {
        let x = (x + self.attn.forward(&self.ln_1.forward(x)?, emb)?)?;
        let x = (&x + self.mlp.forward(&self.ln_2.forward(&x)?)?)?;
        Ok(x)
    }
}

// =============================================================================
// The Model
// =============================================================================

struct SpectralGPT {
    wte: Tensor,
    wpe: Tensor,
    blocks: Vec<Block>,
    ln_f: LayerNorm,
    lm_head: Linear,
    spectral: bool,
}

impl SpectralGPT {
    fn new(
        vocab_size: usize,
        spectral: bool,
        varmap: &VarMap,
        device: &Device,
    ) -> Result<Self> {
        let vb = VarBuilder::from_varmap(varmap, DType::F32, device);

        // Always frozen harmonic embeddings (the established best from Phase 17)
        let wte = build_harmonic_table(vocab_size, N_EMBD, device)?;
        let wpe = build_positional_table(BLOCK_SIZE, N_EMBD, device)?;

        let mut blocks = Vec::new();
        for i in 0..N_LAYER {
            blocks.push(Block::new(
                vb.pp(format!("blocks.{i}")),
                device,
                spectral,
            )?);
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
        let mode_str = if spectral { "spectral" } else { "standard" };
        println!("  {mode_str} model: {n_params} trainable parameters");

        Ok(Self {
            wte,
            wpe,
            blocks,
            ln_f,
            lm_head,
            spectral,
        })
    }

    fn forward(&self, idx: &Tensor, targets: Option<&Tensor>) -> Result<(Tensor, Option<Tensor>)> {
        let (b, t) = idx.dims2()?;

        let idx_flat = idx.flatten_all()?;
        let tok_emb = self.wte.index_select(&idx_flat, 0)?;
        let tok_emb = tok_emb.reshape((b, t, N_EMBD))?;
        let pos_emb = self.wpe.i(0..t)?;
        let emb = tok_emb.broadcast_add(&pos_emb)?;

        // In spectral mode, pass the original embedding to every layer
        // for attention computation. The hidden state evolves through layers
        // but attention patterns are fixed by the harmonic geometry.
        let emb_ref = if self.spectral { Some(&emb) } else { None };

        let mut x = emb.clone();
        for block in &self.blocks {
            x = block.forward(&x, emb_ref)?;
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
    model: &SpectralGPT,
    dataset: &Dataset,
    device: &Device,
) -> Result<Vec<Vec<f32>>> {
    let (x, _) = dataset.get_batch("val", device)?;
    let (b, t) = x.dims2()?;
    let head_dim = N_EMBD / N_HEAD;

    let idx_flat = x.flatten_all()?;
    let tok_emb = model.wte.index_select(&idx_flat, 0)?.reshape((b, t, N_EMBD))?;
    let pos_emb = model.wpe.i(0..t)?;
    let emb = tok_emb.broadcast_add(&pos_emb)?;
    let mut hidden = emb.clone();

    let mut all_entropies = Vec::new();

    for block in &model.blocks {
        let normed = block.ln_1.forward(&hidden)?;
        let attn = &block.attn;

        // Compute attention scores
        let scores = if let Some(ref c_attn) = attn.c_attn {
            // Standard mode — scores from learned Q/K
            let qkv = c_attn.forward(&normed)?;
            let q = qkv.narrow(D::Minus1, 0, N_EMBD)?;
            let k = qkv.narrow(D::Minus1, N_EMBD, N_EMBD)?;

            let q = q
                .reshape((b, t, N_HEAD, head_dim))?
                .transpose(1, 2)?
                .contiguous()?;
            let k = k
                .reshape((b, t, N_HEAD, head_dim))?
                .transpose(1, 2)?
                .contiguous()?;

            let scale = 1.0 / (head_dim as f64).sqrt();
            let k_t = k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
            (q.matmul(&k_t)? * scale)?
        } else {
            // Spectral mode — scores from embedding interference
            let emb_heads = emb
                .reshape((b, t, N_HEAD, head_dim))?
                .transpose(1, 2)?
                .contiguous()?;

            let scale = 1.0 / (head_dim as f64).sqrt();
            let emb_t = emb_heads
                .transpose(D::Minus2, D::Minus1)?
                .contiguous()?;
            (emb_heads.matmul(&emb_t)? * scale)?
        };

        let mask_slice = attn.mask.i((.., .., ..t, ..t))?.broadcast_as(scores.shape())?;
        let scores = (scores + mask_slice)?;
        let att_weights = ops::softmax(&scores, D::Minus1)?;

        // Entropy: H = -sum(p * log(p)), add epsilon for numerical stability
        let log_att = (att_weights.clone() + 1e-10)?.log()?;
        let neg_plogp = (att_weights.clone() * log_att)?.neg()?;
        let token_entropy = neg_plogp.sum(D::Minus1)?;
        let head_entropy = token_entropy.mean(D::Minus1)?.mean(0)?;
        let entropies: Vec<f32> = head_entropy.to_vec1()?;
        all_entropies.push(entropies);

        // Continue forward pass for next layer
        let v = if let Some(ref c_attn) = attn.c_attn {
            let qkv = c_attn.forward(&normed)?;
            qkv.narrow(D::Minus1, 2 * N_EMBD, N_EMBD)?
                .reshape((b, t, N_HEAD, head_dim))?
                .transpose(1, 2)?
                .contiguous()?
        } else {
            attn.c_v.as_ref().unwrap().forward(&normed)?
                .reshape((b, t, N_HEAD, head_dim))?
                .transpose(1, 2)?
                .contiguous()?
        };

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
    model: &SpectralGPT,
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
    n_params: usize,
}

fn train_mode(
    mode_name: &str,
    spectral: bool,
    dataset: &Dataset,
    device: &Device,
) -> Result<TrainResult> {
    println!("\n{}", "=".repeat(60));
    println!("  Training: {}", mode_name.to_uppercase());
    println!(
        "  Attention: {}",
        if spectral {
            "spectral interference (no Q/K)"
        } else {
            "standard (learned Q/K/V)"
        }
    );
    println!("{}", "=".repeat(60));

    let varmap = VarMap::new();
    let model = SpectralGPT::new(dataset.vocab_size, spectral, &varmap, device)?;

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
        n_params,
    })
}

// =============================================================================
// Main
// =============================================================================

fn main() -> Result<()> {
    let device = Device::cuda_if_available(0)?;
    let device_name = if device.is_cuda() { "CUDA" } else { "CPU" };

    println!("{}", "=".repeat(60));
    println!("  Phase 19: Spectral Interference Attention");
    println!("  Can embedding interference replace learned Q/K?");
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

    println!("\n  Spectral attention architecture:");
    println!("    Head 0: dims  0-31  (harmonics  1-16, low frequency)");
    println!("    Head 1: dims 32-63  (harmonics 17-32)");
    println!("    Head 2: dims 64-95  (harmonics 33-48)");
    println!("    Head 3: dims 96-127 (harmonics 49-64, high frequency)");
    println!("    Attention = dot(emb_band_i, emb_band_j) / sqrt(32)");
    println!("    V = learned, Q/K = the embedding itself");

    println!("\n  Phase 17/18 reference results:");
    println!("    Frozen standard:     val 3.0793-3.0912");
    println!("    Harmonic heads (P18): val 3.2511 (-5.2%, uniform attn)");

    // Two modes to test
    let result_standard = train_mode("frozen_standard", false, &dataset, &device)?;
    let result_spectral = train_mode("spectral", true, &dataset, &device)?;

    let results = vec![&result_standard, &result_spectral];

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
        } else if std_val > 0.0 {
            let pct = (1.0 - val_l / std_val) * 100.0;
            format!("{pct:+.1}%")
        } else {
            "—".to_string()
        };
        let param_diff = if result.n_params < result_standard.n_params {
            let saved = result_standard.n_params - result.n_params;
            format!("-{}K", saved / 1000)
        } else {
            format!("{}K", result.n_params / 1000)
        };
        println!(
            "  {:<20} {:>10.4} {:>12.4} {:>10} {:>8}",
            result.mode_name, val_l, train_l, diff, param_diff
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

    // Check if spectral heads show differentiation across heads
    let spectral_entropy = &result_spectral.entropy;
    let head_avgs: Vec<f32> = (0..N_HEAD)
        .map(|h| {
            spectral_entropy.iter().map(|layer| layer[h]).sum::<f32>() / N_LAYER as f32
        })
        .collect();
    let entropy_range = head_avgs.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        - head_avgs.iter().cloned().fold(f32::INFINITY, f32::min);

    println!("\n  Spectral head entropy range: {:.3}", entropy_range);
    if entropy_range > 0.1 {
        println!("    Different frequency bands produce DIFFERENT attention patterns.");
        println!("    Low-freq heads vs high-freq heads show genuine specialisation.");
    } else {
        println!("    All frequency bands produce similar attention patterns.");
    }

    // =========================================================================
    // Summary
    // =========================================================================
    println!("\n{}", "=".repeat(60));
    println!("  SUMMARY");
    println!("{}", "=".repeat(60));

    let spectral_val = result_spectral.history.last().unwrap().2;
    let param_savings = if result_standard.n_params > result_spectral.n_params {
        result_standard.n_params - result_spectral.n_params
    } else {
        0
    };

    println!();
    if spectral_val < std_val {
        let pct = (1.0 - spectral_val / std_val) * 100.0;
        println!(
            "  Spectral interference OUTPERFORMS standard attention by {:.1}%.",
            pct
        );
        println!(
            "  With {} fewer parameters (no Q/K weights).",
            param_savings
        );
        println!("  Harmonic embedding interference is a viable replacement for learned Q/K.");
    } else if spectral_val < std_val * 1.05 {
        let pct = (spectral_val / std_val - 1.0) * 100.0;
        println!(
            "  Spectral interference within {:.1}% of standard attention.",
            pct
        );
        println!(
            "  With {} fewer parameters (no Q/K weights).",
            param_savings
        );
        println!("  Close enough to suggest harmonic interference has potential with refinement.");
    } else {
        let pct = (spectral_val / std_val - 1.0) * 100.0;
        println!(
            "  Spectral interference {:.1}% worse than standard attention.",
            pct
        );
        println!("  Removing Q/K projections costs too much expressiveness.");
    }

    // Compare with Phase 18
    println!();
    println!("  vs Phase 18 (harmonic Q/K weights, same parameter count):");
    if spectral_val < 3.2511 {
        let pct = (1.0 - spectral_val / 3.2511) * 100.0;
        println!(
            "    Spectral interference {:.1}% BETTER than constrained Q/K.",
            pct
        );
        println!("    Skipping Q/K entirely works better than constraining Q/K.");
    } else {
        let pct = (spectral_val / 3.2511 - 1.0) * 100.0;
        println!(
            "    Spectral interference {:.1}% worse than constrained Q/K.",
            pct
        );
    }

    // Entropy comparison
    println!();
    let avg_entropy = |r: &TrainResult| -> f32 {
        let all: Vec<f32> = r.entropy.iter().flat_map(|v| v.iter().copied()).collect();
        all.iter().sum::<f32>() / all.len() as f32
    };

    let std_e = avg_entropy(&result_standard);
    let spec_e = avg_entropy(&result_spectral);

    println!(
        "  Average attention entropy: standard={:.2}, spectral={:.2}",
        std_e, spec_e
    );
    if spec_e < 4.0 {
        println!("  Spectral heads show non-trivial attention patterns (not uniform).");
    }
    if entropy_range > 0.1 {
        println!("  Different frequency bands attend differently — band-specific structure exists.");
    }

    println!();
    println!("  Weight matrices dumped to weights/{{frozen_standard,spectral}}/");
    println!("  Run 'cargo run --release --bin analyze' for spectral analysis.");
    println!();
    println!("{}", "=".repeat(60));

    Ok(())
}
