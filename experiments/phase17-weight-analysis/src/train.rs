// Phase 17: Weight Spectral Analysis — GPU Training + Weight Dump
//
// Trains a character-level harmonic transformer in three modes (baseline,
// harmonic, frozen) and dumps all weight matrices to binary files for
// DFT spectral analysis.
//
// Based on rust-transformer/src/main.rs with these changes:
//   - CUDA support via feature flag (--features cuda)
//   - 2000 training iterations (full run)
//   - Batch size 64 (GPU-friendly)
//   - Weight serialization after each mode's training
//
// Usage:
//   cargo run --release --bin train                  # CPU
//   cargo run --release --bin train --features cuda  # GPU (recommended)

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
// Configuration — tuned for full training run
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

/// Build causal mask: 0.0 for allowed positions, -inf for blocked
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
    fn new(vb: VarBuilder, device: &Device) -> Result<Self> {
        let ln_1 = layer_norm(N_EMBD, candle_nn::LayerNormConfig::default(), vb.pp("ln_1"))?;
        let attn = CausalSelfAttention::new(vb.pp("attn"), device)?;
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
        mode: &str,
        varmap: &VarMap,
        device: &Device,
    ) -> Result<Self> {
        let vb = VarBuilder::from_varmap(varmap, DType::F32, device);

        let wte = match mode {
            "baseline" => {
                vb.pp("wte").get_with_hints(
                    (vocab_size, N_EMBD),
                    "weight",
                    candle_nn::Init::Randn {
                        mean: 0.0,
                        stdev: 0.02,
                    },
                )?
            }
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
            "frozen" => {
                build_harmonic_table(vocab_size, N_EMBD, device)?
            }
            _ => panic!("Unknown mode: {mode}"),
        };

        let wpe = match mode {
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

        let mut blocks = Vec::new();
        for i in 0..N_LAYER {
            blocks.push(Block::new(vb.pp(format!("blocks.{i}")), device)?);
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
        println!("  {mode} model: {n_params} trainable parameters");

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
// Binary Tensor I/O
// =============================================================================

/// Write a tensor in binary format: [ndims: u32] [dim...: u32] [values...: f32]
/// All values little-endian.
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

/// Dump all trained weights from a VarMap to binary files.
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

    // Dump deterministic reference tables (for frozen mode these aren't in VarMap)
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

    #[allow(dead_code)]
    fn decode(&self, tokens: &[u32]) -> String {
        tokens.iter().map(|&t| self.itos[t as usize]).collect()
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

fn train_model(
    mode: &str,
    dataset: &Dataset,
    device: &Device,
) -> Result<Vec<(usize, f32, f32)>> {
    println!("\n{}", "=".repeat(60));
    println!("  Training: {}", mode.to_uppercase());
    println!("{}", "=".repeat(60));

    let varmap = VarMap::new();
    let model = HarmonicGPT::new(dataset.vocab_size, mode, &varmap, device)?;

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

    // Dump trained weights for spectral analysis
    dump_weights(&varmap, mode, dataset.vocab_size, device)?;

    Ok(history)
}

// =============================================================================
// Main
// =============================================================================

fn main() -> Result<()> {
    let device = Device::cuda_if_available(0)?;
    let device_name = if device.is_cuda() { "CUDA" } else { "CPU" };

    println!("{}", "=".repeat(60));
    println!("  Phase 17: Weight Spectral Analysis — Training");
    println!("  Harmonic Transformer — 2000 iters, batch 64");
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

    let mut all_results = Vec::new();

    for mode in &["baseline", "harmonic", "frozen"] {
        let history = train_model(mode, &dataset, &device)?;
        all_results.push((mode.to_string(), history));
    }

    // =========================================================================
    // Comparison
    // =========================================================================
    println!("\n{}", "=".repeat(60));
    println!("  COMPARISON: Final Validation Loss");
    println!("{}", "=".repeat(60));
    println!();
    println!(
        "  {:<12} {:>10} {:>12}",
        "Mode", "Val Loss", "Train Loss"
    );
    println!(
        "  {:<12} {:>10} {:>12}",
        "-".repeat(12),
        "-".repeat(10),
        "-".repeat(12)
    );

    for (mode, history) in &all_results {
        let (_, train_l, val_l) = history.last().unwrap();
        println!("  {:<12} {:>10.4} {:>12.4}", mode, val_l, train_l);
    }

    let baseline_val = all_results[0].1.last().unwrap().2;
    let harmonic_val = all_results[1].1.last().unwrap().2;
    let frozen_val = all_results[2].1.last().unwrap().2;

    println!();
    if harmonic_val < baseline_val {
        let pct = (1.0 - harmonic_val / baseline_val) * 100.0;
        println!(
            "  Harmonic embeddings OUTPERFORM baseline by {:.1}% on val loss.",
            pct
        );
    } else {
        println!("  Harmonic embeddings underperform baseline.");
    }

    if frozen_val < baseline_val * 1.1 {
        println!("  Frozen harmonic embeddings within 10% of baseline --");
        println!("  geometric structure alone carries most of the signal.");
    }

    println!();
    println!("  Weight matrices dumped to weights/{{baseline,harmonic,frozen}}/");
    println!("  Run 'cargo run --release --bin analyze' for spectral analysis.");
    println!();
    println!("{}", "=".repeat(60));

    Ok(())
}
