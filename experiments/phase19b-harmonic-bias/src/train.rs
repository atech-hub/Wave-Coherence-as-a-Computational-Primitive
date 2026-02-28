// Phase 19b: Harmonic Attention Bias
//
// Tests whether an additive harmonic interference bias improves or
// accelerates standard attention, WITHOUT constraining or removing Q/K.
//
// Key insight: Phases 18 and 19 both failed by removing the model's
// ability to learn attention patterns. This experiment PRESERVES full
// learned Q/K while adding a harmonic prior as a hint.
//
// Architecture:
//   Standard:  score = Q·K^T / sqrt(d)
//   Biased:    score = Q·K^T / sqrt(d) + λ_h * interference_h(i,j)
//
// λ_h is a learnable scalar per head per layer. If the model finds
// the harmonic bias useful, λ stays positive. If not, λ → 0 and the
// model ignores it. The bias is a free hint, not a constraint.
//
// The interference term for head h is the dot product of embedding
// sub-vectors for that head's frequency band — same as Phase 19,
// but ADDED to learned scores instead of replacing them.
//
// Modes:
//   frozen_standard — frozen harmonic embeddings, standard attention (baseline)
//   harmonic_bias   — standard attention + additive harmonic bias (λ per head)
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
    c_attn: Linear,
    c_proj: Linear,
    // Harmonic bias: learnable λ per head, shape (1, N_HEAD, 1, 1)
    // None for standard mode
    lambda: Option<Tensor>,
    mask: Tensor,
    n_head: usize,
    n_embd: usize,
}

impl CausalSelfAttention {
    fn new(
        vb: VarBuilder,
        device: &Device,
        harmonic_bias: bool,
    ) -> Result<Self> {
        let c_attn = linear(N_EMBD, 3 * N_EMBD, vb.pp("c_attn"))?;
        let c_proj = linear(N_EMBD, N_EMBD, vb.pp("c_proj"))?;
        let mask = build_causal_mask(BLOCK_SIZE, device)?;

        let lambda = if harmonic_bias {
            // Initialize λ at 0.1 — a small positive value so the bias
            // starts active. The model can learn to increase or decrease it.
            let lambda = vb.pp("lambda").get_with_hints(
                (1, N_HEAD, 1, 1),
                "weight",
                candle_nn::Init::Const(0.1),
            )?;
            Some(lambda)
        } else {
            None
        };

        Ok(Self {
            c_attn,
            c_proj,
            lambda,
            mask,
            n_head: N_HEAD,
            n_embd: N_EMBD,
        })
    }

    /// Forward pass.
    /// `x` — current hidden state
    /// `emb` — original harmonic embedding (for bias computation)
    fn forward(&self, x: &Tensor, emb: Option<&Tensor>) -> Result<Tensor> {
        let (b, t, c) = x.dims3()?;
        let head_dim = c / self.n_head;

        // Standard Q/K/V from learned projection — always
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

        // Standard attention scores
        let scale = 1.0 / (head_dim as f64).sqrt();
        let k_t = k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        let mut scores = (q.matmul(&k_t)? * scale)?;

        // Add harmonic interference bias if enabled
        if let (Some(lambda), Some(emb)) = (&self.lambda, emb) {
            // Partition embedding by frequency band per head
            let emb_heads = emb
                .reshape((b, t, self.n_head, head_dim))?
                .transpose(1, 2)?
                .contiguous()?;

            // Interference: dot product of embedding sub-vectors
            let emb_t = emb_heads
                .transpose(D::Minus2, D::Minus1)?
                .contiguous()?;
            let interference = (emb_heads.matmul(&emb_t)? * scale)?;

            // score = Q·K^T/sqrt(d) + λ * interference
            // broadcast_mul handles (1, N_HEAD, 1, 1) * (b, N_HEAD, t, t) with gradient flow
            scores = (scores + interference.broadcast_mul(lambda)?)?;
        }

        let mask = self.mask.i((.., .., ..t, ..t))?.broadcast_as(scores.shape())?;
        let att = (scores + mask)?;
        let att = ops::softmax(&att, D::Minus1)?;
        let y = att.matmul(&v)?;
        let y = y.transpose(1, 2)?.contiguous()?.reshape((b, t, c))?;
        self.c_proj.forward(&y)
    }

    /// Get current λ values as a Vec<f32> (one per head).
    fn get_lambda_values(&self) -> Option<Vec<f32>> {
        self.lambda.as_ref().map(|l| {
            l.flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        })
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
    fn new(
        vb: VarBuilder,
        device: &Device,
        harmonic_bias: bool,
    ) -> Result<Self> {
        let ln_1 = layer_norm(N_EMBD, candle_nn::LayerNormConfig::default(), vb.pp("ln_1"))?;
        let attn = CausalSelfAttention::new(vb.pp("attn"), device, harmonic_bias)?;
        let ln_2 = layer_norm(N_EMBD, candle_nn::LayerNormConfig::default(), vb.pp("ln_2"))?;
        let mlp = MLP::new(vb.pp("mlp"))?;
        Ok(Self {
            ln_1,
            attn,
            ln_2,
            mlp,
        })
    }

    fn forward(&self, x: &Tensor, emb: Option<&Tensor>) -> Result<Tensor> {
        let x = (x + self.attn.forward(&self.ln_1.forward(x)?, emb)?)?;
        let x = (&x + self.mlp.forward(&self.ln_2.forward(&x)?)?)?;
        Ok(x)
    }
}

// =============================================================================
// The Model
// =============================================================================

struct BiasedGPT {
    wte: Tensor,
    wpe: Tensor,
    blocks: Vec<Block>,
    ln_f: LayerNorm,
    lm_head: Linear,
    harmonic_bias: bool,
}

impl BiasedGPT {
    fn new(
        vocab_size: usize,
        harmonic_bias: bool,
        varmap: &VarMap,
        device: &Device,
    ) -> Result<Self> {
        let vb = VarBuilder::from_varmap(varmap, DType::F32, device);

        // Always frozen harmonic embeddings
        let wte = build_harmonic_table(vocab_size, N_EMBD, device)?;
        let wpe = build_positional_table(BLOCK_SIZE, N_EMBD, device)?;

        let mut blocks = Vec::new();
        for i in 0..N_LAYER {
            blocks.push(Block::new(
                vb.pp(format!("blocks.{i}")),
                device,
                harmonic_bias,
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
        let mode_str = if harmonic_bias { "harmonic_bias" } else { "standard" };
        println!("  {mode_str} model: {n_params} trainable parameters");
        if harmonic_bias {
            println!("  (includes {} lambda parameters: {} per head x {} layers)",
                N_HEAD * N_LAYER, N_HEAD, N_LAYER);
        }

        Ok(Self {
            wte,
            wpe,
            blocks,
            ln_f,
            lm_head,
            harmonic_bias,
        })
    }

    fn forward(&self, idx: &Tensor, targets: Option<&Tensor>) -> Result<(Tensor, Option<Tensor>)> {
        let (b, t) = idx.dims2()?;

        let idx_flat = idx.flatten_all()?;
        let tok_emb = self.wte.index_select(&idx_flat, 0)?;
        let tok_emb = tok_emb.reshape((b, t, N_EMBD))?;
        let pos_emb = self.wpe.i(0..t)?;
        let emb = tok_emb.broadcast_add(&pos_emb)?;

        let emb_ref = if self.harmonic_bias { Some(&emb) } else { None };

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

    /// Report λ values across all layers.
    fn report_lambda(&self) {
        if !self.harmonic_bias {
            return;
        }
        println!("\n  Lambda values (learned harmonic bias strength):");
        for (i, block) in self.blocks.iter().enumerate() {
            if let Some(lambdas) = block.attn.get_lambda_values() {
                print!("    layer {i}:");
                for (h, &l) in lambdas.iter().enumerate() {
                    print!("  h{h}={l:+.6}");
                }
                let avg: f32 = lambdas.iter().sum::<f32>() / lambdas.len() as f32;
                println!("  avg={avg:+.6}");
            }
        }
    }
}

// =============================================================================
// Attention Entropy Measurement
// =============================================================================

fn measure_attention_entropy(
    model: &BiasedGPT,
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

        // Compute attention scores (both modes use c_attn)
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
        let mut scores = (q.matmul(&k_t)? * scale)?;

        // Add harmonic bias if present
        if let Some(ref lambda) = attn.lambda {
            let emb_heads = emb
                .reshape((b, t, N_HEAD, head_dim))?
                .transpose(1, 2)?
                .contiguous()?;
            let emb_t = emb_heads
                .transpose(D::Minus2, D::Minus1)?
                .contiguous()?;
            let interference = (emb_heads.matmul(&emb_t)? * scale)?;
            scores = (scores + interference.broadcast_mul(lambda)?)?;
        }

        let mask_slice = attn.mask.i((.., .., ..t, ..t))?.broadcast_as(scores.shape())?;
        let scores = (scores + mask_slice)?;
        let att_weights = ops::softmax(&scores, D::Minus1)?;

        // Entropy
        let log_att = (att_weights.clone() + 1e-10)?.log()?;
        let neg_plogp = (att_weights.clone() * log_att)?.neg()?;
        let token_entropy = neg_plogp.sum(D::Minus1)?;
        let head_entropy = token_entropy.mean(D::Minus1)?.mean(0)?;
        let entropies: Vec<f32> = head_entropy.to_vec1()?;
        all_entropies.push(entropies);

        // Continue forward pass
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
    model: &BiasedGPT,
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
    final_lambdas: Option<Vec<Vec<f32>>>, // [layer][head]
}

fn train_mode(
    mode_name: &str,
    harmonic_bias: bool,
    dataset: &Dataset,
    device: &Device,
) -> Result<TrainResult> {
    println!("\n{}", "=".repeat(60));
    println!("  Training: {}", mode_name.to_uppercase());
    println!(
        "  Attention: {}",
        if harmonic_bias {
            "standard + harmonic bias (λ per head)"
        } else {
            "standard (learned Q/K/V)"
        }
    );
    println!("{}", "=".repeat(60));

    let varmap = VarMap::new();
    let model = BiasedGPT::new(dataset.vocab_size, harmonic_bias, &varmap, device)?;

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

            // Report λ at each eval point
            if harmonic_bias {
                model.report_lambda();
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

    // Capture final lambda values
    let final_lambdas = if harmonic_bias {
        let lambdas: Vec<Vec<f32>> = model
            .blocks
            .iter()
            .filter_map(|b| b.attn.get_lambda_values())
            .collect();
        Some(lambdas)
    } else {
        None
    };

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
        final_lambdas,
    })
}

// =============================================================================
// Main
// =============================================================================

fn main() -> Result<()> {
    let device = Device::cuda_if_available(0)?;
    let device_name = if device.is_cuda() { "CUDA" } else { "CPU" };

    println!("{}", "=".repeat(60));
    println!("  Phase 19b: Harmonic Attention Bias");
    println!("  Does a harmonic prior improve standard attention?");
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

    println!("\n  Harmonic bias architecture:");
    println!("    score = Q·K^T/sqrt(d) + lambda * dot(emb_h, emb_h^T)/sqrt(d)");
    println!("    lambda: learnable scalar per head per layer (init=0.1)");
    println!("    If lambda -> 0: model ignores harmonic prior");
    println!("    If lambda stays +: harmonic prior provides useful information");

    println!("\n  Reference results:");
    println!("    Frozen standard:      val ~3.08-3.09");
    println!("    Spectral/Phase 18:    val ~3.25 (uniform attention)");

    // Two modes
    let result_standard = train_mode("frozen_standard", false, &dataset, &device)?;
    let result_biased = train_mode("harmonic_bias", true, &dataset, &device)?;

    let results = vec![&result_standard, &result_biased];

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
    println!("  {:>10}", "Bias gain");
    println!("  {}", "-".repeat(60));

    let std_hist = &result_standard.history;
    let bias_hist = &result_biased.history;
    for i in 0..std_hist.len() {
        let (step, _, std_v) = std_hist[i];
        let (_, _, bias_v) = bias_hist[i];
        let gain = (1.0 - bias_v / std_v) * 100.0;
        println!(
            "  {:>6}  {:>16.4}  {:>16.4}  {:>+9.1}%",
            step, std_v, bias_v, gain
        );
    }

    // =========================================================================
    // Lambda Evolution
    // =========================================================================
    if let Some(ref lambdas) = result_biased.final_lambdas {
        println!("\n{}", "=".repeat(60));
        println!("  LAMBDA VALUES (final, after 2000 iterations)");
        println!("  Positive = model uses harmonic prior");
        println!("  Zero/negative = model ignores/opposes harmonic prior");
        println!("{}", "=".repeat(60));
        println!();

        let mut all_lambdas: Vec<f32> = Vec::new();
        for (layer, layer_lambdas) in lambdas.iter().enumerate() {
            print!("  layer {layer}:");
            for (h, &l) in layer_lambdas.iter().enumerate() {
                print!("  h{h}={l:+.6}");
                all_lambdas.push(l);
            }
            println!();
        }

        let avg_lambda: f32 = all_lambdas.iter().sum::<f32>() / all_lambdas.len() as f32;
        let positive_count = all_lambdas.iter().filter(|&&l| l > 0.01).count();
        let negative_count = all_lambdas.iter().filter(|&&l| l < -0.01).count();
        let near_zero = all_lambdas.len() - positive_count - negative_count;

        println!();
        println!("  Average lambda: {avg_lambda:+.4}");
        println!(
            "  Positive (>0.01): {}, Near zero: {}, Negative (<-0.01): {}",
            positive_count, near_zero, negative_count
        );

        if positive_count > all_lambdas.len() / 2 {
            println!("  Majority of heads retain the harmonic prior — it provides useful signal.");
        } else if near_zero > all_lambdas.len() / 2 {
            println!("  Majority of heads drive lambda near zero — harmonic prior is redundant.");
        } else if negative_count > 0 {
            println!("  Some heads actively oppose the harmonic prior — it interferes with learning.");
        }
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

    let biased_val = result_biased.history.last().unwrap().2;
    let improvement = (1.0 - biased_val / std_val) * 100.0;

    println!();
    if biased_val < std_val {
        println!(
            "  Harmonic bias IMPROVES standard attention by {:.2}% on val loss.",
            improvement
        );
        println!("  The harmonic prior provides information that helps the model.");
    } else if (biased_val - std_val).abs() / std_val < 0.005 {
        println!(
            "  Harmonic bias MATCHES standard attention ({:.2}% difference).",
            improvement.abs()
        );
        println!("  The prior neither helps nor hurts — model learns to ignore it.");
    } else {
        println!(
            "  Harmonic bias {:.2}% worse than standard attention.",
            improvement.abs()
        );
    }

    // Check early convergence
    if result_standard.history.len() > 2 && result_biased.history.len() > 2 {
        let std_early = result_standard.history[1].2;
        let bias_early = result_biased.history[1].2;
        let early_gain = (1.0 - bias_early / std_early) * 100.0;

        println!();
        if early_gain > 0.5 {
            println!(
                "  Early convergence: harmonic bias {:.1}% ahead at step {}.",
                early_gain, result_standard.history[1].0
            );
            println!("  The harmonic prior accelerates early training.");
        } else if early_gain < -0.5 {
            println!(
                "  Early convergence: harmonic bias {:.1}% behind at step {}.",
                early_gain.abs(), result_standard.history[1].0
            );
        } else {
            println!(
                "  Early convergence: identical at step {} ({:.1}% difference).",
                result_standard.history[1].0, early_gain.abs()
            );
        }
    }

    println!();
    println!("{}", "=".repeat(60));

    Ok(())
}
