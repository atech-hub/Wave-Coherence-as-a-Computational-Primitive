// Pure Rust Transformer — Full Training Pipeline, Zero Dependencies
//
// Trains a character-level transformer on Shakespeare entirely on CPU.
// No crates. Manual backpropagation. Parallel matmul via std::thread::scope.
//
// Three modes:
//   1. baseline  — random Gaussian embeddings, trainable (weight-tied)
//   2. harmonic  — harmonic phase embeddings, trainable
//   3. frozen    — harmonic phase embeddings, NOT trainable
//
// Architecture: 4 layers, 4 heads, 128 dim, 256 context, ~842K params.

mod adam;
mod model;
mod rng;
mod tensor;

use adam::{AdamW, Param};
use model::{Transformer, TransformerGrad, N_EMBD, BLOCK_SIZE};
use rng::Rng;

// =============================================================================
// Training configuration
// =============================================================================

const BATCH_SIZE: usize = 32;
const MAX_ITERS: usize = 2000;
const EVAL_INTERVAL: usize = 250;
const EVAL_ITERS: usize = 50;
const GEN_CHARS: usize = 500;

// =============================================================================
// Dataset
// =============================================================================

struct Dataset {
    train: Vec<usize>,
    val: Vec<usize>,
    vocab_size: usize,
    itos: Vec<char>,
}

impl Dataset {
    fn new(text: &str) -> Self {
        let mut chars: Vec<char> = text.chars().collect::<std::collections::BTreeSet<_>>().into_iter().collect();
        chars.sort();
        let vocab_size = chars.len();
        let stoi: std::collections::HashMap<char, usize> = chars.iter().enumerate().map(|(i, &c)| (c, i)).collect();

        let data: Vec<usize> = text.chars().map(|c| stoi[&c]).collect();
        let n = (data.len() as f64 * 0.9) as usize;

        Dataset {
            train: data[..n].to_vec(),
            val: data[n..].to_vec(),
            vocab_size,
            itos: chars,
        }
    }

    fn decode(&self, tokens: &[usize]) -> String {
        tokens.iter().map(|&t| self.itos[t]).collect()
    }

    /// Sample a random batch. Returns (inputs, targets) each of shape
    /// [BATCH_SIZE][BLOCK_SIZE] as flat usize vectors.
    fn get_batch(&self, split: &str, rng: &mut Rng) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
        let data = match split {
            "train" => &self.train,
            "val" => &self.val,
            _ => panic!("Unknown split: {split}"),
        };
        let max_start = data.len() - BLOCK_SIZE - 1;

        let mut inputs = Vec::with_capacity(BATCH_SIZE);
        let mut targets = Vec::with_capacity(BATCH_SIZE);
        for _ in 0..BATCH_SIZE {
            let start = rng.usize(max_start);
            inputs.push(data[start..start + BLOCK_SIZE].to_vec());
            targets.push(data[start + 1..start + BLOCK_SIZE + 1].to_vec());
        }
        (inputs, targets)
    }
}

// =============================================================================
// Count optimizer parameters
// =============================================================================

fn count_optimizer_params(model: &Transformer) -> usize {
    let frozen = model.mode == "frozen";
    let mut count = 0;

    // Embeddings (2: wte, wpe)
    if !frozen { count += 2; }

    // Per block: ln1 gamma, ln1 beta, attn.c_attn weight, attn.c_attn bias,
    //            attn.c_proj weight, attn.c_proj bias,
    //            ln2 gamma, ln2 beta, mlp.c_fc weight, mlp.c_fc bias,
    //            mlp.c_proj weight, mlp.c_proj bias = 12 per block
    count += model.blocks.len() * 12;

    // ln_f: gamma, beta
    count += 2;

    // lm_head: weight (no bias)
    if !model.weight_tied { count += 1; }

    count
}

/// Collect mutable params + grads for optimizer step.
fn collect_params<'a>(
    model: &'a mut Transformer,
    grad: &'a TransformerGrad,
) -> Vec<Param<'a>> {
    let frozen = model.mode == "frozen";
    let weight_tied = model.weight_tied;
    let mut params = Vec::new();

    // Embeddings
    if !frozen {
        params.push(Param { value: &mut model.wte, grad: &grad.d_wte, weight_decay: false });
        params.push(Param { value: &mut model.wpe, grad: &grad.d_wpe, weight_decay: false });
    }

    // Blocks
    for (i, block) in model.blocks.iter_mut().enumerate() {
        let bg = &grad.block_grads[i];

        // LN1
        params.push(Param { value: &mut block.ln_1.gamma, grad: &bg.ln1_grad.d_gamma, weight_decay: false });
        params.push(Param { value: &mut block.ln_1.beta, grad: &bg.ln1_grad.d_beta, weight_decay: false });

        // Attention c_attn
        params.push(Param { value: &mut block.attn.c_attn.weight, grad: &bg.attn_grad.c_attn_grad.d_weight, weight_decay: true });
        params.push(Param {
            value: block.attn.c_attn.bias.as_mut().unwrap(),
            grad: bg.attn_grad.c_attn_grad.d_bias.as_ref().unwrap(),
            weight_decay: false,
        });

        // Attention c_proj
        params.push(Param { value: &mut block.attn.c_proj.weight, grad: &bg.attn_grad.c_proj_grad.d_weight, weight_decay: true });
        params.push(Param {
            value: block.attn.c_proj.bias.as_mut().unwrap(),
            grad: bg.attn_grad.c_proj_grad.d_bias.as_ref().unwrap(),
            weight_decay: false,
        });

        // LN2
        params.push(Param { value: &mut block.ln_2.gamma, grad: &bg.ln2_grad.d_gamma, weight_decay: false });
        params.push(Param { value: &mut block.ln_2.beta, grad: &bg.ln2_grad.d_beta, weight_decay: false });

        // MLP c_fc
        params.push(Param { value: &mut block.mlp.c_fc.weight, grad: &bg.mlp_grad.c_fc_grad.d_weight, weight_decay: true });
        params.push(Param {
            value: block.mlp.c_fc.bias.as_mut().unwrap(),
            grad: bg.mlp_grad.c_fc_grad.d_bias.as_ref().unwrap(),
            weight_decay: false,
        });

        // MLP c_proj
        params.push(Param { value: &mut block.mlp.c_proj.weight, grad: &bg.mlp_grad.c_proj_grad.d_weight, weight_decay: true });
        params.push(Param {
            value: block.mlp.c_proj.bias.as_mut().unwrap(),
            grad: bg.mlp_grad.c_proj_grad.d_bias.as_ref().unwrap(),
            weight_decay: false,
        });
    }

    // Final layer norm
    params.push(Param { value: &mut model.ln_f.gamma, grad: &grad.ln_f_grad.d_gamma, weight_decay: false });
    params.push(Param { value: &mut model.ln_f.beta, grad: &grad.ln_f_grad.d_beta, weight_decay: false });

    // lm_head (skip if weight-tied, since it shares wte)
    if !weight_tied {
        params.push(Param { value: &mut model.lm_head.weight, grad: &grad.lm_head_grad.d_weight, weight_decay: true });
    }

    params
}

// =============================================================================
// Training
// =============================================================================

fn estimate_loss(
    model: &Transformer,
    dataset: &Dataset,
    rng: &mut Rng,
) -> (f64, f64) {
    let mut train_loss = 0.0;
    let mut val_loss = 0.0;

    for _ in 0..EVAL_ITERS {
        let (inputs, targets) = dataset.get_batch("train", rng);
        let input_refs: Vec<&[usize]> = inputs.iter().map(|v| v.as_slice()).collect();
        let all_targets: Vec<usize> = targets.into_iter().flatten().collect();
        let (logits, _) = model.forward_batch(&input_refs);
        train_loss += logits.cross_entropy_loss(&all_targets);
    }

    for _ in 0..EVAL_ITERS {
        let (inputs, targets) = dataset.get_batch("val", rng);
        let input_refs: Vec<&[usize]> = inputs.iter().map(|v| v.as_slice()).collect();
        let all_targets: Vec<usize> = targets.into_iter().flatten().collect();
        let (logits, _) = model.forward_batch(&input_refs);
        val_loss += logits.cross_entropy_loss(&all_targets);
    }

    (train_loss / EVAL_ITERS as f64, val_loss / EVAL_ITERS as f64)
}

struct LossEntry {
    step: usize,
    train: f64,
    val: f64,
}

fn train_model(
    mode: &str,
    dataset: &Dataset,
    seed: u64,
) -> (Transformer, Vec<LossEntry>) {
    println!("\n{}", "=".repeat(60));
    println!("  Training: {}", mode.to_uppercase());
    println!("{}", "=".repeat(60));

    let mut rng = Rng::new(seed);
    let mut model = Transformer::new(dataset.vocab_size, mode, &mut rng);
    let n_params = model.param_count();
    println!("  {} model: {} trainable parameters", mode, n_params);

    let n_opt_params = count_optimizer_params(&model);
    let mut optimizer = AdamW::new(n_opt_params);

    let mut history = Vec::new();
    let start = std::time::Instant::now();

    for iter_num in 0..MAX_ITERS {
        // Evaluate periodically
        if iter_num % EVAL_INTERVAL == 0 || iter_num == MAX_ITERS - 1 {
            let (train_l, val_l) = estimate_loss(&model, dataset, &mut rng);
            let elapsed = start.elapsed().as_secs_f64();
            println!(
                "  step {:>5} | train loss {:.4} | val loss {:.4} | {:.1}s",
                iter_num, train_l, val_l, elapsed
            );
            history.push(LossEntry {
                step: iter_num,
                train: train_l,
                val: val_l,
            });
        }

        // Training step: batched forward + backward
        let (inputs, targets) = dataset.get_batch("train", &mut rng);
        let input_refs: Vec<&[usize]> = inputs.iter().map(|v| v.as_slice()).collect();
        let all_targets: Vec<usize> = targets.into_iter().flatten().collect();

        let (logits, cache) = model.forward_batch(&input_refs);
        let d_logits = logits.cross_entropy_backward(&all_targets);
        let mut grad = model.backward_batch(&d_logits, &cache);

        // For weight-tied baseline, merge lm_head grad into wte
        if model.weight_tied {
            grad.d_wte.add_inplace(&grad.lm_head_grad.d_weight.transpose());
        }

        // Optimizer step
        let mut params = collect_params(&mut model, &grad);
        optimizer.step(&mut params);

        // Sync weight tying: lm_head.weight = wte^T
        if model.weight_tied {
            model.lm_head.weight = model.wte.transpose();
        }
    }

    let total = start.elapsed().as_secs_f64();
    println!("  Training complete in {:.1}s", total);

    (model, history)
}

// =============================================================================
// Text generation
// =============================================================================

fn generate(model: &Transformer, seed_token: usize, max_chars: usize, rng: &mut Rng) -> Vec<usize> {
    let mut tokens = vec![seed_token];

    for _ in 0..max_chars {
        // Use last BLOCK_SIZE tokens as context
        let start = if tokens.len() > BLOCK_SIZE { tokens.len() - BLOCK_SIZE } else { 0 };
        let context = &tokens[start..];

        let (logits, _) = model.forward(context);

        // Take logits for last position
        let t = logits.rows;
        let last_row = logits.row(t - 1);

        // Temperature scaling (0.8)
        let temp = 0.8;
        let scaled: Vec<f64> = last_row.iter().map(|&x| x / temp).collect();

        // Top-k filtering (k=40)
        let k = 40.min(scaled.len());
        let mut indexed: Vec<(usize, f64)> = scaled.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let threshold = indexed[k - 1].1;

        let filtered: Vec<f64> = scaled.iter().map(|&v| if v >= threshold { v } else { f64::NEG_INFINITY }).collect();

        // Softmax
        let max_val = filtered.iter().copied().fold(f64::NEG_INFINITY, |a, b| if b > a { b } else { a });
        let exps: Vec<f64> = filtered.iter().map(|&v| (v - max_val).exp()).collect();
        let sum: f64 = exps.iter().sum();
        let probs: Vec<f64> = exps.iter().map(|&e| e / sum).collect();

        let next = rng.categorical(&probs);
        tokens.push(next);
    }

    tokens
}

// =============================================================================
// Main
// =============================================================================

fn main() {
    let nthreads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    println!("{}", "=".repeat(60));
    println!("  Harmonic Transformer -- Pure Rust, Zero Dependencies");
    println!("  No crates. No GPU. Manual backpropagation.");
    println!("  CPU threads: {}", nthreads);
    println!("{}", "=".repeat(60));

    // Load Shakespeare
    let text = std::fs::read_to_string("data/input.txt")
        .expect("Failed to read data/input.txt. Download with:\n  \
                 mkdir data && curl -sL https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o data/input.txt");

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
        model::N_LAYER,
        model::N_HEAD,
        N_EMBD
    );
    println!("  Context: {} characters", BLOCK_SIZE);
    println!("  Batch: {} | Iters: {} | Eval every: {}", BATCH_SIZE, MAX_ITERS, EVAL_INTERVAL);
    println!("  Expected: -ln(1/{}) = {:.4} initial loss", dataset.vocab_size, -(1.0 / dataset.vocab_size as f64).ln());

    // Train all three modes
    let modes = ["baseline", "harmonic", "frozen"];
    let seeds = [42, 42, 42];
    let mut all_results: Vec<(String, Vec<LossEntry>)> = Vec::new();
    let mut all_models: Vec<Transformer> = Vec::new();

    for (mode, seed) in modes.iter().zip(seeds.iter()) {
        let (model, history) = train_model(mode, &dataset, *seed);
        all_results.push((mode.to_string(), history));
        all_models.push(model);
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
        let final_entry = history.last().unwrap();
        println!(
            "  {:<12} {:>10.4} {:>12.4}",
            mode, final_entry.val, final_entry.train
        );
    }

    // =========================================================================
    // Convergence
    // =========================================================================
    println!("\n{}", "=".repeat(60));
    println!("  CONVERGENCE: Loss at Each Checkpoint");
    println!("{}", "=".repeat(60));
    println!();
    print!("  {:>6}", "Step");
    for mode in &modes {
        print!("  {:>12}", mode);
    }
    println!();
    print!("  {:>6}", "----");
    for _ in &modes {
        print!("  {:>12}", "----------");
    }
    println!();

    let max_entries = all_results.iter().map(|(_, h)| h.len()).max().unwrap_or(0);
    for i in 0..max_entries {
        let step = if i < all_results[0].1.len() {
            all_results[0].1[i].step
        } else {
            0
        };
        print!("  {:>6}", step);
        for (_, history) in &all_results {
            if i < history.len() {
                print!("  {:>12.4}", history[i].val);
            } else {
                print!("  {:>12}", "");
            }
        }
        println!();
    }

    // =========================================================================
    // Sample generation
    // =========================================================================
    println!("\n{}", "=".repeat(60));
    println!("  SAMPLE GENERATION ({} characters each)", GEN_CHARS);
    println!("{}", "=".repeat(60));

    let newline_token = dataset.itos.iter().position(|&c| c == '\n').unwrap_or(0);

    for (i, mode) in modes.iter().enumerate() {
        println!("\n  --- {} ---", mode.to_uppercase());
        let mut gen_rng = Rng::new(123);
        let tokens = generate(&all_models[i], newline_token, GEN_CHARS, &mut gen_rng);
        let text_out = dataset.decode(&tokens);
        for line in text_out.split('\n') {
            println!("  {}", line);
        }
    }

    // =========================================================================
    // Summary
    // =========================================================================
    println!("\n{}", "=".repeat(60));
    println!("  SUMMARY");
    println!("{}", "=".repeat(60));

    let baseline_val = all_results[0].1.last().unwrap().val;
    let harmonic_val = all_results[1].1.last().unwrap().val;
    let frozen_val = all_results[2].1.last().unwrap().val;

    println!();
    println!(
        "  Baseline (random init, trainable):     {:.4}",
        baseline_val
    );
    println!(
        "  Harmonic (structured init, trainable):  {:.4}",
        harmonic_val
    );
    println!(
        "  Frozen   (structured, NOT trainable):   {:.4}",
        frozen_val
    );
    println!();

    if harmonic_val < baseline_val {
        let pct = (1.0 - harmonic_val / baseline_val) * 100.0;
        println!(
            "  Harmonic embeddings OUTPERFORM baseline by {:.1}% on val loss.",
            pct
        );
    } else if harmonic_val > baseline_val {
        let pct = (harmonic_val / baseline_val - 1.0) * 100.0;
        println!(
            "  Harmonic embeddings underperform baseline by {:.1}% on val loss.",
            pct
        );
    } else {
        println!("  Harmonic embeddings MATCH baseline on val loss.");
    }

    if frozen_val < baseline_val * 1.1 {
        println!("  Frozen harmonic embeddings within 10% of baseline --");
        println!("  geometric structure alone carries most of the signal.");
    } else if frozen_val < baseline_val * 1.5 {
        println!("  Frozen harmonic embeddings within 50% of baseline --");
        println!("  geometry provides a useful starting point.");
    } else {
        println!("  Frozen harmonic embeddings significantly worse --");
        println!("  pure geometry insufficient without adaptation.");
    }

    // Convergence speed check
    if all_results[0].1.len() >= 3 {
        let baseline_early = all_results[0].1[2].val; // step 500
        let mut harmonic_reached = None;
        for entry in &all_results[1].1 {
            if entry.val <= baseline_early {
                harmonic_reached = Some(entry.step);
                break;
            }
        }
        if let Some(step) = harmonic_reached {
            if step < 500 {
                println!(
                    "  Harmonic reached baseline's step-500 loss ({:.4}) at step {} --",
                    baseline_early, step
                );
                if step > 0 {
                    println!("  {:.1}x faster convergence.", 500.0 / step as f64);
                }
            }
        }
    }

    println!();
    println!("  Built in pure Rust. Zero crates. Manual backpropagation.");
    println!("  CPU threads: {}. No GPU required.", nthreads);
    println!();
    println!("{}", "=".repeat(60));
}
