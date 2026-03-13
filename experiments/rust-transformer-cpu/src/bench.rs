// Microbenchmark — profiles each phase of a training step.
// Run with: cargo run --release --bin bench

use std::time::Instant;

// Pull in everything from the library
#[path = "rng.rs"]
mod rng;
#[path = "tensor.rs"]
mod tensor;
#[path = "model.rs"]
mod model;

use model::{Transformer, N_EMBD, BLOCK_SIZE};
use rng::Rng;
use tensor::Mat;

const BATCH_SIZE: usize = 32;
const T: usize = BLOCK_SIZE; // 256

fn main() {
    let nthreads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    println!("=== Training Step Profiler ===");
    println!("CPU threads: {}", nthreads);
    println!("Batch: {} sequences x {} tokens = {} rows", BATCH_SIZE, T, BATCH_SIZE * T);
    println!();

    let mut rng = Rng::new(42);
    let vocab_size = 65;
    let model = Transformer::new(vocab_size, "harmonic", &mut rng);

    // Create fake batch
    let batch_data: Vec<Vec<usize>> = (0..BATCH_SIZE)
        .map(|_| (0..T).map(|_| rng.usize(vocab_size)).collect())
        .collect();
    let batch_refs: Vec<&[usize]> = batch_data.iter().map(|v| v.as_slice()).collect();
    let targets: Vec<usize> = (0..BATCH_SIZE * T)
        .map(|_| rng.usize(vocab_size))
        .collect();

    // Warm up
    {
        let (logits, cache) = model.forward_batch(&batch_refs);
        let d_logits = logits.cross_entropy_backward(&targets);
        let _grad = model.backward_batch(&d_logits, &cache);
    }

    println!("--- Timing 5 iterations ---\n");

    for iter in 0..5 {
        println!("Iteration {}:", iter);

        let t0 = Instant::now();
        let (logits, cache) = model.forward_batch(&batch_refs);
        let fwd_time = t0.elapsed();

        let t1 = Instant::now();
        let d_logits = logits.cross_entropy_backward(&targets);
        let ce_time = t1.elapsed();

        let t2 = Instant::now();
        let _grad = model.backward_batch(&d_logits, &cache);
        let bwd_time = t2.elapsed();

        let total = fwd_time + ce_time + bwd_time;
        println!("  forward:    {:>7.1}ms", fwd_time.as_secs_f64() * 1000.0);
        println!("  CE backward:{:>7.1}ms", ce_time.as_secs_f64() * 1000.0);
        println!("  backward:   {:>7.1}ms", bwd_time.as_secs_f64() * 1000.0);
        println!("  TOTAL:      {:>7.1}ms", total.as_secs_f64() * 1000.0);
        println!();
    }

    // Now profile individual layers of the forward pass
    println!("--- Forward pass breakdown (1 iteration) ---\n");

    let all_ids: Vec<usize> = batch_refs.iter().flat_map(|seq| seq.iter().copied()).collect();

    let t0 = Instant::now();
    let tok_emb = model.wte.gather_rows(&all_ids);
    let pos_slice = model.wpe.slice_rows(0, T);
    let mut pos_tiled = Mat::zeros(BATCH_SIZE * T, N_EMBD);
    for s in 0..BATCH_SIZE {
        pos_tiled.copy_rows_from(s * T, &pos_slice);
    }
    let mut h = tok_emb.add(&pos_tiled);
    println!("  embedding:  {:>7.1}ms", t0.elapsed().as_secs_f64() * 1000.0);

    for (i, block) in model.blocks.iter().enumerate() {
        let t0 = Instant::now();

        // LN1
        let tln1 = Instant::now();
        let (ln1_out, _ln1_cache) = block.ln_1.forward(&h);
        let ln1_time = tln1.elapsed();

        // Attention
        let tattn = Instant::now();
        let (attn_out, _attn_cache) = block.attn.forward_batch(&ln1_out, BATCH_SIZE, T);
        let attn_time = tattn.elapsed();

        let x2 = h.add(&attn_out);

        // LN2
        let tln2 = Instant::now();
        let (ln2_out, _ln2_cache) = block.ln_2.forward(&x2);
        let ln2_time = tln2.elapsed();

        // MLP
        let tmlp = Instant::now();
        let (mlp_out, _mlp_cache) = block.mlp.forward(&ln2_out);
        let mlp_time = tmlp.elapsed();

        h = x2.add(&mlp_out);

        let block_time = t0.elapsed();
        println!(
            "  block[{}]:   {:>7.1}ms  (LN1:{:.1} Attn:{:.1} LN2:{:.1} MLP:{:.1})",
            i,
            block_time.as_secs_f64() * 1000.0,
            ln1_time.as_secs_f64() * 1000.0,
            attn_time.as_secs_f64() * 1000.0,
            ln2_time.as_secs_f64() * 1000.0,
            mlp_time.as_secs_f64() * 1000.0,
        );
    }

    let t0 = Instant::now();
    let (h, _ln_f_cache) = model.ln_f.forward(&h);
    println!("  ln_f:       {:>7.1}ms", t0.elapsed().as_secs_f64() * 1000.0);

    let t0 = Instant::now();
    let (_logits, _lm_head_cache) = model.lm_head.forward(&h);
    println!("  lm_head:    {:>7.1}ms", t0.elapsed().as_secs_f64() * 1000.0);

    // Profile a single matmul at different sizes
    println!("\n--- Matmul benchmarks ---\n");
    let sizes = [
        (256, 128, 384, "Per-seq c_attn (256x128)@(128x384)"),
        (8192, 128, 384, "Batched c_attn (8192x128)@(128x384)"),
        (256, 32, 256, "Per-head Q@K^T (256x32)@(32x256)"),
        (8192, 128, 512, "Batched c_fc   (8192x128)@(128x512)"),
        (8192, 512, 128, "Batched c_proj (8192x512)@(512x128)"),
    ];

    for (m, k, n, label) in sizes {
        let a = Mat::randn(m, k, 0.0, 0.1, &mut rng);
        let b = Mat::randn(k, n, 0.0, 0.1, &mut rng);

        // Warm up
        let _ = a.matmul(&b);

        let t0 = Instant::now();
        let iters = 5;
        for _ in 0..iters {
            let _ = a.matmul(&b);
        }
        let avg = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        let flops = 2.0 * m as f64 * k as f64 * n as f64;
        let gflops = flops / (avg / 1000.0) / 1e9;
        println!(
            "  {} => {:.1}ms ({:.1} GFLOP/s)",
            label, avg, gflops
        );
    }
}
