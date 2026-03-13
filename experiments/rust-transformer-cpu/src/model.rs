// Transformer model — forward pass, backward pass, all hand-coded.
// No autograd. Each layer returns a cache on forward, consumes it on backward.
//
// Architecture (matches Python harmonic_transformer.py):
//   Pre-norm residual blocks: x = x + attn(layernorm(x))
//   4 layers, 4 heads, 128 dim, 256 context
//   MLP: 128→512→128 with GELU
//   ~842K parameters (harmonic mode)

use crate::rng::Rng;
use crate::tensor::Mat;

// =============================================================================
// Configuration
// =============================================================================

pub const N_LAYER: usize = 4;
pub const N_HEAD: usize = 4;
pub const N_EMBD: usize = 128;
pub const HEAD_DIM: usize = N_EMBD / N_HEAD; // 32
pub const BLOCK_SIZE: usize = 256;
pub const MLP_DIM: usize = 4 * N_EMBD; // 512

// =============================================================================
// Linear layer
// =============================================================================

pub struct Linear {
    pub weight: Mat, // (out_features, in_features) stored transposed as (in, out) for matmul
    pub bias: Option<Mat>, // (1, out_features)
}

pub struct LinearCache {
    input: Mat, // (T, in)
}

pub struct LinearGrad {
    pub d_weight: Mat,
    pub d_bias: Option<Mat>,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, bias: bool, rng: &mut Rng) -> Self {
        // Xavier-like init: N(0, 0.02)
        let weight = Mat::randn(in_features, out_features, 0.0, 0.02, rng);
        let bias = if bias {
            Some(Mat::zeros(1, out_features))
        } else {
            None
        };
        Self { weight, bias }
    }

    /// Scaled init for residual projection layers: std = 0.02 / sqrt(2 * n_layer).
    pub fn new_residual(in_features: usize, out_features: usize, bias: bool, rng: &mut Rng) -> Self {
        let std = 0.02 / (2.0 * N_LAYER as f64).sqrt();
        let weight = Mat::randn(in_features, out_features, 0.0, std, rng);
        let bias = if bias {
            Some(Mat::zeros(1, out_features))
        } else {
            None
        };
        Self { weight, bias }
    }

    /// forward: y = x @ W + b, returns (y, cache)
    pub fn forward(&self, x: &Mat) -> (Mat, LinearCache) {
        let cache = LinearCache { input: x.clone() };
        let mut y = x.matmul(&self.weight);
        if let Some(ref b) = self.bias {
            y = y.add(b);
        }
        (y, cache)
    }

    /// backward: given d_y, return (d_x, LinearGrad)
    pub fn backward(&self, d_y: &Mat, cache: &LinearCache) -> (Mat, LinearGrad) {
        // d_x = d_y @ W^T
        let d_x = d_y.matmul(&self.weight.transpose());
        // d_W = x^T @ d_y
        let d_weight = cache.input.transpose().matmul(d_y);
        // d_b = sum_rows(d_y)
        let d_bias = self.bias.as_ref().map(|_| d_y.sum_axis0());
        (d_x, LinearGrad { d_weight, d_bias })
    }

    pub fn param_count(&self) -> usize {
        let w = self.weight.data.len();
        let b = self.bias.as_ref().map_or(0, |b| b.data.len());
        w + b
    }
}

// =============================================================================
// LayerNorm
// =============================================================================

pub struct LayerNorm {
    pub gamma: Mat, // (1, C)
    pub beta: Mat,  // (1, C)
    eps: f64,
}

pub struct LayerNormCache {
    x_hat: Mat,   // normalized input
    std_inv: Vec<f64>, // 1/std for each row
}

pub struct LayerNormGrad {
    pub d_gamma: Mat,
    pub d_beta: Mat,
}

impl LayerNorm {
    pub fn new(dim: usize) -> Self {
        Self {
            gamma: Mat::ones(1, dim),
            beta: Mat::zeros(1, dim),
            eps: 1e-5,
        }
    }

    /// forward: x_hat = (x - mean) / std, y = gamma * x_hat + beta
    pub fn forward(&self, x: &Mat) -> (Mat, LayerNormCache) {
        let t = x.rows;
        let c = x.cols;
        let mut x_hat = Mat::zeros(t, c);
        let mut std_inv = Vec::with_capacity(t);

        for r in 0..t {
            let row = x.row(r);
            let mean: f64 = row.iter().sum::<f64>() / c as f64;
            let var: f64 = row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / c as f64;
            let inv = 1.0 / (var + self.eps).sqrt();
            std_inv.push(inv);
            let out_row = x_hat.row_mut(r);
            for j in 0..c {
                out_row[j] = (row[j] - mean) * inv;
            }
        }

        // y = gamma * x_hat + beta (broadcast)
        let mut y = Mat::zeros(t, c);
        for r in 0..t {
            let xh = x_hat.row(r);
            let out = y.row_mut(r);
            for j in 0..c {
                out[j] = self.gamma.data[j] * xh[j] + self.beta.data[j];
            }
        }

        let cache = LayerNormCache {
            x_hat,
            std_inv,
        };
        (y, cache)
    }

    /// backward: given d_y, return (d_x, LayerNormGrad)
    pub fn backward(&self, d_y: &Mat, cache: &LayerNormCache) -> (Mat, LayerNormGrad) {
        let t = d_y.rows;
        let c = d_y.cols;
        let n = c as f64;

        // d_gamma = sum over T of (d_y * x_hat), d_beta = sum over T of d_y
        let mut d_gamma = Mat::zeros(1, c);
        let mut d_beta = Mat::zeros(1, c);
        for r in 0..t {
            for j in 0..c {
                d_gamma.data[j] += d_y.at(r, j) * cache.x_hat.at(r, j);
                d_beta.data[j] += d_y.at(r, j);
            }
        }

        // d_x: per row
        // dx_hat = dy * gamma
        // mean_dx_hat = mean(dx_hat)
        // mean_dx_hat_xhat = mean(dx_hat * x_hat)
        // dx = (1/std) * (dx_hat - mean_dx_hat - x_hat * mean_dx_hat_xhat)
        let mut d_x = Mat::zeros(t, c);
        for r in 0..t {
            let inv = cache.std_inv[r];
            // Compute dx_hat for this row, and the two means
            let mut mean_dxh = 0.0;
            let mut mean_dxh_xh = 0.0;
            for j in 0..c {
                let dxh = d_y.at(r, j) * self.gamma.data[j];
                mean_dxh += dxh;
                mean_dxh_xh += dxh * cache.x_hat.at(r, j);
            }
            mean_dxh /= n;
            mean_dxh_xh /= n;

            let out = d_x.row_mut(r);
            for j in 0..c {
                let dxh = d_y.at(r, j) * self.gamma.data[j];
                out[j] = inv * (dxh - mean_dxh - cache.x_hat.at(r, j) * mean_dxh_xh);
            }
        }

        (d_x, LayerNormGrad { d_gamma, d_beta })
    }
}

// =============================================================================
// Causal Self-Attention
// =============================================================================

pub struct CausalSelfAttention {
    pub c_attn: Linear,  // (C, 3C) — projects to Q, K, V
    pub c_proj: Linear,  // (C, C) — output projection
}

/// Per-head attention cache for backward.
struct HeadCache {
    q: Mat,       // (T, head_dim)
    k: Mat,       // (T, head_dim)
    v: Mat,       // (T, head_dim)
    attn: Mat,    // (T, T) softmax weights
}

#[allow(dead_code)]
pub struct AttentionCache {
    attn_cache: LinearCache,
    proj_cache: LinearCache,
    heads: Vec<HeadCache>,
    t: usize,
}

pub struct AttentionGrad {
    pub c_attn_grad: LinearGrad,
    pub c_proj_grad: LinearGrad,
}

impl CausalSelfAttention {
    pub fn new(rng: &mut Rng) -> Self {
        Self {
            c_attn: Linear::new(N_EMBD, 3 * N_EMBD, true, rng),
            c_proj: Linear::new_residual(N_EMBD, N_EMBD, true, rng),
        }
    }

    /// forward for a single sequence (T, C).
    pub fn forward(&self, x: &Mat) -> (Mat, AttentionCache) {
        let t = x.rows;

        // Project to QKV: (T, C) → (T, 3C)
        let (qkv, attn_cache) = self.c_attn.forward(x);

        // Split into Q, K, V: each (T, C)
        let q_full = qkv.slice_cols(0, N_EMBD);
        let k_full = qkv.slice_cols(N_EMBD, 2 * N_EMBD);
        let v_full = qkv.slice_cols(2 * N_EMBD, 3 * N_EMBD);

        let scale = 1.0 / (HEAD_DIM as f64).sqrt();

        // Process each head
        let mut heads = Vec::with_capacity(N_HEAD);
        let mut out_parts: Vec<Mat> = Vec::with_capacity(N_HEAD);

        for h in 0..N_HEAD {
            let col_start = h * HEAD_DIM;
            let col_end = col_start + HEAD_DIM;

            let q = q_full.slice_cols(col_start, col_end); // (T, hd)
            let k = k_full.slice_cols(col_start, col_end);
            let v = v_full.slice_cols(col_start, col_end);

            // Attention scores: Q @ K^T * scale → (T, T)
            let mut scores = q.matmul(&k.transpose()).scale(scale);

            // Causal mask: set upper triangle to -inf
            for i in 0..t {
                for j in (i + 1)..t {
                    *scores.at_mut(i, j) = f64::NEG_INFINITY;
                }
            }

            // Softmax → (T, T)
            let attn = scores.softmax();

            // Weighted sum: attn @ V → (T, hd)
            let head_out = attn.matmul(&v);

            heads.push(HeadCache { q, k, v, attn });
            out_parts.push(head_out);
        }

        // Concatenate heads: (T, C)
        let mut concat = Mat::zeros(t, N_EMBD);
        for (h, part) in out_parts.iter().enumerate() {
            concat.set_cols(h * HEAD_DIM, part);
        }

        // Output projection: (T, C) → (T, C)
        let (y, proj_cache) = self.c_proj.forward(&concat);

        let cache = AttentionCache {
            attn_cache,
            proj_cache,
            heads,
            t,
        };
        (y, cache)
    }

    /// backward: given d_y (T, C), return (d_x (T, C), AttentionGrad)
    #[allow(dead_code)]
    pub fn backward(&self, d_y: &Mat, cache: &AttentionCache) -> (Mat, AttentionGrad) {
        let t = cache.t;
        let scale = 1.0 / (HEAD_DIM as f64).sqrt();

        // Backward through output projection
        let (d_concat, c_proj_grad) = self.c_proj.backward(d_y, &cache.proj_cache);

        // Split d_concat into per-head gradients
        let mut d_q_full = Mat::zeros(t, N_EMBD);
        let mut d_k_full = Mat::zeros(t, N_EMBD);
        let mut d_v_full = Mat::zeros(t, N_EMBD);

        for h in 0..N_HEAD {
            let col_start = h * HEAD_DIM;
            let col_end = col_start + HEAD_DIM;
            let d_head_out = d_concat.slice_cols(col_start, col_end); // (T, hd)

            let hc = &cache.heads[h];

            // d_attn = d_head_out @ V^T → (T, T)
            let d_attn = d_head_out.matmul(&hc.v.transpose());

            // d_V = attn^T @ d_head_out → (T, hd)
            let d_v = hc.attn.transpose().matmul(&d_head_out);

            // Softmax backward: d_scores = softmax_backward(attn, d_attn)
            let d_scores = hc.attn.softmax_backward(&d_attn);

            // Apply causal mask to gradient (positions where scores were -inf have attn=0,
            // so softmax_backward already zeros them, but let's be safe)
            let mut d_scores_masked = d_scores;
            for i in 0..t {
                for j in (i + 1)..t {
                    *d_scores_masked.at_mut(i, j) = 0.0;
                }
            }

            // Scale
            let d_scores_scaled = d_scores_masked.scale(scale);

            // d_Q = d_scores @ K → (T, hd)
            let d_q = d_scores_scaled.matmul(&hc.k);

            // d_K = d_scores^T @ Q → (T, hd)
            let d_k = d_scores_scaled.transpose().matmul(&hc.q);

            // Write into full-width gradient matrices
            d_q_full.set_cols(col_start, &d_q);
            d_k_full.set_cols(col_start, &d_k);
            d_v_full.set_cols(col_start, &d_v);
        }

        // Merge Q, K, V gradients → (T, 3C) for c_attn backward
        let mut d_qkv = Mat::zeros(t, 3 * N_EMBD);
        d_qkv.set_cols(0, &d_q_full);
        d_qkv.set_cols(N_EMBD, &d_k_full);
        d_qkv.set_cols(2 * N_EMBD, &d_v_full);

        // Backward through c_attn
        let (d_x, c_attn_grad) = self.c_attn.backward(&d_qkv, &cache.attn_cache);

        (
            d_x,
            AttentionGrad {
                c_attn_grad,
                c_proj_grad,
            },
        )
    }

    pub fn param_count(&self) -> usize {
        self.c_attn.param_count() + self.c_proj.param_count()
    }
}

// =============================================================================
// MLP
// =============================================================================

pub struct MLP {
    pub c_fc: Linear,   // (C, 4C)
    pub c_proj: Linear,  // (4C, C)
}

pub struct MLPCache {
    fc_cache: LinearCache,
    fc_out: Mat,        // pre-GELU activations for backward
    proj_cache: LinearCache,
}

pub struct MLPGrad {
    pub c_fc_grad: LinearGrad,
    pub c_proj_grad: LinearGrad,
}

impl MLP {
    pub fn new(rng: &mut Rng) -> Self {
        Self {
            c_fc: Linear::new(N_EMBD, MLP_DIM, true, rng),
            c_proj: Linear::new_residual(MLP_DIM, N_EMBD, true, rng),
        }
    }

    pub fn forward(&self, x: &Mat) -> (Mat, MLPCache) {
        let (fc_out, fc_cache) = self.c_fc.forward(x);
        let gelu_out = fc_out.gelu();
        let (y, proj_cache) = self.c_proj.forward(&gelu_out);
        let cache = MLPCache {
            fc_cache,
            fc_out,
            proj_cache,
        };
        (y, cache)
    }

    pub fn backward(&self, d_y: &Mat, cache: &MLPCache) -> (Mat, MLPGrad) {
        // Backward through c_proj
        let (d_gelu_out, c_proj_grad) = self.c_proj.backward(d_y, &cache.proj_cache);

        // GELU backward: d_fc_out = d_gelu_out * gelu'(fc_out)
        let gelu_deriv = cache.fc_out.gelu_backward();
        let d_fc_out = d_gelu_out.hadamard(&gelu_deriv);

        // Backward through c_fc
        let (d_x, c_fc_grad) = self.c_fc.backward(&d_fc_out, &cache.fc_cache);

        (d_x, MLPGrad { c_fc_grad, c_proj_grad })
    }

    pub fn param_count(&self) -> usize {
        self.c_fc.param_count() + self.c_proj.param_count()
    }
}

// =============================================================================
// Transformer Block
// =============================================================================

pub struct Block {
    pub ln_1: LayerNorm,
    pub attn: CausalSelfAttention,
    pub ln_2: LayerNorm,
    pub mlp: MLP,
}

#[allow(dead_code)]
pub struct BlockCache {
    ln1_cache: LayerNormCache,
    attn_cache: AttentionCache,
    ln2_cache: LayerNormCache,
    mlp_cache: MLPCache,
}

pub struct BlockGrad {
    pub ln1_grad: LayerNormGrad,
    pub attn_grad: AttentionGrad,
    pub ln2_grad: LayerNormGrad,
    pub mlp_grad: MLPGrad,
}

impl Block {
    pub fn new(rng: &mut Rng) -> Self {
        Self {
            ln_1: LayerNorm::new(N_EMBD),
            attn: CausalSelfAttention::new(rng),
            ln_2: LayerNorm::new(N_EMBD),
            mlp: MLP::new(rng),
        }
    }

    /// forward: x = x + attn(ln1(x)); x = x + mlp(ln2(x))
    pub fn forward(&self, x: &Mat) -> (Mat, BlockCache) {
        // Branch 1: attention
        let (ln1_out, ln1_cache) = self.ln_1.forward(x);
        let (attn_out, attn_cache) = self.attn.forward(&ln1_out);
        let x2 = x.add(&attn_out); // residual

        // Branch 2: MLP
        let (ln2_out, ln2_cache) = self.ln_2.forward(&x2);
        let (mlp_out, mlp_cache) = self.mlp.forward(&ln2_out);
        let y = x2.add(&mlp_out); // residual

        let cache = BlockCache {
            ln1_cache,
            attn_cache,
            ln2_cache,
            mlp_cache,
        };
        (y, cache)
    }

    /// backward: given d_y, return (d_x, BlockGrad)
    #[allow(dead_code)]
    pub fn backward(&self, d_y: &Mat, cache: &BlockCache) -> (Mat, BlockGrad) {
        // Backward through second residual: d_y splits to d_x2 and d_mlp_out
        let d_mlp_out = d_y;
        let d_x2_from_residual = d_y;

        // Backward through MLP
        let (d_ln2_out, mlp_grad) = self.mlp.backward(d_mlp_out, &cache.mlp_cache);

        // Backward through ln2
        let (d_x2_from_ln2, ln2_grad) = self.ln_2.backward(&d_ln2_out, &cache.ln2_cache);

        // Combine: d_x2 = d_x2_from_residual + d_x2_from_ln2
        let d_x2 = d_x2_from_residual.add(&d_x2_from_ln2);

        // Backward through first residual: d_x2 splits to d_x and d_attn_out
        let d_attn_out = &d_x2;
        let d_x_from_residual = &d_x2;

        // Backward through attention
        let (d_ln1_out, attn_grad) = self.attn.backward(d_attn_out, &cache.attn_cache);

        // Backward through ln1
        let (d_x_from_ln1, ln1_grad) = self.ln_1.backward(&d_ln1_out, &cache.ln1_cache);

        // Combine: d_x = d_x_from_residual + d_x_from_ln1
        let d_x = d_x_from_residual.add(&d_x_from_ln1);

        (
            d_x,
            BlockGrad {
                ln1_grad,
                attn_grad,
                ln2_grad,
                mlp_grad,
            },
        )
    }

    pub fn param_count(&self) -> usize {
        2 + N_EMBD * 2  // ln gamma+beta * 2
        + self.attn.param_count()
        + self.mlp.param_count()
    }
}

// =============================================================================
// Transformer (full model)
// =============================================================================

pub struct Transformer {
    pub wte: Mat,       // (vocab_size, C) — token embeddings
    pub wpe: Mat,       // (block_size, C) — positional embeddings
    pub blocks: Vec<Block>,
    pub ln_f: LayerNorm,
    pub lm_head: Linear,
    pub mode: String,
    pub vocab_size: usize,
    pub weight_tied: bool,
}

#[allow(dead_code)]
pub struct TransformerCache {
    input_ids: Vec<usize>,  // for embedding backward
    t: usize,
    block_caches: Vec<BlockCache>,
    ln_f_cache: LayerNormCache,
    lm_head_cache: LinearCache,
}

pub struct TransformerGrad {
    pub d_wte: Mat,
    pub d_wpe: Mat,
    pub block_grads: Vec<BlockGrad>,
    pub ln_f_grad: LayerNormGrad,
    pub lm_head_grad: LinearGrad,
}

impl Transformer {
    pub fn new(vocab_size: usize, mode: &str, rng: &mut Rng) -> Self {
        let mut blocks = Vec::with_capacity(N_LAYER);
        for _ in 0..N_LAYER {
            blocks.push(Block::new(rng));
        }

        let ln_f = LayerNorm::new(N_EMBD);
        let lm_head = Linear::new(N_EMBD, vocab_size, false, rng);

        // Build embeddings based on mode
        let (wte, wpe, weight_tied) = match mode {
            "baseline" => {
                let wte = Mat::randn(vocab_size, N_EMBD, 0.0, 0.02, rng);
                let wpe = Mat::randn(BLOCK_SIZE, N_EMBD, 0.0, 0.02, rng);
                (wte, wpe, true) // weight tying for baseline
            }
            "harmonic" | "frozen" => {
                let wte = build_harmonic_embeddings(vocab_size, N_EMBD);
                let wpe = build_positional_encoding(BLOCK_SIZE, N_EMBD);
                (wte, wpe, false)
            }
            _ => panic!("Unknown mode: {mode}"),
        };

        let mut model = Self {
            wte,
            wpe,
            blocks,
            ln_f,
            lm_head,
            mode: mode.to_string(),
            vocab_size,
            weight_tied,
        };

        // For baseline with weight tying, lm_head.weight = wte^T
        // wte is (vocab, C), lm_head needs (C, vocab) for x @ W
        if weight_tied {
            model.lm_head.weight = model.wte.transpose();
        }

        model
    }

    /// Count trainable parameters.
    pub fn param_count(&self) -> usize {
        let emb = if self.mode == "frozen" {
            0 // frozen embeddings don't count
        } else {
            self.wte.data.len() + self.wpe.data.len()
        };
        let blocks: usize = self.blocks.iter().map(|b| b.param_count()).sum();
        let ln_f = N_EMBD * 2; // gamma + beta
        let lm_head = self.lm_head.param_count();
        // Weight tying: lm_head shares wte, don't double-count
        let head = if self.weight_tied { 0 } else { lm_head };
        emb + blocks + ln_f + head
    }

    /// Forward pass for a single sequence.
    /// input_ids: &[usize] of length T, each in [0, vocab_size).
    /// Returns (logits: Mat(T, vocab_size), cache).
    pub fn forward(&self, input_ids: &[usize]) -> (Mat, TransformerCache) {
        let t = input_ids.len();
        assert!(t <= BLOCK_SIZE, "sequence length {} exceeds block_size {}", t, BLOCK_SIZE);

        // Token embeddings: gather rows
        let tok_emb = self.wte.gather_rows(input_ids); // (T, C)

        // Positional embeddings: first T rows
        let pos_emb = self.wpe.slice_rows(0, t); // (T, C)

        // Combined embedding
        let x = tok_emb.add(&pos_emb); // (T, C)

        // Transformer blocks
        let mut h = x;
        let mut block_caches = Vec::with_capacity(N_LAYER);
        for block in &self.blocks {
            let (out, cache) = block.forward(&h);
            block_caches.push(cache);
            h = out;
        }

        // Final layer norm
        let (h, ln_f_cache) = self.ln_f.forward(&h);

        // Language model head: (T, C) → (T, vocab)
        let (logits, lm_head_cache) = self.lm_head.forward(&h);

        let cache = TransformerCache {
            input_ids: input_ids.to_vec(),
            t,
            block_caches,
            ln_f_cache,
            lm_head_cache,
        };
        (logits, cache)
    }

    /// Backward pass.
    /// d_logits: from cross_entropy_backward, shape (T, vocab_size).
    /// Returns TransformerGrad.
    #[allow(dead_code)]
    pub fn backward(&self, d_logits: &Mat, cache: &TransformerCache) -> TransformerGrad {
        // Backward through lm_head
        let (d_h, lm_head_grad) = self.lm_head.backward(d_logits, &cache.lm_head_cache);

        // Backward through final layer norm
        let (mut d_h, ln_f_grad) = self.ln_f.backward(&d_h, &cache.ln_f_cache);

        // Backward through blocks (reverse order)
        let mut block_grads = Vec::with_capacity(N_LAYER);
        for i in (0..N_LAYER).rev() {
            let (d_x, bg) = self.blocks[i].backward(&d_h, &cache.block_caches[i]);
            block_grads.push(bg);
            d_h = d_x;
        }
        block_grads.reverse(); // restore forward order

        // d_h is now gradient w.r.t. embedded = tok_emb + pos_emb
        // d_tok_emb = d_h, d_pos_emb = d_h (since addition broadcasts)

        // Embedding gradient: scatter-add into wte-shaped matrix
        let mut d_wte = Mat::zeros(self.vocab_size, N_EMBD);
        d_wte.scatter_add(&cache.input_ids, &d_h);

        // Positional gradient: sum over all positions present
        // d_wpe has shape (BLOCK_SIZE, C), but only first T rows get gradient
        let mut d_wpe = Mat::zeros(BLOCK_SIZE, N_EMBD);
        for r in 0..cache.t {
            let row = d_wpe.row_mut(r);
            for j in 0..N_EMBD {
                row[j] += d_h.at(r, j);
            }
        }

        TransformerGrad {
            d_wte,
            d_wpe,
            block_grads,
            ln_f_grad,
            lm_head_grad,
        }
    }
}

// =============================================================================
// Harmonic embedding construction
// =============================================================================

/// Build harmonic token embeddings: theta_c = c * 2pi / vocab_size
/// embed[c] = [cos(theta), sin(theta), cos(2*theta), sin(2*theta), ...]
/// Scaled by 1/sqrt(n_harmonics).
pub fn build_harmonic_embeddings(vocab_size: usize, n_embd: usize) -> Mat {
    let n_harmonics = n_embd / 2;
    let scale = 1.0 / (n_harmonics as f64).sqrt();
    let mut data = vec![0.0; vocab_size * n_embd];

    for c in 0..vocab_size {
        let theta = c as f64 * 2.0 * std::f64::consts::PI / vocab_size as f64;
        for h in 0..n_harmonics {
            let n = (h + 1) as f64;
            let phase = n * theta;
            data[c * n_embd + h * 2] = phase.cos() * scale;
            data[c * n_embd + h * 2 + 1] = phase.sin() * scale;
        }
    }

    Mat::from_vec(data, vocab_size, n_embd)
}

/// Build positional encoding with log-frequency scaling (like "Attention Is All You Need").
/// freq_scale = 1 / 10000^(2h / n_embd)
pub fn build_positional_encoding(max_len: usize, n_embd: usize) -> Mat {
    let n_harmonics = n_embd / 2;
    let scale = 1.0 / (n_harmonics as f64).sqrt();
    let mut data = vec![0.0; max_len * n_embd];

    for pos in 0..max_len {
        for h in 0..n_harmonics {
            let freq = 1.0 / 10000.0_f64.powf(2.0 * h as f64 / n_embd as f64);
            let phase = pos as f64 * freq;
            data[pos * n_embd + h * 2] = phase.cos() * scale;
            data[pos * n_embd + h * 2 + 1] = phase.sin() * scale;
        }
    }

    Mat::from_vec(data, max_len, n_embd)
}

// =============================================================================
// Batched forward/backward — CPU-efficient
//
// Linear, LayerNorm, MLP operate on (B*T, C) in one big matmul.
// Only attention splits into per-sequence (T, T) processing.
// This gives 28 threads real work: 8192 rows instead of 256.
// =============================================================================

// --- Batch cache types ---

pub struct AttentionBatchCache {
    attn_lin_cache: LinearCache,         // (B*T, 3C)
    proj_lin_cache: LinearCache,         // (B*T, C)
    per_seq_heads: Vec<Vec<HeadCache>>,  // [B][N_HEAD]
    b: usize,
    t: usize,
}

pub struct BlockBatchCache {
    ln1_cache: LayerNormCache,
    attn_cache: AttentionBatchCache,
    ln2_cache: LayerNormCache,
    mlp_cache: MLPCache,
}

pub struct TransformerBatchCache {
    all_ids: Vec<usize>,
    b: usize,
    t: usize,
    block_caches: Vec<BlockBatchCache>,
    ln_f_cache: LayerNormCache,
    lm_head_cache: LinearCache,
}

// --- Batched attention (parallel across sequences) ---

impl CausalSelfAttention {
    /// Batched forward: x is (B*T, C).
    /// Linear projections are one big matmul (parallel via row-split).
    /// Attention is parallelized across B sequences (one thread per sequence).
    pub fn forward_batch(&self, x: &Mat, b: usize, t: usize) -> (Mat, AttentionBatchCache) {
        // One big QKV projection: (B*T, C) @ (C, 3C) → (B*T, 3C)
        let (qkv, attn_lin_cache) = self.c_attn.forward(x);

        let scale = 1.0 / (HEAD_DIM as f64).sqrt();

        // Parallel attention across sequences — each thread processes one sequence
        let mut concat_data = vec![0.0f64; b * t * N_EMBD];
        let chunk_elems = t * N_EMBD;

        let per_seq_heads: Vec<Vec<HeadCache>> = std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(b);
            let mut data_rest = concat_data.as_mut_slice();
            let qkv_ref = &qkv;

            for seq in 0..b {
                let (chunk, rest) = data_rest.split_at_mut(chunk_elems);
                data_rest = rest;

                handles.push(scope.spawn(move || {
                    let row_start = seq * t;
                    let qkv_s = qkv_ref.slice_rows(row_start, row_start + t);
                    let q_full = qkv_s.slice_cols(0, N_EMBD);
                    let k_full = qkv_s.slice_cols(N_EMBD, 2 * N_EMBD);
                    let v_full = qkv_s.slice_cols(2 * N_EMBD, 3 * N_EMBD);

                    let mut heads = Vec::with_capacity(N_HEAD);

                    for h in 0..N_HEAD {
                        let cs = h * HEAD_DIM;
                        let ce = cs + HEAD_DIM;
                        let q = q_full.slice_cols(cs, ce);
                        let k = k_full.slice_cols(cs, ce);
                        let v = v_full.slice_cols(cs, ce);

                        let mut scores = q.matmul(&k.transpose()).scale(scale);
                        for i in 0..t {
                            for j in (i + 1)..t {
                                *scores.at_mut(i, j) = f64::NEG_INFINITY;
                            }
                        }
                        let attn = scores.softmax();
                        let head_out = attn.matmul(&v);

                        // Write into chunk (this sequence's portion of all_concat)
                        for r in 0..t {
                            for j in 0..HEAD_DIM {
                                chunk[r * N_EMBD + cs + j] = head_out.at(r, j);
                            }
                        }

                        heads.push(HeadCache { q, k, v, attn });
                    }

                    heads
                }));
            }

            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        let all_concat = Mat::from_vec(concat_data, b * t, N_EMBD);

        // One big output projection: (B*T, C) @ (C, C) → (B*T, C)
        let (y, proj_lin_cache) = self.c_proj.forward(&all_concat);

        let cache = AttentionBatchCache {
            attn_lin_cache,
            proj_lin_cache,
            per_seq_heads,
            b,
            t,
        };
        (y, cache)
    }

    /// Batched backward: d_y is (B*T, C).
    /// Parallel across B sequences for attention gradients.
    pub fn backward_batch(&self, d_y: &Mat, cache: &AttentionBatchCache) -> (Mat, AttentionGrad) {
        let b = cache.b;
        let t = cache.t;
        let scale = 1.0 / (HEAD_DIM as f64).sqrt();

        // Backward through output projection: (B*T, C)
        let (d_concat, c_proj_grad) = self.c_proj.backward(d_y, &cache.proj_lin_cache);

        // Parallel attention backward across sequences
        let mut d_qkv_data = vec![0.0f64; b * t * 3 * N_EMBD];
        let chunk_elems = t * 3 * N_EMBD;

        std::thread::scope(|scope| {
            let mut data_rest = d_qkv_data.as_mut_slice();
            let d_concat_ref = &d_concat;
            let heads_ref = &cache.per_seq_heads;

            for seq in 0..b {
                let (chunk, rest) = data_rest.split_at_mut(chunk_elems);
                data_rest = rest;

                scope.spawn(move || {
                    let row_start = seq * t;
                    let d_concat_s = d_concat_ref.slice_rows(row_start, row_start + t);
                    let stride = 3 * N_EMBD;

                    for h in 0..N_HEAD {
                        let cs = h * HEAD_DIM;
                        let ce = cs + HEAD_DIM;
                        let d_head_out = d_concat_s.slice_cols(cs, ce);
                        let hc = &heads_ref[seq][h];

                        let d_attn = d_head_out.matmul(&hc.v.transpose());
                        let d_v = hc.attn.transpose().matmul(&d_head_out);
                        let d_scores = hc.attn.softmax_backward(&d_attn);

                        let mut d_scores_masked = d_scores;
                        for i in 0..t {
                            for j in (i + 1)..t {
                                *d_scores_masked.at_mut(i, j) = 0.0;
                            }
                        }
                        let d_scores_scaled = d_scores_masked.scale(scale);

                        let d_q = d_scores_scaled.matmul(&hc.k);
                        let d_k = d_scores_scaled.transpose().matmul(&hc.q);

                        // Write d_q, d_k, d_v into chunk
                        // Layout: [Q cols | K cols | V cols] per row
                        for r in 0..t {
                            for j in 0..HEAD_DIM {
                                chunk[r * stride + cs + j] = d_q.at(r, j);
                                chunk[r * stride + N_EMBD + cs + j] = d_k.at(r, j);
                                chunk[r * stride + 2 * N_EMBD + cs + j] = d_v.at(r, j);
                            }
                        }
                    }
                });
            }
        });

        let d_qkv = Mat::from_vec(d_qkv_data, b * t, 3 * N_EMBD);

        // Backward through QKV projection: (B*T, 3C)
        let (d_x, c_attn_grad) = self.c_attn.backward(&d_qkv, &cache.attn_lin_cache);

        (d_x, AttentionGrad { c_attn_grad, c_proj_grad })
    }
}

// --- Batched block ---

impl Block {
    /// Batched forward: x is (B*T, C).
    pub fn forward_batch(&self, x: &Mat, b: usize, t: usize) -> (Mat, BlockBatchCache) {
        let (ln1_out, ln1_cache) = self.ln_1.forward(x);
        let (attn_out, attn_cache) = self.attn.forward_batch(&ln1_out, b, t);
        let x2 = x.add(&attn_out);

        let (ln2_out, ln2_cache) = self.ln_2.forward(&x2);
        let (mlp_out, mlp_cache) = self.mlp.forward(&ln2_out);
        let y = x2.add(&mlp_out);

        let cache = BlockBatchCache {
            ln1_cache,
            attn_cache,
            ln2_cache,
            mlp_cache,
        };
        (y, cache)
    }

    /// Batched backward: d_y is (B*T, C).
    pub fn backward_batch(&self, d_y: &Mat, cache: &BlockBatchCache) -> (Mat, BlockGrad) {
        // Second residual
        let (d_ln2_out, mlp_grad) = self.mlp.backward(d_y, &cache.mlp_cache);
        let (d_x2_from_ln2, ln2_grad) = self.ln_2.backward(&d_ln2_out, &cache.ln2_cache);
        let d_x2 = d_y.add(&d_x2_from_ln2);

        // First residual
        let (d_ln1_out, attn_grad) = self.attn.backward_batch(&d_x2, &cache.attn_cache);
        let (d_x_from_ln1, ln1_grad) = self.ln_1.backward(&d_ln1_out, &cache.ln1_cache);
        let d_x = d_x2.add(&d_x_from_ln1);

        (d_x, BlockGrad { ln1_grad, attn_grad, ln2_grad, mlp_grad })
    }
}

// --- Batched transformer ---

impl Transformer {
    /// Batched forward: processes B sequences of length T together.
    /// Linear/LN/MLP operate on (B*T, C). Attention splits per-sequence.
    /// Returns logits (B*T, vocab) and cache for backward.
    pub fn forward_batch(&self, batch: &[&[usize]]) -> (Mat, TransformerBatchCache) {
        let b = batch.len();
        let t = batch[0].len();
        assert!(t <= BLOCK_SIZE);

        // Flatten all input IDs
        let all_ids: Vec<usize> = batch.iter().flat_map(|seq| seq.iter().copied()).collect();

        // Token embeddings: (B*T, C)
        let tok_emb = self.wte.gather_rows(&all_ids);

        // Positional embeddings: tile (T, C) → (B*T, C)
        let pos_slice = self.wpe.slice_rows(0, t);
        let mut pos_tiled = Mat::zeros(b * t, N_EMBD);
        for s in 0..b {
            pos_tiled.copy_rows_from(s * t, &pos_slice);
        }

        let mut h = tok_emb.add(&pos_tiled);

        // Blocks
        let mut block_caches = Vec::with_capacity(N_LAYER);
        for block in &self.blocks {
            let (out, cache) = block.forward_batch(&h, b, t);
            block_caches.push(cache);
            h = out;
        }

        // Final LN + lm_head
        let (h, ln_f_cache) = self.ln_f.forward(&h);
        let (logits, lm_head_cache) = self.lm_head.forward(&h);

        let cache = TransformerBatchCache {
            all_ids,
            b,
            t,
            block_caches,
            ln_f_cache,
            lm_head_cache,
        };
        (logits, cache)
    }

    /// Batched backward: d_logits is (B*T, vocab).
    pub fn backward_batch(&self, d_logits: &Mat, cache: &TransformerBatchCache) -> TransformerGrad {
        let (d_h, lm_head_grad) = self.lm_head.backward(d_logits, &cache.lm_head_cache);
        let (mut d_h, ln_f_grad) = self.ln_f.backward(&d_h, &cache.ln_f_cache);

        let mut block_grads = Vec::with_capacity(N_LAYER);
        for i in (0..N_LAYER).rev() {
            let (d_x, bg) = self.blocks[i].backward_batch(&d_h, &cache.block_caches[i]);
            block_grads.push(bg);
            d_h = d_x;
        }
        block_grads.reverse();

        // Embedding gradients
        let mut d_wte = Mat::zeros(self.vocab_size, N_EMBD);
        d_wte.scatter_add(&cache.all_ids, &d_h);

        let mut d_wpe = Mat::zeros(BLOCK_SIZE, N_EMBD);
        // Sum positional gradient across all B sequences
        for s in 0..cache.b {
            for r in 0..cache.t {
                let src = d_h.row(s * cache.t + r);
                let dst = d_wpe.row_mut(r);
                for j in 0..N_EMBD {
                    dst[j] += src[j];
                }
            }
        }

        TransformerGrad {
            d_wte,
            d_wpe,
            block_grads,
            ln_f_grad,
            lm_head_grad,
        }
    }
}
