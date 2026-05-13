# Wave Coherence Engine Patterns: Defensive Publication

**Authors:** Marco Da Cunha (Independent Researcher) and Claude (Anthropic)
**License:** MIT (same as parent framework)
**Purpose:** Defensive prior art publication to prevent patent enclosure of implementation patterns derived from Wave Coherence as a Computational Primitive.

**Legal intent:** This document constitutes a defensive publication under established intellectual property law. All engine patterns, architectures, and implementation methods described herein are published as prior art. Any patent application covering substantially similar implementations filed after this publication date is challengeable on prior art grounds. This document is timestamped via Git commit history and archived via Zenodo DOI.

**Structure:** This file is Part 2 of 3. See ENGINE-PATTERNS-INDEX.md for the master index.

---

**Part 2 of 3 — Patterns 71-111: Architecture & Training Patterns**
**Dates:** March 22, 2026 (71-80); March 30, 2026 (88-92); April 2, 2026 (93-111)

**Status:** All bodies complete (71-111).

---

## 71. Perturbative Kerr-ODE Engine (Telecom-Inspired Single-Pass)

### 71.1 Core Architecture

A nonlinear ODE computation engine that replaces iterative numerical integration (RK4, Euler) with a single-pass analytical perturbation theory approximation. Derived from telecom DSP techniques (ESSFM, Learned DBP, Volterra series) adapted for neural network FFN layers. One dispatch replaces 192 dispatches (16 RK4 steps × 12 passes each).

**Implementation pattern:**
- Compute linear solution: decay (exp(-γ)) × rotation (cos(ω), sin(ω))
- Self-phase modulation: |Z_linear|² per band
- Cross-phase modulation: stencil convolution [1,1,0,1,1] over neighbour magnitudes
- Phase perturbation: δφ = α × SPM + β × XPM
- Correction: r_out = r_lin - δφ × s_lin; s_out = s_lin + δφ × r_lin

**Validated:** MSE 0.000005 vs RK4-16 baseline. Trains better (loss 2.97 vs 3.07). 14x wall-clock speedup.

### 71.2 GPU Shader Implementation

WGSL compute shader where each thread computes one (position, band) element. Each thread recomputes its neighbours' linear solutions in-register (4 extra trig ops) to avoid synchronisation barriers or second dispatch pass. Workgroup size 64, dispatch = ceil(n_pos × n_bands / 64).

---

## 72. Block-Diagonal Output Projection Engine

### 72.1 Core Architecture

An output projection replacing dense matrix multiplication with N independent group_size × group_size blocks. Each block processes its band group independently.

**Implementation pattern:**
- Partition embedding into N groups (e.g., 6 groups of 128 for 768-dim)
- Each group: own weight matrix + bias vector
- Forward: split → independent linear transforms → concatenate
- Backward: gradients flow independently per group
- Parameter reduction: 6x at 768-dim (98K vs 590K)

### 72.2 Enum Abstraction

Rust OutProjWeights enum with Dense and BlockDiagonal variants exposing identical method interfaces (forward, backward_dx, flatten_into, unflatten_from, weights_flat, bias_flat, param_count). Consumer code is variant-agnostic. Checkpoint format stores group count in header.

### 72.3 GPU Block-Diagonal Shader

WGSL shader where each thread computes one output element. Thread determines its group from output index (group = out_i / group_size), reads only from its group's weight block. Single dispatch for all positions and groups.

---

## 73. Frozen Harmonic Coherence Attention

### 73.1 Core Architecture

Attention where harmonic coherence scores cos(n × (θ_i - θ_j)) replace trained Q×K dot products. Attention weights determined by harmonic structure, not learned parameters. Multi-head: each head uses a different harmonic number. Frozen (no gradient) — the mathematical structure IS the representation.

**Validated:** Frozen harmonic embeddings match trained embeddings (Test 25). Zero parameters for attention computation.

---

## 74. Parallel Block Formulation (GPT-J for Wave-Coherent Networks)

### 74.1 Core Architecture

Transformer block where attention and FFN operate in parallel on the same layer-normed input, outputs summed. One norm instead of two. Enables concurrent GPU execution of attention and FFN branches.

**Pattern:** x = x + attn(LN(x)) + FFN(LN(x)) instead of sequential x = x + attn(LN(x)); x = x + FFN(LN(x))

---

## 75. Multi-Tier Compute Architecture (Single Binary, Three Backends)

### 75.1 Core Architecture

Training engine compiling into a single binary with three selectable backends via CLI flag. All tiers share model architecture, weight format, and checkpoint system.

**Tiers:**
- CPU: Pure Rust, RK4-16 ODE, sequential attention. Correctness reference.
- wgpu: WGSL compute shaders. Fused RK4 or perturbative ODE. Vulkan/Metal/DX12. AMD, Intel, NVIDIA.
- Candle CUDA: HuggingFace Candle, perturbative ODE via tensor ops with true autograd, block-diagonal out_proj.

**Selection:** `--gpu` (wgpu), `--candle` (CUDA), default (CPU). All produce identical results (validated to 3.58e-7).

### 75.2 ComputeBackend Trait

Rust trait abstracting forward/backward operations. CPU and GPU implement the same trait. Consumer code is backend-agnostic.

---

## 76. WCHK Self-Describing Checkpoint Format

### 76.1 Core Architecture

Binary checkpoint embedding all architectural parameters in the header. Any compatible engine reconstructs the model from header alone.

**Format:** Magic "WCHK" + version + vocab_size + n_layers + out_proj_groups + iteration + lr + optimizer state size + RNG state. Body: flat f32 params + Adam moments. Version 2 adds block-diagonal support. Loss in filename for quick comparison.

---

## 77. Dual-Maestro Global Coordination Engine

### 77.1 Core Architecture

Two-stage bottleneck conditioning both input and output of the Kerr-ODE. Each maestro: Linear(d_model, maestro_dim) → GELU → Linear(maestro_dim, d_model). Pre-ODE maestro conditions input, post-ODE maestro regulates output.

**Pattern:** pre = x + maestro_in(x); ode_out = kerr_ode(pre); regulated = ode_out + maestro_out(x); output = out_proj(regulated)

**Validated:** Maestro-Add + curriculum = 98.1% of MLP at 44% params. Different mechanisms stack (coordination vs staging).

---

## 78. Ping-Pong Buffer GPU Consistency

### 78.1 Core Architecture

GPU buffer management where forward/backward alternate between two pre-allocated buffer sets. Eliminates per-dispatch allocation. Weight buffers uploaded once, updated in-place after optimizer. Buffer pool with cache-by-pointer reuses same GPU buffer within iteration.

---

## 79. Pipeline Monitor and Diagnostic Engine

### 79.1 Core Architecture

Training monitor providing real-time pipeline visibility. Per-section timing, VRAM monitoring (cudarc mem_get_info), gradient norm tracking, NaN recovery (skip step + log + continue), JSONL telemetry, checkpoint NaN guard (refuse save on NaN/Inf), loss in checkpoint filename.

---

## 80. MLP Weight Structure Analysis (Null Finding — Defensive Publication)

### 80.1 Methodology

Diagnostic analysis of trained transformer MLP weights to determine whether they contain exploitable structure for wave-based compression. Performed on Qwen 2.5 0.5B (896-dim, 24 layers).

**Four analyses:** SVD effective rank (1% threshold), activation statistics (norm ratio, cosine similarity), 1D DFT power spectrum across weight rows, zero-out layer importance (perplexity impact).

### 80.2 Findings (Null Result)

- **Effective rank: FULL (896/896 all layers, all projections).** No low-rank structure.
- **Frequency structure: FLAT (33.3%/33.2%/33.5% low/mid/high).** No wave structure in weights.
- **No near-identity layers.** Cosine similarity input→output is negative (-0.05 to -0.3).
- **Bookend importance.** Layers 0 (+32%) and 23 (+47%) most critical.

### 80.3 Implication

Trained MLP weights do not contain hidden wave structure translatable to ODE parameters. Wave-coherent FFN layers must be trained from scratch, not by parameter conversion. This establishes a boundary: the wave representation is a training architecture, not a compression codec for pre-trained dense matrices.

---

## 81. Physics-Bounded Adaptive Regulation (AGC for Neural ODE)

### 81.1 Core Architecture

An automatic gain control system that regulates the magnitude of signals entering a nonlinear ODE layer during neural network training. The regulation adapts to the model's learned operating range via exponential moving averages while respecting a physics-derived stability ceiling.

**The problem unique to neural ODE architectures:** A learned pre-conditioner (maestro) naturally increases magnitudes during training to carry more information through the ODE's nonlinear dynamics. Without regulation, magnitudes exceed the ODE's stability threshold (phase shift > 90°), causing numerical divergence. Fixed regulation (hard clamps) throttles the model — the pre-conditioner fights the clamp, creating V-shaped loss divergence.

**Implementation pattern:**
- **Floor** (min_threshold): prevents the ODE from receiving too-weak signal for meaningful computation
- **Ceiling** (max_threshold): derived from ODE physics — M < √(π/2 / (α + 4β)) where α, β are coupling constants
- **Adaptive range**: EMA of observed magnitudes (mean + 3σ) sets the compression threshold between floor and ceiling
- **Knee compressor**: below threshold, signal passes unchanged (zero compression tax on normal operation). Above threshold, smooth compression on the excess only: `output = threshold + threshold × tanh((excess) / threshold)`

### 81.2 Electronics Analogy Progression

The design follows a signal processing evolution, each step proven necessary by test data:

| Stage | Electronics equivalent | Implementation | Limitation |
|-------|----------------------|----------------|------------|
| Fixed clamp | Resistor | Hard cutoff at constant | Clips signal, V-shape divergence |
| Higher clamp | Larger resistor | Higher constant | Eventually outgrown (92% clamped) |
| Soft clamp | Zener diode | tanh compression | Over-compresses normal signal (17%) |
| Adaptive | AGC circuit | EMA-based threshold | No upper bound → ODE blows up |
| **Physics-bounded AGC** | **AGC + rail voltage** | **EMA + ODE ceiling** | **Adapts freely within stable range** |

### 81.3 ODE Stability Derivation

For the Kerr-ODE with coupling constants α, β and dt=1.0:
- Phase shift per step: δφ = (α + 4β) × M²
- Stability requires δφ < π/2
- Maximum stable magnitude: M_max = √(π/2 / (α + 4β))
- At α = β = 0.01: M_max ≈ 5.6 (ceiling set to 6.0 for margin)
- At α = β = 0.1: M_max ≈ 1.77 (explains why α=0.1 caused immediate NaN at small dims)

### 81.4 Validated Results

Five controlled tests at 256-dim, 12 layers, 512 BPE, 20K iterations:

| Regulation | Best loss | Rolling avg stable? | V-shape? |
|-----------|----------|-------------------|----------|
| Hard 2.5 | 4.16 | No — rising | YES |
| Hard 5.0 | 3.75 | No — mild rise | Delayed |
| Soft tanh | 3.83 | Yes through 16K | Mild late |
| AGC no ceiling | 4.57 | No — NaN at iter 4K | Blew up |
| **AGC + ceiling 6.0** | **3.76** | **Yes through 20K** | **NO** |

The physics-bounded AGC eliminated V-shape divergence while matching the best individual loss. Rolling averages descended monotonically through 20K iterations — something no fixed regulation achieved.

---

## 88. Learnable ODE Backward — Gradient Flow Through Kerr-ODE RK4 Integration

### 88.1 Core Pattern

Direct backpropagation through the full RK4 ODE integration, replacing the identity pass-through that treated the ODE as a fixed transform for gradient purposes. Each layer's coupling constants (α self-modulation, β cross-modulation, γ damping) become learnable parameters with per-layer gradient flow.

**Implementation pattern:**
- Forward pass caches all RK4 intermediate states (16 steps × 4 k-values × n_bands × 2)
- Backward unrolls the RK4 in reverse, applying the Jacobian of the Kerr derivative at each evaluation point
- Parameter gradients (d_alpha, d_beta, d_gamma_raw) accumulated across all RK4 steps and all sequence positions
- `--freeze-ode` flag preserves legacy identity backward for A/B comparison
- Memory cost: ~22MB at batch=4, seq=64, 4 layers (feasible on consumer GPU)

### 88.2 Three-Tier Implementation

| Tier | Forward | Backward | Status |
|------|---------|----------|--------|
| CPU | RK4-16 with caching | Direct backprop through cached states | Implemented |
| wgpu | RK4 via WGSL shaders | CPU fallback (ODE <10% of matmul cost) | Implemented |
| Candle | Perturbative tensor ops | Autograd (automatic, already worked) | No change needed |

### 88.3 Validated Results

| Run | Loss | vs Frozen | Iters |
|-----|------|-----------|-------|
| Frozen cycling 10K | 4.48 | baseline | 10K |
| Learnable cycling 10K | 3.76 | -0.72 (16% better) | 10K |
| Learnable sustained 30K | 3.18 | -1.01 (24% better) | 30K |
| Previous all-time best | 3.91 | learnable beat in 10K vs 70K | 70K (7 cycles) |

The learnable ODE achieved better loss in one 10K cycle than the frozen ODE achieved in seven cycles totalling 70K iterations. Training time reduced from ~83 minutes to ~15 minutes for superior results.

### 88.4 Key Finding: The Model Was Handcuffed, Not Lazy

The frozen ODE backward masked the true capability of the architecture for the entire project history. The model could not learn its own coupling dynamics — it learned *around* the ODE through maestro and out_proj, but could not learn *through* it. Multiple engine issues (channel drift, layer integration failure, expression bottleneck) traced to this single root cause. Fixing the root cause resolved multiple symptoms simultaneously.

---

## 89. Per-Layer Coupling Self-Organisation — Depth-Dependent Specialisation

### 89.1 Core Pattern

When given learnable ODE parameters, the model spontaneously develops depth-dependent coupling specialisation without any architectural constraint. The first layer maintains high self-coupling (per-band specialist), while deeper layers reduce self-coupling and maintain/increase cross-coupling (cross-band specialists).

### 89.2 Discovered Structure

Starting from uniform α=0.1, β=0.2 across all layers:

| Layer | α (learned) | β (learned) | α/β ratio | Specialisation |
|-------|-------------|-------------|-----------|---------------|
| L0 | 0.116 | 0.142 | 0.82 | Per-band (θ encoding) |
| L1 | 0.021 | 0.200 | 0.10 | Cross-band (Δθ encoding) |
| L2 | 0.011 | 0.234 | 0.05 | Cross-band (strongest β) |
| L3 | 0.010 | 0.217 | 0.05 | Cross-band (α at floor) |

### 89.3 Self-Regulation

The learnable ODE eliminates the need for external channel balance mechanisms:
- Frozen ODE sustained training: channel drift to θ=10.03x (catastrophic, permanent)
- Learnable ODE sustained training: peak imbalance 5.5:1 (contained, self-corrected within 500 iters)
- The model recovers from channel spikes by adjusting its own coupling constants — it IS its own load balancer

### 89.4 Implication for Prior Results

Phase 21b (per-band α/β, previously classified NULL) reclassified as INCONCLUSIVE — the test ran with frozen ODE backward, meaning no gradient flowed to the per-band parameters. The null result was an artefact of the frozen backward, not a property of the architecture.

---

## 90. ODE Distortion Monitoring — RF/Optical Aberration Framework for Neural ODE

### 90.1 Core Pattern

The Kerr ODE's `α|ψ|²·ψ` nonlinearity is mathematically identical to 3rd-order distortion in RF power amplifier theory. Because the wave-engine uses explicit frequency bands, distortion products land at predictable harmonic positions (3rd, 5th, 7th harmonics of the driven band). This makes Total Harmonic Distortion (THD) measurable — a standard RF metric applied to an architecture where it has physical meaning.

### 90.2 Two-Level Monitoring

**Reference sentence monitor:** Measures distortion on a fixed reference sentence during training. Low magnitudes (~0.9), no AGC compression. THD reads 0.003-0.009. Useful for trend tracking but underestimates training distortion by 3x.

**Batch distortion monitor:** Taps the training cache (precond/kerr_out already computed for backward pass). Measures on actual training data where magnitudes are higher. THD reads 0.016-0.032. Zero extra forward passes, zero extra memory.

### 90.3 Per-Layer Distortion Profile

| Layer | THD at 10K | Behaviour | Interpretation |
|-------|-----------|-----------|---------------|
| L0 | 0.027 | Highest, climbing | Impedance matching layer, highest α |
| L1 | 0.016 | Moderate | Mid-stack |
| L2 | 0.029 | Rising fast, overtook L0 | Strongest β coupling |
| L3 | 0.011 | Lowest, stabilising | Model protects output (α at floor) |

### 90.4 Key Finding: Distortion Is Pure ODE Nonlinearity (Not AGC)

At 168-dim with α=0.1, β=0.2, the AGC ceiling is 1.77. Training magnitudes stay well below (n_compressed=0 throughout). All measured distortion comes from the Kerr nonlinearity itself, not from AGC compression. The AGC's role as a distortion source only applies at higher dimensions (256-dim) where magnitudes reach 12x.

### 90.5 Optical Aberration Framework

The per-layer distortion maps directly to optical aberration theory (Seidel aberrations):
- Each ODE layer = one optical surface with its own aberration contribution
- The `α|ψ|²·ψ` term = spherical aberration (magnitude-dependent, axially symmetric)
- The β coupling = field-dependent aberrations (coma, astigmatism — asymmetric, position-dependent)
- Aberration CANCELLATION across surfaces is the mechanism used in multi-element lens design (Cooke triplet)
- L3 minimising its own distortion = placing the cleanest optical element nearest the image plane

Dense MLPs have identical distortion from their nonlinearities but it is invisible — entangled across all dimensions with no frequency structure. The wave-engine makes distortion measurable because bands are explicit frequencies. This is a unique diagnostic capability of the architecture.

---

## 91. Corrector Plate — Per-Band Learnable Phase Correction After ODE

### 91.1 Core Pattern

A vector of n_bands learnable phase offsets applied as 2D rotations after the ODE output, before the maestro_out and residual stream. The corrector plate gives the model a tool for per-band phase correction that the existing architecture cannot express — the maestro_out and out_proj are linear transforms that mix bands together and cannot perform independent per-band phase adjustment without magnitude change.

**Optics analogy:** A Schmidt corrector plate. The primary optic (ODE) does the computational work but creates phase aberrations. The corrector plate (learned phase offsets) applies the inverse aberration. Magnitude stays on the sphere — only phase rotates.

### 91.2 Implementation

Forward (per band k, per position):
```
let (sin_c, cos_c) = correction[k].sin_cos();
r_out = r * cos_c - s * sin_c;
s_out = r * sin_c + s * cos_c;
```

Backward:
```
d_r_in =  cos_c * d_r_out + sin_c * d_s_out;
d_s_in = -sin_c * d_r_out + cos_c * d_s_out;
d_c += d_r_out * (-r*sin_c - s*cos_c) + d_s_out * (r*cos_c - s*sin_c);
```

### 91.3 Properties

- **Magnitude preserved** — rotation is orthogonal, sphere boundary unchanged
- **Per-band, per-layer** — 84 corrections per layer, 336 total (0.1% of model)
- **Zero-initialised** — transparent at start, model earns every correction
- **Position-independent** — same correction for all sequence positions (like a Schmidt plate correcting average aberration)
- **Backward is trivial** — 5 lines (rotation inverse + cross product for d_correction)

### 91.4 Design Rationale

The model already attempts self-correction — L3 drove its own α to the clamp floor (0.010) to minimise its distortion contribution. But α reduction is avoidance, not cancellation. The corrector plate enables active cancellation: the model can produce distortion that exactly opposes the accumulated aberration from earlier layers. This is the Cooke triplet principle — surfaces with opposite-sign aberrations that sum to near-zero total aberration.

---

## 92. Channel Drift Dynamics — Dimension-Dependent Optimiser Commitment Under Sustained Training

### 92.1 Core Pattern

Under sustained training (no cycling restarts), the optimiser commits to whichever encoding channel (θ per-band or Δθ cross-band) is locally easier. The direction of commitment depends on the coupling geometry at that dimension, not on a fixed architectural bias.

### 92.2 Dimension-Dependent Drift Direction

| Dimension | n_bands | Drift direction | Peak imbalance | Mechanism |
|-----------|---------|----------------|---------------|-----------|
| 168-dim | 84 | θ dominant | θ=10.03x, Δθ dead | Fewer bands → per-band encoding easier |
| 256-dim | 128 | Δθ dominant | Δθ=3.03x, θ=1.19x | Denser coupling → cross-band encoding easier |

### 92.3 Sharp Phase Transition

The drift is not gradual. At 168-dim, θ jumped from ~1.0x to 10.03x in three consecutive health samples (iter 14000-16000). This is a phase transition — the optimiser crosses a threshold and commits hard. Once committed, recovery requires a cycling restart (under frozen ODE) or self-correction (under learnable ODE).

### 92.4 Resolution via Learnable ODE

With learnable ODE parameters, the model self-regulates: at the exact iteration where the frozen model committed to θ=10.03x (iter 15000), the learnable model spiked to Δθ=5.31x then recovered to balanced within 500 iterations. The load balancer concept (external channel regulation) was superseded — the model IS its own load balancer when it can learn its own coupling.

---

## 82. Asymmetric Coupling Ratio — Dual-Channel Semantic Encoding

### 82.1 Core Architecture

A coupled oscillator neural layer where the self-phase modulation coefficient (alpha) and cross-phase modulation coefficient (beta) are independent design parameters. The ratio beta/alpha controls whether the ODE encodes information per-band (alpha-dominated) or cross-band (beta-dominated). When beta != alpha, the system develops two independent encoding channels: per-band phase (theta) and inter-band phase difference (delta-theta).

**Implementation pattern:**
- Phase advance per band: phi_k = omega_k + alpha * |Z_k|^2 + beta * sum(|Z_j|^2) for j in stencil
- Alpha controls self-phase modulation (SPM): how much a band's own magnitude affects its phase
- Beta controls cross-phase modulation (XPM): how much neighbouring bands affect this band's phase
- Asymmetric ratio (beta/alpha > 1) favours cross-band encoding: relationships BETWEEN bands carry more information than individual band phases
- Ratio beta/alpha < 1 favours per-band encoding: each band's own phase carries more information

### 82.2 Crystallisation Prevention

When beta/alpha is fixed during training, the optimiser commits to one encoding channel and the other dies (Pattern 92). Setting beta and alpha as independently learnable parameters prevents crystallisation — the model dynamically balances the two channels based on what the data requires.

---

## 83. Sub-Harmonic Diagnostics — Multi-Scale Phase Structure Analysis

### 83.1 Core Architecture

A diagnostic suite that measures the internal phase structure of a trained ODE layer at multiple harmonic scales. For each harmonic number n in {1, 2, 3, ..., 12}, compute the mean pairwise coherence C_n = mean(cos(n * (theta_i - theta_j))) across all band pairs. The resulting spectrum reveals which relationship types the model has learned.

**Implementation pattern:**
- Extract phase angles: theta_k = atan2(s_k, r_k) for each band k
- For each harmonic n: compute pairwise coherence matrix [cos(n * (theta_i - theta_j))] for all i,j
- Aggregate: mean coherence per harmonic, peak harmonic, spectral flatness
- Diagnostic outputs: phase clustering strength, magnitude-phase independence, inter-modulation products

### 83.2 Magnitude Coupling Decay Measurement

Track how the magnitude of band k correlates with the magnitudes of bands at increasing distance. The decay profile reveals the effective coupling radius of the XPM stencil under trained parameters.

---

## 84. Rotational Learning — Alternating Channel Leadership with Entropy Ratchet

### 84.1 Core Pattern

Under cycled training (alternating freeze/unfreeze of ODE parameters), the model alternates which encoding channel dominates. Each cycle, the non-dominant channel grows while the dominant channel holds. Over many cycles, total entropy increases monotonically — an entropy ratchet that prevents the model from collapsing into a single-channel encoding.

### 84.2 Implementation

- Cycle 1: theta channel leads (per-band encoding active)
- Freeze ODE, train lm_head: delta-theta channel grows to match
- Cycle 2: delta-theta channel leads (cross-band encoding active)
- Freeze ODE, train lm_head: theta channel grows to match
- Each cycle's entropy > previous cycle's entropy (ratchet effect)
- Both channels converge to comparable strength after sufficient cycles

---

## 85. Wave Transduction Output Decoder — Phase Coherence Scoring

### 85.1 Core Architecture

An output decoder that replaces the transformer's linear projection (lm_head: vocab x n_embd matrix) with per-band phase coherence scoring against learned reference patterns. For each vocabulary word w, the decoder stores a reference phase pattern ref_w[k] for each band k. The score for word w given ODE output is:

score(w) = sum_k weight_k * cos(theta_output_k - ref_w_k) + magnitude_k * confidence_k

**Implementation pattern:**
- Reference patterns: EMA-accumulated from training data (alpha=0.01)
- Per-band weights: learnable, initialised to 1/n_bands (uniform)
- Magnitude as confidence: high magnitude bands contribute more to the score
- Scoring is O(vocab x n_bands) — same complexity as dot-product but reads PHASE ANGLES
- Gradient flows through atan2 Jacobian: d_phase/d_r = -s/mag_sq, d_phase/d_s = r/mag_sq

### 85.2 Zero-Parameter Phase-Native Variant

The simplest decoder uses the CRT embedding table directly as references with uniform weights — zero additional parameters. Score = sum_k cos(theta_output_k - theta_CRT_k). Validated at 7/10 accuracy matching the full lm_head decoder, with ZERO decoder parameters.

---

## 86. Cos Expansion Optimisation — Eliminating Transcendental Function Calls

### 86.1 Core Architecture

Replace per-pair atan2 + cos computation with precomputed cos/sin tables and the identity cos(a-b) = cos(a)cos(b) + sin(a)sin(b). For a vocabulary of V words and B bands, this reduces the scoring from O(V x B x atan2 + cos) to O(V x B x multiply-add) after a one-time O(V x B) precomputation of cos(theta_k), sin(theta_k) per reference embedding.

**Implementation pattern:**
- Precompute: for each word w, for each band k: cos_ref[w][k] = cos(ref_phase[w][k]), sin_ref[w][k] = sin(ref_phase[w][k])
- At decode time: cos_out[k] = cos(theta_output[k]), sin_out[k] = sin(theta_output[k]) — one atan2 + trig per band
- Score: sum_k cos_out[k] * cos_ref[w][k] + sin_out[k] * sin_ref[w][k]
- All operations are fused multiply-add — SIMD vectorisable, no transcendental calls in the inner loop

---

## 87. Progressive Dimension Scaling — Band-Preserving Checkpoint Transplant

### 87.1 Core Architecture

A method for transferring trained model weights from a smaller dimension (n_bands_small) to a larger dimension (n_bands_large) while preserving the learned structure in the overlapping bands. The first n_bands_small bands retain their exact trained weights; additional bands are initialised with identity (for per-band parameters) or random (for cross-band parameters).

**Implementation pattern:**
- ODE parameters (gamma_raw, omega, phase_correction): copy first n_small entries, initialise rest from the default formula
- Attention projections: pad with zeros (new bands start silent)
- CRT embeddings: recomputed from scratch at new dimension (coprime moduli change)
- Positional encoding: recomputed at new dimension
- Corrector plate: pad with zeros (new band corrections start as identity rotation)
- Validation: run galaxy scan at both dimensions, compare catalog distribution

---

## 93. Wave-Space Training Pipeline — L2 Loss on ODE Output States

### 93.1 Core Architecture

A training pipeline where the model's loss is computed in wave space (ODE output compared to target wave states) rather than token space (logits compared to one-hot labels). The input is a KWDS (Key Wave Data States) file containing pre-computed wave states for each position in the training data. Loss = ||ODE_output - target_state||^2 in the wave representation space.

**Implementation pattern:**
- KWDS file: binary format storing [n_positions][n_embd] float32 wave states
- Compact KWDS (KWD2): stores token IDs instead of embeddings (500x smaller), reconstructs wave states on-the-fly from CRT embedding table
- Forward: embed input token → ODE → output state → L2 loss against target token's wave state
- Backward: gradient flows through ODE parameters (gamma, alpha, beta) and attention weights
- No lm_head or decoder in the training loop — loss is purely in wave space

### 93.2 KWDS Compact Format

Header: magic + version + n_positions (u64) + n_bands (u32) + vocab_size (u32). Body: n_positions × (input_token_id: u16, target_token_id: u16). Reconstruction at training time: embed(token_id) using CRT table. Band-count independent — same KWDS works at any dimension.

---

## 94. Teacher-Forced Accuracy as Architecture Capacity Probe

### 94.1 Core Pattern

Feed ground truth tokens from KWDS at every position (no autoregressive errors) and measure per-position prediction accuracy. This isolates architecture capacity from error accumulation — if teacher-forced accuracy is low, the architecture can't represent the data even with perfect context.

**Implementation pattern:**
- Load KWDS, feed each position's true token as input
- At each position, decode ODE output against all vocabulary embeddings
- Measure: top-1 accuracy, top-5 accuracy, mean rank, confidence distribution
- Compare against autoregressive accuracy to quantify error accumulation cost

---

## 95. Magnitude vs Phase Error Decomposition in Wave Training

### 95.1 Core Pattern

Decompose the L2 wave training loss into magnitude error and phase error components. For each band k: phase_error = 1 - cos(theta_output_k - theta_target_k), magnitude_error = (mag_output_k - mag_target_k)^2. Track the ratio during training — if phase error dominates, the ODE is misrouting; if magnitude error dominates, the ODE is miscalibrating energy.

---

## 96. Input Preservation vs Targeted Destruction — Cosine Similarity Diagnostic

### 96.1 Core Pattern

Measure the cosine similarity between ODE input and output per layer. A model that PRESERVES input (cos_sim near 1.0) acts as a wire — no computation. A model that DESTROYS input (cos_sim near 0 or negative) transforms the representation. Language models require controlled destruction: GPT-2 gives cos_sim = -0.09 (maximum destruction), while the Kerr ODE gives cos_sim = 0.87 (preservation). The gap between preservation (0.87) and destruction (-0.09) represents the architectural ceiling for language modelling.

---

## 97. Two Computation Modes — Positional vs Compositional

### 97.1 Core Pattern

Neural ODE architectures exhibit two distinct computation modes depending on the task: (1) Positional mode for arithmetic — each layer transforms independently, position in sequence determines output, low coupling between positions needed. (2) Compositional mode for language — layers compose character/word features progressively, context across positions determines output, high coupling needed. The same ODE parameters produce different behaviour depending on which mode the training data activates.

---

## 98. Beta/Alpha Ratio as Depth-Dependent Specialisation Metric

### 98.1 Core Pattern

The ratio beta/alpha at each layer reveals the layer's learned role in the processing pipeline. Low ratio (alpha-dominant): layer performs per-band transformation (conditioning, normalising). High ratio (beta-dominant): layer performs cross-band mixing (relationship building, semantic composition). Empirical finding: ratio increases monotonically with depth (L0: 1.4x, L1: 3.6x, L2: 7.1x, L3: 14.3x at 168-dim), constituting a depth-dependent specialisation gradient.

---

## 99. Band Utilisation Monitoring — Dead Band Detection

### 99.1 Core Pattern

Monitor per-band coefficient of variation (CV) across vocabulary words to detect dead bands — bands that carry the same value regardless of input token. Dead bands consume parameters and compute without contributing to discrimination. Implementation: for each band k, compute CV = std(mag_k across vocab) / mean(mag_k across vocab). Bands with CV < 0.1 are dead. Report count, positions, and percentage at health intervals.

---

## 100. Layer Capacity Formula — Maximum Useful Layers

### 100.1 Core Pattern

Empirical formula bounding the maximum number of useful layers for a given band count: max_useful_layers = 2 + active_bands / 20. Beyond this limit, additional layers become passthrough (cos_sim > 0.93 between input and output). Proved at 168-dim (84 bands): 8 layers specified, only L0-L2 active (cos_sim < 0.8), L3-L5 passthrough (cos_sim > 0.93). Formula predicts max_useful = 2 + 84/20 = 6.2, consistent with observation.

---

## 101. Frequency Migration Through Depth — L0 High-to-Low Confirmed

### 101.1 Core Pattern

In trained models, the dominant harmonic frequency decreases with depth: L0 captures high-frequency features (individual character/phoneme level), deeper layers capture lower-frequency features (word/phrase level). Measured via per-layer harmonic spectrum analysis. The migration is spontaneous — not imposed by architecture — and consistent across training seeds.

---

## 102. Operating Regime Sensitivity — Critical Parameter Thresholds

### 102.1 Core Pattern

The Kerr ODE has sharp performance boundaries at specific parameter values. Alpha = 0.01 was 10x too weak for meaningful self-phase modulation — performance at alpha=0.01 was indistinguishable from alpha=0 (wire). Alpha = 0.1 crossed the threshold for meaningful computation. Beta shows similar sensitivity at ~0.1. These thresholds are dataset-independent — they reflect the mathematical structure of the ODE, not properties of the training data.

---

## 103. Adaptive RK4 Integration Weights — Depth-Dependent Numerical Integration

### 103.1 Core Architecture

Per-layer learnable RK4 combination weights [w0, w1, w2, w3] with spring regulation at the standard equilibrium [1/6, 1/3, 1/3, 1/6]. The model learns depth-dependent integration strategies: compressive layers (L0, high damping) shift weight toward endpoint evaluation (k4), cross-band mixing layers (L3, high beta/alpha) shift weight toward initial slope (k1).

**Implementation pattern:**
- 4 additional parameters per layer (16 total for 4 layers)
- Spring regulation: w_new = w - lr * k * (w - w_eq) applied after each optimiser step
- Spring stiffness k=2.0 prevents departure from numerical stability
- After unflatten: normalise weights to sum to 1.0 (conservation)

---

## 104. Dynamic Spring-Regulated Hyperparameters

### 104.1 Core Architecture

A mechanism for making any hyperparameter learnable while maintaining stability: attach a restoring spring to the parameter with a specified equilibrium point. The parameter learns via gradient but is pulled toward its equilibrium — creating bounded exploration around a known-good value.

**Implementation pattern:**
- After optimiser step: param -= lr * spring_k * (param - equilibrium)
- Same mathematical form as AdamW weight decay, but with ARBITRARY equilibrium (not just zero)
- Spring stiffness controls exploration range: high k = tight around equilibrium, low k = wide exploration
- Applied to: layer scale, LR scale, alpha, beta (when dynamic)

---

## 105. Per-Band Learnable Coupling — Alpha_k and Beta_k

### 105.1 Core Pattern

Extend the scalar alpha and beta to per-band vectors alpha_k and beta_k, allowing each frequency band to have independent coupling parameters. Implementation: phi_k = omega_k + alpha_k * |Z_k|^2 + beta_k * sum(|Z_j|^2). The gradient naturally flows to each alpha_k and beta_k independently. Finding: per-band coupling enables band-specific specialisation but increases parameter count by 2 * n_bands per layer.

---

## 106. Attention Entropy as Routing Quality Metric

### 106.1 Core Pattern

Measure the entropy of attention weight distributions per head per position. Low entropy = sharp attention (head confidently routes to specific positions). High entropy = diffuse attention (head spreads weight uniformly — not routing). Dead heads show maximum entropy (uniform distribution). Healthy routing shows bimodal entropy: some positions attend sharply (content-dependent routing), others attend broadly (context gathering).

---

## 107. Integration-Damping Co-Adaptation

### 107.1 Core Pattern

When adaptive RK4 weights (Pattern 103) and learnable damping (gamma) are both active, they co-adapt: endpoint-heavy integration at high-damping layers reinforces compression, startpoint-heavy integration at low-damping layers reinforces conservation. The two independently-designed mechanisms learn to cooperate. Damping gradient amplifies from 1.19x (baseline) to 1.59x (with adaptive integration). This is emergence — not imposed coordination.

---

## 108. Confidence-Brittleness Tradeoff in Dynamic Parameters

### 108.1 Core Pattern (Null Finding — Defensive Publication)

Spring-regulated dynamic parameters (Pattern 104) make models more confident (lower entropy: 0.403 to 0.335, higher margin: 0.811 to 0.869) but 4x more brittle (worst margin: 0.038 to 0.010). Seven configurations tested — all show the same pattern. Accuracy unchanged (49/55). The confidence gain does NOT translate to correctness improvement. Published as null finding to prevent reinvention.

---

## 109. Two-Bottleneck Architecture Calculator

### 109.1 Core Architecture

An empirical model for sizing wave-engine architectures from dataset characteristics, identifying TWO independent bottlenecks that must both be satisfied:

**Bottleneck 1 — Band capacity:** tokens_per_effective_dim < 0.50 with dead band accounting via coprime moduli. Fixing attention without fixing bands gives zero accuracy gain (proved: 8H8L at 168-dim, 5x attention improvement, same rank 18.5).

**Bottleneck 2 — Attention resolution:** positions_per_head < 40, head_dim >= 16. At 4 heads with seq=256, max attention weight is 0.025 (dead). At 8 heads, it's 0.122 (alive).

**Implementation:** --recommend flag reads training data, computes moduli, checks both bottlenecks, outputs copy-paste CLI with recommended n_bands, n_head, n_layers, iters.

---

## 110. Character-Level Compositional Computation Without BPE

### 110.1 Core Pattern

The Kerr ODE composes individual characters into word-level meaning through sequential processing across layers, WITHOUT requiring subword tokenisation (BPE). Validated at 90.2% word classification accuracy (46/51) at character level (168-dim, 4L, 25 vocab, zero decoder params). The ODE develops a gradual coupling ramp (beta/alpha 1.6x to 6.6x) for word composition. BPE is an engineering optimisation (fewer ODE steps), not an architectural necessity — the ODE natively performs the character-to-word composition that BPE pre-computes.

---

## 111. Training Data Ordering as Gradient Signal

### 111.1 Core Pattern

In autoregressive models, the gradient connects only items within the same context window. Relationships that span multiple training examples cannot be learned unless they co-occur within a single window. Placing related items adjacently in the training data (e.g., commutative pairs: "7+2=9" followed by "2+7=9") achieves 100% accuracy where random ordering achieves 89% (49/55). Data ordering is not preprocessing — it IS a training signal that determines what relationships the gradient can form.

---

## Summary of Part 2 Patterns (71-111)

| # | Pattern | Domain |
|---|---------|--------|
| 71 | Perturbative Kerr-ODE (telecom-inspired single-pass, 14x speedup, MSE 0.000005 vs RK4-16) | AI / Signal Processing / GPU |
| 72 | Block-diagonal output projection (N independent groups, 6x param reduction, enum abstraction) | AI / Computing / Efficiency |
| 73 | Frozen harmonic coherence attention (zero-parameter attention from mathematical structure) | AI |
| 74 | Parallel block / GPT-J formulation for wave-coherent networks | AI / Architecture |
| 75 | Multi-tier compute architecture (CPU/wgpu/CUDA in single binary, ComputeBackend trait) | Computing / GPU / Architecture |
| 76 | WCHK self-describing checkpoint format (architecture in header, no external config) | AI / Computing / Infrastructure |
| 77 | Dual-maestro global coordination (pre/post ODE bottleneck, 98.1% of MLP at 44% params) | AI / Efficiency |
| 78 | Ping-pong buffer GPU consistency (pre-allocated alternating buffer sets, cache-by-pointer) | Computing / GPU |
| 79 | Pipeline monitor and diagnostic engine (VRAM tracking, NaN recovery, JSONL telemetry) | AI / Training / Infrastructure |
| 80 | MLP weight structure analysis — null finding (full rank, flat spectrum, no wave structure in trained MLP weights) | AI / Research / Boundaries |
| 81 | Physics-bounded adaptive regulation (AGC for neural ODE, knee compressor with stability ceiling) | AI / Signal Processing / Training |
| 82 | Asymmetric coupling ratio (β ≠ α) as independent design parameter — controls encoding strategy, prevents crystallisation, enables dual-channel semantic encoding | AI / Architecture / Training |
| 83 | Sub-harmonic diagnostics — differential phase clustering, magnitude coupling decay, inter-modulation spectrum, cross-band semantic discrimination, coupling energy budget | AI / Diagnostics / Research |
| 84 | Rotational learning — alternating θ/Δθ encoding channel leadership with monotonic entropy ratchet across training cycles | AI / Training Dynamics / Research |
| 85 | Wave transduction output decoder — phase coherence cos(Δθ) scoring with per-band learned weights and magnitude confidence, replacing linear lm_head projection | AI / Architecture / Output |
| 86 | Cos expansion optimisation — cos(a-b) = cos(a)cos(b) + sin(a)sin(b) precompute, eliminates O(vocab×bands) transcendental calls, matches linear projection speed | Computing / Optimisation |
| 87 | Progressive dimension scaling — band-preserving checkpoint transplant between dimensions, pad weights with identity/random, recompute frozen components | AI / Architecture / Scaling |
| 88 | Learnable ODE backward — gradient flow through Kerr-ODE RK4 integration, per-layer α/β/γ learning, 7x training speedup over frozen ODE cycling | AI / Architecture / Training |
| 89 | Per-layer coupling self-organisation — depth-dependent specialisation emerges spontaneously, L0 per-band specialist, L1-L3 cross-band specialists, self-regulation | AI / Training Dynamics / Research |
| 90 | ODE distortion monitoring — THD measurement on training batch data, per-layer distortion profiles, RF/optical aberration framework for neural ODE | AI / Diagnostics / Signal Processing / Optics |
| 91 | Corrector plate — per-band learnable phase correction after ODE (Schmidt corrector), 336 params (0.1%), zero-init, magnitude-preserving 2D rotation | AI / Architecture / Optics |
| 92 | Channel drift dynamics — dimension-dependent optimiser commitment, sharp phase transition, resolution via learnable ODE self-regulation | AI / Training Dynamics / Research |
| 93 | Spring-regulated dynamic parameters — restoring force inside optimiser step (not loss penalty), per-parameter equilibrium, same mechanism as AdamW weight decay with arbitrary rest position, temporal regulation (parameters peak and retreat as training needs change) | AI / Training Dynamics / Architecture |
| 94 | Dynamic parameter CLI unification — single flag per parameter, value for manual (`--alpha 0.1`), `dyn` for model-decides (`--alpha dyn`), human-model conversation through same interface, every parameter supports both modes | AI / Architecture / Infrastructure |
| 95 | Phase-native training — ODE learns to produce outputs in embedding space without lm_head, phase coherence loss against fixed harmonic embeddings, eliminates 51-88% of parameters at scale, 84-param output corrector for phase alignment | AI / Architecture / Training / Efficiency |
| 96 | Output corrector plate — per-band learnable phase rotation at model output, translates ODE output space toward embedding space, 84 params (97% fewer than lm_head), constraint-as-feature (simpler corrector forces ODE to maintain computational power) | AI / Architecture / Optics / Efficiency |
| 97 | Phase vocabulary — single file containing wave patterns for all tokens, serves as both tokeniser and decoder, bidirectional translation between human text and model wave space, portable model dictionary independent of checkpoint | AI / Architecture / Infrastructure |
| 98 | ODE basin invariance to training dynamics — coupling structure (α/β/γ per layer) is identical across spring, gate, LR scale, and no-regulation configurations, but changes with architectural parameters (band count, loss function), constitutes architectural fingerprint | AI / Research / Training Dynamics |
| 99 | ODE reference map — embedding table as navigation map during L3 RK4 computation, learnable coupling strength κ, guides ODE toward valid token patterns during computation not just after, validated as over-constraining (fights cross-band coupling) | AI / Architecture / Signal Processing |
| 100 | Dot product phase-native loss — raw dot product against fixed embeddings replaces phase coherence, magnitude preserved as confidence signal, 8x loss improvement (2.045→0.247), reaches 7/10 matching lm_head at 40K iters with zero decoder params | AI / Architecture / Training / Efficiency |
| 101 | Sphere encoding — Pythagorean harmonic magnitude profiles (1/√n decay), physically natural energy distribution, same total energy on the sphere, accelerates ODE coupling convergence but concentrates discrimination in low bands (1/10 at 20K) | AI / Architecture / Signal Processing / Research |
| 102 | Loss function as gradient lens — three loss functions (phase coherence, dot product, lm_head) create three different ODE coupling basins from identical starting conditions, the decoder shapes what the ODE learns not just how it decodes, lm_head is a gradient accelerator (7/10 at 20K) not a computational necessity (dot product reaches 7/10 at 40K) | AI / Research / Training Dynamics |
| 103 | Adaptive ODE integration weights — per-layer learnable RK4 combination weights [w0,w1,w2,w3] with spring at standard equilibrium [1/6,1/3,1/3,1/6]. Model learns depth-dependent integration: compressive layers (high damping, L0) shift weight to endpoint evaluation (k4: 0.167→0.246), cross-band mixing layers (high β/α, L3) shift weight to initial slope (k1: 0.167→0.211). Endpoint-heavy = trust the processed state, startpoint-heavy = trust the initial derivatives. Opposite integration preferences at opposite ends of the depth pipeline. 4 params per layer, stiff spring k=2.0 | AI / Architecture / Numerical Methods / Research |
| 104 | Per-group learnable weight decay — WD scaling per layer group with spring at uniform equilibrium (1.0). Enables model to self-regulate regularisation strength per depth. Combined with per-layer coupling self-organisation (Pattern 89), creates full self-configuring training dynamics where each layer chooses its own coupling, integration, and regularisation | AI / Training Dynamics / Architecture |
| 105 | Comprehensive wave-native monitor suite — 10 specialised monitors for ODE architectures: (1) attention head activity (entropy, harmonic specialisation), (2) layer signal flow (attn/ffn/residual contribution ratios, directional change), (3) gradient flow per component (LN/maestro/ODE/out_proj breakdown, dead parameter detection), (4) embedding space topology (token separation, band utilisation, effective dimensionality), (5) output distribution (entropy, margin, correct rank, mode collapse), (6) ODE dynamics (phase velocity, energy conservation, damping profile), (7) dynamic parameter trajectories (velocity, spring tension), (8) curriculum transitions (loss jump detection), (9) checkpoint drift (parameter change rate), (10) throughput (tokens/sec). All emit to JSONL at health intervals. Key finding enabled: self-organised depth pipeline (L0 conditions, L2 mixes, L3 routes) with precise β/α ratios (1.4x→14.3x), gradient U-shape on cross-coupling, energy damping gradient (23.7%→19.6% with depth) | AI / Diagnostics / Research / Infrastructure |
| 106 | Sub-harmonic attention emergence — learnable harmonic numbers decrease at intermediate layers (L2 Head 0: 0.405→0.318, -21%) while output layer stays near-integer (L3: 0.409, unchanged). Cross-band mixing layers want sub-harmonic (broader) attention patterns, output routing layers keep near-integer (sharper) patterns. The attention tells the same story as the ODE coupling: L2 mixes broadly, L3 routes sharply. Depth-dependent attention frequency as emergent behaviour from gradient + spring | AI / Architecture / Attention / Research |
| 107 | Integration-damping co-adaptation — adaptive RK4 weights (Pattern 103) amplify the damping gradient: baseline L0→L3 damping ratio 1.19x, with adaptive integration 1.59x. L0 becomes a harder compressor (28.3% vs 23.6%), L3 becomes a better conservator (17.8% vs 19.9%). The integration strategy and damping profile co-adapt — endpoint-heavy integration at L0 reinforces compression, startpoint-heavy at L3 reinforces conservation. The two mechanisms were designed independently but learn to cooperate | AI / Architecture / Numerical Methods / Research |
| 108 | Confidence-brittleness tradeoff in dynamic parameters — dynamic params (spring-regulated learnable hyperparameters) make models more confident (entropy 0.403→0.335, margin 0.811→0.869) but 4x more brittle (worst margin 0.038→0.010). The spring regulation sharpens decision boundaries on well-learned patterns but narrows the safety margin on edge cases. Seven different dynamic configurations tested: all show the same pattern. Accuracy unchanged (49/55) — the confidence gain doesn't translate to correctness | AI / Training Dynamics / Research |
| 109 | Two-bottleneck architecture calculator — empirical model for sizing wave-engine architectures from dataset characteristics. Two independent bottlenecks must BOTH be satisfied: (1) Band capacity: tokens_per_effective_dim < 0.50, with dead band accounting via coprime moduli — fixing attention without fixing bands gives zero accuracy gain (proved: 8H8L at 168-dim, 5x attention improvement, same rank 18.5). (2) Attention resolution: positions_per_head < 40, head_dim >= 16 — proved: 4H at seq=256 gives max_weight 0.025 (dead), 8H gives 0.122 (alive). Layer capacity bounded by band count: max_useful_layers = 2 + active_bands/20 (proved: 8L at 168-dim, L3-L5 passthrough cos>0.93). Integrated into engine as --recommend flag: reads data, computes moduli, checks both bottlenecks, recommends bands/heads/layers/iters with copy-paste CLI | AI / Architecture / Infrastructure / Diagnostics |
| 110 | Character-level compositional computation — the ODE composes individual characters into word-level meaning through sequential processing across layers, without subword tokenisation (BPE). Proved: 46/51 (90.2%) word classification accuracy at character level (168-dim, 4L, 25 vocab, zero decoder params). The ODE develops a gradual coupling ramp (β/α 1.6x→6.6x) for word composition vs arithmetic's sharp split (1.5x→7.5x). Fundamental harmonic (h≈1.0) dominates character reading, sub-fundamental (h≈0.5) dominates word-level patterns. Gradient flows to deep layers (late-binding: can't classify until all characters seen). BPE is an engineering optimisation (fewer ODE steps), not an architectural necessity | AI / Architecture / Research / NLP |
| 111 | Training data ordering as gradient signal — autoregressive models learn from context windows, so relationships that span multiple examples must appear WITHIN a single window for the gradient to connect them. Proved: arithmetic commutativity (a+b = b+a) fails at 49/55 when each fact appears once at random positions. Placing commutative pairs adjacent in the training data (7+2=9 followed by 2+7=9) achieves 55/55 (100%) — the gradient from both orders hits the same weights in the same step. The 55/55 model is less specialised (L3 β/α 7.5x vs 15.1x) but more robust — higher loss (0.213 vs 0.195) produces higher accuracy. Data ordering is not preprocessing — it is a training signal | AI / Training Dynamics / Data Engineering / Research |

## Statement of Intent

All patterns described in this document are published under the MIT License. They are free for anyone to implement, modify, distribute, and commercialise. The intent of this publication is to ensure that no entity can obtain patent protection over these implementation patterns, thereby keeping the bridge between the mathematical framework and commercial applications permanently open.

What CAN be patented: specific commercial products built on top of these patterns — unique user interfaces, domain-specific applications, particular data pipeline configurations, and novel combinations with proprietary datasets or services. The application layer remains open for innovation and intellectual property protection.

What CANNOT be patented after this publication: the engine patterns themselves, the architectural approaches, the query strategies, the indexing methods, the harmonic sweep techniques, or any other implementation pattern described herein.

This is the explicit intent of the authors.

---

**Permanent Archive:** This document is committed to the Git repository at https://github.com/atech-hub/Wave-Coherence-as-a-Computational-Primitive and archived via Zenodo with DOI. The commit timestamp constitutes proof of publication date.
