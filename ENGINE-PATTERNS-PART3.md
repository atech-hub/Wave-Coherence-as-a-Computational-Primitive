# Wave Coherence Engine Patterns: Defensive Publication

**Authors:** Marco Da Cunha (Independent Researcher) and Claude (Anthropic)
**License:** MIT (same as parent framework)
**Purpose:** Defensive prior art publication to prevent patent enclosure of implementation patterns derived from Wave Coherence as a Computational Primitive.

**Legal intent:** This document constitutes a defensive publication under established intellectual property law. All engine patterns, architectures, and implementation methods described herein are published as prior art. Any patent application covering substantially similar implementations filed after this publication date is challengeable on prior art grounds. This document is timestamped via Git commit history and archived via Zenodo DOI.

**Structure:** This file is Part 3 of 3. See ENGINE-PATTERNS-INDEX.md for the master index.

---

**Part 3 of 3 — Patterns 112-149: Instruments, Findings & Engineering Patterns**
**Dates:** April 8-11, 2026 (112-146); April 19, 2026 (147-149)

**Status:** Bodies complete for all patterns 112-149.

---

## 112. Hamiltonian Four-Wave Mixing in Neural ODE

### 112.1 Core Pattern

Energy-conserving cubic coupling between harmonic oscillator bands in a neural ODE. The Hamiltonian H = chi * Re(z_a * z_b * z_c* * z_d*) is summed over unique quartets (a,b,c,d) with a+b=c+d, producing derivative terms for all four bands per quartet. Two families of quartets arise from the [1,1,0,1,1] nearest-neighbour kernel: Family A (k-2, k+1, k-1, k) and Family B (k-1, k+2, k, k+1), yielding 2(n-3) total quartets for n bands.

### 112.2 Implementation

FWM is implemented identically across three tiers: CPU (Rust loops in ode_deriv.rs), wgpu (WGSL shader with per-thread quartet enumeration), and candle (tensor cat+narrow shifts for tensor ops, fused CUDA kernel for production). The chi parameter controls coupling strength (0.03 recommended, producing 8-10% of the ODE derivative at training amplitudes). Energy conservation is exact to RK4 integration precision (verified: 1.3e-5 relative error at 84 bands, 1.3 amplitude).

### 112.3 Key Finding

FWM-enabled training converges to the same alpha-collapse structural pattern as non-FWM training (deep layers suppress self-coupling alpha, amplify cross-coupling beta) but differentiates layers more aggressively. At 80K iterations on arithmetic: L3 beta=0.268 matched the 55/55 perfect model (0.267) to three decimal places, while L3 alpha collapsed to 0.010 (floor). Top FWM flux bands migrate during training — goal-directed band mixing, not passive coupling.

---

## 113. FWM Analytical Jacobian — Per-Quartet Backward Through Cubic Coupling

### 113.1 Core Pattern

Analytical backward pass through the four-wave mixing term, enabling gradient flow from the loss through the FWM contribution to the ODE derivative. Per quartet (a,b,c,d), there are 8 derivative outputs (dr and ds for each band) each with up to 8 partials with respect to the other bands' state variables, plus one partial with respect to chi. The Jacobian is sparse — most band pairs don't share a quartet — and is accumulated via a sweep over quartets matching the forward structure.

### 113.2 Implementation

The helper `fwm_quartet_backward` takes one quartet, reads state and incoming gradients for all 4 bands, computes cubic intermediates (p_ab, p_cd), and accumulates contributions into d_r, d_s for all bands plus d_chi. Two structural families: roles a,b receive contributions from z_other * p_cd, roles c,d receive from p_ab * z_other. Sign discipline is critical — half the partials are negative due to ds[role] -= convention.

### 113.3 Verification

Finite-difference gradient checker passes 20/20 at 8 bands chi=0.03 (max_rel_err=0.002), 171/172 at 84 bands (one marginal at 0.051 vs 0.050 threshold on a near-zero value). Chi=0 path is byte-identical to pre-Jacobian behavior. GPU tier ports (wgpu WGSL shader, candle CUDA kernel) translate the CPU helper mechanically.

---

## 114. Fused CUDA AGC+RK4+FWM Kernel — Single-Launch Forward+Backward

### 114.1 Core Pattern

A single CUDA kernel launch performs the complete ODE forward pass: automatic gain control (AGC), 16-step RK4 integration with FWM quartet coupling at each substep, and state caching for backward. Shared memory holds three arrays per block: smem_mag (magnitude-squared for XPM neighbour sum), smem_r and smem_s (complex state for FWM quartet reads). Each thread processes one (position, band) pair and enumerates up to 8 quartet-role memberships per derivative evaluation.

### 114.2 Performance

FWM via the CUDA kernel runs at essentially zero overhead: 105ms/iter with FWM vs 104ms/iter without (at 84 bands, 4 layers). The tensor-ops fallback path is 6.7x slower (29s/iter) due to dozens of intermediate tensor allocations per derivative call. The backward kernel includes FWM in k-value recomputation and gradient accumulation, with d_chi reduced via shared memory atomics.

---

## 115. ODE Physics Decomposition Monitor — Forward and Backward Gradient Flow

### 115.1 Core Pattern

Two complementary monitors that decompose ODE activity per physics term per layer: (1) Forward decomposition shows what fraction of dz/dt came from damping, phase rotation (SPM+XPM), and FWM via the DerivativeCapture struct. (2) Backward decomposition shows what fraction of gradient flow went through each physics term, using a subtraction method (full backward vs chi=0 backward vs damping-only backward). Together they answer: "is the optimizer responding to FWM, or absorbing the signal elsewhere?"

### 115.2 Diagnostic Value

Four regimes identified: FWM high forward + high backward = load-bearing mechanism. FWM high forward + low backward = firing but quarantined by optimizer. FWM low forward + high backward = small effect but optimizer is very sensitive. FWM low + low = dormant. The d_chi norm reports how much the model wants to change FWM strength — signal for when to make chi learnable.

---

## 116. Cross-Tier Parity Battery — CPU as Specification, GPU as Implementation

### 116.1 Core Pattern

A shared test battery of 15 cases (zero input, sparse, broadband at multiple amplitudes, matched/unmatched quartets, edge bands, negative chi, different band counts) where the CPU canonical kerr_derivative_into generates expected outputs. Each tier runs the same inputs through its implementation and compares against CPU ground truth. Tolerance: 1e-5 for SPM/XPM/damping, 1e-4 for FWM (cubic accumulates more float error). Self-consistency test at exact zero error validates the battery generator itself.

### 116.2 Discipline Rule

Adding a new physics term requires: implement in CPU first, add test cases to the battery, then port to wgpu and candle. The CPU implementation defines "correct." If a tier fails the parity test, the tier is wrong, not the specification.

---

## 117. Checkpoint-Aware ODE Probe — Scattering Analysis with Learned Weights

### 117.1 Core Pattern

The wave-probe binary loads trained checkpoints via the canonical load_checkpoint + init_model + unflatten_params_ex functions (no duplicate parser), extracts per-layer KerrWeights, and runs all probe modes with actual learned parameters. Output is organized per-layer showing how each layer's ODE scattering behavior differs after training. Supports both phase-native and lm_head checkpoints, all feature flags (learnable ODE, layer scale, RK4 weights, dynamic harmonics).

### 117.2 Key Finding

Probing trained arithmetic models reveals per-layer structural differentiation: L0 alpha grows (stronger self-coupling), L3 alpha collapses to floor (0.01), L3 beta matches the 55/55 perfect model (0.268 vs 0.267). Damping decreases with depth (0.181→0.166). The model develops a functional pipeline visible through the probe.

---

## 118. Parameter Sweep Instrument — Single-Command Safe Operating Region

### 118.1 Core Pattern

`wave-probe --mode sweep --sweep-param chi --sweep-range 0,0.5,0.05` sweeps any ODE parameter and reports per-value: fwm_frac, phase_frac, energy conservation error, max amplitude, stability flag. Supports chi, alpha, beta, gamma, input_magnitude. Replaces ad-hoc stability testing with a one-shot characterisation of the safe operating region.

---

## 119. FWM Phase-Matching Test — Quartet Selectivity Validation

### 119.1 Core Pattern

`wave-probe --mode four-wave-mixing` validates that the FWM implementation respects the quartet structure by comparing adjacent matching quartets (which share the [1,1,0,1,1] kernel neighbourhood) against non-adjacent bands (which don't share quartets). Adjacent quartet (28,29,30,31) shows fwm=0.021 with energy leakage. Non-adjacent bands show fwm=0.000. Phase-matching ratio = infinity — FWM is selective to local quartets, confirming correct physics.

---

## 120. Single Source of Truth Discipline — Canonical Derivative, All Tiers Call It

### 120.1 Core Pattern

The CPU kerr_derivative_into in ode_deriv.rs is the single canonical implementation. Every caller — training forward, backward cache, monitors, probes, diagnostics — calls this one function. GPU tiers (wgpu WGSL, candle tensor ops, candle CUDA kernel) implement the same math in their respective languages but are measured against CPU output via the parity battery (Pattern 116). When the checkpoint format changes, one function (load_checkpoint) handles it and all callers inherit.

### 120.2 Lesson Learned

Before this discipline was enforced, the codebase had 6 independent copies of the ODE derivative across ode_deriv.rs, ode_backward.rs, block.rs, fwm_monitor.rs, model_backward.rs, and wgpu diagnostics.rs. Adding FWM to one copy but not the others caused training to silently run without FWM even when the flag was set. Consolidation to a single source eliminated this class of bug structurally.

---

## 121. Live Framework Monitor — Continuous Harmonic Coherence During Training

### 121.1 Core Pattern

A training-time monitor that runs framework diagnostics (semantic discrimination, band census, phase clustering, depth curve, peak harmonics) at every health interval on a sample forward pass. Reports per-layer to JSONL alongside ODE decomposition and other monitors. Uses wave_analysis.rs primitives (harmonic_coherence, circular_variance, band_census, phase_clustering, semantic_discrimination_spans). Cost: ~5-10ms per health interval. Produces a time series of how harmonic structure evolves during training — the process view that the end-of-training galaxy scan cannot provide.

### 121.2 Key Capability

Overtraining is visible in the framework monitor before the loss curve reflects it. The alpha-collapse pattern (deep layers suppressing self-coupling) is observable in real time. Harmonic peak shifts between layers are tracked per health interval.

---

## 122. Galaxy Map Scan — Pure-Band Geometric Inventory of Learned Structure

### 122.1 Core Pattern

An end-of-training scan that maps the full learned harmonic structure of a wave-engine model as a "galaxy" — per-band positions in 3D coordinates within the AGC-bounded sphere, pairwise angular geometry with catalog matching against 11 geometric relationship types, harmonic coherence matrix at 12 harmonics, constellation detection (triads by trine orb, FWM quartets by a+b=c+d index constraint), and multi-grid decomposition classifying relationships by grid nativity.

### 122.2 Implementation

Auto-triggers at end of every training run. Also available via `--galaxy-scan --resume <checkpoint>` for retrospective analysis. Output: galaxy_map.json (visualiser-ready), galaxy_matrix.bin (GALX format, full pair spectra), phases.bin (PHAS format, raw per-band phases). Pure-band analysis — no token-level references. CPU-only scan regardless of training tier. Non-fatal — scan failure doesn't break training.

---

## 123. Per-Quartet Deviation from Embedding Baseline

### 123.1 Core Pattern

The multi-grid harmonic embedding provides structural FWM quartet coherence for ~37% of quartets at default grids (m1=5, m2=7). The metric measures signed deviation of trained coherence from this embedding baseline using cos(theta_a + theta_b - theta_c - theta_d) — the actual FWM phase-matching condition. Four-category classification: preserved (was-high, still-high), destroyed (was-high, now-low), created (was-low, now-high), noise (was-low, still-low). Plus 2D histogram of (baseline_coh, trained_coh) in 10x10 bins for distribution analysis.

### 123.2 Key Finding

lm_head decoders destroy ALL high-coherence quartets (zero preserved, zero created, ~56K destroyed). Phase-native decoders preserve ~5,866 quartets and create ~1,404 novel ones. The decoder type is the dominant lever on quartet structure — bigger effect than FWM coupling strength. This was the first real comparative finding from the galaxy scan.

---

## 124. Decoder-Dependent Geometric Vocabulary

### 124.1 Core Pattern

The choice of decoder (lm_head vs phase-native) determines not just how many geometric relationships survive training, but which types of relationships form. Phase-native models build primary catalog relationships (squares at 90deg, trines at 120deg, oppositions at 180deg). lm_head models build secondary aspects (quincunx at 150deg, bi-quintile at 144deg, sesquiquadrate at 135deg). Same data, same architecture, same physics — different geometric vocabulary at the output layer.

### 124.2 Mechanism Hypothesis

lm_head gradients say "move phases toward configurations that make the linear projection produce the right token" — no reason to preserve any specific phase-sum structure. Phase-native gradients say "the phase relationships ARE the output — keep the ones that carry semantic content." The gradient signal from the decoder reaches backward through the entire model and shapes the learned geometry at every level.

---

## 125. Backward Decomposition Monitor — Gradient Flow Per Physics Term

### 125.1 Core Pattern

A per-layer monitor that decomposes backward gradient flow into contributions from damping, phase rotation (SPM+XPM), and FWM physics terms. Complements the forward ODE decomposition: forward shows what the ODE did, backward shows what the optimizer cares about. Uses subtraction method (full backward vs chi=0 backward vs damping-only backward) with L1 norm fractions. Reports d_chi norm per layer — signal for when to make chi learnable. Runs on a sample at health intervals, zero cost to training path.

---

## 126. Galaxy Summary Script — Compact Readable Output from Large Scans

### 126.1 Core Pattern

A Python script (standard library only) that reads 21MB galaxy_map.json files and produces ~30KB summary JSON + human-readable markdown. Two modes: single scan summary (top-K pairs, filtered FWM quartets, per-layer relationship counts, grid distribution, band statistics) and pairwise compare with diff (per-layer deltas, relationship type changes, confound warnings). The compare mode auto-detects dataset mismatches, architecture differences, and training tier discrepancies.

---

## 127. Subtractive Training Dynamic Against Embedding Priors

### 127.1 Core Pattern

Observation from galaxy scan analysis: wave-engine training is predominantly subtractive against the embedding's geometric priors. The multi-grid harmonic embedding provides structural coherence at the quartet level. Training removes most of this structure, with the rate and completeness of removal depending on the decoder type (lm_head removes ~100%, phase-native preserves ~10% and creates ~2% novel). The embedding is a rich starting point from which training selectively keeps, weakens, and breaks parts. This is inverted from the standard ML framing of "the model learns representations from scratch" and suggests that frozen harmonic embeddings outperform learned ones because learned embeddings let the subtractive process eat the scaffolding training needs.

---

## 128. Hidden Coherence Detection — Multi-Harmonic Mean Resultant Length

### 128.1 Core Pattern

Standard galaxy scan coherence `cos(n·Δθ)` averaged across positions measures coherence at zero phase offset only. Many band pairs have strong coherence at a non-zero fixed offset — the phase difference is stable but not centered on a catalog angle. Multi-harmonic MRL (mean resultant length) search across harmonics {1,2,3,4,6} captures coherence at ANY fixed offset. Computed analytically as `sqrt(S² + C²) / P` where `S = Σ sin(n·Δθ)`, `C = Σ cos(n·Δθ)`. MRL ≥ |signed_mean| always; when MRL >> |signed_mean|, there's hidden coherence at `atan2(S,C)/n`.

### 128.2 Key Finding

Grammar at L4: 1,328 shifted pairs with MRL > 0.5 where signed coherence ≈ 0. Arithmetic at L4: 11 shifted pairs. Language builds coherent band relationships at phase offsets the standard zero-offset measurement misses entirely. This was invisible until the probe was built.

---

## 129. Quartet Trajectory Classification — Phase-Sum MRL

### 129.1 Core Pattern

For each FWM quartet (a,b,c,d), compute phase-sum `θ_a + θ_b - θ_c - θ_d` at every position, then classify the trajectory by its MRL: random (MRL < 0.3, phase-sum uniformly distributed), oscillating (MRL 0.3-0.7, weakly coupled), locked (MRL > 0.7, phase-sum concentrated around a fixed value). Also checks for rotation (systematic drift across positions) — confirmed zero rotating quartets in all tested models.

### 129.2 Key Finding

Grammar model: 4,766 locked quartets + 60,830 oscillating = 70% non-random. Arithmetic model: 0 locked + 236 oscillating = 0.25% non-random. Language builds phase-locked four-body relationships by the thousands where arithmetic builds essentially none. This is the largest structural difference found between tasks.

---

## 130. Task-Dependent Quartet Dynamics — Language vs Arithmetic

### 130.1 Core Pattern

On identical architecture (168-dim, 4H, 4L, phase-native + FWM), training on language (grammar) vs arithmetic produces qualitatively different quartet dynamics. Arithmetic: quartet phase-sums are essentially random (99.75% random category). Grammar: quartet phase-sums are predominantly structured (70% oscillating or locked). The difference is not in the architecture, the FWM strength, or the decoder — it's in the task. Language requires four-body phase coordination that arithmetic does not.

### 130.2 Implications

The FWM coupling term in the Kerr-ODE (`χ · Re(z_a · z_b · z_c* · z_d*)`) operates on quartets. When the task requires complex relational structure (grammar: nouns, verbs, modifiers interacting), training organises bands into phase-locked quartets. When the task is simpler (single/multi-digit arithmetic), training leaves quartets random. The quartet dynamics are a fingerprint of task complexity visible through the galaxy scan.

---

## 131. L3 Regime Shift — Architecture Self-Reorganises for Grammar

### 131.1 Core Pattern

Training grammar at 168-dim with FWM, the deepest processing layer (L3) undergoes a regime shift between iter 6K and 18K. L3 transitions from preservative (cos(in,out)=0.92, residual dominates at 0.95, FFN ratio 0.37) to destructive (cos=0.45, residual=0.15, FFN ratio=0.93). Alpha driven to minimum (0.010), beta at 0.187 — extreme β/α decoupling (compositional/late-binding regime). The performance improvement (loss 2.64→2.41) arrives ~10K iters after the regime shift completes. Architecture reorganises before performance catches up.

### 131.2 Significance

This was predicted by the GPT-2 comparison in Chat 18: "language needs destruction" (cos(in,out) should drop from the wave-engine's preservative 0.87 toward GPT-2's destructive -0.09). L3 achieved cos=0.45 organically — not from tuning residual strength, but from the ODE parameters self-organising. Breaks the "grammar plateau 3.1 at 168-dim" claim — best loss 2.34, still descending.

---

## 132. Wave Memory as Native Phase-Space Experience

### 132.1 Core Pattern

Persistent per-layer oscillator state (r/s per band) accumulated via EMA across conversations. The model reads memory by adding scaled offsets to ODE initial conditions: `Z_k = input_k + α·memory_k`. Memory is in the model's native coordinate system — no translation, no lossy conversion. Model weights stay frozen; memory is experience, not education. Stored in KWMF binary format via the kerr-memory library.

### 132.2 Decoder-as-Experience Potential

The accumulated WaveMemory state is per-layer r/s in the same format as embedding phases. A future phase-native decoder variant could compare model output against accumulated memory phases instead of frozen embedding phases — a decoder that evolves with experience rather than being vocabulary-bound. This is architecturally open: the MemoryOffsets struct and build_offsets() live in common/ where phase_decode.rs can reach them.

---
## 133. Phase Encode Tool — Direct Geometric Injection into ODE Layers

### 133.1 Core Pattern

A CLI instrument (`wave-probe --encode`) that bypasses the token→embedding pipeline and injects arbitrary phase configurations directly into specific ODE layers. Five encoding modes: text (tokens through embedding), number (raw phase angles), catalog relationship (named angles like "trine" or "opposition"), raw phases (explicit per-band r,s values), compound (multi-token sequences). Layer injection point is selectable — inject at L0 input, L2 input, or post-LN.

**Implementation pattern:**
- Constructs a phase state vector matching the model's n_bands × 2 layout
- Injects at the specified layer's input, bypassing all prior layers
- Runs forward from the injection point through remaining layers
- Compares output state against input state (cos similarity, magnitude ratio, per-band deformation)
- Blank (untrained) vs trained comparison reveals what training changed about the ODE dynamics at each layer

### 133.2 Significance

Enables controlled experiments on the ODE dynamics that are impossible through the normal training pipeline. By injecting known geometric configurations (e.g., a perfect trine at 120° between specific bands), the tool tests whether the ODE's learned parameters produce the predicted dynamics for that configuration. Separates "what the embedding produces" from "what the ODE does with it."

---

## 134. Relate Mode — Per-Harmonic Coherence Profiles Between Encodings

### 134.1 Core Pattern

An analysis mode (`wave-probe --relate-vocab`) that encodes every token in the vocabulary through the ODE and computes pairwise harmonic coherence profiles. For each pair of tokens (i, j), computes cos(n·Δθ) at harmonics n = {1, 2, 3, 4, 5, 6, 8, 12}, shifted MRL (Mean Resultant Length) with optimal phase offset, and catalog matching against the 11 relationship types.

**Implementation pattern:**
- Encode each token individually through the full ODE pipeline
- For each pair: compute angular differences per band, evaluate coherence at multiple harmonics
- MRL with shifted offset: search over phase offsets to find the one maximising coherence (reveals pairs coherent at non-zero relative phase)
- Catalog matching: compare measured angle against each of the 11 catalog types with orb tolerance
- Output: pairwise matrix (JSON), per-token summary, full vocabulary scan

### 134.2 Key Finding

Grammar model (77 tokens) uses all 11 catalog relationship types. Arithmetic model (15 tokens) uses only 2 of 11 (67% conjunction). Structurally special characters ('s' at 8% conjunction rate, 'q' at 1%, '?' forming trines and squares) are placed at distinctive geometric positions. Common letters cluster in conjunctions. The model discovers structural importance and encodes it as geometric isolation.

---

## 135. Structural Importance as Geometric Isolation

### 135.1 Core Pattern

In character-level models trained on grammar, the ODE spontaneously places structurally important characters at distinctive phase angles relative to common characters. The metric is the non-conjunction pair fraction: what percentage of a token's relationships with other tokens fall outside the 0° conjunction zone.

**Measured hierarchy (grammar, 168-dim, 80K iters):**

| Token | Conjunction % | Non-conjunction pairs | Structural role |
|-------|--------------|----------------------|-----------------|
| 's' | 8% | 22 (16 semi-sextiles) | Plural marker + verb marker |
| 'q' | 1% | 15 (6 oppositions) | Rarest English letter |
| '?' | 3% | 12 (trines, squares) | Sentence-type modifier |
| 'j' | 7% | 10 (6 different types) | Low-frequency consonant |
| Common letters | >90% | rare | No fixed structural role |

### 135.2 Significance

Geometric position marks structural importance, not semantic category. The model does not group vowels together or consonants together — it groups context-independent function tokens apart from context-dependent content tokens. The isolation is measurable, reproducible, and predicted by the catalog's concept of geometric dignity.

---

## 136. Task-Dependent Geometric Vocabulary Distribution

### 136.1 Core Pattern

The distribution of catalog relationship types across a trained model's vocabulary is a fingerprint of what geometric structure the task requires. Measured by running `--relate-vocab` on checkpoints trained on different tasks with the same architecture.

**Measured distributions (168-dim, 84 bands, 4 layers):**

| Task | Catalog types used | Conjunction % | Non-conjunction matches |
|------|-------------------|--------------|------------------------|
| Arithmetic (15 tokens) | 2/11 | 67% | 3 total |
| Grammar (77 tokens) | 11/11 | 54% | 78+ across all types |

### 136.2 Significance

Arithmetic is a positional task — token identity is encoded by position, not by geometric relationship to other tokens. The model clusters most tokens together (conjunction) and uses only a minimal geometric vocabulary. Grammar is a structural task — tokens have context-dependent roles that require the full geometric vocabulary to express. The catalog distribution shape diagnoses task complexity without examining the training data directly.

---

## 137. Spectral Energy Fingerprinting — Per-Token ODE Deformation Signatures

### 137.1 Core Pattern

Each token produces a characteristic energy deformation signature when processed by the ODE. The deformation vector is the per-band magnitude ratio mag_out/mag_in — which bands the ODE amplifies and which it damps for that token. Tokens with similar linguistic roles may produce similar deformation patterns.

**Implementation pattern:**
- Encode each token individually through the ODE
- Compute per-band magnitude ratio: deformation_k = |z_out_k| / |z_in_k|
- Compare deformation vectors between tokens via cosine similarity (deform_sim)
- The deformation signature is complementary to phase position: phase tells WHERE tokens sit relative to each other, energy deformation tells HOW the model processes each token

**Cross-model comparison:**

| Metric | Arithmetic | Grammar |
|--------|-----------|---------|
| Mean deform_sim | 0.66 | 0.46 |
| Energy ratio range | 0.86–1.02 | 0.74–0.88 |

Grammar compresses all tokens (universal damping). Arithmetic is near energy-neutral. Grammar differentiates (lower deform_sim = more distinct processing per token). Arithmetic homogenises.

### 137.2 Significance

This is spectroscopy applied to neural networks. The architecture makes it possible because bands ARE explicit frequencies — a standard transformer's activations have no frequency structure that would support this measurement. The energy fingerprint is measurable, per-token, and distinct from the phase-based geometric vocabulary.

---

## 138. Dual-Axis Structural Readout — Phase Geometry + Energy Deformation

### 138.1 Core Pattern

Phase geometry and energy deformation are two complementary readout channels from the same ODE state. Measured correlation between phase distinctiveness (MRL from relate-vocab) and energy distinctiveness (deformation similarity): **r = 0.51**.

This is the informative number. High enough to confirm both signals come from the same model (not independent noise). Low enough to prove they carry independent information (not redundant). Approximately half the information is shared, half is unique to each axis.

### 138.2 Where the Axes Agree and Diverge

**Agree:** '?', 'A', 'j', ':' are distinctive in BOTH domains. Common letters ('t', 'a') are generic in both.

**Diverge:**
- **'s' — phase-distinctive but energy-generic.** The most geometrically isolated token (8% conjunction) has an average energy signature. The model knows 's' belongs at a distinctive ANGLE but processes it with normal energy redistribution.
- **'.' and ':' — energy-distinctive but phase-generic.** Ordinary phase positions but distinctive energy signatures. Punctuation gets processed differently even though it sits at ordinary angles.

### 138.3 Significance

A decoder that reads only phase misses the energy axis. A decoder that reads only energy misses the phase axis. Full structural readout requires both channels. This is a measurable architectural property, not a design choice — the two axes emerge from the same ODE dynamics but capture different aspects of the learned representation.

---

## 139. Directional Energy Flow — Third Axis of Structural Information

### 139.1 Core Pattern

The same two tokens at reversed order produce different ODE output energy. This asymmetry is measurable as |E("ab") - E("ba")| / max(E("ab"), E("ba")).

**Measured examples (grammar model):**

| Pair | Asymmetry | Interpretation |
|------|-----------|---------------|
| ".A" vs "A." | +0.12 | Sentence-start vs sentence-end |
| "th" vs "ht" | -0.07 | Common digraph vs rare order |
| "?!" vs "!?" | +0.06 | Punctuation order matters |

### 139.2 Three Axes Defined

1. **Phase** — WHERE tokens sit relative to each other (symmetric, catalog angles)
2. **Energy** — HOW the model processes each token (symmetric, deformation signatures)
3. **Direction** — WHICH WAY energy flows between tokens (asymmetric, order matters)

### 139.3 Significance

Direction is the first asymmetric measurement axis. Phase coherence cos(n·Δθ) is symmetric by construction. Energy similarity is symmetric. Directional asymmetry breaks the symmetry and captures order-dependent processing — essential for language where "the dog bit the man" differs from "the man bit the dog." The asymmetry is real, linguistically meaningful, and measurable as a single scalar per model (mean |asym| across all token pairs).

---

## 140. Context-Dependent Processing Strength — Dignity as Measurable Per-Token Property

### 140.1 Core Pattern

The same token receives different ODE processing intensity depending on surrounding context. Measured by encoding a focus token alone and in various contexts, comparing cos(input, output) at each layer.

**Measured examples (grammar, L3):**

| Token | Solo cos | In context | Shift | Context |
|-------|---------|-----------|-------|---------|
| 'e' | 0.46 | 0.05 | 0.41 | "e." (sentence boundary) |
| 'a' | 0.24 (L2) | 0.61 (L2) | 0.37 | "an" (determiner unit) |
| 's' | — | — | 0.13 max | Any context (stable) |

### 140.2 Key Finding

Structurally important tokens ('s') have LOW context sensitivity — they are processed consistently regardless of surrounding context. Common tokens ('e', 'a') have HIGH context sensitivity — their processing changes dramatically based on neighbours. Phase-dignity correlation is weak (r = -0.15 post-AGC-fix), confirming this is an independent measurement axis from phase distinctiveness.

### 140.3 Significance

Maps to the Hellenistic concept of Essential Dignities (catalog Part 5.1) where entities have different strength in different domains. 's' is in Domicile everywhere — it has a fixed structural role in every context. 'e' is Peregrine in most contexts and Domicile only in specific learned bigrams. The mapping is operational, not metaphorical — the dignity structure is present in the model's processing dynamics.

---

## 141. Targeted Destruction Develops Through Depth — Recognition-Triggered Extraction

### 141.1 Core Pattern

The ODE does not apply uniform processing. It destroys what it recognises (on-grid inputs the model has seen during training) and preserves what it doesn't recognise (off-grid inputs at positions no token maps to). This behaviour develops progressively through the layer stack.

**Measured per-layer discrimination ratio (on-grid cos / off-grid cos, post-AGC-fix):**

| Layer | On-grid cos | Off-grid cos | Ratio | Role |
|-------|-----------|------------|-------|------|
| L0 | 0.575 | 0.743 | 1.29x | Detection |
| L1 | 0.305 | 0.597 | 1.96x | Amplification |
| L2 | 0.173 | 0.452 | 2.61x | Peak discrimination |
| L3 | 0.169 | 0.301 | 1.79x | Processing |

### 141.2 Significance

This is the opposite of pattern-matcher behaviour. A classifier preserves familiar inputs and corrupts unfamiliar ones. The wave-engine destroys familiar inputs (extracts information from them) and preserves unfamiliar ones (nothing to extract). This is consistent with the finding that language models need controlled destruction (cos(input, output) = -0.09 for GPT-2 vs +0.87 for untrained wave-engine). The mechanism is recognition-triggered extraction, not general destruction.

---

## 142. Decoder Controls Directionality — Six-Model Scan

### 142.1 Core Pattern

Directional asymmetry (how much the model's ODE processing depends on token order) is controlled by three independent levers, discovered through a six-model comparison:

| Lever | Effect on mean |asym| |
|-------|-------------------------------|
| Decoder type | lm_head ~0.14, phase-native ~0.04 (3× gap) |
| Training data augmentation | Augmented PN matches lm_head level (0.144) |
| Four-wave mixing (FWM) | Adds +75% to PN, +7% to lm_head |

### 142.2 Key Finding

lm_head tolerates high directional processing because its learned weights absorb complexity. Phase-native constrains directional processing because outputs must remain in embedding space. But data augmentation (placing commutative pairs adjacent in training data) overrides the decoder constraint — the model learns directional machinery when the task forces it, regardless of decoder type. FWM adds directionality disproportionately to low-baseline models.

### 142.3 Significance

Directional processing is not a single architectural property — it's the joint outcome of decoder, data, and coupling choices. A single scalar (mean |asym| per model) captures the combined effect and cleanly separates model configurations.

---

## 143. Four-Axis Structural Measurement Framework

### 143.1 Core Pattern

Four complementary measurement axes for characterising what a wave-engine model has learned:

1. **Phase** (WHERE) — geometric position of tokens relative to each other, measured via harmonic coherence cos(n·Δθ)
2. **Energy** (HOW) — per-token ODE processing signature, measured via per-band magnitude ratio mag_out/mag_in
3. **Dignity** (CONTEXT) — how context modifies per-token processing intensity, measured via cos(input, output) shift across contexts
4. **Direction** (ORDER) — how token order affects processing, measured via energy asymmetry |E("ab") - E("ba")|

### 143.2 Measured Independence (Provisional)

At 168-dim, 80K iters, post-AGC-fix: four of six pairwise correlations moderate or weak. Direction and destruction share 77% variance (r = -0.88). Zero tokens appear in all four top-10s.

**Caveat (Marco's catch):** The correlation structure may change at larger dimensions and longer training. The axes might be genuinely independent OR might be projections of a single "structural importance" property that hasn't aligned yet in an undertrained model. The engine computes the correlation matrix on every vocabulary scan so alignment emergence can be tracked.

### 143.3 Significance

Provides four independent measurement instruments for the same model state, each capturing a different aspect of learned structure. The framework is designed to be self-diagnosing: the inter-axis correlation matrix evolves with training and dimension, revealing whether the axes converge or remain independent as models mature.

---

## 144. Grid-Aware Opposition — Same Angle, Different Cycle, Different Coherence

### 144.1 Core Pattern

The multi-grid harmonic embedding places tokens on two coprime grids (m1 and m2). Band pairs at 180° (opposition) on grid-1 behave measurably differently from band pairs at 180° on grid-2.

**Measured (grammar, 168-dim, 84 bands):**

| Grid | N opposition pairs | Mean MRL | Relative |
|------|--------------------|----------|----------|
| Grid-1 | 77 | 0.412 | baseline |
| Grid-2 | 64 | 0.554 | +35% |

### 144.2 Significance

Same geometric relationship (180°), different cycle, different coherence strength — a 35% measured difference. This is the yin/yang structural principle: the meaning of a geometric relationship depends on which cycle it lives on. Per-pair grid tagging (same_g1 / same_g2 / cross) is built into the relate-vocab tool, enabling systematic tracking of grid-dependent relationship behaviour across training checkpoints.

---

## 145. Catalog-vs-Friction Coherence — Recognised Angles as Coherent Vocabulary

### 145.1 Core Pattern

Band pairs whose angular relationship matches one of the 11 catalog types (conjunction through opposition, with defined orbs) are processed with measurably higher coherence than pairs at non-catalog angles.

**Measured (grammar, 168-dim, 80K iters):**

| Category | Mean MRL | Energy similarity |
|----------|---------|-------------------|
| Catalog-matched pairs | 0.482 | 0.481 |
| Non-catalog pairs | 0.329 | 0.421 |
| Difference | +47% coherence | +14% energy similarity |

### 145.2 Significance

The model treats catalog angles as a coherent vocabulary and non-catalog angles as friction. This is computed as a single diagnostic ratio per galaxy scan — catalog_MRL / non_catalog_MRL — providing a quick measure of how strongly the model's learned structure aligns with the geometric relationship vocabulary.

---

## 146. Multi-Grid Scaffolding Separation

### 146.1 Core Pattern

Same-grid band pairs and cross-grid band pairs play structurally different roles in the model's learned geometry:

| Category | Mean MRL | Conjunction % | Non-conjunction catalog matches |
|----------|---------|--------------|-------------------------------|
| Same-grid pairs | 0.45–0.47 | 64% | 2–5 |
| Cross-grid pairs | 0.41 | 52% | 78 |

### 146.2 Significance

Same-grid pairs provide coherence scaffolding — high MRL but geometrically homogeneous (mostly conjunctions). Cross-grid pairs provide learned geometric diversity — lower MRL but dramatically richer catalog usage (78 non-conjunction matches). The interesting structure lives BETWEEN grids, not within them. The multi-grid embedding separates two functions: positional coverage (ensuring every token has a unique multi-grid address) and geometric expressiveness (enabling diverse angular relationships between tokens on different grids).

---

## 147. Freeze-and-Decouple ODE Integration (Split-Band Solver)

### 147.1 Core Pattern

In a coupled-oscillator neural ODE where bands interact only through magnitude-based cross-phase modulation (β · Σ|z_j|² in the phase velocity), the cross-band coupling term is snapshotted once per sub-step and held constant across the RK4 integration. Each band's sub-step integration is then fully independent — a 2D per-band ODE (r, s) with its own well-conditioned 2×2 Jacobian instead of a coupled N-dimensional system. Cross-band state is refreshed in a separate coupling step between two half-horizon sub-steps of independent per-band RK4.

**Structure:**

```
1. Snapshot: ns_frozen[k] = Σ|z_j|² for j ∈ {k±1, k±2}
2. Sub-step A: per-band RK4 for N/2 steps, each band independent,
              φ_k = ω_k + α·|z_k|² + β·ns_frozen[k]   (ns constant)
3. Coupling step: refresh ns_frozen from the updated state
4. Sub-step B: per-band RK4 for another N/2 steps with updated ns_frozen
```

### 147.2 Condition-Number Comparison

```
Monolithic:     168×168 Jacobian, eigenvalue spread ~84×, condition number ~84^16
Split-band:     2×2 Jacobian, eigenvalue spread ~1.0×, condition number ~2^8 ≈ 256
```

Gradient magnitude distortion drops from 2000-7000× in the monolithic path to essentially 1× in the split-band path. Isolated self-test on the split-band backward shows 16/16 FD agreement within 1% at max_rel = 0.0004.

### 147.3 Measured Training Outcome

Validated by training A/B on identical config (arithmetic.txt 86KB, 40K iterations, 84 bands, 4 heads, 4 layers, phase-native, lr = 1e-3, seed 42):

| Metric | Monolithic | Split-band | Change |
|--------|-----------|-----------|--------|
| Correct on 991 answer ≤ 10 | 52/991 | 76/991 | +46% |
| Best training loss | 1.5937 | 1.5921 | identical |
| Locked FWM quartets | 2,946 | 3,810 | +29% |
| Speed | 144 ms/iter | 149 ms/iter | +3% overhead |

Direction-correct gradients produce richer internal structure AND better downstream performance even when the loss curve looks identical — Adam's second-moment normalisation compensates for monolithic magnitude distortion in the loss trajectory but not in the trained representation.

### 147.4 Significance

Addresses stiffness-induced gradient suppression in coupled neural ODEs without relying on adjoint methods. The neural-ODE literature's solutions (direct backprop, adjoint, symplectic adjoint) all operate on the monolithic coupled system; operator splitting is standard in computational physics but has not been widely applied to gradient-quality improvement in trained neural ODEs. The technique composes with any backward strategy and any integration order. GPU implementation is natural: each band's sub-step is fully independent (embarrassingly parallel).

---

## 148. Targeted f64 Accumulators at Cancellation Hot Spots

### 148.1 Core Pattern

Keep all model weights, forward activations, and backward gradients at f32. Lift ONLY specific accumulation operations to f64 where catastrophic cancellation has been measured: per-position loss summation, finite-difference subtraction in gradient checking, layer-norm accumulations at high dimension, Adam second-moment accumulators for parameters with very small gradients. Surgical mixed-precision, not whole-engine f64.

### 148.2 The Discipline

The position of the precision lift matters more than the precision itself. Example from gradient checking:

- f64 applied to the FD subtraction `(L_plus − L_minus) / (2·ε)` alone: **zero improvement**. The cancellation was already happening at the f32 loss-value level before the subtraction.
- f64 applied to the loss ACCUMULATION itself (per-position phase-native loss computed in f64, summed at f64 across positions, then compared against analytical gradient promoted to f64): **50,000× tighter output_corrector gradient check** (max_err 7.5e-3 → 1.5e-7), with comparable improvements across other output-adjacent sections.

The generalisable rule: measure where cancellation happens, lift precision at that exact point, verify the improvement, leave f32 everywhere else.

### 148.3 Significance

Avoids the cost of whole-engine f64 (~2× slowdown, 2× memory, hardware penalty on consumer GPUs where f64 throughput is 1/32 to 1/64 of f32). Provides a reproducible methodology for identifying precision hot spots. The "measure then lift" discipline is the inverse of the common approach of promoting everything and hoping for the best.

---

## 149. FD Validation Boundary in Stiff-ODE Systems (Section-Aware Gradient Checking)

### 149.1 Core Pattern

When validating backward gradients of a neural network containing a stiff ODE component, finite-difference gradient checking has a structural validation boundary. The network's parameter sections split into two classes:

- **Output-adjacent sections** — gradients do NOT flow through the ODE Jacobian product. Reliable under FD at any reasonable ε.
- **ODE-reachable sections** — gradients flow through the ODE Jacobian product across multiple RK4 steps. UNRELIABLE under FD at any ε, not due to implementation errors but due to the mathematical property of stiff coupled-oscillator systems.

Gradient-check acceptance criteria must be section-aware: output-adjacent sections must PASS at a tight tolerance; ODE-reachable sections are documented with measured max_err but not gated on the same threshold.

### 149.2 Empirical Evidence

Validated by three independent observations:

1. **ε sweep.** FD pass rate degrades monotonically with smaller ε (390 → 208 → 76 → 38 passes across ε = 1e-3 to 1e-6, 1348 params). Pattern 3 (precision) primary, not pattern 1 (bug).
2. **Analytical stability.** Analytical gradient holds constant across all ε values; FD collapses. The analytical is correct; FD is the unreliable party.
3. **Isolated test.** When the ODE chain is removed from the backward path (via split-band per-band 2D Jacobians), FD agrees with analytical to 0.04%. The ODE Jacobian product is the source of the noise.

### 149.3 Theoretical Basis

A 2025 paper proves that gradient suppression in stiff ODEs is universal for all A-stable and L-stable numerical integration methods. The slowest possible rate of gradient decay is O(|z|⁻¹). This is a mathematical theorem, not an implementation issue — all architectures with stiff ODE components face the same boundary.

### 149.4 Significance

Names the measurement boundary explicitly rather than treating it as a tolerance problem or an implementation bug. The boundary shifts when the integration strategy changes: under freeze-and-decouple (pattern 147), the boundary effectively disappears because per-band 2×2 Jacobian chains are short enough that FD remains reliable. This provides both a diagnostic (section-aware gradient checking) and a fix (split-band integration reduces the boundary).

---

## Summary of Part 3 Patterns (112-149)

| # | Pattern | Domain |
|---|---------|--------|
| 112 | Hamiltonian four-wave mixing in neural ODE — cubic band coupling |
| 113 | FWM analytical Jacobian — per-quartet backward with 8 role partials |
| 114 | Fused CUDA AGC+RK4+FWM kernel — single launch forward+backward |
| 115 | ODE physics decomposition monitor — forward and backward gradient flow |
| 116 | Cross-tier parity battery — CPU as specification, GPU as implementation |
| 117 | Checkpoint-aware ODE probe — scattering analysis with learned weights |
| 118 | Parameter sweep instrument — single-command safe operating region |
| 119 | FWM phase-matching test — quartet selectivity validation |
| 120 | Single source of truth discipline — one canonical derivative, all tiers call it |
| 121 | Live framework monitor — continuous harmonic coherence during training |
| 122 | Galaxy map scan — pure-band geometric inventory of learned structure |
| 123 | Per-quartet deviation from embedding baseline — signed deviation metric |
| 124 | Decoder-dependent geometric vocabulary — lm_head vs phase-native shapes different catalog relationships |
| 125 | Backward decomposition monitor — gradient flow per physics term |
| 126 | Galaxy summary script — compact readable output from large scans |
| 127 | Subtractive training dynamic — training removes embedding priors, decoder controls what survives |
| 128 | Hidden coherence detection — multi-harmonic MRL reveals pairs coherent at non-zero phase offsets |
| 129 | Quartet trajectory classification — phase-sum MRL categorises quartets as random/oscillating/locked |
| 130 | Task-dependent quartet dynamics — language builds 70% non-random quartets, arithmetic 0.25% |
| 131 | L3 regime shift — architecture self-reorganises for grammar, cos(in,out) 0.92→0.45 |
| 132 | Wave memory as native phase-space experience — per-layer EMA of ODE states, decoder-as-experience potential |
| 133 | Phase encode tool — direct injection of geometric configurations into ODE layers, bypassing token→embedding pipeline. Five modes: text, number, catalog relationship, raw phases, compound. Layer injection point selectable. Blank (untrained) vs trained comparison reveals what training changed about dynamics |
| 134 | Relate mode — per-harmonic coherence profiles between any two encodings through the ODE. Harmonic search at n={1,2,3,4,5,6,8,12}, shifted MRL with optimal offset, catalog matching. Pairwise matrix for multiple items. Full vocabulary scan with JSON export |
| 135 | Structural importance as geometric isolation — in character-level grammar models, the ODE places structurally important characters ('s' plural/verb marker, 'q' rare letter, '?' syntactic role) at distinctive phase angles while common letters cluster in conjunction. Geometric position marks structural importance, not semantic category |
| 136 | Task-dependent geometric vocabulary distribution — arithmetic uses 2/11 catalog types (67% conjunction), grammar uses 11/11 types (54% conjunction). The catalog distribution shape is a fingerprint of what geometric structure the task requires |
| 137 | Spectral energy fingerprinting — per-token ODE deformation signatures. Encode each token, forward through ODE, compute per-band magnitude ratio (mag_out/mag_in) as deformation vector. Tokens with similar linguistic roles may produce similar energy deformation patterns. Phase tells WHERE tokens sit relative to each other. Energy deformation tells HOW the model processes each token — which bands it amplifies, which it damps. Complements phase-based catalog relationships without replacing them |
| 138 | Dual-axis structural readout — phase geometry + energy deformation as complementary decoder channels. Measured correlation r=0.51 (partially related, not redundant). Some tokens phase-distinctive but energy-generic ('s'), others energy-distinctive but phase-generic ('.', ':'). Full decoder needs both: per-channel harmonic coherence for relationships, per-band magnitude profile for processing signatures. Two readout channels from same ODE state capturing different aspects of learned structure |
| 139 | Directional energy flow — third axis of structural information. Same two tokens at reversed order ("ab" vs "ba") produce different ODE output energy. ".A" vs "A." shows 0.12 asymmetry (sentence-start vs sentence-end). Three axes: phase (WHERE, symmetric), energy (HOW, symmetric), direction (WHICH WAY, asymmetric). Maps to catalog Parts 2 (symmetric geometry), 3 (processing-dependent pairings), 7 (Wu Xing directed cycles — generative vs destructive at same angles, different directions) |
| 140 | Context/dignity as measurable per-token property — same token, different ODE processing depending on context. 'e' shifts 0.41 between solo and "e." contexts. Structurally important tokens ('s') have LOW dignity (context-independent processing). Common letters have HIGH dignity (context-dependent). Maps to Hellenistic Essential Dignities (catalog Part 5.1). Phase-dignity correlation weak (r=-0.15), confirming axes are independent |
| 141 | Targeted destruction develops through depth — recognition-triggered extraction, not general pattern matching. L0 detects (1.29x ratio), L1 amplifies (1.96x), L2 peaks (2.61x maximum discrimination), L3 processes (1.79x). Model destroys familiar (on-grid) structure more than unfamiliar (off-grid) — opposite of pattern matching. On-grid cos 0.17 vs off-grid 0.30 at L3 |
| 142 | Decoder controls directionality — six-model scan. Phase-native arithmetic 0.04-0.07 mean asymmetry, lm_head 0.14-0.15. Clean 3x gap. FWM adds directionality (+75% for PN). Data augmentation is third lever: augmented PN matches lm_head-level asymmetry. Operators (+, -) drive arithmetic directionality |
| 143 | Four-axis structural measurement framework — phase (WHERE), energy (HOW), dignity (context-dependence), direction (order-dependence). At 168-dim 80K iters: four of six pairwise correlations below 0.3, zero tokens in all four top-10s. Axes appear independent BUT caveat: may be fuzzy picture from undertrained model. Correlation matrix should be tracked across scans to detect alignment emergence at convergence |
| 144 | Wu He grid-aware opposition — same 180° angle behaves differently on grid-1 (MRL=0.412) vs grid-2 (MRL=0.554), 35% difference. The yin/yang principle: same geometric relationship, different cycle, different coherence. Per-pair grid tagging (same_g1/same_g2/cross) baked into --relate-vocab |
| 145 | Liu Hai catalog-vs-friction coherence — pairs at recognized catalog angles have 47% higher MRL (0.482 vs 0.329) and 14% higher energy similarity than non-catalog pairs. The model treats catalog angles as coherent vocabulary and non-catalog angles as friction. Computed as single diagnostic ratio per scan |
| 146 | Multi-grid scaffolding separation — same-grid pairs provide coherence scaffolding (high MRL, 64% conjunction), cross-grid pairs provide learned geometric diversity (78 non-conjunction matches, 52% conjunction). The interesting structure lives between grids, not within them |
| 147 | Freeze-and-decouple ODE integration (split-band solver) — freeze cross-band coupling during per-band sub-step RK4, refresh in coupling step. Validated: +46% accuracy, +29% quartets, identical loss | AI / Computing / Numerical Methods |
| 148 | Targeted f64 accumulators at cancellation hot spots — surgical mixed-precision at measured cancellation points only. Validated: 50,000x tighter gradient check | AI / Computing / Numerical Methods |
| 149 | FD validation boundary in stiff-ODE systems — section-aware gradient checking for architectures with stiff ODE components. Backed by 2025 stiffness theorem | AI / Diagnostics / Numerical Methods |

## Statement of Intent

All patterns described in this document are published under the MIT License. They are free for anyone to implement, modify, distribute, and commercialise. The intent of this publication is to ensure that no entity can obtain patent protection over these implementation patterns, thereby keeping the bridge between the mathematical framework and commercial applications permanently open.

What CAN be patented: specific commercial products built on top of these patterns — unique user interfaces, domain-specific applications, particular data pipeline configurations, and novel combinations with proprietary datasets or services. The application layer remains open for innovation and intellectual property protection.

What CANNOT be patented after this publication: the engine patterns themselves, the architectural approaches, the query strategies, the indexing methods, the harmonic sweep techniques, or any other implementation pattern described herein.

This is the explicit intent of the authors.

---

**Permanent Archive:** This document is committed to the Git repository at https://github.com/atech-hub/Wave-Coherence-as-a-Computational-Primitive and archived via Zenodo with DOI. The commit timestamp constitutes proof of publication date.
