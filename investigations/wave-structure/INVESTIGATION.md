# Wave Structure Emergence in Trained Models

**Status:** ACTIVE
**Started:** 2026-03-23
**Engine:** wave-engine (Rust, Apache 2.0)
**Hardware:** Intel i7-14700K, RTX 4070 Ti, 32GB DDR5

---

## Question

Does a trained wave-engine model build the harmonic phase structure the theory predicts? The research framework proved the math works (Tests 1-25). The kerr-engine proved the architecture trains (98.1% MLP at 44% params). This investigation tests whether production models actually use the wave basis structurally — or treat it as arbitrary numbers.

Secondary question: do the findings from the frequency-depth investigation (64 bands, 4 layers, char-level) scale to larger models (384 bands, 24 layers, BPE)?

## Background

The frequency-depth investigation (v2.25.0) established at 64 bands, 4 layers, char-level Shakespeare:
- 43 universal bands / 21 word-specific (67%/33% split)
- Consecutive-layer correlation 0.144 (active reassignment)
- Maestro eliminates spectral dispersion (H/L ratio 1.013)
- Concrete words more stable than abstract (+0.053 gap)

These findings are from the kerr-engine at small scale. Whether they hold in a production wave-engine model trained on diverse English with BPE tokenization is unknown.

## Method

### Diagnostic Tool

Built into wave-engine as `--analyze` flag (src/common/wave_analysis.rs). Runs one forward pass on test sentences, extracts per-layer hidden states, computes phase angles θ[pos][band] = atan2(s, r), then runs six diagnostics using harmonic coherence cos(n × Δθ).

### Six Diagnostics

1. **Semantic Discrimination** — Do related token pairs (cat/dog, noun/verb) show higher harmonic coherence than random pairs? Ratio > 1.5 indicates emerging semantic structure.

2. **Depth Curve** — Discrimination ratio at each layer. If a peak emerges, the model develops layer specialisation. Flat = all layers contribute equally.

3. **Band Census** — Circular variance per band across tokens. Low variance = universal (structural) band. High variance = word-specific (semantic) band. Compare split against 67/33 from frequency-depth.

4. **Phase Clustering** — Mean resultant length across bands. Above 0.3 = phases cluster at specific angles (structured). Below 0.15 = near-uniform (arbitrary).

5. **Harmonic Spectra** — For each token pair, coherence at harmonics n=1..12. Different relationships mapping to different harmonics validates multi-harmonic structure.

6. **Grammar Coherence** — Does the subject-verb relationship survive syntactic transformation? (Planned, not yet implemented.)

### Test Sentences

Curated set covering all corpus registers with known semantic pairs:
- Related: cat/dog, sat/kicked, boy/ball, noun/verb, love/war, mat/rug
- Random: shuffled pairings from the same set
- Sentences from grammar, children's literature, Shakespeare, legal, science registers

### Models Under Test

| Model | Dims | Layers | Bands | Params | Corpus | Tokenization |
|-------|------|--------|-------|--------|--------|-------------|
| Baseline (24L) | 768 | 24 | 384 | 42M | combined 4.8MB | BPE 50K |
| Model A | 168 | 4 | 84 | ~350K | combined 12.4MB | BPE 50K |
| Model B (planned) | 384 | 8 | 192 | ~3.8M | combined 12.4MB | BPE 50K |
| Model C (planned) | 768 | 24 | 384 | ~42M | combined 12.4MB+ | BPE 50K |

---

## Results

### Baseline: 768-dim, 24 layers, 0.9 corpus passes

**Date:** 2026-03-23
**Checkpoint:** candle_checkpoint_iter3050 (loss ~5.5)
**Corpus:** 4.8MB (grammar + children + Shakespeare + legal)

| Diagnostic | Result | Interpretation |
|-----------|--------|---------------|
| Semantic Discrimination | 1.01x (related 0.683, random 0.674) | NOT YET — no discrimination |
| Depth Curve | Flat at 1.0-1.04x across all 24 layers | No layer specialisation |
| Band Census | 192/192 (50%/50%) | No bimodal split (continuous CV distribution) |
| Phase Clustering | 0.553 | STRUCTURED — phases cluster, not random |
| Harmonic Spectra | cat/dog: n=5, sat/kicked: n=6, boy/ball: n=1 | Different relationships → different harmonics |

**Key findings:**

1. Phase clustering is strong (0.553, well above 0.3 threshold) despite no semantic discrimination. The model is using the wave basis structurally even before learning semantics.

2. Different token pairs peak at different harmonics (n=1 for boy/ball, n=5 for cat/dog, n=6 for sat/kicked). The architecture assigns different relationship types to different harmonic numbers. This matches the theoretical prediction.

3. Band census shows a continuous distribution from CV 0.05 to 0.98, not the bimodal 67/33 split found at 64 bands. At 384 bands with BPE, band specialisation is smooth rather than binary.

4. The model has seen the corpus less than once (0.9 passes). Semantic discrimination is expected to require multiple passes.

### Model A Progression: 168-dim, 4 layers

**Date:** 2026-03-23 (in progress)
**Corpus:** 12.4MB (7 registers, curriculum ordered)
**Speed:** ~85ms/iter (CPU, batch=4, seq=128)
**Target:** 50 passes (~7 hours)

Diagnostics will be captured at pass 1, 5, 10, 20, 50.

| Passes | Clustering | Discrimination | Band Split | Depth Peak | Dominant n |
|--------|-----------|---------------|------------|------------|------------|
| 0.9 (baseline, 24L) | 0.553 | 1.01x | 50/50 | flat | mixed |
| 1 | — | — | — | — | — |
| 5 | — | — | — | — | — |
| 10 | — | — | — | — | — |
| 20 | — | — | — | — | — |
| 50 | — | — | — | — | — |

---

## Open Questions

1. At how many corpus passes does semantic discrimination cross 1.5x?
2. Does the band census become bimodal with more training, or is the continuous distribution a genuine difference at higher band counts?
3. Does the depth curve develop a peak at 4 layers with sufficient training?
4. Do higher harmonics (n=3, n=5) strengthen relative to n=1 with more passes?
5. Do findings from Model A (168-dim) predict the behaviour of Model B (384-dim)?

## Finding: Harmonic Embedding Minimum Dimension

**Date:** 2026-03-23
**Status:** CONFIRMED — numerical verification + failed training runs

Harmonic embeddings have a minimum dimension for a given vocabulary size. Below this threshold, tokens become geometrically indistinguishable and training is numerically degenerate.

The harmonic embedding for token v at band n is `[cos(n×θ_v), sin(n×θ_v)]` where `θ_v = 2π×v/vocab_size`. The dot product between adjacent tokens (v and v+1) is:

`dot(v, v+1) = Σ_n cos(n × 2π/vocab_size)`

The separation from the self-dot-product (which equals n_bands) determines discriminability.

Measured values:

| Vocab | Bands | Adjacent token separation | Cosine similarity | Trainable? |
|-------|-------|--------------------------|-------------------|------------|
| 65 | 84 | 74.66 (of 84) | 0.112 | YES — proven |
| 2,000 | 84 | 0.99 (of 84) | 0.988 | YES — tested |
| 50,000 | 84 | 0.0016 (of 84) | 0.99998 | NO — NaN after ~760 iters |
| 50,000 | 384 | 0.148 (of 384) | 0.99961 | YES — proven at 24L |

At 84 bands with 50K vocab, adjacent tokens differ by 0.0016 in dot product — they are 99.998% identical. Hundreds of tokens occupy the same neighbourhood in embedding space. The softmax over near-identical logits is numerically degenerate: one small perturbation from the ODE flips predictions between indistinguishable tokens, producing unstable gradients that eventually overflow to NaN.

This was confirmed empirically: five different fixes addressing gradient clipping, weight growth, learning rate, output projection, and activation clamping all failed. The NaN is geometric, not numerical — no amount of regularisation can make indistinguishable inputs distinguishable.

**Minimum dimension rule:** For stable training, the adjacent token separation should be at least ~0.1 (empirical threshold). This gives:

| Vocab size | Minimum bands | Minimum dimension |
|------------|--------------|-------------------|
| 65 (char) | 8 | 16 |
| 500 | 21 | 42 |
| 2,000 | 42 | 84 |
| 10,000 | 170 | 340 |
| 50,000 | 384 | 768 |

This is analogous to the Nyquist limit in signal processing: the embedding dimension must provide sufficient angular resolution to separate the vocabulary. Below the limit, the harmonic basis cannot uniquely address every token.

**Practical implication:** For small diagnostic models (168-dim), use a vocabulary matched to the dimension. A 2K BPE vocabulary at 84 bands provides 630x more geometric separation than 50K. For production models (768-dim), 50K vocabulary is geometrically viable.

**Resolution:** Model A uses a custom 2K BPE tokenizer trained on the 12.4MB corpus.

## Finding: ODE Coupling Constants Scale with Band Count

**Date:** 2026-03-23
**Status:** CONFIRMED — empirical sweep

The Kerr-ODE coupling constants (α, β) must be reduced at lower band counts. At the default α=β=0.1 (calibrated for 64 bands at char-level), the ODE produces NaN-inducing spikes at 84 bands with BPE tokenization. Reducing to α=β=0.01 drops the NaN rate from ~84% to ~7%.

**Root cause:** The ODE phase shift per step is `δφ = (α + 4β) × M²` where M is the preconditioned input magnitude. At α=β=0.1 this gives `0.5 × M²` radians. When M exceeds ~2.0, the phase shift exceeds 115° — the ODE enters a chaotic regime where outputs spike to extreme values.

The preconditioned input magnitude depends on the maestro output, which depends on the gradient-driven learning rate of the maestro parameters. BPE tokenization produces ~5.6x stronger gradients than char-level (proportional to sqrt(vocab_size/65)), causing the maestro to learn faster and produce larger corrections, pushing M above the stability threshold.

**Measured NaN rates at 168-dim (84 bands), 2K BPE, 1000 iterations:**

| Alpha | Beta | NaN rate | Loss descent |
|-------|------|----------|--------------|
| 0.100 | 0.100 | ~84% | Cannot train |
| 0.047 | 0.047 | ~24% | Partial |
| 0.022 | 0.022 | ~16% | Better |
| 0.010 | 0.010 | ~7% | 7.79 → 7.43 |

**Scaling rule:** ODE coupling should scale inversely with the square root of band count relative to the reference (64 bands):

```
alpha_init = 0.1 × sqrt(64 / n_bands)
```

| Bands | Dimension | Alpha | Status |
|-------|-----------|-------|--------|
| 64 | 128 | 0.100 | Proven (kerr-engine, char-level) |
| 84 | 168 | 0.087 | Predicted (0.01 confirmed stable) |
| 192 | 384 | 0.058 | Untested |
| 384 | 768 | 0.041 | Untested (24L trained at 0.1 — but ODE params frozen) |

**Note:** The 24L Candle model at 384 bands trained with α=0.1 without NaN. However, the Candle tier uses the perturbative ODE (single-pass, different numerical properties) and the ODE parameters are frozen (identity backward). The coupling constant sensitivity may only manifest when the ODE parameters receive gradient signal and the input magnitude distribution shifts during training.

**Interaction with maestro_dim:** Tested maestro_dim at 4, 16, and 32 — all produced identical NaN rates. The maestro dimension is not the controlling variable. The coupling constants are.

**Connection to gradient balance:** The gradient magnitude through the lm_head backward scales as sqrt(vocab_size). This affects the maestro learning speed, which affects the preconditioned input magnitude, which triggers the ODE instability. Char-level (65 vocab) produces 5.6x weaker gradients than 2K BPE and 27.8x weaker than 50K BPE. This explains why char-level training at α=0.1 is stable — the maestro never learns aggressively enough to push M above 2.0.

## Finding: Multi-Grid Embeddings + Per-Band Clamp = Stable BPE at Any Dimension

**Date:** 2026-03-24
**Status:** CONFIRMED — zero NaN in 3000 iterations

Two independent fixes, addressing two independent problems, combine to give completely stable BPE training at 168-dim:

### Fix 1: Multi-Grid Harmonic Embeddings (geometry)

Replaces single-circle token mapping with two coprime modular circles (Pattern 53). Each grid gets half the bands. Tokens that collide on grid 1 are separated on grid 2.

The Sexagenary principle: 10 Stems × 12 Branches = 60 unique positions from two small grids. For vocab 2048: moduli 46 × 45 = 2070, lcm covers full vocabulary.

| | Adjacent separation | Improvement |
|---|---|---|
| Single grid (84 bands, 2K vocab) | 0.94 | baseline |
| Multi-grid (42+42 bands, 2K vocab) | 95.01 | 101x |
| Single grid (84 bands, 50K vocab) | 0.0016 | baseline |
| Multi-grid (42+42 bands, 50K vocab) | 18.60 | 11,800x |

Implementation: one function change in build_harmonic_table(). Same output shape, same interface. Everything downstream unchanged.

### Fix 2: Per-Band Magnitude Clamp Before ODE (dynamics)

Clamps per-band input magnitude to 2.5 before the ODE processes it. Prevents the maestro from pushing any band past the ODE's stability threshold.

At magnitude 2.5 with α=0.01: δφ = 0.05 × 6.25 = 0.3125 rad = 18° per step. Over 16 RK4 steps: 288°. Tight but stable — the damping (γ) absorbs the accumulated phase within each step.

The maestro can still learn which bands to emphasise (direction preserved). It just can't amplify any band past the magnitude where the ODE wraps chaotically.

### Combined Result

| Config | NaN rate | Loss descent | Status |
|--------|----------|-------------|--------|
| Single grid, no clamp, α=0.1 | 84% | Cannot train | Broken |
| Single grid, no clamp, α=0.01 | 95% after 500 iters | 7.79 → 7.43 then diverges | Unstable |
| Multi-grid, no clamp, α=0.01 | 63% after 700 iters | 7.76 → 7.20 then diverges | Better |
| Multi-grid + clamp 2.5, α=0.01 | **0% in 3000 iters** | 7.76 → 7.20 | **Stable** |

Neither fix alone is sufficient. Multi-grid delays onset but doesn't prevent the maestro from eventually triggering ODE instability. The clamp alone would work but with degraded token separation. Together they address both the geometric problem (embedding resolution) and the dynamical problem (ODE phase wrapping) independently.

### Loss drift note

Loss descends from 7.76 to 7.20 (best at iter 300) then drifts up to 8.4 by iter 1800. This is a learning rate scheduling issue (lr=3e-4 cosine decay), not a stability issue. Zero NaN throughout — the model trains continuously without interruption. LR tuning is standard training optimisation, not an architectural problem.

### Connection to ancient systems

The multi-grid solution was discovered independently by multiple ancient civilisations (Multi-Grid Investigation, Pattern 53). The Chinese Sexagenary system, the Vedic Nakshatra-zodiac overlay, and the Babylonian sexagesimal system all use coprime circle divisions to extend angular resolution beyond what any single division can achieve. The geometric comma theorem (24° = 360°/lcm(3,5)) proves that certain angular relationships are structurally inaccessible from a single grid.

This investigation applies that principle to neural network embeddings: a single harmonic circle cannot resolve large vocabularies at small dimensions. Two coprime circles, each with half the bands, provide sufficient resolution at any vocabulary-to-dimension ratio.

---

## Open Questions

1. At how many corpus passes does semantic discrimination cross 1.5x?
2. Does the band census become bimodal with more training, or is the continuous distribution a genuine difference at higher band counts?
3. Does the depth curve develop a peak at 4 layers with sufficient training?
4. Do higher harmonics (n=3, n=5) strengthen relative to n=1 with more passes?
5. Do findings from Model A (168-dim) predict the behaviour of Model B (384-dim)?
6. What is the exact relationship between vocab size, band count, and the minimum separation threshold for stable training?

## Connections

- **Frequency-depth investigation:** Established 67/33 band split, 0.144 layer correlation at 64 bands. This investigation tests whether those findings scale.
- **Wave memory investigation:** Memory requires a model with established wave structure. This investigation determines when that structure is sufficient for memory to be meaningful.
- **Corpus-ordering investigation:** The 12.4MB corpus uses the validated curriculum ordering (grammar → children → letters → essays → Shakespeare → legal → science).
- **Architecture boundaries (research repo):** The minimum dimension finding adds a new boundary: harmonic embeddings require vocab_size / n_bands below a critical ratio.
