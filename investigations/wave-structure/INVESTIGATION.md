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
