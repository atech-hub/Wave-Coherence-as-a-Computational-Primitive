# Spherical Coherence Investigation: From Circle to Sphere

## Status: Mechanism proven — awaiting real-world validation (Option A)

## Context

The Wave Coherence framework encodes relationships as harmonic phase angles on a circle using `cos(n × Δθ)`. This achieves 96.8% of MLP performance at 42.6% of parameters (Phase A), with a remaining ~3.2% gap.

The Kerr aliasing test identified the gap as architectural: ~4.9% of spectral energy sits in Kerr stop-bands, and the MLP coupling profile is flat across all band distances. The circle framework cannot capture this energy because it operates in one dimension — phase only.

But every embedding pair (cos_val, sin_val) has TWO properties: a phase angle and a magnitude. The framework extracts the phase and discards the magnitude. Frozen embeddings have constant magnitude (0.125) — they sit exactly on the unit circle. Trained embeddings have wildly variable magnitude (0.003 to 0.076, 51.5% coefficient of variation) — the optimizer pushes tokens off the ring. Training builds a second dimension of structure that the framework cannot read.

The circle was always a sphere. We had one eye closed.

This investigation tests whether the discarded magnitude dimension carries usable information, how to extract it without destroying the circle's detection capabilities, and whether it can close part of the remaining performance gap.

---

## Phase 1: The Chebyshev / Legendre Distinction

**File:** `spherical_coherence_test.rs` — 8 tests, zero dependencies

**Question:** Does spherical coherence (Legendre polynomials) reduce to circular coherence (Chebyshev polynomials) on the equator?

**Why this matters:** If P_l(cos γ) equals cos(n × Δθ) when restricted to the equator, the sphere is just a wrapper around the circle. If they differ, the sphere is a genuinely different coherence system with its own properties.

**The answer: They differ.**

| | Circle (Chebyshev) | Sphere (Legendre) |
|---|---|---|
| Trine (60°) at n/l=3 | cos(3×60°) = **−1.000** | P_3(cos 60°) = **−0.4375** |
| Square (90°) at n/l=2 | cos(2×90°) = **−1.000** | P_2(cos 90°) = **−0.500** |
| Exact match (0°) | 1.0 | 1.0 |
| Opposition (180°) | (−1)^n | (−1)^l |

They agree at the endpoints (0° and 180°) and disagree at every intermediate angle. T_2(x) = 2x² − 1 vs P_2(x) = (3x² − 1)/2 — different functions. The sphere is not a wrapper. It is a genuinely different coherence system.

**What passed (6/8):** Exact match, opposition, latitude discrimination, sphere-exclusive relationships, mode count (5.8× more capacity through l=10), Legendre properties.

**What failed (2/8) — these are the findings:** Equator equivalence fails (proves Legendre ≠ Chebyshev). Harmonic family detection fails (intermediate angles behave differently). The failures are diagnostic, not bugs.

**Result: The sphere is a different system, not a generalisation of the circle.**

---

## Phase 2: Relationship Map — Circle vs Sphere

**File:** `spherical_relationship_map.rs` — 5-part analysis, zero dependencies

**Question:** Does the sphere detect relationships the circle misses?

**The answer: No. The opposite.**

- **442 circle-only detections** (circle |c| > 0.95, sphere |s| < 0.50)
- **0 sphere-only detections**
- Every named relationship (trine, square, sextile, opposition) detected by circle but muted by sphere at higher harmonics
- Divergence grows with harmonic number (n=1 identical, n=15 significant divergence)

The sphere is strictly less sensitive than the circle for resonance detection on the equator. Legendre spreads energy where Chebyshev concentrates it. Replacing cos(nΔθ) with P_l(cos γ) would destroy 442 detection capabilities.

**Where the sphere wins: latitude.**

Same Δφ = 90° produces completely different coherence at different latitudes:

| Latitude | P_2 | P_4 | Circle (both) |
|----------|-----|-----|---------------|
| Equator | −0.500 | +0.375 | −1.0 / +1.0 |
| 45° | −0.125 | −0.289 | −1.0 / +1.0 |
| 75° | +0.806 | +0.426 | −1.0 / +1.0 |

The circle sees all three as identical. The sphere sees three completely different relationships.

**Conclusion: Hybrid architecture, not replacement.** Keep cos(nΔθ) for azimuthal coherence. ADD the elevation dimension for discrimination. Circle + sphere, not sphere replacing circle.

---

## Phase 3: Three Combiners Tested

**File:** `hybrid_coherence_test.rs` — 7-part analysis, zero dependencies

**Question:** How should the circle and sphere dimensions combine?

Three architectures tested:

**1. Gated (safe choice):**
```
H = cos(n × Δφ) × [(1−β) + β × P_l(cos Δθ)]
```
- 553/553 circle detections preserved (100%)
- 137/180 elevation angles discriminated at β=0.5
- Backward compatible at Δθ=0 by construction
- ρ = 0.988 vs circle — barely reranks. A dimmer switch on the same picture.

**2. Product:** `cos(nΔφ) × P_l(cos Δθ)` — too aggressive, elevation miss zeroes signal.

**3. Sum:** `α × cos(nΔφ) + (1−α) × P_l(cos Δθ)` — breaks backward compatibility.

**4. Embedded (the discovery):**
```
φ_eff = φ + α × (r − r_mean) / r_std
then cos(n × Δφ_eff)
```
Magnitude adjusts phase. Then the circle detector runs on the adjusted signal.

- ρ = 0.029 vs circle — near-zero rank correlation
- Sees a completely different world
- Not a modulation — a transformation

**Result: Embedded wins.** The gated combiner is too gentle. The embedded combiner is the only method that produces a genuinely different view of the data.

---

## Phase 4: Sweep v3 on Trained Embeddings

**File:** `sweep_v3_hybrid_coherence.rs`

**Question:** Does the hybrid method find new relationships in real trained embeddings?

**Finding 1:** Magnitude signal exists. Trained embeddings have 51.5% coefficient of variation across 64 bands. Frozen/harmonic are flat at 0.125. Training creates real magnitude structure.

**Finding 2:** Circle is saturated. With 65 tokens and n=1..15, 100% of 133,120 pairs score |cos(nΔφ)| > 0.9 for at least one harmonic. Zero weak pairs to rescue.

**Finding 3:** Zero rescues, zero losses. The hybrid found nothing new AND lost nothing.

**Why:** The gated combiner modulates, it doesn't amplify. If the circle signal is near zero, the product is near zero regardless of the gate value. The sphere's role is discrimination (splitting similar scores), not detection (finding new relationships).

**Result: Null for detection.** But this reveals the sphere's true purpose — not finding more relationships, but providing finer-grained ranking within relationships the circle already detects.

---

## Phase 5: Embedded vs Gated — Head-to-Head

**File:** `hybrid_vs_embedded_eval.rs` — 4 metrics, zero dependencies

| Metric | Gated β=0.4 | Embedded α=0.5 | Winner |
|--------|-------------|----------------|--------|
| Score spread (std) | 0.1015 | 0.1203 | Embedded |
| Known-pair gap/std | −0.026 | +0.031 | Embedded (faint) |
| Spearman ρ vs circle | 0.988 | 0.029 | — |
| View | Same as circle | Completely different | — |

The gated combiner barely changes rankings (ρ = 0.988). The embedded combiner creates an entirely new ranking (ρ = 0.029). Neither strongly separates known pairs at 65 tokens — the vocabulary is too small.

**Architecture decision: Embedded wins, needs bigger test.**

---

## Phase 6: Synthetic Proof — The Decisive Result

**File:** `embedded_synthetic_test.rs` — 500 tokens, 5 groups, 32 bands, zero dependencies

**Setup:** 500 tokens in 5 groups of 100. Same-group tokens share similar phase (±0.05 rad jitter) with linear magnitude gradient (0.5 to 1.5). The circle can find groups but CANNOT rank within groups. The embedded method should rank using magnitude.

**Test 1 — Group detection:** Both methods detect same-group pairs equally. Embedded doesn't break detection.

**Test 2 — Within-group ranking (the key test):**

| Method | ρ (distance vs coherence) | Interpretation |
|--------|--------------------------|----------------|
| Circle | −0.0008 | Completely blind |
| Embedded α=0.1 | **−0.9928** | Near-perfect ranking |
| Embedded α=0.2 | −0.7347 | Good but degrading |
| Embedded α=0.5 | −0.2304 | Diluted |
| Embedded α=1.0 | −0.1424 | Too aggressive |

**Test 3 — Retrieval precision:**

| Method | Top-10 accuracy |
|--------|----------------|
| Circle | **12%** (random) |
| Embedded α=0.1 | **100%** (every neighbour found) |

Circle returned: [97, 19, 98, 23, 73, 15, 31, 79, 47, 32] — scattered randomly within group.
Embedded returned: [47, 53, 54, 55, 49, 56, 44, 52, 48, 51] — exact neighbours.

**Test 4 — Coherence gradient:**

Circle: flat wall at 0.9992 for ALL within-group distances.
Embedded: smooth monotonic decline from 0.9992 → 0.9490 as distance grows.

The circle sees a wall. The embedded method sees a landscape.

**α = 0.1 is the sweet spot.** A 10% nudge. Enough to create ranking, small enough to preserve group detection.

**Result: MECHANISM PROVEN.** The magnitude carries real within-group information. The embedded formula extracts it into the circle coherence function. Detection preserved. Ranking transformed.

---

## External Validation: Zelenka et al. (2024)

**Paper:** "Combining Machine Learning with Recurrence Analysis for resonance detection" — Czech Academy of Sciences (arXiv:2412.19683v1)

**Their finding:** Resonances in 2D dynamical systems (orbital mechanics around Kerr black holes) are detected cleanly. In 4D phase space, detection fails. They recover partial signal by embedding higher-dimensional data back into a form their 2D detector can read.

**Our finding:** Resonances on the circle (1D, Chebyshev) are detected sharply — 442 strong detections. On the sphere (2D, Legendre) — zero. Same pattern: resonance sensitivity drops as dimensionality increases.

**Independent convergence:** They didn't replace their 2D tools with 4D tools. They embedded 4D data into 2D features. We did the same: keep the circle (sharp detector), use magnitude to adjust what the circle sees. Different domain, same architectural solution.

---

## Summary: What Each Phase Established

| Phase | Finding | Status |
|-------|---------|--------|
| 1 | Legendre ≠ Chebyshev — sphere is a different system | Proven |
| 2 | Sphere detects 0 relationships circle misses, loses 442 | Proven |
| 3 | Gated combiner too gentle (ρ=0.988), embedded transforms (ρ=0.029) | Proven |
| 4 | Magnitude signal exists in trained embeddings (51.5% CV) | Measured |
| 5 | Embedded method produces genuinely different view | Proven |
| 6 | Embedded achieves 100% retrieval vs 12% circle on synthetic data | Proven |
| 7 | Linear z-score: α*=0.001 for 23/23, but discrimination gap ~zero | Proven |
| 8 | Quantile confirms structural bottleneck — T16 is the wall, not outliers | Proven |

**The embedded formula:**
```
φ_eff = φ + α × (r − r_mean) / r_std     (α = 0.1)
```

**What it does:** Gently adjusts phase based on magnitude. Tokens with higher-than-average magnitude shift slightly one direction. Lower-than-average shift the other. The circle detector then sees these shifts as phase differences, enabling fine-grained ranking within groups.

**What's proven:** The mechanism works on controlled synthetic data.

**What's not proven:** Whether trained embeddings develop magnitude structure that correlates with semantic similarity in real-world tasks.

---

## Phase 7: Three-Mode Backward Compatibility Harness

**File:** `three_mode_harness.rs` — 23 tests × 3 modes + alpha sweep, zero dependencies

**Question:** Does the embedded coherence formula preserve ALL existing circle properties when magnitudes vary?

**Setup:** Every test from the original 23-test suite runs in three modes:
- **Mode A (Circle):** cos(n × Δφ) — baseline control
- **Mode B-Uniform:** Embedded with all magnitudes = 1.0 — MUST reduce to circle exactly
- **Mode B-Varied:** Embedded with α=0.1, magnitudes ~ U[0.108, 1.892], CV ≈ 51.5%

**Results at α = 0.1:**

| Mode | Score | Purpose |
|------|-------|---------|
| A (Circle) | 23/23 | Baseline verified |
| B-Uniform | 23/23 | Mathematical guarantee — embedded reduces to circle when magnitudes uniform |
| A/BU Agreement | 23/23 | Perfect match (epsilon < 1e-12) |
| B-Varied | 11/23 | Shows which properties survive magnitude perturbation |

**T22 (Kernel Admissibility): PASS in all 3 modes.** Symmetry, normalization, PSD, spectral scaling mathematically guaranteed regardless of magnitude variation. The anchor holds.

**The surprise:** Plan predicted ~21/23 for B-Varied, actual is 11/23. The plan estimated ~5.7° max phase shift (1-sigma), but uniform [0.108, 1.892] reaches ~1.7 sigma, giving:
- Max per-entity adjustment: ~10° (not 5.7°)
- Worst-case pair delta: ~20°
- At n=3: ~60° effective shift → cos(60°) = 0.5 (well below 0.95 threshold)

**Alpha sweep — finding the operating point:**

| alpha | Pass | Discrimination gap |
|-------|------|--------------------|
| 0.100 | 11/23 | 0.026908 |
| 0.050 | 15/23 | 0.006890 |
| 0.010 | 21/23 | 0.000279 |
| 0.005 | 21/23 | 0.000070 |
| 0.003 | 22/23 | 0.000025 |
| 0.001 | 23/23 | 0.000003 |

**Operating point:** α* = 0.001 gives 23/23 backward compatibility, but discrimination gap drops to 0.000003 — effectively zero.

**Test sensitivity ranking (most sensitive first):**
- T16 (360 resolution) breaks first at α = 0.003
- T8 (wave = linear scan), T21 (harmonic sweep) break at α = 0.005
- T14 (harmonic orthogonality) breaks at α = 0.030
- 11 tests never fail regardless of alpha

**Conclusion:** The linear z-score formula `adj = α × (mag − μ) / σ` cannot simultaneously maintain backward compatibility AND provide discrimination at 51.5% CV. The operating window is effectively zero.

---

## Phase 8: Quantile Variant — Confirming the Structural Bottleneck

**File:** `variant_quantile_sweep.rs` — linear vs quantile head-to-head, zero dependencies

**Question:** Does bounding the adjustment to [-α, +α] via rank percentile mapping open the operating window?

**Formula:** `adj = α × (2 × rank_percentile(mag) − 1)`

**Hypothesis:** The 1.7-sigma outlier problem broke tests at α=0.1. Quantile maps every magnitude to [0,1] regardless of distribution shape — no outliers, no fat tails. Should allow higher α while preserving backward compatibility.

**Head-to-head results:**

| alpha | Linear Pass | Linear Gap | Quantile Pass | Quantile Gap | Winner |
|-------|-------------|------------|---------------|--------------|--------|
| 0.100 | 11/23 | 0.030545 | 16/23 | 0.010412 | Quantile |
| 0.050 | 16/23 | 0.007670 | 19/23 | 0.002607 | Quantile |
| 0.010 | 20/23 | 0.000307 | 20/23 | 0.000104 | Tie |
| 0.005 | 20/23 | 0.000077 | 22/23 | 0.000026 | Quantile |
| 0.001 | 23/23 | 0.000003 | 23/23 | 0.000001 | Tie |

**Quantile is more robust:** 16 tests never fail (vs linear's 11). At every alpha, quantile passes equal or more tests. But both hit 23/23 at the same α* = 0.001, with effectively zero discrimination.

**The bottleneck is T16.** 360-resolution (1° discrimination) breaks at α = 0.003 for both variants. T16 tests whether the circle can uniquely identify each of 360 entities spaced 1° apart. Any phase perturbation above ~0.003 rad destroys this. No formula variant can escape this geometry.

**Remaining 4 variants abandoned.** Tanh, per-band alpha, selective application, and band selection all perturb phase. T16 constrains all of them identically. Two data points (linear + quantile) are sufficient to establish the structural bound.

**The pivot:** T16 is not the problem — it's the proof that the circle works. Sub-degree precision and magnitude ranking are two different jobs. A microscope and a scale measure different things. Embedding magnitude into phase asks one channel to do both. The data says this is structurally impossible.

**Architectural decision: Dual channel, not embedded.** Circle score (detection + precision) and magnitude score (within-group ranking) should be reported separately per pair. The circle output is never perturbed. T16 never sees a perturbation. Magnitude adds information alongside.

---

## Next: Option A — Word-Level Transformer (Dual-Channel)

The synthetic tests used uniform random magnitudes — maximum chaos, 51.5% CV. A real optimizer doesn't produce random magnitudes. It produces magnitudes that serve the loss function. The global CV might be high but the local CV within semantically meaningful groups could be much lower.

**The fundamental question:** Does training build magnitude structure that correlates with meaning?

If yes — the information exists and we find a way to use it (dual channel, separate scoring, or something we haven't thought of yet).

If no — magnitude is noise shaped by optimizer dynamics, and the investigation closes honestly.

**Measure magnitude properties FIRST, before running coherence tests:**
1. CV per harmonic band
2. CV within semantic word families (names, verbs, emotions, etc.)
3. Correlation between magnitude distance and semantic distance
4. Distribution shape (uniform? peaked? clustered?)

These measurements tell us what the optimizer built before we try to exploit it.

**Three embedding modes to train:**
1. **Baseline** — standard `nn.Embedding`, fully trainable (control)
2. **Frozen** — harmonic cos/sin embeddings, fully frozen (existing approach)
3. **Magnitude** — harmonic phases frozen, per-band magnitudes trainable (the hypothesis)

---

## References

**[Zelenka2024]** Zelenka, O., Kopáček, O., & Lukes-Gerakopoulos, G. (2024). "Combining Machine Learning with Recurrence Analysis for resonance detection." Czech Academy of Sciences. arXiv:2412.19683v1. — Their finding that resonance detection degrades in higher dimensions and their embedding-back-to-lower-dimensional-detector solution directly informed the architectural decision to use the embedded method (φ_eff) instead of the gated combiner. Independent convergence from astrophysical orbital mechanics on deformed Kerr black hole spacetimes.

---

## Open Questions

1. Does the magnitude dimension help close the ~3.2% Kerr gap?
2. ~~Is 192-dim (128 phase + 64 elevation) worth the 50% size increase?~~ **SUPERSEDED: Dual-channel approach — magnitude is a separate score, not embedded in phase.**
3. What does the MLP coupling profile look like when magnitude structure is present?
4. Does multi-grid × embedded give compounding benefit?
5. ~~Is Chebyshev or Legendre better?~~ **ANSWERED: Chebyshev for detection, Legendre for latitude.**
6. ~~Can a formula variant open the operating window?~~ **ANSWERED: No. T16 (1° resolution) constrains all phase-perturbation methods identically. Two variants tested, same wall. Dual channel is the path.**
7. **NEW:** Does training build magnitude structure that correlates with semantic similarity? (Option A answers this)
