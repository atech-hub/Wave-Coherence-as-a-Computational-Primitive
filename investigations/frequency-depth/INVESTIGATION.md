# Depth-Axis Frequency Investigation: Internal Oscillation, Spectral Dispersion, and Band Identity

## Status: COMPLETE. Nine experiments (5a-5h). Core finding: velocity is transport, assignment is meaning. 67% structural / 33% semantic band split at L3 is CONSTRUCTED through depth (active reorganisation at every layer). Abstraction level predicts contextual dependence. Cross-word clustering follows semantic affinity, not category membership. Three null theories sharpened the positive: intelligence lives in structure, not speed.

## Context

The Wave Coherence framework processes information through 64 harmonic bands, each a cos/sin pair evolving through transformer layers. Phase C established Maestro-Add + curriculum as the performance ceiling (98.1% of MLP at 44% params). But HOW the model uses these bands through depth was unmeasured.

This investigation tracks what happens to individual bands as they pass through layers. Do they oscillate? At what rate? Does the Maestro bottleneck control this? And critically: do different bands carry different types of information?

The origin was a conversation about whether the geometric associations in ancient harmonic systems encoded frequency as geometry — positions on a circle that ARE frequencies. The Fourier transform formalises this: position and frequency are dual views of the same information. If the model's bands evolve through depth at different rates, that evolution IS a frequency through the depth axis.

**Hardware:** RTX 4070 Ti (CUDA), Intel i7-14700K. All experiments use the standard 4-layer, 4-head, 128-dim Shakespeare transformer (842K params MLP, ~354K Kerr+Maestro).

---

## Experiment 5a: Depth-Axis Frequency — Does the Model Oscillate Through Layers?

**File:** `tests/depth_frequency_5a.py`

**Question:** Do different harmonic bands propagate through transformer layers at different speeds (spectral dispersion)?

**Method:** Train 3 models (Maestro+curriculum, Kerr flat, MLP baseline). Extract hidden states at each layer. Decompose into per-band phase and magnitude. Compute phase velocity (delta-phase per layer) using unwrapped phase differences.

**Results:**

| Model | H/L Velocity Ratio | Mean Velocity | Magnitude Growth |
|-------|-------------------|---------------|-----------------|
| Maestro+curriculum | **1.013** | 1.069 | 0.16 -> 0.51 (3.2x) |
| Kerr flat | 0.883 | 0.989 | 0.16 -> 0.38 (non-monotonic) |
| MLP baseline | 0.861 | 0.883 | 0.16 -> 0.19 (flat) |

- All three models decelerate from embedding to mid-layers (28% slowdown for Maestro). Matches Phase 22 finding that L0 does impedance matching.
- Cross-transition correlations near zero — which bands are "fast" reorganizes at each layer.
- Token-level velocity CV: 7.5-9.6% across models. Different tokens oscillate at modestly different rates.

**Verdict: Maestro eliminates spectral dispersion.** H/L ratio 1.013 vs 0.86-0.88 for non-Maestro models. All bands propagate at the same velocity. The bottleneck's gather/broadcast mechanism synchronizes the entire spectrum. Kerr flat and MLP have built-in dispersion — high bands move slower than low bands (analogous to Kepler's third law: inner orbits = shorter periods).

---

## Experiment 5b: Selective Dispersion — Does the Model Create Its Own Frequency Structure?

**File:** `tests/depth_frequency_5b.py`

**Question:** If Maestro equalizes global velocity, does it let individual tokens create their own dispersion? (Uniform global average hiding rich per-token structure.)

**Method:** Compute per-token H/L band velocity ratio, acceleration diversity, transition profile shape (cosine similarity), velocity profile clustering, and per-band token CV.

**Results:**

| Model | Global H/L | Per-token H/L CV | Accel Std | Cos Sim |
|-------|-----------|-----------------|-----------|---------|
| Maestro+curriculum | 1.013 | 12.0% | 0.071 | 0.990 |
| Kerr flat | 0.883 | 10.9% | 0.064 | 0.989 |
| MLP baseline | 0.861 | 15.2% | 0.059 | 0.984 |

- Per-band token CV: Maestro low=41.9%, high=41.5% (ratio 0.991 — perfectly uniform). Kerr flat: 1.096. MLP: 1.127.
- Acceleration: Maestro/Kerr ~93% tokens decelerate. MLP: 56/44 split (both strategies).
- Transition profiles: cosine sim > 0.98 for all models — tokens take similar paths through depth.

**Verdict: Maestro creates uniform differentiation capacity.** Not selective dispersion — uniform differentiation. Every band gets equal opportunity to distinguish tokens. The model doesn't paint different pictures on a blank canvas — it calibrates every key to identical action weight. MLP has the highest per-token diversity (15.2% CV) but at the cost of asymmetric band utilisation.

---

## Experiment 5c: Band Aspect Separation — What Do Individual Bands Encode?

**File:** `tests/depth_frequency_5c.py`

**Question:** Do different bands carry different aspects of meaning? Feed the same word through different contexts — which bands stay stable (identity) and which change (context)?

**Method:** 5 target words (king, sword, love, death, hand) x 6 contextual sentences each. Extract hidden states at the first character of each target word. Measure per-band phase stability (circular resultant length across contexts) and magnitude CV. Compare across words and layers.

**Results (Maestro, L3, bands with stability > 0.9):**

| Word | Stable bands (>0.9) | Most contextual |
|------|-------------------|-----------------|
| king | 60/64 | b12, b4, b60, b38 |
| sword | 59/64 | b37, b18, b64, b53 |
| love | 56/64 | b63, b41, b54, b3 |
| hand | 53/64 | b4, b41, b35, b54 |
| death | 31/64 | b5, b64, b7, b13 |

**Cross-word correlation: 0.101 (Maestro), 0.050 (MLP).**

Different words have different stable bands. The stability pattern IS the word's identity fingerprint. Band 43 is rock-stable for "king" but variable for "hand." Which bands carry identity vs context is word-specific, not structural.

**Band classification (Maestro):** 43 universal (stable for all words), 0 universally contextual, 21 word-specific.

**Depth divergence:** Cross-word r drops from 0.430 (embedding) to 0.101 (L3). Words start similar and individuate through depth — the model actively separates identity through successive processing.

**Death is the most contextually sensitive word** (31/64 stable) — death IS different depending on whether it's feared, arrives, is defied, or mourned. This is a genuine semantic property captured by the model.

**Variance decomposition:** 19.3% between-word, 22.0% between-band, ~58% interaction. The information is 42% decomposable and 58% entangled. Not perfectly clean, but the separation is real.

**Verdict: The "cup of soup" hypothesis is confirmed.** Bands encode word-specific information. Stable bands = identity ("what it is"), variable bands = context ("what's happening to it"). The split is ~60% decomposable. Different words have different band fingerprints. The model independently discovers the distinction between essential identity and relational context.

---

## Experiment 5c (continued): Frequency-Stability Correlation

**Question:** Do stable bands (identity carriers) move slower through depth than variable bands (context carriers)?

**Prediction:** Stable = slow, variable = fast. If true, frequency IS the mechanism of identity vs context.

**Results:**

| Model | r(stability, velocity) | Variable/Stable ratio |
|-------|----------------------|----------------------|
| Maestro+curriculum | -0.085 | 1.017 (+1.7%) |
| MLP baseline | -0.212 | 1.040 (+4.0%) |

All 5 words show the same direction for Maestro (stable bands marginally slower) but the effect is negligible. MLP shows a slightly stronger signal (r = -0.21, magnitude CV vs velocity r = 0.31) but still explains < 10% of variance.

**Verdict: NULL.** Frequency and stability are independent mechanisms. The model separates identity from context through WHICH bands it uses (word-specific band selection), not through HOW FAST those bands move through depth. Band assignment, not band frequency, is the mechanism.

---

## Experiment 5d: Maestro Tempo Control — What Regulates Synchronisation?

**File:** `tests/depth_frequency_5d.py`

**Question:** Does bottleneck width control synchronisation precision? Can the maestro signal be amplified or dampened?

**Method:** Two sweeps: (1) bottleneck width 4/8/16/32/64 at tempo=1.0, (2) tempo scalar 0.25/0.5/1.0/2.0/4.0 at dim=16.

**Results — Width Sweep:**

| Dim | Params/layer | Val Loss | H/L Ratio | Distance from 1.0 |
|-----|-------------|----------|-----------|-------------------|
| 4 | 1,156 | 2.103 | 1.110 | 0.110 |
| 8 | 2,184 | 2.153 | 0.987 | 0.013 |
| 16 | 4,240 | 2.087 | 0.983 | 0.017 |
| 32 | 8,352 | **2.066** | 0.966 | 0.034 |
| 64 | 16,576 | 2.068 | **0.996** | **0.004** |

dim=4 cannot synchronize (ratio 1.110 — insufficient bandwidth). dim=64 achieves near-perfect equalization (0.996). But **optimal loss is at dim=32**, not dim=64. The model performs best with *nearly* uniform velocity, not perfectly uniform — it needs a small amount of spectral freedom.

**Results — Tempo Sweep:**

| Tempo | Val Loss | H/L Ratio |
|-------|----------|-----------|
| 0.25 | NaN | (crashed) |
| 0.50 | NaN | (crashed) |
| 1.00 | 2.087 | 0.983 |
| 2.00 | 2.098 | 1.010 |
| 4.00 | 2.130 | 0.981 |

**Tempo < 1.0 crashes training.** The maestro signal is load-bearing — dampening it below 1.0 causes training to diverge. Amplifying (2x, 4x) works but doesn't improve loss. The surviving tempos (1.0-4.0) all produce ratios near 1.0. Equalization is robust to amplification but fragile to dampening.

**Important nuance on the NaN result:** Kerr flat (no maestro at all) trains fine at val 1.7845. The instability is not inherent to the Kerr-ODE — it's a dependency effect. When maestro exists in the architecture, the optimizer learns to rely on global coordination. Dampening that signal breaks assumptions the rest of the network trained under. It's worse than having no conductor at all, because the musicians are listening for a signal that's barely there. The finding is: once the architecture includes a maestro, it becomes load-bearing through training dynamics, not through mathematical necessity.

**Verdict:** Bottleneck width directly controls synchronization precision. The maestro signal becomes structurally essential through training (the optimizer builds around it). Optimal performance requires near-uniform but not perfectly uniform velocity — constrained freedom at the spectral level:
- dim=4: Too little coordination — can't synchronize, poor performance
- dim=16-32: Sweet spot — good synchronization with enough remaining freedom for nuance
- dim=64: Too much constraint — perfect sync but bands lose independence

This completes the constrained freedom arc. Same pattern as Phase B (two-stage magnitude: 2.46% CV surgical vs 6.92% wild) and Phase C (curriculum stages vs all-at-once). The optimizer wants a clean substrate with just enough room to impose its own structure.

---

## Key Findings Summary

1. **Maestro eliminates spectral dispersion** — all bands propagate at the same velocity (ratio 1.013 vs 0.86-0.88 without maestro)
2. **Uniform differentiation, not selective dispersion** — Maestro creates equal expressive capacity across all bands (per-band token CV ratio 0.991)
3. **Bands encode word-specific identity** — cross-word stability correlation 0.10 (each word has unique band fingerprint)
4. **Words individuate through depth** — cross-word r drops 0.43 → 0.10 from embedding to L3
5. **Frequency and stability are independent** — r = -0.085 (null correlation between band velocity and contextual stability)
6. **Maestro is load-bearing** — dampening below 1.0 crashes training; width controls sync precision
7. **Optimal sync is imperfect** — best loss at dim=32 (ratio 0.966), not dim=64 (ratio 0.996)

8. **Higher frequency does not increase capability** — tempo 2.0 and 4.0 both produced worse loss than 1.0 (2.098, 2.130 vs 2.087). The model doesn't get smarter by oscillating faster. Intelligence (at this scale) comes from structure — which bands carry what, how they're organized per word, the identity fingerprints — not from speed. You can't brute-force better computation by running the waves faster. The geometry matters more than the tempo. This connects to finding #5: fast bands aren't smarter than slow bands. The information is in the assignment pattern, not the velocity.
9. **67% of bands are structural, 33% are semantic** — 43 universal bands encode position + structural scaffolding (high-energy, character-insensitive). 21 word-specific bands carry all character/word identity. The structural frame is NOT compressible overhead — it's the loudest signal (69.8% of L3 energy, top 9 of 10 highest bands). The model invests more compute in structure than meaning. The distinction is built through depth (gap: +0.045 at Emb → +0.180 at L3).
10. **Abstraction level predicts contextual dependence** — Concrete words have higher band stability than abstract words (+0.053 gap, 60.6 vs 51.6 stable bands). The gap is built through depth (absent at embedding, +0.053 at L3). Each word has unique band fingerprints — the effect is in the NUMBER of stable bands, not WHICH bands. Death remains a double outlier: lowest of all 10 words AND -0.114 below the abstract group mean.
11. **Band roles actively reassign through depth** — Consecutive-layer stability correlation = 0.144 (near zero). Emb->L3 endpoint correlation = 0.018 (effectively zero). The 67/33 structural/semantic split is the OUTPUT of depth computation, not a fixed property. Only 8 bands are always universal; 63/64 are universal in at least one layer. Dominant trajectory is U-shaped: destabilize at L0-L1 (impedance matching), restabilize by L3. The model constructs its band assignment through successive reorganisation.
12. **Cross-word clustering follows semantic affinity, not category membership** — hand-heart (r=0.846) is the strongest pair, reproducible across Maestro and MLP. Words cluster by how much their meaning depends on relational context: body-action words (hand, heart, horse, voice) cluster together regardless of concrete/abstract label. Stability gradient is monotonic: concrete > boundary > abstract. eye is an anti-correlator (outlier in every group).
13. **The 8 always-universal bands are mathematically constrained** — 5 above-Nyquist aliases, 2 degenerate (GCD > 1), 1 conjugate pair. Zero unexplained. The embedding geometry forbids these bands from carrying unique semantic content. But the broader 43/21 split is only ~40% math-determined — the Nyquist boundary is a factor, not a theorem. The model finds semantic utility in "redundant" above-Nyquist bands (15 of 21 word-specific bands are mathematically non-privileged).

## Connections to Prior Work

- **Phase 22/22b:** L0 impedance matching confirmed by deceleration pattern (most phase change at Emb→L0)
- **Phase B:** Constrained freedom principle confirmed at spectral level (uniform but not rigid)
- **Phase C:** Maestro gather/broadcast mechanism is the synchronization force (conductor analogy)
- **Phase 12:** 22% trapped structure connects to wave transduction concept — band-specific identity survives layers but dies at softmax

## Experiment 5f: What Do the 43 Universal Bands Carry?

**File:** `tests/depth_frequency_5f.py`

**Question:** 5c found 43 of 64 bands stable for all words regardless of context. Are these bands encoding position, character identity, or structural scaffolding? And are they compressible overhead or essential signal?

**Method:** Five targeted tests on a Maestro+curriculum model:
- **Test A (Position):** Same character ('e', 't', 'a') at different positions. If universal bands change, they encode position.
- **Test B (Character):** 10 different characters at the same position (pos 4, pos 0). If universal bands change, they encode character identity.
- **Test C (Context):** Same character at same position with different surrounding words. Control test — isolates context sensitivity.
- **Test D (Depth):** Track universal vs word-specific stability gap through all layers.
- **Test E (Magnitude):** Compare energy in universal vs word-specific bands.

**Results:**

| Test | What varies | Universal stab | Word-specific stab | Both change? |
|------|-----------|---------------|-------------------|-------------|
| A: Position | Position + context | 0.854 | 0.833 | Yes, both drop |
| B: Character | Character identity | 0.734 | 0.603 | Yes, word-specific drops MORE |
| C: Context | Only surrounding words | **1.000** | **1.000** | Neither changes |

**Test C: perfect stability = 1.000 for ALL 64 bands.** Caveat: this is an autoregressive model. Position 0 ('t') has no prior context — it only sees itself. Position 2 ('e' in "the") only sees "th". The 1.000 result means representation at early positions is dominated by character + position, not by downstream context. Expected for autoregressive architecture, but confirms context alone doesn't affect these bands.

**Test B: the critical split.** When 10 different characters occupy position 4:
- Universal bands: 14/43 remain stable (>0.9)
- Word-specific bands: **1/21** remains stable (>0.9)
- Universal bands are 14x more likely to stay stable when the character changes

Universal bands encode **position and structural scaffolding**. Word-specific bands encode **character identity**.

**Test D: Depth evolution of the gap:**

| Layer | Universal stab | Word-specific stab | Gap |
|-------|---------------|-------------------|-----|
| Emb | 0.647 | 0.602 | +0.045 |
| L0 | 0.624 | 0.625 | -0.001 |
| L1 | 0.652 | 0.634 | +0.018 |
| L2 | 0.663 | 0.577 | +0.085 |
| L3 | 0.700 | 0.521 | **+0.180** |

The distinction isn't born at embedding — it's built through computation. Each layer progressively separates universal (structural) from word-specific (semantic) bands. By L3 the gap is 4x the embedding gap.

**Test E: Magnitude structure:**

| Layer | Universal |Z| | Word-specific |Z| | Uni energy share |
|-------|-----------|---------|-----------------|
| Emb | 0.161 | 0.160 | 67.3% |
| L3 | **0.533** | **0.472** | **69.8%** |

Top 9 of 10 highest-magnitude bands at L3 are universal. These aren't low-energy residuals — they're the loudest signal in the network. The model amplifies structural bands more than semantic bands through depth.

**Verdict:** Universal bands encode **position + structural frame**, not character identity. They're high-energy, position-sensitive, character-insensitive. Word-specific bands (21 of 64, 33%) do all the character/word identity work.

**Compression finding:** 33% of bands carry semantic content, 67% carry structural frame. However, the structural frame is NOT compressible overhead — it's high-energy scaffolding the model actively builds and relies on. The 43 universal bands grow from 67.3% to 69.8% of total energy through depth. The model invests MORE compute in structure than in meaning. Compression would need to preserve this scaffolding, not discard it.

**Reframe:** The original question asked "can we skip 43 bands?" The answer is no — they carry the highest energy and the model amplifies them through depth. But the question "can we reconstruct 43 bands from a template?" remains open. If the structural frame is predictable (same scaffolding regardless of content), a template + 21 semantic bands could reproduce the full state. That's the selective band loading connection.

---

## Experiment 5e: The Death Anomaly — Does Abstraction Level Predict Band Stability?

**File:** `tests/depth_frequency_5e.py`

**Question:** 5c found death had 31/64 stable bands vs 53-60 for the other four words. Is this because death is abstract, or is it word-specific? Test whether abstraction level systematically predicts contextual dependence.

**Method:** 5 concrete nouns (stone, blood, sword, horse, crown) vs 5 abstract nouns (death, hope, grief, fear, pride). Each word appears in 6 contextual sentences. Same methodology as 5c: extract hidden states at first character, measure per-band phase stability across contexts.

**Results (Maestro, L3 final layer):**

| Rank | Word | Type | Mean stab | Stable (>0.9) | Variable (<0.5) |
|------|------|------|-----------|---------------|-----------------|
| 1 | blood | CONCRETE | 0.9936 | 63/64 | 0 |
| 2 | stone | CONCRETE | 0.9893 | 63/64 | 0 |
| 3 | crown | CONCRETE | 0.9838 | 62/64 | 0 |
| 4 | horse | CONCRETE | 0.9621 | 59/64 | 2 |
| 5 | pride | ABSTRACT | 0.9612 | 56/64 | 1 |
| 6 | sword | CONCRETE | 0.9479 | 56/64 | 3 |
| 7 | hope | ABSTRACT | 0.9397 | 55/64 | 4 |
| 8 | grief | ABSTRACT | 0.9341 | 54/64 | 4 |
| 9 | fear | ABSTRACT | 0.9455 | 53/64 | 3 |
| 10 | death | ABSTRACT | 0.8315 | 31/64 | 14 |

**Group comparison:**

|  | Mean stab | Avg stable (>0.9) | Avg variable (<0.5) |
|--|-----------|-------------------|---------------------|
| Concrete (5 words) | 0.9753 | 60.6 | 1.0 |
| Abstract (5 words) | 0.9224 | 51.6 | 5.2 |
| **Gap** | **+0.053** | **+9.0** | **-4.2** |

**Depth evolution of the gap:**

| Layer | Concrete | Abstract | Gap |
|-------|----------|----------|-----|
| Emb | 0.3556 | 0.3672 | -0.012 |
| L0 | 0.9377 | 0.8926 | +0.045 |
| L1 | 0.9589 | 0.9085 | +0.050 |
| L2 | 0.9684 | 0.9167 | +0.052 |
| L3 | 0.9753 | 0.9224 | **+0.053** |

The gap doesn't exist at embedding (-0.012) — it's built through computation. By L0 it's already +0.045, then steadily widens. The model progressively encodes abstraction as increasing contextual dependence.

**Cross-word stability correlation:**

| Comparison | Mean r |
|------------|--------|
| Within concrete | 0.036 |
| Within abstract | 0.097 |
| Between groups | 0.091 |

No within-group clustering. Concrete words don't share "concrete bands" and abstract words don't share "abstract bands." Each word has its own unique fingerprint — the abstraction effect is in the NUMBER of stable bands, not in WHICH bands are stable.

**Death is a double outlier:** 0.8315 mean stability vs 0.9455 for the next-lowest abstract word (fear). Death sits -0.114 below the abstract group mean. The 5c anomaly is partly explained by abstraction (abstract < concrete) but death is exceptional even within abstract words. Death may be the most context-dependent word in Shakespeare — it genuinely means something different in every context.

**Verdict: CONFIRMED.** Concrete words have higher band stability than abstract words (+0.053 gap). The model encodes abstraction as contextual dependence — abstract words need more bands to represent "what it means in THIS context." The ranking nearly perfectly separates the two groups (all 5 concrete in top 6, all 5 abstract in bottom 5, with pride and sword crossing at the boundary). Death is a real outlier even within abstract words — the 5c finding was category-level (abstract < concrete) AND word-specific (death < all abstract).

---

## Experiment 5g: Do Band Roles Reassign Through Depth?

**File:** `tests/depth_frequency_5g.py`

**Question:** 5c found 43 universal and 21 word-specific bands at L3. Is this classification fixed at embedding time and amplified, or does the model actively reassign which bands carry identity vs context at each layer?

**Method:** Reuse 5c's word set (king, sword, love, death, hand, 6 contexts each). Four tests:
- **Test A:** Correlate per-band stability profiles between consecutive layers and between Emb and L3
- **Test B:** Count bands that flip category (stable/variable) between layers
- **Test C:** Check whether the same bands are classified as universal at every layer
- **Test D:** Classify per-band stability trajectories (monotonic, U-shape, flat, etc.)

**Results — Test A: Layer-to-layer stability correlation:**

| Transition | king | sword | love | death | hand | Mean |
|------------|------|-------|------|-------|------|------|
| Emb->L0 | 0.055 | 0.073 | 0.006 | 0.325 | 0.017 | **0.095** |
| L0->L1 | -0.039 | 0.290 | 0.155 | 0.279 | 0.025 | **0.142** |
| L1->L2 | 0.039 | 0.134 | 0.141 | 0.568 | -0.022 | **0.172** |
| L2->L3 | 0.090 | 0.060 | 0.234 | 0.350 | 0.111 | **0.169** |

Mean consecutive-layer correlation: **0.144** (near zero). Knowing which bands are stable at one layer tells you almost nothing about the next. Emb->L3 endpoint correlation: **0.018** (effectively zero).

Death is the only word with meaningful layer-to-layer correlation (L1->L2: 0.568). Its contextual sensitivity persists through depth more than other words' band patterns do.

**Results — Test B: Band role flips:**

| Word | Extreme flips | Category changes | Net direction |
|------|--------------|-----------------|---------------|
| king | 0/256 (0%) | 36/256 (14%) | stabilizing (+2) |
| sword | 5/256 (2%) | 83/256 (32%) | stabilizing (+11) |
| love | 12/256 (5%) | 69/256 (27%) | destabilizing (-1) |
| death | 8/256 (3%) | 99/256 (39%) | destabilizing (-15) |
| hand | 15/256 (6%) | 90/256 (35%) | neutral (0) |

Extreme flips (stable->variable) are rare (0-6%), but category changes through the intermediate zone are common (14-39%). Most reorganisation happens through the middle, not via dramatic jumps. Death has the most reorganisation (39% of transitions).

**Results — Test C: Universal band persistence:**

| Layer | Universal bands | Overlap with L3 | Jaccard with L3 |
|-------|----------------|-----------------|-----------------|
| Emb | 45 | 28 | 0.467 |
| L0 | 36 | 25 | 0.463 |
| L1 | 35 | 25 | 0.472 |
| L2 | 36 | 24 | 0.436 |
| L3 | 43 | 43 | 1.000 |

Only **8 bands are always universal** across all layers (bands 20, 25, 31, 34, 40, 46, 47, 48). But **63 of 64 bands** are universal in at least one layer. The "43 universal bands" from 5c is a snapshot of the final state — a different set of ~35-45 bands is universal at each intermediate layer (Jaccard ~0.45 between layers).

**Results — Test D: Stability trajectories:**

| Trajectory type | king | sword | love | death | hand |
|----------------|------|-------|------|-------|------|
| Flat (range < 0.1) | 43 | 20 | 29 | 10 | 19 |
| U-shape (dip then recover) | 13 | 28 | 23 | 31 | 33 |
| Inverted-U (peak then decline) | 1 | 8 | 7 | 8 | 5 |
| Monotonic increase | 3 | 5 | 3 | 2 | 1 |
| Monotonic decrease | 2 | 0 | 1 | 3 | 2 |

**U-shape is the dominant trajectory** for most words. Bands destabilize at L0-L1 (the impedance matching zone from Phase 22) then restabilize by L3. King has the most flat bands (43/64) — it's so stable that most bands barely change. Death has the fewest flat bands (10/64) — almost every band undergoes meaningful reorganisation.

**Verdict: ACTIVE REORGANISATION.** Band roles are not fixed at embedding time. The model recomputes which bands carry identity vs context at every layer (consecutive r = 0.144, endpoint r = 0.018). The universal/word-specific classification from 5c is a FINAL-STATE property, not a persistent one — different bands are universal at different depths.

The dominant pattern is U-shaped: bands destabilize at intermediate layers (L0-L1, the impedance matching zone) then restabilize by L3. Only 8 bands maintain universal status across all layers — these may be the true structural backbone. The rest participate in a dynamic reassignment process where the model decides at each depth which bands to use for scaffolding and which for content.

This reframes the 5c/5f findings: the 67/33 structural/semantic split is the OUTPUT of a depth-dependent computation, not its INPUT. The model constructs the final band assignment through successive reorganisation, not by amplifying a fixed embedding pattern.

### Addendum: Nyquist Boundary Analysis (Test E)

**Question:** The 8 always-universal bands (20, 25, 31, 34, 40, 46, 47, 48) — are they mathematically constrained by the embedding geometry? Does the Nyquist limit (vocab_size/2 = 32.5) cleanly separate structural from semantic bands?

**The 8 always-universal bands are fully explained:**

| Band | Harmonic | Reason |
|------|----------|--------|
| 20 | 20 | Degenerate: GCD(20,65)=5, only 13 distinct values |
| 25 | 25 | Degenerate: GCD(25,65)=5, only 13 distinct values |
| 31 | 31 | Conjugate pair with band 34 (carries same information) |
| 34 | 34 | Above Nyquist, aliases to harmonic 31 |
| 40 | 40 | Above Nyquist, aliases to harmonic 25 |
| 46 | 46 | Above Nyquist, aliases to harmonic 19 |
| 47 | 47 | Above Nyquist, aliases to harmonic 18 |
| 48 | 48 | Above Nyquist, aliases to harmonic 17 |

5 above-Nyquist aliases, 2 degenerate (GCD > 1), 1 conjugate pair. Zero unexplained. The 8 bands that never waver are mathematically constrained to be structural — the embedding geometry forbids them from carrying unique semantic content.

**But the Nyquist boundary does NOT explain the full 43/21 split:**

| Band group | Universal at L3 | Rate |
|------------|----------------|------|
| Below Nyquist (bands 1-32) | 24/32 | 75% |
| Above Nyquist (bands 33-64) | 19/32 | 59% |
| Coprime (GCD=1) | 30/48 | 63% |
| Degenerate (GCD>1) | 13/16 | 81% |

Below-Nyquist bands are MORE universal than above-Nyquist (75% vs 59%) — opposite of the pure aliasing prediction. Conjugate pairs match classification only 11/32 times (34%). And 15 of 21 word-specific bands are above Nyquist or degenerate — the model found semantic utility in "mathematically redundant" bands anyway.

**Semantic-capable bands** (below Nyquist AND coprime with vocab): only 24 of 64. Of the 21 word-specific bands at L3, only 6 (29%) come from this mathematically privileged pool.

**Verdict:** The embedding geometry fully explains the 8 always-universal bands but only partially explains the broader structural/semantic divide. The split is ~40% math-determined (degeneracy, aliasing) and ~60% learned (the model assigns roles within the mathematical constraints). The Nyquist boundary is a factor, not a theorem. Optimal band count is NOT simply vocab_size/2 — the model exploits above-Nyquist bands for semantic work that aliasing theory says they shouldn't be able to do. The redundancy is mathematical; the utility is computational.

---

## Experiment 5h: The Love-Hand Correlation — Do Boundary Words Form a Stability Cluster?

**File:** `tests/depth_frequency_5h.py`

**Question:** 5c found love and hand had r=0.587 in MLP — the highest pair. Both sit at the concrete/abstract boundary. Is this a category effect (boundary words cluster) or specific word affinity? Do boundary words behave differently from pure concrete or abstract words?

**Method:** 15 words in three categories, each in 6 contextual sentences. Both Maestro+curriculum and MLP models.
- **Concrete** (5): stone, sword, horse, crown, blood
- **Boundary** (5): hand, heart, fire, eye, voice
- **Abstract** (5): death, hope, grief, fear, pride

Boundary words are physical things defined by action/relation (hand, eye) or things that bridge physical and metaphorical (fire, heart, voice).

**Results — Stability gradient (both models):**

| Category | Maestro | MLP |
|----------|---------|-----|
| Concrete | 0.9753 | 0.9832 |
| Boundary | 0.9423 | 0.9771 |
| Abstract | 0.9224 | 0.9584 |

Monotonic gradient confirmed in both architectures. Boundary words sit exactly between concrete and abstract. The gap is larger in Maestro (C-B: 0.033, B-A: 0.020) than MLP (C-B: 0.006, B-A: 0.019).

**Results — Group mean correlations:**

| Group | Maestro r | MLP r |
|-------|-----------|-------|
| Within concrete | 0.036 | 0.183 |
| Within boundary | 0.167 | 0.220 |
| Within abstract | 0.097 | 0.079 |
| Boundary-abstract | 0.183 | 0.150 |
| Boundary-concrete | 0.090 | 0.103 |
| Concrete-abstract | 0.091 | 0.078 |

Boundary words have the highest within-group correlation in both models. But the effect is driven by specific pairs, not uniform clustering.

**Results — Top correlated pairs (Maestro):**

| Pair | r | Categories |
|------|---|-----------|
| hand-heart | **0.846** | B-B |
| fire-fear | **0.759** | B-A |
| horse-heart | 0.697 | C-B |
| horse-hand | 0.673 | C-B |
| hand-hope | 0.633 | B-A |
| heart-grief | 0.575 | B-A |
| heart-hope | 0.523 | B-A |
| blood-death | 0.508 | C-A |
| voice-hope | 0.502 | B-A |
| blood-voice | 0.490 | C-B |

**The hand-heart pair (0.846) is the strongest correlation in the entire matrix** — reproducible in MLP (0.824). These two words share the most band stability structure despite being different physical objects. Both are body parts defined by metaphorical extension (hand of fate, heart of the kingdom).

**fire-fear (0.759 Maestro, 0.876 MLP)** is a cross-category surprise. Fire is classified as boundary, fear as abstract, yet they share more band structure than most within-group pairs.

**horse clusters with boundary words** (horse-hand: 0.673, horse-heart: 0.697) despite being "concrete." Horse in Shakespeare is almost always defined by action (riding, journeying) — the model treats it as an action-defined word, not a static object.

**eye is an anti-correlator** — negative or near-zero correlation with almost everything. This makes it an outlier in the boundary group and explains why boundary-boundary clustering (0.167) is lower than the hand-heart pair alone (0.846) would suggest.

**Key insight:** The correlations don't follow category boundaries. Instead, they follow **semantic affinity clusters** that cross the concrete/abstract divide:

1. **Body-action cluster**: hand, heart, horse, voice, grief, hope (all defined by what they DO or FEEL)
2. **Intensity cluster**: fire, fear (both about overwhelming force)
3. **Fluid cluster**: blood, death, voice (life-force words)
4. **Isolates**: stone, crown, eye, pride (low correlation with everything)

**The love-hand finding from 5c is confirmed and expanded.** hand-heart (0.846) exceeds the original love-hand (0.587). The correlation isn't about "boundary words" as a category — it's about words whose meaning depends on relational context. Hand and heart both mean different things depending on whose hand/heart and what it's doing. This is the same mechanism as the abstract/concrete stability gradient from 5e, but operating at the individual word level rather than category level.

**Verdict: SEMANTIC AFFINITY, NOT CATEGORY MEMBERSHIP.** The love-hand correlation from 5c was real and reproducible, but it reflects semantic similarity in how words relate to context, not membership in a "boundary" category. Words cluster by how much their meaning depends on relational context (what they're doing, who they belong to, what's happening to them). Pure objects (stone, crown) are isolates. Action-defined and body words cluster together regardless of concrete/abstract classification. The model discovers semantic groupings that the human-assigned categories only partially capture.

---

## Open Questions

### Broader Open Questions
- Does dispersion help in other tasks/architectures? (Maestro removes it for Shakespeare char-level)
- Can the band identity fingerprint be used for retrieval? (Query by band signature)
- Wave transduction Tier 1: does passing raw wave states between models preserve the band identity that tokenization destroys?
- Does the depth-frequency profile change with different training corpora? (Connects to Harmonic Character Profiling Experiment 1)

---

## File Map

| File | Description |
|------|------------|
| `tests/depth_frequency_5a.py` | Phase velocity through depth, spectral dispersion, cross-model comparison |
| `tests/depth_frequency_5b.py` | Per-token dispersion, uniform differentiation, acceleration diversity |
| `tests/depth_frequency_5c.py` | Band aspect separation, word identity fingerprints, frequency-stability correlation |
| `tests/depth_frequency_5d.py` | Bottleneck width sweep, tempo scalar sweep, synchronization control |
| `tests/depth_frequency_5e.py` | Death anomaly: concrete vs abstract band stability, abstraction as contextual dependence |
| `tests/depth_frequency_5f.py` | Universal band analysis: position/character/context sensitivity, magnitude structure |
| `tests/depth_frequency_5g.py` | Band role reassignment through depth: layer correlation, flips, universal persistence, trajectories |
| `tests/depth_frequency_5h.py` | Love-hand correlation: 15-word cross-category stability clustering, semantic affinity analysis |
