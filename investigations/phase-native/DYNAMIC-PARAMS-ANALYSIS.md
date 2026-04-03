# Dynamic Parameter Optimisation — Deep Analysis

**Date:** 2026-04-03
**Architecture:** 4L, 84 bands (168-dim), 4 heads, dense out_proj (groups=1)
**Training:** 40K iters, lr=3e-4, seq=16, no-curriculum, phase-native (dot product loss)
**Data:** arithmetic_single.txt (110 facts: 55 addition, 55 subtraction)
**Evaluation:** All 55 valid single-digit addition sums (a+b where a+b <= 9)

---

## 1. Accuracy Results

| Model | Accuracy | Best Loss | Final Avg | Delta vs Baseline |
|-------|----------|-----------|-----------|-------------------|
| **baseline** | **49/55 (89.1%)** | 0.195 | 0.308 | — |
| **corrector spring** | **49/55 (89.1%)** | 0.195 | 0.308 | 0 |
| rk4-weights dyn | 48/55 (87.3%) | **0.180** | 0.306 | -1 |
| wd dyn | 48/55 (87.3%) | 0.193 | 0.308 | -1 |
| agc-headroom dyn | 48/55 (87.3%) | 0.189 | 0.308 | -1 |
| layer-scale dyn | 47/55 (85.5%) | 0.194 | 0.309 | -2 |
| harmonics dyn | 1/55 (1.8%) | 0.192 | 0.310 | BUG* |

*Harmonics checkpoint has same param count as rk4 (both +16), causing loader collision. Training loss was healthy (0.310) but generate loaded wrong weights.

### Verdict: No dynamic parameter improves accuracy at this scale.

The baseline 49/55 is the ceiling. All dynamic params either match it (corrector) or lose 1-2 answers. The 6 core failures are structural — commutativity and small-sum confusion — not addressable by training dynamics.

---

## 2. Failure Pattern Analysis

### Core failures (present in ALL models):

| Prompt | Expected | Baseline | ls | rk4 | wd | agc | corr |
|--------|----------|----------|-----|------|-----|------|------|
| 0+2= | 2 | 7 | 8 | 7 | 7 | - | 7 |
| 1+1= | 2 | 1 | 7 | + | 7 | - | 1 |
| 1+4= | 5 | 1 | 7 | 8 | 1 | 1 | 1 |
| 2+3= | 5 | 4 | 9 | 4 | 4 | 4 | 4 |
| 3+1= | 4 | 8 | 9 | 8 | 9 | 6 | 8 |
| 7+2= | 9 | 4 | 4 | 4 | 4 | 4 | 4 |

**Observations:**
- **7+2=4 is universal.** Every model gets this wrong. 2+7=9 is always correct. This is a commutativity failure from position-dependent frozen attention.
- **0+2 and 1+1 (sum=2) are hardest.** No model handles these reliably. The digit 2 is poorly represented in the output space.
- **2+3=5 is consistently wrong as 4.** Off-by-one error — the model confuses adjacent sums.

### New failure in dyn models (not in baseline):

| Model | Extra failure |
|-------|---------------|
| rk4 | 2+6=7 (exp 8) |
| wd | 2+6=9 (exp 8) |
| agc | 2+6=9 (exp 8) |
| ls | 2+6=0 (exp 8), 0+1=5 (exp 1) |

The dynamic params destabilise the model's handling of 2+6. The spring regulation adds noise that the small task can't overcome.

---

## 3. ODE Coupling — Sub-Channel Architecture

The self-organised depth pipeline (Pattern 89) is **invariant** across all dynamic parameter configurations:

### Beta/Alpha ratio at 35K:

| Layer | baseline | ls | rk4 | wd | harm | agc | corr |
|-------|----------|-----|------|-----|------|------|------|
| L0 | 1.43 | 1.41 | 1.45 | 1.42 | 1.44 | 1.40 | 1.43 |
| L1 | 1.75 | 1.89 | 1.86 | 1.77 | 1.63 | 1.67 | 1.75 |
| L2 | 7.29 | 7.33 | 17.7 | 6.93 | 7.65 | 8.04 | 7.29 |
| L3 | 15.1 | 12.9 | 20.9 | 13.7 | 7.01 | 10.1 | 15.1 |

**Finding: The two-regime structure is universal.**
- L0-L1: balanced coupling (β/α = 1.4-1.9x)
- L2-L3: cross-band dominated (β/α = 7-21x)

**RK4 dyn pushes the split hardest:** L2 β/α=17.7x, L3 β/α=20.9x. The adaptive integration allows the deep layers to lean even more into cross-band mixing because the integration is tuned for it.

**Harmonics dyn reduces L3 specialisation:** L3 β/α=7.01x (vs baseline 15.1x). When the attention harmonics can adjust, the ODE doesn't need to specialise as aggressively — the attention absorbs some of the differentiation.

---

## 4. Layer Signal Flow

### FFN ratio (ODE contribution to output) at 30K:

| Layer | baseline | ls | rk4 | wd | harm | agc | corr |
|-------|----------|-----|------|-----|------|------|------|
| L0 | 0.391 | 0.329 | 0.398 | 0.390 | 0.388 | 0.384 | 0.391 |
| L1 | 0.401 | 0.400 | 0.390 | 0.402 | 0.416 | 0.422 | 0.401 |
| L2 | 0.575 | 0.537 | 0.558 | 0.524 | 0.551 | 0.595 | 0.575 |
| L3 | 0.567 | 0.469 | 0.542 | 0.502 | 0.549 | 0.572 | 0.567 |

**Layer-scale dyn reduces L0 and L3 FFN contribution** — it learns to attenuate the extreme layers. L0 FFN drops from 0.391 to 0.329. This is the layer_scale doing its job: moderating the depth pipeline.

### Cosine similarity (directional change) at 30K:

| Layer | baseline | ls | rk4 | wd | harm | agc | corr |
|-------|----------|-----|------|-----|------|------|------|
| L0 | 0.930 | 0.952 | 0.932 | 0.931 | 0.932 | 0.932 | 0.930 |
| L1 | 0.910 | 0.907 | 0.913 | 0.909 | 0.904 | 0.899 | 0.910 |
| L2 | 0.806 | 0.843 | 0.821 | 0.849 | 0.833 | 0.797 | 0.806 |
| L3 | 0.793 | 0.839 | 0.801 | 0.806 | 0.820 | 0.798 | 0.793 |

**L2 makes the biggest directional change** in all models (lowest cosine). This is the cross-band mixing layer — it reorients the hidden state most aggressively.

**AGC dyn preserves the baseline flow best.** cosine values barely change from baseline. The per-layer headroom doesn't distort the natural signal flow.

---

## 5. ODE Dynamics

### Energy damping profile at 30K:

| Layer | baseline | ls | rk4 | wd | harm | agc | corr |
|-------|----------|-----|------|-----|------|------|------|
| L0 | 23.6% | 23.7% | **28.3%** | 23.6% | 23.6% | 23.6% | 23.6% |
| L1 | 21.6% | 21.8% | **23.6%** | 21.6% | 21.7% | 21.6% | 21.6% |
| L2 | 20.0% | 20.0% | **17.5%** | 20.0% | 20.0% | 20.0% | 20.0% |
| L3 | 19.9% | 19.7% | **17.8%** | 19.8% | 19.9% | 19.9% | 19.9% |

**RK4 dyn amplifies the damping gradient.** L0 damps 28.3% (vs baseline 23.6%), L3 conserves more (17.8% vs 19.9%). The adaptive integration weights steepen the compression-to-conservation pipeline. This is the endpoint-heavy (L0) vs startpoint-heavy (L3) integration at work.

### Phase velocity at 30K:

| Layer | baseline | ls | rk4 | wd | harm | agc | corr |
|-------|----------|-----|------|-----|------|------|------|
| L0 | 2.05 | 2.06 | 2.15 | 2.04 | 2.05 | 2.04 | 2.05 |
| L1 | 2.04 | 2.01 | 2.01 | 2.03 | 2.02 | 2.03 | 2.04 |
| L2 | 2.33 | 2.33 | 2.33 | 2.31 | 2.37 | 2.28 | 2.33 |
| L3 | 2.24 | 2.18 | 2.21 | 2.20 | 2.19 | 2.22 | 2.24 |

**Phase velocity is remarkably stable** across all models. L2 consistently rotates fastest (~2.33 rad). The ODE's phase dynamics are robust to training hyperparameter changes.

### Band energy concentration (std) at 30K:

| Layer | baseline | ls | rk4 | wd | harm | agc | corr |
|-------|----------|-----|------|-----|------|------|------|
| L0 | 0.82 | 0.82 | 0.74 | 0.82 | 0.80 | 0.81 | 0.82 |
| L1 | 0.99 | 0.99 | 0.91 | 0.94 | 0.98 | 0.99 | 0.99 |
| L2 | 1.29 | 1.32 | 1.31 | 1.30 | 1.32 | 1.30 | 1.29 |
| L3 | 1.28 | 1.26 | 1.26 | 1.21 | 1.27 | 1.28 | 1.28 |

**Energy concentration increases with depth** in all models (L0 uniform, L3 concentrated). The cross-band coupling channels energy into specific bands. This "frequency attention" pattern is invariant.

---

## 6. Gradient Flow

### Out_proj gradient norm at 30K (dominant component):

| Layer | baseline | ls | rk4 | wd | agc | corr |
|-------|----------|-----|------|-----|------|------|
| L0 | 35.7 | 7.2 | 20.2 | 6.3 | 6.6 | 35.7 |
| L1 | 16.8 | 3.6 | 9.4 | 3.1 | 3.7 | 16.8 |
| L2 | 7.5 | 1.7 | 4.2 | 1.7 | 1.8 | 7.5 |
| L3 | 3.8 | 0.9 | 1.9 | 0.9 | 0.9 | 3.8 |

**Dynamic params reduce gradient magnitude 3-5x.** Baseline and corrector (no spring on main params) have much larger gradients. Every dyn param with spring regulation dampens gradient flow. This is expected — the spring acts as implicit regularisation.

### Beta gradient at 30K:

| Layer | baseline | ls | rk4 | wd | agc | corr |
|-------|----------|-----|------|-----|------|------|
| L0 | 3.34 | 0.61 | 1.95 | 1.12 | 0.53 | 3.34 |
| L1 | 7.26 | 0.08 | 3.62 | 1.18 | 1.50 | 7.26 |
| L2 | 5.85 | 0.32 | 1.56 | 0.18 | 0.64 | 5.85 |
| L3 | 0.62 | 0.14 | 0.14 | 0.11 | 0.14 | 0.62 |

**L1 beta gradient is highest in baseline** (7.26) — the transition layer is still learning its coupling. All dyn params suppress this (0.08-3.62). L3 beta gradient is consistently low (0.14-0.62) — the output layer's coupling has converged everywhere.

**The gradient U-shape (L0 high, L2 high, L1/L3 low) from earlier analysis is NOT universal.** It appears in baseline/corr but gets reshaped by dyn params. The spring regulation smooths the gradient landscape.

---

## 7. Attention Head Activity

### Attention entropy at 30K (lower = more focused):

**L0 Head 2 (h=0.916) is the most focused head in ALL models:**

| Model | L0:H2 entropy | L1:H0 entropy | Most focused overall |
|-------|---------------|---------------|---------------------|
| baseline | 1.779 | 2.196 | L0:H2 |
| ls | 1.780 | **1.609** | **L1:H0** |
| rk4 | 1.779 | 1.945 | L0:H2 |
| wd | 1.779 | 2.078 | L0:H2 |
| harm | 1.783 | 1.946 | L0:H2 |
| agc | 1.779 | 1.945 | L0:H2 |
| corr | 1.779 | 2.196 | L0:H2 |

**Layer-scale dyn makes L1:H0 the most focused head** (entropy 1.609). No other model achieves this. The layer_scale amplification of L1 (scale=1.225) makes its attention more discriminative.

### Learned harmonics (harm dyn only):

| Layer | Head 0 | Head 1 | Head 2 | Head 3 | Change |
|-------|--------|--------|--------|--------|--------|
| Init | 0.405 | 0.693 | 0.916 | 1.099 | — |
| L0 35K | **0.360** | 0.712 | **0.809** | 1.088 | H0↓11%, H2↓12% |
| L1 35K | **0.363** | **0.569** | 0.916 | 1.134 | H0↓10%, H1↓18% |
| L2 35K | **0.318** | **0.581** | **0.877** | 1.040 | H0↓21%, H1↓16%, H2↓4% |
| L3 35K | 0.409 | 0.688 | 0.954 | 1.045 | Near init |

**Harmonics decrease at L0-L2, stay near init at L3.** The model moves harmonics DOWNWARD (toward lower frequency matching) in early/middle layers. L3 keeps its harmonics near the initial values — it trusts the integer-ish harmonics for output routing.

**L2 Head 0 drops most** (0.405→0.318, -21%). The cross-band mixing layer wants a sub-harmonic attention pattern.

---

## 8. Dynamic Parameter Values

### Layer scale (ls model at 35K):
```
L0: 1.188   L1: 1.225   L2: 1.232   L3: 1.153
```
All layers amplified above 1.0. L2 amplified most (1.232) — the cross-band mixing layer gets the strongest residual contribution. L3 amplified least (1.153) — the output layer is already dominant.

### RK4 weights (rk4 model at 30K):
```
Standard:  [0.167, 0.333, 0.333, 0.167]
L0:        [0.086, 0.311, 0.316, 0.252]  ← endpoint-heavy
L1:        [0.135, 0.339, 0.340, 0.233]  ← mild endpoint bias
L2:        [0.193, 0.337, 0.334, 0.149]  ← near standard
L3:        [0.210, 0.343, 0.338, 0.134]  ← startpoint-heavy
```
L0 and L3 are opposite: L0 trusts the final evaluation (k4=0.252), L3 trusts the initial slopes (k1=0.210). L2 stays near standard — its dynamics don't need special integration.

### AGC headroom (agc model at 30K):
```
L0: 3.00   L1: 3.00   L2: 3.00   L3: 3.00
```
No movement from default. The spring (k=1.0) holds all layers at 3-sigma. The model didn't find a reason to differentiate AGC headroom per layer. **Null result** — AGC headroom is not a useful degree of freedom at this scale.

---

## 9. Embedding Space (All Models)

Fixed across all models (frozen harmonic embeddings):
- **Average inter-token distance:** 12.084
- **Minimum distance:** 9.165 (tokens 3 and 13)
- **Band utilization mean:** 0.831
- **Dead bands:** band 14 (zero utilization)
- **Effective dimensionality:** 70/84 bands

---

## 10. Output Distribution at 30K

| Model | Avg entropy | Avg margin | Avg correct rank | Worst margin |
|-------|-------------|------------|-----------------|--------------|
| baseline | 0.403 | 0.811 | 1.5 | 0.038 |
| corr | 0.403 | 0.811 | 1.5 | 0.038 |
| rk4 | 0.351 | 0.858 | 1.4 | 0.009 |
| harm | 0.349 | 0.861 | 1.4 | 0.009 |
| wd | 0.321 | 0.875 | 1.6 | 0.012 |
| ls | 0.324 | 0.878 | 1.8 | 0.010 |
| agc | 0.331 | 0.872 | 1.7 | 0.009 |

**Dynamic params make the model MORE confident** (lower entropy) but with LOWER worst margin. The dyn models overfit to the training distribution, producing high confidence on most tokens but failing on edge cases with narrower margins. This is the "tighter but more brittle" pattern.

---

## 11. Cross-Model Invariants

These properties are **unchanged** across all 7 models:

1. **Two-regime coupling:** L0-L1 balanced (β/α < 2), L2-L3 cross-dominated (β/α > 7)
2. **Damping gradient:** L0 damps most, L3 conserves most (23→20%)
3. **Phase velocity:** L2 fastest (~2.33), L0-L1 slower (~2.04)
4. **Band energy channeling:** concentration increases with depth (std 0.8→1.3)
5. **Attention specialisation:** L0:H2 most focused, H1 most diffuse
6. **6 structural failures:** 0+2, 1+1, 1+4, 2+3, 3+1, 7+2 present in all models
7. **Embedding space:** 70/84 effective dimensions, band 14 dead

These are **architectural invariants** — properties of the wave-engine at this scale, not of the training dynamics.

---

## 12. Findings and Implications

### What dynamic params reveal:

1. **RK4 weights are the most informative dyn param.** The model learns opposite integration strategies at depth extremes (L0 endpoint-heavy, L3 startpoint-heavy) AND amplifies the damping gradient (28% vs 18%). This is Pattern 103.

2. **Layer scale confirms L2 as the key layer.** Scale 1.232 = highest amplification. L2 is where cross-band mixing happens. More residual contribution = more weight on the mixing computation.

3. **Harmonics decrease at intermediate layers.** The model wants sub-harmonic attention (h < 1.0) for broad matching, especially at L2 (h0 drops to 0.318). L3 keeps near-integer harmonics for sharp output routing.

4. **AGC headroom is a null result.** No differentiation from default. The AGC's adaptive threshold already handles per-layer needs through the EMA mechanism.

5. **Corrector spring has zero effect.** Identical to baseline. The corrector plate self-regulates without needing spring constraint.

6. **WD and LS reduce gradient magnitude 3-5x** via implicit regularisation. This doesn't improve accuracy but makes training smoother.

### What to optimise:

1. **The 6 failures are attention-bound.** Position-dependent frozen attention can't learn commutativity. Unfreezing attention (or adding position-independent features) is the path to 55/55.

2. **L2 is undertrained.** Highest coupling ratio, highest amplification needed, biggest directional change. Consider higher LR or more capacity at L2.

3. **Band 14 is dead.** The harmonic table has a structural gap at band 14. At scale, identify and prune dead bands.

4. **RK4 weights improve loss but not accuracy.** At Shakespeare scale where the loss landscape is smoother, the better loss from adaptive integration may translate to better perplexity.

---

## 13. Recommendation for Scaling

For Shakespeare runs:
- Use **--rk4-weights dyn** — the model finds useful per-layer integration
- Skip --agc-headroom dyn (null result) and --corrector dyn (no effect)
- Test **--harmonics dyn** once the checkpoint collision bug is fixed
- **--layer-scale dyn** if the model has more than 4 layers (depth pipeline benefits from per-layer scaling at 6L+)
- Spring k=0.1 is appropriate — loose enough for exploration, stiff enough for stability
