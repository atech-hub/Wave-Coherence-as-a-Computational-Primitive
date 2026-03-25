# ODE Magnitude Regulation — Physics-Bounded Adaptive Gain Control

**Status:** ACTIVE
**Started:** 2026-03-25
**Engine:** wave-engine (Rust, Apache 2.0)
**Hardware:** Intel i7-14700K, RTX 4070 Ti, 32GB DDR5

---

## Question

How should the magnitude of signals entering the Kerr-ODE be regulated during training? The maestro pre-conditioner learns to increase magnitudes as training progresses, but the ODE has a physics-based stability limit. What regulation mechanism allows the model to use its full dynamic range without exceeding the ODE's stable operating regime?

This problem is unique to neural architectures using nonlinear ODE layers. Standard transformers don't have it — their MLP magnitudes are controlled by activation functions and layer normalisation. Fiber optics (Kerr NLS) has fixed launch power. The wave-engine has a LEARNED pre-conditioner that actively increases magnitudes, feeding a nonlinear oscillator with a hard stability boundary.

## Background

The Kerr-ODE phase shift per step is:

```
δφ = (α + 4β) × M²
```

where M is the per-band input magnitude. For stability, δφ < π/2 (90°). At α = β = 0.01 with the perturbative ODE (single-step, dt=1.0):

```
M < √(π/2 / 0.05) ≈ 5.6
```

The maestro pre-conditioner transforms the input before the ODE: `precond = x + maestro_in(x)`. As training progresses, the maestro learns that higher magnitudes carry more information through the ODE's nonlinear dynamics. Without regulation, the maestro pushes magnitudes past the stability threshold, causing phase wrapping and NaN.

The original fix (per-band hard clamp at 2.5) prevented NaN but created a new problem: the V-shaped divergence pattern where loss descends, then rises, then partially recovers. Gradient monitors revealed the cause — the maestro was fighting the clamp.

## Method

Five controlled experiments at 256-dim, 12 layers, 512 BPE, 20K iterations, lr=1e-4 cosine schedule. Each test changes only the magnitude regulation mechanism. Gradient monitors log per-iteration: model/lm_head gradient split, per-layer gradient norms, ODE clamp rate, and maximum pre-clamp magnitude.

| Test | Regulation | LR |
|------|-----------|-----|
| A | Hard clamp at 2.5 | 1e-4 |
| B | Hard clamp at 2.5 | 3e-5 |
| C | Hard clamp at 5.0 | 1e-4 |
| D | Soft clamp (tanh), threshold 5.0 | 1e-4 |
| E | Adaptive (AGC), knee compressor | 1e-4 |

---

## Results

### Test A: Hard clamp at 2.5 (baseline)

| Metric | Value |
|--------|-------|
| Best loss | 4.16 at iter 6641 |
| Rolling avg at 8-10K | 6.53 (rising) |
| V-shape divergence | YES — severe |
| Clamp rate progression | 1.3% → 5.9% |
| Max maestro magnitude | 6.8 (2.7× above clamp) |

The maestro learned to push to 6.8 despite the 2.5 ceiling. Clamp rate escalated from 1.3% to 5.9% of bands over training. The gradient distortion from clipping created a feedback loop: push → clip → push harder → more distortion → V-shape divergence.

### Test B: Hard clamp at 2.5, lower LR (3e-5)

| Metric | Value |
|--------|-------|
| Best loss | 4.65 |
| Rolling avg at 8-10K | 6.15 (flat) |
| V-shape | No |
| Clamp rate | 1.4% → 2.3% (stable) |

Lower LR prevents the maestro from pushing hard enough to fight the clamp. Stable but slow — can't reach deep minima. The clamp rate stays low because the maestro never learns aggressively enough to hit the ceiling.

**Implication:** The V-shape correlates with LR, but reducing LR is a workaround, not a fix.

### Test C: Hard clamp at 5.0

| Metric | Value |
|--------|-------|
| Best loss | 3.75 (best of all tests) |
| Rolling avg at 8-10K | 6.06 (flat) |
| Rolling avg at 14-16K | 6.28 (mild rise) |
| V-shape | Delayed — mild rise starts iter 10K |
| Clamp rate progression | 0% → 13% → 30% → 50% → 92% |

Higher threshold gives the maestro room. Best individual loss of any test (3.75). But the maestro outgrows 5.0 too — by iter 14K, 92% of bands are being clamped. The V-shape is delayed but not eliminated.

**Key finding:** ANY fixed threshold will eventually be outgrown. The maestro continuously increases its operating range as it learns.

### Test D: Soft clamp (tanh compression), threshold 5.0

| Metric | Value |
|--------|-------|
| Best loss | 3.83 |
| Rolling avg at 14-16K | 6.02 (descending) |
| V-shape | No — descending through iter 16K |
| Max magnitude (late) | 7.95 |
| Compression zone | 0.8% |

Smooth compression eliminates the V-shape entirely. The maestro pushed to 7.95 but the tanh handled it gracefully — no hard wall, no feedback loop. Rolling average descended through iter 16K, something no previous test achieved.

However, tanh compresses ALL magnitudes, not just above threshold: at mag=4.0, threshold=5.0, the tanh outputs 3.32 (17% reduction on normal signal). This over-compression slows learning compared to Test C.

**Key finding:** Smooth compression eliminates the V-shape. But indiscriminate compression slows learning.

### Test E: Adaptive (AGC) with knee compressor

**Design:** EMA-based Automatic Gain Control. Threshold = mean + 3 × std_dev of observed magnitudes. Knee compressor: below threshold passes unchanged, above threshold smooth compression on excess only.

**Run 1:** AGC adapted correctly (threshold 3.28 → 7.99 in 3600 iters), but a single NaN at iter 3934 poisoned the EMA (mean became NaN), collapsing threshold to the 2.0 floor. The model trained the remaining 16K iters at clamp=2.0 — worse than Test A.

**Fix:** NaN guard — filter non-finite magnitudes before EMA update.

**Run 2 (with NaN guard):** AGC adapted without interruption:

| Iter | AGC Threshold | EMA Mean | EMA Std | Clamps | Max Mag |
|------|--------------|----------|---------|--------|---------|
| 0 | 3.28 | 1.28 | 0.67 | 89 | 5.25 |
| 1200 | 4.56 | 1.51 | 1.02 | 0 | 4.53 |
| 2700 | 5.72 | 1.83 | 1.30 | 3 | 5.85 |
| 3900 | 6.57 | 1.99 | 1.53 | 0 | 5.07 |
| 4200 | 8.57 | 2.29 | 2.09 | 0 | 6.69 |
| 4400 | 10.03 | 2.56 | 2.49 | 0 | 4.63 |
| 4436 | — | — | — | NaN starts | — |

The threshold climbed to 10+ because the 3-sigma formula doesn't account for ODE physics. At threshold 10, magnitudes of 7+ entered the ODE, causing δφ > 180° (chaotic regime). NaN rate reached 48%.

**Key finding:** The AGC concept works (correct adaptation from 3.28 to 5.5) but needs a physics-based ceiling. The ODE stability constraint is non-negotiable.

---

## Key Finding: Physics-Bounded Adaptive Regulation

The regulation system must satisfy three constraints simultaneously:

1. **Floor (min_threshold = 2.0):** Below this, the ODE receives too little signal for meaningful computation. The model can't learn.

2. **Ceiling (max_threshold ≈ 6.0):** Above this, the ODE phase shift exceeds 90° and dynamics become chaotic. Derived from: M < √(π/2 / (α + 4β)) at α = β = 0.01 → M < 5.6. Rounded to 6.0 for margin.

3. **Adaptive within bounds:** The threshold tracks the maestro's actual operating range via EMA. Only true outliers (above mean + 3σ) get compressed. The knee compressor passes normal magnitudes unchanged.

### Comparison of all approaches

| Approach | Floor | Ceiling | Adaptive | V-shape? | Best loss |
|----------|-------|---------|----------|----------|-----------|
| Hard clamp 2.5 | 2.5 | 2.5 | No | YES | 4.16 |
| Hard clamp 5.0 | 5.0 | 5.0 | No | Delayed | 3.75 |
| Soft tanh 5.0 | ~0 | ~5.0 | No | No | 3.83 |
| AGC (no ceiling) | 2.0 | ∞ | Yes | N/A (NaN) | 4.57 |
| **AGC + ceiling** | **2.0** | **6.0** | **Yes** | **NO** | **3.76** |

### The electronics progression

The progression maps directly to signal processing hardware:

| Stage | Electronics | Wave-engine | Problem solved | Problem remaining |
|-------|------------|-------------|----------------|-------------------|
| 1 | Fixed resistor | Hard clamp 2.5 | Prevents NaN | Clips signal, V-shape divergence |
| 2 | Larger resistor | Hard clamp 5.0 | Delays clipping | Eventually outgrown (92% clamped) |
| 3 | Zener diode | Tanh soft clamp | No hard cliff | Over-compresses normal signal (17%) |
| 4 | AGC circuit | Adaptive threshold | Adapts to signal | No upper bound → ODE blows up |
| 5 | AGC + rail voltage | AGC + physics ceiling | Adapts within stable range | Pending test |

The "rail voltage" is the ODE stability limit — the maximum magnitude the oscillator can process without phase wrapping. This is a physical constant of the system, not a tunable parameter.

---

## ODE Stability Derivation

The Kerr-ODE derivative at band k:

```
dZ_k/dt = (-γ_k + i·ω_k)·Z_k + i·(α|Z_k|² + β·Σ|Z_neighbours|²)·Z_k
```

The nonlinear phase shift per integration step:

```
δφ = (α·|Z_k|² + β·Σ|Z_neighbours|²) × dt
```

For the perturbative ODE (dt = 1.0, single step):

```
δφ_max = (α + 4β) × M²    (4 neighbours at maximum)
```

Stability requires δφ < π/2:

```
M < √(π/2 / (α + 4β))
```

At α = β = 0.01:  M < √(π/2 / 0.05) ≈ **5.6**

At α = β = 0.1:   M < √(π/2 / 0.5) ≈ **1.77**

This explains why α=0.1 at 84 bands caused immediate NaN (the maestro easily pushes past 1.77) while α=0.01 was stable until the clamp intervened.

For the RK4-16 solver (dt = 1/16 per substep):

```
M < √(π/2 / (0.05 × 1/16)) ≈ 22.4
```

The perturbative ODE is the binding constraint — its single-step integration requires much tighter magnitude control than RK4-16.

---

## Gradient Monitor Data

The gradient monitors (implemented during this investigation) revealed the mechanism:

### Clamp rate correlates with divergence

| Phase | Iters | Clamp rate | Max mag | What's happening |
|-------|-------|-----------|---------|-----------------|
| Learning | 0-2K | 1.0-1.6% | 4-5 | Maestro learning gently |
| Peak | 3-5K | 4.2-4.8% | 6.4 | Maestro pushing hard — best losses here |
| Pullback | 5-8K | 1.6-1.8% | 4-5 | LR decaying, maestro calms |
| Second spike | 8.5-10K | 3.3-3.4% | 4.8 | Loss starts rising |
| Late | 12-14K | 3.5-4.9% | 6.8 | Clamp climbing again — V-shape |

### Head gradient share is healthy

The lm_head gradient share stays proportional to its parameter share (28-31%) — it doesn't progressively dominate. The divergence is NOT caused by gradient imbalance. This ruled out hypothesis 3 from our differential diagnosis.

### Per-layer gradients show no vanishing

All 12 layers maintain gradient norms of 0.15-0.45 throughout training. No vanishing gradient problem. The issue is specifically the ODE input magnitude, not the gradient flow.

---

## Status

| Approach | Status |
|----------|--------|
| Hard clamp 2.5 | REPLACED — proven V-shape cause |
| Hard clamp 5.0 | TESTED — works as stopgap, outgrown by iter 14K |
| Soft tanh 5.0 | TESTED — eliminates V-shape, over-compresses normal signal |
| AGC (no ceiling) | TESTED — adapts correctly but exceeds ODE stability |
| AGC + NaN guard | TESTED — NaN guard works, threshold still unbounded |
| **AGC + physics ceiling** | **CONFIRMED — best loss 3.76, avg 5.86, zero V-shape, zero NaN** |

The AGC with max_threshold = 6.0 is **CONFIRMED** as the permanent solution. Rolling averages descended monotonically through 20K iterations (6.19 → 5.86) where every previous approach diverged. Best loss 3.76 matches the hard-clamp-5.0 result but with stable averages (5.88 vs 6.31). Zero NaN. The maestro operates freely at mag 5.1-5.6, the ceiling holds at 6.0, and the knee compressor only activates on true outliers (1.4% of bands at iter 19K).

This is now the standard ODE magnitude regulation for all tiers. Committed to wave-engine main branch 2026-03-25.

---

## Cross-References

- ODE coupling scaling: [wave-structure investigation](../wave-structure/INVESTIGATION.md) — finding on α/β vs band count
- Multi-grid embeddings: [multi-grid investigation](../multi-grid/) — the geometric fix that works alongside the dynamical fix
- Maestro ceiling: wave-engine results/MAESTRO-CEILING-FINDING.md — maestro_dim=16 is universal
- Per-band clamp origin: Marco's "capacitors regulate current" analogy (2026-03-23)
- AGC concept origin: Marco's "let the model find its sweet spot, not us" (2026-03-25)
- Pattern candidate for ENGINE-PATTERNS.md: "Physics-Bounded Adaptive Regulation"

## Connections to Other Fields

- **Fiber optics:** Secondini et al. (ESSFM) and Lin et al. (perturbation-aided DBP) — the source papers for the perturbative ODE. Their launch power control is the physical analogue of our magnitude regulation.
- **Radio engineering:** AGC circuits (1930s onward) — the direct analogue for adaptive threshold regulation. Per-channel AGC in multi-channel receivers matches per-layer AGC in the wave-engine.
- **Audio engineering:** Compressor vs limiter distinction — the tanh (limiter) vs knee compressor (studio compressor) maps directly to our Test D vs Test E.
- **Control theory:** The regulation problem is a form of constrained adaptive control where the plant (ODE) has a hard stability boundary and the controller (maestro) is learning.
