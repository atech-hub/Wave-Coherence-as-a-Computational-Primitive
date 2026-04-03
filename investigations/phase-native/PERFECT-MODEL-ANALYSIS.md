# The 55/55 Model — Deep Analysis

**Date:** 2026-04-03
**Architecture:** 4L, 84 bands (168-dim), 4 heads, dense out_proj (groups=1)
**Training:** 40K iters, lr=3e-4, seq=16, no-curriculum, phase-native (dot product loss)
**Data:** arithmetic_augmented.txt (132 facts: original 110 + 22 duplicated hard cases with commutative pairs adjacent)
**Result:** 55/55 (100%) on all valid single-digit additions

---

## How This Model Differs from the 49/55 Baseline

The augmented data (commutative pairs adjacent) produces a model that is **less specialised but more robust**. The architecture is identical — only the training data changed.

---

## ODE Coupling — Less Extreme Depth Specialisation

| Layer | alpha (self) | beta (cross) | beta/alpha | Baseline beta/alpha |
|-------|-------------|-------------|-----------|-------------------|
| L0 | 0.164 | 0.238 | **1.45x** | 1.43x |
| L1 | 0.149 | 0.236 | **1.58x** | 1.75x |
| L2 | 0.038 | 0.251 | **6.58x** | 7.29x |
| L3 | 0.035 | 0.264 | **7.49x** | 15.1x |

L3 beta/alpha **halved** from 15.1x to 7.49x. The perfect model doesn't need L3 to be a pure cross-band mixer. When the training data teaches commutativity directly, L3 can be more balanced.

The two-regime structure (L0-L1 balanced, L2-L3 cross-dominated) still holds, but the gap is narrower. The model distributes computation more evenly across layers.

---

## Layer Signal Flow — The ODE Dominates Everywhere

| Layer | FFN ratio | Attn ratio | Residual ratio | cos(in,out) | Baseline FFN |
|-------|-----------|------------|----------------|-------------|-------------|
| L0 | **0.699** | 0.192 | 1.283 | 0.825 | 0.391 |
| L1 | **0.720** | **0.245** | 1.098 | 0.759 | 0.401 |
| L2 | **0.727** | 0.174 | 0.816 | **0.688** | 0.575 |
| L3 | **0.658** | 0.146 | 0.630 | 0.734 | 0.567 |

**FFN ratio nearly doubled at L0** (0.391 -> 0.699). The ODE does far more work in every layer. The attention contributes less proportionally but L1 attention jumps to 0.245 (from 0.149) — the feature extraction layer pays more attention to the input.

**L2 cos(in,out) = 0.688** — the biggest directional change of any layer in any model tested. The cross-band mixing layer works harder in the perfect model.

**L3 residual ratio = 0.630** — the output is 1.6x larger than the input. L3 is amplifying, not just routing.

---

## ODE Dynamics — Sharper Phase Velocity Gradient

| Layer | Phase velocity | Damping | Energy ratio | Band energy std |
|-------|---------------|---------|--------------|-----------------|
| L0 | **1.69** | 23.7% | 0.763 | **1.11** |
| L1 | 1.93 | 21.8% | 0.782 | 1.04 |
| L2 | 2.11 | 20.3% | 0.798 | 1.15 |
| L3 | **2.31** | 19.5% | 0.805 | 1.27 |

Phase velocity gradient steepened: **L3/L0 ratio = 1.37x** (vs baseline 1.09x). L0 slows down (1.69 vs 2.05) for more careful input conditioning. L3 speeds up (2.31 vs 2.24) for faster output routing.

**L0 band energy std jumped to 1.11** (from 0.82 in baseline). The input layer concentrates energy into specific bands — it does more per-band work instead of uniform conditioning.

Damping gradient preserved: 23.7% (L0) -> 19.5% (L3). The compression-to-conservation pipeline is invariant.

---

## Gradient Flow — L0 Dominates

| Layer | out_proj | ODE | beta_grad | alpha_grad |
|-------|----------|-----|-----------|-----------|
| L0 | 14.4 | **11.8** | **9.06** | 2.87 |
| L1 | 7.7 | 2.1 | 0.82 | 1.09 |
| L2 | 3.2 | 0.7 | 0.23 | 0.32 |
| L3 | 1.4 | **3.1** | **2.43** | 0.83 |

**L0 ODE gradient is enormous (11.8)** — double the baseline (5.0). The beta gradient at L0 is 9.06 (vs baseline 3.34). The perfect model puts most of its gradient budget into L0's cross-band coupling. The input layer works hardest to handle both token orderings.

**Gradient U-shape is back and stronger:** L0=9.06, L1=0.82, L2=0.23, L3=2.43. The edges learn, the middle is settled. L3's gradient is higher than baseline (2.43 vs 0.62) — the output layer is still adapting to the more diverse token orderings.

---

## Attention — L1 Becomes More Active

| Layer | H0 (h=0.4) | H1 (h=0.7) | H2 (h=0.9) | H3 (h=1.1) | Most focused |
|-------|------------|------------|------------|------------|-------------|
| L0 | ent=2.11 | ent=2.64 | ent=2.19 | ent=2.55 | H0 |
| **L1** | ent=2.20 | ent=2.19 | **ent=1.78** | ent=2.66 | **H2** |
| L2 | ent=2.55 | ent=2.50 | ent=1.89 | ent=2.31 | H2 |
| L3 | ent=2.32 | ent=2.51 | ent=2.06 | ent=2.48 | H2 |

**L1 Head 2 (h=0.916) is the most focused head** (entropy 1.78). In the baseline, L0:H2 was most focused. The perfect model shifts attention sharpness from L0 to L1 — the feature extraction layer does the focused discrimination.

L1 attention ratio = 0.245 (highest of any layer). The feature extraction layer is where the model learns commutativity — it needs to identify operands regardless of position.

---

## Output Distribution — Better Calibrated

| Metric | Perfect | Baseline | Meaning |
|--------|---------|----------|---------|
| Entropy | 0.418 | 0.403 | Slightly less confident |
| Margin | 0.831 | 0.811 | Wider average margin |
| Correct rank | **1.2** | 1.5 | More answers at rank 1 |
| Worst margin | 0.004 | 0.038 | Tighter worst case |

The perfect model is **less confident on average** (higher entropy) but has **better correct ranking** (1.2 vs 1.5). More answers in the top position despite less aggressive softmax sharpening.

This is the opposite of the confidence-brittleness pattern seen with dynamic parameters. The augmented data produces a model that is more *accurate* rather than more *confident*.

---

## Loss Trajectory

| Iter | Loss | Interpretation |
|------|------|---------------|
| 0 | 20.47 | Random |
| 5K | 1.32 | Structure forming |
| 10K | 0.98 | Learning arithmetic |
| 20K | 0.67 | Refining answers |
| 30K | 0.36 | Approaching convergence |
| 35K | 0.30 | Still improving |
| Best | **0.213** (iter 36377) | |
| Final avg | 0.332 | |

**Loss is HIGHER than baseline** (0.213 vs 0.195 best, 0.332 vs 0.308 avg). The augmented data is harder to fit because the model must satisfy both `7+2=9` and `2+7=9` simultaneously — it can't exploit position bias.

**Higher loss produces higher accuracy.** This is the key insight: the model that fits the training data worse (by our loss metric) actually generalises better (by our accuracy metric). The augmented data prevents the model from finding the "easy" position-dependent shortcut that the original data allows.

---

## Embedding Space (unchanged)

- Average inter-token distance: 12.084
- Minimum distance: 9.165 (tokens 3 and 13)
- Band utilization: 0.831 mean, band 14 dead
- Effective dimensionality: 70/84

Frozen harmonic embeddings — identical across all models.

---

## Summary: What Perfect Looks Like

The 55/55 model differs from the 49/55 baseline in five ways:

1. **Less extreme L3 specialisation** (beta/alpha 7.5x vs 15.1x) — commutativity from data, not ODE compensation
2. **FFN dominates** (0.66-0.73 ratio) — the ODE does most of the work at every layer
3. **L1 attention sharpens** (ratio 0.245, H2 entropy 1.78) — feature extraction becomes the focus
4. **Phase velocity gradient steepens** (L0 slows, L3 speeds) — more differentiated pipeline
5. **Higher loss, higher accuracy** — can't exploit position shortcuts, learns genuine computation

The architecture was always capable of 55/55. The training data just needed to present commutativity within single context windows so the gradient could teach it.
