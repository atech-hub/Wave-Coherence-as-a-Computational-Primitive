# The Corrector Plate — When the Model Knew the Problem but Lacked the Tool

**Status:** CONFIRMED
**Date:** 2026-03-30
**Engine:** wave-engine (Rust, Apache 2.0)
**Hardware:** Intel i7-14700K, RTX 4070 Ti, 32GB DDR5

---

## Question

The learnable ODE backward (Investigation 13) produced loss 3.18 — a breakthrough. Per-layer coupling self-organised beautifully. But sustained training still produced noisy output. The loss was excellent. The vocabulary was rich. The text was fragmented. Something was degrading the output that wasn't captured by the loss function.

What was corrupting the signal between the model's internal representation and its output?

## The Clue from RF Engineering

Marco found an article about nonlinear distortion in RF power amplifiers. The mathematics were familiar:

- RF power amplifier: `α|x|²·x` — third-order intermodulation distortion
- Wave-engine ODE: `α|ψ|²·ψ` — Kerr self-phase modulation

They are the same equation. The Kerr nonlinearity IS third-order distortion, expressed in the language of nonlinear optics instead of RF engineering. And because the wave-engine uses explicit frequency bands, the distortion products land at predictable harmonic positions — measurable Total Harmonic Distortion (THD).

Dense MLPs have identical nonlinear distortion from their activation functions. But it's invisible — entangled across all dimensions with no frequency structure. The wave-engine makes it measurable because bands are explicit frequencies. This is a unique property of the architecture.

## Building the Distortion Monitor

The existing health monitor measured distortion on a fixed reference sentence — "The cat sat on the mat." The reference sentence had magnitudes around 0.9. The AGC never fired (n_compressed=0). THD read 0.003. Everything looked clean.

But what about the actual training data, where magnitudes grow to 12x and the AGC compresses 4100 bands per iteration?

The batch distortion monitor taps the training cache — the `precond` (ODE input) and `kerr_out` (ODE output) vectors that are already computed and cached for the backward pass. Zero extra forward passes. Zero extra memory. The data is already there. We just weren't looking at it.

Results on actual training data (168-dim BPE, 10K iters):

| Iter | L0 THD | L1 THD | L2 THD | L3 THD | Batch vs Reference |
|------|--------|--------|--------|--------|-------------------|
| 0 | 0.000 | 0.000 | 0.000 | 0.000 | — |
| 2000 | 0.011 | 0.007 | 0.007 | 0.006 | — |
| 5000 | 0.021 | 0.012 | 0.018 | 0.013 | — |
| 8000 | 0.027 | 0.016 | 0.029 | 0.011 | 3× higher than reference |
| 10000 | 0.030 | 0.017 | 0.031 | 0.012 | 3× higher than reference |

Three findings:

1. **THD climbs monotonically.** Distortion is real and accumulates with training.
2. **n_compressed = 0 throughout.** At 168-dim, the AGC never fires. All distortion is pure ODE nonlinearity, not AGC compression.
3. **L3 has the lowest THD (0.012) and stabilises.** The deepest layer — the one closest to the lm_head — minimises its own distortion.

The reference sentence was measuring the clean signal. The training data has 3× more distortion. We were measuring the wrong signal.

## The Model Already Knew

L3 drove its own α to the clamp floor (0.010). Nobody told it to. The optimizer found that reducing L3's self-coupling improved loss — the side effect is less distortion at the output. The model was attempting self-correction.

But α reduction is avoidance, not cancellation. L3 was saying: "I know there's a problem, so I'll create less of it." What it couldn't say was: "Here's the opposite distortion to cancel what came before me."

The model knew the problem. It just didn't have the right tool.

## The Optics Analogy

Marco uploaded a University of Arizona lecture on optical aberrations — lens design from Prof. Jose Sasian. The Cooke triplet example was the key:

Individual surfaces in a multi-element lens have massive aberrations. Surface 1: W₀₄₀=6.77. Surface 3: W₀₄₀=-16.16. Surface 6: W₀₄₀=14.94. But the SUM across all six surfaces is only 2.66. Each surface's aberration is designed to cancel the others.

The lens designer doesn't make each surface perfect. They design the system so imperfections cancel across the optical path.

In the wave-engine, each layer's ODE creates phase distortion. L0's distortion profile could theoretically cancel L2's. But the model needs a tool to apply the correction — and the existing architecture doesn't provide one. The maestro_out and out_proj are linear transforms that mix bands together. They cannot do independent per-band phase correction without magnitude change.

Marco said: "We are inside a sphere — correct AGC alone can't work — we need a mechanism to cancel or minimise the ripple."

Then he said: "What if we give it a tool and see if it will use it?"

## The Corrector Plate

In telescope optics, a Schmidt corrector plate is a thin optical element placed in the light path to cancel aberrations from the primary mirror. The mirror does the heavy lifting (focusing light). The corrector does one thing: fix the phase errors the mirror created. No magnitude change. No mixing. Just per-position phase adjustment.

The wave-engine equivalent: a vector of `n_bands` learnable phase offsets, applied as 2D rotations after the ODE, before the maestro_out. Per-band. Per-layer. Zero-initialised — transparent at start.

```rust
let (sin_c, cos_c) = correction[k].sin_cos();
r_out = r * cos_c - s * sin_c;
s_out = r * sin_c + s * cos_c;
```

The forward is two lines. The backward is five. The cost is 84 parameters per layer, 336 total — 0.1% of the model. Magnitude stays on the sphere. Only phase rotates.

The model starts with all corrections at zero. It earns every correction through gradient descent. If the model doesn't need the corrector, the values stay at zero and it's a no-op.

## What Happened When We Gave It the Tool

### The coupling reorganisation

Without corrector (learnable ODE, 168-dim char-level):
| Layer | α | β | Strategy |
|---|---|---|---|
| L0 | 0.116 | 0.142 | Moderate self-coupling |
| L1 | 0.021 | 0.200 | Cross-band |
| L2 | 0.011 | 0.234 | Cross-band |
| L3 | 0.010 | 0.217 | Everything low — self-protection |

With corrector (same configuration):
| Layer | α | β | Strategy |
|---|---|---|---|
| L0 | **0.314** | 0.022 | **3× higher α** — strong self-coupling |
| L1 | 0.225 | 0.235 | Balanced — sole cross-band mixer |
| L2 | 0.227 | 0.010 | High self-coupling, β at floor |
| L3 | 0.068 | 0.079 | Low but not at floor |

The model completely reorganised its depth strategy. L0 cranked α from 0.116 to 0.314 — three times higher. It was pushing the ODE HARDER because it had a tool to correct the resulting distortion.

It's like giving a race car driver better brakes. They don't drive slower. They drive faster because they can brake later.

Three of four layers chose high self-coupling. Without the corrector, they couldn't afford to — the distortion would accumulate. With it, they were free to use the nonlinearity at full power.

### The loss

| Config | Best loss | Notes |
|---|---|---|
| Without corrector (learnable ODE, sustained 30K) | 3.18 | Previous all-time best |
| **With corrector (char-level, 15K)** | **2.70** | **New best — 0.48 improvement** |
| With corrector (BPE, 6.5K — partial run) | 3.67 | Beat 3.76 baseline at 65% training time |
| With corrector (BPE, 30K sustained) | 3.24 | THD flat throughout |

Loss 2.70 from a 333K parameter model with 336 extra parameters. A 15% improvement from 0.1% more parameters.

### The distortion profile

The sustained 30K BPE run with corrector:

| Iter | L0 THD | L1 THD | L2 THD | L3 THD |
|------|--------|--------|--------|--------|
| 7000 | 0.007 | 0.005 | 0.004 | 0.003 |
| 15000 | 0.008 | 0.006 | 0.005 | 0.005 |
| 25000 | 0.008 | 0.007 | 0.006 | 0.006 |

Compare to without corrector: 0.027-0.032 and climbing.

THD is flat. It's not climbing. The corrector plate reduced distortion 4× AND stopped it from accumulating. L0 has been stable at 0.008 since iter 8000. The depth-decreasing pattern holds: L0 > L1 > L2 > L3 at every checkpoint.

### Dual-channel encoding unlocked

Without corrector in the forward pass:
- θ (per-band) discrimination: 2.17×
- Δθ (cross-band) discrimination: 1.04× — dead

With corrector active in the forward pass:
- θ discrimination: 1.86×
- Δθ discrimination: **1.59×** — alive

The corrector plate made Δθ semantically viable. The model's heavy investment in cross-band coupling (high β) was wasted without the corrector — phase distortion scrambled the cross-band relationships before the lm_head could read them. The corrector cleans up the phase errors and suddenly the Δθ channel carries real semantic signal.

The model didn't just get a new tool. It got a second encoding channel.

## What the Model Was Telling Us

Every degree of freedom we gave the model, it used intelligently:

1. **Learnable α/β:** The model self-organised per-layer coupling in 10K iterations.
2. **Corrector plate:** The model reorganised its entire depth strategy around having phase correction.
3. **γ per band:** The model learned which bands to suppress and which to amplify.

The model was never failing to learn. It was failing to express what it learned, because the architecture constrained what it could do with the nonlinearity. Each new degree of freedom didn't just improve a number. It changed the model's strategy.

L3 drove α to the floor: "I know there's a problem but I can only avoid it."
L0 with the corrector cranked α to 0.314: "Now I can fix the problem, so I'll push harder."

The model was throttling itself. The corrector removed the throttle.

## Technical Notes

### Position-independence
The corrector applies the same phase rotation to all sequence positions. Different tokens produce different magnitudes through the ODE, causing position-dependent distortion. The corrector corrects the average aberration — like a Schmidt plate that corrects the primary mirror's average error but not the field-dependent part. This is a known limitation. Position-dependent correction would require n_bands × seq_len parameters, which is expensive.

### Corrector vs maestro competition
The maestro_out also processes the ODE output. The corrector provides a capability the maestro cannot express — per-band phase rotation without magnitude change, without cross-band mixing. In practice, the model uses both: the corrector handles per-band phase errors, the maestro handles cross-band coordination. They don't compete.

### Checkpoint compatibility
The corrector adds 336 parameters to the checkpoint. Extended checkpoint format (flatten_params_ex) handles this. Old checkpoints load with corrector values at zero (transparent).

## The Lesson

The model knew there was a problem. L3 drove its coupling to the floor to avoid creating distortion. But avoidance is not the same as correction. The Cooke triplet doesn't avoid aberrations — it cancels them with opposite-sign aberrations from different elements.

The corrector plate gave the model the ability to cancel rather than avoid. The cost was 336 parameters — 0.1% of the model. The benefit was loss 2.70 and a complete reorganisation of how the model uses its depth.

When a model throttles itself, don't ask how to make it work harder within the constraint. Ask what constraint it's working around, and remove it.
