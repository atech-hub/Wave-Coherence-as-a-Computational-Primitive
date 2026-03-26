# Operating Regime Discovery — Why the Model Couldn't Speak

**Status:** CONFIRMED
**Date:** 2026-03-26
**Engine:** wave-engine (Rust, Apache 2.0)
**Hardware:** Intel i7-14700K, RTX 4070 Ti, 32GB DDR5

---

## Question

The wave-engine builds beautiful internal structure — 0.988 phase clustering, bimodal band census, harmonic coherence between related tokens. The diagnostics prove the wave basis organises itself. But when served through wave-server, the model produces garbage: newlines, special characters, whitespace. No words. No structure. No reasoning.

Why does a model with proven internal structure fail to generate text?

## The Situation

After solving V-shape divergence (AGC investigation) and achieving stable multi-epoch training at 256-dim, we served the best checkpoint (loss 3.77, 448K params, 512 BPE). The output was `\n\n\n\n\n`. At 168-dim BPE (loss 3.76), the output was `%, ), S, *`. The 768-dim Candle model (loss 3.12, 15.5M params) was never tested for output.

Three dimension tiers, two tokenizers, multiple clamp approaches — all producing stable training, all producing garbage output. The architecture appeared broken at a fundamental level.

## The Diagnosis That Failed

Initial hypotheses for why the model couldn't generate text:

1. **Frozen attention can't compose** — attention always routes based on phase similarity (cos(n × Δθ)), can't learn "in this context, attend to that word." Without compositional routing, the model builds representations but can't USE them.

2. **ODE trajectory sensitivity** — the Kerr-ODE amplifies small weight changes through 16 nonlinear integration steps. The loss landscape may be too rough for gradient descent across multiple epochs.

3. **Energy runaway** — the ODE is an open system. No constraint says energy out = energy in. Over training, magnitudes grow, representations drift outward. "A galaxy without dark matter — the stars fly apart."

Each hypothesis was plausible. Each had a test. We spent hours implementing and testing energy conservation (the "dark matter" fix). Results:

| Energy conservation | Best loss | Stability | Output |
|---|---|---|---|
| None | 3.76 | Diverges at 10K (168-dim) | Garbage |
| Hard (E_out = E_in) | 5.57 | Holds but too tight | Garbage |
| Soft (10% budget) | 5.61 | Still diverges | Garbage |
| Per-band-group (5%/20%) | Testing | Testing | — |

Energy conservation helped stability but killed learning depth. The model couldn't reach low enough loss to generate text. We were solving the wrong problem.

## The Turning Point

Marco asked: "What about the char-level model? It actually had more — via the token, it showed signs of language."

He was right. An earlier 168-dim model trained with char-level tokenization had produced this output through LM Studio:

```
ANLIORD:
Cate athe the fother thit ispinemed foremen
Ge andiry longie:
Ang d t lerdnd thisthed thes
```

Character names with colons. Articles ("the"). Word-like structures. Dialogue format. This wasn't a different architecture — it was the SAME architecture with different settings. The model had been on the edge of speaking. We'd walked away from it to pursue BPE.

## Reflecting on What Worked

The char-level model that showed signs of language was trained on the kerr-engine — the wave-engine's predecessor. We compared settings:

| Parameter | Kerr-engine (produced text) | Wave-engine (produced garbage) |
|---|---|---|
| **ODE coupling α, β** | **0.1, 0.1** | **0.01, 0.01** |
| **Out projection** | **Dense (full n_embd × n_embd)** | **Block-diagonal (6 groups)** |
| **AGC ceiling** | **None (no regulation)** | **6.0** |
| Learning rate | 3e-4 | 1e-4 |
| Vocabulary | 65 (char-level) | 512 (BPE) |
| Best loss | 2.05 | 3.76 |
| Trainable params | 354K | 77K |

Three compounding errors. Not one — three. Each individually degraded the model. Together they made it incapable of generating text.

## The Three Errors

### Error 1: ODE coupling reduced to 10% (α=0.01 vs 0.1)

The Kerr-ODE's nonlinear phase shift per step is:
```
δφ = (α + 4β) × M²
```

At α=0.01, mag=2.0: δφ = 0.05 × 4 = 0.2 rad = **11°**. This is barely nonlinear — essentially a damped rotation. The self-phase modulation and cross-phase modulation that give the architecture its computational power are effectively turned off.

At α=0.1, mag=2.0: δφ = 0.5 × 4 = 2.0 rad = **115°**. This is real nonlinear coupling. Bands genuinely interact. The ODE transforms the signal, not just rotates it.

**Why we reduced it:** At 84 bands with BPE tokenization, α=0.1 caused NaN at ~84% rate. We reduced coupling to fix NaN. But the NaN was a magnitude problem — the maestro pushed magnitudes past the ODE's stability threshold. The correct fix was magnitude regulation (AGC), not weakening the coupling.

**The analogy:** It's like turning down the engine power to fix overheating, instead of fixing the cooling system. The engine runs cool at 10% power but can't move the car.

### Error 2: Block-diagonal out_proj starved the model

The out_proj is where the ODE's per-band representations get mixed into token-level predictions. It's the only place where band 1's information can influence band 84's contribution to the output.

| Out proj | Groups | Params per layer | Total model params | Best loss |
|---|---|---|---|---|
| Dense (1 group) | 1 | 28,392 | 171K | **2.25** |
| Block-diagonal (6 groups) | 6 | 4,872 | 77K | 3.02 |

Block-diagonal saves 83% of out_proj parameters at 168-dim. At 768-dim where out_proj is 96.3% of all parameters, this savings is essential. At 168-dim where total params are already tiny, it starves the model of its ability to compose predictions across bands.

**Why we used it:** Block-diagonal was developed for 768-dim where out_proj dominates. We applied it uniformly to all dimensions without testing whether small models could afford the parameter reduction.

### Error 3: AGC ceiling too high for the coupling

The AGC ceiling and ODE coupling are linked by physics:
```
safe_magnitude = √(π/2 / (α + 4β))
```

| α | Safe magnitude | Our ceiling | Problem |
|---|---|---|---|
| 0.01 | 5.6 | 6.0 | Ceiling matches — but coupling too weak to compute |
| 0.1 | 1.77 | 6.0 | Ceiling 3.4× above safe limit — ODE chaos |
| **0.1** | **1.77** | **1.0** | **Ceiling below safe limit — sphere boundary** |

When we corrected the coupling to α=0.1 but kept the ceiling at 6.0, the model produced 779 NaN in 1000 iterations. The magnitudes exceeded the ODE's stability threshold at the higher coupling strength.

The correct ceiling for α=0.1 was discovered by sweep:

| Ceiling | Best loss | NaN |
|---|---|---|
| **1.0** | **2.25** | **0** |
| 1.5 | 2.36 | 0 |
| 2.0 | 2.35 | 0 |
| 6.0 | 4.22 | 779 |

**The finding:** Tighter ceiling = better loss. Ceiling 1.0 forces the model to encode ALL information in phase — the dimension where the architecture is designed to read it. This is the sphere boundary from the spherical investigation: "the circle was always a sphere."

## The Sphere Boundary as Dark Matter

Marco's dark matter analogy was correct, but we implemented it in the wrong place.

We tried energy conservation AFTER the ODE — scaling output energy to match input energy. This worked for stability (the galaxy held together) but killed learning depth (couldn't form stars). The energy conservation was too blunt — it prevented the ODE from doing its job.

The actual dark matter is the AGC ceiling BEFORE the ODE. At ceiling=1.0, the ODE's input magnitudes stay near the unit circle. The ODE transforms phase angles through nonlinear coupling. Magnitude barely changes. All information lives in phase — exactly where cos(n × Δθ) reads it, exactly where harmonic coherence operates, exactly where the band census measures structure.

The spherical investigation (Phase 10) had already proven this:
- Phase carries semantics (20x clustering)
- Magnitude amplifies when phase leads (383x)
- Magnitude is an amplifier, not a carrier

Ceiling=1.0 forces the model to USE the architecture as designed. The binding force isn't energy conservation — it's the sphere boundary that keeps representations on the circle where the harmonic mathematics operates.

## The Combined Fix

All three errors fixed together:

| | Old (garbage) | New (speaks) |
|---|---|---|
| Coupling | α=0.01 (10% power) | **α=0.1 (full power)** |
| Out proj | Block-diagonal 6 groups (77K params) | **Dense (171K params)** |
| AGC ceiling | 6.0 (wrong for any α) | **1.0 (sphere boundary)** |
| Best loss | 3.76 (BPE) / 4.10 (char) | **2.25 (char)** |
| Output | \n\n\n or %, ), S, * | **"the", "you", "she", "our"** |

The model at 171K params, loss 2.25, produces English word fragments, Shakespeare character names with colons, dialogue format, and proper punctuation placement. Not fluent — but structured. The architecture generates text.

## What the Kerr-Engine Knew

The kerr-engine (predecessor) ran at α=0.1 with dense out_proj and no magnitude regulation. It produced word fragments at 354K params, loss 2.05. It worked because:

1. Char-level tokenization at 65 vocab keeps magnitudes naturally low
2. Dense out_proj gives full cross-band mixing
3. Strong coupling enables real nonlinear computation

It also hit limitations — NaN at larger scales, no multi-epoch stability, single ODE solver. The wave-engine was built to fix these limitations. But in fixing them, we accidentally turned off the features that made the kerr-engine work:

- Fixed NaN by reducing coupling (instead of adding AGC) → turned off computation
- Added block-diagonal out_proj for parameter efficiency → starved small models
- Set AGC ceiling from α=0.01 physics → wrong ceiling when coupling is corrected

The wave-engine's improvements (AGC, multi-grid, configurable dims, three tiers) are real and necessary. But they need to be combined with the kerr-engine's operating regime (strong coupling, dense out_proj at small scale) to produce text.

## The Linked Parameter Discovery

The three corrected parameters are not independent. They form a coherent operating regime:

```
Strong coupling (α=0.1) → real nonlinear computation
         ↓
Requires tight AGC ceiling (≤ 1.77, use 1.0) → sphere boundary
         ↓
Phase-only representations → need full cross-band mixing
         ↓
Dense out_proj at small scale → all bands contribute to predictions
```

At larger scales, the regime shifts:

| Scale | Coupling | Ceiling | Out proj | Why |
|---|---|---|---|---|
| 168-dim | α=0.1 | 1.0-2.0 | Dense (1) | Small model needs full mixing and strong computation |
| 256-dim | α=0.05 | 2.5-3.0 | Dense or 4 groups | Transition zone — needs testing |
| 768-dim | α=0.01-0.05 | 3.0-6.0 | Block-diagonal (6) | Out proj dominates — must compress |

The coupling and ceiling are linked by ODE physics. The out_proj choice is linked by parameter budget. Together they define the operating regime for each dimension tier.

## Methodology Lesson

The path to this discovery was non-obvious. We spent a full day on energy conservation, per-band regulation, and frozen attention hypotheses — all reasonable, all wrong.

The breakthrough came from Marco's instinct to go back to basics: "the char-level model actually had more." Instead of theorising about what was wrong, he asked what had worked before. The comparison between the working kerr-engine and the non-working wave-engine revealed three concrete differences that could be tested independently.

The dark matter analogy was also correct — but pointed to the wrong mechanism initially. Energy conservation after the ODE was the wrong implementation. The AGC ceiling before the ODE was the right one. The analogy survived; the implementation changed.

**Lesson for future investigations:** When the model fails to produce expected output, compare against the last configuration that DID produce output. The difference set is finite and testable. Theoretical hypotheses about architectural limitations are unlimited and mostly wrong.

---

## Results Summary

### The progression of output quality

| Config | Loss | Output |
|---|---|---|
| 168-dim, α=0.01, block-diag, BPE | 3.76 | %, ), S, * (garbage) |
| 256-dim, α=0.01, block-diag, BPE | 3.77 | \n\n\n\n (newlines only) |
| 168-dim, α=0.1, block-diag, char | 3.02 | Letters and spaces, no words |
| **168-dim, α=0.1, dense, char, ceiling=2.0** | **2.35** | **Word fragments emerging** |
| **168-dim, α=0.1, dense, char, ceiling=1.0** | **2.25** | **"the", "you", "she", "our"** |

### AGC ceiling sweep (α=0.1, dense out_proj, 168-dim char)

| Ceiling | Best loss | Interpretation |
|---|---|---|
| 1.0 | **2.25** | Sphere boundary — best |
| 1.5 | 2.36 | Slightly worse |
| 2.0 | 2.35 | Similar to 1.5 |
| 6.0 | 4.22 (779 NaN) | ODE chaos |

### Block-diagonal vs dense out_proj (α=0.1, ceiling=2.0, 168-dim char)

| Out proj | Params | Best loss |
|---|---|---|
| Dense (1 group) | 171K | **2.35** |
| Block-diagonal (6 groups) | 77K | 3.02 |

---

## Status

**CONFIRMED.** The wave-engine produces English word fragments at 168-dim with the correct operating regime: α=0.1, AGC ceiling=1.0, dense out_proj, char-level tokenization.

BPE tokenization with the corrected settings is pending testing.

---

## Cross-References

- **ODE regulation investigation:** `investigations/ode-regulation/INVESTIGATION.md` — solved V-shape divergence (AGC), prerequisite for this discovery
- **Spherical investigation:** `investigations/spherical/INVESTIGATION.md` — Phase 10 proved phase carries semantics, magnitude amplifies. Ceiling=1.0 forces phase-only encoding.
- **Multi-grid investigation:** `investigations/multi-grid/INVESTIGATION.md` — engineering application of coprime grid embeddings
- **Frequency-depth investigation:** `investigations/frequency-depth/INVESTIGATION.md` — band census, Nyquist boundary, role reassignment
- **Wave structure investigation:** `investigations/wave-structure/INVESTIGATION.md` — 0.988 phase clustering, bimodal bands confirmed
- **Engine patterns:** ENGINE-PATTERNS.md — Pattern 62 (implicit regularisation via ODE constraints)
- **Origin:** Marco's January 27 conversation — "energy efficiency as the fundamental architectural clue"
- **Sphere boundary origin:** Marco's dark matter analogy — "a galaxy falling apart because there is no dark matter to hold it together"
- **Methodology origin:** Marco's instinct — "the char-level model actually had more"

## Connections to Other Fields

- **Fiber optics:** Launch power in optical fibers is tightly controlled. Too high → nonlinear impairments. Too low → noise dominates. The operating regime is a narrow window where the Kerr effect is useful but not destructive. Our coupling/ceiling pair is the same principle.
- **Radio engineering:** AGC circuits with a rail voltage. The AGC adapts within bounds set by the power supply. Our ceiling is the rail voltage — set by ODE physics, not by tuning.
- **Neuroscience:** Neural oscillations operate within a narrow amplitude range. The critical brain hypothesis: too much excitation → seizure, too little → coma. The healthy brain operates at the edge of chaos. Our ceiling=1.0 places the ODE at its optimal operating point — enough nonlinearity for computation, not enough for chaos.
- **Musical instruments:** A violin string vibrates within a range set by the bridge and nut. Too much force → the string snaps. Too little → no sound. The instrument's design constrains the vibration to the range that produces music. The AGC ceiling constrains the ODE to the range that produces computation.
