# Architecture Boundaries: Where Harmonic Structure Helps and Where It Does Not

This document details the established boundaries of the wave coherence framework within transformer architectures. Every boundary is backed by experimental evidence across Phases 17-22d and the Phase A/B integration tests.

---

## Summary

| Layer | Harmonic structure | Result | Evidence |
|-------|-------------------|--------|----------|
| Embeddings | **Helps** | Frozen outperforms learned by 2.8% | Phases 1-16, 17, A, B |
| Retrieval | **Helps** | Per-channel sweep beats cosine similarity | Tests 21, 24, Phase 16 |
| FFN computation | **Partially replaces** | Kerr-ODE at 98.1% of MLP, 44% params | Phases 20-22d, A, B, C |
| Attention Q/K | **Hurts** | Must remain unconstrained | Phases 18, 19, 19b |
| Weight matrices | **No effect** | Spectrally flat regardless | Phases 17, 17b |

---

## The Substrate Incompatibility Principle

Matrix multiplication is structurally blind to frequency. A matrix treats every element as an independent grid position — row 3, column 7 has no relationship to row 3, column 8. It cannot know that column pairs encode cos/sin of the same harmonic. When harmonic embeddings pass through matmul, the wave structure is invisible to the operation.

This is analogous to pushing analogue waves through transistors (discrete switches). The transistor is structurally incompatible with continuous signals. A capacitor stores and releases charge continuously, responds to rate-of-change, and naturally selects frequencies. Capacitors and inductors form resonant LC circuits — native wave processors.

Matrix multiplication is the transistor of neural computation. Harmonic coherence is the capacitor. The framework works for representation (vectors you can decompose) and retrieval (comparison via frequency-aware functions). It fails inside the network where the computation primitive — matmul — has no concept of frequency bands, phase, or resonance.

---

## Embedding Layer — Harmonic Structure Helps

Frozen harmonic embeddings outperform both random initialisation (by 2.8%) and trainable harmonic embeddings (by 0.4%). The geometric structure provided by `cos(n * theta)` is not merely a useful initialisation — it is a sufficient embedding substrate. The model does not need to learn its embeddings; it needs them to be structured.

Phase B adds a refinement: two-stage magnitude training (freeze magnitude during phase organisation, free after stabilisation) achieves 95.2% of MLP at 43.1% parameters. The coupling principle — phase builds structure, magnitude amplifies — expressed as a training schedule.

Evidence: Phase 17 (frozen vs learned), Phase A (full stack), Phase B (two-stage), cross-language Rust reproduction.

---

## Weight Layer — No Effect

Weight matrices remain spectrally flat regardless of embedding type, training data, or curriculum. All modes show identical spectral profiles: 88.3% of bands needed for 90% energy, 0% band sparsity. The optimiser (AdamW) determines weight spectral profile, not the input structure.

Tested twice: Phase 17 (three embedding modes) and Phase 17b (frequency curriculum pre-training). Both null. Weight spectral sparsity requires explicit optimiser intervention, not input structure.

---

## Attention Layer — Harmonic Structure Hurts

Four independent approaches tested, all degrade or match standard performance:

**Phase 18 — Constrained Q/K:** Harmonic-structured Q/K projections produce uniform attention entropy (4.56 across all heads and layers — the theoretical maximum). The model cannot discriminate between tokens. 5.2% worse than standard. The 2-dimensional bottleneck (emphasising only the cos/sin pair for each head's harmonic order) destroys the model's ability to form useful attention patterns.

**Phase 19 — Replace Q/K with embedding interference:** Identical result to Phase 18 (3.2503 vs 3.2511). Confirms Phase 18's constrained Q/K converged to producing the same uniform attention as having no Q/K at all. The ~3.25 ceiling is what a transformer achieves with uniform attention + learned V/MLP.

**Phase 19b — Additive harmonic bias:** Preserves full learned Q/K, adds interference as bias with learnable per-head lambda. PyTorch verification (Corrective Finding #7) showed lambda does learn: low-frequency heads amplify (lambda up to +0.54), high-frequency heads suppress (lambda down to -0.08). The model detects frequency-dependent structure but cannot exploit it for prediction. 0.4% worse.

The Q/K projection is where the model learns unrestricted token-to-token relationships. Constraining, replacing, or biasing it with harmonic structure impairs the model because harmonic embedding dot products encode token identity, not token relevance.

---

## FFN Layer — Kerr-ODE Partially Replaces MLP

The search for a wave-native FFN computation primitive progressed through three architectures:

**Phase 20/20b — LC Circuit Layer:** Per-band processing + linear cross-band coupling. 148 params/layer (Phase 20) and 13,440 params/layer (Phase 20b). Both underperform MLP by 21-23% regardless of capacity. The bottleneck is architectural: per-band nonlinear + cross-band linear cannot match dense MLP expressiveness. The missing primitive was nonlinear multi-band fusion.

**Phase 21 — Kerr-ODE Layer:** Nonlinear optics ODE with |Z|^2 cross-band coupling adapted from the Kerr effect in coupled optical resonators. 16,642 params/layer (7.9x reduction vs MLP). Cuts the gap from 21.3% to 7.7%. The Kerr nonlinearity provides intensity-dependent frequency coupling — the nonlinear multi-band fusion the LC layer lacked.

**Phase 21b — Per-band Kerr:** Giving each band its own alpha/beta bought only 0.54pp over scalar. Integration depth (8 vs 4 steps) bought 4x more. The scalar Kerr coefficient was already the right abstraction.

**Phase 22 — Inverse analysis:** Binary reversibility split: L0 is 100% reversible (spectral remixing), L1-L3 are 100% irreversible-nonlinear (genuine computation). Zero damping-irreversibility.

**Phase 22b — Analytical L0:** Reversible does not equal replaceable. Post-hoc substitution is catastrophic (+163%). Training from scratch with analytical L0 works (+0.68% cost, 25% ODE compute saving). L0 performs impedance matching — near-identity conditioning that downstream layers are calibrated to expect.

**Phase 22c — Wider clamps:** [-50,50] recovers ~30% of the MLP gap. Unclamped hurts due to Euler transient spikes reaching 178 million magnitude.

**Phase 22d — RK4 integration:** The 178M spikes were 100% Euler artifacts. Under RK4, peak magnitudes drop to 6.5. RK4 improves 1.71% over Euler. The remaining ~6.5% gap is the architectural ceiling.

**Phase A — Full stack integration:** All components assembled. 96.8% of MLP at 42.6% parameters. Beats the 93.5% component-level ceiling — the components synergise.

**Phase B — Two-stage magnitude training:** 95.2% of MLP at 43.1% parameters in a controlled 7-variant same-run comparison. Band-aware routing hurts by 7-9% — layers need full-spectrum access.

---

## Corrective Findings

Seven findings discovered through cross-language and cross-framework validation:

1. **Bucket resolution threshold floor** — must exceed `cos(2π / B)` to avoid neighbour leakage
2. **Nonlinear orb falloff** — cosine curve is concave, not linear
3. **Directed distance for asymmetric operations** — shortest-path destroys directionality
4. **Harmonic-scaled Nyquist floor** — threshold floor is `cos(n × 2π / B)`, not `cos(2π / B)`
5. **Overtone conflation** — signed mean coherence resolves fundamental from overtones
6. **Conjugate symmetry in resonance** — rfft middle coefficients need weight 2 (discovered in Rust)
7. **Candle autograd limitation** — frozen tensor products don't propagate gradients (discovered in PyTorch)

---

## The Remaining Gap

The 1.9% gap between the full Kerr-ODE + Maestro + curriculum stack (98.1%) and MLP baseline (100%) is architectural. It is not:
- Integration quality (RK4 confirmed, Phase 22d)
- Clamping (eliminated under RK4)
- Per-band expressiveness (scalar abstraction sufficient, Phase 21b)
- Band routing (hurts performance, Phase B)
- Coupling mechanism (dispersive null — three mechanisms tested, none help, Phase C)
- Learnable kernel weights (0.24pp improvement, negligible, Phase C)

It is the cost of |Z|^2 cross-band coupling versus dense matmul. The Kerr nonlinearity operates on nearest-neighbour bands; MLP operates on all dimensions simultaneously. The gap is the price of locality in frequency space.

**The gap narrows with scale — this was the open question, now answered.**

**Depth convergence (Phase C):** The locality gap closes with depth: 4.88% at 4L → 3.98% at 6L → 2.70% at 7L. Approximately 1pp per 1.5 additional layers. MLP also benefits from depth (fair comparison at equal layer count), but the Kerr gap closes proportionally faster. Extrapolation suggests <1% at 12-15 layers.

**Bandwidth scaling is non-monotonic (Phase C):** The locality penalty peaks at 48-96 bands (4.5-5.4%), then collapses to 0.35% at 128 bands. At 128 bands, MLP (3.17M params) overfits catastrophically without weight decay. Kerr (1.34M params) remains stable — the ODE structure acts as implicit regularisation. At high parameter-to-data ratios, the structural constraint transitions from penalty to advantage.

**Maestro bottleneck — the efficiency finding (Phase C):** A 16D squeeze-and-excitation pathway provides additive global coordination at 3.7% parameter cost. Closes 1.80pp at 4L (4.88% → 3.09%). Consistent across all depths (+0.4pp at 7L). Multiplicative fusion hurts; the bottleneck must correct, not replace, local computation.

**Coupling interventions that don't help (Phase C nulls):** 9-band + curriculum don't stack (both attack coupling reach from different angles, but overlap). Three dispersive coupling mechanisms — Laplacian, FFT global, per-band quadratic — all near-null. The bands are Fourier components of embedding vectors, not physical waves; dispersive terms from wave physics don't transfer.

**Maestro + curriculum stack (Phase C integrated test):** The two interventions attack different mechanisms — maestro provides global coordination per step, curriculum stages band introduction. Combined: 1.91% gap (down from 3.09% Maestro alone, 3.13% curriculum alone). Two-stage magnitude adds nothing on top (1.93%). The magnitude freedom is redundant when coordination + staging are both present.

**The efficiency framing:** The architecture does not beat MLP. It achieves equivalent capability at half the parameters. At 4L + Maestro + curriculum: 98.1% performance at 44% parameters, same forward pass depth. Fewer parameters = less memory, less bandwidth, less energy per forward pass. At scale, the ODE provides implicit regularisation that MLP requires explicit tuning to match.
