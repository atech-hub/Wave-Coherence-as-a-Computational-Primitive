[![DOI](https://zenodo.org/badge/1153530777.svg)](https://doi.org/10.5281/zenodo.18607190)

# Wave Coherence as a Computational Primitive

A validated mathematical framework that uses phase encoding on the unit circle and harmonic coherence as a universal relationship detection operator. A single function — `cos(n * (θ_a - θ_b))` — detects exact matches, harmonic families, oppositions, fuzzy proximity, and multi-type relationships, matching or exceeding the expressiveness of traditional WHERE and JOIN operations for relationship-dense queries.

## Key Results

**Database query primitive.** A single harmonic coherence scan (`cos(3 * Δθ)`) finds 75 related entities across 3 groups in one pass. The equivalent SQL requires 2-3 JOINs and an explicit relation table. The wave model discovers relationships from geometry; the relational model must enumerate them. Zero false positives across 25 validated tests. ([Full test results](experiments/RESULTS.md))

**LLM architecture substrate.** Harmonic embeddings serve as structural priors for transformers — frozen phase-encoded embeddings (zero gradient, zero training) match or outperform learned embeddings. A Kerr-ODE layer adapted from nonlinear optics replaces matrix multiplication in the FFN, achieving **98.1% of MLP performance at 44% of parameters** with a maestro bottleneck for global coordination and progressive curriculum (Phase C). The gap closes with depth (~1pp per 1.5 layers) and the ODE structure provides implicit regularisation — stable where MLP overfits. ([Architecture details](docs/KERR-ODE-MATHEMATICS.md))

**Important: novel architecture.** The Kerr-ODE is NOT a standard transformer. Trained models do not work with LM Studio, Ollama, llama.cpp, or Hugging Face Transformers natively — these clients have no code path for ODE integration or stencil coupling. However, the [Wave Engine](https://github.com/atech-hub/wave-engine) includes a built-in OpenAI-compatible API (`wave-engine --serve`) that any chat UI can connect to. One binary: train, generate, and serve. GGUF export and HF model class integration are documented in [ENGINE-PATTERNS.md](ENGINE-PATTERNS.md) (Pattern 68) as prior art.

**Cosine similarity blindness.** Standard cosine similarity returns 0.0000 between vectors with strong harmonic relationships. A per-channel harmonic sweep recovers coherence of 1.0000 at the correct harmonic. One similarity score hides independent channels of structured information. Confirmed on both synthetic data and production transformer embeddings (all-MiniLM-L6-v2). ([Test 21 and Test 24 details](experiments/RESULTS.md#phase-16-wave-packet-engine))

## Origin

This work emerged from an unconventional observation: multiple ancient traditions independently discovered the same geometric relationship patterns by dividing circles into segments and cataloging which angles produce meaningful relationships. When stripped of interpretive layers, what remains is a complete taxonomy of relationship types — symmetric, asymmetric, directed, structural, context-dependent, compound — unified on a single mathematical substrate.

## What Is and Is Not New

**Not new (established mathematics):**
- The equation `cos(n × (θ_a - θ_b))` — harmonic coherence, standard in Fourier analysis
- Phase encoding on the unit circle — standard in signal processing
- Fourier uniqueness theorem; cosine similarity as a comparison measure

**Potentially new (the application and synthesis):**
- Using harmonic coherence as a database query operator, replacing JOINs with frequency-parameterized scans
- The geometric relationship catalog — a comprehensive taxonomy derived from cross-civilizational analysis of circle-division systems, stripped of interpretive layers
- Harmonic fingerprinting with closed-form collision resolution: `n = ⌈arccos(t) / Δθ⌉`
- The proposal and partial demonstration that these primitives serve as a substrate for LLM embeddings and FFN computation
- Multi-grid harmonic composition — incommensurate grids extend harmonic resolution via lcm coverage
- The geometric comma theorem — 24° = 360°/lcm(3,5), a provable incompatibility between 3-fold and 5-fold circular symmetry

We make no claim of priority over established mathematics. The contribution is recognising that established tools solve a specific class of problems more elegantly than current methods, and compiling the relationship type catalog that defines what they can express.

## Research Note — Emergent Properties

Progressive curriculum training (Phases 6-15) produces qualitatively different internal organisation compared to standard training, including: internal representations that exceed output vocabulary capacity (22.3% trapped structure), structured attractor dynamics under free-running conditions, self-monitoring confidence signals on mid-band harmonics, and faster knowledge absorption with reduced catastrophic forgetting. These properties emerge from training methodology, not explicit design. Their implications scale with model size and should be considered carefully when applying curriculum-based harmonic training to larger architectures.

## Architecture Summary

The full integrated stack (Phases A/B/C):

| Component | Approach | Key Finding |
|-----------|----------|-------------|
| Embeddings | Frozen harmonic (no gradient) | Frozen outperforms learned by 2.8% |
| Layer 0 (FFN) | Analytical per-band linear transform | Impedance matching — near-identity conditioning |
| Layers 1-3 (FFN) | Kerr-ODE with RK4 + maestro bottleneck | Nonlinear multi-band fusion + global coordination |
| ODE backward | Full backprop through 16-step RK4 | Per-layer α/β/γ self-organisation — model designs its own depth hierarchy |
| Corrector plate | Per-band learned phase rotation after ODE | 336 params (0.1%), enables 3× higher nonlinearity, dual-channel encoding |
| Attention | Frozen harmonic coherence scoring | Zero-parameter attention from mathematical structure |
| AGC | Dynamic per-layer ceiling tracking learned coupling | Safety boundary adapts as model learns — prevents divergence at scale |
| Training | No curriculum needed with learnable ODE | γ damping serves as internal band activation schedule |

**Result:** Loss 3.07 at 256-dim (128 bands, 598K params) — new project record. Loss 2.70 at 168-dim char-level (333K params). The learnable ODE backward was the root cause fix: loss 3.76 in 15 minutes vs 3.91 in 83 minutes with frozen ODE. Per-layer coupling self-organises — L0 high self-coupling, L1-L3 cross-band specialists, deepest layer protects output. The corrector plate (Investigation 14) enabled the model to push 3× harder on nonlinearity by providing phase error correction. THD stable over 30K sustained training. At 128 bands, Kerr's structural constraint provides implicit regularisation — stable where MLP overfits.

For the complete experimental record across 34 phases and 15 investigations including null results, corrective findings, and established architectural boundaries, see [experiments/RESULTS.md](experiments/RESULTS.md) and [investigations/](investigations/).

**Training + inference engine:** The [Wave Engine](https://github.com/atech-hub/wave-engine) (Rust, Apache 2.0) is the unified training and inference binary. Train, analyze, generate, and serve — all modes use the same forward pass, eliminating training/inference mismatch. Three compute tiers: CPU (RK4-16 ODE with cached backward), wgpu GPU (fused ODE forward+backward via WGSL shaders, 1152 dispatches in one encoder), and Candle CUDA. Learnable ODE backward, corrector plate, dynamic per-layer AGC, batch distortion monitoring, BPE tokenization, WCHK self-describing checkpoints. KV-cache for fast autoregressive generation (4.7× speedup). OpenAI-compatible API via `--serve` flag (`--features serve`). Successor to the [Kerr Engine](https://github.com/atech-hub/kerr-engine) and [Wave Server](https://github.com/atech-hub/wave-server) (both now deprecated — use wave-engine for all work).

**Wave memory:** The [Kerr Memory](https://github.com/atech-hub/kerr-memory) library (Rust, Apache 2.0) provides persistent wave state that accumulates experience across conversations. The model weights stay frozen; a separate 1.5KB file shifts the Kerr-ODE’s starting position. The model reads and writes memory in its native harmonic coordinate system — no translation layer. ~920 lines, zero dependencies. Integrated into the server via `--memory` flag.

For the mathematical analysis of where harmonic structure helps and where it does not (the substrate incompatibility boundary), see [docs/ARCHITECTURE-BOUNDARIES.md](docs/ARCHITECTURE-BOUNDARIES.md).

## Established Boundaries

Null results are findings. The framework has clear boundaries established through systematic testing:

| Layer | Harmonic structure | Evidence |
|-------|-------------------|----------|
| Embeddings | **Helps** — frozen outperforms learned | Phases 1-16, 17, A, B |
| Retrieval | **Helps** — per-channel sweep beats cosine similarity | Tests 21, 24, Phase 16 |
| FFN computation | **Partially replaces** — Kerr-ODE at 98.1% of MLP | Phases 20-22d, A, B, C |
| Attention Q/K | **Hurts** — must remain unconstrained | Phases 18, 19, 19b |
| Weight matrices | **No effect** — spectrally flat regardless | Phases 17, 17b |
| Harmonic embedding vocab | **Multi-grid required at scale** — single-circle embeddings degenerate at high vocab/bands ratio; coprime dual-grid (Pattern 53) resolves any vocabulary at any dimension | [Wave Structure Investigation](investigations/wave-structure/INVESTIGATION.md) |

Full analysis: [docs/ARCHITECTURE-BOUNDARIES.md](docs/ARCHITECTURE-BOUNDARIES.md)

## Investigations

Deep-dives into specific questions, each self-contained with narrative, tests, and conclusions.

**[Multi-Grid Harmonic Investigation](investigations/multi-grid/INVESTIGATION.md)** — Six anomalies in ancient geometric catalogues tested. Three independent mathematical layers identified: flat harmonics on matched grids, non-uniform metric coherence, and structural rules. Includes the geometric comma theorem and the Sexagenary compression proof.

**[Spherical Coherence Investigation](investigations/spherical/INVESTIGATION.md)** — Discovery that embedding magnitude carries usable information discarded by cosine similarity. Phase carries semantics (20x-383x clustering). Magnitude amplifies phase but cannot replace it. The coupling principle: phase first, magnitude second.

**[Frequency-Depth Investigation](investigations/frequency-depth/INVESTIGATION.md)** — 13 findings across 8 experiments (5a-5h) tracking how harmonic bands evolve through transformer layers. Maestro eliminates spectral dispersion (H/L ratio 1.013). 43 universal bands carry structural scaffolding (~70% energy); 21 word-specific bands carry semantic content. Band roles actively reassign at every layer (consecutive-layer correlation ~0.14). Concrete words more stable than abstract. Word clustering follows semantic affinity, not human-imposed categories. Three null hypotheses killed honestly.

**[Corpus Ordering Investigation](investigations/corpus-ordering/INVESTIGATION.md)** — 6 findings across 54 training runs (3 tests, 5-seed robustness). Sequential diversity pre-training beats single-corpus at equal target exposure. Order matters — wrong order is worse than no pre-training. The mechanism is diversity, not complexity. Legal text is easiest to model. Three-stage beats two-stage. Diversity is more efficient, not more powerful. All results produced by the Kerr Engine (Rust, Apache 2.0), now succeeded by the [Wave Engine](https://github.com/atech-hub/wave-engine).

**[Wave Memory Investigation](investigations/wave-memory/INVESTIGATION.md) (in progress)** — persistent wave state that accumulates experience across conversations by modifying Kerr-ODE initial conditions. The model weights never change; a separate 512-byte memory file shifts where the ODE starts. 5 experiments complete: stochastic resonance confirmed (α=0.05 gives -8.8% perplexity improvement from random noise alone), accumulation converges over 20 conversations, topic separation null at character-level (captures corpus texture, not topic — bounded by model capacity), reset is bit-identical to baseline, anomaly detection catches spikes before affecting output. The mechanism works, is stable, is safe, and is inspectable. Semantic resolution depends on model capacity — word-level and BPE tokenization expected to enable topic separation. Architecture: [ENGINE-PATTERNS.md](ENGINE-PATTERNS.md) (Patterns 69-70).

**[Operating Regime Discovery](investigations/operating-regime/INVESTIGATION.md) (confirmed)** — Three compounding errors preventing the wave-engine from producing English: (1) coupling 10x too weak (α=0.01 vs 0.1), (2) block-diagonal out_proj starving capacity, (3) AGC ceiling too wide. The parameters form a linked system — coupling constant determines the AGC ceiling which determines whether the model can concentrate on the unit sphere. Fixed: α=0.1, dense out_proj, ceiling=1.0 produces English words immediately. The "compare against what worked" methodology: what kerr-engine learned in 128-dim, wave-engine needed to learn at 168-dim with the same physics.

**[Harmonic Scaling Investigation](investigations/harmonic-scaling/INVESTIGATION.md) (168-dim complete)** — The model that wouldn't stop learning. 21+ training cycles across two coupling regimes at 168-dim. β is an independent design parameter — β=0.2 doubles cross-band coupling, enables dual-channel encoding, prevents crystallisation. Rotational learning: the model alternates θ and Δθ encoding channels, each rotation tightening entropy. 3.38x per-band discrimination from 1.36 MB. Twelve corrections documented honestly. All at 340K params.

**[256-dim Scaling Investigation](investigations/256-dim-scaling/INVESTIGATION.md) (active)** — Can the model speak? Scaled from 168-dim's best checkpoint to 256-dim (128 bands, 8 heads, 597K params). Loss 3.78 at C5 — new all-time record, crushing 168-dim's 3.91 in half the iterations. Entropy at 0.433 (half of 168-dim's lifetime best). Δθ at 0.98x approaching dual-channel threshold smoothly instead of oscillating. Gradient balance flipped to 79% model / 21% head. Vocabulary producing grammar concept clusters: "sentence", "phrase", "means", "action", "being", "relinquish".

**[The Decoder Was Never Necessary](investigations/phase-native/INVESTIGATION.md) (complete)** — Phase-native training via spherical geometry. 49/55 (89.1%) arithmetic accuracy with zero decoder parameters. The lm_head is optional — the dot product against frozen embeddings matches it at 40K. Phase coherence failed (5/10) because magnitude normalisation kills the self-coupling gradient. The dot product preserves magnitude as confidence. Confirmed by the spherical Pythagorean theorem. Self-organised depth pipeline: L0 compresses (23.7% damping, β/α=1.4x), L3 routes (19.6% damping, β/α=14.3x). 10-monitor suite provides complete visibility. Defensive publication: [ENGINE-PATTERNS.md](ENGINE-PATTERNS.md) (Patterns 100-105).

**[Output Decoding Investigation](investigations/output-decoding/INVESTIGATION.md) (complete)** — The model knows more than it can say. Five output decoders tested at 168-dim: tied embeddings (null — rigid decoder), wave transduction V1 (85 params, phase-native, 5.84), wave transduction V2 (dual-channel, no improvement), unfrozen wave decode (86K params, 5.25 — broke frozen ceiling), low-rank 32 (38K params, 4.62 — best efficiency). Conservation argument validated: phase coherence preserves what dot products destroy. Low-rank wins on efficiency for 256-dim. Defensive publication: Patterns 85-86.

**[Wave Structure Emergence](investigations/wave-structure/INVESTIGATION.md) (superseded by Harmonic Scaling)** — Initial wave structure diagnostics. Phase clustering at 0.553, multi-grid coprime embeddings (11,800x improvement), per-band ODE clamping. Built into the engine as `--analyze` flag. Findings incorporated into the Harmonic Scaling investigation.

**[MLP Weight Structure Analysis](investigations/mlp-analysis/INVESTIGATION.md) (complete — null finding)** — SVD and DFT analysis of all 72 MLP weight matrices across 24 layers of Qwen 2.5 0.5B. Full effective rank 896/896 everywhere, flat frequency spectrum (33/33/33% low/mid/high), no near-identity layers. The "translate existing model to waves" path does not exist. Wave-engine models must be trained from scratch — efficiency gains come from learning a different representation, not compressing existing MLP weights. Defensive publication: [ENGINE-PATTERNS.md](ENGINE-PATTERNS.md) (Pattern 80).

**[ODE Magnitude Regulation](investigations/ode-regulation/INVESTIGATION.md) (complete)** — How should signals entering the Kerr-ODE be regulated during training? 5 controlled tests proving physics-bounded adaptive gain control eliminates V-shape training divergence. The maestro pre-conditioner naturally increases magnitudes — fixed clamps throttle the model. AGC with physics-derived ceiling (M < sqrt(pi/2 / (alpha+4*beta))) lets the model self-regulate: rolling average descends monotonically through 20K iterations for the first time at BPE scale. Electronics analogy: fixed resistor, zener diode, AGC + rail voltage. Defensive publication: [ENGINE-PATTERNS.md](ENGINE-PATTERNS.md) (Pattern 81).

**[The Handcuffed Model — ODE Backward](investigations/ode-backward/INVESTIGATION.md) (confirmed)** — Six symptoms, six partial fixes, one root cause: the identity backward line. The ODE backward was `d_precond = d_kerr_out` — the entire RK4 integration invisible to the optimizer. Fix: full backpropagation through 16-step RK4. Per-layer coupling self-organisation emerges from uniform starting conditions. Loss 3.76 in 15 minutes vs 3.91 in 83 minutes. The model was never lazy — it was handcuffed. Defensive publication: [ENGINE-PATTERNS.md](ENGINE-PATTERNS.md) (Patterns 88-89).

**[The Corrector Plate](investigations/corrector-plate/INVESTIGATION.md) (confirmed)** — RF distortion meets telescope optics. The Kerr nonlinearity IS 3rd-order distortion. THD measurable because bands are explicit frequencies. The model drove L3's α to the floor — it knew distortion was a problem but could only avoid it, not cancel it. The corrector plate (336 params, 0.1% of model) gave it per-band phase correction. L0 tripled its nonlinearity. Loss 2.70. THD flat over 30K sustained. Dual-channel encoding unlocked. Defensive publication: [ENGINE-PATTERNS.md](ENGINE-PATTERNS.md) (Patterns 90-91).

**[The Scaling Wall](investigations/scaling-wall/INVESTIGATION.md) (confirmed)** — Three divergences at 256-dim, three wrong diagnoses (AGC, curriculum, α clamp). Actual cause: block-diagonal out_proj created 29% body / 71% head parameter imbalance. Dense out_proj rebalanced to 56/44. Loss 3.07 — new 256-dim record. The β that ran away to 0.41 at groups=8 settled at 0.20 at groups=1. The model didn't need extreme coupling when it had enough body capacity. Defensive publication: [ENGINE-PATTERNS.md](ENGINE-PATTERNS.md) (Pattern 92).

## Documents

| File | Description |
|------|-------------|
| [docs/geometric-relationship-catalog.md](docs/geometric-relationship-catalog.md) | Complete catalog: 5 traditions, 26 division systems, 35+ relationship types |
| [docs/wave-mechanics-stripped-catalog.md](docs/wave-mechanics-stripped-catalog.md) | Pure mathematical specification — domain interpretation removed |
| [docs/MATHEMATICS.md](docs/MATHEMATICS.md) | Formal mathematical foundations in standard notation |
| [docs/KERR-ODE-MATHEMATICS.md](docs/KERR-ODE-MATHEMATICS.md) | Kerr-ODE mathematical foundations — ODE system, integration, reversibility |
| [docs/ARCHITECTURE-BOUNDARIES.md](docs/ARCHITECTURE-BOUNDARIES.md) | Where harmonic structure helps and where it does not |
| [experiments/RESULTS.md](experiments/RESULTS.md) | Complete experimental record: 34 phases, all results, all nulls |
| [ENGINE-PATTERNS.md](ENGINE-PATTERNS.md) | Defensive publication: 87 engine pattern families as prior art |
| src/ | Rust validation suite — 25 tests, zero dependencies |
| python/ | Python translation of full test suite |
| experiments/ | 34 training experiments with PyTorch |
| investigations/ | 15 deep-dive investigations: multi-grid, spherical, frequency-depth, corpus-ordering, wave-memory, wave-structure, MLP analysis, ODE regulation, operating-regime, harmonic-scaling, output-decoding, 256-dim-scaling, ODE backward, corrector plate, scaling wall |

## Reproduce the Validation

### Rust (primary)

```bash
git clone <this-repo>
cd wave-coherence-computational-primitive
cargo run
```

Requires only a Rust toolchain (edition 2024). No external dependencies.

### Python

```bash
cd python
python run_tests.py
```

Requires Python 3.10+. No external dependencies.

Both produce identical results: 25 tests, all passing. Tests 24-25 run separately (real embedding analysis and character-level transformer — see [experiments/RESULTS.md](experiments/RESULTS.md) for details).

### Rust Harmonic Transformer (Test 25 cross-language reproduction)

```bash
cd rust-transformer
cargo run --release
```

Confirms identical pattern in pure Rust/candle: harmonic outperforms by 1.8%, frozen matches baseline.

### Rust Math Experiments (cross-language validation)

```bash
cd experiments/rust-experiments
cargo run
```

14 tests validating Phases 4, 5, and 16 in pure Rust. Discovered Corrective Finding #6 (conjugate symmetry in resonance).

### Expected Output

```
=== Wave Mechanics Test Program ===

Test 1:  PASS  (Exact match, zero false positives)
Test 2:  PASS  (3rd harmonic detects 0°, 120°, 240°)
Test 3:  PASS  (2nd harmonic detects 0°, 180°)
Test 4:  PASS  (Fuzzy scores: 1.000 > 0.924 > 0.556 > 0.0)
Test 5:  PASS  (Multi-attribute AND via product)
Test 6:  PASS  (All 4 directed cycle traversals correct)
Test 7:  PASS  (Structural pairs independent of geometry)
Test 8:  PASS  (Wave scan = linear scan, 10/10 matches identical)
Test 9:  PASS  (Single scan found 75 entities across 3 groups)
Test 10: PASS  (Broad: 3 targets, Narrow: 1 target, same position)
Test 11: PASS  (Harmonic fingerprinting: predicted n matches actual at 2°, 1°, 0.1°)
Test 12: PASS  (Mutual amplification: ordering and ratios exact)
Test 13: PASS  (5-node cycle: 20/20 pairs, 4 types × 5, zero conflicts)
Test 14: PASS  (Harmonic orthogonality: zero cross-talk between n=3, 4, 5, 6)
Test 15: PASS  (Wraparound: symmetric scores at 0°/360° boundary)
Test 16: PASS  (Scale: 360 values, 0 false positives, harmonic-scaled Nyquist validated)
Test 17: PASS  (Density scaling: sparse clean, degradation at density, harmonic scales with separation)
Test 18: PASS  (Bucket index: all queries match full scan, ~13% selectivity at 1000 entities)
Test 19: PASS  (2D torus index: compound queries correct, multiplicative selectivity over 1D)
Test 20: PASS  (Dynamic mutation: remove/insert/update, all queries correct throughout)
Test 21: PASS  (Harmonic sweep: 5 planted relationships recovered, cosine similarity blind to all, 0 false positives)
Test 22: PASS  (Kernel admissibility: symmetry, normalization, positive semi-definiteness, spectral scaling all verified)
Test 23: PASS  (Fundamental harmonics: triadic→n=3, opposition→n=2, quadrant→n=4, noise→none)
Test 24: PASS  (Real embeddings: spectral variance 3x syn/ant, 7x syn/unrel, cosine blind spot confirmed)
Test 25: PASS  (Harmonic transformer: -2.2% vs baseline, frozen matches baseline, no tokens needed)

=== RESULTS: 25 passed, 0 failed out of 25 ===
ALL TESTS PASSED
```

## Potential Applications

**Database query engine** — Phase-encoded entities with coherence-based scanning. Self-indexing (the circle IS the index), sub-linear queries, zero maintenance. Compound relational queries as interference patterns rather than JOINs. Dynamic mutation without global rebuild. ([Tests 18-20](experiments/RESULTS.md))

**LLM architecture** — Harmonic embeddings as structural priors. Frozen embeddings that outperform learned ones. Kerr-ODE as wave-native FFN replacement. Progressive curriculum for structure-first training. Language-independent embedding generation (any language that computes trigonometric functions). ([Phases 17-B](experiments/RESULTS.md))

**Knowledge graph / RAG** — Typed retrieval surfacing not just "documents about X" but "documents about things that enable X" — relationship-typed retrieval that cosine similarity alone cannot express.

## Related Work

| Reference | Connection |
|-----------|------------|
| Listopad (2025a) — [ResonanceDB](https://arxiv.org/abs/2509.09691) | Phase-aware retrieval validates resonance-based scoring over flat embeddings |
| Listopad (2025b) — [Phase-Coded Memory](https://arxiv.org/abs/2511.11848) | Extends resonance retrieval into dynamic inference loops |
| Sun et al. (2019) — [RotatE](https://arxiv.org/abs/1902.10197) | Relations as rotations in complex space for knowledge graphs |
| Moriya (2025) — [SECT](https://arxiv.org/abs/2505.17754) | Coherence kernel admissibility conditions validated in our Test 22 |
| Wang (2025) — Defierithos | Self-resonance field replacing transformer attention with wave interference |
| Pal et al. (2024) — [Kerr nonlinearity](https://arxiv.org/abs/2404.05646) | Coupled Lugiato-Lefever equation adapted for Phase 21's Kerr-ODE layer |
| Kato et al. (2024) — [Neural ODE fusion](https://arxiv.org/abs/2407.12937) | ODE integration as computation primitive for multi-band signal processing |
| Zelenka et al. (2024) — [Resonance detection](https://arxiv.org/abs/2412.19683) | Recurrence quantifiers carry resonant imprints regardless of dimensionality |
| Luo et al. (2025) — [DyMixOp](https://arxiv.org/abs/2508.13490) | Local-Global Mixing transformation inspired maestro bottleneck (Phase C) |

Full citation details: [docs/REFERENCES.md](docs/REFERENCES.md)

## Attribution

This work is a collaboration between Marco Da Cunha (conceptual framework, key theoretical insights, architectural direction) and Claude (Anthropic's AI assistant — mathematical formalization, documentation, test design, and code generation).

The core insights — that ancient geometric relationship catalogs encode a complete taxonomy of structural relationships, that harmonics are infinite and geometric invariants persist across all frequencies, and that these primitives map onto LLM embedding and FFN layers — originated from Marco Da Cunha's observations and questions during extended collaborative sessions.

## License

- **Code:** MIT License
- **Documents:** Creative Commons Attribution 4.0 International (CC BY 4.0)

This work is published as prior art to ensure it remains freely available and unpatentable. Use it, extend it, build on it, commercialize implementations of it. The ideas belong to everyone.

## Why Open

Publishing establishes prior art. Prior art prevents patents. What belongs to everyone cannot be taken by anyone.

---

*"The patterns are mathematical facts, not cultural inventions. Every civilization that studied circles found the same ones."*
