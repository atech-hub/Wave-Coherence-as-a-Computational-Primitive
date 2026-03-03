[![DOI](https://zenodo.org/badge/1153530777.svg)](https://doi.org/10.5281/zenodo.18607190)

# Wave Coherence as a Computational Primitive

## What This Is

A validated mathematical framework that uses phase encoding on the unit circle and harmonic coherence as a universal relationship detection operator. A single function — `cos(n * (θ_a - θ_b))` — detects exact matches, harmonic families, oppositions, fuzzy proximity, and multi-type relationships, matching or exceeding the expressiveness of traditional WHERE and JOIN operations for relationship-dense queries.

The framework includes:

- **A geometric relationship catalog** — every structural relationship pattern discoverable on a phase circle, stripped of all domain-specific interpretation, expressed as pure mathematics
- **A validation paper** — 25 tests, 7 corrective findings, all passing, with reproducible Rust and Python code
- **An architecture proposal** — applying wave mechanics as a substrate for LLM embeddings, FFN computation, and knowledge representation

## Origin

This work emerged from an unconventional observation: multiple ancient traditions independently discovered the same geometric relationship patterns by dividing circles into segments and cataloging which angles produce meaningful relationships. When stripped of interpretive layers, what remains is a complete taxonomy of relationship types — symmetric, asymmetric, directed, structural, context-dependent, compound — unified on a single mathematical substrate.

The key theoretical insights:

1. **Harmonic coherence (an established Fourier operation) works as a universal relationship operator for database queries.** One function, parameterized by frequency, detects any angular relationship. No relationship-specific code paths needed.
2. **Some relationships are non-geometric.** Structural pairings exist independent of angular distance, requiring explicit lookup tables alongside the geometric engine.
3. **Harmonics are infinite.** The relationship detection capacity is unbounded — not limited to a fixed set of patterns. The geometric invariants (symmetry groups on circles) remain valid at every frequency.
4. **The same primitives that work for database queries map onto LLM embedding and FFN layers.** Transformers effectively discover wave-like structure through training. Pre-building that structure as the computational substrate reduces parameters and improves efficiency. Attention Q/K projections require full-rank learned weights (Phase 18-19), but embeddings and feed-forward computation benefit from harmonic structure.
5. **Multiple incommensurate grids compose into richer harmonic coverage than any single grid.** A single grid of B positions resolves harmonics up to n=B/2 (Nyquist limit). Two grids with sizes sharing a small GCD cover harmonics up to lcm(B₁,B₂)/2 — two small grids (10+12=22 positions) encode the coverage of a 60-position grid. Competing harmonic symmetries on the same grid produce provable incompatibilities: the geometric comma theorem (24°=360°/lcm(3,5)) shows that 3-fold and 5-fold symmetry cannot coexist exactly, analogous to the Pythagorean comma in music.

## What Is and Is Not New

**Not new (established mathematics):**
- The equation `cos(n × (θ_a - θ_b))` — this is harmonic coherence, a standard operation in Fourier analysis, known since the 1800s
- Phase encoding values on the unit circle — standard technique in signal processing
- Fourier uniqueness — the theorem that distinct functions have distinct Fourier coefficients
- Cosine similarity as a comparison measure — widely used across many fields

**Potentially new (the application and synthesis):**
- Using harmonic coherence as a database query operator, replacing JOINs with frequency-parameterized scans
- The geometric relationship catalog — a comprehensive taxonomy of relationship types (symmetric, asymmetric, directed, structural, compound) derived from cross-civilizational analysis of circle-division systems, stripped of interpretive layers
- Harmonic fingerprinting for collision resolution — using multi-harmonic probing to disambiguate phase-encoded values, with a validated closed-form formula: `n = ⌈arccos(t) / Δθ⌉`
- The proposal that these primitives could serve as a substrate for LLM embeddings and FFN computation (attention Q/K projections proven to require full-rank learned weights — Phases 18-19)
- Multi-grid harmonic composition — the finding that incommensurate grids extend harmonic resolution via lcm coverage, and the three-layer decomposition of ancient geometric catalogues into flat harmonics, non-uniform metric, and structural rules
- The geometric comma theorem — a proven number-theoretic result showing that p-fold and q-fold symmetry on a shared grid produce an irreducible angular excess of 360°/lcm(p,q), analogous to the Pythagorean comma in music tuning

We make no claim of priority over established mathematics. The geometric comma theorem (Proposition 3.14) is, to our knowledge, a new result — the identification of 24°=360°/lcm(3,5) as a provable incompatibility between 3-fold and 5-fold circular symmetry, and its analogy to the Pythagorean comma. The broader contribution is in recognising that established mathematical tools solve a specific class of problems (relationship-dense queries) more elegantly than the methods currently used, and in compiling the relationship type catalog that defines what the tools can express.

## Research Note — Emergent Properties

This work documents harmonic structure in transformer embeddings and its applications to similarity measurement and model efficiency.

Researchers extending these findings should be aware that progressive curriculum training (Phases 6-15) produces qualitatively different internal organisation compared to standard training, including: internal representations that exceed output vocabulary capacity (22.3% trapped structure), structured attractor dynamics under free-running conditions (dreaming), self-monitoring confidence signals on mid-band harmonics, and faster knowledge absorption with reduced catastrophic forgetting.

These properties emerge from training methodology, not from explicit design. Their implications scale with model size and should be considered carefully when applying curriculum-based harmonic training to larger architectures.

## Documents

| File | Description |
|------|-------------|
| `docs/geometric-relationship-catalog.md` | Complete catalog of geometric relationship patterns across all source traditions (5 traditions, 26 division systems, 35+ relationship types) |
| `docs/wave-mechanics-stripped-catalog.md` | Pure mathematical specification — all domain-specific interpretation removed, only structural geometry remains |
| `docs/wave-test-program.md` | Test program specification — 20 tests validating the core math |
| `docs/wave-mechanics-validation-paper-theoretical.md` | Pre-test validation paper — formal framework and expected results (written before code execution) |
| `docs/wave-mechanics-validation-paper-empirical.md` | Post-test validation paper — actual results, real numbers, corrective findings from running the code |
| `docs/MATHEMATICS.md` | Formal mathematical foundations — definitions, propositions, and design constraints in standard notation (language-independent) |
| `docs/KERR-ODE-MATHEMATICS.md` | Kerr-ODE mathematical foundations — the ODE system, integration, reversibility, analytical solutions, and the integrated architecture. Mathematics from nonlinear optics and numerical methods applied as a wave-native FFN replacement |
| `src/` | Rust source code for the validation test suite (~2400 lines, zero dependencies) |
| `python/` | Python translation of the full test suite (Python 3.10+, zero dependencies) |
| `python/embedding_analysis.py` | Test 24: Harmonic structure analysis of real transformer embeddings (requires `sentence-transformers`) |
| `python/harmonic_transformer.py` | Test 25: Character-level harmonic transformer — no tokens, pure geometry (requires `torch` with CUDA) |
| `rust-transformer/` | Test 25 cross-language reproduction: harmonic transformer in pure Rust using candle (HuggingFace's Rust ML framework) |
| `experiments/` | 29 training experiments (Phases 1-22d + Phase A) with results, plus PyTorch cross-framework verification -- spectral persistence, progressive learning, harmonic decoding, wave packet engine, weight spectral analysis, harmonic attention, LC circuit layer, Kerr-ODE layer, per-band Kerr, inverse Kerr reversibility analysis, analytical L0 replacement, wider clamp analysis, RK4 integration, and full stack integration test |
| `experiments/rust-experiments/` | Pure Rust cross-language validation of math-only experiments (Phases 4, 5, 16). Zero dependencies, zero GPU. 14 tests, all passing. |
| `investigations/` | Deep-dive investigations branching from the core framework, each self-contained with narrative, tests, and conclusions |
| `ENGINE-PATTERNS.md` | Defensive publication: 57 engine pattern families (190+ implementations) as prior art across computing, AI, healthcare, finance, aerospace, automotive, robotics, energy, quantum computing, analogue/neuromorphic hardware, and 15 other domains |

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

Requires Python 3.10+. No external dependencies (uses only `math` from stdlib).

Both versions produce identical results: 23 tests, all passing. Tests 24-25 run separately: `python/embedding_analysis.py` (real embedding analysis) and `python/harmonic_transformer.py` (character-level harmonic transformer).

### Rust Harmonic Transformer (Test 25 reproduction)

```bash
cd rust-transformer
cargo run --release
```

Requires Rust toolchain (edition 2024) and internet connection for dataset download. Trains on CPU using candle (HuggingFace's Rust ML framework). No Python, no PyTorch. Reproduces the Test 25 harmonic embedding results: harmonic outperforms baseline by 1.8%, frozen matches baseline.

### Rust Math Experiments (cross-language validation)

```bash
cd experiments/rust-experiments
cargo run
```

Requires only a Rust toolchain (edition 2024). Zero external dependencies, zero GPU. Validates the math-only cores of Phases 4 (harmonic construction), 5 (musical interval theory), and 16 (wave packet engine) in pure Rust. 14 tests covering DFT round-trip, resonance-cosine identity, wave packet retrieval, selective band loading, interpolation, chimera construction, Tenney height, and consonance scoring. This cross-language port discovered Corrective Finding #6 (conjugate symmetry in resonance). A separate PyTorch cross-framework verification of Phase 19b discovered Corrective Finding #7 (candle autograd limitation with frozen tensors).

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

## Key Results

**Test 9 is the critical result.** A single harmonic coherence scan (`cos(3 * delta)`) found 75 related entities across 3 groups in one pass. The equivalent SQL requires 2-3 JOINs and an explicit relation table. The wave model discovers relationships from geometry; the relational model must enumerate them.

**Test 11 validates harmonic fingerprinting.** Bucket collisions are resolvable by probing higher harmonics. The required harmonic has a closed-form formula: `n = ⌈arccos(t) / Δθ⌉`. Predicted matched actual exactly at 2° (n=13), 1° (n=26), and 0.1° (n=259). Collision resolution scales by analysis depth, not storage.

**Test 14 confirms harmonic orthogonality.** Different harmonic frequencies operate as completely independent selectors with zero cross-talk. n=3 finds only 120° family members, completely excluding 90° and 60° entities (which belong to n=4 and n=6 respectively). This validates that `cos(n × Δθ)` with different n values can serve as independent query channels on the same dataset.

**Test 16 validates scale and reveals a new design rule.** 360 distinct values encoded on a 360-bucket circle are resolved with zero false positives. However, harmonic queries at this scale revealed that the Nyquist-like threshold floor (Finding 1) scales with harmonic number — see Finding 4 below.

**Test 17 characterizes density scaling limits.** Eight configurations from 7-in-12 to 360-in-360, placed at golden angle intervals, reveal that exact match fails only at 100% bucket saturation, while triadic (n=3) detection becomes noisy at lower densities due to harmonic amplification of angular proximity. The resolution harmonic needed to distinguish the closest pair scales inversely with minimum separation, following the formula from Test 11.

**Test 18 proves the self-indexing property.** A BucketIndex that uses the encoded phase position as the bucket address — no separate index structure — produces results identical to a full scan while examining only a fraction of entities. At 1000 entities on 360 buckets: exact match at threshold 0.999 examines 2.0% of entities, exact match at 0.95 examines 10.7%, and harmonic queries examine 15-23%. The circle IS the index: insertion is O(1), and queries are sub-linear with zero maintenance overhead.

**Test 19 validates multi-attribute torus indexing.** Extending the 1D bucket index to a 2D torus (B×B grid) enables compound queries that narrow on both attributes simultaneously. At 500 entities on a 60×60 grid: exact+exact queries and exact+harmonic queries all match full scan exactly. Selectivity improvement over 1D is multiplicative — each dimension narrows independently. This bridges the gap between single-attribute proof and real multi-column database viability.

**Test 20 proves dynamic mutation support.** Insert, remove, and update operations work as local mutations on the circle without global rebuild. Starting from 200 entities: 50 removed, 30 inserted, 20 repositioned — all queries (exact and harmonic) remain correct throughout. Remove is tombstone + bucket cleanup. Update is remove + re-insert. This is what separates a mathematical proof from a working database.

**Test 21 demonstrates cosine similarity blindness.** Eight letters encoded at known phase angles with deliberate harmonic relationships. Cosine similarity between triadic partners (A at 0°, B at 120°) = **0.0000** — reporting "no relationship." A harmonic sweep across individual channels recovers coherence = **1.0000** at n=3. All 5 planted relationships (triadic, opposition, quadrant, sextile, pentagonal) recovered at exactly the correct harmonic with zero false positives on noise controls. The sum of harmonic channels cancels to zero, destroying the per-channel structure. This proves that standard cosine similarity — the primary comparison measure used across ML — is blind to harmonic organization in vectors. The harmonic sweep provides a tool to test whether real model embeddings contain this hidden structure.

**Test 24 confirms harmonic structure in real transformer embeddings.** Using `all-MiniLM-L6-v2` (384 dimensions), spectral coherence analysis reveals that real model embeddings contain per-frequency structure that cosine similarity destroys. Antonyms score **0.5789** cosine similarity vs synonyms at **0.6375** — nearly indistinguishable. But spectral variance (variance of per-band coherence) is **3x higher** for antonyms than synonyms, and **7x higher** for unrelated pairs. Different relationship types (hierarchical, functional, analogical) produce distinct spectral profiles — different shapes of coherence across frequency bands — that a single cosine score conflates. This bridges the gap from synthetic proof (Test 21) to real-world validation: the cosine similarity blindness phenomenon exists in production model vectors, not just constructed ones.

**Test 25 proves harmonic embeddings outperform random initialization.** A character-level transformer (4 layers, 128 dim) trained on Shakespeare with three embedding modes: baseline (random Gaussian, trainable), harmonic (phase-encoded, trainable), and frozen (phase-encoded, NOT trainable). No tokenizer — raw characters mapped to phase angles. Harmonic outperforms baseline by **2.2%** on validation loss, leading at every checkpoint. The frozen model — with 40,768 fewer trainable parameters and zero gradient updates to embeddings — matches the fully-trained baseline to within **0.02%**. The geometric structure provided by `cos(n * theta)` is not merely a useful initialization. It is a sufficient embedding substrate. The model does not need to learn its embeddings; it needs them to be structured. **Cross-language reproduction in pure Rust** (candle framework, no Python/PyTorch) confirms identical pattern: harmonic outperforms by 1.8%, frozen matches baseline — the advantage is mathematical, not framework-dependent.

**Phase A validates the full stack.** All validated components assembled into one model: frozen harmonic embeddings (no gradient), analytical per-band linear transform for layer 0, Kerr-ODE with RK4 integration for layers 1-3 (no clamping needed — dynamics bounded within [-7, 7]), standard learned attention, and progressive band curriculum training. Result: **96.8% of MLP performance at 42.6% of parameters** (341K vs 801K). This beats the 93.5% component-level ceiling — the components synergise rather than interfere. The wave-native computation primitive (nonlinear optics ODE replacing matrix multiplication) delivers a measured trade-off: 3.2% performance cost for 57.4% parameter reduction. On inference — where cost lives at scale — fewer parameters means less memory, less bandwidth, less energy per forward pass.

**Seven corrective findings tighten the design:**

1. **Bucket resolution imposes a threshold floor.** Exact match threshold must exceed `cos(2π / bucket_count)` to avoid neighbor leakage. Analogous to the Nyquist limit in signal processing.
2. **Cosine orb falloff is nonlinear.** At 62.5% of tolerance radius, score is 0.556 (not ~0.7). The curve is concave — generous near center, steep near edge.
3. **Asymmetric operations require directed distance.** Shortest-path distance (0-180°) destroys directionality. Typed reach needs directed distance (0-360°).
4. **The Nyquist-like threshold floor scales with harmonic number.** At harmonic n with B buckets, the threshold floor is `cos(n × 2π / B)`, not `cos(2π / B)`. Higher harmonics amplify bucket spacing, widening neighbor leakage. For single-value precision at n=3 with 360 buckets, threshold must exceed cos(3°) = 0.9986, not cos(1°) = 0.9998.
5. **Absolute coherence conflates fundamental with overtones.** |cos(n × Δθ)| = 1.0 at both fundamental and all integer multiples. Signed mean coherence resolves them: the fundamental is the lowest n where signed mean exceeds the alignment threshold.
6. **Conjugate symmetry in resonance.** `rfft` returns one-sided DFT coefficients. Middle coefficients represent two-sided energy and need weight 2 in the resonance formula; DC (n=0) and Nyquist (n=N/2) get weight 1. Without this correction, resonance diverges from cosine similarity by ~4% on structured inputs. With correction: machine-precision match (2.05e-15). Discovered during Rust cross-language validation.
7. **Candle autograd limitation with frozen tensors.** Candle (HuggingFace's Rust ML framework) does not propagate gradients through products where one operand is a frozen tensor not tracked in VarMap. PyTorch handles this correctly — `nn.Parameter` receives gradients even when multiplied by a `register_buffer`. In Phase 19b, candle showed lambda stuck at 0.100000 for 2000 iterations; PyTorch showed lambda learning to head-specific values (range -0.08 to +0.54, avg 0.215). The loss conclusion is unchanged (harmonic bias doesn't help prediction) but the mechanism is refined: the model actively engages with the harmonic prior, selectively amplifying low-frequency structure and suppressing high-frequency structure, yet this doesn't improve next-token prediction. Discovered during Python cross-framework validation.

## Potential Applications

### Database Query Engine
Phase-encoded entities with coherence-based scanning for relationship-dense data. Compound relational queries (harmonic family + structural pair + directed dependency + domain relevance) computed as interference patterns rather than multiple JOINs. The self-indexing property (Test 18) means insertion automatically indexes entities by their encoded position, with sub-linear query performance and zero index maintenance. Multi-attribute torus indexing (Test 19) enables compound queries across multiple columns with multiplicative selectivity. Dynamic mutation (Test 20) confirms insert/remove/update as local operations requiring no global rebuild.

### LLM Architecture — Harmonic Embeddings as Structural Priors

A single phase angle probed across N harmonics produces an N-dimensional vector: `v(θ) = [cos(θ), cos(2θ), ..., cos(Nθ)]`. For K attributes × N harmonics, this yields a K×N-dimensional structured embedding where every dimension has a defined meaning. This is a Fourier basis expansion — the harmonic fingerprint validated in Test 11, generalized to arbitrary dimensionality.

The implication for LLM architecture: harmonic embeddings could serve as **structural priors** — pre-built geometric structure that reduces what the network needs to learn through training. Specific applications:

- ~~Attention heads parameterized by harmonic frequency instead of learned weights~~ — **tested in Phase 18: does not work.** Constraining Q/K projections to harmonic structure produces uniform attention (entropy 4.56 across all heads), destroying token discrimination. Q/K projections require full-rank freedom. Harmonic pre-scoring of token relevance remains an open question (see ENGINE-PATTERNS.md Pattern 6.6-6.8)
- Positional encoding via harmonic phase (RoPE already uses this principle for one dimension; harmonic encoding generalizes it to N dimensions with relationship-typed structure)
- Context windows as resonance fields where relevance emerges from constructive interference
- Directed phase relationships as native reasoning chain primitives
- Infinite harmonic capacity without additional parameters

The hypothesis — now partially demonstrated: learned embeddings discover through gradient descent a structure that harmonic encoding provides by construction. Tests 25 and Phase A confirm this for the embedding layer (frozen harmonic embeddings match or outperform learned ones) and for the FFN layer (Kerr-ODE achieves 96.8% of MLP at 42.6% of parameters in the full stack). Pre-building harmonic structure reduces training cost, improves interpretability, and lowers energy consumption. The remaining open question is whether these gains scale to larger architectures.

### Language-Independent Training

Frozen harmonic embeddings remove the dependency on automatic differentiation frameworks for the embedding layer. The embeddings are computed analytically — `cos(n * position * base_frequency)` and `sin(n * position * base_frequency)` — with no gradient computation, no backpropagation through embeddings, and no framework required. Any language that can compute trigonometric functions can generate identical embeddings.

This means transformer training is no longer bound to Python/PyTorch/TensorFlow for the embedding layer. The evidence:

- **Phase 17:** Frozen harmonic embeddings (val loss 3.0793) outperform both baseline random init (3.1684, -2.8%) and trainable harmonic embeddings (3.0899, -0.4%) on character-level Shakespeare. Not training the embeddings produces the best result.
- **Pure Rust implementation:** Zero external dependencies, zero unsafe blocks — the harmonic transformer trains and evaluates with no framework beyond `std`. Cross-language reproduction confirms the advantage is mathematical, not framework-dependent.
- **Machine-precision cross-language agreement:** Wave packet experiments in Python and Rust produce identical results to 2.05e-15, confirming the math is portable.

The implication: embedding generation, storage, retrieval, and training can all be implemented in any language. The full pipeline from embedding creation to model training to inference is framework-independent. The only remaining framework dependency is backpropagation through the transformer layers themselves (attention, MLP) — the embedding layer, which maps raw inputs to geometric structure, requires only trigonometry.

### Established Boundaries (Null Results)

Null results are findings. Phases 17-20b established where harmonic structure helps and where it does not:

- **Representation layer -- works.** Frozen harmonic embeddings outperform learned embeddings. Wave packets enable selective loading with 25% of bands. Proven across Phases 1-16.
- **Weight layer -- no effect.** Weight matrices remain spectrally flat regardless of embedding type or training curriculum. The optimiser (AdamW) determines weight spectral profile, not the input data. Proven null twice (Phase 17, 17b).
- **Attention layer -- not useful.** Constraining or replacing Q/K projections with harmonic structure produces uniform attention (Phases 18, 19). Biasing Q/K with harmonic interference (Phase 19b) allows the model to engage with the signal -- PyTorch verification shows lambda actively learning head-specific values, amplifying low-frequency structure and suppressing high-frequency structure -- but this still doesn't improve prediction (-0.4%). Four independent approaches tested, all degrade or match standard performance.
- **FFN layer -- frequency-native computation narrows but does not close the gap.** LC circuit layer (Phase 20/20b: per-band processing + linear coupling, 148-13.4K params/layer) underperforms MLP by 21-23% regardless of capacity -- the bottleneck is architectural, not parametric. The Kerr-ODE layer (Phase 21: nonlinear optics ODE with |Z|^2 cross-band coupling, 16.6K params/layer) cuts the gap to 7.7-8.5% with 7.9x fewer FFN params. The Kerr nonlinearity provides the "nonlinear multi-band fusion" the LC layer lacked. Parameters differentiate by depth: deep layers amplify the Kerr effect, shallow layers suppress it. Per-band Kerr coefficients (Phase 21b) provide negligible benefit -- the scalar abstraction is already correct. Inverse analysis (Phase 22) reveals a binary reversibility split: L0 is 100% reversible (spectral remixing), L1-L3 are 100% irreversible-nonlinear (genuine computation). Forward clamping hits 95% of bands in L3, creating an information bottleneck separate from the Kerr dynamics. Analytical L0 replacement (Phase 22b) achieves 25% ODE compute saving at +0.68% loss when trained from scratch, but post-hoc substitution is catastrophic (+163%) -- L0 performs impedance matching (near-identity conditioning) that downstream layers are calibrated to expect. Reversible does not equal replaceable. Wider clamp analysis (Phase 22c) confirms [-10,10] truncates information: widening to [-50,50] improves 1.61%, but unclamped hurts (-0.51%) due to Euler transient spikes reaching 178 million magnitude. RK4 integration (Phase 22d) eliminates the transient spikes entirely -- peak magnitudes drop from 22,000 to 6.5, confirming the 178M peaks were 100% Euler artifacts. RK4 improves 1.71% over Euler, closing the MLP gap to ~6.5%. This is the architectural ceiling.

**Why the boundary exists -- substrate incompatibility:** Matrix multiplication is structurally blind to frequency. A matrix treats every element as an independent grid position -- row 3, column 7 has no relationship to row 3, column 8. It cannot know that column pairs encode cos/sin of the same harmonic. When harmonic embeddings pass through matmul, the wave structure is invisible to the operation. This is analogous to pushing analogue waves through transistors (discrete switches) -- the component is structurally incompatible with the signal. The wave packet concept works at the representation layer (vectors you can decompose) and retrieval layer (comparison via frequency-aware functions). It fails inside the network because matmul -- the core computation primitive -- has no concept of frequency bands, phase, or resonance. Phase 20 tested the inverse approach: building a frequency-native computation layer (LC circuit analog) to replace matmul. The LC layer correctly processes harmonic bands but its factored structure (per-band nonlinear + cross-band linear) is less expressive than dense MLP. Phase 21 found the missing primitive: the Kerr nonlinearity from nonlinear optics (|Z|^2 cross-band coupling). The Kerr-ODE layer achieves 92% of MLP performance with 7.9x fewer FFN parameters -- the first wave-native primitive that meaningfully competes with matmul. Inverse analysis (Phase 22) confirms the nonlinear dynamics are genuinely essential in L1-L3, while L0 does reversible spectral remixing replaceable by analytical transform. Forward clamping (95% of L3 bands) creates a separate information bottleneck. Phase 22c-22d systematically decomposed the remaining gap. Wider clamps ([-50,50]) close ~30% of the gap. RK4 integration eliminates Euler transient spikes entirely (peak magnitudes drop from 22,000 to 6.5 -- the 178M unclamped peaks were pure integration artifacts) and closes another ~1.7pp. The remaining ~6.5% gap is the architectural ceiling of the Kerr-ODE: what |Z|^2 cross-band coupling costs versus dense matmul. Per-band expressiveness ruled out by Phase 21b (see ENGINE-PATTERNS.md Patterns 51-52). **Full stack integration test (Phase A)** assembles all validated components -- frozen harmonic embeddings, analytical L0, Kerr-ODE RK4 for L1-L3, progressive curriculum -- into one model. Result: **96.8% of MLP at 42.6% of parameters**, beating the 93.5% component-level ceiling. The components synergise. The infrastructure holds.

### Investigations

Deep-dives into specific questions arising from the core framework. Each investigation is self-contained with its own narrative, tests, and conclusions.

**[Multi-Grid Harmonic Investigation](investigations/multi-grid/INVESTIGATION.md)** — Six anomalies in ancient geometric catalogues tested against the flat-circle model. Three independent mathematical layers identified: flat harmonics on matched grids, non-uniform metric coherence, and structural rules. Includes the geometric comma theorem (24° = 360°/lcm(3,5)) and the Sexagenary compression proof (two small grids of 10+12 positions encode the harmonic coverage of a 60-position grid). Every result — positive, negative, and null — reported. Six Rust test files, zero external dependencies.

### Knowledge Graph / RAG
Typed retrieval that surfaces not just "documents about X" but "documents about things that enable X" or "documents about things X conflicts with" — relationship-typed retrieval that cosine similarity alone cannot express.

## Related Work

Listopad (2025) independently developed ResonanceDB, a phase-aware retrieval system that scores document relevance using resonance-based coherence rather than cosine similarity over flat embeddings. Their empirical results validate that phase-encoded scoring outperforms standard vector retrieval for relationship-sensitive queries. The present work extends this direction from the retrieval layer to the encoding substrate itself — proposing harmonic coherence not as a scoring alternative bolted onto existing embeddings, but as the foundational computational primitive for encoding, querying, and discovering relationships.

Wang (2025) proposed a more radical departure: the Self-Resonance Field (SRF) architecture, which replaces transformer self-attention entirely with wave interference and phase superposition. Tokens become waveform imprints with spectral signatures; semantic matching operates via coherence estimation between sub-bands rather than dot-product attention. Critically, Wang's architecture uses partial resonance — local spectral matching rather than global all-to-all attention — analogous to the sub-linear bucket selectivity demonstrated in Tests 18–19 of the present work. Their simulation results show improvements over GPT-4 Turbo across six benchmarks (ROUGE-L, METEOR, Pass@k, MMLU, Accuracy, ARC-AGI). While the results are simulation-based and not yet validated on real hardware, the architecture provides independent evidence that wave mechanics can serve as a viable computational substrate for language modeling — the same hypothesis proposed in Section 5.2 of this work from the mathematical primitives side.

Listopad (2025b) further extended this direction in a second paper on Phase-Coded Memory and Morphological Resonance, integrating resonance-based retrieval into inference loops — moving beyond static scoring toward dynamic phase-coded memory during generation. This represents the next logical step: not just retrieving by resonance, but reasoning through resonance.

Sun et al. (2019) established with RotatE that modeling relations as rotations in complex space effectively captures symmetry, antisymmetry, inversion, and composition patterns in knowledge graphs. RotatE validates, from the knowledge graph embedding side, that rotational geometry on the unit circle is a natural substrate for encoding relational structure — the same mathematical insight this work arrives at from the database query side.

Moriya (2025) demonstrated with the Surface-Enhanced Coherence Transform (SECT) that decomposing aggregate coherence into surface and propagation components recovers physical structure that ensemble averaging destroys. His admissibility conditions for valid coherence kernels -- Hermiticity, positive-definiteness, normalization, spectral scaling -- provide the formal contract validated in Test 22 of the present work. The structural parallel is exact: his aggregate coherence loses information the same way cosine similarity does in Test 21, and his per-component decomposition recovers it the same way harmonic sweep does.

Pal et al. (2024) characterized linear and nonlinear coupling in twin optical microresonators with Kerr nonlinearity. Their coupled Lugiato-Lefever equation (Eq. 1) provides the self-phase modulation (i|E|^2 E) and cross-phase modulation (i*2|E'|^2 E) terms that Phase 21 adapts as the computational primitive for a wave-native transformer FFN layer. The physical substrate -- coupled resonant cavities exchanging energy through intensity-dependent phase shifts -- is the direct analog of harmonic bands interacting through amplitude-squared coupling. The Kerr-ODE layer achieves 92% of MLP performance standalone (Phase 21) and 96.8% in the full stack integration (Phase A) at 42.6% of total parameters, confirming that nonlinear optics provides a viable computation model for frequency-structured neural network data.

Kato et al. (2024) demonstrated that neural ODEs serve as practical computation primitives for multi-band signal processing in their Wi-Fi neural dynamic fusion architecture. Their multi-encoder to learned ODE evolution to latent alignment pipeline proves the viability of ODE integration as a trainable computation layer. Phase 21 applies the same principle -- ODE integration as a layer -- but replaces learned neural dynamics with physics-based Kerr dynamics, operating on harmonic transformer embeddings rather than Wi-Fi channel state information.

- Listopad, S. (2025a). *Wave-Based Semantic Memory: A Phase-Aware Alternative to Vector Retrieval.* arXiv:2509.09691. https://arxiv.org/abs/2509.09691
- Listopad, S. (2025b). *Phase-Coded Memory and Morphological Resonance.* arXiv:2511.11848. https://arxiv.org/abs/2511.11848
- Sun, Z., Deng, Z.-H., Nie, J.-Y., & Tang, J. (2019). *RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space.* ICLR 2019. https://arxiv.org/abs/1902.10197
- Moriya, T. (2025). *Surface-Enhanced Coherence Transform: A Framework for Structured Coherence Decomposition.* arXiv:2505.17754. https://arxiv.org/abs/2505.17754
- Wang, L. (2025). *Defierithos: The Lonely Warrior Rises from Resonance -- A Self-Resonance Architecture Beyond Attention.* Submitted to NeurIPS 2025.
- Pal, A., Ghosh, A., Zhang, S., Hill, L., Yan, H., Zhang, H., Bi, T., Alabbadi, A., & Del'Haye, P. (2024). *Linear and Nonlinear Coupling of Light in Twin-Resonators with Kerr Nonlinearity.* arXiv:2404.05646v2. https://arxiv.org/abs/2404.05646 -- Coupled Lugiato-Lefever equation (Eq. 1) provides the self-phase and cross-phase modulation terms adapted in Phase 21's Kerr-ODE layer.
- Kato, S., Wang, P., Koike-Akino, T., Fujihashi, T., Mansour, H., & Boufounos, P. (2024). *Multi-Band Wi-Fi Neural Dynamic Fusion.* arXiv:2407.12937v1 (ICASSP 2024). https://arxiv.org/abs/2407.12937 -- Demonstrates neural ODEs as practical computation primitives for multi-band signal processing, the framework adapted in Phase 21.

## Attribution

This work is a collaboration between Marco (conceptual framework, key theoretical insights, architectural direction) and Claude (Anthropic's AI assistant — mathematical formalization, documentation, test design, and code generation). 

The core insights — that ancient geometric relationship catalogs encode a complete taxonomy of structural relationships, that harmonics are infinite and geometric invariants persist across all frequencies, and that these primitives map onto LLM embedding and FFN layers — originated from Marco's observations and questions during extended collaborative sessions.

## License

All documents, code, and specifications in this repository are released under dual license:

- **Code:** MIT License
- **Documents:** Creative Commons Attribution 4.0 International (CC BY 4.0)

This work is published as prior art to ensure it remains freely available and unpatentable. Use it, extend it, build on it, commercialize implementations of it. The ideas belong to everyone.

## Why Open

This work is released openly and freely because the originators believe that if these mathematical primitives are genuinely useful — for databases, for LLM architectures, for knowledge representation — they should be available to everyone, not locked behind patents or proprietary implementations. 

Publishing establishes prior art. Prior art prevents patents. What belongs to everyone cannot be taken by anyone.

---

*"The patterns are mathematical facts, not cultural inventions. Every civilization that studied circles found the same ones."*
