# Experiments

Exploratory experiments on a 4-layer, 4-head, 128-dim harmonic transformer trained on Shakespeare (~1.1M chars, 842K parameters, 65 tokens). All experiments run with CUDA, 3000 training steps, batch 64, lr 3e-4. Results are from a small model — findings should scale with model size.

## Paper Cross-Reference

Each experiment below maps to the framework validated by the core test suite (Tests 1-25 in `src/` and `python/`). The core tests establish the mathematical primitives; these experiments test whether those primitives manifest in trained transformer internals.

### Core Test Suite (in repo root)

| Test | Location | What it validates |
|------|----------|-------------------|
| Tests 1-23 | `src/main.rs`, `python/run_tests.py` | Full mathematical framework — phase encoding, harmonic coherence, kernel admissibility, self-indexing, torus composition, dynamic mutation |
| Test 24 | `python/embedding_analysis.py` | Harmonic structure in real transformer embeddings (all-MiniLM-L6-v2, 384d). Spectral variance: synonyms 0.0031, antonyms 0.0094, unrelated 0.0215. Cosine blind spot confirmed on production embeddings. |
| Test 25 | `python/harmonic_transformer.py` | Character-level harmonic transformer. Harmonic 1.5223 vs baseline 1.5570 vs frozen 1.5567. Frozen embeddings (zero gradient updates) match fully-trained baseline. Cross-validated in Rust (`rust-transformer/`). |

### Training Experiments (this directory)

| # | File | What it tests | Key finding |
|---|------|---------------|-------------|
| 1 | `spectral_persistence.py` | Do harmonic channels survive through layers? | Partial — n=4, n=11, n=14 persist strongly (correlation 0.36-0.47) |
| 2 | `geometric_relations_probe.py` | Do channels stay independent? | 92.5% independence at final layer. Network actively disentangles. |
| 3 | `knowledge_editing.py` | Can we edit knowledge via weight surgery? | No — knowledge lives in MLP weights, not embeddings |
| 3b | `harmonic_injection.py`, `harmonic_injection_v2.py` | Can we swap identity via embedding geometry? | Yes — 80.7% swap rate, frozen model matches trainable |
| 4 | `harmonic_construction.py` | Can novel harmonic vectors produce coherent output? | Yes — 0.991 interpolation correlation, 100% coherent output |
| 5 | `musical_harmonics.py` | Do musical intervals predict channel behavior? | +0.454 correlation with edit safety. Character types form classifiable spectral chords. |
| 6 | `progressive_learning.py` | Does structure-first training outperform standard? | Yes — 1.5585 vs 1.5876 val loss. 5/5 faster knowledge absorption. |
| 7 | `concept_composition.py` | Do characters compose into word representations? | 33M-fold context divergence. Semantic clustering at character level (king-lord 0.721). |
| 8 | `init_convergence.py` | Do different seeds converge to the same structure? | No — cross-run similarity is zero. Independence: 87.8% ± 0.8% (harmonic) vs 85.5% ± 1.9% (baseline). |
| 9 | `commitment_point.py` | Where does the model commit to a prediction? | layer3_mlp — half the accuracy in one step. Mid+High bands are minimum viable set. |
| 10 | `early_exit.py` | Can the model skip layers for easy tokens? | 2-4% compute saving at 97-99% accuracy. Newlines: 50.9% exit at 97.8% accuracy. |
| 11 | `chord_flow.py` | Can word-level chords replace tokens? | Boundaries detectable (+0.055 to +0.128 gap) but composition works by differentiation. |
| 12 | `natural_expression.py` | How does the progressive model's internal space differ? | Fewer dimensions (5 vs 6 for 50% variance), richer dreams (maintains diversity vs collapses), 22.3% trapped structure. |
| 13 | `expression_curriculum.py` | Can richer output heads unlock trapped structure? | Deep head: best accuracy (0.535). Linear head: best knowledge preservation (11.6x efficiency). |
| 14 | `shakespeare_knowledge.py` | Does the model know Shakespeare? | P("discontent")=0.39, P("Juliet")=0.28. Mid bands 1.6x more active during confident predictions. |
| 15 | `harmonic_decoder.py` | Can mid-band confidence guide decoding? | Harmonic beam 0.186 vs greedy 0.164 (+13.4%). Adaptive beam width from confidence signal. |
| — | `sweep-test/` | Rust reproduction of harmonic sweep (Test 21) | Independent validation — same results, different language. |

### Key Cross-References

| Paper topic | Primary evidence | Supporting evidence |
|-------------|-----------------|---------------------|
| Harmonic coherence as query operator | Tests 1-3, 9, 14 | — |
| Cosine similarity blindness | Test 21 | Test 24 (real embeddings) |
| Self-indexing property | Tests 18-20 | — |
| Kernel admissibility | Test 22 | — |
| Harmonic embeddings as structural priors | Test 25 | Phase 1 (persistence), Phase 2 (independence) |
| Progressive curriculum training | Phase 6 | Phase 12 (internal landscape), Phase 13 (expression) |
| Emergent confidence signals | Phase 14 | Phase 15 (decoder), Phase 9 (commitment) |
| Knowledge representation | Phase 3, 3b | Phase 4 (construction), Phase 7 (composition) |
| Cross-seed reproducibility | Phase 8 | — |
| Five corrective findings | Tests 4, 8, 10, 16, 23 | — |

## Results

See [RESULTS.md](RESULTS.md) for all numerical data tables.
