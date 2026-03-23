# Wave Memory Investigation

**Status:** IN PROGRESS (5/7 experiments complete, BPE extension planned)
**Started:** 2026-03-15
**Engine:** kerr-engine v0.3.0 (experiments 1-5), wave-engine (planned BPE extension)
**Library:** kerr-memory v0.1.0 (Apache 2.0, public)
**Hardware:** Intel i7-14700K, RTX 4070 Ti

---

## Question

Can persistent ODE initial conditions serve as an experience accumulator for Kerr-ODE models? Standard transformers compute hidden states fresh from input every time. The Kerr-ODE evolves from initial conditions through dynamics. If those initial conditions are modified by accumulated conversational experience, the same model produces different output — same education, different life experience.

## Architecture

- **Model checkpoint (frozen):** Defines ω, γ, α, β, maestro, attention weights. The education. Never changes during inference.
- **Wave memory file (mutable):** Stores accumulated phase/magnitude states. Seeds the Kerr-ODE initial conditions. The experience. Changes with each conversation.
- **File format:** KWMF (Kerr Wave Memory Format). ~1.5KB for a 64-band, 3-layer model.
- **Accumulation:** Exponential moving average (EMA) with configurable beta. After each conversation, the ODE's trajectory is averaged into the memory file.
- **Reset:** Delete the file. Output returns to bit-identical baseline (Experiment 4).
- **Inspection:** Run harmonic census on the memory file. Anomaly detection catches spikes before they reach the model (Experiment 5).

## Method

Seven experiments, stop-at-null principle. Each experiment has a clear question, a measurable metric, and a threshold for pass/fail.

Base model: kerr-engine, 128-dim, 64 bands, 4 layers, ~354K params, 3000 iterations on Shakespeare, character-level tokenization.

## Completed Experiments

### Experiment 1: Injection Sensitivity Sweep — PASS

**Question:** Do ODE initial conditions affect generation quality?

Random memory offsets at alpha values from 0.001 to 1.0. Measured perplexity vs baseline.

**Key finding:** Stochastic resonance. Random noise at alpha=0.05 IMPROVES perplexity by 8.8%. The nonlinear Kerr coupling amplifies useful signal from random perturbation. Sweet spot: 0.05-0.20. Below: below threshold. Above 0.50: overwhelms input.

### Experiment 2: Accumulation Stability — PASS

**Question:** Does EMA accumulation converge over multiple conversations?

20 conversations × 200 tokens each. Beta=0.99. Tracked per-layer energy and growth rate.

**Key finding:** Converges. Growth rate declines monotonically: 265% → 9.5% over 20 conversations. Layer ordering stable throughout: L0 > L1 > L2, ratio ~40:34:25. Same top bands (30, 55-57) dominate every conversation. Beta=0.99 recommended.

### Experiment 3: Semantic Memory — HONEST NULL (char-level) / PARTIAL+ (word-level)

**Question:** Can memory distinguish topic-specific text (love vs war Shakespeare)?

**Char-level result:** Correlation 0.987 between love and war memories. Memory captures corpus texture, not topic. Same top bands dominate both.

**Word-level rerun result:** Same top bands BUT 2x energy difference, reordered peaks (love=B35, war=B61). Generation tonally distinct: love → "fair", "give thee" vs war → "dishonour", "death" from same neutral prompt. Semantic tone influence confirmed.

**Boundary:** Resolution scales with tokenization. Character-level = texture. Word-level = tone. BPE = predicted to enable content-level separation.

### Experiment 4: Memory Reset Safety — PASS

**Question:** Does removing memory restore exact baseline?

**Result:** Bit-identical output after memory deletion. Zero residual effects.

### Experiment 5: Harmonic Census Inspectability — PASS

**Question:** Can anomalous memory states be detected before affecting output?

**Result:** Spike at 50x energy in band 32 caught immediately across all 3 layers. Detection threshold 9.69, spike energy 25.0. Guard before the thing it guards.

## Planned Experiments

### Experiment 6: Damping as Retention Schedule

**Question:** Do trained γ (damping) values predict which bands accumulate memory fastest?

**Hypothesis:** Bands with lower damping retain more signal per conversation, accumulating faster. If confirmed, the ODE's own dynamics define the memory's retention policy — no external scheduling needed.

### Experiment 7: Structured vs Random Memory

**Question:** Is real accumulated memory better than the 8.8% random baseline from Experiment 1?

**Method:** Compare perplexity with (a) real memory from 20 conversations, (b) random noise at alpha=0.05, (c) no memory. If real > random > none, the accumulation captures useful structure beyond stochastic resonance.

### Extension: BPE Scale (Model A)

**Question:** Does semantic tone influence (Experiment 3 word-level finding) survive the jump to BPE tokenization on diverse English?

**Method:** Train Model A (168-dim, 4L) to 50 passes on 12.4MB corpus. Accumulate love-themed and war-themed memories via wave-server. Compare generation from neutral prompts.

**Depends on:** Model A reaching sufficient training (wave-structure investigation).

## Findings Summary

| Experiment | Result | Key Number |
|-----------|--------|-----------|
| 1. Injection sensitivity | PASS | -8.8% perplexity (stochastic resonance) |
| 2. Accumulation stability | PASS | Growth rate 265% → 9.5% (converging) |
| 3. Semantic memory | NULL (char) / PARTIAL+ (word) | 0.987 correlation (char), tonal influence (word) |
| 4. Reset safety | PASS | Bit-identical baseline restoration |
| 5. Anomaly detection | PASS | 50x spike caught at 3σ |
| 6. Damping retention | PLANNED | — |
| 7. Structured vs random | PLANNED | — |

## Connections

- **Pattern 69:** Persistent Wave Memory State for ODE-Based Neural Architectures (ENGINE-PATTERNS.md)
- **Pattern 70:** Versioned Wave Memory with Checkpoint/Rollback Semantics
- **kerr-memory:** Implementation library (public, Apache 2.0, ~920 lines, zero deps)
- **wave-server:** Serves models with --memory flag, accumulates during conversations
- **wave-structure investigation:** Determines when model has sufficient structure for memory to be meaningful

## Detailed Results

See RESULTS.md in this directory for full experimental data tables.
