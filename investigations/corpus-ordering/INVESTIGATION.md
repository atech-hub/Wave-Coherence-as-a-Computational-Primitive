# Corpus Ordering Investigation

**Status:** COMPLETE (Tests 1-3). Open questions remain for future work.
**Started:** 2026-03-13
**Engine:** kerr-engine (Rust, Apache 2.0) — must be public before these results are published
**Hardware:** Intel i7-14700K (20 cores, 28 threads), RTX 4070 Ti (idle — CPU-only training)

---

## Question

Does the order in which corpora are presented during training affect final model quality? If so, what principle governs optimal ordering?

## Background

The Python corpus texture sweep (v2.22.0) found that different literary corpora produce similar harmonic distributions — Shakespeare and children's literature tied (0.931 vs 0.930 trine/square ratio). However, the ranking was seed-unstable and the investigation was paused.

The kerr-engine enables rapid experimentation: 30 training runs in 22 minutes (parallelized across 28 threads). This makes seed robustness testing practical.

## Architecture

- 4-layer Kerr-ODE transformer (128-dim, 64 harmonic bands)
- ~354K trainable params (vocab-dependent)
- Progressive curriculum: 8 bands (0-999 iters) → 24 bands (1000-1999) → 64 bands (2000-3000)
- Character-level tokenization
- Adam optimizer, lr=3e-4, batch=4, seq=64

## Corpora

| Corpus | Source | Size | Vocab | Character |
|--------|--------|------|-------|-----------|
| Shakespeare | `data/input.txt` (tinyshakespeare) | 1.1M chars | 65 | Archaic, complex syntax, dramatic structure, nested clauses |
| Children | `data/children.txt` (Grimm + Andersen + Aesop) | 1.0M chars | 89 | Simple vocabulary, short sentences, explicit structure |
| Legal | `data/legal.txt` (Blackstone's Commentaries) | 1.0M chars | ~80 | Formal prose, narrow vocabulary, repetitive structure, long periods |

---

## Test 1: Initial Corpus Ordering (single seed)

**Date:** 2026-03-13
**Seed:** 42 (model), 1337 (training)

| Run | Training | Iters | Final val loss |
|-----|----------|-------|---------------|
| 1 | Shakespeare only | 3000 | 2.2024 |
| 2 | Children only | 3000 | 2.0772 |
| 3 | Children → Shakespeare | 3000 + 3000 | 2.1357 |
| 4 | Shakespeare → Children | 3000 + 3000 | 1.9524 |

**Initial finding:** Shakespeare → Children produces the lowest val loss. Sequential training beats single-corpus (in the right order). Wrong order (Children → Shakespeare) is worse than children alone.

**Caveat:** Single seed. Required robustness testing.

---

## Test 1b: Seed Robustness (5 seeds)

**Date:** 2026-03-13
**Seeds:** 42, 100, 200, 300, 400
**Total runs:** 30 training sessions (20 standalone + 10 sequential)
**Wall time:** 22 minutes (parallelized, 10 concurrent jobs)

### Results

| Seed | Shak Only | Child Only | Child→Shak | Shak→Child |
|------|-----------|------------|------------|------------|
| 42   | 2.2024    | 2.0772     | 2.1357     | **1.9524** |
| 100  | 2.1989    | 2.0471     | 2.1153     | **1.9537** |
| 200  | 2.2355    | 2.0750     | 2.1407     | **1.9958** |
| 300  | 2.2041    | 2.0885     | 2.1257     | **1.9680** |
| 400  | 2.2201    | 2.0529     | 2.1129     | **1.9697** |

### Verdict: ROBUST (5/5)

**Shakespeare → Children wins every seed.** No overlapping ranges between any adjacent configuration:

1. **Shak→Child** (1.952 – 1.996) — best
2. **Child only** (2.047 – 2.089)
3. **Child→Shak** (2.113 – 2.141)
4. **Shak only** (2.199 – 2.236) — worst

### Observations

1. **Sequential always beats single-corpus (in the right order).** Shak→Child beats both baselines at every seed.
2. **Wrong order is worse than children alone.** Child→Shak (2.11-2.14) underperforms child-only (2.05-2.09) at every seed. The Shakespeare phase damages representations built on simple text.
3. **Consistent ranking.** The ordering is identical across all 5 seeds with zero overlap between adjacent pairs. This is not marginal — it's definitive.

---

## Test 2: Legal as Extreme Case (3 seeds)

**Date:** 2026-03-13
**Seeds:** 42, 100, 200
**Hypothesis:** If "complexity builds capacity," then Shak→Child should beat Legal→Child. Legal constrains through repetition, not complexity. If the mechanism is complexity-specific, legal pre-training should produce worse foundations.
**Total runs:** 15 training sessions
**Wall time:** 18 minutes

### Results

| Seed | Child Only | Shak Only | Legal Only | Shak→Child | Legal→Child |
|------|-----------|-----------|-----------|------------|-------------|
| 42   | 2.0772    | 2.2024    | 2.0132    | 1.9524     | **1.9453**  |
| 100  | 2.0471    | 2.1989    | 2.0121    | **1.9537** | 1.9635      |
| 200  | 2.0750    | 2.2355    | 2.0611    | **1.9958** | 1.9760      |

### Verdict: HYPOTHESIS REJECTED

**Legal→Child and Shak→Child are statistically indistinguishable.** Differences are within noise (~0.02). No consistent winner — Legal wins seed 42, Shakespeare wins seeds 100 and 200.

### Surprises

1. **Legal standalone beats both other corpora.** Legal (2.01-2.06) < Children (2.05-2.08) < Shakespeare (2.20-2.24). Blackstone's formal, structured prose is the easiest to model of the three.

2. **Legal pre-training is equally effective as Shakespeare.** The "complexity builds capacity" hypothesis is wrong. Both pre-trainings produce equally good foundations.

### Revised Interpretation

The mechanism is not "hard packaging builds capacity." It is **"different packaging builds transferable representations."** Any corpus sufficiently different from the target forces the model to develop flexible internal structure that transfers well. Shakespeare and legal text are both very different from children's literature — in vocabulary, syntax, and register — and both produce equal benefit.

The constrained-first principle from band curriculum (8→24→64) still holds, but the corpus-level analogue is: **train on something different first, then consolidate on target.** The first corpus doesn't need to be harder — it needs to be different enough to prevent shallow solutions.

---

## Test 3: Three-Stage Curriculum + Iteration Control (3 seeds)

**Date:** 2026-03-13
**Seeds:** 42, 100, 200
**Purpose:** (a) Does three-stage beat two-stage? (b) Does diversity beat raw iteration count?
**Total runs:** 9 training sessions

### Results

| Seed | Legal→Shak→Child (9K) | Child→Shak→Legal (9K) | Child 9K control |
|------|----------------------|----------------------|------------------|
| 42   | 1.9078               | 1.7580               | 1.8384           |
| 100  | 1.8980               | 1.7683               | 1.8598           |
| 200  | 1.9073               | 1.8192               | 1.8465           |

Note: Config 1 eval on children, Config 2 eval on legal, Config 3 eval on children.

### Verdict: NUANCED

**Three-stage beats two-stage:** Legal→Shak→Child (1.90) improves over Shak→Child from Test 1b (1.95). More diversity helps.

**But 9K single-corpus beats three-stage:** Child 9K (1.84) beats Legal→Shak→Child (1.90). Given enough iterations on the target corpus, raw exposure overtakes the diversity benefit.

### Key insight: diversity is more *efficient*, not more *powerful*

Comparing models at equal iterations on children’s text (3000):

| Config | Total iters | Iters on children | Val loss (children) |
|---|---|---|---|
| Child only 3K | 3000 | 3000 | 2.05–2.09 |
| Shak→Child | 6000 | 3000 | 1.95–2.00 |
| Legal→Shak→Child | 9000 | 3000 | 1.90–1.91 |
| Child only 9K | 9000 | 9000 | 1.84–1.86 |

At equal target-corpus exposure, diversity always wins. But the 9K control had 3x more exposure to the target and wins on raw familiarity. Diversity is more efficient per iteration on the target corpus — it reaches better performance with less target-corpus exposure — but given unlimited iterations, raw repetition on the target eventually catches up.

**Practical implication:** If training budget is limited, pre-train on diverse corpora first. If training budget is unlimited, just train on the target corpus longer. The diverse path gets you there faster but the endpoint is the same.

---

## Summary of Findings

1. **Sequential corpus training beats single-corpus training at equal target-corpus exposure** — robust across 5 seeds, zero overlap (Test 1b)
2. **Order matters** — wrong order is worse than single-corpus (Test 1b)
3. **The mechanism is diversity, not complexity** — legal and Shakespeare produce equivalent pre-training benefit (Test 2)
4. **Legal text is the easiest to model** — lowest standalone val loss of all three corpora (Test 2)
5. **Three-stage beats two-stage** — more diversity stages improve results (Test 3)
6. **Diversity is more efficient, not more powerful** — sufficient iterations on the target corpus eventually overtakes the diversity benefit (Test 3)

## Reproducibility

All results produced by kerr-engine with deterministic seeds. Scripts:
- `run_seed_robustness.sh` — Test 1b (5 seeds × 4 configs)
- `run_test2_legal.sh` — Test 2 (3 seeds × 5 configs)
- `run_test3_threestage.sh` — Test 3 (3 seeds × 3 configs)
- JSON summaries in `seed_robustness/`, `test2_legal/`, and `test3_threestage/`

Engine: `kerr-engine train [data] [iters] [batch] [seq] [lr] --seed N`

---

## Open Questions

- Does the diversity principle hold for corpora from the same genre? (e.g., two different Shakespeare plays — same domain, same register, minimal "difference")
- Is there a minimum corpus size or minimum domain distance below which pre-training doesn't help?
- How does this interact with word-level tokenization?
- What is the crossover point — how many target-corpus iterations does it take for single-corpus to catch up with diversity-pretrained? (Between 3K and 9K based on current data)

## Methodological Notes

- Sequential runs use 6000 total iterations (3000 per corpus) vs 3000 for single-corpus baselines. The iteration count confound is addressed two ways: (a) wrong-order sequential at 6000 iters underperforms single-corpus at 3000 iters, ruling out iteration count as the driver; (b) Test 3 shows Child-9K (9000 iters) beats three-stage (9000 iters) at equal total iterations, confirming diversity is an efficiency gain, not an iteration-count artefact.
- Vocab resize: when resuming on a corpus with larger vocabulary, new lm_head rows are added with fresh random init. Initial loss spike (~6.0) recovers within 300 iterations. Existing weights transfer intact.
- All training uses progressive curriculum (8→24→64 bands). The curriculum resets on resume — the second corpus trains through all three band stages again.
- Val loss is computed on a held-out 10% split of the *current* corpus, not the previous one. So Shak→Child val loss is measured on children's text, not Shakespeare.
