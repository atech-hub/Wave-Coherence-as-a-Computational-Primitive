# The Scaling Wall — When the Architecture Was Sound but the Accounting Was Wrong

**Status:** CONFIRMED
**Date:** 2026-03-31
**Engine:** wave-engine (Rust, Apache 2.0)
**Hardware:** Intel i7-14700K, RTX 4070 Ti, 32GB DDR5

---

## Question

The wave-engine works at 168-dim. Learnable ODE, corrector plate, per-layer self-organisation — all validated. Loss 2.70 at char-level, 3.67 at BPE. The architecture is sound.

256-dim should be better. More bands (128 vs 84), more capacity, more room for dual-channel encoding. The frozen ODE already achieved loss 3.71 at 256-dim. The learnable ODE should beat it decisively.

Instead, 256-dim diverged. Loss climbed from 6.9 to 12.5. Three different configurations tried, three failures. Same architecture that works at 168-dim.

Why does the architecture break when it gets wider?

## The First Divergence: β Runaway

The first 256-dim attempt used `--out-proj-groups 8` (block-diagonal output projection), learnable ODE, corrector plate, curriculum training. The fix from a pre-existing bounds error: `n_embd % out_proj_groups` must be zero, so groups=8 was chosen for 256-dim (128 bands).

The run diverged at iter 12K. The coupling constants told the story:

| Layer | α at 15K | β at 15K | Problem |
|---|---|---|---|
| L0 | 0.150 | 0.198 | Normal |
| L1 | 0.233 | **0.376** | β pushing against 0.5 clamp |
| L2 | 0.182 | **0.412** | β at 82% of clamp ceiling |
| L3 | 0.023 | 0.162 | Normal |

β was running away. At 168-dim, β settled at 0.18-0.23. At 256-dim, L1 and L2 pushed past 0.37-0.41 and were still climbing. The 0.5 clamp was the only thing preventing them from going higher.

The loss spike coincided with the curriculum transition from 32 to 64 bands. The model had learned coupling for 32-band geometry. When 64 bands appeared, the neighbour sum (β coupling) effectively doubled. The combination of aggressive coupling plus sudden band count increase produced instability.

Diagnosis: AGC ceiling mismatch. The ceiling was set at 1.0, computed from the initial α=0.1, β=0.2. But at β=0.41, the physics ceiling is `sqrt(π/2 / (0.15 + 4×0.41))` = 0.93. The ODE was operating above its stability limit.

## The Second Divergence: Dynamic AGC

The dynamic per-layer AGC was implemented — the ceiling tracks the learned α/β automatically, tightening when coupling increases. The β clamp was raised. The curriculum was removed (the learnable ODE controls its own band activation through γ damping).

Same result. Loss climbed from 6.9 to 12.5. No NaN, no crash — just loss going up. The AGC tracked correctly (ceiling dropped to 1.109). β stayed stable at 0.19-0.21 — the runaway was fixed. But the model still couldn't learn.

ODE params at 15K:

| Layer | α | β | AGC ceiling |
|---|---|---|---|
| L0 | 0.048 | 0.211 | 1.109 |
| L1 | 0.177 | 0.231 | 1.109 |
| L2 | **0.500** | 0.194 | 1.109 |
| L3 | 0.204 | 0.010 | 1.109 |

L2 hit the α=0.5 clamp ceiling. The model was screaming for more capacity. But the AGC was fine, the coupling was stable, no NaN. The loss climbed anyway.

The AGC wasn't the problem. The curriculum wasn't the problem.

## The Diagnosis

Code noticed something nobody had checked: the parameter balance.

At 168-dim with dense out_proj (groups=1):

```
Body params: 161K (48%)
Head params: 172K (52%)
Total: 334K
```

At 256-dim with block-diagonal out_proj (groups=8):

```
Body params: 107K (29%)
Head params: 262K (71%)
Total: 369K
```

29% body, 71% head.

The lm_head at 256-dim is `vocab_size × n_embd` = 1024 × 256 = 262K parameters. With block-diagonal out_proj (groups=8), each group is only 32×32 = 1K params per layer, total body capacity: 107K.

The head had more than twice the parameters of the body. The gradient was dominated by the head — it learned to predict tokens from whatever the body gave it, but the body didn't have enough capacity or gradient share to build good representations. The head learned statistics. The body couldn't learn structure.

The model at 256-dim groups=8 was a tiny body driving a huge head. Like a moped engine in a truck chassis.

## The Fix

Dense out_proj (groups=1). At 256-dim, the dense out_proj is 256×256 = 65K params per layer instead of 8 × (32×32) = 8K. This rebalances:

```
Body params: 336K (56%)
Head params: 262K (44%)
Total: 598K
```

56% body, 44% head. The body can learn.

## The Result

| Config | Body/Head | Best loss | Outcome |
|---|---|---|---|
| 256-dim groups=8 | 29/71 | 6.26 (diverged) | β runaway, loss climbing |
| 256-dim groups=8 + dynamic AGC | 29/71 | 6.26 (diverged) | AGC fixed, still diverging |
| 256-dim groups=8 + no curriculum | 29/71 | 6.26 (diverged) | Curriculum removed, still diverging |
| **256-dim groups=1 (dense)** | **56/44** | **3.07** | **Converged — new project record** |

Loss 3.07 at 256-dim. The new all-time best for this dimension. The architecture works. The accounting just had to balance.

Additional metrics from the successful run:

| Metric | Value | Notes |
|---|---|---|
| THD at 9K | L0=0.005, L3=0.003 | Lower than 168-dim — cleaner |
| AGC ceiling | 1.32 → 1.34 | Barely moved — coupling stable |
| β range | 0.15 → 0.20 | Calm, near sweet spot |
| n_compressed | 0 | AGC never fired |
| Output vocabulary | grammar + Shakespeare | "subject", "noun", "DUKE", "death" |

The β that ran away to 0.41 at groups=8 settled at 0.20 at groups=1. The model didn't need extreme coupling when it had enough body capacity. The runaway was a symptom of starvation, not a property of the architecture.

## The Three Misdiagnoses

1. **"The AGC ceiling is wrong."** Partially true — the static ceiling was wrong for learned coupling. The dynamic AGC fixed this. But it didn't fix the divergence because the AGC wasn't the cause.

2. **"The curriculum is wrong."** Partially true — the curriculum creates discontinuities that the learnable ODE can't adapt to fast enough. Removing it helped stability. But it didn't fix the divergence because the curriculum wasn't the cause.

3. **"The α clamp is too low."** L2 hitting α=0.5 looked like a ceiling problem. But L2 was pushing α because it couldn't get enough computation from its limited parameters. Give it more parameters (dense out_proj) and α settles at 0.15 — nowhere near the clamp.

Three plausible hypotheses. Three partial fixes. The actual cause: the model was starving for body parameters.

## Why This Wasn't Obvious

At 168-dim, the parameter balance was 48/52 — close to even. Nobody checked whether it would hold at 256-dim because the assumption was "more dimensions = more capacity = better." The block-diagonal out_proj (groups=8) was chosen because groups=6 (the default from 768-dim) didn't divide 256 evenly. The divisibility fix suggested groups=8 as valid. Architecturally valid. Computationally wrong.

The ratio flipped because the lm_head scales as `vocab × n_embd` (linear in width) while the block-diagonal out_proj scales as `groups × (n_embd/groups)²` (sublinear in width for constant groups). As the model gets wider, the head grows linearly but the body grows sublinearly. The imbalance gets worse at every dimension increase.

This is why dense out_proj was required at the proven 168-dim configuration. The project notes already said so: "Block-diagonal out_proj starves ≤256-dim models; dense out_proj required at these scales." The note existed. The lesson was forgotten when a different groups value was needed for divisibility.

## The Lesson

When an architecture works at one scale and fails at another, check the accounting before checking the physics. The ODE was fine. The AGC was fine. The corrector plate was fine. The coupling dynamics were fine. The parameter budget was wrong.

A model needs balanced capacity to learn. The head decodes; the body encodes. If the head has twice the body's parameters, it learns to decode noise — and the noise is all the body can produce with its limited capacity.

The fix wasn't a new algorithm or a new architectural insight. It was arithmetic: count the parameters, check the ratio, rebalance.

The sysadmin would recognise this. It's the same as a server with a fast network card and a slow disk. The network accepts requests beautifully. The disk can't serve them. The bottleneck isn't where you expect — it's where the capacity doesn't match the demand.

## Technical Notes

### Parameter counts by configuration

| Config | ln | maestro_in | maestro_out | out_proj | lm_head | Total |
|---|---|---|---|---|---|---|
| 168-dim groups=1 | 1.3K×4 | 3.0K×4 | 3.0K×4 | **28.2K×4** | 172K | 334K |
| 256-dim groups=8 | 2.0K×4 | 4.4K×4 | 4.4K×4 | **8.2K×4** | 262K | 369K |
| 256-dim groups=1 | 2.0K×4 | 4.4K×4 | 4.4K×4 | **65.5K×4** | 262K | 598K |

The out_proj is the swing factor. Dense: 65K per layer. Block-diagonal groups=8: 8K per layer. 8× difference in the body's projection capacity.

### Dynamic AGC — a valid fix for a different problem

The dynamic per-layer AGC (74 lines of code) correctly solves the coupling-tracks-ceiling problem. When β climbs, the ceiling tightens automatically. This prevents the β runaway and keeps the ODE in its stable regime. The fix is correct and necessary — it just wasn't sufficient because the divergence had a different cause.

### Curriculum retirement at learnable ODE

The curriculum was designed when the ODE was frozen — the model had no way to control band activation. With learnable ODE, γ (damping) per band serves as an internal curriculum. The model suppresses bands it's not ready for and activates them when needed. The external curriculum is redundant and creates harmful discontinuities.
