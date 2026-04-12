# Methodology — Research Discipline Rules for the Wave-Engine Investigations

**Status:** DRAFT — initial codification, will be extended as new catches occur
**Date started:** 2026-04-11
**Context:** These rules emerged during the Geometric Vocabulary investigation (April 9–11 2026) after six framing catches in roughly 48 hours. Each rule is tied to a specific instance where the catch actually happened — this is not abstract methodology, it's the pattern-of-thought that kept the investigation honest in practice.

---

## Why this document exists

The wave-engine is novel enough that most of the vocabulary available for describing it comes from other domains where the words meant something slightly different. "Destruction," "independence," "alignment," "recognition," "structure" — all of these carry baggage from signal processing, neural network literature, physics, or information theory. When we reach for those words to describe what we're measuring, we almost always reach too far. The data supports a narrower claim than the word implies, and if the framing isn't caught early it calcifies into the investigation as a conclusion.

Every framing mistake we've caught has been the same shape: **a scalar projection of a higher-dimensional state, named with a word that implied more than the scalar earned.** Six catches in 48 hours strongly suggests this isn't occasional — it's the default failure mode for this kind of research, and it needs explicit countermeasures.

This document lists the rules we've developed, each tied to the specific catch that produced it. The rules are not abstract principles — they're tactical discipline for this particular investigation.

---

## Rule 1: Don't name scalar projections with narrative words

**The catch:** April 9, 2026. Opus described the FWM quartet phase-sum coherence collapse as "the movement in coherence space is entirely leftward" and treated small-magnitude coherence as "destruction." Marco noticed that "leftward" is a political word with directional baggage, and that coherence is a signed bounded projection of a vector rotation — not a scalar magnitude that can be "destroyed." Same day, a second version: after Opus walked back "destroyed," Code checked per-position values and concluded "the four bands became independent / the quartets became free." Same shape of mistake, opposite narrative direction — imposing a freedom story on a variance measurement without verifying whether the variance was signal or noise.

**The rule:** When a measurement is a scalar projection of a higher-dimensional state, do not describe it with words that imply the full dimensional structure. Specifically: avoid "destroyed," "freed," "liberated," "collapsed," "leftward/rightward," "recognised," "understood," "decided." These words describe cognitive or directional actions, not numerical values. Use neutral quantitative language ("the scalar decreased," "the magnitude fluctuated around zero," "the variance increased without angular structure") until enough measurements are in hand to justify the agentive framing.

**When to apply:** Every time you're about to write a sentence where the subject is "the model" and the verb is an action word. Stop and check: does the word describe what the numbers show, or does it describe what the narrative wants the numbers to show?

---

## Rule 2: Comparisons with different datasets or sizes are confounded

**The catch:** April 9, 2026 morning. Code compared perfect.bin (trained on arithmetic.txt, 87KB) against arith_lmhead_fwm_80k.bin (trained on arithmetic_augmented.txt, 902 bytes) and reported "the FWM model has 5% more significant pairs than the perfect model." Marco caught the confound within seconds. The numbers were real, the arithmetic was correct, the comparison was scientifically meaningless because the two models had seen fundamentally different data (100× size difference). The 5% could have come from FWM, or from data size, or from data content, or from any combination.

**The rule:** Before reporting a comparative result ("model A has more X than model B"), explicitly verify that A and B differ in only one variable. If they differ in more than one (training data, training length, vocabulary, architecture, dimension, decoder type, chi), the comparison is confounded and should be either (a) run as a controlled study with only one variable varying, or (b) reported honestly as "these two models differ in multiple ways, and the difference in X could come from any of them." Never report a single-cause explanation for a multi-variable difference.

**When to apply:** Any comparative statement across two or more checkpoints. Also any time you're tempted to say "X model shows more Y" — stop and ask whether the models actually differ in only one relevant dimension.

---

## Rule 3: Breadcrumbs in undertrained models aren't nulls

**The catch:** April 11, 2026 morning. The multi-resolution harmonic probe tested whether extending the harmonic sweep to Vedic Varga resolutions (n=9, 27, 60) would reveal structure that the standard n=1..12 sweep missed. Most of the extended-harmonic signal turned out to be embedding grid arithmetic artifacts (bands 10 and 19 at n=60, present in both arithmetic and grammar — a grid interaction, not learned structure). Opus framed this as "standard harmonics capture most learned structure" — a null result. Marco caught it: "I think you're dismissing it too soon. We had a breadcrumb. Don't forget the model we scan is at 80K iter, barely learned anything, and may not have a lot of freedom on 168-dim." The n=9 cluster at grammar L0 grid-2 bands (present in grammar, absent in arithmetic, with a plausible mechanism) was real, just small and confounded by undertraining.

**The rule:** A small signal in an undertrained model is not the same as no signal. Before declaring a hypothesis dead, explicitly check: (a) is the model converged? (b) does the model have dimensional headroom, or is it near capacity? (c) is the signal absent, or is it present but attenuated? If the answers are "no, no, present-but-attenuated," the correct status is "breadcrumb, needs re-test at convergence," not "null." Nulls should be reserved for signals that fail to appear under conditions where they should clearly manifest if the hypothesis were correct.

**When to apply:** Every time you're about to write "no effect" or "standard X captures most Y" or "the result doesn't support the hypothesis." Stop and check whether the test conditions were adequate to detect the hypothesised effect at its expected magnitude.

---

## Rule 4: A clean pattern with an anomaly has more levers than you think

**The catch:** April 11, 2026 afternoon. The directional asymmetry scan across six models produced a clean 2× pattern: lm_head arithmetic at 0.14–0.15, phase-native arithmetic at 0.04–0.07. "Directionality is decoder-controlled" — tidy, mechanistically plausible, 5 of 6 models fit. The sixth (perfect.bin) was listed as phase-native but showed lm_head-level asymmetry (0.144). Opus's initial instinct was to flag the anomaly as "perfect.bin may be miscategorised" — implicitly suggesting the anomaly might be data entry error rather than real signal. Code verified the checkpoint metadata: perfect.bin is genuinely phase-native. The anomaly was real. Investigation revealed the third lever: training data augmentation (commutative pairs adjacent in the data) forces the model to learn directional processing regardless of decoder type. The final finding — three independent levers — is stronger than the original "decoder controls directionality" would have been.

**The rule:** When a clean pattern has one anomaly, do not discard the anomaly to preserve the pattern. Verify mechanically (checkpoint metadata, training config, data source) whether the anomaly is real. If it is, the anomaly is telling you that your framing is incomplete — there's a lever you haven't identified. Find the lever before finalising the finding. The final result will be more complete and more honest than the original clean pattern would have been.

**When to apply:** Every time a test produces a pattern that mostly fits with one exception. Do not reach for "miscategorisation" or "data error" as the first explanation. The anomaly is usually real, and it usually contains information.

---

## Rule 5: Check mature data before declaring fundamental properties

**The catch:** April 11, 2026 late afternoon. The axis intersection probe (Test 5) showed four pairwise correlations below 0.3 and zero tokens appearing in all four top-10 lists. By the three-case interpretation rule Opus had written hours earlier, this was Case 3: four independent properties. Opus was about to write "four separate metrics, verdict: independent" into the engine spec. Marco caught it: "my brain is telling me you may be watching a fuzzy picture where everything is out of focus because the models have little training and only 168-dim. A fully reasoned model with mature training may show alignment." The correlation numbers were real, but they came entirely from data sources that were underdeveloped — 80K iters, 168-dim, band utilisation near ceiling. At low signal-to-noise ratio, independent axes and unaligned-because-undertrained axes look the same.

**The rule:** Before declaring a conclusion about the wave-engine's fundamental properties (are the axes independent, does X cause Y, does concept A translate to mechanism B), explicitly check whether the supporting data comes from a mature, converged model at a dimension with headroom. If not, hold the conclusion as provisional and design instruments that will let you re-test when mature data is available. Do not commit to engine design decisions based on undertrained data as if the data were definitive — expose raw measurements so future analysis can re-verify or refute.

**When to apply:** Every time you're about to make a claim about the architecture itself (as opposed to a claim about a specific checkpoint's behaviour). The former requires mature data; the latter is valid at any training stage.

---

## Rule 6: Verify numerical reproduction when porting probes to engine

**The catch:** April 11, 2026 evening. The targeted destruction probe in Python showed a depth pipeline: L0 1.29× → L1 1.96× → L2 2.61× → L3 1.79×. This was the basis for the "L0 detects, L1 amplifies, L2 peaks, L3 processes" interpretation. Code baked the destruction measurement into the engine as part of the catalog axes feature, tested it, and found the engine's profile was: L0 0.99× → L1 0.97× → L2 1.00× → L3 1.19×. Same conceptual measurement, completely different per-layer shape. The direction of the effect survived (L3 still shows targeted destruction) but the depth pipeline claim didn't survive the port. Root cause: the Python probe measured pure Kerr-ODE behaviour; the engine measured full-block behaviour including attention and residual stream. Both measurements are valid — they just measure different things.

**The rule:** When baking a Python finding into the engine, explicitly verify numerical reproduction of the probe's reference output, not just directional agreement. Different code paths in the engine include different architectural components, and the same conceptual measurement can produce quantitatively different results depending on which path it flows through. If the numbers don't match, the difference itself is a finding — it tells you which architectural component is responsible for the smoothing, the amplification, or the reshaping. Document both measurements and name what each one measures.

**When to apply:** Every time a Python probe's result is baked into the engine. Before merging the engine version, run both and compare line-by-line. If the numbers differ, don't dismiss the discrepancy — investigate it.

---

## The meta-rule

Every rule above was produced by the same observation loop: Opus wrote a claim that went beyond the data, Marco or Code caught it, the team investigated, the investigation produced a richer understanding than the original claim would have, and the catch was codified as a rule. This loop is load-bearing for the investigation and should be protected.

**Protection looks like this:**
1. When someone catches a framing error, do not smooth it over. Document the catch as part of the narrative. The catch is evidence that the discipline is working.
2. When a rule is proposed, tie it to a specific instance that motivated it. Abstract rules are harder to apply than rules with examples.
3. When in doubt about whether a claim is a rule violation, err on the side of caution. The cost of over-restraining claims is small (slower progress, more "needs more data" tags). The cost of under-restraining is large (calcified wrong framings that future analysis builds on).

The goal is not to be right on the first try. The goal is to be honest about what the data says and catch ourselves quickly when we overreach. Six catches in 48 hours is not a sign of failure — it's a sign that the discipline is functioning and the investigation is producing stronger findings than the raw data alone would yield.

---

## Future catches

This section will be extended as new framing catches occur. Each new rule should be tied to a specific instance, should fit the same tactical format as the six above, and should be added in chronological order so the history of the investigation's methodological development is preserved.

## Rule 7: Silent numerical drift from missing state initialisation

**The catch:** April 11-12, 2026. The encode/relate forward path used `wave_block_forward` → `dual_maestro_forward_cached` → `kerr_ode_forward_cpu` which had no AGC clamping. Training uses `ffn_forward_via_backend` which includes AGC. The encode path didn't crash — it produced plausible-looking numbers. BPE models produced NaN (caught quickly), but character-level models produced wrong-but-plausible destruction profiles (0.99-1.19x) where the correct values were 1.29-3.11x. The entire probe-vs-engine discrepancy (framing catch #6) was caused by this bug, not by attention smoothing as originally hypothesised.

**The rule:** When adding a new forward path through the model, explicitly verify it routes through the same initialisation sequence as training. Missing initialisation doesn't always crash — it can produce wrong numbers that look plausible and pass surface-level sanity checks. The canonical check: run the same input through training path and new path, compare output values numerically. If they disagree, something is missing.

**When to apply:** Every time a new CLI mode, diagnostic tool, or analysis path is added that creates its own forward pass through the model.

---

## Rule 8: Correlation results on buggy measurement paths are untrustworthy

**The catch:** April 12, 2026. The axis intersection (Test 5) was re-run after the AGC bug fix. Four of six correlations changed substantially. `dignity_inv ↔ destruction` went from -0.63 to +0.03 (artifact vanished). `direction ↔ destruction` went from -0.63 to -0.88 (real correlation was stronger than measured). The "four independent axes" verdict was retracted. Marco's fuzzy-picture catch (Rule 5) was vindicated — but by a different mechanism than expected (broken measurement, not undertrained model).

**The rule:** When a measurement path bug is found and fixed, ALL results that went through that path must be re-verified. Don't assume "the direction was probably right even if the magnitude was off." Correlation analysis is particularly sensitive — a bug that adds noise to one axis can create, destroy, or invert correlations between axes. Re-run, compare, and document which numbers changed and which didn't.

**When to apply:** After any bug fix to a measurement path. Specifically: re-run the highest-level analysis that depended on the fixed path, compare old vs new numbers, and update the investigation with both.
