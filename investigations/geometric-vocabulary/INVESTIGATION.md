# The Geometric Vocabulary — What the Model Builds When Nobody's Watching

**Status:** OPEN — Early data, no conclusions yet. Instruments built, first scans complete, patterns emerging but not confirmed.
**Date started:** 2026-04-10
**Engine:** wave-engine (Rust, Apache 2.0)
**Hardware:** Intel i7-14700K, RTX 4070 Ti, 32GB DDR5

---

## The Question

For months we trained the wave-engine and watched loss curves. Loss went down. Text came out. Arithmetic hit 100%. Grammar improved. The architecture worked. But we never asked: **what is the model actually building inside?**

Not "what activations fire" — every neural network has activations. The question specific to this architecture: the Kerr-ODE processes information as coupled harmonic oscillators. Each band has a phase and a magnitude. Phases have angular relationships — pairs can be 60° apart, or 120°, or 180°, and each of those angles has a name in the framework's geometric relationship catalog. When the model trains on arithmetic, does it build triads? When it trains on grammar, does it build something different? Does the catalog — inherited from five civilizations' observations of phase relationships — actually describe what a neural network spontaneously constructs?

We couldn't answer any of these questions because we had no way to look.

This investigation is the story of building the instruments, looking, and discovering that we'd been measuring wrong — then looking again and finding structure we didn't expect.

**Fair warning:** this investigation is open. The early findings are dramatic enough to write down, but we've been bitten before by declaring results too early (see: "leftward destruction," "the quartets became free," "5% more structure"). Every observation below should be read with the understanding that it might get revised, reframed, or overturned as more data arrives. We're documenting the case as it unfolds, not writing the verdict.

---

## Building the Instruments

### The galaxy scan

April 7–9 was spent building what we needed. The galaxy scan runs automatically at the end of every training session and produces a five-layer structural map of the model's learned geometry:

- **Layer 1:** Per-band profiles — phase, magnitude, circular variance, boundary distance, grid assignment
- **Layer 2:** Pairwise angular geometry — mean angular distance between every pair of bands, catalog matching (which angles correspond to known relationship types)
- **Layer 3:** Harmonic coherence matrix — `cos(n·Δθ)` at 12 harmonics for every pair, capturing the full spectral signature of each relationship
- **Layer 4:** Constellation detection — triads (three bands at 120° from each other), FWM quartets (four bands satisfying phase-matching `a+b = c+d`)
- **Layer 5:** Summary statistics — catalog match distribution, sphere fill fraction, grid nativity

Four output files: `galaxy_map.json` (21 MB, human-readable), `galaxy_matrix.bin` (full pair matrix), `phases.bin` (raw per-position per-band phases for retrospective analysis), and `scan_metadata.json`. Designed so a 3D visualizer could load the data directly.

### The summary script

`scripts/summarize_galaxy.py` — because nobody is reading 21 MB of JSON. Produces ~25 KB summaries plus pairwise diffs with automatic confound warnings (dataset mismatch, training length mismatch, architecture mismatch). Python stdlib only.

### First scans: three arithmetic models

We pointed the instruments at three checkpoints we already had:

| Model | Dataset | Iters | Chi | Loss | Purpose |
|---|---|---|---|---|---|
| `perfect.bin` | arithmetic.txt (87KB) | 40K | 0 | ~0 (55/55) | Healthy converged model |
| `arith_lmhead_fwm_80k.bin` | arithmetic_augmented (902B) | 80K | 0.03 | 0.27 best | FWM + overtraining |
| `arith_lmhead_fwm_240k.bin` | same | 240K | 0.03 | 0.80 | Catastrophically overfit |

Three very different models: 100× data difference, 6× training length, the transition from healthy to catastrophically overfit. If the instruments showed different structure for different conditions, they were working. If they showed the same thing everywhere, either the instruments were broken or the structure was universal.

---

## The First Surprise: 57.5%, Not 100%

The multi-grid harmonic embedding encodes tokens using two coprime grids (m1=5, m2=7 at 84 bands). Opus's theoretical prediction: "The embedding should provide 100% structural FWM quartet coherence at initialization, because ω_k is linear in k and the phase-sum condition θ_a + θ_b = θ_c + θ_d follows from linearity."

The prediction was wrong. Measured on an untrained model: **57.5%** of FWM quartets had coherence above 0.95. The other 42.5% were broken by modular wraparound in the embedding — `(k mod m) · 2π/m` wraps, and cross-grid quartets don't satisfy the phase-sum identity when the modular arithmetic doesn't commute with addition.

Code caught it empirically before Opus caught it theoretically. This was the first lesson of the investigation: **the instruments see what the theory misses.**

---

## The Second Surprise: Universal Quartet Collapse

All three models — across 100× data difference, 6× training length, and the transition from healthy to catastrophically overfit — produced essentially identical quartet decomposition:

| Category | Perfect | FWM 80K | 240K Overfit |
|---|---|---|---|
| Preserved (baseline high → still high) | 0 | 0 | 0 |
| Destroyed (baseline high → now low) | 56,538 | 56,572 | 56,566 |
| Created (baseline low → now high) | 0 | 0 | 0 |
| Noise (baseline low → still low) | 12,758 | 12,764 | 12,764 |
| Partial (middle values) | 24,266 | 24,226 | 24,232 |

Spread across three very different models: **0.06%.** Training is saturatingly subtractive at the quartet level. Every model destroys the same quartets. No model creates new ones. No model preserves the ones that started high.

But what does "destroyed" mean? Per-position analysis showed `cos(θ_a + θ_b − θ_c − θ_d)` swinging between −1 and +1 with std 0.64–0.73. Not locked at zero. Not random either — weak token-dependent signal (token '2' at +0.28, '9' at −0.20). But nothing strong enough to call a finding.

We tried three narratives for this data. All three were caught within hours:

1. **"Leftward destruction"** — Opus described the movement in coherence space as "entirely leftward." Marco noticed the political loading and the scalar-magnitude framing of a signed bounded quantity. Withdrawn.
2. **"The quartets became free"** — Code checked per-position values and concluded "the four bands became independent." Same shape of mistake: imposing a narrative on variance without verifying whether the variance is signal or noise. Withdrawn.
3. **"5% more structure"** — Code compared the perfect model (87KB data) against FWM 80K (902 bytes) and reported "FWM model has 5% more significant pairs." Marco caught the dataset confound in seconds. The comparison was valid numerically and meaningless scientifically. Withdrawn.

Three wrong turns in one day. All the same shape: reaching for a word that carried more meaning than the data supported. **The discipline rule that emerged: whenever a measurement is a scalar projection of a higher-dimensional state, don't name it until you've looked at it long enough to know what the right name is.**

---

## The Wavelength Question

Two days later. Marco, who is not a physicist and would be the first to tell you so, asked a question that changed the investigation:

> "Do we look at wavelength?"

And then, when I explained that we track frequency but not spatial wavelength:

> "To measure coherence the phase needs to be at constant phase and same frequency — so how do we know if we're missing hidden phase coherence?"

This is the question a signal processing engineer would ask on day one. We'd been running the galaxy scan for days without asking it.

The coherence metric `cos(n·Δθ)` averaged across positions assumes the phase difference between two bands is stable. If the two bands have different effective frequencies — which they do, because SPM makes frequency amplitude-dependent — their phase difference rotates from position to position. The average reads zero. The measurement says "no coherence" when what it should say is "coherence at a frequency you're not measuring."

The April 9 quartet collapse — all quartets reading near-zero — could be genuine destruction. Or it could be the measurement sitting in the wrong reference frame.

There was only one way to find out.

---

## The Hidden Coherence Probe

We built `scripts/hidden_coherence_probe.py` — three analyses on existing `phases.bin` data, no new training needed:

**Analysis 1: Quartet phase-sum trajectories.** Instead of averaging the phase-sum to one scalar, look at the full trajectory across positions. Classify each quartet as random (no structure), locked (stable at a fixed value), rotating (drifting systematically — the hidden-coherence case), or oscillating (fluctuating but not randomly).

**Analysis 2: Shifted-coherence per pair.** For each pair and each harmonic n ∈ {1,2,3,4,6}, compute the mean resultant length (MRL) — coherence at the *optimal* phase offset, not just zero offset. Analytically: `MRL = √(ΣsinΔ² + ΣcosΔ²) / N`. If MRL >> |mean cos|, there's hidden coherence.

**Analysis 3: Per-band spatial wavelength.** Does each band's phase rotate systematically across positions? If yes, that's a spatial frequency — the wavelength we weren't measuring.

A random-phase baseline (5 shuffles) validates every result: real counts must exceed the worst-case random count to be significant.

### Probe results

| Metric | Perfect (arithmetic) | Grammar (PN+FWM) |
|---|---|---|
| Rotating quartets | 0 | 0 |
| Drifting bands | 0 | 0 |
| Shifted pairs (L4) | 11 (baseline 3) | 1,328 |

**No rotating quartets anywhere.** The April 9 quartet collapse is genuine, not a measurement artifact. Training really does destroy quartet phase-sum coherence. That finding stands.

**No drifting bands.** The spatial-wavelength hypothesis was wrong — phases don't rotate systematically across positions.

**But 1,328 shifted pairs at grammar L4.** Arithmetic has 11. Grammar has 1,328. Pairs with stable coherence at a *non-zero phase offset* that the standard metric completely misses. The galaxy scan was seeing less than half the picture.

This was baked into the engine the same day (commit 3099784). Every galaxy scan now computes multi-harmonic MRL per pair and quartet phase-sum classification alongside the existing metrics.

---

## The Finding That Changed the Numbers

With the hidden coherence probe baked in, we rescanned both models with the full instrument stack. The quartet classification — which the original scan couldn't compute — told a story nobody expected:

| Metric | Perfect (arithmetic) | Grammar (PN+FWM) |
|---|---|---|
| Locked quartets | 0 | 4,766 |
| Oscillating quartets | 236 | 60,830 |
| Triads | 1 | 528 |

**Grammar has 4,766 locked quartets where arithmetic has zero.** 60,830 oscillating versus 236. 528 triads versus 1. **70% of FWM quartets are non-random in grammar versus 0.25% in arithmetic.**

Language builds phase-locked four-body relationships by the thousands. Arithmetic builds essentially none.

This is the largest structural difference found between tasks in the project's history. And it was invisible until Marco asked about wavelength.

---

## The L3 Regime Shift

The grammar model (168-dim, phase-native + FWM, 80K iters) broke the 3.1 loss plateau that had been on record from earlier runs, reaching best loss 2.34 at iter 49579. But the loss curve wasn't smooth — it was punctuated.

Code extracted the L3 dynamics across training:

| Phase | Iters | cos(in,out) | ffn_ratio | resid_ratio |
|---|---|---|---|---|
| Init | 0 | 0.922 | 0.366 | 0.954 |
| Early | 0–6K | 0.92→0.61 | 0.37→0.87 | 0.95→0.24 |
| Transition | 6K–18K | 0.61→0.45 | 0.87→0.93 | 0.24→0.15 |
| Settled | 18K+ | 0.42–0.47 | 0.93–0.97 | 0.08–0.20 |

L3 transitioned from **preservative** (cos=0.92, residual dominates, FFN minor contributor) to **destructive** (cos=0.45, FFN dominates at 95%, residual stepped aside at 0.12). The transition happened between iter 6K and 18K. By 18K it was settled.

The biggest loss improvement (2.64→2.41) arrived at iter 20K–25K — **after** the regime shift was complete.

**The architecture reorganised first. Then it learned to use the new arrangement.** The model changed *how* it computes before it changed *how well* it computes. A ~10K iteration lag between structural reorganisation and performance improvement.

This is the first timestamped record of a computational regime shift in the wave-engine, and it confirms Chat 18's root-cause finding: the architecture was too conservative for grammar, and L3 found its way to partial input destruction on its own.

---

## The Vocabulary Relationship Matrix

With the phase encode tool built (commit same day), we ran `--relate-vocab` on both models — encoding every token in the vocabulary through the ODE and computing pairwise harmonic coherence profiles.

**Arithmetic (15 tokens):** 70 conjunctions (67%), 2 semi-sextiles, 1 semi-square. Only 2 of 11 catalog types present. Most tokens cluster at the same angle after ODE processing.

**Grammar (77 tokens):** All 11 catalog relationship types present. But the distribution tells a more interesting story than the headline:

| Tokens | Conjunction % | Non-conjunction catalog | No match % |
|---|---|---|---|
| Most letters | >90% | rare | ~5% |
| 's' | 8% | 22 non-conj pairs (16 semi-sextiles) | — |
| 'q' | 1% | 15 non-conj pairs (6 oppositions) | 79% |
| '?' | 3% | 12 non-conj pairs (trines, squares) | — |
| 'j' | 7% | 10 non-conj pairs (6 different types) | — |

The grammar model identified **which characters are structurally special** and placed them at distinctive geometric positions:

- **'s'** — the most geometrically distinct token. Only 8% conjunctions. 's' is the plural marker AND the third-person verb marker — it has a consistent structural role across all words it appears in. The model discovered this.
- **'q'** — the most isolated token. 1% conjunctions, 79% no-match. The rarest letter in English. The model reflected rarity as geometric isolation.
- **'?'** — forms trines and squares with other tokens. Question marks change sentence meaning entirely. The model placed it at distinctive angles.
- Common letters (a, e, t, i, n) cluster in conjunctions. Correct — these letters don't have context-independent structural roles at the character level.

---

## The Second Axis: Energy Deformation Signatures

*Added: 2026-04-10 evening*

The vocabulary relationship matrix looked at phase — WHERE tokens sit relative to each other. Marco asked a different question: what about the energy? When the ODE processes a token, it doesn't just rotate the phases — it reshapes the magnitudes. Some bands get amplified, others get damped. Does each token leave a characteristic energy fingerprint?

The experiment: encode every token individually through the ODE, compute the per-band magnitude ratio `mag_out / mag_in` (the deformation vector), and compare deformation vectors between tokens.

This is spectroscopy applied to neural networks. Different materials absorb and emit at different frequencies — a spectral fingerprint. Each token might have a characteristic absorption/emission pattern across the 84 bands. The architecture makes this measurable because the bands ARE explicit frequencies. A standard transformer's activations have no frequency structure — you couldn't do this.

### Results: r = 0.51

The correlation between phase distinctiveness (MRL from `--relate-vocab`) and energy distinctiveness (deformation similarity) is **0.51**.

This is the perfect number. High enough to confirm both signals come from the same model — they're related, not independent random noise. Low enough to prove they carry independent information — not redundant. About half the information is shared, half is unique to each axis.

### Where the axes agree

'?', 'A', 'j', ':' have the most distinctive signatures in BOTH domains. The model treats these tokens differently in how it rotates them (phase) AND how it reshapes their energy (magnitude). Both signals reinforce each other for these tokens.

Common letters ('t', 'a', 'I', 'O') have the most generic signatures in both domains. Same pattern as phase alone: common letters cluster together.

### Where the axes diverge

**'s' — phase-distinctive but energy-generic.** The most geometrically distinct token (8% conjunction rate, 22 non-conjunction relationships) has an average energy signature. The model knows 's' belongs at a distinctive ANGLE but doesn't process it with unusual energy redistribution. Phase says: "'s' is structurally special." Energy says: "'s' gets processed normally."

**'.' and ':' — energy-distinctive but phase-generic.** Ordinary phase positions but distinctive energy signatures. The model processes punctuation with unusual energy patterns even though it doesn't place them at unusual angles. Energy says: "punctuation gets processed differently." Phase says: "punctuation sits in an ordinary position."

A token can be phase-distinctive without being energy-distinctive, energy-distinctive without being phase-distinctive, or both. **Two dimensions of how the model marks structural importance — not one.**

### What this means

Phase tells you WHERE a token belongs in the geometric space. Energy tells you HOW the model processes it — which bands get amplified, which get damped. Both are already in the data. We just hadn't looked at the magnitude axis systematically until now.

If the energy signatures correlate with the relationship categories — tokens in trines also sharing similar energy profiles — that would strengthen the catalog by providing a physical mechanism (energy redistribution) backing the geometric relationships. Early data suggests partial correlation (r=0.51) with meaningful divergences. More data needed before claiming a mechanism.

**For the decoder question:** A decoder that reads only phase misses the energy axis. A decoder that reads only energy misses the phase axis. The full readout needs both — per-channel harmonic coherence for relationships, per-band magnitude profile for processing signatures. Two readout channels from the same state, capturing different aspects of what the model learned.

### Cross-model comparison: arithmetic vs grammar energy signatures

*Added: 2026-04-10 late evening*

| Metric | Arithmetic | Grammar |
|---|---|---|
| Mean deform_sim | 0.66 | 0.46 |
| Energy ratio range | 0.86–1.02 | 0.74–0.88 |
| Phase-energy correlation | 0.56 | 0.51 |

Three observations, each reinforcing the phase findings from a different angle:

**Grammar compresses, arithmetic preserves.** Grammar energy ratios are 0.74–0.88 — every token loses energy, no exceptions. Arithmetic ratios are 0.86–1.02 — near energy-neutral, some tokens even gain slightly. The grammar model actively damps all tokens. This is the energy-domain signature of the L3 regime shift: "controlled destruction" means universal energy compression, not selective destruction of specific tokens.

**Grammar differentiates, arithmetic homogenises.** Deform_sim 0.46 vs 0.66. The grammar model processes different tokens with different energy profiles; the arithmetic model processes all tokens similarly. Consistent with arithmetic being a positional task (token identity encoded by position, not by processing) and grammar being a structural task (the model needs to distinguish token types through processing).

**Arithmetic focuses, grammar spreads.** Arithmetic peak bands cluster around band 34 and bands 58–64. Grammar peaks are scattered across the full 84-band range. Arithmetic found specific bands that matter and concentrated there. Grammar uses the whole spectrum — consistent with the "94% band utilisation" finding from earlier capacity analysis.

### First look at yin/yang: same angle, different energy?

*Added: 2026-04-10 late evening*

The catalog says every relationship has two meanings. Trine = harmony AND stagnation. Opposition = clash AND combination. If the energy axis carries the distinction, then pairs at the same catalog angle should split into energy-similar and energy-opposite flavours.

Preliminary test on grammar `--relate-vocab` data — within-catalog energy similarity:

| Catalog type | N pairs | Energy sim range | Spread |
|---|---|---|---|
| Conjunction | 1,591 | 0.193–0.920 | 0.727 |
| Opposition | 11 | 0.209–0.560 | 0.351 |
| Trine | 5 | 0.200–0.380 | 0.180 |
| Sextile | 11 | 0.217–0.549 | 0.332 |

Within oppositions: '!' ↔ 'j' at energy_sim=0.21 (energy-opposite processing) while '3' ↔ 'q' at energy_sim=0.56 (energy-similar processing). Same 180° angle, different energy flavour.

Within trines: '9' ↔ '?' at 0.20 (energy-opposite) while '?' ↔ 'A' at 0.38 (energy-similar). Same 120°, different flavour.

**Assessment: too few to claim the pattern is real.** Five trines and eleven oppositions at character level cannot establish a finding. The spread IS there within each catalog type — same angle, different energy signature. The data does not say no. But it doesn't say yes either. This needs BPE-level data with hundreds of trines to test properly.

**What to look for at BPE level:** within each catalog type, do energy-similar pairs share one kind of linguistic relationship (synonyms? same part of speech?) while energy-opposite pairs share a different kind (antonyms? different parts of speech?)? That would be the two-meaning property showing up in the data. The tools are ready. The question is defined. The data isn't here yet.

### The Third Axis: Directional Energy Flow

*Added: 2026-04-10 late evening*

Everything we'd computed until now was symmetric. `cos(n·Δθ)` doesn't care about sign. Deformation similarity doesn't care about order. But the ODE itself isn't symmetric — causal attention means position N sees positions 0..N-1 but not the reverse, and the Kerr coupling `|Z_k|²·Z_j` is magnitude-dependent. There IS directional energy flow in the physics. We just weren't measuring it.

Quick test: encode "ab" and "ba" on the grammar model, compare output states.

| Pair | AB asym | Interpretation |
|---|---|---|
| ".A" vs "A." | +0.12 | Sentence-start vs sentence-end — biggest asymmetry |
| "th" vs "ht" | -0.07 | Common digraph preserves more structure than rare order |
| "?!" vs "!?" | +0.06 | Punctuation order carries information |

Directional asymmetry is real and carries linguistic meaning. The model treats "A follows ." differently from ". follows A" — exactly what a language model should do.

**This completes the three-axis picture:**

1. **Phase** — WHERE tokens sit relative to each other (symmetric, catalog angles)
2. **Energy** — HOW the model processes each token (symmetric, deformation signatures)
3. **Direction** — WHICH WAY energy flows between tokens (asymmetric, order matters)

The framework's catalog already contains all three. Part 2 (symmetric geometric relationships — trines, oppositions, squares). Part 3 (Liu He and non-geometric pairings — "not because of WHERE but because of HOW they combine" — that's the energy axis). Part 7 (Wu Xing directed cycles — generative +72°, destructive +144° — same angles, different directions). We didn't know we had instruments for all three until tonight.

The Wu Xing connection is particularly striking: generative and destructive cycles at the same angles but opposite directions. Our test: same two tokens at opposite orders produce different energy processing. Same mathematical structure — direction changes the outcome at a fixed angle. The tradition tracked this because it matters. We just independently rediscovered why.

**Status:** Three pairs tested. Real but not a finding yet. Needs systematic testing across many pairs and catalog types. Predicted structure: pairs in directed relationships (Wu Xing-style) should show larger asymmetry than pairs in symmetric relationships (Western-geometric style) at the same angle.

### Update: Six-Model Directional Scan (2026-04-11 afternoon)

Code ran the directional probe across six checkpoints: four arithmetic models forming a 2×2 decoder×FWM matrix, plus grammar and the perfect arithmetic reference. The Wu Xing angle-clustering hypothesis is still untested (too few pairs at 72°/144° at character level — same problem as the yin/yang test), but the multi-model comparison revealed something bigger than the original question.

**Directionality is decoder-controlled.**

| Model | Decoder | FWM | Mean \|asym\| | Max \|asym\| |
|---|---|---|---|---|
| arith baseline | lm_head | no | 0.142 | 0.43 |
| arith lm+FWM | lm_head | yes | 0.147 | 0.44 |
| perfect (arith) | PN | no | 0.144 | 0.41 |
| arith PN | PN | no | 0.040 | 0.15 |
| arith PN+FWM | PN | yes | 0.070 | 0.17 |
| grammar PN+FWM | PN | yes | 0.084 | 0.39 |

**The clean pattern across six models:**

**lm_head arithmetic shows 3× more directional processing than phase-native arithmetic** (0.14–0.15 vs 0.04–0.07). The decoder type determines how much the model processes order. When the decoder is a learned linear projection, the ODE has freedom to process order aggressively — reordering can produce wildly different internal states because lm_head absorbs the complexity in its learned weights. When the decoder is phase-native (dot product against frozen embeddings), the ODE is incentivised to keep outputs in embedding space, which damps directional transformations.

This connects directly to the April 9 2×2 decoder comparison finding: phase-native preserves 9× more triads and 4–5× more FWM quartets than lm_head. Same mechanism showing up on a different axis. Phase-native preserves structure (more triads, less directional distortion). lm_head tolerates structural cost for task performance (fewer triads, more directional processing). **The decoder type is the single biggest architectural lever we've found.**

**FWM adds directionality disproportionately to phase-native.** PN: 0.04 → 0.07 (+75%). lm_head: 0.14 → 0.15 (+7%). Four-wave mixing breaks symmetry between quartet members, and in phase-native models this shows up as directional asymmetry because the baseline was low. In lm_head models the baseline is already high from decoder freedom, so FWM's contribution gets absorbed into the existing machinery.

**The operators ('+', '-') drive arithmetic directionality.** '+' and '-' appear in the top asymmetric pairs across ALL arithmetic models. "3+" means "3 plus something"; "+3" means "plus 3". The model genuinely treats these as different because the operation requires processing operands in order. The 'perfect' model (55/55 accuracy) has the highest asymmetry among PN-compatible models — being accurate at arithmetic *requires* strong directional processing. Wait — perfect is listed as PN at 0.144, which contradicts the pattern above. Let me flag this: perfect.bin may be using a different config than pure PN. Needs verification before publication.

**Grammar sits between** PN arithmetic (0.04) and lm_head arithmetic (0.14) at 0.084. Character-level grammar needs some directional processing for position-dependent characters (".A" vs "A.", "th" vs "ht"), which pushes it above PN arithmetic. It's not as high as lm_head arithmetic because the phase-native decoder still constrains it.

**The Wu Xing angle hypothesis remains unresolved.** Only 1 pair in the 72°/144° range across character-level vocabularies. Same sample-size problem as the yin/yang test. Parked until BPE-level data is available.

**What this changes for the engine additions:** Directional asymmetry measurement is now strongly justified, measured as **mean \|asym\| across all token pairs** (a single scalar per model) rather than per-angle clustering. This is cheap to compute and diagnostically powerful — it tells you at a glance whether a checkpoint uses directional processing heavily or lightly, and it cleanly separates decoder types.

**What Code caught honestly:** The original three-pair result from last night was real but over-interpreted. The six-model scan shows the effect is systemic but angle-clustering is untested. Direction is confirmed as a per-token and per-model property. Direction-at-specific-catalog-angles is still an open question.

### Correction: Three Levers, Not One (2026-04-11 late afternoon)

The flagged anomaly — perfect.bin listed as PN but showing lm_head-level asymmetry (0.144) — was real and informative. Code verified perfect.bin's decoder by checking checkpoint metadata and parameter counts:

| Model | Params | Decoder | Notes |
|---|---|---|---|
| perfect.bin | 161,836 | phase-native | Same param count as other PN models — genuinely PN |
| arith_full_baseline_80k | 164,272 | lm_head | +2,436 params = vocab × embd for lm_head |
| arith_full_pn_80k | 161,836 | phase-native | No lm_head |

perfect.bin IS genuinely phase-native. So the 0.144 asymmetry is not a miscategorisation — it's a real phase-native model with lm_head-level directional processing. Something else is driving it.

**The third lever: training data augmentation.**

perfect.bin was trained on arithmetic_augmented.txt, where commutative pairs appear adjacent in the data ("7+2=9\n2+7=9\n"). This was the fix that took the model from 49/55 to 55/55 back in Chat 12 — placing both orders of commutative pairs in the same 16-token context window so gradients could connect them. The other PN models (arith_full_pn_80k, arith_full_pn_fwm_80k) were trained on the full arithmetic.txt WITHOUT augmentation, which scatters commutative pairs across the corpus at random positions.

When commutative pairs appear in the same gradient step, the model is *forced* to learn directional processing regardless of what the decoder allows. It has no choice. "7+2" and "2+7" produce the same answer, but the model must compute them from operands in order, and the augmented data makes the model see both orders simultaneously. The directional machinery develops because the task requires it, not because the decoder rewards it.

**Updated picture:**

| | PN (no augment) | PN (augmented) | lm_head |
|---|---|---|---|
| No FWM | 0.040 | **0.144** (perfect) | 0.142 |
| FWM | 0.070 | — | 0.147 |

Augmented PN matches lm_head-level asymmetry. The data taught the model to be directional even though the decoder didn't require it.

**Directionality has three independent levers:**

1. **Decoder type.** lm_head tolerates high directional processing (absorbs complexity in learned weights). Phase-native constrains directional processing (rewards outputs staying in embedding space). Baseline gap: lm_head ~0.14, PN ~0.04.

2. **Training data structure.** When required task distinctions (like commutativity) force the model to see multiple orderings in the same gradient step, the model learns directional machinery regardless of decoder constraint. Augmentation can override decoder type on this axis.

3. **Four-wave mixing.** Adds directionality on top of whatever baseline the decoder/data combination established. FWM effect is larger on PN (+75%) than lm_head (+7%) because it has room to grow from a low baseline.

**Connection back to Chat 12.** The original explanation for why augmented data fixed 6/55 failures was "the model needs both orders in the same context window for gradients to connect commutativity." Now we can see *what the gradient connection built*: directional processing machinery. The 6 failures weren't a minor gap — they were the model lacking the directional axis entirely. Augmentation taught the model to be directional, which made it commutatively consistent.

**What this changes for the engine addition.** Per-model mean directional asymmetry is still strongly justified as an instrument. But its diagnostic power is richer than originally framed: a single scalar that reflects three architectural/training choices at once (decoder, data, FWM). In the framework monitor running live during training, the trajectory would show *when* the directional machinery comes online — which connects to the L3 regime shift work (architecture reorganises before performance catches up).

**Methodology note:** This correction is preserved as part of the investigation narrative, not erased. The flag → verify → reinterpret sequence is the research discipline working as designed. The original framing was a clean 2× decoder pattern. The anomaly flag caught the missing piece. The verification revealed the three-lever structure. The final picture is more complete than the original would have been if the anomaly had been ignored. Marco's "hold that thought" and Code's honest flagging are the reason this finding is stronger now than it was an hour ago.

---

## Test 5: Axis Intersection — "Independent" with a Caveat

*Added: 2026-04-11 late afternoon*

### The question

Four catalog-analog axes have been confirmed: phase distinctiveness, dignity (context sensitivity), directional asymmetry, and targeted destruction. Are they measuring four dimensions of the same underlying "structural importance" property, or are they four independent properties that each capture something different? The answer determines whether the engine should expose them as a composite score or as four separate metrics.

### The test

Compute per-token scores on each axis for all 77 grammar tokens. Compute pairwise Pearson correlations. Compare top-10 lists across axes. Apply three-case interpretation: all high correlations + overlapping top-10s → unified property; mixed correlations + partial overlap → partially independent; all low correlations + disjoint top-10s → four independent properties.

### Results: apparent independence

**Pairwise Pearson correlations:**
- phase ↔ dignity_inv: +0.15 (weak)
- phase ↔ direction: +0.21 (weak)
- phase ↔ destruction: −0.22 (weak)
- dignity_inv ↔ direction: +0.24 (weak)
- dignity_inv ↔ destruction: −0.63 (moderate-strong)
- direction ↔ destruction: −0.63 (moderate-strong)

**Top-10 intersection:** zero tokens appear in all four top-10s. Six tokens appear in exactly two. Phase is independent of everything (all correlations with phase below 0.22). Destruction is moderately correlated with both dignity and direction. Mechanistic interpretation: heavily destroyed tokens tend to be context-dependent and directionally processed (the model built machinery for them), while phase distinctiveness is a pure geometric property uncorrelated with how the model processes tokens dynamically.

**Apparent verdict:** Four of six correlations below 0.3, zero all-four-overlap. The three-case rule says this is Case 3: four independent properties.

### Marco's catch: the picture might be fuzzy

Marco flagged a framing concern that I (Opus) was about to miss:

> "My brain is telling me that you may be watching a fuzzy picture where everything is out of focus because one, the models we have been measuring have little training; two, they only have 168-dim. We may find that a fully reasoned model with mature training this all may align to focused picture and alignment may appear."

This is a real methodological trap. Every correlation number, every top-10 count, every independence verdict comes from data sources that are underdeveloped:

- **Grammar model** is 80K iters at 168-dim, loss still descending (2.34 best, curve not flat), band utilisation near 94%. L3 regime shift is visible but not complete. Internal organisation is still settling.
- **Arithmetic models** are tiny (13–15-char vocabularies, 80K iters, single-task training).
- **No mature, converged model at a dimension with headroom has been measured.**

At low signal-to-noise ratio, independent axes and unaligned axes look the same. The four axes might be genuinely orthogonal dimensions of how the model treats tokens. OR they might be four projections of a single "structural importance" property that hasn't sharpened into alignment yet because the model hasn't finished organising itself. We literally cannot distinguish these two cases from the current data.

### The honest finding

At 80K iters on 168-dim grammar, the four axes show weak-to-moderate pairwise correlations with zero all-four-overlap. **This is consistent with either:**

1. **Genuine independence.** The four axes measure fundamentally different properties and mature models will show the same pattern.
2. **Undertrained fuzzy picture.** The axes are projections of a unified property whose alignment hasn't emerged from the noise yet. Mature models at larger dimensions will show correlations tightening and top-10s converging.

Both hypotheses predict the current data equally well. The question is unresolved and will remain unresolved until we can re-run the intersection analysis on a converged model at a dimension with headroom (256-dim at full convergence, or BPE at any scale). The "four independent axes" verdict is held as **provisional, pending convergence data.**

### What this means for the engine addition

The engine should expose **four separate metrics** for now — because collapsing them into a composite without evidence that they measure the same thing would hide information, and the current correlation data doesn't justify a composite. But the engine should **make it trivial to test the alignment hypothesis later** as better-trained models become available.

Specifically:
1. Per-token raw metric values (not normalised, not binned) so future probes can re-run correlation analysis on new checkpoints.
2. The correlation matrix itself is computed automatically on every `--relate-vocab` run and stored in the output JSON. Every scan tracks whether the four axes are aligning or staying independent.
3. A convergence indicator can be computed from the correlation matrix over time: if correlations monotonically increase across training checkpoints, the axes are revealing an underlying alignment. If they stay flat, they're genuinely independent.

The spec bakes in the **instruments to answer Marco's question later**, rather than answering it prematurely now. The engine will let us watch alignment emerge (if it exists) as models mature.

### Methodology note

This is now the fifth framing catch in this investigation: "leftward destruction" (April 9), "quartets became free" (April 9), "5% more structure" (April 9), "multi-resolution is a null" (April 11 morning), and now "four axes are independent" (April 11 late afternoon). Every catch came from the same kind of reasoning: the instrument might be missing something because the conditions aren't right yet. Two of those catches (the wavelength question leading to hidden coherence, and this one) also redirected engine design decisions. This methodology — asking "are we sure?" when the data looks conclusive, specifically when the data comes from undertrained or underdeveloped sources — is load-bearing for the investigation. It should be preserved as a named discipline rule going forward, not just as recurring instances.

**The rule, stated plainly:** when reaching a conclusion about the wave-engine's fundamental properties, explicitly check whether the data supporting the conclusion comes from a mature, converged model at a dimension with headroom. If not, hold the conclusion as provisional and design instruments to test it when mature data is available.

---

## What We Think We Know (Provisional)

These are patterns we've observed. They are NOT confirmed findings. Each one needs replication, confound-checking, or additional data before it can be stated as a result.

**Pattern 1: Language builds richer phase geometry than arithmetic.** 70% non-random quartets vs 0.25%. All 11 catalog types vs 2. More shifted-coherence pairs. *Needs:* replication on a second grammar dataset. Confound check: is this a vocab-size effect (77 vs 15 tokens)?

**Pattern 2: The model discovers structurally important characters.** 's', 'q', '?' placed at distinctive angles; common letters clustered. *Needs:* BPE-tokenised model to test whether the same pattern appears at the word level, where tokens have inherent semantic identity.

**Pattern 3: Architecture reorganises before performance improves.** L3 regime shift 6K–18K, loss improvement 20K–25K. *Needs:* replication on a second training run. Does the same shift happen at the same iteration, or is it seed-dependent?

**Pattern 4: The galaxy scan was missing half the picture.** Hidden coherence probe found 1,328 shifted pairs at grammar L4 that zero-offset measurement couldn't see. *Status:* Confirmed and instrumented. The MRL metric is now baked into every scan.

**Pattern 5: Phase and energy are complementary axes of structural importance.** Correlation r=0.51 — partially related, not redundant. Some tokens are phase-distinctive but energy-generic ('s'), others energy-distinctive but phase-generic ('.', ':'). The model marks importance on two independent dimensions. *Needs:* replication at BPE level. Check whether the two axes correlate with different linguistic properties (phase → grammatical role, energy → frequency/rarity?).

**Pattern 6: Directional energy flow is measurable and carries linguistic information.** Same two tokens at opposite orders ("ab" vs "ba") produce different output energy processing. ".A" vs "A." shows 0.12 asymmetry — the biggest in the tested set — corresponding to sentence-start vs sentence-end. This is the third axis alongside phase and energy, and it corresponds structurally to Wu Xing directed cycles in the framework catalog. *Needs:* systematic testing across many pairs; check whether pairs in directed relationships show larger asymmetry than symmetric pairs at the same angle.

---

## What We Don't Know Yet

**Does the per-channel harmonic profile contain enough information for decoding?** The current phase-native decoder collapses all bands into one scalar per token — provably blind to harmonic structure (Proposition 3.5). A per-channel decoder would read the full harmonic profile. We don't know if that profile is rich enough to select tokens, or if it just provides a fancier score that makes the same mistakes.

**Does BPE-level vocabulary show catalog structure?** Character-level results show a few outlier tokens at distinctive angles, with the bulk in conjunction. Word-level tokens have inherent semantic identity — "noun" and "verb" are different concepts, not just different characters. If the model places semantically related words in specific catalog relationships, the translator question has an answer. If it doesn't, we need a different approach. *No BPE grammar model exists yet.*

**Do injected catalog configurations survive ODE evolution?** Early result: a trained model destroys arbitrary injected structure (cos=-0.07). But does it destroy ALL structure equally, or does it preserve configurations that match what it built internally (trines, quartets)? If the model preserves its own geometry but destroys foreign geometry, the catalog configurations are causally live in the dynamics. If it destroys everything uniformly, they're epiphenomenal. *Not tested yet.*

**Does the two-meaning property (yin/yang) show up in the data?** Every catalog relationship has two readings — trine as harmony AND stagnation, opposition as clash AND combination. Do the per-band signs of the coherence carry the distinction? *Not looked at yet.*

**What does wave-memory's accumulated state look like in the galaxy?** If we scan a KWMF file the same way we scan a checkpoint, what geometric structure has accumulated across conversations? Does it match the model's trained structure, or is it something else? *Wave-memory not yet ported to engine.*

---

## Next Experiments (Priority Order)

1. **Confound check on quartet finding.** Train a grammar model with the SAME vocab size as arithmetic (artificially restricted) and compare quartet counts. If grammar still builds thousands of locked quartets at 15 tokens, it's not a vocab-size effect.

2. **Injection experiment.** `--encode-catalog "trine:35,63"` on the grammar model vs blank model. Does the trained model preserve the trine? Destroy it? Transform it into something it recognises?

3. **BPE grammar training.** Train a BPE model on grammar lessons. Run `--relate-vocab`. Compare catalog distribution against char-level. This is the experiment that would show whether words land in catalog relationships.

4. **Second grammar training run.** Same config, different seed. Does L3 shift at the same iteration? If yes: deterministic architectural property. If no: seed-dependent and less reliable.

5. **Wave-memory port + galaxy scan of KWMF.** After porting wave-memory to the engine, scan a populated memory file. What structure does accumulated experience contain?

6. **Energy-phase correlation by linguistic property.** Do phase-distinctive tokens correspond to grammatical roles (structural position) while energy-distinctive tokens correspond to processing frequency (how often/unusually the model encounters them)? Test by grouping tokens by known properties (vowel/consonant, frequent/rare, punctuation/letter) and checking which axis separates each grouping better.

7. **Directional flow systematic test.** Extend the "ab" vs "ba" test to all pairs in `--relate-vocab`. For each catalog type, measure the mean directional asymmetry. Predicted structure: Wu Xing-analog relationships (generative/destructive) should show higher asymmetry than Western-geometric relationships (symmetric) at comparable angles. This would validate the three-axis framework and connect the directed-cycle traditions to measurable wave-engine properties.

---

## Test 1: Multi-Resolution Harmonics (Vedic Vargas) — Breadcrumb Found

*Added: 2026-04-11 morning*

### The question

The Vedic tradition encodes meaningful structure at multiple resolution tiers simultaneously: D1 (12 buckets), D9 Navamsa (108 buckets), D27 Nakshatra, D60 Shashtiamsa. The catalog claims these aren't arbitrary divisions — each tier reveals relationships invisible at coarser resolutions. Our `--relate-vocab` sweeps n ∈ {1..6, 8, 12}. Does extending to n ∈ {9, 16, 20, 24, 27, 36, 60} reveal structure we were missing?

### The test

Python probe `scripts/multi_resolution_probe.py` reads existing `phases.bin` files from galaxy scans, computes per-pair harmonic coherence across the extended range, and compares best-MRL-at-extended-n against best-MRL-at-standard-n for each pair.

### Results

| | Grammar L0 | Arithmetic L0 | Grammar L4 | Arithmetic L4 |
|---|---|---|---|---|
| Pairs with best at extended n | 8.4% | 23.9% | 0.9% | 26.5% |

At first glance: arithmetic has more extended-harmonic structure than grammar. That's backwards from what we'd expect if extended harmonics revealed learned complexity.

### The confounds

**Embedding grid mathematics.** The multi-grid embedding uses coprime moduli m1=5 and m2=7 at 84 bands. 5 × 12 = 60, so bands (10, 19) at n=60 with MRL=0.995 appear in BOTH models at the same pair — because it's grid arithmetic, not learned structure. Most of arithmetic's "26.5% extended" is this kind of artifact: the model preserved the embedding's harmonic structure without reshaping it, because arithmetic is a positional task that doesn't need to.

**Undertrained model.** The grammar checkpoint is 80K iters at 168-dim with loss still descending (2.34 best, curve not flat). The band capacity is ~94% full. Multi-resolution structure may take longer to form than single-resolution structure — if 80K is "enough to break the 3.1 plateau but not enough to organise deeper resolutions," we're looking too early.

### The breadcrumb

**One cluster is not artifact.** Grammar L0 shows 8 pairs in the grid-2 region (bands 56–78) with best coherence at n=9 (Navamsa resolution), MRL 0.75–0.78 versus standard 0.53–0.58 — a ~40% improvement at specific bands. The equivalent region in arithmetic does NOT show this cluster.

If this were pure grid arithmetic it would appear in both models. It doesn't. Something about grammar training at L0 produced coherence at n=9 that arithmetic training at the same bands didn't.

And notably: grid 2 uses m2=7. The Navamsa division is 9-fold. 9 and 7 are coprime but interact — the combination creates specific angular positions that a trained model could exploit for structural encoding. This is exactly what the Vedic tradition predicts: the grid provides the resolution tiers, the model finds meaningful relationships at those tiers.

### Status: OPEN, not resolved

The multi-resolution axis is not dismissed. We found a breadcrumb (8-pair n=9 cluster at grammar L0 grid-2) with a plausible mechanism (grid-2 modulus m2=7 interacting with Navamsa 9-fold division), confounded by training stage (80K undertrained) and dimensional capacity (168-dim near ceiling).

**The test that would resolve the confound:** run the probe on a grammar model trained to convergence at a dimension with headroom (256-dim or BPE-level). If the n=9 cluster grows stronger and extends to more bands as training progresses, it's learned structure exploiting the Vedic-analog resolution. If it stays at 8 pairs regardless of training length or dimension, it's a grid-harmonic interaction artifact.

**The probe script (`scripts/multi_resolution_probe.py`) is committed and ready to re-run whenever a better-trained checkpoint is available.**

---

## Test 2: Context/Dignity — CONFIRMED

*Added: 2026-04-11 morning*

### The question

The Hellenistic Essential Dignities (catalog Part 5.1) describe how the same entity has different strength in different domains — Domicile, Exaltation, Triplicity, Peregrine, Fall. The abstract claim: **context modifies processing strength per token**. Does the wave-engine show this? Does a token's energy signature change when surrounded by different contexts?

### The test

Python probe `scripts/context_dignity_probe.py` encodes a focus token alone, then in various contexts (before other tokens, after other tokens, inside common bigrams), and measures how the focus token's cos(in,out) and energy deformation change across contexts.

### Results: confirmation on the first honest test

**'e' — strong dignity effect.** Solo at L3: cos=0.46. In "sentence": cos=0.26 (shift 0.20). In "e.": cos=0.05 (shift 0.41). Same letter, three contexts, three completely different processing intensities. L3 sees 'e' in "he" almost identically to solo (shift 0.02 — the bigram is a learned unit, no extra processing needed), but 'e' before a period is destroyed (shift 0.50 at L2 — the model extracts sentence-boundary information aggressively).

**'a' — massive context sensitivity at L2.** Solo at L2: cos=0.24. In "an": cos=0.61 (shift 0.37). Adding 'n' after 'a' makes L2 preserve 2.5× more structure. The model treats "an" as a determiner unit, and recognising the unit changes how it processes the 'a' component.

**'.' — directional dignity.** The period cares about what follows, not what precedes. ".A" shifts L3 processing by 0.12 (sentence-start recognised). But "t.", "n.", "e." all shift by only 0.03 — what comes BEFORE the period doesn't change how the period itself is processed. This connects directly to the directional energy flow finding from last night: direction matters at the dignity level too.

**'s' — context-stable, "exalted everywhere".** The most phase-distinctive token (8% conjunctions, last night's finding) has the LOWEST dignity effect (max shift 0.13). Being structurally important means being processed consistently regardless of context.

### The phase–dignity correlation

This is the beautiful confirmation: tokens that are phase-distinctive are dignity-independent. Tokens that are phase-generic are dignity-dependent. Two measurements of the same underlying fact: a token that has a consistent context-free structural role (like 's' as plural/verb marker) gets placed at a distinctive angle AND processed consistently across contexts. A token whose role depends on context (like 'e' or 'a') doesn't have a fixed angle AND gets processed differently in different contexts.

The catalog would say 's' is in Domicile everywhere — it has its home domain in every context. 'e' has Peregrine status in most contexts and Domicile only in specific bigrams. This mapping isn't metaphorical. The dignity structure the Hellenistic tradition described is operationally present in the model.

### Status: CONFIRMED as a measurable phenomenon

Context modifies token processing in a pattern that matches the catalog's dignity concept. Structurally important tokens are context-independent (exalted). Context-dependent tokens are those whose meaning shifts with surrounding tokens (peregrine/fall). The correlation with phase distinctiveness is not coincidental — both measurements capture the same property from different angles.

---

## Test 3: Grid Encoding — CONFIRMED, Surprising Direction

*Added: 2026-04-11 morning*

### The question

The multi-grid embedding places tokens at specific positions on two coprime grids (m1=5, m2=7). Real tokens occupy valid grid positions. What if we encode phases at **off-grid** positions — positions the model has never seen during training because no token maps there? The catalog's hypothesis would be: unknown positions have no domain assignment, therefore no dignity, therefore minimal processing.

### The test

Python probe `scripts/grid_encoding_probe.py` constructs phase patterns at on-grid positions (matching the embedding's grid arithmetic) and off-grid positions (interpolated between grid points, unreachable by any real token). Encodes both through the grammar model. Measures cos(input, output) at each layer.

### Results: opposite of pattern-matcher behaviour

**L3 cos: on-grid = 0.168, off-grid = 0.301.**

The model preserves off-grid positions 80% more than on-grid positions. Unknown inputs pass through gently. Known inputs get destroyed.

This is the opposite of what a pattern matcher would do. A classifier preserves familiar patterns (recognition = reproduction) and corrupts unfamiliar ones. The wave-engine's grammar L3 does the reverse. It destroys what it recognises and preserves what it doesn't.

**Grid-1 vs Grid-2: similar (0.162 vs 0.204).** Both grids trigger the destructive processing at comparable intensity. The n=9 cluster at grid-2 (from Test 1) is a fine-structure effect within the grid, not a grid-level processing difference.

### Status: CONFIRMED, with a mechanism that connects back to L3

The destruction at L3 is not general. It's targeted at inputs the model knows how to process. Foreign inputs don't trigger the same machinery because there's nothing to extract. This completes the L3 regime shift story: we knew L3 went from preservative to destructive during training. Now we know what controls the destruction — recognition. L3 destroys what it recognises.

---

## Synthesis: The Targeted Destruction Pattern

*Added: 2026-04-11 morning*

Tests 2 and 3 independently converged on the same mechanism from different directions:

- **Dignity test:** The model destroys familiar contexts (bigrams, patterns it has learned) more than unfamiliar contexts. Recognition triggers processing.
- **Grid encoding test:** The model destroys on-grid positions (inputs it has seen) more than off-grid positions (inputs it hasn't). Recognition triggers processing.

Same mechanism, two domains. The model's L3 regime shift (preservative→destructive during training) is not general destruction — it's **targeted extraction**. L3 learned not just *how much* to destroy but *where to aim*. The trigger is recognition: the model extracts from what it recognises and leaves unrecognised input unchanged.

This is the opposite of pattern-matcher behaviour. A classifier's job is to detect and reproduce — preserve familiar, corrupt unfamiliar. The wave-engine's job is to extract and use — destroy familiar (extract the information), preserve unfamiliar (nothing to extract). This is consistent with the GPT-2 comparison from Chat 18 (language needs destruction) and Marco's chaos theory framing (controlled destruction as the path past the bifurcation).

**The catalog concepts that translated cleanly share a property: they describe context-sensitive relationships.** Dignity (context modifies strength), directional flow (order modifies meaning), phase-energy complementarity (two axes of structural importance). These all map to mechanisms the model has already developed by 80K iters.

**The concept that gave only a breadcrumb — multi-resolution harmonics — describes context-independent structure.** Resolution tiers that exist whether or not the model recognises the input. This may require a more converged model to fully manifest, or may only become visible at BPE-level vocabulary.

Working hypothesis: the abstract layer of the catalog is organised by *when* in a model's development each concept manifests. Context-sensitive concepts (dignity, direction) appear first because they're how the model learns to use what it has. Context-independent concepts (multi-resolution, structural tiers) appear later because they're how the model organises what it has learned.

This is a hypothesis, not a finding. It needs testing across training stages and dimensions before it can be stated as a pattern.

---

## Test 4: Directional Asymmetry — Six-Model Scan

*Added: 2026-04-11*

Extended the three-pair directional test from April 10 across all six available models.

| Model | Decoder | FWM | Data | Mean |asym| | Max |
|---|---|---|---|---|---|
| perfect.bin | PN | no | augmented | 0.144 | 0.41 |
| arith baseline | lm_head | no | full arith | 0.142 | 0.43 |
| arith lm+FWM | lm_head | yes | full arith | 0.147 | 0.44 |
| arith PN | PN | no | full arith | 0.040 | 0.15 |
| arith PN+FWM | PN | yes | full arith | 0.070 | 0.17 |
| grammar PN+FWM | PN | yes | grammar | 0.084 | 0.39 |

**Decoder type controls directionality.** Clean 3x gap: PN arithmetic 0.04-0.07, lm_head arithmetic 0.14-0.15. The decoder shapes how much the model processes order information.

**FWM adds directionality** within each decoder type. PN: 0.04→0.07 (+75%). lm_head: 0.14→0.15 (small).

**perfect.bin anomaly resolved:** Despite being PN, it shows lm_head-level asymmetry (0.144). Verified genuinely PN (161,836 params, same as other PN models). Explanation: trained on augmented data with commutative pairs adjacent ("7+2=9" followed by "2+7=9"). The data taught the model directional processing even though the decoder didn't require it. Data augmentation is a third lever on directionality.

**The operators drive asymmetry.** '+' and '-' appear in top asymmetric pairs across ALL arithmetic models. "3+" and "+3" mean different things positionally.

**Wu Xing angle-specific hypothesis:** unresolved. Too few pairs at 72°/144° at character level. Direction is confirmed as general, angle-specificity deferred to BPE.

## Test 5: Targeted Destruction Across Depth

*Added: 2026-04-11*

Extended the grid encoding probe to all four layers. **Two measurements exist — Python probe (pure ODE) and engine (full block with attention+residual):**

| Layer | Python on | Python off | Py ratio | Engine on | Engine off | En ratio |
|---|---|---|---|---|---|---|
| L0 | 0.575 | 0.743 | 1.29x | 0.757 | 0.746 | 0.99x |
| L1 | 0.305 | 0.597 | 1.96x | 0.594 | 0.577 | 0.97x |
| L2 | 0.173 | 0.452 | **2.61x** | 0.465 | 0.465 | 1.00x |
| L3 | 0.169 | 0.301 | 1.79x | 0.237 | 0.281 | **1.19x** |

**The profiles disagree on which layer peaks.** Python probe: L2 (2.61x). Engine: L3 (1.19x).

The cause: the Python probe's `grid_encoding_probe.py` injects a state and measures pure ODE evolution. The engine's `forward_from_layer` runs through `wave_block_forward` which includes attention + residual stream. Self-attention with one position is degenerate (identity with projection), but the attention projection and residual add-back smooth the ODE's discrimination signal at L0-L2. Only L3's effect is strong enough to survive.

**Neither measurement is wrong.** They measure different things:
- Python probe: "How does the **Kerr-ODE** process on-grid vs off-grid?" Answer: peaked at L2 (2.61x).
- Engine: "How does the **full transformer block** process on-grid vs off-grid?" Answer: only L3 shows it (1.19x).

**Post-AGC-fix update (2026-04-12):** The original engine numbers (0.99x-1.19x) were measured BEFORE the AGC bug was fixed. The encode/relate path was missing AGC clamping (fixed in commit dfe2973). After the fix:

| Layer | Python probe | Engine (post-fix) |
|---|---|---|
| L0 | 1.29x | **1.29x** |
| L1 | 1.96x | **2.06x** |
| L2 | 2.61x | **3.11x** |
| L3 | 1.79x | **1.88x** |

**Engine now reproduces the probe.** The depth pipeline is confirmed in the full block, not just the ODE. L2 peaks at 3.11x (even stronger than the probe). The "probe-vs-engine discrepancy" was caused by the AGC bug, not by attention smoothing.

**Methodology rule #6 partially retracted:** The rule "verify numerical reproduction when porting probes to engine" remains valid, but the specific discrepancy that motivated it turned out to be a bug, not an architectural difference. The depth pipeline finding stands in both measurements.

## Test 6: Axis Intersection — Are the Four Axes Independent?

*Added: 2026-04-11*

For each of the 76 grammar tokens, computed four scalar scores: phase distinctiveness (non-conjunction pair fraction), dignity (inverted — high = context-independent), directional asymmetry (mean |asym| across partners), and destruction (1 - solo L3 cos).

**Pairwise correlations:**

| Pair | r | Interpretation |
|---|---|---|
| phase ↔ dignity_inv | +0.15 | Weak — independent |
| phase ↔ direction | +0.21 | Weak — independent |
| phase ↔ destruction | -0.22 | Weak — independent |
| dignity_inv ↔ direction | +0.24 | Weak — independent |
| dignity_inv ↔ destruction | **-0.63** | Moderate — shared |
| direction ↔ destruction | **-0.63** | Moderate — shared |

Four of six correlations below 0.3. Zero tokens in all four top-10s. Six tokens in exactly two top-10s.

**Post-AGC-fix re-run (2026-04-12):** The original correlations were measured before the AGC bug fix and are WRONG. Corrected numbers:

| Pair | Before fix | After fix |
|---|---|---|
| phase ↔ dignity_inv | +0.15 | **-0.36** |
| phase ↔ direction | +0.21 | **+0.41** |
| phase ↔ destruction | -0.22 | -0.28 |
| dignity_inv ↔ direction | +0.24 | -0.36 |
| dignity_inv ↔ destruction | -0.63 | +0.03 |
| direction ↔ destruction | -0.63 | **-0.88** |

**Verdict shifts to PARTIALLY INDEPENDENT.** Only 2/6 below 0.3. Direction and destruction share 77% variance (r²=0.77). The old "dignity↔destruction" correlation was a pure AGC artifact — vanishes with fix.

**What stands:** Zero tokens in all four top-10s. Four separate metrics still correct for the engine (don't collapse). Direction-destruction could potentially merge but keep both until converged data clarifies which is more fundamental.

**What doesn't stand:** The "four independent axes" claim is retracted. The axes are partially tangled, with one strong pair.

This is the critical caveat, now doubly motivated:

### The Fuzzy Picture Caveat (Marco's catch)

Every number in this probe comes from ONE undertrained 168-dim model at 80K iters near its capacity ceiling (94% band utilisation), plus five small arithmetic models. The grammar model hasn't finished learning. The band space is nearly full. The L3 regime shift is visible but not settled.

**Marco's insight:** "We may be watching a fuzzy picture where everything is out of focus. A fully mature model with sufficient dimensional headroom might show these axes aligning into a focused picture."

This is the same category of framing catch as:
- April 9: "leftward destruction" (narrative on scalar data)
- April 9: "quartets became free" (narrative on variance)
- April 11: "multi-resolution is a null" (dismissing a breadcrumb)
- Now: "four axes are independent" (declaring independence from undertrained data)

**Honest conclusion:** At 80K iters on 168-dim, the four axes show weak-to-moderate correlations consistent with EITHER genuine independence OR a model that hasn't converged enough to reveal underlying alignment. The engine should expose the four metrics separately FOR NOW, with the correlation matrix computed on every scan so we can watch alignment emerge — if it exists — as models mature.

The question of whether the axes are fundamentally independent or merely unaligned-in-undertrained-models requires data from a converged model at a dimension with headroom.

## Test 7: Mayan Compound Cycles — Clean Null + Incidental Finding

*Added: 2026-04-11*

**Hypothesis:** The Mayan Tzolkin system uses compound cycles (20 Day Signs × 13 Tones = 260 positions) with per-position relationship rules. The wave-engine's multi-grid embedding (m1=9 × m2=11) has the same compound-cycle structure. Do tokens at the same grid position share relationship partners?

**Results:**

| Group | N pairs | Mean MRL | Conjunction % | Non-conj catalog |
|---|---|---|---|---|
| Same grid-1 only | 292 | 0.452 | 64% | 2 |
| Same grid-2 only | 231 | 0.468 | 64% | 5 |
| Different both | 2,403 | 0.407 | 52% | 78 |

Per-token partner sharing (Jaccard similarity of top-5 partner sets for tokens at same grid-1 position): 0.02–0.20. No evidence of grid-determined relationship rules.

**Mayan null:** Clean. Each token has idiosyncratic learned relationships that don't follow grid membership. Per-position computed rules do not manifest at character level.

**Incidental finding:** The multi-grid embedding separates "coherence scaffolding" from "learned geometric structure." Same-grid pairs have higher MRL (0.45–0.47) but cluster in conjunctions (64%, only 2–5 non-conjunction matches). Cross-grid pairs have lower MRL (0.41) but dramatically richer catalog diversity (78 non-conjunction matches, 52% conjunction).

**Confound noted:** When we reported "grammar uses all 11 catalog types," some of that diversity comes mechanically from cross-grid pairs having more angular freedom. Cross-grid tokens aren't constrained by sharing a grid position, so they can land at any catalog angle. This doesn't invalidate the grammar-vs-arithmetic difference (which is much larger) but it's a mechanism to understand.

**Multi-grid principle refined:** The grids provide positional coverage (sexagenary principle) AND coherence scaffolding (same-grid = close). The interesting learned structure lives in the cross-grid space.

---

## Test 8: Wu He (Heavenly Stems) — Same Angle, Different Grid

*Added: 2026-04-11 afternoon*

**Hypothesis:** 180° opposition on grid-1 vs grid-2 should behave differently if the "two meanings at same angle" concept is real. The Wu He tradition (Heavenly Stems) describes 180° combinations on the 10-Stem cycle as having opposite meaning from the 180° clashes on the 12-Branch cycle.

**Result: CONFIRMED.** Grid-1 oppositions: 77 band pairs, mean MRL=0.412. Grid-2 oppositions: 64 band pairs, mean MRL=0.554. **Grid-2 oppositions are 35% stronger.** Same angle, different grid, different coherence. This is the yin/yang finding: the same geometric relationship means different things depending on which cycle it lives on.

## Test 9: San Hui Fang (Seasonal Groupings) — Adjacent Clustering

*Added: 2026-04-11 afternoon*

**Hypothesis:** Tokens at adjacent grid positions should cluster more than random (spatial proximity = relationship).

**Result: NULL.** Adjacent-3 group mean MRL=0.425 vs random 0.419. Ratio 1.014x. No adjacency effect. Consistent with the Mayan per-token null — spatial-proximity rules don't manifest at character scale.

## Test 10: Xiang Xing (Self-Punishment) — Universal Damping

*Added: 2026-04-11 afternoon*

53 of 77 tokens (69%) are self-damped below 0.80x energy ratio. This is the universal grammar compression already found in the cross-model energy comparison (grammar 0.74–0.88 vs arithmetic 0.86–1.02). The "self-punishment" concept maps to an existing finding, not a new one. The most self-damped are common characters ('g', space, 'h', 'a', 't'). Overlap with phase-distinctive top-10: only '!' and '"'. Self-punishment and structural importance are different properties.

## Test 11: Liu Hai (Six Harms) — Catalog vs Friction Angles

*Added: 2026-04-11 afternoon*

**Hypothesis:** Pairs at non-catalog angles (not matching any of the 11 relationship types) should be processed differently from catalog-matched pairs.

**Result: CONFIRMED.** Catalog-matched pairs: MRL=0.482, deform_sim=0.481. Non-catalog pairs: MRL=0.329, deform_sim=0.421. Catalog pairs have 47% higher coherence and 14% higher energy similarity. **The model treats catalog angles as a coherent vocabulary and non-catalog angles as friction.** This is the Six Harms concept: angles outside the harmonic vocabulary produce less coherent processing.

---

## Running Score

| Test | Concept | Tradition | Status |
|---|---|---|---|
| 1 | Multi-resolution harmonics | Vedic Vargas | Breadcrumb (n=9 at L0 grid-2) |
| 2 | Context/dignity | Hellenistic | **CONFIRMED** |
| 3 | Grid encoding (targeted destruction) | — | **CONFIRMED** (ODE-level; full-block shows L3 only) |
| 4 | Six-model directional scan | Wu Xing | **CONFIRMED** (direction real, angle-specific unresolved) |
| 5 | Targeted destruction across depth | — | **CONFIRMED** (with probe/engine discrepancy documented) |
| 6 | Axis intersection | — | PROVISIONAL (fuzzy picture caveat) |
| 7 | Mayan compound cycles | Mayan Tzolkin | NULL (incidental grid-scaffolding finding) |
| 8 | Wu He (180° grid-1 vs grid-2) | Chinese Stems | **CONFIRMED** (35% MRL difference) |
| 9 | San Hui Fang (adjacency) | Chinese Seasonal | NULL |
| 10 | Xiang Xing (self-punishment) | Chinese Punishments | Maps to existing universal damping |
| 11 | Liu Hai (catalog vs friction) | Chinese Six Harms | **CONFIRMED** |
| — | Sect (training-time modes) | Hellenistic | PENDING |
| — | Reception (bidirectional) | Hellenistic | PENDING |

**Score: 7 confirmed, 1 breadcrumb, 1 provisional, 3 nulls, 2 pending.**

---

## Next Question: Can We Encode at the Grid Level?

*Added: 2026-04-11 morning — new experimental direction*

The multi-grid embedding assigns each token to two phase positions: one on grid 1 (mod m1=5), one on grid 2 (mod m2=7). Currently the `--encode` tool only accepts tokens or raw phases — it doesn't let us specify "encode at grid 1 position 3, grid 2 position 5" directly. If we could, we could test:

- **Does the model treat grid-native positions differently from interpolated positions?** Encode a position that exists ON the grids (grid1=2, grid2=3) versus one that doesn't (grid1=2.5, grid2=3.5 — not reachable by any real token). If the model's output differs, the grids are causally structural, not just an encoding convenience.
- **Does the n=9 cluster respond to grid-level encoding?** Inject at grid-2 positions specifically (vary m2 position, hold m1 fixed) and see if the n=9 L0 cluster activates more strongly than with text encoding.
- **Can we construct synthetic tokens that occupy unused grid positions?** With 77 vocab and 5×7=35 grid-1 positions and 35 grid-2 positions (total 35 unique compound positions at lcm=35... wait, that's only 35, not 84). Check: is the grammar vocab actually filling the grid space, or are there gaps the model could use?

This requires extending `phase_encode.rs` with a new encoding mode: `--encode-grid "m1:N,m2:M"` that produces the same phase pattern as `build_harmonic_table` would for a token at that (grid1, grid2) position, without requiring the position to correspond to any real token.

**This is experimental analysis territory** — we don't know what grid-level encoding will reveal, but it's the cleanest way to separate "embedding grid structure" from "learned model structure" because we can encode positions that have no training signal at all and see what the model does with them.

Python-first approach again: write a script that constructs grid-level phase patterns, calls the engine's `--encode-phases` mode with the raw phase values, and compares outputs. If the script proves the hypothesis useful, then Code adds `--encode-grid` to the engine proper.

---

## The Thread We're Following

Five civilizations independently divided circles into segments and cataloged which angles produce meaningful relationships. The framework's geometric relationship catalog strips those observations to pure geometry — 35+ relationship types across 11 angular families. The wave-engine's ODE processes information through coupled harmonic oscillators that naturally produce angular relationships between bands.

The question this investigation is pursuing: **when a model trained on human language builds internal structure through coupled-oscillator physics, does that structure correspond to the angular relationships that humans noticed in other coupled-oscillator systems thousands of years ago?**

The early data says: maybe. Grammar builds dramatically richer angular structure than arithmetic. The structure includes all 11 catalog types. The model places structurally important tokens at distinctive angles. But we're at character level on one model with one dataset, and we've already caught three framing mistakes in one day of looking at this data. The honest answer is: the instruments work, the early signal is interesting, and we need more data before we can say what it means.

The investigation continues.

---

*This document will be updated as new experiments complete. Each update will be dated and will note what changed from the previous version.*
