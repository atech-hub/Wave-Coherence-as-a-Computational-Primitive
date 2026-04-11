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

**Status:** Three pairs tested. Real but not a finding yet. Needs systematic testing across many pairs and catalog types. Predicted structure: pairs in directed relationships (Wu Xing-style) should show larger asymmetry than pairs in symmetric relationships (Western geometric-style) at the same angle.

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

## The Thread We're Following

Five civilizations independently divided circles into segments and cataloged which angles produce meaningful relationships. The framework's geometric relationship catalog strips those observations to pure geometry — 35+ relationship types across 11 angular families. The wave-engine's ODE processes information through coupled harmonic oscillators that naturally produce angular relationships between bands.

The question this investigation is pursuing: **when a model trained on human language builds internal structure through coupled-oscillator physics, does that structure correspond to the angular relationships that humans noticed in other coupled-oscillator systems thousands of years ago?**

The early data says: maybe. Grammar builds dramatically richer angular structure than arithmetic. The structure includes all 11 catalog types. The model places structurally important tokens at distinctive angles. But we're at character level on one model with one dataset, and we've already caught three framing mistakes in one day of looking at this data. The honest answer is: the instruments work, the early signal is interesting, and we need more data before we can say what it means.

The investigation continues.

---

*This document will be updated as new experiments complete. Each update will be dated and will note what changed from the previous version.*
