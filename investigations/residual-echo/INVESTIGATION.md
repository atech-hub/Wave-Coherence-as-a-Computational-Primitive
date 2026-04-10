# The Model Learned to Be a Wire — Residual Echo in Phase-Native Decoding

**Status:** CLOSED. I/Q coherent detection REJECTED. Loss 0.48 was a measurement artifact (low loss only at health check intervals, 99.8% of iterations at 3.4). Arithmetic comparison: I-only 0.167/55-of-55 vs I/Q 0.312/0-of-6. Generation output: garbage. The standard dot product (I-only) IS the correct receiver. Phase-native plateau on grammar remains an open architectural question.
**Date:** 2026-04-06
**Engine:** wave-engine (Rust, Apache 2.0)
**Dimension:** 384-dim (192 bands), 6 layers, 122-token character vocabulary (Shakespeare)

---

## The Question

Phase-native decoding works at 168-dim with small vocabularies (55/55 arithmetic, 46/51 words). Does it scale to grammar at 384-dim with 122 characters?

The non-phase-native path (learned lm_head) reached loss 2.25 on grammar. Phase-native plateaued at 3.35. Why?

---

## The Diagnosis Chain

### Step 1: Is it the architecture?

CPU phase-native at 168-dim reached loss 2.05 on arithmetic (Investigation 12). Candle phase-native at 384-dim plateaued at 3.5 on grammar. Same architecture, different tier and dimension.

**Finding:** CPU tier reaches lower phase-native loss than candle. This suggests a candle-specific issue, not an architectural limit.

### Step 2: Is it the logit scale?

The 384-dim phase-native run started at loss 164. Random should be ln(122) = 4.8. That's 34x above random. The softmax was completely saturated from iteration 0.

At 168-dim, the phase-dot arithmetic run started at loss 21.2 (7.8x above random) but overcame it through training because the magnitudes were smaller.

The dot product against the embedding table produces logits proportional to n_embd. At 384-dim, logits are ~2.3x larger than at 168-dim. Without scaling by 1/√n_embd, the softmax collapses.

**Fix applied:** `logits *= 1.0 / sqrt(n_embd)`. Initial loss dropped from 164 to 8.55 (predicted 8.37 — math confirmed).

**Result:** Loss improved from 3.5 to 3.29 best. Better, but still plateaued. Scaling was one problem but not the only problem.

### Step 3: Is it the attention?

At seq=256, the frozen harmonic attention produces nearly uniform weights:

```
Best head:      max_weight = 0.015  (97% of uniform)
For comparison: arithmetic (seq=16) max_weight = 0.280
```

The frozen harmonic coherence cos(n × Δφ) doesn't differentiate 256 positions well enough. The attention is effectively blind at long sequences.

**Finding:** Attention is dead at seq=256. But the non-phase-native path also has the same frozen attention and reaches loss 2.25. So attention isn't the root cause of the phase-native gap.

### Step 4: The layer flow tells the story

Comparing health monitor data between the two runs:

**Non-phase-native (working, loss 2.25):**
```
L0: cos(in,out)=0.72  FFN=0.69  Attn=0.12  (model transforms the input)
L5: cos(in,out)=0.95  FFN=0.28  Attn=0.14  (residual grows, normal)
```

**Phase-native (stuck at 3.4):**
```
L0: cos(in,out)=0.25  FFN=0.98  Attn=0.03  (L0 does massive work)
L5: cos(in,out)=1.00  FFN=0.08  Attn=0.001 (output IS the input — wire)
```

L5 cos(in,out) = 1.000. The final layer's output is identical to its input. The model's computation vanished. Attention died (0.001). The FFN contributes 0.08 — nearly nothing.

**The model learned to be a wire.**

### Step 5: Why?

The dot product against frozen embeddings measures:

```
dot(output, embedding[v]) = dot(residual + computation, embedding[v])
                          = dot(residual, embedding[v]) + dot(computation, embedding[v])
                            ↑ ECHO (dominates)            ↑ SIGNAL (tiny)
```

The residual stream carries the input embeddings through unchanged. At each layer, `output = input + attention + FFN`. The FFN (ODE computation) adds a small perturbation. After 6 layers, the output is ~98% residual, ~2% computation.

The dot product rewards the echo. The model's optimal strategy is: **don't compute**. Pass the input through unchanged. The echo provides a better match score than any computation could improve upon.

The model rationally learned NOT to think.

### Step 6: The carrier wave analogy

In radio, the input signal is the carrier wave. The information (music) is a small modulation on the carrier. To hear the music, you must **demodulate** — strip the carrier and extract the modulation.

The wave-engine has the same problem. The residual stream is the carrier. The ODE computation is the modulation. The dot product is a broadband detector that picks up carrier + modulation together. Since the carrier is 50x stronger, the detector is deaf to the signal.

---

## The Root Cause

**Phase-native decoding rewards the residual echo over the learned computation.** The model's rational response is to minimize its own contribution and let the input pass through. This is why:

1. Loss plateaus — the model stops learning because NOT computing gives a better score
2. Attention dies — attending to other positions introduces noise on the echo
3. ODE params go pathological — the model fights a broken incentive structure
4. Generation collapses to `"""""` — the most common character in the echo dominates

The lm_head (non-phase-native) doesn't have this problem because it's a SEPARATE learned projection. It can learn to amplify the computation and suppress the echo. The frozen embedding table can't.

---

## The Proposed Fix: Delta Decode

**Subtract the input token's embedding from the output before the dot product.**

```
Current:   logits = dot(residual + computation, embeddings)     ← echo wins
Proposed:  logits = dot(output - input_embedding, embeddings)   ← only computation
```

The subtraction removes the carrier wave. The dot product then measures only the model's computation — what it PRODUCED, not what it RECEIVED.

### Why this preserves base knowledge

The model's computation already contains the base data. The ODE processed the input through 6 layers of nonlinear coupling and attention cross-referencing. The computation IS the digested form of the input. Subtracting the raw input doesn't remove information — it removes the echo.

Analogy: eating an apple. Once digested, the nutrients are in your blood. You don't need the apple anymore. The ODE digested the input through 6 layers. The FFN output IS the digested form. The raw input is the peel.

### Why the dot product still works

The embedding table IS the base vocabulary. The model's computation was built from embeddings. The delta lives in the same wave space as the embeddings. The dot product can decode the delta because they speak the same language.

The model needs to predict token B from context including token A:
- Currently: dot(echo_of_A + tiny_prediction_of_B, embedding[B]) — echo of A dominates
- After delta: dot(prediction_of_B, embedding[B]) — prediction speaks directly

---

## Key Insight: The Prism Connection

The per-band amplitude scaling (prism layer, output_scale) becomes more important with delta decode. The delta is a different kind of signal — the model's computation, not the input echo. Different bands contribute differently to the delta than to the echo. The prism adapts the amplitude profile for this demodulated signal.

Full optical pipeline:
```
Input (light) → ODE on sphere (lens system) → Corrector (phase alignment) →
Prism (amplitude weighting) → Demodulator (subtract carrier) → Detector (dot product)
```

Every stage has a physics-grounded role. Every learnable stage was found through investigation, not guesswork.

---

## Experimental Design

### Test 1: 168-dim CPU, grammar, delta-decode vs echo (direct comparison)
- Same config, only difference is `--delta-decode` flag
- **Metrics:** loss trajectory, L5 cos(in,out), FFN ratios, attention activity
- **Success criteria:** loss descends past the echo-mode plateau; L5 cos(in,out) < 1.0; FFN ratio increases at deep layers

### Test 2: ODE self-organisation
- With echo removed, the ODE should learn instead of being suppressed
- Monitor α, β evolution — should differentiate across layers
- Compare to non-PN ODE differentiation (L0 β=0.193, L5 β=0.201)

### Test 3: Scale to 384-dim on candle (if 168-dim succeeds)
- `--candle --cuda-kernel --phase-native --delta-decode`
- Full grammar test at the dimension that previously failed

---

## Data Supporting the Diagnosis

### Layer flow comparison (iter 5000 health checks)

| Metric | Non-PN L0 | Non-PN L5 | PN L0 | PN L5 |
|--------|----------|----------|-------|-------|
| cos(in,out) | 0.72 | 0.95 | 0.25 | **1.00** |
| FFN ratio | 0.69 | 0.28 | 0.98 | **0.08** |
| Attn ratio | 0.12 | 0.14 | 0.03 | **0.001** |
| Resid ratio | 0.67 | 0.92 | 0.25 | **0.92** |

### Loss trajectory comparison

| Iter | Non-PN | PN (unscaled) | PN (scaled) |
|------|--------|---------------|-------------|
| 0 | 5.31 | 164.0 | 8.55 |
| 500 | 2.77 | 3.88 | 3.77 |
| 2000 | 2.59 | 3.63 | 3.53 |
| 5000 | 2.46 | 3.55 | 3.46 |
| 10000 | 2.43 | 3.41 | — |
| Best | **2.25** | 3.35 | 3.29 |

### ODE parameters (PN run — pathological)

```
L0: α=0.01 (collapsed to minimum — ODE killed its own coupling)
L1: α=0.49, β=0.97 (5x above init — thrashing to compensate)
L5: α=0.10, β=0.20 (unchanged — deep layers gave up)
```

---

## Connection to Previous Investigations

- **Investigation 5 (Output Decoding):** Found lm_head destroys wave structure but is necessary. Delta-decode preserves wave structure AND removes the echo.
- **Investigation 12 (Phase-Native):** Proved 55/55 arithmetic without decoder. This investigation explains why it doesn't scale to grammar: the echo problem is manageable at 15 tokens but fatal at 122.
- **Investigation 10 (Harmonic Scaling):** Frozen attention works at short sequences but fails at seq=256. Not the root cause of the PN gap, but a contributing factor.
- **Corrector Plate:** Phase correction alone doesn't fix the echo. The prism (amplitude scaling) helps but doesn't address the fundamental residual dominance.

---

## Status

- [x] Root cause hypothesised: residual echo dominates phase-native dot product
- [x] Confirmed by data: L5 cos(in,out)=1.000, FFN ratio 0.08 (at 384-dim)
- [x] Fix designed: delta-decode (subtract input embedding before dot product)
- [x] Spec written: `specs/DELTA-DECODE-SPEC.md`
- [x] Logit scaling fix verified: 1/√n_embd reduces init loss from 164 to 8.55
- [x] Prism layer (output_scale) already implemented and wired in CPU tier
- [x] Delta-decode tested at 168-dim CPU — **NEUTRAL** (same plateau at 3.4)
- [x] I/Q monitor implemented and run — **PHASE HYPOTHESIS REJECTED**
- [ ] Translation problem remains open — see Step 7 below

---

## Step 7: I/Q Monitor Kills the Phase Hypothesis

Before implementing coherent detection, we built a monitor to observe where the signal actually lives. The I/Q channel monitor decomposes the dot product into:
- **I channel** (in-phase): Σ(r_out × r_emb + s_out × s_emb) — measures amplitude correlation
- **Q channel** (quadrature): Σ(s_out × r_emb - r_out × s_emb) — measures phase correlation

### Results (168-dim, 4L, grammar, phase-native)

| Iter | I_disc | Q_disc | IQ_ratio | Phase_std | I_rank | Q_rank |
|------|--------|--------|----------|-----------|--------|--------|
| 0 | 7.2 | 2.9 | 0.44 | 1.72 | 41 | 46 |
| 500 | 21.2 | 1.2 | 0.20 | 1.62 | 22 | 59 |
| 1000 | 25.9 | -6.0 | 0.27 | 1.53 | 28 | 62 |
| 1500 | 36.9 | -5.1 | 0.21 | 1.45 | 18 | 52 |

### Interpretation

**The signal is in I, not Q. The phase hypothesis is wrong.**

- I discrimination grows 5x over training (7.2 → 36.9) — the amplitude channel IS learning
- Q discrimination goes NEGATIVE (-6.0) — the phase channel actively hurts prediction
- IQ ratio drops (0.44 → 0.21) — the model moves signal FROM phase TO amplitude
- Phase_std decreases (1.72 → 1.45) — the model REDUCES phase variation over training
- I rank improves (41 → 18), Q rank worsens (46 → 62)

The ODE does NOT compute via phase modulation despite being based on the Kerr effect. The model learned to use the AMPLITUDE changes that result from the coupled dynamics, not the phase shifts themselves. The Kerr nonlinearity produces phase shifts, but the training signal flows through amplitude.

**The dot product IS the correct receiver.** It hears everything the model produces. The bottleneck is not detection.

### What this means

1. Coherent detection (Q channel, I/Q mix) would HURT, not help
2. Delta-decode is neutral because the echo isn't the bottleneck at 168-dim
3. The plateau at 3.4 is genuine capacity limitation, not a decoder mismatch
4. The translation problem is real but the cause is not I vs Q

---

## Step 8: Delta-Decode Results

Tested at 168-dim, 4L, grammar:

| Iter | Delta | No-Delta | Difference |
|------|-------|----------|------------|
| 0 | 4.98 | 5.39 | -0.41 (delta starts lower) |
| 500 | 3.66 | 3.63 | +0.03 |
| 1000 | 3.51 | 3.51 | 0.00 |
| 2000 | 3.46 | 3.43 | +0.03 |

Delta decode starts at a better initial loss (residual echo IS removed) but converges to the same plateau. The model finds the same solution either way at this scale.

Note: The residual echo (L5 cos=1.000) was observed at 384-dim, 6L — not at 168-dim, 4L. The delta-decode test was run at the wrong scale. The echo may still be a factor at 384-dim but this test is inconclusive on that question.

---

## Revised Understanding

The investigation produced three confirmed findings and one open question:

**Confirmed:**
1. Logit scaling (1/√n_embd) is required for phase-native at 384-dim — without it, softmax saturates from iter 0
2. The ODE computes in AMPLITUDE, not phase — despite the Kerr effect being a phase phenomenon
3. The dot product IS the correct receiver for what the ODE produces

**Open:**
4. Why does phase-native plateau ~1 loss point above non-phase-native at the same config?
   - Not detection mismatch (I/Q proves dot product is correct)
   - Not residual echo at 168-dim (delta decode is neutral)
   - Possibly capacity-related (frozen embeddings are harder targets than learned lm_head)
   - Possibly residual echo at 384-dim only (untested — delta decode at 384-dim not run)
   - Possibly frozen attention at seq=256 (both PN and non-PN affected equally)

---

## Update: New Data on Open Question #4 (2026-04-10)

The grammar plateau has moved. Phase-native + FWM at 168-dim reached **best loss 2.34** at iter 49579 (80K run), breaking the 3.1 plateau that was on record. The gap between phase-native and non-phase-native may be narrower than this investigation measured.

Three new pieces of evidence bear on open question #4:

**1. L3 regime shift.** The grammar model's L3 transitioned from preservative (cos=0.92, residual dominates) to destructive (cos=0.45, FFN at 95%) between iter 6K–18K. This is the opposite of the "model learned to be a wire" finding from this investigation's 384-dim run. At 168-dim with FWM, the model found its way to controlled destruction organically. Whether this would also happen at 384-dim is untested.

**2. Hidden coherence.** The galaxy scan was measuring coherence only at zero phase offset. A hidden coherence probe found 1,328 pairs at grammar L4 with stable coherence at non-zero offsets that the dot product readout doesn't see. The dot product — confirmed as "correct" in this investigation — is correct for what it measures, but Proposition 3.5 (Cosine Similarity Blindness) proves it has structural blind spots. A per-channel harmonic decoder would access structure the dot product collapses.

**3. Language builds richer geometry than arithmetic.** Grammar: 4,766 locked FWM quartets, 60,830 oscillating, 528 triads. Arithmetic: 0 locked, 236 oscillating, 1 triad. Phase-native preserves 9× more triads than lm_head (2×2 comparison, April 9). The decoder shapes the model's learned geometry, and phase-native builds dramatically more structure that its own readout can't fully utilise.

**Revised framing:** The plateau is not purely capacity-related. It's at least partly a decoder-readout mismatch: phase-native training builds rich geometric structure, but the dot-product readout collapses that structure into one scalar per token. The model builds more than it can say.

This investigation's core findings remain valid. The update is that "the dot product is the correct receiver" may need the qualification: correct for what it measures, but not measuring everything the model builds.

See Geometric Vocabulary investigation for the full data and next experiments.
