# 256-dim Scaling — Can the Model Speak?

**Status:** ACTIVE — C6 confirms rotational learning at 256-dim. Δθ crossed 1.0x (1.68x).
**Date:** 2026-03-28 (updated 2026-03-29)
**Engine:** wave-engine (Rust, Apache 2.0)
**Hardware:** Intel i7-14700K, RTX 4070 Ti, 32GB DDR5
**Dimension:** 256-dim (128 bands), 4 layers, 1K BPE, 597K params
**Starting point:** Scaled from β=0.2 C7 (168-dim all-time best: θ=3.38x, loss 3.91)

---

## The Question

168-dim proved the wave architecture can learn. Twelve cycles, two rotational recoveries, 3.38x semantic discrimination, entropy dropping to 0.842 — all in 1.36 MB. The model knows "predicate", "clause", "composition", "RICHARD." It knows its corpus deeply.

But it can't compose a sentence. It can't respond to a prompt. It produces domain vocabulary in random order — smart words, no grammar. The expression bottleneck at 168-dim: 51% of params in the lm_head, 85% of gradient going to the translator instead of the thinker.

256-dim changes two things at once:

1. **The gradient flips.** lm_head drops from 51% to 44% of params. The model/head gradient split goes from 15/85 to **79/21**. The ODE gets 5x more training signal than it ever got at 168-dim.

2. **More room.** 128 bands instead of 84. Eight attention heads instead of four. 32-dim head vectors. The attention has enough resolution for multi-token coordination — the kind needed for word order, subject-verb agreement, prompt conditioning.

The question isn't whether the model learns at 256-dim. 168-dim proved it learns. The question is whether it can express what it learns. Can a 597K param model, given the right gradient balance and enough harmonic bandwidth, produce structured English?

---

## The Transplant

We scaled the best 168-dim checkpoint (β=0.2 C7, loss 3.91, θ=3.38x) to 256-dim using `--scale`:

- Bands 1-84: transplanted directly from C7 (learned weights preserved)
- Bands 85-128: initialised fresh (random weights, frozen harmonic embeddings)
- Attention: expanded from 4 heads to 8 heads (new heads initialised fresh)
- Out_proj: expanded from [168×168] to [256×256] (old block preserved, rest initialised)

The model inherits 168-dim's semantic structure and extends it with fresh capacity.

---

## C1 — The Transplant Works

| Metric | 168-dim C1 (10K) | 168-dim C4 (40K) | **256-dim C1 (10K)** |
|--------|-----------------|-----------------|---------------------|
| Best loss | 4.48 | 4.10 | **4.09** |
| θ disc. | 1.71x | 1.31x | **2.73x** |
| Phase clustering | 0.347 | 0.380 | **0.387** |
| Entropy | 0.911 | 0.887 | **0.456** |
| Gradient split (model/head) | 15/85 | 15/85 | **79/21** |

Loss starts at 6.26 (not random 6.93 — the transplanted weights give it a head start) and reaches 4.09 in one cycle. It took 168-dim **four cycles** (40K iters) to reach 4.10.

θ discrimination at 2.73x out of the gate. The transplanted bands from C7 carry their semantic structure intact into the larger model. At 168-dim C1 it was 1.71x — the transplant preserved and even strengthened the signal.

And entropy at 0.456 — already half of 168-dim's lifetime best (0.842). The 128-band space allows dramatically tighter inter-modulation structure from the very first cycle.

---

## C2 — Three Surprises

| Metric | 168-dim C2 (20K) | **256-dim C2 (20K)** |
|--------|-----------------|---------------------|
| Loss | 4.18 | **4.11** |
| θ disc. | 3.21x | 1.09x |
| Δθ disc. | 0.84x | 0.86x |
| Entropy | 0.897 | **0.455** |
| Cross/Self | 7.82x | 7.89x |
| Most coupled band | 10.8x | **17.19x (band 84)** |

### Surprise 1: Entropy 0.455

At 168-dim, the lowest entropy ever measured — after twelve cycles and two rotational recoveries — was 0.842. 256-dim hit **0.455** in cycle two. Nearly twice as concentrated.

More bands doesn't just mean more room. It means the model can build dramatically tighter harmonic structure. This is what we predicted ("entropy drops faster at 256-dim") but the magnitude is striking — not slightly faster, nearly 2x tighter in 1/6 of the cycles.

### Surprise 2: Band 84 is the most coupled

The most-coupled band is exactly at the transplant boundary. Bands 1-84 carry C7's learned structure. Bands 85-128 are fresh. The ODE's nearest-neighbour coupling is strongest at the seam — the model is actively stitching old and new bands together.

The new bands are learning by resonating with the old ones. The transplant isn't just a warm start — it's a scaffold that the fresh bands grow from.

### Surprise 3: Grammar vocabulary at 20K iters

Comprehension: "person", "pronoun", "adjectives", "object", "conjunction", "number", "present", "simple", "possess", "speak", "find", "come", "make"

Grammar terms AND verbs. At 168-dim, grammar vocabulary appeared at C4 (40K iters) and verbs came later. 256-dim has both at C2 (20K iters) — twice as fast.

The verb presence is new. "speak", "find", "come", "make" — action words, not category labels. At 168-dim the model learned what parts of speech are named. At 256-dim it's learning what they do.

---

## C3 — The Oscillation Begins, But Different

| Metric | C1 (10K) | C2 (20K) | **C3 (30K)** | Trend |
|--------|----------|----------|-------------|-------|
| Best loss | 4.09 | 4.11 | **4.09** | Plateau |
| Phase clustering | 0.387 | 0.339 | **0.346** | Dipped then recovering |
| θ disc. | 2.73x | 1.09x | **1.61x** | Spiked C1, oscillating |
| Δθ disc. | 0.70x | 0.86x | **0.88x** | Slowly climbing |
| Entropy | 0.456 | 0.455 | **0.433** | Dropping |
| Cross/Self | 7.88x | 7.89x | **7.89x** | Invariant |
| Most coupled | band 85 (17.3x) | band 84 (17.2x) | **band 85 (16.2x)** | Transplant boundary, easing |

### The oscillation is familiar but gentler

At 168-dim: C1 θ spike (1.71x) → C2 θ peak (3.21x) → C3 Δθ spike (2.90x, θ crashes to 0.90x). Wild swings.

At 256-dim: C1 θ spike (2.73x) → C2 θ dip (1.09x) → C3 θ recovery (1.61x). The θ channel dipped and is climbing back. Meanwhile Δθ is slowly rising (0.70→0.86→0.88) — no dramatic spike, just steady growth. The model has enough room with 128 bands to develop both channels gradually instead of swinging between them.

This could be the difference between 84 bands and 128. At 168-dim, the model couldn't fit both encoding strategies simultaneously — it had to oscillate. At 256-dim, there may be enough capacity for both to grow together without competing.

### Loss plateau at 4.09 — then it breaks

Three cycles at 4.09. The model is building internal structure (entropy dropping 0.456→0.433, discrimination climbing) but loss isn't moving. This is the same pattern 168-dim showed — the ODE organises while the loss waits. (The plateau broke at C4 — see below.)

### Comprehension keeps broadening

C3 produces: "subject", "action", "pronoun", "plural", "modif(y)", "number", "verb", "compound", "object", "words", "present", "heart", "name"

This is richer than C2. New additions: "subject", "action", "compound", "heart". The model is acquiring both grammar metalanguage ("subject", "compound") and corpus vocabulary ("heart", "name") simultaneously. At 168-dim this breadth took 50-70K iters.

### The transplant boundary is easing

Most-coupled band dropped from 17.3x (C1) to 16.2x (C3). The seam between old and new bands is integrating. The initial stitching is intense, then relaxes as the new bands absorb structure from the transplanted ones.

---

## C4 — The Plateau Breaks

| Metric | C3 (30K) | **C4 (40K)** |
|--------|----------|-------------|
| Best loss | 4.09 | **3.93** |
| θ disc | 1.61x | **1.80x** |
| Δθ disc | 0.88x | 0.86x |
| Entropy | 0.433 | 0.437 |
| Most coupled | band 85 (16.2x) | band 84 (14.6x) |

Loss broke through to 3.93 — surpassing 168-dim's all-time best (3.91 at 70K iters) in only 40K iters. θ discrimination climbed to 1.80x, the highest sustained level at 256-dim. The transplant boundary continues easing (16.2x → 14.6x).

Comprehension: "adverb", "form", "quest", "world", "English", "thought", "object", "adjective", "meaning", "class", "words", "lord", "letter", "adverbs", "relinquish". "Relinquish" — a four-syllable, low-frequency word with precise meaning. At 168-dim the model was producing category labels. At 256-dim it's producing words that describe actions and concepts.

---

## C5 — New All-Time Record

| Metric | C4 (40K) | **C5 (50K)** |
|--------|----------|-------------|
| Best loss | 3.93 | **3.78** |
| θ disc | 1.80x | 1.37x |
| Δθ disc | 0.86x | **0.98x** |
| Entropy | 0.437 | 0.483 |

**Loss 3.78** — new all-time record across both dimensions. Crushing 168-dim's lifetime best (3.91).

**Δθ hit 0.98x** — right at the dual-channel threshold. At 168-dim, Δθ crossed 1.0x after wild oscillations. At 256-dim the approach is smooth and steady (0.70 → 0.86 → 0.88 → 0.86 → 0.98). If C6 crosses, both channels will be active without the anti-correlation.

θ dipped from 1.80x to 1.37x — the beginning of the familiar oscillation. But at 168-dim this dip was violent (3.38x → 0.77x). At 256-dim: 1.80x → 1.37x — gentle. The channels are trading leadership without crashing.

Comprehension: "sentence", "phrase", "verbs", "words", "verb", "means", "sentences", "God", "death", "different", "know", "action", "being", "object", "adjectives". Grammar concept clusters emerging — the vocabulary is becoming denser and more semantically grouped.

---

## C6 — The Rotation Begins

| Metric | C5 (50K) | **C6 (60K)** |
|--------|----------|-------------|
| Best loss | **3.78** | 3.97 |
| θ disc | 1.37x | 0.76x |
| Δθ disc | 0.98x | **1.68x** |
| Entropy | 0.483 | **0.453** |

The familiar pattern — but with a crucial difference.

At 168-dim, C6 was the dip where BOTH channels dropped below 1.0x (θ=0.77, Δθ=0.90). It looked like crystallisation. At 256-dim C6, θ dropped (0.76x) but **Δθ surged to 1.68x**. The model didn't lose both channels — it traded leadership cleanly. This is what 168-dim tried to do but couldn't until C7.

**Rotational learning confirmed at 256-dim.** The model alternates which channel leads, same as 168-dim. But the rotation is cleaner — no cycle where both channels collapse simultaneously. With 128 bands there's enough room for one channel to surge while the other reorganises.

Entropy dropped to 0.453 — back near the C3 low of 0.433. The model tightens structure during the rotation, same as 168-dim's entropy ratchet.

**Prediction:** C7 should see θ recover (possibly to a new record) while Δθ holds or eases. If both channels are above 1.0x at C7, that's the dual encoding that 168-dim achieved at C4 and C8 but couldn't sustain.

---

## The 168-dim vs 256-dim Comparison

Three cycles in, the differences are already clear:

| Metric | 168-dim at C3 (30K) | 256-dim at C3 (30K) | What it means |
|--------|-------------------|-------------------|--------------|
| Loss | 4.17 | **4.09** | Same ballpark, 256 slightly better |
| θ disc. | 0.90x | **1.61x** | 256 stronger (168 was in Δθ spike) |
| Entropy | 0.878 | **0.433** | 256 is **2x tighter** — fundamentally different |
| Comprehension | "mixed, grammar noise" | **"subject", "action", "verb", "compound"** | 256 has broader vocabulary, earlier |
| Gradient to ODE | 15% | **79%** | 256 gives the model 5x more learning signal |

The entropy gap is the headline. 168-dim never reached 0.433 in twelve cycles. 256-dim is there in three. The model isn't just bigger — it's building qualitatively tighter structure.

---

## What We're Watching For

### Composition (the big question)
Can the model produce word sequences conditioned on the prompt? At 168-dim, "Hello" produced domain vocabulary in random order. At 256-dim, with 8 attention heads and 79/21 gradient split, prompt conditioning becomes physically possible.

### Loss continues dropping *(CONFIRMED at C5)*
Plateau broke at C4 (3.93), then C5 pushed to 3.78. The model hasn't plateaued again yet. Where does it settle?

### Both-channel encoding with clean rotation *(CONFIRMED)*
Δθ crossed 1.0x at C6 (1.68x) while θ dipped (0.76x). Unlike 168-dim where both channels collapsed simultaneously at C6, 256-dim trades leadership cleanly. The anti-correlation is still present but without the destructive both-channel collapse. C7 should tell us if both can be above 1.0x simultaneously.

### The transplant boundary resolves *(IN PROGRESS)*
Band 84-85 coupling easing: 17.3x → 16.2x → 14.6x → 16.8x → 15.9x. Trending down but not monotonic — the model is still actively working the seam.

---

## Technical Details

| Parameter | 168-dim | 256-dim |
|-----------|---------|---------|
| Bands | 84 | 128 |
| Embedding dim | 168 | 256 |
| Attention heads | 4 | 8 |
| Head dim | 42 | 32 |
| Layers | 4 | 4 |
| Total params | 340K | 597K |
| lm_head params | 172K (51%) | 262K (44%) |
| Model params | 168K (49%) | 335K (56%) |
| Gradient split | 15/85 | **79/21** |

Settings: α=0.1, β=0.2, AGC ceiling=1.0, dense out_proj, 1K BPE, grammar+Shakespeare corpus.

---

## Cycle Log

| Cycle | Iters | Loss | θ disc | Δθ disc | Entropy | Most coupled | Comprehension highlights |
|-------|-------|------|--------|---------|---------|-------------|------------------------|
| C1 | 10K | 4.09 | **2.73x** | 0.70x | 0.456 | band 85 (17.3x) | (transplant baseline) |
| C2 | 20K | 4.11 | 1.09x | 0.86x | 0.455 | band 84 (17.2x) | "pronoun", "adjectives", "conjunction", "speak", "find", "make" |
| C3 | 30K | 4.09 | 1.61x | 0.88x | **0.433** | band 85 (16.2x) | "subject", "action", "verb", "compound", "object", "heart", "name" |
| C4 | 40K | 3.93 | 1.80x | 0.86x | 0.437 | band 84 (14.6x) | "adverb", "meaning", "thought", "adjective", "quest", "relinquish" |
| C5 | 50K | **3.78** | 1.37x | 0.98x | 0.483 | band 84 (16.8x) | "sentence", "phrase", "means", "sentences", "action", "being", "know" |
| **C6** | **60K** | 3.97 | 0.76x | **1.68x** | 0.453 | band 84 (15.9x) | **Δθ crossed 1.0x — rotational learning confirmed at 256-dim** |

---

## Cross-References

- 168-dim investigation: [Harmonic Scaling](../harmonic-scaling/INVESTIGATION.md) — all findings that led here
- Output decoder experiments: [Output Decoding](../output-decoding/INVESTIGATION.md) — expression bottleneck, five decoders tested
- Operating regime: [Operating Regime](../operating-regime/INVESTIGATION.md) — why α=0.1, ceiling=1.0, dense out_proj
- Defensive publication: [ENGINE-PATTERNS.md](../../ENGINE-PATTERNS.md) (Pattern 87 — progressive dimension scaling)
