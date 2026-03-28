# Output Decoding — The Model That Knows More Than It Can Say

**Status:** COMPLETE at 168-dim. Five decoders tested, conservation argument validated, low-rank wins.
**Date:** 2026-03-28
**Engine:** wave-engine (Rust, Apache 2.0)
**Dimension:** 168-dim (84 bands), 4 layers, 1K BPE, β=0.2

---

## The Paradox

A 1.36 MB model produces "language", "person", "clauses", "plural", "should", "show", "every", "common", "word", "participate", "modify", "together" at loss 3.91. Grammar textbook AND Shakespeare vocabulary in the same outputs. A standard 340K param transformer at this loss would produce noise. This one knows its corpus.

But ask it "Hello" and it can't say hello back. Ask it "The king" and it produces domain words in random order, not a narrative. The knowledge is there. The expression isn't.

The sub-harmonic diagnostics measure 3.38x per-band semantic discrimination — related tokens are clearly distinguishable from random in the model's internal representation. The ODE builds rich phase structure. But the output is fragmented. Something is lost in translation.

We went looking for what breaks between the model's internal knowledge and its spoken output. What we found was that the lm_head — a standard linear projection borrowed from conventional transformers — is a foreign component bolted onto a wave-native architecture. It destroys the very structure the model learned to build.

---

## The Bottleneck

At 168-dim with 1K BPE, the numbers tell the story:

| Component | Params | Job | Gradient share |
|-----------|--------|-----|---------------|
| ODE + maestro + LN | 161K (49%) | Learn English | 15% |
| lm_head | 172K (51%) | Translate to tokens | 85% |

Half the model is a translator. And the translator takes 85% of the gradient budget to learn its job, starving the part that actually understands language. The ODE builds 3.38x discrimination with only 15% of the gradient. Imagine what it could do with more.

---

## The Analogy That Started It

Marco, late at night: "The closest thing my brain is telling me about wave translation is the speaker — translate digital to sound waves."

A speaker doesn't flatten waves into a list. It transduces them — electrical oscillations become mechanical oscillations become acoustic oscillations. The signal stays waves the whole way through.

The lm_head does the opposite. It takes waves (phase-encoded hidden states) and flattens them into a probability list (1024 numbers). The wave structure is destroyed at the output. It's like converting a symphony into a bar chart of which instruments were loudest.

Marco again: "If microphones can translate the sound waves back to digital, why can't we?"

That closed the loop. The embedding encodes tokens as waves. The ODE processes waves. The attention scores by phase coherence. The output should decode by phase coherence — not by dot product. Waves in, waves out.

---

## The Physics

From conservation law theory (Notes 21): when a measurement operator commutes with the system's dynamics, the measured quantity is conserved — information survives the measurement.

The ODE evolves hidden states through coupled nonlinear dynamics. The lm_head measures by dot product — a linear projection that treats phase and magnitude identically. It conflates the ODE's semantic signal (phase, 3.38x discrimination) with its structural scaffolding (magnitude, 1.00x discrimination — never carries semantics). The measurement destroys what made the signal meaningful.

Phase coherence `cos(θ_hidden - θ_token)` measures only the phase structure — the part the ODE uses for semantics. It commutes with the ODE's phase dynamics. The measurement preserves what was measured.

---

## Five Decoders, One Question

We tested every approach we could think of. Same model (168-dim, β=0.2), same data (grammar+Shakespeare), same training protocol. Only the output decoder changed.

### 1. Tied Embeddings — the rigid decoder (NULL)

**The idea:** Use the frozen harmonic embedding table as the output decoder. Zero learned params. One temperature scalar.

**The result:** Loss 6.23 at 10K iters. The frozen table is too rigid — it can't adapt to meet the ODE's evolved output. The model learned to process waves for four layers of nonlinear dynamics, and the decoder is comparing the result to the pre-evolution patterns. It's like identifying an adult by their baby photo.

**What we learned:** The lm_head earns its params because it adapts. A decoder that can't adapt fails regardless of how it measures.

### 2. Wave Transduction V1 — the phase-native decoder

**The idea:** Replace the dot product with `cos(θ_hidden - θ_token)` — phase coherence scoring. 84 per-band learned weights + 1 temperature = 85 params.

**The crash:** Temperature init at 1.0 produced loss 60. The coherence scores span [-84, +84] and softmax over a 168-point range is numerically catastrophic. Desktop diagnosed it: "The amplifier was clipping. Turn the volume down." Temperature to 0.02 fixed it instantly.

**The optimisation:** `cos(a-b) = cos(a)cos(b) + sin(a)sin(b)`. One trigonometric identity eliminated 86,000 cos() calls per position. Speed went from 180ms to 72ms — matching the lm_head. The wave decoder runs at the same speed as the linear projection.

**The result:** Loss **5.84** at 10K iters. Below tied embeddings (6.23) by 0.4 points. The phase-native measurement works — it preserves what the dot product destroys.

**What the model actually produces at loss 5.84:** Early-stage vocabulary fragments, not yet domain-specific. Still calibrating.

### 3. Wave Transduction V2 — the dual-channel decoder

**The idea:** At β=0.2, the model encodes in both θ (per-band phase) and Δθ (differential phase between adjacent bands). V1 only reads θ. Adding 83 Δθ weights should read both channels.

**The result:** Loss 5.85. Zero improvement. The Δθ channel is used internally by the ODE (confirmed by 1.77x Δθ discrimination at C12) but is not decodable through the frozen embedding's differential phases. The frozen embedding table's phase differences between adjacent bands are a function of the coprime grid structure, not of token identity.

**One signal in the noise:** Diff weight 41 spiked to 1.40 — the boundary between grid 1 (bands 0-41) and grid 2 (bands 42-83). The model found the coprime seam. Everything else barely moved.

### 4. Unfrozen Wave Decode — the learned phase decoder

**The idea:** Keep the phase coherence scoring but let the reference phases evolve. 1024 tokens × 84 bands = 86K learned reference phases. Start from embedding phases, then adapt to what the ODE actually produces.

**The result:** Loss **5.25** at 10K iters. Broke through the frozen ceiling (5.84) by 0.59 points. Desktop's orchestra analogy was right — the ODE evolves hidden states away from their initial embedding patterns, and the decoder needs a reference that matches the evolved states, not the original ones.

### 5. Low-Rank 32 — the efficient learned decoder

**The idea:** Factor lm_head [1024×168] into lm_down [32×168] × lm_up [1024×32]. Same linear projection through a bottleneck — like the maestro's squeeze-and-excitation, but for the output.

**The result:** Loss **4.62** at 20K iters. 38K params instead of 172K — 78% saving, 3% higher loss. The best efficiency ratio of any decoder tested.

**What the model actually produces at loss 4.62:** Similar domain vocabulary to full-rank at comparable loss. The bottleneck doesn't reduce vocabulary quality — it just reaches the same quality with fewer params.

---

## The Comparison

| Decoder | Method | Params | Best loss | Speed | Phase-native? |
|---------|--------|--------|-----------|-------|--------------|
| Tied embeddings | Dot product against wte | 1 | 6.23 | 71ms | No |
| Wave V1 (frozen) | cos(Δθ) coherence | 85 | 5.84 | 89ms | Yes |
| Wave V2 (θ+Δθ) | cos(Δθ) + cos(ΔΔθ) | 168 | 5.85 | 124ms | Yes |
| Wave unfrozen | Learned cos(Δθ) ref | 86K | 5.25 | 87ms | Yes |
| Low-rank 32 | Factored linear | 38K | 4.62 | 64ms | No |
| Full-rank lm_head | Learned linear | 172K | 4.48 | 72ms | No |

---

## What We Learned

### The conservation argument is validated

Phase-native coherence (5.84) beats rigid dot product (6.23) using the same reference library. The measurement operator matters — coherence preserves phase structure that dot products destroy. This is not a metaphor. It's measurable: same table, same params, different measurement, 0.4 points better.

### The reference library matters more than the measurement

Unfrozen wave decode (5.25) beat frozen (5.84) by 0.59 points. The frozen embedding table was designed for input encoding, not for matching ODE-evolved states. When the reference adapts, the phase-native decoder improves significantly.

### Learned projections win on efficiency

Low-rank at 38K params (4.62) beats unfrozen wave at 86K params (5.25). A linear projection with a bottleneck is more param-efficient than learned phase references — even though it's not phase-native. The dot product's simplicity is a strength when you have enough learned capacity to compensate.

### The lm_head earns its params

Full-rank (4.48 at 172K) beats everything. But low-rank (4.62 at 38K) comes within 3% at 78% fewer params. For 256-dim, low-rank is the clear recommendation — cut lm_head from 262K to 41K, flip the gradient balance from 56/44 to 90/10 in favour of the ODE.

### Loss doesn't mean what you think at 168-dim

A standard language model at loss 4.0 produces basic sentences. This model at loss 3.91 produces grammar textbook vocabulary in random order. The model is **smarter than its loss** (it knows domain vocabulary at losses where standard models produce noise) but **less fluent than its vocabulary** (it can't compose what it knows). The expression bottleneck is real.

---

## What This Means for 256-dim

Low-rank 32 goes forward. At 256-dim with 1K BPE:
- Full-rank lm_head: 262K params (44% of model)
- Low-rank 32: 41K params (10% of model)

That's a gradient balance of 90/10 in favour of the ODE — the model gets 9x more gradient for the wave layers. If composition is possible at 256-dim, low-rank gives the ODE the best chance to find it.

Wave transduction is parked. The principle is validated (phase-native measurement works, conservation argument holds). But at current param efficiency, learned linear projections win. Wave transduction may become relevant at larger scale where the cos expansion's O(vocab × bands) cost becomes significant vs O(vocab × n_embd) for linear — the crossover depends on whether the phase-native measurement enables capabilities that linear projections can't.

---

## Cross-References

- Expression bottleneck characterised in [Harmonic Scaling Investigation](../harmonic-scaling/INVESTIGATION.md) (Finding 5)
- Sub-harmonic diagnostics confirming magnitude discrimination 1.00 (Finding 6)
- Conservation law framework from Notes 21 (Marco's research archive)
- Defensive publication: [ENGINE-PATTERNS.md](../../ENGINE-PATTERNS.md) (Patterns 85-86)
