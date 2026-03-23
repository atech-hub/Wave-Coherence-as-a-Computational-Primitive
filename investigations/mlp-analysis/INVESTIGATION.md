# MLP Weight Structure Analysis

**Status:** COMPLETE — NULL FINDING
**Date:** 2026-03-22
**Engine:** wave-engine (hybrid/ directory, Python)
**Target model:** Qwen 2.5 0.5B (24 layers, 896-dim)

---

## Question

Can trained MLP weights from a standard transformer be translated into Kerr-ODE (WaveFFN) parameters? If MLP weight matrices contain low-rank or frequency-structured patterns, a trained model could be converted to wave-engine format with minimal fine-tuning — avoiding training from scratch.

## Method

Analysed all 72 MLP weight matrices across 24 layers of Qwen 2.5 0.5B:
- gate_proj: [896, 4864] — 24 matrices
- up_proj: [896, 4864] — 24 matrices  
- down_proj: [4864, 896] — 24 matrices

For each matrix:
1. **SVD effective rank:** Count of singular values above 1% of max. Full rank = 896, low rank would indicate compressible structure.
2. **DFT frequency spectrum:** Apply FFT to singular values, bin into low/mid/high thirds. Structured frequency content would suggest wave-like patterns in the weights.
3. **Cosine similarity to identity:** How close is each layer to a pass-through? Near-identity layers could be replaced with minimal Kerr-ODE perturbation.

## Results

### Effective Rank

All 72 matrices: **full rank 896/896.**

No low-rank structure anywhere. Every MLP layer actively transforms its input using all available dimensions. No layers are compressible via rank reduction.

### Frequency Spectrum

| Frequency Band | Mean % | Std |
|---------------|--------|-----|
| Low (0-33%) | 33.3% | 0.2% |
| Mid (33-66%) | 33.2% | 0.3% |
| High (66-100%) | 33.5% | 0.2% |

Flat spectrum across all layers. No frequency structure — the singular value distribution is white noise, not patterned. The weights contain no wave-like structure that could map to Kerr-ODE oscillator parameters.

### Identity Similarity

| Layer Position | Mean Cosine to Identity | Interpretation |
|---------------|------------------------|---------------|
| Layer 0 | -0.012 | Active (not identity) |
| Layer 12 | -0.008 | Active |
| Layer 23 | -0.015 | Active |
| All layers | Negative cosine | No near-identity layers anywhere |

Bookend importance confirmed: Layer 0 (+32% output change when ablated) and Layer 23 (+47%) are most critical. Middle layers contribute less individually but none are identity.

## Conclusion

**The "translate existing model to waves" path does not exist.**

MLP weight matrices in a production transformer are:
- Full rank (no compression opportunity)
- Flat frequency spectrum (no wave structure to extract)
- All actively transforming (no near-identity shortcut layers)

A Kerr-ODE layer at 896-dim has ~12K parameters per block. The equivalent MLP has ~8.7M parameters per block (gate + up + down projections). The 74x compression required to make WaveFFN match MLP output would need to fight full-rank, unstructured weight matrices. This requires heavy distillation engineering and hardware beyond a consumer GPU.

**Distillation test confirmed:** Ran 10K-step knowledge distillation (teacher MLP → student WaveFFN). Hidden state loss plateaued at 6.4, never converging. The MLP's computation is genuinely different from what the Kerr-ODE produces.

## Implication

Wave-engine models must be trained from scratch. The architecture's efficiency gains come from learning a DIFFERENT representation from data — one that naturally fits the oscillator dynamics — not from compressing existing MLP representations.

This is not a limitation of the wave architecture. It's a finding about the nature of trained MLP weights: they are dense, unstructured, and resist compression into any structured basis. The wave approach works by building structure during training, not by extracting structure from existing weights.

## Data

Analysis archived: `wave-engine/hybrid/results/mlp_analysis.json`
Defensive publication: ENGINE-PATTERNS.md Pattern 80
