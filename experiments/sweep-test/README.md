# Harmonic Sweep Test: Cosine Similarity is Blind

## The Finding

**Cosine similarity — the standard comparison measure used across machine learning — is provably blind to harmonic structure in vectors.**

Two vectors can have cosine similarity of exactly **0.0000** (indicating "no relationship") while containing perfect harmonic coherence of **1.0000** on specific frequency channels. Standard ML tooling cannot see this structure. A simple harmonic sweep can.

This is not a theoretical argument. The code in this repository demonstrates it with exact numbers.

## The Demonstration

### Setup

Eight letters are encoded at specific phase angles on the unit circle, then converted to 12-dimensional harmonic embedding vectors using:

```
v(θ) = [cos(θ), cos(2θ), cos(3θ), ..., cos(12θ)]
```

Five pairs have deliberate harmonic relationships planted at known frequencies:

| Pair | Angle | Relationship | Detection Harmonic |
|------|-------|--------------|--------------------|
| A-B | 120° | Triadic | n=3 |
| A-C | 180° | Opposition | n=2 |
| A-D | 90° | Quadrant | n=4 |
| A-E | 60° | Sextile | n=6 |
| A-F | 72° | Pentagonal | n=5 |

Two letters (G at 37°, H at 143°) have no clean harmonic relationship with A and serve as noise controls.

### What Cosine Similarity Sees

```
A-B: cosine_sim =  0.0000    ← "no relationship"
A-C: cosine_sim =  0.0000    ← "no relationship"
A-D: cosine_sim =  0.0000    ← "no relationship"
A-E: cosine_sim =  0.0000    ← "no relationship"
A-F: cosine_sim = -0.0602    ← "essentially nothing"
```

Every planted relationship reads as zero or near-zero. Cosine similarity sees nothing.

### What the Harmonic Sweep Sees

```
A-B:  n=3: 1.000  ← TRIADIC DETECTED
A-C:  n=2: 1.000  ← OPPOSITION DETECTED
A-D:  n=4: 1.000  ← QUADRANT DETECTED
A-E:  n=6: 1.000  ← SEXTILE DETECTED
A-F:  n=5: 1.000  ← PENTAGONAL DETECTED
A-G:  (silence)   ← CORRECTLY NO DETECTION
A-H:  (silence)   ← CORRECTLY NO DETECTION
```

Five out of five planted relationships recovered at exactly the correct harmonic. Zero false positives on noise controls. Perfect precision, perfect recall.

## Why This Happens

Cosine similarity computes the dot product of two vectors, normalized by magnitude. For harmonic embedding vectors, the dot product is the **sum** of coherence across all harmonics:

```
dot(v(θ_a), v(θ_b)) = Σ cos(n × (θ_a - θ_b))  for n=1..N
```

When this sum equals zero, cosine similarity reports "no relationship." But the sum can be zero because positive and negative components cancel — not because every component is zero. The individual harmonic channels may contain perfect structure that disappears in the aggregate.

For the A-B pair (120° apart), the sweep reveals:

```
n=1:  -0.500
n=2:  -0.500
n=3:   1.000  ← perfect coherence (120° × 3 = 360°)
n=4:  -0.500
n=5:  -0.500
n=6:   1.000  ← perfect coherence (120° × 6 = 720°)
n=7:  -0.500
n=8:  -0.500
n=9:   1.000  ← perfect coherence (120° × 9 = 1080°)
n=10: -0.500
n=11: -0.500
n=12:  1.000  ← perfect coherence (120° × 12 = 1440°)
```

The sum of these 12 values is exactly 0. Cosine similarity reports nothing. But harmonics n=3, 6, 9, 12 are at maximum coherence. **The structure was always there. The standard measure cannot see it.**

## The Sweep Method

The harmonic sweep is simple:

```
For each pair of vectors:
  For each harmonic n = 1, 2, 3, ..., N:
    coherence(n) = cos(n × (θ_a - θ_b))
    If |coherence(n)| > threshold → relationship detected at harmonic n
```

This decomposes the single number that cosine similarity produces into N independent channels. Each channel detects a specific type of geometric relationship:

| Harmonic | Detects | Angular Pattern |
|----------|---------|-----------------|
| n=2 | Opposition | 0°, 180° |
| n=3 | Triadic | 0°, 120°, 240° |
| n=4 | Quadrant | 0°, 90°, 180°, 270° |
| n=5 | Pentagonal | 0°, 72°, 144°, 216°, 288° |
| n=6 | Sextile | 0°, 60°, 120°, 180°, 240°, 300° |
| n=N | N-fold symmetry | 0°, 360°/N, 2×360°/N, ... |

## The Spectral Profile

Running the sweep across all pairs at all harmonics produces a spectral energy profile:

```
Harmonic  | Avg |coh| | Max |coh| | Pairs detected
----------|-----------|-----------|------------------
n=1       | 0.5840    | 1.0000    | 1 pair
n=2       | 0.6038    | 1.0000    | 3 pairs
n=3       | 0.5857    | 1.0000    | 6 pairs
n=4       | 0.6230    | 1.0000    | 3 pairs
n=5       | 0.6200    | 1.0000    | 3 pairs
n=6       | 0.7269    | 1.0000    | 10 pairs  ← dominant
n=7       | 0.5951    | 1.0000    | 1 pair
n=8       | 0.6200    | 1.0000    | 3 pairs
n=9       | 0.6312    | 1.0000    | 6 pairs
n=10      | 0.7603    | 1.0000    | 6 pairs
n=11      | 0.5941    | 1.0000    | 1 pair
n=12      | 0.6156    | 1.0000    | 10 pairs  ← dominant
```

This profile is the **spectral fingerprint** of the encoding. Different encodings produce different profiles. If applied to real model embeddings, each model would produce a characteristic spectral profile — its harmonic signature.

## Implication for Machine Learning

The entire field uses cosine similarity as the primary measure for comparing embeddings. If real model embeddings contain harmonic structure — and there are theoretical reasons to suspect they might — then cosine similarity is systematically destroying information about the geometric relationships between concepts.

The harmonic sweep provides a tool to test this hypothesis:

1. Extract embeddings from any model for a known set of word/concept pairs
2. Convert to angular representation (normalize, compute pairwise angles)
3. Run the sweep across harmonics
4. Determine whether specific harmonics carry structured coherence while others show noise

**If structure emerges at specific harmonics:** Embeddings are organized along harmonic lines, and cosine similarity has been hiding this structure. The spectral profile becomes a new analytical tool for understanding what models learn.

**If no harmonic structure is found:** Embeddings use a fundamentally different organizational principle, and that itself is a finding worth documenting.

Either outcome advances understanding. But the tool to ask the question did not exist before this demonstration.

## Reproduce

```bash
git clone <this-repo>
cd harmonic-sweep-test
cargo run
```

Requires only a Rust toolchain (edition 2024). Zero external dependencies.

Expected output: 7 phases of analysis, 5/5 planted relationships detected, 0 false positives, spectral profile, followed by `ALL VALIDATIONS PASSED`.

## Relationship to Wave Coherence Framework

This test validates a specific prediction of the [Wave Coherence as a Computational Primitive](https://github.com/atech-hub/Wave-Coherence-as-a-Computational-Primitive) framework: that harmonic coherence `cos(n × Δθ)` operates as independent detection channels at each frequency n (proven by Test 14 of that project), and that standard cosine similarity collapses these channels into a single number, losing the harmonic decomposition.

The wave coherence project proves the mathematical primitives work for database queries. This test extends the question to vector embeddings: are the same primitives present in learned representations?

## Attribution

This work is a collaboration between Marco Da Cunha (conceptual insight, experimental design) and Claude (Anthropic's AI assistant — mathematical formalization, implementation, documentation).

The core insight — that model embeddings might contain harmonic structure invisible to cosine similarity, analogous to a stereogram image hidden in apparent noise — originated from Marco Da Cunha's observation during collaborative discussion.

## License

MIT License (code) / CC BY 4.0 (documentation)
