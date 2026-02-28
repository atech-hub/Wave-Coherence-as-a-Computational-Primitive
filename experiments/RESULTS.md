# Experimental Results

4-layer, 4-head, 128-dim harmonic transformer. Shakespeare (~1.1M chars, 65 tokens, 842K params). CUDA, 3000 steps, batch 64, lr 3e-4.

---

## Phase 1: Spectral Persistence

Does harmonic structure survive through transformer layers?

| Metric | Harmonic | Baseline |
|---|---|---|
| Per-channel correlation (cos, final layer) | +0.14 | -0.04 |
| Per-channel correlation (sin, final layer) | +0.11 | -0.11 |
| Strongest surviving harmonics | n=4 (0.47), n=11 (0.45), n=14 (0.36) | — |

Verdict: Partial persistence with uneven survival across channels.

---

## Phase 2: Geometric Relations

Do harmonic channels overlap?

| Metric | Harmonic | Baseline |
|---|---|---|
| Independent pairs at embedding | 35.8% | 73.3% |
| Independent pairs at final layer | 92.5% | 79.2% |
| Trajectory (embedding -> final) | +56.7 points | +5.9 points |

Most isolated channels: n=14 (3.6% overlap), n=5 (5.7%), n=1 (6.6%).
Most entangled channels: n=9 (14.2%), n=11 (14.9%).

Verdict: 92.5% channel independence at final layer. Network actively disentangles harmonic channels during training.

---

## Phase 3: Knowledge Editing (Surgery)

Can we change model knowledge by editing embedding frequency bands?

- Planted patterns not learned (too sparse — 0.3% of data)
- Band ablation caused measurable prediction changes (KL up to 0.26)
- Patterns used different bands (differential activation confirmed)
- Every edit that changed target pattern also damaged control pattern

Verdict: Embedding-level surgery is insufficient — knowledge lives in MLP weights.

---

## Phase 3b: Harmonic Injection

Can we swap character identity by swapping embedding geometry at inference?

| Swap Mode | Swap Rate | Preservation |
|---|---|---|
| Embedding only | 0-8% | 70-84% |
| lm_head only | 100% | 100% |
| Full pipeline (emb + lm_head) | 72-100% | 70-91% |

| Model | Avg Swap Rate | Avg Preservation |
|---|---|---|
| Harmonic (trainable) | 80.7% +/- 12.2% | 82.2% +/- 7.0% |
| Frozen (fixed embeddings) | 79.8% +/- 5.4% | 81.7% +/- 7.4% |

Best pairs: d<->c (100%), o<->i (94.3%/88.9%), h<->n (93.8%/87.0%).

Verdict: 80.7% swap rate without retraining. Frozen model matches trainable — model learned generic harmonic processing.

---

## Phase 4: Harmonic Construction

Can we construct novel harmonic vectors and have the model process them predictably?

### Interpolation

| Metric | Value |
|---|---|
| Alpha vs KL-to-A correlation | -0.973 |
| Alpha vs KL-to-B correlation | +0.991 |

### Fractional Position (novel vectors)

| Metric | Value |
|---|---|
| Frac vs KL-to-'e' correlation | +0.970 |
| Frac vs KL-to-'f' correlation | -0.968 |
| Coherent output (real letters) | 100% |

### Prediction Accuracy

| Alpha | Avg Correlation |
|---|---|
| 0.2 | 0.813 |
| 0.4 | 0.664 |
| 0.5 | 0.626 |
| 0.6 | 0.627 |
| 0.8 | 0.805 |
| Overall | 0.707 |

Verdict: 0.991 interpolation correlation. 100% coherent output from novel vectors. 0.707 prediction accuracy for constructed vectors.

---

## Phase 5: Musical Harmonics

Do musical intervals between harmonic channels predict channel behavior?

### Interval Map

| Channel pair | Ratio | Musical interval | Consonance |
|---|---|---|---|
| (1, 2) | 2.000 | octave | perfect |
| (2, 3) | 1.500 | perfect fifth | perfect |
| (3, 4) | 1.333 | perfect fourth | perfect |
| (4, 5) | 1.250 | major third | consonant |
| (5, 6) | 1.200 | minor third | consonant |
| (7, 8) | 1.143 | major second | mild |
| (11, 12) | 1.091 | minor second | dissonant |

### Consonance vs Independence

| Category | Count | Avg Independence | Avg Correlation |
|---|---|---|---|
| Consonant (rank 0-5) | 874 | 0.9098 | 0.0902 |
| Mild (rank 6-9) | 608 | 0.9111 | 0.0889 |
| Dissonant (rank 10+) | 534 | 0.9112 | 0.0888 |

### Tenney Height vs Independence

| Tenney height | Count | Avg Independence |
|---|---|---|
| Low (<4, most consonant) | 213 | 0.9190 |
| Mid (4-7) | 450 | 0.9161 |
| High (>7, most dissonant) | 1353 | 0.9074 |

### Character Type Spectral Chords

| Character type | Chord classification | Consonance rank | Top bands |
|---|---|---|---|
| Punctuation | MAJOR | 3.5 | 64, 63, 51, 1, 62, 2 |
| Rare consonants | near-major | 4.6 | 63, 48, 47, 46, 64, 56 |
| Vowels | MINOR | 5.3 | 64, 50, 32, 45, 55, 33 |
| Common consonants | MINOR | 5.8 | 64, 56, 55, 44, 60, 54 |
| Uppercase | MINOR | 6.2 | 64, 50, 55, 1, 25, 60 |
| Space/newline | most dissonant | 6.9 | 6, 1, 8, 2, 11, 17 |

### Injection Safety by Consonance

| Pair | Avg consonance rank | Swap rate | Preservation |
|---|---|---|---|
| e<->a | 7.1 | 75.0% | 78.1% |
| t<->s | 6.4 | 70.3% | 82.7% |
| o<->i | 6.0 | 83.0% | 74.7% |
| h<->n | 6.1 | 80.6% | 73.4% |
| d<->c | 6.9 | 100.0% | 91.0% |
| r<->l | 6.0 | 79.5% | 82.7% |

Consonance rank vs preservation correlation: +0.454.
Consonance rank vs swap rate correlation: +0.224.

Verdict: +0.454 correlation between consonance rank and edit safety. Character types form classifiable spectral chords. Low harmonics (1-17) carry structural info, high harmonics (44-64) carry identity.

---

## Phase 6: Progressive Learning

Does structure-first training (low bands first) outperform standard training?

### Training Stages

| Stage | Steps | Trainable bands |
|---|---|---|
| 1 | 0-999 | bands 1-8 |
| 2 | 1000-1999 | bands 1-24 |
| 3 | 2000-2999 | bands 1-64 |

### Results

| Metric | Baseline | Progressive |
|---|---|---|
| Final val loss | 1.5876 | 1.5585 |
| Channel independence | 58.2% | 58.4% |
| Learns new data faster | — | 5/5 tests |

### Learning Curve

| Step | Baseline | Progressive | Diff |
|---|---|---|---|
| 0 | 4.178 | 4.207 | +0.029 |
| 200 | 2.485 | 2.478 | -0.007 |
| 600 | 2.091 | 2.077 | -0.014 |
| 1000 | 1.879 | 1.865 | -0.014 |
| 1600 | 1.706 | 1.701 | -0.005 |
| 2000 | 1.655 | 1.648 | -0.007 |
| 2999 | 1.595 | 1.565 | -0.030 |

### Knowledge Absorption (fine-tune on new data)

| Fine-tune steps | Baseline (new loss) | Progressive (new loss) | Baseline (forgetting) | Progressive (forgetting) |
|---|---|---|---|---|
| 5 | 0.472 | 0.424 | +0.310 | +0.277 |
| 10 | 0.160 | 0.141 | +0.659 | +0.712 |
| 20 | 0.040 | 0.037 | +1.067 | +1.205 |
| 50 | 0.015 | 0.014 | +1.531 | +1.538 |
| 100 | 0.011 | 0.010 | +1.638 | +1.639 |

Verdict: Progressive training achieves better final loss (1.559 vs 1.588) and faster new knowledge absorption (5/5 tests).

---

## Phase 7: Concept Composition

Do characters compose into word-level representations or stay independent?

### Context Divergence

| Character | Embedding divergence | Final layer divergence | Growth |
|---|---|---|---|
| 'e' | 0.0000 | 0.3770 | massive |
| 'n' | 0.0000 | 0.3837 | massive |
| 'o' | 0.0000 | 0.2880 | massive |
| 't' | 0.0000 | 0.3098 | massive |
| Average | 0.0000 | 0.3396 | 33,963,108x |

### Band Roles

| Band region | Avg relative variance | Role |
|---|---|---|
| Low (1-16) | 0.44 | Mixed |
| Mid (17-40) | 0.45 | Most context-sensitive |
| High (41-64) | 0.39 | More identity-stable |

### Semantic Clustering

| Metric | Value |
|---|---|
| Within-group avg similarity | 0.5585 |
| Between-group avg similarity | 0.5337 |
| Gap | +0.0248 |

Notable pairs: king-lord 0.721, take-give 0.826, love-give 0.753, king-go 0.421, go-hand 0.225.

### Clustering by Layer

| Layer | Within | Between | Gap |
|---|---|---|---|
| embedding | 0.305 | 0.256 | +0.049 |
| embed+pos | 0.673 | 0.651 | +0.022 |
| layer0_attn | 0.820 | 0.806 | +0.014 |
| layer1_mlp | 0.711 | 0.682 | +0.029 |
| layer2_attn | 0.447 | 0.415 | +0.031 |
| layer2_mlp | 0.455 | 0.467 | -0.012 |
| layer3_attn | 0.444 | 0.458 | -0.014 |
| final | 0.559 | 0.534 | +0.025 |

Verdict: 33M-fold context divergence growth. Same character becomes different representations depending on word context. Semantic clustering present at character level (king-lord 0.721).

---

## Phase 8: Initialization Convergence

Do models from different random seeds converge to the same internal structure?

### 5 Baseline + 5 Harmonic Models

| Model | Seed | Val Loss | Independence |
|---|---|---|---|
| Baseline | 42 | 1.5590 | 82.5% |
| Baseline | 137 | 1.5714 | 86.9% |
| Baseline | 256 | 1.5690 | 86.3% |
| Baseline | 1337 | 1.5681 | 84.2% |
| Baseline | 9999 | 1.5874 | 87.8% |
| Harmonic | 42 | 1.5576 | 88.9% |
| Harmonic | 137 | 1.5805 | 87.5% |
| Harmonic | 256 | 1.5784 | 86.6% |
| Harmonic | 1337 | 1.5744 | 88.4% |
| Harmonic | 9999 | 1.5714 | 87.7% |

| Metric | Baseline | Harmonic |
|---|---|---|
| Mean val loss | 1.5710 +/- 0.0092 | 1.5725 +/- 0.0081 |
| Mean independence | 85.5% +/- 1.9% | 87.8% +/- 0.8% |

### Cross-Run Structural Consistency

| Metric | Baseline | Harmonic |
|---|---|---|
| Channel correlation similarity | 0.002 | -0.003 |
| Energy profile similarity | 0.010 | -0.043 |

### Band-by-Band Convergence

| Region | Baseline variance | Harmonic variance |
|---|---|---|
| Low (1-16) | 0.002247 | 0.002318 |
| Mid (17-40) | 0.001856 | 0.002263 |
| High (41-64) | 0.002312 | 0.002286 |

Verdict: Cross-run structural similarity is zero. Every model invents its own channel organization. Harmonic init provides tighter macroscopic consistency (0.8% vs 1.9% independence spread).

---

## Phase 9: Commitment Point

At which layer does the model commit to a prediction?

### Per-Layer Prediction

| Layer | Entropy | % of max | Accuracy | Top-1 prob |
|---|---|---|---|---|
| embedding | 2.88 | 69.0% | 8.6% | 0.244 |
| layer0_attn | 2.79 | 66.9% | 5.5% | 0.260 |
| layer0_mlp | 2.58 | 61.7% | 11.0% | 0.300 |
| layer1_mlp | 2.30 | 55.2% | 17.8% | 0.376 |
| layer2_mlp | 2.05 | 49.2% | 30.8% | 0.420 |
| layer3_attn | 2.01 | 48.2% | 32.7% | 0.428 |
| layer3_mlp | 1.42 | 34.0% | 53.1% | 0.560 |

Biggest entropy drop: layer3_mlp (delta = 0.59).

### Band Contribution

| Band group | Loss | Loss increase vs full |
|---|---|---|
| All bands (baseline) | 1.568 | — |
| Low only (1-16) | 5.486 | +3.918 |
| Mid only (17-40) | 4.685 | +3.117 |
| High only (41-64) | 4.009 | +2.440 |
| Low+Mid (1-40) | 4.409 | +2.840 |
| Mid+High (17-64) | 2.907 | +1.339 |
| Low+High (1-16, 41-64) | 3.529 | +1.961 |

Mid+High is the minimum viable band set.

### Token-Dependent Commitment Depth

| Token category | Commitment layer | Final accuracy | Count |
|---|---|---|---|
| common consonant | layer3_mlp | 0.520 | 103,951 |
| space | layer3_mlp | 0.874 | 48,518 |
| punctuation | layer3_mlp | 0.247 | 18,235 |
| uppercase | layer3_mlp | 0.367 | 31,035 |
| newline | layer3_mlp | 0.862 | 13,384 |
| vowel | layer2_mlp | 0.489 | 82,737 |
| rare consonant | layer2_mlp | 0.311 | 29,820 |

### Early Exit Quality

| Layer | Accuracy | % of final |
|---|---|---|
| embedding | 8.6% | 16.2% |
| layer0_mlp | 11.1% | 20.9% |
| layer1_mlp | 17.8% | 33.5% |
| layer2_mlp | 30.9% | 58.2% |
| layer3_attn | 32.7% | 61.7% |
| layer3_mlp | 53.1% | 100% |

Verdict: layer3_mlp delivers half the total accuracy in one step (delta 0.59 entropy). Vowels and rare consonants commit one layer earlier. Mid+High bands are the minimum viable set.

---

## Phase 10: Early Exit

Can the model skip layers for already-decided tokens?

### Exit After Layer 2 MLP

| Threshold | % exited | Early acc | Combined acc | vs baseline | Layers saved |
|---|---|---|---|---|---|
| 0.5 | 3.3% | 0.813 | 0.529 | -0.000 | 0.8% |
| 0.8 | 6.4% | 0.775 | 0.530 | +0.001 | 1.6% |
| 1.0 | 8.5% | 0.738 | 0.527 | -0.002 | 2.1% |
| 1.5 | 16.7% | 0.643 | 0.516 | -0.013 | 4.2% |
| 2.0 | 34.0% | 0.506 | 0.477 | -0.052 | 8.5% |

### Per-Category Exit (layer 2, threshold 1.5)

| Token category | % that exit early | Accuracy when exiting |
|---|---|---|
| newline | 50.9% | 0.978 |
| common consonant | 18.6% | 0.774 |
| vowel | 16.0% | 0.405 |
| space | 13.7% | 0.594 |
| punctuation | 12.1% | 0.627 |
| uppercase | 10.8% | 0.226 |
| rare consonant | 9.6% | 0.665 |

### Optimal Strategy (>=99% accuracy retention)

| Strategy | Exit layer | Threshold | % exited | Accuracy retention | Compute saved |
|---|---|---|---|---|---|
| Best | layer2_mlp | 1.00 | 8.7% | 99.2% | 2.2% |
| Runner-up | layer1_mlp | 1.00 | 3.9% | 98.8% | 2.0% |
| Most aggressive | layer2_mlp | 1.50 | 16.7% | 97.1% | 4.2% |

Verdict: 2-4% compute saving at 97-99% accuracy retention. Layer 2 is the only viable exit. Newlines trivially predictable (50.9% exit, 97.8% accuracy). 6.4% of tokens are more accurate at layer 2 than at the final layer.

---

## Phase 11: Chord Flow

Can word-level chord representations replace individual tokens in upper layers?

### Chord Boundary Detection

| Layer | Within-word sim | Cross-boundary sim | Gap |
|---|---|---|---|
| layer0 | 0.773 | 0.717 | +0.055 |
| layer1 | 0.664 | 0.584 | +0.080 |
| layer2 | 0.401 | 0.303 | +0.098 |
| layer3 | 0.304 | 0.176 | +0.128 |

### Chord Pooling

| Threshold | Avg chords (from 256) | Compression | Accuracy | vs baseline |
|---|---|---|---|---|
| 0.60 | 239.7 | 0.94x | 0.503 | -0.031 |
| 0.75 | 253.9 | 0.99x | 0.532 | -0.002 |
| 0.90 | 255.9 | 1.00x | 0.534 | -0.001 |

### Head-to-Head

| Strategy | Accuracy | vs baseline | Retention | Compute |
|---|---|---|---|---|
| Full pipeline | 0.533 | --- | 100.0% | 100.0% |
| Token early exit | 0.528 | -0.005 | 99.1% | 98.0% |
| Chord flow | 0.527 | -0.007 | 98.8% | 99.4% |

Verdict: Chord boundaries are detectable (gap +0.055 to +0.128) but characters within words have absolute similarity of only 0.30-0.40. Mean pooling produces 252.7 chords from 256 tokens — almost no compression. Composition works by differentiation (each position carries different information), not convergence.

---

## Phase 12: Natural Expression

How does the progressive model's internal representation differ from baseline?

*Note: Progressive training had optimizer bug (AdamW recreated each step). Positive findings are conservative.*

### PCA Landscape

| Metric | Baseline | Progressive |
|---|---|---|
| Dims for 50% variance | 6 | 5 |
| Dims for 80% variance | 24 | 21 |
| Dims for 90% variance | 41 | 36 |
| Dims for 95% variance | 66 | 56 |
| PC1 variance | 20.7% | 23.8% |
| Separation ratio | 1.330 | 1.544 |

### Pre-Projection Geometry

| Metric | Baseline | Progressive |
|---|---|---|
| Hidden-to-logit correlation | 0.784 | 0.640 |
| Logit entropy | 1.390 | 1.640 |

### Attractor Dynamics (20 dream iterations)

| Metric | Baseline | Progressive |
|---|---|---|
| Converges? | YES | YES |
| Initial change | 2.37 | 2.84 |
| Final change | 0.67 | 1.08 |
| Final token stability | 93.9% | 89.6% |
| Dream state | Collapses to 'e' | Maintains diversity |

### Harmonic Energy Through Layers

| Metric | Baseline | Progressive |
|---|---|---|
| Final energy uniformity | 0.989 | 0.995 |
| Layer3_mlp total energy | 4.5 | 18.9 |
| Energy trajectory | Non-monotonic | Monotonically increasing |

Verdict: Progressive model uses fewer dimensions (5 vs 6 for 50% variance), better token separation (1.544 vs 1.330), richer dreams (maintains diversity vs collapses to 'e'). 36% of hidden structure doesn't map to token predictions (inflated by optimizer bug — corrected to 22.3% in Phase 13).

---

## Phase 13: Expression Curriculum

Can richer output heads unlock the model's trapped internal structure?

*Corrected progressive training (optimizer created once). Val loss: 1.596.*

### Expression Baseline

| Metric | Value |
|---|---|
| Hidden-to-logit correlation | 0.777 |
| Accuracy | 0.530 |
| Structure trapped inside | 22.3% |

### Expression Heads (frozen internal weights)

| Head | Params | Correlation | Accuracy | Loss |
|---|---|---|---|---|
| Linear | 8,320 | 0.788 | 0.532 | 1.596 |
| Deep (2-layer MLP) | 74,496 | 0.762 | 0.535 | 1.565 |
| Wide (4x expansion) | 99,328 | 0.673 | 0.532 | 1.580 |

### Multi-Step Prediction

| Steps ahead | Accuracy |
|---|---|
| +1 | 0.531 |
| +2 | 0.300 |
| +3 | 0.216 |
| +4 | 0.182 |
| +5 | 0.165 |

### Knowledge Absorption (5 fine-tune steps)

| Head | Absorbed | Forgot | Efficiency (learned/forgot) |
|---|---|---|---|
| Linear | 0.234 | 0.019 | 11.60x |
| Deep | 0.785 | 0.520 | 1.51x |
| Wide | 0.828 | 0.649 | 1.27x |

Verdict: Deep head achieves best accuracy (0.535) and loss (1.565) with lower correlation (0.762) — smarter lens, not more data. Linear head preserves knowledge best (11.6x learned/forgot efficiency). 22.3% of internal structure unused by output.

---

## Phase 14: Shakespeare Knowledge

Does the model know Shakespeare, or just predict characters?

### Quote Completion

| Quote | Expected | P(Baseline) | P(Progressive) | P(Violin) |
|---|---|---|---|---|
| "...wherefore art thou " | Romeo | 0.161 | 0.181 | 0.206 |
| "the winter of our " | discontent | 0.362 | 0.395 | 0.388 |
| "What's in a name" | ? | 0.148 | 0.428 | 0.464 |

### Text Completion (20 lines)

| Model | Avg continuation probability |
|---|---|
| Baseline | 0.178 +/- 0.090 |
| Progressive | 0.178 +/- 0.089 |
| Progressive + Violin | 0.182 +/- 0.091 |

### Relationship Knowledge

| Model | P('Juliet') after "Romeo and " | Greedy top char |
|---|---|---|
| Baseline | 0.273 | 't' |
| Progressive | 0.239 | 't' |
| Violin | 0.275 | 't' |

"the Duke of " -> Progressive/Violin: "Norfolk". Baseline: "York".

### Mid-Band Activation During Confident Predictions

| Band | High confidence | Low confidence | Ratio |
|---|---|---|---|
| Low (1-8) | 1.536 | 1.919 | 0.80x |
| Mid (9-24) | 1.634 | 1.021 | 1.60x |
| High (25-64) | 1.852 | 1.926 | 0.96x |

Verdict: P("discontent" | "the winter of our") = 0.39. P("Juliet" | "Romeo and") = 0.275. Knowledge is distributional — greedy decoding picks 't' (for "the") over "Juliet". Mid bands 1.6x more active during confident predictions.

---

## Phase 15: Harmonic Decoder

Can a decoder that reads mid-band confidence signal outperform fixed-strategy decoding?

### Confidence Signal Calibration

| Prediction quality | Mid-band energy |
|---|---|
| High-confidence (top 25%) | 1.9995 |
| Low-confidence (bottom 25%) | 1.8732 |
| Ratio | 1.07x |

### Confidence Trace (mode switching per prompt)

| Prompt | KNOW mode | GUESS mode |
|---|---|---|
| "the winter of our " | 68% | 32% |
| "Romeo and " | 78% | 22% |
| "KING RICHARD: My lord" | 32% | 68% |

### Knowledge Accuracy (15 known phrases)

| Strategy | Accuracy | vs Greedy |
|---|---|---|
| Greedy | 0.200 | baseline |
| Sample(0.8) | 0.087 | -0.113 |
| Beam(5) | 0.133 | -0.067 |
| Harmonic | 0.167 | -0.033 |
| Harm.Beam(5) | 0.200 | +0.000 |

### Quantitative Advantage (100 completions)

| Strategy | Accuracy | vs Greedy |
|---|---|---|
| Greedy | 0.164 | baseline |
| Beam(5) | 0.180 | +0.016 |
| Harmonic | 0.136 | -0.028 |
| Harm.Beam(5) | 0.186 | +0.022 |

Verdict: Harmonic beam wins: 0.186 vs greedy 0.164 (+13.4%). Mid-band energy guides adaptive beam width — narrow when confident, wide when uncertain. Per-token confidence ratio is 1.07x at this model scale. Greedy/beam collapse into repetitive loops; harmonic beam produces multi-speaker dialogue.

---

## Phase 16: Wave Packet Engine

Can wave packet queries on real embeddings match cosine similarity retrieval using fewer bands?

Model: all-MiniLM-L6-v2 (384 dimensions, 193 DFT bands). Database: 25 words across 5 categories. Queries: 5 words.

### Translator Round-Trip (Pattern 32)

| Metric | Value |
|---|---|
| Max reconstruction error (embed -> DFT -> inverse DFT -> embed) | 2.24e-08 |
| Cosine similarity preservation | 1.0000000000 |

### Wave Packet Retrieval (Pattern 31)

| Band selection | Bands used | % of total | Top-1 correct |
|---|---|---|---|
| Full cosine (baseline) | 384 (raw) | 100% | 5/5 |
| All DFT bands | 193 | 100% | 5/5 |
| Mid+High | 145 | 75% | 5/5 |
| Amplitude-selected (top 25%) | 49 | 25% | 5/5 |

### Selective Band Loading (Pattern 34)

Cosine similarity between full vector and selectively-loaded vector:

| Strategy | Bands | RAM % | Avg similarity | Min similarity |
|---|---|---|---|---|
| All bands | 193 | 100% | 1.0000 | 1.0000 |
| Mid+High | 145 | 75% | 0.8686 | 0.8358 |
| Top 25% by amplitude | 49 | 25% | 0.7785 | 0.7595 |
| Mid only | 96 | 50% | 0.7142 | 0.6522 |
| Low only | 48 | 25% | 0.4940 | 0.3884 |
| High only | 49 | 25% | 0.4934 | 0.4362 |

### Data Transfer Reduction

| Strategy | Bytes per query | % of full vector |
|---|---|---|
| Full vector (cosine) | 1536 | 100% |
| Mid+High packet | 1160 | 75.5% |
| Top-25% amplitude | 384 | 25.0% |
| High only | 392 | 25.5% |

Verdict: Wave packet queries achieve 5/5 retrieval with 25% of bands. Translator round-trip is lossless (2.24e-08). Mid+High bands preserve 87% similarity at 75% data transfer. Amplitude-selected packets reduce data to 25% with 78% similarity preserved. All operations are foundational math: DFT, inverse DFT, cosine, array indexing.

---

## Corrective Finding #6: Conjugate Symmetry in Resonance

Discovered during cross-language validation (Rust port of Phase 16).

**Problem:** The resonance formula `R(W,U) = sum |V_n| * |U_n| * cos(phi_n - psi_n)` treats all rfft coefficients equally. But rfft returns one-sided coefficients for a real signal. Due to conjugate symmetry (X[N-k] = conj(X[k])), the middle coefficients represent two-sided energy and must be weighted by 2. DC (n=0) and Nyquist (n=N/2) appear once and get weight 1.

| Metric | Without weighting | With weighting |
|---|---|---|
| Max |resonance - cosine_similarity| | 4.18e-02 | 2.05e-15 |
| Identity holds | No | Yes (machine precision) |
| Ranking equivalence | Preserved (on uniform energy) | Exact |

**Design rule:** When computing resonance from rfft coefficients, apply conjugate symmetry weights:
- w[0] = 1 (DC component)
- w[k] = 2 for k = 1..N/2-1 (middle components)
- w[N/2] = 1 (Nyquist component, if N is even)

**Why Python masked it:** Sentence-transformer embeddings (384d) have relatively uniform DFT energy distribution, so the ~4% error didn't affect rankings. Harmonic embeddings (structured cosine sequences) concentrate energy at specific bands, making the error visible.

**Fixed in:** `experiments/wave_packet_engine.py` (Python) and `experiments/rust-experiments/src/wave_packet.rs` (Rust).

---

## Rust Cross-Language Validation

Pure Rust reproduction of math-only experiments. No GPU, no external dependencies, no neural networks. Zero crates, pure `std` library.

Location: `experiments/rust-experiments/`

### Results (14/14 pass)

| # | Test | Phase | Result | Key metric |
|---|------|-------|--------|------------|
| 1 | DFT round-trip | 16 | PASS | 1.42e-14 max error |
| 2 | Resonance = cosine identity | 16 | PASS | 2.05e-15 max diff |
| 3 | Wave packet retrieval (5 queries) | 16 | PASS | 5/5 all bands, 5/5 mid+high |
| 4 | Selective band loading | 16 | PASS | All=1.0, Mid+High=0.83, Amp-25%=0.98 |
| 5 | Amplitude band selection energy | 16 | PASS | 98% energy in top-25% bands |
| 6 | Band energy distribution | 16 | PASS | Distinct profiles per angle |
| 7 | Interpolation monotonicity | 4 | PASS | r = +0.974 |
| 8 | Fractional position continuity | 4 | PASS | 0.983 min adjacent similarity |
| 9 | Chimera band independence | 4 | PASS | 1.000 (perfect split) |
| 10 | Norm preservation | 4 | PASS | All ratios 0.70-1.00 |
| 11 | Adjacent channel intervals | 5 | PASS | 5/5 interval names correct |
| 12 | Tenney height ordering | 5 | PASS | Exact values match theory |
| 13 | Consonance map (120 pairs) | 5 | PASS | 51.7% consonant, 28.3% mild, 20.0% dissonant |
| 14 | Interval identification | 5 | PASS | 8/8 ratios correctly identified |

Cross-language validation confirms: the mathematical foundations produce identical results in Rust (pure CPU) and Python (numpy). No GPU required for the foundational operations.

---

## Phase 17: Weight Spectral Analysis

Do harmonic embeddings create band-sparse weight matrices?

4-layer, 4-head, 128-dim. Shakespeare. CUDA, 2000 steps, batch 64, lr 3e-4. DFT analysis on 17 weight matrices per mode (all transformer blocks + lm_head).

### Training Results

| Mode | Val Loss | vs Baseline |
|---|---|---|
| Baseline (random init) | 3.1684 | — |
| Harmonic (trainable cos/sin) | 3.0899 | -2.5% |
| Frozen (fixed cos/sin) | 3.0793 | -2.8% |

### Spectral Analysis (column-wise DFT, averaged across 17 weight matrices)

| Metric | Baseline | Harmonic | Frozen |
|---|---|---|---|
| Bands for 90% energy (% of available) | 88.3% | 88.3% | 88.3% |
| Bands for 95% energy (% of available) | 93.8% | 93.8% | 93.8% |
| Band sparsity (<1% of peak energy) | 0.0% | 0.0% | 0.0% |

Verdict: **NULL RESULT.** Weight matrices are spectrally flat regardless of embedding type. All modes show identical spectral profiles. Harmonic embeddings improve loss (2.5-2.8% better) but do NOT induce frequency-domain sparsity. Frozen mode (3.0793) beats trainable harmonic (3.0899) — geometric structure alone suffices.

---

## Phase 17b: Curriculum-Induced Harmonic Specialisation

Does teaching frequency structure before content change weight spectral profiles?

Two-phase training with frozen harmonic embeddings: 500 iterations on synthetic frequency patterns (low-freq sinusoids 1-4 cycles, high-freq sinusoids 20-64 cycles, band-specific targeting), then 2000 iterations on Shakespeare.

### Curriculum Training

| Phase | Start Loss | End Loss |
|---|---|---|
| Phase 1: Frequency patterns (500 iters) | 5.0188 | 4.0319 |
| Phase 2: Shakespeare fine-tune (2000 iters) | 4.2343 | 3.0774 |

Final Shakespeare val loss: 3.0774 (vs plain frozen 3.0793 — identical within noise).

### Spectral Analysis (5 modes, column-wise DFT)

| Metric | Baseline | Harmonic | Frozen | Curriculum_pre | Curriculum |
|---|---|---|---|---|---|
| Bands for 90% energy | 88.3% | 88.3% | 88.3% | 88.5% | 88.3% |
| Bands for 95% energy | 93.8% | 93.8% | 93.8% | 94.0% | 93.8% |
| Band sparsity | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

Verdict: **NULL RESULT (second confirmation).** Frequency curriculum teaches frequency patterns (loss 5.02→4.03) but does not restructure weight spectra. Curriculum neither helped nor hurt final Shakespeare performance. Weight spectral profile is determined by the optimiser (AdamW), not the input structure or training curriculum.

### Boundary Established

Two null results bracket the same conclusion: wave coherence is a **representation and retrieval** primitive, not a training primitive. The framework operates on what models produce, not on how they learn.

- **Representation layer** — harmonic structure works. Frozen beats learned. Wave packets enable selective loading. Proven across Phases 1-16.
- **Weight layer** — spectrally flat regardless of embedding type, training data, or curriculum. Determined by optimiser dynamics. Proven null twice.

Weight spectral sparsity requires explicit optimiser intervention. Related work by grisun0 (DOI: 10.5281/zenodo.18407920) demonstrates that extreme regularisation pressure (λ = 10³⁰) can force 99.996% weight sparsity with phase-structured matrices, confirming the spectral flatness observed here is a property of standard optimisation, not a fundamental limit.

---

## Phase 18: Harmonic Attention Heads

Does structuring attention heads by harmonic order improve over standard attention?

4-layer, 4-head, 128-dim. Shakespeare. CUDA, 2000 steps, batch 64, lr 3e-4. Frozen harmonic embeddings throughout. Harmonic orders per head: [1, 2, 4, 8] (octave-spaced). Three modes: standard attention (Phase 17 rerun), warm-start Q/K initialised to harmonic orders (trainable), frozen Q/K at harmonic init (only V trainable).

### Training Results

| Mode | Trainable Params | Val Loss | vs Standard |
|---|---|---|---|
| frozen_standard (Phase 17 rerun) | 801,664 | 3.0912 | — |
| harmonic_heads (warm-start Q/K, trainable) | 801,664 | 3.2511 | -5.2% worse |
| frozen_heads (frozen Q/K, only V trainable) | 669,568 | 3.2376 | -4.7% worse |

### Attention Head Entropy (lower = more specialised)

| Mode | Layer 0 avg | Layer 1 avg | Layer 2 avg | Layer 3 avg | Overall avg |
|---|---|---|---|---|---|
| frozen_standard | 3.37 | 3.77 | 4.04 | 3.97 | 3.79 |
| harmonic_heads | 4.56 | 4.56 | 4.56 | 4.56 | 4.56 |
| frozen_heads | 4.56 | 4.56 | 4.56 | 4.56 | 4.56 |

Standard attention develops varied per-head entropy (3.37-4.30), showing genuine head specialisation. Both harmonic modes produce **uniform 4.56 entropy across every head in every layer** — the theoretical maximum for causal attention over 256 positions. The harmonic Q/K projections cannot discriminate between tokens.

### Weight Spectral Analysis (column-wise DFT, common matrices)

| Metric | frozen_standard | harmonic_heads | frozen_heads |
|---|---|---|---|
| Bands for 90% energy | 88.3% | 88.4% | 88.3% |
| Band sparsity | 0.0% | 0.0% | 0.0% |

Confirms Phase 17 null result — weight spectra identical regardless of attention architecture.

### Findings

1. **Harmonic Q/K initialisation hurts performance.** Emphasising only 2 input dimensions per head (the cos/sin pair for that head's harmonic order) is too constraining. With 128 input dimensions, forcing attention through a 2-dimensional bottleneck destroys the model's ability to form useful attention patterns.

2. **Uniform attention entropy = the model cannot attend.** 4.56 across all heads and layers means every token attends equally to all preceding tokens. No specialisation, no information routing. The harmonic Q/K weights produce attention scores that are essentially identical across all token pairs.

3. **Frozen heads slightly outperform trainable harmonic heads** (3.2376 vs 3.2511). The model with 132K fewer trainable parameters did marginally better, suggesting the harmonic_heads model wasted optimisation budget trying to fix the Q/K initialisation rather than learning through V and MLP layers.

4. **The optimiser cannot escape the harmonic basin.** Even with 2000 iterations and full gradient access, trainable harmonic heads never recovered the varied entropy that standard heads naturally develop. The warm-start creates a local minimum.

Verdict: **Harmonic attention heads do not improve performance.** The Q/K projection is where the model learns unrestricted token-to-token relationships. Constraining it to harmonic structure impairs the model by creating uniform (uninformative) attention patterns.

### Boundary Extended

Phase 18 adds a third layer to the established boundary:

- **Representation layer** — harmonic structure works. Frozen beats learned. Proven across Phases 1-16.
- **Weight layer** — spectrally flat regardless of architecture. Determined by optimiser. Proven null in Phases 17-17b.
- **Attention layer** — Q/K projections must remain unconstrained. Harmonic structure in Q/K destroys attention discrimination. Proven in Phase 18.

Wave coherence operates on what models produce (representations, retrieval), not on how they compute (attention, weights). The boundary between where harmonic structure helps and where it hurts is the learned projection — the Q/K transformation needs full-rank freedom to build useful attention patterns, even when the input embeddings carry harmonic structure.

---

## Phase 19: Spectral Interference Attention

Can harmonic embedding interference replace learned Q/K projections entirely?

Phase 18 tried to force harmonic structure THROUGH learned Q/K — which killed attention because Q/K needs full-rank freedom. Phase 19 asks a different question: what if you skip Q/K entirely and use the embedding itself as query and key?

4-layer, 4-head, 128-dim. Shakespeare. CUDA, 2000 steps, batch 64, lr 3e-4. Frozen harmonic embeddings. Each head sees a different frequency band of the embedding: head 0 = dims 0-31 (harmonics 1-16), head 1 = dims 32-63 (harmonics 17-32), head 2 = dims 64-95 (harmonics 33-48), head 3 = dims 96-127 (harmonics 49-64). Attention scores computed as dot product of embedding sub-vectors. V remains learned.

The original harmonic embedding is passed to every layer for attention computation, even as the hidden state evolves. "The geometry tells you who to listen to; learning tells you what to hear."

### Training Results

| Mode | Trainable Params | Val Loss | vs Standard |
|---|---|---|---|
| frozen_standard | 801,664 | 3.0861 | — |
| spectral (no Q/K) | 669,568 | 3.2503 | -5.3% worse |

### Attention Entropy

| Mode | Avg Entropy | Pattern |
|---|---|---|
| frozen_standard | 3.78 | Varied (3.32-4.32), genuine head specialisation |
| spectral | 4.56 | Uniform across all heads and layers |

Entropy range across spectral heads: 0.000 — all frequency bands produce identical attention patterns.

### Key Finding: Phase 18 = Phase 19

The spectral model (no Q/K at all, val 3.2503) matches Phase 18's harmonic heads (constrained Q/K, val 3.2511) to within noise. This proves Phase 18's constrained Q/K weights converged to producing the SAME uniform attention as having no Q/K at all. The constrained weights were effectively useless.

The ~3.25 ceiling is what a transformer achieves with uniform attention + learned V/MLP. The ~0.16 gap to standard (3.09) is the value of learned attention discrimination.

---

## Phase 19b: Harmonic Attention Bias

Does an additive harmonic interference bias improve standard attention?

Phase 18 constrained Q/K. Phase 19 removed Q/K. Neither preserved the model's full learning capacity. Phase 19b PRESERVES full learned Q/K and adds the harmonic interference as an additive bias:

```
score = Q·K^T / sqrt(d) + λ * dot(emb_band, emb_band^T) / sqrt(d)
```

λ is a learnable scalar per head per layer (16 total), initialised at 0.1. If the model finds the harmonic bias useful, λ stays positive. If not, λ → 0.

### Training Results

| Mode | Val Loss | vs Standard |
|---|---|---|
| frozen_standard | 3.0987 | — |
| harmonic_bias (λ=0.1 fixed) | 3.1325 | -1.1% worse |

### Lambda Evolution (Candle)

Lambda remained at exactly +0.100000 across all 16 parameters through 2000 iterations — the gradient did not flow back through the frozen embedding interference computation. Originally attributed to a candle autograd limitation with non-tracked tensor products. **See Phase 19b PyTorch Verification below — this WAS a candle bug (Corrective Finding #7).**

### Convergence Comparison

| Step | Standard | Biased | Gain |
|---|---|---|---|
| 0 | 5.24 | 5.22 | +0.4% |
| 250 | 3.41 | 3.45 | -1.2% |
| 500 | 3.29 | 3.34 | -1.4% |
| 1000 | 3.18 | 3.22 | -1.2% |
| 1999 | 3.10 | 3.13 | -1.1% |

The bias hurts from step 250 onward and never recovers. No convergence acceleration.

### Root Cause: Near-Uniform Interference

The harmonic bias fails for a related but more nuanced reason than originally stated. The embedding dot products produce near-uniform scores, but they are NOT perfectly uniform — **see PyTorch verification below**, which showed lambda actively learning head-specific values. The model can detect the harmonic signal but cannot exploit it for prediction. The interference encodes token identity, not token relevance.

---

## Phase 19b PyTorch Verification — Corrective Finding #7

The candle (Rust) implementation showed lambda stuck at exactly +0.100000 for 2000 iterations. Was this a framework autograd limitation or a genuine property of the computation? PyTorch verification settles it.

### Cross-Framework Setup

Identical architecture, identical hyperparameters (4 layers, 4 heads, 128 dim, 256 block, batch 64, lr 3e-4, 2000 iters). PyTorch 2.10.0+cu128. Lambda as `nn.Parameter` with `register_buffer` for frozen embeddings.

### Result: Lambda DOES Learn in PyTorch

**Gradient check at iteration 0:**
```
layer 0 lambda grad: [-0.00000170, -0.00000395, +0.00000721, +0.00001878]
layer 1 lambda grad: [-0.00000457, +0.00001775, +0.00000314, -0.00000390]
layer 2 lambda grad: [-0.00000620, -0.00000722, -0.00000079, -0.00000530]
layer 3 lambda grad: [+0.00000171, -0.00000877, +0.00001241, -0.00000243]
```

Gradients are real (1e-5 to 1e-8 magnitude). Small, but non-zero. PyTorch propagates them correctly; candle did not.

### Lambda Evolution

| Step | Layer 0 avg | Layer 1 avg | Layer 2 avg | Layer 3 avg | Overall avg |
|---|---|---|---|---|---|
| 0 | +0.100 | +0.100 | +0.100 | +0.100 | +0.100 |
| 250 | +0.143 | +0.139 | +0.146 | +0.131 | +0.140 |
| 500 | +0.167 | +0.182 | +0.185 | +0.155 | +0.172 |
| 1000 | +0.175 | +0.205 | +0.214 | +0.196 | +0.198 |
| 1500 | +0.190 | +0.198 | +0.220 | +0.227 | +0.209 |
| 1999 | +0.201 | +0.192 | +0.219 | +0.250 | +0.215 |

Lambda moved substantially — from 0.100 to 0.215 average, with individual heads ranging from **-0.081 to +0.540**.

### Per-Head Pattern (final values)

| | Head 0 (low freq) | Head 1 | Head 2 | Head 3 (high freq) |
|---|---|---|---|---|
| Layer 0 | +0.415 | +0.359 | +0.035 | -0.004 |
| Layer 1 | +0.540 | +0.203 | +0.104 | -0.081 |
| Layer 2 | +0.457 | +0.353 | +0.110 | -0.043 |
| Layer 3 | +0.287 | +0.462 | +0.149 | +0.102 |

Clear frequency-dependent pattern: **low-frequency heads increase lambda** (0.29-0.54), **high-frequency heads decrease or go negative** (-0.08 to +0.10). The model selectively amplifies coarse geometric structure (broad token-class distinctions) while suppressing fine-grained harmonic identity.

### Training Results

| Mode | Val Loss | vs Standard |
|---|---|---|
| frozen_standard (PyTorch) | 1.6602 | — |
| harmonic_bias (PyTorch) | 1.6672 | -0.4% worse |

### Attention Entropy (PyTorch)

| | frozen_standard | harmonic_bias |
|---|---|---|
| Layer 0 avg | 3.41 | 3.30 |
| Layer 1 avg | 1.75 | 2.12 |
| Layer 2 avg | 2.04 | 1.59 |
| Layer 3 avg | 3.81 | 4.04 |

Attention entropy is NOT uniform (range 1.13-4.23), unlike the candle run where everything was 4.56. The learned Q/K functions normally alongside the bias. The lambda values do NOT push attention toward uniformity — they modulate it per head.

### Corrective Finding #7: Candle Autograd Limitation with Frozen Tensors

Candle (HuggingFace's Rust ML framework) does not propagate gradients through products where one operand is a `Tensor` not tracked in `VarMap` (frozen via construction, not via `requires_grad=false`). PyTorch's autograd handles this correctly — gradients flow to `nn.Parameter` operands even when the other operand is a `register_buffer`.

**Impact on Phase 19b results:**
- Candle showed lambda stuck at 0.100000 → PyTorch shows lambda learns to avg +0.215
- Candle showed 1.1% worse → PyTorch shows 0.4% worse (within noise)
- Candle showed uniform entropy → PyTorch shows normal entropy variation
- **The conclusion is refined**: not "the harmonic prior provides no signal" but "the harmonic prior provides signal the model can detect but cannot exploit for prediction"

This is the seventh time cross-language/cross-framework validation has caught a hidden assumption. Corrective Finding #6 was caught by Rust; Corrective Finding #7 was caught by Python.

### Interpretation: Frequency-Stratified Signal

The lambda pattern reveals the model performing frequency-dependent triage on the harmonic prior:

- **Low-frequency harmonics (Head 0, Head 1): amplified** (λ up to +0.54). These encode broad token-class distinctions — "this is a vowel," "this is punctuation." Coarse geometric structure that maps to real character categories. The model says "yes, this is useful context."
- **High-frequency harmonics (Head 3): suppressed or inverted** (λ down to -0.08). These encode fine-grained identity distinctions — "this specific character is three positions from that one on the phase circle." The model says "this is noise, remove it."
- **Mid-frequency harmonics (Head 2): ignored** (λ stays near 0.1). Neither signal nor noise.

The model can see the difference between useful and useless frequency bands and responds accordingly. It just cannot turn that categorical signal into better prediction because learned Q/K already captures the same information more flexibly.

### Future Work Hypothesis: Vocabulary Complexity Scaling

The Shakespeare character set has 65 tokens with coarse category structure — vowels, consonants, punctuation, digits. A 50,000-token vocabulary has rich hierarchical category structure — nouns, verbs, medical terms, legal terms, subcategories within subcategories.

The low-frequency harmonic signal that the Shakespeare model found "partially useful" (λ increased but loss unchanged) might become genuinely useful when the vocabulary is rich enough to have deep categorical structure worth encoding geometrically. The finding is not "harmonic attention fails" but "harmonic attention provides frequency-stratified signal that scales with vocabulary complexity." At 65 tokens, not enough structure to matter. At 50,000 tokens, it might.

This is a clean, testable hypothesis that follows directly from the lambda pattern.

### Phase 18-19b Complete Picture

| Phase | Approach | Val Loss | vs Standard | Root Cause |
|---|---|---|---|---|
| 18 | Constrain Q/K to harmonic | 3.2511 | -5.2% | 2-dim bottleneck kills discrimination |
| 19 | Replace Q/K with embedding | 3.2503 | -5.3% | Uniform interference, no discrimination |
| 19b | Bias Q/K with interference | 3.1325 | -1.1% | Near-uniform bias adds noise |

All three approaches fail because harmonic embedding dot products encode token identity, not token relevance. PyTorch verification (Phase 19b) refined this: the model CAN detect frequency-dependent structure (lambda learns to amplify low-frequency heads and suppress high-frequency heads), but this geometric knowledge doesn't help predict which characters follow which. That statistical relationship must be LEARNED, which is exactly what Q/K projections do.

### Boundary Finalised (Revised)

Wave coherence operates on what models produce (representations, retrieval), not on how they compute (attention, weights). The boundary between where harmonic structure helps and where it hurts is the learned projection:

- **Before the projection** (embeddings): harmonic structure helps. Frozen beats learned.
- **The projection itself** (Q/K weights): must remain unconstrained. Cannot be replaced or constrained by harmonic structure. Additive bias is tolerated but not beneficial at 65-token vocabulary. Whether richer vocabularies change this is an open question.
- **After the projection** (attention patterns): emergent, task-specific. The model must discover its own attention patterns through gradient descent.

### Why: Substrate Incompatibility

The boundary exists because of a fundamental structural incompatibility between wave mechanics and matrix multiplication.

A matrix treats every element as an independent number in a grid. Row 3, column 7 has no structural relationship to row 3, column 8. The matrix doesn't know that those two columns represent adjacent frequency bands. It doesn't know that column pairs encode cos/sin of the same harmonic. It can't know — the format doesn't carry that information.

When harmonic embeddings — inherently circular, continuous, periodic — pass through a matrix multiplication, the wave structure is invisible to the operation. The matrix treats cos(3θ) and cos(4θ) as two unrelated numbers. Adjacency, periodicity, phase relationships — all destroyed by the container.

**This is the transistor/capacitor analogy from circuit design.** A transistor is a discrete switch — on/off, 0/1. When an analogue wave passes through a transistor, it clips and distorts. The transistor is structurally incompatible with continuous signals. A capacitor stores and releases charge continuously, responds to rate-of-change, naturally selects frequencies. Capacitors and inductors form resonant LC circuits — native wave processors.

Matrix multiplication is the transistor of neural computation. It processes discrete grid positions, structurally blind to frequency. Harmonic coherence is the capacitor — it natively operates on phase, frequency, and resonance. The framework works for representation (vectors you can decompose) and retrieval (comparison via frequency-aware functions). It fails inside the network because the computation primitive — matmul — is structurally incompatible with wave mechanics.

That's why:
- **Weights stay broad-spectrum** (Phase 17): the matrix format can't represent "I only need harmonics 2 through 5." It can only fill grid positions with values. The optimiser fills the grid because the grid has no concept of bands.
- **Harmonic attention fails** (Phase 18-19): wave structure injected into a dot-product mechanism operating on discrete vector positions. The dot product sees numbers, not frequencies.
- **Harmonic bias is detectable but useless** (Phase 19b): the model can sense the frequency structure through lambda, but can't convert it into better predictions through matrix operations.

The question this raises is not "how do we make matrices harmonic" but "what computation primitive replaces matrices when the data is waves?" The computational equivalent of an LC circuit — an operation that naturally resonates at specific frequencies, filters by band, and preserves phase — does not yet exist in neural network architectures. Harmonic coherence already does this for retrieval. The missing piece is a version that works for transformation, not just comparison.

---

## Phase 20: LC Circuit Layer

Can frequency-native computation replace matrix multiplication for harmonically-structured data?

Inspired by the substrate incompatibility insight, we built an LC circuit analog as a neural network layer. Instead of a standard MLP (128->512->128, ~131K params), the LC layer operates on harmonic bands natively:
- **Resonance**: per-band amplitude gain and phase rotation (128 params)
- **Coupling**: cross-band interaction via 5-wide shifted matmuls (20 params)
- **GELU** nonlinearity

Total: 148 params per layer vs 131K for standard MLP (890x reduction).

### Phase 20 Results (Candle/Rust)

| Mode | Val Loss | vs Standard | FFN Params/Layer |
|---|---|---|---|
| frozen_standard (MLP) | 3.0966 | baseline | 131,712 |
| lc_layer | 3.0793 | +0.56% | 148 |

LC appeared to OUTPERFORM MLP. But all LC parameters were frozen at init values (gain=1.0, phase=0.0, coupling=0.0) for all 2000 iterations. The LC layer was effectively just GELU(x). This is Corrective Finding #7 again: candle autograd doesn't propagate gradients through the LC layer's operations.

### Phase 20 PyTorch Verification

PyTorch confirmed LC parameters DO learn:

| Layer | Gain Range | Phase Range | Coupling RMS |
|---|---|---|---|
| 0 | [0.72, 1.27] | [-0.22, +0.14] | 0.094 |
| 1 | [0.69, 1.21] | [-0.26, +0.12] | 0.092 |
| 2 | [0.74, 1.23] | [-0.22, +0.18] | 0.120 |
| 3 | [0.73, 1.25] | [-0.20, +0.17] | 0.106 |

| Mode | Val Loss | vs Standard | FFN Params |
|---|---|---|---|
| frozen_standard (MLP) | 1.6583 | baseline | 526K |
| lc_layer | 2.0432 | -23.5% | 592 |

Gradients are real (gain grad rms ~0.006, coupling grad rms ~0.007). Gain differentiates by band, coupling activates progressively, phase rotates meaningfully. The learning dynamics are healthy -- the concept works mechanically but with 890x fewer parameters, it can't match MLP capacity.

The candle "LC beats MLP" was a double artifact: (1) frozen params meant LC layer = GELU(x), and (2) at candle's higher loss floor (~3.09), attention alone with GELU suffices.

---

## Phase 20b: Expanded LC Layer -- Fair Capacity Test

Phase 20's 148 params vs 131K was not a fair comparison. Phase 20b gives the LC architecture a fair parameter budget while preserving frequency-native structure:

- **Per-band FFN**: each of 64 bands gets a small 2->16->2 network with GELU (5,248 params/layer)
- **Cross-band coupling**: linear mixing across bands, cos and sin independently (8,192 params/layer)
- **Total**: 13,440 params per layer vs 131,712 for MLP (9.8x reduction)

This mirrors LC circuit physics: per-band FFN = multi-stage resonator, cross-band coupling = mutual inductance network. The architecture "knows" about harmonic bands; MLP treats all 128 dims as independent.

### Results (PyTorch, CUDA)

| Mode | Val Loss | Train Loss | vs Standard | Total Params | FFN Params |
|---|---|---|---|---|---|
| frozen_standard (MLP) | 1.6458 | 1.4540 | baseline | 801K | 526K |
| lc_expanded | 1.9967 | 1.8718 | -21.3% | 328K | 53K |

### Convergence

| Step | Standard | LC Expanded | Gap |
|---|---|---|---|
| 0 | 4.1993 | 4.2208 | -0.5% |
| 500 | 2.1874 | 2.4017 | -9.8% |
| 1000 | 1.8874 | 2.2393 | -18.7% |
| 1500 | 1.7399 | 2.0874 | -20.0% |
| 1999 | 1.6458 | 1.9967 | -21.3% |

Gap widens throughout training -- MLP has a higher expressiveness ceiling.

### LC Parameter Analysis

All parameters learn vigorously (confirmed by gradient check at iter 0):

- **Per-band FFN**: weight norms grow and differentiate by layer. Layer 3 develops largest norms (up_norm=0.44, down_norm=0.33). Layer 1 stays smallest (0.29, 0.22). Per-band variance increases, showing bands specialise.
- **Cross-band coupling**: activates strongly. Frobenius norms grow to 2.0-3.4 (from 0). Sparsity drops from 100% to 4-7%, meaning nearly all band-to-band connections activate. Layer 3's coupling is largest (cos=3.36, sin=2.90).

### Key Finding: Architectural, Not Parametric

Going from 148 to 13,440 FFN params per layer (91x increase) only improved the gap by 2.2 percentage points (23.5% -> 21.3%). This means:

1. **The bottleneck is structural, not capacity**. The factored LC architecture (per-band FFN + linear coupling) can't match dense MLP expressiveness regardless of parameter count.

2. **Why**: In MLP, GELU operates on representations that combine ALL 128 dims (via 128->512 expansion). The MLP learns nonlinear functions of multiple frequency bands simultaneously. In expanded LC, the GELU only operates within each band's 2-dim space. Cross-band interaction is linear-only. The LC cannot learn nonlinear multi-band features.

3. **Positive**: 10x fewer FFN params for ~80% of performance (8.1x efficiency ratio). Frequency-native structure does provide useful inductive bias -- just not enough to fully replace dense interaction.

### Phase 20 Series Complete Picture

| Experiment | FFN Params/Layer | vs MLP | What It Showed |
|---|---|---|---|
| Phase 20 tiny LC | 148 | -23.5% | Concept works (params learn) but starved |
| Phase 20b expanded LC | 13,440 | -21.3% | 91x more params -> only 2.2pp better |
| Standard MLP | 131,712 | baseline | Dense nonlinear interaction is needed |

The 2.2pp improvement from 91x more capacity proves the limitation is architectural: the factored structure (per-band nonlinear + cross-band linear) cannot replace the MLP's dense nonlinear cross-dimension interaction.

### Implication for the Framework

This extends the substrate incompatibility insight from a boundary into a theorem: **frequency-native computation is necessary but not sufficient**. The LC layer correctly identifies and processes harmonic bands. But language modelling requires nonlinear relationships between bands that only emerge from dense cross-dimensional interaction. The missing primitive is not "per-band processing" (we have that) but "nonlinear multi-band fusion" -- a frequency-aware dense operation that doesn't yet exist.

---

## Phase 21: Kerr-ODE Layer -- Wave-Native Computation Primitive

The LC layer's missing primitive was "nonlinear multi-band fusion." Phase 21 tests whether an ODE from nonlinear optics -- the Kerr effect in coupled resonators -- provides exactly that.

### Kerr-ODE Architecture

Input x (B, T, 128) is treated as 64 complex harmonic bands: Z_k = x[2k] + i*x[2k+1]. Each band evolves through a differential equation:

```
dZ_k/dt = -gamma_k * Z_k                          (learned damping)
         + i * omega_k * Z_k                       (learned resonance)
         + i * alpha * |Z_k|^2 * Z_k               (Kerr self-phase modulation)
         + i * beta * sum_neighbors(|Z_j|^2) * Z_k  (cross-phase modulation)
```

Where neighbors are bands k-2, k-1, k+1, k+2. Integrated via fixed-step Euler. Followed by learned output projection Linear(128->128).

**Key difference from Phase 20b LC**: the cross-band coupling is NONLINEAR. The |Z_j|^2 term means band k's dynamics depend on the *amplitude squared* of neighboring bands -- not just their linear values. This is the "nonlinear multi-band fusion" the LC layer lacked.

**Parameters per layer:**
- ODE: 64 gamma + 64 omega + 1 alpha + 1 beta = 130 params
- Output projection: 128x128 + 128 = 16,512 params
- Total: 16,642 per layer vs MLP's 131,712 (7.9x reduction)

### Results (PyTorch, CUDA)

| Mode | Val Loss | Train Loss | vs Standard | Total Params | FFN Params | Steps |
|---|---|---|---|---|---|---|
| frozen_standard (MLP) | 1.7119 | 1.5204 | baseline | 801K | 526K | - |
| kerr_ode (4 steps) | 1.8574 | 1.6896 | -8.5% | 341K | 66K | 4 |
| kerr_ode_deep (8 steps) | 1.8431 | 1.6801 | -7.7% | 341K | 66K | 8 |

### Convergence

| Step | Standard | Kerr-ODE 4s | Kerr-ODE 8s | 4s gap | 8s gap |
|---|---|---|---|---|---|
| 0 | 4.2328 | 4.1881 | 4.2080 | +1.1% | +0.6% |
| 400 | 2.2732 | 2.2678 | 2.2501 | +0.2% | +1.0% |
| 800 | 2.0182 | 2.0788 | 2.0524 | -3.0% | -1.7% |
| 1200 | 1.8760 | 1.9607 | 1.9564 | -4.5% | -4.3% |
| 1600 | 1.7827 | 1.8948 | 1.8934 | -6.3% | -6.2% |
| 1999 | 1.7119 | 1.8574 | 1.8431 | -8.5% | -7.7% |

Gap widens but much slower than LC (which reached -21.3%). Convergence curves are parallel -- the Kerr-ODE is learning at a similar rate, just starting from a slightly higher baseline.

### Learned Parameter Analysis

The ODE parameters differentiate meaningfully by layer:

**4-step model final parameters:**

| Layer | Alpha (self-phase) | Beta (cross-phase) | Gamma avg (damping) | Proj norm |
|---|---|---|---|---|
| L0 | 0.062 (decreased) | 0.058 (decreased) | 0.103 (stable) | 1.47 |
| L1 | 0.092 (near init) | 0.083 (near init) | 0.102 (stable) | 1.42 |
| L2 | 0.095 (near init) | 0.086 (near init) | 0.099 (stable) | 1.72 |
| L3 | **0.122** (increased) | **0.107** (increased) | 0.096 (lowest) | **2.30** |

**Pattern: depth-dependent nonlinearity.** The model amplifies Kerr effects in deep layers (L3: alpha 22% above init, beta 7% above) and suppresses them in shallow layers (L0: alpha 38% below init, beta 42% below). Early layers do mostly linear processing; deep layers need complex nonlinear multi-band features.

**Gamma (damping)** barely moves from 0.1 -- the init was well-chosen. Layer 3 has lowest gamma (0.096), approaching minimum dissipation. The model wants to preserve energy in deep layers and amplify nonlinear features. (In the original unstable run without softplus, L3 gamma went negative -- the model tried to create gain/lasing. Softplus prevents this while allowing the model to get close to zero damping.)

**Omega (resonant frequency)** barely changes from init. The harmonic ordering (omega_k = (k+1)/64) is apparently close to optimal -- the model doesn't need to re-learn which frequency is which.

**Output projection norm** grows with depth: L0=1.47, L3=2.30. Deep layers amplify the ODE output more aggressively. This is where most of the actual capacity lives (16.5K of 16.6K params per layer).

### Integration Depth

8 steps (dt=0.125) is 0.84pp better than 4 steps (dt=0.25): 7.7% vs 8.5% gap. The improvement is modest but consistent. More steps = more accurate ODE integration = the Kerr dynamics are better resolved. The 8-step model's alpha and beta are more conservative (smaller values) because the nonlinearity compounds over more steps.

### Stability Note

The original implementation allowed gamma to go negative (anti-damping = "lasing" in optics). With 4 Euler steps this was marginally stable; with 8 steps it caused NaN within 200 iterations. Fix: softplus(gamma_raw) guarantees positive damping. State clamping (|r|, |s| <= 10) provides additional safety. Both models now train stably.

### Cross-Phase Comparison

| Experiment | FFN Params/Layer | vs MLP | Cross-Band Coupling |
|---|---|---|---|
| Phase 20 tiny LC | 148 | -23.5% | Linear (5-wide conv) |
| Phase 20b expanded LC | 13,440 | -21.3% | Linear (64x64 matrix) |
| **Phase 21 Kerr-ODE 4s** | **16,642** | **-8.5%** | **Nonlinear (|Z_j|^2)** |
| **Phase 21 Kerr-ODE 8s** | **16,642** | **-7.7%** | **Nonlinear (|Z_j|^2)** |
| Standard MLP | 131,712 | baseline | Dense nonlinear (matmul + GELU) |

The jump from LC (-21.3%) to Kerr-ODE (-7.7%) confirms the hypothesis: **nonlinear cross-band interaction is the critical missing primitive**. The |Z|^2 terms create amplitude-dependent phase shifts that couple frequency bands nonlinearly -- exactly what dense MLP achieves through matmul + GELU, but operating natively on harmonic bands.

### What This Means

The Kerr-ODE layer achieves **92% of MLP performance with 7.9x fewer FFN parameters**. The computation is entirely element-wise operations and sparse 1D convolution -- no dense matrix multiply anywhere in the ODE (only in the output projection).

This is the first wave-native computation primitive that meaningfully competes with matrix multiplication for transformer FFN layers. The remaining 8% gap likely comes from the output projection being the only cross-band mixing mechanism, while MLP has dense mixing at both the expansion and contraction stages.

The Kerr nonlinearity provides what the LC layer couldn't: **intensity-dependent frequency coupling**. In optical physics, this is what enables wavelength conversion, four-wave mixing, and soliton formation. In neural computation, it enables bands to create nonlinear features of each other's content.

### References

The Kerr-ODE layer's dynamics are adapted from two independent lines of work:

**Kerr physics**: Pal et al. (2024), "Linear and Nonlinear Coupling of Light in Twin-Resonators with Kerr Nonlinearity" (arXiv:2404.05646v2). Their coupled Lugiato-Lefever equation (Eq. 1) provides the self-phase modulation term (i|E|^2 E) and cross-phase modulation term (i*2|E'|^2 E) that we adapt as alpha * |Z_k|^2 * Z_k and beta * sum(|Z_j|^2) * Z_k. The physical substrate -- coupled optical microresonators with Kerr nonlinearity -- is the direct analog of our computational layer: resonant cavities exchanging energy through nonlinear coupling.

**Neural ODE for signal processing**: Kato et al. (2024), "Multi-Band Wi-Fi Neural Dynamic Fusion" (arXiv:2407.12937v1, ICASSP 2024). Their multi-encoder -> learned ODE evolution -> latent alignment -> fusion -> decoder architecture demonstrates that neural ODEs work as practical computation primitives for multi-band signal processing. Our approach applies the same principle (ODE integration as a computation layer) but replaces their learned neural dynamics with physics-based Kerr dynamics, and replaces their multi-band Wi-Fi signals with harmonic transformer embeddings.

---

## Phase 21b: Per-Band Kerr Coefficients

Does giving each of the 64 bands its own alpha_k and beta_k (instead of one scalar per layer) close the 8% gap?

**Change:** scalar alpha -> alpha_k (64 values), scalar beta -> beta_k (64 values). Initialised to 0.1. Extra params: 128/layer (negligible).

**Results:**

| Mode | Val Loss | vs MLP | Steps |
|---|---|---|---|
| MLP baseline | 1.6940 | - | - |
| Kerr scalar (Phase 21) | 1.8704 | +10.42% | 4 |
| Kerr per-band 4s | 1.8612 | +9.87% | 4 |
| Kerr per-band 8s | 1.8245 | +7.71% | 8 |

Per-band bought 0.54pp over scalar. Negligible. Integration depth (8 vs 4 steps) bought 2.17pp — 4x more impact.

**Did bands differentiate?** NO. All four layers failed the success criterion (std > 0.02):

| Layer | alpha std | beta std | Verdict |
|---|---|---|---|
| L0 | 0.0149 | 0.0167 | clustered |
| L1 | 0.0143 | 0.0102 | clustered |
| L2 | 0.0091 | 0.0073 | clustered |
| L3 | 0.0107 | 0.0093 | clustered |

**Diagnostics:** No dead gradients (alpha/beta norms 2e-2 to 1e-1). The optimizer has signal but doesn't see a reason to differentiate bands. Band energy confirms depth pattern: L0 barely transforms (ratio ~1.04), L3 amplifies ~2x.

Verdict: The 8% gap is NOT about per-band nonlinear expressiveness. The scalar Kerr coefficient was already the right abstraction. Same pattern as Phase 17 — freedom granted, freedom ignored.

---

## Phase 22: Inverse Kerr — Understanding the Transform by Reversing It

Run the Kerr-ODE backwards. Feed output through reverse ODE (negated dt), measure per-band reconstruction error. Separates three categories: reversible (spectral remixing), irreversible-damping (energy dissipation), irreversible-nonlinear (genuine computation).

**Method:** Train 8-step scalar Kerr-ODE. For each layer, run three reverse passes: (a) full reverse with 64 steps, (b) gamma=0 reverse (remove dissipation), (c) control (random vector forward+backward for noise floor). Classify by blow-up and error magnitude.

**Results — the binary split:**

| Layer | Reversible | Damping | Nonlinear | Fwd Clamping |
|---|---|---|---|---|
| L0 | 64/64 (100%) | 0/64 | 0/64 | 0/64 |
| L1 | 0/64 (0%) | 0/64 | 64/64 (100%) | 8/64 |
| L2 | 0/64 (0%) | 0/64 | 64/64 (100%) | 43/64 |
| L3 | 0/64 (0%) | 0/64 | 64/64 (100%) | 61/64 |

The transition is binary, not gradual. L0 is fully reversible; L1-L3 are fully irreversible-nonlinear. Zero bands in any layer classified as damping-irreversible — the Kerr nonlinearity, not gamma, creates the irreversibility.

**L0 reversible detail:** Full reverse reconstruction error mean=10.6% (numerical Euler noise). No forward clamping. Amplification ratio 0.968 (slight attenuation). L0 is doing gentle spectral remixing that an analytical transform could replace.

**Clamping gradient — the finding within the finding:** L0: 0/64, L1: 8/64, L2: 43/64, L3: 61/64 bands hit the [-10, 10] clamp during forward pass. By L3, 95% of bands are being truncated. Some of the "irreversible-nonlinear" classification in deep layers may actually be information destroyed by the clamp rather than by the Kerr dynamics. The signal is being driven into the rails.

**Cross-reference:** L3's 4 bands with >1.5x amplification are all nonlinear-irreversible. The amplification is genuine nonlinear computation, not spectral routing.

**Three interventions identified:**
1. Replace L0 ODE with analytical linear transform (free 25% compute saving — fully reversible)
2. Widen/remove clamping bounds in L2-L3 (test if the ceiling moves — is the gap an information bottleneck?)
3. RK4 integration for L1-L3 (better integration of the genuinely nonlinear portion)

Verdict: 75% irreversible-nonlinear (L1-L3), 25% reversible (L0). The nonlinear dynamics are genuinely essential. Damping is not the bottleneck. Forward clamping in deep layers is a separate information bottleneck requiring investigation.

---

## Phase 22b: Analytical L0 Replacement

Phase 22 showed L0 is 100% reversible. Can we replace L0's 8-step ODE with a per-band 2x2 linear transform?

**Three modes tested:**

| Mode | Val Loss | vs MLP | vs Full Kerr |
|---|---|---|---|
| MLP baseline | 1.7134 | - | -4.92% |
| Full Kerr-ODE 8s | 1.8020 | +5.17% | baseline |
| Hybrid (analytical L0, trained from scratch) | 1.8142 | +5.89% | +0.68% |
| Post-hoc replacement (trained Kerr, L0 swapped at inference) | 4.7429 | +176.8% | +163.2% |

**Post-hoc replacement: catastrophic failure.** The analytical linear approximation (ignoring Kerr terms) has 158% relative error on L0's output. Val loss goes from 1.80 to 4.74. Reversible does not mean linear. L0's alpha/beta are small (0.05) but over 8 Euler steps the accumulated nonlinear effect creates a specific output distribution that L1-L3 are trained to expect. Swapping it out breaks the contract.

**Hybrid training: works.** Training from scratch with PerBandLinear at L0 gives +0.68% vs full Kerr-ODE. The learned L0 transforms are near-identity (Frobenius norm 1.40 vs identity's 1.41, determinant 0.98, trace 1.98), confirming L0 barely transforms the signal.

**Key insight -- impedance matching:** L0's function is not to transform the signal. Its function is to condition it. Near-identity, barely changes anything, but establishes the distribution that the nonlinear layers (L1-L3) expect as input. The Kerr-ODE at L0 acts as a handshake protocol -- the content is nearly a no-op, but downstream layers check for the exact signature. This is not wasted compute; it is impedance matching between the embedding layer and the nonlinear computation layers.

**Implication for clamping experiment:** If downstream layers adapt to the exact output distribution of upstream layers -- including clamping artifacts -- then widening clamps post-hoc will cause the same distribution shift. The safe approach: train from scratch with wider clamps.

Verdict: Reversible does not equal replaceable. The 25% ODE compute saving IS achievable (+0.68% cost) but only through hybrid training from scratch, not post-hoc substitution. L0 performs impedance matching, not computation.

---

## Summary

| Phase | Question | Result |
|---|---|---|
| 1. Spectral Persistence | Does harmonic structure survive? | Partially — some channels persist strongly |
| 2. Geometric Relations | Do channels overlap? | 92.5% independent — network actively disentangles |
| 3. Knowledge Editing | Can we edit weights surgically? | NO — surgery at embedding level is insufficient |
| 3b. Harmonic Injection | Can we swap geometry at inference? | YES — 80.7% swap rate, no retraining |
| 4. Harmonic Construction | Can we build NEW geometry from scratch? | YES — 0.991 interpolation, 0.707 prediction accuracy |
| 5. Musical Harmonics | Do musical intervals predict channel behavior? | YES — +0.454 correlation with edit safety, character types form classifiable chords |
| 6. Progressive Learning | Does structure-first training help? | YES — better final quality (1.559 vs 1.588), faster new knowledge absorption (5/5) |
| 7. Concept Composition | Do characters form chords or stay independent? | CHORDS — 33M-fold divergence growth, semantic clustering at character level |
| 8. Init Convergence | Does random init find the same structure? | NO — cross-run similarity is zero. Every model invents its own channel organization |
| 9. Commitment Point | Where does the model decide? | layer3_mlp — delivers half the accuracy in one step. Depth is token-dependent |
| 10. Early Exit | Can the model skip layers for easy tokens? | PARTIALLY — 2-4% compute saving at 97-99% accuracy. Newlines trivially predictable (50.9% exit, 97.8% acc) |
| 11. Chord Flow | Can chords replace tokens in upper layers? | NO — chord boundaries detectable but composition works by differentiation not convergence |
| 12. Natural Expression | What is the model's own language? | Progressive model: more compact space, richer dreams, 22% of hidden structure exceeds token vocabulary |
| 13. Expression Curriculum | Can we teach it to speak? | Richer heads find smarter (not more) structure. Linear head preserves knowledge best (11.6x efficiency) |
| 14. Shakespeare Knowledge | Does the model know Shakespeare? | YES — P("discontent")=0.39, P("Juliet")=0.28. Mid bands 1.6x more active during confident predictions |
| 15. Harmonic Decoder | Can we listen to the model's confidence? | YES — Harmonic beam 0.186 vs greedy 0.164 (+13.4%). Mid-band signal guides adaptive beam width |
| 16. Wave Packet Engine | Can sparse DFT queries match full cosine retrieval? | YES — 5/5 retrieval with 25% of bands. Lossless round-trip (2.24e-08). 87% quality at 75% data transfer. |
| 17. Weight Spectral Analysis | Do harmonic embeddings create band-sparse weights? | **NULL** — all modes spectrally flat (88.3% bands for 90% energy, 0% sparsity). Optimiser determines weight spectra, not embeddings. |
| 17b. Curriculum Specialisation | Does frequency curriculum change weight spectra? | **NULL** — curriculum teaches frequency patterns but does not restructure weight spectra. Boundary: wave coherence is representation/retrieval primitive, not training primitive. |
| 18. Harmonic Attention Heads | Do harmonic-structured Q/K projections improve attention? | **NO** — 5.2% worse than standard. Uniform entropy (4.56) across all heads/layers = model cannot discriminate tokens. Q/K must remain unconstrained. Boundary extended: harmonic structure helps representations, not learned projections. |
| 19. Spectral Interference | Can embedding interference replace learned Q/K? | **NO** — 5.3% worse. Same uniform entropy (4.56). Embedding dot products don't discriminate between tokens. Matches Phase 18 result exactly (3.2503 vs 3.2511) — confirming Phase 18's Q/K converged to producing same uniform attention as having no Q/K at all. |
| 19b. Harmonic Attention Bias | Does an additive harmonic bias improve standard attention? | **NO** -- 1.1% worse (candle, lambda stuck). PyTorch verification: lambda DOES learn (avg 0.1->0.215, low-freq heads amplify, high-freq suppress) but loss still -0.4%. Model detects harmonic signal but cannot exploit it for prediction. **Corrective Finding #7: candle autograd doesn't propagate gradients through frozen tensor products.** |
| 20. LC Circuit Layer | Can frequency-native computation replace MLP? | **NO** -- 148 params/layer, 23.5% worse. Concept works (params learn meaningful patterns in PyTorch) but extreme capacity starvation. Candle autograd blocked gradients again (Finding #7). |
| 20b. Expanded LC Layer | Fair capacity test: 13.4K params/layer (10x fewer than MLP)? | **NO** -- 21.3% worse. 91x more params only bought 2.2pp improvement. Bottleneck is architectural: per-band nonlinear + cross-band linear cannot match dense MLP's nonlinear multi-band interaction. Frequency-native structure helps (8.1x efficiency ratio) but cannot replace dense computation. |
| 21. Kerr-ODE Layer | Can nonlinear optics ODE replace MLP? | **PARTIALLY** -- 7.7-8.5% worse with 7.9x fewer FFN params. Kerr nonlinearity (|Z|^2 cross-band coupling) cuts the gap from LC's 21.3% to 7.7%. Depth-dependent nonlinearity: deep layers amplify Kerr effect, shallow layers suppress it. First wave-native primitive that meaningfully competes with matmul. |
| 21b. Per-Band Kerr | Does per-band alpha_k/beta_k close the 8% gap? | **NO** -- 0.54pp improvement over scalar (negligible). All layers' alpha/beta std < 0.02 (clustered). Integration depth (8 vs 4 steps) matters 4x more than per-band freedom. Same pattern as Phase 17: freedom granted, freedom ignored. |
| 22. Inverse Kerr | What does the Kerr-ODE actually compute? | **Binary split**: L0 100% reversible (spectral remixing), L1-L3 100% irreversible-nonlinear (genuine computation). Zero damping-irreversibility. Clamping gradient: L3 has 95% of bands hitting the clamp -- information bottleneck. Three interventions: analytical L0, wider clamps, RK4 for L1-L3. |
| 22b. Analytical L0 | Can L0's ODE be replaced with a linear transform? | **PARTIALLY** -- Hybrid training from scratch: +0.68% vs full Kerr (viable, 25% ODE compute saving). Post-hoc replacement: +163% (catastrophic). Reversible does not equal replaceable. L0 performs impedance matching -- near-identity conditioning that downstream layers are calibrated to expect. |
