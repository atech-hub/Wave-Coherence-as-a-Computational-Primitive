# Wave Coherence as a Computational Primitive: Mathematical Foundations

---

## Abstract

This document presents the mathematical foundations of the wave coherence framework in standard notation, independent of any programming language. All definitions, propositions, and constraints are stated formally and referenced to their empirical validations (Tests 1--25). The framework encodes discrete attribute values as points on the unit circle and uses a single parameterised function --- the harmonic coherence operator --- as a universal relationship detector. We establish its algebraic properties, characterise its resolving power, prove its kernel admissibility, and identify four design constraints that bound its operating regime.

---

## 1. Preliminaries

Let **S**&sup1; denote the unit circle, identified with the interval [0, 2&pi;) under addition modulo 2&pi;.

All angles are in radians unless stated otherwise. We write &Delta;&theta; = &theta;_a &minus; &theta;_b for the signed angular difference and |&Delta;&theta;|_c = min(|&Delta;&theta;|, 2&pi; &minus; |&Delta;&theta;|) for the shortest-arc (symmetric) distance on **S**&sup1;.

---

## 2. Core Definitions

**Definition 2.1** (Phase Encoding).
Let *B* &in; **Z**&sup;+ be the *bucket count* (resolution parameter). The phase encoding of a discrete value *k* &in; {0, 1, ..., *B* &minus; 1} is

> &theta;_k = 2&pi;k / *B*

mapping each value to a point on **S**&sup1;. The encoding is injective for *k* &lt; *B*.

**Definition 2.2** (Coherence).
The *coherence* between two encoded phases &theta;_a, &theta;_b &in; **S**&sup1; is

> *C*(&theta;_a, &theta;_b) = cos(&theta;_a &minus; &theta;_b)

with range [&minus;1, 1]. This is the inner product of two unit vectors and measures alignment: *C* = 1 (identical), *C* = 0 (orthogonal), *C* = &minus;1 (diametrically opposite).

**Definition 2.3** (Harmonic Coherence).
For *n* &in; **Z**&sup;+, the *n*th-harmonic coherence is

> *C*_n(&theta;_a, &theta;_b) = cos(*n*(&theta;_a &minus; &theta;_b))

The parameter *n* determines the class of relationships detected:

| *n* | Detected relationship | Angular separation |
|-----|----------------------|-------------------|
| 1 | Identity (exact match) | 0 |
| 2 | Opposition | &pi; |
| 3 | Triadic family | 2&pi;/3 |
| 4 | Quadrant | &pi;/2 |
| *n* | *n*-fold symmetric | 2&pi;/*n* |

In general, *C*_n(&theta;_a, &theta;_b) = 1 if and only if *n*(&theta;_a &minus; &theta;_b) &equiv; 0 (mod 2&pi;), i.e., when |&Delta;&theta;| is an integer multiple of 2&pi;/*n*.

**Definition 2.4** (Symmetric Angular Distance).
The shortest-arc distance on **S**&sup1; is

> *d*(&theta;_a, &theta;_b) = min(|&theta;_a &minus; &theta;_b| mod 2&pi;, &ensp; 2&pi; &minus; |&theta;_a &minus; &theta;_b| mod 2&pi;)

with range [0, &pi;].

**Definition 2.5** (Directed Angular Distance).
The counterclockwise distance from &theta;_a to &theta;_b is

> *d*&#8407;(&theta;_a, &theta;_b) = (&theta;_b &minus; &theta;_a) mod 2&pi;

with range [0, 2&pi;). Note that *d*&#8407;(&theta;_a, &theta;_b) &ne; *d*&#8407;(&theta;_b, &theta;_a) in general.

**Definition 2.6** (Orb Function --- Fuzzy Matching).
For a target angular distance &tau; and tolerance radius *r* &gt; 0, the orb function is

> *O*(&delta;; *r*) = cos(&delta;&pi; / 2*r*) &ensp; if &delta; &le; *r*; &ensp; 0 &ensp; otherwise

where &delta; = |*d*(&theta;_a, &theta;_b) &minus; &tau;|. The falloff is concave (cosine, not linear): at 62.5% of the tolerance radius, the score is 0.556, not 0.375. Maximal at &delta; = 0 (score 1.0), continuously vanishing at &delta; = *r*.

**Definition 2.7** (Multi-Attribute Conjunction).
For entities with *K* encoded attributes, each producing coherence scores *C*^(1), *C*^(2), ..., *C*^(*K*), the compound coherence is

> *C*(&mathbf{a}, &mathbf{b}) = &prod;_{k=1}^{K} *C*^(k)(&theta;_a^(k), &theta;_b^(k))

Multiplication implements logical AND: the compound score is high only when all attribute-level scores are high. The combined selectivity approaches the product of individual selectivities.

**Definition 2.8** (Harmonic Embedding Vector).
For *N* harmonics, the harmonic embedding of phase &theta; is the vector

> **v**(&theta;) = (cos &theta;, cos 2&theta;, cos 3&theta;, ..., cos *N*&theta;) &ensp; &in; **R**^*N*

Each component is the *n*th harmonic coherence with the reference angle &theta; = 0. The full embedding across *K* attributes and *N* harmonics per attribute has dimensionality *KN*.

**Definition 2.9** (Energy Concentration).
For a set of phases {&theta;_i}, the *energy fraction* at harmonic *n* is

> &eta;_n = (1/|*P*|) &sum;_{(i,j) &in; *P*} |*C*_n(&theta;_i, &theta;_j)| &ensp; / &ensp; &sum;_{m=1}^{N} (1/|*P*|) &sum;_{(i,j) &in; *P*} |*C*_m(&theta;_i, &theta;_j)|

where *P* is the set of distinct pairs. The *fundamental harmonic* is the lowest *n* at which the signed mean coherence exceeds a threshold (e.g., 0.95), identifying the dominant structural periodicity.

**Definition 2.10** (Multi-Grid Phase Encoding).
Let {*B*₁, *B*₂, ..., *B*_K} be a set of grid sizes (bucket counts). The phase encoding of a discrete value on grid *B*_i is

> &theta;_k^(i) = 2&pi;k / *B*_i

Each grid induces a different angular spacing between adjacent positions. On grid *B*_i, the spacing is 2&pi;/*B*_i. The coherence between two values at harmonic *n* on grid *B*_i is *C*_n^(*B*_i)(&theta;_a, &theta;_b) = cos(*n*(&theta;_a^(i) &minus; &theta;_b^(i))).

**Definition 2.11** (Grid Aliasing).
On a grid of size *B*, harmonic *n* is *aliased* to harmonic *n* mod *B*. The coherence values are identical:

> *C*_n^(*B*)(&theta;_a, &theta;_b) = *C*_{n &thinsp; mod &thinsp; B}^(*B*)(&theta;_a, &theta;_b) &ensp; for all &theta;_a, &theta;_b on the grid

This follows from cos(*n* &middot; *k* &middot; 2&pi;/*B*) = cos((*n* mod *B*) &middot; *k* &middot; 2&pi;/*B*) for integer *k*. Harmonics above *n* = *B*/2 fold back to lower frequencies, analogous to the Nyquist sampling limit in signal processing. Example: harmonic *n* = 27 on a 12-grid is indistinguishable from *n* = 3 (since 27 mod 12 = 3).

**Definition 2.12** (Composite Grid Coverage).
For two grids of sizes *B*₁ and *B*₂, the *composite Nyquist limit* is

> *n*_max = lcm(*B*₁, *B*₂) / 2

The composite grid resolves all harmonics from *n* = 1 to *n*_max without aliasing, using *B*₁ + *B*₂ encoding positions instead of lcm(*B*₁, *B*₂). The harmonic coverage gain over single-grid encoding is:

> gain = lcm(*B*₁, *B*₂) / max(*B*₁, *B*₂)

For *B*₁ = 12, *B*₂ = 10: lcm = 60, *n*_max = 30, gain = 5&times;, using 22 positions instead of 60. For *K* grids, the composite limit extends to lcm(*B*₁, *B*₂, ..., *B*_K) / 2.

**Definition 2.13** (Legendre Coherence).
For two points on a sphere with angular separation &gamma;, the degree-*l* spherical coherence is

> *C*_l^sphere(*a*, *b*) = *P*_l(cos &gamma;)

where *P*_l is the Legendre polynomial of degree *l*, computed via the three-term recurrence:

> *P*_0(*x*) = 1, &ensp; *P*_1(*x*) = *x*, &ensp; *P*_{*l*+1}(*x*) = ((2*l*+1) &middot; *x* &middot; *P*_l(*x*) &minus; *l* &middot; *P*_{*l*&minus;1}(*x*)) / (*l*+1)

For points encoded as unit vectors **u**, **v** on the sphere: cos &gamma; = **u** &middot; **v** (dot product). Computational cost: 3 multiplies + 1 add per degree, same class as cos(*n*&Delta;&theta;). The Legendre kernel satisfies the admissibility conditions of Proposition 3.4: *P*_l(1) = 1 (normalisation), *P*_l(cos &gamma;) = *P*_l(cos &gamma;) (symmetry), |*P*_l(*x*)| &le; 1 for *x* &in; [&minus;1, 1] (boundedness). Through degree *l* = 10, the sphere provides 121 modes (&Sigma;(2*l*+1) for *l* = 0..10) versus the circle's 21 (2*n*+1 for cosine + sine at each harmonic, but only 2 per *n*).

**Definition 2.14** (Embedded Coherence).
For embeddings with phase &phi; and magnitude *r*, the *magnitude-adjusted phase* is

> &phi;_eff = &phi; + &alpha; &middot; (*r* &minus; *r*_mean) / *r*_std

where *r*_mean and *r*_std are the population magnitude statistics and &alpha; &ge; 0 is a tuning parameter. The *embedded coherence* at harmonic *n* is

> *C*_n^emb(*a*, *b*) = cos(*n* &middot; (&phi;_eff_a &minus; &phi;_eff_b))

At &alpha; = 0, this reduces exactly to standard harmonic coherence *C*_n(&theta;_a, &theta;_b). At &alpha; &gt; 0, magnitude differences between entities produce phase shifts that the cosine function converts into coherence gradients, enabling within-group ranking that pure phase coherence cannot provide.

---

## 3. Propositions

**Proposition 3.1** (Encoding Correctness).
*For any two values k, l &in; {0, ..., B &minus; 1}, the phase-encoded coherence scan C(&theta;_k, &theta;_l) = 1 if and only if k = l. Coherence scanning is equivalent to exact-match comparison.*

*Validation:* Test 8 --- phase-encoded scan matches linear value comparison on all 10/10 queries with identical results.

---

**Proposition 3.2** (Harmonic Orthogonality).
*Let m, n &in; **Z**&sup;+ with m &ne; n. The set of phase pairs satisfying C_m = 1 is disjoint from the set satisfying C_n = 1 (excluding the trivial pair &Delta;&theta; = 0). Formally:*

> *{&Delta;&theta; &in; (0, 2&pi;) : C_m(&Delta;&theta;) = 1} &cap; {&Delta;&theta; &in; (0, 2&pi;) : C_n(&Delta;&theta;) = 1} = &empty; &ensp; when gcd(m,n) = 1*

*Different harmonic frequencies operate as independent selectors with zero cross-talk.*

*Validation:* Test 14 --- harmonics *n* = 3, 4, 5, 6 tested pairwise; each detects only its own family members and completely excludes entities belonging to other families.

---

**Proposition 3.3** (Collision Resolution).
*Let two values be encoded at phases &theta;_a, &theta;_b with angular difference &Delta;&theta; = |&theta;_a &minus; &theta;_b| &gt; 0. Let t &in; (0, 1) be a coherence threshold. The minimum harmonic at which C_n(&theta;_a, &theta;_b) &lt; t is*

> *n*_diverge = &lceil; arccos(*t*) / &Delta;&theta; &rceil;

*This is deterministic and closed-form --- collision resolution scales by analysis depth (increasing n), not storage (increasing B).*

*Validation:* Test 11 --- prediction matches measurement exactly at three scales:

| &Delta;&theta; | Predicted *n* | Measured *n* |
|---|---|---|
| 2&deg; (0.0349 rad) | 13 | 13 |
| 1&deg; (0.0175 rad) | 26 | 26 |
| 0.1&deg; (0.00175 rad) | 259 | 259 |

---

**Proposition 3.4** (Kernel Admissibility).
*The harmonic coherence function C_n(&theta;_a, &theta;_b) = cos(n(&theta;_a &minus; &theta;_b)) satisfies the four admissibility conditions for a valid coherence kernel (cf. Moriya, 2025):*

*(i) Symmetry (Hermiticity):* &ensp; *C_n(&theta;_a, &theta;_b) = C_n(&theta;_b, &theta;_a) &ensp; for all n, &theta;_a, &theta;_b*

*(ii) Normalization:* &ensp; *C_n(&theta;, &theta;) = 1 &ensp; for all n, &theta;*

*(iii) Positive semi-definiteness:* &ensp; *For any finite set {&theta;_1, ..., &theta;_m}, the Gram matrix G_{ij} = C_n(&theta;_i, &theta;_j) has all eigenvalues &ge; 0.*

*(iv) Spectral scaling:* &ensp; *The angular resolution at threshold t is arccos(t)/n, which decreases monotonically with n.*

*Proof sketch.* (i) follows from cos being an even function. (ii) follows from cos(0) = 1. (iii) holds because cos(*n*&theta;) is a single Fourier mode on **S**&sup1;, and the Gram matrix of a single non-negative Fourier coefficient is positive semi-definite --- this is a standard result in harmonic analysis on the circle. Computationally verified via all 2&times;2 and 3&times;3 principal minors across 13 test angles and 8 harmonics. (iv) follows from the inverse relationship between *n* and the argument of arccos.

*Validation:* Test 22 --- all four properties verified computationally with zero violations across 624 symmetry checks, 104 normalization checks, and all principal minors.

---

**Proposition 3.5** (Cosine Similarity Blindness).
*The inner product of two harmonic embedding vectors aggregates all harmonic channels into a single scalar:*

> *&langle;**v**(&theta;_a), **v**(&theta;_b)&rangle; = &sum;_{n=1}^{N} cos(n &middot; &Delta;&theta;)*

*This is the Dirichlet kernel. The summation can equal zero even when individual terms equal 1, destroying per-channel harmonic structure.*

*Example.* For &Delta;&theta; = 2&pi;/3 (triadic relationship) with *N* = 12 harmonics:

> *C*_3 = *C*_6 = *C*_9 = *C*_{12} = +1.0 &ensp; (4 channels)
>
> *C*_1 = *C*_2 = *C*_4 = ... = &minus;0.5 &ensp; (8 channels)
>
> &sum; = 4(1.0) + 8(&minus;0.5) = **0.0**

*Cosine similarity reports "no relationship" between entities that are perfectly coherent at n = 3. Per-channel decomposition --- evaluating C_n individually for each n --- recovers the structure that aggregation destroys.*

*Validation:* Test 21 --- five planted relationships (triadic, opposition, quadrant, sextile, pentagonal) all invisible to cosine similarity (scores &lt; 0.01), all recovered by harmonic sweep at their exact harmonics with zero false positives. Test 24 --- confirmed on real word embeddings (all-MiniLM-L6-v2): spectral variance 3&times; higher for synonym/antonym pairs vs 7&times; for synonym/unrelated pairs, while cosine similarity fails to distinguish the structural difference.

---

**Proposition 3.6** (Boundary Invariance).
*The coherence function C_n is continuous and symmetric at the 0/2&pi; boundary. For any &theta;_a near 0 and &theta;_b near 2&pi;:*

> *C_n(&theta;_a, &theta;_b) = C_n(&theta;_b, &theta;_a) = C_n(0, 2&pi; &minus; &epsilon;) &ensp; as &epsilon; &rarr; 0*

*No branch cuts or discontinuities arise from the circular topology.*

*Validation:* Test 15 --- scores verified symmetric at the 0&deg;/360&deg; boundary for all harmonic numbers.

---

**Proposition 3.7** (Self-Indexing).
*The phase encoding &theta;_k = 2&pi;k/B maps each value to a bucket address on [0, B). A query at threshold t examines at most*

> 2 &middot; &lceil; arccos(*t*) / (2&pi;/*B*) &rceil; + 1

*buckets, achieving sub-linear query complexity in the number of entities. For a harmonic query at order n, the spread is:*

> 2 &middot; &lceil; arccos(*t*) / (*n* &middot; 2&pi;/*B*) &rceil; + 1

*buckets per harmonic region.*

*Validation:* Test 18 --- bucket index matches full scan on all queries, examining approximately 2--23% of 1000 entities depending on threshold.

---

**Proposition 3.8** (Torus Composition).
*For K attributes encoded independently on **S**&sup1;, the joint encoding maps to the K-torus **T**^K = **S**&sup1; &times; ... &times; **S**&sup1;. A compound query narrows on all K dimensions simultaneously. The selectivity of the compound query approaches the product of per-dimension selectivities:*

> *S*(&mathbf{q}) &asymp; &prod;_{k=1}^{K} *S*_k(*q*_k)

*Validation:* Test 19 --- 2D torus index with *B* &times; *B* grid achieves multiplicative selectivity improvement over 1D, with all compound query results matching full scan exactly.

---

**Proposition 3.9** (Fundamental Harmonic Detection).
*Given a set of phases with unknown periodic structure, the fundamental harmonic n* is identifiable as the lowest n for which the signed mean pairwise coherence exceeds a threshold &tau;:*

> *n** = min{*n* &in; **Z**&sup;+ : (1/|*P*|) &sum;_{(i,j) &in; *P*} *C*_n(&theta;_i, &theta;_j) &gt; &tau;}

*This distinguishes the fundamental from its integer multiples, which also exhibit high unsigned coherence but have alternating signs that reduce the signed mean.*

*Validation:* Test 23 --- triadic structure correctly identified at *n* = 3, opposition at *n* = 2, quadrant at *n* = 4, noise at none.

---

**Proposition 3.10** (Single-Scan Group Discovery).
*A harmonic coherence scan at order n discovers all phase pairs (&theta;_a, &theta;_b) satisfying n(&theta;_a &minus; &theta;_b) &equiv; 0 (mod 2&pi;) in a single pass. This includes all integer multiples of 2&pi;/n --- the complete n-fold symmetric family --- without enumerating relationships explicitly.*

*Validation:* Test 9 --- a single scan at *n* = 3 found 75 related entities across 3 groups. The equivalent relational operation requires 2--3 JOINs and an explicit relation table.

---

**Proposition 3.11** (Dynamic Mutability).
*The phase-indexed structure supports insertion, deletion, and update as local operations on **S**&sup1;. No global rebuild is required. Query correctness is maintained through arbitrary mutation sequences.*

*Validation:* Test 20 --- remove, insert, and update operations verified with all queries returning correct results throughout the mutation sequence.

---

**Proposition 3.12** (Grid Aliasing Identity).
*For any grid of size B, harmonic n, and positions k, l &in; {0, ..., B &minus; 1}:*

> *cos(n &middot; (k &minus; l) &middot; 2&pi;/B) = cos((n mod B) &middot; (k &minus; l) &middot; 2&pi;/B)*

*Proof.* Let &Delta;k = k &minus; l. Then n &middot; &Delta;k &middot; 2&pi;/B = (n mod B) &middot; &Delta;k &middot; 2&pi;/B + q &middot; &Delta;k &middot; 2&pi; for some integer q. Since cos is 2&pi;-periodic and &Delta;k is an integer, the q &middot; &Delta;k &middot; 2&pi; term vanishes. &square;

*Corollary.* A grid of size B cannot distinguish harmonic n from harmonic n + B. The maximum resolvable harmonic is n = B/2 (the grid Nyquist limit). All harmonics above this limit fold to a lower alias.

*Validation:* Curvature Test 6a --- cos(27 &middot; k &middot; 30&deg;) = cos(3 &middot; k &middot; 30&deg;) confirmed for all 12 positions on a 12-grid. The 27th harmonic is perfectly aliased to the 3rd.

---

**Proposition 3.13** (Composite Grid Resolution).
*For two grids of coprime-ish sizes B₁ and B₂, the composite grid resolves harmonics that neither grid can resolve alone. Specifically, for any harmonic n &le; lcm(B₁, B₂)/2, there exists at least one grid B_i such that n mod B_i &ne; m mod B_i for all m &lt; n with m &le; B_i/2. That is, harmonic n is not aliased on at least one of the two grids.*

*Example.* Harmonic n = 5 on B₁ = 12: aliased (5 &lt; 6 = B/2, so it IS resolved on 12-grid, but cos(5 &middot; k &middot; 30&deg;) never reaches 1.0 for any grid position because no grid position is an integer multiple of 72&deg;). On B₂ = 10: positions at k &middot; 36&deg; include 0&deg;, 72&deg;, 144&deg;, 216&deg;, 288&deg; --- cos(5 &middot; k &middot; 36&deg;) = +1.0 for all even k. The 5th harmonic is *native* to the 10-grid. The 12-grid sees n = 5 as a partial match (coherence 0.300 at best grid positions); the 10-grid sees it as exact (coherence 1.000).

*Validation:* Curvature Test 6b --- cos(5 &middot; k &middot; 36&deg;) = +1.0 for all Yang-stem positions on the 10-grid. Combined (12 + 10) grid covers harmonics n = 1 through 30 from 22 encoding positions.

---

**Proposition 3.14** (Geometric Comma).
*For p-fold and q-fold symmetry on a B-position grid with gcd(p, q) &lt; min(p, q), there exist configurations where no single metric on the B-grid can make both symmetries resonate perfectly for all member sets. The angular excess (comma) is:*

> *comma = 360&deg; / lcm(p, q)*

*This is a number-theoretic identity, not an optimisation failure.*

*Example.* For 3-fold (trine, 120&deg;) and 5-fold (quintile, 72&deg;) symmetry on a 12-grid: comma = 360&deg;/15 = 24&deg;. A non-uniform metric that makes all quintile pairs have exactly 72&deg; geodesic distance forces one of four trine triads to have a leg measuring 144&deg; instead of 120&deg; --- an excess of exactly 24&deg;. The other three triads are unaffected.

*Analogue.* The Pythagorean comma in music: 12 perfect fifths overshoot 7 octaves by ~23.46 cents. Both commas arise from rational divisions of a circle that cannot coexist exactly. Both have magnitude determined by the least common multiple of the competing periods.

*Validation:* Curvature Test 3 --- 24&deg; comma confirmed. 3 of 4 triads coexist with quintile constraints; Triad 3 (Tiger-Horse-Dog) absorbs the full 24&deg; excess in one leg. The comma is structural: no metric optimisation can eliminate it.

---

**Proposition 3.15** (Chebyshev/Legendre Equivalence Conditions).
*The Chebyshev polynomial T_n(x) = cos(n &middot; arccos(x)) and the Legendre polynomial P_l(x) agree at the endpoints:*

> *T_n(1) = P_l(1) = 1 &ensp; for all n, l*
>
> *T_n(&minus;1) = P_l(&minus;1) = (&minus;1)^n &ensp; for all n = l*

*They disagree at every intermediate value x &in; (&minus;1, 1). For x = cos(60&deg;) = 0.5: T_3(0.5) = cos(180&deg;) = &minus;1.0, but P_3(0.5) = &minus;0.4375. This follows from the distinct polynomial definitions: T_2(x) = 2x&sup2; &minus; 1 versus P_2(x) = (3x&sup2; &minus; 1)/2.*

*Consequence: circle coherence cos(n&Delta;&theta;) and sphere coherence P_l(cos &gamma;) are not interchangeable at intermediate angles. They are genuinely different coherence systems that coincide only at exact match (&Delta;&theta; = 0) and opposition (&Delta;&theta; = &pi;).*

*Validation:* Spherical Test 1 --- equator equivalence test confirms T_n &ne; P_l at 60&deg;, 90&deg;, and 120&deg; for l &ge; 2.

---

**Proposition 3.16** (Embedded Coherence Backward Compatibility).
*At &alpha; = 0, the embedded coherence (Definition 2.14) reduces exactly to standard harmonic coherence:*

> *C_n^emb(a, b)|_{&alpha;=0} = cos(n &middot; (&phi;_a &minus; &phi;_b)) = C_n(&theta;_a, &theta;_b)*

*At &alpha; &gt; 0, if magnitudes are equal (r_a = r_b), the effective phase shift is zero and the score is again identical to standard coherence:*

> *&phi;_eff_a &minus; &phi;_eff_b = (&phi;_a &minus; &phi;_b) + &alpha; &middot; (r_a &minus; r_b) / r_std = &phi;_a &minus; &phi;_b*

*The embedded method modifies scores only when magnitudes differ. It is a strict generalisation: every result of the standard coherence function is a special case of the embedded coherence function.*

*Validation:* Spherical Test 6 --- group detection preserved at α=0.1 (both methods detect same-group pairs equally). All circle detections maintained.

---

## 4. Design Constraints

**Constraint 4.1** (Threshold Floor).
*For an encoding with B buckets, adjacent values have coherence cos(2&pi;/B). The exact-match threshold must satisfy*

> *t* &gt; cos(2&pi; / *B*)

*to avoid false positives from neighbouring buckets. This is analogous to the Nyquist sampling limit: values closer than one bucket width are irresolvable at n = 1.*

| *B* | 2&pi;/*B* | Minimum *t* |
|-----|---------|------------|
| 12 | 30&deg; | 0.8660 |
| 60 | 6&deg; | 0.9945 |
| 100 | 3.6&deg; | 0.9980 |
| 360 | 1&deg; | 0.9998 |

*Validation:* Test 8 --- threshold sensitivity demonstrated at *B* = 12.

---

**Constraint 4.2** (Nonlinear Orb Falloff).
*The orb function O(&delta;; r) = cos(&delta;&pi;/2r) follows a cosine curve, not linear degradation. The practical consequence is that scores remain higher than a linear model predicts near the centre and drop more steeply near the boundary:*

| &delta; / *r* | Cosine score | Linear score |
|-------------|-------------|-------------|
| 0.000 | 1.000 | 1.000 |
| 0.250 | 0.924 | 0.750 |
| 0.625 | 0.556 | 0.375 |
| 1.000 | 0.000 | 0.000 |

*Validation:* Test 4 --- fuzzy matching scores measured at four distance fractions.

---

**Constraint 4.3** (Directed vs. Symmetric Distance).
*Symmetric operations (coherence, harmonic family membership) may use the shortest-arc distance d &in; [0, &pi;]. Asymmetric operations (visibility, typed reach, directed traversal) require directed distance d&#8407; &in; [0, 2&pi;). Using symmetric distance for asymmetric operations introduces false symmetry: "A sees B" does not imply "B sees A."*

*Validation:* Test 10 --- asymmetric typed reach requires directed distance to correctly distinguish broad vs. narrow visibility configurations at the same angular position.

---

**Constraint 4.4** (Harmonic Nyquist Limit).
*The threshold floor from Constraint 4.1 scales with harmonic number. At harmonic n with B buckets, the effective angular spacing is n &middot; 2&pi;/B, and the threshold floor becomes*

> *t*_floor(*n*, *B*) = cos(*n* &middot; 2&pi; / *B*)

*A threshold adequate at n = 1 may admit false positives at higher harmonics. The required bucket count for single-value precision at harmonic n with threshold t is:*

> *B* &gt; *n* &middot; 2&pi; / arccos(*t*)

| *n* | Effective angle (*B* = 360) | *t*_floor |
|-----|---------------------------|----------|
| 1 | 1&deg; | 0.9998 |
| 3 | 3&deg; | 0.9986 |
| 6 | 6&deg; | 0.9945 |
| 12 | 12&deg; | 0.9781 |

*Validation:* Test 16 --- harmonic-scaled Nyquist limit confirmed at 360-value scale with zero false positives when thresholds are correctly calibrated.

---

**Constraint 4.5** (Multi-Grid Nyquist Coverage).
*A single grid of size B resolves harmonics n = 1 to n = B/2. To resolve a target harmonic n_target > B/2, either increase B (storage cost) or add a second grid B₂ such that n_target &le; B₂/2 (composition cost). The minimum total encoding positions to cover harmonics 1 through n_target using K grids is:*

> *min(B₁ + B₂ + ... + B_K) &ensp; subject to &ensp; lcm(B₁, ..., B_K)/2 &ge; n_target*

*Grids should be chosen with small pairwise GCDs to maximise coverage per added position. Adding a grid B₂ that divides B₁ (e.g., 24 alongside 12) adds zero new harmonic coverage --- only redundancy.*

| Grid set | Positions | lcm | n_max | Coverage efficiency |
|----------|-----------|-----|-------|-------------------|
| {12} | 12 | 12 | 6 | 0.50 harmonics/position |
| {12, 10} | 22 | 60 | 30 | 1.36 harmonics/position |
| {12, 10, 27} | 49 | 540 | 270 | 5.51 harmonics/position |
| {12, 24} | 36 | 24 | 12 | 0.33 harmonics/position |

*Validation:* Curvature Test 6 --- B={12,10} confirmed to resolve harmonics 1-30. B={12,27} confirmed to extend from n=6 to n=13. Redundant grids (multiples of existing grids) confirmed to add no new coverage.

---

**Constraint 4.6** (Embedded Coherence &alpha; Bounds).
*The tuning parameter &alpha; in the embedded coherence function (Definition 2.14) controls the trade-off between group detection and within-group discrimination. For the phase shift not to exceed one harmonic period at the maximum magnitude deviation:*

> *&alpha; &lt; 2&pi; / (n_max &middot; max(|r &minus; r_mean| / r_std))*

*In practice, &alpha; &in; [0.05, 0.15] preserves group detection while providing useful discrimination. At &alpha; &gt; 0.3, the magnitude-induced phase shifts become large enough to disrupt group detection --- the ranking signal overwhelms the group membership signal. At &alpha; = 0, the method reduces to standard phase coherence with no discrimination capability.*

*Validation:* Spherical Test 6 --- &alpha;=0.1 achieves &rho;=&minus;0.9928 within-group ranking while preserving group detection. &alpha;=0.5 degrades to &rho;=&minus;0.2304. &alpha;=1.0 further degrades to &rho;=&minus;0.1424.

---

## 5. Density Scaling

**Proposition 5.1** (Collision Probability).
*For N entities distributed uniformly among B buckets, the probability of at least one bucket collision follows the birthday problem:*

> *P*(collision) &asymp; 1 &minus; exp(&minus;*N*&sup2; / 2*B*)

*Exact matching remains robust at all sub-saturated densities (N &lt; B). Harmonic queries at order n degrade as the minimum angular separation decreases, with the required resolution harmonic following Proposition 3.3.*

*Validation:* Test 17 --- eight configurations from 7-in-12 to 360-in-360. Exact match robust at all densities below saturation. Harmonic degradation follows predicted pattern.

---

## 6. Connection to Fourier Analysis

The harmonic embedding vector (Definition 2.8) is a Fourier basis expansion restricted to cosine terms. Its inner product yields the Dirichlet kernel:

> &langle;**v**(&theta;_a), **v**(&theta;_b)&rangle; = &sum;_{n=1}^{N} cos(*n* &middot; &Delta;&theta;)

The critical distinction between the harmonic embedding approach and standard cosine similarity is that the former preserves per-component structure while the latter collapses it. Proposition 3.5 demonstrates that this collapse is not merely a loss of precision but a total destruction of detectable relationships.

This parallels Moriya's Surface-Enhanced Coherence Transform (SECT), which shows that decomposing aggregate coherence into surface and propagation components recovers physical structure that ensemble averaging destroys. The principle is the same: **aggregate measures lose structure; per-channel analysis recovers it.**

The harmonic embedding dimensions are *structured* --- each component is the *n*th harmonic of attribute *k*, with a defined geometric meaning. This contrasts with learned embedding dimensions, which are *distributed* --- no individual dimension has interpretable meaning, and structure is emergent from training.

---

## 7. Notation Reference

For readers consulting the accompanying implementation:

| Mathematical notation | Rust implementation | Python implementation |
|---|---|---|
| &theta;_k = 2&pi;k/*B* | `Phase::from_value(k, B)` | `2 * np.pi * k / B` |
| *C*(&theta;_a, &theta;_b) | `phase_a.coherence(&phase_b)` | `np.cos(theta_a - theta_b)` |
| *C*_n(&theta;_a, &theta;_b) | `phase_a.harmonic_coherence(&phase_b, n)` | `np.cos(n * (theta_a - theta_b))` |
| *d*(&theta;_a, &theta;_b) | `phase_a.distance_degrees(&phase_b)` | `min(abs(d), 360 - abs(d))` |
| *d*&#8407;(&theta;_a, &theta;_b) | `phase_a.directed_distance_degrees(&phase_b)` | `(theta_b - theta_a) % (2*pi)` |
| *O*(&delta;; *r*) | `phase_a.fuzzy_match(&phase_b, target, orb)` | `np.cos(delta * pi / (2*orb))` |
| **v**(&theta;) | N/A (implicit in sweep) | `[np.cos(n*theta) for n in range(1,N+1)]` |
| *n*_diverge | Computed in Test 11 | `math.ceil(math.acos(t) / delta_theta)` |

---

## 8. Summary of Empirical Validation

All propositions and constraints are validated by a deterministic test suite of 25 core tests (21 in Rust, 4 in Python) plus 6 curvature investigation tests and 6 spherical investigation tests (Rust), with zero failures and zero external dependencies (Rust) or minimal dependencies (Python: PyTorch, sentence-transformers). Curvature tests validate multi-grid and metric propositions (3.12--3.14, 4.5). Spherical tests validate Chebyshev/Legendre and embedded coherence propositions (3.15--3.16, 4.6).

| Proposition | Statement | Validating test(s) |
|---|---|---|
| 3.1 | Encoding correctness | Test 8 |
| 3.2 | Harmonic orthogonality | Test 14 |
| 3.3 | Collision resolution formula | Test 11 |
| 3.4 | Kernel admissibility | Test 22 |
| 3.5 | Cosine similarity blindness | Tests 21, 24 |
| 3.6 | Boundary invariance | Test 15 |
| 3.7 | Self-indexing | Test 18 |
| 3.8 | Torus composition | Test 19 |
| 3.9 | Fundamental harmonic detection | Test 23 |
| 3.10 | Single-scan group discovery | Test 9 |
| 3.11 | Dynamic mutability | Test 20 |
| 5.1 | Collision probability | Test 17 |
| 3.12 | Grid aliasing identity | Curvature Test 6a |
| 3.13 | Composite grid resolution | Curvature Test 6b |
| 3.14 | Geometric comma | Curvature Test 3 |
| 3.15 | Chebyshev/Legendre equivalence conditions | Spherical Test 1 |
| 3.16 | Embedded coherence backward compatibility | Spherical Test 6 |

| Constraint | Statement | Validating test(s) |
|---|---|---|
| 4.1 | Threshold floor | Test 8 |
| 4.2 | Nonlinear orb falloff | Test 4 |
| 4.3 | Directed vs. symmetric distance | Test 10 |
| 4.4 | Harmonic Nyquist limit | Test 16 |
| 4.5 | Multi-grid Nyquist coverage | Curvature Test 6 |
| 4.6 | Embedded coherence &alpha; bounds | Spherical Test 6 |

---

## References

[1] Listopad, S. (2025). Wave-Based Semantic Memory: A Phase-Aware Alternative to Vector Retrieval. arXiv:2509.09691.

[2] Wang, L. (2025). Defierithos: The Lonely Warrior Rises from Resonance --- A Self-Resonance Architecture Beyond Attention. Submitted to NeurIPS 2025.

[3] Listopad, S. (2025). Phase-Coded Memory and Morphological Resonance. arXiv:2511.11848.

[4] Sun, Z., Deng, Z.-H., Nie, J.-Y., & Tang, J. (2019). RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space. In *Proceedings of ICLR 2019*. arXiv:1902.10197.

[5] Moriya, T. (2025). Surface-Enhanced Coherence Transform: A Framework for Structured Coherence Decomposition. arXiv:2505.17754.
