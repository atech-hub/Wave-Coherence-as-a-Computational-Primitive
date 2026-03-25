# Multi-Grid Harmonic Investigation

## Status: Complete — Six anomalies resolved to three independent mechanisms

## Context

The Wave Coherence framework encodes relationships as harmonic phase angles on a circle and measures them using `cos(n × Δθ)`. On a flat 12-position circle, this achieves 96.8% of MLP performance at 42.6% of parameters (Phase A). Every test in the core suite (Tests 1–25) validates this flat-circle model.

But the ancient geometric relationship catalogues — Chinese, Western, and Vedic systems independently developed over millennia — contain anomalies that the flat circle cannot explain. Specific angular relationships that should be equivalent on a flat circle are assigned opposite meanings. Angles rated as negligible in one tradition are foundational in another. Tolerance windows vary systematically by angle type.

This investigation tests whether these anomalies are arbitrary cultural conventions or signatures of deeper mathematical structure. Six anomalies were identified across three independent traditions. Each was tested with the existing Rust infrastructure using the simplest possible experiment. Every result — positive, negative, and null — is reported.

---

## The Anomalies

| # | Anomaly | Source | Flat-circle prediction | Observed |
|---|---------|--------|----------------------|----------|
| 1 | Liu He / Liu Hai | Chinese | Same angles → same relationship | Same angles, **opposite** meaning |
| 2 | Wu Xing 72°/144° | Chinese vs Western | Strength fixed by angle | 0.30 Western, **fundamental** Chinese |
| 3 | San He vs Wu Xing | Chinese | Systems can coexist | Partial incompatibility |
| 4 | Variable orbs | All traditions | Uniform resonance width | ±8° major, ±2° minor |
| 5 | Vedic Drishti | Vedic | All entities see all angles | **Type-dependent** visibility |
| 6 | Nakshatra / Sexagenary | Vedic / Chinese | One grid sufficient | Multiple incommensurate grids |

---

## Test 1: Liu He / Liu Hai — Path Separation

**File:** `tests/curvature_test1_liu_he_vs_liu_hai.rs`

**Question:** Can a non-uniform metric on the 12-position circle make Liu He pairs (harmony) score HIGH and Liu Hai pairs (harm) score LOW using the same coherence function?

**Why this matters:** Liu He and Liu Hai assign opposite meanings — harmony vs harm — to pairs at identical angular distances (30°, 90°, 150°). On a flat circle, the angle fully determines the relationship. These pairs are indistinguishable. If the geometry is non-uniform, the same coordinate angle can correspond to different geodesic distances depending on which path the relationship traverses.

**Setup:**
- 12 zodiac positions on a circle, 12 segments with variable weights g₀..g₁₁
- Liu He: 6 pairs at angular distances {30°, 30°, 90°, 90°, 150°, 150°}
- Liu Hai: 6 pairs at **identical** angular distances {30°, 30°, 90°, 90°, 150°, 150°}
- Coherence = cos(n × d_geodesic), where d_geodesic = weighted path length
- Flat baseline measured first, then optimized metric

**Phase 1 — Flat baseline:**

Separation = 1.85 × 10⁻¹⁷. Machine-precision zero. The flat circle is provably blind to the distinction.

**Phase 2 — Path analysis:**

The pairs traverse different segments of the circle despite having identical flat distances:

| Segment | Liu He uses | Liu Hai uses | Difference |
|---------|------------|-------------|------------|
| 0 (Rat→Ox) | 3× | 0 | Liu He only |
| 3 (Rabbit→Dragon) | 0 | 3× | Liu Hai only |
| 6 (Horse→Goat) | 3× | 0 | Liu He only |
| 9 (Rooster→Dog) | 0 | 3× | Liu Hai only |

The difference pattern is sinusoidal at k=2 (half-circle period). Liu He paths concentrate around segments 0 and 6. Liu Hai paths concentrate around segments 3 and 9. A k=2 sinusoidal metric — the simplest non-uniform metric — should maximise separation.

**Phase 3 — Optimized metric:**

| Measure | Value |
|---------|-------|
| Separation | **1.999 / 2.000** (theoretical maximum) |
| Liu He mean coherence | +0.9997 (range: +0.9985 to +1.0000) |
| Liu Hai mean coherence | −0.9996 (range: −0.9999 to −0.9988) |
| Overlap | **Zero** — every harmony pair scores above every harm pair |
| Best harmonic | n = 7 |
| Metric shape | k=2 dominant (amplitude 0.549), 6.2× ratio heaviest to lightest |

The k=2 sinusoidal dominance was predicted analytically from the path structure, then confirmed by the optimizer. The metric is smooth and low-frequency — not an overfitted artefact.

**Result: STRONG — Non-uniform metric achieves near-perfect binary classification**

**What this proves:**
- A non-uniform metric CAN distinguish Liu He from Liu Hai using the same coherence function
- The distinction is geometric (path-dependent), not arbitrary
- The k=2 metric shape was predicted analytically, then confirmed by optimization

**What this does NOT prove:**
- That the ancient circle IS curved (only that curvature is consistent with the data)
- That THIS specific metric is the correct one (many metrics achieve separation)
- That the other anomalies have the same explanation

---

## Test 2: Wu Xing — The 0.300 Identity

**File:** `tests/curvature_test2_wu_xing_amplification.rs`

**Question:** Can a non-uniform metric make the 12-position grid "see" the 5th harmonic (72°)?

**Why this matters:** The Western system rates the quintile (72°) at strength 0.30 — near the bottom of their scale. The Chinese Wu Xing system builds its entire generative and destructive element cycle on the same angle — it is foundational. Same angle, wildly different importance. On a flat circle, harmonic strength is a fixed property of the angle. It cannot be simultaneously weak and fundamental.

**The anchor finding:**

The flat n=5 coherence on a 12-position grid is **0.300000**. The Western system's assigned strength for the quintile is 0.30. This is not a correlation — it is an identity. The ancient strength rating IS the literal coherence score. The Western system measured the flat-circle value exactly.

The Chinese system assigned fundamental importance to the same angle because it operates on a geometry where 72° resonates at full strength. Both traditions measured correctly — for different geometries.

**Setup:**
- Wu Xing 5 elements mapped to nearest zodiac positions
- Generative cycle: flat distances alternate 60° and 90°, averaging 72° but never hitting it
- Analytical solution: 5 linear constraints force all generative pairs to exactly 72° geodesic distance
- 7 degrees of freedom remain after satisfying Wu Xing

**Result:**

| Condition | n=5 coherence |
|-----------|--------------|
| Flat 12-grid | **0.300** |
| Wu Xing metric | **1.000** |
| Amplification | **3.33×** |

The destructive cycle (144°) is automatically satisfied — each destructive path is a union of two generative groups, so its geodesic = 2 × 72° = 144°. Free of charge.

**Combined metric — both anomalies simultaneously:**

The 7 remaining degrees of freedom are sufficient to satisfy Wu Xing AND separate Liu He from Liu Hai on a single metric:

| System | Result |
|--------|--------|
| Wu Xing generative (n=5) | cos(5d) = 1.000 for all 5 pairs |
| Wu Xing destructive (n=5) | cos(5d) = 1.000 for all 5 pairs |
| Liu He/Liu Hai separation | **2.000** at n=10 (better than Test 1's 1.999) |
| Liu He/Liu Hai overlap | **None** |

Two independent anomalies, one metric, zero compromise on either.

**Honest cost:** San He triads (120°, n=3) degrade under the combined metric. Two of four triads drop from perfect to near-zero n=3 coherence. The combined metric cannot serve Wu Xing (n=5), Liu He/Liu Hai, AND San He (n=3) perfectly at once.

**Result: STRONG — Curvature amplifies invisible harmonic from 0.30 to 1.00**

**What this proves:**
- The Western 0.30 strength for quintiles is the LITERAL n=5 coherence on a flat 12-grid
- A metric exists that makes the 12-grid see the 5th harmonic perfectly
- Tests 1 and 2 are explained by a single metric with zero compromise
- The solution is partly analytical (exact constraints), not just optimized

**What this does NOT prove:**
- That the ancient circle IS curved (only that curvature is consistent)
- That the San He degradation is acceptable
- That the remaining anomalies follow the same pattern

---

## Test 3: The Geometric Comma — A Theorem

**File:** `tests/curvature_test3_geometric_comma.rs`

**Question:** Is the San He / Wu Xing incompatibility a mathematical necessity or an optimizer artefact?

**Why this matters:** Test 2 showed that a combined metric breaks San He triads. If this is an optimizer failure, a better algorithm might find a metric that satisfies everything. If it is a mathematical necessity, it means the two systems are fundamentally incompatible on one circle — and the ancient tradition of keeping them separate was the correct response.

**The proof:**

Wu Xing constraints force specific segment-weight sums:
- Each of 5 element groups sums to 2.4 (total = 12.0 = normalisation)

San He Triad 3 (Tiger-Horse-Dog) has a leg spanning segments {10, 11, 0, 1}. Wu Xing forces:
- g₁₀ + g₁₁ = 2.4 (Dog→Rat group)
- g₀ + g₁ = 2.4 (Rat→Tiger group)
- Forced sum = 4.8
- Required for 120° geodesic = 4.0
- **Excess = 0.8 weight-units = 24°**

This is a direct algebraic contradiction. No metric exists that satisfies both.

**The comma:**

| Quantity | Value | Derivation |
|----------|-------|-----------|
| Excess | 24° | = 2 × 72° − 120° |
| As fraction of circle | 1/15 | = 360° / lcm(3, 5) |
| Affected triads | T3 only | Tiger-Horse-Dog |
| Unaffected triads | T1, T2, T4 | All compatible with Wu Xing |

**The Pythagorean analogy:**

In music, 12 perfect fifths overshoot 7 octaves by ~23.46 cents — the Pythagorean comma. Two rational divisions of the octave that cannot coexist exactly. Every tuning system in history is a compromise between them.

In geometry, 2 quintile steps (2 × 72° = 144°) overshoot 1 trine step (120°) by 24°. Two rational divisions of the circle that cannot coexist exactly. The 12-position circle forces a choice between pure 3-fold and pure 5-fold symmetry.

Both commas are determined by the least common multiple: lcm(3, 5) = 15.

**Best compromise metric (Wu Xing + T1 + T2 + T4):**

Satisfying everything except Triad 3 leaves 1 degree of freedom. Under this compromise:

| System | Status |
|--------|--------|
| Wu Xing (all 5 pairs at 72°) | **Perfect** |
| San He T1 (Rat-Dragon-Monkey) | **Perfect** |
| San He T2 (Ox-Snake-Rooster) | **Perfect** |
| San He T3 (Tiger-Horse-Dog) | **Broken** — legs at 105.9°, 110.1°, 144.0° |
| San He T4 (Rabbit-Goat-Pig) | **Perfect** |
| Liu He/Liu Hai separation | 0.716 at n=14 (reduced from 2.0 unconstrained) |

Triad 3's Dog→Tiger leg absorbs the full 24° comma, shifting from a trine (120°) to near a quintile (144°).

**Result: THEOREM — The incompatibility is mathematical, not computational**

**What this proves:**
- San He / Wu Xing incompatibility is a theorem, not an optimizer failure
- Exactly 1 of 4 triads is broken, with a precise 24° excess
- The comma = 360° / lcm(3, 5) — a number-theoretic identity
- The ancient catalogue's treatment of the two systems as separate and non-unified correctly reflects the underlying geometry

**What this does NOT prove:**
- That the ancient practitioners knew the geometric reason for the incompatibility
- That a higher-dimensional embedding (e.g., a torus) might resolve it
- That the remaining anomalies follow the same pattern

---

## Test 4: Variable Orbs — Null Result

**File:** `tests/curvature_test4_variable_orbs.rs`

**Question:** Does the non-uniform metric predict which aspects get wide vs narrow tolerance windows?

**Why this matters:** The traditions assign variable orbs (tolerance windows): ±8° for major aspects, ±2° for minor aspects. On a uniform circle, resonance width should be uniform. Variable orbs could be a curvature signature — or they could have a simpler explanation.

**The simpler explanation holds:**

The flat orb is arccos(τ) / n — purely determined by harmonic number. Lower harmonics have wider peaks. Higher harmonics have narrower peaks. This alone correlates r = 0.887 with traditional orb assignments.

| Metric | Correlation with tradition | Mean absolute error |
|--------|--------------------------|---------------------|
| Flat (1/n) | 0.887 | 1.32° |
| Compromise (Test 3) | 0.897 | 0.98° |
| Combined (Test 2) | 0.844 | 2.22° |

The compromise metric adds a trivial +0.01 to correlation. The combined metric (high curvature) is actively worse — it systematically shrinks all orbs. Curvature adds noise, not signal.

One genuine anomaly remains: the quintile (flat predicts 5.2°, tradition says 2.0°). Neither flat nor curved models explain this.

**Result: NULL — Orbs are a flat harmonic property, not a curvature signature**

**What this proves:**
- Variable orbs are primarily explained by harmonic number (1/n peak width)
- This is a flat-circle property — curvature adds nothing
- The curvature story has a boundary: it explains compatibility/incompatibility (Tests 1–3), not resonance width

---

## Test 5: Vedic Drishti — Type-Dependent Visibility

**File:** `tests/curvature_test5_vedic_drishti.rs`

**Question:** Can per-type metrics reproduce the Vedic visibility tables through coherence alone?

**Why this matters:** In Vedic tradition, different graha (planetary types) see different angles. Mars sees {90°, 180°, 210°}. Jupiter sees {120°, 180°, 240°}. Saturn sees {60°, 180°, 270°}. On a flat circle, all entities see all angles equally. Type-dependent visibility would require type-dependent geometry.

**Finding 1 — Jupiter IS the 6th harmonic:**

Jupiter's drishti {120°, 180°, 240°} is symmetric about 180°. On the flat circle: cos(6 × 120°) = cos(6 × 180°) = cos(6 × 240°) = +1.000. Perfect resonance. No curvature needed. No directed distance needed. Jupiter's visibility is a flat-space property.

**Finding 2 — Mars and Saturn are not single-harmonic phenomena:**

Mars's drishti includes 210° (7/12 of the circle). No single harmonic n makes cos(n × 210°) = +1 without also resonating at unwanted angles. The optimizer returned near-flat metrics (all weights ≈ 1.00) for both Mars and Saturn — it could not find any metric that helps.

The problem is structural: {90°, 180°, 210°} is not a harmonic set. It cannot be generated by any single `cos(n × d)` function regardless of the metric. These are combinatorial rules — the tradition stores them as lookup tables because they don't reduce to single-parameter functions.

**Finding 3 — Directed distance doesn't help:**

Mars's 210° forward is geometrically equivalent to 150° backward on a shortest-path metric. Directed distance was tested to break this symmetry. The optimizer found identical scores for both modes — directed distance provides no additional discrimination power here.

**Result: SPLIT — Jupiter = flat harmonic, Mars/Saturn = irreducibly tabular**

**What this proves:**
- Some visibility rules (Jupiter) ARE harmonic phenomena on flat circles
- Other rules (Mars, Saturn) are NOT reducible to single-harmonic coherence
- Curvature adds nothing to drishti prediction
- The tradition's use of both harmonic rules and lookup tables reflects genuine mathematical differences

---

## Test 6: Incommensurate Divisions — The Multi-Grid Discovery

**File:** `tests/curvature_test6_incommensurate_divisions.rs`

**Question:** Why do the traditions maintain cycle counts that don't divide evenly into each other?

**Why this matters:** Most division systems use factors of 360° (12, 36, 60). But the Vedic Nakshatra system uses 27 divisions (13.333° each) and the Chinese Sexagenary system combines 12 and 10 cycles. These are incommensurate with each other and with the standard 12-grid. Curvature might explain why non-standard divisions are necessary — or the explanation might be simpler.

### Part A: Nakshatra 27

On a 12-point grid, cos(27k × 30°) = cos(3k × 30°) for all k. The 27th harmonic is perfectly aliased to the 3rd harmonic. The 12-grid literally cannot tell them apart. This is a Nyquist sampling limit: the 12-grid resolves harmonics up to n=6. Everything above folds back.

The 27-grid extends resolution from n=6 to n=13 — more than doubling the harmonic range. And 27 is optimal for this purpose:
- gcd(27, 12) = 3 → minimal shared structure, maximising new information
- lcm(27, 12) = 108 → rich combined structure (12 factors)
- 27 = 3³ → preserves trine compatibility

**Curvature cannot help here.** 12 points fundamentally cannot reconstruct 27-fold structure regardless of metric.

### Part B: Sexagenary 12 × 10 — The Reframing

This is the finding that reframes the entire investigation.

The 5 Yang Heavenly Stems sit at positions 0°, 72°, 144°, 216°, 288° on the 10-grid (36° segments). On this grid: cos(5 × k × 36°) = +1.0 for all Yang positions. **The 5th harmonic is native to the 10-grid.** No curvature needed. No forcing. Just the right grid for the right symmetry.

The Chinese system didn't warp the 12-grid to make the 5th harmonic work. They built a 10-grid where it works natively. Don't warp the space — build the right space.

**The compression:** The 12-grid covers harmonics n=1–6. The 10-grid covers n=1–5. Together (lcm = 60), they cover n=1–30. Two small grids (10 + 12 = 22 positions) give the harmonic coverage of a 60-position grid. The Sexagenary system is data compression — encoding a large cycle through two small incommensurate ones.

**Result: SAMPLING PHENOMENON — Multi-grid, not curvature**

This changes the interpretation of Test 2. Curvature CAN amplify the 5th harmonic on the 12-grid (0.30 → 1.00). But the ancients' actual solution was more elegant — they built a separate grid where the 5th harmonic lives natively, then composed the two grids into one system.

---

## The Three-Layer Architecture

The six anomalies resolve into three mathematically independent mechanisms:

### Layer 1: Flat Harmonics on Matched Grids (primary)

Each harmonic family on the grid where it is native. Most of the catalogue lives here.

- Western aspects on the 12-grid: trine (n=3), square (n=4), sextile (n=6)
- Wu Xing on the 10-grid: generative/destructive cycles (n=5) — native, no curvature
- Jupiter drishti: n=6 resonance on the 12-grid
- Variable orbs: 1/n peak width (r=0.887 correlation with tradition)
- Wu Xing strength 0.30: literal cos(5d) on the flat 12-grid

### Layer 2: Non-Uniform Metric (secondary)

Type-dependent path weighting within a single grid. Real but bounded — operates within one grid, does not cross grids.

- Liu He/Liu Hai: path-dependent separation of same-angle pairs (1.999/2.0 separation)
- Geometric comma: 24° = 360°/lcm(3,5) — proven theorem
- San He compatibility: 3 of 4 triads coexist with Wu Xing on one metric
- Operates below the Nyquist limit of its grid

### Layer 3: Structural Rules (tertiary)

Irreducible to any geometry or metric. Must be stored as lookup tables.

- Mars drishti {90°, 180°, 210°} — not a harmonic set, no single n discriminates
- Saturn drishti {60°, 180°, 270°} — same structural issue
- The traditions store these as tables because they don't reduce to functions

### Cross-Cutting: Multi-Grid Sampling (Nyquist extension)

Incommensurate grid counts capture frequencies that no single grid can sample.

- Nakshatra 27: extends resolution from n=6 to n=13
- Sexagenary 12×10: lcm = 60, covers n=1 to 30 from two small grids
- Grid affinity table determines which grid carries each harmonic natively

Each layer handles what the others cannot. The traditions maintained them as separate systems because they ARE mathematically separate — independent layers that compose but do not reduce.

---

## Practical Implications

### For the Database Engine

1. **Multi-grid indexing** (primary): Route queries to the grid that matches the target harmonic. Don't force all queries onto one grid.
2. **Metric weighting** (secondary): Apply within a grid for type-dependent relationships (Liu He/Liu Hai style path separation).
3. **Structural lookup** (tertiary): Irreducible rules must be declared, not computed. Some relationships are tables, not functions.
4. **Grid affinity table**: The query planner uses harmonic-to-grid mapping to route efficiently. Two small grids (10 + 12 = 22 positions) provide the coverage of one 60-position grid.

### For the Transformer Architecture

If the 12-grid and 10-grid compose into richer representations than either alone, harmonic embeddings on multiple incommensurate grids may capture structure that a single grid misses. Different grids serve as different basis sets for different frequency ranges.

**UPDATE (2026-03-24): This prediction was implemented and validated.**

The wave-engine's harmonic embeddings map each token to a phase angle: θ_v = 2π × v / vocab_size. On a single circle, adjacent tokens in a large vocabulary become geometrically indistinguishable (see wave-structure investigation — Harmonic Embedding Minimum Dimension).

Measured separation at 84 bands:

| Vocab | Single grid separation | Multi-grid separation | Improvement |
|-------|----------------------|----------------------|-------------|
| 2,000 | 0.94 | 95.01 | **101×** |
| 50,000 | 0.0016 | 18.60 | **11,800×** |

The fix applies the Sexagenary principle directly: two coprime moduli (m1, m2) near √vocab_size, each grid gets half the bands. Tokens that collide on grid 1 (same v mod m1) are separated on grid 2 (different v mod m2). The lcm(m1, m2) ≥ vocab_size guarantees every token has a unique combined position.

Implementation: `wave-engine/src/common/embed.rs` — `build_harmonic_table()` with `find_coprime_moduli()`. Published as Pattern 53 in ENGINE-PATTERNS.md.

This is a direct application of Test 6's finding: "Don't warp the space — build the right space." The ancient Sexagenary system used two incommensurate grids (10 × 12) to encode a 60-position cycle. The wave-engine uses two coprime grids to encode vocabularies that a single grid cannot resolve. Same principle, same mathematics, separated by millennia.

---

## Scorecard

| Test | Anomaly | Result | Mechanism | Layer |
|------|---------|--------|-----------|-------|
| 1 | Liu He / Liu Hai | **STRONG** | Non-uniform metric (path separation) | 2 |
| 2 | Wu Xing 72° | **STRONG** | Curvature on 12-grid / native on 10-grid | 1 + 2 |
| 3 | Geometric comma | **THEOREM** | 3/5 incompatibility = 24° | 2 |
| 4 | Variable orbs | **NULL** | Flat harmonic (1/n peak width) | 1 |
| 5 | Vedic Drishti | **SPLIT** | Jupiter = flat n=6 / Mars, Saturn = tabular | 1 + 3 |
| 6 | Nakshatra / Sexagenary | **SAMPLING** | Multi-grid Nyquist extension | Cross-cutting |

Six anomalies investigated. Every one resolved to a specific mechanism. No hand-waving, no narrative ahead of data. Three independent mathematical layers identified, with clean boundaries between them.

---

## Appendix: The Curvature Boundary

Curvature explains which pairs are compatible or incompatible. It does NOT explain resonance width (orbs), type-dependent visibility (Drishti), or cross-grid sampling (Nakshatra/Sexagenary). The boundary is clean:

| Domain | Curvature helps? | Better explanation |
|--------|-----------------|-------------------|
| Same-angle opposite-meaning pairs | **Yes** | Path-dependent geodesic |
| Harmonic amplification within a grid | **Yes** | Metric alignment |
| Symmetry incompatibility | **Yes** (proves it) | Number-theoretic theorem |
| Resonance width | **No** | 1/n peak width (flat) |
| Type-dependent visibility | **No** | Flat harmonic (Jupiter) or tabular (Mars/Saturn) |
| Incommensurate divisions | **No** | Multi-grid Nyquist sampling |
