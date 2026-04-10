# Geometric Relationship Catalog

## Purpose

This document catalogs every known system for dividing a 360° phase circle and the relationship types defined at each resolution. It is the foundational reference for ResonanceDB's query algebra. Every entry is stripped of divinatory interpretation — what remains is pure geometry: how many divisions, what angles, what properties.

This catalog is a **data structure specification**. The engine loads it. Adding a tradition or resolution tier requires no code changes — only extending this catalog.

---

## Part 1: Phase Division Systems

Every tradition divides a 360° circle into segments. Different division counts create different resolution tiers for spatial indexing.

### 1.1 Summary Table

| ID | Tradition | System Name | Divisions | Segment Size | Total Buckets | Notes |
|----|-----------|-------------|-----------|-------------|---------------|-------|
| W12 | Western | Zodiac Signs | 12 | 30.000° | 12 | Base resolution |
| W36 | Western | Decans | 36 | 10.000° | 36 | 3 per sign |
| C12 | Chinese | Earthly Branches (地支) | 12 | 30.000° | 12 | Same count as Western, different relationship rules |
| C10 | Chinese | Heavenly Stems (天干) | 10 | 36.000° | 10 | Independent offset cycle |
| C60 | Chinese | Sexagenary (干支) | 60 | 6.000° | 60 | Compound of C12 × C10 (LCM) |
| V12 | Vedic | Rashi | 12 | 30.000° | 12 | Equivalent to W12 |
| V2 | Vedic | Hora (D2) | 2 per sign | 15.000° | 24 | Binary split |
| V3 | Vedic | Drekkana (D3) | 3 per sign | 10.000° | 36 | Same as W36 Decans |
| V4 | Vedic | Chaturthamsa (D4) | 4 per sign | 7.500° | 48 | |
| V7 | Vedic | Saptamsa (D7) | 7 per sign | 4.286° | 84 | |
| V9 | Vedic | Navamsa (D9) | 9 per sign | 3.333° | 108 | Most important sub-chart |
| V10 | Vedic | Dasamsa (D10) | 10 per sign | 3.000° | 120 | |
| V12D | Vedic | Dwadasamsa (D12) | 12 per sign | 2.500° | 144 | |
| V16 | Vedic | Shodasamsa (D16) | 16 per sign | 1.875° | 192 | |
| V20 | Vedic | Vimsamsa (D20) | 20 per sign | 1.500° | 240 | |
| V24 | Vedic | Chaturvimsamsa (D24) | 24 per sign | 1.250° | 288 | |
| V27 | Vedic | Saptavimsamsa (D27) / Nakshatra | 27 total | 13.333° | 27 | Independent division (not per-sign) |
| V30 | Vedic | Trimsamsa (D30) | 30 per sign | 1.000° | 360 | Degree-level precision |
| V40 | Vedic | Khavedamsa (D40) | 40 per sign | 0.750° | 480 | |
| V45 | Vedic | Akshavedamsa (D45) | 45 per sign | 0.667° | 540 | |
| V60 | Vedic | Shashtiamsa (D60) | 60 per sign | 0.500° | 720 | Maximum classical precision |
| V81 | Vedic | Nava-Navamsa (D81) | 81 per sign | 0.370° | 972 | Extended system |
| V108 | Vedic | Ashtottaramsa (D108) | 108 per sign | 0.278° | 1296 | Extended system |
| V144 | Vedic | Dwadas-Dwadamsa (D144) | 144 per sign | 0.208° | 1728 | Extended system |
| M20 | Mayan | Day Signs (Kin) | 20 | 18.000° | 20 | |
| M13 | Mayan | Galactic Tones | 13 | 27.692° | 13 | Independent cycle |
| M260 | Mayan | Tzolkin | 260 | 1.385° | 260 | Compound of M20 × M13 |

### 1.2 Resolution Tiers for Database Indexing

Grouped by effective precision (segment size), these form a hierarchy the query planner selects from:

```
Tier 0 (coarsest):  2 buckets     — 180.00° — Binary classification
Tier 1:             10-13 buckets — 27-36°  — Broad grouping (C10, M13, W12, C12, V12)
Tier 2:             20-27 buckets — 13-18°  — Medium grouping (M20, V27 Nakshatra)
Tier 3:             36-60 buckets — 6-10°   — Fine grouping (W36/V3 Decans, V4, C60)
Tier 4:             84-144 buckets — 2.5-4.3° — High precision (V7, V9, V10, V12D)
Tier 5:             192-360 buckets — 1-1.9° — Very high precision (V16, V20, V24, V30, M260)
Tier 6:             480-720 buckets — 0.5-0.75° — Maximum classical (V40, V45, V60)
Tier 7:             972-1728 buckets — 0.2-0.4° — Extended precision (V81, V108, V144)
```

**Query planner logic:** For exact-match queries, use highest available tier. For broad affinity queries, use Tier 1-2. The planner selects the tier where expected result count makes scan cost worthwhile — same principle as traditional database index selection, but the tiers come from 3000+ years of empirical optimization across civilizations.

---

## Part 2: Relationship Types — Symmetric

These relationships have equal strength in both directions: if A relates to B, then B relates to A with the same strength.

### 2.1 Western Aspects

| ID | Name | Angle | Orb (tolerance) | Harmonic | Strength | DB Operation |
|----|------|-------|-----------------|----------|----------|-------------|
| W-CON | Conjunction | 0° | ±8° | 1st | 1.00 | Exact match |
| W-OPP | Opposition | 180° | ±8° | 2nd | 0.90 | Complement / negation |
| W-TRI | Trine | 120° | ±8° | 3rd | 0.85 | Harmonic family (same element group) |
| W-SQU | Square | 90° | ±7° | 4th | 0.80 | Tension / constraint |
| W-SEX | Sextile | 60° | ±6° | 6th | 0.60 | Weak affinity / opportunity |
| W-SSX | Semi-sextile | 30° | ±2° | 12th | 0.30 | Adjacent / minimal relation |
| W-SSQ | Semi-square | 45° | ±2° | 8th | 0.40 | Mild friction |
| W-SES | Sesquiquadrate | 135° | ±2° | 8th | 0.40 | Mild friction (complement of semi-square) |
| W-QUI | Quincunx | 150° | ±2° | 12th | 0.35 | Inconjunct / adjustment required |
| W-QNT | Quintile | 72° | ±2° | 5th | 0.30 | Creative link |
| W-BQN | Bi-quintile | 144° | ±2° | 5th | 0.30 | Creative link (complement) |

**Harmonic series explanation:** The nth harmonic divides 360° by n. Conjunction = 360°/1. Opposition = 360°/2. Trine = 360°/3. Square = 360°/4. And so on. Lower harmonics = stronger relationships. This is identical to signal processing — fundamental frequency has most energy, overtones progressively less.

**Orb = tolerance window for fuzzy matching.** An angle of 118° still counts as a trine (120° ± 8°). In database terms: range query width around the target angle.

### 2.2 Chinese San He (三合 — Three Harmonies)

Equivalent to Western trine (120° separation), but with explicit element assignment per group.

| Group | Members (Branch positions) | Angular separation | Resulting Element |
|-------|---------------------------|-------------------|-------------------|
| SH-1 | Rat(0°), Dragon(120°), Monkey(240°) | 120° | Water |
| SH-2 | Ox(30°), Snake(150°), Rooster(270°) | 120° | Metal |
| SH-3 | Tiger(60°), Horse(180°), Dog(300°) | 120° | Fire |
| SH-4 | Rabbit(90°), Goat(210°), Pig(330°) | 120° | Wood |

**Properties:** Symmetric. Three-way mutual. Same geometric principle as Western trine — entities 120° apart are harmonically related. But Chinese system explicitly names what the harmony *produces* (an element), which in DB terms is an emergent property from the group.

### 2.3 Chinese Liu Chong (六冲 — Six Clashes)

Direct opposition (180°), equivalent to Western opposition.

| Pair | Branch positions | Angular separation |
|------|-----------------|-------------------|
| LC-1 | Rat(0°) — Horse(180°) | 180° |
| LC-2 | Ox(30°) — Goat(210°) | 180° |
| LC-3 | Tiger(60°) — Monkey(240°) | 180° |
| LC-4 | Rabbit(90°) — Rooster(270°) | 180° |
| LC-5 | Dragon(120°) — Dog(300°) | 180° |
| LC-6 | Snake(150°) — Pig(330°) | 180° |

**Properties:** Symmetric. Pairwise. Direct conflict. Geometrically identical to Western opposition.

---

## Part 3: Relationship Types — Non-Geometric (Structural Pairings)

These are the critical innovation — relationships NOT derivable from angular distance alone. They require explicit lookup tables. This is what makes the catalog more than just "divide circle, measure angles."

### 3.1 Chinese Liu He (六合 — Six Harmonies)

Six specific pairs where compatibility is structurally defined, NOT based on angular proximity.

| Pair | Branch positions | Angular separation | Resulting Element | Geometric rule? |
|------|-----------------|-------------------|-------------------|-----------------|
| LH-1 | Rat(0°) — Ox(30°) | 30° | Earth | NO — 30° is semi-sextile in Western (very weak) |
| LH-2 | Tiger(60°) — Pig(330°) | 90° (or 270°) | Wood | NO — 90° is square in Western (tension) |
| LH-3 | Rabbit(90°) — Dog(300°) | 150° (or 210°) | Fire | NO — 150° is quincunx in Western (adjustment) |
| LH-4 | Dragon(120°) — Rooster(270°) | 150° (or 210°) | Metal | NO — same quincunx angle |
| LH-5 | Snake(150°) — Monkey(240°) | 90° (or 270°) | Water | NO — square angle |
| LH-6 | Horse(180°) — Goat(210°) | 30° | Earth | NO — semi-sextile angle |

**This is the most important finding in the catalog.** Liu He pairs have strong harmony at angles that Western geometry classifies as weak or tense. The relationship exists not because of WHERE entities sit on the circle, but because of HOW they combine (Yin-Yang pairing — each pair has one Yang and one Yin branch).

**DB implication:** You need a lookup table for structural pairings that overrides or supplements geometric distance. This is like having a "foreign key" relationship that isn't derivable from column values — it must be declared.

### 3.2 Chinese Liu Hai (六害 — Six Harms)

Structurally defined conflict pairs, also non-geometric.

| Pair | Branch positions | Angular separation |
|------|-----------------|-------------------|
| HA-1 | Rat(0°) — Goat(210°) | 150° |
| HA-2 | Ox(30°) — Horse(180°) | 150° |
| HA-3 | Tiger(60°) — Snake(150°) | 90° |
| HA-4 | Rabbit(90°) — Dragon(120°) | 30° |
| HA-5 | Monkey(240°) — Pig(330°) | 90° |
| HA-6 | Rooster(270°) — Dog(300°) | 30° |

**Properties:** Symmetric. Pairwise. Harm/friction. Mixed angular distances — no single geometric rule explains all pairs.

### 3.3 Chinese Xiang Xing (相刑 — Punishments)

Multi-entity destructive patterns. Some triangular, some pairwise, one self-referential.

| Pattern | Members | Type | Angular pattern |
|---------|---------|------|----------------|
| XP-1 | Rat(0°) — Rabbit(90°) | Ungrateful punishment | 90° pair |
| XP-2 | Tiger(60°), Snake(150°), Monkey(240°) | Persistent punishment | Triangle at 90°/90°/180° |
| XP-3 | Ox(30°), Goat(210°), Dog(300°) | Bullying punishment | Triangle at 180°/90°/90° |
| XP-4 | Dragon(120°) self | Self-punishment | 0° |
| XP-5 | Horse(180°) self | Self-punishment | 0° |
| XP-6 | Rooster(270°) self | Self-punishment | 0° |
| XP-7 | Pig(330°) self | Self-punishment | 0° |

**DB implication:** Self-punishment = entity self-conflict detection. Triangular punishments = three-body problem patterns. These are compound constraints — multi-entity queries that test for specific destructive configurations.

### 3.4 Chinese San Hui Fang (三会方 — Seasonal Groupings)

Three consecutive branches forming seasonal affinity groups.

| Group | Members | Angular span | Resulting Element |
|-------|---------|-------------|-------------------|
| SF-1 | Tiger(60°), Rabbit(90°), Dragon(120°) | 60° span | Wood (Spring) |
| SF-2 | Snake(150°), Horse(180°), Goat(210°) | 60° span | Fire (Summer) |
| SF-3 | Monkey(240°), Rooster(270°), Dog(300°) | 60° span | Metal (Autumn) |
| SF-4 | Pig(330°), Rat(0°), Ox(30°) | 60° span | Water (Winter) |

**Properties:** Three adjacent positions (30° intervals). Spatial proximity grouping. In DB terms: range bucket — entities within a 60° arc are in the same "season."

### 3.5 Chinese Heavenly Stems Wu He (天干五合 — Five Combinations)

Pairwise combinations of the 10 Heavenly Stems (36° cycle).

| Pair | Stem positions | Angular separation | Resulting Element |
|------|---------------|-------------------|-------------------|
| WH-1 | Jia(0°) — Ji(180°) | 180° | Earth |
| WH-2 | Yi(36°) — Geng(216°) | 180° | Metal |
| WH-3 | Bing(72°) — Xin(252°) | 180° | Water |
| WH-4 | Ding(108°) — Ren(288°) | 180° | Wood |
| WH-5 | Wu(144°) — Gui(324°) | 180° | Fire |

**Properties:** All at 180° on the 10-division cycle. Opposition on the Stems cycle produces combination (not clash). This is the opposite of Liu Chong on the Branches cycle, where 180° means clash. Same angle, different cycle, opposite meaning. The cycle determines interpretation.

---

## Part 4: Relationship Types — Asymmetric

These relationships have different strength or reach depending on the entity's TYPE, not just its position.

### 4.1 Vedic Drishti (दृष्टि — Planetary Aspects)

In Vedic astrology, different planet types "see" different angles. All see 180° (opposition). Some see additional angles.

| Entity Type | Aspects (angles seen) | Strength per angle |
|-------------|----------------------|-------------------|
| Default (all) | 180° | Full (1.0) |
| Mars-type | 90° (4th), 180° (7th), 210° (8th) | Full at all three |
| Jupiter-type | 120° (5th), 180° (7th), 240° (9th) | Full at all three |
| Saturn-type | 60° (3rd), 180° (7th), 270° (10th) | Full at all three |

**Critical property:** These are ONE-DIRECTIONAL. Mars at 0° aspects (sees) an entity at 90°, but that entity at 90° does NOT automatically aspect Mars back (unless it is also a Mars-type entity whose own Drishti covers the reverse angle).

**DB implication:** Query reach depends on the querying entity's type. A "security" entity might see relationships that a "UI" entity cannot, because its type grants it broader aspect angles. This is not about what exists in the data — it's about what each entity type is ALLOWED to find.

### 4.2 Chinese Wu Xing (五行 — Five Phases / Elements)

Directed cycles between five elements. Two separate cycle types.

**Generative Cycle (相生 — Sheng):**

```
Wood(0°) → Fire(72°) → Earth(144°) → Metal(216°) → Water(288°) → Wood(0°)
```

| From | To | Direction | Angle step |
|------|----|-----------|-----------|
| Wood | Fire | Generates | +72° |
| Fire | Earth | Generates | +72° |
| Earth | Metal | Generates | +72° |
| Metal | Water | Generates | +72° |
| Water | Wood | Generates | +72° |

**Destructive Cycle (相克 — Ke):**

```
Wood(0°) → Earth(144°) → Water(288°) → Fire(72°) → Metal(216°) → Wood(0°)
```

| From | To | Direction | Angle step |
|------|----|-----------|-----------|
| Wood | Earth | Destroys | +144° |
| Earth | Water | Destroys | +144° |
| Water | Fire | Destroys | +144° |
| Fire | Metal | Destroys | +144° |
| Metal | Wood | Destroys | +144° |

**Properties:** Both cycles are DIRECTED — A generates B does NOT mean B generates A. Generative = +72° step (quintile). Destructive = +144° step (bi-quintile). These are 5th-harmonic relationships. Note that Western astrology assigns quintile/bi-quintile low strength (0.30), but in Chinese system these are THE fundamental relationship types.

**DB implication:** Dependency chains. A enables B (generative). A conflicts with C (destructive). Can traverse: "what does upgrading A enable transitively?" (follow generative chain). "What does removing A break?" (follow destructive chain in reverse).

---

## Part 5: Relationship Types — Context-Dependent

Relationship strength changes based on external context, not just the entities involved.

### 5.1 Hellenistic Essential Dignities

Entity strength varies by WHICH DOMAIN (sign) it occupies. Same entity, different context, different power.

| Dignity Level | Strength Multiplier | Description |
|--------------|-------------------|-------------|
| Domicile | 1.00 | Entity in its home domain — maximum relevance |
| Exaltation | 0.90 | Entity in a domain where it excels — very strong |
| Triplicity | 0.70 | Entity in a sympathetic domain — moderate strength |
| Term | 0.50 | Entity in a partially compatible domain — reduced |
| Face/Decan | 0.30 | Entity has minimal foothold — weak |
| Peregrine | 0.15 | Entity in a domain where it has no standing — very weak |
| Detriment | 0.10 | Entity in opposite of its home domain — dysfunction |
| Fall | 0.05 | Entity in opposite of its exaltation — severe weakness |

**Properties:** Strength is a function of (entity_type, domain). Not pairwise between entities — it modifies one entity's individual power before relationship calculations. A "security" entity in the "infrastructure" domain might be at Domicile (1.0), but in the "marketing" domain at Peregrine (0.15).

**DB implication:** Domain-weighted relevance scoring. Query results ranked not just by phase coherence but by how relevant each entity is IN THE CURRENT QUERY CONTEXT. Equivalent to TF-IDF where the "document" is the domain.

### 5.2 Hellenistic Sect

Binary context switch that modifies the entire query algebra.

| Context | Name | Effect |
|---------|------|--------|
| Mode A | Diurnal (Day) | Certain entity types gain strength, others lose |
| Mode B | Nocturnal (Night) | Reversed — entities that were strong become weak |

**Implementation:** Two sets of dignity modifiers. System has a current mode. Query results weighted differently depending on active mode.

**DB implication:** Think "edit mode" vs "read mode" in a CMS — different relationships are prominent. Or "development" vs "production" context — same entities, different relationship weights. The mode is not per-entity but per-session or per-query.

### 5.3 Hellenistic Reception (Mutual)

When two entities are each in the other's "home domain," their relationship amplifies.

| Pattern | Condition | Amplification |
|---------|-----------|--------------|
| Mutual Reception | A in B's domicile AND B in A's domicile | Coherence × 1.5 |
| Mixed Reception | A in B's domicile OR B in A's domicile (one-way) | Coherence × 1.2 |
| Generosity | A in B's exaltation (one-way) | Coherence × 1.1 |

**DB implication:** Bidirectional reference detection. When entity A references entity B AND entity B references entity A, the relationship is stronger than either unidirectional reference alone. This is computable from the data — no lookup table needed, just cross-reference detection.

---

## Part 6: Compound Pattern Templates

Multi-clause conditions that, when all met simultaneously, assert an emergent property.

### 6.1 Vedic Yogas (योग — Combinations)

Named patterns with specific preconditions and emergent results. These are essentially MATERIALIZED VIEW definitions triggered by pattern matching over the field.

**Structure of a Yoga:**

```
name: string
clauses: [
  { entity_type: X, condition: <position/relationship/dignity> },
  { entity_type: Y, condition: <position/relationship/dignity> },
  ...
]
emergent_property: string
strength: float
```

**Example patterns (translated to database concepts):**

| Pattern Name | Clauses | Emergent Property |
|-------------|---------|-------------------|
| Hub Node | Entity has ≥5 inbound refs AND ≥3 outbound refs AND has category tag | "hub" status |
| Bridge | Entity connects two otherwise-unconnected clusters | "bridge" status |
| Orphan | Entity has 0 inbound refs AND 0 outbound refs AND age > threshold | "orphan" status |
| Authority Chain | A refs B, B refs C, C is tagged "authoritative" | A inherits "authority-linked" |
| Mutual Cluster | A↔B, B↔C, C↔A (triangle of mutual refs) | All three gain "cluster-member" |

**Properties:** Reactive — when any entity changes state, all Yoga patterns that reference its type are re-evaluated. If pattern matches, emergent property is asserted. If pattern breaks, property is retracted. This is reactive materialized views driven by the relationship field.

---

## Part 7: Directed Cycle Systems

Complete cycles where position determines role in a chain.

### 7.1 Wu Xing Cycles (detailed)

Five positions forming two interlocking directed graphs:

```
Positions: 0°, 72°, 144°, 216°, 288° (5 equally spaced)

Generative cycle (step = +1 position):
  0 → 1 → 2 → 3 → 4 → 0

Destructive cycle (step = +2 positions):
  0 → 2 → 4 → 1 → 3 → 0

Weakening cycle (reverse of generative, step = -1):
  0 → 4 → 3 → 2 → 1 → 0

Controlling cycle (reverse of destructive, step = -2):
  0 → 3 → 1 → 4 → 2 → 0
```

**Four relationship types from one 5-position system:**

| Relationship | Step | Meaning | DB Operation |
|-------------|------|---------|-------------|
| Generates | +1 | A enables/feeds B | Dependency (A required for B) |
| Destroys | +2 | A conflicts with/overrides C | Conflict detection |
| Weakens | -1 | A drains/exhausts E | Resource consumption |
| Controls | -2 | A is regulated/limited by D | Constraint enforcement |

### 7.2 Mayan Tzolkin Relationships

260-position cycle (20 × 13) with specific relationship rules:

| Relationship | Offset | Count of occurrences |
|-------------|--------|---------------------|
| Same Day Sign | multiples of 20 | 13 per sign |
| Same Tone | multiples of 13 | 20 per tone |
| Polar Kin | +130 positions | Opposite in cycle |
| Analog Kin | calculated per sign | Supportive partner |
| Antipode Kin | calculated per sign | Challenge partner |
| Occult Kin | calculated per sign | Hidden helper |
| Guide Kin | tone-dependent | Directional influence |

**Properties:** Multi-dimensional cycle. Position is (sign, tone) tuple. Different relationship types emerge from different dimensional projections of the same position.

---

## Part 8: The Earthly Branches — Complete Position Table

This is the master reference for Chinese system calculations. All angular positions assume Rat = 0°, proceeding counterclockwise in 30° increments.

| Position | Branch | Animal | Element | Yin/Yang | Angle |
|----------|--------|--------|---------|----------|-------|
| 1 | 子 Zi | Rat | Water | Yang | 0° |
| 2 | 丑 Chou | Ox | Earth | Yin | 30° |
| 3 | 寅 Yin | Tiger | Wood | Yang | 60° |
| 4 | 卯 Mao | Rabbit | Wood | Yin | 90° |
| 5 | 辰 Chen | Dragon | Earth | Yang | 120° |
| 6 | 巳 Si | Snake | Fire | Yin | 150° |
| 7 | 午 Wu | Horse | Fire | Yang | 180° |
| 8 | 未 Wei | Goat | Earth | Yin | 210° |
| 9 | 申 Shen | Monkey | Metal | Yang | 240° |
| 10 | 酉 You | Rooster | Metal | Yin | 270° |
| 11 | 戌 Xu | Dog | Earth | Yang | 300° |
| 12 | 亥 Hai | Pig | Water | Yin | 330° |

### Complete Relationship Matrix for Earthly Branches

For each branch, every relationship type it participates in:

| Branch | San He (120°) | Liu He (pair) | Liu Chong (180°) | Liu Hai (harm) | Xing (punishment) | San Hui (season) |
|--------|--------------|---------------|------------------|---------------|-------------------|-----------------|
| Rat(0°) | Dragon, Monkey | Ox | Horse | Goat | Rabbit | Pig, Ox |
| Ox(30°) | Snake, Rooster | Rat | Goat | Horse | Goat, Dog | Pig, Rat |
| Tiger(60°) | Horse, Dog | Pig | Monkey | Snake | Snake, Monkey | Rabbit, Dragon |
| Rabbit(90°) | Goat, Pig | Dog | Rooster | Dragon | Rat | Tiger, Dragon |
| Dragon(120°) | Rat, Monkey | Rooster | Dog | Rabbit | Self | Tiger, Rabbit |
| Snake(150°) | Ox, Rooster | Monkey | Pig | Tiger | Tiger, Monkey | Horse, Goat |
| Horse(180°) | Tiger, Dog | Goat | Rat | Ox | Self | Snake, Goat |
| Goat(210°) | Rabbit, Pig | Horse | Ox | Rat | Ox, Dog | Snake, Horse |
| Monkey(240°) | Rat, Dragon | Snake | Tiger | Pig | Tiger, Snake | Rooster, Dog |
| Rooster(270°) | Ox, Snake | Dragon | Rabbit | Dog | Self | Monkey, Dog |
| Dog(300°) | Tiger, Horse | Rabbit | Dragon | Rooster | Ox, Goat | Monkey, Rooster |
| Pig(330°) | Rabbit, Goat | Tiger | Snake | Monkey | Self | Rat, Ox |

---

## Part 9: The Heavenly Stems — Complete Position Table

| Position | Stem | Element | Yin/Yang | Angle (on 10-division circle) |
|----------|------|---------|----------|-------------------------------|
| 1 | 甲 Jia | Wood | Yang | 0° |
| 2 | 乙 Yi | Wood | Yin | 36° |
| 3 | 丙 Bing | Fire | Yang | 72° |
| 4 | 丁 Ding | Fire | Yin | 108° |
| 5 | 戊 Wu | Earth | Yang | 144° |
| 6 | 己 Ji | Earth | Yin | 180° |
| 7 | 庚 Geng | Metal | Yang | 216° |
| 8 | 辛 Xin | Metal | Yin | 252° |
| 9 | 壬 Ren | Water | Yang | 288° |
| 10 | 癸 Gui | Water | Yin | 324° |

### Stem Combinations (Wu He)

| Pair | Stems | Angle separation (on 10-cycle) | Product |
|------|-------|-------------------------------|---------|
| 1 | Jia(0°) + Ji(180°) | 180° | Earth |
| 2 | Yi(36°) + Geng(216°) | 180° | Metal |
| 3 | Bing(72°) + Xin(252°) | 180° | Water |
| 4 | Ding(108°) + Ren(288°) | 180° | Wood |
| 5 | Wu(144°) + Gui(324°) | 180° | Fire |

### Stem Clashes (Xiang Chong)

| Pair | Stems | Relationship |
|------|-------|-------------|
| 1 | Jia(Wood Yang) vs Geng(Metal Yang) | Metal destroys Wood |
| 2 | Yi(Wood Yin) vs Xin(Metal Yin) | Metal destroys Wood |
| 3 | Bing(Fire Yang) vs Ren(Water Yang) | Water destroys Fire |
| 4 | Ding(Fire Yin) vs Gui(Water Yin) | Water destroys Fire |

---

## Part 10: Implementation Data Structures

### 10.1 Division System

```rust
struct DivisionSystem {
    id: &'static str,        // "W12", "V9", "C60", etc.
    tradition: Tradition,     // Western, Vedic, Chinese, Mayan, Hellenistic
    divisions: u32,           // Total bucket count
    segment_degrees: f64,     // 360.0 / divisions
    per_sign: bool,           // true = divisions per 30° sign, false = divisions of full 360°
    tier: u8,                 // Resolution tier (0-7)
}
```

### 10.2 Relationship Type

```rust
struct RelationshipType {
    id: &'static str,         // "W-TRI", "LH-3", "WX-GEN", etc.
    tradition: Tradition,
    category: RelationshipCategory,  // Symmetric, NonGeometric, Asymmetric, ContextDependent, DirectedCycle
    
    // For geometric relationships:
    angle: Option<f64>,       // Target angle in degrees (None for non-geometric)
    orb: Option<f64>,         // Tolerance window (None for exact structural pairings)
    harmonic: Option<u32>,    // Which harmonic (None for non-harmonic)
    
    // For structural pairings (Liu He, Liu Hai, etc.):
    pairs: Option<Vec<(u32, u32)>>,  // Explicit pair list by position index
    
    // For asymmetric (Drishti):
    entity_type_dependent: bool,
    type_angles: Option<HashMap<EntityType, Vec<f64>>>,
    
    // For directed cycles (Wu Xing):
    cycle_step: Option<i32>,  // +1 generative, +2 destructive, -1 weakening, -2 controlling
    
    // Common:
    base_strength: f64,       // 0.0 to 1.0
    symmetric: bool,
    description: &'static str,  // Human-readable DB operation name
}
```

### 10.3 Compound Pattern (Yoga)

```rust
struct CompoundPattern {
    name: String,
    clauses: Vec<PatternClause>,
    emergent_property: String,
    strength: f64,
}

struct PatternClause {
    entity_type: Option<EntityType>,  // None = any type
    condition: ClauseCondition,
}

enum ClauseCondition {
    MinInboundRefs(u32),
    MinOutboundRefs(u32),
    HasTag(String),
    InDomain(String),
    RelatedTo { target: EntityRef, relationship: String },
    DignityAtLeast(f64),
    InPhaseRange { start: f64, end: f64 },
}
```

### 10.4 Context Modifier

```rust
struct ContextModifier {
    mode: String,                           // "diurnal", "nocturnal", "edit", "view", etc.
    dignity_adjustments: HashMap<(EntityType, String), f64>,  // (type, domain) → multiplier
    relationship_weight_overrides: HashMap<String, f64>,       // relationship_id → weight override
}
```

---

## Part 11: Cross-Tradition Synthesis

### Operations That Multiple Traditions Agree On

| Operation | Western | Chinese | Vedic | Angle | Consensus |
|-----------|---------|---------|-------|-------|-----------|
| Exact match | Conjunction | — | Conjunction | 0° | Universal |
| Complement | Opposition | Liu Chong | Drishti (all types) | 180° | Universal |
| Harmonic family | Trine | San He | — | 120° | Strong (2/3) |
| Tension/constraint | Square | — | Mars Drishti | 90° | Moderate |
| Weak affinity | Sextile | — | Saturn Drishti | 60° | Moderate |

### Operations Unique to Each Tradition

| Operation | Tradition | What it provides | No equivalent elsewhere |
|-----------|-----------|-----------------|----------------------|
| Non-geometric pairing | Chinese Liu He | Compatibility not from position but from type-matching | ✓ |
| Directed dependency chain | Chinese Wu Xing | A→B→C→D→E→A generative/destructive cycles | ✓ |
| Type-dependent query reach | Vedic Drishti | Different entity types see different relationship angles | ✓ |
| Multi-resolution indexing | Vedic Vargas | Same data at 12/27/108/720 bucket resolutions | ✓ |
| Compound reactive patterns | Vedic Yogas | Multi-clause patterns that assert emergent properties | ✓ |
| Domain-dependent relevance | Hellenistic Dignities | Same entity, different strength per domain | ✓ |
| Binary context switching | Hellenistic Sect | Global mode changes which relationships are prominent | ✓ |
| Bidirectional amplification | Hellenistic Reception | Mutual references strengthen both directions | ✓ |
| Harm relationships | Chinese Liu Hai | Friction pairs distinct from opposition | ✓ |
| Punishment patterns | Chinese Xiang Xing | Multi-body destructive configurations including self-conflict | ✓ |
| Seasonal proximity | Chinese San Hui | Adjacent-position grouping into functional clusters | ✓ |
| Dual-cycle compound identity | Chinese Sexagenary | Two independent phase angles combine into 60-position space | ✓ |

---

## Part 12: Naming Convention for Implementation

The engine does NOT use astrological names in its API. This table maps tradition-specific names to database operation names.

| Tradition Name | Engine Operation Name | Function Signature |
|---------------|---------------------|-------------------|
| Conjunction | `exact_match` | `query_exact(field, attribute, value)` |
| Opposition | `complement` | `query_complement(field, entity_id)` |
| Trine | `harmonic_family` | `query_harmonic(field, entity_id, harmonic: 3)` |
| Square | `tension_link` | `query_harmonic(field, entity_id, harmonic: 4)` |
| Sextile | `weak_affinity` | `query_harmonic(field, entity_id, harmonic: 6)` |
| San He | `triad_group` | `query_triad(field, entity_id)` |
| Liu He | `structural_pair` | `query_structural_pair(field, entity_id, pair_table: "liu_he")` |
| Liu Chong | `structural_clash` | `query_structural_pair(field, entity_id, pair_table: "liu_chong")` |
| Liu Hai | `structural_harm` | `query_structural_pair(field, entity_id, pair_table: "liu_hai")` |
| Xiang Xing | `destructive_pattern` | `query_pattern(field, entity_id, pattern_table: "xiang_xing")` |
| Wu Xing (gen) | `dependency_chain` | `query_directed_cycle(field, entity_id, step: +1)` |
| Wu Xing (dest) | `conflict_chain` | `query_directed_cycle(field, entity_id, step: +2)` |
| Drishti | `typed_reach` | `query_typed_reach(field, entity_id, entity_type)` |
| Nakshatra | `bucket_at_resolution` | `index_at_resolution(field, tier: 2)` |
| Navamsa | `bucket_at_resolution` | `index_at_resolution(field, tier: 4)` |
| Yoga | `compound_pattern` | `evaluate_pattern(field, pattern_def)` |
| Dignity | `domain_relevance` | `score_in_domain(entity_id, domain)` |
| Sect | `context_mode` | `set_query_context(mode)` |
| Reception | `mutual_reference` | `detect_mutual_refs(field, entity_id)` |

---

## Part 13: What to Test First (Proof-of-Concept Priority)

### Phase 1: Core geometry (validates encoding)

1. **Exact match (0°)** — conjunction. If this doesn't return identical results to `SELECT WHERE`, encoding is broken.
2. **Complement (180°)** — opposition. Tests that phase distance calculation works.
3. **Harmonic family (120°)** — trine/San He. Tests that harmonic grouping works.

### Phase 2: Non-geometric relationships (validates the unique value)

4. **Structural pairing (Liu He)** — tests lookup-table relationships that override geometry.
5. **Directed cycle (Wu Xing)** — tests chain traversal.

### Phase 3: Advanced features (validates scalability of the algebra)

6. **Multi-resolution indexing (Vedic Vargas)** — tests that Tier 1 vs Tier 4 actually improves query performance.
7. **Type-dependent reach (Drishti)** — tests asymmetric query patterns.
8. **Domain relevance (Dignity)** — tests context-weighted scoring.
9. **Compound patterns (Yoga)** — tests reactive pattern matching.

### Phase 4: Integration

10. **All of the above composed in a single query** — "Find entities that are in the same harmonic family as X, compatible by structural pairing with Y, not in conflict chain with Z, and have domain relevance > 0.5 in the current context."

---

*This catalog contains 5 traditions, 26 division systems, 35+ relationship types, 4 cycle systems, and 8 context modifiers. Together they form a comprehensive vocabulary of structural relationship classes documented across the traditions surveyed, applicable to phase-encoded databases.*
