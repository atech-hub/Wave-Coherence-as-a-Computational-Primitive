// Curvature Test 3: The Geometric Comma
//
// QUESTION: Is the San He / Wu Xing incompatibility a NECESSARY mathematical
// impossibility, or an artifact of the optimizer?
//
// ANSWER (analytical proof by contradiction):
// It is NECESSARY. No metric on the 12-segment circle can make both
// Wu Xing (all generative pairs at 72° geodesic) and ALL four San He triads
// (all trine pairs at 120° geodesic) simultaneously exact.
//
// Specifically: Triad 3 (Tiger-Horse-Dog) is incompatible with Wu Xing.
// Triads 1, 2, 4 are all compatible.
//
// The deficit = 24° = 2 × 72° - 120°.
// This is the geometric comma — the exact analog of the Pythagorean comma
// in music theory (where 12 perfect fifths overshoot 7 octaves by ~23 cents).
//
// Here: 2 quintile steps (72° each) overshoot 1 trine step (120°) by 24°.
// The 3-fold and 5-fold harmonics CANNOT coexist on one curved 12-segment
// circle. This explains why the Chinese system treats San He and Wu Xing
// as separate frameworks that interact but don't unify.
//
// COMPILE: rustc curvature_test3_geometric_comma.rs -o curvature_test3
// RUN:     ./curvature_test3

use std::f64::consts::PI;

const N_POS: usize = 12;
const SEG_DEG: f64 = 30.0;

const NAMES: [&str; 12] = [
    "Rat", "Ox", "Tiger", "Rabbit", "Dragon", "Snake",
    "Horse", "Goat", "Monkey", "Rooster", "Dog", "Pig",
];

// San He triads (4 groups of 3 positions, 120° apart on flat circle)
const TRIAD_NAMES: [&str; 4] = [
    "T1: Rat-Dragon-Monkey",
    "T2: Ox-Snake-Rooster",
    "T3: Tiger-Horse-Dog",
    "T4: Rabbit-Goat-Pig",
];

// Each triad has 3 legs (4 segments each, clockwise)
// Leg segments: position a to position a+4, traversing segments {a, a+1, a+2, a+3}
const TRIADS: [[(usize, usize); 3]; 4] = [
    [(0, 4), (4, 8), (8, 0)],     // T1: Rat→Dragon→Monkey→Rat
    [(1, 5), (5, 9), (9, 1)],     // T2: Ox→Snake→Rooster→Ox
    [(2, 6), (6, 10), (10, 2)],   // T3: Tiger→Horse→Dog→Tiger
    [(3, 7), (7, 11), (11, 3)],   // T4: Rabbit→Goat→Pig→Rabbit
];

// Wu Xing group structure on the 12 segments
// Groups: {0,1}, {2,3,4}, {5,6}, {7,8,9}, {10,11}
// Each sum = 2.4 for 72° geodesic
const WX_GROUPS: [(usize, usize); 5] = [
    (0, 2),   // segments 0,1
    (2, 5),   // segments 2,3,4
    (5, 7),   // segments 5,6
    (7, 10),  // segments 7,8,9
    (10, 12), // segments 10,11
];

const WX_GROUP_LABELS: [&str; 5] = [
    "g0+g1", "g2+g3+g4", "g5+g6", "g7+g8+g9", "g10+g11",
];

// Liu He / Liu Hai (from Tests 1-2)
const LIU_HE: [(usize, usize); 6] = [
    (0, 1), (2, 11), (3, 10), (4, 9), (5, 8), (6, 7),
];
const LIU_HAI: [(usize, usize); 6] = [
    (0, 7), (1, 6), (2, 5), (3, 4), (8, 11), (9, 10),
];

// Wu Xing generative pairs
const WX_GEN: [(usize, usize); 5] = [
    (0, 2), (2, 5), (5, 7), (7, 10), (10, 0),
];

// --- Geometry ---

fn forward_distance(a: usize, b: usize, g: &[f64; 12]) -> f64 {
    let mut dist = 0.0;
    let mut pos = a;
    while pos != b {
        dist += g[pos] * SEG_DEG;
        pos = (pos + 1) % N_POS;
    }
    dist
}

fn geodesic_distance(a: usize, b: usize, g: &[f64; 12]) -> f64 {
    if a == b { return 0.0; }
    forward_distance(a, b, g).min(forward_distance(b, a, g))
}

fn coherence(distance_deg: f64, n: u32) -> f64 {
    (n as f64 * distance_deg * PI / 180.0).cos()
}

fn mean_coherence_6(pairs: &[(usize, usize); 6], g: &[f64; 12], n: u32) -> f64 {
    pairs.iter()
        .map(|&(a, b)| coherence(geodesic_distance(a, b, g), n))
        .sum::<f64>() / 6.0
}

fn format_metric(g: &[f64; 12]) -> String {
    let parts: Vec<String> = g.iter().map(|x| format!("{:.3}", x)).collect();
    format!("[{}]", parts.join(", "))
}

/// Get the segments on the clockwise 4-segment leg from position a to a+4
fn leg_segments(start: usize) -> [usize; 4] {
    [start % 12, (start + 1) % 12, (start + 2) % 12, (start + 3) % 12]
}

/// Sum of metric weights for a 4-segment leg
fn leg_sum(start: usize, g: &[f64; 12]) -> f64 {
    let segs = leg_segments(start);
    segs.iter().map(|&s| g[s]).sum()
}

/// Which Wu Xing groups does a leg overlap?
fn leg_wx_groups(start: usize) -> Vec<(usize, Vec<usize>)> {
    let segs = leg_segments(start);
    let mut groups = Vec::new();
    for (gi, &(gs, ge)) in WX_GROUPS.iter().enumerate() {
        let overlap: Vec<usize> = segs.iter()
            .filter(|&&s| {
                let s_mod = s;
                if gs < ge { s_mod >= gs && s_mod < ge }
                else { s_mod >= gs || s_mod < ge }
            })
            .cloned()
            .collect();
        if !overlap.is_empty() {
            groups.push((gi, overlap));
        }
    }
    groups
}

// --- Main phases ---

fn phase1_constraint_setup() {
    println!("--- Phase 1: Constraint Systems ---\n");

    println!("  WU XING constraints (for 72° geodesic per generative pair):");
    println!("  Each group of segments sums to 2.4 (= 72°/30°):");
    for i in 0..5 {
        let (start, end) = WX_GROUPS[i];
        let segs: Vec<String> = (start..end).map(|s| format!("g{}", s % 12)).collect();
        println!("    {} = {} = 2.4", WX_GROUP_LABELS[i], segs.join("+"));
    }
    println!("    Total: 5 × 2.4 = 12.0 (= normalization) ✓\n");

    println!("  SAN HE constraints (for 120° geodesic per trine leg):");
    println!("  Each leg of 4 segments sums to 4.0 (= 120°/30°):");
    for t in 0..4 {
        println!("    {}:", TRIAD_NAMES[t]);
        for &(a, b) in TRIADS[t].iter() {
            let segs = leg_segments(a);
            let seg_str: Vec<String> = segs.iter().map(|s| format!("g{}", s)).collect();
            println!("      {}→{}: {} = 4.0",
                     NAMES[a], NAMES[b], seg_str.join("+"));
        }
    }
    println!();
}

fn phase2_compatibility_check() {
    println!("--- Phase 2: Per-Triad Compatibility with Wu Xing ---\n");

    for t in 0..4 {
        println!("  {}:", TRIAD_NAMES[t]);

        let mut compatible = true;
        for &(a, _b) in TRIADS[t].iter() {
            let segs = leg_segments(a);
            let seg_str: Vec<String> = segs.iter().map(|s| format!("g{}", s)).collect();

            // Compute the forced sum under Wu Xing constraints
            let groups = leg_wx_groups(a);
            let mut forced_sum = 0.0f64;
            let mut group_parts: Vec<String> = Vec::new();
            let mut is_fully_determined = true;

            for (gi, overlap) in &groups {
                let (gs, ge) = WX_GROUPS[*gi];
                let group_size = if ge > gs { ge - gs } else { 12 - gs + ge };
                if overlap.len() == group_size {
                    // Entire group is in this leg
                    forced_sum += 2.4;
                    group_parts.push(format!("{}=2.4", WX_GROUP_LABELS[*gi]));
                } else {
                    // Partial group — not fully determined
                    is_fully_determined = false;
                    let partial: Vec<String> = overlap.iter().map(|s| format!("g{}", s)).collect();
                    group_parts.push(format!("({})", partial.join("+")));
                }
            }

            if is_fully_determined {
                let status = if (forced_sum - 4.0).abs() < 1e-10 {
                    "= 4.0 ✓"
                } else {
                    compatible = false;
                    if forced_sum > 4.0 {
                        "≠ 4.0 ✗ EXCESS"
                    } else {
                        "≠ 4.0 ✗ DEFICIT"
                    }
                };
                println!("      {} = {} = {:.1} {}",
                         seg_str.join("+"), group_parts.join(" + "),
                         forced_sum, status);
            } else {
                println!("      {} = {} (partially free)",
                         seg_str.join("+"), group_parts.join(" + "));
            }
        }

        let verdict = if compatible { "COMPATIBLE ✓" } else { "INCOMPATIBLE ✗" };
        println!("    Verdict: {}\n", verdict);
    }
}

fn phase3_proof_and_comma() {
    println!("--- Phase 3: The Geometric Comma ---\n");

    println!("  THE CONTRADICTION (Triad 3, leg Dog→Tiger):");
    println!("    Dog(10) → Tiger(2): segments {{10, 11, 0, 1}}");
    println!("    Wu Xing forces: g10+g11 = 2.4 (Dog→Rat group)");
    println!("                     g0+g1  = 2.4 (Rat→Tiger group)");
    println!("    Forced sum: 2.4 + 2.4 = 4.8");
    println!("    Required for 120°: 4.0");
    println!("    EXCESS: 0.8 weight-units = 24°\n");

    println!("  THE COMMA:");
    println!("    2 × (Wu Xing step) - (San He step) = 2 × 72° - 120° = 24°");
    println!("    This is EXACT. Not an approximation. Not an optimizer artifact.");
    println!("    It is a theorem: no metric on a 12-segment circle can make both");
    println!("    the 5th harmonic (72°) and the 3rd harmonic (120°) exact for all");
    println!("    positions simultaneously.\n");

    println!("  WHY IT'S NECESSARY:");
    println!("    The 12-segment circle is divided by Wu Xing into groups:");
    println!("    {{0,1}}, {{2,3,4}}, {{5,6}}, {{7,8,9}}, {{10,11}}");
    println!("    Group sizes alternate: 2, 3, 2, 3, 2 segments.");
    println!("    A San He trine leg always spans 4 segments.");
    println!("    When a leg spans TWO 2-segment groups:");
    println!("      forced sum = 2.4 + 2.4 = 4.8 (= 144°)");
    println!("      needed sum = 4.0 (= 120°)");
    println!("      excess = 0.8 (= 24°)");
    println!("    This happens for EXACTLY one triad: Tiger-Horse-Dog.\n");

    println!("  THE PYTHAGOREAN ANALOGY:");
    println!("    In music: 12 perfect fifths overshoot 7 octaves by ~23.46 cents.");
    println!("      (3/2)^12 / 2^7 ≈ 1.01364");
    println!("    In geometry: 2 quintile steps overshoot 1 trine step by 24°.");
    println!("      2 × 72° - 120° = 24°  (= 360°/15 = 6.67% of circle)");
    println!("    Both: two rational divisions of a circle that cannot coexist exactly.");
    println!("    Both: known empirically for centuries, explained by number theory.\n");

    // Numeric verification
    let comma_deg = 2.0 * 72.0 - 120.0;
    let comma_fraction = comma_deg / 360.0;
    println!("  NUMERIC VERIFICATION:");
    println!("    Comma = 2 × 72° - 120° = {:.1}°", comma_deg);
    println!("    Comma / 360° = {:.6} = 1/{:.0}", comma_fraction, 1.0 / comma_fraction);
    println!("    = 360° / lcm(3,5) = 360° / 15 = 24° ✓");
    println!();
}

fn phase4_best_compromise() -> [f64; 12] {
    println!("--- Phase 4: Best Compromise Metric ---\n");

    println!("  Satisfying: Wu Xing + Triads 1, 2, 4 (all except T3)");
    println!("  The constraints fully determine 11 of 12 weights:\n");

    // Derived analytically:
    // From WX + T1: g2+g3 = 1.6, g4 = 0.8, g7 = 0.8, g8+g9 = 1.6
    // From WX + T2: g0 = 0.8, g1 = 1.6, g8 = 0.8, g9 = 0.8
    // From WX + T4: g2 = 0.8, g3 = 0.8, g10 = 1.6, g11 = 0.8
    // Free: g5 ∈ [0.05, 2.35], g6 = 2.4 - g5

    println!("    g0  = 0.8  (from WX + T2)");
    println!("    g1  = 1.6  (from g0+g1=2.4)");
    println!("    g2  = 0.8  (from WX + T4)");
    println!("    g3  = 0.8  (from g2+g3=1.6)");
    println!("    g4  = 0.8  (from WX + T1)");
    println!("    g5  = FREE (only remaining degree of freedom)");
    println!("    g6  = 2.4 - g5");
    println!("    g7  = 0.8  (from WX + T1)");
    println!("    g8  = 0.8  (from WX + T2)");
    println!("    g9  = 0.8  (from g8+g9=1.6 and g8=0.8)");
    println!("    g10 = 1.6  (from WX + T4)");
    println!("    g11 = 0.8  (from g10+g11=2.4)");
    println!("\n    1 degree of freedom: g5 ∈ [0.05, 2.35]\n");

    // Scan g5 for best Liu He/Liu Hai separation
    let mut best_g5 = 1.2f64;
    let mut best_sep = 0.0f64;
    let mut best_n = 1u32;
    let steps = 1000;

    for i in 0..=steps {
        let g5 = 0.05 + (2.30 * i as f64 / steps as f64);
        let metric = build_compromise_metric(g5);

        for n in 1..=24 {
            let lh = mean_coherence_6(&LIU_HE, &metric, n);
            let lhai = mean_coherence_6(&LIU_HAI, &metric, n);
            let sep = (lh - lhai).abs();
            if sep > best_sep {
                best_sep = sep;
                best_n = n;
                best_g5 = g5;
            }
        }
    }

    let metric = build_compromise_metric(best_g5);

    println!("  Best Liu He/Liu Hai separation: {:.6} at n={} (g5={:.3})", best_sep, best_n, best_g5);
    println!("  Metric: {}\n", format_metric(&metric));

    // Verify all constraints
    println!("  Constraint verification:");

    // Wu Xing
    print!("    Wu Xing generative (n=5): ");
    let mut all_wx = true;
    for &(a, b) in WX_GEN.iter() {
        let d = geodesic_distance(a, b, &metric);
        if (d - 72.0).abs() > 0.01 { all_wx = false; }
    }
    println!("{}", if all_wx { "all pairs at 72° ✓" } else { "FAILED ✗" });

    // San He triads
    for t in 0..4 {
        let mut triad_ok = true;
        let mut legs: Vec<f64> = Vec::new();
        for &(a, b) in TRIADS[t].iter() {
            let d = geodesic_distance(a, b, &metric);
            legs.push(d);
            if (d - 120.0).abs() > 0.5 { triad_ok = false; }
        }
        let leg_str: Vec<String> = legs.iter().map(|d| format!("{:.1}°", d)).collect();
        let status = if triad_ok { "✓" } else { "✗ (comma)" };
        println!("    {}: legs [{}] {}",
                 TRIAD_NAMES[t], leg_str.join(", "), status);
    }

    // Liu He / Liu Hai
    let lh = mean_coherence_6(&LIU_HE, &metric, best_n);
    let lhai = mean_coherence_6(&LIU_HAI, &metric, best_n);
    let dir = if lh > lhai { "He>Hai" } else { "Hai>He" };
    println!("    Liu He/Liu Hai: sep={:.6} at n={} ({})", best_sep, best_n, dir);

    println!();
    metric
}

fn build_compromise_metric(g5: f64) -> [f64; 12] {
    let g6 = 2.4 - g5;
    [0.8, 1.6, 0.8, 0.8, 0.8, g5, g6, 0.8, 0.8, 0.8, 1.6, 0.8]
}

fn phase5_triad3_anatomy(metric: &[f64; 12]) {
    println!("--- Phase 5: Triad 3 Under the Compromise Metric ---\n");

    let flat = [1.0f64; 12];

    println!("  Tiger(2) - Horse(6) - Dog(10):\n");
    for &(a, b) in TRIADS[2].iter() {
        let d_flat = geodesic_distance(a, b, &flat);
        let d_curved = geodesic_distance(a, b, metric);
        let error = d_curved - 120.0;
        let c3_flat = coherence(d_flat, 3);
        let c3_curved = coherence(d_curved, 3);
        let segs = leg_segments(a);
        let weights: Vec<String> = segs.iter().map(|&s| format!("{:.1}", metric[s])).collect();
        println!("    {}→{}:", NAMES[a], NAMES[b]);
        println!("      segments: {:?}, weights: [{}]", segs, weights.join(", "));
        println!("      flat: {:.1}°,  curved: {:.1}°,  error: {:+.1}°", d_flat, d_curved, error);
        println!("      cos(3d): flat={:+.6}, curved={:+.6}", c3_flat, c3_curved);
        println!();
    }

    // The comma in n=3 coherence terms
    let triad3_c3: Vec<f64> = TRIADS[2].iter()
        .map(|&(a, b)| coherence(geodesic_distance(a, b, metric), 3))
        .collect();
    let mean_c3: f64 = triad3_c3.iter().sum::<f64>() / 3.0;

    println!("  Triad 3 mean n=3 coherence: {:+.6} (perfect would be +1.000000)", mean_c3);
    println!("  Triads 1,2,4 mean n=3 coherence: +1.000000 (verified above)");
    println!();

    // Compare all triads
    println!("  Summary — n=3 coherence per triad leg:");
    println!("    {:30} {:>7} {:>7} {:>7} {:>8}", "Triad", "Leg 1", "Leg 2", "Leg 3", "Mean");
    for t in 0..4 {
        let cohs: Vec<f64> = TRIADS[t].iter()
            .map(|&(a, b)| coherence(geodesic_distance(a, b, metric), 3))
            .collect();
        let mean: f64 = cohs.iter().sum::<f64>() / 3.0;
        let marker = if t == 2 { " ← comma" } else { "" };
        println!("    {:30} {:+.4} {:+.4} {:+.4}  {:+.4}{}",
                 TRIAD_NAMES[t], cohs[0], cohs[1], cohs[2], mean, marker);
    }
    println!();
}

fn phase6_full_scorecard(metric: &[f64; 12]) {
    println!("--- Phase 6: Full Scorecard ---\n");

    println!("  One metric, all anomalies:\n");

    // Wu Xing
    let wx_cohs: Vec<f64> = WX_GEN.iter()
        .map(|&(a, b)| coherence(geodesic_distance(a, b, metric), 5))
        .collect();
    let wx_mean: f64 = wx_cohs.iter().sum::<f64>() / 5.0;
    println!("    Wu Xing n=5:      mean coh = {:+.6} (flat: +0.300)", wx_mean);

    // Liu He / Liu Hai
    let mut best_sep = 0.0f64;
    let mut best_n_lh = 1u32;
    for n in 1..=24 {
        let lh = mean_coherence_6(&LIU_HE, metric, n);
        let lhai = mean_coherence_6(&LIU_HAI, metric, n);
        let sep = (lh - lhai).abs();
        if sep > best_sep {
            best_sep = sep;
            best_n_lh = n;
        }
    }
    println!("    Liu He/Liu Hai:   sep = {:.6} at n={} (max: 2.0)", best_sep, best_n_lh);

    // San He triads
    for t in 0..4 {
        let cohs: Vec<f64> = TRIADS[t].iter()
            .map(|&(a, b)| coherence(geodesic_distance(a, b, metric), 3))
            .collect();
        let mean: f64 = cohs.iter().sum::<f64>() / 3.0;
        let status = if mean > 0.99 { "✓ perfect" }
            else if mean > 0.5 { "~ partial" }
            else { "✗ broken" };
        println!("    San He {} n=3: mean coh = {:+.6} {}",
                 TRIAD_NAMES[t], mean, status);
    }
    println!();
}

fn phase7_verdict() {
    println!("=== VERDICT ===\n");

    println!("THE INCOMPATIBILITY IS NECESSARY.");
    println!("  Triad 3 (Tiger-Horse-Dog) requires a leg sum of 4.0.");
    println!("  Wu Xing forces that leg sum to 4.8.");
    println!("  The excess is EXACTLY 24° = 2×72° - 120°.");
    println!("  No metric can eliminate it. It is a theorem.\n");

    println!("THE COMMA = 24° = 360°/15.");
    println!("  15 = lcm(3,5). The 3rd and 5th harmonics create a");
    println!("  geometric incompatibility whose magnitude is determined");
    println!("  by their least common multiple.\n");

    println!("THE CATALOG ALREADY KNOWS.");
    println!("  The Chinese system treats San He (triads) and Wu Xing (five phases)");
    println!("  as separate frameworks that interact but don't unify.");
    println!("  They catalogued the incompatibility without proving it.\n");

    println!("WHAT HOLDS:");
    println!("  - Wu Xing (5th harmonic): PERFECT on curved circle");
    println!("  - Liu He/Liu Hai: near-perfect separation on same circle");
    println!("  - San He Triads 1, 2, 4: PERFECT on same circle");
    println!("  - San He Triad 3: BROKEN by 24° comma\n");

    println!("WHAT THIS MEANS:");
    println!("  The ancient traditions arrived at different 'tuning systems'");
    println!("  for the same circle. Like equal temperament vs just intonation,");
    println!("  each system sacrifices exactness somewhere to gain it elsewhere.");
    println!("  The comma is the price. The traditions catalogued which price");
    println!("  they chose to pay.");
}

// --- Main ---

fn main() {
    println!("=== Curvature Test 3: The Geometric Comma ===");
    println!("QUESTION: Is the San He / Wu Xing incompatibility necessary?");
    println!("STATUS:   Internal research — not for publication until proven.\n");

    phase1_constraint_setup();
    phase2_compatibility_check();
    phase3_proof_and_comma();
    let metric = phase4_best_compromise();
    phase5_triad3_anatomy(&metric);
    phase6_full_scorecard(&metric);
    phase7_verdict();
}
