// Curvature Test 2: Wu Xing Harmonic Amplification
//
// QUESTION:
// The Wu Xing (Five Phases) operates at 72° intervals — the 5th harmonic.
// But 72° doesn't exist on the 12-position zodiac grid (positions are at
// multiples of 30°). No zodiac pair is exactly 72° apart.
//
// The Western system rates 72° as "weak" (strength 0.30, orb ±2°).
// The Chinese system makes it THE fundamental relationship.
//
// Can a non-uniform metric make the 12-grid "see" the 5th harmonic by
// warping geodesic distances from 60°/90° to exactly 72°?
//
// KEY DISCOVERY (analytical):
// The Wu Xing generative cycle, mapped to nearest zodiac positions, gives
// flat distances that alternate 60°, 90°, 60°, 90°, 60° — averaging 72°.
// The metric constraints that correct these to exactly 72° are:
//   g_0+g_1 = 2.4,  g_2+g_3+g_4 = 2.4,  g_5+g_6 = 2.4,
//   g_7+g_8+g_9 = 2.4,  g_10+g_11 = 2.4
// These are 5 constraints on 12 variables, leaving 7 degrees of freedom.
// The destructive cycle (144°) is AUTOMATICALLY satisfied.
// Total constraint sum = 5 × 2.4 = 12.0 = normalization. Consistent.
//
// BONUS TEST: Can a SINGLE metric satisfy both Wu Xing resonance AND
// Liu He/Liu Hai separation (from Test 1)?
//
// COMPILE: rustc curvature_test2_wu_xing_amplification.rs -o curvature_test2
// RUN:     ./curvature_test2

use std::f64::consts::PI;

// --- Constants ---

const N_POS: usize = 12;
const SEG_DEG: f64 = 30.0;

const NAMES: [&str; 12] = [
    "Rat", "Ox", "Tiger", "Rabbit", "Dragon", "Snake",
    "Horse", "Goat", "Monkey", "Rooster", "Dog", "Pig",
];

// Wu Xing elements mapped to nearest zodiac positions:
//   Wood = 0° → Rat (0°),  offset 0°
//   Fire = 72° → Tiger (60°),  offset -12°
//   Earth = 144° → Snake (150°),  offset +6°
//   Metal = 216° → Goat (210°),  offset -6°
//   Water = 288° → Dog (300°),  offset +12°
const WX_ELEMENTS: [&str; 5] = ["Wood", "Fire", "Earth", "Metal", "Water"];
const WX_ZODIAC: [usize; 5] = [0, 2, 5, 7, 10]; // Rat, Tiger, Snake, Goat, Dog
const WX_IDEAL_DEG: [f64; 5] = [0.0, 72.0, 144.0, 216.0, 288.0];
const WX_ACTUAL_DEG: [f64; 5] = [0.0, 60.0, 150.0, 210.0, 300.0];

// Generative cycle: Wood→Fire→Earth→Metal→Water→Wood (+72° steps)
const WX_GEN: [(usize, usize); 5] = [
    (0, 2),   // Rat → Tiger:  60° flat, should be 72°
    (2, 5),   // Tiger → Snake: 90° flat, should be 72°
    (5, 7),   // Snake → Goat:  60° flat, should be 72°
    (7, 10),  // Goat → Dog:    90° flat, should be 72°
    (10, 0),  // Dog → Rat:     60° flat, should be 72°
];

const WX_GEN_NAMES: [&str; 5] = [
    "Wood->Fire", "Fire->Earth", "Earth->Metal", "Metal->Water", "Water->Wood",
];

// Destructive cycle: Wood→Earth→Water→Fire→Metal→Wood (+144° steps)
const WX_DEST: [(usize, usize); 5] = [
    (0, 5),   // Rat → Snake:  150° flat, should be 144°
    (5, 10),  // Snake → Dog:   150° flat, should be 144°
    (10, 2),  // Dog → Tiger:   120° flat, should be 144°
    (2, 7),   // Tiger → Goat:  150° flat, should be 144°
    (7, 0),   // Goat → Rat:    150° flat, should be 144°
];

const WX_DEST_NAMES: [&str; 5] = [
    "Wood->Earth", "Earth->Water", "Water->Fire", "Fire->Metal", "Metal->Wood",
];

// San He triads (120° apart) — should be unaffected
const SAN_HE: [(usize, usize); 4] = [
    (0, 4),   // Rat-Dragon: 120°
    (1, 5),   // Ox-Snake: 120°
    (2, 6),   // Tiger-Horse: 120°
    (3, 7),   // Rabbit-Goat: 120°
];

// Liu He / Liu Hai from Test 1
const LIU_HE: [(usize, usize); 6] = [
    (0, 1), (2, 11), (3, 10), (4, 9), (5, 8), (6, 7),
];
const LIU_HAI: [(usize, usize); 6] = [
    (0, 7), (1, 6), (2, 5), (3, 4), (8, 11), (9, 10),
];

// --- PRNG ---

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self { Rng(if seed == 0 { 1 } else { seed }) }

    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn f64(&mut self) -> f64 { (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 }
    fn range(&mut self, lo: f64, hi: f64) -> f64 { lo + (hi - lo) * self.f64() }
}

// --- Geometry (same as Test 1) ---

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

fn mean_coherence_pairs(pairs: &[(usize, usize)], g: &[f64; 12], n: u32) -> f64 {
    let sum: f64 = pairs.iter()
        .map(|&(a, b)| coherence(geodesic_distance(a, b, g), n))
        .sum();
    sum / pairs.len() as f64
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

// --- Wu Xing metric construction ---

/// Build a metric from 7 free parameters, satisfying Wu Xing constraints.
/// Constraints: g_0+g_1=2.4, g_2+g_3+g_4=2.4, g_5+g_6=2.4,
///              g_7+g_8+g_9=2.4, g_10+g_11=2.4
/// Free params: g_0, g_2, g_3, g_5, g_7, g_8, g_10
fn build_wx_metric(free: &[f64; 7]) -> Option<[f64; 12]> {
    let g0 = free[0];
    let g2 = free[1];
    let g3 = free[2];
    let g5 = free[3];
    let g7 = free[4];
    let g8 = free[5];
    let g10 = free[6];

    let g1 = 2.4 - g0;
    let g4 = 2.4 - g2 - g3;
    let g6 = 2.4 - g5;
    let g9 = 2.4 - g7 - g8;
    let g11 = 2.4 - g10;

    let metric = [g0, g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11];

    // Check all weights positive
    let min_weight = 0.05;
    if metric.iter().any(|&x| x < min_weight) {
        return None;
    }
    Some(metric)
}

/// Uniform Wu Xing metric: 1.2 for 2-seg groups, 0.8 for 3-seg groups
fn uniform_wx_metric() -> [f64; 12] {
    [1.2, 1.2, 0.8, 0.8, 0.8, 1.2, 1.2, 0.8, 0.8, 0.8, 1.2, 1.2]
}

// --- Phases ---

fn phase1_flat_analysis() {
    println!("--- Phase 1: Wu Xing on the 12-Grid (Flat) ---\n");

    let flat = [1.0f64; 12];

    // Mapping
    println!("  Wu Xing to zodiac mapping (nearest position):");
    for i in 0..5 {
        let offset = WX_ACTUAL_DEG[i] - WX_IDEAL_DEG[i];
        println!(
            "    {} ({:.0}°) -> {} ({:.0}°), offset {:+.0}°",
            WX_ELEMENTS[i], WX_IDEAL_DEG[i],
            NAMES[WX_ZODIAC[i]], WX_ACTUAL_DEG[i], offset
        );
    }

    // Generative cycle distances
    println!("\n  Generative cycle (should be 72° each):");
    let mut gen_sum = 0.0f64;
    for i in 0..5 {
        let (a, b) = WX_GEN[i];
        let d = geodesic_distance(a, b, &flat);
        let err = d - 72.0;
        gen_sum += d;
        println!(
            "    {:15} {}->{:10} flat={:5.1}° error={:+.1}°",
            WX_GEN_NAMES[i], NAMES[a], NAMES[b], d, err
        );
    }
    println!("    Average flat distance: {:.1}° (ideal: 72.0°)", gen_sum / 5.0);

    // n=5 coherence on flat grid
    println!("\n  Coherence at n=5 (5th harmonic, detecting 72°):");
    println!("    {:15} {:>7} {:>10}", "Pair", "dist", "cos(5d)");
    for i in 0..5 {
        let (a, b) = WX_GEN[i];
        let d = geodesic_distance(a, b, &flat);
        let c = coherence(d, 5);
        println!("    {:15} {:>6.1}°  {:>+.6}", WX_GEN_NAMES[i], d, c);
    }
    let gen_mean = mean_coherence_pairs(&WX_GEN, &flat, 5);
    println!("    Mean n=5 coherence: {:+.6}", gen_mean);
    println!("    (Far from 1.0 — the 12-grid cannot see the 5th harmonic on flat geometry)");
    println!();
}

fn phase2_analytical_metric() {
    println!("--- Phase 2: Analytical Metric ---\n");

    println!("  Constraints for all generative pairs to have 72° geodesic:");
    println!("    Rat->Tiger  (2 segs): g_0 + g_1 = 2.4");
    println!("    Tiger->Snake (3 segs): g_2 + g_3 + g_4 = 2.4");
    println!("    Snake->Goat  (2 segs): g_5 + g_6 = 2.4");
    println!("    Goat->Dog   (3 segs): g_7 + g_8 + g_9 = 2.4");
    println!("    Dog->Rat    (2 segs): g_10 + g_11 = 2.4");
    println!("    Total: 5 × 2.4 = 12.0 = normalization. CONSISTENT.");
    println!("    Degrees of freedom: 12 - 5 = 7");

    let wx_metric = uniform_wx_metric();
    println!("\n  Simplest valid metric (uniform within groups):");
    println!("    {}", format_metric(&wx_metric));

    // Verify generative pairs
    println!("\n  Generative cycle verification:");
    for i in 0..5 {
        let (a, b) = WX_GEN[i];
        let d = geodesic_distance(a, b, &wx_metric);
        let c = coherence(d, 5);
        println!(
            "    {:15} geodesic={:6.2}° cos(5×d)={:+.10}",
            WX_GEN_NAMES[i], d, c
        );
    }

    // Verify destructive pairs
    println!("\n  Destructive cycle (should be auto-satisfied at 144°):");
    for i in 0..5 {
        let (a, b) = WX_DEST[i];
        let d = geodesic_distance(a, b, &wx_metric);
        let c5 = coherence(d, 5);
        println!(
            "    {:15} geodesic={:6.2}° cos(5×d)={:+.10}",
            WX_DEST_NAMES[i], d, c5
        );
    }

    let gen_mean = mean_coherence_pairs(&WX_GEN, &wx_metric, 5);
    let dest_mean = mean_coherence_pairs(&WX_DEST, &wx_metric, 5);
    println!("\n  Generative mean cos(5×d): {:+.10}", gen_mean);
    println!("  Destructive mean cos(5×d): {:+.10}", dest_mean);
    println!("  Both PERFECT (1.0). The analytical metric works.");
    println!();
}

fn phase3_cross_effects() {
    println!("--- Phase 3: Cross-Harmonic Effects ---\n");

    let flat = [1.0f64; 12];
    let wx = uniform_wx_metric();

    // San He triads at n=3 (should be unaffected)
    println!("  San He triads at n=3 (trine, 120°):");
    println!("    {:15} {:>7} {:>7} {:>10} {:>10}", "Pair", "flat", "curved", "flat n=3", "curved n=3");
    for &(a, b) in SAN_HE.iter() {
        let d_flat = geodesic_distance(a, b, &flat);
        let d_wx = geodesic_distance(a, b, &wx);
        let c_flat = coherence(d_flat, 3);
        let c_wx = coherence(d_wx, 3);
        println!(
            "    {}-{:10} {:>6.1}° {:>6.1}°  {:>+.6} {:>+.6}",
            NAMES[a], NAMES[b], d_flat, d_wx, c_flat, c_wx
        );
    }
    let san_he_flat = mean_coherence_pairs(&SAN_HE, &flat, 3);
    let san_he_wx = mean_coherence_pairs(&SAN_HE, &wx, 3);
    println!("    Mean n=3: flat={:+.6}, curved={:+.6}", san_he_flat, san_he_wx);

    // Full harmonic profile comparison
    println!("\n  Wu Xing generative pairs — harmonic profile (flat vs curved):");
    println!("    {:>3}  {:>10}  {:>10}  {:>8}", "n", "flat", "curved", "change");
    for n in 1..=12 {
        let f = mean_coherence_pairs(&WX_GEN, &flat, n);
        let c = mean_coherence_pairs(&WX_GEN, &wx, n);
        let change = c - f;
        let marker = if n == 5 { " <-- target" } else { "" };
        println!("    {:>3}  {:>+.6}  {:>+.6}  {:>+.6}{}", n, f, c, change, marker);
    }

    // Liu He / Liu Hai under Wu Xing metric
    println!("\n  Liu He / Liu Hai separation under Wu Xing metric:");
    let mut best_sep = 0.0f64;
    let mut best_n = 1u32;
    for n in 1..=24 {
        let lh = mean_coherence_6(&LIU_HE, &wx, n);
        let lhai = mean_coherence_6(&LIU_HAI, &wx, n);
        let sep = (lh - lhai).abs();
        if sep > best_sep {
            best_sep = sep;
            best_n = n;
        }
    }
    let lh = mean_coherence_6(&LIU_HE, &wx, best_n);
    let lhai = mean_coherence_6(&LIU_HAI, &wx, best_n);
    let dir = if lh > lhai { "He>Hai" } else { "Hai>He" };
    println!("    Best separation: {:.6} at n={} ({})", best_sep, best_n, dir);
    println!("    (For reference, Test 1 achieved 1.999)");
    println!();
}

fn phase4_combined_optimization() -> ([f64; 12], u32, f64) {
    println!("--- Phase 4: Combined Optimization ---");
    println!("  Finding a SINGLE metric that satisfies:");
    println!("    1. Wu Xing constraints (all generative pairs at 72° geodesic)");
    println!("    2. Maximum Liu He / Liu Hai separation\n");

    let n_restarts = 5000;
    let n_steps = 1000;
    let max_harmonic = 24u32;

    println!("  {} restarts × {} steps, harmonics 1..{}", n_restarts, n_steps, max_harmonic);
    println!("  7 free parameters within Wu Xing constraints");
    println!("  Searching...\n");

    let mut rng = Rng::new(137);
    let mut global_best_metric = uniform_wx_metric();
    let mut global_best_n = 1u32;
    let mut global_best_sep = 0.0f64;

    for _ in 0..n_restarts {
        // Random starting point within constraints
        let mut free = [0.0f64; 7];
        loop {
            free[0] = rng.range(0.1, 2.3);  // g_0
            free[1] = rng.range(0.1, 1.5);  // g_2
            free[2] = rng.range(0.1, 1.5);  // g_3
            free[3] = rng.range(0.1, 2.3);  // g_5
            free[4] = rng.range(0.1, 1.5);  // g_7
            free[5] = rng.range(0.1, 1.5);  // g_8
            free[6] = rng.range(0.1, 2.3);  // g_10
            if build_wx_metric(&free).is_some() {
                break;
            }
        }
        let mut metric = build_wx_metric(&free).unwrap();

        let mut cur_sep = 0.0f64;
        let mut cur_n = 1u32;
        for n in 1..=max_harmonic {
            let lh = mean_coherence_6(&LIU_HE, &metric, n);
            let lhai = mean_coherence_6(&LIU_HAI, &metric, n);
            let sep = (lh - lhai).abs();
            if sep > cur_sep {
                cur_sep = sep;
                cur_n = n;
            }
        }

        // Hill climbing within constraints
        for _ in 0..n_steps {
            let param_idx = (rng.next_u64() % 7) as usize;
            let delta = rng.range(-0.3, 0.3);
            let old_val = free[param_idx];
            free[param_idx] += delta;

            if let Some(new_metric) = build_wx_metric(&free) {
                let mut new_sep = 0.0f64;
                let mut new_n = 1u32;
                for n in 1..=max_harmonic {
                    let lh = mean_coherence_6(&LIU_HE, &new_metric, n);
                    let lhai = mean_coherence_6(&LIU_HAI, &new_metric, n);
                    let sep = (lh - lhai).abs();
                    if sep > new_sep {
                        new_sep = sep;
                        new_n = n;
                    }
                }
                if new_sep > cur_sep {
                    metric = new_metric;
                    cur_sep = new_sep;
                    cur_n = new_n;
                } else {
                    free[param_idx] = old_val;
                }
            } else {
                free[param_idx] = old_val;
            }
        }

        if cur_sep > global_best_sep {
            global_best_sep = cur_sep;
            global_best_n = cur_n;
            global_best_metric = metric;
        }
    }

    println!("  Best Liu He/Liu Hai separation: {:.6} at n={}", global_best_sep, global_best_n);
    println!("  Metric: {}", format_metric(&global_best_metric));

    // Verify Wu Xing constraints still hold
    println!("\n  Wu Xing constraint verification:");
    let groups: [(usize, usize); 5] = [(0, 2), (2, 5), (5, 7), (7, 10), (10, 12)];
    let group_labels = ["g0+g1", "g2+g3+g4", "g5+g6", "g7+g8+g9", "g10+g11"];
    for i in 0..5 {
        let (start, end) = groups[i];
        let end_idx = if end == 12 { 12 } else { end };
        let sum: f64 = (start..end_idx)
            .map(|j| global_best_metric[j % 12])
            .sum();
        println!("    {} = {:.6} (target: 2.400)", group_labels[i], sum);
    }

    // Verify generative coherence
    println!("\n  Wu Xing generative at n=5:");
    for i in 0..5 {
        let (a, b) = WX_GEN[i];
        let d = geodesic_distance(a, b, &global_best_metric);
        let c = coherence(d, 5);
        println!("    {:15} d={:.2}° cos(5d)={:+.8}", WX_GEN_NAMES[i], d, c);
    }

    println!();
    (global_best_metric, global_best_n, global_best_sep)
}

fn phase5_analysis(metric: &[f64; 12], harmonic: u32, separation: f64) {
    println!("--- Phase 5: Combined Metric Analysis ---\n");

    let flat = [1.0f64; 12];

    // Liu He / Liu Hai per-pair breakdown
    println!("  Liu He pairs at n={}:", harmonic);
    let mut lh_cohs = [0.0f64; 6];
    let lh_names = ["Rat-Ox","Tiger-Pig","Rabbit-Dog","Dragon-Rooster","Snake-Monkey","Horse-Goat"];
    for i in 0..6 {
        let (a, b) = LIU_HE[i];
        let d = geodesic_distance(a, b, metric);
        let c = coherence(d, harmonic);
        lh_cohs[i] = c;
        println!("    {:18} curved={:6.1}° coh={:+.6}", lh_names[i], d, c);
    }
    let lh_mean = lh_cohs.iter().sum::<f64>() / 6.0;
    println!("    Mean: {:+.6}\n", lh_mean);

    println!("  Liu Hai pairs at n={}:", harmonic);
    let mut lhai_cohs = [0.0f64; 6];
    let lhai_names = ["Rat-Goat","Ox-Horse","Tiger-Snake","Rabbit-Dragon","Monkey-Pig","Rooster-Dog"];
    for i in 0..6 {
        let (a, b) = LIU_HAI[i];
        let d = geodesic_distance(a, b, metric);
        let c = coherence(d, harmonic);
        lhai_cohs[i] = c;
        println!("    {:18} curved={:6.1}° coh={:+.6}", lhai_names[i], d, c);
    }
    let lhai_mean = lhai_cohs.iter().sum::<f64>() / 6.0;
    println!("    Mean: {:+.6}\n", lhai_mean);

    // Overlap check
    let lh_min = lh_cohs.iter().cloned().fold(f64::INFINITY, f64::min);
    let lh_max = lh_cohs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let lhai_min = lhai_cohs.iter().cloned().fold(f64::INFINITY, f64::min);
    let lhai_max = lhai_cohs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let no_overlap = lh_min > lhai_max || lhai_min > lh_max;
    println!("  Liu He range:  [{:+.4}, {:+.4}]", lh_min, lh_max);
    println!("  Liu Hai range: [{:+.4}, {:+.4}]", lhai_min, lhai_max);
    println!("  Overlap: {}", if no_overlap { "NONE" } else { "Some pairs overlap" });

    // Compare with Test 1 unconstrained result
    println!("\n  Comparison with Test 1 (unconstrained):");
    println!("    Test 1 separation: 1.999 (no Wu Xing constraint)");
    println!("    Test 2 separation: {:.3} (Wu Xing constrained)", separation);
    let retention = separation / 1.999 * 100.0;
    println!("    Retained: {:.1}% of unconstrained performance", retention);

    // San He triads still intact?
    println!("\n  San He triads at n=3 (must remain intact):");
    for &(a, b) in SAN_HE.iter() {
        let d_flat = geodesic_distance(a, b, &flat);
        let d_curved = geodesic_distance(a, b, metric);
        let c_flat = coherence(d_flat, 3);
        let c_curved = coherence(d_curved, 3);
        println!(
            "    {}-{:10} flat_n3={:+.6} curved_n3={:+.6}",
            NAMES[a], NAMES[b], c_flat, c_curved
        );
    }

    // Metric shape
    println!("\n  Metric shape:");
    let max_g = metric.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_g = metric.iter().cloned().fold(f64::INFINITY, f64::min);
    println!("    Range: [{:.3}, {:.3}] (ratio {:.1}x)", min_g, max_g, max_g / min_g);

    for s in 0..12 {
        let bar_len = (metric[s] * 20.0) as usize;
        let bar: String = std::iter::repeat('#').take(bar_len).collect();
        println!(
            "    seg {:2} ({:>8}->{:8}) g={:.3} {}",
            s, NAMES[s], NAMES[(s + 1) % N_POS], metric[s], bar
        );
    }

    // Verdict
    println!("\n=== VERDICT ===\n");

    println!("WU XING RESONANCE:");
    println!("  The 5th harmonic is INVISIBLE on a flat 12-grid.");
    println!("  A non-uniform metric makes it PERFECT (coherence = 1.0).");
    println!("  The solution is ANALYTICAL (exact, not optimized).");
    println!("  The destructive cycle is FREE (auto-satisfied).\n");

    if separation > 1.5 {
        println!("COMBINED METRIC:");
        println!("  A SINGLE metric satisfies BOTH anomalies simultaneously:");
        println!("    - Wu Xing: perfect 5th harmonic resonance");
        println!("    - Liu He/Liu Hai: {:.3} separation (Test 1: 1.999)", separation);
        println!("  The two anomalies are COMPATIBLE — they can coexist");
        println!("  on the same curved circle.");
    } else if separation > 0.5 {
        println!("COMBINED METRIC:");
        println!("  Partial compatibility. Wu Xing constraints reduce but");
        println!("  don't eliminate Liu He/Liu Hai separation ({:.3} vs 1.999).", separation);
        println!("  The anomalies are partially compatible.");
    } else {
        println!("COMBINED METRIC:");
        println!("  Wu Xing and Liu He/Liu Hai require DIFFERENT metrics.");
        println!("  Separation only {:.3} under Wu Xing constraints.", separation);
        println!("  The anomalies may need different explanations.");
    }
}

// --- Main ---

fn main() {
    println!("=== Curvature Test 2: Wu Xing Harmonic Amplification ===");
    println!("QUESTION: Can a non-uniform metric make the 12-grid see the 5th harmonic?");
    println!("BONUS:    Can the same metric also separate Liu He from Liu Hai?");
    println!("STATUS:   Internal research — not for publication until proven.\n");

    phase1_flat_analysis();
    phase2_analytical_metric();
    phase3_cross_effects();
    let (metric, best_n, best_sep) = phase4_combined_optimization();
    phase5_analysis(&metric, best_n, best_sep);
}
