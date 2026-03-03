// Curvature Test 1: Can a non-uniform metric separate Liu He from Liu Hai?
//
// QUESTION:
// On a flat circle, Liu He (六合, Six Harmonies) and Liu Hai (六害, Six Harms)
// have IDENTICAL angular distance distributions: {30°, 30°, 90°, 90°, 150°, 150°}.
// Standard coherence cos(n × Δθ) returns identical scores for both sets.
// They are mathematically indistinguishable on flat geometry.
//
// Can a NON-UNIFORM metric — variable segment weights around the circle — create
// different geodesic distances for the two sets, making them distinguishable?
//
// METHOD:
// 1. Assign weights g_0..g_11 to the 12 segments of the zodiac circle
// 2. Geodesic distance = sum of weighted segment lengths along shortest path
// 3. Curved coherence = cos(n × d_geodesic)
// 4. Search for metrics that maximize |mean(Liu He coh) - mean(Liu Hai coh)|
//
// SIGNIFICANCE:
// If a metric separates them, the "same angle, opposite meaning" anomaly
// has a geometric explanation: the space is curved, and the same coordinate
// distance corresponds to different geodesic distances depending on WHERE
// on the circle you are.
//
// If no metric separates them, the Liu He/Liu Hai distinction is genuinely
// non-geometric and the catalog was right to call it "structural."
//
// COMPILE: rustc curvature_test1_liu_he_vs_liu_hai.rs -o curvature_test1
// RUN:     ./curvature_test1

use std::f64::consts::PI;

// --- Constants ---

const N_POS: usize = 12;
const SEG_DEG: f64 = 30.0; // 360° / 12 positions

const NAMES: [&str; 12] = [
    "Rat", "Ox", "Tiger", "Rabbit", "Dragon", "Snake",
    "Horse", "Goat", "Monkey", "Rooster", "Dog", "Pig",
];

// Yin(1) / Yang(0) — alternating, starting Yang
const YIN_YANG: [u8; 12] = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1];

// Liu He: Six Harmonies — declared compatible despite angular oddness
// Every pair is Yang-Yin. Angular distances: {30, 90, 150, 150, 90, 30}
const LIU_HE: [(usize, usize); 6] = [
    (0, 1),   // Rat(Y,0°)    — Ox(y,30°):      30°
    (2, 11),  // Tiger(Y,60°) — Pig(y,330°):     90° (3 seg ccw)
    (3, 10),  // Rabbit(y,90°)— Dog(Y,300°):    150° (5 seg ccw)
    (4, 9),   // Dragon(Y,120°)—Rooster(y,270°): 150° (5 seg cw)
    (5, 8),   // Snake(y,150°)— Monkey(Y,240°):  90° (3 seg cw)
    (6, 7),   // Horse(Y,180°)— Goat(y,210°):    30°
];

// Liu Hai: Six Harms — declared harmful despite SAME angle set
// Every pair is also Yang-Yin. Angular distances: {150, 150, 90, 30, 90, 30}
const LIU_HAI: [(usize, usize); 6] = [
    (0, 7),   // Rat(Y,0°)    — Goat(y,210°):   150° (5 seg ccw)
    (1, 6),   // Ox(y,30°)    — Horse(Y,180°):  150° (5 seg cw)
    (2, 5),   // Tiger(Y,60°) — Snake(y,150°):   90° (3 seg cw)
    (3, 4),   // Rabbit(y,90°)— Dragon(Y,120°):  30°
    (8, 11),  // Monkey(Y,240°)—Pig(y,330°):      90° (3 seg cw)
    (9, 10),  // Rooster(y,270°)—Dog(Y,300°):     30°
];

const LIU_HE_NAMES: [&str; 6] = [
    "Rat-Ox", "Tiger-Pig", "Rabbit-Dog",
    "Dragon-Rooster", "Snake-Monkey", "Horse-Goat",
];

const LIU_HAI_NAMES: [&str; 6] = [
    "Rat-Goat", "Ox-Horse", "Tiger-Snake",
    "Rabbit-Dragon", "Monkey-Pig", "Rooster-Dog",
];

// --- PRNG (xorshift64, Marsaglia 2003) ---

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Rng { state: if seed == 0 { 1 } else { seed } }
    }

    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    fn next_f64(&mut self) -> f64 {
        // 53 bits of precision in [0, 1)
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }
}

// --- Core geometry ---

/// Weighted distance from position a to position b going clockwise.
/// Each segment i (connecting position i to position i+1) has weight g[i].
fn forward_distance(a: usize, b: usize, g: &[f64; 12]) -> f64 {
    let mut dist = 0.0;
    let mut pos = a;
    while pos != b {
        dist += g[pos] * SEG_DEG;
        pos = (pos + 1) % N_POS;
    }
    dist
}

/// Shortest weighted path around the circle (min of clockwise and counter-clockwise).
fn geodesic_distance(a: usize, b: usize, g: &[f64; 12]) -> f64 {
    if a == b {
        return 0.0;
    }
    forward_distance(a, b, g).min(forward_distance(b, a, g))
}

/// Harmonic coherence at a geodesic distance.
fn coherence(distance_deg: f64, n: u32) -> f64 {
    (n as f64 * distance_deg * PI / 180.0).cos()
}

/// Segments traversed going clockwise from a to b.
fn path_segments_forward(a: usize, b: usize) -> Vec<usize> {
    let mut segs = Vec::new();
    let mut pos = a;
    while pos != b {
        segs.push(pos);
        pos = (pos + 1) % N_POS;
    }
    segs
}

/// Segments on the shortest path (flat metric).
fn shortest_path_segments(a: usize, b: usize) -> Vec<usize> {
    let flat = [1.0; 12];
    if forward_distance(a, b, &flat) <= forward_distance(b, a, &flat) {
        path_segments_forward(a, b)
    } else {
        path_segments_forward(b, a)
    }
}

// --- Analysis helpers ---

fn mean_coherence(pairs: &[(usize, usize); 6], g: &[f64; 12], n: u32) -> f64 {
    pairs.iter()
        .map(|&(a, b)| coherence(geodesic_distance(a, b, g), n))
        .sum::<f64>()
        / 6.0
}

/// Find the harmonic (1..max_n) with the largest absolute separation.
/// Returns (best_n, abs_separation, signed_separation).
fn best_separation(g: &[f64; 12], max_n: u32) -> (u32, f64, f64) {
    let mut best_n = 1u32;
    let mut best_abs = 0.0f64;
    let mut best_signed = 0.0f64;
    for n in 1..=max_n {
        let lh = mean_coherence(&LIU_HE, g, n);
        let lhai = mean_coherence(&LIU_HAI, g, n);
        let signed = lh - lhai;
        if signed.abs() > best_abs {
            best_abs = signed.abs();
            best_signed = signed;
            best_n = n;
        }
    }
    (best_n, best_abs, best_signed)
}

fn normalize(g: &mut [f64; 12]) {
    let sum: f64 = g.iter().sum();
    if sum > 0.0 {
        let scale = N_POS as f64 / sum;
        for x in g.iter_mut() {
            *x *= scale;
        }
    }
}

fn format_metric(g: &[f64; 12]) -> String {
    let parts: Vec<String> = g.iter().map(|x| format!("{:.3}", x)).collect();
    format!("[{}]", parts.join(", "))
}

// --- Phase 1: Flat Baseline ---

fn phase1_flat_baseline() {
    println!("--- Phase 1: Flat Baseline ---");
    println!("All segment weights = 1.0 (uniform circle)\n");

    let flat = [1.0f64; 12];

    // Show per-pair distances
    println!("  {:18} {:>7}    {:18} {:>7}", "Liu He pair", "dist", "Liu Hai pair", "dist");
    println!("  {:18} {:>7}    {:18} {:>7}", "----------", "----", "-----------", "----");
    for i in 0..6 {
        let (a1, b1) = LIU_HE[i];
        let (a2, b2) = LIU_HAI[i];
        let d1 = geodesic_distance(a1, b1, &flat);
        let d2 = geodesic_distance(a2, b2, &flat);
        println!(
            "  {:18} {:>6.1}°    {:18} {:>6.1}°",
            LIU_HE_NAMES[i], d1, LIU_HAI_NAMES[i], d2
        );
    }

    // Sorted distance multisets
    let mut lh_dists: Vec<f64> = LIU_HE.iter()
        .map(|&(a, b)| geodesic_distance(a, b, &flat))
        .collect();
    let mut lhai_dists: Vec<f64> = LIU_HAI.iter()
        .map(|&(a, b)| geodesic_distance(a, b, &flat))
        .collect();
    lh_dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
    lhai_dists.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let lh_str: Vec<String> = lh_dists.iter().map(|d| format!("{:.0}°", d)).collect();
    let lhai_str: Vec<String> = lhai_dists.iter().map(|d| format!("{:.0}°", d)).collect();
    println!("\n  Liu He  sorted distances: {{{}}}", lh_str.join(", "));
    println!("  Liu Hai sorted distances: {{{}}}", lhai_str.join(", "));
    println!("  IDENTICAL — flat geometry cannot distinguish them.");

    // Harmonic sweep
    println!("\n  Harmonic coherence sweep:");
    let mut max_sep = 0.0f64;
    for n in 1..=12 {
        let lh = mean_coherence(&LIU_HE, &flat, n);
        let lhai = mean_coherence(&LIU_HAI, &flat, n);
        let sep = lh - lhai;
        max_sep = max_sep.max(sep.abs());
        println!(
            "    n={:2}: Liu He = {:+.6}, Liu Hai = {:+.6}, sep = {:+.10}",
            n, lh, lhai, sep
        );
    }
    println!("\n  BASELINE: max |separation| = {:.2e}", max_sep);
    if max_sep < 1e-10 {
        println!("  Confirmed: separation is zero to machine precision.");
    }
    println!();
}

// --- Phase 2: Path Analysis ---

fn phase2_path_analysis() {
    println!("--- Phase 2: Path Analysis ---");
    println!("Which circle segments does each pair's shortest path traverse?\n");

    // Build segment usage matrix
    let mut lh_usage = [0u32; 12];
    let mut lhai_usage = [0u32; 12];

    println!("  Liu He paths:");
    for i in 0..6 {
        let (a, b) = LIU_HE[i];
        let segs = shortest_path_segments(a, b);
        for &s in &segs {
            lh_usage[s] += 1;
        }
        let seg_str: Vec<String> = segs.iter().map(|s| s.to_string()).collect();
        println!("    {:18} ({} segs): [{}]", LIU_HE_NAMES[i], segs.len(), seg_str.join(","));
    }

    println!("\n  Liu Hai paths:");
    for i in 0..6 {
        let (a, b) = LIU_HAI[i];
        let segs = shortest_path_segments(a, b);
        for &s in &segs {
            lhai_usage[s] += 1;
        }
        let seg_str: Vec<String> = segs.iter().map(|s| s.to_string()).collect();
        println!("    {:18} ({} segs): [{}]", LIU_HAI_NAMES[i], segs.len(), seg_str.join(","));
    }

    // Segment usage comparison
    println!("\n  Segment usage (how many pairs traverse each segment):");
    println!("    Seg  Pos->Pos+1       Liu He  Liu Hai  Diff");
    println!("    ---  ---------        ------  -------  ----");
    for s in 0..12 {
        let diff = lh_usage[s] as i32 - lhai_usage[s] as i32;
        let marker = if diff > 0 {
            " <-- Liu He dominant"
        } else if diff < 0 {
            " <-- Liu Hai dominant"
        } else {
            ""
        };
        println!(
            "    {:2}   {:>6}->{:6}      {:1}       {:1}      {:+2}{}",
            s,
            NAMES[s],
            NAMES[(s + 1) % N_POS],
            lh_usage[s],
            lhai_usage[s],
            diff,
            marker
        );
    }

    // Key insight
    println!("\n  KEY INSIGHT: Segments 0,6 are used 3x by Liu He, 0x by Liu Hai.");
    println!("              Segments 3,9 are used 3x by Liu Hai, 0x by Liu He.");
    println!("              The paths are STRUCTURALLY DIFFERENT despite identical flat distances.");
    println!("              A metric that makes segments 0,6 light and 3,9 heavy will");
    println!("              compress Liu He distances while stretching Liu Hai distances.");
    println!("              This is a k=2 sinusoidal modulation (period = half circle).");
    println!();
}

// --- Phase 3: Structured Metrics ---

fn test_metric_brief(name: &str, g: &[f64; 12]) {
    let (n, sep, signed) = best_separation(g, 24);
    let dir = if signed > 0.0 { "He>Hai" } else { "Hai>He" };
    println!("  {:40} n={:2}  sep={:.6}  ({})", name, n, sep, dir);
}

fn phase3_structured_metrics() {
    println!("--- Phase 3: Structured Metrics ---\n");

    // 3a: Yin-Yang alternating
    let mut m = [0.0f64; 12];
    for i in 0..12 {
        m[i] = if YIN_YANG[i] == 0 { 0.7 } else { 1.3 };
    }
    normalize(&mut m);
    test_metric_brief("Yin-Yang alternating (Y=0.7, y=1.3)", &m);

    // 3b: Sinusoidal k=1 (single bulge)
    for i in 0..12 {
        m[i] = 1.0 + 0.7 * (2.0 * PI * i as f64 / 12.0).cos();
    }
    normalize(&mut m);
    test_metric_brief("Sinusoidal k=1, A=0.7", &m);

    // 3c: Sinusoidal k=2 (predicted optimal from path analysis)
    for i in 0..12 {
        m[i] = 1.0 + 0.7 * (2.0 * PI * 2.0 * i as f64 / 12.0).cos();
    }
    normalize(&mut m);
    test_metric_brief("Sinusoidal k=2, A=0.7 (predicted best)", &m);

    // 3d: Sinusoidal k=2, stronger amplitude
    for i in 0..12 {
        m[i] = 1.0 + 0.9 * (2.0 * PI * 2.0 * i as f64 / 12.0).cos();
    }
    normalize(&mut m);
    test_metric_brief("Sinusoidal k=2, A=0.9", &m);

    // 3e: Sinusoidal k=3 (triangle)
    for i in 0..12 {
        m[i] = 1.0 + 0.6 * (2.0 * PI * 3.0 * i as f64 / 12.0).cos();
    }
    normalize(&mut m);
    test_metric_brief("Sinusoidal k=3, A=0.6", &m);

    // 3f: Sinusoidal k=6 (alternating, same as Yin-Yang)
    for i in 0..12 {
        m[i] = 1.0 + 0.6 * (2.0 * PI * 6.0 * i as f64 / 12.0).cos();
    }
    normalize(&mut m);
    test_metric_brief("Sinusoidal k=6, A=0.6", &m);

    // 3g: Hand-crafted: seg 0,6 very light, seg 3,9 very heavy
    m = [0.2, 0.8, 1.0, 2.0, 1.0, 0.8, 0.2, 0.8, 1.0, 2.0, 1.0, 0.8];
    normalize(&mut m);
    test_metric_brief("Hand-crafted (0,6 light / 3,9 heavy)", &m);

    // 3h: Extreme version
    m = [0.1, 0.5, 1.0, 2.5, 1.0, 0.5, 0.1, 0.5, 1.0, 2.5, 1.0, 0.5];
    normalize(&mut m);
    test_metric_brief("Extreme (0,6 = 0.1 / 3,9 = 2.5)", &m);

    // 3i: Inverted (0,6 heavy, 3,9 light — should flip sign)
    m = [2.0, 0.8, 1.0, 0.2, 1.0, 0.8, 2.0, 0.8, 1.0, 0.2, 1.0, 0.8];
    normalize(&mut m);
    test_metric_brief("Inverted (0,6 heavy / 3,9 light)", &m);

    println!();
}

// --- Phase 4: Optimization Search ---

fn phase4_optimization() -> ([f64; 12], u32, f64, f64) {
    println!("--- Phase 4: Optimization Search ---");

    let n_restarts = 5000;
    let n_steps = 1000;
    let max_harmonic = 24u32;

    println!("  {} random restarts x {} hill-climbing steps", n_restarts, n_steps);
    println!("  Harmonics tested: 1..{}", max_harmonic);
    println!("  Searching...\n");

    let mut rng = Rng::new(42);
    let mut global_best_metric = [1.0f64; 12];
    let mut global_best_n = 1u32;
    let mut global_best_sep = 0.0f64;
    let mut global_best_signed = 0.0f64;
    let mut improvements = 0u32;

    for _ in 0..n_restarts {
        // Random starting metric
        let mut metric = [0.0f64; 12];
        for g in metric.iter_mut() {
            *g = rng.range(0.2, 2.5);
        }
        normalize(&mut metric);

        let (mut cur_n, mut cur_sep, mut cur_signed) = best_separation(&metric, max_harmonic);

        // Hill climbing
        for _ in 0..n_steps {
            let i = (rng.next_u64() % 12) as usize;
            let j = (rng.next_u64() % 12) as usize;
            if i == j {
                continue;
            }

            let delta = rng.range(0.01, 0.4);
            if metric[i] < delta + 0.05 {
                continue; // don't let weights go near zero
            }

            let old_i = metric[i];
            let old_j = metric[j];
            metric[i] -= delta;
            metric[j] += delta;
            // Sum preserved, no need to normalize

            let (new_n, new_sep, new_signed) = best_separation(&metric, max_harmonic);
            if new_sep > cur_sep {
                cur_sep = new_sep;
                cur_n = new_n;
                cur_signed = new_signed;
            } else {
                metric[i] = old_i;
                metric[j] = old_j;
            }
        }

        if cur_sep > global_best_sep {
            global_best_sep = cur_sep;
            global_best_n = cur_n;
            global_best_signed = cur_signed;
            global_best_metric = metric;
            improvements += 1;
        }
    }

    println!("  Improvements found: {}", improvements);
    println!("  Best separation:    {:.6} at n={}", global_best_sep, global_best_n);
    println!(
        "  Direction:          {}",
        if global_best_signed > 0.0 {
            "Liu He HIGHER (harmony scores above harm)"
        } else {
            "Liu Hai HIGHER (harm scores above harmony)"
        }
    );
    println!("  Best metric:        {}", format_metric(&global_best_metric));
    println!();

    (global_best_metric, global_best_n, global_best_sep, global_best_signed)
}

// --- Phase 5: Full Analysis of Best Result ---

fn phase5_analysis(metric: &[f64; 12], harmonic: u32, separation: f64, signed: f64) {
    println!("--- Phase 5: Best Result Analysis ---");
    println!("Harmonic n={}, separation={:.6}\n", harmonic, separation);

    let flat = [1.0f64; 12];

    // Per-pair breakdown: Liu He
    println!("  Liu He pairs (harmony):");
    println!("    {:18} {:>7} {:>8} {:>10}", "Pair", "flat", "curved", "coh(n)");
    println!("    {:18} {:>7} {:>8} {:>10}", "----", "----", "------", "------");
    let mut lh_cohs = [0.0f64; 6];
    for i in 0..6 {
        let (a, b) = LIU_HE[i];
        let flat_d = geodesic_distance(a, b, &flat);
        let curved_d = geodesic_distance(a, b, metric);
        let c = coherence(curved_d, harmonic);
        lh_cohs[i] = c;
        println!(
            "    {:18} {:>6.1}° {:>7.1}°  {:>+.6}",
            LIU_HE_NAMES[i], flat_d, curved_d, c
        );
    }
    let lh_mean = lh_cohs.iter().sum::<f64>() / 6.0;
    let lh_min = lh_cohs.iter().cloned().fold(f64::INFINITY, f64::min);
    let lh_max = lh_cohs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    println!("    Mean: {:+.6}  Range: [{:+.4}, {:+.4}]\n", lh_mean, lh_min, lh_max);

    // Per-pair breakdown: Liu Hai
    println!("  Liu Hai pairs (harm):");
    println!("    {:18} {:>7} {:>8} {:>10}", "Pair", "flat", "curved", "coh(n)");
    println!("    {:18} {:>7} {:>8} {:>10}", "----", "----", "------", "------");
    let mut lhai_cohs = [0.0f64; 6];
    for i in 0..6 {
        let (a, b) = LIU_HAI[i];
        let flat_d = geodesic_distance(a, b, &flat);
        let curved_d = geodesic_distance(a, b, metric);
        let c = coherence(curved_d, harmonic);
        lhai_cohs[i] = c;
        println!(
            "    {:18} {:>6.1}° {:>7.1}°  {:>+.6}",
            LIU_HAI_NAMES[i], flat_d, curved_d, c
        );
    }
    let lhai_mean = lhai_cohs.iter().sum::<f64>() / 6.0;
    let lhai_min = lhai_cohs.iter().cloned().fold(f64::INFINITY, f64::min);
    let lhai_max = lhai_cohs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    println!(
        "    Mean: {:+.6}  Range: [{:+.4}, {:+.4}]\n",
        lhai_mean, lhai_min, lhai_max
    );

    // Overlap check
    let clean_separation = if signed > 0.0 {
        lh_min > lhai_max
    } else {
        lhai_min > lh_max
    };
    println!(
        "  Overlap: {}",
        if clean_separation {
            "NONE — every Liu He pair scores differently from every Liu Hai pair"
        } else {
            "Some individual pairs overlap (but means still separate)"
        }
    );

    // Full harmonic sweep at best metric
    println!("\n  Full harmonic sweep at best metric:");
    println!("    {:>3}  {:>10}  {:>10}  {:>10}", "n", "Liu He", "Liu Hai", "separation");
    println!("    {:>3}  {:>10}  {:>10}  {:>10}", "---", "------", "-------", "----------");
    for n in 1..=24 {
        let lh = mean_coherence(&LIU_HE, metric, n);
        let lhai = mean_coherence(&LIU_HAI, metric, n);
        let sep = lh - lhai;
        let marker = if n == harmonic { " <-- best" } else { "" };
        println!(
            "    {:>3}  {:>+.6}  {:>+.6}  {:>+.8}{}",
            n, lh, lhai, sep, marker
        );
    }

    // Metric interpretation
    println!("\n  Metric shape analysis:");
    let mean_g: f64 = metric.iter().sum::<f64>() / 12.0;
    let max_g = metric.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_g = metric.iter().cloned().fold(f64::INFINITY, f64::min);
    println!("    Mean weight: {:.3}", mean_g);
    println!("    Range: [{:.3}, {:.3}] (ratio {:.1}x)", min_g, max_g, max_g / min_g);

    // Find heaviest and lightest segments
    let mut indexed: Vec<(usize, f64)> = metric.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("    Heaviest: seg {} ({}->{}, g={:.3})", indexed[0].0,
             NAMES[indexed[0].0], NAMES[(indexed[0].0 + 1) % N_POS], indexed[0].1);
    println!("    Lightest: seg {} ({}->{}, g={:.3})", indexed[11].0,
             NAMES[indexed[11].0], NAMES[(indexed[11].0 + 1) % N_POS], indexed[11].1);

    // Check k=2 sinusoidal fit
    println!("\n  Sinusoidal decomposition of best metric:");
    for k in 0..=6 {
        let mut cos_sum = 0.0f64;
        let mut sin_sum = 0.0f64;
        for i in 0..12 {
            let angle = 2.0 * PI * k as f64 * i as f64 / 12.0;
            cos_sum += metric[i] * angle.cos();
            sin_sum += metric[i] * angle.sin();
        }
        let amplitude = (cos_sum * cos_sum + sin_sum * sin_sum).sqrt() / 6.0;
        let phase = sin_sum.atan2(cos_sum) * 180.0 / PI;
        let marker = if k == 2 { " <-- predicted dominant" } else { "" };
        println!("    k={}: amplitude={:.4}, phase={:+.1}°{}", k, amplitude, phase, marker);
    }

    // Verdict
    println!("\n=== VERDICT ===");
    if separation > 0.5 {
        println!("STRONG CASE ({:.1}% separation)", separation * 100.0);
        println!("A non-uniform metric cleanly separates Liu He from Liu Hai.");
        println!("The 'same angle, opposite meaning' anomaly has a geometric explanation:");
        println!("different paths through differently-weighted regions of the circle");
        println!("produce different geodesic distances despite identical coordinate distances.");
    } else if separation > 0.2 {
        println!("MEASURABLE CASE ({:.1}% separation)", separation * 100.0);
        println!("Non-uniform metric provides partial separation.");
        println!("Curvature contributes to the distinction but may not fully explain it.");
    } else if separation > 0.05 {
        println!("WEAK CASE ({:.1}% separation)", separation * 100.0);
        println!("Curvature alone is insufficient for reliable separation.");
    } else {
        println!("NO CASE ({:.2}% separation)", separation * 100.0);
        println!("Non-uniform metric cannot distinguish the pairs.");
        println!("The Liu He/Liu Hai distinction is genuinely non-geometric.");
    }
}

// --- Main ---

fn main() {
    println!("=== Curvature Test 1: Liu He / Liu Hai Metric Separation ===");
    println!("QUESTION: Can a non-uniform metric distinguish harmony from harm");
    println!("          at identical angular distances?");
    println!("STATUS:   Internal research — not for publication until proven.\n");

    phase1_flat_baseline();
    phase2_path_analysis();
    phase3_structured_metrics();
    let (best_metric, best_n, best_sep, best_signed) = phase4_optimization();
    phase5_analysis(&best_metric, best_n, best_sep, best_signed);
}
