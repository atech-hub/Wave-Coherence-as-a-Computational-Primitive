//! Curvature Test 4: Variable Orbs
//! Can a non-uniform metric predict which aspects get wide vs narrow tolerance windows?
//!
//! ANOMALY #3: All traditions assign different "orbs" (tolerance windows) to different
//! aspect types. Major aspects (trine, square, opposition) get ±7-10°. Minor aspects
//! (quintile, semi-sextile, quincunx) get ±1-2°. On a flat circle, the orb depends
//! ONLY on the harmonic number n (higher n = narrower peak). On a curved circle,
//! the orb also depends on WHERE on the circle the aspect falls — the local metric
//! weight determines how fast coherence decays with perturbation.
//!
//! Internal research — not for publication until proven.

use std::f64::consts::PI;

// ── Metrics from previous tests ──────────────────────────────────────────

const FLAT: [f64; 12] = [1.0; 12];

// Test 3 compromise: Wu Xing + T1 + T2 + T4, only 1 DOF (g5=1.129)
const COMPROMISE: [f64; 12] = [
    0.800, 1.600, 0.800, 0.800, 0.800, 1.129,
    1.271, 0.800, 0.800, 0.800, 1.600, 0.800,
];

// Test 2 combined: Wu Xing + Liu He/Liu Hai optimal
const COMBINED: [f64; 12] = [
    1.798, 0.602, 0.598, 1.201, 0.601, 0.599,
    1.801, 0.600, 0.600, 1.200, 1.802, 0.598,
];

// ── Aspect definitions ──────────────────────────────────────────────────

struct Aspect {
    name: &'static str,
    alpha: f64,    // flat angle in degrees
    n: u32,        // natural harmonic (lowest n where cos(n*alpha) = +1)
    trad_orb: f64, // traditional orb (degrees), modern consensus approximate
}

// Traditional orbs from Ptolemaic + modern synthesis:
// Major (Ptolemaic): opposition ±10°, trine ±8°, square ±7°, sextile ±6°
// Minor: quintile ±2°, semi-sextile ±2°, quincunx ±2°
const ASPECTS: [Aspect; 7] = [
    Aspect { name: "Semi-sextile", alpha:  30.0, n: 12, trad_orb: 2.0 },
    Aspect { name: "Sextile",      alpha:  60.0, n:  6, trad_orb: 6.0 },
    Aspect { name: "Quintile",     alpha:  72.0, n:  5, trad_orb: 2.0 },
    Aspect { name: "Square",       alpha:  90.0, n:  4, trad_orb: 7.0 },
    Aspect { name: "Trine",        alpha: 120.0, n:  3, trad_orb: 8.0 },
    Aspect { name: "Quincunx",     alpha: 150.0, n: 12, trad_orb: 2.0 },
    Aspect { name: "Opposition",   alpha: 180.0, n:  2, trad_orb: 10.0 },
];

// ── Continuous geodesic on piecewise-constant metric ─────────────────────

/// Forward (counterclockwise) geodesic distance from `from` to `to`.
/// The metric g[i] gives the weight of the i-th 30° segment.
fn forward_continuous(from_deg: f64, to_deg: f64, g: &[f64; 12]) -> f64 {
    let from = from_deg.rem_euclid(360.0);
    let to_n = to_deg.rem_euclid(360.0);
    let span = if to_n > from + 1e-10 { to_n - from } else { to_n + 360.0 - from };
    if span < 1e-10 { return 0.0; }

    let mut dist = 0.0;
    let mut pos = from;
    let mut left = span;

    for _ in 0..14 {
        if left < 1e-10 { break; }
        let seg = ((pos.rem_euclid(360.0)) / 30.0).floor() as usize % 12;
        let seg_end = (seg as f64 + 1.0) * 30.0;
        let mut to_boundary = seg_end - pos.rem_euclid(360.0);
        if to_boundary < 1e-10 { to_boundary = 30.0; }
        let step = left.min(to_boundary);
        dist += step * g[seg];
        pos += step;
        left -= step;
    }
    dist
}

/// Geodesic distance (shortest path) between two continuous positions.
fn geodesic_continuous(theta1: f64, theta2: f64, g: &[f64; 12]) -> f64 {
    let diff = (theta1 - theta2).abs().rem_euclid(360.0);
    if diff < 1e-10 || (360.0 - diff).abs() < 1e-10 { return 0.0; }
    forward_continuous(theta1, theta2, g).min(forward_continuous(theta2, theta1, g))
}

fn coherence(dist_deg: f64, n: u32) -> f64 {
    (n as f64 * dist_deg * PI / 180.0).cos()
}

// ── Orb computation ─────────────────────────────────────────────────────

/// Compute the orb (half-width above threshold) for a specific aspect at a specific position.
/// Sweeps δ from -max_delta to +max_delta, finds the widest contiguous above-threshold region.
fn compute_orb(theta: f64, alpha: f64, n: u32, g: &[f64; 12], threshold: f64) -> f64 {
    let step = 0.1;
    let max_delta = 15.0;
    let n_steps = (2.0 * max_delta / step) as usize + 1;

    let mut max_run = 0usize;
    let mut current_run = 0usize;

    for i in 0..n_steps {
        let delta = -max_delta + i as f64 * step;
        let d = geodesic_continuous(theta, theta + alpha + delta, g);
        let c = coherence(d, n);
        if c >= threshold {
            current_run += 1;
            if current_run > max_run { max_run = current_run; }
        } else {
            current_run = 0;
        }
    }

    max_run as f64 * step / 2.0 // half-width = orb
}

/// Compute orb statistics for an aspect across all starting positions (1° resolution).
fn orb_stats(alpha: f64, n: u32, g: &[f64; 12], threshold: f64) -> (f64, f64, f64, f64, Vec<f64>) {
    let mut orbs = Vec::with_capacity(360);
    for theta_i in 0..360 {
        orbs.push(compute_orb(theta_i as f64, alpha, n, g, threshold));
    }
    let mean = orbs.iter().sum::<f64>() / orbs.len() as f64;
    let min = orbs.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = orbs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let std = (orbs.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / orbs.len() as f64).sqrt();
    (mean, min, max, std, orbs)
}

// ── Correlation ─────────────────────────────────────────────────────────

fn correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut vx = 0.0;
    let mut vy = 0.0;
    for i in 0..x.len() {
        let dx = x[i] - mx;
        let dy = y[i] - my;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    if vx < 1e-20 || vy < 1e-20 { return 0.0; }
    cov / (vx.sqrt() * vy.sqrt())
}

/// Mean absolute error between two vectors.
fn mae(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y.iter()).map(|(&a, &b)| (a - b).abs()).sum::<f64>() / x.len() as f64
}

// ── Main ────────────────────────────────────────────────────────────────

fn main() {
    println!("=== Curvature Test 4: Variable Orbs ===");
    println!("QUESTION: Does metric curvature predict traditional orb widths?");
    println!("STATUS:   Internal research — not for publication until proven.\n");

    let threshold = 0.9;
    let ac = (threshold as f64).acos() * 180.0 / PI;

    // ── Phase 1: Flat baseline ──────────────────────────────────────────

    println!("--- Phase 1: Flat Baseline ---\n");
    println!("  Coherence threshold: cos(n × d) >= {:.1}", threshold);
    println!("  arccos({:.1}) = {:.2}°\n", threshold, ac);

    let trad_orbs: Vec<f64> = ASPECTS.iter().map(|a| a.trad_orb).collect();
    let flat_orbs: Vec<f64> = ASPECTS.iter().map(|a| ac / a.n as f64).collect();

    println!("  {:15} {:>3} {:>9} {:>9} {:>9}", "Aspect", "n", "Flat orb", "Trad orb", "Ratio");
    println!("  {:15} {:>3} {:>9} {:>9} {:>9}", "------", "-", "--------", "--------", "-----");
    for (i, asp) in ASPECTS.iter().enumerate() {
        let ratio = flat_orbs[i] / asp.trad_orb;
        println!("  {:15} {:>3} {:>8.1}° {:>8.1}° {:>8.2}",
                 asp.name, asp.n, flat_orbs[i], asp.trad_orb, ratio);
    }

    let r_flat = correlation(&flat_orbs, &trad_orbs);
    let mae_flat = mae(&flat_orbs, &trad_orbs);
    println!("\n  Flat <-> Traditional:");
    println!("    Correlation: {:.4}", r_flat);
    println!("    MAE: {:.2}°", mae_flat);

    // What flat gets wrong:
    println!("\n  Flat model anomalies (ratio far from 1.0):");
    for (i, asp) in ASPECTS.iter().enumerate() {
        let ratio = flat_orbs[i] / asp.trad_orb;
        if ratio > 1.5 || ratio < 0.67 {
            println!("    {} : flat {:.1}° vs trad {:.1}° (ratio {:.2}) — {}",
                     asp.name, flat_orbs[i], asp.trad_orb, ratio,
                     if ratio > 1.0 { "predicted too WIDE" } else { "predicted too NARROW" });
        }
    }

    // ── Phase 2: Curved orbs ────────────────────────────────────────────

    let metrics: [(&str, &[f64; 12]); 2] = [
        ("Compromise (Test 3)", &COMPROMISE),
        ("Combined (Test 2)", &COMBINED),
    ];

    let mut all_curved_means: Vec<Vec<f64>> = Vec::new();

    for (mi, &(mname, g)) in metrics.iter().enumerate() {
        println!("\n--- Phase {}: Curved Orbs — {} ---\n", mi + 2, mname);
        println!("  Metric: [{}]\n",
                 g.iter().map(|x| format!("{:.3}", x)).collect::<Vec<_>>().join(", "));

        let mut curved_means = Vec::new();
        let mut all_orb_data: Vec<Vec<f64>> = Vec::new();

        println!("  {:15} {:>3} {:>8} {:>8} {:>8} {:>8} {:>8}",
                 "Aspect", "n", "Mean", "Min", "Max", "Std", "Trad");
        println!("  {:15} {:>3} {:>8} {:>8} {:>8} {:>8} {:>8}",
                 "------", "-", "----", "---", "---", "---", "----");

        for asp in ASPECTS.iter() {
            let (mean, min, max, std, orbs) = orb_stats(asp.alpha, asp.n, g, threshold);
            curved_means.push(mean);
            all_orb_data.push(orbs);

            println!("  {:15} {:>3} {:>7.1}° {:>7.1}° {:>7.1}° {:>7.2}° {:>7.1}°",
                     asp.name, asp.n, mean, min, max, std, asp.trad_orb);
        }

        let r_curved = correlation(&curved_means, &trad_orbs);
        let mae_curved = mae(&curved_means, &trad_orbs);
        println!("\n  Curved <-> Traditional:");
        println!("    Correlation: {:.4}", r_curved);
        println!("    MAE: {:.2}°", mae_curved);
        println!("  vs Flat baseline:");
        println!("    Corr improvement: {:+.4} ({:.4} -> {:.4})", r_curved - r_flat, r_flat, r_curved);
        println!("    MAE  improvement: {:+.2}° ({:.2}° -> {:.2}°)", mae_flat - mae_curved, mae_flat, mae_curved);

        // Per-aspect improvement
        println!("\n  Per-aspect comparison:");
        println!("  {:15} {:>8} {:>8} {:>8} {:>9}", "Aspect", "Flat", "Curved", "Trad", "Improved?");
        println!("  {:15} {:>8} {:>8} {:>8} {:>9}", "------", "----", "------", "----", "---------");
        for (i, asp) in ASPECTS.iter().enumerate() {
            let flat_err = (flat_orbs[i] - asp.trad_orb).abs();
            let curv_err = (curved_means[i] - asp.trad_orb).abs();
            let improved = if curv_err < flat_err - 0.05 { "YES" }
                           else if curv_err > flat_err + 0.05 { "worse" }
                           else { "same" };
            println!("  {:15} {:>7.1}° {:>7.1}° {:>7.1}° {:>9}",
                     asp.name, flat_orbs[i], curved_means[i], asp.trad_orb, improved);
        }

        // ── Position-dependent orb map ──────────────────────────────────
        // Show for the most interesting aspects: quintile (biggest flat anomaly)
        // and trine (most important major aspect)
        println!("\n  Position-dependent orb map (every 30°):");
        let interesting = [2usize, 4]; // quintile, trine
        for &ai in &interesting {
            let asp = &ASPECTS[ai];
            println!("    {} (α={}°, n={}):", asp.name, asp.alpha, asp.n);
            print!("      θ:   ");
            for t in (0..360).step_by(30) {
                print!("{:>6}°", t);
            }
            println!();
            print!("      orb: ");
            for t in (0..360).step_by(30) {
                print!("{:>6.1}°", all_orb_data[ai][t]);
            }
            println!();
            print!("      g:   ");
            for t in (0..360).step_by(30) {
                let seg = t / 30;
                print!("{:>6.2}", g[seg]);
            }
            println!();
        }

        all_curved_means.push(curved_means);
    }

    // ── Phase 4: Local metric analysis ──────────────────────────────────

    println!("\n--- Phase 4: Local Metric -> Orb Relationship ---\n");
    println!("  Theory: orb(θ) ∝ 1/g(θ+α)  (higher metric weight at endpoint -> narrower orb)");
    println!("  Testing with Combined metric (widest metric variation):\n");

    for asp in ASPECTS.iter() {
        let mut orbs = Vec::new();
        let mut inv_g_at_target = Vec::new();
        for theta_i in 0..360 {
            let theta = theta_i as f64;
            let orb = compute_orb(theta, asp.alpha, asp.n, &COMBINED, threshold);
            let target = (theta + asp.alpha).rem_euclid(360.0);
            let seg = (target / 30.0).floor() as usize % 12;
            orbs.push(orb);
            inv_g_at_target.push(1.0 / COMBINED[seg]);
        }
        let r = correlation(&orbs, &inv_g_at_target);
        println!("  {:15} : orb ~ 1/g(target) correlation = {:+.4}", asp.name, r);
    }

    // ── Phase 5: The harmonic-number hypothesis ─────────────────────────

    println!("\n--- Phase 5: What Actually Explains Orb Width? ---\n");

    // Flat model: orb = arccos(τ) / n
    // This is purely harmonic-number-dependent. Does curvature add anything?

    let n_values: Vec<f64> = ASPECTS.iter().map(|a| a.n as f64).collect();
    let inv_n: Vec<f64> = n_values.iter().map(|&n| 1.0 / n).collect();

    let r_inv_n_trad = correlation(&inv_n, &trad_orbs);
    println!("  1/n <-> Traditional orb:   r = {:.4}  (harmonic-number explanation)", r_inv_n_trad);
    println!("  Flat <-> Traditional orb:  r = {:.4}  (= same thing, since flat orb ∝ 1/n)", r_flat);

    for (mi, &(mname, _g)) in metrics.iter().enumerate() {
        let r = correlation(&all_curved_means[mi], &trad_orbs);
        println!("  {} <-> Traditional: r = {:.4}", mname, r);
    }

    // Does curvature break the 1/n scaling?
    println!("\n  Curvature effect = curved_mean - flat_orb:");
    println!("  {:15} {:>8} {:>8} {:>8}", "Aspect", "Flat", "Curved", "Δ(curved-flat)");
    println!("  {:15} {:>8} {:>8} {:>8}", "------", "----", "------", "--------------");
    for (i, asp) in ASPECTS.iter().enumerate() {
        let delta = all_curved_means[1][i] - flat_orbs[i]; // using combined metric
        println!("  {:15} {:>7.2}° {:>7.2}° {:>+7.2}°",
                 asp.name, flat_orbs[i], all_curved_means[1][i], delta);
    }

    // ── Phase 6: Alternative — does the metric VARIANCE predict orb? ────

    println!("\n--- Phase 6: Metric Variance Along Aspect Paths ---\n");
    println!("  For each aspect, compute the variance of metric weights along");
    println!("  the geodesic path. Higher variance -> more position-dependent -> wider spread.\n");

    for asp in ASPECTS.iter() {
        // For each starting position, collect the metric values along the path
        let mut path_variances = Vec::new();
        for theta_i in 0..360 {
            let theta = theta_i as f64;
            // Collect segment weights along the path
            let mut weights = Vec::new();
            let start_seg = (theta / 30.0).floor() as usize % 12;
            let n_segs = (asp.alpha / 30.0).ceil() as usize;
            for s in 0..n_segs {
                weights.push(COMBINED[(start_seg + s) % 12]);
            }
            let mean_w = weights.iter().sum::<f64>() / weights.len() as f64;
            let var = weights.iter().map(|&w| (w - mean_w).powi(2)).sum::<f64>() / weights.len() as f64;
            path_variances.push(var);
        }
        let mean_var = path_variances.iter().sum::<f64>() / path_variances.len() as f64;
        let (mean_orb, _, _, std_orb, _) = orb_stats(asp.alpha, asp.n, &COMBINED, threshold);
        println!("  {:15} : path metric variance = {:.4}, orb std = {:.2}°, orb mean = {:.1}°",
                 asp.name, mean_var, std_orb, mean_orb);
    }

    // ── Phase 7: Multi-threshold robustness ─────────────────────────────

    println!("\n--- Phase 7: Robustness Across Thresholds ---\n");
    println!("  Does the ranking hold at different coherence thresholds?\n");

    let thresholds: [f64; 4] = [0.5, 0.7, 0.9, 0.95];
    println!("  Flat <-> Trad correlation at each threshold:");
    for &tau in &thresholds {
        let ac_tau = tau.acos() * 180.0 / PI;
        let flat_tau: Vec<f64> = ASPECTS.iter().map(|a| ac_tau / a.n as f64).collect();
        let r = correlation(&flat_tau, &trad_orbs);
        println!("    τ={:.2}: r={:.4}", tau, r);
    }

    println!("\n  Combined metric <-> Trad correlation at each threshold:");
    for &tau in &thresholds {
        let mut means = Vec::new();
        for asp in ASPECTS.iter() {
            let (mean, _, _, _, _) = orb_stats(asp.alpha, asp.n, &COMBINED, tau);
            means.push(mean);
        }
        let r = correlation(&means, &trad_orbs);
        println!("    τ={:.2}: r={:.4}", tau, r);
    }

    // ── Phase 8: Summary + Verdict ──────────────────────────────────────

    println!("\n=== SUMMARY ===\n");

    println!("  {:15} {:>8} {:>8} {:>8} {:>8}",
             "Aspect", "Flat", "Compr.", "Comb.", "Trad");
    println!("  {:15} {:>8} {:>8} {:>8} {:>8}",
             "------", "----", "------", "-----", "----");
    for (i, asp) in ASPECTS.iter().enumerate() {
        println!("  {:15} {:>7.1}° {:>7.1}° {:>7.1}° {:>7.1}°",
                 asp.name, flat_orbs[i], all_curved_means[0][i],
                 all_curved_means[1][i], asp.trad_orb);
    }
    println!();

    let r_comp = correlation(&all_curved_means[0], &trad_orbs);
    let r_comb = correlation(&all_curved_means[1], &trad_orbs);
    println!("  Correlation with tradition:");
    println!("    Flat:       {:.4}", r_flat);
    println!("    Compromise: {:.4}", r_comp);
    println!("    Combined:   {:.4}", r_comb);

    println!("\n=== VERDICT ===\n");

    if (r_comb - r_flat).abs() < 0.02 && (r_comp - r_flat).abs() < 0.02 {
        println!("  CURVATURE DOES NOT SIGNIFICANTLY PREDICT VARIABLE ORBS.");
        println!("  The orb pattern is dominated by HARMONIC NUMBER (1/n),");
        println!("  which is a FLAT-CIRCLE property. Curvature adds position-");
        println!("  dependent variation but does not change the aspect-level ranking.");
        println!();
        println!("  This is an HONEST NULL RESULT for anomaly #3.");
        println!("  Variable orbs are explained by the coherence function's");
        println!("  natural peak width: cos(n*d) has half-width arccos(τ)/n.");
        println!("  Higher harmonics = narrower peaks = smaller orbs. Simple.");
    } else if r_comb > r_flat + 0.05 || r_comp > r_flat + 0.05 {
        println!("  CURVATURE IMPROVES ORB PREDICTION.");
        println!("  The curved metric shifts orb widths toward traditional values");
        println!("  beyond what harmonic number alone explains.");
    } else {
        println!("  CURVATURE DEGRADES ORB PREDICTION.");
        println!("  The metric optimized for other anomalies (Liu He/Liu Hai,");
        println!("  Wu Xing) actually makes orb prediction worse. The orb system");
        println!("  may be tuned to flat geometry even if other systems are not.");
    }

    println!();
    println!("  WHAT HOLDS from Tests 1-3:");
    println!("    - Liu He/Liu Hai: curvature explains opposite meanings at same angle");
    println!("    - Wu Xing: curvature explains 72° resonance on 30°-grid (0.300 identity)");
    println!("    - Geometric comma: 24° incompatibility between 3rd and 5th harmonics");
    println!("  WHAT TEST 4 SHOWS:");
    if (r_comb - r_flat).abs() < 0.02 && (r_comp - r_flat).abs() < 0.02 {
        println!("    - Variable orbs are NOT a curvature signature");
        println!("    - They are a HARMONIC NUMBER signature (flat-circle property)");
        println!("    - The orb ranking is: opposition > trine > square > sextile > minor");
        println!("    - This follows directly from n: 2 < 3 < 4 < 6 < 12 -> wider to narrower");
    } else {
        println!("    - Curvature has a measurable effect on orb widths");
        println!("    - Details above show which aspects are affected and by how much");
    }
}
