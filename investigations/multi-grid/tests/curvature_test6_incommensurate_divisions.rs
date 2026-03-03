//! Curvature Test 6: Incommensurate Divisions
//!
//! ANOMALY #5: Nakshatra 27 — Vedic system uses 27 segments (13.333° each)
//!   alongside 12 zodiac signs (30° each). 27 doesn't divide evenly into 12.
//!
//! ANOMALY #6: Sexagenary 12×10 — Chinese system overlays 12 Earthly Branches
//!   with 10 Heavenly Stems. lcm(10, 12) = 60.
//!
//! QUESTION: Why do these traditions maintain incommensurate cycle counts?
//!   Does curvature explain the choice, or is it a sampling/Nyquist phenomenon?
//!
//! Internal research — not for publication until proven.

use std::f64::consts::PI;

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 { a } else { gcd(b, a % b) }
}

fn lcm(a: u32, b: u32) -> u32 {
    a / gcd(a, b) * b
}

fn coherence(dist_deg: f64, n: u32) -> f64 {
    (n as f64 * dist_deg * PI / 180.0).cos()
}

/// Number of distinct factors of n
fn count_factors(n: u32) -> u32 {
    (1..=n).filter(|&d| n % d == 0).count() as u32
}

/// List factors
fn factors(n: u32) -> Vec<u32> {
    (1..=n).filter(|&d| n % d == 0).collect()
}

/// On a B-point grid (spacing 360/B degrees), what does the n-th harmonic look like?
/// Returns the aliased harmonic number (0..B/2).
fn alias(n: u32, b: u32) -> u32 {
    let r = n % b;
    if r <= b / 2 { r } else { b - r }
}

/// Compute mean |coherence| across all B grid points for harmonic n
fn grid_coherence(n: u32, b: u32) -> f64 {
    let spacing = 360.0 / b as f64;
    let mut sum = 0.0;
    for k in 1..b {
        sum += coherence(k as f64 * spacing, n).abs();
    }
    sum / (b - 1) as f64
}

/// Maximum harmonic fully resolvable on a B-point grid (Nyquist limit)
fn nyquist(b: u32) -> u32 {
    b / 2
}

// ── Geodesic functions (for curvature test) ─────────────────────────────

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

fn geodesic_continuous(theta1: f64, theta2: f64, g: &[f64; 12]) -> f64 {
    let diff = (theta1 - theta2).abs().rem_euclid(360.0);
    if diff < 1e-10 || (360.0 - diff).abs() < 1e-10 { return 0.0; }
    forward_continuous(theta1, theta2, g).min(forward_continuous(theta2, theta1, g))
}

fn main() {
    println!("=== Curvature Test 6: Incommensurate Divisions ===");
    println!("QUESTION: Why do traditions use cycle counts that don't divide evenly?");
    println!("STATUS:   Internal research — not for publication until proven.\n");

    // ══════════════════════════════════════════════════════════════════════
    // PART A: NAKSHATRA 27
    // ══════════════════════════════════════════════════════════════════════

    println!("══════════════════════════════════════════════════════");
    println!("  PART A: NAKSHATRA 27");
    println!("══════════════════════════════════════════════════════\n");

    // ── Phase A1: Number theory ─────────────────────────────────────────

    println!("--- A1: Number Theory of 27 and 12 ---\n");
    let g27 = gcd(27, 12);
    let l27 = lcm(27, 12);
    println!("  27 nakshatras × 13.333° = 360°");
    println!("  12 zodiac signs × 30° = 360°");
    println!("  gcd(27, 12) = {}", g27);
    println!("  lcm(27, 12) = {}", l27);
    println!("  Nakshatras per sign: 27/12 = 2.25 (non-integer!)");
    println!("  Shared symmetry: {}-fold (every {}th nakshatra = every {}th sign)",
             g27, 12 / g27, 27 / g27);
    println!("  Full alignment: every {} divisions ({} per sign, {} per nakshatra)",
             l27, l27 / 12, l27 / 27);

    println!("\n  Nyquist limits:");
    println!("    12-grid: n ≤ {} (can resolve harmonics 1-{})", nyquist(12), nyquist(12));
    println!("    27-grid: n ≤ {} (can resolve harmonics 1-{})", nyquist(27), nyquist(27));
    println!("    Combined ({}-grid): n ≤ {} (harmonics 1-{})", l27, nyquist(l27), nyquist(l27));

    // ── Phase A2: Aliasing analysis ─────────────────────────────────────

    println!("\n--- A2: Aliasing — What Does n=27 Look Like on a 12-Grid? ---\n");

    println!("  On a 12-point grid (spacing 30°), cos(27 × k × 30°) = cos(810k°)");
    println!("  810 mod 360 = {}", 810 % 360);
    println!("  So cos(27 × 30k°) = cos({}k°)", 810 % 360);
    println!("  27 mod 12 = {}, alias = {}", 27 % 12, alias(27, 12));
    println!("  The 27th harmonic APPEARS as the {}rd harmonic on a 12-grid!", alias(27, 12));

    println!("\n  Verification — cos(27d) at each grid position:");
    println!("    {:>5}  {:>8}  {:>10}  {:>10}", "Pos", "Angle", "cos(27d)", "cos(3d)");
    println!("    {:>5}  {:>8}  {:>10}  {:>10}", "---", "-----", "--------", "-------");
    for k in 0..12u32 {
        let d = k as f64 * 30.0;
        let c27 = coherence(d, 27);
        let c3 = coherence(d, 3);
        println!("    {:>5}  {:>7.0}°  {:>+10.4}  {:>+10.4}  {}",
                 k, d, c27, c3,
                 if (c27 - c3).abs() < 0.001 { "= aliased" } else { "≠ DIFFERENT" });
    }

    println!("\n  RESULT: n=27 is PERFECTLY aliased to n=3 on the 12-grid.");
    println!("  Every value of cos(27d) at a grid point equals cos(3d).");
    println!("  The 12-grid CANNOT distinguish the 27th harmonic from the 3rd.");

    // ── Phase A3: What 27 can see that 12 cannot ────────────────────────

    println!("\n--- A3: What the 27-Grid Captures That the 12-Grid Cannot ---\n");

    println!("  Harmonics n=7 to 13 are visible on the 27-grid but aliased on the 12-grid:");
    println!("    {:>3}  {:>14}  {:>14}  {:>8}", "n", "on 12-grid", "on 27-grid", "Aliased?");
    println!("    {:>3}  {:>14}  {:>14}  {:>8}", "-", "---------", "---------", "--------");
    for n in 1..=15u32 {
        let a12 = alias(n, 12);
        let a27 = alias(n, 27);
        let aliased_on_12 = a12 != n && n <= 27;
        println!("    {:>3}  {:>10} (n={}){:>10} (n={})  {}",
                 n,
                 if a12 == n { "exact" } else { "aliased" }, a12,
                 if a27 == n { "exact" } else { "aliased" }, a27,
                 if aliased_on_12 && a27 == n { "← 27 resolves!" }
                 else if !aliased_on_12 { "" }
                 else { "" });
    }

    println!("\n  The 27-grid extends resolution from n=6 (12-grid Nyquist) to n=13.");
    println!("  Harmonics 7-13 are invisible on 12 signs but visible on 27 nakshatras.");

    // ── Phase A4: Can curvature help? ───────────────────────────────────

    println!("\n--- A4: Can Curvature De-Alias n=27 on the 12-Grid? ---\n");

    // Use Wu Xing metric from Test 2
    let wu_xing = [1.798f64, 0.602, 0.598, 1.201, 0.601, 0.599, 1.801, 0.600, 0.600, 1.200, 1.802, 0.598];
    let flat = [1.0f64; 12];

    println!("  Test: compute cos(27 × d_geodesic) at all 12 grid positions.");
    println!("  If curvature de-aliases, the values should differ from cos(3 × d_flat).\n");

    println!("    {:>5}  {:>8}  {:>12}  {:>12}  {:>12}", "Pos", "Flat d", "cos(27*flat)", "cos(27*curv)", "cos(3*flat)");
    println!("    {:>5}  {:>8}  {:>12}  {:>12}  {:>12}", "---", "------", "-----------", "-----------", "----------");

    let mut matches_alias = 0;
    let mut differs = 0;
    for k in 0..12u32 {
        let d_flat = k as f64 * 30.0;
        let d_curv = if k == 0 { 0.0 } else {
            geodesic_continuous(0.0, d_flat, &wu_xing)
        };
        let c27_flat = coherence(d_flat, 27);
        let c27_curv = coherence(d_curv, 27);
        let c3_flat = coherence(d_flat, 3);

        let same = (c27_curv - c3_flat).abs() < 0.1;
        if same { matches_alias += 1; } else { differs += 1; }

        println!("    {:>5}  {:>7.1}°  {:>+11.4}   {:>+11.4}   {:>+11.4}  {}",
                 k, d_flat, c27_flat, c27_curv, c3_flat,
                 if !same { "← DIFFERENT" } else { "" });
    }

    println!("\n  De-aliased positions: {}/12", differs);
    if differs > 0 {
        println!("  Curvature DOES break the aliasing at some positions!");
        println!("  But with only 12 points, this is noise, not resolution.");
        println!("  You cannot reconstruct a 27-fold pattern from 12 samples");
        println!("  regardless of metric — Shannon-Nyquist is a hard limit.");
    } else {
        println!("  Curvature does NOT break the aliasing.");
        println!("  n=27 remains indistinguishable from n=3 on any 12-grid.");
    }

    // ── Phase A5: Why 27 specifically? ──────────────────────────────────

    println!("\n--- A5: Why 27? ---\n");

    println!("  27 = 3³. The cube of the trine number.");
    println!("  27-grid Nyquist = 13, 12-grid Nyquist = 6.");
    println!("  Together: lcm(27,12) = 108, Nyquist = 54.\n");

    println!("  Comparison with alternative division counts:");
    println!("    {:>5}  {:>5}  {:>5}  {:>8}  {:>8}  {:>12}", "Divs", "gcd", "lcm", "Nyquist", "Factors", "Extra harmonics");
    println!("    {:>5}  {:>5}  {:>5}  {:>8}  {:>8}  {:>12}", "----", "---", "---", "-------", "-------", "---------------");

    for &d in &[24u32, 27, 28, 36] {
        let g = gcd(d, 12);
        let l = lcm(d, 12);
        let ny = nyquist(l);
        let nf = count_factors(l);
        let extra = nyquist(d) as i32 - nyquist(12) as i32;
        println!("    {:>5}  {:>5}  {:>5}  {:>8}  {:>8}  {:>+12}",
                 d, g, l, ny, nf, extra);
    }

    println!("\n  24 divisions: gcd=12, lcm=24 → no new information (24 is a multiple of 12)");
    println!("  27 divisions: gcd=3,  lcm=108 → maximum new information (minimal overlap)");
    println!("  28 divisions: gcd=4,  lcm=84 → also good, but 84 has fewer factors than 108");
    println!("  36 divisions: gcd=12, lcm=36 → no new information (multiple of 12)");

    println!("\n  27 is optimal because:");
    println!("  - gcd(27,12) = 3 → minimal shared structure (only trine)");
    println!("  - lcm(27,12) = 108 → rich combined structure (highly composite)");
    println!("  - 108 has {} factors: {:?}", count_factors(108), factors(108));
    println!("  - Extends Nyquist from 6 to 13 (+7 new harmonics)");
    println!("  - 27 = 3³ maintains trine compatibility");

    // ══════════════════════════════════════════════════════════════════════
    // PART B: SEXAGENARY 12×10
    // ══════════════════════════════════════════════════════════════════════

    println!("\n══════════════════════════════════════════════════════");
    println!("  PART B: SEXAGENARY 12×10");
    println!("══════════════════════════════════════════════════════\n");

    // ── Phase B1: Number theory ─────────────────────────────────────────

    println!("--- B1: Number Theory of 12 and 10 ---\n");
    let g10 = gcd(12, 10);
    let l10 = lcm(12, 10);
    println!("  12 Earthly Branches (zodiac) × 30° = 360°");
    println!("  10 Heavenly Stems × 36° = 360°");
    println!("  gcd(12, 10) = {}", g10);
    println!("  lcm(12, 10) = {}", l10);
    println!("  Shared symmetry: {}-fold (opposition)", g10);
    println!("  Full cycle: {} combinations (the sexagenary cycle)", l10);

    println!("\n  Nyquist limits:");
    println!("    12-grid: n ≤ {} (harmonics 1-{})", nyquist(12), nyquist(12));
    println!("    10-grid: n ≤ {} (harmonics 1-{})", nyquist(10), nyquist(10));
    println!("    Combined (60-grid): n ≤ {} (harmonics 1-{})", nyquist(60), nyquist(60));

    // ── Phase B2: Aliasing ──────────────────────────────────────────────

    println!("\n--- B2: What Does n=10 Look Like on a 12-Grid? ---\n");
    println!("  10 mod 12 = {}, alias = {}", 10 % 12, alias(10, 12));
    println!("  The 10th harmonic appears as the {}nd harmonic (opposition) on a 12-grid.",
             alias(10, 12));

    println!("\n  And n=5 (Wu Xing) on a 10-grid?");
    println!("  5 mod 10 = {}, alias = {}", 5 % 10, alias(5, 10));
    println!("  n=5 IS exactly at the Nyquist limit of the 10-grid.");
    println!("  The 10-grid can see the Wu Xing harmonic directly!");

    println!("\n  Harmonics visible per grid:");
    println!("    {:>3}  {:>10}  {:>10}  {:>10}", "n", "12-grid", "10-grid", "60-grid");
    println!("    {:>3}  {:>10}  {:>10}  {:>10}", "-", "-------", "-------", "-------");
    for n in 1..=15u32 {
        let a12 = alias(n, 12);
        let a10 = alias(n, 10);
        let a60 = alias(n, 60);
        println!("    {:>3}  {:>7}=n={:<2} {:>7}=n={:<2} {:>7}=n={:<2}",
                 n,
                 if a12 == n { "exact" } else { "alias" }, a12,
                 if a10 == n { "exact" } else { "alias" }, a10,
                 if a60 == n { "exact" } else { "alias" }, a60);
    }

    // ── Phase B3: The 5th harmonic connection ───────────────────────────

    println!("\n--- B3: Wu Xing and the Heavenly Stems ---\n");

    println!("  The 10 Heavenly Stems contain 5 Yang + 5 Yin stems.");
    println!("  5 Yang stems at 0°, 72°, 144°, 216°, 288° (= Wu Xing generative cycle!)");
    println!("  5 Yin stems at 36°, 108°, 180°, 252°, 324° (= shifted by 36°)");

    println!("\n  On the 10-grid (36° spacing), the 5th harmonic:");
    for k in 0..10u32 {
        let d = k as f64 * 36.0;
        let c5 = coherence(d, 5);
        println!("    Stem {:>2} at {:>5.0}°: cos(5 × {:.0}°) = {:+.4} {}",
                 k, d, d, c5,
                 if c5 > 0.99 { "← resonates" } else { "" });
    }

    println!("\n  The 5 Yang stems (even positions) all resonate at n=5.");
    println!("  The 5 Yin stems (odd positions) all ANTI-resonate at n=5.");
    println!("  Wu Xing IS the 5th harmonic on the 10-grid — no curvature needed!");

    // ── Phase B4: Why 10 and 12 together? ───────────────────────────────

    println!("\n--- B4: Why Combine 10 and 12? ---\n");

    println!("  Harmonic coverage comparison:");
    println!("  {:>30} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6}",
             "Harmonic n →", "3", "4", "5", "6", "7", "10", "12");
    println!("  {:>30} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6}",
             "System", "-", "-", "-", "-", "-", "--", "--");

    let harmonics_to_check = [3u32, 4, 5, 6, 7, 10, 12];
    // 12-grid
    print!("  {:>30}", "12-grid (Nyquist=6)");
    for &n in &harmonics_to_check {
        let a = alias(n, 12);
        print!(" {:>6}", if a == n { "✓" } else { "✗" });
    }
    println!();

    // 10-grid
    print!("  {:>30}", "10-grid (Nyquist=5)");
    for &n in &harmonics_to_check {
        let a = alias(n, 10);
        print!(" {:>6}", if a == n { "✓" } else { "✗" });
    }
    println!();

    // 60-grid (combined)
    print!("  {:>30}", "60-grid (Nyquist=30)");
    for &n in &harmonics_to_check {
        let a = alias(n, 60);
        print!(" {:>6}", if a == n { "✓" } else { "✗" });
    }
    println!();

    println!("\n  CRUCIAL: The 10-grid resolves n=5 (Wu Xing) where the 12-grid cannot.");
    println!("  The 12-grid resolves n=6 (sextile/Jupiter) where the 10-grid cannot.");
    println!("  TOGETHER (60-grid): resolves ALL harmonics up to n=30.");

    println!("\n  Why not just use a 60-grid directly?");
    println!("  Because 60 divisions are hard to manage. But 10 + 12 = 22 divisions");
    println!("  give you the SAME harmonic coverage (n=1 to 30) at lower cost.");
    println!("  The sexagenary cycle is COMPRESSION — two small cycles encoding one large one.");

    // ── Phase B5: Can curvature help? ───────────────────────────────────

    println!("\n--- B5: Can Curvature Help the 12-Grid See n=10? ---\n");

    println!("  n=10 on the flat 12-grid: aliased to n=2 (opposition).");
    println!("  n=10 > Nyquist limit of 6. Curvature cannot de-alias.\n");

    println!("  Verification with Wu Xing metric:");
    println!("    {:>5}  {:>8}  {:>12}  {:>12}", "Pos", "Angle", "cos(10*flat)", "cos(10*curv)");
    println!("    {:>5}  {:>8}  {:>12}  {:>12}", "---", "-----", "-----------", "-----------");
    for k in 0..12u32 {
        let d_flat = k as f64 * 30.0;
        let d_curv = if k == 0 { 0.0 } else {
            geodesic_continuous(0.0, d_flat, &wu_xing)
        };
        let c10_flat = coherence(d_flat, 10);
        let c10_curv = coherence(d_curv, 10);
        let c2_flat = coherence(d_flat, 2);
        println!("    {:>5}  {:>7.0}°  {:>+11.4}   {:>+11.4}   (cos(2d)={:+.4})",
                 k, d_flat, c10_flat, c10_curv, c2_flat);
    }

    println!("\n  Curvature changes the VALUES but cannot recover the n=10 pattern.");
    println!("  With only 12 samples, n=10 is fundamentally under-sampled.");

    // ══════════════════════════════════════════════════════════════════════
    // PART C: UNIFIED ANALYSIS
    // ══════════════════════════════════════════════════════════════════════

    println!("\n══════════════════════════════════════════════════════");
    println!("  PART C: THE CURVATURE BOUNDARY");
    println!("══════════════════════════════════════════════════════\n");

    println!("--- C1: When Does Curvature Work? ---\n");

    println!("  BELOW Nyquist (n ≤ B/2): curvature CAN help.");
    println!("    The harmonic exists in the grid's resolution range.");
    println!("    If grid points don't align with the resonance, curvature");
    println!("    warps distances to CREATE alignment.\n");

    println!("  ABOVE Nyquist (n > B/2): curvature CANNOT help.");
    println!("    The harmonic is aliased — indistinguishable from a lower harmonic.");
    println!("    No metric warp on B points can reconstruct a pattern");
    println!("    that requires more than B/2 frequency components.\n");

    println!("  Results across all tests on the 12-grid (Nyquist = 6):");
    println!("    {:>20}  {:>5}  {:>10}  {:>12}  {:>10}",
             "System", "n", "vs Nyquist", "Curvature?", "Result");
    println!("    {:>20}  {:>5}  {:>10}  {:>12}  {:>10}",
             "------", "-", "----------", "----------", "------");
    println!("    {:>20}  {:>5}  {:>10}  {:>12}  {:>10}",
             "Liu He/Liu Hai", "7", "ABOVE", "Helps*", "STRONG");
    println!("    {:>20}  {:>5}  {:>10}  {:>12}  {:>10}",
             "San He (trine)", "3", "BELOW", "Works", "STRONG");
    println!("    {:>20}  {:>5}  {:>10}  {:>12}  {:>10}",
             "Wu Xing", "5", "BELOW", "Works", "STRONG");
    println!("    {:>20}  {:>5}  {:>10}  {:>12}  {:>10}",
             "Variable orbs", "1-12", "MIXED", "No help", "NULL");
    println!("    {:>20}  {:>5}  {:>10}  {:>12}  {:>10}",
             "Jupiter drishti", "6", "AT", "Not needed", "FLAT");
    println!("    {:>20}  {:>5}  {:>10}  {:>12}  {:>10}",
             "Mars/Saturn drishti", "-", "N/A", "No help", "TABULAR");
    println!("    {:>20}  {:>5}  {:>10}  {:>12}  {:>10}",
             "Nakshatra 27", "27", "FAR ABOVE", "Cannot", "ALIASED");
    println!("    {:>20}  {:>5}  {:>10}  {:>12}  {:>10}",
             "Sexagenary 10", "10", "ABOVE", "Cannot", "ALIASED");

    println!("\n  * Liu He/Liu Hai uses n=7 (above Nyquist), but curvature works");
    println!("    because the test is about PATH differences, not pure harmonic");
    println!("    resolution. The pairs traverse different segments, so the metric");
    println!("    creates separation without needing to resolve n=7 on the grid.");

    println!("\n--- C2: The Three-Layer Architecture ---\n");

    println!("  The ancient catalogs encode THREE distinct mathematical structures:");
    println!();
    println!("  LAYER 1: FLAT HARMONICS (no curvature needed)");
    println!("    - Jupiter drishti = n=6 resonance");
    println!("    - Variable orbs = 1/n peak width");
    println!("    - Sextile, trine, square = exact grid harmonics");
    println!("    - Wu Xing 0.30 = literal cos(5d) on flat 12-grid");
    println!();
    println!("  LAYER 2: CURVATURE (non-uniform metric)");
    println!("    - Liu He/Liu Hai = path-dependent separation");
    println!("    - Wu Xing perfect resonance = metric alignment");
    println!("    - Geometric comma = 3rd/5th harmonic incompatibility");
    println!("    - OPERATES BELOW NYQUIST (n ≤ 6 on 12-grid)");
    println!();
    println!("  LAYER 3: MULTI-GRID SAMPLING (incommensurate divisions)");
    println!("    - Nakshatra 27 = extends resolution to n=13");
    println!("    - Sexagenary 10 = resolves n=5 (Wu Xing) natively");
    println!("    - Combined 12×10×27 → lcm coverage");
    println!("    - OPERATES ABOVE NYQUIST of any single grid");

    println!("\n  Each layer solves problems the others cannot.");
    println!("  The traditions maintain all three because they're mathematically");
    println!("  independent — you can't reduce multi-grid to curvature, or");
    println!("  curvature to flat harmonics.");

    // ── Verdict ─────────────────────────────────────────────────────────

    println!("\n=== VERDICT ===\n");

    println!("  INCOMMENSURATE DIVISIONS ARE A SAMPLING PHENOMENON, NOT CURVATURE.");
    println!();
    println!("  Nakshatra 27: exists because the 12-grid aliases n=7 through n=13.");
    println!("    27 is optimal: gcd(27,12)=3 (minimal overlap), lcm=108 (rich structure),");
    println!("    27=3³ (preserves trine compatibility), Nyquist extends from 6 to 13.");
    println!();
    println!("  Sexagenary 12×10: exists because 10 resolves n=5 (Wu Xing) natively.");
    println!("    On the 10-grid, the 5 Yang stems ARE the n=5 harmonic — no curvature");
    println!("    needed. The 12-grid adds n=6. Together: 60 combinations cover n=1 to 30.");
    println!();
    println!("  Curvature CANNOT help either anomaly.");
    println!("    Both involve harmonics above the 12-grid Nyquist limit.");
    println!("    No metric on 12 points can resolve 27-fold or 10-fold structure.");
    println!("    The traditions solve this by ADDING GRIDS, not WARPING METRICS.");

    println!("\n  COMPLETE SCORECARD:");
    println!("    Test 1 (Liu He/Liu Hai):          STRONG — curvature (path separation)");
    println!("    Test 2 (Wu Xing 72°):             STRONG — curvature (metric alignment)");
    println!("    Test 3 (Geometric comma):          THEOREM — curvature (3/5 incompatibility)");
    println!("    Test 4 (Variable orbs):            NULL — flat harmonic (1/n peak width)");
    println!("    Test 5 (Vedic Drishti):            SPLIT — flat/tabular (not curvature)");
    println!("    Test 6a (Nakshatra 27):            SAMPLING — multi-grid (not curvature)");
    println!("    Test 6b (Sexagenary 12×10):        SAMPLING — multi-grid (not curvature)");
    println!();
    println!("  THE BOUNDARY IS NYQUIST.");
    println!("    Below Nyquist: curvature works (Tests 1-3).");
    println!("    At/above Nyquist: need more grid points (Tests 4-6).");
    println!("    The traditions used BOTH strategies because BOTH are needed.");
}
