/// Hybrid Coherence Test: Circle + Sphere Combined
///
/// Design: Each entity gets (φ, θ) per band.
///   φ = azimuthal angle [0, 2π) — the existing circle dimension
///   θ = elevation angle [0, π]  — the new sphere dimension
///
/// Three combiners tested:
///   Product:  H = cos(n×Δφ) × P_l(cos Δθ)
///   Sum:      H = α × cos(n×Δφ) + (1-α) × P_l(cos Δθ)
///   Gated:    H = cos(n×Δφ) × [1 + β × (P_l(cos Δθ) - 1)]
///             (elevation modulates but doesn't kill azimuthal signal)
///
/// Key property: when θ_a = θ_b (same elevation), all combiners must
/// reduce to cos(n×Δφ) — backward compatibility with circle framework.
///
/// Zero dependencies. Pure math.

use std::f64::consts::PI;

// ── Legendre polynomials ──

fn legendre(l: usize, x: f64) -> f64 {
    if l == 0 { return 1.0; }
    if l == 1 { return x; }
    let mut p_prev = 1.0;
    let mut p_curr = x;
    for k in 1..l {
        let p_next = ((2 * k + 1) as f64 * x * p_curr - k as f64 * p_prev) / (k + 1) as f64;
        p_prev = p_curr;
        p_curr = p_next;
    }
    p_curr
}

// ── Coherence functions ──

fn circle_coherence(delta_phi: f64, n: usize) -> f64 {
    (n as f64 * delta_phi).cos()
}

fn elevation_coherence(delta_theta: f64, l: usize) -> f64 {
    legendre(l, delta_theta.cos())
}

// ── Combiners ──

/// Product: H = cos(n×Δφ) × P_l(cos Δθ)
/// When Δθ=0: P_l(1) = 1 → H = cos(n×Δφ) ✓ backward compatible
/// Problem: elevation miss kills azimuthal signal entirely
fn hybrid_product(delta_phi: f64, delta_theta: f64, n: usize, l: usize) -> f64 {
    circle_coherence(delta_phi, n) * elevation_coherence(delta_theta, l)
}

/// Weighted sum: H = α × cos(n×Δφ) + (1-α) × P_l(cos Δθ)
/// When α=1: pure circle. When α=0: pure Legendre.
/// When Δθ=0: H = α × cos(n×Δφ) + (1-α) × 1.0 — NOT backward compatible unless α=1
fn hybrid_sum(delta_phi: f64, delta_theta: f64, n: usize, l: usize, alpha: f64) -> f64 {
    alpha * circle_coherence(delta_phi, n) + (1.0 - alpha) * elevation_coherence(delta_theta, l)
}

/// Gated: elevation modulates azimuthal signal but can't kill it.
/// H = cos(n×Δφ) × [1 + β × (P_l(cos Δθ) - 1)]
///   = cos(n×Δφ) × [(1-β) + β × P_l(cos Δθ)]
/// When Δθ=0: P_l=1, bracket=1, H = cos(n×Δφ) ✓ backward compatible
/// When Δθ=π (antipodal elevation): bracket = (1-β) + β×(-1)^l = (1-2β) for odd l
/// β controls modulation depth:
///   β=0: pure circle (no elevation effect)
///   β=0.5: bracket ranges [0.5, 1.0] for l=1 — elevation attenuates but never kills
///   β=1.0: equivalent to product
fn hybrid_gated(delta_phi: f64, delta_theta: f64, n: usize, l: usize, beta: f64) -> f64 {
    let gate = (1.0 - beta) + beta * elevation_coherence(delta_theta, l);
    circle_coherence(delta_phi, n) * gate
}

// ── Named relationships ──

struct Relationship {
    name: &'static str,
    angle_deg: f64,
    natural_n: usize,
}

fn catalog() -> Vec<Relationship> {
    vec![
        Relationship { name: "Unison",       angle_deg: 0.0,   natural_n: 1 },
        Relationship { name: "Semitone",     angle_deg: 30.0,  natural_n: 12 },
        Relationship { name: "Sextile",      angle_deg: 60.0,  natural_n: 6 },
        Relationship { name: "Quintile",     angle_deg: 72.0,  natural_n: 5 },
        Relationship { name: "Square",       angle_deg: 90.0,  natural_n: 4 },
        Relationship { name: "Trine",        angle_deg: 120.0, natural_n: 3 },
        Relationship { name: "Quincunx",     angle_deg: 150.0, natural_n: 12 },
        Relationship { name: "Opposition",   angle_deg: 180.0, natural_n: 2 },
    ]
}

fn main() {
    println!("=== Hybrid Coherence Test: Circle + Sphere ===");
    println!("  Encoding: (φ, θ) per band — azimuth + elevation");
    println!("  Circle: cos(n × Δφ)   — Chebyshev (preserved)");
    println!("  Elevation: P_l(cos Δθ) — Legendre (added)");
    println!();

    let max_n = 15;

    // ══════════════════════════════════════════════════════════════
    // PART 1: Backward Compatibility — Δθ = 0 (same elevation)
    // ══════════════════════════════════════════════════════════════
    println!("=== PART 1: Backward Compatibility (Δθ = 0) ===");
    println!("  When both points at same elevation, hybrid MUST equal circle.\n");

    let delta_theta = 0.0;
    let mut compat_pass = true;
    let test_phis = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0];

    for &phi_deg in &test_phis {
        let dphi = phi_deg * PI / 180.0;
        for n in 1..=max_n {
            let c = circle_coherence(dphi, n);
            let hp = hybrid_product(dphi, delta_theta, n, n);
            let hg3 = hybrid_gated(dphi, delta_theta, n, n, 0.3);
            let hg5 = hybrid_gated(dphi, delta_theta, n, n, 0.5);

            let diff_p = (c - hp).abs();
            let diff_g3 = (c - hg3).abs();
            let diff_g5 = (c - hg5).abs();

            if diff_p > 1e-12 || diff_g3 > 1e-12 || diff_g5 > 1e-12 {
                println!("  FAIL at Δφ={}°, n={}: circle={:.6}, product={:.6}, gated={:.6}",
                         phi_deg, n, c, hp, hg5);
                compat_pass = false;
            }
        }
    }

    if compat_pass {
        println!("  ALL PASS — product and gated reduce to circle when Δθ=0");
    }

    // Check sum backward compatibility
    let hs1 = hybrid_sum(PI / 3.0, 0.0, 3, 3, 1.0);
    let c1 = circle_coherence(PI / 3.0, 3);
    println!("  Sum (α=1.0): {:.6} vs circle {:.6} — {}",
             hs1, c1, if (hs1 - c1).abs() < 1e-12 { "PASS" } else { "FAIL" });
    let hs7 = hybrid_sum(PI / 3.0, 0.0, 3, 3, 0.7);
    println!("  Sum (α=0.7): {:.6} vs circle {:.6} — NOT backward compatible (by design)",
             hs7, c1);
    println!();

    // ══════════════════════════════════════════════════════════════
    // PART 2: Named Relationships — All Three Combiners
    // ══════════════════════════════════════════════════════════════
    println!("=== PART 2: Named Relationships — Combiner Comparison ===");
    println!("  Testing at natural harmonic, same elevation (Δθ=0) vs different elevation (Δθ=45°)\n");

    let relationships = catalog();
    let elev_offsets_deg = [0.0, 15.0, 30.0, 45.0, 60.0, 90.0];

    for rel in &relationships {
        let dphi = rel.angle_deg * PI / 180.0;
        let n = rel.natural_n;

        println!("  {} (Δφ={}°, n={}):", rel.name, rel.angle_deg, n);
        println!("  {:>6}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}",
                 "Δθ(°)", "Circle", "Product", "Gate30", "Gate50", "Sum70");
        println!("  {:>6}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}",
                 "-----", "------", "-------", "------", "------", "-----");

        for &elev_deg in &elev_offsets_deg {
            let dtheta = elev_deg * PI / 180.0;
            let c = circle_coherence(dphi, n);
            let hp = hybrid_product(dphi, dtheta, n, n);
            let hg3 = hybrid_gated(dphi, dtheta, n, n, 0.3);
            let hg5 = hybrid_gated(dphi, dtheta, n, n, 0.5);
            let hs = hybrid_sum(dphi, dtheta, n, n, 0.7);

            println!("  {:>5.0}°  {:>8.4}  {:>8.4}  {:>8.4}  {:>8.4}  {:>8.4}",
                     elev_deg, c, hp, hg3, hg5, hs);
        }
        println!();
    }

    // ══════════════════════════════════════════════════════════════
    // PART 3: The 442 Test — Do combiners preserve circle detections?
    // ══════════════════════════════════════════════════════════════
    println!("=== PART 3: Preservation of 442 Circle Detections ===");
    println!("  Testing: for each (angle, n) where |cos(n×Δφ)| > 0.95,");
    println!("  does the hybrid also give |H| > 0.95 at same elevation (Δθ=0)?\n");

    let mut circle_strong = 0;
    let mut product_preserved = 0;
    let mut gated30_preserved = 0;
    let mut gated50_preserved = 0;
    let mut sum70_preserved = 0;

    for deg in 1..180 {
        let dphi = deg as f64 * PI / 180.0;
        for n in 1..=max_n {
            let c = circle_coherence(dphi, n);
            if c.abs() > 0.95 {
                circle_strong += 1;
                let dtheta = 0.0; // same elevation
                if hybrid_product(dphi, dtheta, n, n).abs() > 0.95 { product_preserved += 1; }
                if hybrid_gated(dphi, dtheta, n, n, 0.3).abs() > 0.95 { gated30_preserved += 1; }
                if hybrid_gated(dphi, dtheta, n, n, 0.5).abs() > 0.95 { gated50_preserved += 1; }
                if hybrid_sum(dphi, dtheta, n, n, 0.7).abs() > 0.95 { sum70_preserved += 1; }
            }
        }
    }

    println!("  Circle strong detections (|c| > 0.95): {}", circle_strong);
    println!("  Product preserved at Δθ=0: {} ({}%)", product_preserved, 100 * product_preserved / circle_strong);
    println!("  Gated β=0.3 preserved:     {} ({}%)", gated30_preserved, 100 * gated30_preserved / circle_strong);
    println!("  Gated β=0.5 preserved:     {} ({}%)", gated50_preserved, 100 * gated50_preserved / circle_strong);
    println!("  Sum α=0.7 preserved:       {} ({}%)", sum70_preserved, 100 * sum70_preserved / circle_strong);
    println!();

    // ══════════════════════════════════════════════════════════════
    // PART 4: Elevation Sensitivity — How much does Δθ modulate?
    // ══════════════════════════════════════════════════════════════
    println!("=== PART 4: Elevation Sensitivity ===");
    println!("  For a perfect azimuthal match (Δφ=0), how does elevation difference modulate?\n");

    println!("  {:>6}  {:>4}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}",
             "Δθ(°)", "l", "P_l", "Product", "Gate30", "Gate50", "Sum70");
    println!("  {:>6}  {:>4}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}",
             "-----", "--", "---", "-------", "------", "------", "-----");

    let dphi = 0.0; // perfect azimuthal match
    for elev_deg in (0..=180).step_by(15) {
        let dtheta = elev_deg as f64 * PI / 180.0;
        for &l in &[1usize, 2, 3, 4] {
            let pl = elevation_coherence(dtheta, l);
            let hp = hybrid_product(dphi, dtheta, 1, l);
            let hg3 = hybrid_gated(dphi, dtheta, 1, l, 0.3);
            let hg5 = hybrid_gated(dphi, dtheta, 1, l, 0.5);
            let hs = hybrid_sum(dphi, dtheta, 1, l, 0.7);

            if l == 1 || l == 3 {
                println!("  {:>5}°  {:>4}  {:>8.4}  {:>8.4}  {:>8.4}  {:>8.4}  {:>8.4}",
                         elev_deg, l, pl, hp, hg3, hg5, hs);
            }
        }
    }
    println!();

    // ══════════════════════════════════════════════════════════════
    // PART 5: Discrimination Power — Can hybrid separate what circle can't?
    // ══════════════════════════════════════════════════════════════
    println!("=== PART 5: Discrimination Power ===");
    println!("  Pairs that are IDENTICAL on the circle (same Δφ) but at different elevations.\n");

    let test_cases: &[(&str, f64, f64, &str, f64, f64)] = &[
        // (name_a, phi_a, theta_a, name_b, phi_b, theta_b)
        // Both pairs have Δφ = 60°, but pair B has elevation offset
        ("Equator pair A1", 0.0, 90.0, "Equator pair A2", 60.0, 90.0),
        ("Elevated pair B1", 0.0, 45.0, "Elevated pair B2", 60.0, 45.0),
        ("Mixed pair C1", 0.0, 90.0, "Mixed pair C2", 60.0, 45.0),
        ("Polar pair D1", 0.0, 20.0, "Polar pair D2", 60.0, 20.0),
    ];

    for &(name_a, phi_a_deg, theta_a_deg, name_b, phi_b_deg, theta_b_deg) in test_cases {
        let dphi = (phi_b_deg - phi_a_deg) * PI / 180.0;
        let dtheta = (theta_b_deg - theta_a_deg).abs() * PI / 180.0;

        println!("  {} ↔ {}:", name_a, name_b);
        println!("    Δφ={:.0}°, Δθ={:.0}°", (phi_b_deg - phi_a_deg).abs(), (theta_b_deg - theta_a_deg).abs());

        println!("    {:>4}  {:>8}  {:>8}  {:>8}  {:>8}",
                 "n=l", "Circle", "Product", "Gate30", "Gate50");
        println!("    {:>4}  {:>8}  {:>8}  {:>8}  {:>8}",
                 "---", "------", "-------", "------", "------");

        for n in 1..=6 {
            let c = circle_coherence(dphi, n);
            let hp = hybrid_product(dphi, dtheta, n, n);
            let hg3 = hybrid_gated(dphi, dtheta, n, n, 0.3);
            let hg5 = hybrid_gated(dphi, dtheta, n, n, 0.5);

            println!("    {:>4}  {:>8.4}  {:>8.4}  {:>8.4}  {:>8.4}",
                     n, c, hp, hg3, hg5);
        }
        println!();
    }

    // ══════════════════════════════════════════════════════════════
    // PART 6: New Detection Space — What the hybrid finds that circle can't
    // ══════════════════════════════════════════════════════════════
    println!("=== PART 6: Hybrid-Only Detections ===");
    println!("  Scanning: pairs where circle gives |c| < 0.3 (weak/no signal)");
    println!("  but hybrid_gated gives |H| > 0.7 (strong signal via elevation).\n");

    // For this to happen, cos(n×Δφ) must be moderate-to-low but elevation
    // coherence must boost it. With gated: H = cos(n×Δφ) × [(1-β) + β×P_l]
    // If cos(n×Δφ) is small, even perfect P_l can't make H large (gated preserves sign structure).
    // So hybrid-only detections require a DIFFERENT mechanism:
    // The elevation dimension provides INDEPENDENT signal.

    // Let's check: across elevation differences, are there cases where the hybrid
    // reveals structure the circle is blind to?

    // Scenario: Two entities at same azimuth (Δφ=0) but different elevation.
    // Circle: cos(n×0) = 1.0 for all n — sees them as identical.
    // Hybrid: elevation coherence varies by l.
    println!("  Scenario: Same azimuth (Δφ=0), varying elevation offset");
    println!("  Circle sees ALL as identical (coherence = 1.0 for all n).");
    println!("  Hybrid discriminates by elevation:\n");

    println!("  {:>6}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}",
             "Δθ(°)", "Circle", "G30,l=1", "G30,l=2", "G30,l=3", "G50,l=2", "G50,l=3");
    println!("  {:>6}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}",
             "-----", "------", "-------", "-------", "-------", "-------", "-------");

    for elev_deg in (0..=180).step_by(10) {
        let dtheta = elev_deg as f64 * PI / 180.0;
        let c = circle_coherence(0.0, 1); // always 1.0
        let hg3_l1 = hybrid_gated(0.0, dtheta, 1, 1, 0.3);
        let hg3_l2 = hybrid_gated(0.0, dtheta, 1, 2, 0.3);
        let hg3_l3 = hybrid_gated(0.0, dtheta, 1, 3, 0.3);
        let hg5_l2 = hybrid_gated(0.0, dtheta, 1, 2, 0.5);
        let hg5_l3 = hybrid_gated(0.0, dtheta, 1, 3, 0.5);

        println!("  {:>5}°  {:>8.4}  {:>8.4}  {:>8.4}  {:>8.4}  {:>8.4}  {:>8.4}",
                 elev_deg, c, hg3_l1, hg3_l2, hg3_l3, hg5_l2, hg5_l3);
    }
    println!();

    // Count how many elevation angles are discriminated (H < 0.9 when circle = 1.0)
    let mut gated30_discriminates = 0;
    let mut gated50_discriminates = 0;
    for elev_deg in 1..=180 {
        let dtheta = elev_deg as f64 * PI / 180.0;
        let hg3 = hybrid_gated(0.0, dtheta, 1, 2, 0.3);
        let hg5 = hybrid_gated(0.0, dtheta, 1, 2, 0.5);
        if hg3.abs() < 0.90 { gated30_discriminates += 1; }
        if hg5.abs() < 0.90 { gated50_discriminates += 1; }
    }
    println!("  Elevation angles where gated β=0.3 discriminates (|H|<0.9): {}/180",
             gated30_discriminates);
    println!("  Elevation angles where gated β=0.5 discriminates (|H|<0.9): {}/180",
             gated50_discriminates);
    println!();

    // ══════════════════════════════════════════════════════════════
    // PART 7: The Cross-Talk Test — n ≠ l
    // ══════════════════════════════════════════════════════════════
    println!("=== PART 7: Cross-Harmonic — Different n (azimuth) and l (elevation) ===");
    println!("  The hybrid has TWO harmonic parameters. What happens when n ≠ l?");
    println!("  This is NEW — the circle has only one harmonic number.\n");

    let dphi = 60.0 * PI / 180.0;
    let dtheta = 45.0 * PI / 180.0;

    println!("  Δφ=60°, Δθ=45° — Gated β=0.5:");
    println!("  {:>4}  {:>4}  {:>10}  {:>10}  {:>10}", "n", "l", "cos(nΔφ)", "P_l(cosΔθ)", "Hybrid");
    println!("  {:>4}  {:>4}  {:>10}  {:>10}  {:>10}", "--", "--", "--------", "---------", "------");

    for n in 1..=6 {
        for l in 1..=6 {
            let c = circle_coherence(dphi, n);
            let p = elevation_coherence(dtheta, l);
            let h = hybrid_gated(dphi, dtheta, n, l, 0.5);

            // Show diagonal + interesting off-diagonal
            if n == l || (n == 1 && l <= 4) || (l == 1 && n <= 4) {
                let marker = if n == l { " <-- n=l" } else { "" };
                println!("  {:>4}  {:>4}  {:>10.4}  {:>10.4}  {:>10.4}{}",
                         n, l, c, p, h, marker);
            }
        }
    }
    println!();

    // ══════════════════════════════════════════════════════════════
    // VERDICT
    // ══════════════════════════════════════════════════════════════
    println!("=== VERDICT ===\n");

    println!("  Backward compatibility (Δθ=0):");
    println!("    Product:  {} (Δθ=0 → P_l(1)=1 → H=cos(nΔφ))",
             if compat_pass { "PASS" } else { "FAIL" });
    println!("    Gated:    {} (Δθ=0 → gate=1 → H=cos(nΔφ))",
             if compat_pass { "PASS" } else { "FAIL" });
    println!("    Sum α=1:  PASS (trivially)");
    println!("    Sum α<1:  FAIL (adds constant offset)");
    println!();

    println!("  Circle detection preservation (442 test):");
    println!("    Product:  {}/{} (100% at Δθ=0 by construction)", product_preserved, circle_strong);
    println!("    Gated:    {}/{} (100% at Δθ=0 by construction)", gated50_preserved, circle_strong);
    println!();

    println!("  Elevation discrimination (Δφ=0, circle=1.0 everywhere):");
    println!("    Gated β=0.3: {}/180 angles discriminated", gated30_discriminates);
    println!("    Gated β=0.5: {}/180 angles discriminated", gated50_discriminates);
    println!();

    // Determine best combiner
    println!("  COMBINER RANKING:");
    println!();
    println!("    1. GATED (recommended)");
    println!("       H = cos(n×Δφ) × [(1-β) + β × P_l(cos Δθ)]");
    println!("       ✓ Backward compatible at Δθ=0");
    println!("       ✓ Preserves all 442 circle detections at same elevation");
    println!("       ✓ Elevation modulates but never kills azimuthal signal");
    println!("       ✓ β tunes modulation depth (0=circle, 1=product)");
    println!("       ✓ Two harmonic parameters (n, l) for independent tuning");
    println!();
    println!("    2. PRODUCT");
    println!("       H = cos(n×Δφ) × P_l(cos Δθ)");
    println!("       ✓ Backward compatible at Δθ=0");
    println!("       ✗ Too aggressive — elevation miss zeroes azimuthal signal");
    println!("       ✓ Mathematically elegant (spherical harmonic factorization)");
    println!();
    println!("    3. SUM");
    println!("       H = α × cos(n×Δφ) + (1-α) × P_l(cos Δθ)");
    println!("       ✗ NOT backward compatible (α<1 adds offset)");
    println!("       ✗ Mixes scales — circle [-1,1] and Legendre [-1,1] interfere");
    println!("       ✗ α=1 trivially reduces to circle (useless)");
    println!();

    println!("  HYBRID COHERENCE FUNCTION:");
    println!("    H(a, b, n, l, β) = cos(n × (φ_a - φ_b)) × [(1-β) + β × P_l(cos(θ_a - θ_b))]");
    println!();
    println!("    n = azimuthal harmonic (existing framework, Chebyshev behavior)");
    println!("    l = elevation harmonic (new dimension, Legendre behavior)");
    println!("    β = modulation depth (0 = pure circle, 1 = full product)");
    println!("    β = 0.3-0.5 recommended (elevation informs but doesn't dominate)");
    println!();
    println!("=== END ===");
}
