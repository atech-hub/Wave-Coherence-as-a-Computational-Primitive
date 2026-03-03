/// Spherical Coherence Test: Can P_l(cos γ) extend our framework to the full sphere?
///
/// The simplest possible test of spherical harmonics as a coherence primitive.
///
/// Circle (what we have):  cos(n * Δθ)           — one angle, one harmonic number
/// Sphere (what we test):  P_l(cos γ)            — two angles, Legendre polynomials
///
/// Key questions:
///   1. Does the equator reproduce circle behavior exactly?
///   2. Can the sphere distinguish points that the circle cannot?
///   3. Do harmonic families, opposition, and fuzzy matching still work?
///
/// Zero dependencies. Pure math.

use std::f64::consts::PI;

// ── Spherical point ──

#[derive(Clone, Copy)]
struct SpherePoint {
    theta: f64,  // polar angle (0 = north pole, π = south pole, π/2 = equator)
    phi: f64,    // azimuthal angle (0..2π, longitude)
}

impl SpherePoint {
    fn new(theta: f64, phi: f64) -> Self {
        SpherePoint { theta, phi }
    }

    /// Point on the equator at azimuthal angle φ.
    fn equator(phi: f64) -> Self {
        SpherePoint { theta: PI / 2.0, phi }
    }

    /// Cartesian coordinates on the unit sphere.
    fn to_xyz(&self) -> (f64, f64, f64) {
        let x = self.theta.sin() * self.phi.cos();
        let y = self.theta.sin() * self.phi.sin();
        let z = self.theta.cos();
        (x, y, z)
    }

    /// Angular distance γ between two points (via dot product of unit vectors).
    fn angular_distance(&self, other: &SpherePoint) -> f64 {
        let (x1, y1, z1) = self.to_xyz();
        let (x2, y2, z2) = other.to_xyz();
        let dot = x1 * x2 + y1 * y2 + z1 * z2;
        // Clamp for numerical safety
        dot.clamp(-1.0, 1.0).acos()
    }

    /// cos(γ) — the argument to Legendre polynomials.
    fn cos_gamma(&self, other: &SpherePoint) -> f64 {
        let (x1, y1, z1) = self.to_xyz();
        let (x2, y2, z2) = other.to_xyz();
        let dot = x1 * x2 + y1 * y2 + z1 * z2;
        dot.clamp(-1.0, 1.0)
    }
}

// ── Legendre polynomials ──

/// P_l(x) via recurrence: P_0=1, P_1=x, P_{l+1} = ((2l+1)*x*P_l - l*P_{l-1}) / (l+1)
fn legendre(l: usize, x: f64) -> f64 {
    if l == 0 { return 1.0; }
    if l == 1 { return x; }
    let mut p_prev = 1.0; // P_0
    let mut p_curr = x;   // P_1
    for k in 1..l {
        let p_next = ((2 * k + 1) as f64 * x * p_curr - k as f64 * p_prev) / (k + 1) as f64;
        p_prev = p_curr;
        p_curr = p_next;
    }
    p_curr
}

/// Spherical coherence: P_l(cos γ) — the sphere's version of cos(n * Δθ).
fn spherical_coherence(a: &SpherePoint, b: &SpherePoint, l: usize) -> f64 {
    legendre(l, a.cos_gamma(b))
}

/// Circular coherence: cos(n * Δθ) — our existing framework.
fn circular_coherence(theta_a: f64, theta_b: f64, n: usize) -> f64 {
    (n as f64 * (theta_a - theta_b)).cos()
}

// ── Tests ──

fn test_1_equator_equivalence() -> bool {
    println!("--- Test 1: Equator Equivalence ---");
    println!("  Do P_l(cos γ) on equator and cos(n*Δθ) give identical results?\n");

    let buckets = 12;
    let mut max_diff = 0.0f64;
    let mut all_match = true;

    // Place 12 points on the equator (like our circle test)
    let points: Vec<SpherePoint> = (0..buckets)
        .map(|i| SpherePoint::equator(2.0 * PI * i as f64 / buckets as f64))
        .collect();

    println!("  {:>5}  {:>5}  {:>4}  {:>12}  {:>12}  {:>10}",
             "PtA", "PtB", "l/n", "P_l(cos γ)", "cos(nΔθ)", "Diff");
    println!("  {:>5}  {:>5}  {:>4}  {:>12}  {:>12}  {:>10}",
             "---", "---", "---", "----------", "---------", "----");

    // Test point 0 vs all others, for l=1..6
    for j in 0..buckets {
        for l in 1..=6 {
            let sph = spherical_coherence(&points[0], &points[j], l);
            let circ = circular_coherence(points[0].phi, points[j].phi, l);
            let diff = (sph - circ).abs();
            if diff > max_diff { max_diff = diff; }
            if diff > 1e-10 { all_match = false; }

            // Print selected pairs
            if j <= 6 && l <= 3 {
                println!("  {:>5}  {:>5}  {:>4}  {:>12.6}  {:>12.6}  {:>10.2e}",
                         0, j, l, sph, circ, diff);
            }
        }
    }

    println!("  ...");
    println!("  Max difference across all {} pairs × 6 harmonics: {:.2e}", buckets, max_diff);

    let pass = all_match;
    println!("  RESULT: {}\n", if pass { "PASS — sphere reduces to circle on equator" } else { "FAIL" });
    pass
}

fn test_2_exact_match() -> bool {
    println!("--- Test 2: Exact Match on Sphere ---");
    println!("  P_l(cos 0) = 1 for all l (same point = perfect coherence)\n");

    let p = SpherePoint::new(1.0, 2.0); // arbitrary point
    let mut pass = true;

    for l in 0..=10 {
        let c = spherical_coherence(&p, &p, l);
        let ok = (c - 1.0).abs() < 1e-12;
        if !ok { pass = false; }
        if l <= 6 || !ok {
            println!("  P_{}(cos 0) = {:.10}  {}", l, c, if ok { "OK" } else { "FAIL" });
        }
    }

    println!("  RESULT: {}\n", if pass { "PASS" } else { "FAIL" });
    pass
}

fn test_3_opposition() -> bool {
    println!("--- Test 3: Opposition (Antipodal Points) ---");
    println!("  P_l(cos π) = (-1)^l  (north vs south pole)\n");

    let north = SpherePoint::new(0.0, 0.0);
    let south = SpherePoint::new(PI, 0.0);
    let mut pass = true;

    let gamma = north.angular_distance(&south);
    println!("  Angular distance: {:.4} rad ({:.1}°)", gamma, gamma * 180.0 / PI);
    println!();

    for l in 0..=8 {
        let c = spherical_coherence(&north, &south, l);
        let expected = if l % 2 == 0 { 1.0 } else { -1.0 };
        let ok = (c - expected).abs() < 1e-10;
        if !ok { pass = false; }
        println!("  P_{}(cos π) = {:>8.4}  expected {:>5.1}  {}",
                 l, c, expected, if ok { "OK" } else { "FAIL" });
    }

    println!("\n  Same as circle: cos(n*π) = (-1)^n");
    println!("  RESULT: {}\n", if pass { "PASS" } else { "FAIL" });
    pass
}

fn test_4_latitude_discrimination() -> bool {
    println!("--- Test 4: Latitude Discrimination (SPHERE-ONLY) ---");
    println!("  Can the sphere distinguish points that share the same longitude");
    println!("  but differ in latitude? The circle CANNOT do this.\n");

    // Three points at the same longitude (φ=0), different latitudes
    let equator    = SpherePoint::new(PI / 2.0, 0.0);       // θ=90°
    let mid_north  = SpherePoint::new(PI / 4.0, 0.0);       // θ=45°
    let near_pole  = SpherePoint::new(PI / 8.0, 0.0);       // θ=22.5°

    // On the circle, all three would be at φ=0 → identical → coherence = 1.0
    println!("  Circle view: all at φ=0 → cos(n*0) = 1.0 for all n (indistinguishable)");
    println!();

    println!("  Sphere view:");
    let pairs: &[(&str, &SpherePoint, &str, &SpherePoint)] = &[
        ("Equator",    &equator,   "Mid-North",  &mid_north),
        ("Equator",    &equator,   "Near-Pole",  &near_pole),
        ("Mid-North",  &mid_north, "Near-Pole",  &near_pole),
    ];

    let mut any_discrimination = false;

    for (name_a, pa, name_b, pb) in pairs {
        let gamma = pa.angular_distance(pb);
        println!("  {} vs {} (γ = {:.1}°):", name_a, name_b, gamma * 180.0 / PI);
        for l in 1..=6 {
            let c = spherical_coherence(pa, pb, l);
            if (c - 1.0).abs() > 0.01 { any_discrimination = true; }
            println!("    P_{}(cos γ) = {:>8.4}", l, c);
        }
        println!();
    }

    let pass = any_discrimination;
    println!("  RESULT: {}\n", if pass {
        "PASS — sphere discriminates latitude (circle cannot)"
    } else {
        "FAIL — no discrimination"
    });
    pass
}

fn test_5_harmonic_families() -> bool {
    println!("--- Test 5: Harmonic Families on Sphere ---");
    println!("  Points at angular distance γ = π/n should give P_n(cos(π/n)) ≈ specific values.");
    println!("  Do harmonic relationships emerge naturally?\n");

    let origin = SpherePoint::new(PI / 2.0, 0.0);

    // Points at various angular distances along the equator
    println!("  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}",
             "Δφ (°)", "γ (°)", "P_1", "P_2", "P_3", "P_4", "P_6");
    println!("  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}",
             "------", "-----", "---", "---", "---", "---", "---");

    let angles_deg = [0.0, 30.0, 45.0, 60.0, 90.0, 120.0, 180.0];
    let mut pass = true;

    for &deg in &angles_deg {
        let rad = deg * PI / 180.0;
        let target = SpherePoint::equator(rad);
        let gamma = origin.angular_distance(&target);

        let p1 = spherical_coherence(&origin, &target, 1);
        let p2 = spherical_coherence(&origin, &target, 2);
        let p3 = spherical_coherence(&origin, &target, 3);
        let p4 = spherical_coherence(&origin, &target, 4);
        let p6 = spherical_coherence(&origin, &target, 6);

        println!("  {:>7.1}°  {:>7.1}°  {:>8.4}  {:>8.4}  {:>8.4}  {:>8.4}  {:>8.4}",
                 deg, gamma * 180.0 / PI, p1, p2, p3, p4, p6);
    }

    // Verify: at 60° separation, P_3 should detect the trine (cos(3*60°) = cos(180°) = -1 on circle)
    // On sphere at equator, this should match
    let trine_point = SpherePoint::equator(PI / 3.0); // 60°
    let p3_trine = spherical_coherence(&origin, &trine_point, 3);
    let c3_trine = circular_coherence(0.0, PI / 3.0, 3);
    let trine_match = (p3_trine - c3_trine).abs() < 1e-10;
    println!("\n  Trine check (60°): P_3 = {:.6}, cos(3*60°) = {:.6}, match: {}",
             p3_trine, c3_trine, trine_match);
    if !trine_match { pass = false; }

    // Verify: at 90° separation, P_2 detects the square
    let square_point = SpherePoint::equator(PI / 2.0); // 90°
    let p2_square = spherical_coherence(&origin, &square_point, 2);
    let c2_square = circular_coherence(0.0, PI / 2.0, 2);
    let square_match = (p2_square - c2_square).abs() < 1e-10;
    println!("  Square check (90°): P_2 = {:.6}, cos(2*90°) = {:.6}, match: {}",
             p2_square, c2_square, square_match);
    if !square_match { pass = false; }

    println!("  RESULT: {}\n", if pass { "PASS" } else { "FAIL" });
    pass
}

fn test_6_sphere_exclusive_relationships() -> bool {
    println!("--- Test 6: Sphere-Exclusive Relationships ---");
    println!("  Relationships that ONLY exist on the sphere, impossible on the circle.\n");

    // Two pairs with identical longitude separation but different latitudes
    // Circle sees them as identical. Sphere sees them as different.

    // Pair A: both on equator, 60° apart in φ
    let a1 = SpherePoint::equator(0.0);
    let a2 = SpherePoint::equator(PI / 3.0);

    // Pair B: both at 30° latitude, 60° apart in φ
    let b1 = SpherePoint::new(PI / 6.0, 0.0);        // near north pole
    let b2 = SpherePoint::new(PI / 6.0, PI / 3.0);    // same latitude, 60° east

    let gamma_a = a1.angular_distance(&a2);
    let gamma_b = b1.angular_distance(&b2);

    println!("  Pair A: equator, Δφ=60° → γ = {:.2}°", gamma_a * 180.0 / PI);
    println!("  Pair B: lat=30°N, Δφ=60° → γ = {:.2}°", gamma_b * 180.0 / PI);
    println!("  Circle: both pairs have Δφ=60° → cos(n*60°) identical");
    println!("  Sphere: different γ → different P_l values");
    println!();

    let mut any_diff = false;

    println!("  {:>4}  {:>12}  {:>12}  {:>12}  {:>12}",
             "l", "Pair A (eq)", "Pair B (30N)", "Circ Δφ=60°", "Sphere diff");
    println!("  {:>4}  {:>12}  {:>12}  {:>12}  {:>12}",
             "--", "----------", "-----------", "----------", "-----------");

    for l in 1..=8 {
        let ca = spherical_coherence(&a1, &a2, l);
        let cb = spherical_coherence(&b1, &b2, l);
        let cc = circular_coherence(0.0, PI / 3.0, l);
        let diff = (ca - cb).abs();
        if diff > 0.01 { any_diff = true; }
        println!("  {:>4}  {:>12.6}  {:>12.6}  {:>12.6}  {:>12.6}",
                 l, ca, cb, cc, diff);
    }

    println!();
    println!("  Pair A matches circle (both on equator).");
    println!("  Pair B diverges — latitude compresses angular distance.");
    println!("  This is NEW encoding capacity the circle doesn't have.");

    let pass = any_diff;
    println!("  RESULT: {}\n", if pass {
        "PASS — sphere encodes relationships circle cannot"
    } else {
        "FAIL — no sphere-exclusive discrimination"
    });
    pass
}

fn test_7_mode_count() -> bool {
    println!("--- Test 7: Mode Count — Why the Sphere Has More Capacity ---");
    println!("  Circle: harmonic n gives 2 modes (cos nθ, sin nθ)");
    println!("  Sphere: degree l gives (2l+1) modes (Y_l^m for m = -l..+l)\n");

    let mut circle_total = 0;
    let mut sphere_total = 0;

    println!("  {:>6}  {:>14}  {:>14}  {:>10}",
             "l / n", "Circle modes", "Sphere modes", "Ratio");
    println!("  {:>6}  {:>14}  {:>14}  {:>10}",
             "-----", "------------", "------------", "-----");

    for l in 0..=10 {
        let cm = if l == 0 { 1 } else { 2 };
        let sm = 2 * l + 1;
        circle_total += cm;
        sphere_total += sm;
        println!("  {:>6}  {:>14}  {:>14}  {:>9.1}x",
                 l, cm, sm, sm as f64 / cm as f64);
    }

    println!("  {:>6}  {:>14}  {:>14}  {:>9.1}x",
             "Total", circle_total, sphere_total, sphere_total as f64 / circle_total as f64);

    println!("\n  Through l=10: sphere has {:.1}x more modes than circle.",
             sphere_total as f64 / circle_total as f64);
    println!("  This is the extra encoding capacity the sphere provides.");

    let pass = sphere_total > circle_total;
    println!("  RESULT: {}\n", if pass { "PASS" } else { "FAIL" });
    pass
}

fn test_8_legendre_properties() -> bool {
    println!("--- Test 8: Legendre Polynomial Properties (Sanity) ---");
    println!("  Verify mathematical correctness of P_l implementation.\n");

    let mut pass = true;

    // Property 1: P_l(1) = 1 for all l
    print!("  P_l(1) = 1: ");
    for l in 0..=15 {
        let v = legendre(l, 1.0);
        if (v - 1.0).abs() > 1e-12 {
            println!("FAIL at l={}: {}", l, v);
            pass = false;
        }
    }
    if pass { println!("OK (l=0..15)"); }

    // Property 2: P_l(-1) = (-1)^l
    print!("  P_l(-1) = (-1)^l: ");
    let mut ok = true;
    for l in 0..=15 {
        let v = legendre(l, -1.0);
        let expected = if l % 2 == 0 { 1.0 } else { -1.0 };
        if (v - expected).abs() > 1e-10 {
            println!("FAIL at l={}: {} vs {}", l, v, expected);
            pass = false;
            ok = false;
        }
    }
    if ok { println!("OK (l=0..15)"); }

    // Property 3: P_l(0) for even l
    print!("  P_l(0) known values: ");
    let known: &[(usize, f64)] = &[
        (0, 1.0), (1, 0.0), (2, -0.5), (3, 0.0), (4, 3.0 / 8.0),
        (5, 0.0), (6, -5.0 / 16.0),
    ];
    let mut ok2 = true;
    for &(l, expected) in known {
        let v = legendre(l, 0.0);
        if (v - expected).abs() > 1e-10 {
            println!("FAIL at l={}: {} vs {}", l, v, expected);
            pass = false;
            ok2 = false;
        }
    }
    if ok2 { println!("OK (l=0..6)"); }

    // Property 4: |P_l(x)| <= 1 for |x| <= 1
    print!("  |P_l(x)| <= 1 for |x| <= 1: ");
    let mut ok3 = true;
    for l in 0..=20 {
        for i in 0..=100 {
            let x = -1.0 + 2.0 * i as f64 / 100.0;
            let v = legendre(l, x);
            if v.abs() > 1.0 + 1e-10 {
                println!("FAIL at l={}, x={}: P={}", l, x, v);
                pass = false;
                ok3 = false;
            }
        }
    }
    if ok3 { println!("OK (l=0..20, 101 x-values)"); }

    println!("  RESULT: {}\n", if pass { "PASS" } else { "FAIL" });
    pass
}

// ── Main ──

fn main() {
    println!("=== Spherical Coherence Test ===");
    println!("  Circle: cos(n * Δθ)    — our current framework");
    println!("  Sphere: P_l(cos γ)     — natural extension");
    println!();

    let mut passed = 0;
    let mut failed = 0;

    let tests: Vec<(&str, fn() -> bool)> = vec![
        ("Equator Equivalence", test_1_equator_equivalence),
        ("Exact Match", test_2_exact_match),
        ("Opposition", test_3_opposition),
        ("Latitude Discrimination", test_4_latitude_discrimination),
        ("Harmonic Families", test_5_harmonic_families),
        ("Sphere-Exclusive", test_6_sphere_exclusive_relationships),
        ("Mode Count", test_7_mode_count),
        ("Legendre Properties", test_8_legendre_properties),
    ];

    for (_name, test_fn) in &tests {
        if test_fn() { passed += 1; } else { failed += 1; }
    }

    println!("=== RESULTS: {} passed, {} failed out of {} ===\n",
             passed, failed, passed + failed);

    if failed == 0 {
        println!("ALL TESTS PASSED.");
        println!();
        println!("=== IMPLICATIONS ===");
        println!();
        println!("  1. BACKWARD COMPATIBLE: P_l on equator = cos(n*Δθ) exactly.");
        println!("     Our entire existing framework is a special case.");
        println!();
        println!("  2. STRICTLY MORE EXPRESSIVE: sphere discriminates points");
        println!("     that the circle maps to identical positions.");
        println!();
        println!("  3. MODE SCALING: degree l gives (2l+1) modes vs circle's 2.");
        println!("     Through l=10: ~5.7x more encoding capacity.");
        println!();
        println!("  4. PURE MATH: Legendre recurrence is 3 multiplies + 1 add per step.");
        println!("     No GPU. No dependencies. Same computational philosophy.");
        println!();
        println!("  5. OPEN QUESTION: Does this extra capacity help close the ~5% Kerr gap?");
        println!("     The flat coupling profile might gain structure on the sphere.");
    } else {
        println!("SOME TESTS FAILED — review output above.");
    }
}
