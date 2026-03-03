/// Spherical Relationship Map: Full Comparison of Circle vs Sphere Coherence
///
/// Runs every known relationship type through both systems:
///   Circle: cos(n × Δθ)   — Chebyshev T_n(cos Δθ)
///   Sphere: P_l(cos γ)    — Legendre polynomials
///
/// For each relationship × each harmonic degree, computes both values and the
/// divergence. The divergence map shows exactly WHERE the sphere offers something
/// the circle doesn't.
///
/// Part 1: Named relationships (astrological aspects, musical intervals)
/// Part 2: Dense sweep at 1° resolution, all harmonics 1..15
/// Part 3: Divergence analysis — which (angle, harmonic) pairs differ most?
/// Part 4: Sphere-only test — latitude variations the circle can't see
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

/// Circle: cos(n × Δθ) = T_n(cos Δθ)
fn circle_coherence(delta_theta: f64, n: usize) -> f64 {
    (n as f64 * delta_theta).cos()
}

/// Sphere: P_l(cos γ) — on equator, cos γ = cos Δφ, so γ = Δφ
fn sphere_coherence(delta_theta: f64, l: usize) -> f64 {
    legendre(l, delta_theta.cos())
}

// ── Sphere point for latitude tests ──

struct SpherePoint { theta: f64, phi: f64 }

impl SpherePoint {
    fn cos_gamma(&self, other: &SpherePoint) -> f64 {
        let (x1, y1, z1) = (
            self.theta.sin() * self.phi.cos(),
            self.theta.sin() * self.phi.sin(),
            self.theta.cos(),
        );
        let (x2, y2, z2) = (
            other.theta.sin() * other.phi.cos(),
            other.theta.sin() * other.phi.sin(),
            other.theta.cos(),
        );
        (x1 * x2 + y1 * y2 + z1 * z2).clamp(-1.0, 1.0)
    }
}

// ── Named relationships ──

struct Relationship {
    name: &'static str,
    angle_deg: f64,
    category: &'static str,
    // What circle detects at its "natural" harmonic
    natural_n: usize,
}

fn catalog() -> Vec<Relationship> {
    vec![
        Relationship { name: "Unison",      angle_deg: 0.0,   category: "exact",    natural_n: 1 },
        Relationship { name: "Semitone",    angle_deg: 30.0,  category: "music",    natural_n: 12 },
        Relationship { name: "Sextile",     angle_deg: 60.0,  category: "aspect",   natural_n: 6 },
        Relationship { name: "Quintile",    angle_deg: 72.0,  category: "aspect",   natural_n: 5 },
        Relationship { name: "Square",      angle_deg: 90.0,  category: "aspect",   natural_n: 4 },
        Relationship { name: "Trine",       angle_deg: 120.0, category: "aspect",   natural_n: 3 },
        Relationship { name: "Quincunx",    angle_deg: 150.0, category: "aspect",   natural_n: 12 },
        Relationship { name: "Opposition",  angle_deg: 180.0, category: "aspect",   natural_n: 2 },
        // Musical intervals beyond semitone
        Relationship { name: "Whole tone",  angle_deg: 60.0,  category: "music",    natural_n: 6 },
        Relationship { name: "Minor 3rd",   angle_deg: 90.0,  category: "music",    natural_n: 4 },
        Relationship { name: "Major 3rd",   angle_deg: 120.0, category: "music",    natural_n: 3 },
        Relationship { name: "Perfect 4th", angle_deg: 150.0, category: "music",    natural_n: 12 },
        Relationship { name: "Tritone",     angle_deg: 180.0, category: "music",    natural_n: 2 },
        // Fuzzy / intermediate
        Relationship { name: "Near-match",  angle_deg: 5.0,   category: "fuzzy",    natural_n: 1 },
        Relationship { name: "Slight off",  angle_deg: 15.0,  category: "fuzzy",    natural_n: 1 },
        Relationship { name: "Quarter",     angle_deg: 45.0,  category: "other",    natural_n: 8 },
        Relationship { name: "Arbitrary 1", angle_deg: 37.0,  category: "other",    natural_n: 1 },
        Relationship { name: "Arbitrary 2", angle_deg: 137.0, category: "other",    natural_n: 1 },
    ]
}

// ── Main ──

fn main() {
    println!("=== Spherical Relationship Map ===");
    println!("  Circle: cos(n × Δθ) = T_n(cos Δθ) — Chebyshev");
    println!("  Sphere: P_l(cos γ)                 — Legendre");
    println!();

    let max_n = 15;

    // ══════════════════════════════════════════════════════════════
    // PART 1: Named Relationships — Full harmonic profile comparison
    // ══════════════════════════════════════════════════════════════
    println!("=== PART 1: Named Relationships ===\n");

    let relationships = catalog();

    for rel in &relationships {
        let delta = rel.angle_deg * PI / 180.0;
        println!("  {} ({}°, {}):", rel.name, rel.angle_deg, rel.category);
        println!("  {:>4}  {:>10}  {:>10}  {:>10}  {:>8}",
                 "n/l", "Circle", "Sphere", "Diff", "Status");
        println!("  {:>4}  {:>10}  {:>10}  {:>10}  {:>8}",
                 "---", "------", "------", "----", "------");

        let mut max_diff = 0.0f64;
        let mut max_diff_n = 0;

        for n in 1..=max_n {
            let c = circle_coherence(delta, n);
            let s = sphere_coherence(delta, n);
            let diff = (c - s).abs();

            if diff > max_diff {
                max_diff = diff;
                max_diff_n = n;
            }

            let status = if diff < 0.001 {
                "SAME"
            } else if diff < 0.05 {
                "close"
            } else if (c.abs() - 1.0).abs() < 0.01 && (s.abs() - 1.0).abs() > 0.1 {
                "CIRCLE!"  // circle detects perfectly, sphere misses
            } else if (s.abs() - 1.0).abs() < 0.01 && (c.abs() - 1.0).abs() > 0.1 {
                "SPHERE!"  // sphere detects perfectly, circle misses
            } else if diff > 0.3 {
                "DIVERGE"
            } else {
                "differ"
            };

            // Print natural harmonic + neighbors, and any large divergences
            let is_natural = n == rel.natural_n;
            let show = n <= 6 || is_natural || diff > 0.3 || n == max_n;
            if show {
                let marker = if is_natural { " <-- natural" } else { "" };
                println!("  {:>4}  {:>10.4}  {:>10.4}  {:>10.4}  {:>8}{}",
                         n, c, s, diff, status, marker);
            }
        }

        println!("  Peak divergence: n={}, diff={:.4}", max_diff_n, max_diff);
        println!();
    }

    // ══════════════════════════════════════════════════════════════
    // PART 2: Dense Sweep — 1° resolution, all harmonics
    // ══════════════════════════════════════════════════════════════
    println!("=== PART 2: Dense Divergence Sweep (1° steps, n=1..{}) ===\n", max_n);

    // For each angle, find the harmonic with maximum divergence
    println!("  {:>6}  {:>6}  {:>10}  {:>10}  {:>10}",
             "Angle", "Worst n", "Circle", "Sphere", "Divergence");
    println!("  {:>6}  {:>6}  {:>10}  {:>10}  {:>10}",
             "-----", "------", "------", "------", "----------");

    let mut global_max_div = 0.0f64;
    let mut global_max_angle = 0.0f64;
    let mut global_max_n = 0;

    // Collect for summary statistics
    let mut div_by_n = vec![0.0f64; max_n + 1]; // sum of divergences per harmonic
    let mut div_count_by_n = vec![0usize; max_n + 1];
    let mut angles_with_big_div = Vec::new(); // (angle, n, div)

    for deg in 0..=180 {
        let delta = deg as f64 * PI / 180.0;
        let mut worst_diff = 0.0f64;
        let mut worst_n = 1;

        for n in 1..=max_n {
            let c = circle_coherence(delta, n);
            let s = sphere_coherence(delta, n);
            let diff = (c - s).abs();

            div_by_n[n] += diff;
            div_count_by_n[n] += 1;

            if diff > worst_diff {
                worst_diff = diff;
                worst_n = n;
            }
        }

        if worst_diff > global_max_div {
            global_max_div = worst_diff;
            global_max_angle = deg as f64;
            global_max_n = worst_n;
        }

        if worst_diff > 0.5 {
            angles_with_big_div.push((deg as f64, worst_n, worst_diff));
        }

        // Print every 10° + extremes
        if deg % 10 == 0 || worst_diff > 0.8 {
            let c = circle_coherence(delta, worst_n);
            let s = sphere_coherence(delta, worst_n);
            println!("  {:>5}°  {:>6}  {:>10.4}  {:>10.4}  {:>10.4}",
                     deg, worst_n, c, s, worst_diff);
        }
    }

    println!();
    println!("  Global max divergence: {:.4} at angle={}°, n={}",
             global_max_div, global_max_angle, global_max_n);
    println!("  Angles with divergence > 0.5: {}", angles_with_big_div.len());
    println!();

    // ══════════════════════════════════════════════════════════════
    // PART 3: Per-Harmonic Divergence Profile
    // ══════════════════════════════════════════════════════════════
    println!("=== PART 3: Mean Divergence by Harmonic Number ===\n");
    println!("  {:>4}  {:>12}  {:>12}",
             "n/l", "Mean |diff|", "Interpretation");
    println!("  {:>4}  {:>12}  {:>12}",
             "---", "-----------", "--------------");

    for n in 1..=max_n {
        let mean = if div_count_by_n[n] > 0 {
            div_by_n[n] / div_count_by_n[n] as f64
        } else { 0.0 };

        let interp = if mean < 0.01 {
            "identical"
        } else if mean < 0.05 {
            "nearly same"
        } else if mean < 0.15 {
            "moderate split"
        } else if mean < 0.30 {
            "significant"
        } else {
            "MAJOR divergence"
        };

        println!("  {:>4}  {:>12.4}  {}", n, mean, interp);
    }
    println!();

    // ══════════════════════════════════════════════════════════════
    // PART 4: Where Circle Hits ±1 but Sphere Doesn't (and vice versa)
    // ══════════════════════════════════════════════════════════════
    println!("=== PART 4: Detection Failures — Where Systems Disagree on Strong Signal ===\n");
    println!("  Looking for cases where one system gives |coherence| > 0.95");
    println!("  and the other gives |coherence| < 0.50 (qualitative disagreement)\n");

    let mut circle_only = Vec::new();
    let mut sphere_only = Vec::new();

    for deg in 1..180 {
        let delta = deg as f64 * PI / 180.0;
        for n in 1..=max_n {
            let c = circle_coherence(delta, n);
            let s = sphere_coherence(delta, n);

            if c.abs() > 0.95 && s.abs() < 0.50 {
                circle_only.push((deg, n, c, s));
            }
            if s.abs() > 0.95 && c.abs() < 0.50 {
                sphere_only.push((deg, n, c, s));
            }
        }
    }

    println!("  CIRCLE detects (|c|>0.95) but SPHERE misses (|s|<0.50):");
    if circle_only.is_empty() {
        println!("    (none)");
    } else {
        println!("    {:>6}  {:>4}  {:>10}  {:>10}", "Angle", "n", "Circle", "Sphere");
        println!("    {:>6}  {:>4}  {:>10}  {:>10}", "-----", "--", "------", "------");
        for &(deg, n, c, s) in &circle_only {
            println!("    {:>5}°  {:>4}  {:>10.4}  {:>10.4}", deg, n, c, s);
        }
    }
    println!();

    println!("  SPHERE detects (|s|>0.95) but CIRCLE misses (|c|<0.50):");
    if sphere_only.is_empty() {
        println!("    (none)");
    } else {
        println!("    {:>6}  {:>4}  {:>10}  {:>10}", "Angle", "n", "Circle", "Sphere");
        println!("    {:>6}  {:>4}  {:>10}  {:>10}", "-----", "--", "------", "------");
        for &(deg, n, c, s) in &sphere_only {
            println!("    {:>5}°  {:>4}  {:>10.4}  {:>10.4}", deg, n, c, s);
        }
    }
    println!();

    // ══════════════════════════════════════════════════════════════
    // PART 5: Latitude Dimension — What the Circle Cannot See
    // ══════════════════════════════════════════════════════════════
    println!("=== PART 5: Latitude — Sphere-Only Encoding Capacity ===\n");
    println!("  Same Δφ on equator vs at various latitudes.");
    println!("  Circle: identical. Sphere: different angular distance → different coherence.\n");

    let test_phis = [30.0f64, 60.0, 90.0, 120.0];
    let latitudes = [0.0f64, 30.0, 45.0, 60.0, 75.0]; // 0 = equator, degrees from equator

    for &dphi_deg in &test_phis {
        let dphi = dphi_deg * PI / 180.0;
        println!("  Δφ = {}°:", dphi_deg);
        println!("  {:>8}  {:>8}  {:>10}  {:>10}  {:>10}  {:>10}",
                 "Lat", "γ (°)", "P_1", "P_2", "P_3", "P_4");
        println!("  {:>8}  {:>8}  {:>10}  {:>10}  {:>10}  {:>10}",
                 "---", "-----", "---", "---", "---", "---");

        for &lat_deg in &latitudes {
            let theta = PI / 2.0 - lat_deg * PI / 180.0; // convert latitude to polar angle
            let a = SpherePoint { theta, phi: 0.0 };
            let b = SpherePoint { theta, phi: dphi };
            let cg = a.cos_gamma(&b);
            let gamma_deg = cg.acos() * 180.0 / PI;

            println!("  {:>7.0}°  {:>7.1}°  {:>10.4}  {:>10.4}  {:>10.4}  {:>10.4}",
                     lat_deg, gamma_deg,
                     legendre(1, cg), legendre(2, cg),
                     legendre(3, cg), legendre(4, cg));
        }

        // Circle reference (equator only, one value)
        let cc1 = circle_coherence(dphi, 1);
        let cc2 = circle_coherence(dphi, 2);
        let cc3 = circle_coherence(dphi, 3);
        let cc4 = circle_coherence(dphi, 4);
        println!("  {:>8}  {:>8}  {:>10.4}  {:>10.4}  {:>10.4}  {:>10.4}",
                 "Circle", format!("{:.1}", dphi_deg),
                 cc1, cc2, cc3, cc4);
        println!();
    }

    // ══════════════════════════════════════════════════════════════
    // VERDICT
    // ══════════════════════════════════════════════════════════════
    println!("=== VERDICT ===\n");

    println!("  Systems match at n/l = 1 (mean divergence < 0.01).");
    println!("  Divergence grows with harmonic number.");
    println!();

    println!("  Circle-only detections (strong signal sphere misses): {}", circle_only.len());
    println!("  Sphere-only detections (strong signal circle misses): {}", sphere_only.len());
    println!();

    if !circle_only.is_empty() {
        println!("  IMPORTANT: The circle detects {} relationships at |c|>0.95",
                 circle_only.len());
        println!("  where the sphere gives |s|<0.50. These are relationships that");
        println!("  WOULD BE LOST if we naively replaced cos(nΔθ) with P_l(cos γ).");
        println!();

        // Summarize by harmonic number
        let mut by_n = std::collections::HashMap::new();
        for &(_, n, _, _) in &circle_only {
            *by_n.entry(n).or_insert(0) += 1;
        }
        let mut sorted: Vec<_> = by_n.into_iter().collect();
        sorted.sort_by_key(|&(n, _)| n);
        println!("  Circle-only detections by harmonic:");
        for (n, count) in &sorted {
            println!("    n={}: {} angle(s)", n, count);
        }
        println!();
    }

    if circle_only.is_empty() && sphere_only.is_empty() {
        println!("  No qualitative detection disagreements — systems are");
        println!("  quantitatively different but never catastrophically wrong.");
    } else if !circle_only.is_empty() && sphere_only.is_empty() {
        println!("  The sphere is STRICTLY LESS SENSITIVE at detecting harmonic");
        println!("  resonances on the equator. The Legendre polynomial 'spreads'");
        println!("  the energy across angles differently than Chebyshev.");
        println!("  The sphere's advantage must come from LATITUDE, not from");
        println!("  replacing circle coherence with Legendre coherence.");
        println!();
        println!("  IMPLICATION: The right architecture is likely HYBRID —");
        println!("  keep cos(nΔθ) for azimuthal coherence, ADD P_l for elevation.");
        println!("  Circle + sphere, not sphere replacing circle.");
    } else if circle_only.is_empty() && !sphere_only.is_empty() {
        println!("  The sphere detects relationships the circle cannot.");
        println!("  Direct replacement of cos(nΔθ) with P_l(cos γ) may be viable.");
    }

    println!();
    println!("=== END ===");
}
