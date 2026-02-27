// Musical Interval Theory — pure math from Phase 5.
//
// Maps harmonic channel ratios to musical intervals.
// Tenney height, interval identification, consonance scoring.
// Pure number theory, no neural network required.

/// Named musical intervals as (name, numerator, denominator).
pub const MUSICAL_INTERVALS: &[(&str, u32, u32)] = &[
    ("unison", 1, 1),
    ("minor second", 16, 15),
    ("major second", 9, 8),
    ("minor third", 6, 5),
    ("major third", 5, 4),
    ("perfect fourth", 4, 3),
    ("tritone", 45, 32),
    ("perfect fifth", 3, 2),
    ("minor sixth", 8, 5),
    ("major sixth", 5, 3),
    ("minor seventh", 16, 9),
    ("major seventh", 15, 8),
    ("octave", 2, 1),
];

/// Consonance ranking (lower = more consonant).
pub const CONSONANCE_RANK: &[(&str, u32)] = &[
    ("unison", 0),
    ("octave", 1),
    ("perfect fifth", 2),
    ("perfect fourth", 3),
    ("major third", 4),
    ("minor third", 5),
    ("major sixth", 6),
    ("minor sixth", 7),
    ("major second", 8),
    ("minor seventh", 9),
    ("major seventh", 10),
    ("minor second", 11),
    ("tritone", 12),
];

/// Greatest common divisor (Euclid's algorithm).
pub fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Tenney height of ratio p/q: log2(p * q) where p/q is in lowest terms.
/// Lower = more consonant.
pub fn tenney_height(p: u32, q: u32) -> f64 {
    assert!(p > 0 && q > 0, "tenney_height: p and q must be positive");
    let g = gcd(p, q);
    let p_reduced = p / g;
    let q_reduced = q / g;
    ((p_reduced as f64) * (q_reduced as f64)).log2()
}

/// Identify the closest named musical interval for a given ratio.
/// Returns (name, distance_from_pure_ratio).
pub fn identify_interval(ratio: f64) -> (&'static str, f64) {
    // Reduce ratio to within one octave (1.0 to 2.0)
    let mut r = ratio;
    while r > 2.0 {
        r /= 2.0;
    }
    while r < 1.0 {
        r *= 2.0;
    }

    let mut best_name = "unknown";
    let mut best_dist = f64::INFINITY;

    for &(name, num, den) in MUSICAL_INTERVALS {
        let target = num as f64 / den as f64;
        let dist = (r - target).abs();
        if dist < best_dist {
            best_dist = dist;
            best_name = name;
        }
    }
    (best_name, best_dist)
}

/// Consonance score for the ratio between harmonic channels n_a and n_b.
/// Returns (tenney_height, interval_name, consonance_rank).
pub fn consonance_score(n_a: u32, n_b: u32) -> (f64, &'static str, u32) {
    if n_a == 0 || n_b == 0 {
        return (f64::INFINITY, "undefined", 99);
    }
    let hi = n_a.max(n_b);
    let lo = n_a.min(n_b);
    let ratio = hi as f64 / lo as f64;

    // Tenney height using reduced fraction
    let g = gcd(hi, lo);
    let p = hi / g;
    let q = lo / g;
    let th = ((p as f64) * (q as f64)).log2();

    let (name, _dist) = identify_interval(ratio);
    let rank = CONSONANCE_RANK
        .iter()
        .find(|&&(n, _)| n == name)
        .map(|&(_, r)| r)
        .unwrap_or(13);

    (th, name, rank)
}
