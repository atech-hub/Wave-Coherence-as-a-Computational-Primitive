/// Variant 1: Quantile Mapping Sweep
///
/// Tests whether quantile-based phase adjustment opens the operating window
/// that the linear z-score formula could not.
///
/// Linear (baseline): adj = α × (mag − μ) / σ     → unbounded, outlier-sensitive
/// Quantile:          adj = α × (2 × percentile − 1) → bounded to [-α, +α]
///
/// Key insight: rank percentile maps every magnitude to [0, 1] regardless of
/// distribution shape. No outliers, no fat tails. The 1.7-sigma problem that
/// broke 12/23 tests at α=0.1 cannot occur — worst case is ±α.
///
/// Zero dependencies. Edition 2024.

use std::f64::consts::PI;

// =============================================================================
// RNG — xorshift64, deterministic
// =============================================================================

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Rng { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    /// Uniform [0, 1)
    fn uniform(&mut self) -> f64 {
        (self.next_u64() & 0x1FFFFFFFFFFFFF) as f64 / (1u64 << 53) as f64
    }
}

// =============================================================================
// Coherence functions — linear (baseline) and quantile (variant 1)
// =============================================================================

/// Original linear z-score coherence
fn coherence_linear(phi_a: f64, phi_b: f64, n: u32,
                    mag_a: f64, mag_b: f64,
                    alpha: f64, r_mean: f64, r_std: f64) -> f64 {
    let adj_a = if r_std > 1e-15 { alpha * (mag_a - r_mean) / r_std } else { 0.0 };
    let adj_b = if r_std > 1e-15 { alpha * (mag_b - r_mean) / r_std } else { 0.0 };
    (n as f64 * ((phi_a + adj_a) - (phi_b + adj_b))).cos()
}

/// Quantile mapping coherence — adj bounded to [-alpha, +alpha]
fn coherence_quantile(phi_a: f64, phi_b: f64, n: u32,
                      pct_a: f64, pct_b: f64,
                      alpha: f64) -> f64 {
    let adj_a = alpha * (2.0 * pct_a - 1.0);
    let adj_b = alpha * (2.0 * pct_b - 1.0);
    (n as f64 * ((phi_a + adj_a) - (phi_b + adj_b))).cos()
}

// =============================================================================
// Variant enum — selects which coherence function to use
// =============================================================================

#[derive(Clone, Copy, PartialEq)]
enum Variant {
    Linear,
    Quantile,
}

// =============================================================================
// Mode configuration
// =============================================================================

#[derive(Clone)]
struct ModeConfig {
    #[allow(dead_code)]
    name: &'static str,
    alpha: f64,
    magnitudes: Vec<f64>,
    r_mean: f64,
    r_std: f64,
    variant: Variant,
    percentiles: Vec<f64>,
}

impl ModeConfig {
    fn mode_a() -> Self {
        ModeConfig {
            name: "A (Circle)", alpha: 0.0, magnitudes: vec![],
            r_mean: 0.0, r_std: 0.0,
            variant: Variant::Linear, percentiles: vec![],
        }
    }

    fn mode_b_uniform() -> Self {
        ModeConfig {
            name: "B-Uniform", alpha: 0.1, magnitudes: vec![],
            r_mean: 0.0, r_std: 0.0,
            variant: Variant::Linear, percentiles: vec![],
        }
    }

    fn mode_b_varied_linear(n: usize, alpha: f64) -> Self {
        let mut rng = Rng::new(42);
        let mags: Vec<f64> = (0..n).map(|_| 0.108 + rng.uniform() * (1.892 - 0.108)).collect();
        let mean = mags.iter().sum::<f64>() / mags.len() as f64;
        let std = (mags.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / mags.len() as f64).sqrt();
        ModeConfig {
            name: "B-Varied", alpha, magnitudes: mags,
            r_mean: mean, r_std: std,
            variant: Variant::Linear, percentiles: vec![],
        }
    }

    fn mode_b_varied_quantile(n: usize, alpha: f64) -> Self {
        let mut rng = Rng::new(42);
        let mags: Vec<f64> = (0..n).map(|_| 0.108 + rng.uniform() * (1.892 - 0.108)).collect();
        let mean = mags.iter().sum::<f64>() / mags.len() as f64;
        let std = (mags.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / mags.len() as f64).sqrt();

        // Compute rank percentiles
        let mut indexed: Vec<(usize, f64)> = mags.iter().enumerate()
            .map(|(i, &m)| (i, m)).collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let mut percentiles = vec![0.0; n];
        let denom = if n > 1 { (n - 1) as f64 } else { 1.0 };
        for (rank, &(orig_idx, _)) in indexed.iter().enumerate() {
            percentiles[orig_idx] = rank as f64 / denom;
        }

        ModeConfig {
            name: "B-Varied", alpha, magnitudes: mags,
            r_mean: mean, r_std: std,
            variant: Variant::Quantile, percentiles,
        }
    }

    fn mag(&self, i: usize) -> f64 {
        if self.magnitudes.is_empty() { 1.0 } else { self.magnitudes[i % self.magnitudes.len()] }
    }

    fn pct(&self, i: usize) -> f64 {
        if self.percentiles.is_empty() { 0.5 } else { self.percentiles[i % self.percentiles.len()] }
    }

    fn coh(&self, phi_a: f64, phi_b: f64, n: u32, idx_a: usize, idx_b: usize) -> f64 {
        match self.variant {
            Variant::Linear => {
                coherence_linear(phi_a, phi_b, n, self.mag(idx_a), self.mag(idx_b),
                                 self.alpha, self.r_mean, self.r_std)
            }
            Variant::Quantile => {
                coherence_quantile(phi_a, phi_b, n, self.pct(idx_a), self.pct(idx_b),
                                   self.alpha)
            }
        }
    }
}

// =============================================================================
// Test result
// =============================================================================

struct TestResult {
    id: u32,
    name: &'static str,
    pass_a: bool,
    pass_bu: bool,
    pass_bv: bool,
    structural: bool,
}

impl TestResult {
    fn new(id: u32, name: &'static str, a: bool, bu: bool, bv: bool) -> Self {
        TestResult { id, name, pass_a: a, pass_bu: bu, pass_bv: bv, structural: false }
    }

    fn structural(id: u32, name: &'static str, pass: bool) -> Self {
        TestResult { id, name, pass_a: pass, pass_bu: pass, pass_bv: pass, structural: true }
    }
}

fn deg2rad(d: f64) -> f64 { d * PI / 180.0 }

// =============================================================================
// T1–T23: All 23 tests (identical to three_mode_harness.rs)
// =============================================================================

fn test_01(modes: &[ModeConfig]) -> TestResult {
    let n_entities = 12;
    let target_idx = 7;
    let phases: Vec<f64> = (0..n_entities).map(|i| deg2rad(i as f64 * 30.0)).collect();
    let target_phi = phases[target_idx];
    let mut passes = [false; 3];
    for (mi, mode) in modes.iter().enumerate() {
        let self_coh = mode.coh(target_phi, target_phi, 1, target_idx, target_idx);
        let matches: usize = (0..n_entities)
            .filter(|&i| mode.coh(target_phi, phases[i], 1, target_idx, i) > 0.99)
            .count();
        passes[mi] = (self_coh - 1.0).abs() < 1e-12 && matches == 1;
    }
    TestResult::new(1, "Exact match", passes[0], passes[1], passes[2])
}

fn test_02(modes: &[ModeConfig]) -> TestResult {
    let phases: Vec<f64> = (0..12).map(|i| deg2rad(i as f64 * 30.0)).collect();
    let target = phases[0];
    let threshold = 0.95;
    let mut passes = [false; 3];
    for (mi, mode) in modes.iter().enumerate() {
        let c120 = mode.coh(target, phases[4], 3, 0, 4);
        let c240 = mode.coh(target, phases[8], 3, 0, 8);
        let c90 = mode.coh(target, phases[3], 3, 0, 3);
        passes[mi] = c120 > threshold && c240 > threshold && c90 < threshold;
    }
    TestResult::new(2, "Harmonic family n=3", passes[0], passes[1], passes[2])
}

fn test_03(modes: &[ModeConfig]) -> TestResult {
    let phases: Vec<f64> = (0..12).map(|i| deg2rad(i as f64 * 30.0)).collect();
    let target = phases[0];
    let threshold = 0.95;
    let mut passes = [false; 3];
    for (mi, mode) in modes.iter().enumerate() {
        let c180 = mode.coh(target, phases[6], 2, 0, 6);
        let c90 = mode.coh(target, phases[3], 2, 0, 3);
        let matches: Vec<usize> = (0..12)
            .filter(|&i| mode.coh(target, phases[i], 2, 0, i) > threshold)
            .collect();
        passes[mi] = c180 > threshold && c90 < threshold
            && matches.contains(&0) && matches.contains(&6) && matches.len() == 2;
    }
    TestResult::new(3, "Opposition n=2", passes[0], passes[1], passes[2])
}

fn test_04(modes: &[ModeConfig]) -> TestResult {
    let offsets = [0.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 90.0];
    let target = deg2rad(0.0);
    let mut passes = [false; 3];
    for (mi, mode) in modes.iter().enumerate() {
        let scores: Vec<f64> = offsets.iter().enumerate()
            .map(|(i, &d)| mode.coh(target, deg2rad(d), 1, 0, i + 1))
            .collect();
        let monotone = scores.windows(2).all(|w| w[0] >= w[1] - 1e-12);
        passes[mi] = monotone;
    }
    TestResult::new(4, "Fuzzy falloff", passes[0], passes[1], passes[2])
}

fn test_05(modes: &[ModeConfig]) -> TestResult {
    let vendors = [30.0, 30.0, 30.0, 200.0];
    let cats = [120.0, 240.0, 120.0, 120.0];
    let tv = deg2rad(30.0);
    let tc = deg2rad(120.0);
    let mut passes = [false; 3];
    for (mi, mode) in modes.iter().enumerate() {
        let scores: Vec<f64> = (0..4).map(|i| {
            let vc = mode.coh(tv, deg2rad(vendors[i]), 1, 0, i);
            let cc = mode.coh(tc, deg2rad(cats[i]), 1, 0, i);
            vc * cc
        }).collect();
        passes[mi] = scores[0] > 0.99 && scores[2] > 0.99
            && scores[0] > scores[1] && scores[0] > scores[3];
    }
    TestResult::new(5, "Multi-attribute product", passes[0], passes[1], passes[2])
}

fn test_06() -> TestResult {
    let size = 5;
    let chain = |start: usize, step: i32, depth: usize| -> Vec<usize> {
        let mut result = vec![start];
        let mut cur = start as i32;
        for _ in 0..depth {
            cur = (cur + step).rem_euclid(size);
            result.push(cur as usize);
        }
        result
    };
    let gen_path = chain(0, 1, 3);
    let dest = chain(0, 2, 3);
    let gen2 = chain(3, 1, 2);
    let weak = chain(0, -1, 4);
    let ctrl = chain(0, -2, 4);
    let pass = gen_path == vec![0,1,2,3] && dest == vec![0,2,4,1]
        && gen2 == vec![3,4,0] && weak == vec![0,4,3,2,1] && ctrl == vec![0,3,1,4,2];
    TestResult::structural(6, "Directed cycle", pass)
}

fn test_07() -> TestResult {
    let pairs = [(0,1), (2,11), (3,10), (4,9), (5,8), (6,7)];
    let partner = |pos: usize| -> Option<usize> {
        for &(a, b) in &pairs {
            if a == pos { return Some(b); }
            if b == pos { return Some(a); }
        }
        None
    };
    let pass = partner(0) == Some(1)
        && partner(2) == Some(11)
        && partner(3) == Some(10)
        && pairs.iter().all(|&(a,b)| partner(a) == Some(b) && partner(b) == Some(a));
    TestResult::structural(7, "Structural pairs", pass)
}

fn test_08(modes: &[ModeConfig]) -> TestResult {
    let bucket_count = 100u64;
    let entity_count = 1000;
    let target_value = 42u64;
    let threshold = 0.9999;
    let values: Vec<u64> = (0..entity_count).map(|i| (i * 37 + 13) % bucket_count).collect();
    let phases: Vec<f64> = values.iter()
        .map(|&v| 2.0 * PI * v as f64 / bucket_count as f64).collect();
    let target_phi = 2.0 * PI * target_value as f64 / bucket_count as f64;
    let linear: Vec<usize> = values.iter().enumerate()
        .filter(|&(_, &v)| v == target_value).map(|(i, _)| i).collect();
    let mut passes = [false; 3];
    for (mi, mode) in modes.iter().enumerate() {
        let wave: Vec<usize> = (0..entity_count as usize)
            .filter(|&i| mode.coh(target_phi, phases[i], 1, 0, i) > threshold)
            .collect();
        passes[mi] = linear == wave;
    }
    TestResult::new(8, "Wave = linear scan", passes[0], passes[1], passes[2])
}

fn test_09(modes: &[ModeConfig]) -> TestResult {
    let group_centers = [0.0, 90.0, 120.0, 240.0];
    let mut phases = Vec::new();
    let mut groups = Vec::new();
    for (gi, &center) in group_centers.iter().enumerate() {
        for j in 0..25 {
            let offset = (j as f64 - 12.0) * 0.2;
            phases.push(deg2rad(center + offset));
            groups.push(gi);
        }
    }
    let target = deg2rad(3.0);
    let threshold = 0.85;
    let mut passes = [false; 3];
    for (mi, mode) in modes.iter().enumerate() {
        let mut group_counts = [0usize; 4];
        for (i, &phi) in phases.iter().enumerate() {
            if mode.coh(target, phi, 3, 0, i) > threshold {
                group_counts[groups[i]] += 1;
            }
        }
        passes[mi] = group_counts[2] > 0 && group_counts[3] > 0 && group_counts[1] == 0;
    }
    TestResult::new(9, "Harmonic vs JOIN", passes[0], passes[1], passes[2])
}

fn test_10(modes: &[ModeConfig]) -> TestResult {
    let phases: Vec<f64> = (0..12).map(|i| deg2rad(i as f64 * 30.0)).collect();
    let target = phases[0];
    let mut passes = [false; 3];
    for (mi, mode) in modes.iter().enumerate() {
        let broad_count = [2, 4, 8].iter()
            .filter(|&&i| mode.coh(target, phases[i], 3, 0, i) > 0.95)
            .count();
        let narrow_180 = mode.coh(target, phases[6], 2, 0, 6) > 0.95;
        let narrow_90 = mode.coh(target, phases[3], 2, 0, 3) > 0.95;
        passes[mi] = broad_count >= 2 && narrow_180 && !narrow_90;
    }
    TestResult::new(10, "Type-dependent reach", passes[0], passes[1], passes[2])
}

fn test_11(modes: &[ModeConfig]) -> TestResult {
    let pairs = [(5.0, 7.0), (10.0, 11.0), (10.0, 10.1)];
    let threshold = 0.9;
    let mut passes = [false; 3];
    for (mi, mode) in modes.iter().enumerate() {
        let mut divergence_ns = Vec::new();
        for &(a_deg, b_deg) in &pairs {
            let a = deg2rad(a_deg);
            let b = deg2rad(b_deg);
            let max_n = if (a_deg - b_deg).abs() < 0.5 { 1800u32 } else { 180 };
            let mut div_n = 0u32;
            for n in 1..=max_n {
                let c = mode.coh(a, b, n, 0, 1);
                if c.abs() < threshold && div_n == 0 {
                    div_n = n;
                }
            }
            divergence_ns.push(div_n);
        }
        passes[mi] = divergence_ns.iter().all(|&n| n > 0)
            && divergence_ns[1] > divergence_ns[0]
            && divergence_ns[2] > divergence_ns[1];
    }
    TestResult::new(11, "Fingerprint disambiguation", passes[0], passes[1], passes[2])
}

fn test_12(modes: &[ModeConfig]) -> TestResult {
    let a = deg2rad(30.0);
    let b = deg2rad(35.0);
    let mut passes = [false; 3];
    for (mi, mode) in modes.iter().enumerate() {
        let base = mode.coh(a, b, 1, 0, 1);
        let mutual = base * 1.5;
        let oneway = base * 1.2;
        let none = base;
        let ratio_m = mutual / base;
        let ratio_o = oneway / base;
        passes[mi] = mutual > oneway && oneway > none
            && (ratio_m - 1.5).abs() < 0.001
            && (ratio_o - 1.2).abs() < 0.001;
    }
    TestResult::new(12, "Mutual amplification", passes[0], passes[1], passes[2])
}

fn test_13() -> TestResult {
    let size: i32 = 5;
    let steps = [1i32, 2, -1, -2];
    let mut map = vec![vec![None::<i32>; 5]; 5];
    let mut no_conflicts = true;
    let mut all_assigned = true;
    for &step in &steps {
        for start in 0..5i32 {
            let dest = (start + step).rem_euclid(size) as usize;
            if map[start as usize][dest].is_some() {
                no_conflicts = false;
            } else {
                map[start as usize][dest] = Some(step);
            }
        }
    }
    for a in 0..5 {
        for b in 0..5 {
            if a != b && map[a][b].is_none() {
                all_assigned = false;
            }
        }
    }
    let mut counts = [0usize; 4];
    for (si, &s) in steps.iter().enumerate() {
        for a in 0..5 {
            for b in 0..5 {
                if a != b && map[a][b] == Some(s) {
                    counts[si] += 1;
                }
            }
        }
    }
    let pass = all_assigned && no_conflicts && counts.iter().all(|&c| c == 5);
    TestResult::structural(13, "Cycle uniqueness", pass)
}

fn test_14(modes: &[ModeConfig]) -> TestResult {
    let angles = [0.0, 60.0, 72.0, 90.0, 120.0, 180.0, 240.0, 270.0, 288.0, 300.0];
    let phases: Vec<f64> = angles.iter().map(|&d| deg2rad(d)).collect();
    let target = phases[0];
    let threshold = 0.95;
    let mut passes = [false; 3];
    for (mi, mode) in modes.iter().enumerate() {
        let find = |n: u32| -> Vec<usize> {
            (0..phases.len())
                .filter(|&i| mode.coh(target, phases[i], n, 0, i) > threshold)
                .collect()
        };
        let h3 = find(3);
        let h4 = find(4);
        let h5 = find(5);
        let h3_ok = h3.contains(&0) && h3.contains(&4) && h3.contains(&6)
            && !h3.contains(&3) && !h3.contains(&1);
        let h4_ok = h4.contains(&0) && h4.contains(&3) && h4.contains(&5) && h4.contains(&7)
            && !h4.contains(&4) && !h4.contains(&1);
        let h5_ok = h5.contains(&0) && h5.contains(&2) && h5.contains(&8);
        passes[mi] = h3_ok && h4_ok && h5_ok;
    }
    TestResult::new(14, "Harmonic orthogonality", passes[0], passes[1], passes[2])
}

fn test_15(modes: &[ModeConfig]) -> TestResult {
    let angles = [357.0, 358.0, 359.0, 0.0, 1.0, 2.0, 3.0, 180.0];
    let phases: Vec<f64> = angles.iter().map(|&d| deg2rad(d)).collect();
    let target = deg2rad(0.0);
    let mut passes = [false; 3];
    for (mi, mode) in modes.iter().enumerate() {
        let c_1_359 = mode.coh(phases[4], phases[2], 1, 4, 2);
        let near_threshold = 0.99;
        let near: Vec<usize> = (0..phases.len())
            .filter(|&i| mode.coh(target, phases[i], 1, 0, i) > near_threshold)
            .collect();
        passes[mi] = c_1_359 > 0.95 && !near.contains(&7) && near.contains(&3);
    }
    TestResult::new(15, "Wraparound", passes[0], passes[1], passes[2])
}

fn test_16(modes: &[ModeConfig]) -> TestResult {
    let n = 360;
    let phases: Vec<f64> = (0..n).map(|i| 2.0 * PI * i as f64 / n as f64).collect();
    let bucket_angle_cos = (2.0 * PI / n as f64).cos();
    let threshold = (1.0 + bucket_angle_cos) / 2.0;
    let mut passes = [false; 3];
    for (mi, mode) in modes.iter().enumerate() {
        let mut perfect_count = 0;
        for i in 0..n {
            let matches: Vec<usize> = (0..n)
                .filter(|&j| mode.coh(phases[i], phases[j], 1, i, j) > threshold)
                .collect();
            if matches.len() == 1 && matches[0] == i {
                perfect_count += 1;
            }
        }
        passes[mi] = perfect_count == n;
    }
    TestResult::new(16, "360 resolution", passes[0], passes[1], passes[2])
}

fn test_17(modes: &[ModeConfig]) -> TestResult {
    let golden_angle = 137.50776405003785_f64;
    let scenarios: Vec<(usize, u32)> = vec![
        (7, 12), (50, 360), (200, 360), (360, 360),
    ];
    let mut passes = [false; 3];
    for (mi, mode) in modes.iter().enumerate() {
        let mut exact_results = Vec::new();
        for &(n_obj, buckets) in &scenarios {
            let positions: Vec<f64> = (0..n_obj)
                .map(|i| deg2rad((i as f64 * golden_angle) % 360.0)).collect();
            let bucket_angle_cos = (2.0 * PI / buckets as f64).cos();
            let exact_threshold = (1.0 + bucket_angle_cos) / 2.0;
            let mut exact_ok = true;
            for i in 0..n_obj {
                let matches: usize = (0..n_obj)
                    .filter(|&j| mode.coh(positions[i], positions[j], 1, i, j) > exact_threshold)
                    .count();
                if matches != 1 { exact_ok = false; break; }
            }
            exact_results.push(exact_ok);
        }
        let sparse_ok = exact_results[0];
        let degrades = exact_results.iter().any(|&ok| !ok);
        passes[mi] = sparse_ok && degrades;
    }
    TestResult::new(17, "Density scaling", passes[0], passes[1], passes[2])
}

fn test_18() -> TestResult {
    let buckets = 360u32;
    let test_angles = [0.0, 45.0, 90.0, 137.5, 180.0, 270.0, 359.9];
    let mut pass = true;
    for &deg in &test_angles {
        let phi = deg2rad(deg);
        let bucket = ((phi * buckets as f64 / (2.0 * PI)).floor() as u32) % buckets;
        let expected = (deg.floor() as u32) % buckets;
        if bucket != expected { pass = false; }
    }
    TestResult::structural(18, "Bucket index", pass)
}

fn test_19() -> TestResult {
    let bkts = 60u32;
    let golden_angle = 137.50776405003785_f64;
    let silver_angle = 222.49223594996215_f64;
    let n = 50;
    let mut cells_used = std::collections::HashSet::new();
    for i in 0..n {
        let dx = (i as f64 * golden_angle) % 360.0;
        let dy = (i as f64 * silver_angle) % 360.0;
        let bx = ((dx * bkts as f64 / 360.0).floor() as u32) % bkts;
        let by = ((dy * bkts as f64 / 360.0).floor() as u32) % bkts;
        cells_used.insert((bx, by));
    }
    let pass = cells_used.len() > n * 80 / 100 && cells_used.len() > 1;
    TestResult::structural(19, "Torus index", pass)
}

fn test_20() -> TestResult {
    let golden_angle = 137.50776405003785_f64;
    let n_initial = 50usize;
    let mut entries: Vec<(usize, f64)> = Vec::new();
    for i in 0..n_initial {
        let phi = deg2rad((i as f64 * golden_angle) % 360.0);
        entries.push((i, phi));
    }
    let remove_ids: Vec<usize> = (0..10).map(|i| i * 5).collect();
    entries.retain(|&(id, _)| !remove_ids.contains(&id));
    let after_remove = entries.len();
    for i in 0..5 {
        entries.push((100 + i, deg2rad((i as f64 * 12.345 + 77.0) % 360.0)));
    }
    let after_insert = entries.len();
    let pass = after_remove == 40 && after_insert == 45
        && entries.iter().all(|&(id, _)| !remove_ids.contains(&id));
    TestResult::structural(20, "Dynamic mutation", pass)
}

fn test_21(modes: &[ModeConfig]) -> TestResult {
    let letters: Vec<(char, f64)> = vec![
        ('A', 0.0), ('B', 120.0), ('C', 180.0), ('D', 90.0),
        ('E', 60.0), ('F', 72.0), ('G', 37.0), ('H', 143.0),
    ];
    let phases: Vec<f64> = letters.iter().map(|(_, d)| deg2rad(*d)).collect();
    let expected: Vec<(usize, usize, u32)> = vec![
        (0, 1, 3), (0, 2, 2), (0, 3, 4), (0, 4, 6), (0, 5, 5),
    ];
    let detect_threshold = 0.999;
    let noise_threshold = 0.999;
    let mut passes = [false; 3];
    for (mi, mode) in modes.iter().enumerate() {
        let dt = detect_threshold;
        let mut detected = 0;
        for &(i, j, n) in &expected {
            if mode.coh(phases[i], phases[j], n, i, j) > dt {
                detected += 1;
            }
        }
        let noise_clean = [6, 7].iter().all(|&j| {
            (1..=6).all(|n| mode.coh(phases[0], phases[j], n, 0, j) <= noise_threshold)
        });
        passes[mi] = detected == 5 && noise_clean;
    }
    TestResult::new(21, "Harmonic sweep", passes[0], passes[1], passes[2])
}

fn test_22(modes: &[ModeConfig]) -> TestResult {
    let test_angles: Vec<f64> = vec![
        0.0, 30.0, 45.0, 60.0, 72.0, 90.0, 120.0, 137.5, 180.0, 210.0, 270.0, 315.0, 359.0,
    ];
    let phases: Vec<f64> = test_angles.iter().map(|&d| deg2rad(d)).collect();
    let harmonics = [1u32, 2, 3, 4, 5, 6, 8, 12];
    let eps = 1e-10;
    let mut passes = [false; 3];
    for (mi, mode) in modes.iter().enumerate() {
        let mut all_ok = true;
        // Property 1: Symmetry
        for &n in &harmonics {
            for i in 0..phases.len() {
                for j in (i+1)..phases.len() {
                    let fwd = mode.coh(phases[i], phases[j], n, i, j);
                    let rev = mode.coh(phases[j], phases[i], n, j, i);
                    if (fwd - rev).abs() > eps { all_ok = false; }
                }
            }
        }
        // Property 2: Normalization
        for &n in &harmonics {
            for (i, &phi) in phases.iter().enumerate() {
                let self_coh = mode.coh(phi, phi, n, i, i);
                if (self_coh - 1.0).abs() > eps { all_ok = false; }
            }
        }
        // Property 3: PSD (2x2 and 3x3 principal minors)
        for &n in &harmonics {
            let size = phases.len();
            let gram: Vec<Vec<f64>> = (0..size).map(|i|
                (0..size).map(|j| mode.coh(phases[i], phases[j], n, i, j)).collect()
            ).collect();
            for i in 0..size {
                for j in (i+1)..size {
                    let det2 = gram[i][i] * gram[j][j] - gram[i][j] * gram[j][i];
                    if det2 < -eps { all_ok = false; }
                }
            }
            for i in 0..size {
                for j in (i+1)..size {
                    for k in (j+1)..size {
                        let det3 =
                            gram[i][i] * (gram[j][j] * gram[k][k] - gram[j][k] * gram[k][j])
                            - gram[i][j] * (gram[j][i] * gram[k][k] - gram[j][k] * gram[k][i])
                            + gram[i][k] * (gram[j][i] * gram[k][j] - gram[j][j] * gram[k][i]);
                        if det3 < -eps { all_ok = false; }
                    }
                }
            }
        }
        // Property 4: Spectral Scaling
        let threshold: f64 = 0.95;
        let test_ns: Vec<u32> = vec![1, 2, 3, 4, 6, 8, 12];
        let mut prev_res: Option<f64> = None;
        for &n in &test_ns {
            let res = threshold.acos() * 180.0 / PI / n as f64;
            if let Some(prev) = prev_res {
                if res > prev + eps { all_ok = false; }
            }
            prev_res = Some(res);
        }
        passes[mi] = all_ok;
    }
    TestResult::new(22, "Kernel admissibility", passes[0], passes[1], passes[2])
}

fn test_23(modes: &[ModeConfig]) -> TestResult {
    let n_harmonics = 12usize;
    let alignment_threshold = 0.95;
    let groups: Vec<(&str, Vec<f64>, Option<usize>)> = vec![
        ("Triadic", vec![0.0, 120.0, 240.0], Some(3)),
        ("Opposition", vec![0.0, 180.0], Some(2)),
        ("Quadrant", vec![0.0, 90.0, 180.0, 270.0], Some(4)),
        ("Noise", vec![0.0, 37.0, 143.0, 211.0], None),
    ];
    let mut passes = [false; 3];
    for (mi, mode) in modes.iter().enumerate() {
        let mut all_correct = true;
        for &(_, ref angles, expected_fund) in &groups {
            let phases: Vec<f64> = angles.iter().map(|&d| deg2rad(d)).collect();
            let mut signed_sum = vec![0.0f64; n_harmonics];
            let mut pair_count = 0;
            for i in 0..phases.len() {
                for j in (i+1)..phases.len() {
                    for n in 0..n_harmonics {
                        let coh = mode.coh(phases[i], phases[j], (n+1) as u32, i, j);
                        signed_sum[n] += coh;
                    }
                    pair_count += 1;
                }
            }
            let signed_mean: Vec<f64> = signed_sum.iter().map(|s| s / pair_count as f64).collect();
            let fundamental = signed_mean.iter()
                .position(|&m| m > alignment_threshold)
                .map(|i| i + 1);
            if fundamental != expected_fund { all_correct = false; }
        }
        passes[mi] = all_correct;
    }
    TestResult::new(23, "Channel energy (eta)", passes[0], passes[1], passes[2])
}

// =============================================================================
// Run all 23 tests at a given alpha with a given variant
// =============================================================================

fn run_all_at_alpha(alpha: f64, n_mag: usize, variant: Variant) -> (usize, Vec<(u32, bool)>) {
    let mode_a = ModeConfig::mode_a();
    let mode_bu = ModeConfig::mode_b_uniform();
    let mode_bv = match variant {
        Variant::Linear => ModeConfig::mode_b_varied_linear(n_mag, alpha),
        Variant::Quantile => ModeConfig::mode_b_varied_quantile(n_mag, alpha),
    };
    let modes = vec![mode_a, mode_bu, mode_bv];

    let results = vec![
        test_01(&modes), test_02(&modes), test_03(&modes), test_04(&modes),
        test_05(&modes), test_06(), test_07(), test_08(&modes),
        test_09(&modes), test_10(&modes), test_11(&modes), test_12(&modes),
        test_13(), test_14(&modes), test_15(&modes), test_16(&modes),
        test_17(&modes), test_18(), test_19(), test_20(),
        test_21(&modes), test_22(&modes), test_23(&modes),
    ];

    let bv_pass = results.iter().filter(|r| r.pass_bv).count();
    let per_test: Vec<(u32, bool)> = results.iter().map(|r| (r.id, r.pass_bv)).collect();
    (bv_pass, per_test)
}

// =============================================================================
// Within-group discrimination measurement
// =============================================================================

fn measure_discrimination(alpha: f64, variant: Variant) -> (f64, f64, f64) {
    let n_groups = 4;
    let per_group = 25;
    let n_total = n_groups * per_group;
    let group_bases = [0.0_f64, 90.0, 180.0, 270.0];

    let mut mags = vec![0.0f64; n_total];
    let mut phases = vec![0.0f64; n_total];
    for g in 0..n_groups {
        for t in 0..per_group {
            let idx = g * per_group + t;
            phases[idx] = deg2rad(group_bases[g]);
            mags[idx] = 0.3 + 1.4 * (t as f64 / (per_group - 1) as f64);
        }
    }

    let r_mean = mags.iter().sum::<f64>() / mags.len() as f64;
    let r_std = (mags.iter().map(|x| (x - r_mean).powi(2)).sum::<f64>() / mags.len() as f64).sqrt();

    // Compute percentiles for quantile variant
    let percentiles = {
        let mut indexed: Vec<(usize, f64)> = mags.iter().enumerate()
            .map(|(i, &m)| (i, m)).collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let mut p = vec![0.0; n_total];
        let denom = if n_total > 1 { (n_total - 1) as f64 } else { 1.0 };
        for (rank, &(orig_idx, _)) in indexed.iter().enumerate() {
            p[orig_idx] = rank as f64 / denom;
        }
        p
    };

    let mode = ModeConfig {
        name: "B-Varied", alpha, magnitudes: mags,
        r_mean, r_std, variant, percentiles,
    };

    let mut near_scores = Vec::new();
    let mut far_scores = Vec::new();

    for g in 0..n_groups {
        let base = g * per_group;
        for i in 0..per_group {
            for j in (i+1)..per_group {
                let dist = j - i;
                let idx_a = base + i;
                let idx_b = base + j;
                let c = mode.coh(phases[idx_a], phases[idx_b], 1, idx_a, idx_b);
                if dist <= 5 {
                    near_scores.push(c);
                } else if dist >= 15 {
                    far_scores.push(c);
                }
            }
        }
    }

    let near_mean = if near_scores.is_empty() { 0.0 }
        else { near_scores.iter().sum::<f64>() / near_scores.len() as f64 };
    let far_mean = if far_scores.is_empty() { 0.0 }
        else { far_scores.iter().sum::<f64>() / far_scores.len() as f64 };
    let gap = near_mean - far_mean;

    (near_mean, far_mean, gap)
}

// =============================================================================
// Alpha sweep for a given variant
// =============================================================================

fn sweep(variant: Variant, label: &str, formula: &str) {
    println!();
    println!("============================================================");
    println!("  {} SWEEP", label);
    println!("============================================================");
    println!("  Formula: {}", formula);
    println!();

    let n_mag = 1000;

    // Show magnitude distribution stats
    let bv = ModeConfig::mode_b_varied_linear(n_mag, 0.1);
    let cv = bv.r_std / bv.r_mean;
    println!("  Magnitude distribution: U[0.108, 1.892], CV = {:.1}%", cv * 100.0);

    if variant == Variant::Quantile {
        println!("  Quantile mapping: all magnitudes mapped to rank percentile [0, 1]");
        println!("  Max adjustment per entity: +/- alpha (bounded, no outliers)");
        println!("  Max pair delta: 2*alpha (worst case: percentile 0.0 vs 1.0)");
    }
    println!();

    let alphas = [
        0.100, 0.080, 0.060, 0.050, 0.040, 0.030,
        0.025, 0.020, 0.015, 0.010, 0.005, 0.003, 0.001,
    ];

    println!("  {:>8}  {:>10}  {:>10}  {:>10}  {:>12}",
        "alpha", "Pass", "Near mu", "Far mu", "Gap (discr.)");
    println!("  {:>8}  {:>10}  {:>10}  {:>10}  {:>12}",
        "--------", "----------", "----------", "----------", "------------");

    let mut sweet_spot: Option<(f64, f64)> = None;
    let mut best_gap_at_23: Option<(f64, f64)> = None;

    for &alpha in &alphas {
        let (pass_count, _per_test) = run_all_at_alpha(alpha, n_mag, variant);
        let (near, far, gap) = measure_discrimination(alpha, variant);

        let marker = if pass_count == 23 && sweet_spot.is_none() {
            sweet_spot = Some((alpha, gap));
            " <-- 23/23"
        } else { "" };

        if pass_count == 23 {
            if best_gap_at_23.is_none() || gap > best_gap_at_23.unwrap().1 {
                best_gap_at_23 = Some((alpha, gap));
            }
        }

        println!("  {:>8.4}  {:>7}/23   {:>10.6}  {:>10.6}  {:>12.6}{}",
            alpha, pass_count, near, far, gap, marker);
    }

    println!();

    // Analysis
    if let Some((alpha_star, gap_star)) = sweet_spot {
        println!("  FIRST 23/23: alpha* = {:.4}, gap = {:.6}", alpha_star, gap_star);

        // Worst-case phase shift analysis
        if variant == Variant::Quantile {
            let worst_delta = 2.0 * alpha_star; // bounded: percentile 0 vs percentile 1
            let worst_deg = worst_delta * 180.0 / PI;
            println!();
            println!("  Worst-case analysis (quantile):");
            println!("    Per-entity max adjustment: +/- {:.4} rad = +/- {:.2} deg", alpha_star, alpha_star * 180.0 / PI);
            println!("    Worst-case pair delta:     {:.4} rad = {:.2} deg", worst_delta, worst_deg);
            println!("    At n=3 effective shift:    {:.2} deg", worst_deg * 3.0);
            println!("    cos(worst_delta) at n=1:   {:.6}", worst_delta.cos());
            println!("    cos(3*worst_delta) at n=3: {:.6}", (worst_delta * 3.0).cos());
        } else {
            let max_sigma = 1.7;
            let worst_adj = alpha_star * max_sigma;
            let worst_delta = worst_adj * 2.0;
            let worst_deg = worst_delta * 180.0 / PI;
            println!();
            println!("  Worst-case analysis (linear z-score):");
            println!("    Per-entity max adjustment: {:.4} rad = {:.2} deg", worst_adj, worst_adj * 180.0 / PI);
            println!("    Worst-case pair delta:     {:.4} rad = {:.2} deg", worst_delta, worst_deg);
            println!("    At n=3 effective shift:    {:.2} deg", worst_deg * 3.0);
            println!("    cos(worst_delta) at n=1:   {:.6}", worst_delta.cos());
            println!("    cos(3*worst_delta) at n=3: {:.6}", (worst_delta * 3.0).cos());
        }
    } else {
        println!("  No alpha in sweep range achieved 23/23.");
    }

    // Best gap at 23/23
    if let Some((best_alpha, best_gap)) = best_gap_at_23 {
        println!();
        println!("  BEST GAP AT 23/23: alpha = {:.4}, gap = {:.6}", best_alpha, best_gap);

        if best_gap < 1e-6 {
            println!("  VERDICT: Zero discrimination at backward-compatible alpha.");
        } else if best_gap < 0.001 {
            println!("  VERDICT: Marginal discrimination ({:.6}).", best_gap);
        } else if best_gap < 0.01 {
            println!("  VERDICT: Weak but measurable discrimination ({:.6}).", best_gap);
        } else {
            println!("  VERDICT: MEANINGFUL discrimination ({:.6}).", best_gap);
        }
    }

    // Test sensitivity for this variant
    println!();
    println!("  Test sensitivity (first alpha where test fails):");
    let test_alphas = [
        0.001, 0.003, 0.005, 0.010, 0.015, 0.020, 0.025,
        0.030, 0.040, 0.050, 0.060, 0.080, 0.100,
    ];
    let test_names = [
        "Exact match", "Harmonic family n=3", "Opposition n=2", "Fuzzy falloff",
        "Multi-attribute product", "Directed cycle", "Structural pairs",
        "Wave = linear scan", "Harmonic vs JOIN", "Type-dependent reach",
        "Fingerprint disambig.", "Mutual amplification", "Cycle uniqueness",
        "Harmonic orthogonality", "Wraparound", "360 resolution",
        "Density scaling", "Bucket index", "Torus index", "Dynamic mutation",
        "Harmonic sweep", "Kernel admissibility", "Channel energy (eta)",
    ];

    let mut first_fail = vec![f64::NAN; 23];
    for &a in test_alphas.iter().rev() {
        let (_, per_test) = run_all_at_alpha(a, n_mag, variant);
        for (ti, &(_id, pass)) in per_test.iter().enumerate() {
            if !pass {
                first_fail[ti] = a;
            }
        }
    }

    let mut sensitivity: Vec<(usize, f64)> = first_fail.iter().enumerate()
        .filter(|&(_, &a)| !a.is_nan())
        .map(|(i, &a)| (i, a))
        .collect();
    sensitivity.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    if sensitivity.is_empty() {
        println!("    No tests fail at any alpha in sweep range!");
    } else {
        println!("  {:>3}  {:<24}  {:>12}", "ID", "Test", "Breaks at alpha");
        println!("  {:>3}  {:<24}  {:>12}", "---", "------------------------", "---------------");
        for (ti, break_alpha) in &sensitivity {
            println!("  T{:<2}  {:<24}  {:>12.4}", ti + 1, test_names[*ti], break_alpha);
        }
    }

    let never_fail: Vec<usize> = first_fail.iter().enumerate()
        .filter(|&(_, &a)| a.is_nan())
        .map(|(i, _)| i)
        .collect();
    if !never_fail.is_empty() {
        println!();
        println!("  Tests that never fail:");
        for ti in &never_fail {
            println!("    T{}: {}", ti + 1, test_names[*ti]);
        }
    }

    println!();
}

// =============================================================================
// Main
// =============================================================================

fn main() {
    println!("=== Variant 1: Quantile Mapping vs Linear Baseline ===");
    println!("  Comparing two phase adjustment formulas across the full 23-test suite");
    println!();

    // Run baseline (linear) sweep first for comparison
    sweep(Variant::Linear,
          "LINEAR BASELINE (z-score)",
          "adj = alpha * (mag - mean) / std");

    // Run quantile sweep
    sweep(Variant::Quantile,
          "QUANTILE MAPPING",
          "adj = alpha * (2 * rank_percentile - 1)");

    // Head-to-head comparison
    println!("============================================================");
    println!("  HEAD-TO-HEAD COMPARISON");
    println!("============================================================");
    println!();

    let n_mag = 1000;
    let alphas = [0.100, 0.050, 0.030, 0.020, 0.010, 0.005, 0.003, 0.001];

    println!("  {:>8}  {:>12}  {:>12}  {:>12}  {:>12}",
        "alpha", "Linear Pass", "Linear Gap", "Quantile Pass", "Quantile Gap");
    println!("  {:>8}  {:>12}  {:>12}  {:>12}  {:>12}",
        "--------", "------------", "------------", "-------------", "------------");

    for &alpha in &alphas {
        let (lp, _) = run_all_at_alpha(alpha, n_mag, Variant::Linear);
        let (_, _, lg) = measure_discrimination(alpha, Variant::Linear);
        let (qp, _) = run_all_at_alpha(alpha, n_mag, Variant::Quantile);
        let (_, _, qg) = measure_discrimination(alpha, Variant::Quantile);

        let winner = if qp > lp { " Q wins" }
            else if qp == lp && qg > lg { " Q wins" }
            else if lp > qp { " L wins" }
            else { "" };

        println!("  {:>8.4}  {:>9}/23   {:>12.6}  {:>10}/23   {:>12.6}{}",
            alpha, lp, lg, qp, qg, winner);
    }

    println!();
    println!("  KEY QUESTION: Does quantile achieve 23/23 at a HIGHER alpha");
    println!("  than linear (alpha*=0.001), with a LARGER discrimination gap?");
    println!();
}
