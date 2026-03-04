/// Boundary-Contained Embedded Coherence Test
///
/// Hypothesis: high spherical harmonics create potential wells (proven in
/// boundary_plausibility.rs). Within a well, magnitude CV << global 51.5%.
/// If true, the embedded coherence formula works at higher alpha within wells.
///
/// The test:
///   1. Load trained baseline embeddings (65-token Shakespeare)
///   2. Extract per-band magnitude r = sqrt(cos² + sin²) and phase φ = atan2(sin, cos)
///   3. Map mean magnitude to elevation θ ∈ [0, π]
///   4. Assign wells using Legendre polynomial zeros at various l values
///   5. KEY: Compute within-well CV (if << 51.5%, operating window opens)
///   6. Run embedded coherence within wells (well-local vs global normalization)
///   7. Run 23-test harness within wells for backward compatibility
///   8. Measure discrimination gap
///
/// Two normalization schemes compared:
///   B-WellLocal:  within-well magnitudes, well-local mean/std → z-scores re-expand
///   B-WellGlobal: within-well magnitudes, GLOBAL mean/std → z-scores compressed
///
/// Prediction: B-WellGlobal α* >> B-Global α* because within-well z-score range
/// is much smaller with global normalization.
///
/// Zero dependencies. Pure Rust. Same pattern as existing test files.

use std::f64::consts::PI;
use std::convert::TryInto;

// =============================================================================
// Binary tensor reader (from sweep_v3)
// =============================================================================

fn read_tensor(path: &str) -> Option<(Vec<usize>, Vec<f32>)> {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(_) => return None,
    };
    if bytes.len() < 4 { return None; }
    let ndims = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let mut dims = Vec::with_capacity(ndims);
    let mut offset = 4;
    for _ in 0..ndims {
        if offset + 4 > bytes.len() { return None; }
        dims.push(u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize);
        offset += 4;
    }
    let n_values: usize = dims.iter().product();
    let mut values = Vec::with_capacity(n_values);
    for i in 0..n_values {
        let start = offset + i * 4;
        if start + 4 > bytes.len() { break; }
        values.push(f32::from_le_bytes(bytes[start..start + 4].try_into().unwrap()));
    }
    Some((dims, values))
}

fn find_weight_file() -> Option<(Vec<usize>, Vec<f32>)> {
    let candidates = [
        "experiments/phase17-weight-analysis/weights/baseline/wte.weight.bin",
        "../../../experiments/phase17-weight-analysis/weights/baseline/wte.weight.bin",
        "../../experiments/phase17-weight-analysis/weights/baseline/wte.weight.bin",
        "../experiments/phase17-weight-analysis/weights/baseline/wte.weight.bin",
    ];
    // Check command-line argument first
    if let Some(arg) = std::env::args().nth(1) {
        if let Some(result) = read_tensor(&arg) {
            return Some(result);
        }
    }
    for path in candidates {
        if let Some(result) = read_tensor(path) {
            return Some(result);
        }
    }
    None
}

// =============================================================================
// Phase / magnitude extraction (from sweep_v3)
// =============================================================================

fn extract_phases_and_magnitudes(dims: &[usize], values: &[f32]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let vocab_size = dims[0];
    let embed_dim = dims[1];
    let n_bands = embed_dim / 2;
    let mut phases = vec![vec![0.0f64; vocab_size]; n_bands];
    let mut magnitudes = vec![vec![0.0f64; vocab_size]; n_bands];
    for c in 0..vocab_size {
        for k in 0..n_bands {
            let cos_val = values[c * embed_dim + 2 * k] as f64;
            let sin_val = values[c * embed_dim + 2 * k + 1] as f64;
            let mut angle = sin_val.atan2(cos_val);
            if angle < 0.0 { angle += 2.0 * PI; }
            phases[k][c] = angle;
            magnitudes[k][c] = (cos_val * cos_val + sin_val * sin_val).sqrt();
        }
    }
    (phases, magnitudes)
}

// =============================================================================
// Legendre polynomial (recurrence)
// =============================================================================

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

// =============================================================================
// Find zeros of P_l(x) in (-1, 1) and convert to θ = arccos(x)
// =============================================================================

fn legendre_zeros_theta(l: usize) -> Vec<f64> {
    if l == 0 { return vec![]; }
    let mut zeros_x = Vec::new();
    let steps = 10000;
    let mut prev_x = -0.9999;
    let mut prev_p = legendre(l, prev_x);
    for i in 1..=steps {
        let x = -0.9999 + (i as f64 / steps as f64) * 1.9998;
        let p = legendre(l, x);
        if prev_p * p < 0.0 {
            // Sign change — bisect
            let mut lo = prev_x;
            let mut hi = x;
            for _ in 0..60 {
                let mid = (lo + hi) / 2.0;
                let pm = legendre(l, mid);
                if prev_p * pm < 0.0 { hi = mid; } else { lo = mid; prev_p = pm; }
            }
            zeros_x.push((lo + hi) / 2.0);
            prev_p = p; // reset for next interval
        } else {
            prev_p = p;
        }
        prev_x = x;
    }
    // Convert x → θ = arccos(x), sort ascending
    let mut thetas: Vec<f64> = zeros_x.iter().map(|&x| x.acos()).collect();
    thetas.sort_by(|a, b| a.partial_cmp(b).unwrap());
    thetas
}

// =============================================================================
// Well assignment: given elevations and boundary θ values, assign well index
// =============================================================================

fn assign_wells(elevations: &[f64], boundaries: &[f64]) -> Vec<usize> {
    // boundaries are sorted ascending θ values (Legendre zeros)
    // Well 0: [0, boundaries[0])
    // Well k: [boundaries[k-1], boundaries[k])
    // Well N: [boundaries[N-1], π]
    elevations.iter().map(|&theta| {
        let mut well = 0;
        for &b in boundaries {
            if theta >= b { well += 1; } else { break; }
        }
        well
    }).collect()
}

// =============================================================================
// Statistics helpers
// =============================================================================

fn stat_mean(v: &[f64]) -> f64 {
    if v.is_empty() { return 0.0; }
    v.iter().sum::<f64>() / v.len() as f64
}

fn stat_std(v: &[f64]) -> f64 {
    if v.len() < 2 { return 0.0; }
    let m = stat_mean(v);
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
}

fn stat_cv(v: &[f64]) -> f64 {
    let m = stat_mean(v);
    if m.abs() < 1e-15 { return 0.0; }
    stat_std(v) / m
}

// =============================================================================
// RNG — xorshift64, deterministic (from three_mode_harness)
// =============================================================================

struct Rng { state: u64 }

impl Rng {
    fn new(seed: u64) -> Self { Rng { state: seed } }

    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    fn uniform(&mut self) -> f64 {
        (self.next_u64() & 0x1FFFFFFFFFFFFF) as f64 / (1u64 << 53) as f64
    }
}

// =============================================================================
// Coherence function (from three_mode_harness)
// =============================================================================

fn coherence(phi_a: f64, phi_b: f64, n: u32,
             mag_a: f64, mag_b: f64,
             alpha: f64, r_mean: f64, r_std: f64) -> f64 {
    let adj_a = if r_std > 1e-15 { alpha * (mag_a - r_mean) / r_std } else { 0.0 };
    let adj_b = if r_std > 1e-15 { alpha * (mag_b - r_mean) / r_std } else { 0.0 };
    (n as f64 * ((phi_a + adj_a) - (phi_b + adj_b))).cos()
}

// =============================================================================
// ModeConfig (adapted for dynamic naming and well-based construction)
// =============================================================================

#[derive(Clone)]
struct ModeConfig {
    name: String,
    alpha: f64,
    magnitudes: Vec<f64>,
    r_mean: f64,
    r_std: f64,
}

impl ModeConfig {
    fn circle() -> Self {
        ModeConfig { name: "Circle".into(), alpha: 0.0, magnitudes: vec![], r_mean: 0.0, r_std: 0.0 }
    }

    fn from_mags(name: &str, mags: Vec<f64>, alpha: f64) -> Self {
        let r_mean = stat_mean(&mags);
        let r_std = stat_std(&mags);
        ModeConfig { name: name.into(), alpha, magnitudes: mags, r_mean, r_std }
    }

    fn from_mags_global_norm(name: &str, mags: Vec<f64>, alpha: f64, global_mean: f64, global_std: f64) -> Self {
        ModeConfig { name: name.into(), alpha, magnitudes: mags, r_mean: global_mean, r_std: global_std }
    }

    fn b_varied(n: usize, alpha: f64) -> Self {
        let mut rng = Rng::new(42);
        let mags: Vec<f64> = (0..n).map(|_| 0.108 + rng.uniform() * (1.892 - 0.108)).collect();
        let r_mean = stat_mean(&mags);
        let r_std = stat_std(&mags);
        ModeConfig { name: "B-Varied(synth)".into(), alpha, magnitudes: mags, r_mean, r_std }
    }

    fn mag(&self, i: usize) -> f64 {
        if self.magnitudes.is_empty() { 1.0 } else { self.magnitudes[i % self.magnitudes.len()] }
    }

    fn coh(&self, phi_a: f64, phi_b: f64, n: u32, idx_a: usize, idx_b: usize) -> f64 {
        coherence(phi_a, phi_b, n, self.mag(idx_a), self.mag(idx_b),
                  self.alpha, self.r_mean, self.r_std)
    }
}

fn deg2rad(d: f64) -> f64 { d * PI / 180.0 }

// =============================================================================
// TestResult (generic for N modes)
// =============================================================================

struct TestResult {
    id: u32,
    name: &'static str,
    passes: Vec<bool>,
    structural: bool,
}

impl TestResult {
    fn new(id: u32, name: &'static str, passes: Vec<bool>) -> Self {
        TestResult { id, name, passes, structural: false }
    }

    fn structural(id: u32, name: &'static str, pass: bool, n_modes: usize) -> Self {
        TestResult { id, name, passes: vec![pass; n_modes], structural: true }
    }
}

// =============================================================================
// 23 Tests (faithful copies from three_mode_harness, adapted for N modes)
// =============================================================================

fn test_01(modes: &[ModeConfig]) -> TestResult {
    let n_entities = 12;
    let target_idx = 7;
    let phases: Vec<f64> = (0..n_entities).map(|i| deg2rad(i as f64 * 30.0)).collect();
    let target_phi = phases[target_idx];
    let passes: Vec<bool> = modes.iter().map(|mode| {
        let self_coh = mode.coh(target_phi, target_phi, 1, target_idx, target_idx);
        let matches: usize = (0..n_entities)
            .filter(|&i| mode.coh(target_phi, phases[i], 1, target_idx, i) > 0.99)
            .count();
        (self_coh - 1.0).abs() < 1e-12 && matches == 1
    }).collect();
    TestResult::new(1, "Exact match", passes)
}

fn test_02(modes: &[ModeConfig]) -> TestResult {
    let phases: Vec<f64> = (0..12).map(|i| deg2rad(i as f64 * 30.0)).collect();
    let target = phases[0];
    let passes: Vec<bool> = modes.iter().map(|mode| {
        let c120 = mode.coh(target, phases[4], 3, 0, 4);
        let c240 = mode.coh(target, phases[8], 3, 0, 8);
        let c90 = mode.coh(target, phases[3], 3, 0, 3);
        c120 > 0.95 && c240 > 0.95 && c90 < 0.95
    }).collect();
    TestResult::new(2, "Harmonic family n=3", passes)
}

fn test_03(modes: &[ModeConfig]) -> TestResult {
    let phases: Vec<f64> = (0..12).map(|i| deg2rad(i as f64 * 30.0)).collect();
    let target = phases[0];
    let passes: Vec<bool> = modes.iter().map(|mode| {
        let c180 = mode.coh(target, phases[6], 2, 0, 6);
        let c90 = mode.coh(target, phases[3], 2, 0, 3);
        let matches: Vec<usize> = (0..12)
            .filter(|&i| mode.coh(target, phases[i], 2, 0, i) > 0.95)
            .collect();
        c180 > 0.95 && c90 < 0.95
            && matches.contains(&0) && matches.contains(&6) && matches.len() == 2
    }).collect();
    TestResult::new(3, "Opposition n=2", passes)
}

fn test_04(modes: &[ModeConfig]) -> TestResult {
    let offsets = [0.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 90.0];
    let target = deg2rad(0.0);
    let passes: Vec<bool> = modes.iter().map(|mode| {
        let scores: Vec<f64> = offsets.iter().enumerate()
            .map(|(i, &d)| mode.coh(target, deg2rad(d), 1, 0, i + 1))
            .collect();
        scores.windows(2).all(|w| w[0] >= w[1] - 1e-12)
    }).collect();
    TestResult::new(4, "Fuzzy falloff", passes)
}

fn test_05(modes: &[ModeConfig]) -> TestResult {
    let vendors = [30.0, 30.0, 30.0, 200.0];
    let cats = [120.0, 240.0, 120.0, 120.0];
    let tv = deg2rad(30.0);
    let tc = deg2rad(120.0);
    let passes: Vec<bool> = modes.iter().map(|mode| {
        let scores: Vec<f64> = (0..4).map(|i| {
            mode.coh(tv, deg2rad(vendors[i]), 1, 0, i) * mode.coh(tc, deg2rad(cats[i]), 1, 0, i)
        }).collect();
        scores[0] > 0.99 && scores[2] > 0.99 && scores[0] > scores[1] && scores[0] > scores[3]
    }).collect();
    TestResult::new(5, "Multi-attribute product", passes)
}

fn test_06(n_modes: usize) -> TestResult {
    let size: i32 = 5;
    let chain = |start: usize, step: i32, depth: usize| -> Vec<usize> {
        let mut result = vec![start];
        let mut cur = start as i32;
        for _ in 0..depth {
            cur = (cur + step).rem_euclid(size);
            result.push(cur as usize);
        }
        result
    };
    let path_1 = chain(0, 1, 3);
    let path_2 = chain(0, 2, 3);
    let path_3 = chain(3, 1, 2);
    let path_4 = chain(0, -1, 4);
    let path_5 = chain(0, -2, 4);
    let pass = path_1 == vec![0,1,2,3] && path_2 == vec![0,2,4,1]
        && path_3 == vec![3,4,0] && path_4 == vec![0,4,3,2,1] && path_5 == vec![0,3,1,4,2];
    TestResult::structural(6, "Directed cycle", pass, n_modes)
}

fn test_07(n_modes: usize) -> TestResult {
    let pairs = [(0usize,1usize), (2,11), (3,10), (4,9), (5,8), (6,7)];
    let partner = |pos: usize| -> Option<usize> {
        for &(a, b) in &pairs {
            if a == pos { return Some(b); }
            if b == pos { return Some(a); }
        }
        None
    };
    let pass = partner(0) == Some(1) && partner(2) == Some(11) && partner(3) == Some(10)
        && pairs.iter().all(|&(a,b)| partner(a) == Some(b) && partner(b) == Some(a));
    TestResult::structural(7, "Structural pairs", pass, n_modes)
}

fn test_08(modes: &[ModeConfig]) -> TestResult {
    let bucket_count = 100u64;
    let entity_count = 1000;
    let target_value = 42u64;
    let values: Vec<u64> = (0..entity_count).map(|i| (i * 37 + 13) % bucket_count).collect();
    let phases: Vec<f64> = values.iter().map(|&v| 2.0 * PI * v as f64 / bucket_count as f64).collect();
    let target_phi = 2.0 * PI * target_value as f64 / bucket_count as f64;
    let linear: Vec<usize> = values.iter().enumerate()
        .filter(|&(_, &v)| v == target_value).map(|(i, _)| i).collect();
    let passes: Vec<bool> = modes.iter().map(|mode| {
        let wave: Vec<usize> = (0..entity_count as usize)
            .filter(|&i| mode.coh(target_phi, phases[i], 1, 0, i) > 0.9999)
            .collect();
        linear == wave
    }).collect();
    TestResult::new(8, "Wave = linear scan", passes)
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
    let passes: Vec<bool> = modes.iter().map(|mode| {
        let mut group_counts = [0usize; 4];
        for (i, &phi) in phases.iter().enumerate() {
            if mode.coh(target, phi, 3, 0, i) > 0.85 { group_counts[groups[i]] += 1; }
        }
        group_counts[2] > 0 && group_counts[3] > 0 && group_counts[1] == 0
    }).collect();
    TestResult::new(9, "Harmonic vs JOIN", passes)
}

fn test_10(modes: &[ModeConfig]) -> TestResult {
    let phases: Vec<f64> = (0..12).map(|i| deg2rad(i as f64 * 30.0)).collect();
    let target = phases[0];
    let passes: Vec<bool> = modes.iter().map(|mode| {
        let broad_count = [2usize, 4, 8].iter()
            .filter(|&&i| mode.coh(target, phases[i], 3, 0, i) > 0.95).count();
        let narrow_180 = mode.coh(target, phases[6], 2, 0, 6) > 0.95;
        let narrow_90 = mode.coh(target, phases[3], 2, 0, 3) > 0.95;
        broad_count >= 2 && narrow_180 && !narrow_90
    }).collect();
    TestResult::new(10, "Type-dependent reach", passes)
}

fn test_11(modes: &[ModeConfig]) -> TestResult {
    let pairs = [(5.0, 7.0), (10.0, 11.0), (10.0, 10.1)];
    let passes: Vec<bool> = modes.iter().map(|mode| {
        let mut divergence_ns = Vec::new();
        for &(a_deg, b_deg) in &pairs {
            let a = deg2rad(a_deg);
            let b = deg2rad(b_deg);
            let max_n = if (a_deg - b_deg).abs() < 0.5 { 1800u32 } else { 180 };
            let mut div_n = 0u32;
            for n in 1..=max_n {
                if mode.coh(a, b, n, 0, 1).abs() < 0.9 && div_n == 0 { div_n = n; }
            }
            divergence_ns.push(div_n);
        }
        divergence_ns.iter().all(|&n| n > 0)
            && divergence_ns[1] > divergence_ns[0]
            && divergence_ns[2] > divergence_ns[1]
    }).collect();
    TestResult::new(11, "Fingerprint disambig.", passes)
}

fn test_12(modes: &[ModeConfig]) -> TestResult {
    let a = deg2rad(30.0);
    let b = deg2rad(35.0);
    let passes: Vec<bool> = modes.iter().map(|mode| {
        let base = mode.coh(a, b, 1, 0, 1);
        let mutual = base * 1.5;
        let oneway = base * 1.2;
        let ratio_m = mutual / base;
        let ratio_o = oneway / base;
        mutual > oneway && oneway > base
            && (ratio_m - 1.5).abs() < 0.001 && (ratio_o - 1.2).abs() < 0.001
    }).collect();
    TestResult::new(12, "Mutual amplification", passes)
}

fn test_13(n_modes: usize) -> TestResult {
    let size: i32 = 5;
    let steps = [1i32, 2, -1, -2];
    let mut map = vec![vec![None::<i32>; 5]; 5];
    let mut no_conflicts = true;
    let mut all_assigned = true;
    for &step in &steps {
        for start in 0..5i32 {
            let dest = (start + step).rem_euclid(size) as usize;
            if map[start as usize][dest].is_some() { no_conflicts = false; }
            else { map[start as usize][dest] = Some(step); }
        }
    }
    for a in 0..5 { for b in 0..5 {
        if a != b && map[a][b].is_none() { all_assigned = false; }
    }}
    let mut counts = [0usize; 4];
    for (si, &s) in steps.iter().enumerate() {
        for a in 0..5 { for b in 0..5 {
            if a != b && map[a][b] == Some(s) { counts[si] += 1; }
        }}
    }
    let pass = all_assigned && no_conflicts && counts.iter().all(|&c| c == 5);
    TestResult::structural(13, "Cycle uniqueness", pass, n_modes)
}

fn test_14(modes: &[ModeConfig]) -> TestResult {
    let angles = [0.0, 60.0, 72.0, 90.0, 120.0, 180.0, 240.0, 270.0, 288.0, 300.0];
    let phases: Vec<f64> = angles.iter().map(|&d| deg2rad(d)).collect();
    let target = phases[0];
    let passes: Vec<bool> = modes.iter().map(|mode| {
        let find = |n: u32| -> Vec<usize> {
            (0..phases.len()).filter(|&i| mode.coh(target, phases[i], n, 0, i) > 0.95).collect()
        };
        let h3 = find(3); let h4 = find(4); let h5 = find(5);
        let h3_ok = h3.contains(&0) && h3.contains(&4) && h3.contains(&6)
            && !h3.contains(&3) && !h3.contains(&1);
        let h4_ok = h4.contains(&0) && h4.contains(&3) && h4.contains(&5) && h4.contains(&7)
            && !h4.contains(&4) && !h4.contains(&1);
        let h5_ok = h5.contains(&0) && h5.contains(&2) && h5.contains(&8);
        h3_ok && h4_ok && h5_ok
    }).collect();
    TestResult::new(14, "Harmonic orthogonality", passes)
}

fn test_15(modes: &[ModeConfig]) -> TestResult {
    let angles = [357.0, 358.0, 359.0, 0.0, 1.0, 2.0, 3.0, 180.0];
    let phases: Vec<f64> = angles.iter().map(|&d| deg2rad(d)).collect();
    let target = deg2rad(0.0);
    let passes: Vec<bool> = modes.iter().map(|mode| {
        let c_1_359 = mode.coh(phases[4], phases[2], 1, 4, 2);
        let near: Vec<usize> = (0..phases.len())
            .filter(|&i| mode.coh(target, phases[i], 1, 0, i) > 0.99).collect();
        c_1_359 > 0.95 && !near.contains(&7) && near.contains(&3)
    }).collect();
    TestResult::new(15, "Wraparound", passes)
}

fn test_16(modes: &[ModeConfig]) -> TestResult {
    let n = 360;
    let phases: Vec<f64> = (0..n).map(|i| 2.0 * PI * i as f64 / n as f64).collect();
    let threshold = (1.0 + (2.0 * PI / n as f64).cos()) / 2.0;
    let passes: Vec<bool> = modes.iter().map(|mode| {
        let mut perfect = 0;
        for i in 0..n {
            let matches: Vec<usize> = (0..n)
                .filter(|&j| mode.coh(phases[i], phases[j], 1, i, j) > threshold).collect();
            if matches.len() == 1 && matches[0] == i { perfect += 1; }
        }
        perfect == n
    }).collect();
    TestResult::new(16, "360 resolution", passes)
}

fn test_17(modes: &[ModeConfig]) -> TestResult {
    let golden_angle = 137.50776405003785_f64;
    let scenarios: Vec<(usize, u32)> = vec![(7, 12), (50, 360), (200, 360), (360, 360)];
    let passes: Vec<bool> = modes.iter().map(|mode| {
        let mut exact_results = Vec::new();
        for &(n_obj, buckets) in &scenarios {
            let positions: Vec<f64> = (0..n_obj)
                .map(|i| deg2rad((i as f64 * golden_angle) % 360.0)).collect();
            let exact_threshold = (1.0 + (2.0 * PI / buckets as f64).cos()) / 2.0;
            let mut exact_ok = true;
            for i in 0..n_obj {
                let matches: usize = (0..n_obj)
                    .filter(|&j| mode.coh(positions[i], positions[j], 1, i, j) > exact_threshold).count();
                if matches != 1 { exact_ok = false; break; }
            }
            exact_results.push(exact_ok);
        }
        exact_results[0] && exact_results.iter().any(|&ok| !ok)
    }).collect();
    TestResult::new(17, "Density scaling", passes)
}

fn test_18(n_modes: usize) -> TestResult {
    let buckets = 360u32;
    let test_angles = [0.0, 45.0, 90.0, 137.5, 180.0, 270.0, 359.9];
    let mut pass = true;
    for &deg in &test_angles {
        let phi = deg2rad(deg);
        let bucket = ((phi * buckets as f64 / (2.0 * PI)).floor() as u32) % buckets;
        let expected = (deg.floor() as u32) % buckets;
        if bucket != expected { pass = false; }
    }
    TestResult::structural(18, "Bucket index", pass, n_modes)
}

fn test_19(n_modes: usize) -> TestResult {
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
    TestResult::structural(19, "Torus index", pass, n_modes)
}

fn test_20(n_modes: usize) -> TestResult {
    let golden_angle = 137.50776405003785_f64;
    let n_initial = 50usize;
    let mut entries: Vec<(usize, f64)> = (0..n_initial)
        .map(|i| (i, deg2rad((i as f64 * golden_angle) % 360.0))).collect();
    let remove_ids: Vec<usize> = (0..10).map(|i| i * 5).collect();
    entries.retain(|&(id, _)| !remove_ids.contains(&id));
    let after_remove = entries.len();
    for i in 0..5 {
        entries.push((100 + i, deg2rad((i as f64 * 12.345 + 77.0) % 360.0)));
    }
    let after_insert = entries.len();
    let pass = after_remove == 40 && after_insert == 45
        && entries.iter().all(|&(id, _)| !remove_ids.contains(&id));
    TestResult::structural(20, "Dynamic mutation", pass, n_modes)
}

fn test_21(modes: &[ModeConfig]) -> TestResult {
    let phases: Vec<f64> = [0.0, 120.0, 180.0, 90.0, 60.0, 72.0, 37.0, 143.0]
        .iter().map(|&d| deg2rad(d)).collect();
    let expected: Vec<(usize, usize, u32)> = vec![
        (0,1,3), (0,2,2), (0,3,4), (0,4,6), (0,5,5),
    ];
    let passes: Vec<bool> = modes.iter().map(|mode| {
        let detected = expected.iter()
            .filter(|&&(i,j,n)| mode.coh(phases[i], phases[j], n, i, j) > 0.999).count();
        let noise_clean = [6usize, 7].iter().all(|&j| {
            (1..=6).all(|n| mode.coh(phases[0], phases[j], n as u32, 0, j) <= 0.999)
        });
        detected == 5 && noise_clean
    }).collect();
    TestResult::new(21, "Harmonic sweep", passes)
}

fn test_22(modes: &[ModeConfig]) -> TestResult {
    let test_angles: Vec<f64> = vec![
        0.0, 30.0, 45.0, 60.0, 72.0, 90.0, 120.0, 137.5, 180.0, 210.0, 270.0, 315.0, 359.0,
    ];
    let phases: Vec<f64> = test_angles.iter().map(|&d| deg2rad(d)).collect();
    let harmonics = [1u32, 2, 3, 4, 5, 6, 8, 12];
    let eps = 1e-10;
    let passes: Vec<bool> = modes.iter().map(|mode| {
        let mut ok = true;
        // Symmetry
        for &n in &harmonics { for i in 0..phases.len() { for j in (i+1)..phases.len() {
            let fwd = mode.coh(phases[i], phases[j], n, i, j);
            let rev = mode.coh(phases[j], phases[i], n, j, i);
            if (fwd - rev).abs() > eps { ok = false; }
        }}}
        // Normalization
        for &n in &harmonics { for (i, &phi) in phases.iter().enumerate() {
            if (mode.coh(phi, phi, n, i, i) - 1.0).abs() > eps { ok = false; }
        }}
        // PSD (2x2 and 3x3)
        for &n in &harmonics {
            let sz = phases.len();
            let gram: Vec<Vec<f64>> = (0..sz).map(|i|
                (0..sz).map(|j| mode.coh(phases[i], phases[j], n, i, j)).collect()
            ).collect();
            for i in 0..sz { for j in (i+1)..sz {
                let det2 = gram[i][i]*gram[j][j] - gram[i][j]*gram[j][i];
                if det2 < -eps { ok = false; }
            }}
            for i in 0..sz { for j in (i+1)..sz { for k in (j+1)..sz {
                let det3 =
                    gram[i][i]*(gram[j][j]*gram[k][k] - gram[j][k]*gram[k][j])
                    - gram[i][j]*(gram[j][i]*gram[k][k] - gram[j][k]*gram[k][i])
                    + gram[i][k]*(gram[j][i]*gram[k][j] - gram[j][j]*gram[k][i]);
                if det3 < -eps { ok = false; }
            }}}
        }
        // Spectral scaling
        let test_ns: Vec<u32> = vec![1, 2, 3, 4, 6, 8, 12];
        let mut prev_res: Option<f64> = None;
        for &n in &test_ns {
            let res = 0.95_f64.acos() * 180.0 / PI / n as f64;
            if let Some(prev) = prev_res {
                if res > prev + eps { ok = false; }
            }
            prev_res = Some(res);
        }
        ok
    }).collect();
    TestResult::new(22, "Kernel admissibility", passes)
}

fn test_23(modes: &[ModeConfig]) -> TestResult {
    let groups: Vec<(&str, Vec<f64>, Option<usize>)> = vec![
        ("Triadic", vec![0.0, 120.0, 240.0], Some(3)),
        ("Opposition", vec![0.0, 180.0], Some(2)),
        ("Quadrant", vec![0.0, 90.0, 180.0, 270.0], Some(4)),
        ("Noise", vec![0.0, 37.0, 143.0, 211.0], None),
    ];
    let passes: Vec<bool> = modes.iter().map(|mode| {
        let mut all_correct = true;
        for (_, angles, expected_fund) in &groups {
            let phases: Vec<f64> = angles.iter().map(|&d| deg2rad(d)).collect();
            let mut signed_sum = vec![0.0f64; 12];
            let mut pair_count = 0;
            for i in 0..phases.len() { for j in (i+1)..phases.len() {
                for n in 0..12 {
                    signed_sum[n] += mode.coh(phases[i], phases[j], (n+1) as u32, i, j);
                }
                pair_count += 1;
            }}
            let signed_mean: Vec<f64> = signed_sum.iter().map(|s| s / pair_count as f64).collect();
            let fundamental = signed_mean.iter().position(|&m| m > 0.95).map(|i| i + 1);
            if fundamental != *expected_fund { all_correct = false; }
        }
        all_correct
    }).collect();
    TestResult::new(23, "Channel energy (eta)", passes)
}

// =============================================================================
// Run all 23 tests
// =============================================================================

fn run_all(modes: &[ModeConfig]) -> Vec<TestResult> {
    let nm = modes.len();
    vec![
        test_01(modes), test_02(modes), test_03(modes), test_04(modes),
        test_05(modes), test_06(nm), test_07(nm), test_08(modes),
        test_09(modes), test_10(modes), test_11(modes), test_12(modes),
        test_13(nm), test_14(modes), test_15(modes), test_16(modes),
        test_17(modes), test_18(nm), test_19(nm), test_20(nm),
        test_21(modes), test_22(modes), test_23(modes),
    ]
}

fn count_passes(results: &[TestResult], mode_idx: usize) -> usize {
    results.iter().filter(|r| r.passes.get(mode_idx).copied().unwrap_or(false)).count()
}

// =============================================================================
// Print compact summary table
// =============================================================================

fn print_summary(results: &[TestResult], mode_names: &[&str]) {
    let n = mode_names.len();
    print!("  {:>3}  {:<24}", "ID", "Test");
    for name in mode_names { print!("  {:>12}", name); }
    println!();
    print!("  {:>3}  {:<24}", "---", "------------------------");
    for _ in 0..n { print!("  {:>12}", "------------"); }
    println!();

    for r in results {
        let tag = if r.structural { " [S]" } else { "" };
        print!("  T{:<2}  {:<24}", r.id, format!("{}{}", r.name, tag));
        for mi in 0..n {
            let p = r.passes.get(mi).copied().unwrap_or(false);
            print!("  {:>12}", if p { "PASS" } else { "FAIL" });
        }
        println!();
    }

    print!("  {:>3}  {:<24}", "", "TOTAL");
    for mi in 0..n {
        let c = count_passes(results, mi);
        print!("  {:>12}", format!("{}/23", c));
    }
    println!();
}

// =============================================================================
// Alpha sweep for a given mode constructor
// =============================================================================

fn alpha_sweep_mode(
    make_mode: &dyn Fn(f64) -> ModeConfig,
    name: &str,
) -> (f64, f64) {
    let alphas = [
        0.100, 0.080, 0.060, 0.050, 0.040, 0.030,
        0.025, 0.020, 0.015, 0.010, 0.008, 0.005, 0.003, 0.001,
    ];

    let mut alpha_star = 0.0;
    let mut gap_star = 0.0;

    for &alpha in &alphas {
        let circle = ModeConfig::circle();
        let mode = make_mode(alpha);
        let modes = vec![circle, mode];
        let results = run_all(&modes);
        let pass_count = count_passes(&results, 1);
        let (near, far, gap) = measure_discrimination_mode(&make_mode(alpha));

        let marker = if pass_count == 23 && alpha_star == 0.0 {
            alpha_star = alpha;
            gap_star = gap;
            " <--"
        } else { "" };
        println!("    {:>7.4}  {:>5}/23  {:>10.6}  {:>10.6}  {:>10.6}{}",
            alpha, pass_count, near, far, gap, marker);
    }

    (alpha_star, gap_star)
}

// =============================================================================
// Discrimination measurement (adapted for any ModeConfig)
// =============================================================================

fn measure_discrimination_mode(mode_template: &ModeConfig) -> (f64, f64, f64) {
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

    // Use the mode's normalization but substitute these specific magnitudes
    let r_mean = mode_template.r_mean;
    let r_std = mode_template.r_std;
    let alpha = mode_template.alpha;

    let mut near_scores = Vec::new();
    let mut far_scores = Vec::new();

    for g in 0..n_groups {
        let base = g * per_group;
        for i in 0..per_group {
            for j in (i+1)..per_group {
                let dist = j - i;
                let idx_a = base + i;
                let idx_b = base + j;
                let c = coherence(phases[idx_a], phases[idx_b], 1,
                                  mags[idx_a], mags[idx_b], alpha, r_mean, r_std);
                if dist <= 5 { near_scores.push(c); }
                else if dist >= 15 { far_scores.push(c); }
            }
        }
    }

    let near_mean = stat_mean(&near_scores);
    let far_mean = stat_mean(&far_scores);
    (near_mean, far_mean, near_mean - far_mean)
}

// =============================================================================
// MAIN — The 8-step test
// =============================================================================

fn main() {
    println!("=== Boundary-Contained Embedded Coherence Test ===");
    println!("  Hypothesis: within-well magnitudes have lower effective spread,");
    println!("  enabling higher alpha for the embedded formula.");
    println!();

    // =========================================================================
    // Step 1: Load trained embeddings
    // =========================================================================
    let (dims, values, using_real) = match find_weight_file() {
        Some((d, v)) => {
            println!("  Step 1: Loaded real trained embeddings [{} x {}]", d[0], d[1]);
            (d, v, true)
        }
        None => {
            println!("  Step 1: Weight files not found — generating synthetic embeddings");
            println!("    (Run from repo root or pass weight file path as argument)");
            println!("    Generating 65 tokens x 128 dim with ~51.5% CV to match real data...");
            let vocab = 65;
            let dim = 128;
            let mut rng = Rng::new(12345);
            let mut vals = Vec::with_capacity(vocab * dim);
            for _ in 0..vocab {
                for _ in 0..(dim / 2) {
                    // Random phase + random magnitude [0.1, 1.9]
                    let phi = rng.uniform() * 2.0 * PI;
                    let r = 0.1 + rng.uniform() * 1.8;
                    vals.push((r * phi.cos()) as f32);
                    vals.push((r * phi.sin()) as f32);
                }
            }
            (vec![vocab, dim], vals, false)
        }
    };

    let vocab_size = dims[0];
    let embed_dim = dims[1];
    let n_bands = embed_dim / 2;
    println!("    vocab_size={}, embed_dim={}, n_bands={}, real_data={}",
        vocab_size, embed_dim, n_bands, using_real);
    println!();

    // =========================================================================
    // Step 2: Extract phase and magnitude
    // =========================================================================
    let (_phases, magnitudes) = extract_phases_and_magnitudes(&dims, &values);

    // Compute mean magnitude per token (single scalar for embedded formula)
    let mean_mags: Vec<f64> = (0..vocab_size).map(|c| {
        (0..n_bands).map(|k| magnitudes[k][c]).sum::<f64>() / n_bands as f64
    }).collect();

    let global_mean = stat_mean(&mean_mags);
    let global_std = stat_std(&mean_mags);
    let global_cv = if global_mean > 1e-15 { global_std / global_mean * 100.0 } else { 0.0 };

    println!("  Step 2: Magnitude extraction");
    println!("    Mean magnitude per token: mean={:.4}, std={:.4}, CV={:.1}%",
        global_mean, global_std, global_cv);

    // Also per-band CV
    let band_cvs: Vec<f64> = (0..n_bands).map(|k| stat_cv(&magnitudes[k]) * 100.0).collect();
    let avg_band_cv = stat_mean(&band_cvs);
    println!("    Per-band CV: avg={:.1}%, min={:.1}%, max={:.1}%",
        avg_band_cv,
        band_cvs.iter().cloned().fold(f64::INFINITY, f64::min),
        band_cvs.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
    println!();

    // =========================================================================
    // Step 3: Map mean magnitude to elevation
    // =========================================================================
    let min_mag = mean_mags.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_mag = mean_mags.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mag_range = max_mag - min_mag;
    let elevations: Vec<f64> = mean_mags.iter().map(|&r| {
        if mag_range > 1e-12 { PI * (r - min_mag) / mag_range } else { PI / 2.0 }
    }).collect();

    println!("  Step 3: Elevation mapping");
    println!("    Magnitude range: [{:.4}, {:.4}] (span={:.4})", min_mag, max_mag, mag_range);
    println!("    Elevation range: [0, pi] = [0°, 180°]");
    println!();

    // =========================================================================
    // Step 4: Assign wells at various l values
    // =========================================================================
    println!("--- Step 4: Well Assignment ---");
    println!();

    let l_values = [2, 3, 4, 6, 8];

    // Store well info for later steps
    struct WellInfo {
        l: usize,
        n_wells: usize,
        assignments: Vec<usize>,
        well_sizes: Vec<usize>,
        well_mags: Vec<Vec<f64>>,  // magnitudes per well
        largest_well_idx: usize,
    }

    let mut well_infos: Vec<WellInfo> = Vec::new();

    for &l in &l_values {
        let zeros = legendre_zeros_theta(l);
        let assignments = assign_wells(&elevations, &zeros);
        let n_wells = zeros.len() + 1;

        // Collect per-well magnitudes and sizes
        let mut well_sizes = vec![0usize; n_wells];
        let mut well_mags: Vec<Vec<f64>> = vec![vec![]; n_wells];
        for (tok, &well) in assignments.iter().enumerate() {
            if well < n_wells {
                well_sizes[well] += 1;
                well_mags[well].push(mean_mags[tok]);
            }
        }

        let largest_well_idx = well_sizes.iter().enumerate()
            .max_by_key(|&(_, &s)| s).map(|(i, _)| i).unwrap_or(0);

        let zeros_deg: Vec<String> = zeros.iter().map(|&t| format!("{:.1}°", t * 180.0 / PI)).collect();
        println!("  l={}: {} wells, boundaries at [{}]", l, n_wells, zeros_deg.join(", "));
        print!("    Sizes: ");
        for (wi, &sz) in well_sizes.iter().enumerate() {
            let marker = if wi == largest_well_idx { "*" } else { "" };
            print!("W{}={}{} ", wi, sz, marker);
        }
        println!("  (* = largest)");

        well_infos.push(WellInfo { l, n_wells, assignments, well_sizes, well_mags, largest_well_idx });
    }

    println!();

    // =========================================================================
    // Step 5: KEY MEASUREMENT — Within-well CV
    // =========================================================================
    println!("--- Step 5: Within-Well CV (KEY MEASUREMENT) ---");
    println!();
    println!("  Global CV = {:.1}% (the 51.5% wall)", global_cv);
    println!();
    println!("  {:>4}  {:>6}  {:>8}  {:>10}  {:>10}  {:>12}  {:>12}",
        "l", "Well", "Size", "CV(%)", "Mag Range", "Z-Range(Glb)", "Z-Range(Loc)");
    println!("  {:>4}  {:>6}  {:>8}  {:>10}  {:>10}  {:>12}  {:>12}",
        "----", "------", "--------", "----------", "----------", "------------", "------------");

    for info in &well_infos {
        for wi in 0..info.n_wells {
            let mags = &info.well_mags[wi];
            if mags.len() < 2 {
                println!("  {:>4}  {:>6}  {:>8}  {:>10}  {:>10}  {:>12}  {:>12}",
                    info.l, format!("W{}", wi), mags.len(), "n/a", "n/a", "n/a", "n/a");
                continue;
            }
            let w_cv = stat_cv(mags) * 100.0;
            let w_min = mags.iter().cloned().fold(f64::INFINITY, f64::min);
            let w_max = mags.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let w_range = w_max - w_min;

            // Z-score range with GLOBAL normalization
            let z_min_global = if global_std > 1e-15 { (w_min - global_mean) / global_std } else { 0.0 };
            let z_max_global = if global_std > 1e-15 { (w_max - global_mean) / global_std } else { 0.0 };
            let z_range_global = z_max_global - z_min_global;

            // Z-score range with LOCAL normalization
            let local_std = stat_std(mags);
            let local_mean = stat_mean(mags);
            let z_range_local = if local_std > 1e-15 {
                let z_min_l = (w_min - local_mean) / local_std;
                let z_max_l = (w_max - local_mean) / local_std;
                z_max_l - z_min_l
            } else { 0.0 };

            let marker = if wi == info.largest_well_idx { " *" } else { "" };
            println!("  {:>4}  {:>6}  {:>8}  {:>10.1}  {:>10.4}  {:>12.3}  {:>12.3}{}",
                info.l, format!("W{}", wi), mags.len(), w_cv, w_range,
                z_range_global, z_range_local, marker);
        }
        println!();
    }

    // Summary: average within-well CV per l value
    println!("  Summary (wells with >= 3 tokens):");
    println!("  {:>4}  {:>12}  {:>12}  {:>14}  {:>14}",
        "l", "Avg CV(%)", "Wt Avg CV(%)", "Avg Z-Rng(Glb)", "Avg Z-Rng(Loc)");
    println!("  {:>4}  {:>12}  {:>12}  {:>14}  {:>14}",
        "----", "------------", "------------", "--------------", "--------------");

    for info in &well_infos {
        let mut cvs = Vec::new();
        let mut weighted_cv_num = 0.0;
        let mut weighted_cv_den = 0.0;
        let mut z_globals = Vec::new();
        let mut z_locals = Vec::new();

        for wi in 0..info.n_wells {
            let mags = &info.well_mags[wi];
            if mags.len() < 3 { continue; }
            let w_cv = stat_cv(mags) * 100.0;
            cvs.push(w_cv);
            weighted_cv_num += w_cv * mags.len() as f64;
            weighted_cv_den += mags.len() as f64;

            let w_min = mags.iter().cloned().fold(f64::INFINITY, f64::min);
            let w_max = mags.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            if global_std > 1e-15 {
                z_globals.push((w_max - w_min) / global_std);
            }
            let ls = stat_std(mags);
            let lm = stat_mean(mags);
            if ls > 1e-15 {
                let z_min_l = (w_min - lm) / ls;
                let z_max_l = (w_max - lm) / ls;
                z_locals.push(z_max_l - z_min_l);
            }
        }

        let avg_cv = stat_mean(&cvs);
        let wt_cv = if weighted_cv_den > 0.0 { weighted_cv_num / weighted_cv_den } else { 0.0 };
        println!("  {:>4}  {:>12.1}  {:>12.1}  {:>14.3}  {:>14.3}",
            info.l, avg_cv, wt_cv, stat_mean(&z_globals), stat_mean(&z_locals));
    }

    // The critical comparison
    println!();
    let global_z_range = {
        let z_min = if global_std > 1e-15 { (min_mag - global_mean) / global_std } else { 0.0 };
        let z_max = if global_std > 1e-15 { (max_mag - global_mean) / global_std } else { 0.0 };
        z_max - z_min
    };
    println!("  GLOBAL z-score range: {:.3}", global_z_range);
    println!("  At alpha=0.1: max pair delta = {:.4} rad = {:.2} deg",
        0.1 * global_z_range, 0.1 * global_z_range * 180.0 / PI);
    println!("  T16 threshold delta ≈ 0.003 rad = 0.17 deg");
    println!("  → alpha* ≈ 0.003 / {:.3} = {:.4} (matches known result)",
        global_z_range, 0.003 / global_z_range);
    println!();

    // =========================================================================
    // Steps 6-7: 23-test harness — global vs within-well modes
    // =========================================================================
    println!("--- Steps 6-7: 23-Test Harness Comparison ---");
    println!();

    // Baseline: synthetic B-Varied (replicating three_mode_harness result)
    {
        println!("  [Baseline] Synthetic B-Varied (U[0.108,1.892], CV~51.5%):");
        let circle = ModeConfig::circle();
        let bv = ModeConfig::b_varied(1000, 0.1);
        let modes = vec![circle, bv];
        let results = run_all(&modes);
        println!("    Circle: {}/23, B-Varied: {}/23",
            count_passes(&results, 0), count_passes(&results, 1));
        println!();
    }

    // Global mode: real magnitudes, global normalization
    {
        println!("  [B-Global] Real magnitudes, global normalization, alpha=0.1:");
        let circle = ModeConfig::circle();
        let b_global = ModeConfig::from_mags("B-Global", mean_mags.clone(), 0.1);
        let modes = vec![circle, b_global];
        let results = run_all(&modes);
        let fails: Vec<u32> = results.iter()
            .filter(|r| !r.passes.get(1).copied().unwrap_or(true))
            .map(|r| r.id).collect();
        println!("    Circle: {}/23, B-Global: {}/23",
            count_passes(&results, 0), count_passes(&results, 1));
        if !fails.is_empty() {
            let fail_str: Vec<String> = fails.iter().map(|id| format!("T{}", id)).collect();
            println!("    Failures: {}", fail_str.join(", "));
        }
        println!();
    }

    // Per l-value: largest well, well-local and well-global normalization
    println!("  [Within-Well Modes] Largest well per l, alpha=0.1:");
    println!("  {:>4}  {:>6}  {:>8}  {:>14}  {:>14}  {:>14}",
        "l", "Well", "Size", "B-WellLocal", "B-WellGlobal", "B-Varied(syn)");
    println!("  {:>4}  {:>6}  {:>8}  {:>14}  {:>14}  {:>14}",
        "----", "------", "--------", "--------------", "--------------", "--------------");

    for info in &well_infos {
        let wi = info.largest_well_idx;
        let well_m = info.well_mags[wi].clone();
        if well_m.len() < 3 { continue; }

        let circle = ModeConfig::circle();

        // Well-local normalization
        let b_wl = ModeConfig::from_mags(
            &format!("WellLocal(l={},W{})", info.l, wi), well_m.clone(), 0.1);
        let modes_wl = vec![circle.clone(), b_wl];
        let results_wl = run_all(&modes_wl);
        let pass_wl = count_passes(&results_wl, 1);

        // Well-global normalization
        let b_wg = ModeConfig::from_mags_global_norm(
            &format!("WellGlob(l={},W{})", info.l, wi), well_m.clone(), 0.1,
            global_mean, global_std);
        let modes_wg = vec![circle.clone(), b_wg];
        let results_wg = run_all(&modes_wg);
        let pass_wg = count_passes(&results_wg, 1);

        // Synthetic B-Varied for comparison
        let bv = ModeConfig::b_varied(well_m.len(), 0.1);
        let modes_bv = vec![circle.clone(), bv];
        let results_bv = run_all(&modes_bv);
        let pass_bv = count_passes(&results_bv, 1);

        println!("  {:>4}  {:>6}  {:>8}  {:>11}/23   {:>11}/23   {:>11}/23",
            info.l, format!("W{}", wi), well_m.len(), pass_wl, pass_wg, pass_bv);
    }

    println!();

    // =========================================================================
    // Step 8: Alpha sweep — the operating point
    // =========================================================================
    println!("--- Step 8: Alpha Sweep ---");
    println!();
    println!("  {:>7}  {:>7}  {:>10}  {:>10}  {:>10}", "alpha", "pass", "near", "far", "gap");
    println!("  {:>7}  {:>7}  {:>10}  {:>10}  {:>10}", "-------", "-------", "----------", "----------", "----------");

    // Sweep 1: B-Global (all real magnitudes, global normalization)
    println!();
    println!("  [B-Global] All real magnitudes, global normalization:");
    let (a_star_global, gap_global) = alpha_sweep_mode(
        &|alpha| ModeConfig::from_mags("B-Global", mean_mags.clone(), alpha),
        "B-Global",
    );

    // Sweep 2: B-WellGlobal — for each l, use largest well magnitudes with global norm
    // Pick the l value whose largest well has the most tokens
    let best_l_info = well_infos.iter().max_by_key(|info| {
        info.well_sizes[info.largest_well_idx]
    });

    if let Some(info) = best_l_info {
        let wi = info.largest_well_idx;
        let well_m = info.well_mags[wi].clone();
        if well_m.len() >= 3 {
            println!();
            println!("  [B-WellLocal] l={}, W{} ({} tokens), well-local normalization:",
                info.l, wi, well_m.len());
            let wm = well_m.clone();
            let (a_star_wl, gap_wl) = alpha_sweep_mode(
                &|alpha| ModeConfig::from_mags(
                    &format!("WellLocal"), wm.clone(), alpha),
                "B-WellLocal",
            );

            println!();
            println!("  [B-WellGlobal] l={}, W{} ({} tokens), GLOBAL normalization:",
                info.l, wi, well_m.len());
            let wm2 = well_m.clone();
            let gm = global_mean;
            let gs = global_std;
            let (a_star_wg, gap_wg) = alpha_sweep_mode(
                &|alpha| ModeConfig::from_mags_global_norm(
                    &format!("WellGlob"), wm2.clone(), alpha, gm, gs),
                "B-WellGlobal",
            );

            // =========================================================================
            // VERDICT
            // =========================================================================
            println!();
            println!("============================================================");
            println!("  VERDICT");
            println!("============================================================");
            println!();
            println!("  B-Global:      alpha* = {:.4}, discrimination gap = {:.6}", a_star_global, gap_global);
            println!("  B-WellLocal:   alpha* = {:.4}, discrimination gap = {:.6}", a_star_wl, gap_wl);
            println!("  B-WellGlobal:  alpha* = {:.4}, discrimination gap = {:.6}", a_star_wg, gap_wg);
            println!();

            let wg_improvement = if a_star_global > 0.0 { a_star_wg / a_star_global } else { 0.0 };
            let wl_improvement = if a_star_global > 0.0 { a_star_wl / a_star_global } else { 0.0 };
            println!("  WellGlobal improvement: {:.1}x alpha (vs global)", wg_improvement);
            println!("  WellLocal improvement:  {:.1}x alpha (vs global)", wl_improvement);
            println!();

            if a_star_wg > a_star_global * 3.0 {
                println!("  HYPOTHESIS CONFIRMED: Well containment + global normalization");
                println!("  opens the operating window. Within-well z-score range is small");
                println!("  enough that T16 passes at {:.1}x higher alpha.", wg_improvement);
                println!();
                if gap_wg > 0.01 {
                    println!("  AND discrimination is meaningful ({:.6}).", gap_wg);
                    println!("  The embedded formula provides real ranking power within wells.");
                } else if gap_wg > gap_global * 5.0 {
                    println!("  Discrimination improved {:.1}x but still small ({:.6}).",
                        if gap_global > 0.0 { gap_wg / gap_global } else { 0.0 }, gap_wg);
                    println!("  The window opened but the signal may be too weak.");
                } else {
                    println!("  But discrimination is still near zero ({:.6}).", gap_wg);
                    println!("  Higher alpha doesn't buy ranking power here.");
                }
            } else if a_star_wl > a_star_global * 1.5 {
                println!("  PARTIAL: Well-local normalization helps modestly.");
                println!("  The well structure provides some containment, but z-score");
                println!("  re-expansion limits the improvement.");
            } else {
                println!("  HYPOTHESIS FAILS: Well containment does not meaningfully");
                println!("  change the operating point. The T16 wall is structural");
                println!("  regardless of magnitude grouping.");
            }

            println!();
            println!("  Interpretation:");
            println!("    B-WellLocal uses within-well mean/std → z-scores re-expand");
            println!("    B-WellGlobal uses global mean/std → z-scores compressed");
            println!("    If WellGlobal >> WellLocal: the benefit is from containment,");
            println!("    not from different normalization.");
            println!();

        } else {
            println!("  Largest well too small for meaningful sweep.");
        }
    }

    println!("  Done.");
}
