// Wave Coherence Experiments -- Pure Rust
//
// Cross-language validation of the mathematical foundations.
// No GPU, no external dependencies, no neural networks.
// Same math, same results, different language.
//
// Ports the math-only portions of:
//   Phase 16: Wave Packet Engine (DFT, resonance, selective loading)
//   Phase 4:  Harmonic Construction (interpolation, chimera, fractional)
//   Phase 5:  Musical Interval Theory (Tenney height, consonance)

mod construction;
mod dft;
mod harmonic_embed;
mod linalg;
mod musical;
mod wave_packet;

use construction::{chimera, fractional_embedding, interpolate};
use dft::{rfft, irfft};
use harmonic_embed::{deg_to_rad, harmonic_embedding_cos, harmonic_embedding_cossin};
use linalg::{cosine_similarity, correlation, magnitude};
use musical::{consonance_score, identify_interval, tenney_height};
use wave_packet::{
    bands_to_embed, embed_to_bands, make_wave_packet, resonance, select_by_amplitude,
    selective_load,
};

// ============================================================
// Test data generation
// ============================================================

/// Generate test embeddings: 5 groups of 5 angles, tightly clustered within,
/// well-separated between. Uses cosine-only embeddings matching sweep-test.
fn generate_test_data(n_harmonics: usize) -> (Vec<Vec<f64>>, Vec<&'static str>) {
    let groups: &[(&str, &[f64])] = &[
        ("royalty", &[10.0, 12.0, 14.0, 8.0, 16.0]),
        ("animals", &[80.0, 82.0, 84.0, 78.0, 86.0]),
        ("emotions", &[150.0, 152.0, 154.0, 148.0, 156.0]),
        ("tech", &[220.0, 222.0, 224.0, 218.0, 226.0]),
        ("nature", &[290.0, 292.0, 294.0, 288.0, 296.0]),
    ];

    let mut embeddings = Vec::new();
    let mut labels = Vec::new();

    for &(label, angles) in groups {
        for &deg in angles {
            let theta = deg_to_rad(deg);
            embeddings.push(harmonic_embedding_cos(theta, n_harmonics));
            labels.push(label);
        }
    }

    (embeddings, labels)
}

/// Query embeddings: one per group, at the cluster center.
fn generate_queries(n_harmonics: usize) -> (Vec<Vec<f64>>, Vec<&'static str>) {
    let queries: &[(&str, f64)] = &[
        ("royalty", 11.0),
        ("animals", 81.0),
        ("emotions", 151.0),
        ("tech", 221.0),
        ("nature", 291.0),
    ];

    let mut embeddings = Vec::new();
    let mut labels = Vec::new();

    for &(label, deg) in queries {
        let theta = deg_to_rad(deg);
        embeddings.push(harmonic_embedding_cos(theta, n_harmonics));
        labels.push(label);
    }

    (embeddings, labels)
}

// ============================================================
// Phase 16: Wave Packet Engine (6 tests)
// ============================================================

fn test_dft_round_trip() -> bool {
    println!("--- Test 1: DFT Round-Trip ---");
    println!("  embed -> rfft -> irfft -> reconstruct, verify max error < 1e-6");

    let mut pass = true;

    // Anchor: hand-computed DFT of [1.0, 0.0, -1.0, 0.0]
    let simple = vec![1.0, 0.0, -1.0, 0.0];
    let coeffs = rfft(&simple);
    // Expected: X[0] = 0, X[1] = 2+0j, X[2] = 0
    let anchor_ok = coeffs[0].abs() < 1e-10
        && (coeffs[1].re - 2.0).abs() < 1e-10
        && coeffs[1].im.abs() < 1e-10
        && coeffs[2].abs() < 1e-10;
    println!(
        "  Anchor [1,0,-1,0]: X[0]={:.6}, X[1]={:.6}+{:.6}j, X[2]={:.6}  {}",
        coeffs[0].abs(),
        coeffs[1].re,
        coeffs[1].im,
        coeffs[2].abs(),
        if anchor_ok { "OK" } else { "FAIL" }
    );
    if !anchor_ok {
        pass = false;
    }

    // Round-trip anchor
    let reconstructed = irfft(&coeffs, simple.len());
    let anchor_rt_error: f64 = simple
        .iter()
        .zip(reconstructed.iter())
        .map(|(a, b): (&f64, &f64)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    println!(
        "  Anchor round-trip error: {:.2e}  {}",
        anchor_rt_error,
        if anchor_rt_error < 1e-10 { "OK" } else { "FAIL" }
    );
    if anchor_rt_error > 1e-6 {
        pass = false;
    }

    // Round-trip 25 harmonic embeddings at different angles
    let n_harmonics = 64;
    let mut max_error = 0.0_f64;
    for i in 0..25 {
        let theta = deg_to_rad(i as f64 * 14.4); // spread across 360 degrees
        let original = harmonic_embedding_cos(theta, n_harmonics);
        let (coeffs, _, _) = embed_to_bands(&original);
        let reconstructed = bands_to_embed(&coeffs, original.len());
        let error: f64 = original
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b): (&f64, &f64)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        max_error = max_error.max(error);
    }
    println!(
        "  25 embeddings (64-dim) max round-trip error: {:.2e}",
        max_error
    );

    let rt_ok = max_error < 1e-6;
    println!(
        "  Perfect reconstruction: {}",
        if rt_ok { "YES" } else { "NO" }
    );
    if !rt_ok {
        pass = false;
    }

    println!(
        "  RESULT: {}\n",
        if pass { "PASS" } else { "FAIL" }
    );
    pass
}

fn test_resonance_vs_cosine() -> bool {
    println!("--- Test 2: Resonance vs Cosine Similarity ---");
    println!("  Full-band resonance must equal cosine similarity (mathematical identity)");

    let n_harmonics = 32;
    let (db, _) = generate_test_data(n_harmonics);

    // Decompose all
    let decomposed: Vec<_> = db.iter().map(|v| embed_to_bands(v)).collect();

    let mut max_diff = 0.0_f64;
    let mut n_pairs = 0;

    let n_coeffs = decomposed[0].1.len();

    for i in 0..db.len() {
        for j in (i + 1)..db.len() {
            let cos_sim = cosine_similarity(&db[i], &db[j]);
            let all_bands: Vec<usize> = (0..n_coeffs).collect();
            let packet = make_wave_packet(&decomposed[i].1, &decomposed[i].2, &all_bands);
            let res = resonance(&packet, &decomposed[j].1, &decomposed[j].2, n_coeffs);
            let diff = (cos_sim - res).abs();
            max_diff = max_diff.max(diff);
            n_pairs += 1;
        }
    }

    println!("  Tested {} pairs", n_pairs);
    println!("  Max |cosine - resonance|: {:.2e}", max_diff);

    let pass = max_diff < 1e-8;
    println!(
        "  Identity holds: {}",
        if pass { "YES" } else { "NO" }
    );
    println!(
        "  RESULT: {}\n",
        if pass { "PASS" } else { "FAIL" }
    );
    pass
}

fn test_wave_packet_retrieval() -> bool {
    println!("--- Test 3: Wave Packet Retrieval ---");
    println!("  5 queries against 25 items, top-1 must land in correct group");

    let n_harmonics = 32;
    let (db, db_labels) = generate_test_data(n_harmonics);
    let (queries, q_labels) = generate_queries(n_harmonics);

    let db_decomposed: Vec<_> = db.iter().map(|v| embed_to_bands(v)).collect();
    let n_bands = db_decomposed[0].1.len();

    // Band sets
    let all_bands: Vec<usize> = (0..n_bands).collect();
    let mid_start = n_bands / 4;
    let high_start = 3 * n_bands / 4;
    let mid_high_bands: Vec<usize> = (mid_start..n_bands).collect();
    let high_bands: Vec<usize> = (high_start..n_bands).collect();

    let strategies: &[(&str, &[usize])] = &[
        // We'll handle these inline since we can't store Vec references easily
    ];
    let _ = strategies; // unused, we iterate manually

    let mut all_correct = 0;
    let mut mh_correct = 0;
    let mut high_correct = 0;

    println!(
        "  {:12}  {:10}  {:12}  {:12}  {:12}",
        "Query", "Expected", "All bands", "Mid+High", "High only"
    );

    for (qi, q_emb) in queries.iter().enumerate() {
        let (_, q_amps, q_phases) = embed_to_bands(q_emb);

        // All bands
        let pkt_all = make_wave_packet(&q_amps, &q_phases, &all_bands);
        let best_all = (0..db.len())
            .max_by(|&a, &b| {
                let ra = resonance(&pkt_all, &db_decomposed[a].1, &db_decomposed[a].2, n_bands);
                let rb = resonance(&pkt_all, &db_decomposed[b].1, &db_decomposed[b].2, n_bands);
                ra.partial_cmp(&rb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();
        let all_ok = db_labels[best_all] == q_labels[qi];
        if all_ok {
            all_correct += 1;
        }

        // Mid+High
        let pkt_mh = make_wave_packet(&q_amps, &q_phases, &mid_high_bands);
        let best_mh = (0..db.len())
            .max_by(|&a, &b| {
                let ra = resonance(&pkt_mh, &db_decomposed[a].1, &db_decomposed[a].2, n_bands);
                let rb = resonance(&pkt_mh, &db_decomposed[b].1, &db_decomposed[b].2, n_bands);
                ra.partial_cmp(&rb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();
        let mh_ok = db_labels[best_mh] == q_labels[qi];
        if mh_ok {
            mh_correct += 1;
        }

        // High only
        let pkt_high = make_wave_packet(&q_amps, &q_phases, &high_bands);
        let best_high = (0..db.len())
            .max_by(|&a, &b| {
                let ra = resonance(&pkt_high, &db_decomposed[a].1, &db_decomposed[a].2, n_bands);
                let rb = resonance(&pkt_high, &db_decomposed[b].1, &db_decomposed[b].2, n_bands);
                ra.partial_cmp(&rb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();
        let high_ok = db_labels[best_high] == q_labels[qi];
        if high_ok {
            high_correct += 1;
        }

        println!(
            "  {:12}  {:10}  {:12}  {:12}  {:12}",
            q_labels[qi],
            q_labels[qi],
            if all_ok { "OK" } else { "MISS" },
            if mh_ok { "OK" } else { "MISS" },
            if high_ok { "OK" } else { "MISS" },
        );
    }

    println!("  Correct:      {:>5}/5       {:>5}/5       {:>5}/5",
        all_correct, mh_correct, high_correct);

    // All-bands retrieval must be 5/5. Others may degrade.
    let pass = all_correct == 5;
    println!(
        "  RESULT: {}\n",
        if pass { "PASS" } else { "FAIL" }
    );
    pass
}

fn test_selective_band_loading() -> bool {
    println!("--- Test 4: Selective Band Loading ---");
    println!("  Partial reconstruction quality by band loading strategy");

    let n_harmonics = 32;
    let (db, _) = generate_test_data(n_harmonics);

    let db_decomposed: Vec<_> = db.iter().map(|v| embed_to_bands(v)).collect();
    let n_bands = db_decomposed[0].1.len();
    let dim = db[0].len();

    let all_bands: Vec<usize> = (0..n_bands).collect();
    let mid_start = n_bands / 4;
    let high_start = 3 * n_bands / 4;
    let mid_high: Vec<usize> = (mid_start..n_bands).collect();
    let high_only: Vec<usize> = (high_start..n_bands).collect();

    let configs: &[(&str, &[usize], f64)] = &[
        ("All bands", &all_bands, 0.999),
        ("Mid+High", &mid_high, 0.5),
        ("High only", &high_only, 0.0), // no minimum for high-only
    ];

    println!(
        "  {:25}  {:>8}  {:>8}  {:>12}  {:>6}",
        "Strategy", "Avg sim", "Min sim", "Bands loaded", "RAM %"
    );

    let mut pass = true;

    for &(name, bands, min_avg) in configs {
        let mut sims = Vec::new();
        for i in 0..db.len() {
            let partial = selective_load(&db_decomposed[i].0, bands, dim);
            let sim = cosine_similarity(&db[i], &partial);
            sims.push(sim);
        }
        let avg_sim: f64 = sims.iter().sum::<f64>() / sims.len() as f64;
        let min_sim: f64 = sims.iter().cloned().fold(f64::INFINITY, f64::min);
        let ram_pct = 100.0 * bands.len() as f64 / n_bands as f64;

        println!(
            "  {:25}  {:8.4}  {:8.4}  {:>12}  {:5.1}%",
            name,
            avg_sim,
            min_sim,
            bands.len(),
            ram_pct
        );

        if avg_sim < min_avg {
            pass = false;
        }
    }

    // Also test amplitude-selected
    let mut amp_sims = Vec::new();
    for i in 0..db.len() {
        let amp_bands = select_by_amplitude(&db_decomposed[i].1, 75.0);
        let partial = selective_load(&db_decomposed[i].0, &amp_bands, dim);
        let sim = cosine_similarity(&db[i], &partial);
        amp_sims.push(sim);
    }
    let avg_amp_sim: f64 = amp_sims.iter().sum::<f64>() / amp_sims.len() as f64;
    let min_amp_sim: f64 = amp_sims.iter().cloned().fold(f64::INFINITY, f64::min);
    println!(
        "  {:25}  {:8.4}  {:8.4}  {:>12}  {:5.1}%",
        "Top 25% by amplitude",
        avg_amp_sim,
        min_amp_sim,
        "~25%",
        25.0
    );

    println!(
        "  RESULT: {}\n",
        if pass { "PASS" } else { "FAIL" }
    );
    pass
}

fn test_amplitude_band_selection() -> bool {
    println!("--- Test 5: Amplitude Band Selection ---");
    println!("  Top-25% amplitude bands should capture majority of signal energy");

    let n_harmonics = 64;
    let (db, _) = generate_test_data(n_harmonics);

    let mut total_energy_ratio = 0.0;
    let mut count = 0;

    for emb in &db {
        let (_, amps, _) = embed_to_bands(emb);

        let total_energy: f64 = amps.iter().map(|a| a * a).sum();
        if total_energy < 1e-10 {
            continue;
        }

        let selected = select_by_amplitude(&amps, 75.0);
        let selected_energy: f64 = selected.iter().map(|&n| amps[n] * amps[n]).sum();

        total_energy_ratio += selected_energy / total_energy;
        count += 1;
    }

    let avg_ratio = total_energy_ratio / count as f64;
    println!(
        "  Average energy captured by top-25% bands: {:.1}%",
        avg_ratio * 100.0
    );

    let pass = avg_ratio > 0.5; // top-25% should capture >50% of energy
    println!(
        "  Majority captured: {}",
        if pass { "YES" } else { "NO" }
    );
    println!(
        "  RESULT: {}\n",
        if pass { "PASS" } else { "FAIL" }
    );
    pass
}

fn test_band_energy_distribution() -> bool {
    println!("--- Test 6: Band Energy Distribution ---");
    println!("  Different angles should produce different energy profiles");

    let n_harmonics = 32;
    // Generate embeddings at very different angles
    let angles = [0.0, 45.0, 90.0, 180.0, 270.0];
    let mut profiles: Vec<(f64, f64, f64)> = Vec::new(); // (low%, mid%, high%)

    for &deg in &angles {
        let theta = deg_to_rad(deg);
        let emb = harmonic_embedding_cos(theta, n_harmonics);
        let (_, amps, _) = embed_to_bands(&emb);

        let n_bands = amps.len();
        let quarter = n_bands / 4;

        let low_e: f64 = amps[..quarter].iter().map(|a| a * a).sum();
        let mid_e: f64 = amps[quarter..3 * quarter].iter().map(|a| a * a).sum();
        let high_e: f64 = amps[3 * quarter..].iter().map(|a| a * a).sum();
        let total = low_e + mid_e + high_e;

        if total > 1e-10 {
            profiles.push((
                100.0 * low_e / total,
                100.0 * mid_e / total,
                100.0 * high_e / total,
            ));
            let dominant = if low_e > mid_e && low_e > high_e {
                "low"
            } else if mid_e > high_e {
                "mid"
            } else {
                "high"
            };
            println!(
                "  {:6.1} deg:  low {:5.1}%  mid {:5.1}%  high {:5.1}%  dominant: {}",
                deg,
                100.0 * low_e / total,
                100.0 * mid_e / total,
                100.0 * high_e / total,
                dominant
            );
        }
    }

    // Profiles should differ: check that not all identical
    let mut all_same = true;
    if profiles.len() >= 2 {
        for i in 1..profiles.len() {
            let diff = (profiles[0].0 - profiles[i].0).abs()
                + (profiles[0].1 - profiles[i].1).abs()
                + (profiles[0].2 - profiles[i].2).abs();
            if diff > 1.0 {
                all_same = false;
                break;
            }
        }
    }

    let pass = !all_same;
    println!(
        "  Profiles vary: {}",
        if pass { "YES" } else { "NO" }
    );
    println!(
        "  RESULT: {}\n",
        if pass { "PASS" } else { "FAIL" }
    );
    pass
}

// ============================================================
// Phase 4: Harmonic Construction (4 tests)
// ============================================================

fn test_interpolation_monotonicity() -> bool {
    println!("--- Test 7: Interpolation Monotonicity ---");
    println!("  Cosine similarity to source A should increase with alpha");

    let n_harmonics = 32;
    let theta_a = deg_to_rad(30.0);
    let theta_b = deg_to_rad(150.0);
    let vec_a = harmonic_embedding_cossin(theta_a, n_harmonics);
    let vec_b = harmonic_embedding_cossin(theta_b, n_harmonics);

    let alphas: Vec<f64> = (0..=10).map(|i| i as f64 / 10.0).collect();
    let mut sims_to_a = Vec::new();
    let mut sims_to_b = Vec::new();

    println!(
        "  {:>5}  {:>12}  {:>12}",
        "alpha", "sim to A", "sim to B"
    );

    for &alpha in &alphas {
        let blended = interpolate(&vec_a, &vec_b, alpha);
        let sim_a = cosine_similarity(&blended, &vec_a);
        let sim_b = cosine_similarity(&blended, &vec_b);
        sims_to_a.push(sim_a);
        sims_to_b.push(sim_b);
        println!("  {:5.1}  {:12.6}  {:12.6}", alpha, sim_a, sim_b);
    }

    // Check monotonicity: sim_to_a should increase
    let mono_a = sims_to_a
        .windows(2)
        .all(|w| w[1] >= w[0] - 0.001); // small tolerance

    // Check correlation
    let corr_a = correlation(&alphas, &sims_to_a);
    let corr_b = correlation(&alphas, &sims_to_b);

    println!("  Correlation alpha vs sim-to-A: {:+.4} (expect positive)", corr_a);
    println!("  Correlation alpha vs sim-to-B: {:+.4} (expect negative)", corr_b);

    let pass = mono_a && corr_a > 0.95 && corr_b < -0.95;
    println!(
        "  Monotonic: {}",
        if pass { "YES" } else { "NO" }
    );
    println!(
        "  RESULT: {}\n",
        if pass { "PASS" } else { "FAIL" }
    );
    pass
}

fn test_fractional_position() -> bool {
    println!("--- Test 8: Fractional Position Continuity ---");
    println!("  Adjacent fractional positions should have similarity > 0.95");

    let n_harmonics = 32;
    let vocab_size = 65; // Shakespeare vocab
    let idx_a = 20.0; // arbitrary integer position
    let idx_b = 21.0;

    let fractions: Vec<f64> = (0..=10).map(|i| i as f64 / 10.0).collect();
    let mut embeddings = Vec::new();

    for &frac in &fractions {
        let c_val = idx_a + frac * (idx_b - idx_a);
        let emb = fractional_embedding(c_val, vocab_size, n_harmonics);
        embeddings.push(emb);
    }

    // Check adjacent similarity
    let mut min_adjacent_sim = 1.0_f64;
    println!(
        "  {:>5}  {:>12}  {:>16}",
        "frac", "sim to prev", "sim to endpoint A"
    );

    let endpoint_a = &embeddings[0];
    for i in 0..embeddings.len() {
        let sim_to_a = cosine_similarity(&embeddings[i], endpoint_a);
        if i > 0 {
            let sim = cosine_similarity(&embeddings[i], &embeddings[i - 1]);
            min_adjacent_sim = min_adjacent_sim.min(sim);
            println!(
                "  {:5.1}  {:12.6}  {:16.6}",
                fractions[i], sim, sim_to_a
            );
        } else {
            println!(
                "  {:5.1}  {:>12}  {:16.6}",
                fractions[i], "-", sim_to_a
            );
        }
    }

    println!("  Min adjacent similarity: {:.6}", min_adjacent_sim);

    let pass = min_adjacent_sim > 0.95;
    println!(
        "  Continuous: {}",
        if pass { "YES" } else { "NO" }
    );
    println!(
        "  RESULT: {}\n",
        if pass { "PASS" } else { "FAIL" }
    );
    pass
}

fn test_chimera_band_independence() -> bool {
    println!("--- Test 9: Chimera Band Independence ---");
    println!("  Low dims should match source A, high dims should match source B");

    let n_harmonics = 32;
    let theta_a = deg_to_rad(40.0);
    let theta_b = deg_to_rad(200.0);
    let vec_a = harmonic_embedding_cossin(theta_a, n_harmonics);
    let vec_b = harmonic_embedding_cossin(theta_b, n_harmonics);

    let split = n_harmonics / 2; // split at the midpoint
    let chim = chimera(&vec_a, &vec_b, split);

    // Extract low and high portions
    let low_dim = split * 2;
    let chim_low = &chim[..low_dim];
    let a_low = &vec_a[..low_dim];
    let b_low = &vec_b[..low_dim];

    let chim_high = &chim[low_dim..];
    let a_high = &vec_a[low_dim..];
    let b_high = &vec_b[low_dim..];

    let sim_low_a = cosine_similarity(chim_low, a_low);
    let sim_low_b = cosine_similarity(chim_low, b_low);
    let sim_high_a = cosine_similarity(chim_high, a_high);
    let sim_high_b = cosine_similarity(chim_high, b_high);

    println!("  Split at harmonic {}/{}", split, n_harmonics);
    println!("  Low dims:  sim to A = {:.6}, sim to B = {:.6}", sim_low_a, sim_low_b);
    println!(
        "  High dims: sim to A = {:.6}, sim to B = {:.6}",
        sim_high_a, sim_high_b
    );

    // Low should match A perfectly, high should match B perfectly
    let pass = (sim_low_a - 1.0).abs() < 1e-6 && (sim_high_b - 1.0).abs() < 1e-6;
    println!(
        "  Band independence: {}",
        if pass { "PERFECT" } else { "IMPERFECT" }
    );
    println!(
        "  RESULT: {}\n",
        if pass { "PASS" } else { "FAIL" }
    );
    pass
}

fn test_norm_preservation() -> bool {
    println!("--- Test 10: Norm Preservation ---");
    println!("  Interpolated/fractional vectors should maintain comparable magnitude");

    let n_harmonics = 32;
    let theta_a = deg_to_rad(30.0);
    let theta_b = deg_to_rad(150.0);
    let vec_a = harmonic_embedding_cossin(theta_a, n_harmonics);
    let vec_b = harmonic_embedding_cossin(theta_b, n_harmonics);

    let norm_a = magnitude(&vec_a);
    let norm_b = magnitude(&vec_b);

    println!("  Source A norm: {:.6}", norm_a);
    println!("  Source B norm: {:.6}", norm_b);

    let mut pass = true;

    // Interpolated norms
    for alpha_i in &[0, 3, 5, 7, 10] {
        let alpha = *alpha_i as f64 / 10.0;
        let blended = interpolate(&vec_a, &vec_b, alpha);
        let norm = magnitude(&blended);
        let ratio = norm / norm_a;
        let ok = ratio > 0.5 && ratio < 2.0;
        println!(
            "  alpha={:.1} norm={:.6} ratio={:.4} {}",
            alpha,
            norm,
            ratio,
            if ok { "OK" } else { "FAIL" }
        );
        if !ok {
            pass = false;
        }
    }

    // Fractional norms
    let vocab_size = 65;
    for frac_i in &[0, 3, 5, 7, 10] {
        let frac = *frac_i as f64 / 10.0;
        let c_val = 20.0 + frac;
        let emb = fractional_embedding(c_val, vocab_size, n_harmonics);
        let norm = magnitude(&emb);
        let ratio = norm / norm_a;
        let ok = ratio > 0.5 && ratio < 2.0;
        println!(
            "  frac={:.1}  norm={:.6} ratio={:.4} {}",
            frac,
            norm,
            ratio,
            if ok { "OK" } else { "FAIL" }
        );
        if !ok {
            pass = false;
        }
    }

    println!(
        "  RESULT: {}\n",
        if pass { "PASS" } else { "FAIL" }
    );
    pass
}

// ============================================================
// Phase 5: Musical Interval Theory (4 tests)
// ============================================================

fn test_adjacent_intervals() -> bool {
    println!("--- Test 11: Adjacent Channel Intervals ---");
    println!("  (1,2)=octave, (2,3)=fifth, (3,4)=fourth, (4,5)=major third");

    let expected = [
        (1, 2, "octave"),
        (2, 3, "perfect fifth"),
        (3, 4, "perfect fourth"),
        (4, 5, "major third"),
        (5, 6, "minor third"),
    ];

    let mut pass = true;

    println!(
        "  {:>10}  {:>8}  {:>16}  {:>16}  {:>8}",
        "Pair", "Ratio", "Expected", "Got", "Match"
    );

    for &(n, m, expected_name) in &expected {
        let (_th, name, _rank) = consonance_score(n, m);
        let ratio = m as f64 / n as f64;
        let ok = name == expected_name;
        println!(
            "  ({:>2},{:>2})    {:>8.4}  {:>16}  {:>16}  {}",
            n,
            m,
            ratio,
            expected_name,
            name,
            if ok { "OK" } else { "FAIL" }
        );
        if !ok {
            pass = false;
        }
    }

    // Extended table up to n=16
    println!("\n  Full adjacent intervals up to n=16:");
    for n in 1..16 {
        let m = n + 1;
        let (th, name, rank) = consonance_score(n, m);
        let ratio = m as f64 / n as f64;
        let cons_label = if rank <= 5 {
            "consonant"
        } else if rank <= 9 {
            "mild"
        } else {
            "dissonant"
        };
        println!(
            "  ({:>2},{:>2})  {:>8.4}  {:>16}  Tenney {:>5.2}  {}",
            n, m, ratio, name, th, cons_label
        );
    }

    println!(
        "\n  RESULT: {}\n",
        if pass { "PASS" } else { "FAIL" }
    );
    pass
}

fn test_tenney_height_ordering() -> bool {
    println!("--- Test 12: Tenney Height Ordering ---");
    println!("  unison(0) < octave(1.0) < fifth(2.585) < fourth(3.585)");

    let cases = [
        ("unison", 1, 1, 0.0),
        ("octave", 2, 1, 1.0),
        ("perfect fifth", 3, 2, 2.585),
        ("perfect fourth", 4, 3, 3.585),
        ("major third", 5, 4, 4.322),
    ];

    let mut pass = true;
    let mut prev_th = -1.0;

    println!(
        "  {:>16}  {:>6}  {:>10}  {:>10}  {:>8}",
        "Interval", "Ratio", "Expected", "Got", "Order"
    );

    for &(name, p, q, expected_approx) in &cases {
        let th = tenney_height(p, q);
        let order_ok = th > prev_th;
        let value_ok = (th - expected_approx).abs() < 0.01;
        println!(
            "  {:>16}  {:>2}:{:<2}  {:>10.3}  {:>10.3}  {}",
            name,
            p,
            q,
            expected_approx,
            th,
            if order_ok && value_ok { "OK" } else { "FAIL" }
        );
        if !order_ok || !value_ok {
            pass = false;
        }
        prev_th = th;
    }

    println!(
        "  RESULT: {}\n",
        if pass { "PASS" } else { "FAIL" }
    );
    pass
}

fn test_consonance_map() -> bool {
    println!("--- Test 13: Consonance Map ---");
    println!("  All pairs 1..16, verify known pairs appear at correct intervals");

    let mut consonant_count = 0;
    let mut mild_count = 0;
    let mut dissonant_count = 0;
    let mut total_pairs = 0;

    // Check specific known pairs
    let known_pairs = [
        (1, 2, "octave"),
        (2, 3, "perfect fifth"),
        (3, 4, "perfect fourth"),
        (2, 4, "octave"),       // 4/2 = 2:1
        (3, 6, "octave"),       // 6/3 = 2:1
        (4, 6, "perfect fifth"), // 6/4 = 3:2
    ];

    let mut known_ok = true;
    println!("  Checking known pairs:");
    for &(n_a, n_b, expected) in &known_pairs {
        let (_, name, _) = consonance_score(n_a, n_b);
        let ok = name == expected;
        println!(
            "    ({:>2},{:>2}) = {} (expected {}) {}",
            n_a,
            n_b,
            name,
            expected,
            if ok { "OK" } else { "FAIL" }
        );
        if !ok {
            known_ok = false;
        }
    }

    // Count all pairs
    for n in 1..=16_u32 {
        for m in (n + 1)..=16 {
            let (_, _, rank) = consonance_score(n, m);
            total_pairs += 1;
            if rank <= 5 {
                consonant_count += 1;
            } else if rank <= 9 {
                mild_count += 1;
            } else {
                dissonant_count += 1;
            }
        }
    }

    println!("\n  Total pairs (1..16): {}", total_pairs);
    println!(
        "  Consonant (rank<=5): {} ({:.1}%)",
        consonant_count,
        100.0 * consonant_count as f64 / total_pairs as f64
    );
    println!(
        "  Mild (rank 6-9):     {} ({:.1}%)",
        mild_count,
        100.0 * mild_count as f64 / total_pairs as f64
    );
    println!(
        "  Dissonant (rank>9):  {} ({:.1}%)",
        dissonant_count,
        100.0 * dissonant_count as f64 / total_pairs as f64
    );

    let pass = known_ok && total_pairs == 120; // C(16,2) = 120
    println!(
        "  RESULT: {}\n",
        if pass { "PASS" } else { "FAIL" }
    );
    pass
}

fn test_interval_identification() -> bool {
    println!("--- Test 14: Interval Identification ---");
    println!("  Known ratios should map to correct interval names");

    let cases = [
        (1.5, "perfect fifth"),
        (4.0 / 3.0, "perfect fourth"),
        (5.0 / 4.0, "major third"),
        (2.0, "octave"),
        (1.0, "unison"),
        (6.0 / 5.0, "minor third"),
        (3.0, "perfect fifth"), // 3.0 -> reduce to 1.5 within octave
        (9.0 / 8.0, "major second"),
    ];

    let mut pass = true;

    println!(
        "  {:>10}  {:>16}  {:>16}  {:>8}",
        "Ratio", "Expected", "Got", "Match"
    );

    for &(ratio, expected) in &cases {
        let (name, _dist) = identify_interval(ratio);
        let ok = name == expected;
        println!(
            "  {:>10.4}  {:>16}  {:>16}  {}",
            ratio,
            expected,
            name,
            if ok { "OK" } else { "FAIL" }
        );
        if !ok {
            pass = false;
        }
    }

    println!(
        "  RESULT: {}\n",
        if pass { "PASS" } else { "FAIL" }
    );
    pass
}

// ============================================================
// Main
// ============================================================

fn main() {
    println!("=== Wave Coherence Experiments -- Pure Rust ===");
    println!("No GPU, no external dependencies, no neural networks.");
    println!("Cross-language validation of the mathematical foundations.\n");

    let mut passed = 0;
    let mut failed = 0;

    // Phase 16: Wave Packet Engine
    println!("=== PHASE 16: WAVE PACKET ENGINE ===\n");
    if test_dft_round_trip() { passed += 1; } else { failed += 1; }
    if test_resonance_vs_cosine() { passed += 1; } else { failed += 1; }
    if test_wave_packet_retrieval() { passed += 1; } else { failed += 1; }
    if test_selective_band_loading() { passed += 1; } else { failed += 1; }
    if test_amplitude_band_selection() { passed += 1; } else { failed += 1; }
    if test_band_energy_distribution() { passed += 1; } else { failed += 1; }

    // Phase 4: Harmonic Construction
    println!("\n=== PHASE 4: HARMONIC CONSTRUCTION ===\n");
    if test_interpolation_monotonicity() { passed += 1; } else { failed += 1; }
    if test_fractional_position() { passed += 1; } else { failed += 1; }
    if test_chimera_band_independence() { passed += 1; } else { failed += 1; }
    if test_norm_preservation() { passed += 1; } else { failed += 1; }

    // Phase 5: Musical Interval Theory
    println!("\n=== PHASE 5: MUSICAL INTERVAL THEORY ===\n");
    if test_adjacent_intervals() { passed += 1; } else { failed += 1; }
    if test_tenney_height_ordering() { passed += 1; } else { failed += 1; }
    if test_consonance_map() { passed += 1; } else { failed += 1; }
    if test_interval_identification() { passed += 1; } else { failed += 1; }

    let total = passed + failed;
    println!("\n=== RESULTS: {} passed, {} failed out of {} ===", passed, failed, total);
    if failed == 0 {
        println!("ALL TESTS PASSED.");
    } else {
        println!("SOME TESTS FAILED -- review output above.");
    }
}
