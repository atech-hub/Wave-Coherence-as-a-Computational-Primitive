/// Hybrid vs Embedded: Which Architecture Carries More Information?
///
/// Two approaches to combining circle (phase) + sphere (magnitude/elevation):
///
///   GATED HYBRID:
///     H = cos(n×Δφ) × [(1-β) + β × P_l(cos Δθ)]
///     Circle and sphere in parallel. Elevation modulates.
///
///   EMBEDDED HYBRID:
///     Phase adjusted by magnitude before feeding to circle:
///     φ_eff = φ + α × (r - r_mean)  (magnitude shifts the phase)
///     Then: E = cos(n × Δφ_eff)
///     Sphere folded INTO the circle. One coherence function.
///
/// Evaluation on baseline trained embeddings (65 tokens):
///   1. Score spread — wider distribution = more discriminating
///   2. Known-pair separation — related vs unrelated gap
///   3. Rank agreement — do they see the same structure?
///
/// Zero dependencies. Reads weight files read-only.

use std::convert::TryInto;
use std::f64::consts::PI;

// ── Binary tensor reader ──

fn read_tensor(path: &str) -> (Vec<usize>, Vec<f32>) {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("  ERROR reading {}: {}", path, e);
            return (vec![], vec![]);
        }
    };
    if bytes.len() < 4 { return (vec![], vec![]); }
    let ndims = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let mut dims = Vec::with_capacity(ndims);
    let mut offset = 4;
    for _ in 0..ndims {
        if offset + 4 > bytes.len() { return (vec![], vec![]); }
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
    (dims, values)
}

// ── Legendre ──

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

// ── Extract phase + magnitude ──

fn extract(dims: &[usize], values: &[f32]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let vocab = dims[0];
    let embed = dims[1];
    let bands = embed / 2;
    let mut phases = vec![vec![0.0f64; vocab]; bands];
    let mut mags = vec![vec![0.0f64; vocab]; bands];
    for c in 0..vocab {
        for k in 0..bands {
            let cv = values[c * embed + 2 * k] as f64;
            let sv = values[c * embed + 2 * k + 1] as f64;
            let mut angle = sv.atan2(cv);
            if angle < 0.0 { angle += 2.0 * PI; }
            phases[k][c] = angle;
            mags[k][c] = (cv * cv + sv * sv).sqrt();
        }
    }
    (phases, mags)
}

// ── Coherence methods ──

/// Plain circle: cos(n × Δφ)
fn circle_best(phi_a: f64, phi_b: f64, n_max: usize) -> (f64, usize) {
    let mut best = 0.0f64;
    let mut best_n = 1;
    for n in 1..=n_max {
        let c = (n as f64 * (phi_a - phi_b)).cos();
        if c.abs() > best.abs() { best = c; best_n = n; }
    }
    (best, best_n)
}

/// Gated hybrid: cos(n×Δφ) × [(1-β) + β × P_l(cos Δθ)]
fn gated_best(phi_a: f64, phi_b: f64, elev_a: f64, elev_b: f64,
              n_max: usize, l_max: usize, beta: f64) -> (f64, usize, usize) {
    let dtheta = elev_a - elev_b;
    let mut best = 0.0f64;
    let mut best_n = 1;
    let mut best_l = 1;
    for n in 1..=n_max {
        let c = (n as f64 * (phi_a - phi_b)).cos();
        for l in 1..=l_max {
            let gate = (1.0 - beta) + beta * legendre(l, dtheta.cos());
            let h = c * gate;
            if h.abs() > best.abs() { best = h; best_n = n; best_l = l; }
        }
    }
    (best, best_n, best_l)
}

/// Embedded hybrid: adjust phase by magnitude, then circle coherence
/// φ_eff = φ + α × (r - r_mean) / r_std
/// Then cos(n × Δφ_eff)
fn embedded_best(phi_a: f64, phi_b: f64, mag_a: f64, mag_b: f64,
                 r_mean: f64, r_std: f64, alpha: f64, n_max: usize) -> (f64, usize) {
    let adj_a = if r_std > 1e-12 { alpha * (mag_a - r_mean) / r_std } else { 0.0 };
    let adj_b = if r_std > 1e-12 { alpha * (mag_b - r_mean) / r_std } else { 0.0 };
    let phi_eff_a = phi_a + adj_a;
    let phi_eff_b = phi_b + adj_b;
    let mut best = 0.0f64;
    let mut best_n = 1;
    for n in 1..=n_max {
        let c = (n as f64 * (phi_eff_a - phi_eff_b)).cos();
        if c.abs() > best.abs() { best = c; best_n = n; }
    }
    (best, best_n)
}

// ── Statistics ──

fn mean(v: &[f64]) -> f64 { v.iter().sum::<f64>() / v.len() as f64 }

fn std_dev(v: &[f64]) -> f64 {
    let m = mean(v);
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let idx = (p / 100.0 * (sorted.len() - 1) as f64) as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Spearman rank correlation between two score vectors
fn spearman(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    let rank_a = ranks(a);
    let rank_b = ranks(b);
    // Pearson correlation of ranks
    let mean_a = mean(&rank_a);
    let mean_b = mean(&rank_b);
    let mut num = 0.0f64;
    let mut den_a = 0.0f64;
    let mut den_b = 0.0f64;
    for i in 0..n {
        let da = rank_a[i] - mean_a;
        let db = rank_b[i] - mean_b;
        num += da * db;
        den_a += da * da;
        den_b += db * db;
    }
    if den_a < 1e-12 || den_b < 1e-12 { return 0.0; }
    num / (den_a * den_b).sqrt()
}

fn ranks(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    let mut indexed: Vec<(usize, f64)> = v.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let mut r = vec![0.0f64; n];
    for (rank, &(orig_idx, _)) in indexed.iter().enumerate() {
        r[orig_idx] = rank as f64;
    }
    r
}

// ── Known relationships (Shakespeare 65-char vocab) ──
// Vocab: typically \n, space, !, ', ,, -, ., 0-9, :, ;, A-Z, a-z
// We define relationship groups

fn char_at(i: usize) -> char {
    // Standard Shakespeare char-level vocab (65 chars)
    // This is an approximation — the exact mapping depends on the tokenizer
    // We use structural relationships that hold regardless of exact ordering
    let chars: Vec<char> = "\n !',-.0123456789:;ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz".chars().collect();
    if i < chars.len() { chars[i] } else { '?' }
}

fn is_lowercase(c: char) -> bool { c.is_ascii_lowercase() }
fn is_uppercase(c: char) -> bool { c.is_ascii_uppercase() }
fn is_vowel(c: char) -> bool { "aeiouAEIOU".contains(c) }
fn is_digit(c: char) -> bool { c.is_ascii_digit() }
fn is_punct(c: char) -> bool { "!',-.:;".contains(c) }

/// Returns true if this pair is "known related"
fn known_related(i: usize, j: usize) -> bool {
    let a = char_at(i);
    let b = char_at(j);

    // Same letter, different case
    if a.to_ascii_lowercase() == b.to_ascii_lowercase() && a != b
        && (is_lowercase(a) || is_uppercase(a))
        && (is_lowercase(b) || is_uppercase(b)) {
        return true;
    }

    // Both vowels
    if is_vowel(a) && is_vowel(b) && a != b { return true; }

    // Both digits
    if is_digit(a) && is_digit(b) && a != b { return true; }

    // Both punctuation
    if is_punct(a) && is_punct(b) && a != b { return true; }

    false
}

// ── Main ──

fn main() {
    println!("=== Hybrid vs Embedded: Architecture Evaluation ===\n");

    let base = "../experiments/phase17-weight-analysis/weights";

    let n_max = 15;
    let l_max = 6;
    let beta = 0.4;
    let alpha_values = [0.3, 0.5, 1.0, 2.0];

    // Load baseline (trained, has magnitude variation)
    let path = format!("{}/baseline/wte.weight.bin", base);
    let (dims, values) = read_tensor(&path);
    if dims.is_empty() {
        eprintln!("FATAL: Could not read {}", path);
        std::process::exit(1);
    }

    let vocab = dims[0];
    let n_bands = dims[1] / 2;
    println!("  Baseline embeddings: {} tokens × {} bands", vocab, n_bands);

    let (phases, mags) = extract(&dims, &values);

    // Compute per-band magnitude stats for embedding
    let mut band_means = vec![0.0f64; n_bands];
    let mut band_stds = vec![0.0f64; n_bands];
    for k in 0..n_bands {
        band_means[k] = mean(&mags[k]);
        band_stds[k] = std_dev(&mags[k]);
    }

    // Magnitude → elevation for gated
    let mut elevations = vec![vec![0.0f64; vocab]; n_bands];
    for k in 0..n_bands {
        let min_r = mags[k].iter().cloned().fold(f64::INFINITY, f64::min);
        let max_r = mags[k].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max_r - min_r;
        for c in 0..vocab {
            elevations[k][c] = if range > 1e-12 {
                PI * (mags[k][c] - min_r) / range
            } else {
                PI / 2.0
            };
        }
    }

    // ── Compute all pairwise scores ──
    let n_pairs = vocab * (vocab - 1) / 2;
    println!("  Computing {} pairs × 4 methods...\n", n_pairs);

    // Aggregate across bands: average best coherence across all 64 bands
    let mut circle_scores = vec![0.0f64; n_pairs];
    let mut gated_scores = vec![0.0f64; n_pairs];
    let mut embedded_scores: Vec<Vec<f64>> = alpha_values.iter().map(|_| vec![0.0f64; n_pairs]).collect();

    let mut pair_idx = 0;
    let mut pair_is_related = vec![false; n_pairs];

    for i in 0..vocab {
        for j in (i + 1)..vocab {
            pair_is_related[pair_idx] = known_related(i, j);

            let mut sum_c = 0.0f64;
            let mut sum_g = 0.0f64;
            let mut sum_e: Vec<f64> = vec![0.0; alpha_values.len()];

            for k in 0..n_bands {
                let (c, _) = circle_best(phases[k][i], phases[k][j], n_max);
                sum_c += c;

                let (g, _, _) = gated_best(
                    phases[k][i], phases[k][j],
                    elevations[k][i], elevations[k][j],
                    n_max, l_max, beta);
                sum_g += g;

                for (ai, &alpha) in alpha_values.iter().enumerate() {
                    let (e, _) = embedded_best(
                        phases[k][i], phases[k][j],
                        mags[k][i], mags[k][j],
                        band_means[k], band_stds[k],
                        alpha, n_max);
                    sum_e[ai] += e;
                }
            }

            circle_scores[pair_idx] = sum_c / n_bands as f64;
            gated_scores[pair_idx] = sum_g / n_bands as f64;
            for ai in 0..alpha_values.len() {
                embedded_scores[ai][pair_idx] = sum_e[ai] / n_bands as f64;
            }

            pair_idx += 1;
        }
    }

    // ══════════════════════════════════════════════════════════════
    // METRIC 1: Score Spread
    // ══════════════════════════════════════════════════════════════
    println!("=== METRIC 1: Score Spread (wider = more discriminating) ===\n");

    let methods: Vec<(String, &[f64])> = {
        let mut m = vec![
            ("Circle".to_string(), circle_scores.as_slice()),
            (format!("Gated β={}", beta), gated_scores.as_slice()),
        ];
        for (ai, &alpha) in alpha_values.iter().enumerate() {
            m.push((format!("Embed α={}", alpha), embedded_scores[ai].as_slice()));
        }
        m
    };

    println!("  {:>14}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}",
             "Method", "Mean", "Std", "Min", "P25", "P75", "Max");
    println!("  {:>14}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}",
             "------", "----", "---", "---", "---", "---", "---");

    for (name, scores) in &methods {
        let m = mean(scores);
        let s = std_dev(scores);
        let mut sorted: Vec<f64> = scores.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p25 = percentile(&sorted, 25.0);
        let p75 = percentile(&sorted, 75.0);

        println!("  {:>14}  {:>8.4}  {:>8.4}  {:>8.4}  {:>8.4}  {:>8.4}  {:>8.4}",
                 name, m, s,
                 sorted[0], p25, p75, sorted[sorted.len() - 1]);
    }
    println!();

    // ══════════════════════════════════════════════════════════════
    // METRIC 2: Known-Pair Separation
    // ══════════════════════════════════════════════════════════════
    println!("=== METRIC 2: Known-Pair Separation ===\n");

    let n_related = pair_is_related.iter().filter(|&&r| r).count();
    let n_unrelated = n_pairs - n_related;
    println!("  Related pairs: {}, Unrelated pairs: {}\n", n_related, n_unrelated);

    if n_related > 0 {
        println!("  Related pairs include:");
        let mut shown = 0;
        let mut pi = 0;
        for i in 0..vocab {
            for j in (i + 1)..vocab {
                if pair_is_related[pi] && shown < 10 {
                    println!("    '{}' ({}) ↔ '{}' ({})", char_at(i), i, char_at(j), j);
                    shown += 1;
                }
                pi += 1;
            }
        }
        if n_related > 10 { println!("    ... and {} more", n_related - 10); }
        println!();

        println!("  {:>14}  {:>10}  {:>10}  {:>10}  {:>10}",
                 "Method", "Rel Mean", "Unrel Mean", "Gap", "Gap/Std");
        println!("  {:>14}  {:>10}  {:>10}  {:>10}  {:>10}",
                 "------", "--------", "----------", "---", "-------");

        for (name, scores) in &methods {
            let mut rel_scores = Vec::new();
            let mut unrel_scores = Vec::new();
            for (idx, &s) in scores.iter().enumerate() {
                if pair_is_related[idx] {
                    rel_scores.push(s);
                } else {
                    unrel_scores.push(s);
                }
            }
            let rel_mean = mean(&rel_scores);
            let unrel_mean = mean(&unrel_scores);
            let gap = rel_mean - unrel_mean;
            let pooled_std = std_dev(scores);
            let gap_over_std = if pooled_std > 1e-12 { gap / pooled_std } else { 0.0 };

            println!("  {:>14}  {:>10.4}  {:>10.4}  {:>10.4}  {:>10.4}",
                     name, rel_mean, unrel_mean, gap, gap_over_std);
        }
        println!();

        // Also show: best method's top related pairs vs worst unrelated
        println!("  Top 10 related pairs by gated hybrid score:");
        let mut rel_with_scores: Vec<(usize, f64)> = Vec::new();
        for (idx, &s) in gated_scores.iter().enumerate() {
            if pair_is_related[idx] {
                rel_with_scores.push((idx, s));
            }
        }
        rel_with_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        for &(idx, score) in rel_with_scores.iter().take(10) {
            // Decode pair index back to (i, j)
            let mut pi = 0;
            let mut found_i = 0;
            let mut found_j = 0;
            'outer: for i in 0..vocab {
                for j in (i + 1)..vocab {
                    if pi == idx { found_i = i; found_j = j; break 'outer; }
                    pi += 1;
                }
            }
            let cs = circle_scores[idx];
            println!("    '{}' ↔ '{}': circle={:.4}, gated={:.4}, diff={:.4}",
                     char_at(found_i), char_at(found_j), cs, score, score - cs);
        }
    }
    println!();

    // ══════════════════════════════════════════════════════════════
    // METRIC 3: Rank Agreement (Spearman)
    // ══════════════════════════════════════════════════════════════
    println!("=== METRIC 3: Rank Agreement (Spearman Correlation) ===\n");

    let spear_gated = spearman(&circle_scores, &gated_scores);
    println!("  Circle vs Gated:          ρ = {:.6}", spear_gated);

    for (ai, &alpha) in alpha_values.iter().enumerate() {
        let sp = spearman(&circle_scores, &embedded_scores[ai]);
        println!("  Circle vs Embed α={:.1}:    ρ = {:.6}", alpha, sp);
    }

    let best_embed_idx = 1; // α=0.5 as representative
    let sp_ge = spearman(&gated_scores, &embedded_scores[best_embed_idx]);
    println!("  Gated vs Embed α=0.5:     ρ = {:.6}", sp_ge);

    println!();
    println!("  Interpretation:");
    println!("    ρ > 0.99: methods see essentially the same structure");
    println!("    ρ = 0.90-0.99: mostly agree, some reranking");
    println!("    ρ < 0.90: significantly different views of the data");
    println!();

    // ══════════════════════════════════════════════════════════════
    // METRIC 4: Unique Rank Disagreements
    // ══════════════════════════════════════════════════════════════
    println!("=== METRIC 4: Where Do Methods Disagree Most? ===\n");

    // Find pairs where gated and circle disagree most on ranking
    let circle_ranks = ranks(&circle_scores);
    let gated_ranks = ranks(&gated_scores);
    let embed_ranks = ranks(&embedded_scores[best_embed_idx]);

    let mut gated_disagree: Vec<(usize, f64)> = (0..n_pairs)
        .map(|i| (i, (gated_ranks[i] - circle_ranks[i]).abs()))
        .collect();
    gated_disagree.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("  Top 10 rank disagreements: Circle vs Gated");
    println!("  {:>6}  {:>6}  {:>8}  {:>8}  {:>8}  {:>8}",
             "TokA", "TokB", "CircScr", "GateScr", "CircRnk", "GateRnk");
    for &(idx, _rank_diff) in gated_disagree.iter().take(10) {
        let mut pi = 0;
        let mut fi = 0;
        let mut fj = 0;
        'outer2: for i in 0..vocab {
            for j in (i + 1)..vocab {
                if pi == idx { fi = i; fj = j; break 'outer2; }
                pi += 1;
            }
        }
        println!("  {:>4}{}  {:>4}{}  {:>8.4}  {:>8.4}  {:>8.0}  {:>8.0}",
                 char_at(fi), "", char_at(fj), "",
                 circle_scores[idx], gated_scores[idx],
                 circle_ranks[idx], gated_ranks[idx]);
    }
    println!();

    let mut embed_disagree: Vec<(usize, f64)> = (0..n_pairs)
        .map(|i| (i, (embed_ranks[i] - circle_ranks[i]).abs()))
        .collect();
    embed_disagree.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("  Top 10 rank disagreements: Circle vs Embedded α=0.5");
    println!("  {:>6}  {:>6}  {:>8}  {:>8}  {:>8}  {:>8}",
             "TokA", "TokB", "CircScr", "EmbdScr", "CircRnk", "EmbdRnk");
    for &(idx, _rank_diff) in embed_disagree.iter().take(10) {
        let mut pi = 0;
        let mut fi = 0;
        let mut fj = 0;
        'outer3: for i in 0..vocab {
            for j in (i + 1)..vocab {
                if pi == idx { fi = i; fj = j; break 'outer3; }
                pi += 1;
            }
        }
        println!("  {:>4}{}  {:>4}{}  {:>8.4}  {:>8.4}  {:>8.0}  {:>8.0}",
                 char_at(fi), "", char_at(fj), "",
                 circle_scores[idx], embedded_scores[best_embed_idx][idx],
                 circle_ranks[idx], embed_ranks[idx]);
    }
    println!();

    // ══════════════════════════════════════════════════════════════
    // VERDICT
    // ══════════════════════════════════════════════════════════════
    println!("=== VERDICT ===\n");

    // Compare std devs
    let circle_std = std_dev(&circle_scores);
    let gated_std = std_dev(&gated_scores);
    let embed_stds: Vec<f64> = embedded_scores.iter().map(|s| std_dev(s)).collect();

    println!("  SPREAD:");
    println!("    Circle std:      {:.6}", circle_std);
    println!("    Gated std:       {:.6}", gated_std);
    for (ai, &alpha) in alpha_values.iter().enumerate() {
        println!("    Embed α={} std: {:.6}", alpha, embed_stds[ai]);
    }

    let best_spread = if gated_std > circle_std && gated_std >= *embed_stds.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0) {
        "Gated"
    } else if embed_stds.iter().any(|&s| s > circle_std && s >= gated_std) {
        "Embedded"
    } else {
        "Circle (neither adds spread)"
    };
    println!("    Winner: {}", best_spread);
    println!();

    println!("  RANK DIVERGENCE:");
    println!("    Gated ρ vs circle:  {:.4} → {}", spear_gated,
             if spear_gated < 0.90 { "SIGNIFICANTLY different view" }
             else if spear_gated < 0.99 { "some reranking" }
             else { "essentially same" });
    let sp_embed = spearman(&circle_scores, &embedded_scores[best_embed_idx]);
    println!("    Embed ρ vs circle:  {:.4} → {}", sp_embed,
             if sp_embed < 0.90 { "SIGNIFICANTLY different view" }
             else if sp_embed < 0.99 { "some reranking" }
             else { "essentially same" });
    println!();

    if spear_gated < sp_embed {
        println!("  → Gated hybrid provides MORE reranking (different from circle)");
        println!("    It sees structure the circle doesn't.");
    } else if sp_embed < spear_gated {
        println!("  → Embedded hybrid provides MORE reranking (different from circle)");
        println!("    It sees structure the circle doesn't.");
    } else {
        println!("  → Both methods show similar rank divergence from circle.");
    }

    println!();
    println!("=== END ===");
}
