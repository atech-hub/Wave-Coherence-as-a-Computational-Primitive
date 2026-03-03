/// Embedded Coherence: Synthetic Controlled Test
///
/// Can the embedded method rank WITHIN groups that the circle sees as equal?
///
/// Setup:
///   500 tokens in 5 semantic groups (100 each)
///   32 bands per token (64-dim embeddings, pairs of cos/sin)
///
///   Phase (azimuth): tokens in same group share a base phase per band,
///     with small jitter. Different groups have well-separated phases.
///     → Circle can identify groups (high intra-group coherence)
///     → Circle CANNOT rank within groups (all look ~equally close)
///
///   Magnitude: encodes a "semantic strength" gradient within each group.
///     Token 0 in a group = magnitude 0.5, token 99 = magnitude 1.5
///     → Circle ignores this completely
///     → Embedded method adjusts phase by magnitude → breaks ties
///
///   Ground truth: within a group, tokens with similar magnitude index
///     are "more related" than tokens with distant magnitude index.
///     The embedded method should capture this; the circle should not.
///
/// Zero dependencies. Pure math. Fully controlled.

use std::f64::consts::PI;

// ── Parameters ──

const N_GROUPS: usize = 5;
const TOKENS_PER_GROUP: usize = 100;
const N_TOKENS: usize = N_GROUPS * TOKENS_PER_GROUP;
const N_BANDS: usize = 32;
const N_MAX: usize = 15;
const PHASE_JITTER: f64 = 0.05; // small intra-group phase noise (radians)
const MAG_MIN: f64 = 0.5;
const MAG_MAX: f64 = 1.5;

// ── Simple deterministic RNG (xorshift64) ──

struct Rng { state: u64 }

impl Rng {
    fn new(seed: u64) -> Self { Rng { state: seed } }

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

    /// Uniform [-1, 1)
    fn symmetric(&mut self) -> f64 {
        self.uniform() * 2.0 - 1.0
    }
}

// ── Coherence functions ──

fn circle_coherence(phi_a: f64, phi_b: f64, n: usize) -> f64 {
    (n as f64 * (phi_a - phi_b)).cos()
}

fn circle_best(phi_a: f64, phi_b: f64, n_max: usize) -> f64 {
    let mut best = 0.0f64;
    for n in 1..=n_max {
        let c = circle_coherence(phi_a, phi_b, n);
        if c > best { best = c; }
    }
    best
}

fn embedded_best(phi_a: f64, phi_b: f64, mag_a: f64, mag_b: f64,
                 r_mean: f64, r_std: f64, alpha: f64, n_max: usize) -> f64 {
    let adj_a = if r_std > 1e-12 { alpha * (mag_a - r_mean) / r_std } else { 0.0 };
    let adj_b = if r_std > 1e-12 { alpha * (mag_b - r_mean) / r_std } else { 0.0 };
    let phi_eff_a = phi_a + adj_a;
    let phi_eff_b = phi_b + adj_b;
    let mut best = 0.0f64;
    for n in 1..=n_max {
        let c = (n as f64 * (phi_eff_a - phi_eff_b)).cos();
        if c > best { best = c; }
    }
    best
}

// ── Statistics ──

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() { return 0.0; }
    v.iter().sum::<f64>() / v.len() as f64
}

fn std_dev(v: &[f64]) -> f64 {
    let m = mean(v);
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
}

fn spearman(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    let ra = ranks(a);
    let rb = ranks(b);
    let ma = mean(&ra);
    let mb = mean(&rb);
    let mut num = 0.0f64;
    let mut da2 = 0.0f64;
    let mut db2 = 0.0f64;
    for i in 0..n {
        let x = ra[i] - ma;
        let y = rb[i] - mb;
        num += x * y;
        da2 += x * x;
        db2 += y * y;
    }
    if da2 < 1e-12 || db2 < 1e-12 { return 0.0; }
    num / (da2 * db2).sqrt()
}

fn ranks(v: &[f64]) -> Vec<f64> {
    let mut idx: Vec<(usize, f64)> = v.iter().cloned().enumerate().collect();
    idx.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let mut r = vec![0.0f64; v.len()];
    for (rank, &(orig, _)) in idx.iter().enumerate() {
        r[orig] = rank as f64;
    }
    r
}

// ── Main ──

fn main() {
    println!("=== Embedded Coherence: Synthetic Controlled Test ===");
    println!("  {} tokens in {} groups of {}", N_TOKENS, N_GROUPS, TOKENS_PER_GROUP);
    println!("  {} bands, n=1..{}", N_BANDS, N_MAX);
    println!("  Phase jitter: {:.2} rad, Magnitude range: [{:.1}, {:.1}]",
             PHASE_JITTER, MAG_MIN, MAG_MAX);
    println!();

    let mut rng = Rng::new(42);

    // ── Generate synthetic embeddings ──

    // Group base phases: well-separated on the circle per band
    let mut group_base_phase = vec![vec![0.0f64; N_BANDS]; N_GROUPS];
    for g in 0..N_GROUPS {
        for k in 0..N_BANDS {
            // Spread groups evenly, with per-band rotation so groups aren't aligned across bands
            group_base_phase[g][k] = (2.0 * PI * g as f64 / N_GROUPS as f64
                + 0.7 * k as f64).rem_euclid(2.0 * PI);
        }
    }

    // Token phases and magnitudes
    let mut phases = vec![vec![0.0f64; N_TOKENS]; N_BANDS];
    let mut mags = vec![vec![0.0f64; N_TOKENS]; N_BANDS];
    let mut token_group = vec![0usize; N_TOKENS];
    let mut token_index_in_group = vec![0usize; N_TOKENS]; // 0..99, used as ground truth

    for g in 0..N_GROUPS {
        for t in 0..TOKENS_PER_GROUP {
            let token_id = g * TOKENS_PER_GROUP + t;
            token_group[token_id] = g;
            token_index_in_group[token_id] = t;

            // Magnitude: linear gradient within group
            let mag = MAG_MIN + (MAG_MAX - MAG_MIN) * (t as f64 / (TOKENS_PER_GROUP - 1) as f64);

            for k in 0..N_BANDS {
                // Phase: group base + small jitter
                phases[k][token_id] = group_base_phase[g][k]
                    + PHASE_JITTER * rng.symmetric();
                // Magnitude: same across bands (with tiny noise)
                mags[k][token_id] = mag + 0.01 * rng.symmetric();
            }
        }
    }

    // Per-band magnitude stats
    let mut band_means = vec![0.0f64; N_BANDS];
    let mut band_stds = vec![0.0f64; N_BANDS];
    for k in 0..N_BANDS {
        band_means[k] = mean(&mags[k]);
        band_stds[k] = std_dev(&mags[k]);
    }

    println!("  Magnitude stats: mean={:.4}, std={:.4}, CV={:.1}%",
             mean(&band_means), mean(&band_stds),
             100.0 * mean(&band_stds) / mean(&band_means));
    println!();

    // ══════════════════════════════════════════════════════════════
    // TEST 1: Group Detection — Can circle and embedded find groups?
    // ══════════════════════════════════════════════════════════════
    println!("=== TEST 1: Group Detection ===");
    println!("  Both methods should detect same-group pairs.\n");

    // Sample: 50 random intra-group pairs and 50 random inter-group pairs
    let n_sample = 200;
    let mut intra_circle = Vec::new();
    let mut intra_embed = Vec::new();
    let mut inter_circle = Vec::new();
    let mut inter_embed = Vec::new();

    let alpha = 0.3;

    for _ in 0..n_sample {
        // Intra-group pair
        let g = (rng.next_u64() as usize) % N_GROUPS;
        let t1 = g * TOKENS_PER_GROUP + (rng.next_u64() as usize) % TOKENS_PER_GROUP;
        let mut t2 = g * TOKENS_PER_GROUP + (rng.next_u64() as usize) % TOKENS_PER_GROUP;
        while t2 == t1 { t2 = g * TOKENS_PER_GROUP + (rng.next_u64() as usize) % TOKENS_PER_GROUP; }

        let mut sum_c = 0.0f64;
        let mut sum_e = 0.0f64;
        for k in 0..N_BANDS {
            sum_c += circle_best(phases[k][t1], phases[k][t2], N_MAX);
            sum_e += embedded_best(phases[k][t1], phases[k][t2],
                                   mags[k][t1], mags[k][t2],
                                   band_means[k], band_stds[k], alpha, N_MAX);
        }
        intra_circle.push(sum_c / N_BANDS as f64);
        intra_embed.push(sum_e / N_BANDS as f64);

        // Inter-group pair
        let g1 = (rng.next_u64() as usize) % N_GROUPS;
        let mut g2 = (rng.next_u64() as usize) % N_GROUPS;
        while g2 == g1 { g2 = (rng.next_u64() as usize) % N_GROUPS; }
        let t3 = g1 * TOKENS_PER_GROUP + (rng.next_u64() as usize) % TOKENS_PER_GROUP;
        let t4 = g2 * TOKENS_PER_GROUP + (rng.next_u64() as usize) % TOKENS_PER_GROUP;

        let mut sum_c2 = 0.0f64;
        let mut sum_e2 = 0.0f64;
        for k in 0..N_BANDS {
            sum_c2 += circle_best(phases[k][t3], phases[k][t4], N_MAX);
            sum_e2 += embedded_best(phases[k][t3], phases[k][t4],
                                    mags[k][t3], mags[k][t4],
                                    band_means[k], band_stds[k], alpha, N_MAX);
        }
        inter_circle.push(sum_c2 / N_BANDS as f64);
        inter_embed.push(sum_e2 / N_BANDS as f64);
    }

    println!("  {:>14}  {:>10}  {:>10}  {:>10}",
             "Method", "Intra μ", "Inter μ", "Gap");
    println!("  {:>14}  {:>10}  {:>10}  {:>10}",
             "------", "-------", "-------", "---");
    let c_gap = mean(&intra_circle) - mean(&inter_circle);
    let e_gap = mean(&intra_embed) - mean(&inter_embed);
    println!("  {:>14}  {:>10.4}  {:>10.4}  {:>10.4}", "Circle", mean(&intra_circle), mean(&inter_circle), c_gap);
    println!("  {:>14}  {:>10.4}  {:>10.4}  {:>10.4}", format!("Embed α={}", alpha), mean(&intra_embed), mean(&inter_embed), e_gap);
    println!();

    // ══════════════════════════════════════════════════════════════
    // TEST 2: Within-Group Ranking — Can embedded rank by magnitude?
    // ══════════════════════════════════════════════════════════════
    println!("=== TEST 2: Within-Group Ranking (THE KEY TEST) ===");
    println!("  For pairs in the same group, does index distance (magnitude distance)");
    println!("  correlate with coherence score? Circle should not. Embedded should.\n");

    let alpha_values = [0.1, 0.2, 0.3, 0.5, 1.0];

    // For each group, compute all intra-group pairs
    // Ground truth: |index_a - index_b| = semantic distance
    // Lower distance → should have HIGHER coherence

    println!("  {:>14}  {:>10}  {:>10}  {:>10}",
             "Method", "ρ(dist,coh)", "Near μ", "Far μ");
    println!("  {:>14}  {:>10}  {:>10}  {:>10}",
             "------", "-----------", "------", "-----");

    // Collect across all groups
    let mut all_distances = Vec::new();
    let mut all_circle_coh = Vec::new();
    let mut all_embed_coh: Vec<Vec<f64>> = alpha_values.iter().map(|_| Vec::new()).collect();

    for g in 0..N_GROUPS {
        let base = g * TOKENS_PER_GROUP;
        for i in 0..TOKENS_PER_GROUP {
            for j in (i + 1)..TOKENS_PER_GROUP {
                let t1 = base + i;
                let t2 = base + j;
                let dist = (j - i) as f64; // magnitude/index distance

                let mut sum_c = 0.0f64;
                let mut sums_e = vec![0.0f64; alpha_values.len()];

                for k in 0..N_BANDS {
                    sum_c += circle_best(phases[k][t1], phases[k][t2], N_MAX);
                    for (ai, &a) in alpha_values.iter().enumerate() {
                        sums_e[ai] += embedded_best(
                            phases[k][t1], phases[k][t2],
                            mags[k][t1], mags[k][t2],
                            band_means[k], band_stds[k], a, N_MAX);
                    }
                }

                let avg_c = sum_c / N_BANDS as f64;
                all_distances.push(dist);
                all_circle_coh.push(avg_c);
                for ai in 0..alpha_values.len() {
                    all_embed_coh[ai].push(sums_e[ai] / N_BANDS as f64);
                }
            }
        }
    }

    // Split into "near" (distance < 20) and "far" (distance > 80)
    let mut near_circle = Vec::new();
    let mut far_circle = Vec::new();
    let mut near_embed: Vec<Vec<f64>> = alpha_values.iter().map(|_| Vec::new()).collect();
    let mut far_embed: Vec<Vec<f64>> = alpha_values.iter().map(|_| Vec::new()).collect();

    for (idx, &d) in all_distances.iter().enumerate() {
        if d < 20.0 {
            near_circle.push(all_circle_coh[idx]);
            for ai in 0..alpha_values.len() {
                near_embed[ai].push(all_embed_coh[ai][idx]);
            }
        } else if d > 80.0 {
            far_circle.push(all_circle_coh[idx]);
            for ai in 0..alpha_values.len() {
                far_embed[ai].push(all_embed_coh[ai][idx]);
            }
        }
    }

    // Spearman: negative ρ means higher distance → lower coherence (correct behavior)
    let rho_circle = spearman(&all_distances, &all_circle_coh);
    println!("  {:>14}  {:>10.4}  {:>10.4}  {:>10.4}",
             "Circle", rho_circle, mean(&near_circle), mean(&far_circle));

    let mut best_alpha = 0.0f64;
    let mut best_rho = 0.0f64;

    for (ai, &a) in alpha_values.iter().enumerate() {
        let rho = spearman(&all_distances, &all_embed_coh[ai]);
        if rho.abs() > best_rho.abs() { best_rho = rho; best_alpha = a; }
        println!("  {:>14}  {:>10.4}  {:>10.4}  {:>10.4}",
                 format!("Embed α={}", a), rho,
                 mean(&near_embed[ai]), mean(&far_embed[ai]));
    }

    println!();
    println!("  Near = index distance < 20 (similar magnitude)");
    println!("  Far = index distance > 80 (very different magnitude)");
    println!("  Negative ρ = higher distance → lower coherence (desired)");
    println!();

    // ══════════════════════════════════════════════════════════════
    // TEST 3: Retrieval — Given a query, rank within its group
    // ══════════════════════════════════════════════════════════════
    println!("=== TEST 3: Retrieval — Top-K Accuracy ===");
    println!("  Query: token at index 50 in each group.");
    println!("  Task: rank all 99 group members by coherence.");
    println!("  Success: top-10 results should be indices 40-60 (nearby).\n");

    let alpha_test = best_alpha;

    let mut circle_top10_nearby = 0usize;
    let mut embed_top10_nearby = 0usize;
    let mut total_queries = 0usize;

    for g in 0..N_GROUPS {
        let base = g * TOKENS_PER_GROUP;
        let query = base + 50;

        let mut circle_ranking: Vec<(usize, f64)> = Vec::new();
        let mut embed_ranking: Vec<(usize, f64)> = Vec::new();

        for t in 0..TOKENS_PER_GROUP {
            let target = base + t;
            if target == query { continue; }

            let mut sum_c = 0.0f64;
            let mut sum_e = 0.0f64;
            for k in 0..N_BANDS {
                sum_c += circle_best(phases[k][query], phases[k][target], N_MAX);
                sum_e += embedded_best(
                    phases[k][query], phases[k][target],
                    mags[k][query], mags[k][target],
                    band_means[k], band_stds[k], alpha_test, N_MAX);
            }
            circle_ranking.push((t, sum_c / N_BANDS as f64));
            embed_ranking.push((t, sum_e / N_BANDS as f64));
        }

        // Sort by coherence descending
        circle_ranking.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        embed_ranking.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Count top-10 that are within [40, 60]
        let nearby = |idx: usize| -> bool { idx >= 40 && idx <= 60 };

        let c_nearby: usize = circle_ranking.iter().take(10).filter(|(idx, _)| nearby(*idx)).count();
        let e_nearby: usize = embed_ranking.iter().take(10).filter(|(idx, _)| nearby(*idx)).count();

        circle_top10_nearby += c_nearby;
        embed_top10_nearby += e_nearby;
        total_queries += 1;

        // Print detail for group 0
        if g == 0 {
            println!("  Group 0, query=index 50:");
            println!("  Circle top-10: {:?}",
                     circle_ranking.iter().take(10).map(|(idx, _)| *idx).collect::<Vec<_>>());
            println!("  Embed  top-10: {:?}",
                     embed_ranking.iter().take(10).map(|(idx, _)| *idx).collect::<Vec<_>>());
            println!("  Circle nearby in top-10: {}/10", c_nearby);
            println!("  Embed  nearby in top-10: {}/10", e_nearby);
            println!();
        }
    }

    let circle_precision = circle_top10_nearby as f64 / (total_queries * 10) as f64;
    let embed_precision = embed_top10_nearby as f64 / (total_queries * 10) as f64;

    println!("  Average top-10 precision (nearby = indices 40-60):");
    println!("    Circle:          {}/{} = {:.1}%",
             circle_top10_nearby, total_queries * 10, 100.0 * circle_precision);
    println!("    Embed α={:.1}:     {}/{} = {:.1}%",
             alpha_test, embed_top10_nearby, total_queries * 10, 100.0 * embed_precision);
    println!();

    // ══════════════════════════════════════════════════════════════
    // TEST 4: Coherence Gradient — Smooth decay with distance?
    // ══════════════════════════════════════════════════════════════
    println!("=== TEST 4: Coherence Gradient by Distance ===");
    println!("  Average coherence at each index distance (0-99 within group).\n");

    // Bin by distance
    let mut circle_by_dist = vec![Vec::new(); TOKENS_PER_GROUP];
    let mut embed_by_dist = vec![Vec::new(); TOKENS_PER_GROUP];

    for (idx, &d) in all_distances.iter().enumerate() {
        let di = d as usize;
        if di < TOKENS_PER_GROUP {
            circle_by_dist[di].push(all_circle_coh[idx]);
            embed_by_dist[di].push(all_embed_coh[alpha_values.iter().position(|&a| a == alpha_test).unwrap_or(2)][idx]);
        }
    }

    println!("  {:>6}  {:>10}  {:>10}  {:>10}",
             "Dist", "Circle", "Embedded", "Diff");
    println!("  {:>6}  {:>10}  {:>10}  {:>10}",
             "----", "------", "--------", "----");

    for d in (1..TOKENS_PER_GROUP).step_by(5) {
        if !circle_by_dist[d].is_empty() {
            let cm = mean(&circle_by_dist[d]);
            let em = mean(&embed_by_dist[d]);
            println!("  {:>6}  {:>10.4}  {:>10.4}  {:>10.4}",
                     d, cm, em, em - cm);
        }
    }
    println!();

    // ══════════════════════════════════════════════════════════════
    // VERDICT
    // ══════════════════════════════════════════════════════════════
    println!("=== VERDICT ===\n");

    println!("  Group detection (Test 1):");
    println!("    Circle gap: {:.4}", c_gap);
    println!("    Embed gap:  {:.4}", e_gap);
    println!("    → {}", if (c_gap - e_gap).abs() < 0.02 { "Both detect groups equally" }
             else if c_gap > e_gap { "Circle better at groups" }
             else { "Embedded better at groups" });
    println!();

    println!("  Within-group ranking (Test 2 — THE KEY):");
    println!("    Circle ρ(distance, coherence):  {:.4}", rho_circle);
    println!("    Best embed ρ (α={:.1}):          {:.4}", best_alpha, best_rho);
    if best_rho.abs() > rho_circle.abs() * 2.0 && best_rho < -0.1 {
        println!("    → EMBEDDED WINS: magnitude enables within-group ranking");
        println!("      that the circle cannot do.");
    } else if best_rho.abs() > rho_circle.abs() && best_rho < -0.05 {
        println!("    → Embedded shows some ranking ability, moderate improvement.");
    } else {
        println!("    → Neither method ranks within groups effectively.");
    }
    println!();

    println!("  Retrieval precision (Test 3):");
    println!("    Circle: {:.1}%", 100.0 * circle_precision);
    println!("    Embed:  {:.1}%", 100.0 * embed_precision);
    if embed_precision > circle_precision + 0.05 {
        println!("    → Embedded retrieval is substantially better.");
    } else if embed_precision > circle_precision {
        println!("    → Embedded retrieval is slightly better.");
    } else {
        println!("    → No retrieval improvement from embedding.");
    }
    println!();

    println!("  CONCLUSION:");
    if best_rho < -0.2 && embed_precision > circle_precision + 0.1 {
        println!("    The embedded method WORKS. Magnitude carries within-group");
        println!("    information that the circle is blind to. Phase adjustment");
        println!("    by magnitude creates a finer-grained coherence measure.");
        println!("    Ready for real-world testing (Option A: word-level transformer).");
    } else if best_rho < -0.1 || embed_precision > circle_precision + 0.05 {
        println!("    Partial success. Embedded method shows some improvement but");
        println!("    not definitive. May need tuning of α or the embedding function.");
    } else {
        println!("    The embedded method does not improve over circle in this setup.");
        println!("    The magnitude-to-phase adjustment may need a different function");
        println!("    than linear shift, or the synthetic test may be too simple.");
    }

    println!();
    println!("=== END ===");
}
