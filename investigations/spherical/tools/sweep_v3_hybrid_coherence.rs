/// Sweep v3: Hybrid Coherence on Trained Embeddings
///
/// The hybrid formula:
///   H(a, b, n, l, β) = cos(n × Δφ) × [(1-β) + β × P_l(cos Δθ)]
///
/// φ = phase angle (azimuth) — atan2(sin, cos) of each band pair
/// θ = elevation — derived from MAGNITUDE of each band pair
///     magnitude r = sqrt(w[2k]² + w[2k+1]²), mapped to [0, π]
///
/// Key question: does the magnitude (elevation) carry information that
/// reveals relationships the phase-only circle sweep misses?
///
/// Control: frozen embeddings have r ≈ 1.0 everywhere (pure cos/sin).
///          No elevation variation = no new detections expected.
/// Signal:  trained embeddings may have variable r.
///          Elevation variation = potential new detections.
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
    if bytes.len() < 4 {
        eprintln!("  ERROR: file too small: {}", path);
        return (vec![], vec![]);
    }
    let ndims = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let mut dims = Vec::with_capacity(ndims);
    let mut offset = 4;
    for _ in 0..ndims {
        if offset + 4 > bytes.len() {
            eprintln!("  ERROR: truncated dims in {}", path);
            return (vec![], vec![]);
        }
        dims.push(u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize);
        offset += 4;
    }
    let n_values: usize = dims.iter().product();
    let mut values = Vec::with_capacity(n_values);
    for i in 0..n_values {
        let start = offset + i * 4;
        if start + 4 > bytes.len() {
            eprintln!("  ERROR: truncated values at index {} in {}", i, path);
            break;
        }
        values.push(f32::from_le_bytes(bytes[start..start + 4].try_into().unwrap()));
    }
    (dims, values)
}

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

fn hybrid_gated(delta_phi: f64, delta_theta: f64, n: usize, l: usize, beta: f64) -> f64 {
    let gate = (1.0 - beta) + beta * legendre(l, delta_theta.cos());
    circle_coherence(delta_phi, n) * gate
}

// ── Phase + Magnitude extraction ──

/// Extract phase (azimuth) and magnitude per band per token.
/// phases[band][token], magnitudes[band][token]
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

/// Map magnitudes to elevation angles [0, π] per band.
/// Uses per-band min/max normalization.
fn magnitudes_to_elevation(magnitudes: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n_bands = magnitudes.len();
    let vocab_size = if n_bands > 0 { magnitudes[0].len() } else { return vec![]; };

    let mut elevations = vec![vec![0.0f64; vocab_size]; n_bands];

    for k in 0..n_bands {
        let min_r = magnitudes[k].iter().cloned().fold(f64::INFINITY, f64::min);
        let max_r = magnitudes[k].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max_r - min_r;

        for c in 0..vocab_size {
            if range > 1e-12 {
                // Map [min_r, max_r] → [0, π]
                elevations[k][c] = PI * (magnitudes[k][c] - min_r) / range;
            } else {
                // No variation → equator (π/2)
                elevations[k][c] = PI / 2.0;
            }
        }
    }

    elevations
}

// ── Magnitude statistics ──

struct MagStats {
    mean: f64,
    std: f64,
    min: f64,
    max: f64,
    coeff_of_variation: f64, // std/mean — key metric for "is there information?"
}

fn mag_stats(mags: &[f64]) -> MagStats {
    let n = mags.len() as f64;
    let mean = mags.iter().sum::<f64>() / n;
    let variance = mags.iter().map(|m| (m - mean).powi(2)).sum::<f64>() / n;
    let std = variance.sqrt();
    let min = mags.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = mags.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let cv = if mean.abs() > 1e-12 { std / mean } else { 0.0 };
    MagStats { mean, std, min, max, coeff_of_variation: cv }
}

// ── Sweep engine ──

struct SweepResult {
    name: String,
    vocab_size: usize,
    n_bands: usize,
    // Magnitude statistics
    avg_mag_cv: f64,
    bands_with_variation: usize,
    // Detection counts
    circle_strong: usize,  // |cos(nΔφ)| > 0.9
    hybrid_rescued: usize, // circle < 0.5 AND hybrid > 0.7
    hybrid_lost: usize,    // circle > 0.9 AND hybrid < 0.5
    // Best rescues
    top_rescues: Vec<(usize, usize, usize, usize, usize, f64, f64)>,
    // (band, token_a, token_b, n, l, circle_coh, hybrid_coh)
}

fn sweep_mode(
    name: &str,
    phases: &[Vec<f64>],
    elevations: &[Vec<f64>],
    magnitudes: &[Vec<f64>],
    n_max: usize,
    l_max: usize,
    beta: f64,
) -> SweepResult {
    let n_bands = phases.len();
    let vocab_size = if n_bands > 0 { phases[0].len() } else { 0 };

    // Magnitude statistics
    let mut sum_cv = 0.0f64;
    let mut bands_with_var = 0usize;
    for k in 0..n_bands {
        let stats = mag_stats(&magnitudes[k]);
        sum_cv += stats.coeff_of_variation;
        if stats.coeff_of_variation > 0.01 { bands_with_var += 1; }
    }
    let avg_cv = sum_cv / n_bands as f64;

    // Sweep all pairs × all harmonics
    let mut circle_strong = 0usize;
    let mut hybrid_rescued = 0usize;
    let mut hybrid_lost = 0usize;
    let mut top_rescues: Vec<(usize, usize, usize, usize, usize, f64, f64)> = Vec::new();

    for k in 0..n_bands {
        for i in 0..vocab_size {
            for j in (i + 1)..vocab_size {
                let dphi = phases[k][i] - phases[k][j];
                let dtheta = elevations[k][i] - elevations[k][j];

                // Best circle coherence across n
                let mut best_circle = 0.0f64;
                let mut best_circle_n = 1;
                for n in 1..=n_max {
                    let c = circle_coherence(dphi, n).abs();
                    if c > best_circle {
                        best_circle = c;
                        best_circle_n = n;
                    }
                }

                if best_circle > 0.9 { circle_strong += 1; }

                // Best hybrid coherence across (n, l)
                let mut best_hybrid = 0.0f64;
                let mut best_hybrid_n = 1;
                let mut best_hybrid_l = 1;
                for n in 1..=n_max {
                    for l in 1..=l_max {
                        let h = hybrid_gated(dphi, dtheta, n, l, beta).abs();
                        if h > best_hybrid {
                            best_hybrid = h;
                            best_hybrid_n = n;
                            best_hybrid_l = l;
                        }
                    }
                }

                // Circle weak but hybrid strong → RESCUE
                if best_circle < 0.5 && best_hybrid > 0.7 {
                    hybrid_rescued += 1;
                    if top_rescues.len() < 20 || best_hybrid > top_rescues.last().map(|r| r.6).unwrap_or(0.0) {
                        top_rescues.push((k, i, j, best_hybrid_n, best_hybrid_l, best_circle, best_hybrid));
                        top_rescues.sort_by(|a, b| b.6.partial_cmp(&a.6).unwrap());
                        top_rescues.truncate(20);
                    }
                }

                // Circle strong but hybrid weak → LOST
                if best_circle > 0.9 && best_hybrid < 0.5 {
                    hybrid_lost += 1;
                }
            }
        }
    }

    SweepResult {
        name: name.to_string(),
        vocab_size,
        n_bands,
        avg_mag_cv: avg_cv,
        bands_with_variation: bands_with_var,
        circle_strong,
        hybrid_rescued,
        hybrid_lost,
        top_rescues,
    }
}

// ── Main ──

fn main() {
    println!("=== Sweep v3: Hybrid Coherence on Trained Embeddings ===");
    println!("  H(a,b,n,l,β) = cos(n×Δφ) × [(1-β) + β × P_l(cos Δθ)]");
    println!("  φ = phase angle (azimuth), θ = elevation from magnitude");
    println!();

    let base = "../experiments/phase17-weight-analysis/weights";
    let modes: &[(&str, String)] = &[
        ("Frozen",   format!("{}/frozen/_harmonic_table.bin", base)),
        ("Harmonic", format!("{}/harmonic/wte.weight.bin", base)),
        ("Baseline", format!("{}/baseline/wte.weight.bin", base)),
    ];

    let n_max = 15;
    let l_max = 6;
    let beta = 0.4;

    println!("  Parameters: n=1..{}, l=1..{}, β={}", n_max, l_max, beta);
    println!();

    let mut results: Vec<SweepResult> = Vec::new();

    for (name, path) in modes {
        println!("--- Loading {} from {} ---", name, path);

        let (dims, values) = read_tensor(path);
        if dims.is_empty() {
            println!("  SKIPPED\n");
            continue;
        }

        if dims.len() != 2 || dims[1] % 2 != 0 {
            println!("  SKIPPED (unexpected shape {:?})\n", dims);
            continue;
        }

        println!("  Shape: {:?} ({} tokens × {} bands)", dims, dims[0], dims[1] / 2);

        let (phases, magnitudes) = extract_phases_and_magnitudes(&dims, &values);
        let elevations = magnitudes_to_elevation(&magnitudes);
        let n_bands = phases.len();
        let vocab_size = dims[0];

        // ── Magnitude analysis ──
        println!("\n  Magnitude statistics (is there elevation information?):");
        println!("  {:>5}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}",
                 "Band", "Mean", "Std", "Min", "Max", "CV%");
        println!("  {:>5}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}",
                 "----", "----", "---", "---", "---", "---");

        let show_bands: Vec<usize> = {
            let mut b: Vec<usize> = (0..8.min(n_bands)).collect();
            if n_bands > 16 { b.push(n_bands / 2); }
            if n_bands > 2 { b.push(n_bands - 1); }
            b
        };

        let mut total_cv = 0.0f64;
        let mut bands_with_var = 0usize;
        for k in 0..n_bands {
            let s = mag_stats(&magnitudes[k]);
            total_cv += s.coeff_of_variation;
            if s.coeff_of_variation > 0.01 { bands_with_var += 1; }
            if show_bands.contains(&k) {
                println!("  {:>5}  {:>8.4}  {:>8.4}  {:>8.4}  {:>8.4}  {:>7.2}%",
                         k, s.mean, s.std, s.min, s.max,
                         s.coeff_of_variation * 100.0);
            }
        }

        let avg_cv = total_cv / n_bands as f64;
        println!("  Average CV: {:.4}%", avg_cv * 100.0);
        println!("  Bands with CV > 1%: {} / {}", bands_with_var, n_bands);

        if avg_cv < 0.005 {
            println!("  --> LOW VARIATION: magnitudes nearly constant.");
            println!("      Elevation dimension carries minimal information.");
        } else {
            println!("  --> SIGNIFICANT VARIATION: elevation dimension has signal.");
        }
        println!();

        // ── Elevation distribution ──
        println!("  Elevation distribution (band 0, first 10 tokens):");
        print!("   ");
        for c in 0..10.min(vocab_size) {
            print!(" θ{}={:.0}°", c, elevations[0][c] * 180.0 / PI);
        }
        println!();

        // ── Run hybrid sweep ──
        let total_pairs = vocab_size * (vocab_size - 1) / 2 * n_bands;
        println!("  Running hybrid sweep: {} pairs × {} (n,l) combos...",
                 total_pairs, n_max * l_max);

        let result = sweep_mode(name, &phases, &elevations, &magnitudes, n_max, l_max, beta);

        println!("  Done.\n");

        // ── Results ──
        println!("  Results for {}:", name);
        println!("    Circle strong (|c|>0.9):  {} ({:.2}% of pairs)",
                 result.circle_strong,
                 100.0 * result.circle_strong as f64 / total_pairs as f64);
        println!("    Hybrid rescued (c<0.5, H>0.7): {} ({:.4}% of pairs)",
                 result.hybrid_rescued,
                 100.0 * result.hybrid_rescued as f64 / total_pairs as f64);
        println!("    Hybrid lost (c>0.9, H<0.5):    {} ({:.4}% of pairs)",
                 result.hybrid_lost,
                 100.0 * result.hybrid_lost as f64 / total_pairs as f64);
        println!();

        if !result.top_rescues.is_empty() {
            println!("    Top rescues (circle weak → hybrid strong):");
            println!("    {:>5}  {:>5}  {:>5}  {:>4}  {:>4}  {:>8}  {:>8}",
                     "Band", "TokA", "TokB", "n", "l", "Circle", "Hybrid");
            println!("    {:>5}  {:>5}  {:>5}  {:>4}  {:>4}  {:>8}  {:>8}",
                     "----", "----", "----", "--", "--", "------", "------");
            for &(band, ta, tb, n, l, cc, hc) in &result.top_rescues {
                println!("    {:>5}  {:>5}  {:>5}  {:>4}  {:>4}  {:>8.4}  {:>8.4}",
                         band, ta, tb, n, l, cc, hc);
            }
            println!();
        }

        results.push(result);
    }

    // ══════════════════════════════════════════════════════════════
    // COMPARATIVE VERDICT
    // ══════════════════════════════════════════════════════════════
    println!("=== COMPARATIVE VERDICT ===\n");

    println!("  {:>10}  {:>8}  {:>8}  {:>10}  {:>10}  {:>8}",
             "Mode", "Mag CV%", "Bands>1%", "CircStrong", "Rescued", "Lost");
    println!("  {:>10}  {:>8}  {:>8}  {:>10}  {:>10}  {:>8}",
             "----", "------", "--------", "----------", "-------", "----");

    for r in &results {
        println!("  {:>10}  {:>7.3}%  {:>8}  {:>10}  {:>10}  {:>8}",
                 r.name, r.avg_mag_cv * 100.0, r.bands_with_variation,
                 r.circle_strong, r.hybrid_rescued, r.hybrid_lost);
    }
    println!();

    // Interpret
    let frozen = results.iter().find(|r| r.name == "Frozen");
    let harmonic = results.iter().find(|r| r.name == "Harmonic");
    let baseline = results.iter().find(|r| r.name == "Baseline");

    if let (Some(f), Some(h), Some(b)) = (frozen, harmonic, baseline) {
        println!("  MAGNITUDE VARIATION:");
        if f.avg_mag_cv < 0.01 && (h.avg_mag_cv > 0.01 || b.avg_mag_cv > 0.01) {
            println!("    Frozen: flat (CV={:.3}%) — no elevation signal (expected)", f.avg_mag_cv * 100.0);
            println!("    Trained: variable — training creates magnitude structure");
        } else if f.avg_mag_cv < 0.01 && h.avg_mag_cv < 0.01 && b.avg_mag_cv < 0.01 {
            println!("    ALL modes flat — magnitude carries no information.");
            println!("    Elevation dimension is empty in current embeddings.");
        } else {
            println!("    Frozen CV={:.3}%, Harmonic CV={:.3}%, Baseline CV={:.3}%",
                     f.avg_mag_cv * 100.0, h.avg_mag_cv * 100.0, b.avg_mag_cv * 100.0);
        }
        println!();

        println!("  RESCUE ANALYSIS:");
        if b.hybrid_rescued > f.hybrid_rescued {
            println!("    Baseline has MORE rescues than Frozen ({} vs {})",
                     b.hybrid_rescued, f.hybrid_rescued);
            println!("    → Training creates elevation structure that reveals new relationships.");
        } else if f.hybrid_rescued >= b.hybrid_rescued && f.hybrid_rescued > 0 {
            println!("    Frozen has as many or more rescues ({} vs {})",
                     f.hybrid_rescued, b.hybrid_rescued);
            println!("    → Rescues come from the gating mechanism, not from trained magnitudes.");
        } else if f.hybrid_rescued == 0 && b.hybrid_rescued == 0 {
            println!("    No rescues in ANY mode.");
            println!("    → The elevation dimension (from magnitude) does not reveal");
            println!("      new relationships beyond what the circle already detects.");
        }
        println!();

        println!("  LOSS ANALYSIS:");
        let total_lost = f.hybrid_lost + h.hybrid_lost + b.hybrid_lost;
        if total_lost == 0 {
            println!("    ZERO losses across all modes — hybrid preserves all circle detections.");
        } else {
            println!("    Some losses detected — hybrid gate may be too aggressive at β={}.", beta);
        }
    }

    println!();
    println!("=== END ===");
}
