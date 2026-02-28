// Phase 17: Weight Spectral Analysis — DFT Analyzer
//
// Loads weight matrices dumped by the train binary and performs DFT analysis
// to measure frequency-domain sparsity. Compares baseline vs harmonic vs
// frozen embedding modes.
//
// Hypothesis: harmonic embeddings cause weight matrices to concentrate energy
// into fewer frequency bands, meaning computation could be skipped via
// band-selective loading (extending wave packet concepts to training).
//
// Zero candle dependency — only uses dft.rs and std.
//
// Usage:
//   cargo run --release --bin analyze

#[path = "dft.rs"]
mod dft;

use dft::rfft;
use std::fs;
use std::path::Path;

/// Discover available modes by scanning the weights/ directory.
/// Returns sorted list with preferred ordering: baseline, harmonic, frozen,
/// curriculum_pre, curriculum, then any others alphabetically.
fn discover_modes() -> Vec<String> {
    let weights_dir = Path::new("weights");
    if !weights_dir.exists() {
        eprintln!("Error: weights/ directory not found. Run training first.");
        std::process::exit(1);
    }

    let mut modes: Vec<String> = fs::read_dir(weights_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|ft| ft.is_dir()).unwrap_or(false))
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect();

    let preferred = ["frozen_standard", "harmonic_heads", "frozen_heads", "baseline", "harmonic", "frozen", "curriculum_pre", "curriculum"];
    modes.sort_by(|a, b| {
        let a_idx = preferred.iter().position(|&p| p == a).unwrap_or(usize::MAX);
        let b_idx = preferred.iter().position(|&p| p == b).unwrap_or(usize::MAX);
        a_idx.cmp(&b_idx).then(a.cmp(b))
    });

    if modes.is_empty() {
        eprintln!("Error: no mode directories found in weights/.");
        std::process::exit(1);
    }

    modes
}

// =============================================================================
// Binary Tensor I/O
// =============================================================================

/// Read a tensor from binary format: [ndims: u32] [dims...: u32] [values...: f32]
fn read_tensor_binary(path: &str) -> (Vec<usize>, Vec<f32>) {
    let bytes = fs::read(path)
        .unwrap_or_else(|e| panic!("Failed to read {path}: {e}"));
    let mut offset = 0;

    let ndims = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
    offset += 4;

    let mut shape = Vec::with_capacity(ndims);
    let mut total = 1usize;
    for _ in 0..ndims {
        let d = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
        shape.push(d);
        total *= d;
        offset += 4;
    }

    let mut values = Vec::with_capacity(total);
    for _ in 0..total {
        let v = f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        values.push(v);
        offset += 4;
    }

    (shape, values)
}

// =============================================================================
// Weight Selection
// =============================================================================

/// Returns true for weight matrices we want to analyze (not biases, layer norms,
/// or embedding tables).
fn is_analysis_target(name: &str) -> bool {
    if !name.ends_with(".weight") {
        return false;
    }
    // Exclude layer norms
    if name.contains("ln_") {
        return false;
    }
    // Exclude embedding tables and reference dumps
    if name.starts_with("wte") || name.starts_with("wpe") || name.starts_with('_') {
        return false;
    }
    true
}

// =============================================================================
// DFT Analysis
// =============================================================================

struct AxisResult {
    n_bands: usize,
    bands_90: usize,
    bands_95: usize,
    bands_99: usize,
    sparsity_pct: f64, // % of bands carrying < 1% of peak band energy
    top_bands: Vec<usize>, // top 3 band indices by energy
}

/// Perform DFT analysis along one axis of a 2D weight matrix.
///
/// column_wise=true: DFT on each column (vectors of length `rows`)
/// column_wise=false: DFT on each row (vectors of length `cols`)
fn analyze_axis(data: &[f32], rows: usize, cols: usize, column_wise: bool) -> AxisResult {
    let (n_vectors, vec_len) = if column_wise {
        (cols, rows)
    } else {
        (rows, cols)
    };

    let n_bands = vec_len / 2 + 1;
    let mut band_energy = vec![0.0f64; n_bands];

    for i in 0..n_vectors {
        // Extract vector along the chosen axis
        let vector: Vec<f64> = if column_wise {
            (0..vec_len).map(|r| data[r * cols + i] as f64).collect()
        } else {
            (0..vec_len).map(|c| data[i * cols + c] as f64).collect()
        };

        let coeffs = rfft(&vector);

        // Accumulate band energy with conjugate symmetry weights
        for (k, c) in coeffs.iter().enumerate() {
            let energy = c.re * c.re + c.im * c.im;
            let weight = if k == 0 || (vec_len % 2 == 0 && k == n_bands - 1) {
                1.0
            } else {
                2.0
            };
            band_energy[k] += weight * energy;
        }
    }

    let total_energy: f64 = band_energy.iter().sum();
    let max_energy: f64 = band_energy.iter().cloned().fold(0.0f64, f64::max);

    // Sort bands by energy (descending) for concentration metric
    let mut sorted_indices: Vec<usize> = (0..n_bands).collect();
    sorted_indices.sort_by(|&a, &b| {
        band_energy[b]
            .partial_cmp(&band_energy[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut cumulative = 0.0;
    let mut bands_90 = n_bands;
    let mut bands_95 = n_bands;
    let mut bands_99 = n_bands;

    if total_energy > 0.0 {
        for (count, &idx) in sorted_indices.iter().enumerate() {
            cumulative += band_energy[idx];
            let frac = cumulative / total_energy;
            if frac >= 0.90 && bands_90 == n_bands {
                bands_90 = count + 1;
            }
            if frac >= 0.95 && bands_95 == n_bands {
                bands_95 = count + 1;
            }
            if frac >= 0.99 && bands_99 == n_bands {
                bands_99 = count + 1;
            }
        }
    }

    // Sparsity: % of bands with < 1% of peak band energy
    let threshold = 0.01 * max_energy;
    let sparse_count = band_energy.iter().filter(|&&e| e < threshold).count();
    let sparsity_pct = if n_bands > 0 {
        (sparse_count as f64) / (n_bands as f64) * 100.0
    } else {
        0.0
    };

    // Top 3 bands by energy
    let top_bands: Vec<usize> = sorted_indices.iter().take(3).copied().collect();

    AxisResult {
        n_bands,
        bands_90,
        bands_95,
        bands_99,
        sparsity_pct,
        top_bands,
    }
}

// =============================================================================
// Per-Mode Analysis
// =============================================================================

struct WeightAnalysis {
    name: String,
    rows: usize,
    cols: usize,
    col: AxisResult,
    row: AxisResult,
}

fn analyze_mode(mode: &str) -> Vec<WeightAnalysis> {
    let dir = format!("weights/{mode}");

    let mut entries: Vec<String> = fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("Cannot read {dir}: {e}"))
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            if let Some(weight_name) = name.strip_suffix(".bin") {
                if is_analysis_target(weight_name) {
                    Some(weight_name.to_string())
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect();

    entries.sort();

    let mut results = Vec::new();
    for name in &entries {
        let path = format!("{dir}/{name}.bin");
        let (shape, data) = read_tensor_binary(&path);
        assert!(
            shape.len() == 2,
            "Expected 2D tensor for {name}, got {}D",
            shape.len()
        );
        let (rows, cols) = (shape[0], shape[1]);

        let col = analyze_axis(&data, rows, cols, true);
        let row = analyze_axis(&data, rows, cols, false);

        results.push(WeightAnalysis {
            name: name.clone(),
            rows,
            cols,
            col,
            row,
        });
    }

    results
}

// =============================================================================
// Summary Helpers
// =============================================================================

/// Average the fraction (count / total) across all weight analyses.
fn avg_fraction(analyses: &[WeightAnalysis], extractor: fn(&WeightAnalysis) -> (usize, usize)) -> f64 {
    if analyses.is_empty() {
        return 0.0;
    }
    let sum: f64 = analyses
        .iter()
        .map(|a| {
            let (count, total) = extractor(a);
            if total > 0 {
                count as f64 / total as f64
            } else {
                0.0
            }
        })
        .sum();
    sum / analyses.len() as f64 * 100.0
}

/// Average a simple f64 metric across all weight analyses.
fn avg_metric(analyses: &[WeightAnalysis], extractor: fn(&WeightAnalysis) -> f64) -> f64 {
    if analyses.is_empty() {
        return 0.0;
    }
    let sum: f64 = analyses.iter().map(extractor).sum();
    sum / analyses.len() as f64
}

// =============================================================================
// Main
// =============================================================================

fn main() {
    println!("\n{}", "=".repeat(60));
    println!("  Phase 18: Weight Spectral Analysis");
    println!("{}", "=".repeat(60));

    // Discover available modes dynamically
    let modes = discover_modes();
    println!("  Modes found: {}", modes.join(", "));

    // Analyze each mode — store as (mode_name, HashMap<weight_name, analysis>)
    let mut all_analyses: Vec<(String, Vec<WeightAnalysis>)> = Vec::new();
    for mode in &modes {
        print!("  Analyzing {mode} weights...");
        let analyses = analyze_mode(mode);
        println!(" {} matrices", analyses.len());
        all_analyses.push((mode.clone(), analyses));
    }

    // Find weight names common to ALL modes (for cross-mode comparison)
    let common_names: Vec<String> = {
        let first_names: Vec<&str> = all_analyses[0].1.iter().map(|a| a.name.as_str()).collect();
        first_names
            .into_iter()
            .filter(|name| {
                all_analyses[1..].iter().all(|(_, analyses)| {
                    analyses.iter().any(|a| a.name == *name)
                })
            })
            .map(|s| s.to_string())
            .collect()
    };
    println!("  Common weight matrices across all modes: {}", common_names.len());

    // Helper to find analysis by name
    let find_analysis = |analyses: &[WeightAnalysis], name: &str| -> Option<usize> {
        analyses.iter().position(|a| a.name == name)
    };

    // Find reference mode for comparisons (prefer frozen_standard, then first mode)
    let ref_idx = all_analyses
        .iter()
        .position(|(m, _)| m == "frozen_standard")
        .or_else(|| all_analyses.iter().position(|(m, _)| m == "baseline"))
        .unwrap_or(0);
    let baseline_idx = Some(ref_idx);

    // =========================================================================
    // Per-Matrix Results (common weights only)
    // =========================================================================
    println!("\n{}", "=".repeat(60));
    println!("  PER-MATRIX RESULTS (common weights)");
    println!("{}", "=".repeat(60));

    for name in &common_names {
        // Get the analysis for the first mode that has it (they all do, it's common)
        let ref_a = &all_analyses[0].1[find_analysis(&all_analyses[0].1, name).unwrap()];
        let rows = ref_a.rows;
        let cols = ref_a.cols;

        println!("\n--- {name} ({rows}x{cols}) ---");

        // Column-wise DFT
        let col_n_bands = ref_a.col.n_bands;
        println!(
            "  Column-wise DFT ({col_n_bands} bands from {rows}-element columns):"
        );

        print!("    Bands for 90% energy: ");
        for (mode, analyses) in &all_analyses {
            if let Some(idx) = find_analysis(analyses, name) {
                print!(" {mode}={:<4}", analyses[idx].col.bands_90);
            }
        }
        println!();

        print!("    Bands for 95% energy: ");
        for (mode, analyses) in &all_analyses {
            if let Some(idx) = find_analysis(analyses, name) {
                print!(" {mode}={:<4}", analyses[idx].col.bands_95);
            }
        }
        println!();

        print!("    Band sparsity (<1%):  ");
        for (mode, analyses) in &all_analyses {
            if let Some(idx) = find_analysis(analyses, name) {
                print!(" {mode}={:.1}%", analyses[idx].col.sparsity_pct);
            }
        }
        println!();

        print!("    Top-3 bands:          ");
        for (mode, analyses) in &all_analyses {
            if let Some(idx) = find_analysis(analyses, name) {
                print!(" {mode}={:?}", analyses[idx].col.top_bands);
            }
        }
        println!();

        // Row-wise DFT
        let row_n_bands = ref_a.row.n_bands;
        println!(
            "  Row-wise DFT ({row_n_bands} bands from {cols}-element rows):"
        );

        print!("    Bands for 90% energy: ");
        for (mode, analyses) in &all_analyses {
            if let Some(idx) = find_analysis(analyses, name) {
                print!(" {mode}={:<4}", analyses[idx].row.bands_90);
            }
        }
        println!();

        print!("    Band sparsity (<1%):  ");
        for (mode, analyses) in &all_analyses {
            if let Some(idx) = find_analysis(analyses, name) {
                print!(" {mode}={:.1}%", analyses[idx].row.sparsity_pct);
            }
        }
        println!();
    }

    // =========================================================================
    // Summary
    // =========================================================================
    println!("\n{}", "=".repeat(60));
    println!("  SUMMARY");
    println!("{}", "=".repeat(60));

    // Compute metrics using only common weights for fair comparison
    let common_analyses: Vec<(String, Vec<&WeightAnalysis>)> = all_analyses
        .iter()
        .map(|(mode, analyses)| {
            let common: Vec<&WeightAnalysis> = common_names
                .iter()
                .filter_map(|name| {
                    find_analysis(analyses, name).map(|idx| &analyses[idx])
                })
                .collect();
            (mode.clone(), common)
        })
        .collect();

    let avg_frac_common = |analyses: &[&WeightAnalysis], extract: fn(&WeightAnalysis) -> (usize, usize)| -> f64 {
        if analyses.is_empty() { return 0.0; }
        let sum: f64 = analyses.iter().map(|a| {
            let (count, total) = extract(a);
            if total > 0 { count as f64 / total as f64 } else { 0.0 }
        }).sum();
        sum / analyses.len() as f64 * 100.0
    };

    let avg_met_common = |analyses: &[&WeightAnalysis], extract: fn(&WeightAnalysis) -> f64| -> f64 {
        if analyses.is_empty() { return 0.0; }
        let sum: f64 = analyses.iter().map(|a| extract(a)).sum();
        sum / analyses.len() as f64
    };

    let col90: Vec<f64> = common_analyses.iter().map(|(_, a)| avg_frac_common(a, |w| (w.col.bands_90, w.col.n_bands))).collect();
    let col95: Vec<f64> = common_analyses.iter().map(|(_, a)| avg_frac_common(a, |w| (w.col.bands_95, w.col.n_bands))).collect();
    let col_sp: Vec<f64> = common_analyses.iter().map(|(_, a)| avg_met_common(a, |w| w.col.sparsity_pct)).collect();
    let row90: Vec<f64> = common_analyses.iter().map(|(_, a)| avg_frac_common(a, |w| (w.row.bands_90, w.row.n_bands))).collect();
    let row_sp: Vec<f64> = common_analyses.iter().map(|(_, a)| avg_met_common(a, |w| w.row.sparsity_pct)).collect();

    // Find max mode name length for alignment
    let max_name = all_analyses.iter().map(|(m, _)| m.len()).max().unwrap_or(8);

    // Column-wise: bands for 90% energy
    println!("\n  Average bands for 90% energy (column-wise, as % of available bands):");
    for (j, (mode, _)) in all_analyses.iter().enumerate() {
        let val = col90[j];
        if let Some(bi) = baseline_idx {
            if j != bi && col90[bi] > 0.0 {
                let reduction = (1.0 - val / col90[bi]) * 100.0;
                println!("    {:<width$} {:.1}%  ({reduction:+.1}% vs baseline)", mode, val, width = max_name + 1);
                continue;
            }
        }
        println!("    {:<width$} {:.1}%", mode, val, width = max_name + 1);
    }

    // Column-wise: bands for 95% energy
    println!("\n  Average bands for 95% energy (column-wise, as % of available bands):");
    for (j, (mode, _)) in all_analyses.iter().enumerate() {
        let val = col95[j];
        if let Some(bi) = baseline_idx {
            if j != bi && col95[bi] > 0.0 {
                let reduction = (1.0 - val / col95[bi]) * 100.0;
                println!("    {:<width$} {:.1}%  ({reduction:+.1}% vs baseline)", mode, val, width = max_name + 1);
                continue;
            }
        }
        println!("    {:<width$} {:.1}%", mode, val, width = max_name + 1);
    }

    // Column-wise: band sparsity
    println!("\n  Average band sparsity (column-wise, % of bands carrying <1% of peak energy):");
    for (j, (mode, _)) in all_analyses.iter().enumerate() {
        let val = col_sp[j];
        if let Some(bi) = baseline_idx {
            if j != bi && col_sp[bi] > 0.0 {
                let ratio = val / col_sp[bi];
                println!("    {:<width$} {:.1}%  ({ratio:.2}x vs baseline)", mode, val, width = max_name + 1);
                continue;
            }
        }
        println!("    {:<width$} {:.1}%", mode, val, width = max_name + 1);
    }

    // Row-wise: bands for 90% energy
    println!("\n  Average bands for 90% energy (row-wise, as % of available bands):");
    for (j, (mode, _)) in all_analyses.iter().enumerate() {
        let val = row90[j];
        if let Some(bi) = baseline_idx {
            if j != bi && row90[bi] > 0.0 {
                let reduction = (1.0 - val / row90[bi]) * 100.0;
                println!("    {:<width$} {:.1}%  ({reduction:+.1}% vs baseline)", mode, val, width = max_name + 1);
                continue;
            }
        }
        println!("    {:<width$} {:.1}%", mode, val, width = max_name + 1);
    }

    // Row-wise: band sparsity
    println!("\n  Average band sparsity (row-wise, % of bands carrying <1% of peak energy):");
    for (j, (mode, _)) in all_analyses.iter().enumerate() {
        let val = row_sp[j];
        if let Some(bi) = baseline_idx {
            if j != bi && row_sp[bi] > 0.0 {
                let ratio = val / row_sp[bi];
                println!("    {:<width$} {:.1}%  ({ratio:.2}x vs baseline)", mode, val, width = max_name + 1);
                continue;
            }
        }
        println!("    {:<width$} {:.1}%", mode, val, width = max_name + 1);
    }

    // =========================================================================
    // Conclusion
    // =========================================================================
    println!("\n{}", "=".repeat(60));
    println!("  CONCLUSION");
    println!("{}", "=".repeat(60));

    if let Some(bi) = baseline_idx {
        let base_90 = col90[bi];
        let base_sp = col_sp[bi];
        let ref_name = &all_analyses[bi].0;

        println!();
        println!("  Reference mode: {ref_name}");
        for (j, (mode, _)) in all_analyses.iter().enumerate() {
            if j == bi {
                continue;
            }
            let reduction = if base_90 > 0.0 {
                (1.0 - col90[j] / base_90) * 100.0
            } else {
                0.0
            };
            let sp_ratio = if base_sp > 0.0 {
                col_sp[j] / base_sp
            } else {
                1.0
            };

            println!("  {mode}:");
            if reduction > 0.0 {
                println!(
                    "    Concentrates 90% energy into {:.1}% fewer bands than {ref_name}.",
                    reduction
                );
            } else {
                println!("    Does NOT reduce band concentration vs {ref_name}.");
            }
            if sp_ratio > 1.0 {
                println!(
                    "    Band sparsity is {:.2}x higher ({:.1}% vs {:.1}% {ref_name}).",
                    sp_ratio, col_sp[j], base_sp
                );
            } else {
                println!("    Band sparsity is NOT higher than {ref_name}.");
            }
        }
    } else {
        println!();
        println!("  No reference mode found — cannot compute relative comparisons.");
    }

    println!();
    println!("{}", "=".repeat(60));
}
