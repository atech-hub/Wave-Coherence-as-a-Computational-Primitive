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

const MODES: [&str; 3] = ["baseline", "harmonic", "frozen"];

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
    println!("  Phase 17: Weight Spectral Analysis");
    println!("{}", "=".repeat(60));

    // Check all mode directories exist
    for mode in &MODES {
        let dir = format!("weights/{mode}");
        if !Path::new(&dir).exists() {
            eprintln!(
                "Error: {dir}/ not found. Run 'cargo run --release --bin train' first."
            );
            std::process::exit(1);
        }
    }

    // Analyze each mode
    let mut all_analyses: Vec<(&str, Vec<WeightAnalysis>)> = Vec::new();
    for mode in &MODES {
        print!("  Analyzing {mode} weights...");
        let analyses = analyze_mode(mode);
        println!(" {} matrices", analyses.len());
        all_analyses.push((mode, analyses));
    }

    // Verify all modes have the same weight names in the same order
    let n_matrices = all_analyses[0].1.len();
    for (mode, analyses) in &all_analyses {
        assert_eq!(
            analyses.len(),
            n_matrices,
            "{mode} has {} matrices, expected {n_matrices}",
            analyses.len()
        );
    }
    for i in 0..n_matrices {
        let name = &all_analyses[0].1[i].name;
        for (mode, analyses) in &all_analyses[1..] {
            assert_eq!(
                &analyses[i].name, name,
                "{mode} matrix {i} is '{}', expected '{name}'",
                analyses[i].name
            );
        }
    }

    // =========================================================================
    // Per-Matrix Results
    // =========================================================================
    println!("\n{}", "=".repeat(60));
    println!("  PER-MATRIX RESULTS");
    println!("{}", "=".repeat(60));

    for i in 0..n_matrices {
        let name = &all_analyses[0].1[i].name;
        let rows = all_analyses[0].1[i].rows;
        let cols = all_analyses[0].1[i].cols;

        println!("\n--- {name} ({rows}x{cols}) ---");

        // Column-wise DFT
        let col_n_bands = all_analyses[0].1[i].col.n_bands;
        println!(
            "  Column-wise DFT ({col_n_bands} bands from {rows}-element columns):"
        );

        print!("    Bands for 90% energy: ");
        for (mode, analyses) in &all_analyses {
            print!(" {mode}={:<4}", analyses[i].col.bands_90);
        }
        println!();

        print!("    Bands for 95% energy: ");
        for (mode, analyses) in &all_analyses {
            print!(" {mode}={:<4}", analyses[i].col.bands_95);
        }
        println!();

        print!("    Bands for 99% energy: ");
        for (mode, analyses) in &all_analyses {
            print!(" {mode}={:<4}", analyses[i].col.bands_99);
        }
        println!();

        print!("    Band sparsity (<1%):  ");
        for (mode, analyses) in &all_analyses {
            print!(" {mode}={:.1}%", analyses[i].col.sparsity_pct);
        }
        println!();

        print!("    Top-3 bands:          ");
        for (mode, analyses) in &all_analyses {
            print!(" {mode}={:?}", analyses[i].col.top_bands);
        }
        println!();

        // Row-wise DFT
        let row_n_bands = all_analyses[0].1[i].row.n_bands;
        println!(
            "  Row-wise DFT ({row_n_bands} bands from {cols}-element rows):"
        );

        print!("    Bands for 90% energy: ");
        for (mode, analyses) in &all_analyses {
            print!(" {mode}={:<4}", analyses[i].row.bands_90);
        }
        println!();

        print!("    Band sparsity (<1%):  ");
        for (mode, analyses) in &all_analyses {
            print!(" {mode}={:.1}%", analyses[i].row.sparsity_pct);
        }
        println!();
    }

    // =========================================================================
    // Summary
    // =========================================================================
    println!("\n{}", "=".repeat(60));
    println!("  SUMMARY");
    println!("{}", "=".repeat(60));

    // Column-wise: bands for 90% energy (as % of total bands)
    println!("\n  Average bands for 90% energy (column-wise, as % of available bands):");
    let baseline_90 = avg_fraction(&all_analyses[0].1, |a| (a.col.bands_90, a.col.n_bands));
    let harmonic_90 = avg_fraction(&all_analyses[1].1, |a| (a.col.bands_90, a.col.n_bands));
    let frozen_90 = avg_fraction(&all_analyses[2].1, |a| (a.col.bands_90, a.col.n_bands));

    println!("    baseline: {baseline_90:.1}%");
    if baseline_90 > 0.0 {
        let h_reduction = (1.0 - harmonic_90 / baseline_90) * 100.0;
        let f_reduction = (1.0 - frozen_90 / baseline_90) * 100.0;
        println!("    harmonic: {harmonic_90:.1}%  ({h_reduction:+.1}% vs baseline)");
        println!("    frozen:   {frozen_90:.1}%  ({f_reduction:+.1}% vs baseline)");
    } else {
        println!("    harmonic: {harmonic_90:.1}%");
        println!("    frozen:   {frozen_90:.1}%");
    }

    // Column-wise: bands for 95% energy
    println!("\n  Average bands for 95% energy (column-wise, as % of available bands):");
    let baseline_95 = avg_fraction(&all_analyses[0].1, |a| (a.col.bands_95, a.col.n_bands));
    let harmonic_95 = avg_fraction(&all_analyses[1].1, |a| (a.col.bands_95, a.col.n_bands));
    let frozen_95 = avg_fraction(&all_analyses[2].1, |a| (a.col.bands_95, a.col.n_bands));

    println!("    baseline: {baseline_95:.1}%");
    if baseline_95 > 0.0 {
        let h_reduction = (1.0 - harmonic_95 / baseline_95) * 100.0;
        let f_reduction = (1.0 - frozen_95 / baseline_95) * 100.0;
        println!("    harmonic: {harmonic_95:.1}%  ({h_reduction:+.1}% vs baseline)");
        println!("    frozen:   {frozen_95:.1}%  ({f_reduction:+.1}% vs baseline)");
    } else {
        println!("    harmonic: {harmonic_95:.1}%");
        println!("    frozen:   {frozen_95:.1}%");
    }

    // Column-wise: band sparsity
    println!("\n  Average band sparsity (column-wise, % of bands carrying <1% of peak energy):");
    let baseline_sp = avg_metric(&all_analyses[0].1, |a| a.col.sparsity_pct);
    let harmonic_sp = avg_metric(&all_analyses[1].1, |a| a.col.sparsity_pct);
    let frozen_sp = avg_metric(&all_analyses[2].1, |a| a.col.sparsity_pct);

    println!("    baseline: {baseline_sp:.1}%");
    if baseline_sp > 0.0 {
        let h_ratio = harmonic_sp / baseline_sp;
        let f_ratio = frozen_sp / baseline_sp;
        println!("    harmonic: {harmonic_sp:.1}%  ({h_ratio:.2}x vs baseline)");
        println!("    frozen:   {frozen_sp:.1}%  ({f_ratio:.2}x vs baseline)");
    } else {
        println!("    harmonic: {harmonic_sp:.1}%");
        println!("    frozen:   {frozen_sp:.1}%");
    }

    // Row-wise summary
    println!("\n  Average bands for 90% energy (row-wise, as % of available bands):");
    let baseline_r90 = avg_fraction(&all_analyses[0].1, |a| (a.row.bands_90, a.row.n_bands));
    let harmonic_r90 = avg_fraction(&all_analyses[1].1, |a| (a.row.bands_90, a.row.n_bands));
    let frozen_r90 = avg_fraction(&all_analyses[2].1, |a| (a.row.bands_90, a.row.n_bands));

    println!("    baseline: {baseline_r90:.1}%");
    if baseline_r90 > 0.0 {
        let h_reduction = (1.0 - harmonic_r90 / baseline_r90) * 100.0;
        let f_reduction = (1.0 - frozen_r90 / baseline_r90) * 100.0;
        println!("    harmonic: {harmonic_r90:.1}%  ({h_reduction:+.1}% vs baseline)");
        println!("    frozen:   {frozen_r90:.1}%  ({f_reduction:+.1}% vs baseline)");
    } else {
        println!("    harmonic: {harmonic_r90:.1}%");
        println!("    frozen:   {frozen_r90:.1}%");
    }

    let baseline_rsp = avg_metric(&all_analyses[0].1, |a| a.row.sparsity_pct);
    let harmonic_rsp = avg_metric(&all_analyses[1].1, |a| a.row.sparsity_pct);
    let frozen_rsp = avg_metric(&all_analyses[2].1, |a| a.row.sparsity_pct);

    println!("\n  Average band sparsity (row-wise, % of bands carrying <1% of peak energy):");
    println!("    baseline: {baseline_rsp:.1}%");
    if baseline_rsp > 0.0 {
        let h_ratio = harmonic_rsp / baseline_rsp;
        let f_ratio = frozen_rsp / baseline_rsp;
        println!("    harmonic: {harmonic_rsp:.1}%  ({h_ratio:.2}x vs baseline)");
        println!("    frozen:   {frozen_rsp:.1}%  ({f_ratio:.2}x vs baseline)");
    } else {
        println!("    harmonic: {harmonic_rsp:.1}%");
        println!("    frozen:   {frozen_rsp:.1}%");
    }

    // =========================================================================
    // Conclusion
    // =========================================================================
    println!("\n{}", "=".repeat(60));
    println!("  CONCLUSION");
    println!("{}", "=".repeat(60));

    let col_reduction = if baseline_90 > 0.0 {
        (1.0 - harmonic_90 / baseline_90) * 100.0
    } else {
        0.0
    };

    let sparsity_ratio = if baseline_sp > 0.0 {
        harmonic_sp / baseline_sp
    } else {
        1.0
    };

    println!();
    if col_reduction > 0.0 {
        println!(
            "  Harmonic embeddings concentrate 90% of weight energy into {:.1}% fewer",
            col_reduction
        );
        println!("  frequency bands than baseline.");
    } else {
        println!("  Harmonic embeddings did NOT reduce band concentration vs baseline.");
    }

    if sparsity_ratio > 1.0 {
        println!(
            "  Band sparsity is {:.2}x higher with harmonic embeddings,",
            sparsity_ratio
        );
        println!(
            "  meaning {:.1}% of bands carry negligible energy (vs {:.1}% baseline).",
            harmonic_sp, baseline_sp
        );
        println!();
        println!("  This supports the efficiency argument: harmonic models may need");
        println!("  fewer frequency bands in weight computation, extending the wave");
        println!("  packet selective loading concept from retrieval into training.");
    } else {
        println!("  Band sparsity is NOT higher with harmonic embeddings.");
        println!("  The efficiency argument is not supported by this data.");
    }

    println!();
    println!("{}", "=".repeat(60));
}
