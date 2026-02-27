// Linear algebra utilities — pure Rust, no dependencies.
// Dot product, cosine similarity, correlation, magnitude, percentile.

/// Dot product of two equal-length vectors.
pub fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "dot_product: vectors must have same length");
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Euclidean norm (magnitude) of a vector.
pub fn magnitude(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Cosine similarity between two vectors. Returns 0.0 for near-zero magnitude.
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let mag_a = magnitude(a);
    let mag_b = magnitude(b);
    if mag_a < 1e-10 || mag_b < 1e-10 {
        return 0.0;
    }
    dot_product(a, b) / (mag_a * mag_b)
}

/// Pearson correlation coefficient between two equal-length sequences.
/// Returns 0.0 if either sequence has zero variance.
pub fn correlation(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len(), "correlation: sequences must have same length");
    let n = x.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let mean_x: f64 = x.iter().sum::<f64>() / n;
    let mean_y: f64 = y.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    let denom = (var_x * var_y).sqrt();
    if denom < 1e-10 {
        0.0
    } else {
        cov / denom
    }
}

/// Nearest-rank percentile of a slice. `p` is in [0, 100].
#[allow(dead_code)]
pub fn percentile(data: &[f64], p: f64) -> f64 {
    assert!(!data.is_empty(), "percentile: data must not be empty");
    let mut sorted: Vec<f64> = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = (p / 100.0 * (sorted.len() as f64 - 1.0)).round() as usize;
    let idx = idx.min(sorted.len().saturating_sub(1));
    sorted[idx]
}
