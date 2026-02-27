// Harmonic embedding generators — pure Rust, no dependencies.
//
// Two variants matching the project's conventions:
// 1. Cosine-only: [cos(theta), cos(2*theta), ..., cos(N*theta)]
//    Used in sweep-test and core wave mechanics.
// 2. Cosine+sine: [cos(theta), sin(theta), cos(2*theta), sin(2*theta), ...]
//    Used in the harmonic transformer's HarmonicEmbedding class.

use std::f64::consts::PI;

/// Cosine-only harmonic embedding.
/// Matches sweep-test/src/main.rs harmonic_embedding().
pub fn harmonic_embedding_cos(theta: f64, n_harmonics: usize) -> Vec<f64> {
    (1..=n_harmonics)
        .map(|n| (n as f64 * theta).cos())
        .collect()
}

/// Cosine+sine harmonic embedding with 1/sqrt(N) scaling.
/// Matches Python HarmonicEmbedding class.
/// Output dimension: 2 * n_harmonics.
pub fn harmonic_embedding_cossin(theta: f64, n_harmonics: usize) -> Vec<f64> {
    let scale = if n_harmonics > 0 {
        1.0 / (n_harmonics as f64).sqrt()
    } else {
        1.0
    };
    let mut result = Vec::with_capacity(2 * n_harmonics);
    for n in 1..=n_harmonics {
        let angle = n as f64 * theta;
        result.push(angle.cos() * scale);
        result.push(angle.sin() * scale);
    }
    result
}

/// Convert degrees to radians.
pub fn deg_to_rad(deg: f64) -> f64 {
    deg * PI / 180.0
}
