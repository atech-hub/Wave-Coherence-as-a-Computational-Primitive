// Harmonic Construction — pure math from Phase 4.
//
// Vector construction operations: interpolation, fractional positions,
// chimera (mixed harmonic bands). No neural network required.

use crate::harmonic_embed::harmonic_embedding_cossin;

/// Linear interpolation: alpha * vec_a + (1 - alpha) * vec_b.
pub fn interpolate(vec_a: &[f64], vec_b: &[f64], alpha: f64) -> Vec<f64> {
    assert_eq!(
        vec_a.len(),
        vec_b.len(),
        "interpolate: vectors must have same length"
    );
    vec_a
        .iter()
        .zip(vec_b.iter())
        .map(|(a, b)| alpha * a + (1.0 - alpha) * b)
        .collect()
}

/// Construct an embedding at a fractional (non-integer) position
/// using the harmonic formula with continuous theta.
pub fn fractional_embedding(
    c_value: f64,
    vocab_size: usize,
    n_harmonics: usize,
) -> Vec<f64> {
    let theta = c_value * (2.0 * std::f64::consts::PI / vocab_size as f64);
    harmonic_embedding_cossin(theta, n_harmonics)
}

/// Chimera construction: low harmonics from vec_a, high harmonics from vec_b.
/// `split_harmonic` is the harmonic index (0-indexed) where we switch sources.
/// For cos+sin embeddings, each harmonic occupies dimensions [2*h, 2*h+1].
pub fn chimera(vec_a: &[f64], vec_b: &[f64], split_harmonic: usize) -> Vec<f64> {
    assert_eq!(
        vec_a.len(),
        vec_b.len(),
        "chimera: vectors must have same length"
    );
    assert!(vec_a.len() % 2 == 0, "chimera: dimension must be even");
    let n_harmonics = vec_a.len() / 2;
    let mut result = vec![0.0; vec_a.len()];
    for h in 0..n_harmonics {
        let ci = h * 2;
        let si = h * 2 + 1;
        if h < split_harmonic {
            result[ci] = vec_a[ci];
            result[si] = vec_a[si];
        } else {
            result[ci] = vec_b[ci];
            result[si] = vec_b[si];
        }
    }
    result
}
