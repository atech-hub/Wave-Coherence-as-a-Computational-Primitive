// Wave Packet Engine — pure Rust port of Python Phase 16.
//
// DFT-based embedding decomposition, wave packet queries, resonance
// matching, and selective band loading. All foundational math.

use crate::dft::{Complex, rfft, irfft};

/// Decompose an embedding vector into DFT bands.
/// Returns (coefficients, amplitudes, phases).
pub fn embed_to_bands(vector: &[f64]) -> (Vec<Complex>, Vec<f64>, Vec<f64>) {
    let coeffs = rfft(vector);
    let amplitudes: Vec<f64> = coeffs.iter().map(|c| c.abs()).collect();
    let phases: Vec<f64> = coeffs.iter().map(|c| c.angle()).collect();
    (coeffs, amplitudes, phases)
}

/// Reconstruct vector from DFT coefficients.
pub fn bands_to_embed(coeffs: &[Complex], original_length: usize) -> Vec<f64> {
    irfft(coeffs, original_length)
}

/// A single band entry in a wave packet: (frequency_index, amplitude, phase).
#[derive(Debug, Clone)]
pub struct BandEntry {
    pub n: usize,
    pub amplitude: f64,
    pub phase: f64,
}

/// Create a sparse wave packet from selected bands.
/// W = { (n, |V_n|, phi_n) : n in S }
pub fn make_wave_packet(amplitudes: &[f64], phases: &[f64], band_set: &[usize]) -> Vec<BandEntry> {
    band_set
        .iter()
        .filter(|&&n| n < amplitudes.len() && n < phases.len())
        .map(|&n| BandEntry {
            n,
            amplitude: amplitudes[n],
            phase: phases[n],
        })
        .collect()
}

/// Resonance matching: amplitude-weighted phase coherence.
/// R(W, U) = sum w_n * |V_n| * |U_n| * cos(phi_n - psi_n), normalised.
///
/// The `n_coeffs` parameter is the total number of rfft coefficients (N/2+1).
/// This is needed to correctly weight conjugate-symmetric DFT coefficients:
/// DC (n=0) and Nyquist (n=N/2) get weight 1, all others get weight 2.
/// With correct weighting, full-band resonance equals cosine similarity exactly.
pub fn resonance(
    packet: &[BandEntry],
    stored_amplitudes: &[f64],
    stored_phases: &[f64],
    n_coeffs: usize,
) -> f64 {
    let mut score = 0.0;
    let mut query_energy = 0.0;
    let mut stored_energy = 0.0;

    let last_idx = n_coeffs.saturating_sub(1);

    for entry in packet {
        if entry.n >= stored_amplitudes.len() || entry.n >= stored_phases.len() {
            continue;
        }
        let amp_s = stored_amplitudes[entry.n];
        let phase_s = stored_phases[entry.n];

        // Conjugate symmetry weight: DC and Nyquist count once, rest count twice
        let w = if entry.n == 0 || entry.n == last_idx { 1.0 } else { 2.0 };

        score += w * entry.amplitude * amp_s * (entry.phase - phase_s).cos();
        query_energy += w * entry.amplitude * entry.amplitude;
        stored_energy += w * amp_s * amp_s;
    }

    let norm = query_energy.sqrt() * stored_energy.sqrt();
    if norm < 1e-10 {
        0.0
    } else {
        score / norm
    }
}

/// Selective band loading: zero-fill bands not in band_set, reconstruct via IDFT.
pub fn selective_load(
    coeffs: &[Complex],
    band_set: &[usize],
    original_length: usize,
) -> Vec<f64> {
    let mut partial = vec![Complex::zero(); coeffs.len()];
    for &n in band_set {
        if n < coeffs.len() {
            partial[n] = coeffs[n];
        }
    }
    irfft(&partial, original_length)
}

/// Select bands where amplitude exceeds the given percentile threshold.
pub fn select_by_amplitude(amplitudes: &[f64], threshold_percentile: f64) -> Vec<usize> {
    let mut sorted_amps: Vec<f64> = amplitudes.to_vec();
    sorted_amps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((threshold_percentile / 100.0) * (sorted_amps.len() as f64 - 1.0)).round()
        as usize;
    let idx = idx.min(sorted_amps.len().saturating_sub(1));
    let threshold_val = sorted_amps[idx];
    (0..amplitudes.len())
        .filter(|&n| amplitudes[n] >= threshold_val)
        .collect()
}
