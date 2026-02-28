// Discrete Fourier Transform — pure Rust, no dependencies.
//
// Implements rfft (real-input DFT) and irfft (inverse real-input DFT),
// equivalent to numpy.fft.rfft and numpy.fft.irfft.
//
// Uses naive O(N^2) DFT for clarity and zero-dependency purity.
// For N <= 384 this completes in well under 1ms on modern CPUs.

use std::f64::consts::PI;

/// Minimal complex number type for DFT coefficients.
#[derive(Clone, Copy, Debug)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub fn new(re: f64, im: f64) -> Self {
        Complex { re, im }
    }

    pub fn zero() -> Self {
        Complex { re: 0.0, im: 0.0 }
    }

    /// Complex modulus (absolute value).
    #[allow(dead_code)]
    pub fn abs(&self) -> f64 {
        (self.re * self.re + self.im * self.im).sqrt()
    }

    /// Phase angle in radians.
    #[allow(dead_code)]
    pub fn angle(&self) -> f64 {
        self.im.atan2(self.re)
    }

    pub fn add(self, other: Self) -> Self {
        Complex {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }

    #[allow(dead_code)]
    pub fn mul(self, other: Self) -> Self {
        Complex {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }

    /// Multiply by a real scalar.
    #[allow(dead_code)]
    pub fn scale(self, s: f64) -> Self {
        Complex {
            re: self.re * s,
            im: self.im * s,
        }
    }
}

/// Real-input DFT: equivalent to numpy.fft.rfft.
///
/// Input: real vector of length N.
/// Output: N/2+1 complex coefficients.
/// Formula: X[k] = sum_{n=0}^{N-1} x[n] * exp(-j*2*pi*k*n/N) for k=0..N/2.
pub fn rfft(input: &[f64]) -> Vec<Complex> {
    assert!(!input.is_empty(), "rfft: input must not be empty");
    let n = input.len();
    let n_out = n / 2 + 1;
    let mut result = vec![Complex::zero(); n_out];
    for k in 0..n_out {
        let mut sum = Complex::zero();
        for (t, &x) in input.iter().enumerate() {
            let angle = -2.0 * PI * (k as f64) * (t as f64) / (n as f64);
            sum = sum.add(Complex::new(x * angle.cos(), x * angle.sin()));
        }
        result[k] = sum;
    }
    result
}

/// Inverse real-input DFT: equivalent to numpy.fft.irfft(coeffs, n=original_length).
///
/// Input: N/2+1 complex coefficients, original signal length.
/// Output: real vector of length `original_length`.
///
/// Uses conjugate symmetry: for a real signal, X[N-k] = conj(X[k]).
#[allow(dead_code)]
pub fn irfft(coeffs: &[Complex], original_length: usize) -> Vec<f64> {
    assert!(!coeffs.is_empty(), "irfft: coefficients must not be empty");
    assert!(original_length > 0, "irfft: original_length must be positive");
    let n = original_length;
    let n_coeffs = coeffs.len();
    let mut result = vec![0.0; n];

    for t in 0..n {
        let mut sum = 0.0;
        for k in 0..n_coeffs {
            let angle = 2.0 * PI * (k as f64) * (t as f64) / (n as f64);
            let twiddle = Complex::new(angle.cos(), angle.sin());
            let contribution = coeffs[k].mul(twiddle);
            // For k=0 and k=N/2 (if N even), count once.
            // For other k, count twice (conjugate symmetry).
            if k == 0 || (n % 2 == 0 && k == n_coeffs - 1) {
                sum += contribution.re;
            } else {
                sum += 2.0 * contribution.re;
            }
        }
        result[t] = sum / (n as f64);
    }
    result
}
