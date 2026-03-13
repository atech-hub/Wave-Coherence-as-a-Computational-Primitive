// Xorshift64 PRNG with Box-Muller normal distribution.
// Deterministic, no dependencies.

pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        // Ensure non-zero state
        Self {
            state: if seed == 0 { 0xDEAD_BEEF_CAFE_BABE } else { seed },
        }
    }

    /// Raw xorshift64 step — returns a non-zero u64.
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Uniform f64 in [0, 1).
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }

    /// Uniform usize in [0, max).
    pub fn usize(&mut self, max: usize) -> usize {
        (self.uniform() * max as f64) as usize
    }

    /// Standard normal via Box-Muller transform.
    pub fn normal(&mut self) -> f64 {
        loop {
            let u1 = self.uniform();
            let u2 = self.uniform();
            if u1 > 1e-30 {
                let r = (-2.0 * u1.ln()).sqrt();
                let theta = 2.0 * std::f64::consts::PI * u2;
                return r * theta.cos();
            }
        }
    }

    /// Normal with given mean and std.
    pub fn normal_scaled(&mut self, mean: f64, std: f64) -> f64 {
        self.normal() * std + mean
    }

    /// Categorical sampling from a probability distribution.
    /// Returns an index in [0, probs.len()).
    pub fn categorical(&mut self, probs: &[f64]) -> usize {
        let r = self.uniform();
        let mut cumsum = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if r < cumsum {
                return i;
            }
        }
        probs.len() - 1
    }
}
