// AdamW optimizer with decoupled weight decay.
// Weight decay applied only to 2D weight matrices (not biases, not LayerNorm).

use crate::tensor::Mat;

/// A single parameter tracked by the optimizer.
pub struct Param<'a> {
    pub value: &'a mut Mat,
    pub grad: &'a Mat,
    pub weight_decay: bool, // true for 2D weights, false for biases/LN
}

/// AdamW optimizer state for one parameter.
struct AdamState {
    m: Vec<f64>,  // first moment
    v: Vec<f64>,  // second moment
}

pub struct AdamW {
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
    states: Vec<AdamState>,
    t: usize, // step count for bias correction
}

impl AdamW {
    pub fn new(n_params: usize) -> Self {
        let mut states = Vec::with_capacity(n_params);
        for _ in 0..n_params {
            states.push(AdamState {
                m: Vec::new(), // lazily initialized on first step
                v: Vec::new(),
            });
        }
        Self {
            lr: 3e-4,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            states,
            t: 0,
        }
    }

    /// Perform one optimizer step.
    /// `params` must have the same length and order every call.
    pub fn step(&mut self, params: &mut [Param]) {
        self.t += 1;
        let t = self.t as f64;
        let bc1 = 1.0 / (1.0 - self.beta1.powf(t));
        let bc2 = 1.0 / (1.0 - self.beta2.powf(t));

        assert_eq!(params.len(), self.states.len());

        for (param, state) in params.iter_mut().zip(self.states.iter_mut()) {
            let n = param.value.data.len();

            // Lazy init
            if state.m.is_empty() {
                state.m = vec![0.0; n];
                state.v = vec![0.0; n];
            }

            assert_eq!(n, param.grad.data.len());

            for i in 0..n {
                let g = param.grad.data[i];

                // Update moments
                state.m[i] = self.beta1 * state.m[i] + (1.0 - self.beta1) * g;
                state.v[i] = self.beta2 * state.v[i] + (1.0 - self.beta2) * g * g;

                // Bias correction
                let m_hat = state.m[i] * bc1;
                let v_hat = state.v[i] * bc2;

                // Adam update
                param.value.data[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);

                // Decoupled weight decay (only for designated parameters)
                if param.weight_decay {
                    param.value.data[i] -= self.lr * self.weight_decay * param.value.data[i];
                }
            }
        }
    }
}
