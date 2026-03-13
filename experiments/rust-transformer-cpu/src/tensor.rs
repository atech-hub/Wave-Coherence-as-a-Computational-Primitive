// Mat — 2D matrix type with parallel matmul and numerically stable operations.
// Pure Rust, zero dependencies. Uses std::thread::scope for parallelism.

use crate::rng::Rng;

/// 2D matrix stored in row-major order.
#[derive(Clone, Debug)]
pub struct Mat {
    pub data: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

// Number of CPU threads for parallel matmul
fn num_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

// Threshold for parallel matmul. Set high so per-head attention matmuls
// (256×32×256 = 2M) stay single-threaded while batched linear ops
// (8192×128×384 = 400M+) get full thread parallelism.
// This avoids thread contention with the sequence-level parallelism in attention.
const PAR_THRESHOLD: usize = 5_000_000;

impl Mat {
    // =========================================================================
    // Constructors
    // =========================================================================

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    pub fn ones(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![1.0; rows * cols],
            rows,
            cols,
        }
    }

    pub fn from_vec(data: Vec<f64>, rows: usize, cols: usize) -> Self {
        assert_eq!(
            data.len(),
            rows * cols,
            "from_vec: data len {} != {}x{}",
            data.len(),
            rows,
            cols
        );
        Self { data, rows, cols }
    }

    /// Random normal initialization.
    pub fn randn(rows: usize, cols: usize, mean: f64, std: f64, rng: &mut Rng) -> Self {
        let data: Vec<f64> = (0..rows * cols)
            .map(|_| rng.normal_scaled(mean, std))
            .collect();
        Self { data, rows, cols }
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    #[inline]
    pub fn row(&self, r: usize) -> &[f64] {
        let start = r * self.cols;
        &self.data[start..start + self.cols]
    }

    #[inline]
    pub fn row_mut(&mut self, r: usize) -> &mut [f64] {
        let start = r * self.cols;
        &mut self.data[start..start + self.cols]
    }

    #[inline]
    pub fn at(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    #[inline]
    pub fn at_mut(&mut self, r: usize, c: usize) -> &mut f64 {
        &mut self.data[r * self.cols + c]
    }

    // =========================================================================
    // Element-wise operations
    // =========================================================================

    /// Element-wise addition. Supports broadcasting: if `other` has 1 row,
    /// it is broadcast across all rows of `self`.
    pub fn add(&self, other: &Mat) -> Mat {
        if self.rows == other.rows && self.cols == other.cols {
            let data: Vec<f64> = self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(&a, &b)| a + b)
                .collect();
            Mat::from_vec(data, self.rows, self.cols)
        } else if other.rows == 1 && other.cols == self.cols {
            // Broadcast single row
            let mut out = self.clone();
            for r in 0..self.rows {
                let row = out.row_mut(r);
                for (j, val) in row.iter_mut().enumerate() {
                    *val += other.data[j];
                }
            }
            out
        } else {
            panic!(
                "add shape mismatch: ({},{}) + ({},{})",
                self.rows, self.cols, other.rows, other.cols
            );
        }
    }

    #[allow(dead_code)]
    pub fn sub(&self, other: &Mat) -> Mat {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let data: Vec<f64> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a - b)
            .collect();
        Mat::from_vec(data, self.rows, self.cols)
    }

    /// Scalar multiplication.
    pub fn scale(&self, s: f64) -> Mat {
        let data: Vec<f64> = self.data.iter().map(|&x| x * s).collect();
        Mat::from_vec(data, self.rows, self.cols)
    }

    /// Element-wise multiplication (Hadamard product).
    pub fn hadamard(&self, other: &Mat) -> Mat {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let data: Vec<f64> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a * b)
            .collect();
        Mat::from_vec(data, self.rows, self.cols)
    }

    /// In-place accumulate: self += other.
    pub fn add_inplace(&mut self, other: &Mat) {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        for (a, &b) in self.data.iter_mut().zip(other.data.iter()) {
            *a += b;
        }
    }

    // =========================================================================
    // Matrix multiplication
    // =========================================================================

    /// C = A @ B  (self is A, other is B)
    pub fn matmul(&self, other: &Mat) -> Mat {
        assert_eq!(
            self.cols, other.rows,
            "matmul: ({},{}) @ ({},{})",
            self.rows, self.cols, other.rows, other.cols
        );
        let m = self.rows;
        let k = self.cols;
        let n = other.cols;

        if m * k * n < PAR_THRESHOLD {
            self.matmul_single(other)
        } else {
            self.matmul_parallel(other)
        }
    }

    fn matmul_single(&self, other: &Mat) -> Mat {
        let m = self.rows;
        let k = self.cols;
        let n = other.cols;
        let mut out = vec![0.0; m * n];

        for i in 0..m {
            let a_row = &self.data[i * k..(i + 1) * k];
            let out_row = &mut out[i * n..(i + 1) * n];
            for (p, &a_val) in a_row.iter().enumerate() {
                let b_row = &other.data[p * n..(p + 1) * n];
                for (j, b_val) in b_row.iter().enumerate() {
                    out_row[j] += a_val * b_val;
                }
            }
        }

        Mat::from_vec(out, m, n)
    }

    fn matmul_parallel(&self, other: &Mat) -> Mat {
        let m = self.rows;
        let n = other.cols;
        let nthreads = num_threads().min(m);
        let mut out = vec![0.0; m * n];

        // Share immutable refs to self and other, mutable slice of out
        let a_data = &self.data;
        let b_data = &other.data;
        let k = self.cols;

        std::thread::scope(|s| {
            let chunk_size = (m + nthreads - 1) / nthreads;
            let mut out_rest = out.as_mut_slice();

            for t in 0..nthreads {
                let row_start = t * chunk_size;
                let row_end = (row_start + chunk_size).min(m);
                if row_start >= row_end {
                    break;
                }
                let rows_this = row_end - row_start;
                let (chunk, rest) = out_rest.split_at_mut(rows_this * n);
                out_rest = rest;

                s.spawn(move || {
                    for i in 0..rows_this {
                        let global_i = row_start + i;
                        let a_row = &a_data[global_i * k..(global_i + 1) * k];
                        let out_row = &mut chunk[i * n..(i + 1) * n];
                        for (p, &a_val) in a_row.iter().enumerate() {
                            let b_row = &b_data[p * n..(p + 1) * n];
                            for (j, b_val) in b_row.iter().enumerate() {
                                out_row[j] += a_val * b_val;
                            }
                        }
                    }
                });
            }
        });

        Mat::from_vec(out, m, n)
    }

    // =========================================================================
    // Transpose
    // =========================================================================

    pub fn transpose(&self) -> Mat {
        let mut data = vec![0.0; self.rows * self.cols];
        for r in 0..self.rows {
            for c in 0..self.cols {
                data[c * self.rows + r] = self.data[r * self.cols + c];
            }
        }
        Mat::from_vec(data, self.cols, self.rows)
    }

    // =========================================================================
    // Reductions
    // =========================================================================

    /// Sum along axis 0 (collapse rows) → 1×cols result.
    pub fn sum_axis0(&self) -> Mat {
        let mut out = vec![0.0; self.cols];
        for r in 0..self.rows {
            for c in 0..self.cols {
                out[c] += self.data[r * self.cols + c];
            }
        }
        Mat::from_vec(out, 1, self.cols)
    }

    /// Sum along axis 1 (collapse cols) → rows×1 result.
    #[allow(dead_code)]
    pub fn sum_axis1(&self) -> Mat {
        let mut out = vec![0.0; self.rows];
        for r in 0..self.rows {
            for c in 0..self.cols {
                out[r] += self.data[r * self.cols + c];
            }
        }
        Mat::from_vec(out, self.rows, 1)
    }

    /// Max along axis 1 → rows×1 result.
    #[allow(dead_code)]
    pub fn max_axis1(&self) -> Mat {
        let mut out = vec![f64::NEG_INFINITY; self.rows];
        for r in 0..self.rows {
            for c in 0..self.cols {
                let v = self.data[r * self.cols + c];
                if v > out[r] {
                    out[r] = v;
                }
            }
        }
        Mat::from_vec(out, self.rows, 1)
    }

    // =========================================================================
    // Activation functions
    // =========================================================================

    /// Row-wise softmax (numerically stable).
    pub fn softmax(&self) -> Mat {
        let mut out = Mat::zeros(self.rows, self.cols);
        for r in 0..self.rows {
            let row = self.row(r);
            let max_val = row
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, |a, b| if b > a { b } else { a });
            let mut sum = 0.0;
            let out_row = out.row_mut(r);
            for (j, &v) in row.iter().enumerate() {
                let e = (v - max_val).exp();
                out_row[j] = e;
                sum += e;
            }
            let inv_sum = if sum.abs() < 1e-30 { 0.0 } else { 1.0 / sum };
            for v in out_row.iter_mut() {
                *v *= inv_sum;
            }
        }
        out
    }

    /// GELU activation (exact form): x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    pub fn gelu(&self) -> Mat {
        let sqrt_2_over_pi = (2.0 / std::f64::consts::PI).sqrt();
        let data: Vec<f64> = self
            .data
            .iter()
            .map(|&x| {
                let inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
                0.5 * x * (1.0 + inner.tanh())
            })
            .collect();
        Mat::from_vec(data, self.rows, self.cols)
    }

    /// GELU derivative.
    pub fn gelu_backward(&self) -> Mat {
        let sqrt_2_over_pi = (2.0 / std::f64::consts::PI).sqrt();
        let data: Vec<f64> = self
            .data
            .iter()
            .map(|&x| {
                let x3 = x * x * x;
                let inner = sqrt_2_over_pi * (x + 0.044715 * x3);
                let tanh_val = inner.tanh();
                let sech2 = 1.0 - tanh_val * tanh_val;
                let d_inner = sqrt_2_over_pi * (1.0 + 3.0 * 0.044715 * x * x);
                0.5 * (1.0 + tanh_val) + 0.5 * x * sech2 * d_inner
            })
            .collect();
        Mat::from_vec(data, self.rows, self.cols)
    }

    // =========================================================================
    // Loss functions
    // =========================================================================

    /// Cross-entropy loss from logits and integer targets.
    /// logits: (T, V), targets: T integers in [0, V).
    /// Returns scalar loss = -mean(log(softmax[target])).
    pub fn cross_entropy_loss(&self, targets: &[usize]) -> f64 {
        assert_eq!(self.rows, targets.len());
        let mut total = 0.0;
        for r in 0..self.rows {
            let row = self.row(r);
            let max_val = row
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, |a, b| if b > a { b } else { a });
            let mut log_sum_exp = 0.0;
            for &v in row {
                log_sum_exp += (v - max_val).exp();
            }
            log_sum_exp = max_val + log_sum_exp.ln();
            total += log_sum_exp - row[targets[r]];
        }
        total / self.rows as f64
    }

    /// Combined softmax + cross-entropy backward.
    /// Returns d_logits: (T, V) = (softmax - one_hot) / T.
    pub fn cross_entropy_backward(&self, targets: &[usize]) -> Mat {
        let sm = self.softmax();
        let mut grad = sm;
        let n = grad.rows as f64;
        for r in 0..grad.rows {
            grad.row_mut(r)[targets[r]] -= 1.0;
        }
        // Average over batch
        for v in grad.data.iter_mut() {
            *v /= n;
        }
        grad
    }

    // =========================================================================
    // Softmax backward (for attention weights)
    // =========================================================================

    /// softmax backward: given S = softmax output and dS = upstream gradient,
    /// returns d_scores where d_scores[i] = S[i] * (dS[i] - sum(S * dS)).
    /// Applied row-wise.
    pub fn softmax_backward(&self, d_out: &Mat) -> Mat {
        assert_eq!(self.rows, d_out.rows);
        assert_eq!(self.cols, d_out.cols);
        let mut grad = Mat::zeros(self.rows, self.cols);
        for r in 0..self.rows {
            let s = self.row(r);
            let ds = d_out.row(r);
            // dot = sum(s * ds)
            let dot: f64 = s.iter().zip(ds.iter()).map(|(&a, &b)| a * b).sum();
            let out = grad.row_mut(r);
            for j in 0..self.cols {
                out[j] = s[j] * (ds[j] - dot);
            }
        }
        grad
    }

    // =========================================================================
    // Slicing
    // =========================================================================

    /// Extract rows [start..end).
    pub fn slice_rows(&self, start: usize, end: usize) -> Mat {
        assert!(end <= self.rows);
        let data = self.data[start * self.cols..end * self.cols].to_vec();
        Mat::from_vec(data, end - start, self.cols)
    }

    /// Extract columns [start..end) from every row.
    pub fn slice_cols(&self, start: usize, end: usize) -> Mat {
        assert!(end <= self.cols);
        let new_cols = end - start;
        let mut data = Vec::with_capacity(self.rows * new_cols);
        for r in 0..self.rows {
            data.extend_from_slice(&self.data[r * self.cols + start..r * self.cols + end]);
        }
        Mat::from_vec(data, self.rows, new_cols)
    }

    /// Set columns [start..end) from another matrix.
    pub fn set_cols(&mut self, start: usize, src: &Mat) {
        assert_eq!(self.rows, src.rows);
        assert!(start + src.cols <= self.cols);
        for r in 0..self.rows {
            let dest = &mut self.data[r * self.cols + start..r * self.cols + start + src.cols];
            dest.copy_from_slice(src.row(r));
        }
    }

    // =========================================================================
    // Embedding operations
    // =========================================================================

    /// Gather rows by index: for each idx[i], return self[idx[i]].
    pub fn gather_rows(&self, indices: &[usize]) -> Mat {
        let n = indices.len();
        let mut data = Vec::with_capacity(n * self.cols);
        for &idx in indices {
            data.extend_from_slice(self.row(idx));
        }
        Mat::from_vec(data, n, self.cols)
    }

    /// Scatter-add: for each i, self[indices[i]] += grad.row(i).
    pub fn scatter_add(&mut self, indices: &[usize], grad: &Mat) {
        assert_eq!(indices.len(), grad.rows);
        assert_eq!(self.cols, grad.cols);
        for (i, &idx) in indices.iter().enumerate() {
            let row = self.row_mut(idx);
            for (j, &g) in grad.row(i).iter().enumerate() {
                row[j] += g;
            }
        }
    }

    // =========================================================================
    // Concatenation
    // =========================================================================

    /// Copy rows from src into self starting at row `offset`.
    pub fn copy_rows_from(&mut self, offset: usize, src: &Mat) {
        assert_eq!(self.cols, src.cols);
        assert!(offset + src.rows <= self.rows);
        let start = offset * self.cols;
        let len = src.rows * self.cols;
        self.data[start..start + len].copy_from_slice(&src.data);
    }

    /// Horizontal concatenation: [self | other] (same rows, cols add up).
    #[allow(dead_code)]
    pub fn hcat(&self, other: &Mat) -> Mat {
        assert_eq!(self.rows, other.rows);
        let new_cols = self.cols + other.cols;
        let mut data = Vec::with_capacity(self.rows * new_cols);
        for r in 0..self.rows {
            data.extend_from_slice(self.row(r));
            data.extend_from_slice(other.row(r));
        }
        Mat::from_vec(data, self.rows, new_cols)
    }
}
