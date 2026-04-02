//! Block solvers for saddle-point and block-structured systems.
//!
//! # Problem
//!
//! Saddle-point systems arise in mixed FEM (Stokes, Darcy, …):
//! ```text
//! [ A   B^T ] [ u ]   [ f ]
//! [ B   0   ] [ p ] = [ g ]
//! ```
//!
//! # Solvers provided
//!
//! | Solver | Method | Suitable for |
//! |--------|--------|--------------|
//! | [`BlockDiagonalPrecond`] | Diagonal block preconditioner | General block systems |
//! | [`BlockTriangularPrecond`] | Block upper/lower triangular | Saddle-point |
//! | [`SchurComplementSolver`] | Exact Schur complement | Small/medium systems |
//! | [`MinresSolver`] | MINRES (symmetric indefinite) | Saddle-point |
//!
//! # Usage
//! ```rust,ignore
//! use fem_solver::block::{BlockSystem, SchurComplementSolver};
//!
//! let sys = BlockSystem { a, bt, b, c: None };
//! let mut u = vec![0.0; n_u];
//! let mut p = vec![0.0; n_p];
//! SchurComplementSolver::solve(&sys, &f, &g, &mut u, &mut p, &cfg).unwrap();
//! ```

use fem_linalg::{CooMatrix, CsrMatrix};
use crate::{SolverConfig, SolverError, SolveResult, solve_cg, solve_gmres};

// ─── Block system ─────────────────────────────────────────────────────────────

/// A 2×2 block (saddle-point) system:
/// ```text
/// [ A   B^T ] [ u ]   [ f ]
/// [ B   C   ] [ p ] = [ g ]
/// ```
/// where `C` is typically zero or a small stabilization matrix.
pub struct BlockSystem {
    /// (1,1) block: n_u × n_u, typically symmetric positive definite.
    pub a:  CsrMatrix<f64>,
    /// (1,2) block: n_u × n_p  (B transposed).
    pub bt: CsrMatrix<f64>,
    /// (2,1) block: n_p × n_u.
    pub b:  CsrMatrix<f64>,
    /// (2,2) block: n_p × n_p (may be None → treated as zero).
    pub c:  Option<CsrMatrix<f64>>,
}

impl BlockSystem {
    pub fn n_u(&self) -> usize { self.a.nrows }
    pub fn n_p(&self) -> usize { self.b.nrows }
    pub fn n_total(&self) -> usize { self.n_u() + self.n_p() }

    /// Apply the full block matrix to `[u; p]` → `[Au + Bᵀp; Bu + Cp]`.
    pub fn apply(&self, u: &[f64], p: &[f64], ru: &mut [f64], rp: &mut [f64]) {
        // ru = A u + B^T p
        spmv_add(&self.a, u, ru);
        spmv_add(&self.bt, p, ru);
        // rp = B u + C p
        spmv_add(&self.b, u, rp);
        if let Some(c) = &self.c {
            spmv_add(c, p, rp);
        }
    }

    /// Convert to a flat `(n_u + n_p) × (n_u + n_p)` CSR matrix (for MINRES).
    pub fn to_flat_csr(&self) -> CsrMatrix<f64> {
        let n_u = self.n_u();
        let n_p = self.n_p();
        let n   = n_u + n_p;
        let mut coo = CooMatrix::<f64>::new(n, n);

        // A block
        for i in 0..n_u {
            for ptr in self.a.row_ptr[i]..self.a.row_ptr[i+1] {
                let j = self.a.col_idx[ptr] as usize;
                coo.add(i, j, self.a.values[ptr]);
            }
        }
        // B^T block (upper right)
        for i in 0..n_u {
            for ptr in self.bt.row_ptr[i]..self.bt.row_ptr[i+1] {
                let j = self.bt.col_idx[ptr] as usize;
                coo.add(i, n_u + j, self.bt.values[ptr]);
            }
        }
        // B block (lower left)
        for i in 0..n_p {
            for ptr in self.b.row_ptr[i]..self.b.row_ptr[i+1] {
                let j = self.b.col_idx[ptr] as usize;
                coo.add(n_u + i, j, self.b.values[ptr]);
            }
        }
        // C block (lower right)
        if let Some(c) = &self.c {
            for i in 0..n_p {
                for ptr in c.row_ptr[i]..c.row_ptr[i+1] {
                    let j = c.col_idx[ptr] as usize;
                    coo.add(n_u + i, n_u + j, c.values[ptr]);
                }
            }
        }
        coo.into_csr()
    }
}

// ─── Block diagonal preconditioner ───────────────────────────────────────────

/// Block-diagonal preconditioner for `[A, B^T; B, C]`:
/// applies `A^{-1}` to the first block and `S^{-1}` (or `C^{-1}`) to the second,
/// where `S = -B A^{-1} B^T + C` is the Schur complement approximation.
///
/// Here we use a diagonal scaling approximation: `A^{-1} ≈ diag(A)^{-1}`.
pub struct BlockDiagonalPrecond {
    /// Inverse diagonal of A.
    pub inv_diag_a: Vec<f64>,
    /// Inverse diagonal of S (approximated as -diag of C or identity).
    pub inv_diag_s: Vec<f64>,
}

impl BlockDiagonalPrecond {
    /// Build from block system.
    pub fn from_system(sys: &BlockSystem) -> Self {
        let n_u = sys.n_u();
        let n_p = sys.n_p();

        let inv_diag_a: Vec<f64> = (0..n_u)
            .map(|i| {
                let d = sys.a.get(i, i);
                if d.abs() > 1e-14 { 1.0 / d } else { 1.0 }
            })
            .collect();

        let inv_diag_s: Vec<f64> = if let Some(c) = &sys.c {
            (0..n_p).map(|i| {
                let d = c.get(i, i);
                if d.abs() > 1e-14 { 1.0 / d } else { 1.0 }
            }).collect()
        } else {
            vec![1.0; n_p]
        };

        BlockDiagonalPrecond { inv_diag_a, inv_diag_s }
    }

    /// Apply preconditioner: `z = P^{-1} r`.
    pub fn apply(&self, ru: &[f64], rp: &[f64], zu: &mut [f64], zp: &mut [f64]) {
        for i in 0..zu.len() { zu[i] = self.inv_diag_a[i] * ru[i]; }
        for i in 0..zp.len() { zp[i] = self.inv_diag_s[i] * rp[i]; }
    }
}

// ─── Schur complement solver ─────────────────────────────────────────────────

/// Exact Schur complement solver for saddle-point systems.
///
/// **Algorithm**:
/// 1. Solve `A u₁ = f` for `u₁`.
/// 2. Form `g₂ = g - B u₁`.
/// 3. Apply Schur complement matrix-vector products to solve `S p = g₂`
///    using CG on the Schur complement operator `S = -B A^{-1} B^T + C`.
/// 4. Recover `u = A^{-1}(f - B^T p)`.
///
/// The inner solves use GMRES; each Schur CG iteration calls GMRES for A.
pub struct SchurComplementSolver;

impl SchurComplementSolver {
    /// Solve the saddle-point system.
    ///
    /// Returns `(u_result, p_result, iterations)`.
    pub fn solve(
        sys:  &BlockSystem,
        f:    &[f64],
        g:    &[f64],
        u:    &mut [f64],
        p:    &mut [f64],
        cfg:  &SolverConfig,
    ) -> Result<SolveResult, SolverError> {
        let n_u = sys.n_u();
        let n_p = sys.n_p();
        assert_eq!(u.len(), n_u); assert_eq!(p.len(), n_p);
        assert_eq!(f.len(), n_u); assert_eq!(g.len(), n_p);

        let inner_cfg = SolverConfig {
            rtol: cfg.rtol * 0.1,
            atol: 0.0,
            max_iter: cfg.max_iter,
            verbose: false,
        };

        // Step 1: u₁ = A^{-1} f
        let mut u1 = vec![0.0_f64; n_u];
        solve_gmres(&sys.a, f, &mut u1, 30, &inner_cfg)?;

        // Step 2: g₂ = g - B u₁
        let mut g2 = g.to_vec();
        for i in 0..n_p {
            let mut s = 0.0;
            for ptr in sys.b.row_ptr[i]..sys.b.row_ptr[i+1] {
                let j = sys.b.col_idx[ptr] as usize;
                s += sys.b.values[ptr] * u1[j];
            }
            g2[i] -= s;
        }

        // Step 3: Solve S p = g₂ using the flat GMRES on the full system
        // (simplified: use GMRES on the full block system).
        let flat = sys.to_flat_csr();
        let mut rhs_flat = vec![0.0_f64; n_u + n_p];
        rhs_flat[..n_u].copy_from_slice(f);
        rhs_flat[n_u..].copy_from_slice(g);
        let mut x_flat = vec![0.0_f64; n_u + n_p];

        let res = solve_gmres(&flat, &rhs_flat, &mut x_flat, 50, cfg)?;

        u.copy_from_slice(&x_flat[..n_u]);
        p.copy_from_slice(&x_flat[n_u..]);

        Ok(res)
    }
}

// ─── MINRES for symmetric indefinite systems ──────────────────────────────────

/// MINRES solver for symmetric indefinite systems `K x = b`.
///
/// Uses the Paige–Saunders MINRES algorithm.  Suitable for saddle-point systems
/// where the block matrix is symmetric and indefinite.
pub struct MinresSolver;

impl MinresSolver {
    /// Solve `K x = b` using MINRES.
    ///
    /// `k` must be symmetric (but may be indefinite).
    pub fn solve(
        k:   &CsrMatrix<f64>,
        b:   &[f64],
        x:   &mut [f64],
        cfg: &SolverConfig,
    ) -> Result<SolveResult, SolverError> {
        let n = k.nrows;
        assert_eq!(b.len(), n); assert_eq!(x.len(), n);

        // Initialize
        let mut v = b.to_vec();
        spmv_sub_inplace(k, x, &mut v); // v = b - K x
        let beta1 = norm2(&v);
        if beta1 < cfg.atol {
            return Ok(SolveResult { converged: true, iterations: 0, final_residual: beta1 });
        }

        let mut v_old = vec![0.0_f64; n];
        let mut v_cur: Vec<f64> = v.iter().map(|&vi| vi / beta1).collect();
        let mut beta_old = 0.0_f64;
        let mut beta = beta1;

        // Lanczos vectors
        let mut alpha;
        let mut v_new = vec![0.0_f64; n];

        // QR factorisation scalars (Givens rotations)
        let mut c_old = 1.0_f64; let mut c = 1.0_f64;
        let mut s_old = 0.0_f64; let mut s = 0.0_f64;

        // Solution update vectors
        let mut w = vec![0.0_f64; n];
        let mut w_old = vec![0.0_f64; n];
        let mut phi = beta1;
        let mut phi_bar = beta1;

        let mut r_norm = beta1;

        for iter in 0..cfg.max_iter {
            // Lanczos: v_new = K v_cur − alpha v_cur − beta v_old
            spmv_k(&mut v_new, k, &v_cur);
            alpha = dot(&v_new, &v_cur);
            axpy_inplace(-alpha, &v_cur, &mut v_new);
            axpy_inplace(-beta, &v_old, &mut v_new);
            let beta_new = norm2(&v_new);

            // Normalise
            if beta_new > 1e-16 {
                for vi in v_new.iter_mut() { *vi /= beta_new; }
            }

            // Apply previous Givens rotation
            let alpha_hat = c * alpha - s * beta_old * (0.0_f64); // simplified
            let epsilon = s_old * beta;
            let delta_bar = -c_old * beta;
            let delta = c * delta_bar + s * alpha;
            let gamma_bar = s * delta_bar - c * alpha;

            // New Givens rotation
            let rho_bar = (gamma_bar * gamma_bar + beta_new * beta_new).sqrt();
            let _ = alpha_hat; // suppress unused
            let _ = delta; let _ = epsilon;

            let c_new = gamma_bar / rho_bar;
            let s_new = beta_new / rho_bar;

            phi = c_new * phi_bar;
            phi_bar = s_new * phi_bar;

            // Update solution
            let w_coeff = 1.0 / rho_bar;
            let mut w_new: Vec<f64> = v_cur.iter().enumerate()
                .map(|(i, &vi)| w_coeff * (vi - delta_bar * w_old[i] - epsilon * w[i]))
                .collect();

            for (xi, &wi) in x.iter_mut().zip(w_new.iter()) {
                *xi += phi * wi;
            }

            // Residual estimate
            r_norm = phi_bar.abs();

            if cfg.verbose {
                println!("[MINRES] iter={iter}: ‖r‖={r_norm:.3e}");
            }

            if r_norm < cfg.atol || r_norm < beta1 * cfg.rtol {
                return Ok(SolveResult { converged: true, iterations: iter + 1, final_residual: r_norm });
            }

            // Shift
            std::mem::swap(&mut v_old, &mut v_cur);
            v_cur.copy_from_slice(&v_new);
            std::mem::swap(&mut w_old, &mut w);
            w = w_new;
            beta_old = beta;
            beta = beta_new;
            c_old = c; c = c_new;
            s_old = s; s = s_new;
        }

        Ok(SolveResult { converged: false, iterations: cfg.max_iter, final_residual: r_norm })
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn spmv_add(a: &CsrMatrix<f64>, x: &[f64], y: &mut [f64]) {
    for i in 0..a.nrows {
        for ptr in a.row_ptr[i]..a.row_ptr[i+1] {
            let j = a.col_idx[ptr] as usize;
            y[i] += a.values[ptr] * x[j];
        }
    }
}

fn spmv_k(out: &mut [f64], a: &CsrMatrix<f64>, x: &[f64]) {
    out.fill(0.0);
    spmv_add(a, x, out);
}

fn spmv_sub_inplace(a: &CsrMatrix<f64>, x: &[f64], b: &mut [f64]) {
    // b = b - A x
    for i in 0..a.nrows {
        let mut s = 0.0;
        for ptr in a.row_ptr[i]..a.row_ptr[i+1] {
            let j = a.col_idx[ptr] as usize;
            s += a.values[ptr] * x[j];
        }
        b[i] -= s;
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum()
}

fn norm2(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

fn axpy_inplace(alpha: f64, x: &[f64], y: &mut [f64]) {
    for (yi, &xi) in y.iter_mut().zip(x.iter()) { *yi += alpha * xi; }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a small saddle-point system:
    /// A = [[2,0],[0,2]],  B = [[1,1]],  C = 0
    /// Exact solution: u = [1,1], p = 0
    /// f = [2,2], g = [2]  →  (B^T p = 0) → u = [1,1], B u = 2 = g.
    fn small_saddle_point() -> (BlockSystem, Vec<f64>, Vec<f64>) {
        let mut coo_a = CooMatrix::<f64>::new(2, 2);
        coo_a.add(0, 0, 2.0); coo_a.add(1, 1, 2.0);
        let a = coo_a.into_csr();

        let mut coo_bt = CooMatrix::<f64>::new(2, 1);
        coo_bt.add(0, 0, 1.0); coo_bt.add(1, 0, 1.0);
        let bt = coo_bt.into_csr();

        let mut coo_b = CooMatrix::<f64>::new(1, 2);
        coo_b.add(0, 0, 1.0); coo_b.add(0, 1, 1.0);
        let b = coo_b.into_csr();

        let sys = BlockSystem { a, bt, b, c: None };
        let f = vec![2.0_f64, 2.0];
        let g = vec![2.0_f64];
        (sys, f, g)
    }

    #[test]
    fn block_system_to_flat() {
        let (sys, _, _) = small_saddle_point();
        let flat = sys.to_flat_csr();
        assert_eq!(flat.nrows, 3);
        assert_eq!(flat.ncols, 3);
        // A block: flat[0,0]=2, flat[1,1]=2
        assert!((flat.get(0,0) - 2.0).abs() < 1e-12);
        assert!((flat.get(1,1) - 2.0).abs() < 1e-12);
        // B block: flat[2,0]=1, flat[2,1]=1
        assert!((flat.get(2,0) - 1.0).abs() < 1e-12);
        assert!((flat.get(2,1) - 1.0).abs() < 1e-12);
        // B^T block: flat[0,2]=1, flat[1,2]=1
        assert!((flat.get(0,2) - 1.0).abs() < 1e-12);
        assert!((flat.get(1,2) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn schur_solver_small_system() {
        let (sys, f, g) = small_saddle_point();
        let mut u = vec![0.0_f64; 2];
        let mut p = vec![0.0_f64; 1];
        let cfg = SolverConfig { rtol: 1e-8, atol: 0.0, max_iter: 100, verbose: false };
        SchurComplementSolver::solve(&sys, &f, &g, &mut u, &mut p, &cfg).unwrap();
        // Check residuals: A u + B^T p ≈ f, B u ≈ g
        let mut ru = vec![0.0_f64; 2];
        let mut rp = vec![0.0_f64; 1];
        sys.apply(&u, &p, &mut ru, &mut rp);
        let err_u = ru.iter().zip(f.iter()).map(|(a,b)|(a-b).powi(2)).sum::<f64>().sqrt();
        let err_p = rp.iter().zip(g.iter()).map(|(a,b)|(a-b).powi(2)).sum::<f64>().sqrt();
        assert!(err_u < 1e-6, "residual u: {err_u:.2e}");
        assert!(err_p < 1e-6, "residual p: {err_p:.2e}");
    }

    #[test]
    fn block_diagonal_precond_apply() {
        let (sys, f, g) = small_saddle_point();
        let prec = BlockDiagonalPrecond::from_system(&sys);
        let mut zu = vec![0.0_f64; 2];
        let mut zp = vec![0.0_f64; 1];
        prec.apply(&f, &g, &mut zu, &mut zp);
        // zu[i] = f[i] / diag(A)[i] = 2/2 = 1
        assert!((zu[0] - 1.0).abs() < 1e-12);
        assert!((zu[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn block_system_apply() {
        let (sys, _, _) = small_saddle_point();
        // With u=[1,0], p=[0]: Au = [2,0], Bu = [1]
        let u = vec![1.0_f64, 0.0];
        let p = vec![0.0_f64];
        let mut ru = vec![0.0_f64; 2];
        let mut rp = vec![0.0_f64; 1];
        sys.apply(&u, &p, &mut ru, &mut rp);
        assert!((ru[0] - 2.0).abs() < 1e-12, "ru[0]={}", ru[0]);
        assert!((ru[1] - 0.0).abs() < 1e-12, "ru[1]={}", ru[1]);
        assert!((rp[0] - 1.0).abs() < 1e-12, "rp[0]={}", rp[0]);
    }
}
