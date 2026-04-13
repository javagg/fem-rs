//! # Example 13 — Eigenvalue Problem (LOBPCG)  (analogous to MFEM ex13)
//!
//! Finds the smallest eigenvalues and eigenmodes of the Laplacian:
//!
//! ```text
//!   −Δu = λ u    in Ω = [0,1]²
//!     u = 0    on ∂Ω
//! ```
//!
//! In discrete form this is the generalized eigenvalue problem:
//! ```text
//!   K v = λ M v
//! ```
//! where K is the stiffness matrix and M is the mass matrix.
//!
//! The analytical eigenvalues are `λ_{m,n} = π²(m² + n²)` for m,n = 1,2,…
//! Smallest: λ₁₁ = 2π² ≈ 19.739, λ₁₂ = λ₂₁ = 5π² ≈ 49.348, λ₂₂ = 8π² ≈ 78.957.
//!
//! ## Usage
//! ```
//! cargo run --example mfem_ex13
//! cargo run --example mfem_ex13 -- --n 16 --k 6
//! ```

use std::f64::consts::PI;

use fem_assembly::{Assembler, standard::{DiffusionIntegrator, MassIntegrator}};
use fem_mesh::SimplexMesh;
use fem_solver::{lobpcg, LobpcgConfig};
use fem_space::{H1Space, fe_space::FESpace, constraints::{apply_dirichlet, boundary_dofs}};

fn main() {
    let args = parse_args();
    println!("=== fem-rs Example 13: Laplacian eigenvalues (LOBPCG) ===");
    println!("  Mesh: {}×{} subdivisions, {} smallest eigenpairs", args.n, args.n, args.k);

    // ─── 1. Mesh and H¹ space ─────────────────────────────────────────────────
    let mesh = SimplexMesh::<2>::unit_square_tri(args.n);
    let space = H1Space::new(mesh, 1);
    let n = space.n_dofs();
    println!("  DOFs: {n}");

    // ─── 2. Assemble K (stiffness) and M (mass) ───────────────────────────────
    let mut k_mat = Assembler::assemble_bilinear(
        &space, &[&DiffusionIntegrator { kappa: 1.0 }], 3
    );
    let mut m_mat = Assembler::assemble_bilinear(
        &space, &[&MassIntegrator { rho: 1.0 }], 3
    );

    // ─── 3. Apply Dirichlet BCs for eigenvalue problem ────────────────────────
    // Strategy: build reduced system restricted to free (interior) DOFs.
    // This avoids pollution from the boundary penalty modes.
    let dm   = space.dof_manager();
    let bnd  = boundary_dofs(space.mesh(), dm, &[1, 2, 3, 4]);
    let bnd_set: std::collections::HashSet<u32> = bnd.iter().cloned().collect();
    let free: Vec<usize> = (0..n).filter(|&i| !bnd_set.contains(&(i as u32))).collect();
    let nf = free.len();
    println!("  Free (interior) DOFs: {nf}");

    // Extract the free×free submatrices using COO
    let k_free = extract_submatrix(&k_mat, &free);
    let m_free = extract_submatrix(&m_mat, &free);

    // ─── 4. Solve with LOBPCG ─────────────────────────────────────────────────
    let cfg = LobpcgConfig {
        max_iter: 500,
        tol:      1e-8,
        verbose:  false,
    };
    let result = lobpcg(&k_free, Some(&m_free), args.k, &cfg)
        .expect("LOBPCG failed");

    println!("\n  Computed eigenvalues:");
    println!("  {:>4}  {:>14}  {:>14}  {:>12}", "Mode", "Computed λ", "Exact λ (approx)", "Error");

    // Expected eigenvalues (sorted): 2π², 5π², 5π², 8π², 10π², 10π², ...
    let exact: Vec<f64> = analytical_eigenvalues(args.k);

    for i in 0..result.eigenvalues.len() {
        let lam    = result.eigenvalues[i];
        let ex_lam = if i < exact.len() { exact[i] } else { f64::NAN };
        let err    = if ex_lam.is_finite() { (lam - ex_lam).abs() / ex_lam } else { f64::NAN };
        println!("  {:>4}  {:>14.6}  {:>14.6}  {:>12.4e}", i+1, lam, ex_lam, err);
    }

    println!("\n  Converged: {}, iterations: {}", result.converged, result.iterations);
    println!("\nDone.");
}

/// Return the k smallest analytical eigenvalues λ = π²(m²+n²), sorted.
fn analytical_eigenvalues(k: usize) -> Vec<f64> {
    let mut eigs: Vec<f64> = Vec::new();
    let max_mn = 10;
    for m in 1..=max_mn {
        for n in 1..=max_mn {
            eigs.push(PI * PI * (m * m + n * n) as f64);
        }
    }
    eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    eigs.dedup_by(|a, b| (*a - *b).abs() < 1e-8);
    eigs.truncate(k);
    eigs
}

/// Extract the submatrix rows/cols indexed by `free_dofs` (a sorted subset).
fn extract_submatrix(a: &fem_linalg::CsrMatrix<f64>, free: &[usize]) -> fem_linalg::CsrMatrix<f64> {
    let n = free.len();
    // Build reverse map: global index → free index (or usize::MAX if constrained)
    let global_n = a.nrows;
    let mut rev = vec![usize::MAX; global_n];
    for (fi, &gi) in free.iter().enumerate() { rev[gi] = fi; }

    let mut coo = fem_linalg::CooMatrix::<f64>::new(n, n);
    for (fi, &gi) in free.iter().enumerate() {
        for ptr in a.row_ptr[gi]..a.row_ptr[gi+1] {
            let gj = a.col_idx[ptr] as usize;
            let fj = rev[gj];
            if fj != usize::MAX {
                coo.add(fi, fj, a.values[ptr]);
            }
        }
    }
    coo.into_csr()
}

// ─── CLI ─────────────────────────────────────────────────────────────────────

struct Args { n: usize, k: usize }

fn parse_args() -> Args {
    let mut a = Args { n: 12, k: 6 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--n" => { a.n = it.next().unwrap_or("12".into()).parse().unwrap_or(12); }
            "--k" => { a.k = it.next().unwrap_or("6".into()).parse().unwrap_or(6); }
            _ => {}
        }
    }
    a
}
