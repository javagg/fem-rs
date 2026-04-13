//! # Example 22 — Complex-Valued Time-Harmonic Helmholtz (analogous to MFEM ex22)
//!
//! Solves a damped time-harmonic scalar Helmholtz equation using the
//! **2×2 real-block** strategy in `fem_assembly::complex`:
//!
//! ```text
//!   −∇·(a∇u) − ω²b·u + iω·c·u = 0    in Ω = [0,1]²
//! ```
//!
//! with Dirichlet BCs:
//! - Left edge  (tag 4): u = 1+0i  (unit amplitude port)
//! - All other edges:    u = 0+0i
//!
//! The 2×2 real block system is:
//! ```text
//! [ K − ω²M   −ωC ] [ u_re ]   [ 0 ]
//! [ ωC        K−ω²M] [ u_im ] = [ 0 ]
//! ```
//! Solved with GMRES.
//!
//! ## Usage
//! ```
//! cargo run --example ex22_complex
//! cargo run --example ex22_complex -- --n 16 --omega 2.0
//! cargo run --example ex22_complex -- --n 32 --sigma 0.2
//! ```

use fem_assembly::{
    ComplexAssembler, ComplexGridFunction,
    standard::{DiffusionIntegrator, MassIntegrator},
};
use fem_mesh::SimplexMesh;
use fem_solver::{SolverConfig, solve_gmres};
use fem_space::{
    H1Space,
    fe_space::FESpace,
    constraints::boundary_dofs,
};

// ─── CLI ─────────────────────────────────────────────────────────────────────

struct Args { n: usize, omega: f64, sigma: f64 }

fn parse_args() -> Args {
    let mut a = Args { n: 8, omega: 1.5, sigma: 0.1 };
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < raw.len() {
        match raw[i].as_str() {
            "--n"     => { i += 1; a.n     = raw[i].parse().unwrap(); }
            "--omega" => { i += 1; a.omega = raw[i].parse().unwrap(); }
            "--sigma" => { i += 1; a.sigma = raw[i].parse().unwrap(); }
            other     => eprintln!("unknown arg: {other}"),
        }
        i += 1;
    }
    a
}

// ─── main ─────────────────────────────────────────────────────────────────────

fn main() {
    let args  = parse_args();
    let n     = args.n;
    let omega = args.omega;
    let sigma = args.sigma;

    println!("=== fem-rs Example 22: Complex Helmholtz (2×2 real-block) ===");
    println!("  Mesh: {n}×{n},  ω = {omega:.4},  σ = {sigma:.4}");

    // ─── 1. Mesh + H¹ space ──────────────────────────────────────────────────
    let mesh  = SimplexMesh::<2>::unit_square_tri(n);
    let space = H1Space::new(mesh, 1);
    let ndofs = space.n_dofs();
    println!("  DOFs: {ndofs}  (2×{ndofs} flat system)");

    // ─── 2. Assemble complex system ───────────────────────────────────────────
    //   K = ∫ ∇φᵢ·∇φⱼ dx  (diffusion)
    //   M = ∫ φᵢ φⱼ dx     (mass)
    //   C = σ·M             (conductivity damping)
    //   k_re = K − ω²M,  k_im = ωC
    let mut sys = ComplexAssembler::assemble(
        &space,
        &[&DiffusionIntegrator { kappa: 1.0 }],
        &[&MassIntegrator      { rho:   1.0 }],
        &[&MassIntegrator      { rho: sigma }],
        omega,
        3,
    );

    // ─── 3. Assemble zero RHS (BCs drive the solution) ────────────────────────
    let f_re = vec![0.0_f64; ndofs];
    let f_im = vec![0.0_f64; ndofs];
    let mut rhs = sys.assemble_rhs(&f_re, &f_im);

    // ─── 4. Dirichlet BCs ────────────────────────────────────────────────────
    //   tag 4 = left wall  → u = 1 + 0i
    //   tags 1,2,3 = other walls → u = 0 + 0i
    let dm                  = space.dof_manager();
    let mesh_ref            = space.mesh();
    let left_dofs:  Vec<_>  = boundary_dofs(mesh_ref, dm, &[4])
                               .into_iter().map(|d| d as usize).collect();
    let other_dofs: Vec<_>  = boundary_dofs(mesh_ref, dm, &[1, 2, 3])
                               .into_iter().map(|d| d as usize).collect();

    let left_re:  Vec<f64> = vec![1.0; left_dofs.len()];
    let left_im:  Vec<f64> = vec![0.0; left_dofs.len()];
    let other_re: Vec<f64> = vec![0.0; other_dofs.len()];
    let other_im: Vec<f64> = vec![0.0; other_dofs.len()];

    // Apply zero BCs on other walls first, then left wall — so left wall wins
    // at corner nodes that appear in both edge sets.
    sys.apply_dirichlet(&other_dofs, &other_re, &other_im, &mut rhs);
    sys.apply_dirichlet(&left_dofs,  &left_re,  &left_im,  &mut rhs);
    // ─── 5. Build flat 2n×2n system and solve with GMRES ─────────────────────
    let flat = sys.to_flat_csr();
    let mut x = vec![0.0_f64; 2 * ndofs];
    let cfg = SolverConfig {
        rtol:     1e-8,
        atol:     1e-14,
        max_iter: 3000,
        verbose:  false,
        ..SolverConfig::default()
    };
    let res = solve_gmres(&flat, &rhs, &mut x, 50, &cfg)
        .expect("GMRES did not converge");
    println!("  GMRES: {} iters,  residual = {:.3e},  converged = {}",
             res.iterations, res.final_residual, res.converged);

    // ─── 6. Extract solution and verify BCs ──────────────────────────────────
    let gf  = ComplexGridFunction::from_flat(&x);
    let amp = gf.amplitude();
    let amp_max = amp.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let amp_min = amp.iter().cloned().fold(f64::INFINITY,     f64::min);
    println!("  |u| ∈ [{amp_min:.4}, {amp_max:.4}]");

    // BC check: left wall nodes must have u_re ≈ 1
    let max_bc_err = left_dofs.iter()
        .map(|&i| (gf.u_re[i] - 1.0).abs())
        .fold(0.0_f64, f64::max);
    println!("  Max left-BC error: {max_bc_err:.2e}  (should be < 1e-10)");
    assert!(max_bc_err < 1e-10,
            "Left Dirichlet BC not satisfied: max error = {max_bc_err}");

    assert!(res.converged, "GMRES did not converge");
    println!("  ✓ Example 22 passed");
}
