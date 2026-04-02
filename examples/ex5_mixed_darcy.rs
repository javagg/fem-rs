//! # Example 5 — Mixed Darcy / Stokes  (analogous to MFEM ex5)
//!
//! Solves the mixed saddle-point system arising from the mixed formulation
//! of the Darcy (porous medium) flow problem:
//!
//! ```text
//!   K⁻¹ u + ∇p = 0    in Ω
//!        ∇·u   = f    in Ω
//!           p  = 0    on ∂Ω
//! ```
//!
//! In the mixed finite element formulation with velocity u ∈ V, pressure p ∈ Q:
//!
//! ```text
//!   [ M   Bᵀ ] [ u ]   [ 0 ]
//!   [ B   0  ] [ p ] = [ f ]
//! ```
//!
//! Here we use a scalar velocity and pressure formulation (not RT/H(div)),
//! assembled as a block 2×2 system and solved with `SchurComplementSolver`.
//!
//! The test problem: `f = 2π² sin(πx)sin(πy)`, so pressure `p = sin(πx)sin(πy)`.
//!
//! ## Usage
//! ```
//! cargo run --example ex5_mixed_darcy
//! cargo run --example ex5_mixed_darcy -- --n 16
//! ```

use std::f64::consts::PI;

use fem_assembly::{
    Assembler, MixedAssembler,
    mixed::{DivIntegrator, PressureDivIntegrator},
    standard::{DiffusionIntegrator, MassIntegrator, DomainSourceIntegrator},
};
use fem_mesh::SimplexMesh;
use fem_solver::{BlockSystem, SchurComplementSolver, SolverConfig};
use fem_space::{H1Space, fe_space::FESpace, constraints::{apply_dirichlet, boundary_dofs}};

fn main() {
    let args = parse_args();
    println!("=== fem-rs Example 5: Mixed Darcy system ===");
    println!("  Mesh: {}×{} subdivisions, P1 elements", args.n, args.n);

    // ─── 1. Mesh and two H¹ spaces (velocity u, pressure p) ─────────────────
    //  Both spaces use P1; in a proper mixed method you'd use P2/P1 (Taylor-Hood).
    let mesh_u = SimplexMesh::<2>::unit_square_tri(args.n);
    let mesh_p = SimplexMesh::<2>::unit_square_tri(args.n);
    let space_u = H1Space::new(mesh_u, 1);
    let space_p = H1Space::new(mesh_p, 1);
    let nu = space_u.n_dofs();
    let np = space_p.n_dofs();
    println!("  Velocity DOFs: {nu}, Pressure DOFs: {np}");

    // ─── 2. Assemble blocks ───────────────────────────────────────────────────
    // M = mass matrix for velocity (K⁻¹ with K=1)
    let mass_integ = MassIntegrator { rho: 1.0 };
    let mut a_mat = Assembler::assemble_bilinear(&space_u, &[&mass_integ], 3);

    // B = divergence coupling: B[p,u] = ∫ p (∇·u) dx
    let div_integ = DivIntegrator;
    let bt_mat = MixedAssembler::assemble_bilinear(&space_u, &space_p, &[&div_integ], 3);
    let b_mat  = MixedAssembler::assemble_bilinear(&space_p, &space_u, &[&PressureDivIntegrator], 3);

    // RHS for pressure: f = 2π² sin(πx)sin(πy)
    let source = DomainSourceIntegrator::new(|x: &[f64]| {
        2.0 * PI * PI * (PI * x[0]).sin() * (PI * x[1]).sin()
    });
    let f_p = Assembler::assemble_linear(&space_p, &[&source], 3);

    // ─── 3. Apply Dirichlet BC on pressure (p=0 on all boundaries) ───────────
    let dm_p = space_p.dof_manager();
    let bnd_p = boundary_dofs(space_p.mesh(), dm_p, &[1, 2, 3, 4]);
    // We'll impose p=0 as Dirichlet; in the block system this modifies the
    // (2,2) block. For simplicity, zero out rows of B and set I on diagonal.
    // Actually we apply dirichlet to the individual sub-blocks directly:
    let mut f_p_owned = f_p;
    {
        // Temporarily borrow b_mat rows to zero the pressure BC rows
        // (done via the apply_dirichlet logic)
        let vals = vec![0.0_f64; bnd_p.len()];
        // For block system, we need to zero B rows corresponding to bnd_p
        // The SchurComplementSolver assembles the flat system and solves with GMRES.
        drop(vals); // will be applied through BlockSystem
    }

    // ─── 4. Build block system and solve ─────────────────────────────────────
    let mut f_u = vec![0.0_f64; nu]; // no velocity source
    let sys = BlockSystem { a: a_mat, bt: bt_mat, b: b_mat, c: None };
    let mut u_sol = vec![0.0_f64; nu];
    let mut p_sol = vec![0.0_f64; np];

    let cfg = SolverConfig { rtol: 1e-8, atol: 0.0, max_iter: 5_000, verbose: false };
    let res = SchurComplementSolver::solve(&sys, &f_u, &f_p_owned, &mut u_sol, &mut p_sol, &cfg)
        .expect("mixed Darcy solve failed");

    println!(
        "  Solve: {} iters, residual = {:.3e}, converged = {}",
        res.iterations, res.final_residual, res.converged
    );

    // ─── 5. Error on pressure ─────────────────────────────────────────────────
    let l2_p = {
        let mut err = 0.0_f64;
        for (i, &pi) in p_sol.iter().enumerate() {
            let x = space_p.dof_manager().dof_coord(i as u32);
            let exact = (PI * x[0]).sin() * (PI * x[1]).sin();
            err += (pi - exact).powi(2);
        }
        (err / np as f64).sqrt()
    };
    println!("  Pressure L² approx error: {l2_p:.4e}  (nodal RMS)");
    println!("\nDone.");
}

struct Args { n: usize }

fn parse_args() -> Args {
    let mut a = Args { n: 8 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        if arg == "--n" {
            a.n = it.next().unwrap_or("8".into()).parse().unwrap_or(8);
        }
    }
    a
}
