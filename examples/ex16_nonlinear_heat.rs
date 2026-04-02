//! # Example 16 — Nonlinear Heat Equation (Newton)  (analogous to MFEM ex16)
//!
//! Solves the nonlinear heat equation with conductivity κ(u) = 1 + u²:
//!
//! ```text
//!   −∇·(κ(u) ∇u) = f    in Ω = [0,1]²
//!              u = 0    on ∂Ω
//! ```
//!
//! Uses Newton–Raphson iteration with Picard Jacobian:
//! ```text
//!   J(uₙ) Δu = −F(uₙ),    uₙ₊₁ = uₙ + Δu
//!   F(u) = ∫ κ(u) ∇u·∇v dx − ∫ f v dx
//!   J(u) ≈ ∫ κ(u) ∇φⱼ·∇φᵢ dx   (Picard / frozen-κ Jacobian)
//! ```
//!
//! Manufactured solution approach: choose `u* = sin(πx)sin(πy)` and compute
//! `f = −∇·((1+u*²)∇u*)` analytically:
//!
//! For u* = sin(πx)sin(πy), κ(u*) = 1 + sin²(πx)sin²(πy):
//! ```text
//!   f = π²(2 + sin²(πx)sin²(πy)) sin(πx)sin(πy)
//!       − 2π² sin³(πx)sin(πy)cos²(πx) − 2π² sin(πx)sin³(πy)cos²(πy)
//! ```
//! (simplified below)
//!
//! ## Usage
//! ```
//! cargo run --example ex16_nonlinear_heat
//! cargo run --example ex16_nonlinear_heat -- --n 16 --newton-tol 1e-10
//! ```

use std::f64::consts::PI;

use fem_assembly::{Assembler, nonlinear::{NonlinearDiffusionForm, NewtonSolver, NewtonConfig}};
use fem_mesh::SimplexMesh;
use fem_space::{H1Space, fe_space::FESpace, constraints::{apply_dirichlet, boundary_dofs}};

fn main() {
    let args = parse_args();
    println!("=== fem-rs Example 16: Nonlinear heat equation (Newton) ===");
    println!("  Mesh: {}×{} subdivisions, P1 elements", args.n, args.n);
    println!("  κ(u) = 1 + u²,  Newton tol = {:.0e}", args.newton_tol);

    // ─── 1. Mesh and H¹ space ─────────────────────────────────────────────────
    let mesh  = SimplexMesh::<2>::unit_square_tri(args.n);
    let space = H1Space::new(mesh, 1);
    let n     = space.n_dofs();
    println!("  DOFs: {n}");

    // ─── 2. Identify Dirichlet DOFs ───────────────────────────────────────────
    let dm   = space.dof_manager();
    let bnd  = boundary_dofs(space.mesh(), dm, &[1, 2, 3, 4]);
    // Dirichlet: u = 0 on all walls
    let dirichlet: Vec<(usize, f64)> = bnd.iter().map(|&d| (d as usize, 0.0)).collect();

    // ─── 3. Assemble RHS f = manufactured source ─────────────────────────────
    // f = (2 + 3sin²(πx)sin²(πy)) * π² * sin(πx)sin(πy)
    //   This comes from: -div((1+u²)∇u) where u = sin(πx)sin(πy):
    //   ∂u/∂x = π cos(πx) sin(πy),  ∂²u/∂x² = -π² sin(πx)sin(πy)
    //   κ(u) = 1 + sin²(πx)sin²(πy)
    //   -div(κ∇u) = -κ Δu - ∇κ·∇u = κ·2π²·u - ∇κ·∇u
    //   ∇κ = (2u ∂u/∂x, 2u ∂u/∂y)
    //   ∇κ·∇u = 2u(|∂u/∂x|² + |∂u/∂y|²) = 2u · π²(cos²(πx)sin²(πy) + sin²(πx)cos²(πy))
    //          = 2u · π²(cos²(πx)sin²(πy) + sin²(πx)cos²(πy))
    //   Combined: f = (1+u²)·2π²·u - 2u·π²(...)
    use fem_assembly::standard::DomainSourceIntegrator;
    let src = DomainSourceIntegrator::new(|x: &[f64]| {
        let (sx, sy) = ((PI * x[0]).sin(), (PI * x[1]).sin());
        let (cx, cy) = ((PI * x[0]).cos(), (PI * x[1]).cos());
        let u_star = sx * sy;
        let kappa  = 1.0 + u_star * u_star;
        let lap_u  = -2.0 * PI * PI * u_star;
        let grad_kappa_dot_grad_u = 2.0 * u_star * PI * PI *
            (cx * cx * sy * sy + sx * sx * cy * cy);
        -kappa * lap_u - grad_kappa_dot_grad_u
    });
    let mesh2 = SimplexMesh::<2>::unit_square_tri(args.n);
    let space2 = H1Space::new(mesh2, 1);
    let rhs = Assembler::assemble_linear(&space2, &[&src], 5);

    // ─── 4. Build nonlinear form ──────────────────────────────────────────────
    let mut form = NonlinearDiffusionForm::new(
        space,
        |u: f64| 1.0 + u * u,   // κ(u) = 1 + u²
        3,
    );
    form.set_dirichlet(dirichlet);

    // ─── 5. Newton solve ──────────────────────────────────────────────────────
    let cfg = NewtonConfig {
        atol:       args.newton_tol,
        rtol:       args.newton_tol * 1e2,
        max_iter:   50,
        linear_tol: args.newton_tol * 0.1,
        verbose:    true,
    };
    let solver = NewtonSolver::new(cfg);
    let mut u = vec![0.0_f64; n];

    match solver.solve(&form, &rhs, &mut u) {
        Ok(r) => println!("\n  Newton converged: {} iters, ‖F‖ = {:.3e}", r.iterations, r.final_residual),
        Err(r) => println!("\n  Newton did NOT converge: {} iters, ‖F‖ = {:.3e}", r.iterations, r.final_residual),
    }

    // ─── 6. L² error ─────────────────────────────────────────────────────────
    let dm2 = space2.dof_manager();
    let l2 = {
        let mut err = 0.0_f64;
        for i in 0..n {
            let x = dm2.dof_coord(i as u32);
            let u_ex = (PI * x[0]).sin() * (PI * x[1]).sin();
            err += (u[i] - u_ex).powi(2);
        }
        (err / n as f64).sqrt()
    };
    let h = 1.0 / args.n as f64;
    println!("  h = {h:.4e},  nodal RMS error = {l2:.4e}");
    println!("  (Expected O(h²) for P1 manufactured solution)");
    println!("\nDone.");
}

// ─── CLI ─────────────────────────────────────────────────────────────────────

struct Args { n: usize, newton_tol: f64 }

fn parse_args() -> Args {
    let mut a = Args { n: 16, newton_tol: 1e-10 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--n"          => { a.n          = it.next().unwrap_or("16".into()).parse().unwrap_or(16); }
            "--newton-tol" => { a.newton_tol = it.next().unwrap_or("1e-10".into()).parse().unwrap_or(1e-10); }
            _ => {}
        }
    }
    a
}
