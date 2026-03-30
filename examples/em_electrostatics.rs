//! # Electrostatics Example
//!
//! Solves the electrostatic Poisson equation:
//!
//! ```text
//!   -∇·(ε ∇φ) = ρ    in Ω
//!          φ = φ_D    on ∂Ω_D   (Dirichlet)
//!  ε ∂φ/∂n = σ        on ∂Ω_N   (Neumann / surface charge)
//! ```
//!
//! Two built-in test cases are provided:
//!
//! 1. **Parallel plate capacitor** (default)
//!    - Domain: unit square [0,1]²
//!    - φ = 0 on bottom (tag 1), φ = 1 V on top (tag 3)
//!    - Neumann (zero flux) on left/right walls
//!    - Analytical solution: φ(x,y) = y
//!
//! 2. **Point charge** (`--case point_charge`)
//!    - Domain: unit square
//!    - φ = 0 on all boundaries
//!    - ρ = 1/(ε₀) at domain centre (approximated as piecewise constant)
//!    - No analytical solution
//!
//! 3. **From .msh file** (`--mesh path/to/mesh.msh`)
//!    - Reads any GMSH v4.1 ASCII mesh
//!    - Boundary conditions controlled by `--dirichlet-tags` and `--neumann-tags`
//!
//! ## Usage
//! ```
//! cargo run --example em_electrostatics
//! cargo run --example em_electrostatics -- --mesh examples/meshes/coaxial.msh \
//!     --dirichlet-tags 1,2 --case coaxial
//! ```
//!
//! ## Output
//! Writes `output/electrostatics.vtk` for ParaView.

use std::collections::HashMap;

use fem_examples::{
    dirichlet_nodes_fn, l2_error_p1, p1_assemble_poisson,
    p1_gradient_2d, p1_neumann_load, solve_dirichlet_reduced, write_vtk_scalar_vector,
};
use fem_io::read_msh_file;
use fem_mesh::SimplexMesh;

// ============================================================================
// Constants
// ============================================================================

/// Permittivity of free space [F/m]
const EPS0: f64 = 8.854_187_817e-12;

// ============================================================================
// main
// ============================================================================

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = parse_args();

    log::info!("=== fem-rs Electrostatics Example ===");
    log::info!("Case: {}", args.case);

    // -----------------------------------------------------------------------
    // 1. Load / generate mesh
    // -----------------------------------------------------------------------
    let mesh = match &args.mesh_path {
        Some(path) => {
            log::info!("Loading mesh: {path}");
            let msh = read_msh_file(path).expect("failed to read .msh file");
            msh.into_2d().expect("expected a 2-D mesh")
        }
        None => {
            let n = args.mesh_n;
            log::info!("Generating unit-square mesh ({n}×{n} squares, {} triangles)", 2 * n * n);
            SimplexMesh::<2>::unit_square_tri(n)
        }
    };

    log::info!(
        "Mesh: {} nodes, {} elements, {} boundary edges",
        mesh.n_nodes(), mesh.n_elems(), mesh.n_faces()
    );

    // -----------------------------------------------------------------------
    // 2. Problem coefficients and BCs (case-specific)
    // -----------------------------------------------------------------------
    let problem = match args.case.as_str() {
        "parallel_plate" => parallel_plate_problem(&mesh, &args),
        "point_charge"   => point_charge_problem(&mesh, &args),
        "coaxial"        => coaxial_problem(&mesh, &args),
        other => {
            eprintln!("Unknown case '{other}'. Available: parallel_plate, point_charge, coaxial");
            std::process::exit(1);
        }
    };

    // -----------------------------------------------------------------------
    // 3. Assemble stiffness matrix K and load vector f
    // -----------------------------------------------------------------------
    log::info!("Assembling system matrix …");
    let t0 = std::time::Instant::now();

    let (mut mat, mut rhs) = p1_assemble_poisson(
        &mesh,
        |x, y| problem.epsilon(x, y),
        |x, y| problem.rho(x, y),
    );

    // Add Neumann contributions
    p1_neumann_load(&mesh, &problem.neumann_tags, |x, y| problem.sigma(x, y), &mut rhs);

    log::info!("  Assembly: {:.3} ms, NNZ = {}", t0.elapsed().as_secs_f64() * 1e3, mat.nnz());

    // -----------------------------------------------------------------------
    // 4. Identify Dirichlet nodes and solve the reduced free-DOF system
    // -----------------------------------------------------------------------
    let bcs = dirichlet_nodes_fn(&mesh, &problem.dirichlet_tags, |x, y| problem.phi_d(x, y));
    log::info!("  Dirichlet DOFs: {}", bcs.len());

    // -----------------------------------------------------------------------
    // 5. Solve K φ = f  (reduced system, no symmetric elimination artefacts)
    // -----------------------------------------------------------------------
    log::info!("Solving with PCG on free DOFs (tol = {:.0e}) …", args.tol);
    let t1 = std::time::Instant::now();
    let (phi, iters, residual) = solve_dirichlet_reduced(&mat, &rhs, &bcs, args.tol, args.max_iter);
    let solve_time = t1.elapsed().as_secs_f64() * 1e3;

    log::info!(
        "  Converged: {} iters, residual = {:.3e}, time = {:.3} ms",
        iters, residual, solve_time
    );

    // -----------------------------------------------------------------------
    // 6. Post-process: compute E = -∇φ
    // -----------------------------------------------------------------------
    let (gx, gy) = p1_gradient_2d(&mesh, &phi);
    // E = -∇φ
    let ex: Vec<f64> = gx.iter().map(|v| -v).collect();
    let ey: Vec<f64> = gy.iter().map(|v| -v).collect();

    // -----------------------------------------------------------------------
    // 7. Error analysis (if exact solution known)
    // -----------------------------------------------------------------------
    if let Some(exact) = &problem.exact_solution {
        let l2 = l2_error_p1(&mesh, &phi, |x, y| exact(x, y));
        log::info!("  L2 error = {l2:.4e}");
        print_convergence_hint(mesh.n_nodes(), l2);
    }

    // -----------------------------------------------------------------------
    // 8. Output
    // -----------------------------------------------------------------------
    let out_path = "output/electrostatics.vtk";
    log::info!("Writing {out_path} …");
    write_vtk_scalar_vector(out_path, &mesh, &phi, "potential_V", &ex, &ey, "E_field_Vm")
        .expect("VTK write failed");

    print_summary(&phi, &ex, &ey, &mesh);
}

// ============================================================================
// Problem definitions
// ============================================================================

struct Problem {
    epsilon:        Box<dyn Fn(f64, f64) -> f64>,
    rho:            Box<dyn Fn(f64, f64) -> f64>,
    sigma:          Box<dyn Fn(f64, f64) -> f64>,
    phi_d:          Box<dyn Fn(f64, f64) -> f64>,
    dirichlet_tags: Vec<i32>,
    neumann_tags:   Vec<i32>,
    exact_solution: Option<Box<dyn Fn(f64, f64) -> f64>>,
}

impl Problem {
    fn epsilon(&self, x: f64, y: f64) -> f64 { (self.epsilon)(x, y) }
    fn rho(&self,     x: f64, y: f64) -> f64 { (self.rho)(x, y)     }
    fn sigma(&self,   x: f64, y: f64) -> f64 { (self.sigma)(x, y)   }
    fn phi_d(&self,   x: f64, y: f64) -> f64 { (self.phi_d)(x, y)   }
}

/// **Parallel plate capacitor**
///
/// ```text
///   -ε₀ ∇²φ = 0  in [0,1]²
///   φ = 0  on y=0  (bottom, tag 1)
///   φ = 1  on y=1  (top,    tag 3)
///   ∂φ/∂n = 0 on x=0,1  (Neumann, tags 2,4)
///
///   Exact: φ(x,y) = y
/// ```
fn parallel_plate_problem(_mesh: &SimplexMesh<2>, _args: &Args) -> Problem {
    Problem {
        epsilon:        Box::new(|_x, _y| EPS0),
        rho:            Box::new(|_x, _y| 0.0),
        sigma:          Box::new(|_x, _y| 0.0),
        phi_d:          Box::new(|_x, y | y.round()),   // 0 or 1 based on position
        dirichlet_tags: vec![1, 3],
        neumann_tags:   vec![2, 4],
        // Exact: φ = y; note phi_d uses rounded y, which is correct for tags 1 (y=0) and 3 (y=1)
        exact_solution: Some(Box::new(|_x, y| y)),
    }
}

/// **Point charge at centre**
///
/// ```text
///   -ε₀ ∇²φ = δ(x-0.5, y-0.5)    (approximated)
///   φ = 0 on all boundaries
/// ```
fn point_charge_problem(mesh: &SimplexMesh<2>, _args: &Args) -> Problem {
    // Approximate δ by a uniform source over elements containing the centre
    let cx = 0.5f64; let cy = 0.5f64;
    let r_src = 0.1; // source radius
    Problem {
        epsilon: Box::new(|_x, _y| EPS0),
        rho: Box::new(move |x, y| {
            let d = ((x - cx).powi(2) + (y - cy).powi(2)).sqrt();
            if d < r_src { 1.0 / (std::f64::consts::PI * r_src * r_src) } else { 0.0 }
        }),
        sigma:          Box::new(|_x, _y| 0.0),
        phi_d:          Box::new(|_x, _y| 0.0),
        dirichlet_tags: vec![1, 2, 3, 4],
        neumann_tags:   vec![],
        exact_solution: None,
    }
}

/// **Coaxial cable cross-section**
///
/// Inner conductor (tag 1): φ = V_inner
/// Outer conductor (tag 2): φ = 0
///
/// The analytical solution for a coaxial cable with inner radius r_i and outer radius r_o is:
/// φ(r) = V_inner * ln(r/r_o) / ln(r_i/r_o)
fn coaxial_problem(_mesh: &SimplexMesh<2>, args: &Args) -> Problem {
    let v_inner = args.voltage.unwrap_or(1.0);
    let r_i = 0.2f64; // inner conductor radius
    let r_o = 0.8f64; // outer conductor radius
    let ln_ratio = (r_i / r_o).ln();

    Problem {
        epsilon:        Box::new(|_x, _y| EPS0),
        rho:            Box::new(|_x, _y| 0.0),
        sigma:          Box::new(|_x, _y| 0.0),
        phi_d: Box::new(move |x, y| {
            let r = (x * x + y * y).sqrt();
            if r <= r_i * 1.01 {
                v_inner
            } else {
                0.0
            }
        }),
        dirichlet_tags: args.dirichlet_tags.clone().unwrap_or_else(|| vec![1, 2]),
        neumann_tags:   vec![],
        exact_solution: Some(Box::new(move |x, y| {
            let r = (x * x + y * y).sqrt().max(1e-30);
            v_inner * (r / r_o).ln() / ln_ratio
        })),
    }
}

// ============================================================================
// Argument parsing
// ============================================================================

struct Args {
    case:            String,
    mesh_path:       Option<String>,
    mesh_n:          usize,
    tol:             f64,
    max_iter:        usize,
    dirichlet_tags:  Option<Vec<i32>>,
    voltage:         Option<f64>,
}

fn parse_args() -> Args {
    let mut args = Args {
        case:           "parallel_plate".to_string(),
        mesh_path:      None,
        mesh_n:         32,
        tol:            1e-10,
        max_iter:       10_000,
        dirichlet_tags: None,
        voltage:        None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--case"           => { args.case        = it.next().unwrap_or_default(); }
            "--mesh"           => { args.mesh_path   = Some(it.next().unwrap_or_default()); }
            "--n"              => { args.mesh_n      = it.next().unwrap_or("32".into()).parse().unwrap_or(32); }
            "--tol"            => { args.tol         = it.next().unwrap_or("1e-10".into()).parse().unwrap_or(1e-10); }
            "--max-iter"       => { args.max_iter    = it.next().unwrap_or("10000".into()).parse().unwrap_or(10_000); }
            "--dirichlet-tags" => {
                let s = it.next().unwrap_or_default();
                args.dirichlet_tags = Some(s.split(',').filter_map(|x| x.trim().parse().ok()).collect());
            }
            "--voltage"        => { args.voltage     = it.next().and_then(|s| s.parse().ok()); }
            _                  => {}
        }
    }
    args
}

// ============================================================================
// Helpers
// ============================================================================

fn print_summary(phi: &[f64], ex: &[f64], ey: &[f64], mesh: &SimplexMesh<2>) {
    let phi_min = phi.iter().cloned().fold(f64::INFINITY,  f64::min);
    let phi_max = phi.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let e_max = ex.iter().zip(ey.iter())
        .map(|(&ex, &ey)| (ex * ex + ey * ey).sqrt())
        .fold(0.0f64, f64::max);
    println!("\n--- Results ---");
    println!("  Nodes:      {}", mesh.n_nodes());
    println!("  Elements:   {}", mesh.n_elems());
    println!("  φ range:    [{phi_min:.6e}, {phi_max:.6e}] V");
    println!("  |E|_max:    {e_max:.6e} V/m");
    println!("  Output:     output/electrostatics.vtk");
}

fn print_convergence_hint(n_nodes: usize, l2: f64) {
    // Simple hint so users can run with different mesh sizes and observe O(h²) convergence
    let h_approx = 1.0 / (n_nodes as f64).sqrt();
    println!("  h ≈ {h_approx:.4e},  L2 error = {l2:.4e}  (expected O(h²) for P1)");
}
