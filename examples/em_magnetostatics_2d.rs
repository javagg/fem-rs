//! # 2-D Magnetostatics Example (A_z formulation)
//!
//! Solves the 2-D magnetostatic problem in terms of the z-component of the
//! magnetic vector potential A_z:
//!
//! ```text
//!   -∇·(ν ∇A_z) = J_z    in Ω
//!          A_z = 0        on ∂Ω    (homogeneous Dirichlet)
//! ```
//!
//! where:
//! - ν = 1/μ  is the magnetic reluctivity [m/H]
//! - A_z      is the z-component of the magnetic vector potential [Wb/m]
//! - J_z      is the z-component of the current density [A/m²]
//!
//! Once A_z is found, the magnetic flux density is recovered as:
//! ```text
//!   B_x =  ∂A_z/∂y
//!   B_y = -∂A_z/∂x
//! ```
//!
//! ## Built-in test cases
//!
//! 1. **Square conductor** (default)
//!    - Domain: unit square [0,1]²
//!    - J_z = 1 A/m² in the central sub-region [0.3, 0.7]²
//!    - ν = ν₀ = 1/(4π×10⁻⁷) everywhere (free space)
//!    - A_z = 0 on all boundaries
//!
//! 2. **Two-conductor** (`--case two_conductors`)
//!    - Two rectangular conductors with equal and opposite currents
//!    - Illustrates force computation possibility
//!
//! 3. **Transformer core** (`--case transformer`)
//!    - Iron core with μ_r = 1000, copper winding
//!    - Demonstrates heterogeneous materials
//!
//! 4. **From .msh file** (`--mesh path/to/mesh.msh`)
//!
//! ## Usage
//! ```
//! cargo run --example em_magnetostatics_2d
//! cargo run --example em_magnetostatics_2d -- --case two_conductors --n 64
//! cargo run --example em_magnetostatics_2d -- --mesh examples/meshes/coil.msh
//! ```
//!
//! ## Output
//! Writes `output/magnetostatics.vtk` (A_z potential + B-field vectors).

use fem_mesh::MeshTopology as _;

use fem_examples::{
    dirichlet_nodes, solve_dirichlet_reduced, p1_assemble_poisson,
    p1_gradient_2d, write_vtk_scalar_vector,
};
use fem_io::read_msh_file;
use fem_mesh::SimplexMesh;

// ============================================================================
// Physical constants
// ============================================================================

/// Permeability of free space [H/m]
const MU0: f64 = 4.0 * std::f64::consts::PI * 1.0e-7;

/// Reluctivity of free space [m/H]
const NU0: f64 = 1.0 / MU0;

// ============================================================================
// main
// ============================================================================

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = parse_args();

    log::info!("=== fem-rs 2-D Magnetostatics Example ===");
    log::info!("Case: {}", args.case);

    // -----------------------------------------------------------------------
    // 1. Load / generate mesh
    // -----------------------------------------------------------------------
    let mesh = match &args.mesh_path {
        Some(path) => {
            log::info!("Loading mesh: {path}");
            read_msh_file(path)
                .expect("failed to read .msh file")
                .into_2d()
                .expect("expected a 2-D mesh")
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
    // 2. Problem definition (case-specific)
    // -----------------------------------------------------------------------
    let problem = match args.case.as_str() {
        "square_conductor" => square_conductor_problem(&args),
        "two_conductors"   => two_conductors_problem(&args),
        "transformer"      => transformer_problem(&args),
        other => {
            eprintln!("Unknown case '{other}'. Available: square_conductor, two_conductors, transformer");
            std::process::exit(1);
        }
    };

    // -----------------------------------------------------------------------
    // 3. Assemble -∇·(ν ∇A_z) = J_z
    // -----------------------------------------------------------------------
    log::info!("Assembling system matrix …");
    let t0 = std::time::Instant::now();

    let (mut mat, mut rhs) = p1_assemble_poisson(
        &mesh,
        |x, y| problem.nu(x, y),
        |x, y| problem.j_z(x, y),
    );

    log::info!("  Assembly: {:.3} ms, NNZ = {}", t0.elapsed().as_secs_f64() * 1e3, mat.nnz());

    // -----------------------------------------------------------------------
    // 4. Homogeneous Dirichlet BC: A_z = 0 on entire boundary
    // -----------------------------------------------------------------------
    let all_boundary_tags: Vec<i32> = {
        let mut tags: Vec<i32> = mesh.face_tags.iter().copied().collect::<std::collections::HashSet<_>>()
            .into_iter().collect();
        tags.sort();
        tags
    };
    let bcs = dirichlet_nodes(&mesh, &all_boundary_tags);
    log::info!("  Dirichlet DOFs: {}", bcs.len());

    // -----------------------------------------------------------------------
    // 5. Solve K A_z = f  (reduced free-DOF system)
    // -----------------------------------------------------------------------
    log::info!("Solving with PCG on free DOFs (tol = {:.0e}) …", args.tol);
    let t1 = std::time::Instant::now();
    let (az, iters, residual) = solve_dirichlet_reduced(&mat, &rhs, &bcs, args.tol, args.max_iter);
    let solve_time = t1.elapsed().as_secs_f64() * 1e3;

    log::info!(
        "  Converged: {} iters, residual = {:.3e}, time = {:.3} ms",
        iters, residual, solve_time
    );

    // -----------------------------------------------------------------------
    // 6. Post-process: B = ∇×A
    //    B_x =  ∂A_z/∂y    B_y = -∂A_z/∂x
    // -----------------------------------------------------------------------
    let (gx, gy) = p1_gradient_2d(&mesh, &az);
    //  gx = ∂A_z/∂x,  gy = ∂A_z/∂y
    let bx: Vec<f64> = gy.iter().copied().collect();          // B_x =  ∂A_z/∂y
    let by: Vec<f64> = gx.iter().map(|v| -v).collect();      // B_y = -∂A_z/∂x

    // -----------------------------------------------------------------------
    // 7. Derived quantities
    // -----------------------------------------------------------------------
    let b_max = bx.iter().zip(by.iter())
        .map(|(&bxi, &byi)| (bxi * bxi + byi * byi).sqrt())
        .fold(0.0f64, f64::max);

    // Stored magnetic energy W = ∫ ν/2 |B|² dΩ  (element-averaged P0)
    let energy = compute_magnetic_energy(&mesh, &bx, &by, |x, y| problem.nu(x, y));
    log::info!("  |B|_max = {b_max:.4e} T,  W_mag = {energy:.4e} J/m");

    // -----------------------------------------------------------------------
    // 8. Output
    // -----------------------------------------------------------------------
    let out_path = "output/magnetostatics.vtk";
    log::info!("Writing {out_path} …");
    write_vtk_scalar_vector(out_path, &mesh, &az, "Az_Wb_per_m", &bx, &by, "B_field_T")
        .expect("VTK write failed");

    print_summary(&az, &bx, &by, &mesh, energy, b_max);
}

// ============================================================================
// Problem definitions
// ============================================================================

struct Problem {
    nu:  Box<dyn Fn(f64, f64) -> f64>,  // magnetic reluctivity [m/H]
    j_z: Box<dyn Fn(f64, f64) -> f64>,  // current density [A/m²]
}

impl Problem {
    fn nu(&self, x: f64, y: f64) -> f64  { (self.nu)(x, y)  }
    fn j_z(&self, x: f64, y: f64) -> f64 { (self.j_z)(x, y) }
}

/// **Square conductor**: single conductor with uniform current density
///
/// ```text
///  ┌─────────────────────────────┐
///  │                             │
///  │    ┌──────────────┐         │
///  │    │   J_z = J    │  Air    │  A_z = 0 on outer boundary
///  │    └──────────────┘         │
///  │                             │
///  └─────────────────────────────┘
/// ```
fn square_conductor_problem(args: &Args) -> Problem {
    let j_val = args.current_density.unwrap_or(1.0e6); // 1 MA/m²
    let x1 = 0.3f64; let x2 = 0.7f64;
    let y1 = 0.3f64; let y2 = 0.7f64;

    Problem {
        nu:  Box::new(|_x, _y| NU0),
        j_z: Box::new(move |x, y| {
            if x >= x1 && x <= x2 && y >= y1 && y <= y2 { j_val } else { 0.0 }
        }),
    }
}

/// **Two anti-parallel conductors**: net force between conductors
///
/// ```text
///  ┌──────────────────────────────────────────┐
///  │                                          │
///  │  ┌───────┐              ┌───────┐        │
///  │  │ +J_z  │    Air gap   │ -J_z  │        │  A_z = 0 on boundary
///  │  └───────┘              └───────┘        │
///  │                                          │
///  └──────────────────────────────────────────┘
/// ```
fn two_conductors_problem(args: &Args) -> Problem {
    let j_val = args.current_density.unwrap_or(1.0e6);
    Problem {
        nu: Box::new(|_x, _y| NU0),
        j_z: Box::new(move |x, y| {
            if x >= 0.1 && x <= 0.3 && y >= 0.3 && y <= 0.7 {
                 j_val   // positive current (out of page)
            } else if x >= 0.7 && x <= 0.9 && y >= 0.3 && y <= 0.7 {
                -j_val   // negative current (into page)
            } else {
                0.0
            }
        }),
    }
}

/// **Transformer cross-section**
///
/// Simplified transformer with an iron core (μ_r = 1000) and air gaps.
///
/// ```text
///  ┌──────────────────────────────────────────┐
///  │  Iron core (μ_r = 1000)                  │
///  │  ┌────────────┐    ┌────────────┐         │
///  │  │   Winding  │    │   Window   │         │
///  │  │   +J_z     │    │   -J_z     │         │
///  │  └────────────┘    └────────────┘         │
///  └──────────────────────────────────────────┘
/// ```
fn transformer_problem(args: &Args) -> Problem {
    let j_val  = args.current_density.unwrap_or(1.0e6);
    let mu_r_core = 1000.0f64;
    let nu_core = NU0 / mu_r_core;

    Problem {
        nu: Box::new(move |x, y| {
            // Core region: outer frame minus the window opening
            let in_outer = x >= 0.05 && x <= 0.95 && y >= 0.05 && y <= 0.95;
            let in_window = x >= 0.25 && x <= 0.75 && y >= 0.15 && y <= 0.85;
            if in_outer && !in_window { nu_core } else { NU0 }
        }),
        j_z: Box::new(move |x, y| {
            if x >= 0.28 && x <= 0.44 && y >= 0.18 && y <= 0.82 {
                 j_val   // primary winding
            } else if x >= 0.56 && x <= 0.72 && y >= 0.18 && y <= 0.82 {
                -j_val   // secondary winding (return)
            } else {
                0.0
            }
        }),
    }
}

// ============================================================================
// Post-processing
// ============================================================================

/// Compute stored magnetic energy W = ∫ ν/2 |B|² dΩ using element centroids.
fn compute_magnetic_energy(
    mesh: &SimplexMesh<2>,
    bx: &[f64],
    by: &[f64],
    nu_fn: impl Fn(f64, f64) -> f64,
) -> f64 {
    let mut energy = 0.0f64;
    for e in mesh.elem_iter() {
        let ns = mesh.elem_nodes(e);
        let [x0, y0] = mesh.coords_of(ns[0]);
        let [x1, y1] = mesh.coords_of(ns[1]);
        let [x2, y2] = mesh.coords_of(ns[2]);
        let det = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
        let area = det.abs() * 0.5;
        let xc = (x0 + x1 + x2) / 3.0;
        let yc = (y0 + y1 + y2) / 3.0;
        let nu = nu_fn(xc, yc);
        let b2 = bx[e as usize].powi(2) + by[e as usize].powi(2);
        energy += 0.5 * nu * b2 * area; // W = ∫ ν/2 |B|² dΩ per unit depth
    }
    energy
}

// ============================================================================
// Argument parsing
// ============================================================================

struct Args {
    case:              String,
    mesh_path:         Option<String>,
    mesh_n:            usize,
    tol:               f64,
    max_iter:          usize,
    current_density:   Option<f64>,
}

fn parse_args() -> Args {
    let mut args = Args {
        case:            "square_conductor".to_string(),
        mesh_path:       None,
        mesh_n:          32,
        tol:             1e-10,
        max_iter:        10_000,
        current_density: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--case"    => { args.case     = it.next().unwrap_or_default(); }
            "--mesh"    => { args.mesh_path = Some(it.next().unwrap_or_default()); }
            "--n"       => { args.mesh_n   = it.next().unwrap_or("32".into()).parse().unwrap_or(32); }
            "--tol"     => { args.tol      = it.next().unwrap_or("1e-10".into()).parse().unwrap_or(1e-10); }
            "--max-iter"=> { args.max_iter = it.next().unwrap_or("10000".into()).parse().unwrap_or(10_000); }
            "--J"       => { args.current_density = it.next().and_then(|s| s.parse().ok()); }
            _           => {}
        }
    }
    args
}

// ============================================================================
// Summary
// ============================================================================

fn print_summary(
    az: &[f64],
    bx: &[f64],
    by: &[f64],
    mesh: &SimplexMesh<2>,
    energy: f64,
    b_max: f64,
) {
    let az_min = az.iter().cloned().fold(f64::INFINITY,     f64::min);
    let az_max = az.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    println!("\n--- Results ---");
    println!("  Nodes:      {}", mesh.n_nodes());
    println!("  Elements:   {}", mesh.n_elems());
    println!("  A_z range:  [{az_min:.6e}, {az_max:.6e}] Wb/m");
    println!("  |B|_max:    {b_max:.6e} T");
    println!("  W_mag:      {energy:.6e} J/m (per unit depth)");
    println!("  Output:     output/magnetostatics.vtk");
}
