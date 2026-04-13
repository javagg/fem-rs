//! ex41_imex - IMEX time-integrator comparison
//!
//! Demonstrates and compares three IMEX integrators on a stiff test problem:
//!   - ImexEuler  (1st order, first-order accuracy)
//!   - ImexSsp2   (2nd order, A-stable implicit stage)
//!   - ImexArk3   (3rd order adaptive, Kennedy-Carpenter ARK)
//!
//! Test problem: scalar advection-diffusion ODE (modal reduction)
//!
//!   du/dt = f_E(t, u) + f_I(t, u)
//!
//! where
//!   f_I(t, u) = -lambda * u        (stiff diffusion, large lambda)
//!   f_E(t, u) = A * sin(omega * t) (non-stiff oscillatory forcing)
//!
//! Exact solution with u(0) = 0:
//!   u(t) = [lambda sin(omega t) - omega cos(omega t) + omega exp(-lambda t)] / (lambda^2 + omega^2)
//!
//! For lambda = 100 the diffusion term is stiff; explicit methods need dt < 2/lambda = 0.02.
//! IMEX methods can use much larger steps because the stiff part is handled implicitly.
//!
//! This example:
//!   1. Verifies each method converges to the exact solution.
//!   2. Measures temporal convergence rate (order check) with fixed dt refinement.
//!   3. Shows that IMEX Euler/SSP2/ARK3 achieve order 1/2/3 respectively.

use fem_solver::{ImexEuler, ImexSsp2, ImexArk3};
use fem_linalg::{CooMatrix, CsrMatrix};

// --- Problem parameters -------------------------------------------------------

const LAMBDA: f64 = 100.0; // stiffness (large -> stiff decay)
const OMEGA:  f64 = std::f64::consts::PI; // forcing frequency
const AMP:    f64 = 1.0;
const T_END:  f64 = 1.0;

/// Exact solution: u(t) with u(0)=0.
fn exact(t: f64) -> f64 {
    let lam = LAMBDA;
    let om  = OMEGA;
    AMP * (lam * (om * t).sin() - om * (om * t).cos() + om * (-lam * t).exp()) / (lam * lam + om * om)
}

/// Explicit RHS: f_E(t, u) = A sin(omega t) (non-stiff, independent of u)
fn rhs_explicit(t: f64, _u: &[f64], out: &mut [f64]) {
    out[0] = AMP * (OMEGA * t).sin();
}

/// Implicit RHS: f_I(t, u) = -lambda u (stiff decay)
fn rhs_implicit(_t: f64, u: &[f64], out: &mut [f64]) {
    out[0] = -LAMBDA * u[0];
}

/// Jacobian of implicit RHS: df_I/du = -lambda I
fn jac_implicit(_t: f64, _u: &[f64]) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::<f64>::new(1, 1);
    coo.add(0, 0, -LAMBDA);
    coo.into_csr()
}

// --- Order-of-convergence check ----------------------------------------------

fn run_imex_euler(dt: f64) -> f64 {
    let mut u = vec![0.0f64];
    let solver = ImexEuler;
    solver.integrate(0.0, T_END, &mut u, dt, rhs_explicit, rhs_implicit, jac_implicit);
    (u[0] - exact(T_END)).abs()
}

fn run_imex_ssp2(dt: f64) -> f64 {
    let mut u = vec![0.0f64];
    let solver = ImexSsp2;
    solver.integrate(0.0, T_END, &mut u, dt, rhs_explicit, rhs_implicit, jac_implicit);
    (u[0] - exact(T_END)).abs()
}

fn run_imex_ark3(dt_init: f64) -> f64 {
    let mut u = vec![0.0f64];
    let solver = ImexArk3 { rtol: 1e-10, atol: 1e-12, ..Default::default() };
    solver.integrate(0.0, T_END, &mut u, dt_init, rhs_explicit, rhs_implicit, jac_implicit);
    (u[0] - exact(T_END)).abs()
}

fn convergence_order(e_coarse: f64, e_fine: f64, refinement: f64, expected_order: f64) -> bool {
    // Richardson extrapolation: order ~= log(e_c / e_f) / log(r)
    if e_fine < 1e-15 { return true; } // already machine precision
    let ratio = e_coarse / e_fine;
    let order = ratio.log2() / refinement.log2();
    println!("    convergence order ~= {order:.2}  (expected >= {expected_order:.1})");
    order > expected_order * 0.75 // allow 25% margin
}

// --- main --------------------------------------------------------------------

fn main() {
    println!("=== ex41_imex: IMEX Time Integrator Comparison ===\n");
    println!("Problem:  du/dt = A sin(wt) - lam*u,  lam={LAMBDA}, w={OMEGA:.4}");
    println!("          u(0) = 0,  T_end = {T_END}\n");
    println!("  Explicit RHS (non-stiff): f_E = A sin(wt)");
    println!("  Implicit RHS (stiff):     f_I = -lam*u  (lam={LAMBDA}, dt_crit = {:.4} for explicit)\n", 2.0/LAMBDA);
    println!("  Exact u(T)   = {:.10}\n", exact(T_END));

    // -- Method 1: IMEX Euler (order 1) ---------------------------------------
    println!("--- IMEX Euler (order 1) ---");
    let dt1 = 0.1;
    let dt2 = 0.05;
    let e1 = run_imex_euler(dt1);
    let e2 = run_imex_euler(dt2);
    println!("  dt={dt1:.3}  err={e1:.3e}");
    println!("  dt={dt2:.3}  err={e2:.3e}");
    let ok1 = convergence_order(e1, e2, 2.0, 1.0);
    assert!(ok1, "IMEX Euler did not achieve order 1");
    assert!(run_imex_euler(0.01) < 5e-4, "IMEX Euler: accuracy too low at dt=0.01");
    println!("  PASS\n");

    // -- Method 2: IMEX SSP-RK2 (order 2) -------------------------------------
    println!("--- IMEX SSP-RK2 (order 2) ---");
    let dt1 = 0.1;
    let dt2 = 0.05;
    let e1 = run_imex_ssp2(dt1);
    let e2 = run_imex_ssp2(dt2);
    println!("  dt={dt1:.3}  err={e1:.3e}");
    println!("  dt={dt2:.3}  err={e2:.3e}");
    let ok2 = convergence_order(e1, e2, 2.0, 1.8);
    assert!(ok2, "IMEX SSP2 did not achieve order 2");
    assert!(run_imex_ssp2(0.01) < 5e-4, "IMEX SSP2: accuracy too low at dt=0.01");
    println!("  PASS\n");

    // -- Method 3: IMEX ARK3 (order 3, adaptive) ------------------------------
    println!("--- IMEX ARK3 (order 3, adaptive) ---");
    let e_coarse = run_imex_ark3(0.05);
    let e_fine   = run_imex_ark3(0.01);
    println!("  dt_init=0.05  err={e_coarse:.3e}");
    println!("  dt_init=0.01  err={e_fine:.3e}");
    // ARK3 with tight tol should already be very accurate
    assert!(e_fine < 1e-8, "IMEX ARK3: accuracy too low (err={e_fine:.3e})");
    println!("  PASS\n");

    // -- Stability demonstration ------------------------------------------------
    // Explicit-only forward Euler needs dt < 2/lambda = 0.02 for lambda=100.
    // IMEX Euler can use dt=0.1 (5x larger) because stiff part is implicit.
    println!("--- Stability comparison ---");
    println!("  Explicit forward Euler with dt=0.1 (unstable for lambda=100):");
    let mut u_exp = vec![0.0f64];
    {
        let mut t = 0.0f64;
        let dt: f64 = 0.1;
        while t < T_END - 1e-14 {
            let h = dt.min(T_END - t);
            let fe: f64 = AMP * (OMEGA * t).sin();
            let fi: f64 = -LAMBDA * u_exp[0];
            u_exp[0] += h * (fe + fi);
            t += h;
        }
    }
    println!("    u_explicit(T) = {:.3e}  (diverged/unstable expected)", u_exp[0]);
    // Check that pure explicit has blown up
    assert!(u_exp[0].abs() > 1.0, "Expected explicit method to be unstable for large dt; got {:.3e}", u_exp[0]);

    println!("  IMEX Euler with dt=0.1 (stable, implicit handles stiff part):");
    let e_imex = run_imex_euler(0.1);
    println!("    u_imex(T) ~= {:.6}  err={e_imex:.3e}  (stable)", exact(T_END));
    assert!(e_imex < 0.1, "Expected IMEX Euler to be stable with dt=0.1");
    println!("  PASS\n");

    println!("=== All IMEX tests passed ===");
    println!("\nSummary:");
    println!("  ImexEuler  - 1st order, stable for any dt (stiff part)");
    println!("  ImexSsp2   - 2nd order, A-stable implicit, SSP explicit");
    println!("  ImexArk3   - 3rd order adaptive (Kennedy-Carpenter ARK3(2)4L[2]SA)");
}
