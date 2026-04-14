//! mfem_ex26_geom_mg - baseline geometric multigrid V-cycle demo.
//!
//! This is a compact MFEM ex26-style baseline that demonstrates the new
//! `GeomMGHierarchy` + `GeomMGPrecond` solve path on a nested 1D Poisson
//! hierarchy.

use fem_linalg::{CooMatrix, CsrMatrix};
use fem_solver::{solve_vcycle_geom_mg, GeomMGHierarchy, GeomMGPrecond, SolverConfig};

struct SolveResult {
    fine_n: usize,
    max_iter: usize,
    rtol: f64,
    converged: bool,
    iterations: usize,
    final_residual: f64,
    exact_l2_error: f64,
    symmetry_error: f64,
    solution_min: f64,
    solution_max: f64,
    center_value: f64,
    checksum: f64,
}

fn main() {
    let args = parse_args();

    println!("=== mfem_ex26_geom_mg: geometric multigrid baseline ===");
    let result = solve_case(args.fine_n, args.max_iter, args.rtol);

    println!("  fine_n={}, max_iter={}, rtol={:.1e}", result.fine_n, result.max_iter, result.rtol);
    println!(
        "  Solve: converged={}, iters={}, residual={:.3e}",
        result.converged,
        result.iterations,
        result.final_residual
    );
    println!("  exact error = {:.3e}, symmetry error = {:.3e}", result.exact_l2_error, result.symmetry_error);
    println!("  range = [{:.4e}, {:.4e}], center = {:.4e}", result.solution_min, result.solution_max, result.center_value);
    println!("  checksum = {:.8e}", result.checksum);

    assert!(result.converged, "GeomMG did not converge");
    assert!(result.final_residual < 1e-5, "residual too large");

    println!("  PASS");
}

fn solve_case(fine_n: usize, max_iter: usize, rtol: f64) -> SolveResult {
    let fine_n = if fine_n % 2 == 0 { fine_n + 1 } else { fine_n };

    // Build a 3-level nested hierarchy: N -> (N-1)/2 -> ...
    let n0 = fine_n;
    let n1 = (n0 - 1) / 2;
    let n2 = (n1 - 1) / 2;
    assert!(n2 >= 3, "fine_n too small for 3-level hierarchy");

    let a0 = lap1d(n0);
    let a1 = lap1d(n1);
    let a2 = lap1d(n2);
    let p0 = prolong_1d(n0, n1);
    let p1 = prolong_1d(n1, n2);
    let h = GeomMGHierarchy::new(vec![a0.clone(), a1, a2], vec![p0, p1]);

    // Solve A x = 1 with zero initial guess.
    let b = vec![1.0; n0];
    let mut x = vec![0.0; n0];

    let mg = GeomMGPrecond::default();
    let cfg = SolverConfig {
        rtol,
        atol: 0.0,
        max_iter,
        verbose: false,
        ..Default::default()
    };

    let res = solve_vcycle_geom_mg(&a0, &b, &mut x, &h, &mg, &cfg)
        .expect("solve_vcycle_geom_mg failed");

    let x_exact = exact_discrete_solution(n0);
    let exact_l2_error = l2_error(&x, &x_exact);
    let symmetry_error = x
        .iter()
        .zip(x.iter().rev())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f64, f64::max);
    let solution_min = x.iter().copied().fold(f64::INFINITY, f64::min);
    let solution_max = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let center_value = x[n0 / 2];
    let checksum = x
        .iter()
        .enumerate()
        .map(|(i, value)| (i as f64 + 1.0) * value)
        .sum::<f64>();

    SolveResult {
        fine_n: n0,
        max_iter,
        rtol,
        converged: res.converged,
        iterations: res.iterations,
        final_residual: res.final_residual,
        exact_l2_error,
        symmetry_error,
        solution_min,
        solution_max,
        center_value,
        checksum,
    }
}

fn exact_discrete_solution(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let left = i as f64 + 1.0;
            let right = (n - i) as f64;
            0.5 * left * right
        })
        .collect()
}

fn l2_error(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().max(1) as f64;
    let sum = a
        .iter()
        .zip(b.iter())
        .map(|(lhs, rhs)| (lhs - rhs).powi(2))
        .sum::<f64>();
    (sum / n).sqrt()
}

fn lap1d(n: usize) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::<f64>::new(n, n);
    for i in 0..n {
        coo.add(i, i, 2.0);
        if i > 0 {
            coo.add(i, i - 1, -1.0);
        }
        if i + 1 < n {
            coo.add(i, i + 1, -1.0);
        }
    }
    coo.into_csr()
}

fn prolong_1d(nf: usize, nc: usize) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::<f64>::new(nf, nc);
    for i in 0..nf {
        if i % 2 == 1 {
            let j = (i - 1) / 2;
            if j < nc {
                coo.add(i, j, 1.0);
            }
        } else {
            let jr = i / 2;
            if jr > 0 && jr < nc {
                coo.add(i, jr - 1, 0.5);
                coo.add(i, jr, 0.5);
            } else if jr == 0 {
                coo.add(i, 0, 1.0);
            } else {
                coo.add(i, nc - 1, 1.0);
            }
        }
    }
    coo.into_csr()
}

struct Args {
    fine_n: usize,
    max_iter: usize,
    rtol: f64,
}

fn parse_args() -> Args {
    let mut a = Args {
        fine_n: 31,
        max_iter: 80,
        rtol: 1e-6,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--n" => a.fine_n = it.next().unwrap_or("31".into()).parse().unwrap_or(31),
            "--max-iter" => a.max_iter = it.next().unwrap_or("80".into()).parse().unwrap_or(80),
            "--rtol" => a.rtol = it.next().unwrap_or("1e-6".into()).parse().unwrap_or(1e-6),
            _ => {}
        }
    }
    // keep odd sizes so nested levels are exact for this baseline prolongation
    if a.fine_n % 2 == 0 {
        a.fine_n += 1;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ex26_geom_mg_default_case_matches_discrete_solution() {
        let result = solve_case(31, 80, 1e-6);
        assert!(result.converged);
        assert!(result.final_residual < 1.0e-5, "residual too large: {}", result.final_residual);
        assert!(result.exact_l2_error < 1.0e-4, "exact discrete error too large: {}", result.exact_l2_error);
        assert!(result.symmetry_error < 1.0e-10, "symmetry drift too large: {}", result.symmetry_error);
        assert!(result.solution_min > 0.0);
    }

    #[test]
    fn ex26_geom_mg_tighter_tolerance_improves_error_and_residual() {
        let loose = solve_case(31, 80, 1e-6);
        let tight = solve_case(31, 80, 1e-8);
        assert!(loose.converged && tight.converged);
        assert!(tight.final_residual < loose.final_residual,
            "tighter tolerance should reduce residual: loose={} tight={}", loose.final_residual, tight.final_residual);
        assert!(tight.exact_l2_error < loose.exact_l2_error,
            "tighter tolerance should reduce exact error: loose={} tight={}", loose.exact_l2_error, tight.exact_l2_error);
    }

    #[test]
    fn ex26_geom_mg_larger_grid_remains_symmetric_and_accurate() {
        let result = solve_case(63, 80, 1e-6);
        assert!(result.converged);
        assert!(result.iterations <= 60, "too many MG iterations: {}", result.iterations);
        assert!(result.exact_l2_error < 3.0e-4, "large-grid exact error too large: {}", result.exact_l2_error);
        assert!(result.symmetry_error < 1.0e-10, "large-grid symmetry drift too large: {}", result.symmetry_error);
        assert!(result.center_value > 5.0e2, "center value too small: {}", result.center_value);
    }

    #[test]
    fn ex26_geom_mg_even_requested_size_is_rounded_to_nested_odd_grid() {
        let even = solve_case(30, 80, 1e-6);
        let odd = solve_case(31, 80, 1e-6);
        assert_eq!(even.fine_n, 31);
        assert!((even.exact_l2_error - odd.exact_l2_error).abs() < 1.0e-12);
        assert!((even.checksum - odd.checksum).abs() < 1.0e-12);
    }
}

