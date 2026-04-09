//! Lagrange elements on the reference quadrilateral `[-1,1]²`.

use crate::quadrature::quad_rule;
use crate::reference::{QuadratureRule, ReferenceElement};

// ─── Q1 ───────────────────────────────────────────────────────────────────────

/// Bilinear Lagrange element on the reference quad `[-1,1]²` — 4 DOFs.
///
/// Node ordering (counter-clockwise):
/// - 0: (−1,−1)
/// - 1: (+1,−1)
/// - 2: (+1,+1)
/// - 3: (−1,+1)
///
/// Basis: φᵢ = (1 + ξᵢ ξ)(1 + ηᵢ η) / 4
pub struct QuadQ1;

/// Node coordinates (ξ, η) of the 4 Q1 nodes.
const Q1_NODES: [(f64, f64); 4] = [
    (-1.0, -1.0),
    ( 1.0, -1.0),
    ( 1.0,  1.0),
    (-1.0,  1.0),
];

impl ReferenceElement for QuadQ1 {
    fn dim(&self)    -> u8    { 2 }
    fn order(&self)  -> u8    { 1 }
    fn n_dofs(&self) -> usize  { 4 }

    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let (x, y) = (xi[0], xi[1]);
        for (i, &(xi_i, eta_i)) in Q1_NODES.iter().enumerate() {
            values[i] = 0.25 * (1.0 + xi_i * x) * (1.0 + eta_i * y);
        }
    }

    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        let (x, y) = (xi[0], xi[1]);
        for (i, &(xi_i, eta_i)) in Q1_NODES.iter().enumerate() {
            grads[i * 2]     = 0.25 * xi_i  * (1.0 + eta_i * y);
            grads[i * 2 + 1] = 0.25 * eta_i * (1.0 + xi_i  * x);
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule { quad_rule(order) }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        Q1_NODES.iter().map(|&(x, y)| vec![x, y]).collect()
    }
}

// ─── Q2 ───────────────────────────────────────────────────────────────────────

/// Biquadratic serendipity — 9-node Lagrange element on the reference quad `[-1,1]²`.
///
/// Node ordering:
/// - 0: (−1,−1)  corner
/// - 1: (+1,−1)  corner
/// - 2: (+1,+1)  corner
/// - 3: (−1,+1)  corner
/// - 4: ( 0,−1)  edge midpoint
/// - 5: (+1, 0)  edge midpoint
/// - 6: ( 0,+1)  edge midpoint
/// - 7: (−1, 0)  edge midpoint
/// - 8: ( 0, 0)  interior
///
/// Basis: standard tensor-product Lagrange polynomials through the nine nodes.
/// For node at (ξᵢ, ηᵢ), φᵢ = Lᵢ(ξ) · Lᵢ(η) where Lᵢ is the 1-D Lagrange polynomial.
pub struct QuadQ2;

/// Node coordinates (ξ, η) of the 9 Q2 nodes.
const Q2_NODES: [(f64, f64); 9] = [
    (-1.0, -1.0), // 0
    ( 1.0, -1.0), // 1
    ( 1.0,  1.0), // 2
    (-1.0,  1.0), // 3
    ( 0.0, -1.0), // 4
    ( 1.0,  0.0), // 5
    ( 0.0,  1.0), // 6
    (-1.0,  0.0), // 7
    ( 0.0,  0.0), // 8
];

/// Evaluate the three 1-D quadratic Lagrange polynomials on [-1,1]
/// through nodes ξ=-1, ξ=0, ξ=+1:
/// L₀(ξ) = ξ(ξ−1)/2,  L₁(ξ) = 1−ξ²,  L₂(ξ) = ξ(ξ+1)/2
///
/// Returns [L0, L1, L2] and their derivatives [L0', L1', L2'].
#[inline]
fn q2_1d(x: f64) -> ([f64; 3], [f64; 3]) {
    let vals = [
        0.5 * x * (x - 1.0), // L₋₁ (node at -1)
        1.0 - x * x,          // L₀  (node at  0)
        0.5 * x * (x + 1.0), // L₊₁ (node at +1)
    ];
    let ders = [
        0.5 * (2.0 * x - 1.0), // L₋₁'
        -2.0 * x,               // L₀'
        0.5 * (2.0 * x + 1.0), // L₊₁'
    ];
    (vals, ders)
}

/// Map (ξᵢ, ηᵢ) → indices into the 1-D basis:
/// ξ coordinate: -1 → index 0, 0 → index 1, +1 → index 2
/// η coordinate: same.
#[inline]
fn coord_to_idx(c: f64) -> usize {
    if c < -0.5 { 0 } else if c > 0.5 { 2 } else { 1 }
}

impl ReferenceElement for QuadQ2 {
    fn dim(&self)    -> u8    { 2 }
    fn order(&self)  -> u8    { 2 }
    fn n_dofs(&self) -> usize { 9 }

    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let (x, y) = (xi[0], xi[1]);
        let (lx, _) = q2_1d(x);
        let (ly, _) = q2_1d(y);
        for (i, &(xi_i, eta_i)) in Q2_NODES.iter().enumerate() {
            let ix = coord_to_idx(xi_i);
            let iy = coord_to_idx(eta_i);
            values[i] = lx[ix] * ly[iy];
        }
    }

    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        let (x, y) = (xi[0], xi[1]);
        let (lx, dlx) = q2_1d(x);
        let (ly, dly) = q2_1d(y);
        for (i, &(xi_i, eta_i)) in Q2_NODES.iter().enumerate() {
            let ix = coord_to_idx(xi_i);
            let iy = coord_to_idx(eta_i);
            grads[i * 2]     = dlx[ix] * ly[iy];  // ∂φᵢ/∂ξ
            grads[i * 2 + 1] = lx[ix]  * dly[iy]; // ∂φᵢ/∂η
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule { quad_rule(order) }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        Q2_NODES.iter().map(|&(x, y)| vec![x, y]).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_pou(elem: &dyn ReferenceElement) {
        let rule = elem.quadrature(4);
        let mut phi = vec![0.0_f64; elem.n_dofs()];
        for pt in &rule.points {
            elem.eval_basis(pt, &mut phi);
            let s: f64 = phi.iter().sum();
            assert!((s - 1.0).abs() < 1e-13, "POU failed sum={s}");
        }
    }

    fn check_grad_zero(elem: &dyn ReferenceElement) {
        let dim = elem.dim() as usize;
        let rule = elem.quadrature(4);
        let mut g = vec![0.0_f64; elem.n_dofs() * dim];
        for pt in &rule.points {
            elem.eval_grad_basis(pt, &mut g);
            for d in 0..dim {
                let s: f64 = (0..elem.n_dofs()).map(|i| g[i * dim + d]).sum();
                assert!(s.abs() < 1e-12, "grad sum d={d} = {s}");
            }
        }
    }

    #[test] fn quad_q1_pou()       { check_pou(&QuadQ1); }
    #[test] fn quad_q1_grad_zero() { check_grad_zero(&QuadQ1); }

    #[test]
    fn quad_q1_node_dofs() {
        let mut phi = vec![0.0; 4];
        for (i, &(x, y)) in Q1_NODES.iter().enumerate() {
            QuadQ1.eval_basis(&[x, y], &mut phi);
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((phi[j] - expected).abs() < 1e-14,
                    "node {i}, basis {j}: expected {expected}, got {}", phi[j]);
            }
        }
    }

    #[test] fn quad_q2_pou()       { check_pou(&QuadQ2); }
    #[test] fn quad_q2_grad_zero() { check_grad_zero(&QuadQ2); }

    #[test]
    fn quad_q2_node_dofs() {
        let mut phi = vec![0.0; 9];
        for (i, &(x, y)) in Q2_NODES.iter().enumerate() {
            QuadQ2.eval_basis(&[x, y], &mut phi);
            for j in 0..9 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((phi[j] - expected).abs() < 1e-13,
                    "node {i}, basis {j}: expected {expected}, got {}", phi[j]);
            }
        }
    }
}
