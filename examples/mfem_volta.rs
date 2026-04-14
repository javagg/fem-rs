use fem_examples::{apply_dirichlet, dirichlet_nodes_fn, p1_assemble_poisson, pcg_solve};
use fem_mesh::SimplexMesh;

const DEFAULT_BC_SCALE: f64 = 1.0;

fn solve_volta(n: usize) -> (SimplexMesh<2>, Vec<(usize, f64)>, Vec<f64>, usize, f64) {
	solve_volta_with_scale(n, DEFAULT_BC_SCALE)
}

fn solve_volta_with_scale(n: usize, bc_scale: f64) -> (SimplexMesh<2>, Vec<(usize, f64)>, Vec<f64>, usize, f64) {
	let mesh = SimplexMesh::<2>::unit_square_tri(n);
	let (mut k, mut rhs) = p1_assemble_poisson(&mesh, |_, _| 1.0, |_, _| 0.0);
	let bcs = dirichlet_nodes_fn(&mesh, &[1, 3], |_, y| bc_scale * y);
	apply_dirichlet(&mut k, &mut rhs, &bcs);
	let (u, iters, res) = pcg_solve(&k, &rhs, 1e-10, 5000);
	(mesh, bcs, u, iters, res)
}

fn main() {
	let (_mesh, _bcs, u, iters, res) = solve_volta(16);
	println!("mfem_volta done: dofs={}, iters={}, res={:.3e}, u_min={:.3e}, u_max={:.3e}", u.len(), iters, res, u.iter().fold(f64::INFINITY, |a, &v| a.min(v)), u.iter().fold(f64::NEG_INFINITY, |a, &v| a.max(v)));
}

#[cfg(test)]
mod tests {
	use super::*;
	use fem_mesh::topology::MeshTopology;

	fn l2_norm(values: &[f64]) -> f64 {
		values.iter().map(|v| v * v).sum::<f64>().sqrt()
	}

	#[test]
	fn volta_dirichlet_values_are_enforced() {
		let (mesh, bcs, u, _iters, _res) = solve_volta(12);
		for (idx, val) in bcs {
			let node = mesh.node_coords(idx as u32);
			let expected = node[1];
			assert!((val - expected).abs() < 1e-12);
			assert!((u[idx] - expected).abs() < 1e-8, "Dirichlet mismatch at node {}", idx);
		}
	}

	#[test]
	fn volta_solution_is_finite_and_converged() {
		let (_mesh, _bcs, u, _iters, res) = solve_volta(12);
		assert!(res < 1e-8, "PCG residual too large: {}", res);
		assert!(u.iter().all(|v| v.is_finite()));
	}

	#[test]
	fn volta_reproduces_exact_linear_potential_throughout_domain() {
		let (mesh, _bcs, u, _iters, res) = solve_volta(12);
		assert!(res < 1e-8, "PCG residual too large: {}", res);
		for idx in 0..u.len() {
			let node = mesh.node_coords(idx as u32);
			let expected = node[1];
			assert!(
				(u[idx] - expected).abs() < 1e-8,
				"expected exact linear potential at node {}: got {} expected {}",
				idx,
				u[idx],
				expected
			);
		}
	}

	#[test]
	fn volta_solution_scales_linearly_with_boundary_voltage() {
		let (_mesh_half, _bcs_half, u_half, _iters_half, res_half) = solve_volta_with_scale(12, 0.5);
		let (_mesh_full, _bcs_full, u_full, _iters_full, res_full) = solve_volta_with_scale(12, 1.0);
		let norm_half = l2_norm(&u_half);
		let norm_full = l2_norm(&u_full);
		let ratio = norm_full / norm_half.max(1e-30);

		assert!(res_half < 1e-8 && res_full < 1e-8);
		assert!(
			(ratio - 2.0).abs() < 1e-10,
			"expected linear scaling with boundary voltage, got ratio {}",
			ratio
		);
	}

	#[test]
	fn volta_sign_reversal_flips_potential_everywhere() {
		let (mesh_pos, _bcs_pos, u_pos, _iters_pos, res_pos) = solve_volta_with_scale(12, 1.0);
		let (mesh_neg, _bcs_neg, u_neg, _iters_neg, res_neg) = solve_volta_with_scale(12, -1.0);

		assert!(res_pos < 1e-8 && res_neg < 1e-8);
		assert_eq!(u_pos.len(), u_neg.len());

		for idx in 0..u_pos.len() {
			let xy_pos = mesh_pos.node_coords(idx as u32);
			let xy_neg = mesh_neg.node_coords(idx as u32);
			assert!((xy_pos[0] - xy_neg[0]).abs() < 1e-12 && (xy_pos[1] - xy_neg[1]).abs() < 1e-12);
			assert!(
				(u_pos[idx] + u_neg[idx]).abs() < 1e-10,
				"expected electrostatic potential to flip sign at node {}: u_pos={} u_neg={}",
				idx,
				u_pos[idx],
				u_neg[idx]
			);
		}
	}
}
