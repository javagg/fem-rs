use fem_examples::{apply_dirichlet, dirichlet_nodes, p1_assemble_poisson, pcg_solve};
use fem_mesh::SimplexMesh;

const DEFAULT_SOURCE_SCALE: f64 = 1.0e3;

fn solve_tesla(n: usize) -> (SimplexMesh<2>, Vec<f64>, usize, f64) {
	solve_tesla_with_scale(n, DEFAULT_SOURCE_SCALE)
}

fn solve_tesla_with_scale(n: usize, source_scale: f64) -> (SimplexMesh<2>, Vec<f64>, usize, f64) {
	let mesh = SimplexMesh::<2>::unit_square_tri(n);

	let src = |x: f64, y: f64| {
		if (0.3..=0.7).contains(&x) && (0.3..=0.7).contains(&y) {
			source_scale
		} else {
			0.0
		}
	};
	let (mut k, mut rhs) = p1_assemble_poisson(&mesh, |_, _| 1.0, src);
	let bcs = dirichlet_nodes(&mesh, &[1, 2, 3, 4]);
	apply_dirichlet(&mut k, &mut rhs, &bcs);

	let (u, iters, res) = pcg_solve(&k, &rhs, 1e-10, 5000);
	(mesh, u, iters, res)
}

fn main() {
	let (_mesh, u, iters, res) = solve_tesla(16);
	let l2 = u.iter().map(|v| v * v).sum::<f64>().sqrt();
	println!("mfem_tesla done: dofs={}, iters={}, res={:.3e}, ||Az||2={:.3e}", u.len(), iters, res, l2);
}

#[cfg(test)]
mod tests {
	use super::*;
	use fem_mesh::topology::MeshTopology;

	fn l2_norm(values: &[f64]) -> f64 {
		values.iter().map(|v| v * v).sum::<f64>().sqrt()
	}

	#[test]
	fn tesla_solution_is_finite_and_solver_converges() {
		let (_mesh, u, _iters, res) = solve_tesla(12);
		assert!(res < 1e-8, "PCG residual too large: {}", res);
		assert!(u.iter().all(|v| v.is_finite()));
	}

	#[test]
	fn tesla_nonzero_source_generates_nontrivial_field() {
		let (_mesh, u, _iters, _res) = solve_tesla(12);
		let max_abs = u.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
		assert!(max_abs > 1e-6, "expected nontrivial response from source, got max |u|={}", max_abs);
	}

	#[test]
	fn tesla_zero_source_gives_trivial_field() {
		let (_mesh, u, _iters, res) = solve_tesla_with_scale(12, 0.0);
		let max_abs = u.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
		assert!(res < 1e-8, "PCG residual too large: {}", res);
		assert!(max_abs < 1e-12, "expected zero source to give zero field, got max |u|={}", max_abs);
	}

	#[test]
	fn tesla_response_scales_linearly_with_source_strength() {
		let (_mesh_half, u_half, _iters_half, res_half) = solve_tesla_with_scale(12, 5.0e2);
		let (_mesh_full, u_full, _iters_full, res_full) = solve_tesla_with_scale(12, 1.0e3);
		let norm_half = l2_norm(&u_half);
		let norm_full = l2_norm(&u_full);
		let ratio = norm_full / norm_half.max(1e-30);

		assert!(res_half < 1e-8 && res_full < 1e-8);
		assert!(norm_half > 0.0 && norm_full > 0.0);
		assert!(
			(ratio - 2.0).abs() < 1e-10,
			"expected linear source scaling with ratio 2, got {}",
			ratio
		);
	}

	#[test]
	fn tesla_square_source_preserves_xy_symmetry() {
		let (mesh, u, _iters, res) = solve_tesla(12);
		assert!(res < 1e-8, "PCG residual too large: {}", res);

		for idx in 0..u.len() {
			let node = mesh.node_coords(idx as u32);
			let mut match_idx = None;
			for j in 0..u.len() {
				let other = mesh.node_coords(j as u32);
				if (other[0] - node[1]).abs() < 1e-12 && (other[1] - node[0]).abs() < 1e-12 {
					match_idx = Some(j);
					break;
				}
			}

			let j = match_idx.expect("expected reflected node under x<->y symmetry");
			assert!(
				(u[idx] - u[j]).abs() < 1e-10,
				"expected x/y symmetry at node {} mirrored by {}: u_i={} u_j={}",
				idx,
				j,
				u[idx],
				u[j]
			);
		}
	}
}
