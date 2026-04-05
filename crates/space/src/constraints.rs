//! Essential (Dirichlet) boundary condition enforcement and hanging-node
//! constraint application.
//!
//! After assembly, call [`apply_dirichlet`] to modify the stiffness matrix and
//! right-hand side so that constrained DOFs are set to their prescribed values.
//!
//! For non-conforming meshes, call [`apply_hanging_constraints`] to enforce
//! `u_hang = 0.5*(u_a + u_b)` and then [`recover_hanging_values`] after solving.

use fem_core::types::DofId;
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::amr::HangingNodeConstraint;

use crate::dof_manager::{DofManager, EdgeKey, FaceKey};
use crate::hcurl::HCurlSpace;
use crate::hdiv::HDivSpace;

/// Apply Dirichlet boundary conditions to the assembled system `(K, f)`.
///
/// For each DOF in `constrained_dofs`:
/// 1. Zero the row.
/// 2. Set the diagonal to 1.
/// 3. Set `rhs[dof] = value[i]`.
///
/// This is the **non-symmetric** row-zeroing approach — fast and sufficient
/// for most FEM solves.
///
/// # Panics
/// Panics if `constrained_dofs.len() != values.len()`.
pub fn apply_dirichlet(
    mat:              &mut CsrMatrix<f64>,
    rhs:              &mut [f64],
    constrained_dofs: &[DofId],
    values:           &[f64],
) {
    assert_eq!(constrained_dofs.len(), values.len(),
        "constrained_dofs and values must have the same length");
    for (&dof, &val) in constrained_dofs.iter().zip(values.iter()) {
        mat.apply_dirichlet_row_zeroing(dof as usize, val, rhs);
    }
}

/// Identify which DOFs lie on boundary faces with the given tag(s).
///
/// Returns sorted global DOF indices for all boundary nodes (and, for P2,
/// edge-midpoint DOFs) that lie on boundary faces whose tag is in `tags`.
///
/// # Arguments
/// * `mesh`  — mesh providing boundary face data
/// * `dm`    — DOF manager for the space
/// * `tags`  — boundary tags to select (e.g. `&[1, 2, 3, 4]` for all sides)
pub fn boundary_dofs(
    mesh: &dyn fem_mesh::topology::MeshTopology,
    dm:   &DofManager,
    tags: &[i32],
) -> Vec<DofId> {
    use std::collections::HashSet;
    let mut node_set: HashSet<DofId> = HashSet::new();

    for f in 0..mesh.n_boundary_faces() as u32 {
        if tags.contains(&mesh.face_tag(f)) {
            for &node in mesh.face_nodes(f) {
                node_set.insert(node as DofId);
            }
        }
    }

    let mut dof_set: HashSet<DofId> = node_set.clone();

    // For P2, also include edge-midpoint DOFs whose both vertex endpoints are boundary nodes.
    if dm.order == 2 {
        let n_elems = dm.dofs_flat.len() / dm.dofs_per_elem;
        for e in 0..n_elems as u32 {
            let dofs  = dm.element_dofs(e);
            // Vertex DOFs are at positions 0,1,2; edge DOFs at 3,4,5.
            // Edge mapping: edge(v0→v1)=dofs[3], edge(v1→v2)=dofs[4], edge(v0→v2)=dofs[5]
            let edge_pairs = [
                (dofs[0], dofs[1], dofs[3]),
                (dofs[1], dofs[2], dofs[4]),
                (dofs[0], dofs[2], dofs[5]),
            ];
            for (a, b, edge_dof) in edge_pairs {
                if node_set.contains(&a) && node_set.contains(&b) {
                    dof_set.insert(edge_dof);
                }
            }
        }
    }

    let mut out: Vec<DofId> = dof_set.into_iter().collect();
    out.sort_unstable();
    out
}

/// Identify H(curl) DOFs on boundary faces with the given tag(s).
///
/// Collects all edges that lie on tagged boundary faces, then looks up
/// the corresponding global DOF in the space.
pub fn boundary_dofs_hcurl<M: fem_mesh::topology::MeshTopology>(
    mesh: &M,
    space: &HCurlSpace<M>,
    tags: &[i32],
) -> Vec<DofId> {
    use std::collections::HashSet;

    // Collect boundary edges from tagged boundary faces.
    let dim = mesh.dim() as usize;
    let mut boundary_edges: HashSet<EdgeKey> = HashSet::new();

    for f in 0..mesh.n_boundary_faces() as u32 {
        if tags.contains(&mesh.face_tag(f)) {
            let nodes = mesh.face_nodes(f);
            if dim == 2 {
                // Boundary face in 2-D is an edge.
                if nodes.len() >= 2 {
                    boundary_edges.insert(EdgeKey::new(nodes[0], nodes[1]));
                }
            } else {
                // Boundary face in 3-D is a triangle: collect its 3 edges.
                if nodes.len() >= 3 {
                    boundary_edges.insert(EdgeKey::new(nodes[0], nodes[1]));
                    boundary_edges.insert(EdgeKey::new(nodes[1], nodes[2]));
                    boundary_edges.insert(EdgeKey::new(nodes[0], nodes[2]));
                }
            }
        }
    }

    let mut out: Vec<DofId> = boundary_edges
        .iter()
        .filter_map(|ek| space.edge_dof(*ek))
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// Identify H(div) DOFs on boundary faces with the given tag(s).
///
/// In 2-D, boundary faces are edges; in 3-D, they are triangular faces.
pub fn boundary_dofs_hdiv<M: fem_mesh::topology::MeshTopology>(
    mesh: &M,
    space: &HDivSpace<M>,
    tags: &[i32],
) -> Vec<DofId> {
    let dim = mesh.dim() as usize;
    let mut out: Vec<DofId> = Vec::new();

    for f in 0..mesh.n_boundary_faces() as u32 {
        if tags.contains(&mesh.face_tag(f)) {
            let nodes = mesh.face_nodes(f);
            let dof = if dim == 2 {
                if nodes.len() >= 2 {
                    space.edge_face_dof(EdgeKey::new(nodes[0], nodes[1]))
                } else {
                    None
                }
            } else {
                if nodes.len() >= 3 {
                    space.tri_face_dof(FaceKey::new(nodes[0], nodes[1], nodes[2]))
                } else {
                    None
                }
            };
            if let Some(d) = dof {
                out.push(d);
            }
        }
    }

    out.sort_unstable();
    out.dedup();
    out
}

// ─── Hanging-node constraints ───────────────────────────────────────────────

/// Apply hanging-node constraints to the assembled system `(K, f)`.
///
/// For each constraint `u_c = 0.5*(u_a + u_b)`, the constrained DOF is
/// eliminated by substituting the interpolation into the variational form.
///
/// The implementation rebuilds the matrix via COO format to handle new
/// sparsity entries that arise from the distribution step.
///
/// After solving, call [`recover_hanging_values`] to fill in constrained DOFs.
pub fn apply_hanging_constraints(
    mat: &mut CsrMatrix<f64>,
    rhs: &mut [f64],
    constraints: &[HangingNodeConstraint],
) {
    if constraints.is_empty() { return; }

    let n = mat.nrows;

    // Build interpolation matrix P conceptually:
    //   For free DOF i:  u_i = x_i  (identity)
    //   For constrained c: u_c = 0.5*x_a + 0.5*x_b
    //
    // The constrained system is: P^T K P x = P^T f
    // where x has the constrained DOFs set to 0 (they'll be recovered later).
    //
    // In practice, we compute K' = P^T K P and f' = P^T f directly.

    let mut constraint_map = std::collections::HashMap::new();
    for c in constraints {
        constraint_map.insert(c.constrained, (c.parent_a, c.parent_b));
    }

    // Recursively expand a DOF into its free-DOF contributions.
    // Handles chains: if DOF is constrained to parents that are also constrained,
    // the expansion follows through until only free DOFs remain.
    fn expand_dof(
        dof: usize,
        weight: f64,
        constraint_map: &std::collections::HashMap<usize, (usize, usize)>,
        out: &mut Vec<(usize, f64)>,
        depth: usize,
    ) {
        if depth > 20 { return; } // safety guard against cycles
        if let Some(&(a, b)) = constraint_map.get(&dof) {
            expand_dof(a, weight * 0.5, constraint_map, out, depth + 1);
            expand_dof(b, weight * 0.5, constraint_map, out, depth + 1);
        } else {
            out.push((dof, weight));
        }
    }

    // Build K' in COO format.
    let mut coo = CooMatrix::<f64>::new(n, n);

    for i in 0..n {
        let start = mat.row_ptr[i];
        let end = mat.row_ptr[i + 1];

        // Effective row indices: recursively expand if constrained.
        let mut i_targets: Vec<(usize, f64)> = Vec::new();
        expand_dof(i, 1.0, &constraint_map, &mut i_targets, 0);

        for p in start..end {
            let j = mat.col_idx[p] as usize;
            let v = mat.values[p];
            if v.abs() < 1e-30 { continue; }

            // Effective column indices: recursively expand if constrained.
            let mut j_targets: Vec<(usize, f64)> = Vec::new();
            expand_dof(j, 1.0, &constraint_map, &mut j_targets, 0);

            // Add v * alpha_i * alpha_j to K'[ii, jj] for all target pairs.
            for &(ii, ai) in &i_targets {
                for &(jj, aj) in &j_targets {
                    coo.add(ii, jj, v * ai * aj);
                }
            }
        }
    }

    // Set identity rows for constrained DOFs.
    for c in constraints {
        coo.add(c.constrained, c.constrained, 1.0);
    }

    // Build f' = P^T f — also with recursive expansion.
    // Process in reverse topological order (constrained DOFs that depend on
    // other constrained DOFs need those resolved first).
    // Simpler approach: expand each constrained DOF recursively.
    let mut new_rhs = vec![0.0_f64; n];
    for i in 0..n {
        if rhs[i].abs() < 1e-30 { continue; }
        let mut targets = Vec::new();
        expand_dof(i, 1.0, &constraint_map, &mut targets, 0);
        for &(t, w) in &targets {
            new_rhs[t] += w * rhs[i];
        }
    }
    // Zero out constrained DOF RHS.
    for c in constraints {
        new_rhs[c.constrained] = 0.0;
    }
    rhs.copy_from_slice(&new_rhs);

    *mat = coo.into_csr();
}

/// Recover hanging-node DOF values after solving.
///
/// Sets `x[c] = 0.5*(x[a] + x[b])` for each hanging-node constraint.
/// Handles chained constraints by processing in topological order:
/// constraints whose parents are free are resolved first, then constraints
/// whose parents are now resolved, etc.
///
/// Call this after the linear solve and before post-processing.
pub fn recover_hanging_values(
    x: &mut [f64],
    constraints: &[HangingNodeConstraint],
) {
    if constraints.is_empty() { return; }

    let constrained_set: std::collections::HashSet<usize> =
        constraints.iter().map(|c| c.constrained).collect();

    // Topological sort: process constraints whose parents are NOT constrained first.
    let mut remaining: Vec<&HangingNodeConstraint> = constraints.iter().collect();
    let mut resolved = std::collections::HashSet::new();

    // Iterate until all resolved (bounded by constraint count).
    for _ in 0..constraints.len() + 1 {
        let mut progress = false;
        remaining.retain(|c| {
            let a_free = !constrained_set.contains(&c.parent_a) || resolved.contains(&c.parent_a);
            let b_free = !constrained_set.contains(&c.parent_b) || resolved.contains(&c.parent_b);
            if a_free && b_free {
                x[c.constrained] = 0.5 * (x[c.parent_a] + x[c.parent_b]);
                resolved.insert(c.constrained);
                progress = true;
                false // remove from remaining
            } else {
                true // keep
            }
        });
        if remaining.is_empty() || !progress { break; }
    }

    // Handle any remaining (shouldn't happen with valid constraints, but just in case).
    for c in remaining {
        x[c.constrained] = 0.5 * (x[c.parent_a] + x[c.parent_b]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fe_space::FESpace;
    use fem_linalg::CooMatrix;
    use fem_mesh::SimplexMesh;

    fn simple_system() -> (CsrMatrix<f64>, Vec<f64>) {
        let mut coo = CooMatrix::<f64>::new(3, 3);
        coo.add(0, 0,  2.0); coo.add(0, 1, -1.0);
        coo.add(1, 0, -1.0); coo.add(1, 1,  2.0); coo.add(1, 2, -1.0);
        coo.add(2, 1, -1.0); coo.add(2, 2,  2.0);
        (coo.into_csr(), vec![1.0_f64; 3])
    }

    #[test]
    fn apply_dirichlet_zero_bc() {
        let (mut mat, mut rhs) = simple_system();
        apply_dirichlet(&mut mat, &mut rhs, &[0], &[0.0]);
        assert!((mat.get(0, 0) - 1.0).abs() < 1e-14);
        assert!((mat.get(0, 1)).abs() < 1e-14);
        assert!((rhs[0]).abs() < 1e-14);
    }

    #[test]
    fn apply_dirichlet_nonzero_bc() {
        let (mut mat, mut rhs) = simple_system();
        apply_dirichlet(&mut mat, &mut rhs, &[2], &[5.0]);
        assert!((mat.get(2, 2) - 1.0).abs() < 1e-14);
        assert!((rhs[2] - 5.0).abs() < 1e-14);
    }

    #[test]
    fn boundary_dofs_returns_sorted_valid_dofs() {
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let dm   = DofManager::new(&mesh, 1);
        let dofs = boundary_dofs(&mesh, &dm, &[1, 2, 3, 4]);
        assert!(!dofs.is_empty());
        for &d in &dofs {
            assert!((d as usize) < dm.n_dofs, "DOF {d} out of range");
        }
        // Check sorted
        for i in 1..dofs.len() {
            assert!(dofs[i] > dofs[i-1]);
        }
    }

    #[test]
    fn boundary_dofs_p2_includes_edge_midpoints() {
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let dm   = DofManager::new(&mesh, 2);
        let n_nodes = mesh.n_nodes();
        let dofs = boundary_dofs(&mesh, &dm, &[1, 2, 3, 4]);
        // At least some DOFs should be edge-midpoint DOFs (index >= n_nodes)
        let edge_dofs: Vec<_> = dofs.iter().filter(|&&d| d as usize >= n_nodes).collect();
        assert!(!edge_dofs.is_empty(), "no edge-midpoint boundary DOFs found for P2");
    }

    #[test]
    fn boundary_dofs_hcurl_unit_square() {
        use crate::hcurl::HCurlSpace;
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = HCurlSpace::new(mesh, 1);
        let dofs = boundary_dofs_hcurl(space.mesh(), &space, &[1, 2, 3, 4]);
        assert!(!dofs.is_empty(), "should find boundary edge DOFs");
        // 4×4 grid boundary has 4×4 = 16 boundary edges.
        assert_eq!(dofs.len(), 16, "4×4 unit square has 16 boundary edges");
        for &d in &dofs {
            assert!((d as usize) < space.n_dofs(), "DOF {d} out of range");
        }
        // Check sorted
        for i in 1..dofs.len() {
            assert!(dofs[i] > dofs[i - 1]);
        }
    }

    #[test]
    fn boundary_dofs_hdiv_unit_square() {
        use crate::hdiv::HDivSpace;
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = HDivSpace::new(mesh, 0);
        let dofs = boundary_dofs_hdiv(space.mesh(), &space, &[1, 2, 3, 4]);
        assert!(!dofs.is_empty(), "should find boundary face DOFs");
        // Same count as HCurl in 2-D: 16 boundary edges.
        assert_eq!(dofs.len(), 16, "4×4 unit square has 16 boundary edges");
        for &d in &dofs {
            assert!((d as usize) < space.n_dofs(), "DOF {d} out of range");
        }
    }

    // ── Hanging-node constraint tests ────────────────────────────────────────

    #[test]
    fn recover_hanging_values_simple() {
        let mut x = vec![2.0, 6.0, 0.0]; // DOF 2 is hanging between 0 and 1
        let constraints = vec![HangingNodeConstraint {
            constrained: 2, parent_a: 0, parent_b: 1,
        }];
        recover_hanging_values(&mut x, &constraints);
        assert!((x[2] - 4.0).abs() < 1e-14, "expected 0.5*(2+6)=4, got {}", x[2]);
    }

    #[test]
    fn recover_hanging_values_chained() {
        // DOF 2 = mid(0, 1), DOF 3 = mid(1, 2)
        // DOF 2 should be recovered first since its parents are free,
        // then DOF 3 uses the recovered DOF 2.
        let mut x = vec![0.0, 4.0, 0.0, 0.0];
        let constraints = vec![
            HangingNodeConstraint { constrained: 2, parent_a: 0, parent_b: 1 },
            HangingNodeConstraint { constrained: 3, parent_a: 1, parent_b: 2 },
        ];
        recover_hanging_values(&mut x, &constraints);
        // DOF 2 = 0.5*(0 + 4) = 2
        assert!((x[2] - 2.0).abs() < 1e-14, "expected x[2]=2, got {}", x[2]);
        // DOF 3 = 0.5*(4 + 2) = 3
        assert!((x[3] - 3.0).abs() < 1e-14, "expected x[3]=3, got {}", x[3]);
    }

    #[test]
    fn apply_hanging_constraints_chained() {
        // 6-DOF system: DOF 3 = mid(1, 2), DOF 4 = mid(2, 3).
        // DOF 4 depends on DOF 3 which is also constrained.
        // After expansion: DOF 4 = 0.5*(u2 + 0.5*(u1 + u2)) = 0.25*u1 + 0.75*u2.
        let n = 6;
        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n {
            coo.add(i, i, 2.0);
            if i > 0     { coo.add(i, i - 1, -1.0); }
            if i < n - 1 { coo.add(i, i + 1, -1.0); }
        }
        let mut mat = coo.into_csr();
        let mut rhs = vec![1.0; n];

        let constraints = vec![
            HangingNodeConstraint { constrained: 3, parent_a: 1, parent_b: 2 },
            HangingNodeConstraint { constrained: 4, parent_a: 2, parent_b: 3 },
        ];

        apply_hanging_constraints(&mut mat, &mut rhs, &constraints);

        // Constrained rows should be identity.
        assert!((mat.get(3, 3) - 1.0).abs() < 1e-14);
        assert!((mat.get(4, 4) - 1.0).abs() < 1e-14);
        assert!((rhs[3]).abs() < 1e-14);
        assert!((rhs[4]).abs() < 1e-14);
    }

    #[test]
    fn apply_hanging_constraints_identity_row() {
        // 4-DOF system: DOF 2 is constrained to 0.5*(DOF 0 + DOF 1).
        // After apply_hanging_constraints, row 2 should be identity.
        let mut coo = CooMatrix::<f64>::new(4, 4);
        for i in 0..4 {
            coo.add(i, i, 2.0);
            if i > 0     { coo.add(i, i - 1, -1.0); }
            if i < 3     { coo.add(i, i + 1, -1.0); }
        }
        let mut mat = coo.into_csr();
        let mut rhs = vec![1.0; 4];
        let constraints = vec![HangingNodeConstraint {
            constrained: 2, parent_a: 0, parent_b: 1,
        }];

        apply_hanging_constraints(&mut mat, &mut rhs, &constraints);

        // Row 2 should be: K[2,2] = 1, all others 0.
        assert!((mat.get(2, 2) - 1.0).abs() < 1e-14, "K[2,2] should be 1");
        assert!((mat.get(2, 0)).abs() < 1e-14, "K[2,0] should be 0");
        assert!((mat.get(2, 1)).abs() < 1e-14, "K[2,1] should be 0");
        assert!((mat.get(2, 3)).abs() < 1e-14, "K[2,3] should be 0");
        assert!((rhs[2]).abs() < 1e-14, "rhs[2] should be 0");

        // Column 2 should be zero in all other rows.
        assert!((mat.get(0, 2)).abs() < 1e-14, "K[0,2] should be 0");
        assert!((mat.get(1, 2)).abs() < 1e-14, "K[1,2] should be 0");
        assert!((mat.get(3, 2)).abs() < 1e-14, "K[3,2] should be 0");
    }

    #[test]
    fn hanging_constraint_preserves_solvability() {
        // Build a small system, apply constraint, solve, recover.
        // 5-DOF 1-D Laplacian: DOF 2 is hanging between 1 and 3.
        let n = 5;
        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n {
            coo.add(i, i, 2.0);
            if i > 0     { coo.add(i, i - 1, -1.0); }
            if i < n - 1 { coo.add(i, i + 1, -1.0); }
        }
        let mut mat = coo.into_csr();
        let mut rhs = vec![1.0; n];

        // Dirichlet: DOF 0 = 0, DOF 4 = 0.
        apply_dirichlet(&mut mat, &mut rhs, &[0, 4], &[0.0, 0.0]);

        // Hanging constraint: DOF 2 = 0.5*(DOF 1 + DOF 3).
        let constraints = vec![HangingNodeConstraint {
            constrained: 2, parent_a: 1, parent_b: 3,
        }];
        apply_hanging_constraints(&mut mat, &mut rhs, &constraints);

        // Solve with simple direct solver (small enough).
        let mut x = vec![0.0; n];
        // Simple Gauss-Seidel iteration for this small system.
        for _ in 0..1000 {
            for i in 0..n {
                let start = mat.row_ptr[i];
                let end = mat.row_ptr[i + 1];
                let mut s = rhs[i];
                let mut diag = 1.0;
                for p in start..end {
                    let j = mat.col_idx[p] as usize;
                    if j == i { diag = mat.values[p]; }
                    else { s -= mat.values[p] * x[j]; }
                }
                x[i] = s / diag;
            }
        }

        // Recover hanging DOF.
        recover_hanging_values(&mut x, &constraints);

        // x[2] should be average of x[1] and x[3].
        assert!(
            (x[2] - 0.5 * (x[1] + x[3])).abs() < 1e-8,
            "hanging DOF: x[2]={}, 0.5*(x[1]+x[3])={}",
            x[2], 0.5 * (x[1] + x[3])
        );

        // Boundary conditions should hold.
        assert!(x[0].abs() < 1e-10, "x[0] = {}, expected 0", x[0]);
        assert!(x[4].abs() < 1e-10, "x[4] = {}, expected 0", x[4]);
    }
}
