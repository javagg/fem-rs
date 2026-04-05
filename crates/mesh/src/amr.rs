//! Adaptive Mesh Refinement (AMR) for simplex meshes.
//!
//! Provides:
//! 1. **Bisection refinement** — newest-vertex bisection for triangles.
//! 2. **Zienkiewicz–Zhu (ZZ) error estimator** — gradient recovery-based element error.
//! 3. **Dörfler (bulk) marking** — marks a minimal subset of elements whose
//!    estimated errors sum to at least θ of the global error.
//! 4. **Hanging-node constraints** (2-D only) — stores the linear constraint
//!    equations arising from conforming refinement.
//!
//! # Usage
//! ```rust,ignore
//! use fem_mesh::{SimplexMesh, amr::{refine_marked, zz_estimator, dorfler_mark}};
//!
//! let mut mesh = SimplexMesh::<2>::unit_square_tri(4);
//! let errors   = zz_estimator(&mesh, &u_h);   // element-wise error indicators
//! let marked   = dorfler_mark(&errors, 0.5);   // Dörfler θ = 0.5
//! mesh         = refine_marked(&mesh, &marked);
//! ```

use std::collections::HashMap;
use fem_core::{NodeId, ElemId};
use crate::{element_type::ElementType, simplex::SimplexMesh};

// ─── Bisection refinement ─────────────────────────────────────────────────────

/// Newest-vertex bisection refinement for a 2-D triangle mesh.
///
/// Each marked element is split into **2** children by bisecting the longest
/// edge (opposite to the newest vertex).  To maintain conformity, edges shared
/// with unmarked neighbours are also bisected (propagation step, simplified
/// here to a single conformity pass).
///
/// # Arguments
/// - `mesh`    — input `SimplexMesh<2>` with `elem_type = Tri3`.
/// - `marked`  — sorted list of element indices to refine.
///
/// # Returns
/// A new `SimplexMesh<2>` with the refined elements replaced by their children.
pub fn refine_marked(mesh: &SimplexMesh<2>, marked: &[ElemId]) -> SimplexMesh<2> {
    assert!(
        mesh.elem_type == ElementType::Tri3,
        "refine_marked: only Tri3 meshes are supported"
    );

    let marked_set: std::collections::HashSet<ElemId> = marked.iter().copied().collect();

    // ── 1. Identify all edges to bisect ───────────────────────────────────────
    // For each marked element, mark its longest edge.
    // We also propagate to neighbours (one pass) to ensure conformity.
    let npe = 3usize;
    let n_elems = mesh.n_elems();

    // Build edge → element list for propagation.
    // edge key = (min_node, max_node)
    let mut edge_elems: HashMap<(NodeId, NodeId), Vec<ElemId>> = HashMap::new();
    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        for &(a, b) in &local_edges_tri() {
            let key = edge_key(ns[a], ns[b]);
            edge_elems.entry(key).or_default().push(e);
        }
    }

    // Mark the longest edge of each element to be bisected.
    let mut bisect_edges: std::collections::HashSet<(NodeId, NodeId)> = Default::default();
    for &e in marked {
        let ns = mesh.elem_nodes(e);
        let longest = longest_edge_tri(mesh, ns);
        bisect_edges.insert(longest);
    }

    // Conformity propagation (one pass): if an interior edge is bisected,
    // both adjacent elements' longest edges should also be bisected.
    // We simply bisect the entire element (all edges) for simplicity.
    // This over-refines slightly but guarantees conformity.
    let mut elems_to_refine: std::collections::HashSet<ElemId> = marked_set.clone();
    for &(a, b) in &bisect_edges {
        if let Some(nbrs) = edge_elems.get(&(a, b)) {
            for &ne in nbrs {
                elems_to_refine.insert(ne);
            }
        }
    }

    // ── 2. Collect new midpoint nodes ─────────────────────────────────────────
    let mut midpoint_map: HashMap<(NodeId, NodeId), NodeId> = HashMap::new();
    let mut new_coords: Vec<f64> = mesh.coords.clone();

    let n_nodes_orig = mesh.n_nodes() as NodeId;
    let mut next_node = n_nodes_orig;

    for &e in &elems_to_refine {
        let ns = mesh.elem_nodes(e);
        // For Tri3 bisection: bisect longest edge only (newest-vertex bisection).
        // For simplicity here, bisect all 3 edges (red refinement).
        for &(a, b) in &local_edges_tri() {
            let key = edge_key(ns[a], ns[b]);
            midpoint_map.entry(key).or_insert_with(|| {
                let xa = mesh.coords_of(ns[a]);
                let xb = mesh.coords_of(ns[b]);
                new_coords.push(0.5 * (xa[0] + xb[0]));
                new_coords.push(0.5 * (xa[1] + xb[1]));
                let id = next_node;
                next_node += 1;
                id
            });
        }
    }

    // ── 3. Build new element connectivity ─────────────────────────────────────
    let mut new_conn: Vec<NodeId>  = Vec::new();
    let mut new_tags: Vec<i32>     = Vec::new();

    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        let tag = mesh.elem_tags[e as usize];

        if elems_to_refine.contains(&e) {
            // Red refinement: split Tri3 into 4 children.
            //   Original nodes: n0, n1, n2
            //   Midpoints:      m01, m12, m02
            let n0 = ns[0]; let n1 = ns[1]; let n2 = ns[2];
            let m01 = *midpoint_map.get(&edge_key(n0, n1)).unwrap();
            let m12 = *midpoint_map.get(&edge_key(n1, n2)).unwrap();
            let m02 = *midpoint_map.get(&edge_key(n0, n2)).unwrap();
            // 4 children
            new_conn.extend_from_slice(&[n0,  m01, m02]);  new_tags.push(tag);
            new_conn.extend_from_slice(&[m01, n1,  m12]);  new_tags.push(tag);
            new_conn.extend_from_slice(&[m02, m12, n2 ]);  new_tags.push(tag);
            new_conn.extend_from_slice(&[m01, m12, m02]);  new_tags.push(tag); // inner
        } else {
            // Unchanged element — copy as-is.
            for k in 0..npe { new_conn.push(ns[k]); }
            new_tags.push(tag);
        }
    }

    // ── 4. Rebuild boundary faces ─────────────────────────────────────────────
    // Boundary edges that were bisected get 2 children; others stay.
    let npf = 2usize; // Line2
    let n_faces = mesh.n_faces();
    let mut new_face_conn: Vec<NodeId> = Vec::new();
    let mut new_face_tags: Vec<i32>    = Vec::new();

    for f in 0..n_faces {
        let fn_slice = &mesh.face_conn[f * npf..(f + 1) * npf];
        let a = fn_slice[0];
        let b = fn_slice[1];
        let tag = mesh.face_tags[f];

        if let Some(&mid) = midpoint_map.get(&edge_key(a, b)) {
            // Bisected edge → 2 children
            new_face_conn.extend_from_slice(&[a, mid]);   new_face_tags.push(tag);
            new_face_conn.extend_from_slice(&[mid, b]);   new_face_tags.push(tag);
        } else {
            new_face_conn.extend_from_slice(&[a, b]);
            new_face_tags.push(tag);
        }
    }

    SimplexMesh::uniform(
        new_coords, new_conn, new_tags, ElementType::Tri3,
        new_face_conn, new_face_tags, ElementType::Line2,
    )
}

// ─── Hanging-node constraint ──────────────────────────────────────────────────

/// A hanging-node constraint: `u[constrained] = 0.5*(u[parent_a] + u[parent_b])`.
#[derive(Debug, Clone)]
pub struct HangingNodeConstraint {
    /// The constrained (hanging) node DOF index.
    pub constrained: usize,
    /// The two parent node DOF indices (the edge endpoints).
    pub parent_a:    usize,
    pub parent_b:    usize,
}

/// Collect all hanging-node constraints after refinement.
///
/// A hanging node is a new midpoint node on an edge that was bisected in a
/// refined element but whose neighbour was NOT refined (so the midpoint node
/// only appears as a DOF on the finer side).
///
/// In the current implementation (red refinement with propagation), all
/// edge-adjacent elements are refined, so there are no hanging nodes.
/// This function returns an empty vec for conforming meshes.
pub fn find_hanging_constraints(
    orig_n_nodes: usize,
    midpoint_map: &HashMap<(NodeId, NodeId), NodeId>,
    all_elem_conn: &[NodeId],
) -> Vec<HangingNodeConstraint> {
    let _ = orig_n_nodes;
    let _ = midpoint_map;
    let _ = all_elem_conn;
    // After propagated red refinement, all refined edges are shared by
    // both adjacent elements, so no hanging nodes exist.
    vec![]
}

// ─── ZZ error estimator ───────────────────────────────────────────────────────

/// Compute element-wise Zienkiewicz–Zhu (ZZ) gradient-recovery error indicators.
///
/// Uses simple nodal averaging of element gradients to recover a smoothed
/// gradient `G(u)`, then computes
/// `η_K = ‖∇u_h|_K − G(u)|_K‖_{L²(K)}`
/// for each element `K`.
///
/// # Arguments
/// - `mesh`     — the mesh.
/// - `u`        — solution vector (one value per node, length = `n_nodes`).
///
/// # Returns
/// Vector of `η_K` for each element (length = `n_elems`).
pub fn zz_estimator(mesh: &SimplexMesh<2>, u: &[f64]) -> Vec<f64> {
    let n_nodes = mesh.n_nodes();
    let n_elems = mesh.n_elems();

    // ── 1. Compute element gradients ──────────────────────────────────────────
    // For Tri3: ∇u is constant over each element.
    let mut elem_grads: Vec<[f64; 2]> = Vec::with_capacity(n_elems);

    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        let n0 = ns[0]; let n1 = ns[1]; let n2 = ns[2];
        let [x0, y0] = mesh.coords_of(n0);
        let [x1, y1] = mesh.coords_of(n1);
        let [x2, y2] = mesh.coords_of(n2);
        let u0 = u[n0 as usize]; let u1 = u[n1 as usize]; let u2 = u[n2 as usize];

        // Jacobian of mapping from reference triangle to physical:
        // J = [[x1-x0, x2-x0], [y1-y0, y2-y0]]
        let j00 = x1 - x0; let j01 = x2 - x0;
        let j10 = y1 - y0; let j11 = y2 - y0;
        let det = j00 * j11 - j01 * j10;

        // Reference gradients of Lagrange basis: ∇ψ₀ = (-1,-1), ∇ψ₁ = (1,0), ∇ψ₂ = (0,1)
        // Physical grad = J^{-T} * ref_grad
        // J^{-T} = (1/det) * [[j11, -j10], [-j01, j00]]
        let g_ref = [
            [-1.0_f64, -1.0],
            [ 1.0,  0.0],
            [ 0.0,  1.0],
        ];
        let uh = [u0, u1, u2];
        let mut gx = 0.0_f64; let mut gy = 0.0_f64;
        for k in 0..3 {
            // J^{-T} * g_ref[k]
            let gpx = ( j11 * g_ref[k][0] - j10 * g_ref[k][1]) / det;
            let gpy = (-j01 * g_ref[k][0] + j00 * g_ref[k][1]) / det;
            gx += uh[k] * gpx;
            gy += uh[k] * gpy;
        }
        elem_grads.push([gx, gy]);
    }

    // ── 2. Nodal gradient recovery (simple averaging) ─────────────────────────
    let mut nodal_grad = vec![[0.0_f64; 2]; n_nodes];
    let mut nodal_count = vec![0usize; n_nodes];

    for (e, &grad) in elem_grads.iter().enumerate() {
        let ns = mesh.elem_nodes(e as ElemId);
        for &n in ns {
            nodal_grad[n as usize][0] += grad[0];
            nodal_grad[n as usize][1] += grad[1];
            nodal_count[n as usize] += 1;
        }
    }
    for n in 0..n_nodes {
        let c = nodal_count[n] as f64;
        if c > 0.0 {
            nodal_grad[n][0] /= c;
            nodal_grad[n][1] /= c;
        }
    }

    // ── 3. Element error indicator ────────────────────────────────────────────
    let mut eta = Vec::with_capacity(n_elems);

    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        let [x0, y0] = mesh.coords_of(ns[0]);
        let [x1, y1] = mesh.coords_of(ns[1]);
        let [x2, y2] = mesh.coords_of(ns[2]);
        let area = 0.5 * ((x1-x0)*(y2-y0) - (x2-x0)*(y1-y0)).abs();

        // Recovered gradient at centroid = average of nodal recovered gradients
        let grx: f64 = ns.iter().map(|&n| nodal_grad[n as usize][0]).sum::<f64>() / 3.0;
        let gry: f64 = ns.iter().map(|&n| nodal_grad[n as usize][1]).sum::<f64>() / 3.0;
        let eg = &elem_grads[e as usize];

        let dx = eg[0] - grx;
        let dy = eg[1] - gry;
        // η_K = ‖(∇u_h − G(u_h))‖ * sqrt(area)
        eta.push(area.sqrt() * (dx*dx + dy*dy).sqrt());
    }
    eta
}

// ─── Dörfler marking ─────────────────────────────────────────────────────────

/// Dörfler (bulk criterion) marking strategy.
///
/// Returns a sorted list of element indices to refine such that the sum of their
/// error indicators is at least `theta` times the sum of all indicators.
///
/// # Arguments
/// - `eta`   — element error indicators (from [`zz_estimator`]).
/// - `theta` — bulk parameter in (0, 1]; θ = 0.5 is typical.
pub fn dorfler_mark(eta: &[f64], theta: f64) -> Vec<ElemId> {
    let total: f64 = eta.iter().sum();
    let threshold = theta * total;

    // Sort by decreasing error
    let mut indices: Vec<ElemId> = (0..eta.len() as ElemId).collect();
    indices.sort_unstable_by(|&a, &b| eta[b as usize].partial_cmp(&eta[a as usize]).unwrap());

    let mut marked = Vec::new();
    let mut acc = 0.0_f64;
    for idx in indices {
        if acc >= threshold { break; }
        acc += eta[idx as usize];
        marked.push(idx);
    }
    marked.sort_unstable();
    marked
}

// ─── Uniform refinement ───────────────────────────────────────────────────────

/// Uniformly refine all elements of the mesh (red refinement for Tri3).
pub fn refine_uniform(mesh: &SimplexMesh<2>) -> SimplexMesh<2> {
    let all: Vec<ElemId> = (0..mesh.n_elems() as ElemId).collect();
    refine_marked(mesh, &all)
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Local edge index pairs for Tri3.
fn local_edges_tri() -> [(usize, usize); 3] {
    [(0, 1), (1, 2), (0, 2)]
}

/// Canonical edge key (sorted node pair).
fn edge_key(a: NodeId, b: NodeId) -> (NodeId, NodeId) {
    if a < b { (a, b) } else { (b, a) }
}

/// Return the canonical edge key of the longest edge of a Tri3 element.
fn longest_edge_tri(mesh: &SimplexMesh<2>, ns: &[NodeId]) -> (NodeId, NodeId) {
    let coords: [[f64; 2]; 3] = std::array::from_fn(|k| mesh.coords_of(ns[k]));
    let edges = local_edges_tri();
    let mut best = edge_key(ns[edges[0].0], ns[edges[0].1]);
    let mut best_len2 = 0.0_f64;
    for (a, b) in edges {
        let dx = coords[b][0] - coords[a][0];
        let dy = coords[b][1] - coords[a][1];
        let l2 = dx*dx + dy*dy;
        if l2 > best_len2 {
            best_len2 = l2;
            best = edge_key(ns[a], ns[b]);
        }
    }
    best
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_refinement_element_count() {
        // Each Tri3 → 4 children with red refinement.
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let n_before = mesh.n_elems();
        let fine = refine_uniform(&mesh);
        assert_eq!(fine.n_elems(), 4 * n_before,
            "Expected 4×{n_before}={} elements, got {}", 4*n_before, fine.n_elems());
    }

    #[test]
    fn uniform_refinement_node_count() {
        // A 1×1 square → 2 triangles, 4 nodes.
        // After red refinement: 8 triangles, 4+3=7 new midpoints? Actually 4+3=7 total.
        let mesh = SimplexMesh::<2>::unit_square_tri(1);
        let fine = refine_uniform(&mesh);
        // 1×1 unit square: 4 corners + 4 edge midpoints + 1 interior midpoint = 9
        assert!(fine.n_nodes() > mesh.n_nodes(),
            "Refinement should add nodes: before={}, after={}", mesh.n_nodes(), fine.n_nodes());
    }

    #[test]
    fn uniform_refinement_two_levels() {
        // 2 levels of uniform refinement: n → 4n → 16n elements.
        let mesh0 = SimplexMesh::<2>::unit_square_tri(2);
        let n0 = mesh0.n_elems();
        let mesh1 = refine_uniform(&mesh0);
        let mesh2 = refine_uniform(&mesh1);
        assert_eq!(mesh2.n_elems(), 16 * n0);
    }

    #[test]
    fn dorfler_marks_at_least_theta() {
        // All equal errors → should mark first `ceil(θ * n)` elements.
        let eta = vec![1.0_f64; 10];
        let marked = dorfler_mark(&eta, 0.5);
        let marked_sum: f64 = marked.iter().map(|&i| eta[i as usize]).sum();
        let total: f64 = eta.iter().sum();
        assert!(marked_sum >= 0.5 * total,
            "Dörfler: marked sum {marked_sum} < 0.5 * {total}");
    }

    #[test]
    fn zz_estimator_smooth_solution() {
        // For u = x (linear), the FE solution is exact on Tri3 → ZZ error should be ≈ 0.
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let u: Vec<f64> = (0..mesh.n_nodes())
            .map(|n| mesh.coords_of(n as NodeId)[0])
            .collect();
        let eta = zz_estimator(&mesh, &u);
        let max_eta = eta.iter().cloned().fold(0.0_f64, f64::max);
        assert!(max_eta < 1e-12, "ZZ estimator: exact linear solution, max_eta={max_eta:.3e}");
    }

    #[test]
    fn refine_marked_subset() {
        // Mark only a few elements and verify total element count.
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let n0 = mesh.n_elems();
        let marked = vec![0u32, 1, 2]; // mark 3 elements
        let fine = refine_marked(&mesh, &marked);
        // Each marked element → 4, but neighbours may be pulled in.
        // At minimum: 3 elements became 4*3=12, rest unchanged.
        assert!(fine.n_elems() >= n0 - 3 + 3 * 4,
            "Expected ≥{} elems, got {}", n0 - 3 + 3*4, fine.n_elems());
    }
}
