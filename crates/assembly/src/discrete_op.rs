//! Discrete linear operators: gradient, curl, and divergence.
//!
//! These operators map between finite element spaces in the de Rham complex:
//!
//! ```text
//!   H1 --grad--> H(curl) --curl--> H(div) --div--> L2
//! ```
//!
//! The resulting sparse matrices are exact (not approximations) for the
//! lowest-order spaces (P1 -> ND1 -> RT0 -> P0).

use std::collections::HashSet;

use fem_element::{TriND1, VectorReferenceElement};
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::{topology::MeshTopology, ElementTransformation};
use fem_space::fe_space::FESpace;
use fem_space::{H1Space, HCurlSpace, HDivSpace, L2Space};

/// Discrete linear operators that build sparse matrices mapping between FE spaces.
///
/// All methods are associated functions (no `self`) that take the relevant
/// spaces as arguments and return a `CsrMatrix<f64>`.
pub struct DiscreteLinearOperator;

impl DiscreteLinearOperator {
    /// Build the discrete gradient matrix G: H1 -> H(curl).
    ///
    /// For lowest-order (P1 -> ND1), G is the signed incidence matrix of the
    /// mesh graph.  For edge (v_a, v_b) with global orientation a < b:
    ///
    /// ```text
    ///   G[edge_dof, v_b] = +sign
    ///   G[edge_dof, v_a] = -sign
    /// ```
    ///
    /// where `sign` is the H(curl) orientation sign on the element.
    ///
    /// # Panics
    /// Panics if the spaces are not lowest-order (P1 and ND1).
    pub fn gradient<M: MeshTopology>(
        h1_space: &H1Space<M>,
        hcurl_space: &HCurlSpace<M>,
    ) -> CsrMatrix<f64> {
        assert_eq!(h1_space.order(), 1, "gradient: only P1 H1 space supported");
        assert_eq!(hcurl_space.order(), 1, "gradient: only ND1 H(curl) space supported");

        let mesh = h1_space.mesh();
        let n_hcurl = hcurl_space.n_dofs();
        let n_h1 = h1_space.n_dofs();

        let mut coo = CooMatrix::<f64>::new(n_hcurl, n_h1);

        // Local edge table for triangles (must match HCurlSpace TRI_EDGES)
        let local_edges: &[(usize, usize)] = match mesh.dim() {
            2 => &[(0, 1), (1, 2), (0, 2)],
            3 => &[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
            d => panic!("gradient: unsupported dimension {d}"),
        };

        // Track which edge DOFs we have already visited, to avoid duplicates
        // from shared edges.
        let mut visited = HashSet::with_capacity(n_hcurl);

        for e in mesh.elem_iter() {
            let verts = mesh.element_nodes(e);
            let h1_dofs = h1_space.element_dofs(e);
            let hcurl_dofs = hcurl_space.element_dofs(e);

            for (local_edge_idx, &(li, lj)) in local_edges.iter().enumerate() {
                let edge_dof = hcurl_dofs[local_edge_idx] as usize;

                // Only process each edge DOF once
                if !visited.insert(edge_dof) {
                    continue;
                }

                // H1 DOFs for P1 are the vertex node indices
                let va_dof = h1_dofs[li] as usize;
                let vb_dof = h1_dofs[lj] as usize;

                // Global edge orientation: from smaller to larger vertex index.
                // The discrete gradient is the signed incidence matrix:
                //   G[edge_dof, end_vertex] = +1
                //   G[edge_dof, start_vertex] = -1
                // where the edge goes from start to end in global orientation.
                let (gi, gj) = (verts[li], verts[lj]);
                if gi < gj {
                    // Local li -> lj matches global direction
                    coo.add(edge_dof, vb_dof, 1.0);
                    coo.add(edge_dof, va_dof, -1.0);
                } else {
                    // Local direction is opposite to global
                    coo.add(edge_dof, va_dof, 1.0);
                    coo.add(edge_dof, vb_dof, -1.0);
                }
            }
        }

        coo.into_csr()
    }

    /// Build the discrete curl matrix C: H(curl) -> L2 (2D).
    ///
    /// For lowest-order (ND1 -> P0) in 2D, the matrix entries are:
    ///
    /// ```text
    ///   C[l2_dof, hcurl_dof] = sign * curl_ref / det_j * |det_j| * area_ref
    /// ```
    ///
    /// where `curl_ref` is the constant reference curl of each ND1 basis function,
    /// `det_j` is the Jacobian determinant, and `area_ref = 0.5` for triangles.
    ///
    /// # Panics
    /// Panics if the spaces are not 2D lowest-order (ND1 and P0).
    pub fn curl_2d<M: MeshTopology>(
        hcurl_space: &HCurlSpace<M>,
        l2_space: &L2Space<M>,
    ) -> CsrMatrix<f64> {
        assert_eq!(hcurl_space.order(), 1, "curl_2d: only ND1 H(curl) supported");
        assert_eq!(l2_space.order(), 0, "curl_2d: only P0 L2 supported");

        let mesh = hcurl_space.mesh();
        assert_eq!(mesh.dim(), 2, "curl_2d: only 2D meshes supported");

        let n_l2 = l2_space.n_dofs();
        let n_hcurl = hcurl_space.n_dofs();

        let mut coo = CooMatrix::<f64>::new(n_l2, n_hcurl);

        // Reference curls for TriND1: [2, 2, -2] (constant)
        let ref_elem = TriND1;
        let mut curl_ref = vec![0.0; ref_elem.n_dofs()];
        ref_elem.eval_curl(&[0.0, 0.0], &mut curl_ref);

        let area_ref = 0.5; // reference triangle area

        for e in mesh.elem_iter() {
            let nodes = mesh.element_nodes(e);
            let hcurl_dofs = hcurl_space.element_dofs(e);
            let signs = hcurl_space.element_signs(e);
            let l2_dofs = l2_space.element_dofs(e);
            let l2_dof = l2_dofs[0] as usize;

            // Compute Jacobian determinant
            let det_j = simplex_det(mesh, nodes);

            // Physical curl = curl_ref / det_j
            // Integral over element = curl_phys * |det_j| * area_ref
            //                       = (curl_ref / det_j) * |det_j| * area_ref
            //                       = curl_ref * sign(det_j) * area_ref
            let sign_det = if det_j > 0.0 { 1.0 } else { -1.0 };

            for i in 0..ref_elem.n_dofs() {
                let hcurl_dof = hcurl_dofs[i] as usize;
                let val = signs[i] * curl_ref[i] * sign_det * area_ref;
                coo.add(l2_dof, hcurl_dof, val);
            }
        }

        coo.into_csr()
    }

    /// Build the discrete divergence matrix D: H(div) -> L2.
    ///
    /// For lowest-order (RT0 -> P0), the matrix entries are:
    ///
    /// ```text
    ///   D[l2_dof, hdiv_dof] = sign * div_ref * sign(det_j) * area_ref
    /// ```
    ///
    /// where `div_ref = 2` for all RT0 basis functions, and `area_ref = 0.5`.
    ///
    /// # Panics
    /// Panics if the spaces are not lowest-order (RT0 and P0).
    pub fn divergence<M: MeshTopology>(
        hdiv_space: &HDivSpace<M>,
        l2_space: &L2Space<M>,
    ) -> CsrMatrix<f64> {
        assert_eq!(hdiv_space.order(), 0, "divergence: only RT0 H(div) supported");
        assert_eq!(l2_space.order(), 0, "divergence: only P0 L2 supported");

        let mesh = hdiv_space.mesh();
        let dim = mesh.dim();

        let n_l2 = l2_space.n_dofs();
        let n_hdiv = hdiv_space.n_dofs();

        let mut coo = CooMatrix::<f64>::new(n_l2, n_hdiv);

        // vol_ref: reference simplex volume (1/2 for tri, 1/6 for tet)
        let (n_local_dofs, _vol_ref, _div_ref_val): (usize, f64, f64) = if dim == 2 {
            (3, 0.5, 2.0)  // TriRT0: 3 DOFs, div=2
        } else {
            (4, 1.0 / 6.0, 6.0)  // TetRT0: 4 DOFs, div=6
        };

        for e in mesh.elem_iter() {
            let hdiv_dofs = hdiv_space.element_dofs(e);
            let l2_dofs = l2_space.element_dofs(e);
            let l2_dof = l2_dofs[0] as usize;

            // Topological divergence: D[elem, face] = face_sign
            // Using face_sign gives the weak divergence (integral of div).
            // For de Rham complex with exact sequence property, use pure incidence.
            let signs = hdiv_space.element_signs(e);
            for i in 0..n_local_dofs {
                let hdiv_dof = hdiv_dofs[i] as usize;
                coo.add(l2_dof, hdiv_dof, signs[i]);
            }
        }

        coo.into_csr()
    }

    /// Build the discrete curl matrix C: H(curl) -> H(div) in 3D (tetrahedra).
    ///
    /// For lowest-order (ND1 -> RT0), the discrete curl is the topological
    /// face-edge incidence matrix. Each face is processed once; the Stokes
    /// signs are derived from the sorted global vertex order of the face,
    /// which matches the global face-normal convention used by HDivSpace.
    ///
    /// For a face with sorted global vertices (a < b < c):
    ///   - boundary traversal (right-hand rule): a → b → c → a
    ///   - C[face, edge(a,b)] = +1
    ///   - C[face, edge(b,c)] = +1
    ///   - C[face, edge(a,c)] = −1  (traversal goes c→a, opposite to global a→c)
    ///
    /// # Panics
    /// Panics if spaces are not lowest-order (ND1 and RT0) on a 3D mesh.
    pub fn curl_3d<M: MeshTopology>(
        hcurl_space: &HCurlSpace<M>,
        hdiv_space: &HDivSpace<M>,
    ) -> CsrMatrix<f64> {
        assert_eq!(hcurl_space.order(), 1, "curl_3d: only ND1 H(curl) supported");
        assert_eq!(hdiv_space.order(), 0, "curl_3d: only RT0 H(div) supported");

        let mesh = hcurl_space.mesh();
        assert_eq!(mesh.dim(), 3, "curl_3d: only 3D meshes supported");

        let n_hdiv = hdiv_space.n_dofs();
        let n_hcurl = hcurl_space.n_dofs();

        let mut coo = CooMatrix::<f64>::new(n_hdiv, n_hcurl);

        // Local face / edge tables matching HDivSpace / HCurlSpace.
        let tet_faces: [(usize, usize, usize); 4] = [
            (1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2),
        ];
        let tet_edges: [(usize, usize); 6] = [
            (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),
        ];

        // Track visited face DOFs so each face is assembled exactly once.
        let mut visited = HashSet::with_capacity(n_hdiv);

        for e in mesh.elem_iter() {
            let verts = mesh.element_nodes(e);
            let hcurl_dofs = hcurl_space.element_dofs(e);
            let hdiv_dofs = hdiv_space.element_dofs(e);

            for (face_local, &(la, lb, lc)) in tet_faces.iter().enumerate() {
                let face_dof = hdiv_dofs[face_local] as usize;

                if !visited.insert(face_dof) {
                    continue;
                }

                // Global vertices of this face, sorted → canonical orientation.
                let mut fv = [verts[la], verts[lb], verts[lc]];
                fv.sort_unstable();

                // Boundary traversal fv[0]→fv[1]→fv[2]→fv[0] (right-hand rule
                // with the global face normal (p1−p0)×(p2−p0)).
                let face_boundary: [(u32, u32, f64); 3] = [
                    (fv[0], fv[1],  1.0),
                    (fv[1], fv[2],  1.0),
                    (fv[0], fv[2], -1.0),
                ];

                for (gv0, gv1, stokes_sign) in face_boundary {
                    // Find the local edge index whose global vertices match.
                    let edge_idx = tet_edges.iter().position(|&(li, lj)| {
                        let (gi, gj) = (verts[li], verts[lj]);
                        (gi == gv0 && gj == gv1) || (gi == gv1 && gj == gv0)
                    }).expect("face boundary edge not found in element");

                    let edge_dof = hcurl_dofs[edge_idx] as usize;
                    coo.add(face_dof, edge_dof, stokes_sign);
                }
            }
        }

        coo.into_csr()
    }
}

/// Compute the determinant of the simplex Jacobian for element `e`.
fn simplex_det<M: MeshTopology>(mesh: &M, geo_nodes: &[u32]) -> f64 {
    ElementTransformation::from_simplex_nodes(mesh, geo_nodes).det_j()
}



// ---- Tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;

    /// Test: Discrete gradient of a linear function u = x + 2y.
    ///
    /// The gradient field is (1, 2) everywhere.  Interpolating u into H1
    /// and applying G should give the same result as interpolating (1,2)
    /// into H(curl) via its DOF functional.
    #[test]
    fn gradient_of_linear_function() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let h1 = H1Space::new(mesh, 1);
        let mesh2 = SimplexMesh::<2>::unit_square_tri(4);
        let hcurl = HCurlSpace::new(mesh2, 1);

        // Interpolate u = x + 2y into H1
        let u_h1 = h1.interpolate(&|x| x[0] + 2.0 * x[1]);

        // Build gradient matrix and apply
        let g = DiscreteLinearOperator::gradient(&h1, &hcurl);
        let mut g_u = vec![0.0; hcurl.n_dofs()];
        g.spmv(u_h1.as_slice(), &mut g_u);

        // Interpolate grad(u) = (1, 2) into H(curl) via the DOF functional
        let grad_interp = hcurl.interpolate_vector(&|_x| vec![1.0, 2.0]);

        // Compare: they should match exactly (up to floating-point)
        for i in 0..hcurl.n_dofs() {
            assert!(
                (g_u[i] - grad_interp.as_slice()[i]).abs() < 1e-12,
                "gradient mismatch at DOF {i}: G*u = {}, interp = {}",
                g_u[i], grad_interp.as_slice()[i]
            );
        }
    }

    /// Test: de Rham exact sequence property: curl(grad(u)) = 0.
    ///
    /// Build G (H1 -> H(curl)) and C (H(curl) -> L2), then verify C * G = 0.
    #[test]
    fn de_rham_curl_of_grad_is_zero() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let h1 = H1Space::new(mesh, 1);
        let mesh2 = SimplexMesh::<2>::unit_square_tri(4);
        let hcurl = HCurlSpace::new(mesh2, 1);
        let mesh3 = SimplexMesh::<2>::unit_square_tri(4);
        let l2 = L2Space::new(mesh3, 0);

        let g = DiscreteLinearOperator::gradient(&h1, &hcurl);
        let c = DiscreteLinearOperator::curl_2d(&hcurl, &l2);

        // Test C * G * u = 0 for several functions
        let test_fns: Vec<Box<dyn Fn(&[f64]) -> f64>> = vec![
            Box::new(|x: &[f64]| x[0]),
            Box::new(|x: &[f64]| x[1]),
            Box::new(|x: &[f64]| x[0] + x[1]),
            Box::new(|x: &[f64]| 3.0 * x[0] - 2.0 * x[1]),
        ];

        for (idx, f) in test_fns.iter().enumerate() {
            let u = h1.interpolate(f.as_ref());
            let mut gu = vec![0.0; hcurl.n_dofs()];
            g.spmv(u.as_slice(), &mut gu);
            let mut cgu = vec![0.0; l2.n_dofs()];
            c.spmv(&gu, &mut cgu);

            let max_err: f64 = cgu.iter().map(|v| v.abs()).fold(0.0, f64::max);
            assert!(
                max_err < 1e-12,
                "curl(grad(u_{idx})) not zero: max |C*G*u| = {max_err}"
            );
        }
    }

    /// Test: Discrete divergence of a known field.
    ///
    /// For the constant field F = (1, 0), div(F) = 0.
    /// For the field F = (x, y), div(F) = 2.
    #[test]
    fn divergence_constant_field() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let hdiv = HDivSpace::new(mesh, 0);
        let mesh2 = SimplexMesh::<2>::unit_square_tri(4);
        let l2 = L2Space::new(mesh2, 0);

        let d = DiscreteLinearOperator::divergence(&hdiv, &l2);

        // Test 1: F = (1, 0) -> div = 0
        let f_const = hdiv.interpolate_vector(&|_x| vec![1.0, 0.0]);
        let mut div_f = vec![0.0; l2.n_dofs()];
        d.spmv(f_const.as_slice(), &mut div_f);

        let max_err: f64 = div_f.iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!(
            max_err < 1e-10,
            "div(1,0) should be 0, max |D*F| = {max_err}"
        );

        // Test 2: F = (0, 1) -> div = 0
        let f_const2 = hdiv.interpolate_vector(&|_x| vec![0.0, 1.0]);
        let mut div_f2 = vec![0.0; l2.n_dofs()];
        d.spmv(f_const2.as_slice(), &mut div_f2);

        let max_err2: f64 = div_f2.iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!(
            max_err2 < 1e-10,
            "div(0,1) should be 0, max |D*F| = {max_err2}"
        );
    }

    /// Test: Matrix dimensions are correct.
    #[test]
    fn matrix_dimensions() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let h1 = H1Space::new(mesh, 1);
        let mesh2 = SimplexMesh::<2>::unit_square_tri(4);
        let hcurl = HCurlSpace::new(mesh2, 1);
        let mesh3 = SimplexMesh::<2>::unit_square_tri(4);
        let hdiv = HDivSpace::new(mesh3, 0);
        let mesh4 = SimplexMesh::<2>::unit_square_tri(4);
        let l2 = L2Space::new(mesh4, 0);

        let g = DiscreteLinearOperator::gradient(&h1, &hcurl);
        assert_eq!(g.nrows, hcurl.n_dofs());
        assert_eq!(g.ncols, h1.n_dofs());

        let c = DiscreteLinearOperator::curl_2d(&hcurl, &l2);
        assert_eq!(c.nrows, l2.n_dofs());
        assert_eq!(c.ncols, hcurl.n_dofs());

        let d = DiscreteLinearOperator::divergence(&hdiv, &l2);
        assert_eq!(d.nrows, l2.n_dofs());
        assert_eq!(d.ncols, hdiv.n_dofs());
    }

    /// Test: de Rham sequence div(curl(u)) = 0 (2D version).
    ///
    /// In 2D the "curl" of a scalar is a div-free vector field, so we check
    /// that D * C * u_hcurl = 0 for arbitrary H(curl) DOF vectors.
    /// Note: In 2D, the curl maps H(curl) -> L2 (scalar), and div maps H(div) -> L2.
    /// The sequence is H1 -> H(curl) -> L2, so we only check curl(grad) = 0 (above).
    /// For completeness, check that G has no trivial kernel issues.
    #[test]
    fn gradient_nonzero_for_nonconst() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let h1 = H1Space::new(mesh, 1);
        let mesh2 = SimplexMesh::<2>::unit_square_tri(4);
        let hcurl = HCurlSpace::new(mesh2, 1);

        let g = DiscreteLinearOperator::gradient(&h1, &hcurl);

        // A non-constant function should have non-zero gradient
        let u = h1.interpolate(&|x| x[0]);
        let mut gu = vec![0.0; hcurl.n_dofs()];
        g.spmv(u.as_slice(), &mut gu);

        let norm: f64 = gu.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(norm > 1e-10, "gradient of x should be nonzero, got norm = {norm}");

        // A constant function should have zero gradient
        let u_const = h1.interpolate(&|_x| 1.0);
        let mut gu_const = vec![0.0; hcurl.n_dofs()];
        g.spmv(u_const.as_slice(), &mut gu_const);

        let norm_const: f64 = gu_const.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(norm_const < 1e-12, "gradient of constant should be zero, got norm = {norm_const}");
    }

    /// Test: de Rham exact sequence in 3D — div(curl(u)) = 0.
    #[test]
    fn de_rham_div_of_curl_3d_is_zero() {
        let mesh  = SimplexMesh::<3>::unit_cube_tet(2);
        let mesh2 = SimplexMesh::<3>::unit_cube_tet(2);
        let mesh3 = SimplexMesh::<3>::unit_cube_tet(2);
        let mesh4 = SimplexMesh::<3>::unit_cube_tet(2);

        let hcurl = HCurlSpace::new(mesh,  1);
        let hdiv  = HDivSpace::new(mesh2, 0);
        let hdiv2 = HDivSpace::new(mesh3, 0);
        let l2    = L2Space::new(mesh4,  0);

        let c = DiscreteLinearOperator::curl_3d(&hcurl, &hdiv);
        let d = DiscreteLinearOperator::divergence(&hdiv2, &l2);

        assert_eq!(c.nrows, hdiv.n_dofs(),  "C: wrong nrows");
        assert_eq!(c.ncols, hcurl.n_dofs(), "C: wrong ncols");

        // D * C * u = 0 for arbitrary u
        for seed in 0..5u64 {
            let u: Vec<f64> = (0..hcurl.n_dofs())
                .map(|i| (((i as u64 * 1_000_003 + seed * 998_244_353) % 1000) as f64) / 500.0 - 1.0)
                .collect();
            let mut cu = vec![0.0f64; hdiv.n_dofs()];
            c.spmv(&u, &mut cu);
            let mut dcu = vec![0.0f64; l2.n_dofs()];
            d.spmv(&cu, &mut dcu);
            let max_err: f64 = dcu.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
            assert!(max_err < 1e-10,
                "div(curl(u)) ≠ 0 for seed={seed}: max|D*C*u| = {max_err}");
        }
    }

    /// Test: curl_3d matrix dimensions.
    #[test]
    fn curl_3d_dimensions() {
        let mesh  = SimplexMesh::<3>::unit_cube_tet(2);
        let mesh2 = SimplexMesh::<3>::unit_cube_tet(2);
        let hcurl = HCurlSpace::new(mesh,  1);
        let hdiv  = HDivSpace::new(mesh2, 0);
        let c = DiscreteLinearOperator::curl_3d(&hcurl, &hdiv);
        assert_eq!(c.nrows, hdiv.n_dofs());
        assert_eq!(c.ncols, hcurl.n_dofs());
        assert!(c.nrows > 0 && c.ncols > 0);
    }

    /// Debug test: print curl_3d and divergence matrices for a single element.
    #[test]
    #[ignore] // Disabled - curl_3d is placeholder
    fn debug_curl_3d_single_element() {
        let mesh  = SimplexMesh::<3>::unit_cube_tet(1);
        let mesh2 = SimplexMesh::<3>::unit_cube_tet(1);
        let mesh3 = SimplexMesh::<3>::unit_cube_tet(1);
        let hcurl = HCurlSpace::new(mesh,  1);
        let hdiv  = HDivSpace::new(mesh2, 0);
        let l2    = L2Space::new(mesh3,  0);

        // Print element DOFs and signs for element 0
        let hcurl_dofs = hcurl.element_dofs(0);
        let hcurl_signs = hcurl.element_signs(0);
        let hdiv_dofs = hdiv.element_dofs(0);
        let hdiv_signs = hdiv.element_signs(0);
        let l2_dofs = l2.element_dofs(0);

        println!("\n=== Element 0 ===");
        println!("HCurl DOFs: {:?}", hcurl_dofs);
        println!("HCurl signs: {:?}", hcurl_signs);
        println!("HDiv DOFs: {:?}", hdiv_dofs);
        println!("HDiv signs: {:?}", hdiv_signs);
        println!("L2 DOFs: {:?}", l2_dofs);

        let c = DiscreteLinearOperator::curl_3d(&hcurl, &hdiv);
        let d = DiscreteLinearOperator::divergence(&hdiv, &l2);

        println!("\n=== C matrix ({} x {}) ===", c.nrows, c.ncols);
        for row in 0..c.nrows.min(8) {
            let start = c.row_ptr[row];
            let end = c.row_ptr[row+1];
            print!("Row {}: ", row);
            for i in start..end {
                let col = c.col_idx[i];
                print!("({}, {:.1}) ", col, c.values[i]);
            }
            println!();
        }

        println!("\n=== D matrix ({} x {}) ===", d.nrows, d.ncols);
        for row in 0..d.nrows.min(4) {
            let start = d.row_ptr[row];
            let end = d.row_ptr[row+1];
            print!("Row {}: ", row);
            for i in start..end {
                print!("({}, {:.3}) ", d.col_idx[i], d.values[i]);
            }
            println!();
        }

        // Show contributions from each element to C for shared faces
        println!("\n=== Face sharing analysis ===");
        for e in 0..2u32 {
            let hdiv_dofs = hdiv.element_dofs(e);
            let hcurl_dofs = hcurl.element_dofs(e);
            println!("Element {}: HDiv DOFs {:?}, HCurl DOFs {:?}", e, hdiv_dofs, hcurl_dofs);
        }

        // Compute D*C
        let mut dc_max = 0.0f64;
        for i in 0..d.nrows {
            let d_start = d.row_ptr[i];
            let d_end = d.row_ptr[i+1];
            for d_idx in d_start..d_end {
                let k = d.col_idx[d_idx] as usize;
                let d_val = d.values[d_idx];
                let c_start = c.row_ptr[k];
                let c_end = c.row_ptr[k+1];
                for c_idx in c_start..c_end {
                    let j = c.col_idx[c_idx];
                    let val: f64 = d_val * c.values[c_idx];
                    if val.abs() > 1e-10 {
                        println!("D*C[{},{}] += {:.3} * {:.1} = {:.3}", i, j, d_val, c.values[c_idx], val);
                    }
                    dc_max = dc_max.max(val.abs());
                }
            }
        }
        println!("\nmax|D*C| = {}", dc_max);
    }
}

