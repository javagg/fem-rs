# fem-rs ↔ MFEM Correspondence Reference
> Tracks every major MFEM concept and its planned or implemented fem-rs counterpart.
> Use this as the authoritative target checklist for feature completeness.
>
> Status legend: ✅ implemented · 🔨 partial · 🔲 planned · ❌ out-of-scope

---

## Table of Contents
1. [Mesh](#1-mesh)
2. [Reference Elements & Quadrature](#2-reference-elements--quadrature)
3. [Finite Element Spaces](#3-finite-element-spaces)
4. [Coefficients](#4-coefficients)
5. [Assembly: Forms & Integrators](#5-assembly-forms--integrators)
6. [Linear Algebra](#6-linear-algebra)
7. [Solvers & Preconditioners](#7-solvers--preconditioners)
8. [Algebraic Multigrid](#8-algebraic-multigrid)
9. [Parallel Infrastructure](#9-parallel-infrastructure)
10. [I/O & Visualization](#10-io--visualization)
11. [Grid Functions & Post-processing](#11-grid-functions--post-processing)
12. [MFEM Examples → fem-rs Milestones](#12-mfem-examples--fem-rs-milestones)
13. [Key Design Differences](#13-key-design-differences)

---

## 1. Mesh

### 1.1 Mesh Container

| MFEM class / concept | fem-rs equivalent | Status | Notes |
|---|---|---|---|
| `Mesh` (2D/3D unstructured) | `SimplexMesh<D>` | ✅ | Uniform element type per mesh |
| `Mesh` (mixed elements) | `SimplexMesh<D>` + multiple `elem_type` | 🔲 | Phase 2: per-element type array |
| `NCMesh` (non-conforming) | — | 🔲 | Phase 2+: hanging nodes for AMR |
| `ParMesh` | `ParallelMesh<M>` | 🔲 | Phase 10 |
| `Mesh::GetNV()` | `MeshTopology::n_nodes()` | ✅ | |
| `Mesh::GetNE()` | `MeshTopology::n_elements()` | ✅ | |
| `Mesh::GetNBE()` | `MeshTopology::n_boundary_faces()` | ✅ | |
| `Mesh::GetVerticesArray()` | `SimplexMesh::coords` (flat `Vec<f64>`) | ✅ | |
| `Mesh::GetElementVertices()` | `MeshTopology::element_nodes()` | ✅ | |
| `Mesh::GetBdrElementVertices()` | `MeshTopology::face_nodes()` | ✅ | |
| `Mesh::GetBdrAttribute()` | `MeshTopology::face_tag()` | ✅ | Tags match GMSH physical group IDs |
| `Mesh::GetAttribute()` | `MeshTopology::element_tag()` | ✅ | Material group tag |
| `Mesh::bdr_attributes` | `SimplexMesh::face_tags` (unique set) | 🔨 | No dedup utility yet |
| `Mesh::GetDim()` | `MeshTopology::dim()` | ✅ | Returns `u8` (2 or 3) |
| `Mesh::GetSpaceDim()` | same as `dim()` for flat meshes | ✅ | |
| `Mesh::UniformRefinement()` | `refine::uniform_refine()` | 🔲 | Phase 2: bisection for simplex |
| `Mesh::AdaptiveRefinement()` | — | 🔲 | Phase 2+: error-driven AMR |
| `Mesh::GetElementTransformation()` | Jacobian computed inline in assembly | 🔨 | No `ElementTransformation` type yet |
| `Mesh::GetFaceElementTransformations()` | `MeshTopology::face_elements()` | 🔨 | Phase 2: full adjacency map |
| `Mesh::GetBoundingBox()` | utility fn (not yet) | 🔲 | Phase 2 utility |

### 1.2 Element Types

| MFEM element | `ElementType` variant | dim | Nodes | Status |
|---|---|---|---|---|
| `Segment` | `Line2` | 1 | 2 | ✅ |
| Quadratic segment | `Line3` | 1 | 3 | ✅ |
| `Triangle` | `Tri3` | 2 | 3 | ✅ |
| Quadratic triangle | `Tri6` | 2 | 6 | ✅ |
| `Quadrilateral` | `Quad4` | 2 | 4 | ✅ |
| Serendipity quad | `Quad8` | 2 | 8 | ✅ |
| `Tetrahedron` | `Tet4` | 3 | 4 | ✅ |
| Quadratic tet | `Tet10` | 3 | 10 | ✅ |
| `Hexahedron` | `Hex8` | 3 | 8 | ✅ |
| Serendipity hex | `Hex20` | 3 | 20 | ✅ |
| `Wedge` (prism) | `Prism6` | 3 | 6 | ✅ (type only) |
| `Pyramid` | `Pyramid5` | 3 | 5 | ✅ (type only) |
| `Point` | `Point1` | 0 | 1 | ✅ |

### 1.3 Mesh Generators

| MFEM generator | fem-rs equivalent | Status |
|---|---|---|
| `Mesh::MakeCartesian2D()` | `SimplexMesh::unit_square_tri(n)` | ✅ |
| `Mesh::MakeCartesian3D()` | `SimplexMesh::unit_cube_tet(n)` | 🔲 Phase 2 |
| `Mesh::MakePeriodic()` | — | 🔲 Phase 2+ |
| Reading MFEM format | — | ❌ use GMSH instead |
| Reading GMSH `.msh` v4 | `fem_io::read_msh_file()` | ✅ |
| Reading Netgen | — | 🔲 Phase 9 |

---

## 2. Reference Elements & Quadrature

### 2.1 Reference Elements

| MFEM class | fem-rs trait/struct | Status |
|---|---|---|
| `FiniteElement` (base) | `ReferenceElement` trait | 🔲 Phase 3 |
| `Poly_1D` utility | inline basis in `lagrange/` | 🔲 Phase 3 |
| `H1_SegmentElement` | `lagrange::seg::P1Seg`, `P2Seg` | 🔲 Phase 3 |
| `H1_TriangleElement` | `lagrange::tri::P1Tri`, `P2Tri`, `P3Tri` | 🔲 Phase 3 |
| `H1_TetrahedronElement` | `lagrange::tet::P1Tet`, `P2Tet` | 🔲 Phase 3 |
| `H1_QuadrilateralElement` | `lagrange::quad::Q1Quad`, `Q2Quad` | 🔲 Phase 3 |
| `H1_HexahedronElement` | `lagrange::hex::Q1Hex`, `Q2Hex` | 🔲 Phase 3 |
| `ND_TriangleElement` | `nedelec::ND1Tri`, `ND2Tri` | 🔲 Phase 5 |
| `ND_TetrahedronElement` | `nedelec::ND1Tet` | 🔲 Phase 5 |
| `RT_TriangleElement` | `raviart_thomas::RT0Tri`, `RT1Tri` | 🔲 Phase 5 |
| `RT_TetrahedronElement` | `raviart_thomas::RT0Tet` | 🔲 Phase 5 |
| `L2_TriangleElement` | `lagrange::tri::P1TriDG` | 🔲 Phase 5 |

### 2.2 Quadrature Rules

| MFEM class | fem-rs struct | Status |
|---|---|---|
| `IntegrationRule` | `QuadratureRule` | 🔲 Phase 3 |
| `IntegrationRules` (table) | `quadrature.rs` look-up table | 🔲 Phase 3 |
| Gauss-Legendre 1D (orders 1–10) | `gauss_legendre_1d(order)` | 🔲 Phase 3 |
| Gauss-Legendre on triangle | `gauss_triangle(order)` | 🔲 Phase 3 |
| Gauss-Legendre on tet | `gauss_tet(order)` | 🔲 Phase 3 |
| Tensor product (quad, hex) | `tensor_gauss(order, dim)` | 🔲 Phase 3 |
| Gauss-Lobatto | — | 🔲 Phase 3+ |

---

## 3. Finite Element Spaces

### 3.1 Collections (Basis Families)

| MFEM collection | Mathematical space | fem-rs struct | Status |
|---|---|---|---|
| `H1_FECollection(p)` | H¹(Ω): C⁰ scalar Lagrange | `H1Space` | 🔲 Phase 5 |
| `L2_FECollection(p)` | L²(Ω): discontinuous Lagrange | `L2Space` | 🔲 Phase 5 |
| `DG_FECollection(p)` | L²(Ω): DG (element-interior only) | `L2Space` | 🔲 Phase 5 |
| `ND_FECollection(p)` | H(curl): Nédélec tangential | `NedelecSpace` | 🔲 Phase 5 |
| `RT_FECollection(p)` | H(div): Raviart-Thomas normal | `RTSpace` | 🔲 Phase 5 |
| `H1_Trace_FECollection` | H½: traces of H¹ on faces | — | 🔲 Phase 6+ |
| `NURBS_FECollection` | NURBS isogeometric | — | ❌ out of scope |

### 3.2 Finite Element Space (DOF management)

| MFEM method | fem-rs equivalent | Status |
|---|---|---|
| `FiniteElementSpace(mesh, fec)` | `H1Space::new(mesh)` etc. | 🔲 Phase 5 |
| `FES::GetNDofs()` | `FESpace::n_dofs()` | 🔲 Phase 5 |
| `FES::GetElementDofs()` | `FESpace::element_dofs()` | 🔲 Phase 5 |
| `FES::GetBdrElementDofs()` | DOF extractor for faces | 🔲 Phase 5 |
| `FES::GetEssentialTrueDofs()` | `dirichlet_nodes()` in examples | 🔨 Phase 5 |
| `FES::GetTrueDofs()` | — | 🔲 Phase 5 (parallel DOF ownership) |
| `FES::TransferToTrue()` / `Transfer()` | — | 🔲 Phase 5 |
| `DofTransformation` | Edge/face DOF sign flipping | 🔲 Phase 5 (Nédélec) |
| `FES::GetFE()` | `FESpace::element_type()` | 🔨 returns enum; full ref-elem object Phase 5 |

### 3.3 Space Types

| Space | Problem | Phase |
|---|---|---|
| H¹ | Electrostatics, heat, elasticity (scalar) | Phase 5 |
| H(curl) | Maxwell, eddy currents (vector potential) | Phase 5 |
| H(div) | Darcy flow, mixed Poisson | Phase 5 |
| L² / DG | Transport, DG methods | Phase 5 |
| Vector H¹ = [H¹]ᵈ | Elasticity (displacement vector) | Phase 5 |
| Taylor-Hood P2-P1 | Stokes flow | Phase 6 |

---

## 4. Coefficients

MFEM provides a rich coefficient hierarchy for spatially- and
time-varying material properties.  fem-rs uses closures `Fn(x,y)->T`.

| MFEM class | fem-rs | Status |
|---|---|---|
| `ConstantCoefficient(c)` | `|_,_| c` closure | ✅ (examples) |
| `FunctionCoefficient(f)` | `|x,y| f(x,y)` closure | ✅ (examples) |
| `GridFunctionCoefficient` | Evaluate `GridFunction` at quadrature points | 🔲 Phase 6 |
| `PWConstCoefficient` | Piecewise constant per element tag | 🔲 Phase 6 |
| `PWCoefficient` | Piecewise per-domain | 🔲 Phase 6 |
| `VectorCoefficient` | `Fn(&[f64]) -> [f64; D]` | 🔲 Phase 6 |
| `MatrixCoefficient` | `Fn(&[f64]) -> [[f64;D];D]` | 🔲 Phase 6 (anisotropic diffusion) |
| `InnerProductCoefficient` | Composed scalar product | 🔲 Phase 6 |
| `TransformedCoefficient` | Compose with a transform | 🔲 Phase 6 |

---

## 5. Assembly: Forms & Integrators

### 5.1 Bilinear Forms

| MFEM class | fem-rs equivalent | Status |
|---|---|---|
| `BilinearForm(fes)` | `Assembler::assemble_bilinear(integrators)` | 🔲 Phase 6 |
| `BilinearForm::AddDomainIntegrator()` | `assembler.add_domain(integrator)` | 🔲 Phase 6 |
| `BilinearForm::AddBoundaryIntegrator()` | `assembler.add_boundary(integrator)` | 🔲 Phase 6 |
| `BilinearForm::Assemble()` | `Assembler::assemble_bilinear()` | 🔲 Phase 6 |
| `BilinearForm::FormLinearSystem()` | `solve_dirichlet_reduced()` | 🔨 impl. in examples |
| `BilinearForm::FormSystemMatrix()` | `apply_dirichlet()` variants | 🔨 impl. in examples |
| `MixedBilinearForm(trial, test)` | `MixedAssembler` | 🔲 Phase 6+ |

### 5.2 Linear Forms

| MFEM class | fem-rs equivalent | Status |
|---|---|---|
| `LinearForm(fes)` | `Assembler::assemble_linear(integrators)` | 🔲 Phase 6 |
| `LinearForm::AddDomainIntegrator()` | `assembler.add_domain_load(integrator)` | 🔲 Phase 6 |
| `LinearForm::AddBndryIntegrator()` | Neumann load integrator | 🔲 Phase 6 |
| `LinearForm::Assemble()` | `Assembler::assemble_linear()` | 🔲 Phase 6 |

### 5.3 Bilinear Integrators

| MFEM integrator | Bilinear form | fem-rs struct | Status |
|---|---|---|---|
| `DiffusionIntegrator(κ)` | ∫ κ ∇u·∇v dx | `DiffusionIntegrator` | 🔲 Phase 6 |
| `MassIntegrator(ρ)` | ∫ ρ u v dx | `MassIntegrator` | 🔲 Phase 6 |
| `ConvectionIntegrator(b)` | ∫ (b·∇u) v dx | `ConvectionIntegrator` | 🔲 Phase 6 |
| `ElasticityIntegrator(λ,μ)` | ∫ σ(u):ε(v) dx | `ElasticityIntegrator` | 🔲 Phase 6 |
| `CurlCurlIntegrator(μ)` | ∫ μ (∇×u)·(∇×v) dx | `CurlCurlIntegrator` | 🔲 Phase 6 (ND) |
| `VectorFEMassIntegrator` | ∫ u·v dx (H(curl)/H(div)) | `VectorMassIntegrator` | 🔲 Phase 6 |
| `DivDivIntegrator(κ)` | ∫ κ (∇·u)(∇·v) dx | `DivDivIntegrator` | 🔲 Phase 6 (RT) |
| `VectorDiffusionIntegrator` | ∫ κ ∇uᵢ·∇vᵢ (vector Laplacian) | — | 🔲 Phase 6 |
| `BoundaryMassIntegrator` | ∫_Γ α u v ds | — | 🔲 Phase 6 |
| `VectorFEDivergenceIntegrator` | ∫ (∇·u) q dx (Darcy/Stokes) | — | 🔲 Phase 6 |
| `GradDivIntegrator` | ∫ (∇·u)(∇·v) dx | — | 🔲 Phase 6 |
| `DGDiffusionIntegrator` | Interior penalty DG diffusion | — | 🔲 Phase 6+ |
| `TransposeIntegrator` | Transposes a bilinear form | — | 🔲 Phase 6 |
| `SumIntegrator` | Sum of integrators | — | 🔲 Phase 6 |

### 5.4 Linear Integrators

| MFEM integrator | Linear form | fem-rs struct | Status |
|---|---|---|---|
| `DomainLFIntegrator(f)` | ∫ f v dx | `DomainSourceIntegrator` | 🔲 Phase 6 |
| `BoundaryLFIntegrator(g)` | ∫_Γ g v ds | `NeumannIntegrator` | 🔲 Phase 6 |
| `VectorDomainLFIntegrator` | ∫ **f**·**v** dx | — | 🔲 Phase 6 |
| `BoundaryNormalLFIntegrator` | ∫_Γ g (n·v) ds | — | 🔲 Phase 6 |
| `VectorFEBoundaryFluxLFIntegrator` | ∫_Γ f (v·n) ds (RT) | — | 🔲 Phase 6 |

### 5.5 Assembly Pipeline

| MFEM concept | fem-rs equivalent | Status |
|---|---|---|
| `ElementTransformation` | Jacobian `jac`, `det_j`, `jac_inv_t` | 🔨 inline in examples |
| `Geometry::Type` | `ElementType` enum | ✅ |
| Sparsity pattern | `SparsityPattern` built once | 🔲 Phase 6 |
| Parallel assembly | Element loop → ghost DOF AllReduce | 🔲 Phase 10 |

---

## 6. Linear Algebra

### 6.1 Sparse Matrix

| MFEM class | fem-rs struct | Status |
|---|---|---|
| `SparseMatrix` (CSR) | `CsrMatrix<T>` | ✅ |
| `SparseMatrix::Add(i,j,v)` | `CooMatrix::add(i,j,v)` | ✅ |
| `SparseMatrix::Finalize()` | `CooMatrix::into_csr()` | ✅ |
| `SparseMatrix::Mult(x,y)` | `CsrMatrix::spmv(x,y)` | ✅ |
| `SparseMatrix::MultTranspose()` | `CsrMatrix::spmv_add()` or transpose | 🔲 Phase 4 |
| `SparseMatrix::EliminateRowCol()` | `apply_dirichlet_symmetric()` | ✅ |
| `SparseMatrix::EliminateRow()` | `apply_dirichlet_row_zeroing()` | ✅ |
| `SparseMatrix::GetDiag()` | `CsrMatrix::diagonal()` | ✅ |
| `SparseMatrix::Transpose()` | `CsrMatrix::transpose()` | 🔲 Phase 4 |
| `SparseMatrix::Add(A,B)` | — | 🔲 Phase 4 |
| `SparseMatrix::Mult(A,B)` | SpGEMM | 🔲 Phase 4 (AMG needs this) |
| `DenseMatrix` (local dense) | `nalgebra::SMatrix` | ✅ |
| `DenseTensor` | nested matrices | 🔲 Phase 6 |

### 6.2 Vector

| MFEM class | fem-rs struct | Status |
|---|---|---|
| `Vector` | `Vector<T>` | ✅ |
| `Vector::operator +=` | `Vector::axpy(1.0, x)` | ✅ |
| `Vector::operator *=` | `Vector::scale(a)` | ✅ |
| `Vector::operator * (dot)` | `Vector::dot()` | ✅ |
| `Vector::Norml2()` | `Vector::norm()` | ✅ |
| `Vector::Neg()` | `vector.scale(-1.0)` | ✅ |
| `Vector::SetSubVector()` | index slice assignment | 🔲 Phase 4 |
| `BlockVector` | block-partitioned vector | 🔲 Phase 6 (mixed methods) |

---

## 7. Solvers & Preconditioners

### 7.1 Iterative Solvers

| MFEM solver | Problem type | fem-rs module | Status |
|---|---|---|---|
| `CGSolver` | SPD: A x = b | `solver/src/cg.rs` | 🔲 Phase 7 |
| `PCGSolver` | SPD + preconditioner | `solver/src/pcg.rs` | 🔲 Phase 7 |
| `GMRESSolver(m)` | General: A x = b | `solver/src/gmres.rs` | 🔲 Phase 7 |
| `FGMRESSolver` | Flexible GMRES | — | 🔲 Phase 7+ |
| `BiCGSTABSolver` | Non-symmetric | — | 🔲 Phase 7+ |
| `MINRESSolver` | Indefinite symmetric | — | 🔲 Phase 7+ |
| `SLISolver` | Stationary linear iteration | — | 🔲 Phase 7 |
| `NewtonSolver` | Nonlinear F(x)=0 | — | 🔲 Phase 7+ |
| `UMFPackSolver` | Direct (SuiteSparse) | `solver/src/direct.rs` | 🔲 Phase 7 (small systems) |
| `MUMPSSolver` | Parallel direct | — | ❌ |

### 7.2 Preconditioners

| MFEM preconditioner | Type | fem-rs module | Status |
|---|---|---|---|
| `DSmoother` | Jacobi / diagonal scaling | `precond/jacobi.rs` | 🔲 Phase 7 |
| `GSSmoother` | Gauss-Seidel | `SmootherKind::GaussSeidel` | 🔲 Phase 7 |
| Chebyshev smoother | Chebyshev polynomial | `SmootherKind::ChebyshevJacobi` | 🔲 Phase 7 |
| `SparseSmoothedProjection` | ILU-based | `precond/ilu0.rs` | 🔲 Phase 7 |
| `BlockDiagonalPreconditioner` | Block Jacobi | — | 🔲 Phase 7 |
| `BlockTriangularPreconditioner` | Block triangular | — | 🔲 Phase 7+ |
| `SchurComplement` | Elimination for saddle point | — | 🔲 Phase 7+ |

### 7.3 Solver Convergence Monitors

| MFEM concept | fem-rs equivalent | Status |
|---|---|---|
| `IterativeSolver::SetTol()` | `tol` parameter | ✅ (examples) |
| `IterativeSolver::SetMaxIter()` | `max_iter` parameter | ✅ (examples) |
| `IterativeSolver::GetFinalNorm()` | `SolverResult::residual_norm` | 🔲 Phase 7 |
| `IterativeSolver::GetNumIterations()` | `SolverResult::iterations` | 🔲 Phase 7 |
| `IterativeSolver::SetPrintLevel()` | `log::info!` integration | 🔲 Phase 7 |

---

## 8. Algebraic Multigrid

| MFEM / hypre concept | fem-rs equivalent | Status |
|---|---|---|
| `HypreBoomerAMG` (setup) | `AmgSolver::setup(mat)` → `AmgHierarchy` | 🔲 Phase 8 |
| `HypreBoomerAMG` (solve) | `AmgSolver::solve(hierarchy, rhs)` | 🔲 Phase 8 |
| Strength of connection θ | `AmgParams::theta` (default 0.25) | 🔲 Phase 8 |
| Ruge-Stüben C/F splitting | SA-AMG aggregation (preferred) | 🔲 Phase 8 |
| Smoothed aggregation | `aggregation.rs` MIS-based | 🔲 Phase 8 |
| Prolongation P | `AmgLevel::p: CsrMatrix` | 🔲 Phase 8 |
| Restriction R = Pᵀ | `AmgLevel::r: CsrMatrix` | 🔲 Phase 8 |
| Galerkin coarse A_c = R A P | SpGEMM chain | 🔲 Phase 8 |
| Pre-smoother (ω-Jacobi) | `SmootherKind::Jacobi { omega: 0.67 }` | 🔲 Phase 8 |
| Post-smoother | `AmgParams::post_smooth` steps | 🔲 Phase 8 |
| V-cycle | `CycleType::V` | 🔲 Phase 8 |
| W-cycle | `CycleType::W` | 🔲 Phase 8 |
| F-cycle | `CycleType::F` | 🔲 Phase 8 |
| Max levels | `AmgParams::max_levels` (25) | 🔲 Phase 8 |
| Coarse-grid direct solve | `solver::direct::lu_solve` | 🔲 Phase 8 |
| hypre binding | feature `amg/hypre` | 🔲 Phase 8+ |

---

## 9. Parallel Infrastructure

### 9.1 MPI Communicators

| MFEM concept | fem-rs module | Status |
|---|---|---|
| `MPI_Comm` | `parallel::comm::Communicator` | 🔲 Phase 10 |
| `MPI_Allreduce` | `comm::allreduce()` | 🔲 Phase 10 |
| `MPI_Allgather` | `comm::allgather()` | 🔲 Phase 10 |
| `MPI_Send/Recv` | point-to-point DOF exchange | 🔲 Phase 10 |

### 9.2 Distributed Mesh

| MFEM class | fem-rs struct | Status |
|---|---|---|
| `ParMesh` | `ParallelMesh<M>` | 🔲 Phase 10 |
| METIS partitioning | `partition::metis_partition()` | 🔲 Phase 10 |
| Ghost elements | `ParallelMesh::shared_nodes` | 🔲 Phase 10 |
| Global-to-local node map | `node_global_ids`, `node_tag_to_id` | 🔲 Phase 10 |

### 9.3 Parallel Linear Algebra

| MFEM / hypre class | fem-rs struct | Status |
|---|---|---|
| `HypreParMatrix` | `ParCsrMatrix` (diag + off-diag blocks) | 🔲 Phase 10 |
| `HypreParVector` | `ParVector` | 🔲 Phase 10 |
| `HypreParMatrix::Mult()` | `ParCsrMatrix::par_spmv()` | 🔲 Phase 10 |
| `HypreParMatrix::GetDiag()` | `ParCsrMatrix::diag` | 🔲 Phase 10 |
| `HypreParMatrix::GetOffd()` | `ParCsrMatrix::off_diag` | 🔲 Phase 10 |

---

## 10. I/O & Visualization

### 10.1 Mesh I/O

| MFEM format / method | fem-rs | Status |
|---|---|---|
| MFEM native mesh format (read/write) | — | ❌ use GMSH |
| GMSH `.msh` v2 (read) | — | 🔲 Phase 9+ |
| GMSH `.msh` v4.1 ASCII (read) | `fem_io::read_msh_file()` | ✅ |
| GMSH `.msh` v4.1 binary (read) | — | 🔲 Phase 9 |
| Netgen `.vol` (read) | — | 🔲 Phase 9+ |
| Abaqus `.inp` (read) | — | 🔲 Phase 9+ |
| VTK `.vtu` legacy ASCII (write) | `write_vtk_scalar()` | ✅ (examples) |
| VTK `.vtu` XML binary (write) | — | 🔲 Phase 9 |
| HDF5 / XDMF (read/write) | feature `io/hdf5` | 🔲 Phase 9 |
| ParaView GLVis socket | — | ❌ out of scope |

### 10.2 Solution I/O

| MFEM concept | fem-rs | Status |
|---|---|---|
| `GridFunction::Save()` | VTK point data | 🔨 scalar only |
| `GridFunction::Load()` | — | 🔲 Phase 9 |
| Restart files | HDF5 mesh + solution | 🔲 Phase 9 |

---

## 11. Grid Functions & Post-processing

| MFEM class / method | fem-rs equivalent | Status |
|---|---|---|
| `GridFunction(fes)` | `Vec<f64>` (nodal DOF vector) | 🔨 no type yet |
| `GridFunction::ProjectCoefficient()` | `FESpace::interpolate(f)` | 🔲 Phase 5 |
| `GridFunction::ComputeL2Error()` | `l2_error_p1()` | ✅ (examples) |
| `GridFunction::ComputeH1Error()` | — | 🔲 Phase 6 |
| `GridFunction::GetGradient()` | `p1_gradient_2d()` | ✅ (examples) |
| `GridFunction::GetCurl()` | — | 🔲 Phase 6 (ND) |
| `GridFunction::GetDivergence()` | — | 🔲 Phase 6 (RT) |
| `ZZErrorEstimator` (Zienkiewicz-Zhu) | — | 🔲 Phase 2+ |
| `KellyErrorEstimator` | — | 🔲 Phase 2+ |
| `DiscreteLinearOperator` | Gradient, curl, div operators | 🔲 Phase 6 |

---

## 12. MFEM Examples → fem-rs Milestones

Each MFEM example defines a target milestone for fem-rs feature completeness.

### Tier 1 — Core Capability (Phases 6–7)

| MFEM example | PDE | FEM space | BCs | fem-rs milestone |
|---|---|---|---|---|
| **ex1** | −∇²u = 1, u=0 on ∂Ω | H¹ P1/P2 | Dirichlet | Phase 6 smoke test |
| **ex2** | −∇²u = f, mixed BCs | H¹ P1/P2 | Dirichlet + Neumann | Phase 6: `NeumannIntegrator` |
| **ex3** (scalar) | −∇²u + αu = f (reaction-diffusion) | H¹ P1 | Dirichlet | Phase 6: `MassIntegrator` |
| **ex13** | −∇·(ε∇φ) = 0, elasticity | H¹ vector | Mixed | Phase 6: `ElasticityIntegrator` |
| **pex1** | Parallel Poisson | H¹ + MPI | Dirichlet | Phase 10 |

### Tier 2 — Mixed & H(curl)/H(div) (Phase 6+)

| MFEM example | PDE | FEM space | fem-rs milestone |
|---|---|---|---|
| **ex3** (curl) | ∇×∇×**u** + **u** = **f** (Maxwell) | H(curl) Nédélec | Phase 6: `CurlCurlIntegrator` + `ND` space |
| **ex4** | −∇·(**u**) = f, **u** = −κ∇p (Darcy) | H(div) RT + L² | Phase 6: RT space + `DivDivIntegrator` |
| **ex5** | Saddle-point Darcy/Stokes | H(div) × L² | Phase 6: `MixedBilinearForm` |
| **ex22** | Time-harmonic Maxwell (complex coeff.) | H(curl) | Phase 7+ |
| **em_magnetostatics_2d** (this project) | −∇·(ν∇Az) = Jz | H¹ P1 (2D A_z) | ✅ |

### Tier 3 — Time Integration (Phase 7+)

| MFEM example | PDE | Time method | fem-rs milestone |
|---|---|---|---|
| **ex9** (heat) | ∂u/∂t − ∇²u = 0 | BDF1 / Crank-Nicolson | Phase 7: `TimeIntegrator` trait |
| **ex10** (wave) | ∂²u/∂t² − ∇²u = 0 | Leapfrog / Newmark | Phase 7+ |
| **ex14** (DG heat) | ∂u/∂t − ∇²u + b·∇u = 0 | Explicit RK + DG | Phase 7+ |
| **ex16** (elastodynamics) | ρ ∂²**u**/∂t² = ∇·σ | Generalized-α | Phase 7+ |

### Tier 4 — Nonlinear & AMR (Phase 7+)

| MFEM example | Problem | fem-rs milestone |
|---|---|---|
| **ex4** (nonlinear) | −Δu + exp(u) = 0 | Phase 7+: Newton solver |
| **ex6** | AMR Poisson with ZZ estimator | Phase 2+: `refine.rs`, `ZZErrorEstimator` |
| **ex15** | DG advection with AMR | Phase 6+ |
| **ex19** | Incompressible Navier-Stokes | Phase 7+ |

### Tier 5 — HPC & Parallel (Phase 10)

| MFEM example | Problem | fem-rs milestone |
|---|---|---|
| **pex1** | Parallel Poisson (Poisson) | Phase 10: `ParallelMesh` + `ParCsrMatrix` |
| **pex2** | Parallel mixed Poisson | Phase 10 |
| **pex3** | Parallel Maxwell (H(curl)) | Phase 10 |
| **pex5** | Parallel Darcy | Phase 10 |

---

## 13. Key Design Differences

| Aspect | MFEM (C++) | fem-rs (Rust) | Rationale |
|---|---|---|---|
| **Polymorphism** | Virtual classes + inheritance | Traits + generics (zero-cost) | No vtable overhead in inner loop |
| **Index types** | `int` (32-bit signed) | `NodeId = u32` etc. | Half memory; explicit casting |
| **Parallel model** | Always-on `ParMesh`; MPI implicit | Feature-gated `fem-parallel` crate | Same binary works without MPI |
| **Web target** | emscripten (experimental) | `fem-wasm` crate (wasm-bindgen) | First-class JS interop |
| **AMG default** | Ruge-Stüben (classical) | Smoothed Aggregation | Better performance on vector problems |
| **Quadrature** | Hard-coded tables | Generated tables in `quadrature.rs` | Reproducible, testable |
| **Coefficient API** | Polymorphic `Coefficient*` objects | Closures `Fn(f64,f64)->f64` | Simpler ownership semantics |
| **Memory layout** | Column-major `DenseMatrix` | Row-major element buffers; nalgebra for Jacobians | Cache-friendly assembly |
| **Error handling** | Exceptions / abort | `FemResult<T>` everywhere | Propagate, never panic in library |
| **BC application** | `FormLinearSystem()` (symmetric elim.) | `solve_dirichlet_reduced()` (reduced system) | Avoids scale artefacts with small ε |
| **Grid function** | `GridFunction` owns DOF vector + FES ref | `Vec<f64>` + separate `FESpace` ref | Separation of concerns |

---

## Quick Reference: Phase → Features

| Phase | Crates | MFEM equivalents unlocked |
|---|---|---|
| 0 | workspace | — |
| 1 | `core` | Index types, `FemError`, scalar traits |
| 2 | `mesh` | `Mesh`, element types, mesh generators, refinement |
| 3 | `element` | `FiniteElement`, `IntegrationRule`, Lagrange P1–P3 |
| 4 | `linalg` | `SparseMatrix`, `Vector`, COO→CSR assembly |
| 5 | `space` | `FiniteElementSpace`, H1/L2/ND/RT, DOF manager |
| 6 | `assembly` | `BilinearForm`, `LinearForm`, all standard integrators |
| 7 | `solver` | `CGSolver`, `GMRESSolver`, ILU(0), direct |
| 8 | `amg` | `BoomerAMG` (native SA-AMG) |
| 9 | `io` | VTK XML, HDF5, Netgen import |
| 10 | `parallel` | `ParMesh`, `HypreParMatrix`, parallel PCG |
| 11 | `wasm` | Browser-side FEM solver via JS API |
