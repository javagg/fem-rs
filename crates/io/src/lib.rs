//! # fem-io
//!
//! Mesh and solution I/O for fem-rs.
//!
//! ## Modules
//! - [`gmsh`] — GMSH `.msh` v4.1 ASCII reader → `SimplexMesh`

pub mod gmsh;

pub use gmsh::{read_msh, read_msh_file, MshFile};
