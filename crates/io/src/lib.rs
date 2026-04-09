//! # fem-io
//!
//! Mesh and solution I/O for fem-rs.
//!
//! ## Modules
//! - [`gmsh`]           — GMSH `.msh` v4.1 ASCII/binary reader → `SimplexMesh`
//! - [`vtk`]            — VTK UnstructuredGrid `.vtu` XML writer
//! - [`vtk_reader`]     — VTK `.vtu` XML reader (point data arrays)
//! - [`matrix_market`]  — Matrix Market `.mtx` reader/writer

pub mod gmsh;
pub mod vtk;
pub mod vtk_reader;
pub mod matrix_market;

pub use gmsh::{read_msh, read_msh_file, MshFile};
pub use vtk::{DataArray, VtkWriter};
pub use vtk_reader::read_vtu_point_data;
pub use matrix_market::{read_matrix_market, read_matrix_market_coo, write_matrix_market, MmioError};
