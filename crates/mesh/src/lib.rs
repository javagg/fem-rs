//! # fem-mesh
//!
//! Mesh topology and geometry for fem-rs.
//!
//! ## Modules
//! - [`element_type`] — `ElementType` enum (Tri3, Tet4, Hex8, …)
//! - [`boundary`]     — `BoundaryTag` and `PhysicalGroup`
//! - [`topology`]     — `MeshTopology` trait
//! - [`simplex`]      — `SimplexMesh<D>`: concrete unstructured mesh with built-in generators

pub mod boundary;
pub mod element_type;
pub mod simplex;
pub mod topology;

pub use boundary::{BoundaryTag, PhysicalGroup};
pub use element_type::ElementType;
pub use simplex::SimplexMesh;
pub use topology::MeshTopology;
