//! # fem-linalg
//!
//! Sparse and dense linear algebra for fem-rs.
//!
//! ## Modules
//! - [`csr`]    — `CsrMatrix<T>`: CSR sparse matrix with SpMV and BC helpers
//! - [`coo`]    — `CooMatrix<T>`: coordinate-format accumulator → converts to CSR
//! - [`vector`] — `Vector<T>`: heap vector with axpy, dot, norm

pub mod coo;
pub mod csr;
pub mod vector;

pub use coo::CooMatrix;
pub use csr::CsrMatrix;
pub use vector::Vector;
