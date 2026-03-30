//! # fem-parallel
//!
//! MPI-parallel mesh distribution, parallel assembly, and parallel AMG.
//! Requires the `mpi` feature flag; never enable for wasm32 targets.
//!
//! **Status**: stub — Phase 10 not yet implemented.

#[cfg(target_arch = "wasm32")]
compile_error!("`fem-parallel` must not be compiled for the wasm32 target");
