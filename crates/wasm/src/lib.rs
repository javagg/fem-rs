//! # fem-wasm
//!
//! WebAssembly bindings for fem-rs. Exposes a JS-friendly `WasmSolver` API
//! via `wasm-bindgen`. Never links `fem-parallel` or any MPI dependency.
//!
//! Build: `cargo wasm-build`
//!
//! **Status**: stub — Phase 11 not yet implemented.

#[cfg(target_arch = "wasm32")]
pub fn _wasm_placeholder() {}
