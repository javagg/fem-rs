//! Matrix Market (.mtx) file I/O — thin wrapper around linger's mmio module.
//!
//! Supports reading/writing sparse matrices in the standard Matrix Market
//! coordinate format used by SuiteSparse and NIST.
//!
//! # Example
//! ```rust,ignore
//! use fem_io::matrix_market::{read_matrix_market, write_matrix_market};
//!
//! // Read a matrix
//! let a = read_matrix_market("stiffness.mtx").unwrap();
//!
//! // Write a matrix
//! write_matrix_market("output.mtx", &a).unwrap();
//! ```

use std::path::Path;

use fem_linalg::{CooMatrix, CsrMatrix};
use linger::sparse::{
    CooMatrix as LingerCoo, CsrMatrix as LingerCsr,
    read_matrix_market as linger_read, write_matrix_market as linger_write,
};

pub use linger::MmioError;

// ─── Conversion helpers ───────────────────────────────────────────────────────

fn linger_csr_to_fem(lc: LingerCsr<f64>) -> CsrMatrix<f64> {
    CsrMatrix {
        nrows:   lc.nrows(),
        ncols:   lc.ncols(),
        row_ptr: lc.row_ptr().to_vec(),
        col_idx: lc.col_idx().iter().map(|&c| c as u32).collect(),
        values:  lc.values().to_vec(),
    }
}

fn fem_csr_to_linger(a: &CsrMatrix<f64>) -> LingerCsr<f64> {
    LingerCsr::from_raw(
        a.nrows,
        a.ncols,
        a.row_ptr.clone(),
        a.col_idx.iter().map(|&c| c as usize).collect(),
        a.values.clone(),
    )
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Read a Matrix Market `.mtx` file into a `CsrMatrix<f64>`.
///
/// Supports `real general`, `real symmetric`, `integer general`, and `pattern` variants.
pub fn read_matrix_market<P: AsRef<Path>>(path: P) -> Result<CsrMatrix<f64>, MmioError> {
    let lc: LingerCsr<f64> = linger_read(path)?;
    Ok(linger_csr_to_fem(lc))
}

/// Read a Matrix Market `.mtx` file into a `CooMatrix<f64>`.
///
/// Preserves duplicate entries as separate (row, col, val) triplets.
pub fn read_matrix_market_coo<P: AsRef<Path>>(path: P) -> Result<CooMatrix<f64>, MmioError> {
    use linger::sparse::read_matrix_market_coo as linger_read_coo;
    let lc: LingerCoo<f64> = linger_read_coo(path)?;
    let mut coo = CooMatrix::<f64>::new(lc.nrows(), lc.ncols());
    for ((r, c), v) in lc.row_indices().iter().zip(lc.col_indices()).zip(lc.values()) {
        coo.add(*r, *c, *v);
    }
    Ok(coo)
}

/// Write a `CsrMatrix<f64>` to a Matrix Market `.mtx` file.
///
/// Writes in `%%MatrixMarket matrix coordinate real general` format.
pub fn write_matrix_market<P: AsRef<Path>>(path: P, a: &CsrMatrix<f64>) -> Result<(), MmioError> {
    let lc = fem_csr_to_linger(a);
    linger_write(path, &lc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn laplacian_1d(n: usize) -> CsrMatrix<f64> {
        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n {
            coo.add(i, i, 2.0);
            if i > 0     { coo.add(i, i - 1, -1.0); }
            if i < n - 1 { coo.add(i, i + 1, -1.0); }
        }
        coo.into_csr()
    }

    #[test]
    fn roundtrip_matrix_market() {
        let a = laplacian_1d(5);
        let tmp = NamedTempFile::new().unwrap();
        write_matrix_market(tmp.path(), &a).unwrap();
        let b = read_matrix_market(tmp.path()).unwrap();

        assert_eq!(b.nrows, a.nrows);
        assert_eq!(b.ncols, a.ncols);
        // Check all values match
        for i in 0..a.nrows {
            for p in a.row_ptr[i]..a.row_ptr[i + 1] {
                let j = a.col_idx[p] as usize;
                let val_a = a.values[p];
                // Find same (i, j) in b
                let val_b = (b.row_ptr[i]..b.row_ptr[i + 1])
                    .find(|&q| b.col_idx[q] as usize == j)
                    .map(|q| b.values[q])
                    .expect("entry missing in read-back matrix");
                assert!((val_a - val_b).abs() < 1e-14, "value mismatch at ({i},{j})");
            }
        }
    }

    #[test]
    fn read_mtx_from_string() {
        let mtx = b"%%MatrixMarket matrix coordinate real general\n3 3 4\n1 1 4.0\n1 2 1.0\n2 1 1.0\n3 3 9.0\n";
        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(mtx).unwrap();
        tmp.flush().unwrap();
        let a = read_matrix_market(tmp.path()).unwrap();
        assert_eq!(a.nrows, 3);
        assert_eq!(a.ncols, 3);
    }
}
