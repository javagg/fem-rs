use fem_core::Scalar;

/// Compressed Sparse Row matrix.
///
/// - `row_ptr[i]` = index into `col_idx`/`values` of the first entry in row `i`.
/// - `row_ptr[nrows]` = total number of stored non-zeros.
/// - `col_idx` and `values` are aligned: `values[k]` lives at column `col_idx[k]`.
#[derive(Debug, Clone)]
pub struct CsrMatrix<T> {
    pub nrows:   usize,
    pub ncols:   usize,
    pub row_ptr: Vec<usize>,   // length nrows + 1
    pub col_idx: Vec<u32>,
    pub values:  Vec<T>,
}

impl<T: Scalar> CsrMatrix<T> {
    /// Empty matrix with pre-allocated structure arrays.
    pub fn new_empty(nrows: usize, ncols: usize) -> Self {
        Self {
            nrows,
            ncols,
            row_ptr: vec![0; nrows + 1],
            col_idx: Vec::new(),
            values:  Vec::new(),
        }
    }

    /// Number of stored non-zeros.
    pub fn nnz(&self) -> usize { self.values.len() }

    /// Get value at `(row, col)`.  Returns 0 if not stored.
    pub fn get(&self, row: usize, col: usize) -> T {
        let start = self.row_ptr[row];
        let end   = self.row_ptr[row + 1];
        for k in start..end {
            if self.col_idx[k] as usize == col {
                return self.values[k];
            }
        }
        T::zero()
    }

    /// Mutable reference to value at `(row, col)`.
    /// Panics if the entry is not present in the sparsity pattern.
    pub fn get_mut(&mut self, row: usize, col: usize) -> &mut T {
        let start = self.row_ptr[row];
        let end   = self.row_ptr[row + 1];
        for k in start..end {
            if self.col_idx[k] as usize == col {
                return &mut self.values[k];
            }
        }
        panic!("CsrMatrix::get_mut: ({row},{col}) not in sparsity pattern");
    }

    // -----------------------------------------------------------------------
    // Matrix-vector products
    // -----------------------------------------------------------------------

    /// Compute `y = A x`.
    pub fn spmv(&self, x: &[T], y: &mut [T]) {
        assert_eq!(x.len(), self.ncols);
        assert_eq!(y.len(), self.nrows);
        for (row, yi) in y.iter_mut().enumerate() {
            let start = self.row_ptr[row];
            let end   = self.row_ptr[row + 1];
            let mut s = T::zero();
            for k in start..end {
                s += self.values[k] * x[self.col_idx[k] as usize];
            }
            *yi = s;
        }
    }

    /// Compute `y = α A x + β y`.
    pub fn spmv_add(&self, alpha: T, x: &[T], beta: T, y: &mut [T]) {
        assert_eq!(x.len(), self.ncols);
        assert_eq!(y.len(), self.nrows);
        for (row, yi) in y.iter_mut().enumerate() {
            let start = self.row_ptr[row];
            let end   = self.row_ptr[row + 1];
            let mut s = T::zero();
            for k in start..end {
                s += self.values[k] * x[self.col_idx[k] as usize];
            }
            *yi = alpha * s + beta * *yi;
        }
    }

    /// Diagonal vector `d[i] = A[i,i]`.
    pub fn diagonal(&self) -> Vec<T> {
        let mut d = vec![T::zero(); self.nrows];
        for row in 0..self.nrows {
            d[row] = self.get(row, row);
        }
        d
    }

    // -----------------------------------------------------------------------
    // Boundary condition helpers
    // -----------------------------------------------------------------------

    /// Apply a Dirichlet BC for DOF `row`:
    /// - Zero the entire row.
    /// - Set the diagonal to 1.
    /// - Modify `rhs[row] = prescribed_value`.
    ///
    /// Also subtracts the column contribution from other rows to maintain
    /// symmetry (the "symmetric elimination" approach).
    pub fn apply_dirichlet_symmetric(
        &mut self,
        row: usize,
        value: T,
        rhs: &mut [T],
    ) {
        // Subtract column `row` contributions from other rows
        for other_row in 0..self.nrows {
            if other_row == row { continue; }
            let a_ij = self.get(other_row, row);
            if a_ij == T::zero() { continue; }
            rhs[other_row] -= a_ij * value;
            // Zero the off-diagonal entry (other_row, row)
            if let Some(k) = self.find_entry(other_row, row) {
                self.values[k] = T::zero();
            }
        }
        // Zero the entire row, then set diagonal
        let start = self.row_ptr[row];
        let end   = self.row_ptr[row + 1];
        for k in start..end {
            self.values[k] = T::zero();
        }
        if let Some(k) = self.find_entry(row, row) {
            self.values[k] = T::one();
        }
        rhs[row] = value;
    }

    /// Apply Dirichlet BC (row-zeroing only, not symmetric).
    ///
    /// Faster than symmetric elimination; use when symmetry is not required.
    pub fn apply_dirichlet_row_zeroing(
        &mut self,
        row: usize,
        value: T,
        rhs: &mut [T],
    ) {
        let start = self.row_ptr[row];
        let end   = self.row_ptr[row + 1];
        for k in start..end {
            self.values[k] = T::zero();
        }
        if let Some(k) = self.find_entry(row, row) {
            self.values[k] = T::one();
        }
        rhs[row] = value;
    }

    fn find_entry(&self, row: usize, col: usize) -> Option<usize> {
        let start = self.row_ptr[row];
        let end   = self.row_ptr[row + 1];
        for k in start..end {
            if self.col_idx[k] as usize == col { return Some(k); }
        }
        None
    }

    // -----------------------------------------------------------------------
    // Debug
    // -----------------------------------------------------------------------

    /// Convert to a dense `nrows × ncols` row-major matrix (testing only).
    pub fn to_dense(&self) -> Vec<T> {
        let mut d = vec![T::zero(); self.nrows * self.ncols];
        for row in 0..self.nrows {
            let start = self.row_ptr[row];
            let end   = self.row_ptr[row + 1];
            for k in start..end {
                d[row * self.ncols + self.col_idx[k] as usize] = self.values[k];
            }
        }
        d
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coo::CooMatrix;

    fn small_matrix() -> CsrMatrix<f64> {
        // [ 2 -1  0 ]
        // [-1  2 -1 ]
        // [ 0 -1  2 ]
        let mut c = CooMatrix::<f64>::new(3, 3);
        c.add(0, 0,  2.0); c.add(0, 1, -1.0);
        c.add(1, 0, -1.0); c.add(1, 1,  2.0); c.add(1, 2, -1.0);
        c.add(2, 1, -1.0); c.add(2, 2,  2.0);
        c.into_csr()
    }

    #[test]
    fn spmv_tridiag() {
        let a = small_matrix();
        let x = vec![1.0f64, 2.0, 3.0];
        let mut y = vec![0.0f64; 3];
        a.spmv(&x, &mut y);
        // [2-2, -1+4-3, -2+6] = [0, 0, 4]
        assert!((y[0]).abs() < 1e-14);
        assert!((y[1]).abs() < 1e-14);
        assert!((y[2] - 4.0).abs() < 1e-14);
    }

    #[test]
    fn diagonal() {
        let a = small_matrix();
        let d = a.diagonal();
        assert_eq!(d, vec![2.0, 2.0, 2.0]);
    }

    #[test]
    fn dirichlet_row_zeroing() {
        let mut a = small_matrix();
        let mut rhs = vec![1.0f64, 2.0, 3.0];
        a.apply_dirichlet_row_zeroing(0, 5.0, &mut rhs);
        assert!((a.get(0, 1)).abs() < 1e-14, "off-diag should be zero");
        assert!((a.get(0, 0) - 1.0).abs() < 1e-14, "diagonal should be 1");
        assert!((rhs[0] - 5.0).abs() < 1e-14);
    }
}
