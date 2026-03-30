use fem_core::Scalar;
use crate::csr::CsrMatrix;

/// Coordinate-format sparse matrix (accumulates (row, col, value) triples).
///
/// Used during FEM assembly to collect element contributions before converting
/// to CSR with `into_csr()`.  Duplicate `(row, col)` entries are summed.
#[derive(Debug, Clone)]
pub struct CooMatrix<T> {
    pub nrows: usize,
    pub ncols: usize,
    rows: Vec<u32>,
    cols: Vec<u32>,
    vals: Vec<T>,
}

impl<T: Scalar> CooMatrix<T> {
    /// Create an empty sparse matrix of shape `nrows × ncols`.
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self { nrows, ncols, rows: Vec::new(), cols: Vec::new(), vals: Vec::new() }
    }

    /// Reserve capacity for `n` non-zeros (performance hint).
    pub fn reserve(&mut self, n: usize) {
        self.rows.reserve(n);
        self.cols.reserve(n);
        self.vals.reserve(n);
    }

    /// Add a scalar contribution `val` at position `(row, col)`.
    #[inline]
    pub fn add(&mut self, row: usize, col: usize, val: T) {
        self.rows.push(row as u32);
        self.cols.push(col as u32);
        self.vals.push(val);
    }

    /// Add a dense `k × k` element matrix at the DOF index pairs in `dofs`.
    ///
    /// `k_elem` is row-major: `k_elem[i * k + j]` is the (i,j) entry.
    pub fn add_element_matrix(&mut self, dofs: &[usize], k_elem: &[T]) {
        let k = dofs.len();
        debug_assert_eq!(k_elem.len(), k * k);
        for i in 0..k {
            for j in 0..k {
                self.add(dofs[i], dofs[j], k_elem[i * k + j]);
            }
        }
    }

    /// Add a dense element load vector `f_elem` (length `k`) to positions `dofs`.
    pub fn add_element_vec_to_rhs(&self, dofs: &[usize], f_elem: &[T], rhs: &mut [T]) {
        debug_assert_eq!(dofs.len(), f_elem.len());
        for (&d, &v) in dofs.iter().zip(f_elem.iter()) {
            rhs[d] += v;
        }
    }

    /// Number of stored triplets (before deduplication).
    pub fn nnz_raw(&self) -> usize { self.vals.len() }

    /// Convert to CSR, summing duplicate entries.
    ///
    /// Sort by (row, col), then merge duplicates.
    pub fn into_csr(mut self) -> CsrMatrix<T> {
        let nnz = self.vals.len();
        if nnz == 0 {
            return CsrMatrix::new_empty(self.nrows, self.ncols);
        }

        // Sort by (row, col)
        let mut idx: Vec<usize> = (0..nnz).collect();
        idx.sort_unstable_by_key(|&i| (self.rows[i], self.cols[i]));

        let mut row_ptr = vec![0usize; self.nrows + 1];
        let mut col_idx: Vec<u32> = Vec::with_capacity(nnz);
        let mut values: Vec<T>    = Vec::with_capacity(nnz);

        let (mut prev_row, mut prev_col) = (u32::MAX, u32::MAX);

        for &i in &idx {
            let r = self.rows[i];
            let c = self.cols[i];
            let v = self.vals[i];
            if r == prev_row && c == prev_col {
                *values.last_mut().unwrap() += v;
            } else {
                // Fill row_ptr for any skipped rows
                if r != prev_row {
                    for rr in (prev_row as usize + 1)..=(r as usize) {
                        row_ptr[rr] = col_idx.len();
                    }
                }
                col_idx.push(c);
                values.push(v);
                prev_row = r;
                prev_col = c;
            }
        }
        // Fill remaining rows
        let last_row = self.rows[*idx.last().unwrap()] as usize;
        for rr in (last_row + 1)..=(self.nrows) {
            row_ptr[rr] = col_idx.len();
        }

        // Clear to free memory
        self.rows.clear();
        self.cols.clear();
        self.vals.clear();

        CsrMatrix { nrows: self.nrows, ncols: self.ncols, row_ptr, col_idx, values }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coo_to_csr_sum_duplicates() {
        // 3×3 identity via duplicate entries
        let mut coo = CooMatrix::<f64>::new(3, 3);
        for i in 0..3 {
            coo.add(i, i, 0.5);
            coo.add(i, i, 0.5);
        }
        let csr = coo.into_csr();
        assert_eq!(csr.nrows, 3);
        assert_eq!(csr.values.len(), 3); // 3 unique (i,i)
        for i in 0..3 {
            assert!((csr.get(i, i) - 1.0).abs() < 1e-14);
        }
    }

    #[test]
    fn element_matrix_add() {
        let mut coo = CooMatrix::<f64>::new(2, 2);
        let k = [1.0, -1.0, -1.0, 1.0];
        coo.add_element_matrix(&[0, 1], &k);
        let csr = coo.into_csr();
        assert!((csr.get(0, 0) - 1.0).abs() < 1e-14);
        assert!((csr.get(0, 1) + 1.0).abs() < 1e-14);
    }
}
