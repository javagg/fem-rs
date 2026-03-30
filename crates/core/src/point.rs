//! Re-exports of nalgebra point and vector types used as physical coordinates.
//!
//! Use these aliases throughout fem-rs instead of direct nalgebra imports so
//! that the concrete type can be changed in one place if needed.

pub use nalgebra::{
    Point2, Point3,
    Vector2, Vector3,
    Matrix2, Matrix3,
    SMatrix, SVector,
};

/// 2-D physical coordinate (x, y).
pub type Coord2 = Point2<f64>;
/// 3-D physical coordinate (x, y, z).
pub type Coord3 = Point3<f64>;

/// 2-D displacement / gradient vector.
pub type Vec2 = Vector2<f64>;
/// 3-D displacement / gradient vector.
pub type Vec3 = Vector3<f64>;

/// 2×2 matrix (e.g., Jacobian in 2-D).
pub type Mat2x2 = Matrix2<f64>;
/// 3×3 matrix (e.g., Jacobian in 3-D).
pub type Mat3x3 = Matrix3<f64>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coord_construction() {
        let p: Coord2 = Point2::new(1.0, 2.0);
        assert_eq!(p.x, 1.0);
        assert_eq!(p.y, 2.0);

        let q: Coord3 = Point3::new(1.0, 2.0, 3.0);
        assert_eq!(q.z, 3.0);
    }

    #[test]
    fn jacobian_inverse() {
        // 2-D identity Jacobian: det = 1, inverse = identity
        let j = Mat2x2::identity();
        let det = j.determinant();
        let inv = j.try_inverse();
        assert!((det - 1.0).abs() < 1e-14);
        assert!(inv.is_some());
        // J^{-T} of identity is identity
        assert_eq!(inv.unwrap().transpose(), Mat2x2::identity());
    }
}
