/// Physical group tag assigned to a boundary face (edge in 2-D, face in 3-D).
///
/// Tags match the GMSH physical group numbers defined in the `.geo` file.
/// Tag 0 means "untagged" (interior face or boundary without a label).
/// Negative tags are reserved by fem-rs for internal use.
pub type BoundaryTag = i32;

/// Assign a human-readable name to a physical group.
#[derive(Debug, Clone)]
pub struct PhysicalGroup {
    /// Topological dimension of the group (0–3).
    pub dim: u8,
    /// GMSH tag number.
    pub tag: BoundaryTag,
    /// Name as defined in the `.geo` file.
    pub name: String,
}
