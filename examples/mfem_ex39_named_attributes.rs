//! mfem_ex39_named_attributes - baseline named-attribute workflow demo.
//!
//! Demonstrates:
//! 1) GMSH PhysicalNames -> NamedAttributeRegistry
//! 2) named set queries on mesh
//! 3) named-set driven submesh extraction
//! 4) multi-set boundary aggregation (--merge-boundary mode)

use fem_io::read_msh;
use fem_mesh::{extract_submesh_by_name, SimplexMesh};
use std::collections::HashSet;

fn main() {
    let args = parse_args();
    println!("=== mfem_ex39_named_attributes: baseline named set workflow ===");
    if args.merge_boundary {
        println!("  Mode: merge-boundary (inlet + outlet aggregation)");
    }
    if args.intersection_region {
        println!("  Mode: intersection-region (inlet �?outlet)");
    }
    if args.difference_region {
        println!("  Mode: difference-region (inlet \\ outlet)");
    }

    let msh_text = r#"$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
3
2 1 "fluid"
1 1 "inlet"
1 3 "outlet"
$EndPhysicalNames
$Nodes
4
1 0 0 0
2 1 0 0
3 1 1 0
4 0 1 0
$EndNodes
$Elements
6
1 1 2 1 1 1 2
2 1 2 2 2 2 3
3 1 2 3 3 3 4
4 1 2 4 4 4 1
5 2 2 1 1 1 2 3
6 2 2 1 1 1 3 4
$EndElements
"#;

    let msh = read_msh(msh_text.as_bytes()).expect("failed to parse in-memory gmsh");
    let registry = msh.named_attribute_registry();
    let mesh: SimplexMesh<2> = msh.into_2d().expect("expected 2D mesh");

    let fluid_elems = mesh
        .element_ids_for_named_set(&registry, "fluid")
        .expect("missing named set: fluid");
    let inlet_faces = mesh
        .face_ids_for_named_set(&registry, "inlet")
        .expect("missing named set: inlet");
    let outlet_faces = mesh
        .face_ids_for_named_set(&registry, "outlet")
        .expect("missing named set: outlet");

    let fluid_sub = extract_submesh_by_name(&mesh, &registry, "fluid")
        .expect("submesh extraction by named set failed");

    println!(
        "  mesh: n_nodes={}, n_elems={}, n_faces={}",
        mesh.n_nodes(),
        mesh.n_elems(),
        mesh.n_faces()
    );
    println!(
        "  named sets: fluid elems={}, inlet faces={}, outlet faces={}, fluid submesh elems={}",
        fluid_elems.len(),
        inlet_faces.len(),
        outlet_faces.len(),
        fluid_sub.mesh.n_elems()
    );

    if args.merge_boundary {
        let mut merged_boundary: HashSet<u32> = inlet_faces.iter().copied().collect();
        merged_boundary.extend(outlet_faces.iter().copied());
        println!(
            "  merged boundary (inlet �?outlet): {} faces",
            merged_boundary.len()
        );
        assert_eq!(
            merged_boundary.len(),
            inlet_faces.len() + outlet_faces.len()
        );
    }

    if args.intersection_region {
        let inlet_set: HashSet<u32> = inlet_faces.iter().copied().collect();
        let outlet_set: HashSet<u32> = outlet_faces.iter().copied().collect();
        let intersection: HashSet<u32> = inlet_set
            .intersection(&outlet_set)
            .copied()
            .collect();
        println!(
            "  intersection (inlet �?outlet): {} faces",
            intersection.len()
        );
    }

    if args.difference_region {
        let inlet_set: HashSet<u32> = inlet_faces.iter().copied().collect();
        let outlet_set: HashSet<u32> = outlet_faces.iter().copied().collect();
        let difference: HashSet<u32> = inlet_set
            .difference(&outlet_set)
            .copied()
            .collect();
        println!(
            "  difference (inlet \\ outlet): {} faces",
            difference.len()
        );
    }

    assert_eq!(fluid_elems.len(), mesh.n_elems());
    assert!(!inlet_faces.is_empty());
    assert!(!outlet_faces.is_empty());
    assert_eq!(fluid_sub.mesh.n_elems(), mesh.n_elems());

    println!("  PASS");
}

struct Args {
    merge_boundary: bool,
    intersection_region: bool,
    difference_region: bool,
}

fn parse_args() -> Args {
    let mut args = Args {
        merge_boundary: false,
        intersection_region: false,
        difference_region: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--merge-boundary" => { args.merge_boundary = true; }
            "--intersection-region" => { args.intersection_region = true; }
            "--difference-region" => { args.difference_region = true; }
            _ => {}
        }
    }
    args
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn named_attributes_merge_boundary_mode() {
        let msh_text = r#"$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
3
2 1 "fluid"
1 1 "inlet"
1 3 "outlet"
$EndPhysicalNames
$Nodes
4
1 0 0 0
2 1 0 0
3 1 1 0
4 0 1 0
$EndNodes
$Elements
6
1 1 2 1 1 1 2
2 1 2 2 2 2 3
3 1 2 3 3 3 4
4 1 2 4 4 4 1
5 2 2 1 1 1 2 3
6 2 2 1 1 1 3 4
$EndElements
"#;

        let msh = read_msh(msh_text.as_bytes()).expect("failed to parse");
        let registry = msh.named_attribute_registry();
        let mesh: SimplexMesh<2> = msh.into_2d().expect("expected 2D mesh");

        let inlet = mesh
            .face_ids_for_named_set(&registry, "inlet")
            .expect("missing inlet");
        let outlet = mesh
            .face_ids_for_named_set(&registry, "outlet")
            .expect("missing outlet");

        let mut merged: std::collections::HashSet<u32> = inlet.iter().copied().collect();
        merged.extend(outlet.iter().copied());

        assert!(!inlet.is_empty());
        assert!(!outlet.is_empty());
        assert_eq!(merged.len(), inlet.len() + outlet.len());
    }

    #[test]
    fn named_attributes_intersection_mode() {
        let msh_text = r#"$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
3
2 1 "fluid"
1 1 "inlet"
1 3 "outlet"
$EndPhysicalNames
$Nodes
4
1 0 0 0
2 1 0 0
3 1 1 0
4 0 1 0
$EndNodes
$Elements
6
1 1 2 1 1 1 2
2 1 2 2 2 2 3
3 1 2 3 3 3 4
4 1 2 4 4 4 1
5 2 2 1 1 1 2 3
6 2 2 1 1 1 3 4
$EndElements
"#;

        let msh = read_msh(msh_text.as_bytes()).expect("failed to parse");
        let registry = msh.named_attribute_registry();
        let mesh: SimplexMesh<2> = msh.into_2d().expect("expected 2D mesh");

        let inlet = mesh
            .face_ids_for_named_set(&registry, "inlet")
            .expect("missing inlet");
        let outlet = mesh
            .face_ids_for_named_set(&registry, "outlet")
            .expect("missing outlet");

        let inlet_set: std::collections::HashSet<u32> = inlet.iter().copied().collect();
        let outlet_set: std::collections::HashSet<u32> = outlet.iter().copied().collect();
        let intersection: std::collections::HashSet<u32> = inlet_set
            .intersection(&outlet_set)
            .copied()
            .collect();

        // For this mesh, inlet and outlet don't share faces, so intersection is empty
        assert_eq!(intersection.len(), 0);
    }

    #[test]
    fn named_attributes_difference_mode() {
        let msh_text = r#"$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
3
2 1 "fluid"
1 1 "inlet"
1 3 "outlet"
$EndPhysicalNames
$Nodes
4
1 0 0 0
2 1 0 0
3 1 1 0
4 0 1 0
$EndNodes
$Elements
6
1 1 2 1 1 1 2
2 1 2 2 2 2 3
3 1 2 3 3 3 4
4 1 2 4 4 4 1
5 2 2 1 1 1 2 3
6 2 2 1 1 1 3 4
$EndElements
"#;

        let msh = read_msh(msh_text.as_bytes()).expect("failed to parse");
        let registry = msh.named_attribute_registry();
        let mesh: SimplexMesh<2> = msh.into_2d().expect("expected 2D mesh");

        let inlet = mesh
            .face_ids_for_named_set(&registry, "inlet")
            .expect("missing inlet");
        let outlet = mesh
            .face_ids_for_named_set(&registry, "outlet")
            .expect("missing outlet");

        let inlet_set: std::collections::HashSet<u32> = inlet.iter().copied().collect();
        let outlet_set: std::collections::HashSet<u32> = outlet.iter().copied().collect();
        let difference: std::collections::HashSet<u32> = inlet_set
            .difference(&outlet_set)
            .copied()
            .collect();

        // For this mesh, inlet \ outlet = inlet (since they don't intersect)
        assert_eq!(difference.len(), inlet.len());
    }
}

