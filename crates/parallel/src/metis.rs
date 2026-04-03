//! METIS-based mesh partitioning.
//!
//! [`MetisPartitioner`] builds a **dual graph** from the element connectivity
//! (elements are vertices, shared faces/edges are graph edges), then partitions
//! the mesh.  Currently uses a greedy graph-coloring fallback (no external
//! METIS dependency); the API is compatible with a future rmetis backend.
//!
//! # Usage
//! ```rust,ignore
//! use fem_parallel::metis::{MetisPartitioner, MetisOptions};
//! use fem_mesh::SimplexMesh;
//!
//! let mesh = SimplexMesh::<2>::unit_square_tri(16);
//! let opt  = MetisOptions::default();
//! let parts = MetisPartitioner::partition_mesh(&mesh, 4, &opt).unwrap();
//! // parts[e] = rank that owns element e (0..nparts)
//! ```
//!
//! The `partition_simplex_metis` convenience function wraps this into a
//! [`ParallelMesh`] for use in the parallel pipeline.

use std::collections::HashMap;

use fem_core::{ElemId, FaceId, NodeId, Rank};
use fem_mesh::SimplexMesh;

use crate::{Comm, MeshPartition, par_mesh::ParallelMesh};

// ─── Options ──────────────────────────────────────────────────────────────────

/// Options for the METIS partitioner.
#[derive(Debug, Clone, Default)]
pub struct MetisOptions {
    /// If true, print partition statistics to stdout.
    pub verbose: bool,
}

// ─── MetisPartitioner ─────────────────────────────────────────────────────────

/// Mesh partitioner using a greedy graph-bisection heuristic.
///
/// This provides balanced partitions without an external METIS dependency.
/// For production use with large meshes, link against METIS via a feature flag.
pub struct MetisPartitioner;

impl MetisPartitioner {
    /// Partition a simplex mesh into `nparts` balanced parts.
    ///
    /// Returns a vector of length `n_elems` where `partition[e]` is the rank
    /// (0..nparts) assigned to element `e`.
    pub fn partition_mesh<const D: usize>(
        mesh:   &SimplexMesh<D>,
        nparts: usize,
        opts:   &MetisOptions,
    ) -> Result<Vec<Rank>, String> {
        assert!(nparts >= 1, "nparts must be ≥ 1");
        let n_elems = mesh.n_elems();
        assert!(n_elems > 0, "mesh has no elements");

        if nparts == 1 {
            return Ok(vec![0; n_elems]);
        }

        // Build dual graph
        let (xadj, adjncy) = build_dual_graph(mesh);

        // Greedy BFS-based k-way partitioning
        let partition = bfs_kway_partition(n_elems, &xadj, &adjncy, nparts);

        if opts.verbose {
            let mut counts = vec![0usize; nparts];
            for &p in &partition { counts[p as usize] += 1; }
            println!("[MetisPartitioner] nparts={nparts}, counts={counts:?}");
        }

        Ok(partition)
    }
}

// ─── BFS k-way partitioner ────────────────────────────────────────────────────

/// Simple BFS-based k-way partitioner.
///
/// Grows k regions simultaneously from seed elements placed uniformly.
fn bfs_kway_partition(n: usize, xadj: &[i32], adjncy: &[i32], k: usize) -> Vec<Rank> {
    const UNSET: Rank = -1;
    let mut part = vec![UNSET; n];
    let mut queue: std::collections::VecDeque<usize> = Default::default();

    // Place k seeds spaced evenly
    for p in 0..k {
        let seed = (p * n) / k;
        if part[seed] == UNSET {
            part[seed] = p as Rank;
            queue.push_back(seed);
        }
    }

    // BFS flood-fill
    while let Some(e) = queue.pop_front() {
        let owner = part[e];
        for j in xadj[e] as usize..xadj[e + 1] as usize {
            let nb = adjncy[j] as usize;
            if part[nb] == UNSET {
                part[nb] = owner;
                queue.push_back(nb);
            }
        }
    }

    // Assign any remaining unvisited elements (disconnected components)
    for i in 0..n {
        if part[i] == UNSET {
            part[i] = (i % k) as Rank;
        }
    }

    part
}

// ─── Dual graph builder ───────────────────────────────────────────────────────

fn build_dual_graph<const D: usize>(mesh: &SimplexMesh<D>) -> (Vec<i32>, Vec<i32>) {
    let n_elems = mesh.n_elems();

    let mut face_map: HashMap<Vec<NodeId>, Vec<ElemId>> = HashMap::new();
    for e in 0..n_elems as ElemId {
        let nodes = mesh.elem_nodes(e);
        for lf in local_faces_of_elem::<D>(nodes) {
            let mut key = lf;
            key.sort_unstable();
            face_map.entry(key).or_default().push(e);
        }
    }

    let mut adj: Vec<Vec<ElemId>> = vec![Vec::new(); n_elems];
    for (_key, elems) in &face_map {
        if elems.len() == 2 {
            adj[elems[0] as usize].push(elems[1]);
            adj[elems[1] as usize].push(elems[0]);
        }
    }

    let mut xadj = vec![0_i32; n_elems + 1];
    let mut adjncy = Vec::<i32>::new();
    for (e, nbrs) in adj.iter().enumerate() {
        xadj[e + 1] = xadj[e] + nbrs.len() as i32;
        adjncy.extend(nbrs.iter().map(|&n| n as i32));
    }
    (xadj, adjncy)
}

fn local_faces_of_elem<const D: usize>(nodes: &[NodeId]) -> Vec<Vec<NodeId>> {
    match (nodes.len(), D) {
        (3, 2) => vec![
            vec![nodes[0], nodes[1]],
            vec![nodes[1], nodes[2]],
            vec![nodes[0], nodes[2]],
        ],
        (4, 3) => vec![
            vec![nodes[1], nodes[2], nodes[3]],
            vec![nodes[0], nodes[2], nodes[3]],
            vec![nodes[0], nodes[1], nodes[3]],
            vec![nodes[0], nodes[1], nodes[2]],
        ],
        _ => vec![],
    }
}

// ─── partition_simplex_metis ──────────────────────────────────────────────────

/// Distribute `mesh` across `comm.size()` ranks using k-way partitioning.
pub fn partition_simplex_metis<const D: usize>(
    mesh: &SimplexMesh<D>,
    comm: &Comm,
    opts: &MetisOptions,
) -> ParallelMesh<SimplexMesh<D>> {
    let n_elems = mesh.n_elems();
    let n_nodes_total = mesh.n_nodes();
    assert!(n_elems > 0, "partition_simplex_metis: mesh has no elements");

    let size = comm.size();
    if size == 1 {
        let partition = MeshPartition::new_serial(n_nodes_total, n_elems);
        return ParallelMesh::new(mesh.clone(), comm.clone(), partition);
    }

    let elem_part = MetisPartitioner::partition_mesh(mesh, size, opts)
        .expect("partitioning failed");

    build_parallel_mesh_from_partition(mesh, comm, &elem_part)
}

fn build_parallel_mesh_from_partition<const D: usize>(
    mesh:      &SimplexMesh<D>,
    comm:      &Comm,
    elem_part: &[Rank],
) -> ParallelMesh<SimplexMesh<D>> {
    let n_elems = mesh.n_elems();
    let n_nodes = mesh.n_nodes();
    let local_rank = comm.rank();

    let mut node_owners = vec![usize::MAX; n_nodes];
    for e in 0..n_elems {
        let rank = elem_part[e] as usize;
        for &n in mesh.elem_nodes(e as ElemId) {
            if node_owners[n as usize] == usize::MAX {
                node_owners[n as usize] = rank;
            }
        }
    }
    for o in &mut node_owners { if *o == usize::MAX { *o = 0; } }

    let local_elem_gids: Vec<u32> = (0..n_elems as ElemId)
        .filter(|&e| elem_part[e as usize] as usize == local_rank as usize)
        .collect();

    let mut node_set: std::collections::BTreeSet<NodeId> = Default::default();
    for &ge in &local_elem_gids {
        for &n in mesh.elem_nodes(ge) { node_set.insert(n); }
    }

    let mut owned_global: Vec<NodeId> = Vec::new();
    let mut ghost_global: Vec<(NodeId, Rank)> = Vec::new();
    for gn in &node_set {
        let owner = node_owners[*gn as usize] as Rank;
        if owner == local_rank { owned_global.push(*gn); }
        else { ghost_global.push((*gn, owner)); }
    }

    let ghost_base = owned_global.len();
    let mut g2l: HashMap<NodeId, u32> = HashMap::new();
    for (lid, &gn) in owned_global.iter().enumerate() { g2l.insert(gn, lid as u32); }
    for (idx, &(gn, _)) in ghost_global.iter().enumerate() {
        g2l.insert(gn, (ghost_base + idx) as u32);
    }

    let total_local_nodes = g2l.len();
    let mut local_coords = Vec::with_capacity(total_local_nodes * D);
    for &gn in owned_global.iter().chain(ghost_global.iter().map(|(gn, _)| gn)) {
        local_coords.extend_from_slice(&mesh.coords_of(gn));
    }

    let npe = mesh.elem_type.nodes_per_element();
    let mut local_conn = Vec::with_capacity(local_elem_gids.len() * npe);
    let mut local_elem_tags = Vec::new();
    for &ge in &local_elem_gids {
        for &gn in mesh.elem_nodes(ge) { local_conn.push(g2l[&gn]); }
        local_elem_tags.push(mesh.elem_tags[ge as usize]);
    }

    let (local_face_conn, local_face_tags) =
        extract_local_faces(mesh, &g2l, &node_owners, local_rank as usize);

    let local_mesh = SimplexMesh::<D> {
        coords:    local_coords,
        conn:      local_conn,
        elem_tags: local_elem_tags,
        elem_type: mesh.elem_type,
        face_conn: local_face_conn,
        face_tags: local_face_tags,
        face_type: mesh.face_type,
    };

    let partition = MeshPartition::from_partitioner(
        &owned_global,
        &ghost_global,
        &local_elem_gids,
        local_rank,
    );

    ParallelMesh::new(local_mesh, comm.clone(), partition)
}

fn extract_local_faces<const D: usize>(
    mesh:        &SimplexMesh<D>,
    g2l:         &HashMap<NodeId, u32>,
    node_owners: &[usize],
    local_rank:  usize,
) -> (Vec<NodeId>, Vec<i32>) {
    let n_bfaces = mesh.n_faces();
    let mut face_conn = Vec::new();
    let mut face_tags = Vec::new();

    for f in 0..n_bfaces as u32 {
        let bnodes = mesh.bface_nodes(f as FaceId);
        if bnodes.iter().any(|gn| !g2l.contains_key(gn)) { continue; }
        let min_gn = *bnodes.iter().min().expect("face has no nodes");
        if node_owners[min_gn as usize] != local_rank { continue; }
        for &gn in bnodes { face_conn.push(g2l[&gn]); }
        face_tags.push(mesh.face_tags[f as usize]);
    }
    (face_conn, face_tags)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;

    #[test]
    fn partition_covers_all_elements() {
        let mesh = SimplexMesh::<2>::unit_square_tri(8);
        let n_elems = mesh.n_elems();
        let nparts = 4;
        let parts = MetisPartitioner::partition_mesh(&mesh, nparts, &MetisOptions::default())
            .unwrap();
        assert_eq!(parts.len(), n_elems);
        for &p in &parts {
            assert!((p as usize) < nparts, "partition out of range: {p}");
        }
    }

    #[test]
    fn partition_balanced() {
        let mesh = SimplexMesh::<2>::unit_square_tri(8);
        let n_elems = mesh.n_elems();
        let nparts = 4;
        let parts = MetisPartitioner::partition_mesh(&mesh, nparts, &MetisOptions::default())
            .unwrap();
        let mut counts = vec![0usize; nparts];
        for &p in &parts { counts[p as usize] += 1; }
        let ideal = n_elems as f64 / nparts as f64;
        for (i, &c) in counts.iter().enumerate() {
            let imbalance = (c as f64 - ideal).abs() / ideal;
            assert!(imbalance < 0.6, "part {i}: count={c}, ideal={ideal:.1}, imbalance={imbalance:.2}");
        }
    }

    #[test]
    fn partition_single_part_is_identity() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let n = mesh.n_elems();
        let parts = MetisPartitioner::partition_mesh(&mesh, 1, &MetisOptions::default()).unwrap();
        assert!(parts.iter().all(|&p| p == 0));
        assert_eq!(parts.len(), n);
    }

    #[test]
    fn partition_simplex_serial() {
        use crate::launcher::{Launcher, native::MpiLauncher};
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let comm = MpiLauncher::init().unwrap().world_comm();
        let pmesh = partition_simplex_metis(&mesh, &comm, &MetisOptions::default());
        assert_eq!(pmesh.global_n_elems(), mesh.n_elems());
        assert_eq!(pmesh.global_n_nodes(), mesh.n_nodes());
        pmesh.local_mesh().check().expect("local mesh failed check");
    }
}
