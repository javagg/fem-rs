use std::time::Instant;
use std::path::Path;
use std::{env, fs};

use fem_assembly::standard::{DiffusionIntegrator, DomainSourceIntegrator};
use fem_mesh::SimplexMesh;
use fem_parallel::{
    par_solve_gmres_ras, par_solve_pcg_ras, partition_simplex, ParAssembler,
    ParallelFESpace, RasConfig, RasLocalSolverKind,
};
use fem_parallel::launcher::{native::ThreadLauncher, WorkerConfig};
use fem_solver::SolverConfig;
use fem_space::constraints::boundary_dofs;
use fem_space::fe_space::FESpace;
use fem_space::H1Space;

#[test]
#[ignore = "benchmark-style timing test; run explicitly"]
fn ras_benchmark_report_two_ranks() {
    let mesh = SimplexMesh::<2>::unit_square_tri(24);
    let launcher = ThreadLauncher::new(WorkerConfig::new(2));

    launcher.launch(move |comm| {
        let pmesh = partition_simplex(&mesh, &comm);
        let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
        let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());

        let diff = DiffusionIntegrator { kappa: 1.0 };
        let mut a_mat = ParAssembler::assemble_bilinear(&par_space, &[&diff], 2);
        let source = DomainSourceIntegrator::new(|_x: &[f64]| 1.0);
        let mut rhs = ParAssembler::assemble_linear(&par_space, &[&source], 3);

        let dm = par_space.local_space().dof_manager();
        let bc_dofs = boundary_dofs(par_space.local_space().mesh(), dm, &[1, 2, 3, 4]);
        for &d in &bc_dofs {
            let lid = d as usize;
            if lid < par_space.dof_partition().n_owned_dofs {
                a_mat.apply_dirichlet_row(lid, 0.0, rhs.as_slice_mut());
            }
        }

        let cfg = SolverConfig {
            rtol: 1e-8,
            max_iter: 2000,
            ..SolverConfig::default()
        };

        let mut rows: Vec<(String, usize, f64, f64)> = Vec::new();

        {
            let mut run_pcg = |name: &str, ras_cfg: RasConfig| {
                let mut u = fem_parallel::ParVector::zeros(&par_space);
                let t0 = Instant::now();
                let res = par_solve_pcg_ras(&a_mat, &rhs, &mut u, &ras_cfg, &cfg)
                    .unwrap_or_else(|e| panic!("{} failed: {}", name, e));
                let dt = t0.elapsed().as_secs_f64() * 1e3;
                rows.push((name.to_string(), res.iterations, res.final_residual, dt));
            };

            run_pcg(
                "pcg_ras_diag_ov0",
                RasConfig {
                    overlap: 0,
                    local_solver: RasLocalSolverKind::DiagJacobi,
                    ..RasConfig::default()
                },
            );
            run_pcg(
                "pcg_ras_diag_ov1",
                RasConfig {
                    overlap: 1,
                    local_solver: RasLocalSolverKind::DiagJacobi,
                    ..RasConfig::default()
                },
            );
            run_pcg(
                "pcg_ras_ilu0_ov0",
                RasConfig {
                    overlap: 0,
                    local_solver: RasLocalSolverKind::Ilu0,
                    ..RasConfig::default()
                },
            );
            run_pcg(
                "pcg_ras_ilu0_ov1",
                RasConfig {
                    overlap: 1,
                    local_solver: RasLocalSolverKind::Ilu0,
                    ..RasConfig::default()
                },
            );
        }

        {
            let mut run_gmres = |name: &str, ras_cfg: RasConfig| {
                let mut u = fem_parallel::ParVector::zeros(&par_space);
                let t0 = Instant::now();
                let res = par_solve_gmres_ras(&a_mat, &rhs, &mut u, &ras_cfg, 30, &cfg)
                    .unwrap_or_else(|e| panic!("{} failed: {}", name, e));
                let dt = t0.elapsed().as_secs_f64() * 1e3;
                rows.push((name.to_string(), res.iterations, res.final_residual, dt));
            };

            run_gmres(
                "gmres_ras_diag_ov0",
                RasConfig {
                    overlap: 0,
                    local_solver: RasLocalSolverKind::DiagJacobi,
                    ..RasConfig::default()
                },
            );
            run_gmres(
                "gmres_ras_diag_ov1",
                RasConfig {
                    overlap: 1,
                    local_solver: RasLocalSolverKind::DiagJacobi,
                    ..RasConfig::default()
                },
            );
            run_gmres(
                "gmres_ras_ilu0_ov0",
                RasConfig {
                    overlap: 0,
                    local_solver: RasLocalSolverKind::Ilu0,
                    ..RasConfig::default()
                },
            );
            run_gmres(
                "gmres_ras_ilu0_ov1",
                RasConfig {
                    overlap: 1,
                    local_solver: RasLocalSolverKind::Ilu0,
                    ..RasConfig::default()
                },
            );
        }

        if comm.is_root() {
            println!("\n=== RAS Benchmark (2 ranks, mesh=unit_square_tri(24)) ===");
            println!("case,iterations,final_residual,time_ms");
            for (name, it, rr, ms) in &rows {
                println!("{},{},{:.3e},{:.3}", name, it, rr, ms);
            }

            if let Ok(path) = env::var("RAS_BENCH_CSV") {
                let mut csv = String::from("case,iterations,final_residual,time_ms\n");
                for (name, it, rr, ms) in &rows {
                    csv.push_str(&format!("{},{},{:.6e},{:.6}\n", name, it, rr, ms));
                }
                if let Some(parent) = Path::new(&path).parent() {
                    if !parent.as_os_str().is_empty() {
                        fs::create_dir_all(parent).unwrap_or_else(|e| {
                            panic!(
                                "failed to create parent directory for RAS_BENCH_CSV {}: {}",
                                path, e
                            )
                        });
                    }
                }
                fs::write(&path, csv)
                    .unwrap_or_else(|e| panic!("failed to write RAS_BENCH_CSV to {}: {}", path, e));
                println!("ras benchmark csv written to {}", path);
            }
        }

        for (_name, _it, rr, _ms) in rows {
            assert!(rr <= 1e-6, "benchmark run residual too high: {:.3e}", rr);
        }
    });
}
