# RAS DDM Status (2026-04-19)

## Scope

This note tracks the current Domain Decomposition rollout in `fem-parallel`.
The implemented method is Restricted Additive Schwarz (RAS) used as a
preconditioner for parallel Krylov solvers.

## Implemented

- RAS preconditioner API in `crates/parallel/src/par_ras.rs`.
- Solver entrypoints:
  - `par_solve_pcg_ras`
  - `par_solve_gmres_ras`
- Local solver kernels:
  - `DiagJacobi`
  - `Ilu0`
- Overlap support:
  - `overlap = 0`
  - `overlap = 1` (multiplicative two-stage overlap correction)
  - `overlap > 1` currently rejected with a clear error.

## Regression Coverage

Current RAS regression tests are in `crates/parallel/src/par_ras.rs` and include:

- Build-time overlap contract checks.
- Serial and 2-rank convergence checks for PCG + RAS.
- Serial and 2-rank convergence checks for GMRES + RAS.
- Diag and ILU0 local-kernel paths.
- Stability checks across overlap modes.

Run:

```bash
cargo test -p fem-parallel par_ras -- --nocapture
```

## Benchmark Entry

A benchmark-style ignored test is provided in:

- `crates/parallel/tests/ras_benchmark.rs`

Run explicitly:

```bash
cargo test -p fem-parallel ras_benchmark_report_two_ranks -- --ignored --nocapture
```

Optional CSV export:

PowerShell:

```powershell
$env:RAS_BENCH_CSV = "output/ras_benchmark.csv"
cargo test -p fem-parallel ras_benchmark_report_two_ranks -- --ignored --nocapture
```

Bash:

```bash
RAS_BENCH_CSV=output/ras_benchmark.csv cargo test -p fem-parallel ras_benchmark_report_two_ranks -- --ignored --nocapture
```

The benchmark currently reports:

- `pcg_ras_diag_ov0`
- `pcg_ras_diag_ov1`
- `pcg_ras_ilu0_ov0`
- `pcg_ras_ilu0_ov1`
- `gmres_ras_diag_ov0`
- `gmres_ras_diag_ov1`
- `gmres_ras_ilu0_ov0`
- `gmres_ras_ilu0_ov1`

## Current Conclusion

At current mesh/problem settings, overlap=1 is generally beneficial, and
`Ilu0 + overlap=1` is the strongest tested variant for both PCG and GMRES.

## Next Engineering Steps

1. Replace overlap=1 MVP correction with explicit subdomain expansion and
   restriction/prolongation operators.
2. Add overlap-level diagnostics (effective subdomain size, communication cost).
3. Add larger-mesh benchmark points and optional CSV output for trend tracking.
