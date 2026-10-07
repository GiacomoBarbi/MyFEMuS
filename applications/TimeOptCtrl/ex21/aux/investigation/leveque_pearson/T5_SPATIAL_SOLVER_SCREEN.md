# T5 — convection-robust spatial solver for the SCHUR_SWEEP diagonal blocks

**Result: solved without any source change, using PETSc options on the existing
sweep prefixes.**

- **GAMG cannot be rescued by small tuning.** The E block still fails at
  `Pe_cell` ≥ 6.
- **hypre is not in this PETSc build**, so BoomerAMG/AIR could not be tested.
- **Every factorisation-based preconditioner tested is robust** for both
  advection orientations (E, right, left^T), both θ, up to ε = 1/500 and
  66k DOFs: ILU(k), ASM+ILU, block-Jacobi+ILU, and LU.
- **End-to-end**, every shortlisted backend reproduces the exact-sweep KKT
  behaviour (2 outer iterations, the same Schur counts) and the same discrete
  solution as LU (|Δy|, |Δp| ≤ 1e-13; J identical to 12 digits). This holds for
  ε = 1/100 and 1/500, CN and BE, nx up to 64.
- **At these 2D sizes, sparse direct LU is the fastest option end-to-end.**

The production default (GMRES + GAMG) is unchanged. The binary is still
500a2b2e; no source file was modified.

Files:
- scripts: `t5/run_t5.py` (configurations, block screen, scaling screen, KKT;
  `run_t5.py show <cfg>` prints the exact options), `t5/analyze_t5.py`,
  `t5/mpi_check.py`;
- data: `t5/results/`, `t5/logs/`;
- summaries: `t5_block_solver_screen.csv`, `t5_kkt_solver_shortlist.csv`;
- figures: `figures/t5_iterations_vs_peclet.png`, `figures/t5_time_vs_solver.png`.

## 1. PETSc build capabilities (this installation)

Build: PETSc **3.23.5**, `arch-linux-c-openblas-opt`, the build ex21 links
against (checked with `ldd`). Configure options: `--download-openmpi
--download-hdf5 --download-metis --download-parmetis --download-blacs
--download-scalapack --download-mumps --download-suitesparse`, OpenBLAS.

| candidate | available? | exact PETSc options (per prefix) | serial / MPI | comments |
|---|---|---|---|---|
| GAMG (baseline) | yes (built in) | default: `pc_type gamg` | MPI, scalable AMG | defaults read with `-ksp_view` on a real block: smoothed aggregation (`agg_nsmooths 1`), 1 aggressive MIS-2 level, Chebyshev(2)+Jacobi smoother, LU on the coarse level, 3 levels |
| GAMG tuning | yes | `mg_levels_ksp_type richardson`, `mg_levels_pc_type sor`, `pc_gamg_agg_nsmooths 0`, `mg_levels_ksp_max_it N`, `pc_gamg_aggressive_coarsening 0` (also exposed: `pc_gamg_graph_symmetrize` [default TRUE], `pc_gamg_threshold`, `pc_gamg_mat_coarsen_type hem/misk/mis`) | MPI | names verified from `-help` of this build |
| hypre / BoomerAMG / AIR | **NO** | – | – | No `PETSC_HAVE_HYPRE`; `pc_type` has no `hypre` entry. A `libHYPRE.so` exists only in an unrelated conda environment and is not linked. **Not testable without rebuilding PETSc (`--download-hypre`).** |
| ILU(k) | yes | `pc_type ilu`, `pc_factor_levels k` | **sequential only** (PCILU needs SeqAIJ) | – |
| ASM + ILU | yes | `pc_type asm`, `pc_asm_blocks B`, `pc_asm_overlap o`, `sub_pc_type ilu`, `sub_pc_factor_levels k` | MPI | in serial, `pc_asm_blocks 4` emulates 4 subdomains |
| block-Jacobi + ILU | yes | `pc_type bjacobi`, `pc_bjacobi_blocks B`, `sub_pc_type ilu` | MPI | – |
| LU, PETSc native | yes | `pc_type lu` | sequential | – |
| LU, MUMPS | yes | `pc_type lu`, `pc_factor_mat_solver_type mumps` | **MPI** | verified at np = 2, 4 (§7) |
| LU, UMFPACK / KLU (SuiteSparse) | yes | `pc_factor_mat_solver_type umfpack` | sequential | – |
| SuperLU_DIST, STRUMPACK, PaStiX | no | – | – | not configured |

`pc_type` list of this build, from `-help`: nn tfs hmg bddc composite ksp lu icc patch
bjacobi eisenstat deflation vpbjacobi redistribute sor mg pbjacobi cholesky mat qr
svd fieldsplit mpi kaczmarz jacobi telescope redundant cp shell galerkin ilu exotic
gasm gamg none lmvm asm lsc.

## 2. Method

**Prefixes.** No code change. Each logical configuration (`CONFIGS` in
`run_t5.py`) is expanded:
- onto the block-test prefix `block_test_` for screening;
- onto **all four** sweep prefixes for KKT runs: `sweep_right_`,
  `sweep_left_`, `sweep_ay_`, `fieldsplit_y_sweep_`.

All of these call `KSPSetFromOptions`. Every configuration was checked with
`-ksp_view` to take effect.

**Krylov method, unchanged.** It is set by the code: GMRES(50), right
preconditioning, unpreconditioned norm, rtol 1e-10, maxits 500, zero initial
guess. Only the PC changes.

**Blocks.** Each block-test run solves the three sweep blocks:
- `E = M/dt + θL`;
- `right = E + √β N`;
- `left = (E + √β M_ω)ᵀ`, the transposed advection;

for θ = 1 (BE) and θ = ½ (CN diagonal), with 3 seeded random right-hand sides
and 1 smooth one. All LP points use `τ = h_LP = 1/nx` and β = 1e-2.

**Screen points.**

| point | ε | nx | Pe_cell | role |
|---|---|---|---|---|
| A | 1/20 | 32 | 1.25 | easy control |
| B | 1/20 | 16 | 2.5 | first failure of the baseline |
| C | 1/100 | 32 | 6.25 | moderate convection |
| D | 1/500 | 32 | 31 | hard |

**Scaling points** (survivors only):

| point | ε | nx | Pe_cell | DOFs |
|---|---|---|---|---|
| E16 | 1/500 | 16 | 62.5 | 1,089 |
| C64 | 1/100 | 64 | 3.1 | 16,641 |
| D64 | 1/500 | 64 | 15.6 | 16,641 |
| D128 | 1/500 | 128 | 7.8 | 66,049 |

**Not covered by the block test.** The CN `A_left_final` block (last slot) is
`(A + √β·½·M_ω)ᵀ`. It has the same structure and convection orientation as
`left`, with half the extra mass, so it is no harder than `left` at θ = ½. It
is exercised in every CN KKT run (§5).

## 3. Block screen

Entries are the maximum GMRES iterations over the four right-hand sides.
**500!** means a failure (500-iteration cap, KSP reason −3). Within each point
the column order is θ = 1 (E, right, left), then θ = ½ (E, right, left).

| config | A (Pe 1.25) | B (Pe 2.5) | C (Pe 6.25) | D (Pe 31) |
|---|---|---|---|---|
| gamg_base | 24 24 24 · 23 23 23 | **500!** 278 266 · 129 45 44 | **500! ×6** | **500! ×6** |
| gamg_sor | 8 7 7 · 5 5 5 | 6 4 4 · 4 3 3 | **500!** 41 37 · 11 7 7 | **500! 500! 500! 500!** 45 46 |
| gamg_sor_unsm | 9 8 8 · 6 5 5 | 6 4 4 · 4 3 3 | **500!** 36 35 · 11 7 7 | **500! ×4** 45 45 |
| gamg_sor_unsm4 | 5 4 4 · 3 3 3 | 3 2 2 · 2 2 2 | **500!** 23 23 · 7 4 4 | **500! ×4** 30 30 |
| ilu0 | 25 21 21 · 16 14 14 | 20 15 15 · 13 10 10 | 22 18 18 · 14 11 11 | 33 26 26 · 19 17 17 |
| **ilu1** | 6 6 6 · 5 5 5 | 6 5 5 · 5 4 4 | 7 7 7 · 6 6 6 | 16 13 13 · 10 9 9 |
| ilu2 | 5 4 4 · 4 3 3 | 4 3 4 · 3 3 3 | 5 5 5 · 4 4 4 | 10 8 8 · 7 6 6 |
| **asm4o1_ilu1** | 7 6 6 · 5 5 5 | 6 5 5 · 5 5 5 | 8 7 7 · 6 6 6 | 17 13 13 · 10 9 9 |
| asm4o0_ilu1 | 52 44 44 · 33 28 28 | 43 31 30 · 26 19 19 | 42 35 35 · 26 23 23 | 82 68 68 · 54 48 48 |
| bjac4_ilu1 | identical to asm4o0_ilu1 (expected: ASM without overlap is block Jacobi) | | | |
| lu_petsc / lu_mumps / lu_umfpack | 1 everywhere | 1 | 1 | 1 |

**Scaling** (maximum iterations over all 6 blocks; E-block θ = 1 setup [ms] / solve per right-hand side [ms] in brackets):

| config | E16 (Pe 62.5) | C64 | D64 | D128 (66k DOFs) |
|---|---|---|---|---|
| ilu0 | 33 [0/1.2] | 25 [3/16.8] | 30 [3/19.0] | 28 [16/78.5] |
| **ilu1** | 17 [2/1.2] | 6 [33/8.8] | 12 [36/15.0] | 9 [148/47.2] |
| ilu2 | 10 [8/1.5] | 5 [144/14.8] | 8 [145/20.2] | 6 [658/67.2] |
| **asm4o1_ilu1** | 19 [2/5.2] | 6 [15/52.5] | 13 [14/72.8] | 9 [61/272.2] |
| bjac4_ilu1 | 91 [1/3.2] | 48 [2/32.8] | 59 [2/38.8] | 48 [13/140.5] |
| **lu_petsc** | 1 [2/0.2] | 1 [55/3.5] | 1 [56/3.5] | 1 [374/16.5] |
| lu_mumps | 1 [15/0.5] | 1 [125/6.8] | 1 [135/6.8] | 1 [620/30.2] |
| lu_umfpack | 1 [3/0.2] | 1 [49/5.5] | 1 [46/3.8] | 1 [242/20.8] |

**Forward vs transpose.** The factorisation-based candidates handle E (forward
advection) and left (transposed advection) equally well: the left counts equal
or are below the E counts at every point. The tuned GAMG variants rescue
`right`/`left` at C (they carry the extra `√β` mass) but not the plain E
block. **No GAMG variant handles E at Pe_cell ≥ 6.**

## 4. Rejected candidates

| candidate | reason |
|---|---|
| gamg_base (production) | fails at B (θ = 1, E), and at every block of C and D |
| gamg_sor / gamg_sor_unsm / gamg_sor_unsm4 | The defensible nonsymmetric tuning (SOR smoother instead of Chebyshev/Jacobi, unsmoothed aggregation, more smoothing, no aggressive coarsening) rescues B and the mass-augmented blocks at C. **The E block still fails at C and D**, as do D's θ = 1 right/left blocks. Further parameter search was deliberately not pursued. |
| ilu0 | robust (≤ 33), but 3–5× the iterations of ILU(1) with no cost advantage at scale |
| ilu2 | robust (≤ 10) but expensive setup (658 ms at D128) for little gain over ILU(1) |
| bjac4_ilu1, asm4o0_ilu1 | robust but weak (≤ 91 iterations): no overlap |
| lu_mumps, lu_umfpack | robust; kept as alternatives to PETSc LU (MUMPS is the MPI direct path, §7). PETSc native LU is the fastest serial factorisation here. |
| hypre/BoomerAMG/AIR | not available in this build |

**Shortlist (2 iterative + LU):**
- `ilu1`: the best serial iterative option;
- `asm4o1_ilu1`: the MPI-capable iterative option;
- `lu_petsc`: the reference.

## 5. End-to-end KKT (LP §5.2, corrected tracking, β = 1e-2, production KKT tolerances)

Every run in the table: outer reason 2 (converged), **0 sweep failures**.

| config | case | outer its | Schur its/outer | spatial solves | mean / max spatial its | solve wall [s] | total [s] | relative KKT (state/adj) | march | max\|Δy\|, max\|Δp\| vs LU | J vs LU |
|---|---|---|---|---|---|---|---|---|---|---|---|
| lu_petsc | CN 1/100 nx16 | 2 | 9.5 | 2080 | 1 / 1 | 0.44 | 0.7 | 3.5e-15 / 4.7e-15 | 4.3e-15 | ref | ref |
| lu_petsc | CN 1/100 nx32 | 2 | 8.0 | 3584 | 1 / 1 | 3.7 | 5.5 | 1.6e-14 / 2.8e-14 | 1.4e-14 | ref | ref |
| lu_petsc | CN 1/500 nx16 | 2 | 14.0 | 2944 | 1 / 1 | 0.68 | 1.0 | 2.0e-15 / 8.4e-15 | 4.7e-15 | ref | ref |
| lu_petsc | CN 1/500 nx32 | 2 | 11.0 | 4736 | 1 / 1 | 5.2 | 7.2 | 5.6e-15 / 7.6e-15 | 1.5e-14 | ref | ref |
| lu_petsc | BE 1/100 nx32 | 2 | 8.0 | 3584 | 1 / 1 | 3.8 | 5.0 | 1.6e-14 / 3.0e-14 | 1.4e-14 | ref | ref |
| lu_petsc | CN 1/100 nx64 | 2 | 8.0 | 7168 | 1 / 1 | 44.7 | 59.8 | 1.6e-13 / 2.6e-14 | 1.5e-13 | ref | ref |
| lu_petsc | CN 1/500 nx64 | 2 | 8.5 | 7552 | 1 / 1 | 46.8 | 63.5 | 1.1e-13 / 1.9e-14 | 1.9e-13 | ref | ref |
| ilu1 | CN 1/100 nx16 / nx32 / nx64 | 2 / 2 / 2 | 9.5 / 8 / 8 | = LU | 6.4/7, 5.7/6, 4.1/5 | 1.2 / 8.4 / 73.4 | 1.4 / 10.2 / 89.7 | ≤ 1.6e-13 | ≤ 1.5e-13 | ≤ 1e-13, ≤ 1e-14 | identical (12 digits) |
| ilu1 | CN 1/500 nx16 / nx32 / nx64 | 2 / 2 / 2 | 14 / 11 / 8.5 | = LU | 8.7/10, 8.8/10, 7.6/9 | 2.2 / 15.3 / 102.7 | 2.6 / 17.2 / 119.3 | ≤ 1.1e-13 | ≤ 1.9e-13 | ≤ 1e-13, ≤ 1e-14 | identical |
| ilu1 | BE 1/100 nx32 | 2 | 8 | 3584 | 6.8/8 | 9.9 | 11.2 | 1.6e-14 / 3.0e-14 | 1.4e-14 | 1e-14, 1e-16 | identical |
| asm4o1_ilu1 | CN 1/100 nx16 / nx32 / nx64 | 2 / 2 / 2 | 9.5 / 8 / 8 | = LU | 6.5/8, 6.0/6, 5.0/5 | 2.9 / 23.4 / 161.7 | 3.2 / 25.3 / 178.4 | ≤ 1.6e-13 | ≤ 1.4e-13 | ≤ 1e-13, ≤ 1e-14 | identical |
| asm4o1_ilu1 | CN 1/500 nx16 / nx32 / nx64 | 2 / 2 / 2 | 14 / 11 / 8.5 | = LU | 8.7/10, 9.1/10, 8.1/9 | 5.9 / 49.5 / 228.5 | 6.2 / 51.6 / 242.2 | ≤ 1.1e-13 | ≤ 1.9e-13 | ≤ 1e-13, ≤ 1e-14 | identical |
| asm4o1_ilu1 | BE 1/100 nx32 | 2 | 8 | 3584 | 7.2/8 | 31.5 | 32.8 | 1.5e-14 / 3.0e-14 | 1.2e-14 | 1e-13, 1e-15 | identical |

The |Δ| values are at the printed precision of the t = 1 snapshot (12
significant digits).

**Interpretation.**
- The outer/Schur behaviour is identical to exact sweeps: 2 outer iterations,
  the same Schur iterations per outer, the same number of spatial solves.
- The preconditioner does not change the converged discrete solution.
- For comparison, the production GAMG on these cases: reason −11 or a 900 s
  timeout (T4). The *healthy* GAMG case at ε = 1/20, nx = 64 took 101 s in T4.
  LU solves the much harder ε = 1/500, nx = 64 case in 47 s of solve time.

## 6. LU comparison and setup amortisation

**How often a factorisation is reused.** Each sweep context factors its block
**once** (`AdoptSpatialSolver`) and reuses it for every time slot of every
sweep application. The contexts are E (y-split), right, left (+ left_final
under CN), plus `ay` where used. A KKT run here performs 2,000–7,500 spatial
solves on 3–4 factorised matrices, i.e. **hundreds to thousands of solves per
setup**.

**Break-even against LU**, number of solves after which LU is cheaper in total
(from the scaling screen, E block θ = 1, setup and per-solve times):

| point | vs ILU(1) | vs ASM(4, o1)+ILU(1) |
|---|---|---|
| D64 | (56 − 36)/(15.0 − 3.5) ≈ **2** | LU cheaper from the first solve |
| D128 | (374 − 148)/(47.2 − 16.5) ≈ **7** | (374 − 61)/(272 − 16.5) ≈ **1.2** |

**In KKT wall time** (CN, ε = 1/500, nx = 64): LU 47 s, ILU(1) 103 s, ASM 229 s.

**Is LU competitive?** Yes. At the 2D sizes relevant to LP (≤ 66k DOFs per
level), sparse LU is the fastest robust choice by a wide margin. This follows
from setup amortisation and from 1 iteration per solve.

**When this stops holding.** In 3D, or at much larger 2D meshes, LU fill and
memory grow superlinearly. Then an iterative/AMG backend is needed. That is
not tested here.

## 7. MPI relevance (optional check, run)

Production path (`--exact-oc lp52 --solve-only`, distributed solve, JSON
output), ε = 1/100, CN, nx = 32, Nt = 64:

| backend | np | outer its | outer history | max\|Δy\| / max\|Δp\| vs np = 1 | J_track rel. vs np = 1 | wall [s] |
|---|---|---|---|---|---|---|
| lu_mumps | 1 / 2 / 4 | 2 / 2 / 2 | 1, 4.587e-08, 3.6e-14 (identical first two) | 1.6e-14 / 7e-16; 1.1e-15 / 4e-17 | 1.7e-14; 7e-16 | 7.4 / 7.2 / 5.6 |
| asm4o1_ilu1 | 1 / 2 / 4 | 2 / 2 / 2 | same | 1.6e-14 / 7e-16; 1.2e-15 / 5e-17 | 1.6e-14; 1e-15 | 21.5 / 6.3 / 4.3 |

Correctness across ranks is confirmed. The wall times are from a shared
workstation: **no scalability claim** is made.

**Classification.**

| backend | classification |
|---|---|
| ILU(k) (PCILU) | serial-only practical choice (sequential matrix only) |
| PETSc native LU, UMFPACK | serial-only |
| MUMPS LU | MPI-capable direct; moderate rank counts in 2D; not scalable to 3D/large |
| ASM(overlap 1)+ILU(1) | MPI-capable local/preconditioned choice. Iterations may grow with the number of subdomains (one level, no coarse space); untested beyond 4. |
| block-Jacobi+ILU | MPI-capable, weak (no overlap) |
| AMG (GAMG) | the only potentially scalable family available, and **it fails** on these blocks. A convection-aware AMG (hypre AIR) needs a PETSc rebuild. |

## 8. Answers

1. **Can GAMG be made robust by tuning alone?** No, not with defensible small
   tuning. SOR smoothing, unsmoothed aggregation, more smoothing and no
   aggressive coarsening rescue `Pe_cell` 2.5 and the mass-augmented blocks.
   The plain E block still fails at `Pe_cell` ≥ 6 (ε ≤ 1/100).
2. **Is hypre/BoomerAMG available?** No, not in this PETSc build.
3. **Does AIR help?** Not testable here. It needs `--download-hypre` in a new
   PETSc build, plus relinking ex21.
4. **ASM+ILU and block-Jacobi+ILU.**
   - ASM(4 subdomains, overlap 1)+ILU(1): robust everywhere, ≤ 19 iterations,
     MPI-consistent.
   - Block-Jacobi / ASM without overlap: robust but weak (≤ 91 iterations).
5. **Is direct LU competitive?** Yes, and it is the fastest end-to-end at
   these 2D sizes, because the factorisation is amortised over hundreds to
   thousands of solves.
6. **Which candidates handle both A and Aᵀ?** ILU(k), ASM+ILU, block-Jacobi+ILU
   and every LU variant. GAMG handles neither orientation of the E block at
   high Pe.
7. **Is the 1–2-outer-iteration behaviour preserved?** Yes. Every shortlisted
   backend gives 2 outer iterations with the Schur counts of exact sweeps, and
   the same discrete solution.
8. **Setup vs repeated solve.** LU has the highest setup but the cheapest
   solves. It breaks even after about 2–7 solves against ILU(1) and about 1
   against ASM, while a KKT run performs thousands. Rank backends by setup +
   N_solves × solve, not by setup alone.
9. **Most credible for MPI and future 3D?** For moderate MPI in 2D: MUMPS LU
   (exact, consistent) or ASM+ILU. For 3D and large scale: neither is
   convincing. The credible path is a convection-aware AMG (hypre AIR) or
   ASM+ILU with a coarse space. That requires a PETSc with hypre and is
   future work.
10. **One default for all Pe, or a convection-robust backend at large Pe?**
    - **Recommendation:** keep the production default (GMRES+GAMG) **for now**.
      It is exact-solution-validated at low Pe and is the scalable AMG path.
    - Document a **high-Pe spatial backend** selectable through
      `PETSC_OPTIONS` (§9), and use it for every `Pe_cell ≳ 2` run.
    - A Pe-based automatic switch would need a code hook. It was not added,
      because the options already express everything.
    - Changing the default to LU is not recommended yet: it is serial/MUMPS
      only and not 3D-capable.
11. **Is the solver issue resolved enough to move to LPS?** Yes, for the LP 2D
    problem class. With LU or ILU(1)/ASM spatial solves, the unstabilised
    high-Pe Galerkin KKT systems solve robustly (2 outer iterations) and
    reproducibly. LPS can now be studied cleanly, as a discretisation question
    with a working solver.

## 9. Recommended high-Pe spatial configurations (PETSc options; no code change)

Expand the logical options onto all four prefixes. `run_t5.py show <cfg>`
prints the full string.

| configuration | logical options | MPI? |
|---|---|---|
| **Serial, fastest (recommended at these sizes)** | `-pc_type lu` | serial |
| MPI direct | `-pc_type lu -pc_factor_mat_solver_type mumps` | MPI |
| Iterative, MPI-capable | `-pc_type asm -pc_asm_overlap 1 -sub_pc_type ilu -sub_pc_factor_levels 1` (in parallel, one block per rank; `-pc_asm_blocks` only for serial emulation) | MPI |
| Iterative, serial | `-pc_type ilu -pc_factor_levels 1` | serial |

For example, LU on every sweep:

    PETSC_OPTIONS="-sweep_right_pc_type lu -sweep_left_pc_type lu -sweep_ay_pc_type lu -fieldsplit_y_sweep_pc_type lu"

The Krylov method, the tolerance (1e-10) and maxits stay as set in the code.
With LU, GMRES converges in 1 iteration. `ksp_type preonly` (T4's `lu` variant)
is equivalent and slightly cheaper.

## 10. Reproduction

    cd aux/investigation/leveque_pearson/t5
    WORK=<scratch> python3 run_t5.py screen --jobs 3     # 52 block runs, seconds each
    WORK=<scratch> python3 run_t5.py scale  --jobs 2     # 32 block runs
    WORK=<scratch> python3 run_t5.py kkt lu_petsc        # 7 KKT runs each
    WORK=<scratch> python3 run_t5.py kkt ilu1
    WORK=<scratch> python3 run_t5.py kkt asm4o1_ilu1
    WORK=<scratch> python3 mpi_check.py                  # np = 1, 2, 4
    python3 analyze_t5.py

Total cost: block screens ≈ 3 min; KKT ≈ 15 min (the largest is ASM, nx = 64,
242 s); MPI check ≈ 1 min.

## 11. Source code

**No source change.** The stale comment in `Preconditioners.cpp`, which
describes A/A_R/A_L as SPD for CG+GAMG and is true only for the symmetric heat
case, was noted but not edited. The task allows editing it only when the file
is touched for another reason, and it was not.
