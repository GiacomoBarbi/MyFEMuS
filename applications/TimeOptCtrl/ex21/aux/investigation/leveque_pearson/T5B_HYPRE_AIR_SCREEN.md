# T5b — hypre / BoomerAMG / AIR for the convection-dominated sweep blocks

Scope: the **spatial sweep solves only** of the ex21 SCHUR_SWEEP preconditioner (blocks
E = M/dt + θL, right = E + √β N, left = (E + √β M_ω)ᵀ). The following are unchanged:

- the KKT formulation, the matching Schur preconditioner and the convection discretisation;
- the ex21 defaults, ex20 and the Aulisa/Gilliam report;
- the ex21 source: T5b has no source change. The same T4 source is rebuilt against a second PETSc arch.

T4 (`T4_HIGH_PECLET_UNSTABILISED.md`) and T5 (`T5_SPATIAL_SOLVER_SCREEN.md`) are taken
as authoritative. Every block and KKT setting is the T5 one:

- Krylov solver: GMRES(50), right preconditioning, unpreconditioned norm, rtol 1e-10, maxit 500;
- block test: 3 random + 1 smooth RHS per block;
- KKT: LP §5.2, β = 1e-2, τ = 1/nx, Nt = 2nx, production tolerance.

## Verdict

**CASE B (qualified).** One hypre configuration is robust in serial on every block and
KKT case tested, for forward and transpose blocks alike, with no per-case retuning:

    H2c hypre_air1_mrs1 = AIR (restriction_type 1) + max_row_sum 1.0

It takes 7–18 GMRES iterations at Pe_cell 1.25–62.5, both θ, and E/right/left. In 2D it
is **slower than LU and than ASM+ILU**. It is **not yet usable under MPI**: np = 1, 2 and 3
agree with each other to ≤ 2e-14, but np = 4 crashes deterministically with a SIGSEGV
inside hypre's AMG cycle. The crash disappears when the hierarchy is capped at 8 levels;
that is a diagnostic lead, not an adopted fix (see I. MPI).

Every other hypre configuration fails the survival rule:

- AIR without `max_row_sum 1.0` breaks down (KSP_DIVERGED_BREAKDOWN after 1 iteration) on the
  mass-dominated blocks, which are all blocks at D and E16 plus the θ = ½ blocks at low resolution.
- Classical BoomerAMG fails (500 iterations) on θ = 1 blocks whenever it builds a real
  hierarchy. Where it "converges" at high Pe, it has built **one level**, i.e. a single
  SOR sweep, not AMG.

The cause of the AIR breakdown is a hypre **default**, not AIR itself. BoomerAMG marks every
coupling of row i weak when |Σ_j a_ij| > max_row_sum·|a_ii|, and `max_row_sum` defaults
to 0.9. For the mass-dominated Q2 blocks this holds in 100% of rows at D (median ratio 1.09
at θ = 1 and 1.28 at θ = ½), so nothing is coarsened. Setting `max_row_sum 1.0` disables
the test. It is a single fixed option, applied identically to every point, θ and block.

## A. Build

| item | value |
|---|---|
| PETSc | 3.23.5 (same source tree as production) |
| PETSC_DIR | `/home/giacomo/software/petsc` |
| PETSC_ARCH | `arch-linux-c-openblas-hypre` (**new**; `arch-linux-c-openblas-opt` untouched) |
| hypre | 2.32.0 (`HYPRE_RELEASE_VERSION "2.32.0"`, the version PETSc 3.23.5 pins), `libHYPRE-2.32.0.so` |
| compilers | `mpicc`/`mpicxx`/`mpif90` of the existing arch → gcc 7.5.0 (SUSE) |
| MPI | Open MPI 5.0.7, **reused** from `arch-linux-c-openblas-opt` (`--with-mpi-dir`) |
| BLAS/LAPACK | `-L/usr/lib64 -lopenblas` (as production) |
| MUMPS | yes (5.7.3), with ScaLAPACK + BLACS |
| METIS / ParMETIS | yes (v5.1.0-p12 / v4.0.3-p9) |
| SuiteSparse, HDF5 | yes (7.10.1, 1.14.3-p1) |
| `make check` | passed (incl. HYPRE, MUMPS, SuiteSparse, HDF5 tests) |
| ex21 build dir | `/home/giacomo/software/FEMuS/femusbin-control-hypre` (**new**; `femusbin-control` untouched) |
| ex21 binary | `.../femusbin-control-hypre/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21`, sha256 4435f2c5… |

The exact configure command is in `t5b/build_petsc_hypre.sh`. It is the production arch's
configure (`reconfigure-arch-linux-c-openblas-opt.py`) with two changes: `--download-hypre`
is added, and `--with-mpi-dir=<existing arch>` replaces `--download-openmpi`. Package
tarballs are local because bitbucket.org is unreachable from the build sandbox; METIS and
ParMETIS were exported with `git archive` at the pinned tags.

```
env -i HOME=$HOME PATH=/usr/bin:/bin /usr/bin/python3.11 ./configure PETSC_ARCH=arch-linux-c-openblas-hypre \
  --with-mpi-dir=/home/giacomo/software/petsc/arch-linux-c-openblas-opt --download-hypre=$PKGS/v2.32.0.tar.gz \
  --download-blacs=1 --download-hdf5=$PKGS/hdf5-1.14.3-p1.tar.bz2 --download-metis=$PKGS/pkg-metis-v5.1.0-p12.tar.gz \
  --download-parmetis=$PKGS/pkg-parmetis-v4.0.3-p9.tar.gz --download-mumps=$PKGS/MUMPS_5.7.3.tar.gz \
  --download-scalapack=$PKGS/0e8767285b7a201c7b1ff34d2c2bb009534145df.tar.gz --download-suitesparse=$PKGS/v7.10.1.tar.gz \
  '--with-blaslapack-lib=-L/usr/lib64 -lopenblas' --with-debugging=0 --with-shared-libraries=1 --with-x=1 \
  'COPTFLAGS=-O3 -march=native -mtune=native' 'CXXOPTFLAGS=-O3 -march=native -mtune=native' \
  'FOPTFLAGS=-O3 -march=native -mtune=native'
```

**Baseline regression of the hypre binary.** gamg_base, ilu1 and lu_petsc at A–D give
72/72 iteration lists and reasons identical to T5 (`t5b/results/baseline.csv`).
The new binary therefore reproduces T5 exactly.

## B. hypre capabilities exposed by this PETSc (from `-help`, not guessed)

| feature | available? | exact option (prefix omitted) | notes |
|---|---|---|---|
| hypre PC | yes | `-pc_type hypre -pc_hypre_type boomeramg` | also ilu, euclid, pilut, parasails, ams, ads |
| coarsening | yes | `-pc_hypre_boomeramg_coarsen_type` | CLJP, Ruge-Stueben, modifiedRuge-Stueben, Falgout (default), PMIS, HMIS |
| interpolation | yes | `-pc_hypre_boomeramg_interp_type` | classical (default), ext+i, … ; forced to one-point (100) under AIR |
| strength threshold | yes | `-pc_hypre_boomeramg_strong_threshold` | default 0.25 |
| max row sum | yes | `-pc_hypre_boomeramg_max_row_sum` | default 0.9; **1.0 disables** the weak-row test (decisive here) |
| relaxation | yes | `-pc_hypre_boomeramg_relax_type_all/_down/_up/_coarse` | incl. l1-Gauss-Seidel, backward-l1-Gauss-Seidel, l1scaled-Jacobi, symmetric-SOR/Jacobi (default) |
| sweeps | yes | `-pc_hypre_boomeramg_grid_sweeps_all/_down/_up/_coarse` | |
| levels | yes | `-pc_hypre_boomeramg_max_levels` | |
| aggressive coarsening | yes | `-pc_hypre_boomeramg_agg_nl` | incompatible with AIR |
| ILU / Euclid smoothers | yes | `-pc_hypre_boomeramg_smooth_type` + ILU options | not used |
| statistics | yes | `-pc_hypre_boomeramg_print_statistics` | used for the level counts |
| no C/F relaxation | yes | `-pc_hypre_boomeramg_no_CF` | |
| keep transpose | yes | `-pc_hypre_boomeramg_keeptranspose` | not needed |

## C. AIR availability

AIR **is exposed** by PETSc 3.23.5 + hypre 2.32.0:

| feature | available? | exact option | notes |
|---|---|---|---|
| AIR distance-1 / distance-2 | yes | `-pc_hypre_boomeramg_restriction_type 1` / `2` | 0 = classical P^T (default) |
| one-point interpolation | automatic | – | PETSc forces interp_type 100 when restriction_type ≠ 0 |
| R strength threshold | yes (only when AIR on) | `-pc_hypre_boomeramg_strongthresholdR` | default kept |
| R filter threshold | yes (AIR on) | `-pc_hypre_boomeramg_filterthresholdR` | default kept |
| A drop tolerance / type | yes (AIR on) | `-pc_hypre_boomeramg_Adroptol`, `-pc_hypre_boomeramg_Adroptype` | default kept |
| F/C relaxation ordering | yes (AIR on) | `-pc_hypre_boomeramg_prerelax`, `-pc_hypre_boomeramg_postrelax` (strings of F, C, A) | string length must equal `grid_sweeps_down/up`; unset = 0 sweeps |

PETSc's only AIR test is `src/ksp/ksp/tutorials/ex86.c`, 1D upwind.

## D. Configurations

All are applied with exactly the same string to every block (E, right, left), θ and point.
The prefixes are `block_test_` in the block test and `sweep_right_`, `sweep_left_`, `sweep_ay_`
and `fieldsplit_y_sweep_` in the KKT.

| id | name | options (without prefix) |
|---|---|---|
| H0 | hypre_default | `-pc_type hypre -pc_hypre_type boomeramg` |
| H1 | hypre_ns | H0 + `-pc_hypre_boomeramg_coarsen_type HMIS -pc_hypre_boomeramg_interp_type ext+i -pc_hypre_boomeramg_relax_type_all l1-Gauss-Seidel -pc_hypre_boomeramg_strong_threshold 0.5` |
| H2 | hypre_air1 | H0 + `-pc_hypre_boomeramg_restriction_type 1 -pc_hypre_boomeramg_grid_sweeps_down 0 -pc_hypre_boomeramg_grid_sweeps_up 3 -pc_hypre_boomeramg_postrelax F,F,C -pc_hypre_boomeramg_relax_type_all l1scaled-Jacobi` |
| H2b | hypre_air2 | as H2 with `restriction_type 2` |
| H3 | hypre_air1_gs | as H2 with `relax_type_all l1-Gauss-Seidel` |
| H2c | **hypre_air1_mrs1** | H2 + `-pc_hypre_boomeramg_max_row_sum 1.0` |

H2c was added after the diagnostics below had identified `max_row_sum` as the cause of the
breakdown. It then went through the full screen, the T5 scale points, the KKT set and the
MPI check under the same survival rule. The search was H0–H3 plus one diagnosed option;
there was no option grid.

## E. Block screen

Points:

- A: ε 1/20, nx 32
- B: ε 1/20, nx 16
- C: ε 1/100, nx 32
- D: ε 1/500, nx 32 (Pe 31)
- E16: ε 1/500, nx 16 (Pe 62.5)
- H2c additionally at C64 (ε 1/100, nx 64), D64 (ε 1/500, nx 64) and D128 (ε 1/500, nx 128).

Entries are the max iterations over the 4 RHS for the blocks `θ=1: E right left | θ=½: E right left`.
"brk" = KSP_DIVERGED_BREAKDOWN (−5) at iteration 1; "500!" = not converged. The levels
come from `-pc_hypre_boomeramg_print_statistics`.

| config | A | B | C | D | E16 | survives? |
|---|---|---|---|---|---|---|
| H0 hypre_default | 8 7 7 \| 6 6 6 | 6 6 5 \| 5 4 4 | 20 21 20 \| 15 11 11 | 55 37 37 \| 25 20 20 **(1 level)** | 67 32 32 \| 26 19 19 **(1 level)** | **no**: converges only as 1-level SOR at D/E16; fails (500) on θ = 1 blocks as soon as a hierarchy exists (C16 E, D64, D128 — see diagnostics) |
| H1 hypre_ns | ≤ 11 | ≤ 10 | **500!** 89 91 \| 31 19 20 | 72 59 58 \| 45 39 40 (1 level) | 77 50 49 \| 44 35 35 (1 level) | **no** |
| H2 hypre_air1 | 15 13 13 \| 11 10 10 | 9 8 8 \| 8 7 7 | 10 9 9 \| 9 10 10 | **brk** ×6 (1 level) | **brk** ×6 (1 level) | **no** |
| H2b hypre_air2 | ≈ H2 | ≈ H2 | ≈ H2 | **brk** ×6 | **brk** ×6 | **no** |
| H3 hypre_air1_gs | 11 10 10 \| 9 8 8 | 8 7 7 \| 7 6 6 | 11 11 11 \| 10 10 10 | = H1 (1 level, l1-GS) | = H1 (1 level) | **no** (single-level GS at D/E16, not AMG) |
| **H2c hypre_air1_mrs1** | 15 13 13 \| 11 10 10 | 9 8 8 \| 8 7 7 | 10 9 9 \| 8 7 7 | 16 12 12 \| 12 11 11 | 18 12 13 \| 12 11 12 | **yes** |

H2c at the T5 scale points (all RHS converged):

- C64: 10 9 9 \| 8 8 8
- D64: 15 13 13 \| 10 10 10
- D128: 11 11 11 \| 9 9 8

**0 of 48** H2c blocks have a 1-level hierarchy (11–16 levels at the failing points).

T5 references (max iterations over all points and blocks):

- LU: 1.
- ILU(1): ≤ 9 up to D128.
- ASM(4, o1)+ILU(1): ≤ 9.
- GAMG: not rescuable.

H2c's iteration counts at ε = 1/500 (E, θ = 1) are 18, 16, 15 and 11 for nx = 16, 32,
64 and 128. They are bounded, but Pe_cell also falls as nx rises, so this is **not** an
h-independence demonstration.

Cost per E block at D128 (θ = 1; setup s / solve s per RHS):

| solver | setup s | solve s per RHS |
|---|---|---|
| H2c | 0.22 | 0.25 |
| LU (PETSc) | 0.37 | 0.017 |
| ILU(1) | 0.15 | 0.047 |
| ASM+ILU | 0.06 | 0.27 |

### Diagnostics (why H0–H3 fail)

1. **Hierarchy check** (`t5b/results/hierarchy.csv`). At D and E16 *every* configuration H0–H3
   builds **1 level** for all six blocks. At A–C they build 4–8 levels. So the H0 "convergence" at
   D and E16 is a single symmetric-SOR/Jacobi sweep, and AIR's one-point interpolation has
   nothing to work with, hence the breakdown.
2. **Strength probe** (`logs/strength_probe.log`). `strong_threshold` 0.1, 0.05 and 0.01 at D still give
   1 level (H0 and H2).
3. **Coarsening probe** (`logs/coarsen_type_probe.log`). HMIS, PMIS, CLJP and Ruge-Stueben at D still
   give 1 level (H2).
4. **Coarsening onset** (`logs/coarsening_onset.log`). These are the points where some blocks coarsen.
   When classical BoomerAMG *does* coarsen a θ = 1 block it fails:
   - H0 at ε 1/100, nx 16: E 500!;
   - H0 at ε 1/500, nx 64 and nx 128: E, right, left all 500!.

   H2 (AIR) converges in 11–23 iterations wherever it coarsens. It breaks down only on the
   uncoarsened blocks: θ = ½ at ε 1/100 nx 16 and at ε 1/500 nx 64.
5. **Sign structure** (`logs/sign_structure.log`). This was a first hypothesis: positive dominant
   off-diagonals defeating classical strength. The correlation is only partial:
   - nx 128 θ = ½ has 98.4% positive-dominant rows but coarsens;
   - almost every row has negative off-diagonals.

   **This hypothesis is superseded by 6.**
6. **max_row_sum** (`logs/rowsum_structure.log`, `logs/rowsum_probe.log`, `results/rowsum_probe.csv`).
   BoomerAMG makes all couplings of row i weak when |Σ_j a_ij| / |a_ii| > max_row_sum = 0.9.
   For A = M/dt + θL at τ = 1/nx:

   | ε | nx | θ | rows with ratio > 0.9 | median ratio |
   |---|---|---|---|---|
   | 1/100 | 32 | 1 | 0.0% | 0.49 |
   | 1/100 | 32 | ½ | 3.1% | 0.74 |
   | 1/500 | 32 | 1 | **100%** | 1.09 |
   | 1/500 | 32 | ½ | **100%** | 1.28 |
   | 1/500 | 128 | 1 | 0.0% | 0.57 |
   | 1/500 | 128 | ½ | 25.2% | 0.83 |

   This matches the level counts (1 level exactly where the ratio exceeds 0.9 everywhere).
   With `max_row_sum 1.0`:
   - H0 coarsens everywhere (9–14 levels) but then **fails** on θ = 1 blocks at D, E16, C16 and D64
     (500!) and needs 245–431 iterations at θ = ½ E at D and D64. This confirms that classical
     BoomerAMG is the wrong method for these operators.
   - H2 coarsens everywhere (10–16 levels) and converges in 8–18 iterations at C, D, E16, C16 and D64,
     for all blocks and both θ.

   The ratio exceeds 1 because, when diffusion is small relative to M/dt, the
   positive Q2 mass off-diagonals (plus the Galerkin convection) dominate the row sum. This is
   a property of the unstabilised Q2 mass-dominated operator. It is not a θ- or orientation-specific
   effect.

## F. Forward vs transpose

There is **no orientation asymmetry** for AIR when it coarsens. right (E + √β N, forward) and
left ((E + √β M_ω)ᵀ, transpose) differ by at most 1 iteration in every H2c row, and likewise
in the H2 rows where it coarsens:

- D: 12/12 (θ = 1), 11/11 (θ = ½);
- E16: 12/13, 11/12;
- D64: 13/13, 10/10;
- D128: 11/11, 9/8.

In the KKT runs the per-solve maxima are right 8–12 and left 8–12, with summed iteration
counts within 6% (e.g. ε 1/500 nx 64: 21349 vs 20060). The **same option string** serves
both orientations: `keeptranspose`, a different relaxation and directional retuning were not
needed. H0's failures (C16, D64 and D128 with max_row_sum 0.9; D, E16 and C16 with 1.0) hit
E, right and left alike when they occur at θ = 1, so they are not an orientation effect
either. CASE C (forward-only) does not apply.

## G. KKT results (LP §5.2, β = 1e-2, production tolerance, np = 1)

Carried:

- best standard BoomerAMG: H0 (the only BoomerAMG config passing the core points, with the caveat above);
- AIR as first screened (H2), to document its failure in context;
- the survivor H2c;
- references: T5 LU (`lu_petsc`) and ASM+ILU (`asm4o1_ilu1`).

nx = 64 was run because nx = 16 and 32 were healthy for H2c.

| config | scheme | ε | nx | reason | outer | sweep fail. | spatial its mean/max | true res. | solve wall s | RSS MB | max\|Δy\| vs LU | max\|Δp\| vs LU | J_track, J_reg rel. Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **H2c** | CN | 1/100 | 16 | 2 | 2 | 0 | 7.6/8 | 6.0e-15 | 5.6 | 115 | 1e-15 | 1e-16 | 0, 0 |
| **H2c** | CN | 1/100 | 32 | 2 | 2 | 0 | 6.0/7 | 3.2e-14 | 36.1 | 626 | 1e-13 | 1e-14 | 0, 0 |
| **H2c** | CN | 1/500 | 16 | 2 | 2 | 0 | 11.2/12 | 8.6e-15 | 11.6 | 120 | 1e-13 | 1e-15 | 0, 0 |
| **H2c** | CN | 1/500 | 32 | 2 | 2 | 0 | 10.0/11 | 9.4e-15 | 73.2 | 645 | 1e-14 | 1e-15 | 0, 0 |
| **H2c** | BE | 1/100 | 32 | 2 | 2 | 0 | 8.5/9 | 3.4e-14 | 45.1 | 616 | 1e-13 | 1e-15 | 0, 0 |
| **H2c** | CN | 1/100 | 64 | 2 | 2 | 0 | 7.1/8 | 2.8e-13 | 334.6 | 4516 | 1e-13 | 1e-14 | 0, 0 |
| **H2c** | CN | 1/500 | 64 | 2 | 2 | 0 | 8.5/9 | 1.1e-13 | 445.3 | 4524 | 1e-13 | 1e-14 | 0, 0 |
| H0 | CN | 1/100 | 16/32/64 | 2 | 2 | 0 | 12.4/13, 10.0/11, 5.0/6 | ≤ 2.8e-13 | 1.7, 47.6, 382.8 | | ≤ 1e-13 | ≤ 1e-14 | 0, 0 |
| H0 | CN | 1/500 | 16/32/64 | 2 | 2 | 0 | 17.8/21, 17.9/21, 13.5/18 | ≤ 1.1e-13 | 3.6, 23.9, 149.8 | | ≤ 1e-13 | ≤ 1e-14 | 0, 0 |
| H0 | BE | 1/100 | 32 | 2 | 2 | 0 | 20.0/21 | 3.3e-14 | 124.3 | 614 | 1e-13 | 1e-15 | 0, 0 |
| H2 | CN | 1/100 | 16 | **3** | 1 | **96** | 0.5/1 | **4.8e-1** | 0.08 | | 6.3e-1 | 1.7e-2 | **19×** |
| H2 | CN | 1/500 | 16/32/64 | **3** | 1 | **96/192/384** | 0.5/1 | **0.18–0.37** | | | 0.63–0.89 | 0.02–0.03 | **20–29×** |
| H2 | CN, BE | 1/100 | 32, 64 | 2 | 2 | 0 | 7.1–8.8/9 | ≤ 2.8e-13 | 29.7–354.3 | | ≤ 1e-13 | ≤ 1e-14 | 0, 0 |
| LU (T5) | CN | 1/100 | 16/32/64 | 2 | 2 | 0 | 1/1 | | 0.44, 3.7, 44.7 | 109–4495 | ref | ref | ref |
| LU (T5) | CN | 1/500 | 16/32/64 | 2 | 2 | 0 | 1/1 | | 0.68, 5.2, 46.8 | | ref | ref | ref |
| LU (T5) | BE | 1/100 | 32 | 2 | 2 | 0 | 1/1 | | 3.8 | 606 | ref | ref | ref |
| ASM+ILU (T5) | CN | 1/100 | 16/32/64 | 2 | 2 | 0 | 6.1/7, 6.0/6, 5.0/5 | | 2.9, 23.4, 161.7 | | ≤ 1e-13 | ≤ 1e-14 | 0, 0 |
| ASM+ILU (T5) | CN | 1/500 | 16/32/64 | 2 | 2 | 0 | 8.0/9, 8.6/9, 8.0/9 | | 5.9, 49.5, 228.5 | | ≤ 1e-13 | ≤ 1e-14 | 0, 0 |
| ASM+ILU (T5) | BE | 1/100 | 32 | 2 | 2 | 0 | 7.0/7 | | 31.5 | 632 | 1e-13 | 1e-15 | 0, 0 |

Full rows, including Schur and y-split counts and the outer histories, are in `t5b_hypre_kkt.csv`.

- The Δ columns are the max differences between snapshot CSVs, which are written with 12
  significant digits, so 1e-13 to 1e-15 is the snapshot resolution. J_track and J_reg agree
  to all 13 printed digits.
- Every converged run has the same outer history as LU: `1.00e+00 4.59e-08 ~3e-14` at
  ε = 1/100 nx = 32.
- **Preconditioning does not alter the discrete solution.**

**Hazard found (H2 rows).** When a sweep solve breaks down, the sweep returns ~0 and the
outer FGMRES sees an *estimated* residual of exactly 0, so it stops with
`KSP_CONVERGED_ATOL` (reason 3) after 1 iteration. The **true** residual is still 0.18–0.48
and the solution is wrong (J_track off by a factor of 19–29). ex21 does print
`sweep_failures=96…384` and the true residual, but the exact-OC driver's exit code tests
only `reason > 0 && finite`, so the process exits 0.

Any automated use of a non-direct sweep backend must therefore check `sweep_failures == 0`
and the true residual, not only the reason. `analyze_t5b.py` does this (`genuinely_converged`).
This needs no change for T5b, but it is worth a guard (refuse reason > 0 when
`sweep_failures > 0`) before any non-direct backend is used routinely.

## H. Comparison to LU and ASM+ILU

- **Robustness.** H2c ties LU and ASM+ILU: all block and KKT cases pass, with 2 outer
  iterations and identical solutions. No other hypre configuration does.
- **Iterations.** H2c needs 7–18 per block solve; ASM+ILU needs ≤ 9 and LU 1. H2c is not
  better than ASM+ILU at these sizes.
- **Time (serial 2D).** H2c is slower than both:

  | case | H2c | ASM+ILU | LU |
  |---|---|---|---|
  | KKT CN ε 1/500 nx 64 | 445 s | 229 s | 47 s |
  | KKT CN ε 1/100 nx 64 | 335 s | 162 s | 45 s |

  LU wins because the ex21 sweep reuses one factorisation for thousands of right-hand sides
  (2432 right + 2432 left solves at nx 64). H2c pays 8–9 V-cycles with 3 F/F/C post-sweeps
  per solve.
- **Memory.** All are comparable at these sizes (≈ 4.5 GB at nx 64, dominated by the AAO system,
  not the preconditioner).
- **What H2c offers.** It is the only *multilevel* option found that is robust on the
  unstabilised high-Pe blocks, in both orientations, with a bounded iteration count from nx 16 to
  128. Its value would be 3D, or large MPI runs, where LU fill and ASM/ILU iteration growth
  become limiting. Neither was demonstrated here.

## I. MPI (LP §5.2, ε = 1/100, CN, nx = 32, Nt = 64; same harness as T5's `mpi_check.py`)

Candidate H2c (`hypre_air1_mrs1`), with T5's `lu_mumps` and ASM+ILU results for reference
(`t5b/results/mpi_check_t5b.csv`, `mpi_check_t5b_isolate.csv`; logs `t5b/logs/mpi_*`):

| config | np | status | wall s | outer | reason | true res. | max\|Δy\| | max\|Δp\| | J_track rel. Δ |
|---|---|---|---|---|---|---|---|---|---|
| H2c | 1 | done | 35.5 | 2 | 2 | 3.2e-14 | ref | ref | ref |
| H2c | 2 | done | 24.6 | 2 | 2 | 2.8e-14 | 1.6e-14 vs np 1 | 6.8e-16 | 1.6e-14 |
| H2c | 3 | done | 23.8 | 2 | 2 | 2.9e-14 | – | – | J equal to np 1 to 1e-14 |
| H2c | **4** | **SIGSEGV on rank 3 → hang** (killed at 300 s; reproduced twice) | – | – | – | – | – | – | – |
| H2 hypre_air1 | 4 / 3 | done | 14.6 / 16.2 | 2 | 2 | 3.3e-14 / 2.9e-14 | 1.9e-14 (np 3 vs np 4) | 8.4e-16 | 2.0e-14 |
| H0 hypre_default | 4 / 3 | done | 20.5 / 22.3 | 2 | 2 | 3.2e-14 / 2.9e-14 | 1.9e-14 (np 3 vs np 4) | 8.4e-16 | 1.9e-14 |
| T5 lu_mumps | 1/2/4 | done | 7.4/7.2/5.6 | 2 | 2 | ≤ 3.3e-14 | | | |
| T5 asm4o1_ilu1 | 1/2/4 | done | 21.5/6.3/4.3 | 2 | 2 | ≤ 3.3e-14 | | | |

All converged runs have the outer history `1.000e+00 4.587e-08 ~3e-14` and J_track = J_reg to 13–14 digits.

**The np = 4 failure of H2c is an MPI-only crash, not a convergence failure.**

- Stack traces taken with gdb from the hung run:
  - rank 3 receives SIGSEGV inside `hypre_ParCSRMatrixMatvecOutOfPlaceHost` ←
    `hypre_BoomerAMGCycle` ← `PCApply_HYPRE` (heap corruption in `calloc`/`unlink_chunk`);
  - it then deadlocks inside PETSc's signal handler (malloc under a held heap lock), so the job hangs
    instead of aborting;
  - the other ranks wait in `MPI_Waitall` in `hypre_ParCSRCommHandleDestroy`.
- The crash is specific to the combination. Each of these runs:
  - the same options at np = 3;
  - AIR without `max_row_sum 1.0` at np = 4 (at this ε = 1/100 point AIR coarsens without it);
  - default BoomerAMG at np = 4.
- Diagnostic: H2c + `-pc_hypre_boomeramg_max_levels 8` at np = 4 **runs** (2 outer iterations, sweep max 7,
  16.1 s; `t5b/logs/mpi_probe_air1_mrs1_maxlevels8_np4.log`). This points at the very deep
  (12–16 level) AIR hierarchies that `max_row_sum 1.0` produces, whose coarsest levels have
  near-empty or empty rank partitions at np = 4.
- This is a plausible hypre 2.32.0 AIR defect on such levels. It is *not* established.
- `max_levels 8` is **not** adopted. It changes the coarse problem (with `relax_type_all
  l1scaled-Jacobi` the coarsest level is relaxed, not solved), so it would require rescreening
  the whole block and KKT set. That was outside this task's no-grid budget.

Note that "max|Δy| vs np 1" in the isolation CSV compares np 3 with np 4 of the same configuration
(no np 1 run in that batch).

No scaling claims are made (workstation, oversubscription-free but shared).

## J. Recommended high-Pe scalable backend

- **Serial, or any 2D size tested: keep LU.** It is fastest by 5–10× and exact per sweep.
- **MPI baseline: keep ASM(4, overlap 1)+ILU(1)** (T5: np 1/2/4 consistent to round-off).
- **Scalable candidate for later 3D/MPI work:** H2c, i.e. hypre BoomerAMG AIR
  (`restriction_type 1`, one-point interpolation, `grid_sweeps_down 0`, `grid_sweeps_up 3`,
  `postrelax F,F,C`, `relax_type_all l1scaled-Jacobi`) **with `max_row_sum 1.0`**. It is a
  candidate only, and is blocked until the np = 4 crash is resolved. Next steps: a debug or
  valgrind build of hypre, a newer hypre release, and a properly screened coarse-level
  treatment (`max_levels` plus a direct coarse solve) instead of the ad-hoc cap.
- Avoid classical BoomerAMG (H0/H1) for these blocks.
- Never use AIR without `max_row_sum 1.0` on mass-dominated blocks: it breaks down silently.

## K. Source change needed?

**No.** Everything was done through `PETSC_OPTIONS` on the existing prefixes. The only source
change worth considering later, outside T5b's scope, is the outer-convergence guard above:
treat `sweep_failures > 0` as failure.

## L. Defaults unchanged?

**Yes.** The ex21 defaults (low-Pe GAMG) and the source are unchanged, and the production
binary (`femusbin-control`, sha256 500a2b2e…) and the PETSc arch `arch-linux-c-openblas-opt`
are untouched. hypre lives only in the new arch and the new build directory.

## M. Ready for LPS?

**Yes, on the solver side, for the 2D LP §5.2 regime**, as already concluded in T5:

- LU (serial) and ASM+ILU (MPI) solve every unstabilised high-Pe sweep block and KKT case,
  and give solutions identical to round-off.
- T5b adds a robust multilevel candidate (H2c) but does not change that conclusion. It
  does not resolve MPI-scalable AMG for convection, because of the np = 4 crash.

That open item does **not** block LPS. LPS changes the discretisation, not the solver
structure; the stabilised blocks should be rescreened with the same harness (LU, ASM+ILU,
H2c), since LPS adds symmetric positive semi-definite terms that will change the row sums.

The high-Pe Galerkin oscillations identified in T4 remain; they are a discretisation issue
that no spatial solver removes.

## N. Files

`aux/investigation/leveque_pearson/`:

- `T5B_HYPRE_AIR_SCREEN.md` (this report), `t5b_hypre_block_screen.csv`, `t5b_hypre_kkt.csv`
- `figures/t5b_iterations_vs_peclet.png`, `figures/t5b_time_vs_backend.png`
- `t5b/build_petsc_hypre.sh`: exact PETSc configure and build
- `t5b/reproduce_t5b.sh`: full reproduction sequence (FEMuS build, campaign, analysis)
- `t5b/run_t5b.py`: harness (imports `t5/run_t5.py`); configurations H0–H3, H2c; modes
  baseline, screen, screen1, kkt, show
- `t5b/hierarchy_check.py`, `t5b/rowsum_probe.py`, `t5b/rowsum_structure.py`: diagnostics
- `t5b/mpi_check_t5b.py`: MPI check with hang/crash recording
- `t5b/analyze_t5b.py`: tables and figures
- `t5b/results/*.csv`: baseline, screen, screen_hypre_air1_mrs1, hierarchy, rowsum_probe,
  kkt_*, mpi_check_t5b*, snap_*
- `t5b/logs/`: campaign logs and probe logs

The probe logs `strength_probe.log`, `coarsen_type_probe.log`, `coarsening_onset.log` and
`sign_structure.log` came from one-off runs of the block test with the option shown in each
log line, appended to the H0/H2 strings.
