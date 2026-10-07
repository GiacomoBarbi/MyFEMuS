# Solver robustness across the Aulisa/Gilliam convection–diffusion benchmarks (ex21)

Purpose: a self-contained audit for inclusion in the consolidated Aulisa/Gilliam
comparison report. It separates **solver efficiency** from **mathematical
correctness**, and it records which benchmarks stress the SCHUR_SWEEP
preconditioner.

Data:
* `results/aulisa_gilliam_solver_summary.csv` (`scripts/solver_summary.py`),
  built from the existing production logs; no new production runs;
* the EF 9.6 diagnostics in `errorfeedback_95_96/solver_diag/`;
* the MPI check in `errorfeedback_95_96/solver_diag/mpi/`.

Wall times were measured on a shared 12-core node, with up to 12 concurrent
serial jobs. They are comparable only to within a factor of about 2.

## 1. The linear system and the preconditioner

Discretize-then-optimize all-at-once (AAO) system after eliminating the
control `u = −(η/α_u) p`:

    [ E    S  ] [y]   [f]
    [ T   Eᵀ  ] [p] = [g]

* `E` is the space–time state operator, block lower-bidiagonal in time, with
  CN or BE diagonal blocks `M/dt + θL`. Because of convection, `L = K + C`
  (with `C` skew) is **nonsymmetric**, so `E ≠ Eᵀ`.
* `S ∝ (η²/α_u) N` is the control coupling, with `N` the mass on `Ω_b`.
* `T` is the tracking coupling, `M_ω` on `Ω_c` with the CN end weights.

**Solver (SCHUR_SWEEP):**
* outer FGMRES(30), rtol 1e-8, at most 200 iterations;
* PCFIELDSPLIT Schur, FULL factorisation;
* the `y` split is an exact forward time sweep;
* the Schur split is **GMRES(30), rtol 1e-6, capped at 100 iterations**, one
  solve per outer iteration;
* the Schur split is preconditioned by the matching approximation
  `Ŝ⁻¹ = right⁻¹ E left⁻¹`, with `left = Eᵀ + √β M_ω` and `right = E + √β N`;
* each `left` / `right` / `E` application is an exact triangular time sweep
  of spatial GMRES(50) + GAMG solves (rtol 1e-10, PC_RIGHT, cap 500).

**Cost structure.** One outer iteration costs ≈ (Schur iterations + 1) × 3
time sweeps, and each sweep is `N` spatial solves.

## 2. Correctness is established independently of solver efficiency

For every benchmark configuration the following checks hold. None of them
depends on how fast the Krylov solvers converge.

* exact discrete transpose: shell transpose `|⟨Ex,y⟩ − ⟨x,Eᵀy⟩|` ≤ 3e-17;
* reduced-Hessian symmetry ≤ 2e-12;
* FD reduced gradient ≤ 1e-13, with the ex20-style untransposed adjoint
  failing by 23–290 % as a negative control;
* zero-convection regression (61/61 ex20 goldens identical; `v = 0` baseline
  identical);
* manufactured-solution convergence (`EX21_CONVECTION_VALIDATION.md`);
* an independent LU forward march reproducing the production state to
  ≤ 6e-12;
* the **true** (unpreconditioned) KKT residual of every completed solve,
  reported below.

**Distinction kept throughout.** A slow or capped inner Krylov solve is a
statement about *preconditioner efficiency and parameter robustness*. A
completed solve whose true residual is small is a valid discrete optimum,
however expensive it was. Solver inefficiency does not indicate an incorrect
formulation or discretisation.

## 3. Cross-benchmark solver table (SCHUR_SWEEP, completed production solves)

`s/o` = Schur-split GMRES iterations per outer iteration; 100 means every
Schur solve stopped at its cap. Spatial KSP iterations were not recorded by
the production runs; the instrumented EF 9.6 runs are in §5.

| benchmark | dim | colocated | `ν` | `v` | Pe_cell | `α_u` | mesh | N | scheme | outer | Schur its (s/o) | cap hit | true residual | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M25 Ex3 | 1 | no | 2 | 2 | 3.1e-3 | 1e-3 | 160 | 800 | BE | 2 | 15 (7.5) | no | 5.1e-12 | 35 |
| M25 Ex3 | 1 | no | 2 | 2 | 3.1e-3 | 1e-5 | 160 | 800 | BE | 2 | 29 (14.5) | no | 3.0e-11 | 61 |
| M25 Ex3 | 1 | no | 2 | 2 | 3.1e-3 | 1e-7 | 160 | 800 | BE | 2 | 61 (30.5) | no | 3.2e-11 | 117 |
| M25 Ex3 | 1 | no | 2 | 2 | 3.1e-3 | 1e-9 | 160 | 800 | BE | 2 | 175 (87.5) | partly | 3.2e-11 | 264 |
| M25 Ex3 | 1 | no | 2 | 2 | 3.1e-3 | **1e-11** | 160 | 800 | BE | **5** | 500 (100) | yes | 4.4e-10 | 577 |
| M25 Ex3 (primary) | 1 | no | 2 | 2 | 1.3e-2 | 1e-9 | 40 | 1600 | CN | 7 | 700 (100) | yes | 7.1e-9 | 557 |
| M25 Ex4 | 1 | yes | 2 | 2 | 7.8e-4 | 1e-3 … 1e-9 | 640 | 1200 | CN | 2 | 13–16 (6.5–8) | no | ≤ 1.3e-10 | 157–187 |
| M25 Ex4 | 1 | yes | 2 | 2 | 7.8e-4 | **1e-11** | 640 | 1200 | CN | **2** | 15 (7.5) | no | 1.3e-10 | 165 |
| M25 Ex5 | 2 | no | 2 | (2,2) | 4.6e-2 | 1e-7 | 48×16 | 400 | CN | 3 | 300 (100) | yes | 9.2e-11 | 4700 |
| EF 9.5 | 2 | yes | 1 | (2,1) | 0.11 | 1e-3 … 1e-9 | 32×12 | 400 | CN | 2 | 15–17 (7.5–8.5) | no | ≤ 1.6e-13 | 121–156 |
| EF 9.5 | 2 | yes | 1 | (2,1) | 0.11 | 1e-10 / **1e-11** | 32×12 | 400 | CN | **2** | 12 / 9 | no | ≤ 9.2e-14 | 88–121 |
| EF 9.5 | 2 | yes | 1 | (2,1) | 0.027 | 1e-7 … **1e-11** | 128×48 | 200 | CN | **2** | 14–16 (7–8) | no | ≤ 2.3e-13 | 1150–1930 |
| EF 9.6 | 2 | no | 1 | (2,1) | 0.11 | 1e-3 | 32×12 | 400 | CN | 2 | 17 (8.5) | no | 5.1e-13 | 166 |
| EF 9.6 | 2 | no | 1 | (2,1) | 0.11 | 1e-4 | 32×12 | 400 | CN | 2 | 34 (17) | no | 2.1e-13 | 290 |
| EF 9.6 | 2 | no | 1 | (2,1) | 0.11 | 1e-5 | 32×12 | 400 | CN | 2 | 77 (38.5) | no | 2.5e-13 | 515 |
| EF 9.6 | 2 | no | 1 | (2,1) | 0.11 | 1e-6 | 32×12 | 400 | CN | 2 | 168 (84) | partly | 2.8e-12 | 1269 |
| EF 9.6 | 2 | no | 1 | (2,1) | 0.11 | 1e-7 | 32×12 | 400 | CN | 3 | 300 (100) | yes | 1.4e-9 | 2726 |
| EF 9.6 | 2 | no | 1 | (2,1) | 0.11 | 1e-8 | 32×12 | 400 | CN | 7 | 700 (100) | yes | 6.0e-9 | 6337 |
| EF 9.6 | 2 | no | 1 | (2,1) | 0.11 | **1e-9** | 32×12 | 400 | CN | **18** | 1800 (100) | yes | 8.4e-9 | 11534 |
| EF 9.6 | 2 | no | 1 | (2,1) | 0.11 | **1e-10** | 32×12 | 400 | CN | **70** | 7000 (100) | yes | **8.0e-9** | 24483 |
| EF 9.6 | 2 | no | 1 | (2,1) | 0.11 | **1e-11** | 32×12 | 400 | CN | — | — | (yes, §5) | **aborted** | > 37515 |
| EF 9.6 | 2 | no | 1 | (2,1) | 0.055 | 1e-7 | 64×24 | 200 / 400 | CN | 5 / 3 | 500 / 300 (100) | yes | 1.9e-9 / 1.7e-9 | 6746 / 7211 |

* All completed solves have reason 2 (rtol reached) and zero spatial-sweep
  failures.
* "partly": `s/o` between 50 and 100, i.e. some Schur solves were capped.
* The EF 9.6 `α_u = 1e-11` production run was **stopped by request after
  10.4 h**. No value from it is used.

## 4. Where the degradation occurs: case B, not case A

**Answer: the evidence supports case B.** The degradation is localised to a
strongly ill-conditioned configuration; it is not a general failure of the ex21
convection–diffusion preconditioner.

* **Colocated benchmarks are fully robust.** M25 Ex4 (1D) and EF 9.5 (2D, the
  same plant, mesh and data as EF 9.6) take 2 outer iterations and ≈ 8 Schur
  iterations per outer iteration at every `α_u` from 1e-3 down to **1e-11**,
  with true residuals ≤ 2e-10. This covers 8 decades of `α_u` and meshes up
  to 128×48.
* **Non-colocation is a stressor, but not by itself the cause.** In every
  non-colocated case (M25 Ex3, M25 Ex5, EF 9.6) the Schur iterations per outer
  iteration grow as `α_u` decreases, roughly doubling per decade, until the
  100-iteration cap is hit:
  * M25 Ex3: 7.5 → 14.5 → 30.5 → 87.5 → 100 for `α_u` = 1e-3 … 1e-11;
  * EF 9.6: 8.5 → 17 → 38.5 → 84 → 100 for 1e-3 … 1e-7.

  This is consistent with the structure of the matching approximation, which
  pairs `N` with `M_ω`. When the two coincide (colocated) the pairing is exact
  in the regularisation scaling; when they live on disjoint regions it is not.
  This structural observation is not a proved spectral bound.
* **The outer FGMRES degradation is specific to EF 9.6 among the tested
  cases.** With the Schur solve capped, the outer iteration must absorb the
  inexactness.
  * M25 Ex3 tolerates this: **5 outer iterations at `α_u = 1e-11`** (BE,
    N = 800), and 7 at 1e-9 (CN, N = 1600).
  * EF 9.6 does not: 3 → 7 → 18 → 70 outer iterations for `α_u` = 1e-7 …
    1e-10 (≈ 4× per decade at the small end), and for 1e-11 the outer
    residual stagnates (§5).
  * M25 Ex5 was run only at its source value `α_u = 1e-7`, with 3 outer
    iterations, the same as EF 9.6 at 1e-7. No inference is made about 2D
    non-colocation at smaller `α_u` beyond EF 9.6.
* **The relevant stressor is therefore the combination:** non-colocation + 2D
  + very small `α_u` + this particular input/output configuration. In EF 9.6
  that configuration is a narrow `Ω_B` upstream of a narrow `Ω_C`, with
  a Gaussian target of width `σ = 0.15`, and a tail that shows practical
  saturation over the tested regularisation range
  (`ERRORFEEDBACK_96_BENCHMARK.md`). Non-colocation alone (M25 Ex3, 1D) is not
  sufficient. Whether 2D is essential cannot be separated with the present
  data: Ex5 was not pushed below 1e-7.
* **Not convection strength.** All cell Péclet numbers are ≤ 0.11 and standard
  Galerkin is stable throughout. Colocated EF 9.5, with the same velocity
  field and Péclet number as EF 9.6, is fully robust.

## 5. EF 9.6 mechanism (diagnostics)

Instrumented runs (`solver_diag/run_diag.sh`): binary b6b6eb86…; nx 32 × nz
12, CN, SCHUR_SWEEP; driver path, np = 1; algorithm unchanged. Instrumentation:
* existing sweep and Schur counters, printed by the driver;
* per-spatial-solve iteration counts, aggregated per time slot by awk;
* PETSc nested `-log_view`;
* the outer FGMRES residual history;
* each run capped at 3 h.

**Runs** (table: `solver_diag/diag_summary.csv`). `α_u ∈ {1e-10, 1e-11}`
× `N ∈ {50, 100, 200}`. All six were stopped by the 3 h cap before reaching
the outer rtol 1e-8, i.e. a reduction of 1e8. They are diagnostics, not
solutions.

| `α_u` | N | outer its done | outer residual reduction | Schur solves / capped at 100 | spatial solves | spatial its mean (min–max) | per-slot mean range | spatial failures | s / outer it |
|---|---|---|---|---|---|---|---|---|---|
| 1e-10 | 50 | 130 | **387×** | 130 / 130 | 2.04e6 | 18.3 (13–21) | ±0.2 within each sweep | 0 | 83 |
| 1e-11 | 50 | 128 | **43×** | 128 / 128 | 2.01e6 | 18.5 (13–22) | ±0.3 | 0 | 84 |
| 1e-10 | 100 | 67 | **678×** | 67 / 67 | 2.11e6 | 17.5 (13–20) | ±0.2 | 0 | 161 |
| 1e-11 | 100 | 67 | **70×** | 67 / 67 | 2.10e6 | 17.9 (13–21) | ±0.2 | 0 | 161 |
| 1e-10 | 200 | 37 | **2363×** | 37 / 37 | 2.32e6 | 16.5 (12–19) | ±0.3 | 0 | 292 |
| 1e-11 | 200 | 36 | **117×** | 36 / 36 | 2.27e6 | 16.7 (12–19) | ±0.3 | 0 | 300 |

Per sweep type (N = 50), the spatial GMRES iteration counts are:

| sweep | `α_u = 1e-10` | `α_u = 1e-11` |
|---|---|---|
| `y` sweep (Schur matvec) | 20.0 | 20.2 |
| `right` sweep (forward) | 19.8 | 20.7 |
| `left` sweep (backward) | 15.0 | 14.6 |

The same pattern holds at N = 100 and 200.

**Outer residual at equal iteration counts.** After 10 outer iterations,
N = 50: 5.3e-3 at 1e-10 against 4.1e-2 at 1e-11. After 36 iterations,
N = 200: 1.9e-4 against 3.8e-3.

**Timing breakdown** (`solver_diag/timing/`). Same case, `α_u = 1e-10`,
N = 50, outer FGMRES limited to 5 iterations (`-ksp_max_it 5`, outer only),
PETSc nested `-log_view`. Total 232 s:

| component | time | share |
|---|---|---|
| outer KSP | 231.4 s | 99.7 % |
| Schur-split KSP (all its Schur matvecs and PC applications) | 229.7 s | 99.0 % |
| matching-Schur PC applications (`left` + `right` sweeps) | 143.9 s | 62 % |
| forward sweeps (`right`, `y`) | 163.4 s | 70 % |
| backward sweeps (`left`) | 61.1 s | 26 % |
| spatial KSP solves (all sweeps) | 222.5 s | 96 % |

**What the diagnostics establish:**

* **(A) Outer iterations — yes, this is the bottleneck.** At equal outer
  iteration counts the residual reduction at `α_u = 1e-11` is about **10×
  smaller** than at 1e-10, for every N (43× vs 387×, 70× vs 678×, 117× vs
  2363×).
* **The Schur split is always capped.** All 465 Schur-split solves stopped at
  the 100-iteration cap, at both `α_u`. So every outer iteration costs the
  maximum: about 314·N spatial solves, i.e. 83 / 161 / 296 s at N = 50 / 100 /
  200. That cost is the same for both `α_u` and **linear in N**.
* **(B) Spatial iterations — no.** The spatial GMRES+GAMG solves take 12–22
  iterations, with a mean of 16.5–18.5. The means differ by ≤ 0.6 between
  `α_u = 1e-10` and 1e-11, and there were no failures in about 12.8 million
  solves. The wall time sits in spatial solves (96 %) only because of their
  **number**, not their individual cost.
* **(D) Pathological time slots — none.** Within each sweep type the per-slot
  mean iteration count varies by at most ±0.4 across all slots, including the
  first and last.
* **Classification: A**, made expensive by the always-capped Schur split
  (case C in the sense "both" applies only through the capped inner Krylov
  level, not through the spatial solves).
* **Time-step dependence.** Coarser time grids need *more* outer iterations
  for the same reduction:
  * after 36 outer iterations at `α_u = 1e-10`: 2363× at N = 200, 678× at
    N = 100 (after 67), 387× at N = 50 (after 130);
  * the N = 400 production run converged (reduction 1e8) in 70 iterations;
  * at `α_u = 1e-7`: 11 outer iterations at N = 50 (MPI check, §8), 5 at
    N = 200 (nx 64), 3 at N = 400.

  The N = 50 / 100 diagnostics therefore **overstate** the outer count of the
  N = 400 production case. They are valid for the *relative* `α_u` comparison
  at fixed N.
* **Extrapolation, not a measurement.** At N = 400, `α_u = 1e-11`, the outer
  FGMRES would need several hundred iterations at about 600 s each, with its
  30-step restart and 200-iteration cap. That is consistent with the aborted
  10.4 h run still iterating. Whether it would ever reach rtol 1e-8 within the
  200-iteration cap is not established.


**Tuning check** (`EF96_SMALL_ALPHA_SOLVER_TUNING.md`). Cheap N = 25 / 50
proxies varied only PETSc runtime options.
* The Schur-split rtol is never active: every solve is capped.
* Lowering the cap from 100 to 50 removes wasted, stagnating Schur iterations
  and improves residual decades per second by ≈ 2–2.7×.
* A larger restart or a tighter Schur solve does not help.
* The `α_u = 1e-11` degradation persists at every setting.
* Classification: **B, modest improvement only**. The limitation is the
  parameter robustness of the matching-Schur approximation, not the tolerance
  choice.

## 6. EF 9.6 scientific conclusion (unchanged by the solver findings)

* The completed `α_u = 1e-10` solution is a **valid discrete optimum**: true
  KKT residual 8.0e-9, independent march 1.9e-14, zero sweep failures.
* Tail error (sup on `[5, 20]`): 0.1520 at `α_u = 1e-10` against 0.1540 at
  `1e-9`, a 1.3 % change. Over the same step the control norm grows from 1.2e3
  to 5.1e3. Both points are on nx 32 × nz 12.
* **Resolution status.** The tail is spatially converged at `α_u = 1e-7`
  (nx 32 → 64: 0.15 %; N 200 → 400: 3e-6). Full spatial convergence of the
  smallest-`α_u` value 0.1520 was not established because of solver cost. It
  is a practical saturation level on the tested mesh and regularisation
  range.
* Source Table 10: `E₅ = 5.220e-2`.
* This supports **practical saturation over the tested regularisation range**
  (`α_u` = 1e-6 … 1e-10). It is not a proof of an asymptotic floor, and it is
  not a claim that `E₅` is mathematically unreachable.
* The `α_u = 1e-11` production run was **aborted because of solver cost**
  (10.4 h, still iterating). It was not a failed mathematical solve. **No
  numerical value from it is used.**

## 7. Reporting distinctions for the final Aulisa/Gilliam report

* **Same plant ≠ same controller.** ex21 computes the finite-horizon AAO
  optimum. The sources use cascade / REFC controllers.
* **The source's REFC parameters `α, β` are not ex21's `α_u`.** An `α_u` at
  which the AAO tail matches a published level is a *matched tail scale*, not
  parameter equivalence.
* **EF 9.6:** Table 10 is the primary quantitative datum. Fig. 34 sits
  systematically 3.6–5.1 % above it and is used only qualitatively.
* **Missing source initial condition:** handled by IC-sensitivity tests (three
  ICs). The tails are IC-insensitive: ≤ 4e-6 for EF 9.5, ≤ 1.1e-4 for EF 9.6
  at the solver-tolerance level.
* **Source geometry status:** the EF 9.5/9.6 regions are as asserted by the
  user, since the source figures are non-to-scale schematics. Math2025 Ex5
  `(W, H) = (π, 1)` is [B], corroborated numerically.
* **True-residual convergence validates a completed AAO solve** even when the
  preconditioner is inefficient. Cost and validity are separate statements.

## 8. Serial execution and MPI

**Where serial execution is imposed.** Three explicit guards in the ex21
benchmark drivers throw unless the communicator has exactly one rank:
* `include/Benchmark2D.cpp:164` (`RunBenchmark2D`, used by Ex5 and EF
  9.5/9.6);
* `include/Math2025Benchmark.cpp:128` (Ex3/Ex4);
* `include/ConvectionVerification.cpp:122` (convection gates and MMS).

These drivers are serial by design: their gates use a serial LU march, LU
reference adjoints and rank-local arrays. In addition, all campaign scripts
invoke the binary without `mpiexec`. The PETSc matrices are therefore SeqAIJ
(communicator size 1), which is the `MatMult_SeqAIJ` seen in the user's
backtrace.

**Is the production solver MPI-capable for this configuration?** Yes. The
production path (`main()` → `SolveOptimalControl` → `WriteSolution`) is the
ex20 ownership-distributed solver with no rank restriction, and the ex21
convection, source-region and Crank–Nicolson additions assemble over owned
elements.

The EF presets previously forced the serial benchmark driver. A
`--solve-only` flag now keeps the preset's data and takes the production path;
the driver's guard is unchanged. **The benchmark driver, plus the scripts'
missing `mpiexec`, was the only reason the run used one rank.**

**Small MPI check of the production path** (`solver_diag/mpi/`), kept
separate from the scientific conclusions. EF 9.6 data, `--solve-only`,
`α_u = 1e-7`, nx 32 × nz 12, N = 50, CN, SCHUR_SWEEP; np = 1, 2, 4 run one
after another, with 6 serial diagnostic jobs on the same node.

The planned case (`α_u = 1e-10`, N = 100) was replaced: §5 shows it does not
converge within 3 h even at np = 1, so it is not a "small, safe" test.

| np | outer | Schur its (all capped) | spatial its mean y / right / left | true residual | `J_total` rel. diff vs np 1 | wall s |
|---|---|---|---|---|---|---|
| 1 | 11 | 1100 | 19.3 / 18.1 / 17.3 | 1.21e-9 | — | 875 |
| 2 | 11 | 1100 | 19.2 / 18.3 / 17.5 | 1.28e-9 | 1.8e-12 | 836 |
| 4 | 11 | 1100 | 17.7 / 17.2 / 16.4 | 1.18e-9 | 1.0e-12 | 570 |

* `J_track` agrees to ≤ 9e-10 and `J_reg` to ≤ 7e-8 relative, which is the
  outer solver tolerance.
* The outer and Schur iteration counts are identical across np.
* The spatial counts differ slightly, because GAMG aggregates differently in
  parallel.
* Speedup is 1.05× at np 2 and 1.5× at np 4. That is modest: each time step
  is a small spatial problem (≈ 3.3e3 DOFs), the sweeps are sequential in
  time, and the node was shared.

**Conclusion: ex21's production solver runs this configuration correctly
under MPI.** Parallelising the expensive EF 9.6 runs would cut wall time by
at most a small factor at this mesh size. It would not remove the
outer-iteration growth, which is algorithmic (§5).

**What would be needed to run the benchmark *driver* under MPI:** replace its
serial gates with parallel versions:
* the LU forward march (MUMPS / SuperLU_dist);
* the rank-local `on_boundary`, `IntegrateSpatialField` and series loops (add
  reductions).

None of this is needed for the solve itself: `--solve-only` already gives the
same optimum on any np.


## 9. Recommended wording for the final Aulisa/Gilliam report

> All ex21 convection–diffusion solutions reported here are discrete optima
> verified independently of the linear solver, by exact-transpose, reduced-
> gradient, zero-convection and manufactured-solution tests and by the true
> KKT residual of each solve (≤ 8.4e-9). The SCHUR_SWEEP preconditioner is
> robust for the colocated benchmarks (Mathematics 2025 Ex. 4, ErrorFeedback
> 9.5): two outer iterations and about eight Schur iterations per outer
> iteration for α_u from 1e-3 to 1e-11.
>
> In the non-colocated benchmarks the matching Schur approximation loses
> effectiveness as α_u decreases, and the Schur-split solve reaches its
> iteration cap. Mathematics 2025 Ex. 3 still converges in at most seven outer
> iterations down to α_u = 1e-11.
>
> ErrorFeedback 9.6 is the stress case. There the outer iteration count grows
> from 3 (α_u = 1e-7) to 70 (1e-10) at N = 400. Diagnostics at reduced N show
> that the cost growth comes from the outer FGMRES convergence (about ten
> times slower residual reduction at α_u = 1e-11 than at 1e-10), multiplied by
> Schur solves that always run to their cap. The spatial solves average
> 16.5–18.5 iterations without failures. The α_u = 1e-11 production solve was
> therefore aborted for cost, and no value from it is used.
>
> This is a limitation in parameter robustness of the preconditioner in a
> strongly ill-conditioned non-colocated 2D configuration. It is not evidence
> of an error in the formulation, and it does not affect the completed
> solutions. For ErrorFeedback 9.6 the completed α_u = 1e-10 solution (true
> residual 8.0e-9; nx 32 × nz 12, where full spatial convergence was not
> established because of solver cost, whereas the tail is spatially converged
> at α_u = 1e-7) has a tail error of 0.152, against 0.154 at 1e-9, while the
> control norm grows from 1.2e3 to 5.1e3. This indicates practical saturation
> over the tested mesh and regularisation range, about three times above the
> published Table 10 value E₅ = 0.0522, which is not quantitatively
> reproduced on the asserted geometry. It is not a proof of an
> asymptotic floor, nor of the unreachability of E₅.
