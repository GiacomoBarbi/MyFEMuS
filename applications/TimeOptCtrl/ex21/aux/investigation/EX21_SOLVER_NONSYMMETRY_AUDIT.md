# ex21 — solver and preconditioner audit for a nonsymmetric spatial operator

With convection the spatial operator is `L = K + C`, `C_ij = ∫ (v·∇φ_j) φ_i`,
and `L ≠ L^T`. Every ex20 component was inspected for an assumption of
symmetry, SPD, identical forward/adjoint spatial blocks, Cholesky
applicability, symmetric-AMG behaviour, and CG/MINRES use (`grep` of
`KSPCG|MINRES|CHOLESKY|potrf|MAT_SPD|MAT_SYMMETRIC|MatMult(stiffness` over
`include/`, then reading each hit).

Policy (unchanged from the brief): **no broad rewrite**. A component is
either valid as is, fixed by the smallest gated change, or **refused** with an
explicit error. Every change is gated on `HasConvection(config)` (`--convection
on`), so the default path is ex20's, operation for operation (61/61 goldens
bit-identical, `EX21_ZERO_CONVECTION_BASELINE.md`).

| # | component (file) | ex20 assumption | valid with `L ≠ L^T`? | action (ex21) |
|---|---|---|---|---|
| 1 | forward shell `AY` (`SpaceTimeOperators.cpp`) | none (applies `M`, `K`) | **yes** (`K → L`) | none |
| 2 | **adjoint shell `AP`** | `K` symmetric: applies `MatMult(K)`, `MatMult(A_±)` (header: "M and K symmetric, so A_plus and A_minus are their own transposes") | **NO — silently wrong adjoint** | `transpose_spatial` flag: `MatMultTranspose` on `L`, `A_+`, `A_−` (`SpatialProduct`) |
| 3 | assembled `E` (`ay`) | none | yes | none |
| 4 | assembled `E^T` (`ap`) | built as `MatTranspose(ay)` | **yes, exactly** | none |
| 5 | CN data term `(M/dt − K/2) y_0` (`Solver.cpp`) | forward product | yes | none (see #21 for the separate CN/Dirichlet trace fix) |
| 6 | homogeneous Dirichlet elimination (`MatZeroRowsColumns`, `Discretization.cpp`) | header: eliminations "symmetric, so M and K stay symmetric" | yes: rows **and** columns are zeroed in `L`, so `E^T` stays the exact transpose (gate G2) | none |
| 7 | control reconstruction `u = −(η/α_u) p`, objective, KKT diagnostics | `N`, `M_ω` symmetric (still true); residuals through the shells | yes (after #2) | none |
| 8 | outer solver: FGMRES on the 2×2 `MatNest` | nonsymmetric already | yes | none |
| 9 | `ADDITIVE_GAMG`: GMRES + GAMG on `ay` (y split) and `ap` (p split) | GAMG smoothed aggregation is designed for SPD; used only as a **preconditioner** | yes (Krylov method nonsymmetric; PC quality only) | none. Observed: 2D BE/CN 14 outer its, same J as the other PCs |
| 10 | `SCHUR_MATCHING` auxiliary `right = ay + √β(I⊗N)`, `left = ap + √β(W⊗M_ω)` | built from the assembled `ay`/`ap` | **yes** (`ap` is the exact transpose) | none. GMRES+GAMG inner solves |
| 11 | `SCHUR_SWEEP` forward sweeps (`right`, `ay`, y-split) | diagonal `A + √β N`, coupling `M/dt` (BE) or `M/dt − K/2` (CN) | yes (forward = `E`) | none |
| 12 | **`SCHUR_SWEEP` backward sweep (`left` = `E^T` factor)** | reuses `A` and the CN coupling **untransposed** ("A is SPD") | **NO** | `A_left = A^T + √β M_ω`, `A_left_final` likewise, CN coupling `A_coupling^T` (`BuildSpatialSweepMatrices`) |
| 13 | **spatial solve inside every sweep: CG + GAMG** | SPD block | **NO** (CG invalid) | `SetNonsymmetricSpatialSweeps(true)` → GMRES(50), **right** preconditioning (keeps the unpreconditioned residual norm, so the tolerance means the same), GAMG |
| 14 | `MAT_SYMMETRIC` / `MAT_SPD` flags on sweep blocks | true | **NO** (would mislead PETSc) | not set under convection |
| 15 | Stage 8D temporal prototypes (block Jacobi / windowed) | run on the sweep contexts | inherit #12–#14 | none |
| 16 | `SCHUR_BOUNDARY_REDUCED` (`BoundaryReducedSchur.cpp`) | `A`, `A_ZZ` SPD, Cholesky preferred, `MAT_SPD` | **NO** | **refused**: needs Neumann control, and convection is DISTRIBUTED-only (`Config::Validate`) |
| 17 | `SCHUR_DIRICHLET_REDUCED` (`DirichletReducedSchur.cpp`) | dense `LAPACKpotrf` of the reduced Gram, `MAT_SPD` | **NO** | **refused** (Dirichlet control + convection rejected) |
| 18 | Dirichlet control coupling `B_D`, `R`-Cholesky (`DirichletControl.cpp`) | applies `MatMult(K_full)` for both `B_D` and `B_D^T` | **NO** | **refused** (as #17) |
| 19 | ex20 verification batteries `--verify-stage1/-sweeps/-boundary-reduced` (`Verification.cpp`) | symmetric arrangement, symmetric sweeps | not designed for it | **refused** with `--convection on`; replaced by `--verify-convection` |
| 20 | ex20 literature / manufactured benchmarks (`Benchmarks.cpp`) | diffusion-only data | not applicable | **refused** with `--convection on`; replaced by `--convection-mms` |
| 21 | CN + homogeneous Dirichlet **state** BC (not a symmetry issue; found while auditing) | — | **NO**: constrained rows `½y_n + ½y_{n−1} = …` propagate a nonzero initial trace as `(−1)^n y_0,G` | fixed in ex21: `y_0` restricted to zero trace before the CN rhs (gated on CN + Dirichlet state); unreachable from the ex20 CLI |

Direct LU paths (the reduced-Hessian / FD gates and the MMS march in
`ConvectionVerification.cpp`) are kept as correctness references, serial only.

## Observed solver behaviour under convection (`v = (2,1)`, `ν = 0.5`, 2D; `a = v = 2` 1D)

From `logs/verify_convection_*.log` (gate G4):

| case | ADDITIVE_GAMG | SCHUR_MATCHING | SCHUR_SWEEP |
|---|---|---|---|
| 2D BE | converged | converged | converged |
| 2D CN | converged | converged | converged |
| 1D BE | converged | converged | converged |
| 1D CN | converged | **fails (reason −11)** | converged |

**1D CN + SCHUR_MATCHING fails with and without convection** (identical
failure with `--convection off`): the GMRES+GAMG inner solves of the matching
factors stall at 100 iterations and then produce NaN; replacing the PC of
either factor by LU (`PETSC_OPTIONS=-fieldsplit_schur_right_pc_type lu`)
restores convergence. It is a pre-existing GAMG/1D-CN issue, not a symmetry
issue, and is recorded, not fixed. For 1D runs use SCHUR_SWEEP or
ADDITIVE_GAMG.

## Not changed (deliberately)

* No SUPG or other stabilisation (brief, Phase 2).
* GAMG options untouched: smoothed-aggregation GAMG preconditions the
  nonsymmetric blocks adequately at the moderate Péclet numbers tested; a
  nonsymmetric-AMG configuration is a performance question for later.
* Neumann control with convection is refused although #1–#15 would cover it:
  it has not been verified, so it is not exposed.
