# T9 — 3D phase 1: distributed control in ex21 (HEX27 Q2)

*Baseline: commit e0410efe3 (frozen 2D). Scope: 3D, Q2 HEX27 box, distributed
full-domain control and tracking, homogeneous Dirichlet state BC, constant wind
(including w = 0), BE and CN, L2 regularisation, SCHUR_SWEEP, MPI production path.
Everything else in 3D is refused. No commit, push or staging.*

Scripts, logs and results are in `t9/`; the CSVs are in this directory.

## 1. 3D dependency map (written before any code change)

| file | symbol / function | current (2D) assumption | needed 3D change | in phase-1 scope |
|---|---|---|---|---|
| Config.hpp | `Validate` | `dimension ∈ {1, 2}` | accept 3 behind a strict whitelist; refuse everything else in 3D | yes |
| Config.hpp | `Domain {xmin, xmax, zmin, zmax}` | two extents; "z" is mesh axis 1 | add `ymin, ymax`. In 3D: axis 1 = y, axis 2 = z. Phase 1 requires the unit cube | yes (unit cube only) |
| Config.hpp | `MeshConfig {nx, nz}` | `nz` = element count along axis 1 | add `ny`. In 3D: (nx, ny, nz) along axes (0, 1, 2) | yes |
| Config.hpp | `ConvectionConfig {vx, vy}` | vy acts on axis 1 | add `vz` (axis 2). The component index is always the axis index | yes |
| Config.hpp | `convection.field(x, z, w[2])` | 2D variable wind | refused in 3D | no |
| Config.hpp | `DataOverrides {source, initial_state, tracking_target, dirichlet_state}(x, z)` | 2-coordinate callbacks | add 3D callbacks `source3d / initial3d / target3d (x, y, z)`; lifting refused in 3D | yes (3D data) |
| Config.hpp | `EvaluateSource / EvaluateInitialState / EvaluateTrackingTarget(x, z)` | 2 coordinates | optional trailing `axis2` argument; in 3D dispatch to the 3D overrides; the 2D path is untouched. Only dimension-free built-in data (constants, sin_x_minus_t) is allowed in 3D | yes |
| Config.hpp | `EvaluateVelocity(x, z, w[2])` | 2 components | the constant 3D wind is read directly in the 3D assembly; the 2D helper is untouched | yes |
| Config.hpp | `ControlRegionIsWholeDomain` | (x, z) extents | phase 1 forces full-domain control and tracking | yes (full domain only) |
| Discretization.cpp | `BuildMesh` | EDGE3 / QUAD9 | `GenerateCoarseBoxMesh(nx, ny, nz, …, HEX27, "seventh")` | yes |
| Discretization.cpp | `ElementCentroid`, `CentroidIsInBox`, `Tracking/ControlRegionIndex` | centroid (axis 0, axis 1) | 3D returns region 1 (whole domain) for every element; boxes are refused | yes (trivially) |
| Discretization.cpp | convection assembly, `velocity[2]`, `axis < 2` | 2 components | `velocity[3]`, all `dimension` axes; 2D bits unchanged | yes |
| Discretization.cpp | face loop (M_Γ, K_Γ) | 1D faces (edge arc length) | skipped in 3D (as in 1D): no boundary control in phase 1 | no |
| Discretization.cpp | `BuildLpsMatrix`, LPS h_k | 2 extents | refused in 3D | no |
| Discretization.cpp | `ConstrainedStateRows` (homogeneous Dirichlet) | FEMuS `GenerateBdc` boundary flags | dimension-free (FEMuS marks all boundary faces of the HEX27 box); verified | yes |
| Discretization.hpp/.cpp | `OwnedSpatialLayout {x, z}` | axis 0, axis 1 | add `axis2` (filled only in 3D; empty otherwise) | yes |
| Discretization.cpp | `IntegrateSpatialField(exact(x, z))` | 2-coordinate exact field | new `IntegrateSpatialField3D(exact(x, y, z))`; the 2D one is untouched | yes |
| Solver.cpp | data side (`EvaluateSource / Target / Initial` at `layout.x, layout.z`) | 2 coordinates | pass `layout.axis2` | yes |
| SpaceTimeOperators / Preconditioners / Solver KKT | E, Eᵀ, sweeps, FGMRES | dimension-free (spatial matrices only) | none | yes |
| Output.cpp | JSON `spatial_coordinates [x, z]` | 2 coordinates | `[x, y, z]` in 3D only | yes |
| Verification.cpp | `--dump-operators` coords | 2 columns | 3 columns in 3D only | yes |
| ConvectionVerification.cpp | G1 orientation (linear fields x, z; wind (vx, vy)) | 2 axes | three axes in 3D (x, y, z with vx, vy, vz) | yes |
| ConvectionVerification.cpp | G1N boundary identity, G8b natural-BC tests | 2D faces / normals | natural BC is refused in 3D, so these do not run | no |
| ExactOcBenchmark.cpp | all exact cases (x, z), 2D driver | 2D | a separate 3D case and a small 3D driver (new file); the 2D driver is untouched | yes |
| CommandLine.cpp | `--dimension 1\|2`, `--nz`, `--vx/--vy`, presets, run modes | 2D | `--dimension 3`, `--ny`, `--vz`; refuse 2D presets, other run modes and other solvers in 3D | yes |
| Benchmarks / Math2025 / Benchmark2D / MMS | 2D / 1D presets | refused in 3D | no |
| DirichletControl / BoundaryReducedSchur / DirichletReducedSchur | boundary control | refused in 3D | no |

## 2. Coordinate convention

FEMuS stores node coordinates in `_topology->_Sol[0..2]` = (X, Y, Z) in every dimension.
The mapping, chosen so that the 2D interpretation is preserved bitwise:

| dimension | storage axis 0 | storage axis 1 | storage axis 2 | wind |
|---|---|---|---|---|
| 1D | x | — | — | vx |
| 2D | x | **"z"** (historical name; domain zmin/zmax, mesh nz) | — | (vx, vy) on axes (0, 1) |
| 3D | x | **y** (domain ymin/ymax, mesh ny) | **z** (domain zmin/zmax, mesh nz) | (vx, vy, vz) on axes (0, 1, 2) |

The wind component index always equals the storage axis. In 3D the physical
coordinates passed to every 3D callback are (x, y, z) = (axis 0, axis 1, axis 2).
`layout.z` keeps holding storage axis 1 in every dimension, so no 2D variable is
renamed. In 3D it is the physical **y**, and the new `layout.axis2` is the physical z.

## 3. Phase-1 whitelist (dimension = 3)

**Allowed:**

- `--distributed-solve`, full-domain control (`--control-full-domain` or the implied
  default) and full-domain tracking;
- `--state-bc dirichlet` (homogeneous);
- convection off, or on with a constant (vx, vy, vz);
- BE / CN, L2 control;
- `--pc-schur-sweep`, or ADDITIVE_GAMG for BE only;
- the production run mode, `--verify-convection`, `--exact-oc conv-sin-3d` (and
  `--solve-only`), `--dump-operators`.

**Refused with `invalid_argument`:** Neumann and Dirichlet control, H1-space / H1-time,
lifting, named or variable winds, LPS, source regions, control/tracking boxes and
multi-zone, non-unit-cube domains, the natural state BC, all 2D/1D presets
(Math2025, ErrorFeedback, LP, literature, manufactured, MMS), the other exact-OC cases,
the boundary-reduced, Dirichlet-reduced and matching solvers, the time-block / sweep-depth
prototypes, ADDITIVE_GAMG with CN, and 2D-shaped data forms (EXP_MINUS_T_COS2PI,
ONE_MINUS_Y target).

`--dimension 3` combined with any 1D/2D preset or 2D exact case (Math2025 Ex3/4/5,
ErrorFeedback 9.5/9.6, conv-sin3, lift-exp, …) is refused. Those presets overwrite
`config.dimension`, and before the fix found by the regression's restriction section
(§15) they silently ran in 1D/2D. The check now uses the *requested* dimension. `--ny`
is refused unless the final dimension is 3; in 2D the second axis stays `--nz`, and
before T9 `--ny` was an unknown option, so it is still an error in 2D.

The 3D production path must name its whitelist explicitly:
`--dimension 3 --distributed-solve --control-full-domain --tracking-box 0 1 0 1
--state-bc dirichlet …`. The 2D default state BC and tracking region are not
silently reused. `--exact-oc conv-sin(3)-3d` sets all of this itself.

## 4. HEX27 mesh checks (`t9/scripts/t9_mesh.py`, `t9/logs/mesh.log`: 39/39 PASS)

These checks use `--dump-operators` at n = 2, 3, 4, on the unit cube with homogeneous Dirichlet.

- The spatial DOF count is (2n+1)³ (125, 343, 729), with 3 coordinate columns.
- The constrained set is exactly the boundary nodes, (2n+1)³ − (2n−1)³ of them (98, 218, 386), and K = I on it.
- The Q2 element bubble q gives qᵀMq = 1/27000 and qᵀKq = 1/900 exactly (ν = 1).
- M and K (heat) are symmetric.
- The convection part C = L_conv − K_heat satisfies qᵀCq ≈ 1e-19 and |C + Cᵀ|/|C| ≤ 4e-14 on the free DOFs (skew).
- M and K of a convection run equal those of the heat run.

Preallocation is 125 nonzeros per row in 3D (a HEX27 vertex row couples 5³ DOFs); 1D/2D keep 64.

## 5. Exact 3D optimal control (`include/ExactOc3D.{hpp,cpp}`)

The common definitions are:

    S = s(πx) s(πy) s(πz),   s = sin  (conv-sin-3d)   or   s = sin³  (conv-sin3-3d)
    y = cos(2t) S,   p = α_u sin(T − t) S,   u = −(η/α_u) p
    f   = y_t − νΔy + w·∇y − η u
    y_d = y + (p_t + νΔp + w·∇p)/α_t

The parameters are ν = 0.25, α_u = 1e-2, α_t = η = 1, T = 1. The constant wind is (vx, vy, vz) on the storage axes (x, y, z), and is zero when convection is off.

- **conv-sin-3d** is the direct analogue of the 2D `conv-sin`. Its source has a nonzero boundary trace when w ≠ 0 (∂ₓS ≠ 0 on x = 0, 1). The inherited load convention M_elim · I_h f drops that trace. The result is the same h^{5/2} state rate that 2D shows (EX21_CONVECTION_VALIDATION.md §F; 2D conv-sin: 2.30/2.43/2.37).
- **conv-sin3-3d** uses s = sin³, so all data vanish on ∂Ω (the analogue of the 2D `conv-sin3`). This is the reference case for the order-3 claim.

`RunExactOptimalControl3D` (serial) applies the tightened policy of the 2D exact driver:

- outer rtol 1e-12, inner 1e-10, maxits 400;
- an independent LU forward march;
- stationarity αᵤu + ηp;
- FEM-integrated L²(Ω) errors via `IntegrateSpatialField3D`: y at t_n, and p, u at the interval midpoints under CN (t_n under BE).

## 6. Symbolic verification (`t9_symbolic.py`, `logs/symbolic.log`: 69/69 PASS)

The checks are exact sympy simplifications, run for the shapes sin and sin³, each with the winds 0, (1, 0.5, −0.75) and a symbolic (vx, vy, vz):

| check | content |
|---|---|
| K1 | state PDE residual = 0 |
| K2 | adjoint PDE residual = 0 |
| K3 | y = p = u = 0 on all six faces |
| K4 | p(T) = 0 |
| K5 | αᵤu + ηp = 0 |
| K6 | the literal C++ port of the source and target matches sympy to ≤ 1e-12 at 375 points (numeric winds) |

## 7. Heat and w = 0 sanity

**Gates.** The heat and w = 0 rows of `t9_3d_gates.csv` show:

- G1 = 0;
- G2 ≤ 4.4e-17;
- FD ≤ 6.4e-13;
- Hessian ≤ 8.7e-15;
- 2 outer iterations with true residual ≤ 6e-15;
- G7 ≤ 2.3e-12;
- march ≤ 2.1e-15.

As intended, the wrong-adjoint negative controls show nothing without wind; they are identical to the correct adjoint.

**Heat exact ladder (spaceH, sin³, CN, N = 64, LU).** y orders are 3.46, 3.17, 3.08, 3.05, 3.03 (n = 2 → 12). p and u orders are 5.17, 3.46, 3.49, 3.44, 3.35.

**Convection off vs on with w = (0, 0, 0)** (`t9_parity.sh`, `logs/parity/dumps.log`):

- At n = 3, M, K, M_track, M_ctrl_box, N_selected and the coordinates are byte-identical.
- In the production solves (n = 4, N = 16) with LU spatial solves, fields agree to ≤ 5e-15 relative and J to 1 ulp, with identical outer and inner iterations.
- With the iterative backend, fields agree to ≤ 1e-10. That is the inner GMRES+GAMG tolerance: the GAMG setup sees a different (but numerically equal) matrix nonzero pattern, giving 1343 vs 1337 inner iterations.

## 8. Convection operator and algebraic gates (`t9_gates.sh`, `t9_3d_gates.csv`)

The runs cover n = 2, 3 × BE/CN × {heat, w0, wA = (1, 0.5, −0.75), wB = (−2, 1, 0.5)}, 16 runs in all, with the outer KSP at rtol 1e-12.

| gate | correct (max over 16 runs) | negative control |
|---|---|---|
| G1 orientation (x, y, z linear fields) | ≤ 1.2e-14 | transposed C: 2.5 – 4.0 |
| G1 skew on free DOFs | ≤ 1.8e-14 | — |
| G2 shell transpose ⟨Ax, y⟩ = ⟨x, Aᵀy⟩ | ≤ 4.4e-17 | ex20-style adjoint (no −w·∇p): 1.5e-4 – 2.8e-3 under wind |
| G3 finite-difference gradient | ≤ 6.4e-13 | ex20-style: 0.27 – 2.4 |
| G4 Hessian symmetry/consistency | ≤ 2.2e-14 | ex20-style: 0.08 – 0.76 |
| G5 outer iterations / true residual | 2 / ≤ 6e-15 | — |
| G7 gradient consistency | ≤ 2.3e-12 | — |
| G6 independent forward march | ≤ 2.1e-15 | — |

At the production outer rtol 1e-8, G7 is 6.4e-7 after a single outer iteration. That is tolerance, not algebra: at 1e-12 the solve takes 2 outer iterations and reaches 5e-13.

## 9. Spatial convergence (`t9_3d_space_convergence.csv`, `figures/3d_space_convergence.png`)

The setup is CN with N = 64, wind A (or heat), SCHUR_SWEEP with LU spatial solves. The spaceB n = 16 point used GMRES+GAMG spatial solves: 6.9 M space-time unknowns, outer true residual 3.3e-15. The table gives absolute space-time L² errors; orders are relative to the previous n.

| n | sin, wind A: y | p, u | sin³, wind A: y | p, u | sin³, heat: y | p, u |
|---|---|---|---|---|---|---|
| 2 | 1.03e-2 | 2.61e-4 | 5.59e-2 | 2.59e-3 | 5.67e-2 | 2.79e-3 |
| 4 | 1.88e-3 (2.46) | (2.56) | 5.13e-3 (3.45) | (5.10) | 5.16e-3 (3.46) | (5.17) |
| 6 | 7.44e-4 (2.29) | (2.90) | 1.43e-3 (3.16) | (3.49) | 1.43e-3 (3.17) | (3.46) |
| 8 | 3.78e-4 (2.36) | (2.91) | 5.88e-4 (3.08) | (3.47) | 5.89e-4 (3.08) | (3.49) |
| 10 | 2.21e-4 (2.40) | (2.87) | 2.98e-4 (3.05) | (3.41) | 2.98e-4 (3.05) | (3.44) |
| 12 | 1.42e-4 (2.42) | (2.84) | 1.72e-4 (3.03) | (3.32) | 1.72e-4 (3.03) | (3.35) |
| 16 | 7.06e-5 (2.43) | (2.73) | 7.20e-5 (3.02) | (3.09) | — | — |

Every run in these ladders has stationarity ≈ 5e-17 and independent march ≤ 2.3e-14.

**Reading:**

- With sin³ data, the state converges at **order 3**: 3.05 → 3.03 → 3.02, monotone toward 3 from above. Heat and wind agree to three digits.
- p and u are pre-asymptotically super-convergent (3.5 → 3.3). At n = 16 the N = 64 time error (measured in §10: p ≈ 2.7e-7) is no longer negligible against the spatial p error (7.1e-7). Removing it in quadrature gives a spatial p, u order of about 3.4 for 12 → 16. The raw 3.09 at n = 16 is therefore time-floor contaminated and is **not** read as an asymptotic order. No order above 3 is claimed for p and u.
- With sin data, y converges at h^{5/2} (2.43). This is exactly the documented 2D data-trace effect of the load convention, not a 3D defect. sin³ removes it.

## 10. BE temporal convergence (`t9_3d_time_convergence.csv`, `figures/3d_time_convergence.png`)

This is fixed-mesh self-convergence: n = 8, wind A, sin³ data, SCHUR_SWEEP + LU, outer rtol 1e-12, reference N = 256 on the same mesh, so the spatial error cancels. The norm is the space-time L² with HEX27 Simpson weights. Under BE all of y, p, u are compared at t_n.

The "raw" order is log₂ of successive ratios. The "ref-corrected" order solves e₁/e₂ = (Δt₁^q − Δt_ref^q)/(Δt₂^q − Δt_ref^q), which removes the bias of a reference only 4× finer than N = 64.

| N | y | raw / corrected | p | raw / corrected | u | raw / corrected | |J_N − J_256| |
|---|---|---|---|---|---|---|---|
| 4 | 2.28e-3 | | 2.32e-4 | | 2.32e-2 | | 8.6e-5 |
| 8 | 1.15e-3 | 0.99 / 0.97 | 1.23e-4 | 0.92 / 0.89 | 1.23e-2 | 0.92 / 0.89 | 4.7e-5 |
| 16 | 5.59e-4 | 1.04 / 0.99 | 6.19e-5 | 0.99 / 0.93 | 6.19e-3 | 0.99 / 0.93 | 2.4e-5 |
| 32 | 2.62e-4 | 1.09 / 0.99 | 2.96e-5 | 1.07 / 0.96 | 2.96e-3 | 1.07 / 0.96 | 1.1e-5 |
| 64 | 1.12e-4 | 1.22 / **1.00** | 1.29e-5 | 1.20 / **0.97** | 1.29e-3 | 1.20 / **0.97** | 4.9e-6 |

**BE is order 1** for y, p, u and J.

## 11. CN temporal convergence (same setup; p and u are interval quantities)

Coarse interval m is compared with the mean of the two fine midpoints symmetric about its midpoint, which is an O(Δt²) comparison.

| N | y (L²(0,T)) | raw | y on [0.25, T] | raw | y(T) | raw | p | raw / corrected | u | raw / corrected | |J_N − J_256| |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 6.68e-4 | | 6.68e-4 | | 1.27e-3 | | 5.05e-5 | | 5.05e-3 | | 4.1e-6 |
| 8 | 1.55e-4 | 2.11 | 1.52e-4 | 2.14 | 3.74e-4 | 1.76 | 1.54e-5 | 1.71 / 1.71 | 1.54e-3 | 1.71 / 1.71 | 1.0e-6 |
| 16 | 4.63e-5 | 1.74 | 3.89e-5 | 1.96 | 9.92e-5 | 1.91 | 4.17e-6 | 1.89 / 1.88 | 4.17e-4 | 1.89 / 1.88 | 2.6e-7 |
| 32 | 1.81e-5 | 1.35 | 7.94e-6 | 2.29 | 2.47e-5 | 2.01 | 1.06e-6 | 1.98 / 1.96 | 1.06e-4 | 1.98 / 1.96 | 6.5e-8 |
| 64 | 7.58e-6 | 1.26 | 1.49e-6 | 2.41 | 5.90e-6 | 2.06 | 2.53e-7 | 2.06 / **1.99** | 2.53e-5 | 2.06 / **1.99** | 1.6e-8 |

**CN is order 2 for p, u and J** (|ΔJ| ratios 3.9, 4.0, 4.0, 4.2). **y is order 2 away from t = 0** (on [0.25, T] and at T).

The global y L²(0,T) difference loses order (1.35, 1.26) because of a **CN initial layer**. The per-level difference peaks at the first level, t = Δt, and decays in time. The first-level value shrinks only slowly: 1.3e-4, 8.9e-5, 6.8e-5, 5.6e-5, 4.4e-5. This is the classical Crank–Nicolson behaviour for discrete initial data whose stiff modes (λΔt ≫ 1) are not damped. Here those modes are the high-frequency part of the interpolated y₀ on the coarse n = 8 mesh.

The effect belongs to the inherited, frozen CN time discretisation and is not 3D-specific. The same diagnostic on 2D `conv-sin3` with the **frozen 2D binary** (nx = 16) shows it (`t9_cn_layer.sh`, `logs/cn_initial_layer.log`):

- the first-level difference stalls at about 8e-6 for N = 32 → 64;
- the global y order drops to 1.77;
- y on [0.25, T] stays at 2.2.

3D at n = 8 is coarser, so the layer becomes visible earlier. Removing it would need a change of the time discretisation (e.g. a Rannacher start-up), which is outside phase 1: the 2D mathematics is frozen.

In the exact-error metric the layer is invisible. At n = 8 the CN error against the exact solution is spatially dominated from N ≥ 32.

## 12. Solver baseline (`t9_solver.sh`, `t9_3d_solver.csv`)

The setup is n = 8, N = 32 (471,648 space-time unknowns), wind A, sin³ data, the production outer rtol 1e-8, and SCHUR_SWEEP with three spatial backends. ASM uses 4 blocks, overlap 1 and ILU(1). Pe_cell = |w|h/(2ν) with h = 1/8.

| scheme | ν (Pe_cell) | backend | outer | spatial its per solve (max) | wall [s] | MaxRSS [MB] | fields vs LU | J vs LU |
|---|---|---|---|---|---|---|---|---|
| BE | 0.25 (0.34) | LU | 1 | 1 | 11.6 | 1063 | — | — |
| BE | 0.25 | GMRES+GAMG | 1 | 11 | 21.7 | 996 | 2.3e-11 | 2.4e-12 |
| BE | 0.25 | GMRES+ASM/ILU | 1 | 9 | 66.1 | 1204 | 7.4e-11 | 5.6e-11 |
| BE | 0.02 (4.2) | LU | 1 | 1 | 11.8 | 1065 | — | — |
| BE | 0.02 | GMRES+GAMG | **no result** | 500 (cap) | killed at 22 min | 989 | — | — |
| BE | 0.02 | GMRES+ASM/ILU | 1 | 6 | 51.2 | 1204 | 2.4e-10 | 2.5e-12 |
| CN | 0.25 (0.34) | LU | 1 | 1 | 15.8 | 1093 | — | — |
| CN | 0.25 | GMRES+GAMG | 1 | 11 | 27.7 | 1009 | 3.2e-11 | 1.3e-12 |
| CN | 0.25 | GMRES+ASM/ILU | 1 | 7 | 63.4 | 1280 | 2.0e-11 | 6.7e-12 |
| CN | 0.02 (4.2) | LU | 1 | 1 | 14.8 | 1093 | — | — |
| CN | 0.02 | GMRES+GAMG | 1 | 74 | 86.8 | 1015 | 6.6e-10 | 3.2e-11 |
| CN | 0.02 | GMRES+ASM/ILU | 1 | 6 | 52.4 | 1278 | 2.2e-10 | 2.9e-12 |

Notes on the table:

- Every converged run has sweep_failures = 0 and reason 2.
- In the BE ν = 0.02 GAMG run, every `fieldsplit_y_sweep_` GMRES+GAMG solve hits the 500-iteration cap (DIVERGED_ITS) and the `sweep_right_` solves need about 300 iterations (`logs/solver/be_nu0.02_gamg_diag.log`). This is the known high-Pe weakness of plain GAMG on the convection-dominated M/Δt + K + C. It is not tuned here (out of scope).
- The ADDITIVE_GAMG BE smoke run (n = 4, N = 16) converges in 18 outer iterations. ADDITIVE_GAMG with CN is **refused** in 3D, with the message "3D phase 1: Crank-Nicolson needs --pc-schur-sweep (ADDITIVE_GAMG is not robust under CN, T7F/T8D finalisation)".
- In the tight exact ladder at n = 16 (6.9 M unknowns), GAMG needs ≤ 13 spatial iterations per solve, 2 outer iterations, and 1446 s versus 566 s for LU (spaceA).

**Validated baseline:**

- SCHUR_SWEEP with LU spatial solves is the reference (one outer iteration at the production tolerance, both schemes, both Pe).
- GMRES+GAMG is equivalent at Pe_cell ≲ 1.
- ASM+ILU is the robust iterative fallback at Pe_cell ≈ 4.

## 13. MPI consistency (`t9_mpi.sh`, `t9_mpi.py`, `t9_3d_mpi.csv`: 12/12 PASS)

The setup is the production path at n = 6, N = 16, w0 and wA, BE and CN, np = 1/2/4, with LU spatial solves (MUMPS for np > 1) and a distributed mesh (`nrank=4 ownership=distributed`).

- Fields agree with np = 1 to ≤ 9.0e-15 relative and J to ≤ 3.7e-14.
- Outer iteration counts are identical (1), and the outer FGMRES histories agree to ≤ 1.7e-17 relative.
- The same 2 history entries appear in every run.
- The true residual is identical to the printed digits.
- Per-rank MaxRSS is 268 → 184 → 124 MB (w0 BE).

## 14. Memory (`t9_memory.py`, `t9_3d_memory.csv`, `figures/3d_memory_vs_unknowns.png`)

Serial MaxRSS is linear in the number of space-time unknowns U = 3 (2n+1)³ N: **≈ 2.2–2.4 KB per unknown** for n ≥ 4 (proportional fit 2385 B/unknown).

The figure is nearly backend-independent. At n = 16, N = 64 LU takes 16.1 GB and GAMG 14.5 GB (−10%); at n = 8 LU takes 1.06 GB and GAMG 1.00 GB. The cost is therefore dominated by the space-time storage (the all-at-once operators and the outer Krylov basis) and not by the spatial factorisations. This is an inference from the backend-independence; per-object accounting was not done.

| n | N | spatial DOFs | unknowns | MaxRSS (measured / fit) |
|---|---|---|---|---|
| 8 | 64 | 4,913 | 0.94 M | 2.0 GB (measured) |
| 12 | 64 | 15,625 | 3.0 M | 6.7 GB (measured) |
| 16 | 64 | 35,937 | 6.9 M | 16.1 GB LU / 14.5 GB GAMG (measured) |
| 20 | 64 | 68,921 | 13.2 M | ≈ 30 GB (fit) |
| 24 | 64 | 117,649 | 22.6 M | ≈ 51 GB (fit) |
| 32 | 64 | 274,625 | 52.7 M | ≈ 120 GB (fit) |

On this 30 GB workstation the serial limit is about n = 20 at N = 64. Beyond that, the MPI path distributes the storage (§13). The measured MaxRSS values come from the build described in §15.1; the extra miniforge libraries add about 20 MB of mappings per process (268 vs 291 MB at n = 6), which is negligible at these sizes.

## 15. 2D regression

### 15.1 Build-environment incident (found by the regression, not a source change)

The first regression pass reported **ex20 goldens 58/61**: T7_mpi_np2, M_reduced_neumann and M_reduced_over_sweep differed at round-off (1e-18 absolute in the KKT residuals; 1009 vs 1042 inner GAMG iterations). The cause is the **build**, not the T9 sources:

- Adding `ExactOc3D.cpp` to CMakeLists.txt made `make` re-run CMake at 12:03 inside a **conda-activated shell**.
- pkg-config then found miniforge's SLEPc 3.24. Its static link list put a second PETSc (3.24), HYPRE, MUMPS, LAPACK and BLAS from `~/miniforge3/lib` on the ex21 link line, with `-isystem ~/miniforge3/include` on the compile line.
- It also caused `libfemus.so` to be rebuilt at 12:04 with SLEPc references.
- The frozen binary's NEEDED/RUNPATH list contains only the PETSc/OpenBLAS tree.

The proof is a **private clean build**, `t9/scripts/t9_clean_build.sh`:

- It recompiles the current T9 sources with the identical flags minus every miniforge token, and links with the identical library list minus miniforge.
- Its NEEDED + RUNPATH are byte-identical to the frozen binary's.
- It runs against the same `libfemus.so` as the frozen binary.
- With it, the ex20 goldens are **61/61** (`logs/t9_regress_ex20_goldens_CLEAN.log`).
- The frozen binary also gives 61/61 (`…_FROZEN.log`); the shared-tree binary gives 58/61 (`…_CONTAMINATED_BINARY.log`).

**None of the 3D results depend on this:**

- operator dumps are byte-identical between the two binaries (`logs/dumps_clean_vs_contaminated.log`);
- the gate CSV is byte-identical;
- the exact ladder at n = 8, the CN time-ladder point and the GAMG solver run are bit-identical (`logs/spot_clean_binary.log`);
- MPI np > 1 runs differ only by 1e-15 (MUMPS/BLAS round-off), and were re-run with the clean binary.

The **shared build tree must be reconfigured in a sanitised environment** before the next real build. That needs your approval, because the action was blocked as a shared-resource modification. The command:

    cd /home/giacomo/software/FEMuS/femusbin-control
    env -i HOME=$HOME PATH=/home/giacomo/software/petsc/arch-linux-c-openblas-opt/bin:/usr/bin:/bin \
        cmake -U 'LIB_*' -U 'SLEPC_*' -U 'pkgcfg_lib_SLEPC_*' -U prefix_result .
    env -i HOME=$HOME PATH=/home/giacomo/software/petsc/arch-linux-c-openblas-opt/bin:/usr/bin:/bin \
        make femus TimeOptCtrl_ex21

Afterwards, check that `readelf -d .../TimeOptCtrl_ex21 | grep miniforge` is empty. The ex20 executable (built 2026-10-03, clean RUNPATH) has not been relinked.

**Resolved (build sanitisation, same day).** These steps were run with your authorisation:

1. The cache was reconfigured in `env -i HOME=$HOME PATH=<petsc arch>/bin:/usr/bin:/bin` with `cmake -U 'LIB_*' -U 'SLEPC_*' -U 'pkgcfg_lib_SLEPC_*' -U prefix_result .`. The cache now has 0 miniforge lines, and SLEPc is not found, as in the original setup.
2. The 14 orphaned `CMakeFiles/<target>.dir` folders of the SLEPc-only targets (NonLocal ex1–ex11, ex20, ex21 and `testSVD2NormCondNumb`) were moved to a scratchpad backup. All of them were created by the polluted 12:03 configure, and those targets are no longer in the build graph.
3. `make VERBOSE=1 femus TimeOptCtrl_ex21` ran in the same environment: 129 compiles and 2 links, with no miniforge or conda token in the log.

Results:

- `libfemus.so` and `TimeOptCtrl_ex21` have no miniforge in readelf or ldd, a single PETSc 3.23, and no SLEPc or HYPRE.
- The executable's NEEDED and RUNPATH are identical to the frozen binary's.
- The `--dimension 3` preset refusals are present.
- ex20 goldens: 61/61.
- Against the private clean build, every serial output is identical and the operator dumps are byte-identical. MPI np = 2 differs by 2e-15, below the 2.3e-15 run-to-run noise of np = 2 itself.

Details are in `t9/scripts/t9_shared_build_check.sh`, `t9/logs/t9_shared_build_check.log` and `t9/logs/t9_regress_ex20_goldens_SHARED_REBUILT.log`.

### 15.2 Regression with the clean binary (`t9_regress_2d.sh`, OLD = frozen 2D binary sha 8eae9680)

The candidate is the final private clean build (sha256 d9f96b42…, NEEDED/RUNPATH identical to the frozen binary), run with `t9/logs/t9_regress_2d.log`.

| block | result |
|---|---|
| R1 ex20 goldens | **61/61** identical |
| R2 Math2025 Ex3 probe, Ex3 sweep nx40 N1600, Ex4 ref | 3/3 BYTE-IDENTICAL against the archived logs |
| R3 Math2025 Ex5 | BYTE-IDENTICAL |
| R3 ErrorFeedback 9.5 | 4 diff lines: the CONFIG line gained `variant=source` (a label added in an earlier task, after the archived log). T8D finalisation showed the same 4 lines; no numbers differ |
| R4 T0/T1/T2 constant wind (dumps, archived runs, gates) | REGRESSION PASS |
| R5 T3/T3b/T4 representative runs (incl. the expected FAILS case and the block test) | 9/9 IDENTICAL, same exit codes |
| (2) old vs new, every EX20_/EX21_ line, timings stripped (core/shell seconds included) | **55/55 IDENTICAL** |
| (2) zero-wind ex20/ex21 parity configurations, `--dump-json` byte-compare | 34/34 JSON_IDENTICAL, plus 2 cases where **both** binaries refuse identically (boundary-reduced + convection, `logs/t9_refusals_extra.log`) |
| (3) restrictions | every 2D refusal unchanged. The 2 intended acceptances (convection + Dirichlet; inert default core) are unchanged. All 16 T9 3D refusals, `--dimension 4` and `--vz` in 2D refuse |

The (2) block covers:

- Dirichlet heat with L2 / H1s / H1t / H1st, under BE and CN, with the dense core;
- the T8D iterative default (heat, and convection + H1st) and `--verify-dirichlet-core`;
- Neumann additive / sweep / reduced;
- T7N exact and production;
- T7D exact, H1s, production H1st and gates;
- T7F natural-conv exact and the multi-zone + two-source production run;
- the nx = 32 dense core with all regularisations;
- the T6 LPS case;
- the natural-BC verify runs;
- conv-sin3 and lift-exp exact.

Zero-wind parity with ex20 therefore carries over unchanged from the T8D finalisation (EX20 parity CSV).

The first pass of this section (`…_CONTAMINATED_BINARY.log`) also exposed the preset-override hole. `--dimension 3 --math2025-ex3` and `--dimension 3 --exact-oc conv-sin3` were *not refused*: they silently ran in 1D/2D. This is fixed in `CommandLine.cpp` (§3), and the re-run above uses the fixed build. The 3D gate CSV and the MPI battery were re-run with the final build: byte-identical and PASS.

**The shared-tree executable** (`femusbin-control/.../TimeOptCtrl_ex21`, built 12:22) **is stale**: it predates the refusal fix. It is also contaminated (§15.1). Rebuilding it was blocked as a shared-resource modification. Rebuild it with the sanitised commands of §15.1.

## 16. Verdict

**3D DISTRIBUTED CONTROL PHASE 1: VALIDATED** (for the source tree as it stands, built clean).

| item | status |
|---|---|
| exact 3D OC, symbolic | 69/69 |
| HEX27 mesh / operators | 39/39 |
| gates G1–G7 with wrong-adjoint negative controls | all pass / all negatives fire under wind |
| spatial order, sin³ data | y 3.02–3.03 (n = 12, 16); p, u ≥ 3 (3.3–3.5 pre-asymptotic) |
| BE | order 1.00 (y), 0.97 (p, u), ref-corrected |
| CN | order 1.99 (p, u), J order 2; y order 2 on [0.25, T] and at T |
| off vs w = 0 parity | operators byte-identical; LU solves to 5e-15 |
| MPI np 1/2/4 | 12/12 PASS |
| solver baseline | SCHUR_SWEEP + LU reference; GAMG for Pe_cell ≲ 1; ASM/ILU for Pe_cell ≈ 4 |
| 2D regression | all identical (with the clean build) |

Caveats, each documented and none of them a phase-1 blocker:

1. The global y L²(0,T) CN self-convergence on the coarse n = 8 mesh shows the CN initial layer of the inherited, frozen time scheme. The 2D frozen binary shows it identically (§11).
2. Plain GMRES+GAMG spatial solves fail under BE at Pe_cell ≈ 4. This is out of scope; LU and ASM/ILU work.
3. The shared build tree was contaminated and has since been sanitised and rebuilt (§15.1).

## 17. What remains for later 3D phases (not started, per instructions)

- 3D Neumann and Dirichlet **boundary control**. This needs the face loop for 2D faces (QUAD9 face Jacobians, M_Γ, and a surface tangential stiffness in place of the 1D arc-length one), the 3D Dirichlet lifting, and the reduced solvers.
- H1-space surface regularisation, and H1-time with boundary control.
- Control/tracking **boxes**, multi-zone and source boxes. This needs 3D centroid boxes and `--domain` with ymin/ymax.
- Natural (Neumann) state BC: the G1N/G8b boundary identities in 3D.
- **Variable 3D winds** (3-component fields) and LPS in 3D.
- High-Pe 3D studies and a robust iterative spatial solver for convection-dominated 3D (GAMG fails at Pe_cell ≈ 4 under BE).
- Tetrahedra, wedges, unstructured 3D meshes; non-cube domains.
- A memory-lean space-time storage, needed beyond about 13 M unknowns serially (§14).
- Ports of the presets and batteries: Math2025 / ErrorFeedback / MMS / literature in 3D.
- A Rannacher-type start-up, if global-in-time CN state accuracy on coarse meshes matters. This would change the frozen time scheme, in 2D too.

## 18. Files

**Modified** (ex21 only; ex20 untouched):

- `CMakeLists.txt`
- `ex21.cpp`
- `include/CommandLine.cpp`
- `include/Config.hpp`
- `include/ConvectionVerification.cpp`
- `include/Discretization.cpp`
- `include/Discretization.hpp`
- `include/Output.cpp`
- `include/Solver.cpp`
- `include/Verification.cpp`

**New:**

- `include/ExactOc3D.hpp`, `include/ExactOc3D.cpp`;
- this directory: the report, 6 CSVs, `figures/` (3 PNGs), `t9/scripts/` (23 scripts), `t9/logs/`.

**Scripts** (`t9/scripts/`):

| script | role |
|---|---|
| `t9_symbolic.py` | symbolic checks |
| `t9_mesh.py` | mesh checks |
| `t9_gates.sh`, `t9_gates_table.py` | gate battery and table |
| `t9_exact_ladder.sh`, `t9_ladder_table.py` | spatial ladders |
| `t9_selfconv.py` | fixed-Δt spatial self-convergence helper |
| `t9_time.sh`, `t9_time.py` | temporal ladders |
| `t9_cn_layer.sh`, `t9_cn_layer.py` | CN initial-layer diagnostic, 3D and frozen 2D |
| `t9_parity.sh` | off vs w = 0 |
| `t9_solver.sh`, `t9_solver_table.py` | solver baseline |
| `t9_mpi.sh`, `t9_mpi.py` | MPI battery |
| `t9_memory.py` | memory audit |
| `t9_figures.py` | figures |
| `t9_clean_build.sh` | private uncontaminated build |
| `t9_regress_2d.sh`, `t9_regress_suite.sh`, `t9_regress_ex3_ex4.sh`, `t9_regress_mathef.sh` | 2D regression |

Large field dumps (JSON) were kept outside the repository, in the session scratchpad and /tmp.
