# T3 — variable wind, inhomogeneous Dirichlet state data, and LP §5.2 in the LPS-free regime

**Result.**
- Both new capabilities are **validated**:
  - variable velocity `w(x)`: forward MMS, exact optimal control, and all gates;
  - inhomogeneous Dirichlet state data by lifting: exact optimal control and gates.
- The constant-wind / homogeneous path is **byte-identical** to before.
- ex21 solves **Leveque–Pearson §5.2 at ε = 1/20** at the node resolution and
  time step of LP's level l = 7. LP's own LPS stabilisation is **inactive** there
  (verified from their definition).
  - For **β = 1e-2**, the setting of LP's Figs. 2–3, the result has **structural
    agreement** with Fig. 2 and no oscillation.
  - The solver is robust for β = 1e-2, 1e-4 and 1e-6 (1–2 outer iterations).
  - At β ≤ 1e-4 the adjoint develops a thin, under-resolved feature at the corner
    (1, +1). It is driven by the target/boundary-data mismatch, not by convection
    (§7.5). LP report no solution at those β.

This is **same problem, different discretisation**: Q2 instead of Q1 + LPS, DTO
instead of OTD, FGMRES + SCHUR_SWEEP instead of MINRES. **No quantitative
reproduction is claimed.** Nothing here concerns ε = 1/100, 1/500 or Pe ≫ 1.

Files:
- data: `t3/results/*.csv`, `t3/logs/` (campaign, gates, regression, solver_pe, post-processing);
- scripts: `t3/run_t3.py`, `t3/analyze_t3.py`, `t3/lp52_post.py`, `t3/lp52_corner_check.py`,
  `t3/lp52_smallbeta_diag.py`, `t3/lp52_patch_peclet.py`, `t3/regress_constant_wind.py`, `t3/regress_2d.sh`;
- summaries: `variable_wind_validation.csv` (59/59 PASS), `dirichlet_validation.csv` (51/51 PASS),
  `lp52_lowpe_results.csv`;
- figures: `figures/variable_wind_mms_convergence.png`, `figures/lp52_y_t1.png`, `figures/lp52_p_t1.png`.

---

## 1. Implementation

### 1.1 Variable velocity (`Config.hpp`, `Discretization.cpp`)

- `ProblemConfig::ConvectionConfig` gains an optional, time-independent
  `field(config, x, z, w[2])` (plus a name for logs).
- **Null is the default**, and the assembly then uses the constant `(vx, vy)`
  exactly as before: the same local array, the same arithmetic.
- When a field is set, the assembly evaluates it at every **physical quadrature
  point**. The point comes from the isoparametric map, the same way
  `IntegrateSpatialField` maps it.
- The Galerkin term stays `C_ij = ∫ (w·∇φ_j) φ_i`. The adjoint is still the
  `MatMultTranspose` of the assembled state operator; no adjoint convection
  operator is built.
- Two named divergence-free fields, selected with `--wind`:
  - `lp-recirculating`: `(2z(1−x²), −2x(1−z²))`, LP's wind with `x2 = z`;
  - `q2-divfree`: `(½ + x², ¼ − 2xz)`.

  Both have components in Q2, so the orientation gate can be exact.
- `EvaluateVelocity()` is used by the forward MMS source and by the exact-OC
  data. With a constant wind it computes the same products in the same order.
- `Validate()` rejects a field without `--convection on`.

### 1.2 Inhomogeneous Dirichlet data by lifting

Code: `Config.hpp`, `Discretization.{hpp,cpp}`, `Solver.{hpp,cpp}`.

- `DataOverrides::dirichlet_state(config, x, z)` is time-independent `g`, and
  null by default. It is accepted only with the Dirichlet state BC and
  distributed control.
- **Construction**, in `BuildNativeSpatialOperators` and before `K` is
  eliminated:
  - `G` = `g` on FEMuS's constrained rows and 0 elsewhere;
  - `lift_load = −(L_full G)` with its constrained rows zeroed.

  `SpatialOperators` owns both vectors.
- **Free rows.** They read `M_FF ẏ₀ + L_FF y₀ + L_FG g = …`, so the homogeneous
  unknown `y₀ = y − G` sees the constant load `−L_FG g` in every slot.
  - BE: `L y_n`.
  - CN: `L (y_n + y_{n−1})/2`.

  Both contribute `L G` once per slot, so the load is scheme-independent.
  `M_FG ġ = 0` because `g` does not depend on time.
- **Threading.** `BuildDistributedRightHandSide` and
  `SolveOptimalControlOwnershipDistributed` gain an optional trailing
  `Vec lift_load = nullptr`. Null is a no-op.
- **Output.** `SolveOptimalControl` passes `spatial.state_lift_load` and adds `G`
  to the returned state, so production output is the physical `y = y₀ + G`.
- **Unchanged.** The constrained rows, the transpose, the adjoint, the control
  and the observation operators are all untouched.
- **The adjoint and control stay homogeneous on ∂Ω.** This is correct for
  distributed tracking and control. They are not given the state's boundary
  values.
- **Conventions inherited from the homogeneous path:**
  - the source load is `M_elim I_h f` (it drops boundary nodal values of `f`);
  - the tracking residual is evaluated on the free DOFs (it drops the boundary
    residual `g − y_d|_∂Ω`).

  Both are exact when `f|∂Ω = 0` and `y_d|∂Ω = g`. The manufactured tests are
  built that way. For LP §5.2, `f = 0`, but `y_d|∂Ω ≠ g` on `x2 = ±1` near
  `x1 = 1` (§7.5).
- **CN initial trace.** The existing CN fix zeroes `y₀`'s trace in the
  first-step term, which is exactly the homogeneous part of `y₀ = y₀,hom + G`.

### 1.3 Drivers and gates

- **`ExactOcBenchmark.cpp`:**
  - new cases:
    - `lift-const` (`g = 1`);
    - `lift-exp` (`g = ½ + exp((v·x − v_x − v_y)/ν)`, a steady solution for the
      constant wind; nonzero, non-polynomial trace);
    - `lp52`;
  - `--wind` is honoured by `conv-sin3` and by the `lift-*` cases;
  - boundary checks (`max|y − g|`, `max|p|`, `max|u|` over every boundary node and
    slot), a solver-statistics line, and a lift-aware independent march;
  - `--exact-oc-production-tol` (the unmodified production solver policy) and
    `--exact-oc-snapshot`.
- **`ConvectionVerification.cpp`:**
  - G1 orientation for a field: `C·x_lin = M·I_h w_axis`, exact for Q2 `w`;
  - the reduced problem and G4 use the lift load, and G4 compares `y₀ + G`;
  - the MMS source uses `EvaluateVelocity`.
- **`CommandLine.cpp` / `Options.hpp`:** `--wind`, `--exact-oc-production-tol`,
  `--exact-oc-snapshot`.

## 2. Variable-wind validation

`w = (½ + x², ¼ − 2xz)`, `ν = 1/4`, unit square. `max|w| = 2.30`, so the cell
Péclet number `|w|h/(2ν)` is 1.15, 0.58, 0.29, 0.14 and 0.07 for nx = 4…64.
Both components vary in space.

### 2.1 Forward MMS (`--convection-mms sin3 --wind q2-divfree`, T = 1)

| ladder | errors | orders |
|---|---|---|
| space, CN Nt = 512, `‖e(T)‖` rel., nx = 4, 8, 16, 32 | 3.84e-2, 4.36e-3, 5.33e-4, 6.63e-5 | 3.14, 3.03, **3.01** |
| CN, nx = 64, `max_n‖e(t_n)‖`, Nt = 4…64 | 2.12e-3 … 8.18e-6 | 2.02, 2.00, 2.00, **1.99** |
| BE, nx = 64, `max_n‖e(t_n)‖`, Nt = 4…64 | 1.63e-2 … 1.24e-3 | 0.85, 0.93, 0.96, **0.98** |

- The all-at-once residual of the march is at most 3e-13.
- BE `‖e(T)‖` at the last level alone is pre-asymptotic (0.54, 0.40, 0.73,
  0.87). The max-over-time norm is the one the original ladders use
  (`EX21_CONVECTION_VALIDATION.md` §F.1).

### 2.2 Exact optimal control (`--exact-oc conv-sin3 --wind q2-divfree`)

This is the T2 solution, with the wind entering `f` and `y_d`.

| ladder | y | p (all slots) | p, t ≤ T/2 |
|---|---|---|---|
| space, CN Nt = 128, nx = 4 → 32 | orders 3.28, 3.03, **3.01** | 3.86, 3.15, **3.00** | 3.88, 3.16, **3.05** |
| CN time, nx = 32, Nt = 4 → 32 | 2.28, 2.29, 1.60* | 1.79, 1.91, **1.94** | 2.52, 2.16, 1.48* |
| BE time, nx = 32, Nt = 8 → 64 | 1.04, 1.02, **1.01** | 1.02, 1.00, **1.00** | – |

\* At Nt = 32 the error reaches the nx = 32 spatial floor (y ≈ 6.6e-5,
p ≈ 3.8e-4), as in T2.

Solver and algebraic checks on every run:
- 2 outer iterations;
- relative KKT residual ≤ 1.4e-14;
- independent march ≤ 4.5e-14;
- on the boundary: `|y| ≤ 3e-20`, `|p|, |u| ≤ 3e-23`.

### 2.3 Gates (`--verify-convection`)

| gate | q2-divfree, CN / BE (nx = 8) | lp-recirculating + LP lifted data, CN nx = 8 / BE nx = 48 |
|---|---|---|
| orientation `C·x_lin = M I_h w` (transposed orientation) | 4.5e-14 (12.4–13.8) | 2.7e-15 / 9.5e-14 (2.0) |
| skew-symmetry of `C` on the free DOFs (div w = 0) | 1.3e-14 | 1.2e-14 |
| space-time transpose | 8.7e-18 / 4.0e-18 | 7.4e-19 / 8.0e-18 |
| FD reduced gradient (production adjoint) | 2.2e-14 / 6.1e-13 | 5.4e-14 / 1.3e-12 |
| reduced Hessian symmetry | 9.9e-14 / 0 | 5.7e-15 / 1.8e-13 |
| negative control (`L` untransposed) | fails: 53 % / 41 % | fails: 223 % / 98 % |
| G4 production solve | converged | converged |

## 3. Inhomogeneous Dirichlet validation

Cases:
- **`lift-exp`:** constant `v = (1, ½)`, `ν = 1/4`,
  `y = g + cos(2t) S1`, with
  `g = ½ + exp((v·x − v_x − v_y)/ν)` (trace in [0.502, 1.5], non-polynomial),
  and with `p`, `u` as in T2.
  - Since `−νΔg + v·∇g = 0`, `f` vanishes on ∂Ω.
  - `y_d|∂Ω = g`, so both inherited conventions are exact for this case.
- **`lift-const`:** `g = 1`, combined with the **variable** wind. This is the
  feature combination LP need. A constant trace is the only simple one that
  keeps `f|∂Ω = 0` for a variable wind.

| ladder | y | y(T) | p (all slots) | p, t ≤ T/2 |
|---|---|---|---|---|
| lift-exp, space, CN Nt = 128, nx = 4 → 32 | 3.29, 3.03, **3.01** | 3.12, 3.02, 2.94 | 3.88, 3.17, **3.00** | 3.91, 3.17, **3.05** |
| lift-exp, CN time, nx = 32, Nt = 4 → 32 | 2.21, 2.27, 1.59* | 1.71, 1.89, **1.93** | 1.78, 1.91, **1.94** | 2.54, 2.18, 1.47* |
| lift-exp, BE time, nx = 32, Nt = 8 → 64 | 1.01, 1.01, **1.00** | 0.66, 0.81, 0.89 | 1.02, 1.00, **1.00** | 1.01, 1.00, 1.00 |
| lift-const + q2-divfree, space | 3.28, 3.03, **3.01** | 3.14, 3.03, 2.94 | 3.86, 3.15, **3.00** | 3.88, 3.16, **3.05** |
| lift-const + q2-divfree, CN time | 2.20, 2.26, 1.58* | 1.72, 1.90, **1.94** | 1.79, 1.91, **1.94** | – |

\* Spatial floor, as above.

**Boundary values** (all slots, all boundary nodes):
- `max|y − g|` ≤ 1.5e-13. The CN constrained rows `½y_n + ½y_{n−1} = 0` are
  solved iteratively, hence round-off rather than bitwise 0.
- `max|p|, |u|` ≤ 1.1e-16: **the adjoint stays homogeneous**.
- The gate criterion is ≤ 1e-14. "Exactly 0" was an over-strict first
  expectation, corrected before the gate was applied.

**Solver:**
- 1–2 outer iterations;
- relative KKT residual ≤ 6.5e-12;
- independent lift-aware march ≤ 1.7e-12.

**Gates (`lift-exp`, CN / BE nx = 8):**
- transpose 7.3e-18 / 1.6e-17;
- FD gradient 1.7e-12 / 4.4e-12 (the lift enters the rhs of the reduced
  functional);
- Hessian 2.8e-15 / 6.3e-14;
- G4: the production `y = y₀ + G` agrees with the reduced state plus `G` to
  2.2e-11 / 1.0e-11.

## 4. Regression (existing behaviour unchanged)

| gate | result |
|---|---|
| F: constant-wind operator dumps `M, L, N, M_track, E, Eᵀ` (CN and BE) vs the T0 dumps | **12/12 byte-identical** |
| B: 8 archived T1/T2 `--exact-oc` runs (every result key except wall time, plus every per-slot line) | **8/8 identical** |
| T2 `--verify-convection` gate logs (CN, BE) | **identical** (36 lines each) |
| A: ex21 default path vs the ex20 goldens (`be_nonregression.py`) | **61/61 identical** |
| C: Mathematics 2025 Ex. 3 (2 runs) and Ex. 4 archived runs | **3/3 byte-identical** |
| D: Mathematics 2025 Ex. 5 (`d-zero`, CN nx 24×8, Nt 200) | **byte-identical** (413 lines) |
| E: ErrorFeedback 9.5 (α_u 1e-7, CN nx 32×12, Nt 400) | numerical lines identical. Only the CONFIG label line differs: `variant=source` was added to that print in a pre-T3 stage, after the archived log |

## 5. LP §5.2 — source data, re-transcribed from the PDF (pp. 19–20)

| item | value | source |
|---|---|---|
| Ω, T, d | (−1,1)², `t_f = 2`, 2 | [A] |
| ε | 1/20 (this phase) | [A] (Tables 6–8 also use 1/100 and 1/500, not used here) |
| w | `[2x2(1−x1²), −2x1(1−x2²)]ᵀ` | [A] |
| ŷ | `e^{−10(1−x1)}`, time-independent | [A] |
| f | 0 | [A] |
| Dirichlet data | 1 on `∂Ω1 = {1}×[−1,1]` (closed segment, so the corners (1,±1) take 1), 0 elsewhere | [A] |
| initial condition | 1 if `x1 = 1`, else 0, i.e. equal to the boundary data and 0 inside | [A] |
| control, observation | full domain (J (1)) | [A] |
| β | Figs. 2–3: 1e-2; Tables: 1e-1…1e-6 | [A] |
| time | CN, `τ = h = 2^{1−l}` | [A] |
| element | LP Q1 + LPS (δ_k = 0 if the patch Pe ≤ 1); ex21 Q2, no stabilisation | [A]/ex21 |

**Mapping.** ex21's `α_u = β` exactly (`η = α_track = 1`, full-domain `N = M_ω = M`;
audit §3), and `p_LP = −p_ex21`.

## 6. Péclet and LPS status, verified from LP's definition

LP (§2.1):
- patches of 2×2 Q1 elements;
- `δ_k = 0` iff `Pe_k = ‖w_k‖ h_k/(2ε) ≤ 1`, where `w_k` is the wind at the
  patch centroid and `h_k` the patch length in the wind direction.

`t3/lp52_patch_peclet.py` evaluates this on every patch, using the chord of the
patch through its centroid along `w_k`, and also the pessimistic `h_k = 2√2 h`:

| ε = 1/20, l | h | max Pe_k (chord) | max Pe_k (2√2 h) | patches with Pe_k > 1 | LPS |
|---|---|---|---|---|---|
| 5 (LP Figs. 2–3) | 1/16 | 2.33 | 3.30 | 212/256 | **ON** |
| 6 | 1/32 | 1.21 | 1.71 | 136/1024 | **ON** |
| 7 | 1/64 | **0.62** | 0.87 | **0/4096** | **off** |
| 8 | 1/128 | **0.31** | 0.44 | **0/16384** | **off** |

So the audit's estimate (inactive at l = 7, 8) holds under both readings of `h_k`.
LP's own Figs. 2–3 (l = 5) **are stabilised**.

**ex21 runs.** Q2 with `nx = 2^{l−1}` elements has exactly the node set of LP's
Q1 level l. At **Q2 nx = 64** (the LP l = 7 node set, `τ = 1/64`, Nt = 128):
- cell Pe `= max|w|·h_e/(2ε) = 0.63` (element `h_e = 1/32`);
- node-spacing Pe = 0.31.

Two lower levels serve as references:
- nx = 32 (l = 6 node set, cell Pe 1.25, node Pe 0.63);
- nx = 16 (l = 5 node set, cell Pe 2.5, node Pe 1.25).

A literal Q2 nx = 128 run (about 18 GB) was not launched.

## 7. LP §5.2 results (CN, production solver tolerances)

### 7.1 β = 1e-2

| Q2 nx (LP l) | Nt | outer its | relative KKT (state / adjoint) | march | y range (all t) | y(t=1) range | p_LP(t=1) range | wall |
|---|---|---|---|---|---|---|---|---|
| 16 (5) | 32 | 2 | 1.5e-14 / 2.4e-14 | 1.4e-14 | [−2.1e-7, 1] | [−1.9e-7, 1] | [−0.02138, 0.00498] | 12 s |
| 32 (6) | 64 | 1 | 5.3e-11 / 4.4e-8 | 1.8e-11 | [−2.6e-7, 1] | [−2.3e-7, 1] | [−0.02122, 0.00589] | 15 s |
| **64 (7)** | 128 | 1 | 2.3e-11 / 4.5e-8 | 1.2e-11 | [−2.6e-7, 1] | [−2.4e-7, 1] | [−0.02122, 0.00586] | 78 s (4.5 GB) |

- The adjoint residual ~4e-8 matches the production `rtol` of 1e-8.
- Boundary values: `|y − g|` ≤ 7e-13, `|p|` ≤ 4e-14, `u = 0`.
- All fields are finite.

### 7.2 Qualitative comparison with LP Fig. 2 (ε = 1/20, β = 1e-2, l = 5, raster)

| feature | LP Fig. 2 | ex21 (`figures/lp52_y_t1.png`, `lp52_p_t1.png`, nx = 64) |
|---|---|---|
| y range | 0 … 1 | [−2.6e-7, 1] |
| y structure | ≈ 0 in most of Ω; steep layer rising to 1 at x1 = 1 | same: layer at x1 = 1, ≈ 0 elsewhere |
| p sign/range | ≤ 0 mostly, minimum ≈ −20×10⁻³; small positive part | [−0.0212, +0.0059] |
| p structure | trough near x1 ≈ 0.8, deeper toward x2 = −1; small positive lobe near the (1, +1) side | trough at (0.875, −0.67); positive lobe near (0.85, 0.95) |

**Structural / qualitative agreement.** LP's figure is a stabilised Q1 OTD
solution and a raster image, so no digits are compared.

### 7.3 Oscillations and mesh sensitivity (β = 1e-2)

- **Node-to-node wiggle rows** (≥ 3 consecutive alternations of `∂²/∂x1²` above
  1e-3 of the row range): **0** for both y and p on every mesh.
- **Undershoot:** `y_min = −2.6e-7`, i.e. none of significance.
- **Level-to-level differences** at shared nodes (`t3/lp52_corner_check.py`;
  max-norm differences relative to the max of the fine field):

  | | y: RMS | y: max | y: max, dist > 0.25 from (1,±1) | p: RMS | p: max | p: max, dist > 0.25 | p: max, dist > 0.5 |
  |---|---|---|---|---|---|---|---|
  | nx 16→32 | 1.1e-3 | 1.9e-2 | 2.9e-3 | 4.1e-3 | 7.9e-2 | 1.4e-2 | 7.6e-3 |
  | nx 32→64 | 5.7e-4 | 2.3e-2 | **1.7e-4** | 9.4e-4 | 2.8e-2 | 4.0e-3 | **6.4e-4** |

  - The max-norm difference does not decrease because it sits at the two
    **boundary-data discontinuities (1, ±1)**, where the Dirichlet data jump from
    1 to 0. That is a corner singularity, so no max-norm convergence is expected
    there.
  - Away from the corners the differences fall 12–17× per refinement.
  - The corner effect is local and not oscillatory. It is a property of LP's
    data, not of the generic inhomogeneous-Dirichlet support, which converges at
    order 3 for smooth data (§3).

### 7.4 β robustness (solver)

| β | Q2 nx / Nt | outer its | Schur its (per outer) | max spatial GMRES its per solve | sweep failures | relative KKT (adjoint) | wall |
|---|---|---|---|---|---|---|---|
| 1e-2 | 32 / 64 | 1 | 8 | 21 | 0 | 4.4e-8 | 15 s |
| 1e-2 | 64 / 128 | 1 | 8 | 13 | 0 | 4.5e-8 | 78 s |
| 1e-4 | 32 / 64 | 1 | 9 | 22 | 0 | 1.6e-8 | 13 s |
| 1e-4 | 64 / 128 | 1 | 9 | 13 | 0 | 1.3e-8 | 86 s |
| 1e-6 | 32 / 64 | 2 | 8.5 | 24 | 0 | 7.3e-15 | 22 s |
| 1e-6 | 64 / 128 | 1 | 9 | 13 | 0 | 1.4e-8 | 84 s |

At nx = 16 (cell Pe 2.5): 2 outer iterations, 8 Schur iterations per outer, and
spatial GMRES up to 129.

**Robustness holds.** For this full-domain, colocated problem with a variable
wind, ex21's solver stays at 1–2 outer iterations, with flat Schur and spatial
counts, over β = 1e-2…1e-6 and the two LPS-free-equivalent meshes. The counts
are not comparable one-to-one with LP's MINRES counts (15–26 in Table 6); only
the pattern is.

### 7.5 Small β: an under-resolved boundary/corner feature of the adjoint (not convection)

From `t3/lp52_smallbeta_diag.py` and `lp52_post.py`:

| β | nx | max|p_LP| at | dist. to ∂Ω | share of max|p| beyond 0.1 from ∂Ω | odd-even amplitude: boundary band / interior | p wiggle rows | p change nx 32→64 |
|---|---|---|---|---|---|---|---|---|
| 1e-2 | 64 | (0.875, −0.67) | 0.125 | 100 % | 0.04 / 0.015 | 0 | 2.8 % |
| 1e-4 | 64 | (0.953, 0.969) | 0.031 | 22 % | 0.41 / 0.018 | 0 | 27 % |
| 1e-6 | 32 | (0.938, 0.969) | 0.031 | 8 % | 1.24 / 0.015 | 59/65 | – |
| 1e-6 | 64 | (0.969, 0.984) | 0.016 | 2 % | 0.95 / **5.9e-4** | 43/129 | O(1) |

- As β decreases, the adjoint's peak moves into a layer of 1–2 node widths at
  the corner (1, +1). There the target `y_d → 1` while the state is pinned to
  `g = 0` on `x2 = ±1` (`|y_d − g|` on `x2 = ±1` is 0.08 at `x1 = 0.75` and 1
  at the corner).
- This tracking-induced layer sharpens as β → 0 and is **under-resolved** at
  these meshes. The node-to-node wiggles live in the boundary band; in the
  interior the odd-even amplitude is small and falls with refinement
  (β = 1e-6: 1.5e-2 → 5.9e-4).
- **Cell Pe ≤ 0.63 at nx = 64.** This is **not** the unstabilised-convection
  oscillation that LPS addresses. It is a resolution issue of the β-dependent
  layer, plus LP's discontinuous corner data.
- The inherited convention of dropping the boundary tracking residual
  `g − y_d|∂Ω` (§1.2) acts in exactly this band, so it may contribute. This was
  not separated.
- LP publish **no** solution at β = 1e-4 or 1e-6 (iteration counts only), so
  there is nothing to compare. The state is well behaved throughout:
  `y ∈ [−2e-8, 1.023]`. The 1.023 at β = 1e-4, nx = 64 is a 2 % overshoot at
  the same corner.

### 7.6 Solver observation outside the regime (recorded for the high-Pe phase)

On a deliberately coarse LP-wind mesh, **BE** with `dt = 0.125`:

| run | outcome |
|---|---|
| Q2 nx = 8 (cell Pe 5) | the Schur inner solves converge (15–17 its), but the **outer FGMRES stagnates at 0.34** |
| nx = 16 (cell Pe 2.5) | **every spatial GMRES(50)+GAMG solve in the sweeps hits its 500-iteration cap**; the solve aborts with KSP reason −11 and NaN (reported, not silent) |
| nx = 24 (Pe 1.67) and nx = 48 (Pe 0.83) | 2 outer iterations |

CN on nx = 8 and nx = 16 converged. Logs: `t3/logs/solver_pe/`.

So the spatial sweep solver (GMRES+GAMG on `M/dt + L` blocks) is **not robust
for convection-dominated blocks** under BE. This is the first concrete item for
the high-Péclet phase. It is a solver finding, not a discretisation one, and no
claim is made about its cause beyond "GAMG on nonsymmetric, convection-dominated
blocks".

## 8. Acceptance (brief §14)

| criterion | status |
|---|---|
| **Variable wind:** constant-wind regression unchanged | **PASS** (byte-identical dumps, runs, gates; 61/61; Ex3/4/5; EF 9.5 numerics) |
| manufactured variable-wind state at the expected orders | **PASS** (Q2 3.01; CN 2.00; BE 0.98; exact OC 3.01 / 1.94 / 1.00) |
| sign/orientation checks | **PASS** (round-off vs O(1) transposed, both winds) |
| no transpose issue | **PASS** (≤ 1.6e-17; FD ≤ 1.3e-12; Hessian ≤ 1.8e-13) |
| **Inhomogeneous Dirichlet:** smooth manufactured problem at the expected orders | **PASS** (3.01 / 1.94 / 1.00; also with the variable wind) |
| boundary values correct | **PASS** (≤ 1.5e-13) |
| adjoint homogeneous where required | **PASS** (≤ 1.1e-16) |
| true KKT residual small; independent march agrees | **PASS** (≤ 6.5e-12; ≤ 1.7e-12) |
| **LP low-Pe:** source data implemented faithfully | **PASS** (§5) |
| LP's LPS inactive on the chosen resolution | **PASS** (l = 7 node set; max patch Pe 0.62, 0 of 4096 patches active) |
| small true residual | **PASS** (relative ≤ 4.5e-8, at the production rtol; march ≤ 1.2e-11) |
| y, p qualitatively as LP's figures | **PASS** at β = 1e-2 (structural agreement, §7.2) |
| no significant unstabilised oscillation | **PASS** at β = 1e-2 (0 wiggle rows). At β ≤ 1e-4 there are wiggles in a boundary/corner band from an under-resolved β-layer, with Pe ≤ 0.63 (§7.5): documented, not attributed to convection |
| solver iterations controlled | **PASS** (1–2 outer iterations, β = 1e-2…1e-6) |

**This validates the variable-wind / inhomogeneous-Dirichlet extension in a
regime where LP's own stabilisation is inactive.** Nothing is inferred about
ε = 1/100, 1/500 or Pe ≫ 1.

---

**Correction note (T3b/T4, added later; the text above is unchanged as a
record).** T3 did not include the affine tracking term `M_FΓ (g − y_d)_Γ` of
inhomogeneous Dirichlet data. Without it, the adjoint load drops the boundary
tracking residual, and the reported `J_track` was evaluated on the homogeneous
unknown. Both are corrected in `T3B_AFFINE_DIRICHLET_TRACKING.md`.

**Effect on the results above:**
- The lift-exp / lift-const results (§3) are unchanged to round-off, because
  `y_d = g` on ∂Ω there.
- LP §5.2, β = 1e-2: same structure. The mesh sensitivity of p drops from
  2.8e-2 to 7.5e-3, and the structural agreement with LP Fig. 2 is kept.
- LP §5.2, β = 1e-4: the mesh sensitivity of p drops from 27 % to 9.5 %.
- LP §5.2, β = 1e-6: the corner layer at (1, +1) remains. It is
  resolution/physics-driven, not an artifact of the dropped term
  (`lp52_smallbeta_corrected.csv`).

**§7.6 (BE solver failure).** The cause is now isolated: it is the spatial
GMRES+GAMG preconditioner on blocks `M/dt + θL`, which depend only on θ·dt. It
is not the KKT/Schur strategy, which converges in 1–2 outer iterations with
exact spatial solves. See `T4_HIGH_PECLET_UNSTABILISED.md`.
