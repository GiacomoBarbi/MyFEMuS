# T7N — Neumann boundary control for transient convection–diffusion OC (ex21, 2D)

**Scope:**

- 2D only.
- Neumann control only: no Dirichlet control, no H1-space/H1-time regularisation, no 3D.
- Unchanged: the KKT formulation, the stabilisation, LPS, ex20.
- Nothing committed, pushed or staged.

**Result:** Neumann boundary control under convection is **VALIDATED** for the generic
SCHUR_SWEEP path (§14).

- Every item of the definition in step 17 passes.
- The boundary-reduced Schur solver stays refused under convection.
- Dirichlet control stays refused.

Scripts: `scripts/`. Logs: `logs/`. Raw results: `results/`. Tables:

- `t7n_matrix_gates.csv` (30/30 PASS);
- `t7n_convergence.csv`;
- `t7n_solver.csv`.

## 0–1. The optimality system (derived from the current code, then checked on the matrices)

**State** (ex21 strong-form convection, natural state BC; no state elimination with Neumann control):

    y_t − ν Δy + w·∇y = f                    in Ω × (0,T)
    ν ∂_n y = η u                            on Γ_c × (0,T)        (controlled face)
    ν ∂_n y = 0                              on ∂Ω \ Γ_c            (natural)
    y(·,0) = y_0

**Weak form, as assembled:**

- (ẏ, φ) + ν(∇y, ∇φ) + (w·∇y, φ) = (f, φ) + η ∫_{Γ_c} u φ.
- Discretely: M ẏ + L y = F + η M_Γ u, with L = νK + C, C_ij = ∫ (w·∇φ_j) φ_i, and M_Γ the
  trace mass of face Γ_c (`neumann_control_face`; face 1 = {z = 0}).
- `SelectedControlOperator` returns N = M_Γ. Every KKT block, sweep block and the control
  reconstruction consume N generically.

**Objective:**

    J = (α_t/2) ∫_0^T ‖y − y_d‖²_{L²(ω)} + (α_u/2) ∫_0^T ‖u‖²_{L²(Γ_c)}

The regularisation is L2(Γ_c) via M_Γ; there is no H1 term.

**DTO adjoint.** The KKT adjoint row is Eᵀp = α_t W M_ω (y − y_d); ex21's p is minus the
Lagrange multiplier, so u = −(η/α_u)p. Eᵀ applies **Lᵀ**.

- For a discrete p_h: (Lᵀp)_i = ν(∇φ_i, ∇p_h) + ∫ (w·∇φ_i) p_h.
- Integrating by parts (div w = 0): ∫ (w·∇φ_i) p = −∫ φ_i w·∇p + ∫_∂Ω (w·n) φ_i p.

So the continuous counterpart is

    −p_t − ν Δp − w·∇p = α_t (y − y_d) χ_ω,    p(T) = 0,
    ν ∂_n p + (w·n) p = 0                       on ALL of ∂Ω (Γ_c included)

Γ_c is not special for the adjoint: the control enters only the state equation and the
stationarity condition.

**Stationarity** (on Γ_c, in the control metric):

    α_u M_Γ u + η M_Γ p = 0   ⇔   u = −(η/α_u) p|_{Γ_c}

- ex21 reconstructs it nodewise on the active set: the support of M_Γ, i.e. the
  2·nx + 1 nodes on z = 0.
- Sign check: the "wrong-sign" diagnostic gives relative error exactly 2.00 (§9).

**Verified on the matrices, not assumed (gate G1N, §7):** C + Cᵀ = Σ_faces (w·n_f) M_f to
3.5e-14 relative, for three winds and both schemes. The four faces and normals were
identified automatically from the face-mass supports. This is the discrete statement of
the adjoint boundary term.

**CN control:**

- Interval-constant in time (ex20 stage 2): the forcing on interval n is
  η M_Γ u_{n−1/2}, and J_reg = (α_u/2) Σ dt u_{n−1/2}ᵀ M_Γ u_{n−1/2} (exact for piecewise
  constants; the code uses uniform weights for non-Dirichlet control).
- Adjoint and control are compared at interval midpoints t_{n−1/2} (T2 convention).

## 2. Symmetry audit of the Neumann code path

grep over `include/` for `KSPCG|MINRES|CHOLESKY|potrf|MAT_SPD|MAT_SYMMETRIC|MatMult(stiffness|transpose`
and every `ControlType`/`M_Gamma` branch. Then each hit was read.

| # | component | Neumann-path role | assumption | classification |
|---|---|---|---|---|
| 1 | `SelectedControlOperator` (Discretization) | N = M_Γ | M_Γ symmetric (true) | **SAFE** |
| 2 | forward shell AY (SpaceTimeOperators) | applies L, A± | none | **SAFE** |
| 3 | adjoint shell AP | Lᵀ / A±ᵀ via `transpose_spatial = HasConvection` | ex20: "M and K symmetric, so A± are their own transposes" | **SAFE** (transposed under convection since ex21 stage 1, audit #2) |
| 4 | assembled Eᵀ (`ap = MatTranspose(ay)`) | GAMG operand | none | **SAFE** (exact) |
| 5 | KKT (1,2)/(2,1) blocks | (η²/α_u) I⊗N, −α_t W⊗M_ω | N, M_ω symmetric | **SAFE** |
| 6 | CN right-hand side (Solver.cpp:201) | (M/dt − L/2) y_0 in the first state row | forward product: L, not Lᵀ, is correct | **SAFE** |
| 7 | control reconstruction / J_reg / KKT residuals | u = −(η/α_u)p on supp M_Γ; J_reg in M_Γ | none | **SAFE** |
| 8 | sweep blocks (Preconditioners.cpp:180–262) | right = A + √β N; left = Aᵀ + √β M_ω; CN coupling transposed | without convection: `MAT_SPD`/`MAT_SYMMETRIC` flags | **SAFE** (convection branch transposes and sets no flags) |
| 9 | spatial sweep solver | CG+GAMG | CG on a nonsymmetric block | **SAFE** (GMRES(50), right PC under convection) |
| 10 | matching auxiliaries | right = E + √β(I⊗N), left = Eᵀ + √β(W⊗M_ω) | from assembled ay / ap | **SAFE** (algebra); solver quality: §11 |
| 11 | ADDITIVE_GAMG | GMRES+GAMG on ay, ap | preconditioner only | **SAFE** (algebra); solver quality: §11 |
| 12 | **`BoundaryReducedSchur`** | exact Schur on the control support | CHOLMOD/MUMPS **Cholesky**, `MAT_SPD`, A₊/A₋ treated as symmetric in `ApplyTransposeE` (ex20 stage 2: "A₊ and A₋ are symmetric") | **SOLVER-ONLY LIMITATION** → refused under convection (CLI, explicit error) |
| 13 | `--verify-boundary-reduced`, `--verify-stage1/-sweeps` | ex20 batteries | symmetric arrangements | refused under convection (pre-existing, unchanged) |
| 14 | `DirichletControlOperators` | not on the Neumann path | B_Dᵀ uses L | not touched (Dirichlet stays refused) |

**No ACTUAL MATHEMATICAL BUG and no NEEDS-TRANSPOSE item was found on the generic Neumann
path.** The only heat-era symmetry assumption in the Neumann CN derivation lives in the
boundary-reduced solver.

## 3–4. Restrictions changed (minimum)

| restriction | before | after |
|---|---|---|
| `Validate`: convection ⇒ distributed only | rejected Neumann and Dirichlet | **convection + NEUMANN accepted**; convection + DIRICHLET still rejected ("B_D^T is not convection-ready") |
| `--pc-schur-boundary-reduced` + convection | unreachable (Neumann rejected) | **explicitly rejected** with an error ("assumes a symmetric spatial operator; use --pc-schur-sweep …"); never silently replaced |
| default PC for Neumann | ADDITIVE_GAMG | unchanged (no silent mode change). Its CN failure is pre-existing (§11) and caught by the acceptance guard |
| H1-in-time | Dirichlet only | unchanged (still refused for Neumann) |
| ex20 verification batteries | refused with convection | unchanged |

## 5. Manufactured exact OC (`--exact-oc neumann-conv`)

Setup:

- Ω = (0,1)², Γ_c = {z = 0}, constant w = (w_x, w_z) set by `--exact-oc-velocity`
  (default (1, 0.5));
- ν = 0.25, η = 1, α_t = 1, α_u = 1e-2, T = 1;
- tracking over Ω, natural state BC elsewhere.

Derivation and verification: `scripts/t7n_symbolic.py`, log `logs/t7n_symbolic.log`.

    X(x) = 1 − cos 2πx                       (X = X′ = 0 at x = 0, 1)
    Z(z) = e^{−(w_z/ν) z} (2 + cos πz)        (ν Z′ + w_z Z = 0 at z = 0, 1)
    H(z) = z − z²/2                          (H′(0) = 1, H′(1) = 0)
    p = α_u sin(T − t) X Z                    → ν ∂_n p + (w·n) p = 0 on all 4 faces, p(T) = 0
    u = −(η/α_u) p|_{z=0} = −η sin(T − t) X(x) Z(0)          (Z(0) = 3; u(0.5, 0) = −5.05)
    y = cos 2t cos πx cos πz + κ sin(T − t) X H,  κ = η² Z(0)/ν
                                              → −ν ∂_z y = η u on z = 0, ν ∂_n y = 0 elsewhere
    f = y_t − ν Δy + w·∇y,     y_d = y − (−p_t − ν Δp − w·∇p)/α_t

**Symbolic checks (sympy, exact): 12/12 zero.** They cover:

- the state and adjoint PDE residuals;
- the 4 state BCs (3 natural, plus the controlled flux);
- the 4 convective adjoint BCs;
- p(T) = 0;
- stationarity on Γ_c.

The **symmetric-diffusion** adjoint condition ν∂_n p = 0 is violated by exactly w_z p on
z = 0, so the exact solution genuinely exercises the convective boundary term whenever
w_z ≠ 0.

**Implementation** (`ExactOcBenchmark.cpp`, case `NEU_CONV`):

- natural state BC, NEUMANN control on face 1;
- y_d and f from closed-form derivatives;
- **u error in L2(Γ_c)** via `IntegrateBoundaryField`, the same face quadrature that
  assembled M_Γ;
- the exact J_reg is a boundary integral;
- new `EX21_NEUMANN_STATIONARITY` line: ‖M_Γ(α_u u + η p)‖ / ‖α_u M_Γ u‖, max|u| off Γ_c,
  number and z-range of the control nodes.

Controlled face, measured: 2·nx + 1 control nodes, all at z = 0; max|u| off Γ_c = 0
exactly; no Dirichlet elimination (state BC natural; Validate already rejects
Neumann + homogeneous Dirichlet state).

## 6–7. Algebraic gates (`scripts/t7n_gates.sh`, `logs/gates/`, `t7n_matrix_gates.csv`)

`--verify-convection --exact-oc neumann-conv`, nx = 8, Nt = 16, LU sweeps, BE and CN, three winds:

| wind | w | w·n on Γ_c (z = 0) | z = 1 | x = 0 / x = 1 |
|---|---|---|---|---|
| A | (1, 0.5) | −0.5 (inflow) | +0.5 | −1 / +1 |
| B | (1, −0.5) | +0.5 (outflow) | −0.5 | −1 / +1 |
| C | (1, 0) | 0 (tangential) | 0 | −1 / +1 |

**New gates in `ConvectionVerification.cpp`.** They are printed only for natural state BC +
constant wind + no LPS, so every existing (Dirichlet-state) gate log is unchanged.

- **G1N:** the boundary identity C + Cᵀ = Σ_f (w·n_f) M_f.
- **G8b, the boundary-wrong adjoint:** spatial adjoint operator νK − C. That is the correct
  interior operator −νΔp − w·∇p with the symmetric-diffusion condition ν∂_n p = 0; it
  differs from Lᵀ exactly by C + Cᵀ.

Ranges over the 6 runs (all PASS):

| gate | measured | target |
|---|---|---|
| G1 convection orientation (C·lin = v M1) | 3.4e-14 … 5.8e-14 | round-off |
| G1N boundary identity | 3.5e-14 … 3.6e-14 | round-off |
| G2 space-time transpose ⟨Ex,y⟩ − ⟨x,Eᵀy⟩ | 1.1e-18 … 1.8e-17 | round-off |
| G2 assembled vs shell Eᵀ | 2.0e-16 … 2.3e-16 | round-off |
| G3 FD reduced gradient (exact LU / production shell) | 5.8e-14 … 1.7e-12 | ≤ 1e-10 |
| G4 reduced Hessian symmetry (LU / shell) | 1.4e-16 … 8.8e-14 | round-off |
| G5 true KKT residual (production solve) | 3.5e-13 … 1.6e-12 | solver tolerance |
| G6 independent forward march | 7.7e-16 … 4.3e-15 | round-off |
| G7 reduced gradient at the computed optimum | 6.5e-12 … 3.1e-11 | solver tolerance |
| G7 Neumann stationarity on Γ_c, exact-OC runs (24) | 3.8e-17 … 4.7e-17; max\|u\| off Γ_c = 0 | round-off |
| **G8 ex20-style adjoint (L untransposed): transpose / Hessian / FD** | 3.9e-4 … 1.8e-3 / 0.13 … 0.28 / **3.4 … 160** | must fail |
| **G8b boundary-wrong adjoint νK − C: transpose / Hessian / FD** | 8.0e-7 … 2.0e-4 / 6.0e-3 … 0.27 / **1.9 … 19** | must fail |

The boundary-wrong adjoint is detected for every wind, including wind C (tangential on
Γ_c), because the x-faces carry w·n = ±1. With a constant nonzero wind on a rectangle, w·n
cannot vanish on every face.

## 8. Spatial convergence (CN, Nt = 512, LU sweeps; `results/space.csv`)

Errors are L2(0,T; L2(Ω)) for y and p, and **L2(0,T; L2(Γ_c))** for u.

| wind | nx | y | order | p | order | u (Γ_c) | order | J err (rel.) |
|---|---|---|---|---|---|---|---|---|
| A | 4 | 3.99e-2 | | 4.05e-3 | | 3.93e-2 | | 1.1e-2 |
| A | 8 | 4.62e-3 | 3.11 | 2.37e-4 | 4.09 | 3.59e-3 | 3.45 | 1.2e-3 |
| A | 16 | 5.67e-4 | 3.03 | 1.47e-5 | 4.01 | 4.02e-4 | 3.16 | 7.7e-5 |
| A | 32 | 7.05e-5 | **3.01** | 9.78e-7 | **3.91** | 4.89e-5 | **3.04** | 5.8e-6 |
| B | 4 | 4.40e-2 | | 1.32e-2 | | 5.43e-2 | | 4.7e-2 |
| B | 8 | 4.75e-3 | 3.21 | 7.75e-4 | 4.09 | 4.22e-3 | 3.69 | 3.6e-3 |
| B | 16 | 5.71e-4 | 3.06 | 4.81e-5 | 4.01 | 4.25e-4 | 3.31 | 2.3e-4 |
| B | 32 | 7.06e-5 | **3.01** | 3.27e-6 | **3.88** | 4.97e-5 | **3.10** | 1.4e-5 |

- **y:** O(h³), as expected for Q2.
- **u (boundary L2):** O(h³), approached from above.
- **p:** empirically ≈ O(h⁴) in the volume L2 norm on this smooth case. That is a
  superconvergence observation, not claimed in general; its trace on Γ_c, which is u, is
  O(h³).
- The objective converges faster than the fields.

## 9. Temporal convergence (nx = 32; `results/time.csv`), T2 methodology

Columns:

- *L2L2*: the full interval.
- *half*: the window t ≤ T/2 (T2's interior metric).
- *last*: the terminal slot alone.

The adjoint and control are compared at t_{n−1/2} under CN and at t_n under BE.

**BE, Nt 8 → 64:**

| field | wind A orders | wind B orders | verdict |
|---|---|---|---|
| y | 0.96, 0.97, 0.99 | 0.97, 0.99, 0.99 | order 1 ✓ |
| p | 0.95, 0.97, 0.98 | 1.02, 1.01, 1.00 | order 1 ✓ |
| u | 0.79, 0.83, 0.89 | 0.94, 0.94, 0.96 | order 1, approached from below (half window A: 0.87, 0.93, 0.97) |

**CN, Nt 4 → 32:**

| field | metric | wind A orders | wind B orders | verdict |
|---|---|---|---|---|
| y | L2L2 | 1.93, 1.99, 1.89 | 2.02, 2.06, 1.93 | order 2 (at Nt = 32 the nx = 32 spatial floor, 7e-5, is reached) |
| p | L2L2 | 2.09, 2.02, 1.98 | 2.02, 2.00, 1.96 | order 2 ✓ |
| u | t ≤ T/2 | **1.99, 1.97, 1.91** | **2.08, 1.98, 1.80** | order 2 ✓ (floor approached) |
| u | full L2L2 | 1.50, 1.56, 1.75 | 1.59, 1.71, 1.86 | rising towards 2 |

**CN terminal layer.** The control's full-interval rate is held below 2 at coarse Nt by the
last few slots.

- Per-slot u error at nx = 32: t ≈ 0.5 has 6.9e-4 → 1.8e-4 → 6.5e-5 (Nt 16/32/64), while
  the last slot has 9.4e-3 → 3.7e-3 → 1.2e-3 (order 1.35 → 1.62).
- This is the **documented DTO Crank–Nicolson terminal half-step layer**: ex20 stage 1/2
  measured ≈1.56–1.63 max-in-time for adjoint and control, and T2 §6.
- It is amplified in u because u = −(η/α_u)p|_{Γ_c} (1/α_u = 100).
- Two controls (`results/cnlayer.csv`):
  - **finer mesh** nx = 48, wind A, Nt 32/64/128: u full L2L2 order **1.91, 1.92**, last
    slot 1.62 → 1.80. The full-interval order is 2 once the spatial floor (~1.5e-5) is
    removed. y at Nt = 128 reaches its nx = 48 floor.
  - **w = 0 (heat)**, same manufactured structure: the same layer (u last slot
    1.44 → 1.67; interior 1.22 → 1.40). It is a property of the CN DTO scheme with
    boundary-trace control, **not of convection or of the Neumann implementation**.
- **CN control-interpretation diagnostic** (step 13):
  - p compared at interval **midpoints** has order 2.09 / 2.02 / 1.98;
  - the same p compared at **node times** has order 1.33 / 1.13 / 1.05;
  - the wrong-sign comparison gives relative error 2.00 at every Nt.

  So the interval-constant control is interpreted unambiguously, and the sign convention is
  confirmed.

## 10. Literature structural comparison (no numerical reproduction claimed)

**Seymen 2013 thesis (§2.3, eqs. (2.62)–(2.66); full text read in the feature audit):**

- Robin boundary control ε∂_n y + γy = u on Σ, with L2(Σ) regularisation.
- OTD adjoint ε∂_n p + (γ + β·n) p = 0, gradient αu − p = 0.

At **γ = 0** the state condition becomes the pure Neumann flux ε∂_n y = u, and the adjoint
condition becomes **ε∂_n p + (β·n) p = 0**. That is exactly ex21's ν∂_n p + (w·n)p = 0,
which ex21 obtains from the DTO exact transpose and G1N verifies on the matrices.

The gradient sign differs only by ex21's convention p_ex21 = −λ, and ex21 carries η
explicitly.

The bounded literature audit (`EX21_FEATURE_PARITY_AUDIT.md`) found **no close transient
all-at-once pure-Neumann convection–diffusion numerical benchmark**:

- Seymen's numerics are Robin (1D OTD space-time one-shot; 2D γ = 10³, a penalty);
- the AAO Neumann-control preconditioning literature is for the heat equation.

T7N's validation is therefore internal (manufactured exact OC + algebraic gates), with
Seymen as structural support only.

## 11. Solver behaviour (`results/solver.csv`, `logs/solver_attribution.log`, `t7n_solver.csv`)

Manufactured case, wind A, CN, Pe_cell = 0.14, production tolerance:

| solver | nx 16 (Nt 64) | nx 32 (Nt 128) |
|---|---|---|
| SCHUR_SWEEP + LU (reference) | 2 outer, Schur 60 its (max 39), spatial 1, true res 5.2e-12, 0 failures, 3.4 s | 2 outer, Schur 65 (44), true res 1.0e-11, 31.8 s |
| SCHUR_SWEEP + default spatial GMRES+GAMG | 2 outer, spatial ≤ 11, same J to 13 digits, 17 s | 2 outer, spatial ≤ 13, same J, 175 s |
| SCHUR_MATCHING | reason −11 (inner GMRES+GAMG hits 100), **rejected** | same, rejected |
| ADDITIVE_GAMG (Neumann default) | 200 outer, reason −3, **rejected** | timeout (3600 s) |

**Attribution:** these failures are **not caused by convection**.

- They are identical with w = 0, and for α_u = 1 as well as 1e-2 (manufactured case,
  nx = 8).
- On the plain production path with Neumann control and **convection off (the pure ex20
  code)**, ADDITIVE and MATCHING converge under BE (14 / 2 outer) and fail under CN
  (200 / −11), exactly as with convection on (v = 0 or v = (1, 0.5)).
- SCHUR_SWEEP converges in 2 outer iterations in every case.
- ex20's stage-2 report already documents this: "ADDITIVE_GAMG is unusable for Neumann",
  "SCHUR_MATCHING is not robust for CN", with SWEEP and the boundary-reduced Schur as the
  working Neumann solvers.

**Under convection the boundary-reduced solver is refused, so SCHUR_SWEEP is the Neumann
convection solver.** The default PC is not changed silently; ADDITIVE/MATCHING failures
are caught by the T6 acceptance guard (`EX21_SOLVE_REJECTED`, exit 8). As in ex20, SWEEP's
outer count is expected to grow at very small α_u; that is not explored here (no tuning
campaign).

## 12. MPI (`scripts/t7n_mpi.py`, `results/mpi.csv`)

Production path (`--exact-oc neumann-conv --solve-only --dump-json`), wind A, CN, nx = 16,
Nt = 32, MUMPS sweeps:

| np | outer | true res. | max\|Δy\| | max\|Δp\| | max\|Δu\| | J_track / J_reg rel. Δ | outer history |
|---|---|---|---|---|---|---|---|
| 1 | 2 | 3.13e-12 | ref | ref | ref | ref | 1.0, 9.005e-7, 4.92e-13 |
| 2 | 2 | 3.13e-12 | 4.3e-14 | 2.1e-15 | 1.8e-14 | 5.5e-14 / 7.0e-15 | identical (last entry at round-off) |
| 4 | 2 | 3.13e-12 | 3.9e-14 | 8.3e-16 | 1.6e-14 | 5.7e-14 / 6.2e-15 | identical |

**PASS.**

## 13. Natural state boundaries (step 12)

| face | state (continuous / discrete) | adjoint (continuous / discrete) |
|---|---|---|
| Γ_c = {z = 0} (face 1) | ν∂_n y = η u / natural row + η M_Γ u in the state rows | ν∂_n p + (w·n)p = 0 / from Lᵀ (G1N) |
| z = 1, x = 0, x = 1 (faces 3, 4, 2) | ν∂_n y = 0 / natural rows, nothing added | ν∂_n p + (w·n)p = 0 / from Lᵀ (G1N) |

No homogeneous-Dirichlet elimination occurs: state_bc is NATURAL, and Validate rejects
Neumann + Dirichlet state. The control support is exactly the 2·nx + 1 nodes of z = 0.

## 14. Definition of "Neumann convection ready" (step 17)

| # | item | result |
|---|---|---|
| 1 | analytical boundary optimality system derived | **PASS** (§0–1, symbolic §5) |
| 2 | generic path has no hidden symmetry assumption | **PASS** (§2; only the boundary-reduced solver, refused) |
| 3 | exact manufactured OC exists | **PASS** (`neumann-conv`, 12/12 symbolic checks) |
| 4 | state / adjoint / control spatial convergence | **PASS** (y 3.01, p 3.9, u 3.04–3.10 in L2(Γ_c)) |
| 5 | BE first order | **PASS** (y, p 0.97–1.02; u 0.89–0.96 and rising) |
| 6 | CN second order | **PASS** (y, p ≈ 2; u: t ≤ T/2 ≈ 2, full interval 1.92 at nx = 48; documented terminal layer) |
| 7 | FD gradient | **PASS** (≤ 1.7e-12) |
| 8 | transpose gate | **PASS** (≤ 1.8e-17; G1N 3.5e-14) |
| 9 | Hessian symmetry | **PASS** (≤ 8.8e-14) |
| 10 | true KKT residual | **PASS** (≤ 1.6e-12; exact-OC runs ≤ 1e-11) |
| 11 | independent forward solve | **PASS** (≤ 1.5e-14) |
| 12 | wrong-adjoint negative controls fail | **PASS** (ex20-style FD 3.4–160; boundary-wrong FD 1.9–19) |
| 13 | np = 1/2/4 consistency | **PASS** (≤ 4.3e-14) |

**Neumann boundary control under convection: VALIDATED** (SCHUR_SWEEP; LU/MUMPS or default
spatial GAMG in the sweeps; BE and CN; low/moderate Pe).

## 15. Regression (step 16)

**Script:** `scripts/t7n_regress.sh`, output `logs/regress.out`.

- Reference: the pre-T7N binary 3811bf71 (T6 final).
- Tested binary: eea795a5. The final binary 1d772913 differs from it only in the text of
  the boundary-reduced refusal message.
- On the final binary, the gates (identical logs), the restrictions
  (`logs/restrictions.log`) and the ex20 goldens (`logs/regress_ex20_goldens_final.log`)
  were rerun.

| check | result |
|---|---|
| ex20 goldens (stdout, stderr, JSON, exit codes) | **61/61 identical** (eea795a5 and final 1d772913) |
| Mathematics 2025 Ex. 3 (probe + primary), Ex. 4 | **byte-identical** |
| Mathematics 2025 Ex. 5 | **byte-identical** |
| ErrorFeedback 9.5 | numerics identical; the known pre-T3 label difference only |
| T0/T1/T2 constant wind (operator dumps, archived exact-OC runs, T2 gate logs) | **REGRESSION PASS** |
| T3/T4 representative (lp52 GAMG/LU/BE, variable-wind OC, block test, T2 verify-convection, failing GAMG case) | 7/9 identical. **lift-track and lift-const:** solution, J and errors identical to all printed digits; only last-bit round-off in post-processing diagnostics (`direct_rel` 1.4e-16 → 0, `march_rel` 4.674e-13 → 4.675e-13, `stationarity_rel` 3.1e-20 → 3.0e-20), from recompiling `ExactOcBenchmark.cpp` (no logic change on these paths) |
| Neumann heat, pre-T7N vs new: ADDITIVE, SWEEP, **boundary-reduced**; BE and CN | **identical** (6/6) |
| Dirichlet heat, pre-T7N vs new: plain and H1-space + H1-time; BE and CN | **identical** (4/4) |
| T6 LPS lp52 (ε = 1/500, nx = 16) | **identical** |
| distributed natural-BC `--verify-convection` | every pre-existing line identical; **additions only**: the new G1N line (3.5e-14) and the G8b lines (transpose 1.2e-4, Hessian 6.9e-3, FD 0.11 → fails, as required) |
| convection + Dirichlet control | **refused**: "convection is implemented for DISTRIBUTED and NEUMANN control only (Dirichlet boundary control: B_D^T is not convection-ready)" |
| convection + `--pc-schur-boundary-reduced` | **refused**: "… assumes a symmetric spatial operator); use --pc-schur-sweep …" |
| convection + Neumann + `--control-h1-time-weight` | **refused** (H1-time: Dirichlet only, unchanged) |
| convection + `--verify-boundary-reduced` | **refused** (ex20 batteries, unchanged) |

**Note on the T6 logs.** A first launch of the regression wrote into
`leveque_pearson/t6/logs/` before I stopped it:

- `regress_ex3_ex4.log` was restored verbatim from T6's own `regress_t6.out`;
- `regress_ex20_goldens.log` now holds this run's (equally 61/61) output rather than T6's.

The suite now runs from a local copy (`scripts/regress_suite.sh`) that writes only into
`boundary_convection/logs/`.

## 16. What remains before Dirichlet

- **Not in scope / left as is:**
  - high-Pe Neumann control (no solver work);
  - very small α_u with SWEEP;
  - a nonsymmetric boundary-reduced Schur;
  - LPS + Neumann (allowed, but not validated here beyond the feature-audit smoke test).
- **Next task (not started):**
  - convection + true Dirichlet boundary control, with a corrected B_Dᵀ (`ApplyStep` with
    Lᵀ) and a nonsymmetric reduced solver;
  - then its H1-space, H1-time and combined regularisations.

## 17. Files

**Source** (untracked `ex21/`):

- `include/Config.hpp`: Validate accepts convection + NEUMANN; Dirichlet still refused.
- `include/CommandLine.cpp`: boundary-reduced + convection refused; `--exact-oc-velocity`.
- `include/Options.hpp`: velocity option.
- `include/ExactOcBenchmark.{hpp,cpp}`: case `neumann-conv`, velocity override, boundary
  u error, boundary J_reg, `EX21_NEUMANN_STATIONARITY`.
- `include/ConvectionVerification.cpp`: G1N, G8b (natural state BC only).

Binary: production build `femusbin-control`, final sha256 1d772913….

**This folder:**

- `T7N_NEUMANN_CONVECTION.md`, `t7n_matrix_gates.csv`, `t7n_convergence.csv`, `t7n_solver.csv`.
- `scripts/`: `t7n_symbolic.py`, `t7n_gates.sh`, `t7n_run.py`, `t7n_mpi.py`, `t7n_analyze.py`,
  `t7n_regress.sh`, `regress_suite.sh`.
- `results/`, `logs/`.
