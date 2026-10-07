# T7D — true Dirichlet boundary control for transient convection–diffusion OC (ex21, 2D)

*(§1–1b were written BEFORE any source change; §2 onward after the implementation and runs.)*

## 1. Re-derived discrete Dirichlet-control algebra with L ≠ Lᵀ (before any source change)

**Notation:**

- I = free (interior + uncontrolled-boundary) state DOFs; C = controlled DOFs, the
  support of the trace mass M_Γ of Γ_c (face 1, z = 0).
- Q_C extends a C-vector by zero; P_I zeroes the C rows.
- Unhatted spatial matrices: M, and **L = νK + C_w** (C_w the strong-form Galerkin
  convection, row = test). A = M/dt + L (BE); A± = ±M/dt + L/2 (CN).
- Hatted E_I: the space-time operator built from M_I, L_I, with C rows/columns
  eliminated (identity rows on C). With the strong elimination, the physical state is
  **y = P_I y + Q_C (η u)**: y_C = η u.

**State (BE)**, free rows of A(y_I,n + ηQ_C u_n) − (M/dt)(y_I,n−1 + ηQ_C u_{n−1}) = M f_n:

    A_II y_I,n − (M/dt)_II y_I,n−1 + η [ A_IC u_n − (M/dt)_IC u_{n−1} ] = (M f_n)_I      (u_0 = y_0|_C/η data)

so E_I Y + η B_D U = F, with the lower block-bidiagonal control-to-state operator

    (B_D U)_n = P_I [ A Q_C u_n − (M/dt) Q_C u_{n−1} ],    n = 1..N  (the u_0 term is data).

**CN:** (B_D U)_n = P_I [ A₊ Q_C u_n + A₋ Q_C u_{n−1} ], with nodal controls u_n at t_n.

**Objective** (ex20 stage 3; tracking region strictly away from Γ_c, so only free DOFs
are tracked):

    J = (α_t/2) dt Σ_n ŵ_y,n (y_n − y_d,n)ᵀ M_ω (y_n − y_d,n)
      + (α_u/2) dt Σ_n ŵ_u,n u_nᵀ R_Γ u_n
      + (γ_u/(2dt)) Σ_n (u_n − u_{n−1})ᵀ M_Γ (u_n − u_{n−1})        [H1-time, γ_u ≥ 0]

- R_Γ = M_Γ + ℓ²K_Γ.
- ŵ = 1 (BE); ŵ = diag(1, …, 1, ½) (CN, nodal).

**Lagrangian and DTO optimality** (ex21 multiplier P = −λ/dt; derivatives divided by dt):

    adjoint     :  E_Iᵀ P = α_t (Ŵ_y ⊗ M_ω)(Y − Y_d)
    stationarity:  Q_u U − η B_Dᵀ P = (γ_u/dt) q_0,    Q_u = α_u (Ŵ_u ⊗ R_Γ) + (γ_u/dt) T ⊗ M_Γ
    state       :  E_I Y + η B_D U = F

**B_Dᵀ — the exact algebraic transpose.** B_D has diagonal blocks P_I A Q_C and
sub-diagonal blocks −P_I (M/dt) Q_C (BE), or P_I A₊ Q_C and P_I A₋ Q_C (CN). Hence

    BE:  (B_Dᵀ P)_n = Q_Cᵀ [ Aᵀ P_I p_n − (M/dt) P_I p_{n+1} ]     (p_{N+1} := 0)
    CN:  (B_Dᵀ P)_n = Q_Cᵀ [ A₊ᵀ P_I p_n + A₋ᵀ P_I p_{n+1} ]

so the block needed is (A_IC)ᵀ = (Aᵀ)_CI = (M/dt + **Lᵀ**)_CI (BE), or (A±ᵀ)_CI (CN).

**The current code** (`DirichletControlOperators::ApplyBDT`) reuses the forward
`ApplyStep`, i.e. `MatMult(stiffness_)`. It applies **A_CI = (M/dt + L)_CI** instead. The
error is

    (L − Lᵀ)_CI = (C_w − C_wᵀ)_CI,

which is zero without convection (ex20: exact) and nonzero with it. M is symmetric, so
only the stiffness products are affected. The forward B_D, `ApplyAffineStateCorrection`
(forward B_D) and the CN first-row data term (forward (M/dt − L/2) y_0) are correct.

The adjoint E_Iᵀ is applied by the generic AP shell with `transpose_spatial` (hatted L_Iᵀ,
and the explicit CN A±_I blocks transposed). That is already correct (T7N audit).

**Where the error enters:** `ReconstructControl` (u = (η/α_u) R⁻¹B_Dᵀp), the KKT (1,2)
block N_D = B_D R⁻¹ B_Dᵀ (`ApplyND`), `ComputeKktResiduals` (stationarity), and the
Dirichlet-reduced solver (Woodbury with B_Dᵀ).

## 1b. Continuous interpretation (supporting, DTO algebra authoritative)

State: y_t − νΔy + w·∇y = f, y = ηu on Γ_c, ν∂_n y = 0 on ∂Ω∖Γ_c.

Green's identity (div w = 0) for the linearised state δy (δy = ηδu on Γ_c, natural
elsewhere, δy(0) = 0) against the adjoint −p_t − νΔp − w·∇p = α_t(y − y_d)χ_ω, p(T) = 0:

    ∫∫ (α_t(y−y_d)χ_ω) δy = ∫_0^T ∫_∂Ω [ ν p ∂_n δy − ν δy ∂_n p − (w·n) p δy ].

Hence the adjoint boundary conditions are:

- **p = 0 on Γ_c** (the adjoint trace on the controlled face is constrained: the C rows of
  E_Iᵀ are identity rows);
- **ν∂_n p + (w·n) p = 0 on ∂Ω∖Γ_c** (as in T7N).

The remaining Γ_c term gives the gradient

    J′(u) = α_u (u − ℓ² ∂_ss u) − γ_u ∂_tt u − η ν ∂_n p |_{Γ_c}

with natural conditions ∂_s u = 0 at ∂Γ_c (ℓ > 0) and ∂_t u(T) = 0 (γ_u > 0); u(0) = u_0
is data. Because p = 0 on Γ_c, the convective term (w·n)p **drops out of the continuous
gradient**.

Discretely, however, B_Dᵀ p collects the C-row residual of the adjoint operator. Besides
the boundary flux ∫_Γ φ_c ν∂_n p, it contains volume terms from the strip of elements
next to Γ_c, including the convection ∫(w·∇φ_c) p. That is exactly where the L vs Lᵀ
difference lives: its relative size is O(Pe_h), so the wrong B_Dᵀ is only
asymptotically consistent, and it is wrong at every finite h.

## 2. Source changes (minimal; heat path bitwise unchanged)

All in `ex21/include/` (untracked directory, so no git diff exists. The regression of §13
is the evidence that nothing else moved.)

| file | change |
|---|---|
| `DirichletControl.{hpp,cpp}` | `ApplyStep(..., bool transpose)`: its stiffness product is `MatMultTranspose(stiffness_)` when `transpose && transpose_stiffness_` (`transpose_stiffness_ = HasConvection(config)`), else `MatMult` as before. `ApplyBDT` passes `transpose=true`. The forward B_D, the affine state correction and the CN first-row data term are untouched. New `ApplyBDTLegacy` (the old product, diagnostics only), `transposes_stiffness()`, and a `lu_only` overload of `CreateDirichletDirectSolver`, which skips the Cholesky candidates. |
| `DirichletReducedSchur.{hpp,cpp}` | Under convection: A = M/dt + L_I (or the CN A₊) is nonsymmetric. The SPD flags are no longer set; A is factored with LU (`dirichlet_reduced_a_`). Aᵀ (and the transposed CN coupling) is built explicitly and LU-factored (`dirichlet_reduced_at_`) for the backward sweep. The Woodbury core K = R_T + βZᵀWM_ωZ is still SPD (Z from forward responses only), so its dense Cholesky is unchanged. The heat branch is untouched (CHOLMOD Cholesky, as before). |
| `Config.hpp` | Convection + DIRICHLET control is no longer rejected. |
| `ConvectionVerification.cpp` | Dirichlet gates (§4); G1 for Dirichlet runs on the unconstrained L (a DISTRIBUTED copy), because the hatted elimination destroys the C·lin orientation test. |
| `ExactOcBenchmark.cpp` | `--exact-oc dirichlet-conv` (§3): an independent Dirichlet forward march, the `EX21_DIRICHLET_STATIONARITY` line, nodal-in-time u errors in L2(Γ_c). |

The parts that were already correct and were not touched (audited in §1): the E_Iᵀ AP
shell, the forward B_D, N_D = B_D R⁻¹ B_Dᵀ (correct once B_Dᵀ is), and
`ReconstructControl` (likewise).

**Solver restrictions (unchanged, explicit).** Dirichlet control runs with
SCHUR_DIRICHLET_REDUCED (the default) or ADDITIVE_GAMG only:

- ADDITIVE_GAMG has no CLI flag for Dirichlet. ex20 documents that it does not
  converge beyond the smallest Dirichlet problems.
- `--pc-schur-sweep` / `--pc-schur-matching` throw `invalid_argument` ("… need a control
  operator that is block diagonal in time"). N_D couples consecutive times.
- `--pc-schur-boundary-reduced` throws (it requires Neumann control).

Nothing is silently switched.

## 3. Manufactured exact optimal control (`--exact-oc dirichlet-conv`)

**Why it is non-trivial:** for Dirichlet control the tracking region must stay away from
Γ_c (only free DOFs are tracked), and the adjoint must vanish on Γ_c while its normal flux
carries the control. ex20 had no closed-form compatible optimum.

The construction (header comment in `ExactOcBenchmark.cpp`, verified by
`t7d/scripts/t7d_symbolic.py`):

- Ω = (0,1)², Γ_c = {z = 0}, natural state BC elsewhere, w = (0, v_y) (default v_y = 0.5),
  ν = 1/4, α_u = 10⁻², η = 1, tracking box z ∈ (1/4, 1).
- **Adjoint** p = S cos(πx) Q(z,t), with Q = (1 − φ)Q_s + φ B(t) Z_r:
  - Q_s = e^{−az−cτ}[Φ(z − z₁, τ) − Φ(z + z₁, τ)], where τ = T − t, Φ is the 1D heat
    kernel, z₁ = 1.2 lies **outside** Ω, and the odd image gives Q_s(0, t) = 0. Q_s
    solves the homogeneous backward adjoint exactly and is C∞ on the closed domain, with
    Q_s(·, T) = 0.
  - φ is a C∞ step (0 for z ≤ ½, 1 for z ≥ 0.9).
  - Z_r = e^{−(v_y/ν)z}(2 + cos πz) satisfies the Robin adjoint BC νZ_r′ + v_y Z_r = 0
    at z = 1; B = sin(T − t).
  - Hence the adjoint residual r vanishes for z < ½. The tracking (needed only where r ≠ 0)
    lives in z > ½ ⊂ ω.
- **Control** (stationarity α_u(u − ℓ²u_ss) = ην∂_n p on Γ_c): u = U(t) cos πx with
  U = −ηνS ∂_zQ_s(0,t) / (α_u(1 + π²ℓ²)). **ℓ > 0 is covered by the closed form.**
  γ_u > 0 is not: the stationarity becomes a 2-point ODE in t, and the driver refuses it.
- **State** y = A(t) cos πx sin(πz/2) + ηu (y = ηu on Γ_c, ν∂_n y = 0 elsewhere);
  f = y_t − νΔy + w·∇y; y_d = y − r/α_t.
- **Restrictions (refused explicitly):** v_x ≠ 0 (the x-faces would need
  ν∂_x p + v_x n_x p = 0, which cos πx does not satisfy), γ_u ≠ 0, and a velocity field.

**Symbolic verification** (`t7d/logs/symbolic_ell0.log`, `symbolic_ell0.3.log`, all PASS):

| check | result |
|---|---|
| S1 backward adjoint PDE for Q_s | sympy-simplified to exactly 0 |
| S2 Q_s(0, t) = 0 and Q_s(·, T⁻) = 0 | 0, and < 10⁻³⁰⁰ |
| S3 Robin BC of Z_r at z = 1 | exactly 0 |
| S6 / S7 y = ηu on Γ_c; ∂_z y = 0 at z = 1; ∂_x y = ∂_x p = 0 at x = 0, 1 | exactly 0 |
| S8 stationarity on Γ_c (ℓ = 0 and ℓ = 0.3) | exactly 0 |
| S5 the C++ hand-derived Q, Q_z, Q_zz, Q_t vs sympy (60 points incl. the blend zone) | ≤ 1.1e-15 relative |
| S6 C++ source f; S8 C++ U(t) | ≤ 1.1e-15 relative |
| S9 C++ ‖p(t_{n−½})‖_{L2(Ω)} (quadrature) vs mpmath | 5.9e-7 (nx = 16 quadrature) |

**Discrete consistency** (every exact-OC run prints these):

- `kkt_stationarity_rel` ≤ 1e-18;
- max|y_C − ηu| = 0 exactly (strong elimination);
- the independent Dirichlet forward march vs the production state ≈ 1e-15;
- true KKT residual ≤ 1e-14, with 2 outer FGMRES iterations.

## 4. Algebraic gates (`t7d/scripts/t7d_gates.sh`, `t7d/logs/gates/`, `t7d_matrix_gates.csv`)

Setup: `--verify-convection --dirichlet-solve`, nx = 8, N = 8, ν = 0.25, α_u = 10⁻²,
target sin_x_minus_t, BE and CN.

- **Winds:** heat, (1, 0.5), (0, 0.5), (−2, 1), (0, −4).
- **Regularisations:** L2, H1-space ℓ = 0.3, H1-time γ_u = 10⁻³, combined, and heat
  combined.

The gates:

- **G1:** C-orientation on the unconstrained L.
- **G2:** the E shell transpose.
- **G3D:** ⟨B_D u, p⟩ = ⟨u, B_Dᵀ p⟩, with the legacy B_Dᵀ as a negative control.
- **R:** symmetry of the regularisation operator.
- **G4D:** FD gradient of the reduced J(u) = objective(state(u)) (the state solved with
  E_I), compared with g = −ηB_Dᵀλ + α_u dt R u − γ q₀, under 4 adjoint variants: exact
  LU(Eᵀ), production shell, legacy B_Dᵀ, ex20-style untransposed E.
- **G5D:** reduced-Hessian symmetry ⟨Hu₁, u₂⟩ = ⟨u₁, Hu₂⟩, same 4 variants.
- **G6D:** the production solve:
  - true KKT residual;
  - |∇J(u*)| / |dt α_u R u*|;
  - **G7D** the forward march from u* vs the production y;
  - **G8D** stationarity rows of the KKT.
- **G9D** = the legacy rows of G3D, G4D and G5D (the wrong B_Dᵀ must fail).

Results across all 18 runs:

| gate | correct (all 18 runs) | legacy B_Dᵀ (12 convective runs) | ex20-style untransposed E (convective) |
|---|---|---|---|
| G1 C-orientation (convective) | 3.2e-14 – 6.0e-14 | | |
| G2 E-shell transpose | ≤ 9.0e-18 | | negative control 1.6e-4 – 5.9e-3 |
| G3D ⟨B_D u,p⟩ − ⟨u,B_Dᵀp⟩ | ≤ 1.4e-17 | 3.5e-4 – 3.3e-2 | — |
| R symmetry | ≤ 1.1e-14 | | |
| G4D FD gradient (shell = exact-LU) | ≤ 2.7e-12 | 2.2e-4 – 1.41 | 1.6e-2 – 5.1 |
| G5D Hessian symmetry | ≤ 1.2e-15 | 2.1e-7 – 1.3e-3 | 2.1e-6 – 3.4e-3 |
| G6D true KKT residual (1–2 outer its) | ≤ 3.6e-11 | | |
| G6D ‖∇J(u*)‖ / ‖dt α_u R u*‖ | ≤ 3.7e-12 | | |
| G7D forward march from u* vs production y | ≤ 1.0e-10 | | |
| G8D KKT stationarity rows | ≤ 7.3e-18 | | |

**Heat runs:** the legacy B_Dᵀ equals the correct one, and the result is identical to
round-off. It must be: L is symmetric, so the change is a no-op.

**G9D** (the wrong B_Dᵀ must fail) holds in every convective case, BE and CN, every wind
and every regularisation. The legacy FD error is ≥ 2.2e-4, against ≤ 2.7e-12 for the
correct adjoint. Its size depends on the wind and on the regularisation (largest:
L2 CN (1, 0.5) at 1.41, H1-time BE at 0.78; smallest: (0, −4) at 2.2e-4 – 6.0e-4). No
attempt is made to attribute that spread.

## 5. Spatial convergence, L2 Dirichlet control (`t7d_l2_convergence.csv`, `t7d/results/selfconv*.csv`)

### 5a. Against the closed form (CN, Nt = 256, `ladder=spaceCN`)

| nx | ‖e_y‖ L2(L2) | ord | ‖e_p‖ | ord | ‖e_u‖ L2(Γ_c) | ord | MaxRSS |
|---|---|---|---|---|---|---|---|
| 8 | 2.27e-2 | | 1.96e-3 | | 5.73e-2 | | 0.4 GB |
| 16 | 2.97e-3 | 2.93 | 2.65e-4 | 2.89 | 7.87e-3 | 2.86 | 1.3 GB |
| 24 | 7.37e-4 | 3.44 | 8.36e-5 | 2.84 | 1.99e-3 | 3.39 | 2.9 GB |
| 32 | 6.65e-5 | 8.36 | 1.09e-5 | 7.10 | 1.84e-4 | 8.27 | 5.1 GB |
| 48 | 2.50e-5 | 2.42 | 1.97e-6 | 4.21 | 7.13e-5 | 2.34 | 11.9 GB |

These orders are **not** a clean rate, for two reasons:

1. At nx = 48 the CN time error is no longer negligible. The temporal ladder at nx = 48
   (§6) gives e_y ≈ 8.3e-5 at Nt = 128, so ≈ 2.1e-5 at Nt = 256, out of a total of
   2.5e-5. A larger Nt would need (Nt·|C|)² dense memory, 48 GB at nx = 48, Nt = 512.
   That is the reduced solver's limit (§10).
2. Between nx = 24 and 32 the error falls by 11×. That jump is not an h³ rate. It is
   present in the semi-discrete data below as well, so it is spatial, and it is a
   pre-asymptotic **resolution** effect of the manufactured solution itself. The image
   heat kernel Q_s, in the blend zone z ∈ [½, 0.9] at late times (τ = T − t ≲ 0.05), has
   a small amplitude but large derivatives: ∂_z ~ ξ/(2ντ) ≈ 30–60 for ξ ≈ 0.3–0.7. Q2
   only resolves it once h ≲ 1/32. J_exact_track itself, the mesh quadrature of the
   data, still moves at the 1e-3 level up to nx = 24.

### 5b. Semi-discrete spatial self-convergence at fixed dt (no time floor)

Method:

- The same `dirichlet-conv` data, through the production path (`--solve-only`; J
  identical to the exact-OC driver, 1e-12).
- The error is measured against a fine-mesh reference at the **same** dt (so the time
  error cancels), in **true** L2 norms: the coarse Q2 field is prolonged exactly to the
  reference mesh and integrated with element Simpson weights.
- A coarse-node norm was tried first and rejected: it sees Q2 nodal superconvergence.

| ladder | nx | y | ord | u (Γ_c) | ord | p | ord |
|---|---|---|---|---|---|---|---|
| CN, Nt = 32, ref nx = 128 | 8 | 2.23e-2 | | 5.66e-2 | | 1.98e-3 | |
| | 16 | 2.90e-3 | 2.95 | 7.69e-3 | 2.88 | 2.70e-4 | 2.88 |
| | 32 | 6.36e-5 | 5.51 | 1.77e-4 | 5.44 | 1.14e-5 | 4.57 |
| | 64 | 2.54e-6 | 4.65 | 6.87e-6 | 4.69 | 7.38e-7 | 3.95 |
| BE, Nt = 32, ref 128 | 16→32→64 | | 2.92, 5.48, 4.66 | | 2.85, 5.42, 4.70 | | 2.89, 4.56, 3.93 |
| CN, Nt = 8, ref nx = 256 | 16 | 2.65e-3 | | 7.11e-3 | | 2.87e-4 | |
| | 32 | 5.53e-5 | 5.58 | 1.57e-4 | 5.50 | 1.21e-5 | 4.57 |
| | 64 | 2.27e-6 | 4.61 | 6.22e-6 | 4.65 | 7.52e-7 | 4.01 |
| | 128 | 3.85e-8 | 5.88 | 4.77e-8 | (7.03)ᵃ | 9.29e-8 | **3.02** |

ᵃ At nx = 128 the u error (relative 9e-8) sits at the FGMRES tolerance floor (true
residual ~1e-10). It is not evidence of anything.

**What is established:**

- **p converges with the Q2 L2 order 3**: 2.88 at coarse meshes, 3.02 at the finest pair.
- **y and u (L2(Γ_c))** converge at least at order 3. 2.9 is observed at coarse meshes;
  4.6–5.9 is observed in the resolution range nx = 16–128. The ladders do not reach an
  asymptotic regime for y and u. **No superconvergence is claimed**: the > 3 values are
  the pre-asymptotic resolution of the late-time kernel feature (5a), and the same
  pattern appears in BE and CN and at Nt = 8 and Nt = 32.
- The measured **control order** in L2(Γ_c): 2.86–2.88 in the 8 → 16 range; ≥ 3 beyond.

### 5c. Production data (target sin_x_minus_t, w = (1, 0.5)): a corner-compatibility layer

Self-convergence with the same method (Nt = 32, ref nx = 64):

| reg. | y ord (16, 32) | u ord | p ord |
|---|---|---|---|
| L2 BE / CN | 1.99, 1.77 / 1.60, 1.83 | 3.01, 2.86 / 3.03, 2.87 | 2.87, 2.79 / 2.86, 2.77 |
| H1-space ℓ = 0.3, BE | **2.84, 2.82** | 2.97, 2.86 | 2.87, 2.79 |

With ℓ = 0, **98 %** of ‖e_y‖² lies in the two 0.1 × 0.1 boxes at the corners where Γ_c
(Dirichlet, z = 0) meets the natural faces x = 0, 1. The boxes are 2 % of the area. With
ℓ = 0.3 the share is 13 %.

Reason: the natural x-faces require ∂_x y = 0 at the corners. An L2 control has no
reason to satisfy ∂_x u = 0 there, so the state has a corner singularity. H1-space
imposes ∂_s u = 0 at ∂Γ_c (natural BC of R_Γ), which restores compatibility.

The exact benchmark has u = U cos πx (∂_x u = 0 at x = 0, 1) and is compatible. This is
a **property of the data / the L2 Dirichlet problem**, not of the discretisation.

## 6. Temporal convergence (`t7d_l2_convergence.csv`: `timeCN`, `timeBE`; `results/selfconv.csv`)

**Closed form:**

| scheme (mesh) | Nt | e_y | ord | e_p | ord | e_u | ord |
|---|---|---|---|---|---|---|---|
| CN (nx = 48) | 8 | 1.72e-2 | | 1.33e-3 | | 4.86e-2 | |
| | 16 | 5.15e-3 | 1.74 | 2.84e-4 | 2.23 | 1.48e-2 | 1.72 |
| | 32 | 1.30e-3 | 1.98 | 7.13e-5 | 1.99 | 3.81e-3 | 1.96 |
| | 64 | 3.25e-4 | 2.00 | 1.68e-5 | 2.09 | 9.53e-4 | 2.00 |
| | 128 | 8.30e-5 | 1.97 | 4.29e-6 | 1.97 | 2.43e-4 | 1.97 |
| BE (nx = 32) | 8 → 256 | 6.5e-2 → 2.25e-3 | 0.98, 0.95, 0.97, 0.98, 0.99 | 4.4e-3 → 1.69e-4 | 0.90 – 0.97 | 1.52e-1 → 4.78e-3 | 1.07, 0.97, 0.98, 0.99, 0.99 |

**BE: order 1. CN: order 2** for y, p (at interval midpoints) and the **nodal** control
u_n at t_n.

**Endpoint audit** (per-slot errors, `EX21_EXACT_SLOT`):

- **u₀** is data: u₀ = y₀|_C/η = U(0) (U(0) = −0.424, compatible by construction). The
  strong elimination gives max|y_C − ηu| = 0 at every slot, including the first.
- **First step:**
  - CN e_u(t₁) = 2.2e-2, 4.0e-3, 9.8e-4, 3.2e-4, 1.6e-4 for Nt = 8 … 128. That is order ≈ 2
    until it meets the nx = 48 spatial level (≈ 1e-4).
  - BE e_u(t₁) decreases at ≈ 0.7 on average (4.7e-2 → 4.1e-3 over Nt = 8 → 256). This is a
    first-step layer of BE nodal controls; the global BE order is still 1.0.
- **Final step:**
  - u(T) = 0 exactly (Q_s(·, T) = 0); CN e_u(T) collapses (4.7e-3 → 1.2e-8).
  - CN p at the last midpoint t_{N−½}: the relative error is 5–9 %, of a p that is itself
    O(dt) there. This is the documented CN half-step terminal layer (as in T7N); the
    global p order is 2.
  - CN y(T) is order 2 (8.8e-5 → 2.1e-5).
- **CN weights:**
  - tracking and regularisation use ŵ = diag(1, …, 1, ½) on the nodal values n = 1..N;
  - the adjoint is compared at midpoints;
  - B_D carries A₊/A₋;
  - all of these are verified by G4D/G5D, which use the same objective.
- **J_reg endpoint weight (finding, reporting only):** for Dirichlet CN the reported
  J_reg = dt Σ_{n≥1} ŵ_n (α_u/2) u_nᵀR u_n omits the n = 0 trapezoid term
  (dt/2)(α_u/2) u₀ᵀR u₀. u₀ is data, so this is a **constant**: it does not enter the
  gradient, the KKT system or the optimum, and the G4D FD gradient agrees to 1e-14.
  It does make the *reported* CN J_reg a first-order quadrature whenever u₀ ≠ 0
  (here (dt/2)(α_u/2)‖u₀‖² = 2.8e-5 at Nt = 8). The convention is pre-existing (it is
  the ex20 heat convention too) and was **not changed** (out of scope, no effect on the
  solution). The J error in the exact runs is dominated by tracking-data quadrature on
  coarse meshes (5a), not by this term.

**Production data (target sin_x_minus_t; nx = 16, ref Nt = 512):**

- BE is order 1 for all four regularisations. The apparent 1.14–1.22 at Nt = 128 is the
  reference-gap bias of a 4×-finer reference: e(1 − ¼)/e(1 − ⅛) predicts 1.22.
- CN with γ_u > 0 (H1-time, combined): **order 2.0–2.1** for y, u, p.
- CN with γ_u = 0 (L2, H1-space) is **order-reduced**: y 0.73–0.89, u 1.2–1.6, p 1.0–1.4.

  Cause: an **initial incompatibility**. y₀ = 0 gives u₀ = 0, but the optimal L2 control
  is ≈ 0.455 already at t = dt. So the Dirichlet data jumps at t = 0, the state has an
  initial boundary layer (slot errors 0.10, 0.039, 0.019 …), and CN does not damp it.
  With γ_u > 0 the H1-time term makes u start continuously at u₀ (u(dt) = 0.0023), and
  order 2 returns.

  This is the classical CN order reduction for non-smooth data. It is a property of
  that data set, not a defect: the compatible exact benchmark is order 2.
## 7. H1-space (ℓ > 0) — `t7d_h1space.csv`

**Closed form** (ℓ = 0.3; U scaled by 1/(1 + π²ℓ²); symbolic S8 PASS for ℓ = 0.3):

- **Spatial**, CN, Nt = 128: y 1.91e-2 → 2.45e-3 → 5.98e-4 → 6.21e-5 (orders 2.97, 3.48,
  7.87), and the same for u and p. This is the L2 pattern of §5a, with the same
  resolution jump and the Nt = 128 time floor at nx = 32.
- **Temporal:**
  - CN at nx = 32: orders 1.98 / 1.98 for y and 1.94 / 1.99 for u up to Nt = 64, after
    which it reaches the nx = 32 spatial floor;
  - BE: 0.94–0.98.
- Stationarity ≤ 4e-16; y_C = ηu exactly; 2 outer iterations.

**Semi-discrete self-convergence** (exact data, ℓ = 0.3, CN, Nt = 32, ref nx = 128):

| | 8→16 | 16→32 | 32→64 |
|---|---|---|---|
| y | 2.98 | 5.55 | 4.63 |
| u | 2.92 | 5.49 | 4.67 |
| p | 2.88 | 4.64 | 3.97 |

These are identical in character to L2.

**Production data:** H1-space removes the corner layer, so the y order becomes 2.84 /
2.82 (BE), see §5c. Under CN with γ = 0, the initial layer of §6 remains.

**Gates (wind (1, 0.5), ℓ = 0.3):**

| gate | BE | CN |
|---|---|---|
| G3D | 1.35e-17 | 3.7e-18 |
| G4D FD | 8.9e-15 | 6.2e-15 |
| G4D legacy | 3.5e-3 | 8.5e-4 |
| G5D | 0 | 1.7e-16 |
| G6D stationarity | 7.3e-18 | 5.9e-18 |

**Status: H1-space + convection VALIDATED** (closed form, gates, self-convergence).

## 8. H1-time (γ_u > 0) — `t7d_h1time.csv`

There is no closed form: with γ_u > 0 the stationarity is a 2-point boundary-value
problem in t. The `dirichlet-conv` driver refuses γ_u ≠ 0 explicitly. Validation is
therefore by gates and fine-reference self-convergence. The g-device (an augmented
functional with a manufactured control-side linear term) was **not** implemented: it
would need a new source term in the stationarity row, i.e. a KKT change, which is out of
scope.

**Gates (γ_u = 10⁻³, wind (1, 0.5)):**

| gate | BE | CN |
|---|---|---|
| R symmetry | ≤ 1.1e-14 | |
| G3D | 1.35e-17 | 3.7e-18 |
| G4D FD | 2.7e-12 | 1.5e-13 |
| G4D legacy | **0.78** | 0.11 |
| G4D untransposed-E | 0.98 | 2.26 |
| G5D | 0 | 0 |
| G6D true residual | 2.0e-11 | 6.1e-12 |
| G7D march | 9.9e-11 | 2.3e-11 |
| G8D stationarity | 3.5e-18 | 2.9e-18 |

The objective inside G4D includes the H1-time term with U₀ = trace and the
(γ_u/dt) q₀ right-hand side, so the FD check also covers the u₀ coupling of the first
step.

**Self-convergence** (production data):

- **Time** (nx = 16, ref Nt = 512): BE 1.05, 1.06, 1.10, (1.22 ref-gap) for y and u;
  **CN 2.06, 2.00, 1.97, 1.99** for y and 2.10, 2.06, 2.05, 2.09 for u; p 1.98–2.22.
- **Space** (Nt = 32, ref nx = 64):
  - u: 3.00, 2.86;
  - p: 2.87, 2.78;
  - y: 1.85, 1.73. This is the ℓ = 0 corner layer of §5c; H1-time does not regularise
    in space.
- The ‖J_h − J_ref‖ orders are 2.00 (CN time) and 4.0 (space).

**Status: H1-time + convection VALIDATED** algebraically (exact gradient, Hessian
symmetry, KKT, forward march, stationarity) and in convergence by self-convergence
(design orders BE 1 / CN 2 in time). It is **not** validated against a closed-form
optimum, because none exists.

## 9. Combined H1-space + H1-time (ℓ = 0.3, γ_u = 10⁻³)

- **Gates:**

  | gate | BE | CN | heat-combined BE / CN |
  |---|---|---|---|
  | G3D | 1.35e-17 | 3.7e-18 | 2.9e-18 / 0 |
  | G4D FD | 4.2e-14 | 2.6e-14 | 3.3e-14 / 6.3e-16 |
  | legacy FD | 3.5e-3 | 8.5e-4 | = correct (heat) |
  | untransposed-E FD | 5.6e-2 | 1.6e-2 | = correct (heat) |
  | G5D | 8.5e-16 | 0 | |
  | G6D true residual | 1.8e-11 | 5.1e-12 | |
  | G8D stationarity | 6.2e-18 | 5.4e-18 | |

- **Self-convergence:**
  - Time, BE: 1.06, 1.07, 1.11 (1.23 ref-gap).
  - Time, **CN: 2.09, 2.04, 2.01, 2.03** (y); u 2.05–2.10.
  - Space, BE: y 2.92, 2.83 (corner layer removed by ℓ); u 2.90, 2.84; p 2.87, 2.79.
  - Space, CN: y 2.48, 2.14.
- **Production solve:** 2 outer iterations; MPI np = 1/2/4 agree (§11).

**Status: combined VALIDATED** (gates and self-convergence, BE and CN), with the same
no-closed-form caveat as §8.

## 10. Solvers (`t7d_solver.csv`, `t7d/logs/solver/`)

**Audit (Dirichlet + convection):**

| solver | status | evidence |
|---|---|---|
| SCHUR_DIRICHLET_REDUCED (default) | **valid and robust** after the T7D fix (LU of A and of Aᵀ, transposed CN coupling; the Woodbury core is still SPD/Cholesky) | 1–2 outer iterations for every wind ((1,0.5), (0,0.5), (−2,1), (0,−4), (4,2)), BE/CN, nx = 16/32, N = 32/64, all regularisations; true residual ≤ 5.4e-11 |
| ADDITIVE_GAMG | algebraically allowed by `Solver.cpp`, **but no CLI flag reaches it for Dirichlet**. ex20: it does not converge beyond the smallest Dirichlet problems | not exercised; no new flag added (no solver development) |
| SCHUR_SWEEP / SCHUR_MATCHING | **refused explicitly** (`invalid_argument`: N_D not block-diagonal in time) | `refuse_{sweep,matching}_{be,cn}` |
| SCHUR_BOUNDARY_REDUCED | **refused explicitly** (Neumann only) | `refuse_boundary-reduced_*` |
| factor backend overrides | `PETSC_OPTIONS=-dirichlet_reduced_a[t]_pc_factor_mat_solver_type mumps`: honoured, J identical to 4e-16. SuperLU: explicit PETSc error (not installed). The printed `forward_backend` label shows the default candidate, not the override (cosmetic) | `lu_mumps_*` |
| `--pc-time-blocks 4` (inexact block Woodbury prototype) | converges in 17–18 outer iterations, the **same as for heat with the T7N binary** (pre-existing behaviour) | `timeblocks4_*` |

**"Salvage" step 14:** nothing needed salvaging. The reduced solver's algebra is valid
under convection once B_Dᵀ is the true transpose, and its only SPD assumption (the
Woodbury core) still holds. It was fixed minimally (§2), not forced.

**Comparison and cost:**

- Iteration counts do not depend on h, N, wind or regularisation (1–2).
- The cost is set by the **dense Woodbury core**, of dimension Nt·|C| = Nt(2nx + 1):
  - 1056 (nx = 16, N = 32) → 4160 (nx = 32, N = 64);
  - the exact-OC runs reach 16 640 (nx = 32, N = 256: 5.1 GB) and 24 832 (nx = 48,
    N = 256: 11.9 GB);
  - MaxRSS ∝ (Nt·|C|)².
- This, **not robustness**, is the limit of the Dirichlet solver. It is the reason the
  §5a ladder could not use Nt = 512 at nx = 48.
- Wall times in the CSV were measured under concurrent load and are indicative only.

## 11. MPI (np = 1/2/4; `logs/solver/mpi_*`)

Production path, nx = 32, N = 32, BE and CN, three cases: convection (1, 0.5),
heat, and convection + combined H1.

| case | J agreement np = 1/2/4 (relative) | outer its | backends (np = 1 / >1) |
|---|---|---|---|
| conv BE | ≤ 1.4e-15 | 2/2/2 | petsc:lu / mumps:lu |
| conv CN | ≤ 1.7e-15 | 2/2/2 | petsc:lu / mumps:lu |
| heat BE / CN | ≤ 1.5e-15 | 2/2/2 | cholmod:cholesky / mumps:cholesky |
| conv + H1s + H1t BE / CN | ≤ 3.6e-15 | 2/2/2 | petsc:lu / mumps:lu |

Under convection, parallel runs select MUMPS **LU** (the `lu_only` candidates); heat
keeps MUMPS Cholesky.
## 12. Literature (structural only; no numerical reproduction claimed)

From the bounded audit (`literature_feature_map.csv`, `EX21_FEATURE_PARITY_AUDIT.md`):

| source | relation to T7D |
|---|---|
| Seymen 2013 thesis, Ex. 4.2.1 | "Dirichlet" control of convection–diffusion is a **Robin penalty** (γ = 10³ on ∂Ω), CN + SUPG, DTO. Not true Dirichlet control; no exact solution; only cost-functional convergence. Background only. |
| HDG Dirichlet boundary control of convection–diffusion I/II (arXiv 1801.01461, 1801.01056) | True Dirichlet control with L2 regularisation, **steady**. Its optimality system has the same structure as §1b: p = 0 on Γ_c, gradient α u = ∂_n p (up to sign/scaling conventions), and the convective boundary term vanishes from the gradient because p = 0 there. Structural support for §1b only. |
| FE Dirichlet boundary control of parabolic PDEs (arXiv 1410.0136) | Heat equation only: ex20's existing territory. |
| MPIMD15-03 (time-dependent Dirichlet data in FE) | Background on discretising time-dependent Dirichlet inputs (lifting / nodal data); consistent with the nodal-in-time u_n used here. |
| H1-in-time regularisation | None found for convection–diffusion; internal ex20 heat-stage design. |

No close transient, all-at-once, **true**-Dirichlet convection–diffusion benchmark was found.
T7D's validation is internal: the image-method exact OC, the symbolic checks and the
algebraic gates.
## 13. Regression (`t7d/scripts/t7d_regress.sh`, `t7d/logs/regress.out`; OLD = the T7N final binary, sha 1d772913; NEW sha cf170d75)

| block | result |
|---|---|
| R1 ex20 goldens | **61/61 identical** |
| R2 Mathematics 2025 Ex. 3 / Ex. 4 | byte-identical (3 runs) |
| R3 Ex. 5 | byte-identical |
| R3 EF 9.5 | differs only in the `EX21_EF95_CONFIG` label `variant=source`. R3 compares against **archived** logs that predate that label; both the T7N and the T7D binaries print it. This is the same pre-existing 4-line diff recorded in T7N, and it is **not** a T7D change |
| R4 T0–T2 constant wind | REGRESSION PASS |
| R5 T3/T3b/T4 representative | all identical, including the expected-failure cases (exit 8) |
| Dirichlet heat, BE and CN: L2, H1-space, H1-time, combined, plus nx = 16 sin_x | **all IDENTICAL** to T7N, every `EX20_/EX21_` line |
| Neumann heat (additive / sweep / boundary-reduced, BE and CN) | IDENTICAL |
| T7N `neumann-conv` exact OC and Neumann + convection production (BE and CN) | IDENTICAL |
| T6 LPS case; distributed natural-BC gate run; Neumann natural-BC gate run | IDENTICAL |

**Restrictions** (new binary):

- convection + Dirichlet is now **accepted** (the intended change);
- convection + Dirichlet + sweep / matching / boundary-reduced are refused;
- convection + boundary-reduced (Neumann), convection + H1-time Neumann and
  convection + verify-boundary-reduced are still refused;
- `dirichlet-conv` with v_x ≠ 0 or γ_u ≠ 0 is refused.

All refusals are explicit `invalid_argument` errors; nothing is switched silently.

## 14. Classification (2D, convection on, true Dirichlet boundary control)

| feature | algebra (G3D–G9D) | exact-OC convergence | self-convergence | status |
|---|---|---|---|---|
| **L2** | PASS (BE, CN, 4 winds) | BE 1 / CN 2 in time; space ≥ 3 (p → 3.0; y, u pre-asymptotic, > 3) | consistent | **VALIDATED** |
| **H1-space** (ℓ > 0) | PASS | closed form: same as L2 | consistent; removes the L2 corner layer | **VALIDATED** |
| **H1-time** (γ_u > 0) | PASS (FD includes the u₀ coupling) | no closed form (refused explicitly) | BE 1 / CN 2.0 in time; space u 3, p 2.8 | **VALIDATED (algebra + self-convergence); no exact-solution validation** |
| **Combined** | PASS | no closed form | BE 1 / CN 2.0; space 2.8–2.9 (BE) | **VALIDATED (same caveat)** |
| Solver | DIRICHLET_REDUCED: robust, 1–2 outer its, MPI-consistent | | | ready; memory ∝ (Nt·#C)² is the limit |

**Known properties (not defects):**

- L2 Dirichlet with corner-incompatible data has a corner layer: y order ~1.8 in space.
- CN with an initially incompatible control (γ_u = 0, y₀|_C ≠ u_opt(0⁺)) is
  order-reduced.
- The reported CN J_reg omits the constant u₀ endpoint term (§6).

## 15. Not done / out of scope

- **3D:** not started, as instructed.
- The g-device closed form for H1-time: not implemented (it would change the KKT).
- ADDITIVE_GAMG for Dirichlet: not reachable from the CLI. No flag was added.
- `dirichlet-conv` with v_x ≠ 0: needs a different x-profile. It is refused, not
  approximated.
- The reported CN J_reg endpoint convention (§6): documented, not changed.
- Memory of the Dirichlet-reduced solver: limits Nt·|C|. A factored or iterative
  Woodbury core would be new solver development, which is not done here.

## 16. Files

- **Report and CSVs** in `boundary_convection/`:
  - `T7D_DIRICHLET_CONVECTION.md`;
  - `t7d_matrix_gates.csv` (18 gate runs);
  - `t7d_l2_convergence.csv` (exact-OC ladders spaceCN, timeCN, timeBE);
  - `t7d_h1space.csv` (closed-form ℓ = 0.3 ladders);
  - `t7d_h1time.csv` (H1-time and combined self-convergence);
  - `t7d_solver.csv` (audit, comparison, MPI).
- **Scripts** in `t7d/scripts/`:
  - `t7d_symbolic.py`;
  - `t7d_gates.sh`, `t7d_gates_table.py`;
  - `t7d_ladder.sh`, `t7d_ladder_table.py`;
  - `t7d_selfconv.sh`, `t7d_selfconv.py` (the JSON fields, 1.2 GB, are kept **outside**
    the repository via `T7D_JSON`);
  - `t7d_solver.sh`, `t7d_solver_table.py`;
  - `t7d_regress.sh`, `regress_suite.sh` (the T7D copy, which writes to `t7d/logs`).
- **Logs** in `t7d/logs/`: ladder logs, `gates/`, `solver/`, `selfconv/`, the symbolic
  logs, `regress.out` and the R1–R4 logs.
- **Results** in `t7d/results/`: `selfconv.csv`, `selfconv_exact_fine_nt8.csv`.
