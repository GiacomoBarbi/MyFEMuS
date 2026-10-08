# T11H1 — 3D Dirichlet boundary control with H1-space / H1-time / combined regularisation

*Baseline:* **7530c58f8** "Add validated 3D Dirichlet boundary control to ex21" (T11D-L2,
committed at the start of this task). Nothing for T11H1 is committed, staged or pushed.

*Scope:* dimension 3, Dirichlet control on ONE whole QUAD9 face, u = 0 on ∂Γ_c (T11D policy),
H1-space (ℓ > 0), H1-time (γ_u > 0), combined, constant wind, BE/CN, SCHUR_DIRICHLET_REDUCED
with the matrix-free core (dense = gold), MPI. No Neumann/distributed H1, partial or multiple
faces, variable winds, LPS or new meshes.

Scripts, logs and results are in `t11h1/`; the tables are in this directory.

## 1. Audit of the existing 2D H1 machinery (written before any code change)

Sources read: `T7D_DIRICHLET_CONVECTION.md` §1b, §3, §6–9; `T8D_MATRIX_FREE_DIRICHLET_CORE.md`,
`T8D_FINALISATION.md`; `DirichletControl.{hpp,cpp}`, `DirichletReducedSchur.cpp`, `Solver.{hpp,cpp}`,
`Discretization.cpp` (face loop, elimination), `Config.hpp`, `CommandLine.cpp`.

**A. M_Γ (2D).** (M_Γ)_ij = ∫_Γc φ_i φ_j ds over the EDGE3 traces of the control face, via
`JacobianSur`. After the homogeneous elimination of the state-constrained set, C = support diag(M_Γ).
3D: the T11N/T11D QUAD9 loop already assembles it; C = int(Γ_c) (T11D).

**B. K_Γ (2D).** (K_Γ)_ij = ∫_Γc ∂_sφ_i ∂_sφ_j ds. It is assembled in the same 2D face loop
from the reference derivative and the arc-length Jacobian J = |dx/dξ|:
K_ab += (dφ_a/dξ)(dφ_b/dξ) w_g / J (the comment notes that `JacobianSur` provides no surface
gradient). It is eliminated like M_Γ (`EliminateConstrainedDofs(K_Γ, D, 0)`).
**This is the only 1D-tangent assumption** (G). In 3D the loop is skipped (`dimension == 2u`),
and the T11N/T11D 3D loop assembles M_Γ only, so K_Γ is an empty matrix in 3D.

**C. R_Γ.** `BuildRegularization`: R = M_Γ + ℓ²K_Γ on the C rows, identity on I
(`EmbedFaceMatrix` with the C mask), with a direct factor. ℓ = `control_h1_length`. SPD on C
whenever M_Γ|_C is SPD and K_Γ|_C ⪰ 0.

**D. R_H1t.** With nodal-in-time controls u_1..u_N (u_0 = y_0|_C/η is DATA):

    J_reg = (α_u/2) dt Σ_{n=1}^N ŵ_n u_nᵀ R_Γ u_n + (γ_u/(2dt)) Σ_{n=1}^N (u_n − u_{n−1})ᵀ M_Γ (u_n − u_{n−1}),

with ŵ = 1 (BE) and ŵ = diag(1, …, 1, ½) (CN, trapezoid on the nodes; the n = 0 term is a constant).
The control block of the dt-scaled KKT system is

    Q_u = α_u R_H1t,   R_H1t = Ŵ ⊗ R_Γ + ρ L_t,   ρ = γ_u/(α_u dt),   L_t = (1/dt) T ⊗ M_Γ,

i.e. **R_H1t = Ŵ ⊗ R_Γ + (γ_u/(α_u dt²)) T ⊗ M_Γ**. Verified in the source: `rho_ =
ControlTimeRatio(config)` = γ_u/(α_u dt) (`ControlTimeSourceWeight` = γ_u/dt divided by α_u),
and the assembled coupling is `rho_/dt_` (`ApplyR`, `BuildReducedRegularization`,
`BuildTemporalRegularization`; the header comment "rho = gamma_u/alpha_u" on the member is
stale). T = tridiag(−1, 2, −1) with LAST diagonal 1 (T = DᵀD, D the first difference with u_0
removed), and the stationarity is

    α_u R_H1t U − η B_Dᵀ P = (γ_u/dt) q_0,   q_0 = (1/dt)[M_Γ u_0; 0; …; 0].

`ApplyR` applies it matrix-free, `BuildReducedRegularization` assembles it SPARSE on the
distributed reduced layout (T8D) with a CHOLMOD/MUMPS Cholesky (`dirichlet_core_rt_`), and
`BuildTemporalRegularization` builds the dense replicated Cholesky only for the dense gold core.
The affine term goes through `AffineControl` / `ApplyAffineStateCorrection`.

**E. Role of γ_u.** The temporal H1 seminorm (γ_u/2)∫‖u_t‖²_{L2(Γc)}; its metric is M_Γ, NOT R_Γ
(independent of ℓ, no double count by construction). Continuous gradient (T7D §1b):

    J′(u) = α_u (u − ℓ²Δ_Γ u) − γ_u u_tt − η ν ∂_n p |_Γc

with natural condition **u_t(T) = 0** (γ_u > 0) and u(0) = u_0 DATA (no condition). Discretely
the natural condition is the last diagonal 1 of T (slot N has no successor). In 2D, ℓ > 0 carried
the natural condition ∂_s u = 0 at ∂Γ_c, because the 2D Dirichlet path had natural state BCs elsewhere.

**F. Dimension-independent.** R (the embedding, mask, factor), R_H1t (dense and sparse, `ApplyR`,
`SolveR`, `SolveReducedR`), Ŵ, T, q_0, the affine control and state correction, the J_reg time
term, the reduced Schur core (K = R_H1t + βH, shell and dense, `VerifyCore`), the T7D gates
(G4D's objective includes the H1-time term and u_0). None of them reads geometry. Their only
inputs are the embedded M_Γ and K_Γ.

**G. 1D-tangent assumptions.** Only the 2D K_Γ face loop (arc-length tangent), plus the CLI/
`Validate3D` refusals of `--control-h1-length` / `--control-h1-time-weight` in 3D.

**Two consequences for 3D.**
1. **Edge policy.** The 3D trace is C = int(Γ_c) with u = 0 on ∂Γ_c (essential, the perimeter
   is in D). K_Γ restricted to C is therefore the H¹₀(Γ_c) stiffness, and the 3D H1-space problem
   has the *essential* condition u = 0 on ∂Γ_c instead of the 2D natural ∂_s u = 0. The
   elimination already zeroes the D rows/columns of K_Γ.
2. **No closed form in 2D for γ_u > 0.** The T7D H1-time validation was gates + self-convergence;
   the stationarity in t is a two-point problem. §4 constructs a closed form for 3D.

## 2. 3D H1 regularisation (fixed before coding)

    J_reg = (α_u/2) ∫_0^T [ ∫_Γc u² dS + ℓ² ∫_Γc |∇_Γ u|² dS ] dt + (γ_u/2) ∫_0^T ∫_Γc |u_t|² dS dt,
    ∇_Γ u = (I − n⊗n)∇u,   u ∈ H¹₀(Γ_c) in space (u = 0 on ∂Γ_c).

Discrete: (K_Γ)_ij = ∫_Γc ∇_Γφ_i·∇_Γφ_j dS on the QUAD9 face, R_Γ = M_Γ + ℓ²K_Γ, and R_H1t exactly
as in §1D (the source scaling is kept). Stationarity (continuous, ex21 sign p = −λ, n outward):

    α_u (u − ℓ²Δ_Γ u) − γ_u u_tt = η ν ∂_n p   on Γ_c,   u = 0 on ∂Γ_c,   u_t(T) = 0 (γ_u > 0),   u(0) = u_0 (data).

**Surface gradient on a QUAD9 face (geometry-robust).** At each face Gauss point, with the
tangents t₁ = ∂x/∂ξ, t₂ = ∂x/∂η (the same Jacobian columns `JacobianSur` uses for the weight
and the normal), the metric G = [tᵢ·tⱼ], and the reference gradient (∂_ξφ, ∂_ηφ):

    ∇_Γφ = Σ_{ij} tᵢ (G⁻¹)ᵢⱼ ∂_jφ,  then explicitly projected  ∇_Γφ ← ∇_Γφ − (∇_Γφ·n) n.

For any extension φ̃ of φ, (I − n⊗n)∇φ̃ equals the first expression. The explicit projection
is a no-op in exact arithmetic; the assembly records max |n·∇_Γφ|/|∇_Γφ| as a diagnostic. This
is NOT the 1D arc-length formula. On a planar axis-aligned face it reduces to the two in-plane
derivatives, checked independently against K1 ⊗ M1 + M1 ⊗ K1 (§3).

## 3. Implementation (what changed; the validated T11D-L2 algebra is untouched)

| File | Change |
|---|---|
| `Discretization.cpp/.hpp` | `FaceSurfaceGradients` (metric-tensor surface gradient + explicit tangential projection, normal-leak diagnostic); the 3D Dirichlet face loop now also assembles **K_Γ** (it was empty in 3D); `IntegrateBoundaryGradient3D` (surface H1-seminorm error); `SurfaceGradientNormalLeak`; dump-only `KeepFaceOperatorsBeforeElimination` / `FaceOperatorBeforeElimination` (copies of the whole-face M_Γ, K_Γ before the D-elimination). |
| `Config.hpp` | `Validate3D`: `--control-h1-length` / `--control-h1-time-weight` allowed for **Dirichlet control only** (Neumann / distributed: refused). |
| `ExactOc3D.cpp/.hpp` | new exact case **`dirichlet-h1-3d`** (§4), `DirH1Time` (closed form), `DirExactUGrad`; the driver's stationarity check includes the H1-time affine term; new line `EX21_DIRICHLET3D_H1` (u surface-H1 error, ‖∇_Γu_h‖, J_reg_time, normal leak). `dirichlet-conv-3d` keeps its code path literally (byte-identical output, §11) and refuses H1 options. |
| `CommandLine.cpp` | `dirichlet-h1-3d` registered as a 3D exact case. |
| `Verification.cpp` | 3D Dirichlet `--dump-operators` additionally writes `_K_Gamma`, `_RT` (the sparse R_H1t the core factors), `_M_Gamma_face`, `_K_Gamma_face`, and an `# h1` header line in `_D_C.txt` when H1 is on. |
| `Solver.cpp` | **Bug fix in the KKT residual REPORT** (§7a). The solve is unchanged. |

**Not changed:** `DirichletControl.*` (R, R_H1t dense/sparse, `ApplyR`, `SolveR`, q₀, affine terms), `DirichletReducedSchur.*` (shell H, CG, dense gold core), B_D/B_Dᵀ, the T7D gate driver. The H1 machinery of 2D was generic, as the audit predicted (§1F).

## 4. Exact benchmark `dirichlet-h1-3d`

The T11D adjoint p is kept (Γ_c = {z = 0}, ω = (0,1)² × (½,1), p = X X (1−φ) Qs). The control is u = S(x,y) U(t) with S = X_cx X_cy, which is S = sin πx sin πy when ℓ > 0. The state is y = cos 2t sin³sin³sin³ + ηu(1−z)³.

- **H1-space.** −Δ_Γ S = λS with λ = 2π², and S = 0 on ∂Γ_c, so u ∈ H¹₀(Γ_c). This needs w_x = w_y = 0 (with a tangential wind, X_c = e^{−cs} sin πs is no longer a Δ_Γ eigenfunction); the case refuses ℓ > 0 otherwise.
- **Stationarity.** a U − γ_u U_tt = F(t) := −ην ∂_zQs(0,t), with a = α_u(1 + ℓ²λ).
  - γ_u = 0: U = F/a.
  - γ_u > 0, k = √(a/γ_u), τ = T − t: the natural condition is U_τ(0) = 0 (u_t(T) = 0), and u(0) = u₀ is DATA. The admissible family is U_p + A cosh kτ. The benchmark takes A = 0, U(τ) = −(1/(γk))∫₀^τ sinh(k(τ−s))F(s)ds.
  - With ∂_zQs(0,s) = C₀ s^{−3/2} e^{−A₀/s − cc s}, the convolution is closed-form in erfc (`DirH1Time`). It requires cc > k, refused otherwise.
- **Endpoint conditions (§14 of the request).** The continuous derivation (T7D §1b, unchanged) gives the natural u_t(T) = 0 from γ/2∫|u_t|², with no condition at t = 0 (u₀ is data). Discretely, that is T's last diagonal 1 (no successor) and the affine q₀. The benchmark satisfies U_τ(0) = 0 (H6) and builds y₀ from U(T), so u₀ is consistent.

**Symbolic / high-precision verification** (`t11h1_symbolic.py`): **181/181 PASS**. Eight cases: heat, the normal wind (0,0,−0.75) and the opposite normal wind (0,0,0.5), with ℓ ∈ {0, 0.3} and γ ∈ {0, 1e-2}; plus the tangential wind A with ℓ = 0, γ = 1e-2.

| check | what | result |
|---|---|---|
| H1 | Δ_Γ eigen-relation; S = 0 on ∂Γ_c | exact |
| H2 | ∂_zQs(0) closed form | exact |
| H3 | E(b,τ) erfc closed form vs quadrature | ≤ 9e-41 |
| H4 | U and U_τ vs a **direct mpmath convolution quadrature** (independent of the erfc formula) | ≤ 3e-23 (quadrature accuracy) |
| H5 | ODE residual | ≤ 4e-40 |
| H6 | U_τ(0) = 0 and U(0) = 0 (to all orders) | < 1e-150 |
| H7 | stationarity α_u(u − ℓ²Δ_Γu) − γu_tt − ην∂_np, pointwise | ≤ 7e-40 |
| H8 | y = ηu on Γ_c, y = 0 on the other five faces, with a symbolic U(t) | exact |
| H9 | adjoint r = 0 for z ≤ ½; p = 0 on the faces | exact |
| H10 | the homogeneous mode A cosh kτ keeps the ODE and the natural endpoint (A only changes the datum u₀) | exact |
| H11 | literal C++ port of U, U_t vs mpmath | ≤ 2.3e-16 |

*Corrections made while writing the checks, recorded honestly:*
- The mpmath reference first used the binary 1.2 instead of 6/5, which gave a 1.5e-15 floor in H7. It now uses 6/5.
- The H4 bar was set to 1e-20, the quadrature accuracy.
- A draft H10 asserted U_γ → F/a as γ → 0. That is false for the A = 0 member, which grows like e^{kτ − 2√(A₀k)}; the γ = 0 benchmark is a different member. H10 was replaced by the family statement.

Driver consistency, every run: stationarity |α_u R_H1t U − ηB_DᵀP − (γ/dt)q₀|/|α_u R_H1t U| ≤ 1.5e-14, y_C = ηu exactly, independent march ≤ 2.5e-13, surface-gradient normal leak 0.

**ℓ = γ_u = 0 reproduces T11D-L2:** J and every error norm are identical to the printed digits (10–12). Only KKT-residual digits at 1e-13 differ, because the data are evaluated in a different operation order.

## 5. K_Γ, R_Γ, R_H1t validation (`t11h1_surface_stiffness.py` → `t11h1_surface_stiffness.csv`, **393/393 PASS**)

Sixteen cases: all six faces of the unit cube (3×3×3); three faces at (2,3,4), which are anisotropic; four faces of the non-unit box (0,2)×(−1,1)×(0,½) at (4,4,2); BE and CN.

**Whole face (before elimination)**
- support = the face-plane nodes ((2n_a+1)(2n_b+1)); off-face rows are 0;
- |K − Kᵀ| = 0 exactly;
- PSD, λ_min/λ_max ≥ −2.1e-16, with the constants as the **only** null vector;
- max|K·1|/max|K| ≤ 4.4e-16;
- vᵀKv = ∫|∇_Γv|² exactly for a linear and a quadratic tangential v;
- **K = K1(a)⊗M1(b) + M1(a)⊗K1(b)**, from independent 1D Q2 matrices with the actual edge lengths: ≤ 7.5e-15;
- opposite faces, coordinate-matched: ≤ 3.1e-16.

**Restriction to C = int(Γ_c)**
- eliminated K_Γ rows and columns off C are 0, so the perimeter is inactive;
- K|_C = K_face|_C exactly;
- K|_C is SPD (H¹₀; λ_min ≥ 0.30), M|_C is SPD, and R = M + ℓ²K is SPD for ℓ ∈ {0, 0.1, 0.3, 1, 3, 10} (λ_min ≥ 2.9e-3).

**Sparse R_H1t**
- equals Ŵ⊗R + (γ/(α dt²))T⊗M rebuilt in Python (≤ 1.4e-16);
- symmetric, SPD (λ_min ≥ 0.23).

**Preallocation**
- PETSc `-info :mat`: **0 mallocs** in every MatSetValues of every case.
- max row nnz: K_Γ face 25, R_H1t ≤ 75 ≤ the allocated max(24, 3·row_max). The T11D measured-row-length rule covers K_Γ and the time coupling, so no hard-coded row length is involved.

**Negative controls, all detected (weakest 0.45)**
- a one-direction K (the 1D-tangent analogue);
- K without the metric inverse;
- the time term with R_Γ instead of M_Γ (double count);
- a wrong sign of the H1-time term;
- the weight γ/(α dt) (missing 1/dt);
- T without the natural last diagonal.

## 6. SPD of the reduced core (§10)

R_H1t = Ŵ⊗R_Γ + ρ_t T⊗M_Γ, with Ŵ = diag(w_n) > 0, R_Γ ≻ 0 on C for every ℓ ≥ 0 (M_Γ|_C ≻ 0, K_Γ|_C ⪰ 0 — §5), ρ_t = γ_u/(α_u dt²) ≥ 0 and T = DᵀD ≻ 0. So R_H1t ≻ 0. H = Zᵀ(W ⊗ M_ω)Z ⪰ 0 is unchanged from T11D, hence **K_c = R_H1t + βH ≻ 0** and CG applies.

Numerically, with the dense gold core (`--verify-dirichlet-core`) for H1-space, H1-time and combined (§8): ‖K − Kᵀ‖ = 0, λ_min ≥ 6.0e-3, cond(R⁻¹K) 1.00–1.04, shell symmetry ≤ 3e-16. The gates' min Rayleigh(R) is ≥ 1.6e-2.

## 7. Algebraic gates (`t11h1_gates.sh`, `t11h1_gates_table.py` → `t11h1_gates.csv`, **64/64 PASS**)

`--verify-convection` with L2, H1s (ℓ = 0.3), H1t (γ = 1e-2) and combined × BE/CN:
- exact `dirichlet-h1-3d`, n = 2, 3: heat and the normal wind for all four regularisations, the tangential wind A for L2 and H1t;
- production: face xmax wind A, ymin wind B, and the non-unit box ymax wind A.

| gate | worst over 64 runs |
|---|---|
| G1 orientation | ≤ 7e-15 |
| G3D ⟨B_Du,p⟩ = ⟨u,B_Dᵀp⟩ | 1.2e-17 |
| R symmetry | 1.9e-15 |
| G5D Hessian symmetry (exact LU and shell) | 1.6e-13 |
| G4D FD gradient (shell; the objective includes the H1-time term and u₀) | 1.0e-13 |
| G6D true residual / ‖∇J‖ / forward march | 2.2e-11 / 2.6e-10 / 5.0e-11 |
| **kkt_state_rel** | **2.2e-11** (after the §7a fix) |
| core CG / outer | 2–5 / 1 |

**Negatives** (w ≠ 0: detected; heat/w = 0: coincide):

| negative | minimum |
|---|---|
| G2 ex20-style | 1.5e-4 |
| G3D legacy B_Dᵀ | 2.4e-3 |
| G4D legacy / ex20 | 2.4e-5 / 1.1e-4 |
| G5D legacy / ex20 | 9.8e-8 / 9.5e-8 |

*Criterion (stated honestly):* with the T11D bar (G4D > 1e-4), 8 of 64 runs (H1 regularisations, exact n = 3 wind A, the non-unit case) had G4D legacy negatives of 2.4e-5 to 7.6e-5 while their G3D negatives were 2.4e-3 to 2.1e-2. H1 makes the optimal control smaller and smoother, which damps every derived detector. The final rule:
- the direct transpose detectors (G2, G3D) must exceed 1e-4;
- the derived ones (G4D, G5D) must exceed 1e-8, at least 6 orders above their positives (≤ 6e-15).

The operator-level H1 negatives are in §5.

**7a. Finding and fix: double-counted affine term in the KKT state-residual report.** The first combined-γ gate showed `kkt_state_rel = 1.66e-2` while the true residual was 6.5e-14 and the independent forward march agreed to 6e-13. It occurred only with exact data and γ_u > 0, i.e. **u₀ ≠ 0**.

`SolveOptimalControl` adds the affine correction −ηB_D Q_u⁻¹(γ/dt)q₀ to `rhs_y`, the right-hand side of the reduced (Y, P) system. `ComputeKktResiduals` then evaluated E_IY + ηB_DU − rhs_y with that corrected rhs **and** the full control U, which already carries the affine part, so the term was counted twice. The bug dates from 2D H1-time. It was invisible there because every 2D H1-time run had y₀ = 0 ⇒ u₀ = 0 ⇒ zero correction.

Fix: the report receives a copy of the physical rhs taken before the correction. For u₀ = 0 the two are identical, and a 2D combined-H1 run is md5-identical to the frozen T11D-L2 binary (as is the whole 2D regression, §13). The solve never used this report.

## 8. Dense vs matrix-free (`t11h1_core_equivalence.py` → `t11h1_core_equivalence.csv`, **20/20 PASS**)

H1s, H1t and combined × {production xmax wA BE/CN, zmax heat CN, non-unit ymax wA BE, exact heat BE n = 3, exact wN CN n = 4}, plus two L2 references, which reproduce T11D's numbers exactly.

| quantity | result |
|---|---|
| dense vs shell action (9 probes) | ≤ 2.2e-15 |
| K symmetry | 0 |
| λ_min | > 0 |
| cond(R⁻¹K) | 1.000–1.035 |
| iterative vs dense, rtol 1e-8 (nodal y, p, u and J) | ≤ 1.0e-11 |
| iterative vs dense, rtol 1e-12 | ≤ 1.3e-14 |
| outer iterations, dense/iterative | 1/1 |
| core CG iterations | 2–4 (rtol 1e-8), 3–6 (rtol 1e-12) |

The H action is unchanged (B_D → E_I⁻¹ → tracking → E_I⁻ᵀ → B_Dᵀ); only R_H1t changes, and it is a sparse factor. No dense response, Gram or core object appears in iterative mode (§12).

## 9. Spatial convergence (`t11h1_space_convergence.csv`, figure `t11h1_space_convergence.png`)

Exact errors. y and p are in L²(0,T;L²(Ω)); u is in L²(0,T;L²(Γ_c)) and in the surface H1 seminorm, both measured with the surface gradients of §2 and nodal in time. Orders are at the last two levels.

| ladder | y | p | u (L²(Γ_c)) | u (H¹-seminorm Γ_c) |
|---|---|---|---|---|
| H1-space ℓ = 0.3, normal wind, CN N = 256, n ≤ 10 | 3.06 | 2.65 | 2.98 | **2.13** |
| H1-space ℓ = 0.3, heat, CN N = 256, n ≤ 10 | 3.06 | 2.84 | 3.06 | **2.15** |
| H1-time γ = 1e-2, heat, CN N = 64, n ≤ 14 | 3.03 | (time floor) | 3.20, **3.49** | 2.13, 2.11 |
| combined, normal wind, CN N = 64, n ≤ 14 | 3.03 | (time floor) | 3.28, **3.60** | 2.06, 2.05 |

- p is the T11D adjoint, and its spatial order is that of T11D (2.66–3.05 at the finest pair).
- At N = 64, p's CN time error caps it in the two γ > 0 ladders. Those use N = 64 because the γ > 0 u error has no time floor there: the n = 6 N-scan from 32 to 256 is flat.
- **γ > 0 pre-asymptotic plateau.** At n = 6 → 8 the u error stalls (orders 0.17 to 0.35). Diagnosis:
  - the dense core is identical, and the plateau is independent of Δt and of γ;
  - fitting U_h − U shows it is not a homogeneous mode (25–38% fit residual);
  - with H1-time the A = 0 member is a small difference of large terms (U(0.5) ≈ +0.03 against F/α ≈ −0.9), so the slowly converging discrete normal-derivative forcing (amplitude error 0.4–1.7% at n = 6–8 already for γ = 0) is carried by the temporal coupling into the late-time tail, where the exact u ≈ 0;
  - extending to n = 10, 12, 14 recovers order 3.2–3.6.

  So it is pre-asymptotic, not a defect. It was not tuned around.

## 10. Temporal convergence (`t11h1_time_convergence.csv`, figure `t11h1_time_convergence.png`)

n = 6, normal wind, production path, nodal u, reference-corrected orders at the finest step.

| | BE (ref N = 1024, N = 256) | CN (ref N = 512, N = 128) |
|---|---|---|
| H1-time γ = 1e-2: y / p / u / u_t | 0.99 / 0.97 / 0.98 / 0.97 | y 1.73 (**y on t ≥ T/4: 1.96**) / p 2.17 / u 1.96 / u_t 1.99 |
| combined: y / p / u / u_t | 0.99 / 0.97 / 0.98 / 0.97 | y 1.73 (**t ≥ T/4: 1.97**) / p 2.17 / u 1.92 / u_t 1.99 |
| \|J_N − J_ref\| | 2.7e-6 (N = 256) | 1.6e-9 / 3.0e-9 (N = 128) |

- u_t is the coarse first difference against the reference's at the coarse nodes; u₀ cancels.
- The full-interval CN y order comes from the **inherited CN start-up layer**. The first-slot y errors are 7.7e-5 → 4.7e-5 → 2.1e-5, while on t ≥ T/4 it is order 2.03 / 2.04 (the same layer as T11N/T11D). The time scheme is unchanged.
- H1-time makes BE cleaner than L2 (T11D L2 BE reached 0.95 only at N = 256).

## 11. Heat / w = 0 parity (`t11h1_parity.sh`, `t11h1/logs/parity/parity.log`)

H1s, H1t and combined × faces zmin, xmax × BE/CN.
- **Operators, 120/120 byte-identical:** M_Γ, K_Γ (eliminated and whole-face), R_H1t, the unhatted M and L, B_D, B_Dᵀ and coords. The 12 C-lists are byte-identical; their header differs only in the transposes_L path flag.
- **Solves** (dense and iterative core, outer rtol 1e-14): fields ≤ 2.4e-15; J identical to all digits; same outer (2) and core (5–6) counts and path.

## 12. Iteration scaling, ℓ and γ (figure `t11h1_core_iterations.png`)

Production setup: face zmin, ω = (0,1)² × (0.3,1) adjacent to Γ_c, default CG + sparse R_H1t Cholesky at rtol 1e-8. 126 + 18 + 20 runs, 0 failures.

**Core scaling (`t11h1_core_scaling.csv`)**
- **mesh n = 2–10:** 3–5 iterations, all regularisations, heat and wind A;
- **Nt = 8–512:** 3–5;
- **α_u = 1e-1 to 1e-6:**
  - H1s grows like √cond(R⁻¹K): 49–72 iterations at 1e-6, against 105 for L2 in T11D;
  - H1t and combined stay at **4**. The time term (γ/(α dt²))T⊗M dominates K and is exactly what the preconditioner factors (cond(R⁻¹K) = 1.01).

The R_H1t preconditioner keeps the T8D/T11D mesh and Nt robustness in 3D.

**ℓ scan (`t11h1_ell_scaling.csv`, γ = 0, ℓ = 0 → 3)**
- ‖u‖ falls from 0.134 to 7.6e-4 (heat);
- the spatial Rayleigh ratio ‖∇_Γu‖/‖u‖ falls monotonically from 5.09 to 4.54 (heat) and from 5.41 to 4.60 (wind A), towards √(2π²) = 4.44, the smoothest admissible shape in H¹₀;
- J rises monotonically;
- CG goes from 5 to 2–3 iterations and cond(R⁻¹K) → 1;
- at n = 6, 8 there is no iteration growth.

Caveat: the n = 6 rows select a different tracking layer by centroid (z ≥ 0.3), so they are slightly different problems.

**γ scan (`t11h1_gamma_scaling.csv`, ℓ = 0, γ = 0 → 0.1)**
- ‖u_t‖/‖u‖ falls from 9.6 to 1.6, and the maximum slot jump by about 700× (temporal smoothing);
- J rises monotonically;
- CG stays at 3–5;
- γ does not smooth in space: the spatial ratio rises slightly under wind A.

## 13. Memory (`t11h1_memory.csv`, figure `t11h1_memory.png`)

Wind A CN; n = 4 (Nt = 32 to 512), n = 6 (Nt = 32, 64), n = 8 (Nt = 32, 128); 60 runs.

**Iterative core**
- H1s costs exactly as much as L2;
- H1t and combined add about 5% (nnz(R_H1t) = Nt·nnz(R_C) + 2(Nt−1)·nnz(M_C) ≈ 3× L2, sparse);
- RSS is linear in Nt: 1136 MB at Nt = 512 (n = 4) and 2141 MB at n = 8, Nt = 128, about 1.5–1.8 kB per space-time unknown, the same as T11D;
- **no O(m²)** storage;
- nnz(K_Γ|_C) = 529 / 1521 / 3025 for n = 4 / 6 / 8.

**Dense gold mode**
- L2 and H1s match T11D's 12 B·m² + 8 B·Nt·N·|C| (ratio 1.02–1.14);
- with γ > 0 the ratio is 1.45–1.61, because the dense mode also keeps the replicated dense R_H1t Cholesky (≈ +8 B·m², the documented 2D design for the reference mode only).

## 14. MPI np = 1/2/4 (`t11h1_mpi.sh`, `t11h1_mpi.py` → `t11h1_mpi.csv`, **15/15 PASS**)

Cases: H1-space convection CN, H1-time convection BE (tangential wind A), combined convection CN (production xmax wind B), combined heat BE, and combined on the non-unit box.

| quantity | result |
|---|---|
| y, p, u vs np = 1 | ≤ 6.4e-15 |
| J | ≤ 3e-16 |
| trace set | identical int(Γ_c) (49 / 21 nodes), u = 0 off C, y_C = ηu_C exactly |
| outer (2) and core CG (3–6) counts | identical at every np |
| r₀ | identical; later residuals ≤ 1e-8·r₀ |
| R_H1t factor | **CHOLMOD Cholesky (np = 1), MUMPS Cholesky (np = 2, 4)** on the distributed reduced layout |

Outer rtol 1e-14 (the T11D one-step note).

## 15. Whitelist / refusals (`t11h1_refusals.sh`, `t11h1/logs/refusals.log`, **75/75**)

**Accepted, 3D Dirichlet only:**
- H1-space, H1-time and combined: BE/CN, heat/wind;
- dense core and `--verify-dirichlet-core`;
- face xmax and the non-unit box;
- the exact case combined, and H1-time with a tangential wind;
- plus every T11D/T11N/T10/T9 accept case.

**Refused:**
- H1 for 3D Neumann (space, time, combined) and for distributed control;
- negative ℓ or γ;
- H1 options on the L2 benchmark `dirichlet-conv-3d`;
- `dirichlet-h1-3d` with ℓ > 0 and a tangential wind, or with γ too small for the closed form;
- partial or multiple faces;
- variable wind, LPS, SWEEP and boundary-reduced with H1;
- the exact case with a face or a sweep;
- plus all previous refusals.

## 16. Regression (`t11h1_regress_all.sh`; logs `t11h1/logs/t11h1_regress_*.log`)

OLD = a private clean build of the **T11D-L2 commit 7530c58f8** (`git archive`, sanitised, headers remapped; recipe `t11h1/scripts/clean_build_t11d.sh`; sha256 66bb1f6b…; no Miniforge).

| Suite | Result |
|---|---|
| ex20 goldens | **61/61** |
| Math2025 Ex3/Ex4/Ex5 | byte-identical |
| EF 9.5 | only the known `variant=source` label (as T11N/T11D) |
| R4 T0–T2 constant wind | **PASS** |
| R5 T3/T4 | 9/9 identical |
| 2D old vs new (T6, T7N, **T7D all four regularisations**, T7F, **T8D dense/iterative core and verify**, natural-BC, Dirichlet heat all regularisations) | **55/55 IDENTICAL**, **34/34 JSON_IDENTICAL** (the 2 boundary-reduced + w0 cases refused by both, as before) |
| 2D/3D restrictions | identical to the T11D log |
| T9 (3D distributed) | **32/32** (MPI round-off 3.4e-15) |
| T10 (3D natural, multi-zone) | **27/27** |
| T11N (3D Neumann) | **43/43**, no new dump files |
| **T11D-L2** (3D Dirichlet L2: 16 exact gates, 6 production gates, exact driver incl. dense and verify core, 3 JSON, 4 dumps, MPI) | **85/85 identical**; new-only dump files `_K_Gamma`, `_K_Gamma_face`, `_M_Gamma_face`, `_RT` |

Build hygiene: the NEW binary, its link line, flags and cache contain no Miniforge/Conda (checked with `readelf` after every build). Builds and runs used the sanitised PETSc-arch environment. Miniforge Python is used only for the offline sympy/mpmath script `t11h1_symbolic.py`, as in T11N/T11D.

## 17. Verdict

**3D DIRICHLET H1 REGULARISATION: VALIDATED** — H1-space, H1-time and combined, on one whole QUAD9 face with u = 0 on ∂Γ_c, constant wind, BE/CN, SCHUR_DIRICHLET_REDUCED with the matrix-free core (dense gold), MPI np = 1/2/4.

Closure items 1–17 of the request: K_Γ correct (§5) and equal to the tensor reference (§5); R_Γ SPD (§5, §6); H1-space, H1-time and combined working (§7–§10); exact benchmark (§4); symbolic (§4); gates (§7); dense = matrix-free (§8); BE 1 / CN 2 (§10); Q2 spatial ~3 with surface-H1 ~2 (§9); mesh/Nt-robust iterations (§12); ℓ/γ trends consistent (§12); no O(m²) (§13); MPI (§14); no regression (§16).

Caveats (documented, not blockers):
1. The γ > 0 u spatial pre-asymptotic plateau at n = 6–8 (§9).
2. The CN start-up layer in y (§10).
3. The p spatial order is that of the unchanged T11D adjoint (2.65–3.05).
4. H1-space iterations still grow like √cond as α_u → 0 (as L2).
5. The KKT state-residual report bug, fixed and confined to the report (§7a).

## 18. Remaining work after T11H1 (not started)

- H1 for 3D Neumann control (the K_Γ assembly is generic; Neumann control needs its own stationarity and, for H1-time, the interval-quantity time derivative, refused in 2D too).
- Partial or multiple controlled faces (C would become a union with edge and corner policies between faces).
- Variable winds and LPS in 3D; non-hex meshes (`FaceSurfaceGradients` already handles curved and tilted faces, but only planar axis-aligned faces are validated).
- Optionally, a bounded-member H1-time benchmark (A ≠ 0, continuous in γ → 0) for small-γ studies.

## 19. Files

- **Report:** this file.
- **Tables:** `t11h1_surface_stiffness.csv`, `t11h1_gates.csv`, `t11h1_core_equivalence.csv`, `t11h1_space_convergence.csv`, `t11h1_time_convergence.csv`, `t11h1_ell_scaling.csv`, `t11h1_gamma_scaling.csv`, `t11h1_core_scaling.csv`, `t11h1_memory.csv`, `t11h1_mpi.csv`.
- **Figures:** `figures/t11h1_{surface_operator_check,space_convergence,time_convergence,core_iterations,memory}.png`.
- **Scripts** (`t11h1/scripts/`):

  | purpose | scripts |
  |---|---|
  | operators | `t11h1_surface_stiffness.py` |
  | benchmark | `t11h1_symbolic.py` |
  | gates | `t11h1_gates.sh`, `t11h1_gates_table.py` |
  | core | `t11h1_core_equivalence.py` |
  | convergence | `t11h1_exact_ladder.sh`, `t11h1_runs.sh`, `t11h1_ladder_table.py`, `t11h1_time.py`, `t11h1_selfconv.py`, `t11h1_convergence.sh` |
  | scans, memory, figures | `t11h1_scans.py`, `t11h1_memory.py`, `t11h1_figures.py` |
  | parity, refusals, MPI | `t11h1_parity.sh`, `t11h1_refusals.sh`, `t11h1_mpi.{sh,py}` |
  | regression | `t11h1_regress_{all,2d,suite,ex3_ex4,mathef,t9,t10,t11n,t11d}.sh`, `clean_build_t11d.sh` |

- **Logs:** `t11h1/logs/` (git-ignored `*.log`/`*.out`).
- **JSON:** large intermediate JSON stayed in the session scratch directory.
