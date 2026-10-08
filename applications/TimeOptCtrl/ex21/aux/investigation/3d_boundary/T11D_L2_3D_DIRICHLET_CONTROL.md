# T11D-L2 — 3D Dirichlet boundary control, L2 regularisation (ex21, HEX27 / QUAD9)

*Baselines:* e0410efe3 (2D), 8ed57d629 (T9), 577805e36 (T10), **ec825f26d** (T11N, 3D
Neumann; committed at the start of this task).

*Scope:* dimension 3, Dirichlet control on ONE whole face, L2(Γ_c) regularisation, constant
wind, BE/CN, the reduced Dirichlet solver with the matrix-free core, MPI. No H1, partial or
multiple faces, LPS, variable winds or new meshes. Nothing committed, pushed or staged for
T11D.

Scripts, logs and results are in `t11d/`; the CSVs are in this directory.

## 1. Audit of the 2D T7D / T8D implementation (written before any code change)

Sources read: `T7D_DIRICHLET_CONVECTION.md`, `T8D_*`, `DirichletControl.{hpp,cpp}`,
`DirichletReducedSchur.{hpp,cpp}`, `Discretization.cpp` (EliminateDirichletControlDofs,
SpatialOperators), `Solver.cpp`, `ConvectionVerification.cpp`
(`RunDirichletConvectionGates`), `CommandLine.cpp`, `Config.hpp`.

**A. Exact 2D reduced algebra (as implemented).**

- **Sets.** C = support(diag M_Γ) on the control face (`neumann_control_face`, the 2D bottom edge) and I = the rest.
  - `EliminateDirichletControlDofs` copies the *unhatted* M, L (`mass_full`, `stiffness_full`), then eliminates C from M, L, M_track, M_ctrl_box.
  - L gets diagonal 1 and the mass-like operators diagonal 0, so E_I has identity rows on C.
- **Control-to-state.** On the full space-time layout, with Q_C the extension by zero and P_I the zeroing of the C rows:
  - BE: (B_D u)_n = P_I [A Q_C u_n − (M/dt) Q_C u_{n−1}], A = M/dt + L.
  - CN: (B_D u)_n = P_I [A₊ Q_C u_n + A₋ Q_C u_{n−1}], A₊ = M/dt + L/2, A₋ = −M/dt + L/2.
  - u₀ = y₀\|_C/η is data and reaches the first row through the right-hand side.
- **Transpose** (`ApplyBDTImpl`, `transpose = HasConvection`; T7D fix):
  - BE: (B_Dᵀ p)_n = Q_Cᵀ [Aᵀ p_n − (M/dt) p_{n+1}].
  - CN: Q_Cᵀ [A₊ᵀ p_n + A₋ᵀ p_{n+1}].
  - `ApplyBDTLegacy` (forward L) is the negative control.
- **State and objective.** E_I Y = rhs_y − η B_D U; J = (α_t/2)‖y − y_d‖² + (α_u/2) dt Σ_n w_n U_nᵀ R_Γ U_n, with w_n = 1 (BE) and the trapezoidal 1, …, 1, ½ (CN).
- **Regularisation.** For L2, R_Γ = M_Γ, embedded with the identity outside C.
- **Stationarity.** α_u R_T U = η B_Dᵀ P, i.e. u = (η/α_u) R⁻¹ B_Dᵀ p (`ReconstructControl`).
- **Reduced Schur (T8D).** K = R_H1t + β H, with H = B_Dᵀ E_I⁻ᵀ (W_y ⊗ M_track) E_I⁻¹ B_D and β = α_t η²/α_u (`Solver.cpp`: `Prepare(alpha_track*eta*eta/alpha_u)`).
  - It is SPD: H = ZᵀDZ ≥ 0 and R > 0 on C.
  - It is solved by CG on a **distributed** reduced vector (N_t·\|C_owned\| per rank), preconditioned by the sparse direct factor of R_H1t.
  - The dense Cholesky core remains the gold reference (`--dirichlet-core-solver dense`, `--verify-dirichlet-core`).

**B. Dimension-independent.** All of `DirichletControlOperators` (B_D, B_Dᵀ, R, the
reduced layout, sparse R_H1t), `DirichletReducedSchurOperators` (the shell core, CG, the
dense gold core, VerifyCore), the T7D gate driver and the KKT/Solver plumbing. A scan for
coordinates, sorting by geometry or `dimension` finds only `std::sort` of global DOF ids:
there is no geometry.

**C. EDGE3-trace assumptions.**
- The face loop that builds M_Γ and **K_Γ** with a 1D arc-length tangent is 2D-only. T11N added the 3D QUAD9 M_Γ loop for Neumann only.
- The 2D-only face selector: `neumann_control_face` = 1 = bottom, no CLI.
- `OwnedDofIsControlled` (diagnostic mask): the 2D bottom edge.

**D. Consumes only a generic trace DOF list.** The whole Dirichlet machinery: C is read
from diag(M_Γ) after elimination.

**E. Relies on M_Γ, K_Γ or 1D ordering.**
- M_Γ: C and R_Γ (L2).
- K_Γ: R_Γ only for H1-space (ℓ > 0), which is refused in 3D. In 3D K_Γ is empty and ℓ = 0, so R = M_Γ exactly.
- 1D ordering: none.

**Two 2D policies that do NOT carry over:**

1. 2D Dirichlet control uses the **natural** state BC elsewhere, and `Validate` refuses
   Dirichlet control + homogeneous Dirichlet state. T11D's formulation is y = 0 on the
   other five faces, so the combination must be enabled in 3D only. The state-constrained
   set is then D = ∂Ω \ int(Γ_c).
2. The CN-without-sweep refusal of the 3D CLI must not apply to the reduced Dirichlet
   solver.

## 2. 3D Dirichlet formulation (fixed before coding)

    State:      y_t − ν Δy + w·∇y = f          in Ω × (0,T)
                y = η u                         on Γ_c   (one whole face)
                y = 0                           on ∂Ω \ Γ_c
    Control:    u ∈ L²(0,T; L²(Γ_c)),  u = 0 on ∂Γ_c (edge compatibility, below)
    Objective:  J = (α_t/2) ∫ ‖y − y_d‖²_{L²(ω)} + (α_u/2) ∫ ‖u‖²_{L²(Γ_c)}     (no H1)
    Adjoint:    −p_t − ν Δp − w·∇p = α_t (y − y_d) χ_ω,  p = 0 on ∂Ω (all six faces), p(T) = 0
    Stationarity (continuous; ex21 sign p = −λ):
                α_u u = η ν ∂_n p   on Γ_c

On Γ_c, p = 0, so the convective boundary term (w·n)p y vanishes there. Discretely the
stationarity is α_u R_T U = η B_Dᵀ P, where B_Dᵀ p is the variational normal derivative.

**Discrete splitting.** y = P_I y_I + Q_C(η u), where:
- C = int(Γ_c) = the Γ_c nodes **not** on its perimeter;
- D = ∂Ω \ int(Γ_c) = the other five faces, including the perimeter of Γ_c (homogeneous Dirichlet, eliminated as in T9);
- I = the rest (free).

**Edge/corner compatibility.** The perimeter of Γ_c lies on the homogeneous faces.
- The **control space excludes the perimeter nodes**: they are in D, y = 0 there, and u = 0 there by construction. The controlled face does not override the essential BC of its neighbours.
- Implementation: the FEMuS BC callback marks every boundary node Dirichlet **except when called from the controlled face**. FEMuS's `GenerateBdc` only ever sets a node to constrained and never resets it, so the perimeter nodes (shared with a neighbouring face) stay constrained.
- After the homogeneous elimination, M_Γ is zero on D, so C = support(diag M_Γ) = int(Γ_c), with (2n_a − 1)(2n_b − 1) nodes.
- R = M_Γ restricted to C is the face mass of the trace functions that vanish on ∂Γ_c. It is SPD on C.

**B_D, B_Dᵀ** are exactly the 2D formulas of §1A, with the unhatted (C-not-eliminated,
D-eliminated) M and L. They are reused, not duplicated. Under convection Bᵀ uses Lᵀ
(A₊ᵀ, A₋ᵀ).

## 3. Implementation (what changed)

Only the ex21 sources listed here; the Dirichlet algebra (`DirichletControl`,
`DirichletReducedSchur`, the T7D gates, the solver) is reused unchanged.

| File | Change |
|---|---|
| `Config.hpp` | `Validate`: Dirichlet control + homogeneous Dirichlet state allowed in 3D only. `Validate3D`: Dirichlet control needs one whole face, `--state-bc dirichlet`, no control boxes. `OwnedDofIsControlled` (diagnostic mask): the face plane minus its perimeter. |
| `Discretization.{hpp,cpp}` | `InitializeStateSpace(..., free_face)`: the FEMuS BC callback returns *natural* only when called from the controlled face (`g_free_face`), so D = ∂Ω \ int(Γ_c). The T11N QUAD9 face-mass loop now also serves Dirichlet control (M_Γ → C, R). |
| `ex21.cpp` | passes the controlled face to `InitializeStateSpace` for 3D Dirichlet control. |
| `CommandLine.cpp` | `--dirichlet-face3d xmin\|…\|zmax` (names only, as `--neumann-face3d`), its refusals, `dirichlet-conv-3d` as a 3D exact case, the 3D solver whitelist (only `SCHUR_DIRICHLET_REDUCED`), CN allowed without sweep for it, the CN-aware 3D Dirichlet `--dump-operators`. |
| `DirichletControl.cpp` | `BuildReducedRegularization`: preallocation from the measured max row length (the hard 24 was < 25 for a QUAD9 trace row → PETSc "new nonzero caused a malloc", exit 63). |
| `ExactOc3D.{hpp,cpp}` | new case `dirichlet-conv-3d` (§4), its driver branch (Dirichlet march, y_C = ηu, `EX21_DIRICHLET3D_STATIONARITY`), u error on Γ_c at t_n; unused `NatSinF` removed. |
| `Verification.cpp` | 3D Dirichlet `--dump-operators`: unhatted M, L, the C list (+ dt, scheme, transposes_L) and B_D, B_Dᵀ built column by column from `ApplyBD`/`ApplyBDT` on unit space-time vectors. |

## 4. Exact benchmark `dirichlet-conv-3d`

    Γ_c = {z = 0} (face 1), y = 0 on the other faces, ω = (0,1)² × (½,1), ν = ¼, α_u = 1e-2, η = α_t = 1, T = 1
    c_a = w_a/(2ν),  X_c(s) = e^{−cs} sin πs,  τ = T − t
    Qs(z,τ) = e^{−az − cc τ}[G(z − 1.2, τ) − G(z + 1.2, τ)],  G = heat kernel of ν,
              a = w_z/(2ν),  cc = w_z²/(4ν) + ν(2π² + c_x² + c_y²)
    φ = 35s⁴ − 84s⁵ + 70s⁶ − 20s⁷,  s = 2z − 1 on [½,1]  (C³ septic step; 0 below, 1 at z = 1)
    p = X_{cx}(x) X_{cy}(y) (1 − φ(z)) Qs(z,τ)
    u = −(ην/α_u) X X ∂_z Qs(0,τ)          (= η ν ∂_n p / α_u, n = −e_z)
    y = cos 2t · sin³πx sin³πy sin³πz + η u(x,y,t) (1 − z)³
    f = y_t − νΔy + w·∇y,   y_d = y − r/α_t,  r = −p_t − νΔp − w·∇p  (r ≡ 0 for z ≤ ½)

*Design history (why this solution).* The first version used the T7D C^∞ step on
[0.5, 0.9] and the profiles sin / (1 − z). (i) The step's second derivative is huge in thin zones
that 3D meshes up to n = 8 cannot resolve: y order ≈ 2.4, p relative error 0.8, erratic J at
ν = 1 → replaced by the C³ septic step on [½, 1] (z = ½ an element face for even n). (ii) With
the sin interior part and a (1 − z) lifting, y converged like h^{5/2}: the load M_elim I_h f
drops f on the eliminated faces (the T9 effect), so f must vanish on D → sin³ interior part,
(1 − z)³ lifting. (iii) A tangential wind makes the X factors exponential and the lifting
non-polynomial along Γ_c; the clean-order ladder therefore uses the normal wind (0,0,−0.75);
the general wind (1,0.5,−0.75) is reported as well (§9).

**Symbolic verification** (`t11d_symbolic.py`, sympy; winds 0, A = (1,0.5,−0.75), B = (−2,1,0.5),
N = (0,0,−0.75) and a symbolic (v_x,v_y,v_z)): **127/127 PASS** — state PDE (D1), adjoint r = 0
where φ = 0 (D2), y = ηu on Γ_c (D3), y = 0 on the five other faces (D4), p = 0 on all six faces
(D5; z = 0 by image cancellation, z = 1 by 1 − φ(1) = 0), p(T) = 0 as the limit τ → 0⁺
(max_z |Qs| = 1e-172 at τ = 1e-4, D6), stationarity α_u u = ην∂_n p (D7), edge compatibility
u = 0 on ∂Γ_c and y(0)|_Γc = ηu(0) (D8), φ ∈ C³ (D9), and the literal C++ port of f, y_d, y, p,
u against sympy (max rel. 6e-14, D10).

## 5. Trace convention, edge policy, R  (`t11d_trace.py` → `t11d_trace.csv`, **169/169 PASS**)

13 cases: all six faces of the unit cube (3×3×3), three faces at (2,3,4), four faces of the
non-unit box (0,2)×(−1,1)×(0,½) at (4,4,2). For each, from the dump:

- C = support diag(M_Γ) = int(Γ_c) exactly: |C| = (2n_a − 1)(2n_b − 1), all on the face plane,
  no perimeter node.
- The unhatted operators eliminate exactly D = boundary \ int(Γ_c) (y = 0 on the five other
  faces *and on ∂Γ_c*). C ∩ D = ∅. The hatted state operators eliminate D ∪ C (y = P_I y_I + Q_C ηu).
- M_Γ rows off C are zero. R = M_Γ|_C is symmetric (exactly) and SPD (min eigenvalue
  7.8e-3 … ).
- R = [M1(a) ⊗ M1(b)] restricted to the interior face grid (independent 1D Q2 masses),
  relative max entrywise difference ≤ 6.3e-15.
- vᵀRv = ∫ b² dS for the bubble b = Π(s − lo)(hi − s), exactly to 1e-13.

**Edge/corner policy:** u = 0 on ∂Γ_c by construction (the perimeter is in D). The neighbouring
homogeneous faces win at shared edges; the controlled face never overrides their essential BC.

## 6. B_D and B_Dᵀ  (`t11d_bd.py` → `t11d_bd_transpose.csv`, **133/133 PASS**)

The C++ `ApplyBD`/`ApplyBDT` are dumped as explicit space-time matrices and compared with the
BE/CN formulas of §1A rebuilt in Python from the dumped unhatted M, L and C alone. Cases: faces
zmin/xmax × BE/CN × heat/wind A (n = 2, Nt = 3), faces ymax/zmax CN wind B (2,3,3), Nt = 4.

- dumped B_D = formula B_D: ≤ 6.9e-17 rel. Columns off C are zero, rows on C ∪ D are zero (P_I),
  B_D ≠ 0.
- dumped B_Dᵀ = (formula B_D)ᵀ: ≤ 6.9e-17. dumped B_Dᵀ = (dumped B_D)ᵀ: 0. ⟨B_D u, p⟩ = ⟨u, B_Dᵀ p⟩: ≤ 2e-16.
- `transposes_L` = 1 exactly when convection is on.
- **Negative controls.** A forward-L transpose is detected under convection (1.1e-1 … 2.6e-1) and
  coincides for heat (L symmetric). The wrong CN coupling p_{n−1} is detected (1.0).
- **G1D** (exact boundary identity of the 3D split): on the free set I ∪ C, C + Cᵀ = (w·n) M_Γ
  (≤ 2.8e-14 rel. to max|C|) for w·n = 0.75, 1, −1. For a tangential wind it is 0. The plain skew
  identity C + Cᵀ = 0 is violated (the negative): only Γ_c survives the elimination, so the
  `EX21_VC_G1_SKEW_DIRICHLET` ratio (0.375 …) printed by the gates is informational and expected.

## 7. Gates  (`t11d_gates.sh`, `t11d_gates_table.py` → `t11d_gates.csv`, **32/32 PASS**)

`--verify-convection` with the default reduced Dirichlet solver.
- **Exact `dirichlet-conv-3d`:** n = 2, 3 × BE/CN × heat, w0, A, B, normal N, tangential C.
- **Production:** face xmax wind A, ymin wind B, zmax heat, and the non-unit box face ymax wind A, each BE/CN.

| gate | worst over 32 runs |
|---|---|
| G1 orientation (x, y, z) | ≤ 6.2e-15 |
| G2 shell transpose | ≤ 4e-17 |
| G3D ⟨B_D u,p⟩ = ⟨u,B_Dᵀp⟩ | 1.5e-17 |
| R symmetry / min Rayleigh | ≤ 4e-16 / > 0 |
| G5D reduced-Hessian symmetry (exact LU and production shell) | 1.3e-13 |
| G4D FD gradient (exact and shell) | 1.0e-13 |
| G6D production: accepted, outer its, true residual | all, 1, 2.7e-11 |
| G7 ‖∇J(u*)‖/‖dt α_u R u*‖ | 2.9e-10 |
| ‖y_prod − y(u*)‖/‖y‖ | 5.4e-11 |
| core CG iterations / failures | 2–5 / 0 |

**Negative controls** (w ≠ 0: detected; heat and w = 0: coincide to ≤ 3.5e-14):

| negative | minimum over the nonsymmetric runs |
|---|---|
| G2 ex20-style untransposed E | 1.2e-4 |
| G3D legacy forward-L B_Dᵀ | 2.4e-3 |
| G4D FD, legacy B_Dᵀ / ex20-style | 1.2e-3 / 5.6e-3 |
| G5D Hessian symmetry, legacy / ex20-style | 1.3e-5 / 6.3e-6 |

*Criterion (stated honestly):* the first table used one 1e-4 bar for all negatives and flagged
4 runs. Their primary negatives (G2, G3D, G4D) were 2.4e-3 … 5e-1, but the G5D Hessian-*symmetry*
negatives were 6e-6 … 7e-5: a wrong Bᵀ perturbs H only through E⁻¹(L − Lᵀ) and is damped when ω
is far from Γ_c (T11N saw the same). The final criterion is primary negatives > 1e-4 and G5D
negatives > 1e-8, at least 4 orders above the 1e-12 positive bar. Every negative is ≥ 7 orders
above its positive counterpart.

## 8. Reduced core: SPD and dense vs matrix-free  (`t11d_core_equivalence.py` → `t11d_core_equivalence.csv`, **8/8 PASS**)

*SPD proof.* K = R + βH with R = I_{Nt}^{w} ⊗ M_Γ|_C (trapezoid weights > 0; M_Γ|_C SPD, §5),
β = α_t η²/α_u > 0, H = Zᵀ(W ⊗ M_track)Z with Z = E_I⁻¹B_D and W ⊗ M_track ⪰ 0, so H ⪰ 0 and
K ≻ 0. Numerically, via `--verify-dirichlet-core` (dense K assembled column by column):
- ‖K − Kᵀ‖/‖K‖ = 0 and λ_min > 0 (≥ 9.8e-4), with cond(K) 9–25 and cond(R⁻¹K) 1.00–1.22;
- the dense and shell actions agree to ≤ 7.8e-16 over 9 probe vectors each;
- shell symmetry ≤ 3e-16;
- sparse vs dense R_T: 0.

*Solutions.* Five production cases (xmax A BE/CN, ymin B CN, zmax heat BE, non-unit ymax A CN)
compare the full space-time y, p, u **node by node** (JSON) and J. Three exact cases compare J
and the error norms.

| core vs dense | max relative difference |
|---|---|
| iterative at the default rtol 1e-8 | ≤ 4.0e-11 |
| iterative at rtol 1e-12 | ≤ 1.4e-14 |

## 9. Spatial convergence  (`t11d_convergence.sh` → `t11d_space_convergence.csv`; figure `t11d_space_convergence.png`)

Exact errors, CN, N = 256. y and p are measured in L²(0,T;L²(Ω)); u in L²(0,T;L²(Γ_c)), nodal
in time. The default solver is used throughout.

| n | normal wind y | p | u | heat y | p | u |
|---|---|---|---|---|---|---|
| 4 | 5.86e-3 | 7.47e-3 | 9.43e-3 | 5.90e-3 | 2.37e-3 | 8.45e-3 |
| 6 | 1.50e-3 (3.36) | 1.45e-3 (4.05) | 1.44e-3 (4.64) | 1.54e-3 (3.31) | 4.08e-4 (4.34) | 1.70e-3 (3.95) |
| 8 | 6.15e-4 (3.10) | 4.99e-4 (3.70) | 5.59e-4 (3.28) | 6.40e-4 (3.06) | 1.67e-4 (3.10) | 7.38e-4 (2.90) |
| 10 | 3.11e-4 (3.05) | 2.76e-4 (2.66) | 2.87e-4 (2.99) | 3.24e-4 (3.06) | 8.86e-5 (2.84) | 3.72e-4 (3.07) |
| 12 | 1.79e-4 (3.04) | 1.64e-4 (2.84) | 1.63e-4 (3.11) | 1.85e-4 (3.07) | 5.08e-5 (3.05) | 2.06e-4 (3.24) |

The CN N = 256 time error, extrapolated from the n = 6 time ladder (y 7e-6, p 1e-5, u 3e-5),
removed in quadrature, changes the last orders by ≤ 0.08 (u 3.19 / 3.29, p 2.85 / 3.12).

- **y** and **u** (surface) are cleanly order 3.
- **p** is pre-asymptotically superconvergent (~4) and then settles around 3. For the normal
  wind the last steps oscillate (2.66, 2.84; n = 8 → 12 mean 2.74), and n = 14 (~21 GB) does not
  fit on this machine.
- General wind A (n ≤ 8): y 3.09, p 3.88, u 3.54. The tangential-wind lifting caveat of §4 did
  not show at these sizes.
- Every run: stationarity 1.5e-16 … 1.7e-16, independent march ≤ 1.5e-14, 1–2 outer iterations,
  4–6 core CG iterations.

## 10. Temporal convergence  (`t11d_time_convergence.csv`; figure `t11d_time_convergence.png`)

n = 6, normal wind, production path (`--solve-only --dump-json`, core rtol 1e-12). u is compared
at t_n for both schemes (the Dirichlet control is nodal); p at the CN midpoints.

- **CN** (reference N = 512): y, p and u are order 2.00 / 2.22 / 2.07 at N = 128 (reference-corrected
  1.99 / 2.17 / 2.00). u on t ≤ T/2 is 2.04.
- **BE** (reference N = 1024): the orders rise monotonically to 0.95 / 0.97 / 0.95 (reference-corrected,
  N = 256); u on t ≤ T/2 is 1.03. BE is slower to become asymptotic because the exact solution is a
  heat kernel that steepens as t → T.
- **Last-slot u.** In both schemes the last-slot u difference plateaus at ~1e-6 (max|u| = 0.8).
  It is a fixed-mesh spatial floor of the terminal control, not a time effect:
  - it is unchanged with outer rtol 1e-14 (true residual 1e-16);
  - at fixed N = 128, max|u_N| = 3.0e-4 / 7.0e-7 / 8.0e-9 for n = 4 / 6 / 8 (exact u(T) = 0, and ≈ 1e-13 on the last 0.1).

## 11. Heat / w = 0 parity  (`t11d_parity.sh`, `logs/parity/parity.log`)

`--convection off` vs `--convection on` with w = 0, faces zmin and xmax, BE and CN.

- **Operators:** hatted M, K, M_Γ, M_track, coords, the unhatted M and L, the C list, and the explicit
  B_D and B_Dᵀ are **40/40 byte-identical**. The C-list header differs only in its `transposes_L` path flag (0/1).
- **Solves** (dense and iterative core, BE and CN):
  - same solver path (SCHUR_DIRICHLET_REDUCED / EXACT_WOODBURY), same outer and core counts;
  - fields agree to 2.9e-11 (BE) and 4.6e-11 (CN) relative; J agrees to 3.5e-13 / 4.8e-13.
  - This is the outer-tolerance level of a one-step outer solve (cf. §13). w = 0 selects the
    MatMultTranspose path, which is equal in exact arithmetic.

## 12. Core iteration scaling and memory

**Scaling** (`t11d_core_scaling.py` → `t11d_core_scaling.csv`, figure `t11d_core_iterations.png`).
Production data, ω = (0,1)² × (0.3,1) adjacent to Γ_c (the strongest coupling the free-DOF tracking allows),
CG + sparse R_T Cholesky, rtol 1e-8. 47 runs, no failure.

| scan | CG iterations |
|---|---|
| mesh n = 2 … 10 (Nt = 32) | 4–5 (heat, wind A, normal wind) |
| Nt = 8 … 512 (n = 4) | 5 (heat), 4 (wind A) |
| α_u = 1e-1 … 1e-6 | 4 → 105 (heat CN) and 3 → 95 (wind A CN), BE ≤ 43 |

The α_u growth follows √cond(R⁻¹K): cond(R⁻¹K) 1.01 → 1457, the same 1/α_u dependence as 2D T8D.
Convection never needs more iterations than heat.

**Memory** (`t11d_memory.py` → `t11d_memory.csv`, figure `t11d_memory.png`). CN, wind A; n = 4, 6, 8.

- The dense excess RSS_dense − RSS_iter equals the predicted 12 B m² + 8 B Nt N |C| to a ratio of
  1.011 … 1.057 (up to 1.9 GB at m = 12544).
- The iterative RSS is linear in Nt: 305 → 562 → 1078 → 2108 MB for Nt = 128 → 1024 at n = 4,
  about 1.5 kB per space-time DOF. **No O(m²) storage.**
- At Nt = 1024 (m = 50176) the dense core would need about 29 GB extra; the iterative core runs in 2.1 GB.
- Iterative is faster at every size (n = 4, Nt = 256: 10.6 s vs 38.7 s), and J agrees with dense to ≤ 1.7e-13.

## 13. MPI np = 1/2/4  (`t11d_mpi.sh`, `t11d_mpi.py` → `t11d_mpi.csv`, **15/15 PASS**)

Cases: heat BE, convection BE, convection CN (exact data, n = 4, N = 8); production face xmax wind B CN;
the non-unit box face ymax BE.

- The controlled trace is compared node by node: the same int(Γ_c) node set at every np (49 / 21 nodes),
  u = 0 elsewhere including ∂Γ_c, and y_C = ηu_C exactly.
- y, p, u agree with np = 1 to ≤ 8.9e-15, and J to ≤ 2.6e-16.
- Outer and core CG counts are identical at every np.
- r₀ is identical; every later residual is ≤ 1e-8 r₀.
- Outer rtol is 1e-14. At the default 1e-8 the outer stops after **one** step at a 1e-11 residual, and
  the np differences (1e-11 … 4e-11) sit exactly at that level
  (`logs/mpi_table_default_outer_rtol.out`; dense core and CG 1e-14 give the same numbers, so the core is not the cause).
- The T11N history criterion (r_i within 1e-12 r₀) is not applicable here: after one exact-Woodbury
  step, r₁/r₀ = 1e-11 … 4e-9 is itself round-off through the np-dependent direct factors.

## 14. Whitelist / refusals  (`t11d_refusals.sh`, `logs/refusals.log`, **52/52**)

**Refused (24 Dirichlet-specific):**
- missing face, a bad name, or a numeric id;
- `--state-bc natural` (the 2D natural-elsewhere variant is not ported);
- `--dirichlet-face3d` with distributed or Neumann control, or in 2D;
- `--neumann-face3d` with Dirichlet control;
- control boxes, full domain, partial face or union;
- H1-space, H1-time, combined;
- SCHUR_SWEEP, matching, boundary-reduced;
- variable winds, LPS;
- a tracking region touching Γ_c;
- `--verify-dirichlet-core` with the iterative core;
- exact case + face, region or sweep.

**Refused (carried over):** 12 T11N/T10/T9 refusals.

**Accepted (16):**
- all six faces under CN wind A;
- explicit `--pc-schur-dirichlet-reduced`, the dense core, `--dirichlet-core-pc none`, the dense verify;
- heat BE, the non-unit box, the exact case;
- T11N Neumann, T10 distributed, T9 production.

*Test-design notes:* tracking boxes are selected by element centroid. At n = 2 the centroids 0.25 / 0.75
lie on a 0.25–0.75 box edge, so that box is empty (reason 3, zero problem) — a test-box choice, not a
Dirichlet issue; the battery uses boxes strictly containing centroids.

## 15. Mandatory regression  (`t11d_regress_all.sh`; logs `t11d/logs/t11d_regress_*.log`)

OLD = the T11N build: a private clean build of **ec825f26d** (`git archive`, sanitised compile and link,
headers remapped to the archive; recipe `t11d/scripts/clean_build_t11n.sh`; sha256 2cbd12ce…; no
Miniforge). NEW = the current build (sha256 53db08be…). NEW was rebuilt after the last, comment-only
header edit and is byte-identical, so all results above come from the final sources.

| Suite | Result |
|---|---|
| ex20 goldens | **61/61** identical |
| Math2025 Ex3/Ex4, Ex5 | byte-identical |
| EF 9.5 | only the known `variant=source` label, as in T11N |
| R4 T0–T2 constant wind | **REGRESSION PASS** |
| R5 T3/T4 | **9/9** identical |
| 2D old vs new | **55/55 IDENTICAL**, **34/34 JSON_IDENTICAL** (the 2 boundary-reduced + w0 cases are refused identically by both binaries, as in T11N) |
| 2D/3D restrictions | outcome-for-outcome identical to the T11N log |
| T9 invariance | **32/32** identical (gates, exact driver, JSON, dumps); MPI np 2 round-off 3.0e-15 |
| T10 invariance | **27/27** identical |
| T11N invariance (new) | **43/43** identical: Neumann gates (exact heat/A/B/C and production xmax / non-unit ymax, BE/CN), exact driver, JSON (LU and GAMG), operator dumps of three faces; no new dump files for Neumann |

Build hygiene:
- `readelf` / `ldd` of the NEW binary, its `link.txt`, `flags.make` and `CMakeCache.txt` contain no Miniforge/Conda.
- All builds and runs used the sanitised PETSc-arch environment.
- The only use of Miniforge Python is the offline sympy script `t11d_symbolic.py`, as in T11N; it never touches FEMuS.

## 16. Verdict

**3D DIRICHLET BOUNDARY CONTROL (L2): VALIDATED** — one whole face, HEX27/QUAD9, u = 0 on ∂Γ_c,
y = 0 on the other five faces, constant wind, BE/CN, the reduced Dirichlet solver with the
matrix-free CG core, MPI.

Caveats (documented, not blockers):
1. **p order.** Under the normal wind, p's last spatial orders are 2.66 / 2.84 (mean 2.74 for n = 8 → 12)
   after a superconvergent phase. Under heat they are 2.84 / 3.05. y and u are cleanly 3.
2. **BE pre-asymptotics.** BE reaches 0.95–0.97 (u on t ≤ T/2: 1.03) only at N = 256, because the
   kernel steepens towards T.
3. **Last-slot u floor.** It is a spatial floor of the terminal control that vanishes with h.
4. **Default outer tolerance.** At the default outer rtol 1e-8 the solve takes one outer step with
   a 1e-11 residual. This sets the w = 0 parity (3–5e-11) and the default-tolerance MPI agreement (≤ 4e-11).

## 17. Remaining work for T11H1 (not started)

1. **Surface-gradient K_Γ on QUAD9 faces.** The 2D K_Γ uses a 1D arc-length tangent. In 3D, ∇_Γ needs
   the two in-plane derivatives; restrict to int(Γ_c) with the perimeter in D. R_H1s = M_Γ + ℓ² K_Γ is
   then SPD on C, since K_Γ ⪰ 0 and M_Γ ≻ 0.
2. **Lift the refusals.** Allow `--control-h1-length` / `--control-h1-time-weight` for 3D Dirichlet,
   keeping H1 refused for 3D Neumann/distributed unless separately validated.
3. **Reuse the H1-time machinery.** R_H1t (time stiffness ⊗ R_Γ) and its sparse factor are already
   dimension-independent (T8D). Check the sparse Cholesky preallocation (the `BuildReducedRegularization`
   fix of §3 sizes rows from the measured row length, so K_Γ couplings are covered, but this must be
   re-verified).
4. **New exact benchmark.** Stationarity becomes α_u(u − ℓ²Δ_Γ u − γ u_tt) = ην∂_n p with natural
   conditions on ∂Γ_c replaced by u = 0. It needs a new solution and a symbolic check.
5. **Gates.** Re-run G3D/G5D/G4D, the dense-vs-matrix-free core check, iteration scaling in ℓ and γ
   (2D: mesh-independent), memory, MPI, and the regression.

## 18. Files

- **Report:** this file.
- **Tables:** `t11d_trace.csv`, `t11d_bd_transpose.csv`, `t11d_gates.csv`, `t11d_core_equivalence.csv`,
  `t11d_space_convergence.csv`, `t11d_time_convergence.csv`, `t11d_core_scaling.csv`, `t11d_memory.csv`,
  `t11d_mpi.csv`.
- **Figures:** `figures/t11d_space_convergence.png`, `t11d_time_convergence.png`, `t11d_core_iterations.png`,
  `t11d_memory.png`.
- **Scripts** (`t11d/scripts/`):

  | purpose | scripts |
  |---|---|
  | operator checks | `t11d_bd.py`, `t11d_trace.py`, `t11d_symbolic.py` |
  | gates | `t11d_gates.sh`, `t11d_gates_table.py` |
  | core | `t11d_core_equivalence.py`, `t11d_core_scaling.py`, `t11d_memory.py` |
  | convergence | `t11d_exact_ladder.sh`, `t11d_runs.sh`, `t11d_ladder_table.py`, `t11d_time.py`, `t11d_selfconv.py`, `t11d_convergence.sh`, `t11d_figures.py` |
  | parity, refusals, MPI | `t11d_parity.sh`, `t11d_refusals.sh`, `t11d_mpi.sh`, `t11d_mpi.py` |
  | regression | `t11d_regress_{all,2d,suite,ex3_ex4,mathef,t9,t10,t11n}.sh`, `clean_build_t11n.sh` |

- **Logs:** `t11d/logs/`.
- **Large intermediate JSON** (the time ladder, MPI) stayed in the session scratch directory and is not archived.
