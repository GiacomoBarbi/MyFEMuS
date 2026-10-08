# T11N — 3D Neumann boundary control under convection (ex21, HEX27 / QUAD9)

*Baselines:*

- e0410efe3 (2D);
- 8ed57d629 (T9, 3D phase 1);
- **577805e36** (T10, 3D volume features and natural BC; committed at the start of this task).

*Scope:* dimension 3, Neumann control on ONE whole face, L2(Γ_c) regularisation, constant
wind, BE/CN, SCHUR_SWEEP, MPI. No Dirichlet control, H1, LPS, variable winds or non-hex
meshes. Nothing committed, pushed or staged for T11N.

Scripts, logs and results are in `t11n/`; the CSVs are in this directory.

## 1. The current (2D, T7N) Neumann algebra and its dimension audit

Read from `T7N_NEUMANN_CONVECTION.md` and the code (Discretization, Solver,
SpaceTimeOperators, Preconditioners, Config, CommandLine).

| piece | 2D algebra (as implemented) | dimension-independent? |
|---|---|---|
| state coupling | M ẏ + L y = F + η M_Γ u, with L = νK + C; `SelectedControlOperator` returns N = M_Γ for NEUMANN | **yes**: only N is consumed |
| boundary mass | (M_Γ)_ij = ∫_{Γ_c} φᵢφⱼ dS, assembled in the face loop of `BuildNativeSpatialOperators` for `face_id == neumann_control_face`, by JacobianSur | the mass part **yes**. The same loop also builds K_Γ with a **1D arc-length tangent** (`dphi_dxi`, a 2-component tangent). That is edge-only, and the whole loop is guarded `dimension == 2` since T9 |
| control space | the active set = support of diag(N) (`ActiveControlDofs`); u = −(η/α_u)p on it, 0 elsewhere | **yes** |
| adjoint coupling | Eᵀp with Lᵀ (exact transpose); CN coupling transposed | **yes** |
| stationarity | α_u M_Γ u + η M_Γ p = 0 ⇔ u = −(η/α_u)p\|_{Γ_c} | **yes** |
| time discretisation | BE: nodal; CN: interval-constant control, forcing η M_Γ u_{n−½} on interval n, adjoint and control at midpoints | **yes** |
| geometric mask (`OwnedDofIsControlled`, diagnostic `geometric_mask_dofs`) | Neumann ⇒ \|z − zmin\| ≤ tol, i.e. the bottom EDGE (storage axis 1) | **no**: 2D-only, needs the 3D face |
| face selection | `neumann_control_face = 1` (no CLI; always the bottom edge `y0`) | **no**: 3D needs an explicit face selector |
| boundary-reduced Schur | Cholesky / MAT_SPD; A± taken symmetric | heat-era and SPD-specific; refused under convection (T7N) and refused in 3D (T9) |
| ADDITIVE_GAMG / SCHUR_MATCHING | algebraically generic. In 2D they fail for Neumann under CN, with or without convection (T7N §11) | must be audited in 3D, not assumed |
| exact-OC driver | 2D-only (`IntegrateBoundaryField(x, z)`) | the 3D driver needs a 3D surface integral |

**Conclusion:** the KKT, sweep, control-reconstruction and time code is
dimension-independent. The 3D work is:

1. a QUAD9 face mass on a selected face, without the 1D tangent;
2. a 3D face selector with verified orientation;
3. the 3D geometric mask;
4. a 3D surface error integral and the exact benchmark;
5. the whitelist.

## 2. 3D Neumann-control formulation (fixed before coding)

    State:     y_t − ν Δy + w·∇y = f            in Ω × (0,T)
               ν ∂_n y = η u                    on Γ_c        (one whole face)
               ν ∂_n y = 0                      on ∂Ω \ Γ_c   (natural)
    Weak:      M ẏ + L y = F + η M_Γ u,   L = νK + C,   (M_Γ)_ij = ∫_{Γ_c} φᵢ φⱼ dS (QUAD9 face)
    Objective: J = (α_t/2) ∫ ‖y − y_d‖²_{L²(ω)} + (α_u/2) ∫ ‖u‖²_{L²(Γ_c)}
    Adjoint (DTO, exact transpose Lᵀ):
               −p_t − ν Δp − w·∇p = α_t (y − y_d) χ_ω,   p(T) = 0
               ν ∂_n p + (w·n) p = 0            on ALL six faces, Γ_c included
    Stationarity (ex21 sign: p = −λ):
               α_u M_Γ u + η M_Γ p = 0   ⇔   u = −(η/α_u) p|_{Γ_c}

The Robin adjoint condition is not imposed. It is what the exact transpose of L produces,
because C + Cᵀ = Σ_f ∫_f (w·n) φᵢφⱼ (T10 G1N, generalised to all six faces). Γ_c is not
special for the adjoint.

## 3. Face convention and selection

**FEMuS face numbering of the HEX27 box**, taken from the T10 G1N report (JacobianSur outward normals). It is **re-verified at every 3D Neumann assembly**: after building M_Γ, the code compares the integrated outward normal and the area of the selected face with the selector's (axis, sign) and the box extents, and throws on a mismatch above 1e-10.

| id | name | outward normal |
|---|---|---|
| 1 | zmin | (0,0,−1) |
| 2 | ymin | (0,−1,0) |
| 3 | xmax | (+1,0,0) |
| 4 | ymax | (0,+1,0) |
| 5 | xmin | (−1,0,0) |
| 6 | zmax | (0,0,+1) |

**CLI:** `--neumann-face3d xmin|xmax|ymin|ymax|zmin|zmax`. It is 3D-only and required for 3D Neumann, together with an explicit `--neumann-solve`. ex21's default control type is NEUMANN (the 2D bottom edge), so a 3D run without a control flag is refused rather than silently controlled on face 1.

**Refused with Neumann:**
- partial-face patches and unions (`--control-box3d*`, `--control-full-domain`);
- `--neumann-face3d` with distributed control or with an exact case;
- a homogeneous Dirichlet state.

The volume control box is set to the full domain and is inert, since N = M_Γ.

## 4. QUAD9 face mass M_Γ (`t11n_face_mass.py`, `t11n_face_mass.csv`: 117/117 PASS)

Coverage: all six faces at (3,3,3); zmin, xmax and ymax on an anisotropic (2,3,4) mesh; zmin, xmin, ymax and zmax on (0,2)×(−1,1)×(0,½) with (4,4,2).

| check | result |
|---|---|
| support = face-plane nodes; count (2n_a+1)(2n_b+1) | exact, every case |
| symmetry | max\|M − Mᵀ\| = 0 |
| SPD on the trace support | min eigenvalue > 0 (1.19e-3 at (3,3,3)) |
| 1ᵀM_Γ1 = \|Γ_c\| | ≤ 1e-13 (areas 1, 4, 1, …) |
| vᵀM_Γv = ∫_Γ (x + 2y − 3z)² dS (linear interpolant, Gauss-exact) | ≤ 1e-12 |
| **M_Γ = M₁(a) ⊗ M₁(b)** (two independently assembled 1D Q2 masses) | ≤ 1e-14 relative entrywise |
| N_selected = M_Γ; off-face rows of N zero | exact |
| K_Γ | empty (no tangential stiffness in 3D) |

## 5. Trace layout and MPI

- The control space is the support of diag(N) (`ActiveControlDofs`). It is built from PETSc's distributed M_Γ: face nodes on rank boundaries, edges and corners are assembled once and owned once.
- **Exact driver (serial):** active nodes = (2n+1)², all on z = 0, at every n = 2…12, with `active_off_plane = 0`.
- **Production:** `active_dofs` = `geometric_mask_dofs` (the new 3D face mask), with off-active = 0.
- **MPI:** trace node count, node-by-node trace values and off-trace zeros agree at np = 1/2/4 (§13).

## 6. Neumann state/control algebra

The state rows carry η M_Γ u (weak Neumann flux; no Dirichlet elimination; natural state BC).

The coupling is checked three ways:
- assembled M_Γ = the Kronecker face integral (§4);
- M_Γ is symmetric, so the coupling transpose Nᵀ = N appears unchanged in both KKT off-diagonal blocks;
- G6 (independent forward march with the computed u): ≤ 2.2e-15 (§8).

Control reconstruction is u = −(η/α_u)p on supp N, and 0 elsewhere:
- face stationarity ‖M_Γ(α_u u + ηp)‖ / ‖α_u M_Γu‖ ≈ **3.6–4.0e-17** in every exact run;
- max |u| off the face = **0 exactly**.

## 7. Adjoint boundary condition and exact benchmark

**Adjoint boundary condition.** The DTO adjoint is the exact transpose of L. **G1N:** C + Cᵀ = Σ_f ∫_f (w·n)φᵢφⱼ to **≤ 3.1e-14** over all 26 gate runs. The identity uses all six faces, Γ_c included; Γ_c is **not** replaced by plain Neumann.

**`neumann-conv-3d`** (`ExactOc3D.cpp`, `t11n_symbolic.py`): the 3D extension of T7N, with Γ_c = {z = 0}.

    X(s) = 1 − cos 2πs      (X = X' = 0 at s = 0, 1)
    Z = R_kz,  R_k(s) = e^(−ks)(2 + cos πs),  kz = w_z/ν
    p = α_u B X(x) X(y) Z(z),   B = sin(T − t)
    u = −(η/α_u) p|_{z=0} = −η B X(x) X(y) Z(0)
    y = A cos πx cos πy cos πz + κ(t) X(x) X(y) (z − z²/2),   κ = η² B Z(0)/ν
    f = y_t − νΔy + w·∇y
    y_d = y + (p_t + νΔp + w·∇p)/α_t

- On the four lateral faces, p = ∂ₙp = 0. This is forced by smooth compatibility at the edges of Γ_c, where ν∂ₙy = ηu meets ν∂ₙy = 0.
- The Robin term is exercised on **Γ_c** and on z = 1 through w_z ≠ 0. Wind A gives w·n = +0.75 on Γ_c (outflow); wind B gives −0.5 (inflow).
- The algebraic wrong-BC gate G8b uses random vectors, so it detects all six faces.

**Symbolic verification: 100/100 PASS** for w = 0, A, B, C = (1, ½, 0) and a symbolic wind:
- state and adjoint PDEs;
- the controlled flux ν∂ₙy = ηu on Γ_c and ν∂ₙy = 0 on the 5 other faces;
- the Robin condition on all 6 faces;
- p(T) = 0 and stationarity;
- C++ literals for f, y_d and u to ≤ 3.5e-16 relative;
- max|(w·n)p| and the wrong-BC residual on Γ_c: 0.076 (A) and 0.050 (B), both > 0.

## 8. Algebraic gates (`t11n_gates.sh`, `t11n_gates.csv`: 26 runs, all PASS)

The runs cover:
- neumann-conv-3d at n = 2, 3, BE/CN, winds heat / w0 / A / B / C;
- production runs on face **xmax** (wind A) and **ymin** (wind B), n = 3;
- the non-unit domain on face **ymax**, tracking y > 0, T = 1.

The outer rtol is 1e-12.

| gate | correct adjoint (max) | ex20-style adjoint (wind) | wrong natural BC νK − C (wind) |
|---|---|---|---|
| G1N boundary identity | 3.1e-14 | — | — |
| G1 orientation x/y/z | 2.6e-14 | — | — |
| G2 transpose | 3.5e-17 | 1.4e-4 – 3.3e-2 | 2.9e-5 – 1.2e-2 |
| G3 FD gradient | 1.2e-11 | **2.4e-2 – 24** | **9.3e-2 – 55** |
| G4 reduced-Hessian symmetry | 1.2e-13 | 1.6e-2 – 0.95 | 5.8e-4 – 1.7 |
| G5 outer / true KKT residual | 2–3 / 2.0e-13 | — | — |
| G7 gradient consistency | 1.3e-11 | — | — |
| G6 independent forward march | 2.2e-15 | — | — |
| face stationarity / off-face u (exact driver) | ≤ 4.0e-17 / 0 | — | — |

- With w = 0 both wrong adjoints coincide with the exact one (FD ≤ 1.1e-11), as they must.
- Wind C (w·n = 0 on Γ_c) is still detected, through the x- and y-faces.
- A first version of the non-unit case put the tracking region at distance 1 from the control face with T = 0.2. The reduced Hessian was then nearly blind (negative controls 6e-7–1e-5, FD still 2e-3). The redesigned case (tracking adjacent, T = 1) gives O(1) negatives.
- *Wrong face-coupling orientation* (optional negative control) is covered by construction rather than by a run: the runtime face assertion (§3), the geometric support and Kronecker checks (§4), and the exact-solution convergence (§9) would all fail for a wrong face.

## 9. Spatial convergence (`t11n_space_convergence.csv`, `figures/t11n_space_convergence.png`)

CN, N = 64, LU. Exact errors: y, p in L²(0,T; L²(Ω)); **u in L²(0,T; L²(Γ_c))** with the face quadrature of M_Γ.

| n | wind A: y | p | u (Γ_c) | wind B: y | u | heat: y | u |
|---|---|---|---|---|---|---|---|
| 4 | (3.84) | (4.19) | (4.12) | (3.64) | (3.65) | (3.77) | (3.93) |
| 6 | (3.20) | (4.11) | (3.64) | (3.08) | (3.23) | (3.10) | (3.23) |
| 8 | 8.04e-3 (3.10) | (4.00) | 6.66e-3 (3.43) | (3.05) | (3.24) | (3.05) | (3.15) |
| 10 | 4.06e-3 (3.06) | (3.89) | 3.21e-3 (3.27) | (3.03) | (3.16) | (3.03) | (3.08) |
| 12 | 2.33e-3 (3.04) | (3.71) | 1.82e-3 (3.12) | | | | |

The CN N = 64 time error, measured on the same data (§10), is y 2.5e-4, p 3.3e-5, u 4.1e-4. Removing it in quadrature gives, for wind A at n = 10 / 12:

| field | order n = 10 | order n = 12 |
|---|---|---|
| y | 3.07 | **3.06** |
| u | 3.30 | **3.22**, approaching 3 from above, as in 2D T7N (3.45 → 3.16 → 3.04) |

**Verdict: Q2 order 3 for y and for u on Γ_c.** p is empirically superconvergent (≈ 3.7–4, as in 2D T7N's ≈ 4) on this smooth case. That is not claimed in general; no order above 3 is claimed.

## 10. Temporal convergence (`t11n_time_convergence.csv`, `figures/t11n_time_convergence.png`)

Fixed mesh n = 6, reference N = 512 on the same mesh, wind A. u is measured on Γ_c (face Simpson weights). The CN control is interval-constant, so p and u are compared at interval midpoints (T7N convention).

| N | BE y | BE p | BE u | CN y (L²(0,T)) | CN y [¼,T] | CN y(T) | CN p | CN u | CN u last slot |
|---|---|---|---|---|---|---|---|---|---|
| 8 | 0.97 | 1.07 | 0.97 | 1.79 | 1.84 | 1.56 | 2.01 | 1.59 | 1.08 |
| 16 | 0.98 | 1.03 | 0.95 | 1.44 | 1.79 | 1.73 | 2.00 | 1.74 | 1.30 |
| 32 | 0.99 | 1.02 | 0.94 | 1.26 | 2.48 | 1.88 | 2.00 | 1.86 | 1.53 |
| 64 | 0.99 | 1.01 | 0.95 | 1.25 | 2.24 | 1.97 | 2.00 | 1.94 | 1.73 |
| 128 | **1.00** | **1.00** | **0.97** | 1.49 | **2.12** | **2.06** | **2.01** | **1.99** | 1.91 |

The BE orders and the CN p, u orders are reference-corrected; the CN y orders are raw.

- **BE: order 1** for y, p and u on Γ_c.
- **CN: order 2** for p, for u on Γ_c, for J (|ΔJ| ratios ≈ 4), and for y away from t = 0.
- **CN state start-up layer:** the first-level difference decays slowly (2.6e-3 → 8.7e-4), holding the global y order near 1.3–1.5 at these N. The **same diagnostic on the 2D T7N `neumann-conv` case**, run with a binary whose 2D path is unchanged (nx = 8), shows **the same pattern**: global 1.83 → 1.33, [¼, T] 2.1–2.3, y(T) → 2.04. The layer belongs to the inherited CN scheme with boundary-flux control, not to the 3D port.
- **CN control terminal layer:** the u last slot converges at 1.08 → 1.91, as documented in T7N §9 (amplified by 1/α_u). The full-interval u order is 1.99.
- The frozen time scheme is unchanged.

## 11. Solver audit (`t11n_solver.sh`, `t11n_solver.csv`)

neumann-conv-3d data, n = 8, N = 32, wind A (Pe_cell ≈ 0.34), production rtol 1e-8.

| scheme | solver | outer | spatial its/solve (max) | true res. | fields vs LU | wall | MaxRSS |
|---|---|---|---|---|---|---|---|
| BE | SWEEP + LU | 2 | 1 | 5.1e-13 | — | 43 s | 1054 MB |
| BE | SWEEP + GMRES/GAMG | 2 | 11 | 5.1e-13 | 2.5e-15 | 119 s | 987 MB |
| BE | SWEEP + GMRES/ASM-ILU | 2 | 9 | 5.1e-13 | 1.4e-14 | 422 s | 1193 MB |
| CN | SWEEP + LU | 2 | 1 | 5.1e-13 | — | 49 s | 1094 MB |
| CN | SWEEP + GMRES/GAMG | 2 | 11 | 5.1e-13 | 5.9e-15 | 113 s | 1009 MB |
| CN | SWEEP + GMRES/ASM-ILU | 2 | 7 | 5.1e-13 | 1.8e-14 | 289 s | 1280 MB |
| BE | ADDITIVE_GAMG (default PC) | **46** | 22 | 4.8e-8 | 2.0e-6 | 347 s | **2546 MB** |
| CN | ADDITIVE_GAMG | **refused** in 3D (CN needs SWEEP; pre-existing, T7F/T8D) | | | | | |
| BE | SCHUR_MATCHING | **refused** in 3D (2D T7N: fails for Neumann CN even without convection) | | | | | |
| BE | SCHUR_BOUNDARY_REDUCED (convection) | **refused**: SPD/Cholesky, A± taken symmetric (T7N audit #12) | | | | | |
| BE | SCHUR_BOUNDARY_REDUCED (heat) | **refused** in 3D: not proven in 3D, not redesigned here | | | | | |

**Policy: SCHUR_SWEEP is the 3D Neumann solver** (2 outer iterations everywhere; all backends agree to round-off).

ADDITIVE_GAMG stays permitted for BE, as for 3D distributed control. It is algebraically generic (T7N audit #11: SAFE) and converges, but it is not recommended: 46 outer iterations, 8× slower, and 2.4× the memory, because it builds GAMG on the assembled space-time operator.

## 12. Heat limit (`t11n_parity.sh`, `logs/parity/parity.log`)

**Operators:** convection off vs on with w = (0,0,0), faces zmin and xmax. M, K, **M_Γ**, N_selected, M_track and the coordinates are **byte-identical**.

**Solves** (n = 4, N = 16, BE/CN):
- LU: fields ≤ 6.8e-15, J to 1 ulp, outer and inner iterations identical.
- Iterative backend: fields ≤ 6.6e-15, inner 6161 vs 6155.

**Path difference (not algebraic):** with convection on, the transposed blocks are used, and the spatial Krylov changes from CG to GMRES. The `inner=TIME_SWEEP+CG/GAMG` label is static.

## 13. MPI (`t11n_mpi.sh/.py`, `t11n_mpi.csv`: 15/15 PASS)

Cases, at np = 1/2/4 (verified nrank) with MUMPS LU for np > 1:
- neumann-conv-3d BE and CN with wind A;
- heat CN;
- production on face xmax, wind B, BE;
- the non-unit domain on face ymax, CN.

Against np = 1, matched by coordinates:

| quantity | agreement |
|---|---|
| y | ≤ 1.5e-15 |
| p | ≤ 2.8e-14 (relative to max\|p\| ~ α_u) |
| u | ≤ 1.6e-15 |
| **control trace (node by node on the face: 81 / 45 nodes)** | ≤ 1.6e-15 |
| u off the trace | exactly 0 |
| J | ≤ 3.9e-15 |
| outer iterations | 2 at every np |
| FGMRES histories | ≤ 3.4e-15·r₀ |
| true residuals | equal to the printed digits |

## 14. Memory (`t11n_memory.py`, `t11n_memory.csv`, `figures/t11n_memory.png`)

| n (N = 64) | trace DOFs | space-time unknowns | MaxRSS Neumann | T9 distributed | T10 natural |
|---|---|---|---|---|---|
| 4 | 81 | 0.14 M | 291 MB | 306 MB | 287 MB |
| 8 | 289 | 0.94 M | 2011 MB | 2010 MB | 1987 MB |
| 10 | 441 | 1.78 M | 3903 MB | 3883 MB | 3860 MB |
| 12 | 625 | 3.0 M | 6755 MB | 6705 MB | 6682 MB |

Neumann memory equals the same-size distributed runs to within 1% (0.91–0.99× the T9 law). **No dense surface object appears**, and memory stays O(N_t·n_3D). The one exception is ADDITIVE_GAMG (2.4×, §11).

## 15. Updated 3D whitelist and refusals (`t11n_refusals.sh`, `logs/refusals.log`: 38/38)

| accepted in 3D (after T11N) | refused in 3D (explicit `invalid_argument`) |
|---|---|
| distributed control: full / box / multi-zone (T9/T10) | **Dirichlet boundary control** ("not ported (T11D)") |
| **Neumann boundary control on ONE whole face** (`--neumann-solve --neumann-face3d F`, any of the 6 faces) | Neumann without a face, with a bad face name, or with an implicit control type |
| L2 regularisation | H1-space, H1-time, combined H1 (also with Neumann) |
| natural state BC (required with Neumann) | Neumann with the homogeneous Dirichlet state |
| constant wind; BE/CN | partial-face patches and unions (control boxes / full-domain with Neumann) |
| SCHUR_SWEEP (BE/CN); ADDITIVE_GAMG (BE) | SCHUR_BOUNDARY_REDUCED (with or without convection), SCHUR_MATCHING, CN without SWEEP |
| rectangular HEX27 domains (Neumann on non-unit boxes) | variable winds, LPS, 1D/2D presets, 2D region syntax, `--neumann-face3d` in 2D, region or face options with exact cases |
| exact cases conv-sin(3)-3d, natural-conv-3d, **neumann-conv-3d** | non-hex meshes (no CLI path: the 3D mesh is always the HEX27 box) |

The accepted lines are all six faces (CN, wind A), a non-unit domain on ymax, Neumann heat BE, neumann-conv-3d, and the T10 / T9 distributed command lines.

## 16. Regression

| suite | OLD | result |
|---|---|---|
| 2D (`t11n_regress_2d.sh`): ex20 goldens, Math2025 Ex3/4/5, EF 9.5, R4 T0–T2, R5 T3/T4, 55 old-vs-new 2D configurations (T6, T7N 2D Neumann, T7D, T7F, T8D iterative default and verify, Dirichlet heat with all regularisations, natural-BC verifies), zero-wind ex20/ex21 parity | the T10 build (577805e36, private clean build) | **61/61**; byte-identical; EF: only the known `variant=source` label; **REGRESSION PASS**; 9/9; **55/55 IDENTICAL**; **34/34 JSON_IDENTICAL** (the 2 boundary-reduced + convection cases refused identically by both) |
| T9 (`t11n_regress_t9.sh`): 16 gates, exact driver, production JSON, dumps, MPI | the T9 build (8ed57d629) | **31/31 identical** (plus the T10 `_regions3d.txt` file; MPI round-off 2.5e-15) |
| T10 (`t11n_regress_t10.sh`): natural-BC gates (A, B, heat × BE/CN), multi-zone / Dirichlet multi-zone / non-unit gates, natural exact driver, multi-zone / natural / GAMG production JSON, multi-zone + sources dumps | the T10 build | **27/27 identical** |

No existing 1D/2D/3D-distributed behaviour changed.

## 17. Verdict

**3D NEUMANN BOUNDARY CONTROL: VALIDATED**

| # | criterion | evidence |
|---|---|---|
| 1 | face selection | runtime normal/area assertion; 6 faces; geometric mask = active set |
| 2 | QUAD9 M_Γ | 117/117, Kronecker-exact |
| 3 | trace layout | (2n+1)² on the plane, MPI node-by-node |
| 4 | state/control coupling | G6 march 2e-15, stationarity 4e-17, off-face 0 |
| 5 | exact transpose | G2 3.5e-17 |
| 6 | adjoint Robin BC exercised | G1N 3.1e-14; symbolic Robin on Γ_c; wrong-BC negatives O(1) |
| 7 | exact benchmark | neumann-conv-3d |
| 8 | symbolic checks | 100/100 |
| 9 | algebraic gates | 26/26 |
| 10 | spatial order | y 3.04–3.06, u(Γ_c) → 3 from above |
| 11 | BE 1 / CN 2 | §10 |
| 12 | SCHUR_SWEEP robust | 2 outer everywhere, three backends |
| 13 | MPI np = 1/2/4 | 15/15 |
| 14 | no regression | §16 |

Inherited, documented behaviours (not blockers):
- the CN state start-up layer, reproduced identically in 2D T7N;
- the CN control terminal layer (T7N);
- p superconvergence (not claimed);
- ADDITIVE_GAMG is permitted for BE but slow.

## 18. Remaining before T11D (not started)

- **3D Dirichlet boundary control:**
  - the Dirichlet lifting on a QUAD9 face;
  - B_D / B_Dᵀ (DirichletControl) on face DOFs;
  - the reduced Woodbury core (T8D iterative / dense) with a 3D face trace;
  - the Dirichlet-reduced Schur solver in 3D;
  - a 3D Dirichlet exact benchmark (T7D analogue).
- **Surface H1 regularisation:** a surface gradient on QUAD9 faces (the 2D K_Γ uses a 1D arc-length tangent; JacobianSur provides no surface gradient), then H1-space and H1-time for Neumann and Dirichlet control.
- **Partial-face patches, unions of faces, edge controls** (selector and face-mass support), if wanted.
- **The boundary-reduced Schur solver in 3D** (heat): it would need a 3D proof; it stays SPD-specific under convection.
- **The 3D matrix-free Dirichlet core** and its memory.
