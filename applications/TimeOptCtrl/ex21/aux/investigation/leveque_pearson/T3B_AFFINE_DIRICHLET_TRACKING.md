# T3b — affine tracking contribution of inhomogeneous Dirichlet data

*§1 was written before any code was changed. §2 onward record the
implementation and the validation.*

## 1. Derivation

### 1.1 Setting

The physical state at time slot `n` is

    y_n = y0_n + G,

where:
- `y0_n` is the solver's homogeneous unknown, with `y0_{n,Γ} = 0` on the set Γ
  of constrained (Dirichlet) DOFs;
- `G` is the lift, equal to `g` on Γ and 0 on the free set F.

Write `d_n = I_h y_d(t_n)` (nodal values at every DOF), and let `M` be the
**full** (un-eliminated) tracking mass matrix, `M_track_full`. The residual is
`r_n = y_n − d_n`. Partitioned, `r_n,F = y0_n,F − d_n,F` and
`r_n,Γ = g_Γ − d_n,Γ`; the second part is data.

### 1.2 Objective

The discrete tracking functional is (`w_n` = BE 1; CN `(1, …, 1, ½)` for
`n = 1..N`, plus the constant initial node with weight ½):

    J_track = (α_t/2) dt Σ_n w_n r_nᵀ M r_n  [+ (α_t/2) dt (½) r_0ᵀ M r_0  under CN]

Expanding each slot over the F/Γ partition gives

    r_nᵀ M r_n = r_n,Fᵀ M_FF r_n,F + 2 r_n,Fᵀ M_FΓ r_n,Γ + r_n,Γᵀ M_ΓΓ r_n,Γ.

The three terms are:
- **free–free:** quadratic in the unknown;
- **cross term:** linear in the unknown;
- **boundary–boundary:** a constant.

The CN initial term `r_0 = y_0 − d_0` is entirely data; its boundary part is
`g_Γ − d_0,Γ`.

### 1.3 Derivative with respect to the unknown y0_n,F

    ∂J_track/∂y0_n,F = α_t dt w_n [ M_FF (y0_n,F − d_n,F) + M_FΓ (g_Γ − d_n,Γ) ].

### 1.4 What the current code computes (read from the source before any change)

- **Constrained-set elimination** (`Discretization.cpp`,
  `EliminateConstrainedDofs`, diagonal 0) turns `M_track` into
  `M_track,elim`. This matrix equals `M_FF` on F×F and is zero in every Γ row
  and column.
- **Adjoint RHS** (`BuildDistributedRightHandSide`):
  `rhs_p,n = −α_t w_n M_track,elim d_n`, i.e. `−α_t w_n M_FF d_n,F` on F.
- **KKT second row:** `−α_t (W ⊗ M_track,elim) Y0 + Eᵀ P = rhs_p`, i.e.
  `Eᵀ P = α_t W ⊗ M_FF (Y0 − D)`.
- **Result:** the code keeps `M_FF (y0_F − d_F)` and **drops `M_FΓ (g_Γ − d_Γ)`**.

  In the homogeneous case (`g = 0`) the dropped term is `−M_FΓ d_Γ`. That is the
  historical convention of evaluating the tracking residual on F only, which is
  equivalent to replacing the boundary trace of `y_d` by 0.
- **Objective value** (`ReconstructDistributedDataSide`): `J_track` is
  evaluated with `M_track,elim` on the **homogeneous** unknown, because the lift
  is added only afterwards, in `SolveOptimalControl`. The CN initial term also
  uses `M_track,elim`. Both the cross term and the constant are therefore
  missing.

  The cross term is in fact zero only because `M_elim` has no Γ columns. Had it
  been evaluated with the full matrix on `y0`, the physical state on Γ would
  have been taken as 0 instead of `g`.

### 1.5 Correction (the KKT Hessian `M_FF` stays as it is)

- **Adjoint RHS.** ex21's adjoint row is `Eᵀ P − α_t W M_FF Y0 = rhs_p`. The
  correct stationarity is

      Eᵀ P = α_t W [ M_FF (Y0 − D) + M_FΓ (g − D)_Γ ],

  so

      rhs_p,n  +=  +α_t w_n [ M (R_Γ(g − d_n)) ]_F,

  where `R_Γ` keeps the Γ entries of a nodal vector and zeroes the F entries.

  *Erratum, recorded honestly:* the first version of this section (and of the
  code) had a minus sign here. It was caught by the new physical-objective
  finite-difference gate G3P before any campaign run. With the wrong sign G3P
  gave 1.7e-3, twice the 8e-4 of the omitted term. It was corrected in both
  places (§2).
- **Objective:** `J_track` is evaluated with the full `M` on the physical
  state `y0_n + G`, including the CN initial node.
- **Scope of the correction:**
  - It is zero whenever the tracking region contains no element touching Γ, or
    when `d_Γ = g_Γ`.
  - It is applied **only when inhomogeneous Dirichlet data are installed**, so
    the homogeneous path is byte-identical. The homogeneous path keeps its
    historical convention (the trace of `y_d` is dropped). That is documented
    as a known, separate limitation of that path (see §6).
  - The **source load** convention `M_elim I_h f` (it drops `f`'s boundary
    nodal values) is **unchanged**. That is a separate known limitation for
    `f ≠ 0` on ∂Ω. LP §5.2 has `f = 0`.

## 2. Implementation (binary 825465c0; T4 additions on top, which are opt-in only: 500a2b2e)

| file | change |
|---|---|
| `Discretization.hpp` | `struct DirichletLift { Vec g; Vec load; Mat tracking_full; std::vector<unsigned char> constrained; bool affine_tracking; }` replaces the two T3 vectors. `SpatialOperators::lift` owns it. |
| `Discretization.cpp` | When lifting data are installed: `MatDuplicate(M_track)` **before** the constrained-DOF elimination, plus the Γ mask. That is the only new storage: one sparse matrix, and only with lifting data. The eliminated `M_track` (the KKT Hessian `M_FF`) is unchanged. |
| `Solver.cpp` | `AddAffineTracking()` computes `out_n += scale·w_n [M_full R_Γ(g − d_n)]_F`. It is used in three places: <br>• `BuildDistributedRightHandSide`, scale `+α_t`; <br>• `ComputeKktResiduals`, adjoint residual, scale `+α_t` on the tracking side; <br>• the objective in `ReconstructDistributedDataSide`, which evaluates `J_track` with `M_full` on the physical state `y0 + G`, including the CN initial node. <br>All three are gated on the lift being present; null means the previous code path. |
| `Config.hpp`, `CommandLine.cpp` | Diagnostic toggle `lift_affine_tracking` / `--lift-no-affine-tracking`, the negative control. It restores the pre-T3b right-hand side, but the objective stays physical. |
| `ConvectionVerification.cpp` | New gate **G3P**: the finite-difference derivative of the **physical** full-mass objective, independent of `rhs_p`, against the adjoint gradient. |
| `ExactOcBenchmark.cpp` | New case `lift-track`. `J_track` is checked against a direct full-mass recomputation, and `J_h` against the continuous `J` (3-point Gauss in time on 64 subintervals, times element quadrature). |

The source load (`M_elim I_h f`) is **unchanged** (§1.5).

## 3. Manufactured validation (`lift-track`)

The data are `ν = 1/4`, `v = (1, ½)`, `α_u = 1e-2`, `T = 1`. The lift is
`g = ½ + exp((v·x − v_x − v_y)/ν)` (steady, smooth, nonzero trace). The state is
`y = g + cos(2t) sin³πx sin³πz`, and the adjoint is
`p = α_u sin(T−t) sin 2πx sin πz` (s = sin, **not** cubed).

The adjoint vanishes on ∂Ω, but its normal derivative does not. Hence
`y_d − g = α_u sin(T−t) (v·n) ∂_n S2 ≠ 0` on ∂Ω: `2π α_u v_x sin πz sin(T−t)`
on `x = 0, 1`, and `±π α_u v_y sin 2πx sin(T−t)` on `z = 0, 1`. The source `f`
still vanishes on ∂Ω, so the source convention is exact for this case.

The symbolic check (`t3b/t3b_symbolic_check.py`) gives all three optimality
residuals equal to 0, `p(T) = 0`, `y = g` and `p = 0` on ∂Ω, and the nonzero
`y_d − g` above.

| ladder | with the affine term (T3b) | without (negative control = pre-T3b) |
|---|---|---|
| space, CN Nt = 128: `y` | 3.13, 3.03, **3.01** | 3.13, 3.03, 3.01 (y is insensitive) |
| space: `p`, `u` on `t ≤ T/2` | 3.46, 3.25, **3.04** | 3.28, 2.92, **2.67** |
| space: error of `p` (t ≤ T/2) at nx = 32 | **4.61e-5** | 9.42e-5 (**2.04×**, and the ratio grows 1.12, 1.26, 1.58, 2.04) |
| space: `p` on (0,T), nx = 16 → 32, with Nt = 256 at nx = 32 | **3.00** | 2.66 |
| CN time, nx = 32 (p) | 1.75, 1.90, **1.97** | same to 3 digits (temporal errors dominate) |
| BE time, nx = 32 (p / u) | 1.01, 1.00, **0.99** | same |
| `|J_h − J_exact|/J_exact` | 2.7e-2 → 5.3e-6, order **4.1–4.2** | identical to 4 digits |

**Why the whole-interval adjoint gives 2.17 at nx = 32, Nt = 128.** That number
is the CN terminal adjoint layer seen in T2: at Nt = 256 the same mesh gives
order 3.00.

**Why the objective error cannot detect the term.** The objective converges
at order about 4 with or without the term. Its error is quadratic in the state
error, and the state is barely affected. The term is detected by the
**adjoint** and by **G3P**, not by J.

**Gates** (`--verify-convection --exact-oc lift-track`, nx = 8, Nt = 16):

| | CN | BE |
|---|---|---|
| space-time transpose | 7.3e-18 | 1.6e-17 |
| reduced-objective FD gradient | 2.7e-12 | 1.6e-12 |
| reduced Hessian symmetry | 7.2e-14 | 3.8e-14 |
| **G3P (physical full-mass objective)**, with the term | **1.9e-14** | **3.3e-14** |
| **G3P without the term (negative control)** | **8.4e-4 (fails)** | **8.0e-4 (fails)** |
| G4, production solve: `‖∇J(u*)‖/‖dt α_u N u*‖` (at production rtol) | 9.8e-7 | 2.0e-7 |

The reduced-objective FD check passes even without the term, because it
differentiates an objective built from `rhs_p` itself. This is why G3P was
added.

**All lift-track runs with the term:**
- relative KKT residuals ≤ 8.7e-12, including the affine term in the adjoint
  residual;
- independent march ≤ 1.7e-12;
- `|y − g|` on ∂Ω ≤ 1.2e-13;
- `|p|`, `|u|` on ∂Ω ≤ 1.9e-16 (the adjoint stays homogeneous);
- `J_track` equals a direct full-mass recomputation from the returned state to
  1.4e-16.

## 4. Regression

| check | result |
|---|---|
| homogeneous / constant-wind path: operator dumps (12), 8 archived T1/T2 runs, T2 gate logs | **byte-identical** |
| ex20 goldens | **61/61** |
| Mathematics 2025 Ex. 3 (2 runs), Ex. 4, Ex. 5 | **byte-identical** |
| ErrorFeedback 9.5 | numerics identical. The CONFIG label differs (`variant=source`, pre-T3), as in T3 |
| T3 lift cases (`y_d = g` on ∂Ω): lift-exp CN/BE, lift-const + variable wind; errors and `J_track`, `J_reg` | relative change 0, 0, 3.2e-11 (round-off) |
| T4 rebuild (opt-in `--exact-oc-nu`, `--exact-oc-block-test`) vs the T3b binary: lift-track, lp52, variable wind | identical `EX21_` output |

`t3b_tracking_validation.csv`: **34/34 PASS**.

## 5. LP §5.2 small β, OLD (T3) vs NEW (T3b)

Setup: ε = 1/20, CN, Q2 nx = 32 / 64, production tolerances. Script:
`t3b/lp52_old_vs_new.py`. Data: `lp52_smallbeta_corrected.csv`. Figure:
`figures/lp52_corner_old_new.png`.

| β | nx | version | J_track | p_LP(t=1) range | peak, dist. to ∂Ω | odd-even (band / interior) | p wiggle rows | p change nx 32→64 (all / dist ≥ 0.1) | outer its | Schur its/outer |
|---|---|---|---|---|---|---|---|---|---|---|
| 1e-2 | 64 | old | 4.183e-3 | [−0.02122, 0.00586] | (0.875,−0.67), 0.125 | 0.039 / 0.015 | 0/129 | 2.8e-2 / 4.6e-4 | 1 | 8 |
| 1e-2 | 64 | **new** | 4.701e-3 | [−0.02122, 0.00586] | same | 0.046 / 0.015 | 0/129 | **7.5e-3** / 1.0e-3 | 1 | 8 |
| 1e-4 | 64 | old | 8.04e-4 | [−4.43e-4, 1.56e-3] | (0.953, 0.969), 0.031 | 0.41 / 0.018 | 0/129 | 0.27 / 5.6e-3 | 1 | 9 |
| 1e-4 | 64 | **new** | 1.30e-3 | [−4.43e-4, 1.55e-3] | (0.953, 0.984), 0.016 | 0.51 / 0.018 | 0/129 | **0.095** / 3.2e-3 | 1 | 9 |
| 1e-6 | 64 | old | 3.56e-4 | [−1.81e-5, 1.38e-4] | (0.969, 0.984), 0.016 | 0.95 / 5.9e-4 | 43/129 | 1.99 / 3.0e-2 | 1 | 9 |
| 1e-6 | 64 | **new** | 7.24e-4 | [−4.44e-5, 1.84e-4] | same | 1.12 / 4.2e-4 | 44/129 | 2.48 / 0.15 | 1 | 9 |

The nx = 32 rows are in the CSV. Every run converged:
- relative KKT residual ≤ 7e-8, consistent with the production rtol;
- 1–2 outer iterations, 8–9 Schur iterations per outer iteration;
- no spatial sweep failure.

**J_track.** The NEW values are the physical objective. They include the
boundary–boundary constant and the cross term, so they are not comparable with
the OLD (free-DOF) values in absolute terms. The NEW value is **mesh-stable**:
at β = 1e-2 it is 4.703e-3 at nx = 32 and 4.701e-3 at nx = 64, against
3.86e-3 → 4.18e-3 before.

**How much of the small-β corner layer came from the omitted term?**

| β | finding |
|---|---|
| 1e-2 | Part of the nx 32→64 adjoint difference (2.8e-2 → 7.5e-3, about 3.7× less) came from the omitted term. The rest of the qualitative picture is unchanged: same trough, same range, no wiggles. **The structural agreement with LP Fig. 2 remains.** |
| 1e-4 | The mesh-to-mesh change of p drops from 27 % to 9.5 % (about 2.9×). The peak stays at the (1, +1) corner. So a substantial part of the earlier mesh sensitivity was the omitted term. A sharp corner feature remains. |
| 1e-6 | The term **does not** remove the feature. It sharpens it slightly: p range wider, 44/129 wiggle rows (was 43), change 2.48 (was 1.99), all within a 1–2-node band at (1, +1). Interior odd-even amplitudes stay ≤ 4e-4. |

**Conclusion.** The β ≤ 1e-6 corner layer is **physical / resolution-driven**.
- It is a tracking-induced adjoint layer at the corner where `y_d → 1` while the
  state is pinned to 0 on `x2 = 1`, and where LP's boundary data are
  discontinuous.
- With the correct affine term it is, if anything, slightly stronger, because
  the adjoint now "sees" the mismatch `g − y_d` on Γ instead of ignoring it.
- It is under-resolved at these meshes.
- It is **not** an artifact of the dropped term, and with a cell Péclet number
  ≤ 0.63 it is not convective.
- At β ≥ 1e-4 the dropped term was responsible for a large share of the
  mesh sensitivity.

## 6. Limitations still present (documented, unchanged)

1. **Homogeneous Dirichlet path** (`g = 0`): the trace of `y_d` is still
   dropped (historical convention), for byte-identity. Enabling the same
   correction there (term `−M_FΓ d_Γ`) would be a deliberate, separately
   reviewed change of the default path.
2. **Source load:** `M_elim I_h f` drops `f`'s boundary nodal values. This
   matters only for `f ≠ 0` on ∂Ω (order 2.5); LP §5.2 has `f = 0`.

## 7. Verdict

The affine tracking contribution of inhomogeneous Dirichlet data is now
correct:
- in the adjoint right-hand side;
- in the KKT residual check;
- in the objective value.

The evidence: order 3 is restored for the adjoint (2.66 without the term), G3P
passes at 2e-14 (8e-4 without the term), and nothing outside the lifted path
changed.
