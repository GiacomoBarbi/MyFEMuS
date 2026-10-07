# ex21 — convection–diffusion: formulation and validation (Phases 2–4)

Binary sha256 `c5f89f0e…df19e606` (`logs/final_binary.sha256`); all numbers
from `scripts/run_all_gates.sh` on that binary (rerun after the Example 3 and
Example 4 additions; identical to the earlier binaries `e5412ac1…`,
`256681ba…`). Serial runs.

---

## A. State equation, weak form and sign convention (derived, then verified)

ex21 solves

    y_t − ν Δy + v·∇y = f + η u   in Ω × (0, T),   v constant,

the **strong (non-conservative)** convection form. Multiply by a test function
`φ_i`, integrate, integrate by parts **only the diffusion term** (for constant
`v` the strong and conservative forms coincide, `∇·(v y) = v·∇y`):

    (y_t, φ_i) + ν(∇y, ∇φ_i) + (v·∇y, φ_i) = (f, φ_i) + η(u, φ_i) [+ ν ∫_∂Ω ∂_n y φ_i].

With `y_h = Σ_j y_j φ_j`:

    M ẏ + (K + C) y = F + η N u,
    M_ij = ∫ φ_j φ_i,   K_ij = ν ∫ ∇φ_j·∇φ_i,   C_ij = ∫ (v·∇φ_j) φ_i .

**Row `i` is the test function, column `j` the trial function.** `C` is
added to the stiffness at assembly (`Discretization.cpp`), so every ex20 code
path that used `K` now uses

    L = K + C,      L ≠ L^T  whenever v ≠ 0.

The convection term is on the **left** with a **plus** sign, i.e.
`y_t = νΔy − v·∇y + …`. A source written as `A = a d²/dx² − v d/dx`
(Mathematics 2025, eq. (65)) therefore maps to **`ν = a`, `vx = +v`**.

**Verification of the convention (gate G1, not assumed):**

* *Orientation.* For a linear field `y = x` (or `z`), `v·∇y = v_x` is
  constant, so the strong-form row must satisfy `(C y)_i = v_x (M 1)_i`
  **exactly** (quadrature is exact). Natural BC, `C = L − K` from two
  assemblies on the same mesh:

  | case | `‖C·x − vx·M1‖∞/‖M1‖∞` | transposed `‖Cᵀ·x − vx·M1‖∞/‖M1‖∞` |
  |---|---|---|
  | 2D, `v=(2,1)`, axis x | **4.7e-14** | 1.15e+01 |
  | 2D, axis z (`vy`) | **9.0e-14** | 2.30e+01 |
  | 1D, `vx = 2` | **9.8e-13** | 4.75e+01 |

  So `C` is assembled with the intended orientation **and** sign (a wrong sign
  gives `−vx·M1`, a transposed one fails by O(10)). FEMuS
  `add_matrix_blocked` → `MatSetValuesBlocked` is row-oriented; the gate
  confirms it numerically.
* *Skew symmetry.* For `div v = 0`,
  `C_ij + C_ji = ∫_∂Ω (v·n) φ_i φ_j`, which vanishes whenever `i` or `j` is
  interior. After the homogeneous-Dirichlet elimination:
  `‖C + Cᵀ‖_F/‖C‖_F = 1.3e-14` (2D), `2.4e-14` (1D), while
  `‖L − Lᵀ‖_F = 3.41` (2D) — `L` is genuinely nonsymmetric, `C` is exactly
  skew on the free DOFs.
* *MMS* (§F) converges at the optimal rates with `v ≠ 0`; a sign or orientation
  error would leave an O(1) consistency error.

**Scope (as briefed).** Constant `v`, constant scalar `ν`, 1D (`EDGE3`, new
`--dimension 1`) and 2D (`QUAD9`), homogeneous Dirichlet first, standard
Galerkin (no SUPG), DISTRIBUTED control only. Runtime options:

| option | meaning | default |
|---|---|---|
| `--convection on\|off` | enable `C` and the transpose code paths | `off` (= ex20) |
| `--vx`, `--vy` (`--vz` must be 0) | velocity; `vy` acts along the 2nd mesh axis (`z` in the code) | 0 |
| `--nu` / `--diffusivity` | `ν` (ex20 option) | 1 |
| `--dimension 1\|2` | mesh dimension | 2 |
| `--domain xmin xmax zmin zmax` | domain (z ignored in 1D) | unit square |
| `--state-bc natural\|dirichlet` | state BC | natural (= ex20) |

## B. BE and CN blocks (ex20 convention, audited from the code)

Unknowns `y_n ≈ y(t_n)`, `n = 1..N`, `y_0` data; `p` is ex20's dt-scaled
multiplier, `u = −(η/α_u) p` on the active set.

**Backward Euler** (`SpaceTimeOperators.cpp`, `Solver.cpp`):

    (M/dt + L) y_n − (M/dt) y_{n−1} = M f(t_n) + η N u_n,   (n = 1: + (M/dt) y_0 to the rhs)
    E   = blockdiag(M/dt + L) − blocksubdiag(M/dt)
    E^T = blockdiag(M/dt + Lᵀ) − blocksuperdiag(M/dt)

**Crank–Nicolson**:

    A₊ y_n + A₋ y_{n−1} = ½ M (f(t_n) + f(t_{n−1})) + η N u_n,
    A₊ = M/dt + L/2,   A₋ = −M/dt + L/2,   (n = 1: −A₋ y_0 to the rhs)
    E   = blockdiag(A₊)  + blocksubdiag(A₋)
    E^T = blockdiag(A₊ᵀ) + blocksuperdiag(A₋ᵀ),   A₊ᵀ = M/dt + Lᵀ/2, A₋ᵀ = −M/dt + Lᵀ/2

Reduced 2×2 (unchanged from ex20):

    [ E                    (η²/α_u) I⊗N ] [y]   [rhs_y]
    [ −α_t W⊗M_ω               Eᵀ        ] [p] = [rhs_p],   W = I (BE), diag(1,…,1,½) (CN).

**Where `L` and `Lᵀ` appear.** `L` in the diagonal of `E` (BE, CN) and in its
sub-diagonal (CN only); `Lᵀ` in the diagonal of `Eᵀ` and its super-diagonal
(CN). ex20 applied `L` (= `K`, symmetric) in both; ex21's adjoint shell applies
`MatMultTranspose` whenever convection is on, and the assembled `Eᵀ` is
`MatTranspose(E)`. **No continuous-adjoint operator (`ν∆ + v·∇`) is assembled
anywhere**: the discrete adjoint is `Eᵀ`, exactly.

## C. Transpose tests (gate G2) — random space-time vectors, fixed seeds

| case | `\|⟨Ex,y⟩−⟨x,Eᵀy⟩\|/(‖Ex‖‖y‖)` shell | `‖E_aij x − E x‖/‖Ex‖` | `‖Eᵀ_aij y − Eᵀy‖/‖Eᵀy‖` | `‖E_aijᵀ y − Eᵀy‖` | **negative control**: ex20-style adjoint (`L` untransposed) |
|---|---|---|---|---|---|
| 2D BE `v=(2,1)` | **2.1e-18** | 1.5e-16 | 2.1e-16 | 2.1e-16 | 2.0e-04 ✗ |
| 2D CN | **3.0e-17** | 1.8e-16 | 1.9e-16 | 1.9e-16 | 1.0e-03 ✗ |
| 1D BE `vx=2` | **1.0e-17** | 1.4e-16 | 1.7e-16 | 1.7e-16 | 2.3e-04 ✗ |
| 1D CN | **5.0e-18** | 1.4e-16 | 1.4e-16 | 1.4e-16 | 4.7e-04 ✗ |

**Reduced operator.** `H U = ∇J(U) − ∇J(0)` (forward with LU of `E`, adjoint as
listed). Symmetry `|⟨Hu₁,u₂⟩−⟨u₁,Hu₂⟩|/|⟨Hu₁,u₂⟩|`:

| case | adjoint = LU(`Eᵀ` assembled) | adjoint = **production shell** `Eᵀ` (GMRES, 1 it) | ex20-style adjoint |
|---|---|---|---|
| 2D BE | 2.0e-16 | **2.0e-16** | 6.4e-03 ✗ |
| 2D CN | 4.0e-16 | **2.4e-15** | 2.3e-03 ✗ |
| 1D BE | 6.3e-16 | **8.9e-14** | 7.1e-03 ✗ |
| 1D CN | 1.3e-15 | **1.3e-13** | 4.8e-03 ✗ |

Gate configuration: 2D `nx=nz=8`, `N=8`, `dt=0.05`, `ν=0.5`, `v=(2,1)`; 1D
`[0,2]`, `nx=32`, `N=16`, `ν=2`, `vx=2`; homogeneous Dirichlet; distributed box
control; `α_u = 1e-3`, target `sin(x−t)`. Logs `logs/verify_convection_*.log`.

## D. (see C) — the reduced operator is symmetric only with the exact transpose

## E. Reduced-gradient finite-difference test (gate G3)

Objective: the **reported** functional
`J = Σ_n dt[(α_t/2) w_n ‖y_n − y_d‖²_{M_ω} + (α_u/2) u_nᵀ N u_n]`, evaluated by
an independent forward LU solve; gradient
`∇J = dt α_u N u + η N Λ`, `EᵀΛ = dt[α_t (W⊗M_ω) Y + rhs_p]`. Non-optimal
random base point and direction; central differences, step
`ε·‖U₀‖/‖D‖`, `ε = 10⁻¹ … 10⁻⁷`. Best relative error:

| case | LU(`Eᵀ`) | **production shell `Eᵀ`** | ex20-style adjoint |
|---|---|---|---|
| 2D BE | 3.2e-16 | **2.1e-15** | 1.2e-01 ✗ |
| 2D CN | 1.3e-15 | **1.6e-15** | 1.2e-01 ✗ |
| 1D BE | 9.2e-15 | **4.9e-14** | 3.0e-01 ✗ |
| 1D CN | 3.1e-15 | **5.0e-15** | 3.0e-01 ✗ |

Example (2D CN, production shell): `ε = 1e-1 … 1e-6` give rel 1.6e-15,
2.2e-15, 8.0e-14, 5.3e-13, 1.7e-12, 5.8e-12 (cancellation-limited, as
expected for a quadratic `J`). The gate is sensitive: using ex20's adjoint
product with the nonsymmetric `L` gives a **12–30 %** gradient error.

**Gate G4 — the production KKT solve.** Reduced gradient at the production
control `u*`, relative to one of its two cancelling terms:

| case | ADDITIVE_GAMG | SCHUR_MATCHING | SCHUR_SWEEP |
|---|---|---|---|
| 2D BE | 14 its, `\|∇J\|`rel 8.2e-11 | 2 its, 4.0e-13 | 2 its, 1.9e-13 |
| 2D CN | 14 its, 9.7e-10 | 2 its, 4.1e-13 | 2 its, 1.0e-13 |
| 1D BE | 16 its, 3.6e-09 | 2 its, 1.1e-11 | 2 its, 9.5e-14 |
| 1D CN | 20 its, 4.4e-10 | **reason −11** (see H) | 2 its, 1.3e-13 |

and `‖y_prod − E⁻¹(rhs + ηNu*)‖/‖y‖ ≤ 1.4e-11` in every converged case. All
three preconditioners give the same `J` (2D BE `J_track = 4.35206779e-4`,
`J_reg = 1.10640402e-4`).

## F. Manufactured solutions (`--convection-mms sin|sin3`)

`y = cos(2t) s(ξ) [s(ζ)]`, `f = y_t − νΔy + v·∇y`, homogeneous Dirichlet,
`T = 1`. 2D: unit square, `ν = 0.5`, `v = (2,1)`; 1D: `[0,2]`, `ν = 2`,
`vx = 2`. The state equation `EY = rhs_y` is marched with a spatial LU and the
all-at-once residual `‖EY − rhs‖/‖rhs‖` is checked with the production shell
(≤ 4.4e-12 in every run). `L²` error of the FE field by element quadrature.
Orders are `log₂` of successive ratios (`results/mms_ladders.csv`).

### F.1 Time (shape `sin`)

| ladder | errors `max_n ‖e(t_n)‖_L2` (N = 4, 8, 16, 32, 64[, 128, 256]) | observed orders |
|---|---|---|
| 2D BE `v=(2,1)`, nx=64 | 1.58e-2, 8.90e-3, 4.73e-3, 2.43e-3, 1.24e-3 | 0.83, 0.91, 0.96, **0.98** |
| 2D BE `v=0`, nx=64 | 1.69e-2, 9.73e-3, 5.14e-3, 2.64e-3, 1.34e-3 | 0.80, 0.92, 0.96, **0.98** |
| 1D BE `vx=2`, nx=256 | 5.82e-2 … 1.11e-3 | 0.86, 0.93, 0.96, 0.98, 0.99, **0.995** |
| 2D CN `v=(2,1)`, nx=64 | 1.85e-3, 4.69e-4, 1.17e-4, 2.94e-5, 7.84e-6 | 1.98, 2.01, 1.99, 1.91 |
| 2D CN `v=0`, nx=64 | 2.07e-3, 5.20e-4, 1.29e-4, 3.23e-5, 8.08e-6 | 1.99, 2.01, 2.00, **2.000** |
| 1D CN `vx=2`, nx=256 | 7.59e-3 … 1.82e-6 | 2.02, 2.00, 2.00, 2.00, 2.00, **1.999** |

**BE → first order, CN → second order**, with and without convection. The last
2D CN `v=(2,1)` step (1.91) is where the nx=64 spatial error of the `sin` shape
(§F.2) becomes visible; the `v=0` ladder and the 1D ladder show a clean 2.000.

### F.2 Space (CN, `N = 512` in 2D, `N = 4096` in 1D; temporal error negligible)

| ladder | `‖e(T)‖_L2` (nx = 2, 4, 8, 16[, 32]) | orders |
|---|---|---|
| 1D `v=0`, sin | 9.03e-3, 1.15e-3, 1.44e-4, 1.81e-5, 2.26e-6 | 2.98, 2.99, 3.00, **3.00** |
| 1D `vx=2`, **sin** | 1.03e-2, 1.69e-3, 2.90e-4, 4.96e-5, 8.55e-6 | 2.62, 2.54, 2.55, **2.54** |
| 1D `vx=2`, **sin³** | 9.00e-2, 9.16e-3, 1.03e-3, 1.24e-4, 1.54e-5 | 3.30, 3.16, 3.05, **3.01** |
| 1D `v=0`, sin³ | 8.12e-2, 9.19e-3, 1.03e-3, 1.25e-4, 1.54e-5 | 3.14, 3.16, 3.05, **3.01** |
| 2D `v=0`, sin | 6.85e-3, 8.29e-4, 1.03e-4, 1.28e-5 | 3.05, 3.01, **3.00** |
| 2D `v=(2,1)`, **sin** | 8.60e-3, 1.57e-3, 2.99e-4, 5.38e-5 | 2.46, 2.39, **2.47** |
| 2D `v=(2,1)`, **sin³** | 7.00e-2, 5.03e-3, 5.70e-4, 6.94e-5 | 3.80, 3.14, **3.04** |
| 2D `v=0`, sin³ | 7.09e-2, 5.07e-3, 5.72e-4, 6.95e-5 | 3.81, 3.15, **3.04** |

**Q2 → third order in L², with convection, once the data are consistent.**
The `sin` shape with `v ≠ 0` converges at ≈ 2.5. Diagnosis (not assumed —
isolated by the `sin³` shape):

* `sin` with `v = 0`: `f ∝ s` vanishes on `∂Ω` → order 3.
* `sin` with `v ≠ 0`: `f` contains `g(t) v·k cos(k ξ)`, **nonzero on `∂Ω`** →
  order 2.5.
* `sin³` with `v ≠ 0`: `y`, `∇y`, `Δy` and `f` all vanish on `∂Ω` → order 3.

The only difference is the boundary trace of `f`. ex20's load vector is
`M_elim · I_h f`, and the homogeneous-Dirichlet elimination zeroes the
constrained **columns** of `M`, so the boundary nodal values of `f` are
dropped from the interior rows next to the boundary — an O(h) consistency
error in a one-element band, i.e. `h^{5/2}` in L². **This is an inherited
data-assembly convention of ex20, not a property of the convection operator**;
it is not changed in ex21 (the change would alter the default path for any
Dirichlet-state case whose source is nonzero on `∂Ω`, and break ex20 identity
there; whether any of the 61 goldens is such a case was not checked). See K.

## G. Gate summary

| gate | result |
|---|---|
| G1 orientation / sign of `C` | ✔ (≤ 1e-12 vs O(10) for the wrong orientation) |
| G1 skew of `C` on free DOFs | ✔ (≤ 2.4e-14) |
| G2 shell transpose | ✔ (≤ 3e-17) |
| G2 shell = assembled, assembled `Eᵀ` = `Eᵀ` | ✔ (≤ 2.1e-16) |
| G2 reduced Hessian symmetric (production adjoint) | ✔ (≤ 1.3e-13) |
| G3 FD reduced gradient (production adjoint) | ✔ (≤ 4.9e-14) |
| G4 production solve consistent with the reduced gradient | ✔ (except 1D CN SCHUR_MATCHING: solver failure, H) |
| negative controls (ex20-style adjoint) | fail as they must (12–30 % gradient error) |
| MMS time order BE / CN | ✔ 0.98–0.995 / 2.000 |
| MMS space order (consistent data) | ✔ 3.01 (1D), 3.04 (2D) |
| AAO residual of the MMS march | ✔ ≤ 4.4e-12 |

## H. Solver note

1D CN + SCHUR_MATCHING fails (KSP reason −11, NaN) — **identically with
`--convection off`**: the GMRES+GAMG inner solves of the matching factors stall
on 1D Crank–Nicolson space-time matrices; LU in either factor restores
convergence. Pre-existing, not a symmetry issue; see
`EX21_SOLVER_NONSYMMETRY_AUDIT.md`. Use SCHUR_SWEEP (2 outer its) or
ADDITIVE_GAMG in 1D.

## K. Findings inherited from ex20

1. **CN + homogeneous Dirichlet state BC with `y₀ ≠ 0` on `∂Ω`** — the
   eliminated rows read `½y_{n,G} + ½y_{n−1,G} = rhs_G`, so the constrained
   values became `(−1)^n y_{0,G}` (measured: `−1, +1, −1, …` at both ends of a
   1D run with `y₀ ≡ 1`; BE gives 0). **Fixed in ex21** (gated on CN + Dirichlet
   state): `y₀` is restricted to zero trace before the CN rhs; after the fix the
   trace is 0 at every step (`logs/cn_dirichlet_initial_trace_fix.log`). In
   ex20 this combination is unreachable from the CLI (only BE benchmarks set the
   Dirichlet state BC), so ex20 is left as is. Relevant to Mathematics 2025
   Example 1, whose `y₀ = 2x(1−x)` is −4 at the Dirichlet end `x = 2`.
2. **Load vector drops the boundary values of `f` under the Dirichlet
   elimination** (§F.2): order 2.5 instead of 3 whenever `f ≠ 0` on `∂Ω`. Not
   changed. A fix would assemble the load from the uneliminated mass (zeroing
   constrained rows only); it would make ex21's default path differ from ex20
   for such data, so it is a decision for review. It does not affect the Mathematics 2025 plants of Examples 1 and
   3 (`f = 0`; disturbance and control regions are interior).
