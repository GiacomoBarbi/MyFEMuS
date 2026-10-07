# T8D — matrix-free Dirichlet-reduced Woodbury core (ex21, 2D)

*Goal: keep the T7D Dirichlet-reduced formulation and its robustness, and remove the
O((N_t |C|)²) memory of the dense Woodbury core. Nothing in the mathematics changed
(B_D, B_Dᵀ, BE/CN, H1-space and H1-time are untouched). The dense core stays the
default and the reference. No 3D work. Nothing committed, staged or pushed.*

All scripts are in `t8d/scripts/`, logs in `t8d/logs/` and results/figures in
`t8d/results/`. The CSVs are `t8d_core_action.csv`, `t8d_core_iterations.csv`,
`t8d_memory.csv`, `t8d_solution_equivalence.csv` and `t8d_mpi.csv`. The JSON field
dumps used for the comparisons are kept outside the repository (scratchpad).

## 0. Architecture audit

| question | answer (verified in source) |
|---|---|
| 1. Is the dense core shared between ex20 and ex21? | **No: it is copied.** `ex21/include/{DirichletReducedSchur,DirichletControl}.{hpp,cpp}` are forks of the ex20 files. Before T8D they differed from ex20 only by the T7D convection changes (114 changed lines; `t8d/results/ex20_vs_ex21_pre_t8d.diff`). ex21 builds from its own `include/` only (CMakeLists `target_include_directories(... ex21/include)`). **ex20 was not touched.** |
| 2. Which parts of the memory problem are identical? | All of it. Both codes hold **three** dense objects, not one: (i) the responses Y_k (N_t × n_local × \|C\| doubles, distributed by rows), (ii) the Gram blocks Γ_ab (N_t(N_t+1)/2 × \|C\|², **replicated**), (iii) the Cholesky factor of K ((N_t\|C\|)², **replicated**). For γ_u > 0, `DirichletControlOperators` holds a **fourth**: a dense replicated Cholesky of R_H1t ((N_t\|C\|)²), used every outer iteration through N_D. |
| 3. What differs only because ex21 needs A and Aᵀ? | Under convection, ex21 factors A_I by LU and builds Aᵀ_I with its own LU (backward sweep), plus the transposed CN coupling. That doubles the *spatial* factor memory, O(n), and is unrelated to the O(m²) core. Heat runs in ex21 are bitwise ex20 (§15). |
| 4. What can be generic without touching the cluster ex20 branch? | The matrix-free core and the sparse R_H1t touch only `DirichletReducedSchurOperators` (iterative branch, verification), `DirichletControlOperators` (reduced layout, sparse R_H1t, the sparse branch of `SolveR`) and one `ProblemConfig` struct plus CLI. ex20's parallel-in-time work lives elsewhere. The port is a file-local patch (§15). |

## 1. Current reduced algebra (from source)

The T7D KKT system, with the ex20 dt-scaled multiplier:

    E_I Y + η B_D U = F
    E_Iᵀ P = α_t (Ŵ_y ⊗ M_ω)(Y − Y_d)
    Q_u U − η B_Dᵀ P = (γ_u/dt) q₀,    Q_u = α_u R_H1t,
    R_H1t = Ŵ_u ⊗ R_Γ + (ρ/dt) T ⊗ M_Γ,  R_Γ = M_Γ + ℓ²K_Γ,  ρ = γ_u/(α_u dt)
    (`ControlTimeRatio`), T = tridiag(−1, 2, −1) with T_NN = 1.

Eliminating U gives the reduced KKT system used by the solver:

    [ E_I              (η²/α_u) N_D ] [y]   with N_D = B_D R_H1t⁻¹ B_Dᵀ
    [ −α_t Ŵ_y M_ω     E_Iᵀ         ] [p]

Its Schur complement is S_D = E_Iᵀ + β Ŵ_y M_ω E_I⁻¹ N_D, with
**β = α_t η²/α_u** (`Solver.cpp`: `Prepare(alpha_track·eta²/alpha_u)`).

Woodbury with A = E_Iᵀ, U = β(Ŵ_y⊗M_ω)E_I⁻¹B_D, C = R_H1t⁻¹, V = B_Dᵀ gives the
**exact** inverse implemented by `DirichletReducedSchurOperators::Solve`:

    S_D⁻¹ b = w − β E_I⁻ᵀ (Ŵ_y⊗M_ω) E_I⁻¹ B_D K_c⁻¹ B_Dᵀ w,      w = E_I⁻ᵀ b
    K_c = R_H1t + β H,   H = B_Dᵀ E_I⁻ᵀ (Ŵ_y⊗M_ω) E_I⁻¹ B_D = Zᵀ (Ŵ_y⊗M_ω) Z,   Z = E_I⁻¹ B_D.

**What the dense code assembles** (`AssembleDenseK`, formerly inline in `Prepare`):

- The responses Y_k = slot-k free-state response to the unit control of slot 1 at
  face DOF c (E_I lower block-Toeplitz). The BE recursion is
  y₁ = A⁻¹g_c, y₂ = A⁻¹((M_I/dt)y₁ − h_c), y_k = A⁻¹(M_I/dt)y_{k−1}; CN uses (A₊, A₋).
- Γ_ab = Y_aᵀ M_ω Y_b.
- H_{m,m'} = Σ_j ŵ_y(j+m') Γ_{j+d,j}, d = m' − m.
- K = R_H1t + βH, where the R_H1t term is Ŵ_u(m)·R_Γ on the d = 0 blocks, plus
  (ρ/dt)·T_mm·M_Γ (d = 0) and −(ρ/dt)·M_Γ (d = 1).
- Then a dense `potrf`.

## 2. K_c is SPD: proof and numbers

**Proof.**

1. **H = Zᵀ D Z** with D = Ŵ_y ⊗ M_ω symmetric positive semidefinite. Ŵ_y is a positive
   diagonal and M_ω is the tracking mass (PSD, zero off ω). Hence H is symmetric PSD for
   **any** E_I, symmetric or not: the Eᵀ inside H is exactly the transpose of the E in Z
   (T7D: transposed backward sweep, exact B_Dᵀ). Convection cannot break this.
2. **R_H1t is SPD.**
   - Ŵ_u ⊗ R_Γ is SPD: Ŵ_u is a positive diagonal, and R_Γ = M_Γ + ℓ²K_Γ is SPD on C
     (M_Γ SPD on its support, K_Γ PSD).
   - T = DᵀD (first differences with u₀ as data) is SPD, so T ⊗ M_Γ is SPD and
     ρ/dt ≥ 0.
3. **Conclusion.** With β > 0, K_c = R_H1t + βH is SPD. Moreover the spectrum of
   R_H1t⁻¹K_c lies in **[1, 1 + β λ_max(R_H1t⁻¹H)]**. That makes CG legitimate, makes
   R_H1t the natural preconditioner, and predicts that only β ∝ 1/α_u (not h, N_t, ℓ
   or γ_u) can drive the iteration count.

**Numbers** (`--verify-dirichlet-core`, `t8d_core_action.csv`; 32 cases = heat/convection
× BE/CN × L2 / H1-space ℓ = 0.3 / H1-time γ_u = 10⁻³ / combined × (nx, N) = (8, 16),
(16, 32); m = 272 and 1056):

| quantity | range over the 32 cases |
|---|---|
| ‖K − Kᵀ‖_F / ‖K‖_F | **0** (exact) |
| λ_min(K) | 3.4e-3 – 0.10 (> 0) |
| cond(K) | 7.6 – 1.0e3 |
| min random Rayleigh quotient | > 0 everywhere |
| **cond(R_H1t⁻¹K)** | **1.20 – 3.40** (λ_min = 1.000 every time, as the proof predicts) |

## 3. Matrix-free K_c action (implementation)

**`ProblemConfig::DirichletCoreConfig`** holds:

- solver DENSE (default) or ITERATIVE;
- pc RT (default) or NONE;
- rtol (default 1e-10) and maxit (2000);
- verify.

**CLI:** `--dirichlet-core-solver dense|iterative`, `--dirichlet-core-pc rt|none`,
`--dirichlet-core-rtol`, `--dirichlet-core-maxit`, `--verify-dirichlet-core`.
Validate refuses these options without Dirichlet control, and refuses non-positive
tolerances.

**Distributed reduced layout** (`DirichletControlOperators`):

- A reduced vector holds the C entries of a space-time vector: local size
  N_t·|C_owned|, ordered [slot][owned C DOF].
- `GatherReduced` and `ScatterReduced` are **purely local copies**.
- Nothing of size N_t|C| is replicated. The only replicated data are the |C|-long
  position maps.

**Shell action** (`DirichletReducedSchurOperators::ApplyCore`, a MatShell `MATOP_MULT`):

    v → Q_C v → B_D → forward sweep E_I⁻¹ → (Ŵ_y ⊗ M_ω) → backward sweep E_I⁻ᵀ → B_Dᵀ → gather
      → ×β → + R_H1t v (sparse MatMult)

It reuses the T7D sweeps and their spatial factors: LU of A_I and of A_Iᵀ under
convection, Cholesky for heat. No factorisation is rebuilt per MatMult.

**Core solve:** CG (`KSPCG`, prefix `dirichlet_core_`, unpreconditioned-norm stopping
test), called inside `Solve` in place of the dense `potrs`.

**Iterative mode skips** the responses, the Gram blocks and the dense K entirely. For
γ_u > 0 it also skips the dense R_H1t Cholesky: `SolveR` uses the sparse factor of §6.

**Verification mode** (dense mode + `--verify-dirichlet-core`) keeps an unfactored copy
of K and prints:

- `EX21_T8D_CORE_SPD` (symmetry, eigenvalues, the pencil (K, R_H1t), random Rayleigh
  quotients);
- `EX21_T8D_CORE_ACTION` (dense vs shell on 9 structured vectors);
- `EX21_T8D_CORE_SHELL_SYMMETRY`;
- `EX21_T8D_CORE_RT_SPARSE_VS_DENSE`.

## 4. Gold matrix-action test (`t8d_core_action.csv`, all 32 cases)

The nine test vectors are: random ×2, constant, temporal low frequency sin(πt),
temporal high frequency (−1)ⁿ, smooth and oscillatory in the C DOF index, slot-1-only,
and last-slot-only. The boundary "smooth/oscillatory" vectors use the sorted C-index as
coordinate, not the geometric arc length.

| check | worst case |
|---|---|
| ‖K_dense v − K_shell v‖ / ‖K_dense v‖ | **3.6e-14** (most cases 1e-15) |
| shell symmetry \|⟨Kx, y⟩ − ⟨x, Ky⟩\| / (‖Kx‖‖y‖) | 2.1e-16 |
| sparse R_H1t vs dense R block | 1.5e-16 |

**PASS** for heat and convection, BE and CN, and all four regularisations.

## 5–6. Iterative core and the R_H1t preconditioner (P0)

**P0 = R_H1t, exact.**

- R_H1t = Ŵ_u⊗R_Γ + (ρ/dt)T⊗M_Γ is **assembled sparse** (MPIAIJ) on the reduced
  layout. Each row has ≤ 15 nonzeros: the 1D Q2 face stencil × 3 time levels.
- It is factored **once** by the existing direct-solver factory: CHOLMOD Cholesky on one
  rank, MUMPS Cholesky on several.
- Without H1-time it is block diagonal in time (one R_Γ block per slot, ½ on the CN
  last slot); with H1-time it is block tridiagonal. A sparse direct factor is exact in
  both cases. Fill is negligible (N_t|C| unknowns, a banded 1D-in-space ⊗ tridiagonal-
  in-time pattern).
- Fast diagonalisation or nested KSP were not needed.
- The same factor replaces the dense (N_t|C|)² Cholesky of R_H1t in `SolveR` (H1-time).

**Unpreconditioned CG** (diagnostic) is already bounded in h and N_t: 21–29 iterations at
α_u = 10⁻². It grows as α_u decreases: 120 (heat) and 152 (convection) at
α_u = 10⁻⁵.

**P1** (§7 of the brief) was **not needed**; see §12.

## 9. Solution equivalence (`t8d_solution_equivalence.csv`; production path, nx = 16, N = 32, rtol 1e-10)

16 cases: heat/convection × BE/CN × L2 / H1-space / H1-time / combined.

| quantity | dense vs iterative |
|---|---|
| y, p, u | ≤ 1.8e-12 (≤ 4e-14 in 15/16; the exception is a 1-outer-iteration CN run whose true residual is 5.7e-11 in **both** modes) |
| J_track, J_reg | ≤ 1.2e-13 |
| true KKT residual | equal (to 2 digits) |
| outer iterations | identical (1–2) |
| core CG iterations per solve | 6–10 |

**T7D gates with the iterative core** (`t8d/logs/gates/`, the full T7D battery: 18 runs,
heat + 4 winds, BE/CN, L2 / H1-space / H1-time / combined):

- Every non-production gate line (G1, G2, G3D, R, G5D, G4D including the legacy and
  untransposed-E negative controls) is **identical** to the T7D logs.
- The production gate G6D gives:
  - true residual ≤ 3.6e-11;
  - ‖∇J(u*)‖ ≤ 3.7e-12;
  - forward march ≤ 1.0e-10;
  - stationarity ≤ 6e-18;
  - the T7D outer counts.

  Core CG takes 3–7 iterations.

## 10. Memory scaling (`t8d_memory.csv`, `results/memory_vs_core_size.png`)

Setup: production path on the T7D exact data (`--exact-oc dirichlet-conv --solve-only`),
CN, serial, one run at a time.

| nx | N_t | m = N_t\|C\| | dense MaxRSS | iterative MaxRSS | ratio |
|---|---|---|---|---|---|
| 16 | 32 / 64 / 128 / 256 | 1056 … 8448 | 108 / 199 / 454 / 1271 MB | 83 / 124 / 207 / 372 MB | 1.3 – 3.4 |
| 32 | 32 / 64 / 128 / 256 | 2080 … 16640 | 349 / 727 / 1781 / **5077** MB | 228 / 389 / 711 / **1356** MB | 1.5 – 3.7 |
| 48 | 32 / 64 / 128 / 256 | 3104 … 24832 | 805 / 1721 / 4213 / **11823** MB | 469 / 829 / 1551 / **2972** MB | 1.7 – 4.0 |
| 48 | 512 | 49 664 | not run (≈ 37 GB predicted) | **5.8 GB** | |
| 64 | 256 / 512 | 33 024 / 66 048 | not run (≈ 21 / 67 GB) | **5.2 / 10.2 GB** | |
| 96 | 256 | 49 408 | not run (≈ 52 GB) | **11.6 GB** | |
| 48 | 1024 | 99 328 | not run (≈ 128 GB) | **11.5 GB** | |

**What exactly was removed.** The dense − iterative excess at the 12 common points obeys

    excess = 12.03 bytes · m²  +  8.00 bytes · N_t n |C|      (least squares; model/measured = 1.00 – 1.16)

That is exactly 8m² for K, about 4m² for the Gram blocks (8·N_t(N_t+1)/2·|C|²) and
8·N_t·n·|C| for the responses. The "not run" predictions above use this validated model.
No dense run was forced into OOM.

**What remains in iterative mode:** MaxRSS = 68 MB + 1.20e-3 MB × (N_t n), i.e. about
157 doubles per space-time DOF. The log–log slope in m tends to 1 (0.79 → 0.95 as the
size grows). This is the all-at-once data side, the KKT vectors, the inner sweep and
Krylov workspaces: about 247 PETSc vectors at the peak (`-memory_view`). It is
**independent of the outer FGMRES restart** (restart 30 / 5 / 2 give the same MaxRSS,
since there is 1 outer iteration).

**The O(m²) term is gone; memory is O(N_t n).**

## 11. Time to solution (`t8d_memory.csv`, `results/dense_vs_matrixfree_time.png`)

Setup: serial; t_pcsetup contains the dense responses, Gram blocks and Cholesky.

| case | dense: t_pcsetup / t_ksp / total | iterative: t_pcsetup / t_ksp (of which core CG) / total |
|---|---|---|
| nx = 16, N = 32 | 0.14 / 0.12 / 0.6 s | 0.02 / 0.20 (0.09) / 0.6 s |
| nx = 32, N = 128 | 16.7 / 2.5 / 24.3 s | 0.13 / 3.9 (1.6) / 8.8 s |
| nx = 48, N = 256 | **323** / 10.7 / **361** s | 0.34 / 17.6 (8.0) / **39** s |
| nx = 48, N = 1024 (dense impossible) | — | 2.3 / 64 (32) / 146 s |

- The dense setup grows like N_t|C| spatial solves plus the O(N_t²|C|²n) Gram BLAS plus
  the O(m³) Cholesky. At nx = 48, N = 256 that is 90 % of the run.
- The iterative path adds about 13 CG iterations, each costing 2 sweeps, to the outer
  solve.
- **There is no crossover in favour of dense in the tested range.** The two are equal at
  the smallest case (0.6 s) and the iterative path is faster everywhere else (up to 9×).

## 12. Core iteration scaling (`t8d_core_iterations.csv`, `results/core_iterations_vs_nt.png`)

Max CG iterations per core solve at rtol 1e-10; base case nx = 16, N = 32, α_u = 10⁻²,
CN, data sin_x_minus_t:

| sweep | heat, PC = R_H1t | convection, PC = R_H1t | heat, no PC | convection, no PC |
|---|---|---|---|---|
| nx = 8, 16, 32, 64 | **7, 7, 7, 7** | **9, 9, 9, 9** | 26, 26, 26, 25 | 27, 29, 29, 27 |
| N_t = 16 … 256 | 7, 7, 7, 8, 7 | 9, 9, 9, 9, 9 | 27 → 21 | 29 → 24 |
| α_u = 10⁻¹ … 10⁻⁵ | 5, 7, 13, 26, 58 | 6, 9, 16, 34, 74 | 24 … 120 | 25 … 152 |
| ℓ = 0, 0.1, 0.3, 1, 3 | 7, 7, 7, 7, 7 | 9, 9, 8, 8, 8 | | |
| γ_u = 0, 10⁻⁴, 10⁻³, 10⁻², 10⁻¹ | 7, 7, 6, 5, 4 | 9, 8, 7, 5, 4 | | |
| BE (base) | 7 | 9 | | |

These are **mesh-, N_t-, ℓ- and γ_u-independent**, as §2 predicts. The only growth is in
1/α_u: roughly α_u^(−0.3), from κ ≤ 1 + β λ_max(R⁻¹H) with β ∝ 1/α_u. Outer iterations
stay at 1–2 with no core failures. A stronger P1 is not required for the intended
regime; for α_u ≲ 10⁻⁵ it would be the next lever (§18).

## 13. Outer KKT interaction (tolerance ladder, `t8d/results/tolerance.csv`)

Setup: convection, CN, nx = 32, N = 64; L2 and combined H1; α_u = 10⁻² and 10⁻⁴.

| core rtol | outer its (dense: 1–2) | true residual | fields vs dense |
|---|---|---|---|
| 1e-2 | 2 – 6 | ≤ 6e-12 | ≤ 1.5e-9 |
| 1e-4 | 2 – 3 | ≤ 3e-14 | ≤ 1e-11 |
| 1e-6 | 2 (= dense) | ≤ 1.8e-12 | ≤ 6e-10 |
| 1e-8 | 2 (= dense) | ≤ 2e-14 | ≤ 1.4e-13 |
| 1e-10, 1e-12 | 2 (= dense) | ≤ 2e-14 | ≤ 1.4e-13 |

The outer FGMRES is flexible, so an inexact core costs extra outer iterations, never
robustness. **Recommended core rtol: 1e-8.** It is the loosest value whose outer counts
and solution are indistinguishable from dense in every case; 1e-6 already matches the
outer counts. The code default stays 1e-10 (the conservative setting used throughout
this study); nothing was changed silently.

## 14. MPI (`t8d_mpi.csv`; production path, exact data, CN)

**Code audit:**

- Reduced vectors are distributed (local N_t|C_owned|); gather/scatter are local.
- In iterative mode no N_t|C| object is replicated: `reduced_a_/b_` are not
  allocated, and neither are the dense K, the Gram blocks or the responses.
- The H1-time branch returns before the dense `Q_dense_`.
- What remains replicated is O(|C|) (C lists, position maps) and, for γ_u > 0, the
  dense **|C|²** face blocks R_Γ and M_Γ (e.g. 129² at nx = 64).
- All-gathers: |C| integers and the rank counts, once. CG uses scalar all-reduces.
- Sweeps: distributed spatial LU (MUMPS), sequential in time.
- R_H1t: MPIAIJ + MUMPS Cholesky.
- Only the diagnostic `--verify-dirichlet-core` builds dense replicated objects.

| case | np | max field Δ vs np = 1 | J rel Δ | CG its | outer | MaxRSS per rank | total |
|---|---|---|---|---|---|---|---|
| iterative nx = 32, N = 128 | 1 / 2 / 4 | 0 / 7.5e-11 / 9.1e-11 | ≤ 8.8e-12 | 13 / 13 / 13 | 1 | 712 / 411 / 233 MB | 712 / 808 / 894 MB |
| dense nx = 32, N = 128 | 1 / 2 / 4 | 0 / 7.4e-11 / 9.3e-11 | ≤ 8.8e-12 | — | 1 | 1782 / 1351 / 1114 MB | 1782 / 2685 / **4392** MB |
| iterative nx = 48, N = 256 | 1 / 2 / 4 | 0 / 1.8e-10 / 1.6e-10 | ≤ 1.2e-11 | 13 / 13 / 13 | 1 | 2973 / 1629 / **862** MB | 2973 / 3211 / 3305 MB |

- The cross-np field differences are identical for dense and iterative: they are the
  outer tolerance (true residual about 3e-10), not the core.
- Iterative per-rank memory divides by np. Dense per-rank memory floors at the
  replicated K + Gram (≈ 0.8 GB here); its total grows ×2.5 from np = 1 to np = 4.
- Wall time does **not** improve with np in this range: the sweeps are sequential in
  time and these spatial problems are small. **No parallel-scalability claim.**

## 15. ex20 heat relevance and porting note

**Reproduction** (`t8d/results/ex20_heat_comparison.csv`; identical CLI, ex20 defaults,
heat, nx = 16, N = 32; L2 / H1-space / H1-time / combined × BE/CN):

- ex21 **dense** = ex20 **bitwise**, all 8 cases.
- ex21 **iterative** vs ex20: fields ≤ 2.4e-14, J ≤ 1.3e-15, identical outer
  iterations, 8–12 CG iterations.

1. **The matrix-free algorithm reproduces the ex20 heat algebra exactly.** With
   convection off, B_Dᵀ and the sweeps are ex20's.
2. **It can replace the ex20 dense core.** The heat case is the special case (Cholesky
   instead of LU spatial factors, A = Aᵀ). The same SPD proof and the same R_H1t
   preconditioner apply.
3. **Porting to the cluster ex20 branch after its parallel-in-time merge.** The change is
   self-contained:
   - **`DirichletControl.{hpp,cpp}`:** add the reduced-layout API
     (`CreateReducedVector`, `GatherReduced`, `ScatterReduced`), `ReducedRegularization`
     + `SolveReducedR` (sparse R_H1t, about 120 lines), and the `sparse_core_` branches
     in the constructor, `BuildTemporalRegularization` and `SolveR`.
   - **`DirichletReducedSchur.{hpp,cpp}`:**
     - the `iterative_` / `verify_` members and the core statistics;
     - `AssembleDenseK` split out of `Prepare` (the dense path is numerically
       identical, verified bitwise);
     - the iterative branch of `Solve`;
     - `ApplyCore`, `BuildCoreSolver`, `VerifyCore`.
   - **`Config.hpp`:** the `DirichletCoreConfig` struct and its Validate check.
   - **`CommandLine.cpp`:** the five options.
   - **`Solver.cpp`:** VerifyCore after Prepare, and the stats line.

   **Dependence on the parallel-in-time work.** The core only needs the forward and
   backward space-time sweep applications (`RunTimeSweep` on the two contexts) and
   B_D/B_Dᵀ. If the parallel-in-time merge changes how a space-time solve E_I⁻¹ / E_I⁻ᵀ
   is applied (e.g. a parareal or block-parallel sweep), only the two `RunTimeSweep` calls
   in `ApplyCore`/`Solve` need to point at it. They must stay **exact transposes of each
   other** to round-off: CG and the SPD proof need ⟨E⁻¹x, y⟩ = ⟨x, E⁻ᵀy⟩. An inexact
   (iterative) parallel-in-time sweep would make the shell slightly nonsymmetric: use
   flexible CG or tighten the sweep tolerance.

   The reduced layout assumes ex20's rank-major space-time layout (each rank owns all
   time levels of its spatial DOFs). If the cluster branch distributes **time** across
   ranks, `GatherReduced`/`ScatterReduced` and the global numbering in
   `BuildReducedRegularization` must follow that layout. That is the one non-trivial
   adaptation.

   The convection-specific Aᵀ/LU pieces are not needed in ex20.

## 16. Regression (`t8d/scripts/t8d_regress.sh`, `t8d/logs/regress.out`; OLD = the T7F final binary sha 6278531a)

The new binary is sha a2e6f680. The **dense mode is the default and is unchanged**:

| block | result |
|---|---|
| ex20 goldens | **61/61 identical** |
| Math2025 Ex3 / Ex4 / Ex5 | byte-identical |
| EF 9.5 | only the archived `variant=source` label line differs: pre-existing since T7N, not a T8D change |
| T0–T2 constant wind | PASS |
| T3/T3b/T4 representative | identical (9), including the expected failures |
| old vs new, every `EX20_/EX21_` line | **49/49 IDENTICAL** |

The 49 old-vs-new comparisons cover:

- Dirichlet heat with all four regularisations (BE/CN), and the larger nx = 32 dense-core
  cases (convection H1-space, convection H1-time, heat combined);
- the T7D `dirichlet-conv` exact (L2 and H1-space), Dirichlet + convection production
  H1-space + H1-time, and the T7D gates;
- Neumann heat (3 PCs) and T7N (exact and production);
- T7F `natural-conv` exact, and multi-zone + two sources production;
- the feature-audit multi-zone smoke runs, `conv-sin3` / `lift-exp` exact, T6 LPS, and the
  distributed and Neumann natural-BC gates.

**Restrictions:** all previous refusals are unchanged. New explicit refusals:
`--dirichlet-core-solver iterative` or `--verify-dirichlet-core` without Dirichlet
control, an unknown core solver, and core rtol ≤ 0.

## 17. Success criteria

| # | criterion | result |
|---|---|---|
| 1 | matrix-free K_c = dense K_c | ✔ ≤ 3.6e-14 (32 cases) |
| 2 | core SPD | ✔ proof + numbers (exact symmetry, λ_min > 0, κ(R⁻¹K) 1.2–3.4) |
| 3 | iterative solution = dense | ✔ fields ≤ 1.8e-12, J ≤ 1.2e-13 (16 cases) |
| 4 | T7D gates | ✔ all 18; non-production lines identical |
| 5 | no O((N_t\|C\|)²) | ✔ excess = 12.03 B·m² removed exactly; iterative MaxRSS ∝ N_t n |
| 6 | MPI does not replicate a dense core | ✔ code audit + per-rank MaxRSS ∝ 1/np |
| 7 | inner iterations acceptable | ✔ 7–9, independent of h, N_t, ℓ, γ_u; grows only as α_u^(−0.3) |
| 8 | outer robustness | ✔ outer counts identical at core rtol ≤ 1e-6 |
| 9 | heat and convection | ✔ |
| 10 | all four regularisations | ✔ |

## 18. Decision

**CASE A: matrix-free core + R_H1t preconditioner is robust and removes the memory
bottleneck.**

- It is also **faster** than the dense core at every tested size (no crossover;
  9× at nx = 48, N = 256).
- It enables problems the dense core cannot run, up to m ≈ 10⁵ (nx = 48, N = 1024) in
  11.5 GB.
- Practical payoff: the T7D exact-OC ladder now runs at **N_t = 512**. At nx = 32 / 48 /
  64: y = 6.6e-5 / 1.45e-5 / 5.4e-6, p = 1.1e-5 / 1.8e-6 / 6.9e-7,
  u = 1.8e-4 / 3.8e-5 / 1.5e-5. That gives spatial orders 3.1–3.4 between nx = 48 and
  64, past the T7D memory wall (`t8d/logs/exact/`).

**Recommended production strategy (not applied; the default is unchanged).**

- Make `--dirichlet-core-solver iterative --dirichlet-core-pc rt --dirichlet-core-rtol
  1e-8` the Dirichlet default once reviewed.
- Keep `dense` as the reference and small-problem option, and keep
  `--verify-dirichlet-core` as the gold test.
- The only regime where the core iteration count grows noticeably is very small α_u
  (74 iterations at 10⁻⁵ with convection). A low-rank or coarse-temporal correction of
  βH would be the P1 to try there; it was not needed in the validated range.

**Remaining blockers before 3D (Dirichlet path):**

- **None from the core.**
- The remaining memory is the all-at-once space-time storage itself (about 157 vectors
  of size N_t n). That is the true 3D concern: it scales with N_t·n_3D, not with the
  boundary. It is a property of the all-at-once formulation, not of the Dirichlet solver.
- The sequential-in-time sweeps limit parallel speed-up. That was not a T8D goal.

## Files changed

- **`ex21/include/Config.hpp`:** `DirichletCoreConfig` and its Validate check.
- **`CommandLine.cpp`:** the 5 options.
- **`DirichletControl.{hpp,cpp}`:**
  - the reduced layout API and the sparse R_H1t;
  - the `sparse_core_` branches;
  - the destructor.
- **`DirichletReducedSchur.{hpp,cpp}`:**
  - `AssembleDenseK` split out of `Prepare`;
  - the iterative branch of the constructor and of `Solve`;
  - `ApplyCore`, `BuildCoreSolver`, `VerifyCore` and the statistics.
- **`Solver.cpp`:** VerifyCore after Prepare, and the `EX21_T8D_CORE_STATS` line
  (iterative only).

**ex20: untouched.**

---

## Finalisation note (appended after T8D; the report above is historical and unchanged)

- **New ex21 default:** `--dirichlet-core-solver iterative --dirichlet-core-pc rt
  --dirichlet-core-rtol 1e-8` (§13 recommendation, applied).
- **Dense** remains fully available as the reference/gold solver
  (`--dirichlet-core-solver dense`), and reproduces the old default byte-for-byte.
  `--verify-dirichlet-core` now selects the dense reference by itself; combined with an
  explicit `iterative` it is refused.
- **Zero-velocity parity with ex20:**
  - With `--convection off`, ex21 is byte-identical to ex20 for every shared formulation
    and solver; Dirichlet requires the dense core for this, and the default iterative
    core agrees to ≤ 2.3e-14.
  - With `--convection on` and w = 0, the matrices are byte-identical. The solutions agree
    to ≤ 4e-15, but the flag-selected transposed / LU paths are not bitwise, and the
    Neumann boundary-reduced Schur is refused.
- **Scope of solver-default differences:** only the Dirichlet core (dense in ex20,
  iterative in ex21). Distributed and Neumann defaults are unchanged and identical to
  ex20.
- **Details:** `T8D_FINALISATION.md` and `ex20_ex21_zero_velocity_parity.csv`.
