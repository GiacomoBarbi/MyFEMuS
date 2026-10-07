# T6 — adjoint-consistent Local Projection Stabilisation (LPS) for ex21

Scope: discretisation only.

- **Unchanged:** the KKT formulation, the matching Schur preconditioner, outer
  iterations, ex20 and the Aulisa/Gilliam report.
- **Defaults unchanged:** LPS is **off by default** (`--convection-stabilization none`).
- **Solver:** all primary comparisons use **exact LU in all four sweep prefixes**,
  so spatial-solver failure cannot contaminate the stabilisation study.
- **Authoritative inputs:** `LEVEQUE_PEARSON_SOURCE_AUDIT.md`, T4, T5 and T5B.
- **Re-read for this phase:** LP §2.1 (eqs. (7)–(8) and δ_k) was re-read from the
  PDF text layer for this phase (`Numerical Linear Algebra App - 2021 - Leveque - …pdf`, pp. 3–5).

## Summary

| question | answer |
|---|---|
| failure guard | **added, verified.** A solve is accepted only if reason > 0, sweep_failures = 0 and the true residual is ≤ 100·max(rtol‖b‖, atol). The T5b false-convergence case is now rejected (exit 8, `EX21_SOLVE_REJECTED`). Successful runs are unchanged (61/61 goldens) |
| implementation | LP's W (7)–(8), δ_k, Pe_k, h_k and w_k exactly. One Q2 element = one LP 2×2 Q1 patch (the same square on the same node grid). Projection: P0 (LP's patch mean; primary) or P1 (diagnostic) |
| matrix gates | symmetric (exactly 0), PSD (dense eigenvalues ≥ −2e-16·λ_max), W_{−w} = W_w exactly, W·1 = 0, W ≡ 0 below Pe 1, active patches = independent LP count (18/18), MPI-identical |
| KKT gates | transpose, FD gradient, Hessian symmetry, physical objective and production KKT/forward march all pass on the stabilised operator (10 gate runs). The wrong-adjoint negative control fails, as it must |
| order | **P0 lowers Q2 from O(h³) to O(h²)** while active (fixed patch Pe: 2.33 → 2.07). **P1 keeps O(h³).** Time orders (BE 1, CN 2) are preserved |
| oscillations | P0 removes the global odd-even oscillation of under-resolved meshes (nx = 8–16 at ε = 1/500; nx = 8 at ε = 1/100). P1 is too weak to do so |
| side effects | P0 **amplifies the corner undershoots at both data discontinuities** (1, ±1). It can leave a persistent several-% undershoot where Galerkin is clean (ε = 1/500, β = 1e2, nx = 32: −4.7% against −0.3% at t = 1) |
| layer | not smeared: d50/d10 identical to 3–4 digits for nx ≥ 16 |
| T4 all-time undershoot | located. **Early (t ≈ 0.03–0.045) spatial transient at the node next to the discontinuous corner (1, −1).** It is converged in dt and the same under BE and CN, so it is not a time-stepping artefact and not interior convection. It decays slowly with h. LPS does not cure it (P0 worsens it) |
| solver (secondary) | **LPS-P0 makes the production GAMG converge** where unstabilised GAMG fails (blocks: 19–28 iterations against 500; KKT: 2 outer, identical solution). AIR H2c is unchanged |
| verdict | Implementation correct and adjoint-consistent. LP's P0 LPS is a coarse-mesh oscillation suppressor for Q2, with an O(h²) price and corner artefacts. On meshes that resolve the √ε layer (nx ≥ 32 at ε = 1/500), plain Galerkin Q2 is already clean and better |

## 0. Failure guard (step 0)

**Problem (T5b).** A broken spatial sweep returns ≈ 0. The outer FGMRES then sees a zero
*estimated* residual and stops with `KSP_CONVERGED_ATOL` (reason 3), while the true KKT
residual is O(1e-1). The drivers tested only `reason > 0`.

**Fix (smallest possible; no algebra change).**

- `Solver.hpp`: `DistributedOptimalControlSolve` gains `accepted`, `status`, `rhs_norm`
  and `acceptance_tolerance`, plus `kSolveAcceptanceFactor = 100`.
- `Solver.cpp` computes, after the existing true-residual evaluation:
  `status` = `ksp_not_converged` (reason ≤ 0) | `sweep_failures` (> 0) | `true_residual`
  (‖b − Ax‖ > 100·max(rtol‖b‖, atol), with the rtol and atol the outer KSP *actually used*,
  so PETSc option overrides count) | `ok`.
- A rejection is never silent:
  - an `EX21_SOLVE_REJECTED status=… reason=… sweep_failures=… true_residual=… acceptance_tolerance=…`
    line on stdout;
  - `ERROR: ex21 KKT solve REJECTED (…)` on stderr.
- Every driver's exit code now uses `solve.accepted`: `ex21.cpp`, `ExactOcBenchmark` (both
  exits), `Benchmarks` (3), `Math2025Benchmark`, `Benchmark2D`, and `Verification`'s
  reduced-solve check. `EX21_EXACT_SOLVER_STATS` gains `accepted=` and `status=`.
- No other output changes. JSON is untouched, because the ex20 goldens compare it byte-for-byte.

**Negative test** (`t6/guard_test.py`, `results/guard.csv`; LP 5.2, ε = 1/500, nx = 16, CN):

| case | reason | outer | sweep failures | true residual | status | exit | verdict |
|---|---|---|---|---|---|---|---|
| AIR without max_row_sum (T5b H2; hypre binary) — the T5b false convergence | **3** | 1 | 96 | **1.8e-1** | sweep_failures | **8** | PASS (rejected) |
| sweeps starved (Jacobi, ksp_max_it 1) | −11 | 0 | 3104 | nan | ksp_not_converged | 8 | PASS (rejected) |
| production default GAMG (fails at this Pe, T5) | −11 | 0 | 3104 | nan | ksp_not_converged | 8 | PASS (rejected) |
| LU sweeps | 2 | 2 | 0 | 8.6e-15 | ok | 0 | PASS (accepted) |

## 1. The LP formulation, exactly (LP §2.1, eqs. (5), (7), (8))

- Space operator: `L = εK + N + W_w`, with `n_ij = ∫(w·∇φ_j)φ_i` (identical to ex21's `C_ij`).
- **Stabilisation form (7):**
  `w_ij = δ ∫_Ω [w·∇φ_i − π_h(w·∇φ_i)] [w·∇φ_j − π_h(w·∇φ_j)] dΩ`.
  With patchwise δ:
  `vᵀWv = Σ_k δ_k ‖w·∇v − π̃_k‖²_{L2(P_k)} ≥ 0`.
- **Patches:** unions of elements; for Q1, **2 elements in each direction** (2×2).
  π_h may be discontinuous across patch edges.
- **Projection (8):** `π_h(v)|_P = (1/|P|) ∫_P v` — the L2-orthogonal projection onto
  **constants on each patch**.
- **Fluctuation operator:** `κ_h = id − π_h`.
- **Parameter:** `δ_k = h_k/(2‖w_k‖)·(1 − 1/Pe_k)` if `Pe_k > 1`, and **`δ_k = 0` if
  `Pe_k ≤ 1`**, with `Pe_k = ‖w_k‖ h_k/(2ε)`:
  - `‖w_k‖` = ℓ2 norm of the wind **at the patch centroid**;
  - `h_k` = "a measure of the patch length **in the direction of the wind**".
- **Variable wind:** the integrand of (7) contains `w·∇φ` with the actual wind w(x); only
  δ_k uses the centroid value w_k.
- **Symmetry and adjoint consistency:** "with (8) … `W_w = W_wᵀ = W_{−w} =: W`". W is
  quadratic in w, hence even. So the OTD adjoint `εK − N + W_{−w}` equals `Lᵀ`: exactly
  the matrix ex21's DTO adjoint already applies by exact transposition.
- **Theory (LP's references):** L2 order h^{3/2} for Q1 LPS in the convection-dominated regime.

**`h_k` as implemented:** the chord of the patch through its centroid along w_k. For a
rectangle of sides H_x and H_z this is min(H_x/|cos θ|, H_z/|sin θ|), i.e. H/max(|cos θ|, |sin θ|)
for a square. This is the definition already used by `t3/lp52_patch_peclet.py` (T3/T4).
The exact discretisation of "measure … in the direction of the wind" is not given by LP
(audit grade C). This reading reproduces LP's own stated switch exactly (§4, 18/18).

## 2. Q1 source vs Q2 ex21: the extension used

**Geometry.** ex21 runs LP's level l on Q2 with `nx = 2^{l−1}` elements per direction. One
Q2 element of side `2/nx = 2·h_LP` is therefore **exactly one LP patch** (2×2 Q1 elements of
side h_LP), carrying the same 3×3 nodes. So **patch = one Q2 element** reproduces LP's patch
set, centroids, h_k, w_k, Pe_k and δ_k identically. Only the FE basis on the patch changes:
one Q2 instead of four Q1 functions.

**What is projected.** Exactly LP's quantity: the streamline derivative w·∇φ_i of each basis
function, restricted to the patch.

**Reasonable Q2 extensions considered:**

1. **P0 projection on the Q2-element patch** (LP's (8) unchanged). This is the simplest
   source-faithful extension, and the **primary T6 variant, labelled "LPS-P0 (LP)", the
   ex21 Q2 version of LP's LPS**.
2. **P1 (discontinuous, per patch) projection.** This is the projection space of standard
   Q_r LPS theory (P_{r−1}^disc, Braack–Burman, Matthies–Skrzypacz–Tobiska), with the same
   patches and δ_k. Carried as a **labelled diagnostic variant "LPS-P1"**; not LP's operator.
3. 2×2 Q2-element patches (P0 or P1). Rejected: it doubles LP's patch size, so LP's Pe
   switch and δ_k would no longer be reproduced.

**Validity:**

- Any L2 projection computed with the same positive-weight quadrature makes
  `W_k = δ_k Σ_g ω_g (f_g − Πf_g)(f_g − Πf_g)ᵀ` symmetric PSD exactly. Here
  `f_g,i = w(x_g)·∇φ_i(x_g)` and the 4×4 Gauss rule integrates the LP-wind integrand
  (degree 6 per direction) exactly.
- W is even in w for both variants, so `W_{−w} = W_w = W_wᵀ` holds.
- This is **not** a literal reproduction of LP's Q1 discrete operator.

## 3. Implementation

- `Config.hpp`:
  - `ConvectionConfig::Stabilization {NONE, LPS}` (default NONE) and `LpsProjection {P0, P1}`
    (default P0), with documentation;
  - `Validate()` rejects LPS without convection.
- `CommandLine.cpp`:
  - `--convection-stabilization none|lps`, `--lps-projection p0|p1`;
  - `--exact-oc-series <path>` (opt-in diagnostics, below);
  - `Options.hpp` gains `exact_oc_series`.
- `Discretization.hpp/.cpp`:
  - `LpsStats`; `NativeSpatialOperators::W_lps` plus `lps`.
  - **`BuildLpsMatrix(config, mesh, solution_type, wind_sign, stats)`** assembles W
    **as a separate matrix**. It loops over elements (MPI-parallel, owned elements) and
    computes the patch centroid, extents, w_k, h_k, Pe_k and δ_k (skipping the element if
    Pe_k ≤ 1).
  - Per Gauss point: `f_a = w(x_g)·∇φ_a`, plus the patch basis q = {1} (P0) or
    {1, (x−x_c)/H_x, (z−z_c)/H_z} (P1); then `W_k = δ(A − Cᵀ G⁻¹ C)`, symmetrised entrywise.
  - Global statistics come from MPI reductions.
  - In `BuildNativeSpatialOperators`, only when LPS is on:
    - W is assembled and diagnosed (`EX21_LPS` line: active patches, δ range, Pe max, ‖W‖_F,
      symmetry defect, and the ratios to ‖sym(L₀)‖_F and ‖skew(L₀)‖_F);
    - it is then **added to L before the Dirichlet lifting and elimination**, so the lifting
      load −L_FG g and the eliminated operator both carry it;
    - W itself is eliminated like M.
  - When LPS is off none of this code runs. With zero active patches nothing is added.
- **Adjoint:** untouched. ex21 keeps applying the exact transpose of the assembled L (`Lᵀ`
  contains Wᵀ = W automatically).
- `ConvectionVerification.cpp` (`--verify-convection`):
  - G1 now tests the Galerkin term alone (LPS forced off there; it was LPS-free by
    construction before);
  - new **G5** = LPS matrix gates.
- `ExactOcBenchmark.cpp`:
  - `--exact-oc-dump` also writes `_W.txt` when LPS is on;
  - `--exact-oc-nu` now also sets ν for the convective manufactured cases (default 0.25
    unchanged; the manufactured data use `c.diffusivity` throughout);
  - lp52 gains `--exact-oc-series <csv>`: per-slot y min/max with locations, plus the y field
    at the slot of the all-time minimum.

## 4. Matrix-level checks (`t6/matrix_checks.py` → `t6_lps_matrix_checks.csv`: **35/35 PASS**)

**A–D on the in-code G5 gate** (`--verify-convection`, 9 configurations: LP wind ε = 1/20 and
1/500, q2-divfree and lifted q2-divfree at ν = 0.01, BE/CN, P0/P1; 64/64 active patches):

| check | result (all 9) |
|---|---|
| A symmetry ‖W−Wᵀ‖_F/‖W‖_F | **0** (exact) |
| B PSD, min over 20 random x of xᵀWx/(‖W‖‖x‖²) | +0.029 … +0.033 |
| C wind reversal ‖W_w − W_{−w}‖_F/‖W_w‖_F, same active set | **0** (exact), active equal |
| W·1 (w·∇1 = 0) | ≤ 1.2e-16 relative |
| D Pe switch: ν raised until max Pe_k ≤ 1 | **0 active, ‖W‖_F = 0 exactly** |

**Independent dense checks from dumps** (LP wind ε = 1/500 nx = 4 and 8, ε = 1/20 nx = 8,
q2-divfree ν = 0.01 nx = 6; P0 and P1):

- L_lps − L_none = W to ≤ 1.8e-16;
- ‖W − Wᵀ‖ = 0;
- eigenvalues of W on the free DOFs ≥ −1.9e-16·λ_max (PSD).

For the LP wind W has exactly **one** zero eigenvalue on the free DOFs. The wind is the curl
of ψ = (1 − x²)(1 − z²), which lies in Q2 and vanishes on ∂Ω, so w·∇ψ ≡ 0 and Wψ = 0.

**D: Pe activation against the independent LP evaluation** (`t4/post_t4.lp_patch` =
`t3/lp52_patch_peclet.py`): **18/18** (ε × nx) cases match. Active patches and max Pe_k
agree (relative ≤ 1e-5), e.g.:

| ε | nx (LP l) | active ex21 = independent | max Pe_k | δ range |
|---|---|---|---|---|
| 1/20 | 16 (5) | **212/256** | 2.335 | 6.8e-3 … 3.9e-2 |
| 1/20 | 32 (6) | **136/1024** | 1.210 | 9.5e-5 … 2.8e-3 |
| 1/20 | 64 (7) | **0/4096** | 0.615 | — |
| 1/20 | 128 (8) | **0/16384** | 0.310 | — |
| 1/100 | 64 (7) | 3720/4096 | 3.075 | 5.5e-4 … 1.2e-2 |
| 1/500 | 32 (6) | 1024/1024 | 30.24 | 1.6e-2 … 2.4e-1 |
| 1/500 | 128 (8) | 16168/16384 | 7.751 | 4.4e-4 … 1.5e-2 |

**Magnitude** (Frobenius, unstabilised L₀ = νK + N):

| ε, nx | ‖W‖ / ‖sym L₀‖ | ‖W‖ / ‖skew L₀‖ |
|---|---|---|
| 1/500, nx 32 | 10.3 | 3.0 |
| 1/100, nx 64 | 0.58 | 1.7 |
| 1/20, nx 32 | 0.023 | 0.17 |

At high Pe, W dominates the diffusion and is of the order of the convection.

**MPI** (`t6/mpi_lps_check.py`; production path, MUMPS sweeps, ε = 1/500, nx = 32):

- np = 1, 2 and 4 give identical global statistics (1024/1024 active, the same δ range,
  ‖W‖_F 5.918944 (P0) / 2.577033 (P1));
- max|Δy| ≤ 3e-14 and max|Δp| ≤ 2e-15 against np = 1;
- J_track agrees to 1e-14 relative; **PASS**.

## 5. Low-Pe zero-operator regression (step 6)

LP 5.2, ε = 1/20, nx = 64 (all 4096 patches Pe ≤ 1, max 0.615), β = 1e-2, CN and BE, LU
sweeps: **LPS on and LPS off are byte-identical.** This covers every `EX21_LP52_RESULT`,
`EX21_EXACT_SOLVER_STATS` and `EX21_EXACT_OBJECTIVE` field (iterations, residuals, ranges,
J_track, J_reg), the t = 1 snapshot CSV and the all-time series CSV. The only difference is
the added `EX21_LPS … active=0 ||W||_F=0` diagnostic line.

## 6. Manufactured consistency (steps 7E, 8) → `t6_lps_mms.csv`

Forward MMS: `--convection-mms sin3`, q2-divfree wind (max|w| 2.3), homogeneous Dirichlet,
T = 1, ν = 0.01, CN Nt = 1024; error is L2(Ω) at T.

**(a) Fixed ν = 0.01:**

| nx | Galerkin error (order) | LPS-P0 error (order) [active] | LPS-P1 error (order) |
|---|---|---|---|
| 4 | 4.85e-3 | 2.79e-2 [16/16] | 5.53e-3 |
| 8 | 7.04e-4 (2.78) | 3.46e-3 (3.01) [64/64] | 6.79e-4 (3.03) |
| 16 | 8.26e-5 (3.09) | 4.99e-4 (2.79) [256/256] | 8.19e-5 (3.05) |
| 32 | 9.13e-6 (3.18) | 5.10e-5 (3.29) [707/1024] | 9.13e-6 (3.17) |
| 64 | 1.09e-6 (3.07) | 2.64e-6 (4.27) [1179/4096] | 1.09e-6 (3.07) |

**(b) Fixed patch Péclet** (ν = 0.16/nx, max Pe_k ≈ 8–9, *every* patch active on every mesh).
This is the asymptotic test of the stabilised scheme:

| nx | Galerkin (order) | **LPS-P0** (order) | LPS-P1 (order) |
|---|---|---|---|
| 4 | 4.86e-3 | 1.31e-2 | 5.04e-3 |
| 8 | 6.08e-4 (3.00) | 2.61e-3 (**2.33**) | 6.02e-4 (3.07) |
| 16 | 8.26e-5 (2.88) | 4.99e-4 (**2.39**) | 8.19e-5 (2.88) |
| 32 | 1.09e-5 (2.93) | 1.08e-4 (**2.20**) | 1.08e-5 (2.92) |
| 64 | 1.38e-6 (2.97) | 2.57e-5 (**2.07**) | 1.38e-6 (2.97) |

**Diagnosis of the P0 order loss.**

- κ₀ = id − (patch mean) annihilates only constants, so for smooth y,
  ‖κ₀(w·∇y)‖_{L2(P)} = O(h).
- With δ_k ~ h, the consistency term δ(κ₀ w·∇y, κ₀ w·∇v) limits the L2 error to O(h²).
- For LP's Q1, Galerkin itself is O(h²) in L2, so P0 costs nothing in order there. For Q2 it
  removes one order.
- κ₁ = id − (P1 projection) annihilates linears: ‖κ₁ w·∇y‖ = O(h²), and the Q2 order 3 is
  kept. This is the standard Q_r ↔ P_{r−1}^disc pairing.
- The fixed-ν ladder (a) hides the loss: Pe_k → 1 switches patches off as h → 0. Hence the
  pre-asymptotic rate ≈ 3, with a 5–6× larger constant, and the 4.27 recovery at nx = 64.
- At fixed ε the asymptotic order is that of Galerkin (LPS vanishes once Pe_k ≤ 1). In the
  **active** regime P0 is O(h²).

**Exact OC** (`conv-sin3` + q2-divfree, ν = 0.01, CN Nt = 256; y, p, u in L2(0,T;L2)):

- Galerkin and P1: order ≈ 3 for y, p and u.
- P0: the same rates (2.3–3.1, pre-asymptotic as in (a)) with 3–5× larger errors in y and 2.5–4×
  in p and u. For example at nx = 32: y 7.58e-5 against 1.36e-5.

**Time orders** (forward MMS at nx = 32; exact OC at nx = 32):

- BE: 0.83 → 0.98 (MMS) and 1.00 (OC), unchanged by LPS.
- CN: 2.0 for Galerkin.
- P0 + CN flattens at Nt ≥ 32. That is P0's larger **spatial** floor (5.1e-5 at nx = 32),
  not a time-order loss.

## 7. KKT correctness gates with LPS (step 16) → `t6/logs/gates/`, `t6/logs/gates_summary.txt`

`--verify-convection`, LU sweeps. 9 stabilised configurations:

- LP 5.2 at ε = 1/20 and ε = 1/500, CN and BE;
- conv-sin3 + q2-divfree at ν = 0.01, CN and BE;
- lift-const + q2-divfree (inhomogeneous data, affine tracking) at ν = 0.01, CN and BE;
- P1 at ε = 1/500 CN;

plus the unstabilised reference.

| gate | stabilised range | wrong adjoint (negative control) |
|---|---|---|
| G2 space-time transpose ⟨Ex,y⟩ − ⟨x,Eᵀy⟩ | 0 … 1.9e-17 | ex20-style untransposed: 1.5e-4 … 2.0e-3 |
| G2 reduced Hessian symmetry | 2.8e-16 … 7.9e-14 | 0.12 … 0.96 |
| G3 FD reduced gradient, best rel. | 4.2e-16 … 4.0e-12 | 1.2 … 4.4 (FAIL, as required) |
| G3P physical objective (lifted data) | 2.1e-15 … 9.2e-14 | — |
| G4 production KKT: true residual | 4.3e-15 … 1.1e-8 (default outer rtol 1e-8; same scale as T3) | — |
| G4 ‖∇J(u*)‖ / ‖dt α_u N u*‖ | 6.2e-13 … 1.4e-6 | — |
| G1 Galerkin orientation / skew | unchanged (2.7e-15; ‖C + Cᵀ‖/‖C‖ 1.2e-14) | — |

All pass. As expected for an exactly transposed, symmetric W, LPS changes nothing in adjoint
consistency.

## 8. LP 5.2 primary comparison (β = 1e-2, LU sweeps) → `t6_lp52_stabilised.csv`

Setup: CN, τ = 1/nx (Nt = 2nx), T = 2, corrected affine tracking (T3b), the T4 ladder
nx = 8–64.

Metrics at t = 1 are the T4 ones, imported unchanged (`lp52_post.wiggles`,
`post_t4.excess_tv` / `layer`, mesh change at shared nodes): **all-time** min y with its
time and place (new), y(t = 1) min, wiggle rows of y and p, excess TV of y, layer
d50/d10, and max|Δy| against Galerkin.

| ε | nx | variant | y min all-time (t, place) | y(t=1) min | wiggle y / p | excess TV y | d50 / d10 | J_track |
|---|---|---|---|---|---|---|---|---|
| 1/20 | 16 | none / P0 / P1 | −2e-7 (all) | −1.9e-7 | 0/33, 0/33 (all) | 6.0e-8 … 8.5e-8 | 0.0895 / 0.222 (all) | 4.526e-3 / 4.524e-3 / 4.527e-3 |
| 1/20 | 64 | all | identical (0 active) | | | | | 4.7012e-3 |
| 1/100 | 8 | none | −0.8% (t = 2, boundary) | −3.5e-3 | **10/17**, 15/17 | 1.9e-2 | 0.084 / 0.227 | 1.991e-3 |
| 1/100 | 8 | P0 | **−10.6%** (t = dt, corner (1,−1)) | −3.3e-4 | **0/17**, 9/17 | 7.2e-4 | 0.086 / 0.235 | 2.151e-3 |
| 1/100 | 8 | P1 | −3.5% (t = dt) | −8.6e-4 | 6/17, 13/17 | 2.5e-3 | 0.085 / 0.234 | 2.022e-3 |
| 1/100 | 16–64 | all | ≥ −1.7% (P0, first step) / ~0 | ~−1e-7 | 0 / ≤ 19 (P0) | ≤ 6e-8 | equal to 3 digits | ≤ 2% apart (16), 0.002% (64) |
| 1/500 | 8 | none | −1.4% (t = 2) | −4.8e-3 | **14/17**, 15/17 | 4.5e-2 | 0.081 / 0.226 | 1.929e-3 |
| 1/500 | 8 | P0 | **−11.6%** (t = dt, corner) | −3.8e-3 | **2/17**, 8/17 | 2.9e-3 | 0.085 / 0.241 | 2.119e-3 |
| 1/500 | 16 | none | −0.3% (t = 2) | −1.0e-3 | **18/33**, 31/33 | 1.1e-2 | 0.063 / 0.221 | 2.622e-3 |
| 1/500 | 16 | P0 | −3.6% (t = dt, corner) | −4.4e-7 | **0/33**, 24/33 | 2.0e-7 | 0.066 / 0.226 | 2.763e-3 |
| 1/500 | 16 | P1 | −0.05% | −1.4e-4 | 12/33, 30/33 | 2.3e-3 | 0.066 / 0.225 | 2.698e-3 |
| 1/500 | 32 | none / P0 / P1 | ~0 / −0.37% (corner, dt) / ~0 | ~0 | 0/65 (all); p 15 / 50 / 15 | ≤ 8e-7 | 0.0647 / 0.220 (all) | 2.851 / 2.917 / 2.892 e-3 |
| 1/500 | 64 | all | ~0 | 0 | 0/129; p 0 / 110 / 2 | ~0 | 0.0648 / 0.2202 (all) | 2.964 / 2.962 / 2.967 e-3 |

Reading:

- **At β = 1e-2 (strong control), LPS is needed only on the coarsest meshes**: nx ≤ 16 at
  ε = 1/500 (Pe_cell ≥ 31) and nx = 8 at ε = 1/100. There P0 removes the t = 1 y wiggles
  (14 → 2 and 18 → 0 rows), and P1 does not (13 and 12 rows).
- From nx = 32 (ε = 1/500) and nx = 16 (ε = 1/100), Galerkin Q2 is already clean at t = 1,
  and J_track for all variants converges to the same value (0.05% apart at nx = 64).
- P0 raises the p wiggle *count*. The amplitude is small: max|p_P0 − p_Galerkin| = 1.7e-3
  against a p range of 0.18. The pattern is not element-periodic. These are low-amplitude
  perturbations crossing the detector's 1e-3-of-range threshold, not gross oscillations.

**LP Figs. 2–3 setting** (l = 5 ↔ nx = 16; `figures/t6_lp_figs23_l5.png`).

- At ε = 1/20, every variant has y ∈ [0, 1] with the layer at x1 = 1, the p trough at
  (0.875, −0.69) with minimum **−0.0214** (LP: ≈ −0.02), and a positive lobe near (0.875, 0.94).
- At ε = 1/100, the p range is [−0.0148, 0.0175] (Galerkin) against [−0.0151, 0.0162] (P0).
- This is structural agreement only: LP is Q1/OTD and the figures are raster.

## 9. Weak control, ε = 1/500 (step 11) — the scientifically useful cases

All-time min y is listed with its time, place and class, plus the minimum after t > 0.25.

**β = 1e-1:**

| nx | variant | all-time y_min (t, class) | min after t > 0.25 | y(t=1) min | wiggle y / p | excess TV y | d50 / d10 |
|---|---|---|---|---|---|---|---|
| 16 | none | −6.6% (dt, corner) | −1.4e-2 | −8.4e-3 | **26/33**, 31/33 | 9.2e-2 | 0.0500 / 0.1746 |
| 16 | P0 | **−15.8%** (dt, corner) | −5.6e-3 | −1.0e-4 | **0/33**, 16/33 | **2.3e-3** | 0.0516 / 0.1845 |
| 16 | P1 | −9.7% (dt, corner) | −1.3e-2 | −2.3e-3 | 17/33, 29/33 | 1.3e-2 | 0.0514 / 0.1829 |
| 32 | none / P0 / P1 | −5.0% / **−11.1%** / −7.0% (dt, corner) | −3e-5 (all) | ~−5e-6 | 1 / 0 / 0; p 6 / 40 / 4 | 4e-7 / 2.4e-4 / 4e-7 | 0.048 / 0.184 (all) |
| 64 | none / P0 / P1 | −5.1% / −4.9% / −5.0% (dt, corner) | −3e-5 (all) | −5e-6 | 0 (all); p 0 / 60 / 0 | 4e-7 (all) | 0.0458 / 0.1837 (all) |

**β = 1e2 (≈ uncontrolled):**

| nx | variant | all-time y_min (t, class) | min after t > 0.25 | y(t=1) min | wiggle y / p | excess TV y | d50 / d10 |
|---|---|---|---|---|---|---|---|
| 8 | none | −36% (mid, boundary) | −36% | −35% | 15/17, 14/17 | 1.34 | 0.051 / 0.093 |
| 8 | P0 | −20% (dt) | −19% | −18% | 5/17, 9/17 | 0.20 | 0.058 / 0.104 |
| 8 | P1 | −22% (mid) | −22% | −20% | 12/17, 12/17 | 0.48 | 0.057 / 0.103 |
| 16 | none | −10.1% (dt, corner) | −6.6% | −6.3% | **18/33**, 29/33 | 0.38 | 0.0340 / 0.0611 |
| 16 | P0 | −18.5% (dt, corner) | **−14.2%** | **−14.0%** | **9/33**, 18/33 | **0.032** | 0.0348 / 0.0637 |
| 16 | P1 | −12.7% (dt, corner) | −8.5% | −4.9% | 25/33, 25/33 | 0.14 | 0.0348 / 0.0632 |
| 32 | none | −7.2% (dt, corner) | −1.8% | −0.31% | **23/65**, 8/65 | 6.2e-3 | 0.0262 / 0.0616 |
| 32 | P0 | −13.4% (dt, corner) | **−5.5%** | **−4.7%** | **0/65**, 21/65 | 2.4e-3 | 0.0262 / 0.0618 |
| 32 | P1 | −9.2% (dt, corner) | −2.3% | −0.66% | 21/65, 2/65 | 3.7e-3 | 0.0261 / 0.0617 |
| 64 | none | −6.0% (dt, corner) | −2e-8 | −8e-9 | 0/129, 0/129 | 6.7e-5 | 0.0255 / 0.0617 |
| 64 | P0 | −5.8% (dt, corner) | **−0.68%** | **−0.58%** | 1/129, 19/129 | 3.6e-4 | 0.0255 / 0.0617 |
| 64 | P1 | −5.9% (dt, corner) | −2e-8 | −8e-9 | 0/129, 0/129 | 4.9e-5 | 0.0255 / 0.0617 |

Figures:

- `figures/t6_profiles_weak_control.png`: y(x1, x2 = 0, t = 1), all nx, three variants;
- `figures/t6_unstabilised_vs_lps.png`: y(t = 1) fields, β = 1e2, nx = 16 and 64.

At nx = 16 the Galerkin field carries a **domain-wide alternating (odd-even) pattern**,
visible as sign-changing zero contours everywhere. P0 removes it almost entirely,
leaving one localised dip at the corner (1, +1). P1 leaves the global pattern intact.

**Answers for weak control:**

- **Odd-even oscillations:** P0 removes them on the under-resolved meshes: wiggle rows
  26 → 0 and 18 → 9, excess TV 40× and 12× smaller at nx = 16; 23 → 0 rows at nx = 32,
  β = 1e2. P1 is too weak to do so.
- **Undershoot:** P0 does **not** reduce it. The all-time minimum gets worse at nx = 16–32
  (−6.6% → −15.8% and −5.0% → −11.1% for β = 0.1; −7.2% → −13.4% for β = 1e2). For β = 1e2,
  P0 adds a **persistent** undershoot at the corner (1, +1), converged in time, where Galerkin
  is clean (nx = 32: −5.5% against −1.8% after t > 0.25; −4.7% against −0.31% at t = 1;
  nx = 64: −0.6% against 0).
- **Mesh-to-mesh consistency:** away from the corners it improves slightly at nx = 16–32
  (β = 0.1: 7.4e-2 → 3.3e-2 at nx 32; β = 1e2: 0.138 → 0.086). Near the corners it does not
  improve. Including the corners it is mixed: 0.048 → 0.070 at nx = 64, β = 1e2.
- **Layer width:** unchanged (see §J).

**Why P0 makes corner undershoots worse.** The data jump from 1 (x1 = 1) to 0 (x2 = ±1) at both
corners. Through the lifting, W acts on the interpolated discontinuous data in the corner
element (the −W_FG g part of the load). The streamline derivative of that interpolant is
O(1/h) there, and LPS, a linear symmetric stabilisation, has no discrete maximum principle.

The corner (1, +1) is where the flow along the top edge (y = 0) turns down the edge x1 = 1
(y = 1): the start of the characteristic layer. There P0's δ_k is largest relative to the
local diffusion. A cure would require a non-linear, monotone stabilisation or regularised
corner data, both outside T6's scope (§N).

## 10. Location of the all-time undershoot (step 12)

All-time minimum (`--exact-oc-series`), ε = 1/500:

- weak control, nx ≥ 16, **every** variant: at **the node next to the corner (1, −1)**
  (distance ≤ 0.07), at **t = dt** in the 2nx-step runs;
- at nx = 8 and at β = 1e-2 the Galerkin minima are elsewhere (boundary, t = T, < 1.5%).

Time-step study (`undershoot` group; ε = 1/500, β = 1e2, nx = 32):

| scheme | Nt = 64 | Nt = 256 | Nt = 1024 | place, time |
|---|---|---|---|---|
| Galerkin CN | −7.22% | −7.38% | −7.37% | (0.938, −0.969), t = 0.031 → 0.045 |
| Galerkin BE | −4.25% | −6.48% | −7.14% | same node, t = 0.031 → 0.045 |
| P0 CN | −13.4% | −10.6% | −10.5% | same node, t ≈ 0.03 |
| P0 BE | −7.7% | −9.3% | −10.2% | same node |

After t > 0.25, P0 holds a persistent −5.7% (all Nt, both schemes) at (1, +1), against
−0.4 … −2.0% for Galerkin (BE at Nt = 64 is lowest). At t = 1: −4.72% (P0, all) against −0.31 … −0.49% (Galerkin).

**Classification:**

- It is an **early-time transient** of the solution next to the discontinuous corner data,
  at a fixed physical time t ≈ 0.03–0.045.
- It is **converged in dt** (CN Nt 256 → 1024: −7.38% → −7.37%), and BE converges to the
  same value. So it is **not** a CN/BE time-stepping artefact of the discontinuous initial
  data, and not "the first step" as such.
- It is **spatial**, decaying slowly with h: −10.1%, −7.2%, −6.0% at nx = 16, 32, 64 for
  β = 1e2; −6.6%, −5.0%, −5.1% for β = 0.1. That is the behaviour of an under-resolved
  corner singularity.
- It is **not an interior convection oscillation**. After t > 0.25 the Galerkin minimum is
  −3e-5 (β = 0.1, nx ≥ 32) and −2e-8 (β = 1e2, nx = 64).
- The T4 statement "probably early transient" is therefore replaced by: **early spatial
  corner transient at (1, −1), dt-converged, scheme-independent.**
- LPS does not cure it. P0 amplifies it; P1 slightly amplifies it.

## 11. Incidental solver effect (steps 14–15; secondary) → `t6_solver_effect.csv`

Block harness (T5 settings, 3 random + 1 smooth RHS, both θ, E/right/left), max iterations
per point; "500!" = not converged.

| point | GAMG (production) unstabilised | **GAMG + LPS-P0** | AIR H2c unstabilised (levels) | AIR H2c + P0 (levels) |
|---|---|---|---|---|
| A (1/20, 32) | 24 | 22 | 15 (10–11) | 15 (10–11) |
| B (1/20, 16) | **500!** (θ=1 E), 278 | **22** | 9 (9) | 12 (9) |
| C (1/100, 32) | **500!** all blocks | **24** | 10 (12) | 12 (11–12) |
| D (1/500, 32) | **500!** all | **26** | 16 (13–14) | 15 (12–13) |
| E16 (1/500, 16) | **500!** all | **24** | 18 (11) | 14 (10) |
| D64 (1/500, 64) | **500!** all | **28** | 15 (15–16) | 14 (14–15) |

**End-to-end KKT with the production default GAMG** (no options; `kkt_gamg` group,
β = 1e-2, CN):

- **unstabilised:** 4/4 runs hit the 900 s limit (as in T4);
- **LPS-P0:** 4/4 accepted, 2 outer iterations, spatial max 19–22 iterations, solution identical
  to LU + P0 (max|Δy| ≤ 1e-13);
- **LPS-P1:** 4/4 accepted, 38–62 spatial iterations.

So the symmetric PSD streamline term (‖W‖ ~ 3× the convection norm at these Pe) incidentally
makes the blocks GAMG-friendly. AIR H2c, already robust, is unaffected: ±3 iterations and
a similar hierarchy depth; no np = 4 debugging was done (as instructed). LPS was not tuned
for solver behaviour.

## 12. Regression (step 17), LPS off — `t6/regress_t6.sh`, `t6/logs/regress_*`

| check | result |
|---|---|
| ex20 goldens (stdout, stderr, JSON, exit codes) | **61/61 identical** |
| Mathematics 2025 Ex. 3 (probe + primary), Ex. 4 | **byte-identical** |
| Mathematics 2025 Ex. 5 | **byte-identical** |
| ErrorFeedback 9.5 | numerics identical; the known pre-T3 CONFIG label difference only (`variant=source`, same diff as T3b) |
| T0/T1/T2: operator dumps (12), 8 archived exact-OC runs, T2 gate logs | **REGRESSION PASS** |
| T3/T3b/T4 representative runs, T4 binary (500a2b2e) vs new: lift-track, lift-const + q2-divfree BE, variable-wind OC, lp52 ε = 1/20 GAMG, lp52 ε = 1/500 LU CN, lp52 ε = 1/100 LU BE, block test D, T2 verify-convection, and a **failing** GAMG lp52 run | **9/9 identical** `EX21_` output. The failing run keeps exit 8 and now additionally prints `EX21_SOLVE_REJECTED` |
| LPS on, zero active patches (ε = 1/20, nx = 64, CN and BE) | **byte-identical** to LPS off (§5) |

## 13. Acceptance criteria (step 18)

| # | criterion | result |
|---|---|---|
| 1 | symmetry / PSD | **PASS** (exact symmetry, dense eigenvalues ≥ 0, 35/35) |
| 2 | Pe activation = independent LP calculation | **PASS** (18/18; l = 7, 8 at ε = 1/20: 0 active; l = 5, 6: 212, 136) |
| 3 | low-Pe W = 0 exact | **PASS** (W ≡ 0; full run byte-identical) |
| 4 | smooth manufactured convergence | **P0: order 2 (Q2 order lost while active), diagnosed. P1: order 3.** Time orders preserved |
| 5 | KKT transpose / gradient / Hessian gates | **PASS** (all 9 stabilised configurations; negative control fails) |
| 6 | high-Pe oscillations reduced where T4 found them | **P0: yes** for the global odd-even oscillation on under-resolved meshes. P1: marginal |
| 7 | no artificial instability | **Not met by P0:** it amplifies, and partly creates, corner undershoots at the data discontinuities (up to −15.8% all-time, −4.7% persistent at t = 1). P1 mildly. No instability in the solver sense (all runs converge) |
| 8 | layer not grossly smeared | **PASS** (d50/d10 unchanged at nx ≥ 16; ≤ 10% wider d10 only at nx = 8) |

The **implementation passes**. LP's Q1 LPS transferred to Q2 (P0) does **not** pass
criteria 4 and 7 as an unqualified improvement. It is a coarse-mesh oscillation
suppressor with known costs.

## 14. Answers to the scientific questions (step 19)

1. **Is LP-style LPS effective for Q2 ex21?** Partly.
   - P0 removes the global odd-even oscillations of the under-resolved meshes (Pe_cell ≥ 31).
   - It costs one order of accuracy while active, and it amplifies the corner undershoots
     tied to the discontinuous data.
   - The order-preserving P1 variant is too weak to remove the oscillations.
2. **Q2 extension:** patch = one Q2 element (geometrically LP's 2×2 Q1 patch on the same node
   grid). LP's P0 patch-mean projection, δ_k, Pe_k, h_k (wind-direction chord) and w_k
   (centroid) are taken exactly. The variable wind is used inside the integral. P1 is a
   labelled diagnostic variant.
3. **Spatial order:** P0 is **O(h²)** in the active regime (fixed Pe_k: 2.33 → 2.07). P1 is
   **O(h³)**. At fixed ε both revert to Galerkin O(h³) once Pe_k ≤ 1.
4. **When it becomes materially useful:** only where Galerkin oscillates globally. That is
   patch Pe ≳ 30 (cell Pe ≥ 31): ε = 1/500 with nx ≤ 16 (β = 1e-2), nx ≤ 32 at weak control;
   ε = 1/100 with nx = 8. At ε = 1/20 it is never useful.
5. **At β = 1e-2:** essentially not needed. Galerkin is clean at t = 1 from nx = 16 (ε = 1/100)
   and nx = 32 (ε = 1/500), and all variants converge to the same J_track.
6. **Weak control, ε = 1/500:** P0 cures the odd-even oscillations at nx = 16–32 but not the
   undershoot. At nx = 64 Galerkin is already clean, and P0 is slightly worse (a −0.6%
   persistent corner dip).
7. **The 5–10% all-time undershoot** is an early (t ≈ 0.03–0.045), dt-converged,
   scheme-independent spatial corner transient next to (1, −1). It decays slowly with h. LPS
   does not reduce it (P0: −10.5% against −7.4% at nx = 32).
8. **Layer smearing:** none of significance. d50 and d10 at x2 = 0 agree to 3–4 digits for
   nx ≥ 16; at nx = 8 d10 is ≤ 10% wider.
9. **Solver conditioning:** yes, for GAMG, strongly. LPS-P0 turns the failing production GAMG
   into a robust 19–28-iteration solver (KKT: 2 outer, identical solution). AIR H2c is
   unchanged.
10. **Credible convection-dominated distributed-control regime?** Yes, with qualifications.
    - The KKT/adjoint machinery is exact with or without LPS, and the solver side is robust
      (LU, ILU, ASM+ILU; now also GAMG + LPS).
    - Discretisation: credible where the mesh resolves the √ε characteristic layer (ε = 1/500:
      nx ≥ 32 at β = 1e-2, nx ≥ 64 at weak control) with plain Galerkin Q2.
    - On coarser meshes, LPS-P0 gives oscillation-free interiors at O(h²) accuracy, with
      localised corner undershoots.
    - No variant gives a discrete maximum principle at discontinuous corner data.
11. **Open after T6:** see §N.

## N. Remaining open issues

- **Corner undershoots** at discontinuous Dirichlet/initial data (both variants, and Galerkin):
  - a monotone, non-linear stabilisation (AFC, FCT, shock capturing) — not adjoint-consistent
    in the DTO sense without care;
  - or regularised corner data.
- **An LPS that is both order-preserving and effective for Q2.** P1 projection is
  order-preserving but weak with LP's δ_k. Calibrating δ for P1/Q2 would be stabilisation
  tuning, deliberately not done here.
- **BE LP 5.2 comparison.** Only the gates and the undershoot study ran BE with LPS; the β/ε
  maps are CN.
- **hypre AIR np = 4 crash** (T5b), unchanged.
- **Stabilised-block MPI performance:** not studied. The assembly is verified MPI-correct.

## O. Files

Source (untracked `ex21/`):

- `include/Solver.hpp`, `include/Solver.cpp` — acceptance guard;
- `ex21.cpp`, `include/Benchmarks.cpp`, `include/Math2025Benchmark.cpp`,
  `include/Benchmark2D.cpp`, `include/ExactOcBenchmark.cpp`, `include/Verification.cpp` —
  exit codes use `accepted`;
- `include/Config.hpp`, `include/CommandLine.cpp`, `include/Options.hpp` — LPS options,
  `--exact-oc-series`;
- `include/Discretization.hpp`, `include/Discretization.cpp` — `LpsStats`, `BuildLpsMatrix`,
  assembly hook;
- `include/ConvectionVerification.cpp` — G1 LPS-free, G5 gate;
- `include/ExactOcBenchmark.cpp` — W dump, ν for convective cases, lp52 series.

Investigation (`aux/investigation/leveque_pearson/`):

- `T6_LPS_STABILISATION.md`, `t6_lps_matrix_checks.csv`, `t6_lps_mms.csv`,
  `t6_lp52_stabilised.csv`, `t6_solver_effect.csv`;
- `figures/t6_profiles_beta1e-2.png`, `t6_profiles_weak_control.png`, `t6_undershoot_time.png`,
  `t6_unstabilised_vs_lps.png`, `t6_lp_figs23_l5.png`;
- `t6/`:
  - `run_t6.py` (groups mms_space, mms_fixedpe, mms_time, oc_space, oc_time, lowpe, lp52, weak,
    undershoot, kkt_gamg);
  - `guard_test.py`, `gates_t6.sh`, `matrix_checks.py`, `solver_effect.py`, `mpi_lps_check.py`,
    `regress_t6.sh`, `post_t6.py`;
  - `results/` (incl. `snap/`, `series/`), `logs/`.

**Exact options:**

- LPS: `--convection-stabilization lps [--lps-projection p0|p1]`.
- LU sweeps: `PETSC_OPTIONS="-sweep_right_pc_type lu -sweep_left_pc_type lu -sweep_ay_pc_type lu -fieldsplit_y_sweep_pc_type lu"`.
- MPI: the same with `-…pc_factor_mat_solver_type mumps`.
- GAMG: no options.
- AIR H2c: `t5b/run_t5b.py show hypre_air1_mrs1`.

**Binaries:**

- production build `femusbin-control`, final sha256 3811bf71…;
- hypre build `femusbin-control-hypre` rebuilt from the same source;
- the T4 binary 500a2b2e is kept as a scratch copy for the regression.
