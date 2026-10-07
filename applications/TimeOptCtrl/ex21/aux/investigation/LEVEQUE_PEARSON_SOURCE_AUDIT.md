# Leveque & Pearson (NLAA 2022) — source audit for ex21

Audit only. No run, no code change, no benchmark implemented. The frozen
Aulisa/Gilliam campaign and report are untouched.

Tags: **[A]** stated in the paper · **[B]** inferred from the paper's
equations, figures or context (evidence given) · **[C]** missing or ambiguous.

---

## 1. Paper identification

* S. Leveque, J. W. Pearson, *Fast iterative solver for the optimal control of
  time-dependent PDEs with Crank–Nicolson discretization in time*, Numer. Linear
  Algebra Appl. **2022**; 29:e2419, DOI 10.1002/nla.2419 (received 16 Jul 2020,
  accepted 15 Oct 2021; open access).
* Local copies: two files in `~/Downloads`, "Numerical Linear Algebra App - 2021
  - Leveque - Fast iterative solver … with.pdf" and "…with-1.pdf". Both have
  23 pages, and their text layers differ only in the Wiley download-date
  footer. The `-1` copy was used.
* **It is the paper remembered.** Its heat-control Section 5.1 was already
  reproduced by ex20 (`ex20/aux/EX20_LEVEQUE_PEARSON_2022_BENCHMARK.md`, Q1
  harness, analytic solution). That work is relied on below and not redone.
* The paper was read in full (pp. 1–23, including references) from the PDF
  text layer. Figures 2–3 were inspected as rendered pages; they are raster
  images (300 dpi), and Figure 1 is an eigenvalue scatter plot.

## 2. Executive conclusion

1. **Setting.** Leveque–Pearson (LP) solve *distributed, full-domain* optimal
   control of the transient convection–diffusion equation (and the heat
   equation as a special case), with *full-domain* tracking. The method is
   **optimize-then-discretize**, Q1 elements, LPS stabilisation, and
   Crank–Nicolson in time. The resulting system is made symmetric by a block
   transformation and solved by MINRES with a matching Schur-complement
   preconditioner, which is proven robust in h, τ, β (and ε).
2. **Closeness to ex21.**
   * The *continuous* problem is the one ex21 solves, with `ν = ε`, `v = w`,
     `η = 1`, full-domain `B = I`, `α_track = 1` and **`α_u = β` exactly**
     (derived in §3; the functionals are identical).
   * **Space:** with homogeneous Dirichlet boundaries and LP's Assumption 1, LP's
     discrete spatial adjoint operator is *exactly* `Lᵀ`, the matrix ex21 uses,
     provided the stabilisation is absent or symmetric.
   * **State equation in time:** LP's CN state equation is row-for-row ex21's CN
     state equation once LP's nodal-control average is identified with ex21's
     interval control.
   * **Adjoint and tracking in time:** these are discretised differently (LP:
     nodal adjoint with an imposed `p(t_f)=0` and an interval-averaged tracking
     residual; ex21: exact transpose, interval adjoint, nodal trapezoid tracking
     weights).
   * **Answer: (b) a different discretisation of the same continuous OC
     problem**, not the same discrete system (§5–6).
3. **Reproducible now.** Only the heat test (§5.1, Tables 1–3) is reproducible,
   and it has already been done by ex20. ex21 at `v = 0` is bit-identical to
   ex20, so it transfers; rerunning it through the ex21 Q2 production path is a
   small preset (class B).
4. **The convection tests (§5.2, Tables 6–8, Figures 2–3) need new
   capabilities** (class C): a variable, divergence-free (recirculating) wind,
   inhomogeneous Dirichlet state data, Q1, and — wherever the patch Péclet
   number exceeds 1 — LPS stabilisation. They also publish **no solution
   values**: only MINRES counts and raster surface plots. A quantitative
   solution comparison is therefore impossible (class D for numbers). A
   structural and qualitative comparison is possible once the capabilities
   exist.
5. **They do test a genuinely convection-dominated regime.** Element Péclet
   numbers reach `|w|h/(2ε) = 125` (ε = 1/500, l = 3). ex21's whole external
   validation so far is at ≤ 0.22.
6. **Boundary control.**
   * LP list "boundary control problems" as **future work for their CN
     approach** (Conclusions). They do not restrict this to convection or name
     Dirichlet or Neumann.
   * Their own reference 5 (Pearson–Stoll–Wathen 2012, in `~/Downloads`) already
     treats *time-dependent Neumann boundary control of the heat equation* with
     all-at-once preconditioners.
   * No novelty claim is possible or made.
7. **Verdict on ex21** (§13, §17). It is a **correct and internally verified**
   distributed transient convection–diffusion AAO application for **constant
   wind, homogeneous Dirichlet data and low cell Péclet number**, externally
   corroborated in 1D and 2D. It is **not yet mature for the class LP treat**:
   * there is no exact-OC-solution convergence test with convection;
   * there is no variable wind and no inhomogeneous Dirichlet data;
   * there is no stabilisation and no high-Péclet evidence;
   * boundary control is not accepted under convection.

## 3. The continuous problem (§2 of the paper)

| datum | LP | tag |
|---|---|---|
| functional (1) | `J = ½∫₀^{t_f}∫_Ω |y − ŷ|² + (β/2)∫₀^{t_f}∫_Ω |u|²` | [A] |
| state (2) | `y_t − ε∇²y + w·∇y = u + f` in `Ω×(0,t_f)`; `y = g_D` on `∂Ω_D`; `∂y/∂ν = g_N` on `∂Ω_N`; `y(0) = y₀` | [A] |
| wind | `w` divergence-free | [A] |
| Assumption 1 | for convection–diffusion, `w·ν = 0` on `∂Ω_N` (no restriction for a pure Dirichlet problem) | [A] |
| heat case (3)–(4) | `ε = 1`, `w = 0`, Dirichlet on all of `∂Ω`: `y_t − ∇²y = u + f`, `y = g` | [A] |
| control | **distributed on all of Ω**, same FE space as the state; eliminated a priori | [A] |
| observation | **full-domain** tracking `∫_Ω |y − ŷ|²`; no terminal term | [A] |
| optimality system (9) | state with `u = p/β`; adjoint `−p_t − ε∇²p − w·∇p = ŷ − y`, `p = 0` on `∂Ω`, `p(t_f) = 0`; gradient `βu − p = 0` | [A] |
| dimensions | `d ∈ {1,2,3}` in the theory; all experiments `d = 2` | [A] |

**Mapping to ex21.** ex21 solves `y_t − ν∆y + v·∇y = f + ηBu` and minimises
`(α_track/2)∫‖y − y_d‖²_{M_ω} + (α_u/2)∫‖u‖²_N`. With `ν = ε`, `v = w` (the
same sign: both have `+w·∇y` on the left), `η = 1`, `B = I` (`N = M`,
control box = Ω), tracking box = Ω (`M_ω = M`) and `α_track = 1`, the two
functionals and constraints are **identical**, hence **`α_u = β` exactly**.
This was derived, and confirmed numerically, in the ex20 study §5–6. The only
convention difference is the sign of the multiplier: `p_ex21 = −p_LP` (ex20
§6), and ex21's `u = −(η/α_u)p`.

**Convection matrix convention.** LP: `n_ij = ∫(w·∇φ_j)φ_i` [A]. ex21:
`C_ij = ∫(v·∇φ_j)φ_i`. These are **identical**, with the same row = test
function convention.

## 4. The discrete formulation (§2.1, §3, §4)

**Space** [A]: `L = εK + N + W_w`, with `K` the stiffness, `N` the convection
and `M` the mass matrix; the control mass is `M` (same basis).
* `N` is skew for Dirichlet problems.
* `W_w` is the local projection stabilisation (LPS) matrix (7). It is symmetric
  positive semidefinite, with `W_{−w} = W_wᵀ = W_w` for the patchwise-constant
  projection (8).
* The parameter is `δ_k = h_k/(2‖w_k‖)(1 − 1/Pe_k)` if the patch Péclet
  number `Pe_k = ‖w_k‖h_k/(2ε) > 1`, and **`δ_k = 0` otherwise**.
* Patches are 2 elements per direction (Q1). `h_k` is the patch length in the
  wind direction.

SUPG is discussed and *rejected*, because SUPG-DTO is adjoint-inconsistent and
SUPG-OTD is not an optimisation system. Remark 4 states that Theorems 2–3 also
hold **without** stabilisation [A].

**BE** (10)–(12) [A]: `τM yⁿ + (L_E)ᵀpⁿ − M pⁿ⁺¹ = τM ŷⁿ`;
`−M yⁿ + L_E yⁿ⁺¹ − (τ/β) M pⁿ⁺¹ = τ fⁿ⁺¹`, with `L_E = τL + M`, `p^{n_t} = 0` and
`y⁰ = y₀`. The system is symmetrised by replacing the initial and final
conditions.

**CN** (13)–(15) [A]. With `L± = (τ/2)L ± M`, `M̄ = (τ/2)M` and
`M̄_β = τ/(2β) M`, for `n = 0,…,n_t−1`:

    M̄(yⁿ + yⁿ⁺¹) + (L⁺)ᵀpⁿ + (L⁻)ᵀpⁿ⁺¹ = M̄(ŷⁿ + ŷⁿ⁺¹)            (adjoint row, interval n)
    L⁻yⁿ + L⁺yⁿ⁺¹ − M̄_β(pⁿ + pⁿ⁺¹)     = (τ/2)(fⁿ + fⁿ⁺¹)          (state row, interval n)

with `p^{n_t} = 0` (imposed) and `y⁰ = y₀`. After eliminating `y⁰` and
`p^{n_t}`, the unknowns are `y¹…y^{n_t}` and `p⁰…p^{n_t−1}`. The coupling block
`Λ21` is lower block-bidiagonal (`L⁺` diagonal, `L⁻` sub-diagonal) and
`Λ12 = Λ21ᵀ`; `Λ11` and `Λ22` are *non-symmetric* block-bidiagonal mass
matrices.

**Symmetrisation** (16)–(19) [A]. The left transformation
`T = diag(T₁, T₂)`, with bidiagonal identity blocks, gives the symmetric saddle
point `[A Bᵀ; B −C]` with `A = T₁A_D T₁ᵀ`, `C = T₂C_D T₂ᵀ`,
`A_D = blockdiag(M̄)` and `C_D = blockdiag(M̄_β)`. MINRES then applies.

**Preconditioner** (20), (28)–(32) [A]:
* `blockdiag(Â, Ŝ)`, with `Â = T₁ Â_D T₁ᵀ` (20 Chebyshev steps per mass block);
* the matching approximation `Ŝ = (Λ21 + M̂) Â_D⁻¹ (Λ21 + M̂)ᵀ`, with
  `M̂ = τ/(2√β) [M; M M; …]` (block lower-bidiagonal);
* block forward and backward substitution with 3 multigrid V-cycles per
  diagonal block (HSL_MI20 for heat, AGMG for convection; 5 V-cycles on the
  Shishkin mesh).

**Theorem 3** [A]: `λ(Ŝ⁻¹S) ⊂ [1/2, 1]`, independent of h, τ, β and ε, under
Assumption 1. The proof uses `L + Lᵀ = 2(εK + W)` positive definite and the
**same mass matrix `M` in the tracking and control terms**.

## 5. Optimize-then-discretize versus ex21's discretize-then-optimize

**Space: identical under the paper's assumptions.**
* The OTD adjoint operator `−ε∇² − w·∇` discretised by Galerkin gives
  `εK + N_{−w} = εK − N`. For Dirichlet problems (or `w·ν = 0` on Neumann
  parts) `N` is skew, so `εK − N = (εK + N)ᵀ`.
* With LPS, `W_{−w} = W_w = W_wᵀ`.
* Hence LP's spatial adjoint matrix is **exactly `Lᵀ`** — the matrix ex21
  applies by `MatMultTranspose`. LP state this themselves: footnote 1 attributes
  the `(L_E)ᵀ` in the BE system to the skew-symmetry of `N`.
* With **SUPG** this would fail (LP §2.1). With Neumann outflow boundaries
  (`w·ν ≠ 0`) `N` is not skew, and LP's Remark 5 notes that the discrete adjoint
  must then be chosen non-trivially. ex21's exact transpose needs no assumption
  in either case.

**Time: not identical.** Multiply ex21's CN state equation by `τ`. ex21 has
`τA₊ = M + (τ/2)L = L⁺` and `τA₋ = −M + (τ/2)L = L⁻`, so for each interval
`(t_{n−1}, t_n]`:

    ex21 state, interval n:   L⁻ y_{n−1} + L⁺ y_n − τ η N u_n = (τ/2) M_d (f_{n−1} + f_n)

* The source quadrature is trapezoidal in both codes (ex21 `Solver.cpp`,
  "TRAPEZOIDAL average ½(F_n + F_{n−1})").
* With `N = M` and `η = 1`, this is LP's state row **exactly**, under the
  identification `u_n^{ex21} = (p^{n−1} + pⁿ)/(2β)` = LP's trapezoid average of
  its nodal controls over the interval.
* **The state equations therefore coincide row by row.**

The adjoint (KKT for `y_n`) of ex21's discrete functional, with tracking
weights `w_n = (1,…,1,½)` on `y_1…y_N` (`y_0` is data), is:

    ex21 adjoint, node n:   τ w_n M (y_n − ŷ_n) + (L⁺)ᵀ λ_n + (L⁻)ᵀ λ_{n+1} = 0,   λ_{N+1} = 0

Compare this with LP's adjoint row for interval n,
`M̄(yⁿ + yⁿ⁺¹) + (L⁺)ᵀpⁿ + (L⁻)ᵀpⁿ⁺¹ = M̄(ŷⁿ + ŷⁿ⁺¹)`:

| | LP (OTD) | ex21 (DTO) |
|---|---|---|
| adjoint unknown lives at | nodes `t_n` (`pⁿ ≈ p(t_n)`) | intervals (`λ_n`, effectively `t_{n−½}`; ex20 §6 confirmed numerically) |
| tracking residual in the adjoint row | interval average `(rⁿ + rⁿ⁺¹)/2` | nodal `r_n` with trapezoid weight `w_n` |
| terminal condition | `p(t_f) = 0` **imposed** at the last node | `λ_{N+1} = 0`, half a step beyond `t_f`; produces an `O(τ)` terminal layer in `p` (ex20 §27) |
| control | nodal `uⁿ = pⁿ/β`; enters the state as its interval average | one value per interval, `u_n = −(η/α_u)λ_n` |
| regularisation quadrature | implied by `C = T₂C_D T₂ᵀ` | `τ Σ u_nᵀN u_n` (exact for piecewise-constant u) |
| system | non-symmetric, symmetrised by `T` for MINRES | the KKT system of a discrete functional; FGMRES on `[E S; T Eᵀ]` |

The two adjoint recursions are different second-order discretisations of
`−p_t − ε∇²p − w·∇p = ŷ − y`. They are not related by a row transformation in
general, because the tracking residual is sampled differently. **Conclusion:
(b) — a different discretisation of the same continuous OC problem.** ex20
measured the consequence for the heat case: interior time slots converge at the
same second order; the difference is a terminal layer of the adjoint that
converges at first order; both converge to the analytic solution, with
error ratios settling at `0.32–2.25` at `l = 7` and `1.1–1.6` at `l = 8`.
The derivation above is [B] (by this audit, from LP (13)–(15) and ex21's
source). A literal algebraic cross-check on a tiny grid is listed as optional
test T0 (§14).

## 6. Side-by-side KKT (CN, after eliminating initial data)

| block | LP (unknowns `y¹…y^{n_t}`, `p⁰…p^{n_t−1}`) | ex21 (unknowns `y_1…y_N`, `λ_1…λ_N`; control eliminated) |
|---|---|---|
| state operator | `Λ21`: diagonal `L⁺`, sub-diagonal `L⁻` | `τE`: diagonal `L⁺ = M + τL/2`, sub-diagonal `L⁻ = −M + τL/2` — **same** |
| adjoint operator | `Λ12 = Λ21ᵀ`: diagonal `(L⁺)ᵀ`, super-diagonal `(L⁻)ᵀ`; last block `(L⁺)ᵀ` | `τEᵀ`: diagonal `(L⁺)ᵀ`, super-diagonal `(L⁻)ᵀ` — **same pattern**; the last row couples only `λ_N` |
| control coupling | `Λ22 = M̄_β·(I + shift)`: block bidiagonal (`pⁿ + pⁿ⁺¹`) | `S = (τη²/α_u)·blockdiag(N)`: block **diagonal** |
| tracking | `Λ11 = M̄·(I + shift)`: block bidiagonal (`yⁿ + yⁿ⁺¹`) | `T = τ α_track blockdiag(w_n M_ω)`, `w = (1,…,1,½)`: block **diagonal** |
| first time row | `y⁰ = y₀` moved to the right-hand side (`L⁻y₀` in state row 0, `M̄y₀` in adjoint row 0) | `y₀` enters state row 1 through `−L⁻y₀`; the tracking weight at `t_0` is irrelevant (y₀ is data) |
| last time row | `p^{n_t} = 0`; the last state row involves `p^{n_t−1}` only | `λ_{N+1} = 0`; the tracking weight is ½ at `y_N` |
| β scaling | `M̄_β = τ/(2β)M` | `η²τ/α_u` with `α_u = β` |
| signs | `p_LP` | `λ_ex21 = −p_LP` (dt-scaled multiplier) |

## 7. Inventory of every numerical experiment

Common [A]:
* `d = 2`, `Ω = (−1,1)²`, `t_f = 2`, `f = 0`;
* Dirichlet boundary conditions only ("we consider only Dirichlet boundary
  conditions");
* Q1 for state, control and adjoint;
* uniform grid `h = 2^{1−l}`;
* MINRES to a relative residual tolerance of `1e-6`;
* iteration counts start at 0;
* MATLAB R2018b on a 1.70 GHz quad-core i5 with 8 GB RAM.

| # | where | problem | parameters | reported | exact solution | complete for reproduction? |
|---|---|---|---|---|---|---|
| E1 | Fig. 1 | Dirichlet conv.–diff. test, `ε = 1/100`, recirculating `w`, on `(−1,1)²×(0,2)` | `τ = h = 1/8`, `β = 10^{-2…-5}` | eigenvalues of `S̃_int⁻¹ S_int` (scatter) | n/a | ŷ, IC and BC of "a particular Dirichlet test problem" are not stated [C]; it is a property of *their* preconditioner |
| E2 | §5.1, Tables 1–3 | heat control (3)–(4), analytic `y`, `p`, `ŷ` with `s = cos(πx₁/2)cos(πx₂/2)` | `β = 10^{-2,-3,-4}`; `l = 3…8`; BE `τ = h²` (ξ = 1e-3), CN `τ = h` | MINRES `it`, CPU, `y_error`, `p_error` (relative pointwise at argmax) | **yes** (analytic) | **yes**; reproduced by ex20 (§9) |
| E3 | §5.1, Table 4 | as E2 | `β = 1e-3` | DoF and CPU per preconditioner block | n/a | CPU of *their* blocks; not meaningful for ex21 |
| E4 | §5.1, Table 5 | as E2 on a **Shishkin** mesh (transition points 0.5 and −0.6) | `β = 10^{-2,-3,-4}`, `l = 4…8`, 5 V-cycles | MINRES `it`, CPU | yes (same analytic solution) | yes, but it needs non-uniform meshes and reports only solver data |
| E5 | §5.2, Figs. 2–3 | conv.–diff. control (below) | `β = 1e-2`, `l = 5`, `ε = 1/20` (Fig. 2) and `1/100` (Fig. 3) | raster surface plots of `y(·,1)`, `p(·,1)` | no | only qualitative values readable (e.g. `y ∈ [0,1]`, layer at `x₁ = 1`; `p` minimum ≈ −0.02 for ε = 1/20) |
| E6 | §5.2, Tables 6–8 | conv.–diff. control | `ε = 1/20, 1/100, 1/500`; `β = 10^{-1…-6}`; `l = 3…8`; `τ = h` (CN) | MINRES `it` only (10–27) | no | the problem is fully specified, but no solution or objective values exist |

**§5.2 problem (E5, E6)** [A]:
* wind `w = [2x₂(1 − x₁²), −2x₁(1 − x₂²)]ᵀ` — the recirculating wind,
  divergence-free, with `w·ν = 0` on the whole of `∂Ω`, so Assumption 1 holds
  (verified here: `w₁ = 0` on `x₁ = ±1` and `w₂ = 0` on `x₂ = ±1`);
* `y(x,0) = 1` if `x₁ = 1`, 0 otherwise;
* `y = 1` on `∂Ω₁ = {1}×[−1,1]` and `y = 0` on the rest of `∂Ω` (time-independent
  step data, discontinuous at the corners `(1,±1)`);
* `ŷ = e^{−10(1−x₁)}`, time-independent;
* `f = 0`; CN with `τ = h`; full-domain control and tracking.

LPS in §5.2: stated as the method of the paper ("In this work, we thus employ
the adjoint-consistent local projection stabilization") but not restated in
§5.2. **[B]: active wherever the patch Péclet number exceeds 1.**

## 8. Source-data status

| datum | E2 (heat) | E5/E6 (conv.–diff.) |
|---|---|---|
| PDE, Ω, `t_f`, f, BC, IC, ŷ | [A] | [A] |
| `ε`, `w` | [A] (ε = 1, w = 0) | [A] |
| β, l, τ rule | [A] | [A] |
| FE (Q1) | [A] | [A] |
| stabilisation | none (w = 0) | LPS with δ_k (7)–(8) [A in §2.1], used in §5.2 [B] |
| patch shape / `h_k` "in the direction of the wind" | n/a | [A] definition; implementation details (patch alignment, `h_k` for diagonal wind) [C] |
| solver tolerance | [A] 1e-6 relative residual | [A] |
| error metric | [A] (pointwise relative at the argmax) | none reported [C] |
| solution values | analytic [A] | **none** (raster figures only) [C] |
| objective values | none | none [C] |
| ξ (BE preconditioner) | [A] 1e-3 | n/a |

## 9. ex21 reproducibility classification

| # | class | reason |
|---|---|---|
| E2 heat Tables 1–3 | **A in ex20 — done**; **B in ex21** | ex20 reproduced it (Q1 harness, the exact constant shift `w = y − 1` that removes `g = 1`, `α_u = β`). ex21 at `v = 0` is bit-identical to ex20 on all 61 reference cases, so the result transfers. A run through the ex21 *production* path (Q2) needs only a preset with LP's analytic data and the same shift (no new numerics). |
| E3 Table 4 | not applicable | CPU of LP's own preconditioner blocks |
| E4 Shishkin mesh | **C** | non-uniform meshes are not exposed by ex20/ex21; it reports only iteration counts |
| E1 Fig. 1 | **C / not meaningful** | concerns LP's Schur approximation of LP's OTD system; ex21 uses a different block partition (§12). The test problem data are [C]. |
| E5 Figs. 2–3 | **C**, and only qualitative | needs a variable wind, inhomogeneous Dirichlet data and LPS (both figures are at l = 5, where the patch Péclet number is > 1); raster images only |
| E6 Tables 6–8 | **C** for the problem; **not a reproduction target** for the numbers | the same capabilities as E5; the reported quantity is MINRES counts of a different solver |

**What prevents E5/E6 in ex21 today** (verified in the code):
* the velocity is a constant pair (`ConvectionConfig{vx, vy}`, assembled from
  `velocity[2]`);
* the state boundary condition is `NATURAL` or `HOMOGENEOUS_DIRICHLET` only;
* elements are Q2 only in production (`InitializeStateSpace`; ex20 used Q1
  only in a harness);
* there is no stabilisation.

## 10. Boundary control

**What LP say** [A], Section 6 (Conclusions): "Future work includes extending
this new approach to problems with more complex PDE constraints, boundary
control problems, and problems with additional algebraic constraints on the
state and control variables."

* **Scope:** "this new approach" means LP's CN optimize-then-discretize
  all-at-once method with its symmetrising transformation and matching
  preconditioner. The statement is **general**: it is not tied to
  convection–diffusion, and it does not say Dirichlet or Neumann.
* **Related remarks in the same paper**, about *state* boundary conditions, not
  boundary control:
  * §2: "our method can be readily generalized to heat control problems with
    Neumann and mixed boundary conditions";
  * Assumption 1 and Remark 5: with Neumann parts and `w·ν ≠ 0`, `N` is no
    longer skew, the adjoint discretisation must be chosen so that the system
    stays symmetric, and the eigenvalue bound may degrade.
* **Cited prior work.** LP's reference 5 (Pearson, Stoll, Wathen, SIAM J. Matrix
  Anal. Appl. 2012; local copy `~/Downloads/pearson2012.pdf`) treats "the
  Neumann boundary control problem involving Poisson's equation and the heat
  equation", including **time-dependent Neumann boundary control** with
  backward Euler (its §2.3, §3.3).
  * All-at-once preconditioned *heat* boundary control is therefore in the
    literature LP cite.
  * No reference in LP's list is specifically about boundary control of
    convection–diffusion. That says nothing about the wider literature, which
    was not surveyed here.
* **Relevance to ex20/ex21.**
  * ex20 implements Neumann and Dirichlet boundary control for the heat equation.
  * ex21 rejects convection together with boundary control.
  * Implementing it would go **beyond the numerical setting LP demonstrate**
    (their experiments are distributed-only). With Neumann control on outflow
    or inflow boundaries, ex21's exact discrete transpose avoids the adjoint
    consistency question LP raise in Remark 5.
  * **No novelty is claimed.** The literature on boundary control of
    convection–diffusion was not searched.

## 11. Convection strength (Péclet numbers)

Maximum wind `max|w| = 2`, attained at the boundary midpoints `(0,±1)` and
`(±1,0)`, where the wind is aligned with a mesh axis. Element Péclet number in
ex21's convention, `Pe_h = |w| h/(2ε)` with `h = 2^{1−l}`:

| l | h | ε = 1/20 | ε = 1/100 | ε = 1/500 |
|---|---|---|---|---|
| 3 | 1/4 | 5.0 | 25 | 125 |
| 4 | 1/8 | 2.5 | 12.5 | 62.5 |
| 5 | 1/16 | 1.25 | 6.25 | 31.3 |
| 6 | 1/32 | 0.63 | 3.13 | 15.6 |
| 7 | 1/64 | 0.31 | 1.56 | 7.8 |
| 8 | 1/128 | 0.16 | 0.78 | 3.9 |

* LP's LPS switch uses the **patch** Péclet number, with the patch spanning
  2 elements, so roughly twice the values above. LPS is therefore switched off
  everywhere only for **ε = 1/20 at l = 7, 8** [B]. Every other entry of
  Tables 6–8, and both Figures 2–3 (l = 5), use stabilisation at least near the
  maximum wind.
* **Regime:** ε = 1/20 is moderate (element Pe up to 5). **ε = 1/100 and 1/500
  are genuinely convection-dominated** on the coarse and medium grids, with
  boundary layers at `x₁ = 1` (visible in Figs. 2–3).
* **For comparison,** every external ex21 benchmark so far has cell Péclet
  ≤ 0.22 (Mathematics 2025 Ex. 3: ≤ 0.025; Ex. 5: 0.046–0.185; EF 9.5/9.6:
  0.027–0.22).
* **ex21 has never been run at Pe_h > 1.** Unstabilised Q2 Galerkin is expected
  to produce oscillations there. That has not been measured, and no claim about
  ex21's high-Péclet behaviour is made.

## 12. Solver and preconditioner comparison

| aspect | LP | ex21 (SCHUR_SWEEP) |
|---|---|---|
| system | OTD CN system, symmetrised by `T`; saddle point with (1,1) block = tracking mass `A` | DTO KKT `[E S; T Eᵀ]`, non-symmetric |
| Krylov method | MINRES (symmetric), tolerance 1e-6 | FGMRES(30), tolerance 1e-8; inner Schur GMRES(30), 1e-6, cap 100 |
| block eliminated | the mass-type block `A`: `S = C + BA⁻¹Bᵀ` | the state operator: field split with `A00 = E` (exact forward sweep) and the Schur complement in the adjoint variable |
| Schur approximation | `Ŝ = (Λ21 + M̂)Â_D⁻¹(Λ21 + M̂)ᵀ` (matching: captures `C` and `BA⁻¹Bᵀ`; cross terms ≥ 0) | `Ŝ⁻¹ = right⁻¹ E left⁻¹`, `left = Eᵀ + √β M_ω`, `right = E + √β N` (matching between `Eᵀ` and the `S E⁻¹ T` term) |
| inner solves | 3 V-cycles of AMG (HSL_MI20 / AGMG) per diagonal block, block forward/backward substitution | *exact* time sweeps; each spatial block solved by GMRES(50) with GAMG to 1e-10 |
| theory | `λ(Ŝ⁻¹S) ⊂ [1/2, 1]`, independent of h, τ, β, ε (Assumption 1, Dirichlet) | none proved; empirically robust for colocated problems |
| reported robustness | 10–27 iterations over l = 3…8, β = 1e-1…1e-6, ε = 1/20…1/500 | colocated: 2 outer iterations for α_u = 1e-3…1e-11 (Math2025 Ex. 4, EF 9.5); the ex20 LP heat study: 2 outer iterations at every l and β |
| hardware | serial MATLAB | C++/PETSc, serial (MPI-capable production path) |

**Answers.**
1. **Related?** Yes. Both use the "matching" idea (Pearson–Wathen): approximate
   a two-term Schur complement by a product whose expansion reproduces both
   terms, with the regularisation entering as `√β` times a mass matrix next to
   the time-stepping operator. They are applied to **different block
   partitions** — LP eliminate the mass block, ex21 the state block — of
   **different discrete systems** (OTD vs DTO). They are related but not the
   same preconditioner.
2. **The same idea in a different implementation?** The same idea in a
   different *formulation*. ex21's factors are applied exactly (by sweeps)
   rather than by a few V-cycles. ex21's FGMRES tolerates its non-symmetric
   form; LP need the symmetrisation for MINRES.
3. **Untested regimes where LP are robust.** Yes:
   * small ε (1/100, 1/500; element Pe up to 125);
   * a variable recirculating wind;
   * β down to 1e-6 on fine grids (l = 8; about 3.3e7 unknowns in their CN
     setting);
   * non-uniform (Shishkin) meshes.

   ex21's spatial GMRES+GAMG sweeps have not been exercised on
   convection-dominated blocks; AGMG was chosen by LP precisely because it
   suits convection.
4. **EF 9.6-type degradation.** LP's Theorem 3 assumes the **same mass matrix in
   the tracking and control terms** (full-domain observation and control), so
   the setting is colocated by construction. The non-colocated EF 9.6
   configuration lies **outside the hypotheses** of their robustness result, and
   their experiments do not test it. LP's results neither predict nor
   contradict the EF 9.6 behaviour. They are consistent with ex21's empirical
   finding that colocated configurations are robust.
5. **A cleaner solver benchmark?** Yes, for *colocated full-domain* robustness:
   * E6's (l, β, ε) grid is a clean, well-specified robustness map, especially
     in ε, which no Aulisa/Gilliam example offers;
   * it is meaningful as an iteration-pattern comparison (robust or not), not
     as an iteration-count match;
   * it needs the capabilities of §9.

**Do not overfit to iteration counts.**
* Different discrete system (OTD vs DTO), element (Q1 vs Q2), Krylov method
  (MINRES vs FGMRES), inner solver (V-cycles vs exact sweeps), stopping rule
  (1e-6 vs 1e-8) and hardware. Equal counts or times are neither expected nor
  meaningful.
* **Mathematical reproduction** means convergence to LP's analytic solution
  (E2) or qualitative agreement of solution structure (E5).
* **Preconditioner reproduction** means only the *pattern*: boundedness of
  iterations in l, β and ε.

## 13. ex21 capability gap analysis (standard distributed transient convection–diffusion OC)

| area | status | evidence or gap |
|---|---|---|
| mathematical formulation (DTO, exact `Eᵀ`) | **COMPLETE / VALIDATED** | transpose ≤ 3e-17; FD gradient 1e-13…1e-15; the wrong adjoint fails; Hessian symmetry |
| state discretisation, Q2 | **COMPLETE / VALIDATED** (forward) | forward MMS: order 3 in L² for compatible data (2.5 with boundary-incompatible sources, inherited `M_elim I_h f`) |
| discrete adjoint | **COMPLETE / VALIDATED** (algebraic) | as above; *no OC-level manufactured or analytic solution with convection yet* |
| BE | **COMPLETE / VALIDATED** | forward MMS order 1; Math2025 Ex. 3 BE vs CN Richardson agreement |
| CN | **PARTIALLY VALIDATED** | forward MMS order 2; Math2025 Ex. 3/5 converged; heat OC (ex20/LP analytic) order 2 with an O(τ) adjoint terminal layer; no convection OC order study against an exact solution |
| distributed control on subregions / full domain | **COMPLETE / VALIDATED** (subregions, constant wind); full domain with convection **IMPLEMENTED BUT NOT EXTERNALLY VALIDATED** | Math2025 Ex. 3/4/5, EF 9.5/9.6 used subregions; full-domain control and tracking with convection only in the internal gates |
| distributed observation | **COMPLETE / VALIDATED** | same benchmarks |
| target handling (space–time analytic) | **COMPLETE / VALIDATED** | Gaussian, polynomial and smoothstep targets |
| disturbance handling (region-restricted source) | **COMPLETE / VALIDATED** | load gates; Math2025 and EF benchmarks |
| 1D | **COMPLETE / VALIDATED** | Math2025 Ex. 3 (same-problem reproduction), Ex. 4 |
| 2D | **PARTIALLY VALIDATED** | Math2025 Ex. 5 (periodic regime), EF 9.5/9.6; no exact-solution OC test |
| solver correctness | **COMPLETE / VALIDATED** | true-residual checks, independent march, MPI consistency |
| solver robustness | **PARTIALLY VALIDATED** | robust when colocated; documented non-colocated small-α_u stress case (EF 9.6) |
| low / moderate Péclet (≤ 0.25) | **VALIDATED** | every external benchmark |
| high Péclet (> 1) | **NOT VALIDATED**, and no stabilisation | never run; standard Galerkin only |
| variable convection field | **NOT IMPLEMENTED** | constant `(vx, vy)` only |
| inhomogeneous Dirichlet state data | **NOT IMPLEMENTED** in ex21 production | homogeneous only; ex20 harnesses used exact shifts or lifting |
| boundary control (Neumann, Dirichlet) with convection | **NOT IMPLEMENTED** (rejected by design) | available for the heat equation in ex20 |
| reaction term | **NOT IMPLEMENTED**; not needed for LP | — |
| 3D | **OUT OF CURRENT SCOPE** | 2D box meshes only in the campaigns |

## 14. Recommended minimal reproduction campaign

Each test exercises something different; together they close the most
important gaps. Costs assume serial runs on this node.

**T0 (optional, cheap) — algebraic cross-check of the §5 derivation.**
* **What:** on a tiny 1D or 2D grid, assemble LP's CN system (13)–(15)
  literally, next to ex21's CN KKT system, and verify the claimed
  correspondences: identical state rows under the control identification, and
  the stated adjoint differences.
* **Why:** it turns §5's [B] derivation into [A].
* **Validates:** the claim "different discretisation of the same problem".
* **Code:** a stand-alone script (Python/NumPy on dumped matrices); no solver
  change.
* **Cost:** minutes.
* **Acceptance:** state rows equal to round-off; adjoint rows differ exactly by
  the tracking-quadrature terms.

**T1 — LP §5.1 heat control through the ex21 production path (E2).**
* **Why:** an analytic-solution OC benchmark on the exact ex21 binary and
  element (Q2). It anchors CN OC convergence in ex21 itself rather than
  relying on bit-identity with ex20.
* **Validates:** full-domain control and tracking, CN/BE OC convergence, the
  `α_u = β` mapping, and the sign `p = −p_LP`.
* **Data:** all [A].
* **ex21 support:** yes (`v = 0`, homogeneous Dirichlet after the exact shift
  `w = y − 1`).
* **Code:** a preset with LP's analytic ŷ and initial state (data overrides);
  no new numerics.
* **Cost:** l = 3…6, small (ex20: CN l = 6 in about 15 s).
* **Compare:** relative `L²_t L²_x` errors of y, p and u against the analytic
  solution; observed orders; ratios to LP Tables 1–3 (boundedness only).
* **Acceptance:** CN state order ≈ 2 under `τ = h` (Q2 spatial error is
  smaller than Q1, so the temporal error dominates); BE order ≈ 1 in τ; the
  adjoint terminal layer documented as in ex20.

**T2 — LP-style exact OC solution *with convection* (source-inspired, not a
literature reproduction).**
* **Why:** this is the most important gap. It is the first exact optimal-control
  solution (state and adjoint) under convection, testing CN second order of the
  coupled OC system and Q2 spatial order with `Eᵀ` — something no source
  benchmark provides.
* **Construction:** LP §5.1 builds `ŷ` so that a chosen `(y, p)` is optimal.
  With constant `w`:
  * choose `y`, and choose `p` with `p(t_f) = 0` and `p = 0` on `∂Ω`;
  * set `u = p/β`;
  * set `f = y_t − ε∆y + w·∇y − u` (LP's formulation allows `f`);
  * set `ŷ = y + p_t + ε∆p + w·∇p`.

  This mirrors LP's construction and is labelled source-inspired.
* **Validates:** convection OC convergence in space and time, and the adjoint
  under convection.
* **Data:** all analytic.
* **ex21 support:** yes. Analytic `f`, `ŷ` and `y₀` go through the existing data
  overrides; the boundary data vanish if `y` is chosen to vanish on `∂Ω`.
* **Code:** a preset or driver only.
* **Cost:** small (2D, l = 3…6).
* **Compare:** errors in y, p and u; orders in h and τ; Pe_h ≤ 1 first, then
  ε = 1/20 at fine levels.
* **Acceptance:** Q2 spatial order 3 for compatible data and CN order 2,
  away from the documented adjoint terminal layer.

**T3 — LP §5.2 problem at the LPS-free resolution (E5/E6 subset).**
* **Why:** the first comparison with a published convection-dominated-capable
  setting. ε = 1/20 at l = 7, 8 needs no stabilisation in LP's own scheme, so
  ex21's unstabilised Galerkin solves *their* discrete problem class there
  (Q1 vs Q2 and OTD vs DTO aside).
* **Validates:** variable wind, inhomogeneous step Dirichlet data, full-domain
  colocated control, and solver robustness in l and β.
* **Data:** [A] (problem), [B] (LPS inactive at these levels).
* **ex21 support: no** — needs a variable wind and inhomogeneous Dirichlet data.
* **Code:** see §16.
* **Cost:** l = 7 has 128×128 elements, about 66k Q2 DoF per time level and
  128 steps. That is substantial; begin with l = 5, 6, using a Q1 harness or
  l-shifted Q2.
* **Compare:**
  * qualitative `y(·,1)` and `p(·,1)` structure against Figs. 2–3 (range, layer
    location, sign pattern of `p`);
  * outer, Schur and spatial iterations across β = 1e-1…1e-6 (pattern only);
  * objective components, reported as new data, since LP give none.
* **Acceptance:** converged true residuals; bounded iteration counts in l and
  β; solution structure consistent with the figures.

**T4 — ε-robustness and the high-Péclet limit (E6 rows ε = 1/100, 1/500).**
* **Why:** it establishes honestly where unstabilised ex21 stops being reliable
  and whether the SCHUR_SWEEP spatial sweeps (GMRES+GAMG) degrade on
  convection-dominated blocks.
* **Validates:** the high-Péclet behaviour of both the discretisation and the
  solver.
* **ex21 support:** the same as T3. Without LPS the solutions at Pe_h > 1 are
  expected to oscillate. That is a *finding to document*, not a reproduction
  target.
* **Cost:** moderate, l = 3…6.
* **Compare:** iteration counts against ε; a solution oscillation diagnostic
  (overshoot beyond [0,1] in y); spatial GMRES counts.
* **Acceptance:** a clear map of the regime (Pe_h) in which ex21 results can be
  trusted. It also informs the decision on implementing LPS (adjoint-consistent
  by LP's argument) as a separate future task.

**T5 (cheap once T3 exists) — β-robustness map in convection.** The analogue
of Tables 6–8 restricted to Pe_h ≤ 1 (ε = 1/20 at l = 6–8, and constant-wind
T2 problems), β = 1e-1…1e-6. It confirms that colocated full-domain
convection problems stay in ex21's robust regime down to small β.

Ordering: **T2 first** (largest value, no new capability), then **T1**, then
T3–T5 after the two capability additions. T0 can accompany T2.

## 15. Tests not worth reproducing

* **Exact MINRES iteration counts and CPU times (Tables 1–8):** different
  solver, system, element, tolerance and hardware. Only the robustness pattern
  is meaningful.
* **Table 4** (CPU per preconditioner block): a property of LP's preconditioner.
* **Fig. 1** (eigenvalues of `S̃_int⁻¹ S_int`): a property of LP's Schur
  approximation of LP's OTD system; the test problem's data are not stated.
* **Table 5** (Shishkin mesh): requires non-uniform meshes and reports only
  solver data. Low value relative to cost.
* **BE vs CN CPU competition** (the paper's Tables 1–3 headline): already
  answered qualitatively by ex20 (CN about 25× faster at l = 6). It is a
  solver-efficiency claim about LP's own BE preconditioner.
* **Re-running E2 with Q1** in ex21: ex20 did this; T1 with Q2 adds what is new.

## 16. Code changes

**REQUIRED BEFORE THE BENCHMARK**
* **For T0, T1, T2: none.** Only presets or drivers using the existing data
  overrides (analytic f, ŷ and y₀; full-domain control and tracking; CN
  trapezoidal source). T1 uses the exact constant shift; T2 chooses data
  vanishing on `∂Ω`.
* **For T3, T4, T5:**
  1. a **variable velocity field `w(x)`** in the convection assembly. The Gauss
     point loop must evaluate `w` instead of using `(vx, vy)`. `Eᵀ` stays exact
     automatically (MatMultTranspose), but the orientation gate must be
     generalised;
  2. **inhomogeneous, time-independent Dirichlet state data**: either a lifting
     in the driver (as in the ex20 actuator harness) or a production feature
     (non-zero constrained rows plus their right-hand side contribution).

**NICE TO HAVE**
* a **Q1 option** (an ex20-style harness or a configuration switch), for
  like-for-like element comparison with LP and cheaper large-l runs;
* **LPS stabilisation** (adjoint-consistent, symmetric `W`). It is needed
  before any quantitative statement at Pe_h > 1, but it is a separate method
  addition, to be decided after T4;
* a mesh-grading option (Shishkin), only if Table 5 is ever wanted.

**NOT REQUIRED** (verified in the code)
* CN tracking, control and source quadrature: consistent and documented
  (tracking `w = (1,…,1,½)`, control per interval, source trapezoidal like LP);
* the exact transpose and the convection sign convention (identical to LP's
  `N`);
* full-domain control and tracking (`--control-full-domain`, tracking box = Ω;
  `N = M_ω = M`);
* time-dependent analytic targets;
* the `α_u = β` mapping (`η = 1`, `α_track = 1`);
* a reaction term (LP have none).

## 17. Is ex21 already a mature distributed convection–diffusion OC application?

**Not yet, in the sense LP's class demands; yes, within a narrower, clearly
defined scope.**

* **Within scope** (constant wind, homogeneous Dirichlet data, distributed
  control on subregions or the whole domain, low cell Péclet number, BE/CN,
  1D/2D), ex21 is:
  * correct by construction (exact `Eᵀ`) and verified algebraically;
  * forward-convergent at the expected orders;
  * externally corroborated by a same-problem 1D reproduction (Math2025 Ex. 3)
    and a periodic-regime 2D same-problem agreement (Ex. 5);
  * consistent in cross-method tail scales (EF 9.5/9.6).
* **What the standard class still needs:**
  * an **exact optimal-control solution test with convection** (T2);
  * **variable winds** and **inhomogeneous Dirichlet data** (T3);
  * evidence in the **convection-dominated regime** with a decision on
    stabilisation (T4);
  * a broader **solver-robustness** map in ε (T4/T5).

## 18. What remains before ex20 and ex21 are parallel heat and convection–diffusion AAO applications

ex20 (heat) offers distributed, Neumann-boundary and Dirichlet-boundary control,
with BE/CN, analytic-solution benchmarks (including LP §5.1), and several solver
families. For ex21 to be its convection–diffusion counterpart:

1. **An exact OC-solution benchmark with convection** (T2), and T1 on the ex21
   production path. These are the analogues of ex20's analytic heat benchmarks.
2. **Variable wind and inhomogeneous Dirichlet data** (T3 prerequisites),
   which ex20 handled through harness-level shifts or liftings.
3. **A high-Péclet assessment and a stabilisation decision** (T4). This has no
   heat analogue but is intrinsic to convection–diffusion.
4. **Boundary control under convection** (Neumann first, then Dirichlet).
   Currently rejected. It requires the exact transpose of the boundary-control
   operator (already exact in ex20) and attention to inflow/outflow boundaries
   (LP Remark 5). It goes beyond LP's demonstrated setting; no novelty is
   claimed.
5. **Solver coverage**: SCHUR_SWEEP is ex21's validated path. The
   non-colocated small-α_u limitation (EF 9.6) remains documented, and the
   boundary-reduced or Dirichlet-reduced solvers of ex20 have no convection
   counterpart yet.

Items 1–2 are small and enable most of the external evidence; items 3–4 are
genuine method work.

---

## Post-audit validation status (T0, T1, T2 — added after the audit; the sections above are unchanged)

The reports are in `leveque_pearson/`; `validation_summary.csv` lists 75/75 checks as PASS.

| test | report | outcome |
|---|---|---|
| **T0**: algebraic check of the LP CN system against ex21's CN system | `leveque_pearson/T0_CN_MATRIX_CHECK.md` | 49/49 checks pass, on generic matrices and on the operators dumped from the ex21 binary. **§5's [B] derivation is now [A].** One sharpening: under the positional identification `p ↔ −λ`, LP's adjoint operator `Λ12` *is* `dt·Eᵀ` entry for entry. The difference lies only in the tracking (1,1) block (LP: interval-averaged, bidiagonal; ex21: nodal trapezoid, diagonal), in the control (2,2) block (bidiagonal vs diagonal) and in the tracking rhs. For the same data the two solutions differ at O(dt²). Conclusion: **same continuous problem, different discrete KKT systems**. |
| **T2**: exact optimal-control solution with `v = (1, ½)` (source-inspired, not a literature problem) | `leveque_pearson/T2_EXACT_CONVECTION_OC.md` | **Q2 space:** order 3.00 for compatible data (state; adjoint and control on `t ≤ T/2`), and 2.5 for data with a nonzero boundary trace (the inherited `M_elim I_h f` load). **CN time:** order 1.97 (adjoint and control, full interval); a terminal adjoint slot with relative error O(dt) and absolute order rising to 1.89. **BE time:** order 1.00. **Algebra:** transpose 7e-18, FD gradient 5e-13, Hessian symmetry 4e-14, independent march ≤ 4e-13. **Péclet:** cell Pe ≤ 0.56 (not convection-dominated). |
| **T1**: LP §5.1 heat benchmark through the ex21 production path (Q2) | `leveque_pearson/T1_EX21_LP_HEAT.md` | O(h²) under both of LP's rules (adjoint 2.00 / 2.02 on `t ≤ T/2`). CN temporal errors equal ex20's to 0.1–3.5 %. `v = 0` with the transposed path on is identical to it off. |

Effect on §13:
- "discrete adjoint" and "CN" move to **COMPLETE / VALIDATED** for constant wind
  and homogeneous Dirichlet data at low Péclet number;
- "2D" moves to **VALIDATED** for exact-solution convergence (T2);
- "full domain with convection" is now **VALIDATED**.

§14's T3–T5 and §16's REQUIRED items (variable wind, inhomogeneous Dirichlet
data) are unchanged.

**Code added for this phase** (opt-in, no default changed):
- `include/ExactOcBenchmark.{hpp,cpp}`;
- the options `--exact-oc`, `--exact-oc-alpha-u` and `--exact-oc-dump` in
  `CommandLine.cpp` / `Options.hpp`;
- the dispatch in `ex21.cpp` and the source entry in `CMakeLists.txt`.

**Regression:** the ex21 default path matches the ex20 goldens, 61/61 identical.
The Mathematics 2025 Ex. 3 and Ex. 4 archived runs are byte-identical (3/3,
`scripts/regress_ex3_ex4.sh`). Both logs are in `leveque_pearson/logs/regression/`.

## Post-validation note (T3: variable wind, inhomogeneous Dirichlet, LP §5.2 at low Péclet)

Report: `leveque_pearson/T3_VARIABLE_WIND_DIRICHLET.md` (59/59 and 51/51 checks
pass). Two §16 REQUIRED items are now implemented and validated:
- variable velocity `w(x)`, through `--wind`, with the constant path byte-identical;
- time-independent inhomogeneous Dirichlet state data by lifting.

LP §5.2 at ε = 1/20 was solved at the node set and time step of LP l = 7. There,
LP's patch Péclet number is ≤ 0.62 and their LPS is off (§11 re-verified from
the LP definition: `t3/lp52_patch_peclet.py`).

**β = 1e-2:**
- structural agreement with Fig. 2: y layer at x1 = 1; p ∈ [−0.021, 0.006];
- no oscillation;
- only corner-local (1,±1) mesh sensitivity from the discontinuous data.

**Solver:** robust over β = 1e-2…1e-6 (1–2 outer iterations).

**Open items:**
- **Small-β adjoint feature:** at β ≤ 1e-4 there is an under-resolved
  boundary/corner adjoint feature at Pe ≤ 0.63, caused by tracking/boundary
  mismatch, not convection.
- **Spatial sweep solver at high Pe:** it fails on BE convection-dominated blocks
  at cell Pe ≥ 2.5. This is recorded for the high-Pe phase; LPS is not
  implemented.

## Post-validation note (T3b, T4)

**T3b.** The affine tracking term of inhomogeneous Dirichlet data is now
implemented and validated (`leveque_pearson/T3B_AFFINE_DIRICHLET_TRACKING.md`,
34/34 checks).

**T4.** The unstabilised high-Péclet map of LP §5.2
(`leveque_pearson/T4_HIGH_PECLET_UNSTABILISED.md`) separates three findings:
- **Discretisation:** at β = 1e-2, visible Galerkin oscillations appear only
  at `Pe_cell ≥ 25`. With weak control they appear from 12.5–62.5 (at t = 1),
  plus a persistent 5–10 % transient undershoot at ε = 1/500.
- **Spatial solver:** GMRES+GAMG fails for `Pe_cell ≳ 2.5` unless the block is
  mass-dominated. Every ε ≤ 1/100 level up to nx = 64 fails.
- **KKT strategy:** with exact sweeps it converges in 1–2 outer iterations
  everywhere.

**Next:** a spatial preconditioner for convection-dominated blocks, then the
LPS decision.
