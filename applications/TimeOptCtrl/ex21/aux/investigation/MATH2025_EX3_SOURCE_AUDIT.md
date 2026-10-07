# Mathematics 2025, Example 3 — source audit and mapping to ex21

**Source.** Aulisa, Chierici, Gilliam, *A Least-Squares Control Strategy for
Asymptotic Tracking and Disturbance Rejection Using Tikhonov Regularization and
Cascade Iteration*, Mathematics 13 (2025) 3707,
`~/Downloads/mathematics-13-03707.pdf`, Section 8, Example 3, pp. 23–27,
eqs. (65)–(66), (78)–(84), Figures 25–33.

Tags: **[A]** explicitly stated · **[B]** inherited / inferred (evidence given)
· **[C]** missing.

---

## 1. Source specification

| datum | value | tag | evidence |
|---|---|---|---|
| PDE | `z_t = A z + B_d d + B_in u` | [A] | (79) |
| operator | `A = a d²/dx² − v d/dx`, `v > 0` | [A] form | (65), "As in the previous examples, we write the state operator A for our plant as in (65)" |
| **`a`, `v`** | **`a = 2`, `v = 2`** | **[B]** | not restated in Ex. 3. Set in Ex. 1 ("In the system (65), we have set a = 2, v = 2"), restated with the same values in Ex. 5 ("we have set a = 2, u = v = 2"); no other value appears anywhere. Numerically confirmed (`MATH2025_EX3_BENCHMARK.md` §6): `(a, v) = (2, 2)` matches Fig. 33 to RMS 6.2e-6 (extraction noise); `(1,1)`, `(2,0)`, `(2,1)`, `(4,2)` miss by 4.8e-4, 2.6e-3, 1.25e-3, 1.05e-3, i.e. **78–420× worse** |
| domain | `Ω = [0, 1]` | [A] | p. 24 |
| state BC | homogeneous Dirichlet at both ends | [A] | "We impose Dirichlet boundary conditions at the endpoints"; (66) |
| IC of the **plant** (cascade) | `z(x, 0) = x(1 − x)` (BC-compatible) | [A] | (80); confirmed by Fig. 31: plotted `‖e(0)‖_c = 0.10591`, exact `‖x(1−x)‖_{L²(Ω_c)} = 0.10597` |
| IC of the **optimal-control run** | **`z(x,0) = 0`** | **[B]** | not stated. Both curves of Figs. 32/33 start at `|E| ≤ 2e-7`, impossible from `x(1−x)` (0.106 at `t = 0`; control on `Ω_b` cannot act instantaneously on `Ω_c`). ex21 with `z₀ = 0` matches Fig. 33 to RMS 6.2e-6; with `x(1−x)` it misses by 0.106 at `t = 0` (RMS 1.7e-2 on `[0,2]`) and agrees only after `t ≈ 0.5` (benchmark report §5) |
| `Ω_b` | `[1/5, 2/5]` | [A] | Figure 25 / p. 24 |
| `Ω_c` | `[1/2, 7/10]` | [A] | idem |
| `Ω_d` | `[4/5, 9/10]` | [A] | idem |
| `B_in` | extension by zero `E_b : L²(Ω_b) → L²(Ω)` (infinite-dimensional input) | [A] | "B_in = E_b"; "the input operator is an extension operator" |
| `C` | restriction `C ψ = χ_{Ω_c} ψ` (infinite-dimensional output) | [A] | "C = χ_{Ω_c}(x)", "output operator is a restriction operator" |
| `B_d` | extension by zero `E_d` | [A] | "B⁰_d(x) = E_d" |
| reference | `r(x, t) = x sin(t) F(t − 1.5, 1.5)` on `Ω_c` | [A] | (78) |
| disturbance | `d(x, t) = x² sin(2t) F(t − 1.5, 1.5)` on `Ω_d` | [A] | (78) |
| `F` | Ex. 3 says only "a smooth function that, together with its first four derivatives, is zero at t = 0"; the full definition is Ex. 1's: degree-9 polynomial on `[t*−ε, t*+ε]`, `F(−ε) = 0`, `F(ε) = 1`, derivatives 1–4 zero at both ends | [A]/[B] | (69) and p. 16; with `t* = ε = 1.5` the window is `[0, 3]`, consistent with "zero at t = 0". Unique: `F = 126τ⁵ − 420τ⁶ + 540τ⁷ − 315τ⁸ + 70τ⁹`, `τ = t/3` (verified symbolically) |
| error norm | `‖e(t)‖_c = (∫_{Ω_c} \|r − Cz\| dx)^{1/2}` **as printed** | [A] (typo) | the square is missing; the plotted values are the **L²(Ω_c) norm** [B]: Fig. 31 `0.10591` vs L² `0.10597` (the printed formula would give 0.214) |
| horizon `T` | 20 | [B] | every Ex. 3 figure ends at `t = 20`; Fig. 32 shows a terminal deviation of `E_oc` from the (horizon-free) cascade error in the last ≈ 0.3 time units, consistent with an optimal-control horizon ending at 20; ex21 with `T = 25` reproduces the horizon-free cascade curve on `[19.7, 20]` to 3e-6 while `T = 20` shows a terminal effect (benchmark report §5) |
| plotted sampling | uniform, `Δt_out = 0.05` | [B] | 401 vertices on `[0, 20]`, 41 on `[0, 2]` |
| discretisation | COMSOL Multiphysics 6.1, "basic" implementation; the OC system solved by embedding time as an auxiliary coordinate; mesh and time step not given | [A]/[C] | pp. 7–8, 27 |
| cascade | levels `j = 0, 1, 2`, plant driven by `u² = Σ u_j`; `α = 10⁻³ … 10⁻⁹` (Figs. 26–29), `α = 10⁻⁹` for Figs. 30–33 | [A] | p. 24–26 |
| `β` | not stated for Ex. 3 | [C] | (`β = 0.25` in Exs. 1, 5) |
| OC functional | `J_ε = ∫₀ᵀ [ ∫_{Ω_c} ½(z − r)² dx + (ε/2) ∫_{Ω_b} u² dx ] dt` | [A] | p. 26–27 |
| `ε` | `10⁻⁹` | [A] | "with ϵ = 10⁻⁹"; caption of Fig. 32 "ϵ = α = 10⁻⁹" |
| OC system | (82) `z_t = Az + B_d d + B_in(−λ/ε)`; (83) `−λ_t = A*λ + χ_{Ω_c}(r − z)`; (84) `z = λ = 0` on `∂Ω`, `λ(x, T) = 0` | [A] | p. 27 (the `λ` after `A*` is lost in text extraction, visible in the PDF) |
| printed sign | varying the printed `J_ε` (with `+∫λ(z_t − Az − B_in u − B_d d)`) gives (83) exactly but `u = +λ/ε`, not `−λ/ε` | note | one printed sign is a typo; the minimiser of `J_ε` is unique, so it does not affect the comparison |

### Figures used

| figure | content (caption, verified against the PDF) | role here |
|---|---|---|
| 30, 31 | `‖E₀‖_c` (blue), plant `‖e‖_c` (red), `α = 10⁻⁹`, `t ∈ [0,20]` / `[0,2]` | IC / norm checks |
| **32** | **`‖E_oc‖_c` (red) and cascade `‖e₂(t)‖_c` (blue), `ε = α = 10⁻⁹`, `t ∈ [0, 20]`, `y ∈ [0, 0.06]`** | **primary target** |
| **33** | same, `t ∈ [0, 2]`, `y ∈ [0, 0.05]` | **primary target (zoom)** |

No table and no printed number exists for Example 3: the comparison is against
vector-extracted curves (§3).

## 2. Mapping of (82)–(84) to ex21

ex21 minimises

    J = (α_track/2) Σ_n dt w_n ‖y_n − I_h r(t_n)‖²_{M_ω} + (α_u/2) Σ_n dt u_nᵀ N u_n
    s.t. E Y = rhs_y + η (I ⊗ N) U.

| source | ex21 | check |
|---|---|---|
| `∫₀ᵀ∫_{Ω_c} ½(z − r)²` | `α_track = 1`, `M_ω` = mass over `Ω_c` (elements with centroid in `[0.5, 0.7]`), `y_d = I_h r` | factor ½ identical; **integral, not average** (both sides) |
| `(ε/2)∫₀ᵀ∫_{Ω_b} u²` | `α_u = ε = 10⁻⁹`, `N` = mass over `Ω_b`, `u` a Q2 field on the `Ω_b` elements | `ε` multiplies exactly the full `L²(Q_b)` norm, with the same ½: **no rescaling** |
| `B_in u = E_b u` | `η N u`, `η = 1` | `(N u)_i = ∫_{Ω_b} u φ_i` = Galerkin form of `χ_{Ω_b} u` |
| `B_d d = E_d d` | `M_d I_h d` (new `source_regions = {[0.8, 0.9]}`) | `(M_d f)_i = ∫_{Ω_d} f φ_i`, no indicator interpolated |
| `A = 2∂ₓₓ − 2∂ₓ` | `ν = 2`, `vx = +2` | ex21 solves `y_t − νΔy + v·∇y`; sign convention verified (G1, `EX21_CONVECTION_VALIDATION.md`) |
| Dirichlet `z = 0` on `∂Ω` | `--state-bc dirichlet` (row/column elimination) | trace = 0 to 7e-15 (gate) |
| `z(x, 0)` | `--ex3-initial zero` (OC run, primary) or `source` (`x(1−x)`) | gate: `‖I_h z₀ − z₀‖ = 3e-17` (`x(1−x)` ∈ Q2) |
| `λ` backward, `λ(T) = 0` | discrete adjoint `Eᵀ P = …`, block upper-triangular in time: `P` has no slot beyond `N`, i.e. **`λ(T) = 0` is the discrete terminal condition** of the discretize-then-optimize system | no separate `A*` is used; `Eᵀ` exact (gates) |
| `u = ±λ/ε` | `u = −(η/α_u) p` (ex21's dt-scaled, sign-convention multiplier) | ex21 minimises `J_ε` directly; the printed sign typo is immaterial |
| horizon `T` | `T = 20` (`--T`, default of the preset) | [B] |

There is **no functional-scaling mismatch**: `α_track = 1`, `α_u = ε` exactly.
The only discretisation-level differences are the ones inherent to
discretize-then-optimize vs. the source's optimize-then-discretize COMSOL
solve. They vanish under refinement everywhere except in the terminal layer of
the `λ(T) = 0` optimum, which no time step used here (or, evidently, by the
source) resolves at `ε = 10⁻⁹` (benchmark report §5).

Configuration (preset `--math2025-ex3`, `include/Math2025Benchmark.cpp`): 1D
Q2 (`EDGE3`) mesh on `[0, 1]` with `nx` a multiple of 10 (every region endpoint
is an element face, so all region measures are exact), `ν = 2`, `vx = 2`,
homogeneous Dirichlet, distributed control on `[0.2, 0.4]`, tracking on
`[0.5, 0.7]`, source on `[0.8, 0.9]`, `α_track = 1`, `α_u = 10⁻⁹` (not
overridable), `η = 1`, `T = 20` (default). Free: `--nx`, `--steps`, `--T`,
`--time-scheme`, preconditioner, `--ex3-initial`, and the sensitivity-only
`--ex3-coefficients a v`.

## 3. Figure extraction (Phase 3)

Method (`math2025_ex3/scripts/extract_figs.py`, `svg_paths.py`): page 26 →
`pdftocairo -svg` → every stroked path with its accumulated transform. The
axis map comes from the light-grey **grid lines** (5 vertical, 7 / 6
horizontal), whose values are the tick labels read from the PDF **text layer**
(`pdftotext -bbox`), never from a raster. Curves are identified by stroke
colour as the caption states (red `E_oc`, blue `e₂`). Each curve is ONE
polyline whose vertices are the plotted samples.

| quantity | Fig. 32 | Fig. 33 |
|---|---|---|
| time range | `[0, 20]` (grid at 0, 5, 10, 15, 20) | `[0, 2]` (0, 0.5, …, 2) |
| y range | `[0, 0.06]` | `[0, 0.05]` |
| vertices blue / red | 401 / 397 | 41 / 39 |
| output step | 0.05 (snapped; max snap 3.0e-3 in t, quantized page coordinates) | 0.05 (snap 3.5e-7) |
| grid-fit residual x / y | 2.0e-4 / 9.3e-7 | 6.7e-7 / 9.2e-7 |
| value quantum (smallest distinct Δy) | 2.5e-6 | 2.1e-6 |
| `max |blue − red|` | **6.6e-4** (at t → 20) | 1.7e-5 |

Missing red vertices are collinear points dropped by the plot export; they are
reconstructed by linear interpolation. **Estimated extraction uncertainty:
≈ ±3e-6 in value** (quantum + calibration), negligible in time. The two curves
are not literally overlaid: they are separate polylines, identical to ≈ 1e-5
up to `t ≈ 19.6`, then the red one rises above the blue by up to 6.6e-4 at the
end of the horizon. CSVs: `results/math2025_ex3_fig{30,31,32,33}.csv`;
calibration record `results/math2025_ex3_extraction.json`.
