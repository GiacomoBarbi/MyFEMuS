# Convection–diffusion benchmark inventory for ex21

Sources: Mathematics 2025 (Aulisa, Chierici, Gilliam, 13, 3707), Examples 1–6;
ErrorFeedback2026 (3 Oct 2026 manuscript), Examples 9.5–9.6. The
actuator-dynamics paper (`~/Downloads/Wiley_Actuator_Dynamics.pdf`) was
re-checked: it mentions convection–diffusion generators `A = Δ − v·∇` only as
a theoretical extension remark (velocity as a lower-order sectorial
perturbation); all its numerical examples (6.1–6.4) are ODE/heat plants, so it
contributes **no** convection–diffusion benchmark.

Classification: **A** quantitative now · **B** quantitative with an explicit
assumption or figure extraction · **C** structural only.

**Source quantitative data.** Mathematics 2025 does not print absolute
numerical tracking values for its convection–diffusion examples. Its strongest
quantitative information comes from vector figures (extracted) and from ratio
tables (Tables 1–2: ratios of successive cascade errors). ErrorFeedback
9.5–9.6 do provide absolute tail values in Tables 9–10 [A]. However, the
initial condition is unstated [C], and the exact subregion coordinates are not
given textually: they are [B], taken from schematic figures and confirmed by
the user. Every quantitative comparison therefore rests on at least some [B]
data.

## Final classification (after the Example 5 / ErrorFeedback 9.5–9.6 task, 4 Oct 2026)

| benchmark | final class | what was compared | key numbers | report |
|---|---|---|---|---|
| **M25 Ex3** (1D non-colocated, source OC at ε = 1e-9) | **same-problem source-OC reproduction; gold-standard convection benchmark** | ex21 AAO optimum vs the source's own optimal-control error curve | RMS 6.2e-6 on `[0,2]`; 3.4e-5 on `[0,19.7]` | `MATH2025_EX3_BENCHMARK.md` |
| **M25 Ex4** (1D colocated) | **structural colocated benchmark + cascade scale cross-check; no source OC** | no source OC; colocated floor removal and figure-derived cascade tail scales | every cascade `e₅` attained at `α_u ≈ 0.05–0.2 α` | `MATH2025_EX4_BENCHMARK.md` |
| **M25 Ex5** (2D non-colocated, source OC at ε = 1e-7) | **B-strong geometry corroboration; same-problem 2D comparison; strong periodic-regime agreement; unexplained source transient mismatch** | ex21 AAO optimum vs the source `E_oc` (Fig. 46); free response vs Fig. 45 | `(W,H) = (π,1)` beats `(1,1)`, `(π,2)`, `(2π,1)` by 3–1000×; periodic `[3,9.5]` L∞ 1.25e-3 (2.4 %), peaks 0.4 %, phase 0; **unexplained source transient mismatch on `[0,3]`**: under the reconstructed source data, the published transient is not reproduced by the converged ex21 optimum (it is inconsistent with that optimum unless the source control had `‖u‖ ≳ 95` vs 14.5) | `MATH2025_EX5_BENCHMARK.md` |
| **EF 9.5** (2D colocated) | **B cross-method tail-scale benchmark** (colocated; IC-insensitive tail; source controller parameters not identified with `α_u`) | AAO tail vs Table 9 REFC levels on the asserted geometry | tail IC-insensitive (≤ 4e-6); every `E_j` attainable; `α_u(E₅) ≲ 9.3e-11` (finest mesh; **upper bound**, small-`α_u` tail not mesh-converged) | `ERRORFEEDBACK_95_BENCHMARK.md` |
| **EF 9.6** (2D non-colocated) | **B for the qualitative practical-saturation phenomenon; C for the published level `E₅`** | AAO tail vs Table 10 (primary; Fig. 34 qualitative) | practical saturation over the tested range α_u = 1e-6 … 1e-10. Smallest completed point 1e-10: tail 0.152, residual 8.0e-9, nx 32 only; spatial convergence established at α_u = 1e-7, not at 1e-10. 1e-11 aborted for solver cost, no value used. Factor ≈ 3 above `E₅ = 0.0522`; Table 10 not quantitatively reproduced on the asserted geometry. Conditional estimate (periodic-interior approximation): reaching `E₅` would need time-RMS ≈ 9.7e3 or more. No asymptotic-floor or unattainability claim | `ERRORFEEDBACK_96_BENCHMARK.md` |
| **Solver robustness (all of the above)** | **localised stress case, not a general formulation failure** | SCHUR_SWEEP outer / Schur / spatial iterations, true residuals | colocated (Ex4, EF 9.5): 2 outer its to α_u = 1e-11; non-colocated Ex3: ≤ 7; EF 9.6: 3 → 70 outer its (1e-7 → 1e-10), Schur split always capped, spatial solves flat (16–19) | `AULISA_GILLIAM_SOLVER_ROBUSTNESS_AUDIT.md` |
| M25 Ex1 / Ex2 | B / B (not implemented) | scalar outputs need a per-channel tracking operator | — | `MATH2025_EX1_SOURCE_AUDIT.md` |
| M25 Ex6 | C (not implemented) | colocated 2D, figures only; geometry now `(π,1)` [B] via Ex5 | — | — |
| **Wiley actuator-dynamics paper** | **no convection–diffusion numerics** | convection mentioned only as a theoretical remark; examples 6.1–6.4 are ODE/heat | — | this file |

## Source data matrix (updated)

| | M25 Ex1 | M25 Ex2 | **M25 Ex3** | M25 Ex4 | M25 Ex5 | M25 Ex6 | EF 9.5 | EF 9.6 |
|---|---|---|---|---|---|---|---|---|
| dimension | 1D | 1D | **1D** | 1D | 2D | 2D | 2D | 2D |
| domain | `[0,2]` | `[0,2]` | `[0,1]` | `[0,ℓ]`, `ℓ=1` | `[0,W]×[0,H]`; **`(π,1)` [B], corroborated** (Figs. 45–46) | as Ex5 | `(0,π)×(0,1)` | idem |
| PDE / diffusion | `a z_xx − v z_x`, `a=2` | idem | idem, `a=2` **[B]** | idem [B] | `aΔ − V·∇`, `a=2` | idem | `Δ − V·∇` (`a=1`) | idem |
| convection | `v=2` | `v=2` | `v=2` **[B]** | `v=2` [B] | `V=(2,2)` | `V=(2,2)` [B] | `V=(2,1)` | idem |
| BC | hom. Dirichlet | idem | idem | idem | hom. Dirichlet | idem | hom. Dirichlet | idem |
| IC | `2x(1−x)` — **≠0 at x=2** | as Ex1 | `x(1−x)` (plant); OC run **z₀=0 [B]** | not stated; **identified numerically as `x(1−x)` [B]** (Fig. 40, 4 digits) | `xy` (cascade plant, ≠0 on ∂Ω); OC run **z₀=0 [B]** (E_oc(0.1)=0) | `xy` | **[C]**; tail IC-insensitive (≤ 4e-6) | [C]; tail IC-insensitive (≤ 1e-4) |
| input | 2 scalar, `χ_[.25,.35]`, `χ_[.85,.95]` | 1 scalar, `χ` of the union | distributed `L²(Ω_b)`, `[1/5,2/5]` | distributed, `[0.45,0.55]` | distributed, `Ω_b=[0,W/3]×[0,H]` | colocated, figure only | distributed `L²(Ω_B)`, figure only [B] | idem [B] |
| output | 2 scalar integrals `[.55,.65]`, `[1.25,1.35]` | 2 scalar | restriction `L²(Ω_c)`, `[1/2,7/10]` | `Ω_c=Ω_b` | `Ω_c=[W/3,2W/3]×[0,H]` | `Ω_c=Ω_b` | restriction `L²(Ω_C)` | idem |
| colocated | no | no | **no** | yes | no | yes | yes | no |
| reference | `sin t F`, `sin 2t F` (`t*=2, ε=1`) | idem | `x sin t F(t−1.5,1.5)` | as Ex3 | `xy(1−y)(y−½) sin t F` | as Ex5 | `x e^{−(y−½)²/σ²} sin 2t`, `σ=0.15` | idem |
| disturbance | `cos t F` on `[1.55,1.65]` | idem | `x² sin 2t F` on `[4/5,9/10]` | as Ex3, `Ω_d=[0.7,0.8]` | `x(y−½)² sin 2t F` on `Ω_d=[2W/3,W]×[0,H]` | as Ex5 | `x e^{…} cos 3t` | idem |
| **source OC comparison** | no | no | **yes, Figs. 32–33, ε=10⁻⁹** | no | **yes, Fig. 46, ε=10⁻⁷** | no | no | no |
| published absolute metric | **none** (Table 1 = ratios) | none | **none** (vector figures) | none (Table 2 = ratios) | none (figures) | none | **Table 9** `E_j`, j=1..5 | **Table 10** |
| missing data | IC/BC conflict; `B_d` form [B]; window [B] | as Ex1 | OC IC [B], `a,v` [B], `T` [B] | none essential (IC [B] identified, `T = 15` [A], window immaterial) | W, H [B] (resolved numerically), `T = 10` [B], OC IC [B]; unexplained transient | as Ex5 | regions [B] (asserted by the user), IC [C] (immaterial), `T` [B]; Table 9 = Fig. 30 with a one-level index shift | idem; Table 10 ≈ 4 % below its own Fig. 34 |
| ex21 support | ready (1D, conv., distributed, `source_regions`); scalar region-integral *outputs* = tracking on boxes needs per-channel weights — **not implemented** | as Ex1 + single input on a union (supported) | **done** (`--math2025-ex3`) | **done** (`--math2025-ex4`) | **done** (`--math2025-ex5`) | not implemented | **done** (`--errorfeedback-95`) | **done** (`--errorfeedback-96`) |
| **class** | **B** (figure-derived level) | B | **B — gold standard** (source's own OC curve reproduced to ~6e-6) | **C structural + B scale cross-check** (done) | **B-strong** geometry; periodic regime agrees closely; transient mismatch unexplained | C | **B** tail scale (bound) | saturation **B**; level **C** |

Notes.

* Mathematics 2025 Ex1/Ex2 measure **scalar** outputs `y_j = ∫_{Ω_cj} z`.
  ex21's tracking term `‖y − y_d‖²_{L²(Ω_c)}` is a *field* tracking; a scalar
  output tracking `Σ_j |∫_{Ω_cj} z − r_j|²` is a rank-one tracking operator per
  channel (as in the ex20 harnesses) and is not in the production code.
* Ex5's optimal control is at `ε = 10⁻⁷` (not `10⁻⁹`), and its plotted plant
  error uses the BC-incompatible IC `xy`.
* ErrorFeedback 9.5/9.6 details: `reaudit_20261003/FUTURE_EX21_BENCHMARKS.md`
  (ex20 tree).

## Status after the Example 4 / ErrorFeedback 9.5–9.6 audit task (superseded by the final classification above)

* **M25 Ex3** remains the reference (gold-standard) convection benchmark:
  same optimisation problem as ex21, source OC curve reproduced pointwise.
* **M25 Ex4** done: structural (colocated vs non-colocated floor) + figure-scale
  cross-check (`MATH2025_EX4_BENCHMARK.md`).
* **EF 9.5 / 9.6**: audited (`ERRORFEEDBACK_95_96_SOURCE_AUDIT.md`); exact
  geometry not recoverable, IC missing → **C, not implemented**.
* **M25 Ex5**: still blocked on `W`, `H`.

## Ranking of next tasks (as of the Example 4 task; items 1–3 are now done)

Criteria: (1) same optimisation problem as ex21; (2) 1D → 2D extension;
(3) absolute published metric; (4) completeness of source data; (5) need for
unsupported ex21 features; (6) amount of inference.

| rank | task | (1) | (2) | (3) | (4) | (5) | (6) | comment |
|---|---|---|---|---|---|---|---|---|
| 1 | **M25 Ex5** (2D non-colocated, source OC at ε = 1e-7) | **yes** | **yes** | figure | W, H missing; IC `xy` BC-incompatible | none | low once W, H known | the only 2D same-problem benchmark; **ask the authors for W, H** — highest value per datum requested |
| 2 | **EF 9.5** (2D colocated) | no (REFC) | yes | **Table 9** (validated by Fig. 30) | region coordinates [C], IC [C] | Gaussian preset only | moderate (geometry, IC, α_u match) | best printed-metric 2D case; **ask the authors for the region coordinates and IC** |
| 3 | EF 9.6 (2D non-colocated) | no | yes | Table 10 (≈ 4 % off its figure) | as 9.5 | as 9.5 | moderate | structurally interesting (floor in 2D) but the table/figure discrepancy caps precision |
| 4 | M25 Ex6 (2D colocated) | no | yes | figure | as Ex5 | none | as Ex5 | after Ex5 |
| 5 | M25 Ex1 / Ex2 (1D, scalar outputs) | no | no | ratios only | IC/BC conflict | **scalar-output tracking not in ex21** | high | lowest value: needs a new tracking operator and still has no absolute metric |

## Earlier recommendation (superseded by the ranking above)

1. **Mathematics 2025 Example 4** (1D, colocated, same plant family as Ex3):
   cheapest next step — a preset variant of Ex3 (regions only), and the colocated
   case is where the cascade errors decrease geometrically (Table 2), giving a
   structural contrast with Ex3's non-trackable floor. Its IC is unstated, so
   the comparison is tail-only; there is no source OC curve.
2. **Mathematics 2025 Example 5** — the only other source OC comparison, in 2D;
   blocked on the unpublished `W`, `H` (and the BC-incompatible IC). Worth
   asking the authors for `W`, `H`.
3. **ErrorFeedback 9.5** — the only convection case with printed absolute
   values, but the IC and exact regions are missing.
