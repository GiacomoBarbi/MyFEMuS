# Mathematics 2025, Example 5 — source audit (re-audit)

**Source.** Aulisa, Chierici, Gilliam, Mathematics 13 (2025) 3707,
`~/Downloads/mathematics-13-03707.pdf`, Section 8, Example 5, pp. 30–31
(Figures 42–46). Tags: **[A]** stated · **[B]** inherited / inferred (evidence
given) · **[C]** missing. Figure data: vector extraction
(`math2025_ex5/scripts/extract_figs.py`; raw CSVs
`results/math2025_ex5_fig{45,46}_raw.csv`; calibration
`results/math2025_ex5_fig{45,46}_extraction.json`).

## 1. Specification

| datum | value | tag | evidence |
|---|---|---|---|
| operator | `A = aΔ − V·grad`, `V = [u, v]`, (85) | [A] | p. 30 |
| `a`, `u`, `v` | **2, 2, 2** | [A] | "we have set a = 2, u = v = 2, β = 0.25" |
| BC | homogeneous Dirichlet, `D(A) = {φ ∈ H²(Ω): φ|∂Ω = 0}` | [A] | (85) |
| domain | a rectangle `Ω = [0, W] × [0, H]` (Fig. 42) | [A] shape / **[C] values** | Fig. 42 is a schematic with labels `W/3, 2W/3, W` (drawn glyphs) and `H` (text layer); no number anywhere in the example |
| **`W`, `H`** | **`π`, `1`** (hypothesis) | **[B]** | `H = 1`: the factor `y(1−y)(y−0.5)` of `r` vanishes on `y = 0, 1` only for `H = 1`. `W = π`: tested numerically, §4 (the free response of Fig. 45 and the periodic regime of Fig. 46) |
| regions | `Ω_b = [0, W/3] × [0, H]`, `Ω_c = [W/3, 2W/3] × [0, H]`, `Ω_d = [2W/3, W] × [0, H]` | [A] (in units of `W`) | Fig. 42, three vertical strips; "non-colocated" |
| `B_in`, `C`, `B_d` | extension by zero from `L²(Ω_b)`, restriction to `L²(Ω_c)`, extension by zero from `L²(Ω_d)` | [B] | "analogous to the one-dimensional Example 3" |
| reference | `r = x y(1−y)(y−0.5) sin(t) F(t−1.5, 1.5)` (86) | [A] | |
| disturbance | `d = x (y−0.5)² sin(2t) F(t−1.5, 1.5)` (87) | [A] | |
| `F` | order-4 smoothstep on `[0, 3]` (Ex. 1 definition) | [A] | "as in the previous example, F is a smooth function …" |
| plant IC of the cascade runs | `φ(x) = xy` (plant error `e`, Fig. 45) | [A] | "the error obtained when using the cascade control u3 in the plant with initial condition φ(x) = xy is denoted by e(t)" |
| **IC of the optimal-control run** | **not stated** | [C] stated / **[B] identified: `z₀ = 0`** | §3: `E_oc(0.1) = 0.0000`, while any run from `xy` has `‖e(0.1)‖_c ≈ 0.089` (Fig. 45) |
| optimal-control problem | "a system similar to (82)–(84)": `J = ∫₀ᵀ [½‖z − r‖²_{L²(Ω_c)} + (ε/2)‖u‖²_{L²(Ω_b)}] dt` | [A]/[B] | Ex. 3's functional, by reference |
| `ε` | **`10⁻⁷`** | [A] | "the full optimal control error Eoc obtained with the penalty constant ϵ = 10⁻⁷" |
| horizon `T` | **10** | [B] | all time axes of Figs. 43–46 end at 10; the OC curve is sampled up to `t = 9.9` |
| norm | `‖·‖_c = L²(Ω_c)` | [A]/[B] | as Ex. 3 |
| cascade `α`, `β`, levels | `α = 10⁻⁴` (Fig. 43), `10⁻⁷` (Figs. 44, 46), `10⁻⁹` (Fig. 45 caption) ; `β = 0.25`; `j = 0…3` | [A] | documentation only; **not comparable to `α_u`**. Internal inconsistency: the text says "in Figures 44–46, we have fixed α = 10⁻⁷" but the caption of Fig. 45 says `α = 10⁻⁹` |
| claim | "After a short transient, the two errors [e₃, E_oc] become indistinguishable" | [A] | |

## 2. Figure extraction

Pipeline as Examples 3–4 (`pdftocairo -svg` → stroked paths with accumulated
transforms). These panels have **no grid lines**: the axis map is a linear fit
to the dark-grey **tick marks**, whose values are the tick labels of the PDF
text layer (x: 0, 2, …, 10; y: 0, 0.01, …, 0.06 for Fig. 46 and 0 … 0.09 for
Fig. 45).

| | Fig. 45 | Fig. 46 |
|---|---|---|
| calibration residual (t / value) | 1.4e-4 / 2.1e-6 | 1.4e-4 / 1.2e-6 |
| value quantum (smallest vertex step) | 4.4e-6 | 2.9e-6 |
| stroke half-width in value units | — | 3.7e-4 (plotting only; vertices are exact) |
| blue `(0,0,1)` | `e₃`, 197 vertices, cascade grid `Δt = 0.05` | `e₃`, 197 vertices, same grid |
| second curve | red `(1,0,0)`, plant error `e`, 199 vertices | orange `(0.85,0.33,0.10)`, **99 vertices, `t = 0.1 … 9.9`, `Δt = 0.1`** |

**Curve identity in Fig. 46 [B].** The caption gives no colours. Blue is the
cascade colour of every Example 5 panel (Figs. 43–45, same 197-vertex grid).
The orange curve is on its own grid (`Δt = 0.1`, ending at 9.9), the signature
of a separate run, and is listed second in the caption ("‖e₃‖_c, ‖E_oc‖_c").
Both assignments point the same way: **orange = `E_oc`**.

## 3. The initial condition of the optimal-control run

The text attaches `φ = xy` to the *plant error with the cascade control*
(Fig. 45), not to `E_oc`.

* Fig. 45 (red, plant from `xy`): `‖e‖_c = 0.0890, 0.0300, 0.0100, 0.0033`
  at `t = 0.10, 0.15, 0.20, 0.25` — the free decay of `xy` seen on `Ω_c`.
* Fig. 46 (orange, `E_oc`): `0.0000, 0.00002, 0.00007, 0.00019` at
  `t = 0.1, 0.2, 0.3, 0.4` — no IC transient at all.
* An ex21 optimal control started from `xy` has `‖e(0.1)‖_c = 0.065` (BE,
  N = 800; CN rings on this IC and gives 0.17–0.24). The free response of
  `xy` alone is 0.080–0.085 (CN/BE, `dt = 0.0025`). From `0` it has `≈ 0`.
  After `t ≈ 0.7` the two ICs give identical errors.

**Verdict: the OC run started from `z₀ = 0` [B, unambiguous at the 0.065
vs 0.0000 level].** The plant IC `xy` does not apply to `E_oc`.

## 4. Geometry evidence (summary; details in `MATH2025_EX5_BENCHMARK.md`)

Details are in `MATH2025_EX5_BENCHMARK.md`, §3. Two independent observables
are used.

* **Fig. 45, free response of `xy` on `Ω_c`** (controller-independent,
  `t ≤ 0.35`). The ex21/source ratio is 0.94–0.97 for `(π, 1)`; 0.001–0.035
  for `(1, 1)`; 11–400 for `(π, 2)`; and 3.0–4.5 for `(2π, 1)`. The decay
  rate is 22.15 for the source and 22.08 (BE) / 22.72 (CN) for `(π, 1)`. The
  exact slowest rate is 22.74 for `(π, 1)`, against 40.5, 7.9 and 21.2 for
  the alternatives.
* **Fig. 46, `E_oc` peaks** (0.0511). ex21 gives 0.0510 for `(π, 1)`,
  0.0065 for `(1, 1)`, 1.71 for `(π, 2)` and 0.153 for `(2π, 1)`.

**Verdict: `(W, H) = (π, 1)`, [B], strongly corroborated.** It is the only
candidate consistent with both figures. Fig. 46 alone constrains the product
`W^{3/2}·‖y(1−y)(y−½)‖_{L²(0,H)}`; Fig. 45 separates `W` from `H`.

## 5. What is and is not specified

* Fully stated: operator coefficients, BC, region layout (in units of `W`),
  signals, `F`, `ε`, the functional (by reference to Ex. 3), the norm.
* Inferred: `T = 10` (axes), `H = 1` (signal factor), `W` (numerically,
  §4), OC IC `= 0` (§3), curve identity (§2).
* Missing and immaterial to the OC comparison: cascade `α`, `β` semantics.
