# Mathematics 2025, Example 4 — source audit

**Source.** Aulisa, Chierici, Gilliam, Mathematics 13 (2025) 3707,
`~/Downloads/mathematics-13-03707.pdf`, Section 8, Example 4, pp. 27–30
(Figures 34–41, Table 2). Tags: **[A]** stated · **[B]** inherited / inferred
(evidence given) · **[C]** missing. Figure data: vector extraction
(`math2025_ex4/scripts/extract_figs.py`; raw CSVs
`results/math2025_ex4_fig{35..40}_raw.csv`; record
`results/math2025_ex4_extraction.json`).

## 1. Specification

| datum | value | tag | evidence |
|---|---|---|---|
| operator | `A = a d²/dx² − v d/dx`, as (65), domain (66) | [A] | "the same system operator A introduced at the beginning of this section in (65) with domain (66)" |
| `a`, `v` | **2, 2** | **[B]** | not restated; Ex. 1 sets `a = v = 2`, Ex. 5 restates them; Ex. 3 confirmed them numerically (78–420× discrimination). **Ex. 4 itself:** the measured decay rate of Fig. 40's IC transient between `t = 0.2` and `0.4`, `ln(1.4243e-3/2.4838e-5)/0.2 = 20.26`, equals the slowest decay rate `aπ² + v²/(4a) = 20.24` for `a = v = 2` (§3) |
| domain | `Ω = [0, ℓ]`, **`ℓ = 1`** | [A] | "we have chosen ℓ = 1" (footnote-style sentence after Fig. 34) |
| BC | homogeneous Dirichlet (domain (66)) | [A] | |
| **colocated** | **`Ω_b = Ω_c = [0.45, 0.55]`** | [A] | "Ω_b = Ω_c = [0.45, 0.55]"; "a colocated case in which Ω_b = Ω_c" |
| `Ω_d` | `[0.7, 0.8]` | [A] | |
| input / output spaces | as Ex. 3: `B_in` extension by zero from `L²(Ω_b)`, `C` restriction to `L²(Ω_c)` (infinite-dimensional) | [B] | "The main difference between Example 3 and this example is … colocated"; nothing else changes |
| `B_d` | extension by zero from `L²(Ω_d)` | [B] | as Ex. 3 |
| reference, disturbance | `r = x sin t F(t−1.5, 1.5)`, `d = x² sin 2t F(t−1.5, 1.5)` | [A] | "using the same r and d introduced in (78)" |
| `F` | order-4 smoothstep on `[0, 3]` (Ex. 1 definition) | [A]/[B] | as Ex. 3 |
| **IC (plant)** | **not stated** | [C] stated / **[B] identified** | §3: the plant error of Fig. 40 is reproduced to **4 significant digits** by the free response of `z₀ = x(1 − x)` (the Ex. 3 value) on `t ∈ [0.2, 0.4]` |
| cascade | `j = 1…5` levels ("five additional steps"), plant driven by `u⁵ = Σ_{j=0}^5 u_j` | [A] | documentation only |
| `α` | `10⁻³, 10⁻⁵, 10⁻⁷, 10⁻⁹` (Figs. 35–38), `10⁻⁹` (Figs. 39–40), `10⁻⁷, 10⁻⁸, 10⁻⁹` (Fig. 41) | [A] | documentation only; **not comparable to ex21's `α_u`** |
| `β` | not stated | [C] | documentation only |
| source optimal-control run | **none** for Ex. 4 | [A] | unlike Ex. 3 |
| horizon `T` | **15** | **[A]** | "Figure 41 plots the cascade control u⁵(x, t) … at the final time t = 15"; all time axes end at 15 |
| tail window | not stated | [C] → immaterial | the extracted tail sups are window-independent on `[5, 15]`, and the sup over the plotted curves reproduces Table 2 (§2) |
| norm | `‖e(t)‖_c = ‖r(t) − z(t)‖_c` (L² on `Ω_c`, as Ex. 3) | [A]/[B] | p. 28 |
| geometric convergence statement | "we expect the norms of the errors obtained from the cascade algorithm to decrease geometrically because of Lemma 3"; "the error norms decrease geometrically with each iteration"; "the first step … always produces an outlier" | [A] | |
| Table 2 | ratios `lim‖e_j‖/lim‖e_{j−1}‖`, `j = 1…5` (below) | [A] | |
| Fig. 41 | `u⁵(x, 15)` on `Ω_b` for `α = 10⁻⁷, 10⁻⁸, 10⁻⁹`, values up to ≈ 450; "the control becomes progressively rougher" as `α` decreases | [A] | not used quantitatively |

**Table 2** (`lim‖e_j‖/lim‖e_{j−1}‖`):

| j | α = 1e-3 | α = 1e-5 | α = 1e-7 | α = 1e-9 |
|---|---|---|---|---|
| 1 | 0.8851 | 0.5238 | 0.6486 | 0.1369 |
| 2 | 0.8852 | 0.9765 | 0.8622 | 0.8516 |
| 3 | 0.8854 | 0.9817 | 0.9024 | 0.9014 |
| 4 | 0.8856 | 0.9818 | 0.9137 | 0.9202 |
| 5 | 0.8858 | 0.9820 | 0.9215 | 0.9315 |

## 2. Figure-derived absolute levels (semi-quantitative, [B])

Vector extraction of Figs. 35–40 (calibration residual ≤ 2e-4 in t; value
quantum 2e-8 … 6e-6 depending on the axis). Tail sup on `[5, 15]` of the
cascade iteration errors:

| α | `e₁` | `e₂` | `e₃` | `e₄` | `e₅` | extracted ratios j=2..5 | Table 2 j=2..5 |
|---|---|---|---|---|---|---|---|
| 1e-3 | 1.2415e-1 | 1.0991e-1 | 9.7323e-2 | 8.6190e-2 | **7.6343e-2** | .8854 .8854 .8856 .8858 | .8852 .8854 .8856 .8858 |
| 1e-5 | 7.2767e-3 | 7.1037e-3 | 6.9742e-3 | 6.8474e-3 | **6.7239e-3** | .9762 .9818 .9818 .9820 | .9765 .9817 .9818 .9820 |
| 1e-7 | 2.2713e-3 | 1.9582e-3 | 1.7670e-3 | 1.6146e-3 | **1.4879e-3** | .8621 .9024 .9137 .9215 | .8622 .9024 .9137 .9215 |
| 1e-9 | 4.7473e-4 | 4.0428e-4 | 3.6444e-4 | 3.3535e-4 | **3.1238e-4** | .8516 .9014 .9202 .9315 | .8516 .9014 .9202 .9315 |

**The extraction reproduces Table 2 to 4 digits** for every α and level
(largest deviation 3e-4 in a ratio), which validates the extraction and shows
that Table 2's `lim` is the sup over the plotted tail. The plant error of
Fig. 39 (`α = 10⁻⁹`) has the same tail sup as `e₅`, 3.1237e-4. `e₀` is not
plotted, so the `j = 1` ratios cannot be checked. CSV:
`results/math2025_ex4_source_levels.csv`.

## 3. The initial condition — identified numerically

The source states no IC. The cascade controller is built from controller states
started at zero, so (by linearity) the plant error differs from the cascade
error `e₅` exactly by the **free response** of the plant IC seen on `Ω_c`,
`χ_c S(t) z₀`. Where `e₅` is negligible, `‖e_plant(t)‖_c = ‖χ_c S(t) z₀‖`.
Figs. 39–40 start at `t = 0.2` (the value at `t = 0` is not plotted).

ex21 computes `‖χ_c S(t) z₀‖` for a candidate `z₀` (driver output
`EX21_EX4_FREE`). For `z₀ = x(1−x)`, CN, `dt = 0.0025`, nx = 40 and 80
(identical):

| t | Fig. 40 plant | Fig. 40 `e₅` | ex21 `‖χ_c S(t) x(1−x)‖` | ratio |
|---|---|---|---|---|
| 0.20 | 1.4243e-3 | 2.2e-8 | 1.4239e-3 | **1.000** |
| 0.25 | 5.1760e-4 | 2.2e-8 | 5.1747e-4 | **1.000** |
| 0.30 | 1.8807e-4 | 9.3e-8 | 1.8806e-4 | **1.000** |
| 0.35 | 6.8444e-5 | 2.4e-7 | 6.8347e-5 | 1.001 |
| 0.40 | 2.4838e-5 | 5.2e-7 | 2.4839e-5 | **1.000** |
| 0.50 | 3.59e-6 | 1.7e-6 | 3.28e-6 | (`e₅` no longer negligible) |

(BE at the same dt is 10–20 % off on this stiff transient; CN is converged.)
Logs: `math2025_ex4/logs/free_x1mx_{be,cn}_nx{40,80}_T2_nt800.log`.

**Status: IC not stated [C]; numerically identified [B] as `x(1−x)`** in
the sense that matters after `t = 0.2`: the components of `z₀` along the slower
modes match to 1e-3 relative (the decay rate also confirms `a = v = 2`). The
fast components have decayed by ≥ 7 orders by `t = 0.2` and are not
identified; `x(1−x)` is consistent, not uniquely proven.

## 4. Comparison classification

* There is **no source finite-horizon optimal control** for Example 4, so the
  Example-3-type validation (same optimisation problem) is impossible.
* The source publishes **ratios** (Table 2) and **figure-derived absolute
  cascade levels** (validated against Table 2 to 4 digits).
* The source's `α` parametrises the cascade's Tikhonov regularisation, not the
  AAO functional.

**Classification: B/C — structural primary question (does colocation remove
the non-trackable floor?), with a semi-quantitative cross-check of the
attainable tail scale against the figure-derived cascade levels.** No `α_u` is
tuned to manufacture agreement; where a figure-derived level is placed on the
AAO `α_u` curve, it is by interpolation of the sweep and labelled as such.
