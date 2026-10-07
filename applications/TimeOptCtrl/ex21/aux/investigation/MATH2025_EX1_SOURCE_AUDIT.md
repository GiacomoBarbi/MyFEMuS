# Mathematics 2025, Example 1 — source audit (before any coding)

**Source.** E. Aulisa, A. Chierici, D. S. Gilliam, *A Least-Squares Control
Strategy for Asymptotic Tracking and Disturbance Rejection Using Tikhonov
Regularization and Cascade Iteration*, Mathematics **13** (2025) 3707,
`~/Downloads/mathematics-13-03707.pdf` (47 pp., PDF CreationDate 2025-11-19).
Section 8 ("Numerical Simulations"), pp. 15–20, eqs. (65)–(72), Figures 2–18,
Table 1.

**Scope.** Data extraction only. The cascade/Tikhonov controller is NOT
implemented. Tags: **[A]** explicitly stated · **[B]** inferred (with the
evidence) · **[C]** missing.

---

## 1. Plant

| datum | value | tag | evidence / note |
|---|---|---|---|
| operator | `A = a d²/dx² − v d/dx`, `v > 0` | [A] | (65) |
| `a`, `v`, `ℓ` | `a = 2`, `v = 2`, `ℓ = 2`, `Ω = [0, 2]` | [A] | p. 16 "we have set a = 2, v = 2 and … ℓ = 2" |
| BC | homogeneous Dirichlet at both ends | [A] | (65)–(66); (66) writes `φ(L)` for `φ(ℓ)` (typo) |
| adjoint (continuous) | `A* = a d²/dx² + v d/dx`, `D(A*) = D(A)` | [A] | p. 15; **not used by ex21** (discrete transpose instead) |
| state equation | `z_t = A z + B_d d + B_in u` | [A] | (70) |
| initial condition | `z(x, 0) = 2x(1 − x)` | [A] | (71) |
| **IC / BC conflict** | `z(2, 0) = −4 ≠ 0` while `z(2, t) = 0` | ambiguity | the formula vanishes at `x = 0` and `x = 1`, not at `x = ℓ = 2` |
| IC actually simulated | the literal `2x(1−x)` | [B] | Fig. 6 starts at `y₂(0) ≈ −0.07`; `∫_{1.25}^{1.35} 2x(1−x) dx = −0.0782`, whereas `x(2−x)` would give `+0.077` |

**Mapping to ex21 (sign checked, not guessed).** ex21 solves
`y_t − ν Δy + v·∇y = f + η u`, i.e. `y_t = ν y_xx − v_x y_x + …`. Matching
`z_t = a z_xx − v z_x` term by term gives **`ν = a = 2`, `vx = +v = +2`**. The
assembled ex21 term is `C_ij = ∫ (vx φ_j') φ_i`, verified by G1 of
`EX21_CONVECTION_VALIDATION.md` (`C · x = vx · M 1` to 1e-14).

## 2. Actuation, observation, disturbance

| datum | value | tag | note |
|---|---|---|---|
| inputs | `nb = 2`, `B_in u = b₁u₁ + b₂u₂`, `b_j = χ_{Ω_bj}` | [A] | (67) |
| `Ω_b1`, `Ω_b2` | `[0.25, 0.35]`, `[0.85, 0.95]` | [A] | p. 16 |
| outputs | `nc = 2`, `y_j = C_j z = ∫_Ω z c_j dx`, `c_j = χ_{Ω_cj}` (INTEGRALS) | [A] | p. 15, (67) |
| `Ω_c1`, `Ω_c2` | `[0.55, 0.65]`, `[1.25, 1.35]` | [A] | p. 16 |
| disturbance region | `Ω_d = [1.55, 1.65]` | [A] | p. 16 |
| `B_d` | `χ_{Ω_d} d(t)` | [B] | "a disturbance d enters through the region Ω_d"; `D = R^{nd} = R`; operator never written (same situation as ErrorFeedback 9.1/9.2) |
| `G = C(−A)^{-1} B_in` | invertible 2×2, `L = 0` | [A] (claim) | numerical values of `G` NOT printed |

All five regions have width 0.1 and endpoints on the 0.05 lattice, so a uniform
mesh with `h = 0.05/k` resolves them exactly (ex21 selects regions by element
centroid, as ex20).

## 3. Signals

| datum | value | tag |
|---|---|---|
| `r₁(t)` | `sin(t) F(t − 2, 1)` | [A] (69) |
| `r₂(t)` | `sin(2t) F(t − 2, 1)` | [A] (69) |
| `d(t)` | `cos(t) F(t − 2, 1)` | [A] (69) |
| `F(t − t*, ε)` | 0 for `t < t* − ε`, 1 for `t > t* + ε`; on `[t*−ε, t*+ε]` the degree-9 polynomial with `F(−ε) = 0`, `F(ε) = 1`, `F^{(k)}(±ε) = 0`, `k = 1..4` | [A] |

`F` is **uniquely determined**: 10 conditions on 10 coefficients. With
`τ = (t − t* + ε)/(2ε) ∈ [0, 1]` it is the order-4 smoothstep
`F = 126τ⁵ − 420τ⁶ + 540τ⁷ − 315τ⁸ + 70τ⁹` (derived, not printed; satisfies all
10 conditions). Here `t* = 2`, `ε = 1`, so the signals switch on over `[1, 3]`.

## 4. Controller (documentation only — NOT implemented)

| datum | value | tag |
|---|---|---|
| cascade depth | "three extra iterations", `j = 0..3`, plant driven by `u = u³ = Σ_{j=0}^{3} u_j` | [A] |
| `α`, `β` | `α = 10⁻⁹`, `β = 0.25` (Figs. 15–18 also `α = 10⁻⁷`) | [A] |
| software | COMSOL Multiphysics 6.1 for all reported results ("implementation is basic") | [A] p. 7–8 |
| spatial / temporal discretisation | not given | [C] |

## 5. Horizon, tail window, published metrics

| datum | value | tag | note |
|---|---|---|---|
| horizon `T` | 20 | [B] | Figs. 3–4, 7–12, 15–18 all end at `t = 20`; never stated |
| tail window for `lim` | `[10, 20]` | [B] | Figs. 7–12 are drawn on `[10, 20]`; never stated |
| metric definition | `lim ‖e_jk‖ = lim sup_t |e_jk(t)|` of the **cascade iteration errors** | [A]/[B] | Table 1 caption; the window is not given |
| **Table 1** (only table) | ratios `lim‖e_jk‖ / lim‖e_(j−1)k‖`: `k=1`: 8.88e-2, 1.040e-1, 1.006e-1; `k=2`: 8.68e-2, 9.78e-2, 9.82e-2 (j = 1, 2, 3) | [A] | **ratios only** |
| absolute tail levels | **not printed anywhere** | [C] | only figure axes: Fig. 7 `e₀₁` amplitude ≈ 0.06, Fig. 8 `×10⁻³`, Fig. 9 `×10⁻⁴` |
| plant error vs cascade error | `e_k ≈ e_3k` after a short transient | [A] | Figs. 13–14 |
| control effort | Figs. 15–18 plot `u_j` (vector graphics); no number | [B] | |

A chained, figure-based estimate of the final level is possible but is
**not a published number**: e.g. `lim |e₃₁| ≈ 0.06 × 0.0888 × 0.104 × 0.1006
≈ 5.6e-5` (Fig. 7 amplitude × Table 1 ratios). The figures are vector
paths (`pdfimages` lists no raster image on pp. 16–21), so a direct vector
extraction of `e₃₁`, `e₃₂` from Figs. 9 and 12 is feasible, as was done for
ErrorFeedback Figure 14 — it would be a **[B]** quantity.

## 6. Other examples of the same paper (for the inventory)

Only Tables 1 and 2 exist in the paper; **both are ratios**. Example 3
(`Ω = [0,1]`, `Ω_b = [1/5, 2/5]`, `Ω_c = [1/2, 7/10]`, `Ω_d = [4/5, 9/10]`,
infinite-dimensional input/output, `r = x sin t F(t−1.5, 1.5)`,
`d = x² sin 2t F(t−1.5, 1.5)`, `z₀ = x(1−x)` — BC-compatible) is notable: the
source **itself solves the all-at-once optimal control problem (82)–(84)** —
`J_ε = ∫∫_{Ω_c} ½(z−r)² + (ε/2)∫∫_{Ω_b} u²`, `ε = 10⁻⁹`, backward adjoint,
`λ(T) = 0` — and reports (Figs. 32–33) that its tracking error is
indistinguishable from the cascade `e₂`. That is the same problem class as
ex21 (`α_track = 1`, `α_u = ε`, distributed control on `Ω_b`, tracking on
`Ω_c`); its `a`, `v` are not restated (inherited from (65) / Example 1 [B]).

## 7. Verdict

| item | status |
|---|---|
| plant (PDE, BC, coefficients, regions, signals) | **constructible**: all [A] except `B_d = χ_{Ω_d}` [B] and the IC/BC conflict at `x = 2` |
| IC/BC conflict | resolved in ex21 by the Dirichlet elimination (the constrained trace of `y₀` is discarded); this needed the CN fix of §K of the validation report. It affects only the initial transient, not a tail metric on `[10, 20]` |
| horizon, tail window | [B] (`T = 20`, `[10, 20]`) |
| **published comparison metric** | **[C] — no absolute tracking level is printed**; Table 1 gives ratios of cascade iteration errors, which have no AAO counterpart |
| quantitative α_u matching (the ex20 methodology) | **not possible from printed data**; possible only against a figure-derived [B] level |

**Stop condition met** ("literature data are insufficient for exact benchmark
construction"): the plant is constructible, but there is no printed target to
match. Per the task's stop rule, Example 1 was **not** implemented and no
benchmark run was made. Options for review are listed in the final report.
