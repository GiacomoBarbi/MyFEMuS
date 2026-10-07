# Mathematics 2025, Example 3 — ex21 benchmark against the source's own optimal control

Source data, tags and the mapping of (82)–(84): `MATH2025_EX3_SOURCE_AUDIT.md`.
Scripts, logs and per-run series: `math2025_ex3/`; tables:
`results/math2025_ex3_runs.csv`, `results/math2025_ex3_compare_*.json`,
extracted curves `results/math2025_ex3_fig{30,31,32,33}.csv`.
Binary sha256 in `math2025_ex3/logs/binary.sha256`. Serial runs, single-thread
BLAS.

**What is compared.** The source solved the finite-horizon problem
`min J_ε` (82)–(84) itself, with `ε = 10⁻⁹`, and plotted the resulting error
`‖E_oc(t)‖_{L²(Ω_c)}` (Figs. 32–33, red). ex21 solves the same problem
(`α_track = 1`, `α_u = ε = 10⁻⁹`, **not tuned**). Agreement with `E_oc` is a
**solver/discretisation validation** of the same mathematical problem.
Agreement with the cascade error `e₂` (blue) is a secondary cross-method
observation.

## 1. Implementation (Phase 4)

Least-invasive option: a **preset + driver**
(`include/Math2025Benchmark.{hpp,cpp}`, CLI `--math2025-ex3`), on the
unchanged production solver. One generic core feature was needed and added:

* `ProblemConfig::source_regions` (empty = ex20 behaviour). The source load
  becomes `M_d I_h f` with `M_d` the mass matrix over the boxes. This is the
  discretisation of `B_d d = χ_{Ω_d} d` without interpolating a discontinuous
  indicator. Threaded through `BuildDistributedRightHandSide` and
  `SolveOptimalControlOwnershipDistributed` as a defaulted argument; the default
  path is unchanged (goldens, §9).

Reproducible command (primary configuration):

    TimeOptCtrl_ex21 --math2025-ex3 --ex3-initial zero --time-scheme cn \
        --pc-schur-sweep --nx 40 --steps 1600 --no-paraview        # T = 20 default

Driver output: the gates of §2, the solve statistics, `J_track`, `J_reg`,
`‖u‖_{L²}`, sup/RMS of `‖e‖_c` on `[0,20]` and `[10,20]`, and the series
`‖e(t_n)‖_{L²(Ω_c)} = ((y_n − I_h r(t_n))ᵀ M_ω (y_n − I_h r(t_n)))^{1/2}` at every
time level (including `t = 0`).

## 2. Plant verification before optimisation (Phase 5)

From `logs/cn_sweep_nx40_nt1600.log` (identical in every run with `nx` a
multiple of 10) and `logs/verify_{cn,be}.log`:

| gate | result |
|---|---|
| region measures `1ᵀN1`, `1ᵀM_ω1`, `1ᵀM_d1` | 0.200000000000000, 0.200000000000000, 0.100000000000000 (exact) |
| IC `x(1−x)`: `‖I_h z₀ − z₀‖_{L²}`; `‖z₀‖` | 2.5e-17; 0.182574185835 = `1/√30` (exact) |
| IC trace on `∂Ω` | 0 |
| switch-on `F` at `t = 0, 0.75, 1.5, 2.25, 3` | 0, 0.0489273, **0.5**, 0.9510727, 1 (symmetric, as the smoothstep must be) |
| `r(0.6, t)`, `d(0.85, t)` | match the closed forms at `t = 0, …, 10` |
| disturbance load `1ᵀ M_d I_h(x²)` | 0.072333333333333 = `∫_{0.8}^{0.9} x² dx` (15 digits) |
| **convection sign / orientation** (on this config) | `‖C·x − 2·M1‖/‖M1‖ = 3.3e-14` vs transposed 14.5 |
| skew of `C` on free DOFs | 1.3e-14 |
| **state solve independent of the optimiser** (march with `u*`, spatial LU) | `max|Δy|/max|y| = 2e-12` (all runs ≤ 1.3e-11) |
| Dirichlet trace of the optimal state | ≤ 7e-15 |
| **exact-transpose** `⟨Ex,y⟩ − ⟨x,Eᵀy⟩` (CN / BE) | 1.0e-17 / 1.0e-17 |
| reduced Hessian symmetry, production adjoint (CN / BE) | 3.0e-13 / 1.4e-13 |
| **FD reduced gradient**, production adjoint (CN / BE) | 1.9e-15 / 4.8e-15 |
| negative control: ex20-style adjoint | gradient error 37 % / 37 % (as it must fail) |
| production solve, `‖∇J(u*)‖ / ‖dt α_u N u*‖` | 5.6e-4 (CN) / 1.2e-6 (BE) — relative to the *regularisation* term, which is O(α_u = 10⁻⁹): consistent with the outer `rtol = 1e-8` |

## 3. Source curves (Phase 3, summary)

Extraction method and calibration: audit §3. Value uncertainty ≈ ±3e-6;
`E_oc` and `e₂` are distinct polylines, equal to ≈ 1e-5 up to `t ≈ 19.6`
except at the cusp near `t = 12.55`, where they differ by 4.7e-4; at the very
end `E_oc` rises above `e₂` by up to 6.6e-4. **Source's own `E_oc − e₂` spread
on `[0, 19.7]`: RMS 4.1e-5, L∞ 4.7e-4.**

## 4. Refinement, schemes, solvers (Phase 6) — `z₀ = 0`, `T = 20`, `ε = 10⁻⁹`

| run | outer its | true res | `J_track` | `J_reg` | `J_total` | `‖u‖_{L²}` | sup `‖e‖` [10,20] | RMS `‖e‖` [0,20] | wall s |
|---|---|---|---|---|---|---|---|---|---|
| CN nx40 N=400 (dt .05) | 22 | 3.7e-9 | 1.446719e-2 | 1.6545e-6 | — | 57.523 | 0.0571012 | 0.0380357 | 451 |
| CN nx40 N=800 | 13 | 5.4e-9 | 1.446670e-2 | 1.6549e-6 | — | 57.531 | 0.0571018 | 0.0380351 | 526 |
| **CN nx40 N=1600** (dt .0125) | 7 | 7.1e-9 | **1.446633e-2** | **1.6547e-6** | **1.446799e-2** | **57.527** | **0.0571018** | **0.0380351** | 557 |
| CN nx40 N=3200 | 4 | 4.4e-9 | 1.446606e-2 | 1.6543e-6 | 1.446772e-2 | 57.521 | 0.0571018 | 0.0380344 | 591 |
| CN nx10 N=1600 | 6 | 1.1e-8 | 1.446697e-2 | 1.6546e-6 | — | 57.526 | 0.0571034 | 0.0380354 | 227 |
| CN nx20 N=1600 | 6 | 1.3e-8 | 1.446637e-2 | 1.6547e-6 | — | 57.527 | 0.0571020 | 0.0380351 | 302 |
| CN nx80 N=1600 | 8 | 2.2e-9 | 1.446633e-2 | 1.6547e-6 | — | 57.527 | 0.0571018 | 0.0380351 | 1017 |
| BE nx40 N=400 | 2 | 3.1e-12 | 1.451473e-2 | 1.6651e-6 | — | 57.708 | 0.0571256 | 0.0380526 | 24 |
| BE nx40 N=800 | 2 | 4.0e-12 | 1.449068e-2 | 1.6603e-6 | — | 57.624 | 0.0571140 | 0.0380440 | 70 |
| BE nx40 N=1600 | 2 | 2.6e-11 | 1.447853e-2 | 1.6575e-6 | — | 57.576 | 0.0571079 | 0.0380396 | 165 |
| BE nx40 N=3200 | 2 | 3.9e-9 | 1.447233e-2 | 1.6559e-6 | 1.447398e-2 | 57.547 | 0.0571049 | 0.0380373 | 326 |
| BE nx40 N=800 **SCHUR_MATCHING** | 3 | 4.8e-12 | 1.449068e-2 | 1.6603e-6 | — | 57.624 | 0.0571140 | 0.0380440 | 469 |
| CN nx40 N=800 **ADDITIVE_GAMG** | **200, reason −3** | 7.9e-1 | invalid | | | | | | 506 |
| BE nx40 N=800 **ADDITIVE_GAMG** | **200, reason −3** | 1.9e+0 | invalid | | | | | | 344 |

* **Temporal convergence.** BE is first order in `J_track`: differences
  −2.4e-5, −1.2e-5, −6.2e-6; its Richardson limit `1.44661e-2` equals the CN
  value to 1e-7 absolute. CN's tracking curve is dt-converged (§5: unchanged to
  the extraction noise from `dt = 0.05` on); its `J_track` still drifts by
  3–5e-7 per halving — the unresolved terminal layer (§5), which carries a
  first-order-like contribution to the integral. `‖u‖_{L²}` converges to
  ≈ 57.52 (CN) with < 0.02 % variation.
* **Spatial convergence.** `nx = 10 → 80` changes `J_track` by 6e-7, then 4e-8,
  then 0 at the printed digits; `nx = 40` is converged.
* **Solvers.** SCHUR_SWEEP converges for both schemes (BE: 2 outer iterations;
  CN: 22 → 13 → 7 → 4 as dt shrinks — the matching-Schur approximation is weaker
  for CN at `β = α_t η²/α_u = 10⁹`). BE SCHUR_MATCHING gives identical
  results (3 iterations, 25k inner GMRES iterations). ADDITIVE_GAMG **does not
  converge** at `ε = 10⁻⁹` (200 iterations, true residual O(1); exit code 8, not
  silent) — the same block-diagonal-preconditioner failure at small
  regularisation that ex20 already documents; it is not a convection or 1D
  issue. 1D CN SCHUR_MATCHING was not used (documented GAMG failure).

## 5. Pointwise comparison with the source's optimal control (Phase 7)

Primary run CN, nx = 40, N = 1600, `z₀ = 0`, `T = 20` (sampled on the source's
0.05 output grid; the ex21 nodes contain it). Full table for every run in
`results/math2025_ex3_runs.csv`.

| window | vs **`E_oc`** (source OC) L∞ / RMS | relative RMS | vs `e₂` (cascade) L∞ / RMS |
|---|---|---|---|
| Fig. 33, `[0, 2]` | 1.16e-5 / **6.2e-6** | 2.8e-4 | 7.8e-6 / 3.9e-6 (`[0.5,2]`; the source's `e₂` has a one-sample blip of 9.6e-5 at `t = 0.05`) |
| Fig. 32, `[0, 19.7]` | 3.81e-4 (cusp, t = 12.55) / **3.4e-5** | 9.0e-4 | 9.6e-5 (the `t = 0.05` blip) / 8.5e-6 |
| Fig. 32, `[0, 19.7]` without the cusp (`|t − 12.55| > 0.1`) | 2.8e-4 / **2.8e-5** | | 9.6e-5 / 7.1e-6 |
| Fig. 32, terminal `[19.7, 20]` | 8.3e-4 / 3.6e-4 | | 1.7e-4 / 6.6e-5 |
| **for scale: source `E_oc` vs `e₂`, `[0, 19.7]`** | 4.7e-4 / **4.1e-5** | | |

**Peaks** (5 maxima on `[3, 20]`, parabolic refinement): time shift
≤ 6e-4 (an eighth of a 0.05 sample), amplitude difference ≤ 3.5e-6
(extraction-noise level); **no phase shift**. Tail amplitude: sup on `[10, 20]`
0.0571018 (ex21) vs 0.05710 (both source curves). Peaks ex21 / `E_oc`:
`t = 4.6846/4.6846, 7.8474/7.8473, 10.9678/10.9680, 14.1306/14.1311,
17.2510/17.2509`, amplitudes 0.057102/0.057100, 0.057068/0.057066,
0.057102/0.057100, 0.057068/0.057066, 0.057102/0.057098 (differences
1.7–3.5e-6).

**Reading.**

1. On `[0, 2]` ex21 reproduces the source's optimal-control error to
   6e-6 RMS — about twice the extraction quantum.
2. On `[0, 19.7]` ex21 differs from `E_oc` by **less than the source's own two
   curves differ from each other** (RMS 3.4e-5 vs 4.1e-5), and lies closer to
   the cascade curve (RMS 8.5e-6). The largest pointwise gap is at the cusp
   `t ≈ 12.55`, a near-zero minimum of `‖e‖`: there `E_oc = 6.7e-4`,
   `e₂ = 1.9e-4`, ex21 `= 2.9e-4` (identical for every dt and both schemes);
   ex21 lies between the two source curves.
3. **Terminal layer `[19.7, 20]`.** With `λ(T) = 0` and `ε = 10⁻⁹` the
   optimum has a terminal layer far thinner than any time step used. The source
   (optimize-then-discretize, COMSOL) shows `E_oc` rising above `e₂` in its last
   ~3 samples (0.05294 vs 0.05228 at `t = 20`). ex21 (discretize-then-optimize)
   keeps a nonzero discrete adjoint on the last slot; its end value drifts with
   dt (CN 0.05221 → 0.05211, BE 0.05220 → 0.05169) and, under CN, rings
   (odd–even) over the last ≈ 0.4 time units at coarse dt. Neither side resolves
   the layer, so `[19.7, 20]` is **not** a validation window. Evidence that it
   is a horizon effect: ex21 with **`T = 25`** reproduces the horizon-free
   cascade curve on `[19.7, 20]` to **3.1e-6 RMS**, and is unchanged on
   `[0, 19.7]` (RMS vs `E_oc` 3.40e-5, identical).
4. **Initial condition.** With `z₀ = x(1−x)` (eq. (80)) ex21 starts at
   `‖e(0)‖ = 0.1059748` (exact) and misses Fig. 33 by RMS 1.7e-2; with
   `z₀ = 0` it matches to 6.2e-6. The source's OC run therefore used a zero
   initial state [B]; after `t ≈ 0.5` both ICs give the same curve.
5. **`a`, `v`.** At N = 800 the Fig. 33 RMS is 6.2e-6 for `(a, v) = (2, 2)`
   and 4.8e-4, 2.6e-3, 1.25e-3, 1.05e-3 for `(1, 1)`, `(2, 0)`, `(2, 1)`,
   `(4, 2)` — the inherited values are confirmed by **78–420×**.

**Conclusion.** ex21's discretize-then-optimize all-at-once solution of
(82)–(84) agrees with the source's own optimal-control solution, pointwise in
time, to within the source's own internal consistency (its OC-vs-cascade
spread) everywhere except the unresolved terminal layer, with no phase shift
and peak amplitudes equal to the extraction noise. This validates the ex21
solver and discretisation on a source-defined convection–diffusion OC problem.

## 6. Secondary: the cascade curve (`e₂`)

ex21 matches the cascade iteration error to RMS 7–9e-6 on `[0, 19.7]` (better
than it matches `E_oc`), and to 3e-6 on `[19.7, 20]` when the horizon is moved
to `T = 25`. This is a cross-method observation consistent with the source's
own statement that the two curves are "indistinguishable"; it says that at
`ε = α = 10⁻⁹` the 2-level cascade and the finite-horizon optimum reach the same
non-trackable floor (`sup ≈ 0.0571` on the tail). It is **not** a statement of
controller equivalence: the cascade is a forward-only feedback construction,
ex21 a finite-horizon open-loop optimum.

## 7. Inherited-issue check (Phase 8) — verified, not assumed

| issue | relevance here | evidence |
|---|---|---|
| load drops boundary values of `f` (Dirichlet elimination) | **none**: the only source is `χ_{Ω_d} d` with `Ω_d = [0.8, 0.9]` interior; `M_d` has no boundary column | load gate 0.0723333… exact |
| CN initial-trace fix | **no-op**: `z₀ = x(1−x)` and `z₀ = 0` have zero trace | IC gate: trace 0; boundary state ≤ 7e-15 |
| need for SUPG | **none**: global Péclet `vL/a = 1`; cell Péclet `v(h/2)/(2a)` = 0.025, 0.0125, 0.006, 0.003 for nx = 10…80 | spatial ladder monotone and converged (`J_track` 1.446697 → 1.446637 → 1.446633 → 1.446633 e-2); CN/BE agree |
| 1D CN SCHUR_MATCHING GAMG failure | avoided (SWEEP / BE MATCHING used) | — |
| ADDITIVE_GAMG at small `α_u` | **fails** (as in ex20 at small regularisation) | §4 |

## 8. Stop-condition review

| stop condition | status |
|---|---|
| (82)–(84) does not match ex21 after scaling | **no**: exact match, `α_track = 1`, `α_u = ε` |
| `a`/`v` ambiguity affecting the plant | resolved numerically (78–420× discrimination) |
| figure extraction unreliable | **no**: vector paths, value quantum ≈ 2e-6 |
| new adjoint/gradient inconsistency | **no**: all gates pass on this configuration |
| no temporal convergence | **no**: BE first order; CN curve converged; CN J drifts O(3e-7) (terminal layer) |
| Galerkin unstable | **no**: cell Péclet ≤ 0.025 |

## 9. Regression after the core change (`source_regions`)

See `../EX21_ZERO_CONVECTION_BASELINE.md` (rerun on the final binary): 61/61
ex20 goldens bit-identical; zero-convection baseline and convection gates
unchanged.
