# ErrorFeedback 2026, Example 9.6 — non-colocated practical saturation, cross-method benchmark (2D)

Source audit: `ERRORFEEDBACK_95_96_SOURCE_AUDIT.md`. Campaign:
`errorfeedback_95_96/scripts/run_ef_campaign.sh`. Table:
`results/errorfeedback_96_aao_runs.csv` (`summarize.py`). Figure:
`errorfeedback_95_96/ef_alpha_sweep.{png,pdf}` (right panel). Binaries and the
bit-identity check are as for 9.5 (`ERRORFEEDBACK_95_BENCHMARK.md`).

## 0. Classification

**C for the published numerical level `E₅` (Table 10 not quantitatively
reproduced on the asserted geometry); B for the qualitative
practical-saturation phenomenon.**

* **The AAO tail shows practical saturation over the tested regularisation
  range.** No asymptotic floor is claimed. It levels off at ≈ 0.152–0.156
  (nx 32) for `α_u` from 1e-6 to 1e-10, with a residual drift of 0.5–1.3 %
  per decade, while the control effort grows 31× (163 → 5068). This matches
  the source's "leveling-off of the computed iteration errors".
* **Resolution status.** The tail is spatially converged at `α_u = 1e-7`.
  Full spatial convergence of the smallest-`α_u` value 0.1520 at
  `α_u = 1e-10` was not established, because of solver cost; that point was
  run on nx 32 × nz 12 only. It is a practical saturation level on the tested
  mesh and regularisation range.
* **The published `E₅ = 5.220e-2` lies a factor ≈ 3 below the AAO tail
  levels.** This direct AAO-vs-Table-10 disagreement is the primary finding and
  needs no further argument.
* **Secondary, conditional estimate (§5).** Under the periodic-interior
  (turnpike) approximation, a control reaching `E₅` would require a time-RMS
  `‖u(t)‖_{L²(Ω_B)}` of at least approximately 9.7e3, ≈ 8.5× the AAO
  optimum's at `α_u = 1e-10`. This is a conditional lower-bound estimate, not
  a proof of finite-horizon unattainability. Like the source, ex21
  does not assume `ker(G*)` is nontrivial: the range may be dense, and smaller
  errors may be reachable with ever larger controls. The comparison is
  reported as a **disagreement, not forced**. Either the source's data differ
  from the asserted ones (most plausibly the region placement, available only
  as a non-to-scale schematic), or its controller used controls far larger
  than the AAO optimum's, or Table 10 measures a different quantity.
* The tail is IC-insensitive (≤ 1.1e-4 relative, at solver tolerance).

## 1. Data used

As in 9.5 (same plant, signals, `T = 20`, metric and IC handling), except for
the non-colocated regions:

| region | value | tag |
|---|---|---|
| `Ω_B` | `[π/8, π/4] × [1/3, 2/3]` (upstream of `Ω_C` along `V = (2, 1)`) | [B], asserted by the user |
| `Ω_C` | `[3π/8, π/2] × [1/3, 2/3]` | [B] |
| `Ω_d` | `[3π/4, 7π/8] × [1/3, 2/3]` (downstream) | [B] |
| published | Table 10 (primary): `6.355e-2, 5.496e-2, 5.338e-2, 5.264e-2, 5.220e-2`. Fig. 34 lies 3.6–5.1 % above the table and is used qualitatively only | [A] |

`max_t ‖r(t)‖_{L²(Ω_C)} = 0.370`. Uncontrolled, the tracking error is O(0.4).

## 2. Gates (all pass)

| gate | CN | BE |
|---|---|---|
| region measures `Ω_B, Ω_C, Ω_d` | `π/24` each to 1e-15 | |
| load `1ᵀ M_d I_h(x²y²)` | = exact to 2e-16 | |
| orientation, skew | as 9.5 (3.4e-14 / 2.4e-14; 1.2e-14) | same |
| shell transpose | 9.9e-18 | 1.8e-17 |
| reduced-Hessian symmetry, production adjoint | 1.8e-13 | 2.2e-12 |
| FD gradient, production adjoint | 1.1e-14 | 4.6e-14 |
| negative control (ex20-style adjoint) | 292 % (fails as it must) | 288 % |
| gradient at the production optimum | 4.2e-8 | 7.0e-11 |
| independent march vs production | ≤ 6.2e-12 (all runs) | |
| Dirichlet trace | ≤ 8e-14 | |

**Solver at small `α_u`** (SCHUR_SWEEP, nx 32, N 400):

| `α_u` | outer its | true residual |
|---|---|---|
| ≥ 1e-6 | 2 | ≤ 3e-12 |
| 1e-7 | 3 | 1.4e-9 |
| 1e-8 | 7 | 6.0e-9 |
| 1e-9 | 18–19 | 3–8e-9 |
| 1e-10 | 70 | 8.0e-9 |

The non-colocated optimality system becomes harder as `α_u` decreases, as in
Mathematics 2025 Ex. 3. Wall time grows from 3 min (`α_u = 1e-3`) to 24483 s
≈ 6.8 h (`α_u = 1e-10`) (production table, shared node).

## 3. IC sensitivity (nx 32, CN, N 400)

| `α_u` | `z₀ = 0` | `sin x sin πy` | poly | max relative spread |
|---|---|---|---|---|
| 1e-5 | 0.16919598 | 0.16919596 | 0.16919597 | 1.1e-7 |
| 1e-7 | 0.15550258 | 0.15550155 | 0.15550149 | 7.0e-6 |
| 1e-9 | 0.15398999 | 0.15397418 | 0.15397311 | 1.1e-4 |

At `α_u = 1e-9` the spread (1.1e-4) is at the level of the outer-solve
accuracy: 18–19 iterations, true residual 3–8e-9. Either way it is four
orders of magnitude below the factor-3 gap of §5. **Not a STOP condition.**

## 4. Saturation and refinement

`α_u` sweep, nx 32, `z₀ = sin x sin πy`, tail sup on `[5, 20]` (all sups at
interior periodic peaks, `t ≈ 8.65` or `14.9`; terminal layer immaterial,
≤ 2e-4):

| `α_u` | 1e-3 | 1e-4 | 1e-5 | 1e-6 | 1e-7 | 1e-8 | 1e-9 | 1e-10 | 1e-11 |
|---|---|---|---|---|---|---|---|---|---|
| tail sup | 0.3578 | 0.2842 | 0.1692 | 0.1561 | 0.1555 | 0.1548 | 0.1540 | 0.1520 | aborted† |
| tail RMS | 0.2569 | 0.2042 | 0.1217 | 0.1122 | 0.1118 | 0.1113 | 0.1107 | 0.1092 | |
| `‖u‖_{L²}` | 6.6 | 48.4 | 133.8 | 163.1 | 186.8 | 340.0 | 1202.4 | 5067.9 | |

**`α_u = 1e-10` is the smallest completed production point**: 70 outer
iterations, true KKT residual 8.0e-9, independent march 1.9e-14, zero sweep
failures. It is a valid discrete optimum.

† **`α_u = 1e-11`: aborted because of solver cost.** The run was stopped by
request after 10.4 h of wall time, still iterating (a live backtrace showed it
inside the spatial GMRES of the SCHUR_SWEEP time sweeps). This is not a failed
mathematical solve. **No numerical value from it is used.** The partial log is
kept as `logs/ABORTED_sweep_ef96_a1e-11_cn_nx32_nz12_nt400.log`. Nothing
suggests that the 1e-11 point would change the conclusion: the previous decade
already shows practical saturation.

From `1e-6` to `1e-10` the tail moves by 2.7 % while the control effort grows
31× (163 → 5068): **practical saturation over the tested regularisation range**
(not a proved asymptotic floor). A slow drift remains (0.5 %/decade from 1e-7
to 1e-9, then 1.3 % from 1e-9 to 1e-10). The bulk of the error on `Ω_C` is not
substantially reduced from `Ω_B` over the tested regularisation range.
Smaller errors may in principle be attainable with much larger controls.

**Space / time at `α_u = 1e-7`** (CN):

| mesh | N | tail sup |
|---|---|---|
| nx 16, nz 6 | 400 | 0.14898 |
| nx 32, nz 12 | 400 | 0.15550 |
| nx 64, nz 24 | 200 | 0.155733 (5 outer its, residual 1.9e-9) |
| nx 64, nz 24 | 400 | **0.155734** (3 outer its, residual 1.7e-9) |

At `α_u = 1e-7` the tail is **spatially and temporally converged**: nx 32 →
64 changes it by 0.15 %, and N 200 → 400 by 3e-6. nx 16 is 4 % low. **Full
spatial convergence of the smallest-`α_u` value (0.1520 at `α_u = 1e-10`) was
not established because of solver cost**: it was run on nx 32 × nz 12 only.
It is reported as a practical saturation level on the tested mesh and
regularisation range. Péclet number `|V| h_max / 2`: 0.22 / 0.11 /
0.055. The nx 64 runs at N 800 (CN and BE) and nx 128 were dropped for cost:
the non-colocated solves take ≈ 2 h at nx 64.

## 4b. Solver performance (why small `α_u` is expensive)

Full analysis: `AULISA_GILLIAM_SOLVER_ROBUSTNESS_AUDIT.md`.

* **Schur split hits its cap.** The Schur-split GMRES (cap 100) runs
  8.5 / 17 / 38.5 / 84 / 100 iterations per outer iteration at `α_u` = 1e-3 …
  1e-7. From 1e-7 on, every Schur solve stops at the cap.
* **Outer FGMRES degrades.** It needs 2, 2, 2, 2, 3, 7, 18, 70 iterations at
  `α_u` = 1e-3 … 1e-10. Wall time at N = 400 grows from 3 min to 6.8 h.
* **Spatial solves are not the cause.** The time-sweep spatial GMRES solves
  average ≈ 16.5–18.5 iterations (individual solves 12–22), with no failures
  (§5 of the audit). The cost
  explosion is the product of capped Schur solves and growing outer
  iterations; it is not a degradation of the spatial solves.
* **It is specific to this configuration.** Colocated EF 9.5, with the same
  plant, mesh and data, needs 2 outer iterations and ≈ 8 Schur iterations per
  outer iteration down to `α_u = 1e-11`. Non-colocated Mathematics 2025 Ex. 3
  (1D) also caps its Schur solves at small `α_u` but needs only 5 outer
  iterations at 1e-11.
* **Tuning check** (`EF96_SMALL_ALPHA_SOLVER_TUNING.md`). Lowering the Schur
  cap from 100 to 50 gives only ≈ 2–2.7× better work-to-residual-reduction
  (classification B, modest improvement). The `α_u = 1e-11` degradation
  persists, and a production retry is not justified.
* **Validity is unaffected.** Every completed solve reached the outer rtol
  with a true KKT residual ≤ 8.4e-9. Solver inefficiency at small `α_u` does
  not affect the validity of these solutions.

## 5. The published level against the AAO saturation level

* Published `E₅ = 5.220e-2` (Table 10); Fig. 34's `e₅` tail is ≈ 5.4e-2.
* AAO tail sup (nx 32): 0.1555 / 0.1548 / 0.1540 / 0.1520 at `α_u = 1e-7`,
  `1e-8`, `1e-9`, `1e-10`. **Ratio to `E₅` ≈ 2.9–3.0.** This direct
  disagreement is the primary finding. It does not depend on the estimate
  below.
* **Conditional control-size estimate (periodic-interior approximation; not a
  rigorous finite-horizon bound).**
  * The tail sup sits at interior periodic peaks (§4). The forcing is
    `2π`-periodic for all `t ≥ 0` (`d` has no switch-on), and the plant decays
    at rate ≈ 12. So on `[5, 18]` the AAO optimum approximates the
    `α_u`-regularised **periodic** optimum. For any periodic control `u_c`,
    with `mse` the per-period mean of `‖e‖²_{L²(Ω_C)}` and `ms` that of
    `‖u‖²_{L²(Ω_B)}`, optimality gives
    `mse(u_c) + α_u ms(u_c) ≥ mse(u*) + α_u ms(u*)`.
  * A tail with sup 0.0522 has `mse(u_c) ≤ 0.0027`. The AAO optimum has
    `mse(u*)` = 0.01250 / 0.01238 / 0.01225 and time-RMS `‖u*(t)‖` = 42 / 76 /
    269 at `α_u = 1e-7` / `1e-8` / `1e-9`.
  * Hence `rms ‖u_c(t)‖ ≥ 315` (from `α_u = 1e-7`), `≥ 986` (from `1e-8`) and
    `≥ 3.1e3` (from `1e-9`), and **≈ 9.7e3 (from `1e-10`; AAO time-RMS
    1133, `mse(u*)` = 0.01192)**. Under the periodic-interior approximation,
    a control reaching the published `E₅` would therefore require a time-RMS
    of at least approximately 9.7e3.
  * This is a **conditional lower-bound estimate**. It relies on the
    turnpike/periodic approximation of the interior of a `T = 20` horizon and
    on the AAO solution at nx 32. It is not a proof of finite-horizon
    unattainability, and it is not needed for the factor-3 discrepancy above.
    It does not depend on the controller's structure.
* **Reading.**
  1. Under that approximation, Table 10's level on the asserted data would
     need a control about an order of magnitude larger than the AAO optimum
     at `α_u = 1e-9` or `1e-10`, where further regularisation decreases
     already buy only 0.5–1.3 % per decade.
  2. The audit had found Table 10 inconsistent with its own Fig. 34 by ≈ 4 %,
     far smaller than the factor 3 here.
  3. In 9.5, the same plant, signals and metric with the colocated regions
     gave a consistent picture (every `E_j` attainable at moderate `α_u`).
  4. The tension is therefore specific to the non-colocated configuration.
     The most plausible cause is the region placement (`Ω_B` farther from or
     closer to `Ω_C`, or a different `y`-extent), which the source shows only
     as a non-to-scale schematic (audit §3).
  5. The source itself declines to claim a nontrivial `ker(G*)`. Both readings
     — the data differ, or the controller used very large inputs — are
     compatible with its text.
* No `α_u` was tuned, and no geometry was altered to approach `E₅`.

## 6. Conclusion

**Structural agreement:** non-colocation produces practical saturation over the
tested regularisation range in 2D (`α_u` = 1e-6 … 1e-10; smallest completed
point 1e-10, tail 0.152, a practical saturation level on the tested mesh), consistent with the source's "leveling-off of the computed
iteration errors".

**Quantitative disagreement:** on the asserted geometry the AAO tail levels
are ≈ 3× the published `E₅`, and Table 10 is not quantitatively reproduced. A
secondary, conditional estimate (periodic-interior approximation) puts the
control needed for `E₅` at a time-RMS of roughly 9.7e3 or more, ≈ 8.5× the
AAO optimum at `α_u = 1e-10`. No asymptotic floor is claimed, and `E₅` is not
claimed unreachable. Class **C** for the published level; **B** for the
practical-saturation phenomenon itself.

**STOP-condition check:**
* the tail is IC-insensitive;
* gates pass;
* residuals are ≤ 8e-9 (degrading with `α_u`, adequate for the 1e-4-level
  statements made);
* at `α_u = 1e-7` the tail is converged in space (0.15 %) and time (3e-6);
  at `α_u = 1e-10` (nx 32 only) full spatial convergence was not established
  because of solver cost;
* no Galerkin instability (Péclet ≤ 0.22).
