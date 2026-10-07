# EF 9.6 at small `α_u`: is the cost a tuning problem? (bounded solver investigation)

**Question.** Is the extreme cost of ErrorFeedback 9.6 at very small `α_u`
mainly an avoidable consequence of the nested-Krylov tolerances and iteration
caps, or is the SCHUR_SWEEP / matching-Schur family intrinsically ineffective
in this regime?

**Answer: classification B — modest improvement only.** Lowering the
Schur-split iteration cap from 100 to 50 roughly halves the work per outer
iteration without hurting outer progress, a ≈ 2–2.7× gain in residual decades
per second. The `α_u = 1e-11` degradation remains obvious at every setting,
and no setting tested gives a robust ≥ 3× gain with healthy outer
convergence. The limitation is the parameter robustness of the matching-Schur
approximation, not a bad tolerance choice. **A new N = 400, `α_u = 1e-11`
production run is not justified.**

Data and scripts:
* `errorfeedback_95_96/solver_diag/tuning/` (`run_tuning.sh`, `stage1.sh`,
  `stage3.sh`, `summarize_tuning.py`, `logs/`);
* `errorfeedback_95_96/solver_diag/tuning_summary.csv`.

Binary b6b6eb86… (unchanged). Only PETSc runtime options were varied: no code
change, no change of the mathematical algorithm. All runs were serial and
sequential, on the real EF 9.6 spatial problem (nx 32 × nz 12, CN, the same
regions and data, SCHUR_SWEEP, `z₀ = sin x sin πy`). Each run had a fixed outer
budget and a hard `timeout`. Total compute: about 50 min.

## 1. Current nested-Krylov configuration (from the code)

| level | type | restart | rtol | max_it | other | set in | runtime prefix |
|---|---|---|---|---|---|---|---|
| outer | FGMRES | 30 | 1e-8 | 200 | PCFIELDSPLIT Schur, FULL factorisation, Schur pre = USER (matching shell) | `Solver.cpp` ≈ 735–785 (`KktSolverPolicy`) | none: `-ksp_*` (the only unprefixed KSP); `KSPSetFromOptions(outer)` follows the hard-coded values |
| `y` split | PREONLY + PCSHELL (exact forward time sweep) | — | — | — | — | `Solver.cpp` ≈ 812 | `fieldsplit_y_` |
| Schur split | GMRES | 30 | 1e-6 | **100** | PCSHELL = matching `Ŝ⁻¹ = right⁻¹ E left⁻¹` | `Solver.cpp` ≈ 826–839 | **`fieldsplit_schur_ksp_*`**; `KSPSetFromOptions` follows the hard-coded values |
| sweep spatial solves (`right`, `left`, `y`) | GMRES(50), PC_RIGHT, unpreconditioned norm (nonsymmetric under convection) | 50 | 1e-10 (`--spatial-tol`) | 500 | PCGAMG, PETSc defaults, zero initial guess | `Preconditioners.cpp` `CreateSweepSpatialSolver` | `sweep_right_`, `sweep_left_`, `fieldsplit_y_sweep_` |

Every knob tested is already exposed through these prefixes, so no new option
was added. Overrides were verified with `-options_left`: a Schur
`max_it 30` request stops each Schur solve at exactly 30 iterations.

Options used per run: `-ksp_max_it {6|8} -ksp_monitor
-fieldsplit_schur_ksp_converged_reason -fieldsplit_schur_ksp_rtol <r>
-fieldsplit_schur_ksp_max_it <m> [-fieldsplit_schur_ksp_gmres_restart 60]`.

**Metric.**
* efficiency = `log10(r₀ / r_final) / t_ksp` (residual decades per second of
  outer-solve wall time);
* seconds per decade;
* N = 25 is a solver proxy only. No tail or physical conclusion is drawn
  from it.

## 2. Stage 1 (N = 25, 6 outer iterations, 300 s cap each)

| config | Schur rtol / max_it | `α_u` | decades | t s | **decades/s** | s/decade | s/outer | Schur its/outer (capped) | spatial its mean |
|---|---|---|---|---|---|---|---|---|---|
| A (baseline) | 1e-6 / 100 | 1e-10 | 0.872 | 127.5 | 0.0068 | 146 | 21.2 | 100 (6/6) | 18.9 |
| B | 1e-4 / 50 | 1e-10 | 0.914 | 64.2 | 0.0142 | 70 | 10.7 | 50 (6/6) | 19.0 |
| C | 1e-3 / 30 | 1e-10 | 0.453 | 37.3 | 0.0122 | 82 | 6.2 | 30 (6/6) | 19.0 |
| D | 1e-6 / 50 | 1e-10 | 0.914 | 62.6 | **0.0146** | 68 | 10.4 | 50 (6/6) | 19.0 |
| E | 1e-3 / 100 | 1e-10 | 0.872 | 126.5 | 0.0069 | 145 | 21.1 | 100 (6/6) | 18.9 |
| A (baseline) | 1e-6 / 100 | 1e-11 | 0.376 | 124.5 | 0.0030 | 331 | 20.8 | 100 (6/6) | 19.0 |
| B | 1e-4 / 50 | 1e-11 | 0.442 | 64.3 | 0.0069 | 146 | 10.7 | 50 (6/6) | 19.0 |
| C | 1e-3 / 30 | 1e-11 | 0.389 | 40.0 | **0.0097** | 103 | 6.7 | 30 (6/6) | 19.0 |
| D | 1e-6 / 50 | 1e-11 | 0.442 | 65.1 | 0.0068 | 147 | 10.8 | 50 (6/6) | 19.0 |
| E | 1e-3 / 100 | 1e-11 | 0.376 | 129.5 | 0.0029 | 344 | 21.6 | 100 (6/6) | 19.0 |

**What this shows:**

* **The Schur rtol is never active.** Every Schur solve, at every rtol from
  1e-6 to 1e-3 and every cap, stopped at its cap. A ≡ E and B ≡ D give
  identical outer histories. The cap alone sets the inexactness.
* **The cap is over-generous (D vs baseline).** Halving it to 50 gives the
  same or slightly better outer reduction per outer iteration (1e-10: 0.914
  vs 0.872 decades; 1e-11: 0.442 vs 0.376) at half the cost. Schur iterations
  51–100 are wasted work.
* **Cutting further (C, cap 30)** cuts outer progress per iteration at 1e-10
  (0.45 decades) but not clearly at 1e-11. Its efficiency is comparable to
  cap 50.
* **Spatial solves are unaffected** (≈ 19 iterations, no failures). That
  confirms the earlier diagnosis; GAMG was not investigated further.

**Why the cap is wasted: the Schur-split GMRES stagnates.** One baseline
Schur solve at `α_u = 1e-11` (monitor probe):
* residual 5.52e-4 → 3.20e-4 after one iteration;
* → 2.53e-4 after 30;
* → 2.48e-4 after 100.

The total reduction is only a factor of 2.2. Each GMRES(30) restart (at 30,
60, 90) is followed by a near-flat stretch. The matching-preconditioned Schur
operator is barely contracted by GMRES in this regime.

**Under-solving hypothesis (Stage 2): rejected without a run.** Doubling the
Schur work (cap 50 → 100) gave equal or worse outer progress at both `α_u`,
and the Schur solve cannot reach even rtol 1e-3 within 100 iterations. A
tighter target (1e-7 or 1e-8, max_it 200) would only add capped iterations on
a stagnating GMRES. That direction was discarded as instructed.

**Restart (tested because of the visible restart plateaus).** Cap 50 with
Schur restart 60 (i.e. no restart within the cap):

| `α_u` | decades | decades/s | vs cap 50 / restart 30 |
|---|---|---|---|
| 1e-10 | 0.704 | 0.0112 | worse (0.914, 0.0146) |
| 1e-11 | 0.133 | 0.0022 | much worse (0.442, 0.0069) |

Discarded. The outer FGMRES is sensitive to *which* inexact Schur solution it
receives, not just to its accuracy. That sensitivity is itself a sign of an
ill-conditioned outer problem.

**Alternative preconditioners (§6 of the task): none tested.**
* ADDITIVE_GAMG is known to fail at small `α_u` (ex20 and Mathematics 2025
  Ex. 3).
* SCHUR_MATCHING uses the **same** matching approximation `Ŝ`, with inexact
  space-time GMRES/GAMG inner solves in place of exact sweeps. It passes the
  2D convection gates but fails in 1D CN (reason −11). Since the measured
  bottleneck is the stagnation of GMRES on the `Ŝ`-preconditioned Schur
  operator, it cannot address the issue.
* The boundary-reduced and Dirichlet-reduced exact Schur solvers exist only
  for boundary control.

**There is no validated alternative of a different family for distributed
convection–diffusion control.** Step 6 was skipped.

## 3. Stage 3: validation at N = 50 (8 outer iterations, 600 s cap each)

| config | `α_u` | decades (8 its) | t s | **decades/s** | gain vs A | s/outer | Schur its/outer | spatial its mean | outer residual history |
|---|---|---|---|---|---|---|---|---|---|
| A cap 100 | 1e-10 | 1.611 | 329.5 | 0.0049 | 1 | 41.2 | 100 | 18.4 | 2.26e-1 1.14e-1 1.85e-2 9.18e-3 5.59e-3 5.56e-3 5.55e-3 5.54e-3 5.53e-3 |
| B cap 50 | 1e-10 | 1.665 | 166.0 | **0.0100** | **2.0×** | 20.8 | 50 | 18.4 | 2.26e-1 1.47e-1 2.10e-2 1.66e-2 1.11e-2 7.86e-3 7.28e-3 4.91e-3 4.89e-3 |
| C cap 30 | 1e-10 | 0.909 | 99.3 | 0.0092 | 1.9× | 12.4 | 30 | 18.4 | 2.26e-1 1.50e-1 9.79e-2 8.98e-2 6.97e-2 6.69e-2 5.73e-2 4.05e-2 2.79e-2 |
| A cap 100 | 1e-11 | 0.646 | 333.9 | 0.0019 | 1 | 41.7 | 100 | 18.5 | 2.26e-1 1.87e-1 1.04e-1 6.72e-2 6.70e-2 6.37e-2 5.83e-2 5.38e-2 5.10e-2 |
| B cap 50 | 1e-11 | 0.886 | 168.8 | **0.0052** | **2.7×** | 21.1 | 50 | 18.5 | 2.26e-1 1.87e-1 1.43e-1 5.11e-2 4.69e-2 4.67e-2 4.50e-2 3.50e-2 2.94e-2 |
| C cap 30 | 1e-11 | 0.778 | 98.7 | 0.0079 | 4.1× | 12.3 | 30 | 18.5 | 2.26e-1 1.89e-1 1.83e-1 1.52e-1 4.50e-2 3.97e-2 3.97e-2 3.97e-2 3.77e-2 |

**Comparisons:**

* **Equal outer iterations.** B matches or beats the baseline at both `α_u`
  (1.67 vs 1.61 decades; 0.89 vs 0.65). C trails at 1e-10.
* **Equal wall time (≈ 100 s).** The baseline has done only 2 outer
  iterations (1e-10: 1.85e-2; 1e-11: 1.04e-1). B, after 4 iterations, is at
  1.11e-2 and 4.69e-2. C, after 8, is at 2.79e-2 and 3.77e-2.
* **Seconds per outer iteration scale with the cap:** 41 / 21 / 12 s. Spatial
  iterations are unchanged.
* **The 1e-11 degradation persists for every setting.** The efficiency ratio
  1e-10 : 1e-11 is 2.5 (A), 1.9 (B), 1.2 (C). Per outer iteration, 1e-11
  still achieves about half of the 1e-10 reduction or less.
* **No projection to rtol is made.** The histories are clearly
  non-asymptotic: plateaus with occasional jumps, and C's 4.1× rests on one
  jump at iteration 4 followed by a flat plateau.
  * The 3 h baseline diagnostics (`AULISA_GILLIAM_SOLVER_ROBUSTNESS_AUDIT.md`
    §5) show the rate falling further over longer horizons. At N = 50,
    `α_u = 1e-11`: 0.65 decades in the first 8 outer iterations but only 1.63
    decades after 128.
  * Short-window efficiencies therefore overstate the long-run rate, for all
    settings alike.

## 4. Classification: **B — modest improvement only**

* **Robust gain:** cap 50 (B/D), 2.0× at `α_u = 1e-10` and 2.7× at `1e-11`,
  consistently across N = 25 and 50, with no loss per outer iteration.
* **Erratic gain:** cap 30 (C), 1.2–4.1×. It depends on single jumps of a
  non-asymptotic history and is not "healthy outer convergence".
* **Outer behaviour:** the `α_u` degradation and the outer plateaus remain
  for every setting.
* **Mechanism:** the Schur-split GMRES stagnates on the matching-preconditioned
  Schur operator. A higher cap or a longer restart does not help, and a lower
  cap only removes wasted work.

**The main issue is the parameter robustness of the current matching-Schur
approximation in this non-colocated 2D regime, not a bad tolerance choice.**

## 5. Is an N = 400, `α_u = 1e-11` production retry justified? **No.**

* **Computationally.**
  * The aborted baseline ran 10.4 h without converging.
  * At N = 50 the baseline reached only 1.6 of the required 8 decades in 3 h,
    with a decaying rate.
  * A 2–3× constant-factor gain does not change this picture. A retry would
    most likely still take many hours to days with uncertain completion,
    which is not a "reasonable wall time".
* **Scientifically.**
  * The completed `α_u = 1e-10` point (true residual 8.0e-9, tail 0.1520 on
    nx 32) already establishes practical saturation over the tested
    regularisation range: −1.3 % in tail vs `1e-9`, while the control norm
    grows 4×.
  * A better solver cannot change converged optima; it would only make
    smaller `α_u` cheaper to probe.
  * The factor-≈ 3 discrepancy with Table 10 is not a Krylov error, and no
    claim is made that `α_u = 1e-11` would change the practical-saturation
    conclusion.

## 6. Recommended settings for future non-colocated small-`α_u` runs

* **Use `-fieldsplit_schur_ksp_max_it 50`** (rtol irrelevant in this regime;
  e.g. keep 1e-6 or use 1e-4). In the proxies this is ≈ 2–2.7× cheaper per
  decade with no loss per outer iteration.
  * It changes only the inexactness of an inner solve inside FGMRES. The outer
    rtol and the true-residual check still certify the final optimum.
  * It has **not** been validated by a complete N = 400 solve. Before
    adopting it as a production default, rerun one completed point (e.g.
    `α_u = 1e-9` or `1e-10`, N = 400) and confirm identical optima within
    tolerance.
* Keep the Schur restart at 30: restart 60 was worse.
* Do not tighten the Schur rtol: it is never reached.
* Do not rely on MPI to fix the cost. At this mesh size np = 4 gave only
  ≈ 1.5× (audit §8).
* For genuinely smaller `α_u` in non-colocated configurations, a different
  Schur approximation, not tuning, would be needed. That is out of scope here,
  and nothing was designed.

## 7. Scientific conclusions: unchanged

* `α_u = 1e-10` remains a valid discrete optimum.
* The EF 9.6 classification (B for practical saturation, C for the published
  `E₅`), the factor-≈ 3 discrepancy with Table 10, and the "no asymptotic floor
  / no unattainability" caveats are unaffected.
* No benchmark report was changed by this investigation, apart from the
  cross-reference added to `ERRORFEEDBACK_96_BENCHMARK.md` §4b and to
  `AULISA_GILLIAM_SOLVER_ROBUSTNESS_AUDIT.md`.
