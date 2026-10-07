# T8D finalisation — iterative Dirichlet core as the ex21 default; ex20/ex21 zero-velocity parity

*No new physics, no solver research, no 3D. Nothing committed, staged or pushed.*

Scripts, logs and results are in `t8d_final/`. The parity table is
`ex20_ex21_zero_velocity_parity.csv`.

The binaries involved:

| binary | sha | role |
|---|---|---|
| new ex21 | 8eae9680 | this task |
| T8D ex21 | a2e6f680 | reference; dense default |
| ex20 | built 2026-10-03 | unmodified source |

## 1. New defaults

| option | old default | **new default** |
|---|---|---|
| `--dirichlet-core-solver` | dense | **iterative** |
| `--dirichlet-core-pc` | rt | rt |
| `--dirichlet-core-rtol` | 1e-10 | **1e-8** |

**Kept unchanged:** `dense` (the reference / gold solver), `--dirichlet-core-maxit`, and
`--verify-dirichlet-core`. No option was removed.

**Two interface consequences of making a core option the default:**

1. **The refusal now keys on explicit use.** Validate refused the core options without
   Dirichlet control. With iterative as the *default* that would have broken every
   distributed and Neumann run. It now refuses only **explicitly given** `--dirichlet-core-*`
   options (`DirichletCoreConfig::explicitly_set`) and `--verify-dirichlet-core`. The
   default is inert for non-Dirichlet control: verified, no new output lines.
2. **`--verify-dirichlet-core` selects the dense reference itself.** It is a dense-vs-shell
   test. Combining it with an **explicit** `--dirichlet-core-solver iterative` is refused
   with a clear message, not ignored.

## 2. Default-change regression (`t8d_final/results/default_change.csv`, `logs/default/`)

16 cases: heat/convection × BE/CN × L2 / H1-space / H1-time / combined, nx = 16, N = 32,
production path.

| check | result |
|---|---|
| every Dirichlet production run now uses the iterative core by default | ✔ `EX21_T8D_CORE_STATS mode=iterative pc=rt rtol=1.0e-08` in all 16 |
| explicit `--dirichlet-core-solver dense` = old default | ✔ **JSON byte-identical and every log line identical** (16/16; only the JSON file path in the log differs) |
| new default vs old default | fields ≤ 9.1e-12 (≤ 2.6e-14 in 14/16), J_track ≤ 1.8e-13, J_reg ≤ 5.6e-13 |
| outer iterations | identical in all 16 |
| true residual | equal to 2–3 digits |
| core CG iterations | 6–9 |

**The T7D gate battery with the new default** (18 runs, no core flag;
`t8d_final/logs/gates/`, `gates_summary.txt`):

- every non-production gate line is identical to T7D;
- production outer counts are identical;
- true residual ≤ 3.6e-11.

One documented change: in the single-outer-iteration runs, the production consistency
diagnostic ‖∇J(u*)‖ / ‖dt α_u R u*‖ rises from ≤ 3e-12 (dense) to ≤ **1.5e-10** (iterative,
rtol 1e-8). That is the expected footprint of the 1e-8 inner tolerance, still two orders
below the outer tolerance (1e-8). No test tolerance was weakened.

**Exact-OC check (`dirichlet-conv`, BE/CN × L2 / H1-space):**

- explicit dense is IDENTICAL to the old default;
- the new default reproduces the exact-solution errors y, p, u to all printed digits,
  with the same outer iterations.

## 3–7. ex20 vs ex21 zero-velocity parity

ex21 was run in two zero-velocity modes: **"off"** (`--convection off`, the default) and
**"w0"** (`--convection on --vx 0 --vy 0`).

### Spatial operators (`t8d_final/logs/parity_dumps.log`)

`--dump-operators` from the three binaries, for distributed, Neumann and Dirichlet
control: M, K, tracking mass, M_ctrl_whole, M_ctrl_box, M_Γ, the selected control
operator N and the coordinates.

**24/24 byte-identical, in both ex21 modes.** With w = 0 the advective term adds exact
±0, so it changes no bits.

### Solutions (`t8d_final/results/parity_solutions.csv`)

Every solver available for each control type was run with identical flags (ex20
defaults otherwise), nx = 8, N = 16, BE and CN.

| control / solver | ex21 off vs ex20 | ex21 w0 vs ex20 |
|---|---|---|
| distributed / Neumann, ADDITIVE_GAMG (default) | **byte-identical** (fields, JSON, log lines) | byte-identical fields/JSON; log differs only by the label `convection=on` |
| distributed / Neumann, SCHUR_MATCHING | byte-identical (including the identical CN divergence of distributed MATCHING, reason −11, **pre-existing in ex20**) | same as left; label only |
| distributed / Neumann, SCHUR_SWEEP (GMRES+GAMG and LU spatial) | **byte-identical** | ≤ 3.8e-15, same outer its: the **transposed backward-sweep path** (`HasConvection`) |
| Neumann, SCHUR_BOUNDARY_REDUCED | **byte-identical** | **refused** (`--pc-schur-boundary-reduced` keys on the convection flag; T7N restriction) |
| Dirichlet, explicit dense core (all 4 regularisations) | **byte-identical** | ≤ 2.2e-15, same outer its: LU instead of Cholesky, transposed B_Dᵀ / CN coupling |
| Dirichlet, **new default** (iterative core) | ≤ 2.3e-14, J ≤ 1e-15, same outer its | ≤ 2.3e-14 |

**Every** J_track, J_reg and KKT residual is bitwise equal wherever the fields are, and
otherwise agrees to ≤ 2e-15 relative.

### Answers

- **5. Convection off vs on with w = 0.** The **discrete operators are identical**, but
  the two modes are **not** the same code path. `HasConvection(config)` is a *flag* test,
  not a test of the value of w. It selects:
  - `MatMultTranspose(K)` in the adjoint shell (bitwise harmless for symmetric K);
  - Aᵀ-rebuilt backward-sweep blocks and the transposed CN coupling in the sweep
    preconditioners (round-off);
  - LU instead of Cholesky, and the transposed B_Dᵀ, in the Dirichlet solver
    (round-off);
  - the refusal of the boundary-reduced Schur;
  - a `convection` block in the JSON/log.

  These are **path differences only, not bugs**. They are the T7-series design
  (convection-aware paths are chosen by the flag so that a nonzero w can never reach an
  SPD-only path). Left unchanged.
- **6. Neumann.** ex21's convection-capable SCHUR_SWEEP reduces to ex20's: **bitwise**
  with convection off, ≤ 2.8e-15 with w = 0. Control reconstruction, adjoint and
  objective agree in the same way.
- **7. Dirichlet.**
  - ex21 **dense** heat = ex20 **byte-for-byte** (8/8 cases, and in T8D 8/8 at nx = 16).
  - ex21 **iterative** (the new default) agrees with ex20 to ≤ 2.3e-14.
  - The only difference caused by the new default is this inner-tolerance-level deviation.

## 8. Solver / default differences and final mapping

| formulation | default solver (ex20 = ex21) | validated alternative(s) | known limitation |
|---|---|---|---|
| distributed heat | ADDITIVE_GAMG | **SCHUR_SWEEP** (GMRES+GAMG or LU spatial); SCHUR_MATCHING (BE) | ADDITIVE_GAMG under **CN** converges on small problems but fails at nx = 16, N = 32 (reason −3) **identically in ex20 and ex21**, even with ex20's own defaults: pre-existing. SCHUR_MATCHING diverges under CN (−11) in both. → use `--pc-schur-sweep` for CN |
| distributed convection | ADDITIVE_GAMG | **SCHUR_SWEEP** (LU/ILU/ASM at high Pe; GAMG+LPS) | ADDITIVE_GAMG: BE only (CN fails, as for heat), iterations grow with Pe |
| Neumann heat | ADDITIVE_GAMG | **SCHUR_SWEEP**, SCHUR_BOUNDARY_REDUCED, SCHUR_MATCHING | ADDITIVE/MATCHING fail for Neumann CN at realistic sizes (T7N, pre-existing; they converge at nx = 8, N = 16) |
| Neumann convection | ADDITIVE_GAMG | **SCHUR_SWEEP** | boundary-reduced refused under the convection flag (also for w = 0); ADDITIVE/MATCHING not robust (T7N) |
| Dirichlet heat | SCHUR_DIRICHLET_REDUCED, **core: ex20 dense / ex21 iterative (new)** | ex21 `--dirichlet-core-solver dense` (bitwise ex20) | sweep/matching/boundary-reduced refused (N_D not block diagonal); ADDITIVE_GAMG has no CLI path |
| Dirichlet convection | SCHUR_DIRICHLET_REDUCED, iterative core (CG + R_H1t, rtol 1e-8) | dense core (reference, memory O((N_t\|C\|)²)) | CG iterations grow only for small α_u (74 at 10⁻⁵) |

**Default differences between ex20 and ex21:** exactly one, the Dirichlet core (dense →
iterative). Distributed and Neumann defaults are unchanged and identical. The ADDITIVE_GAMG
CN limitation is **not** a correctness issue: non-converged solves are rejected explicitly
by the T6 acceptance guard (exit 8), and ex20 has the same behaviour. So the distributed
default was **not** changed, as instructed. Recommendation: pass `--pc-schur-sweep` for
CN runs; a future default switch for CN is a reasonable small follow-up.

## 10. Regression (`t8d_final/logs/regress.out`; OLD = T8D binary a2e6f680)

| block | result |
|---|---|
| ex20 goldens | **61/61 identical** |
| Math2025 Ex3/Ex4 | byte-identical |
| Ex5 | byte-identical |
| EF 9.5 | only the archived `variant=source` label line (pre-existing since T7N) |
| T0–T2 | PASS |
| T3–T4 representative (incl. expected failures) | identical |
| old vs new, every `EX20_/EX21_` line | **45/49 identical**; the other 4 are the `--exact-oc dirichlet-conv` runs, which take the new default (rerun with explicit dense: identical; default: same errors and outer its, §2) |

The 45 identical old-vs-new comparisons cover:

- **T7N:** Neumann heat (3 PCs, BE/CN) and the T7N exact/production runs;
- **T7D:** Dirichlet heat with all four regularisations, plus the larger dense-core cases,
  production H1-space + H1-time and the gates, all with explicit dense;
- **T7F:** `natural-conv` exact and multi-zone + two sources;
- the audit multi-zone runs, `conv-sin3` / `lift-exp`, T6 LPS, and the natural-BC gates.

**T8D gold verification:** 32/32 cases, worst action 3.6e-14, exact symmetry,
λ_min > 0, max κ(R⁻¹K) = 3.40. This is identical to T8D, now running dense automatically.

**Zero-wind parity:** see §3–7.

**Restrictions:** all previous refusals are unchanged, plus:

- verify with explicit iterative → refused;
- the default core on distributed control → inert (not refused, no effect).

## 11. Verdict

1. **ex21 with convection OFF reduces to ex20: yes, byte-for-byte**, for every shared
   formulation and solver (distributed, Neumann, Dirichlet with the dense core; BE and
   CN; all four Dirichlet regularisations). The only ex21 ≠ ex20 output in this mode is
   Dirichlet with the **new default core** (≤ 2.3e-14).
2. **ex21 with convection ON and w = 0** has **identical discrete matrices** and the same
   algebra. It reduces to ex20 **numerically** (≤ 4e-15) but **not byte-for-byte** on the
   flag-selected transposed / LU paths (SWEEP, Dirichlet), and it **refuses** the Neumann
   boundary-reduced solver.
3. **Mathematical differences:** none at w = 0.
4. **Solver / default / path differences only:**
   - the Dirichlet core default (iterative vs ex20's dense);
   - flag-selected transposed and LU paths, and the boundary-reduced refusal, when
     `--convection on`;
   - the JSON/log `convection` label.
5. **Is the iterative Dirichlet core safe as the production default? Yes.** Dense is
   reproduced exactly on request. The default agrees with dense to the inner tolerance,
   with identical outer counts, all gates pass, and the full regression passes.
6. **Is the 2D codebase ready for the 3D extension? Yes.**
   - The 2D convection feature set is closed (T7F).
   - The Dirichlet memory wall is removed (T8D).
   - ex21 is a strict superset of ex20 at zero velocity.

   Carry these items into 3D, none of them blocking:
   - choose SCHUR_SWEEP explicitly for CN (ADDITIVE_GAMG CN limitation, inherited from
     ex20);
   - the remaining all-at-once memory (about 157 space-time vectors);
   - sequential-in-time sweeps.
