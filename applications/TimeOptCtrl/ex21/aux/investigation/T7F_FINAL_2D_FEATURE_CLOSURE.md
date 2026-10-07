# T7F — final 2D convection feature closure (ex21)

*Scope: close the three remaining 2D gaps of `EX21_FEATURE_PARITY_AUDIT.md`:*

1. *multi-zone control/tracking;*
2. *multiple source regions;*
3. *natural state BC + convection.*

*Nothing else was changed. No 3D work. Nothing committed, staged or pushed.*

All scripts are in `t7f/scripts/` and all logs in `t7f/logs/`. Results are in
`t7f/results/` and in `t7f_multizone.csv`, `t7f_multisource.csv`,
`t7f_natural_bc.csv`, `t7f_solver.csv` and `feature_parity_final.csv`. The ≈1.3 GB of
JSON fields used by the self-convergence and MPI comparisons are kept **outside** the
repository; the scripts point at them through `T7F_JSON`.

## 1. Pre-3D checklist (written first; final column filled at the end)

| feature | previous status (audit + T7N + T7D) | required evidence | final status |
|---|---|---|---|
| full-domain distributed control | VALIDATED (LP T1–T6, exact OC) | none (closed) | VALIDATED |
| subdomain distributed control | VALIDATED (M25 Ex3/Ex5, EF) | none | VALIDATED |
| **multi-zone distributed control** | SMOKE (gates only) | masks/union/overlap; gates; true residual; march; BE 1 / CN 2; Q2 spatial order away from interfaces | **VALIDATED WITH DOCUMENTED LIMITATION** (§2) |
| Neumann boundary control | VALIDATED (T7N) | none | VALIDATED WITH DOCUMENTED LIMITATION (T7N) |
| Dirichlet boundary control (+ H1-space / H1-time / combined) | VALIDATED (T7D) | none | VALIDATED WITH DOCUMENTED LIMITATION (T7D) |
| **multi-zone tracking** | SMOKE | as multi-zone control | **VALIDATED** (§2) |
| full-domain / subdomain / non-colocated tracking; time-dependent target | VALIDATED | none | VALIDATED |
| whole-domain source; single source region | VALIDATED | none | VALIDATED |
| **multiple source regions** | IMPLEMENTED-UNVALIDATED (preset-only) | ≥ 2 regions at once; each contributes; removal; overlap; BE/CN; KKT; march; MPI | **VALIDATED WITH DOCUMENTED LIMITATION** (§3) |
| homogeneous / inhomogeneous Dirichlet state BC | VALIDATED | none | VALIDATED / VALIDATED WITH DOCUMENTED LIMITATION |
| **natural state BC + convection** | SMOKE (gates only) | exact OC with ν∂ₙp + (w·n)p = 0 exercised (w·n ≠ 0, p ≠ 0 on ∂Ω); G1N; gates; Q2 space; BE 1 / CN 2; 2 wind orientations | **VALIDATED** (§4) |
| BE, CN, L2, constant/variable wind, exact transpose, LPS, high-Pe behaviour | VALIDATED (with T5/T6 caveats) | none | unchanged (§9) |

Nothing that was already closed was rerun, except as regression (§8).

## 2. Multi-zone distributed control / tracking

**Problem.**

- Ω = (0,1)², Q2, ν = 1/4, w = (1, 0.5), α_u = 10⁻², target sin_x_minus_t, T = 1.
- Natural state BC; homogeneous-Dirichlet variants are covered by the gates.
- Two disjoint control boxes C₁ = [⅛,⅜]², C₂ = [⅝,⅞]².
- Two tracking boxes T₁ = [½,⅞]×[⅛,⅜] and T₂ = [⅛,½]×[½,⅞]. They are non-colocated:
  no element is shared with C (checked on the matrices).
- All box edges lie on the 1/8 lattice, so the boxes are element-aligned for every
  nx = 8k.

**Why no closed form.** With tracking restricted to ω, an exact optimum needs an
adjoint p that solves the homogeneous backward adjoint equation off ω, vanishes at T,
and is nonzero on the control set C, which is disjoint from ω. Building one requires the
T7D image-kernel machinery, glued to each box. That is the "artificial" case the brief
anticipates. The feature is therefore validated with **algebraic gates plus
fine-reference convergence**.

**Masks** (`t7f/scripts/t7f_masks.{sh,py}`, operator dumps, nx = 8 and 16;
`logs/masks.log`, 34/34 PASS):

| check | result |
|---|---|
| 1ᵀM_ctrl1 = area of C₁ ∪ C₂ = 0.125; 1ᵀM_track1 = area of T₁ ∪ T₂ = 0.234375 | exact |
| M_ctrl(C₁ ∪ C₂) = M_ctrl(C₁) + M_ctrl(C₂); same for tracking | identical, 0.0 |
| N_selected = M_ctrl_box (the union is the control operator) | identical |
| control and tracking share no entry (non-colocated) | 0 |
| overlapping A ∪ B ≡ one box covering A ∪ B (control, tracking, source) | identical, 0.0 |
| … and ≠ M(A) + M(B), i.e. no double counting | differs, 4.4e-3 / 1.1e-3 |
| centroid selection on the non-aligned box (0.3, 0.7)² | area = (#centroids inside · h)² exactly (0.25 at nx = 8, 0.140625 at nx = 16) |

**Overlap precedence, "lowest index wins"** (`t7f_region_index.py`, read from the
`--write-region-fields` VTU, overlapping boxes, nx = 8): every element in A ∩ B carries
index **1** (control: 2 overlap cells; tracking: 3). All 64 cells match the rule.
`logs/region_index.log`: PASS.

**Gates** (§5): multi-zone with natural BC, with homogeneous-Dirichlet BC, and with
overlapping control boxes, all BE and CN. Every check passes, and every negative
control fails.

**Fine-reference self-convergence** (`t7f_selfconv.{sh,py}`, `results/selfconv_mz.csv`;
SCHUR_SWEEP + LU; true L2 norms; u integrated over the control union only):

| ladder | y orders | u orders | p orders | J diff orders |
|---|---|---|---|---|
| time, BE (nx = 16, ref Nt = 512) | 1.12, 1.10, 1.13, (1.24) | 1.04, 1.06, 1.10, (1.22) | 1.03 – 1.10, (1.22) | |
| time, **CN** | **1.92, 1.90, 1.90, 1.91** | **1.98, 2.02, 2.07, 2.15** | 1.94 – 2.16 | 2.00 – 2.07 |
| space, BE (Nt = 32, ref nx = 128) | 2.71, 2.80, 2.72 | **2.98, 2.98, 2.86** | 2.87, 2.90, 2.80 | 3.99 – 4.09 |
| space, CN | 2.32, 2.21, 2.40 | **2.98, 2.98, 2.86** | 2.86, 2.90, 2.79 | 3.99 – 4.09 |

Values in parentheses carry the reference-gap bias of a 4× reference (expected ≈ 1.22).

**Interface limitation, measured.** 85 % (BE) and 97 % (CN) of ‖e_y‖² lie within 1/16
of the control-box boundaries, a band that covers 24 % of the area. The state is driven
by the discontinuous χ_C u, so y has only the interface regularity of a
discontinuous-source parabolic problem. The control and the adjoint keep the Q2 rate
(u 2.98, p 2.9). This is the "region-interface regularity limitation" anticipated by
the brief, not a defect.

**MPI** (§7): np = 1/2/4 agree to ≤ 4.3e-15 in y, p and u, with identical outer
histories.

**Status: multi-zone distributed control — VALIDATED WITH DOCUMENTED LIMITATION.**
The limitations: the state's spatial order is ≈ 2.2–2.8 near control-box interfaces;
selection is by element centroid (axis-aligned boxes); there is no closed-form optimum.

**Status: multi-zone tracking — VALIDATED.** Masks are exact, there is no interface
loss in p, and the tracking forcing enters only through the adjoint, which converges at
2.8–2.9.

## 3. Multiple source / disturbance regions

**Interface (tiny, as permitted).** A new CLI option, `--source-box-add xmin xmax zmin zmax`,
pushes one box onto `ProblemConfig::source_regions`; that list was previously reachable
from presets only.

- Validation and assembly are unchanged.
- The convention is unchanged: one source field f, loaded as M_source I_h f over the
  **union** of the boxes, with each element counted once.
- An **empty** list still means the whole domain.

`--dump-operators` additionally writes `M_source`, but only when source regions exist,
so every existing dump keeps its file set.

**Test** (`t7f_multisource.{sh,py}`, `logs/multisource.log`, `t7f_multisource.csv`;
83/83 PASS):

- Production path, natural BC, w = (1, 0.5), nx = 16, N = 32, SCHUR_SWEEP + LU.
- S₁ = [0,¼]×[¾,1] and S₂ = [¾,1]×[0,¼], in opposite corners.
- The baseline z₀ uses a zero amplitude (`--source 0`), because an empty list means the
  whole domain.

| check (y, p, u; BE and CN) | result |
|---|---|
| both regions contribute: max\|z₁−z₀\| / max\|z₁₂−z₀\|, likewise for region 2 | y 0.99 / 0.68; p 0.88 / 0.12; u 0.88 / 0.12 (BE) |
| the two contributions differ | yes |
| superposition z₁₂ − z₀ = (z₁ − z₀) + (z₂ − z₀) | ≤ 3.8e-12 |
| removing region 1 (z₁₂ − z₂ = z₁ − z₀); removing region 2 | ≤ 3.8e-12 |
| amplitude 2.5: z − z₀ = 2.5 (z₁₂ − z₀) | ≤ 1e-11 |
| two regions differ from a whole-domain source | yes |
| overlapping boxes ≡ a single union box | **bitwise** |
| the baseline does not depend on the boxes | bitwise |
| true KKT residual | ≤ 1.7e-15 |
| MPI np = 1/2/4 | y, p, u ≤ 4.1e-15; J ≤ 8.3e-16; outer iterations 2/2/2 |

Gates with two source regions (natural BC, and Dirichlet BC with multi-zone control and
tracking, BE and CN) pass, including the independent forward march (≤ 1.4e-15).

**Status: multiple source regions — VALIDATED WITH DOCUMENTED LIMITATION.**

- One source field on the union; separate per-region amplitudes are not an interface.
  By the superposition shown above they are equivalent to z₀ + a₁δ₁ + a₂δ₂.
- Element-centroid selection.
- The source **shape** (anything other than the constant `--source` amplitude) is still
  preset/override-only.

## 4. Natural state BC + convection: exact formulation (`--exact-oc natural-conv`)

The problem:

- Ω = (0,1)², distributed control and tracking on the whole domain, **natural state BC
  on all four faces**.
- Constant wind with both components free (default (1, 0.5); any (v_x, v_y) through
  `--exact-oc-velocity`).
- ν = 1/4, α_u = 10⁻².

With R_k(s) = e^{−ks}(2 + cos πs), we have R_k′ = −k R_k wherever sin πs = 0, so at
s = 0 and s = 1. Setting

    p = α_u B(t) R_{v_x/ν}(x) R_{v_y/ν}(z),   B = sin(T − t),
    y = A(t) cos πx cos πz,                    A = cos 2t,
    u = −(η/α_u) p,  f = y_t − νΔy + w·∇y − ηu,  y_d = y + (p_t + νΔp + w·∇p)/α_t,

gives:

- ν∂ₙp + (w·n)p = 0 on all four faces, with **p ≠ 0 there**, so the convective term
  carries the condition;
- ν∂ₙy = 0 on all four faces;
- p(T) = 0.

**Symbolic verification** (`t7f_symbolic.py`, `logs/symbolic.log`, 37/37 PASS for
w = (1, ½) and (−1, −½)):

- N1, the Robin adjoint BC on each face: exactly 0.
- N2, |(w·n)p| on the faces: 3.5e-4 – 5.8e-2. This is exactly the amount by which the
  wrong condition ν∂ₙp = 0 would be violated.
- N3, the state Neumann BC and p(T): exactly 0.
- N4, the C++ literal port of R, p, f and y_d against sympy: ≤ 1e-12.
- N5, the adjoint and state residuals: exactly 0.

A third wind, (2, −1), is covered by the gates.

## 5. Algebraic gates (`t7f_gates.sh`, `results/gates.csv`, 16 runs: nx = 8, N = 8, SCHUR_SWEEP + LU)

| gate | natural-conv (3 winds) | multi-zone (natural / Dirichlet / overlap) | two sources |
|---|---|---|---|
| G1N boundary identity ‖C + Cᵀ − Σ_f (w·n_f) M_f‖ / ‖C + Cᵀ‖ | 3.2e-14 – 3.6e-14 | 3.5e-14 | 3.5e-14 |
| G2 exact transpose | ≤ 1.7e-17 | ≤ 2.2e-17 | ≤ 1.2e-17 |
| ex20-style adjoint (negative) | 2.1e-4 – 3.6e-3 | 1.8e-4 – 9.7e-4 | 6.2e-5 – 1.7e-3 |
| G8b wrong natural BC νK − C (negative) | 5.6e-5 – 3.4e-4 | 1.2e-4 – 2.1e-4 | 1.2e-4 – 2.1e-4 |
| reduced-Hessian symmetry | ≤ 2.0e-11ᵃ | ≤ 1.6e-15 | ≤ 1.3e-15 |
| FD gradient, correct adjoint | ≤ 1.4e-11ᵃ | ≤ 1.7e-14 | ≤ 1.0e-14 |
| FD gradient, ex20-style / wrong-BC adjoint (must fail) | 0.029 – 0.83 / 0.097 – 0.72 | 0.10 – 0.50 / 0.017 – 0.26 | 0.011 – 0.87 / 0.21 – 0.27 |
| true KKT residual (2 outer its) | ≤ 2.4e-12 | ≤ 6.3e-15 | ≤ 2.8e-15 |
| ‖∇J(u*)‖ rel | ≤ 7.9e-12 | ≤ 6.4e-13 | ≤ 3.6e-13 |
| independent forward march | ≤ 1.2e-13 | ≤ 1.5e-15 | ≤ 1.4e-15 |

ᵃ Wind (−1, −½) gives p ∝ e^{4x+2z}, with |J| ≈ 16 – 19 against 2e-3 for the default
wind; the larger relative round-off scales with that. All other runs are ≤ 1e-13.

## 6. Natural BC: convergence (`t7f_natural_bc.csv`, `results/selfconv_natural.csv`)

**Against the closed form:**

| ladder | y | p | u |
|---|---|---|---|
| space, CN, Nt = 256, w = (1, ½), nx 8→16→32 | 3.00, 3.00 | 3.01, 2.88 | 3.01, 2.88 |
| space, nx = 64 | time floor (CN error at Nt = 256 ≈ 2.5e-6 in u) | | |
| space, w = (−1, −½) | 3.60, then time floor at nx = 32 | 2.97, 2.98 | 2.97, 2.98 |
| time, **CN**, nx = 64, w = (1, ½) | 2.14, 2.10, 2.05, 1.94 | 1.96, 1.99, 2.00, 2.00 | 1.96, 1.99, 2.00, 2.00 |
| time, CN, w = (−1, −½) | 2.19, 2.17, 2.11, 2.06 | 1.88 – 1.99 | 1.88 – 1.99 |
| time, **BE**, nx = 32, w = (1, ½) | 0.95, 0.98, 0.99, 1.00, 1.00 | 0.90 – 0.99 | 0.90 – 0.99 |
| time, BE, w = (−1, −½) | 0.90 – 1.00 | 0.48 → 0.97 (pre-asymptotic at coarse dt, large p) | 0.48 → 0.97 |

**Fixed-dt spatial self-convergence** (exact data through `--solve-only`, CN, Nt = 16,
ref nx = 128, true L2 norms):

| wind | y | u | p |
|---|---|---|---|
| (1, ½) | 3.00, 2.99, 2.86ᵇ | 3.01, 2.99, 2.86ᵇ | 3.01, 2.99, 2.86ᵇ |
| (−1, −½) | 3.71, 3.45, 3.06 | 2.97, 2.98, 2.86ᵇ | 2.97, 2.98, 2.86ᵇ |

ᵇ 2× reference-gap bias.

Every run shows:

- stationarity ‖α_u u + ηp‖ ≤ 5.4e-17 relative;
- the independent forward march ≤ 1.8e-12 (≤ 8.6e-14 for w = (1, ½); larger with the 10³× larger wind-(−1, −½) solution);
- a true residual ≤ 1.6e-13.

**Status: natural state BC + convection — VALIDATED.** Q2 order 3 in space, BE 1, CN 2,
two wind orientations with inflow and outflow on every face. The (w·n)p term is
demonstrably active.
## 7. MPI consistency (`t7f_mpi.{sh,py}`, `results/mpi.csv`; multi-source in §3)

Setup: SCHUR_SWEEP with LU (np = 1) or MUMPS LU (np > 1), nx = 16, N = 32. Fields
are matched **by coordinates**: in parallel, the JSON node order follows the partition.
The outer history comes from `-ksp_monitor` on the unprefixed outer FGMRES.

| case | np | max rel Δ y / p / u | J rel Δ | true residual (np = 1) | outer its | outer history max \|Δr_k\| / r₀ |
|---|---|---|---|---|---|---|
| natural-conv BE | 2 / 4 | ≤ 2.1e-15 / 1.3e-15 / 1.3e-15 | ≤ 2.3e-15 | 1.5e-13 (same) | 2 / 2 / 2 | ≤ 3.3e-16 |
| natural-conv CN | 2 / 4 | ≤ 2.4e-15 / 2.0e-15 / 2.0e-15 | ≤ 2.8e-15 | 1.6e-13 (same) | 2 / 2 / 2 | ≤ 2.4e-16 |
| multi-zone BE | 2 / 4 | ≤ 3.2e-15 / 2.1e-15 / 1.6e-15 | 0 | 4.0e-15 (same) | 2 / 2 / 2 | ≤ 7.2e-16 |
| multi-zone CN | 2 / 4 | ≤ 4.3e-15 / 2.5e-15 / 2.0e-15 | 3.6e-16 | 3.2e-15 (same) | 2 / 2 / 2 | ≤ 1.6e-15 |
| two sources BE / CN | 2 / 4 | ≤ 4.1e-15 | ≤ 8.3e-16 | ≤ 1.7e-15 | 2 / 2 / 2 | — |

The first comparison of the histories used |Δr_k| / r_k and flagged the round-off-level
last iterate (r₂ ≈ 1e-13: 7e-6 relative to itself). It is reported relative to the
initial residual instead.

**All PASS.**

## 8. Regression (`t7f/scripts/t7f_regress.sh`, `t7f/logs/regress.out`; OLD = the T7D final binary sha cf170d75, NEW sha 6278531a)

| block | result |
|---|---|
| ex20 goldens | **61/61 identical** |
| Math2025 Ex3 / Ex4 / Ex5 | byte-identical |
| EF 9.5 | only the archived `variant=source` config-label line differs: pre-existing, recorded in T7N and T7D, not a T7F change |
| T0–T2 constant wind | PASS |
| T3/T3b/T4 representative, including the expected failures | identical |
| T7N: Neumann heat (3 PCs), `neumann-conv` exact, Neumann + convection production (BE and CN) | identical |
| T7D: `dirichlet-conv` exact (L2, H1-space), Dirichlet + convection H1-space+H1-time production, Dirichlet gates (BE and CN) | identical |
| Dirichlet heat: L2 / H1-space / H1-time / combined (BE and CN) | identical |
| the feature-audit multi-zone smoke configurations (natural and Dirichlet BC, BE and CN) | identical |
| `conv-sin3` and `lift-exp` exact OC (BE and CN) | identical |
| T6 LPS lp52 | identical |
| distributed and Neumann natural-BC gate runs | identical |

**Restrictions:** everything previously refused is still refused (Dirichlet + sweep /
matching / boundary-reduced, Neumann boundary-reduced and H1-time under convection,
dirichlet-conv with v_x ≠ 0 or γ_u ≠ 0). New explicit refusals: `natural-conv` with a
wind field, and a source box outside the domain.

**No existing default changed.** The source changes are additive only: a new exact-OC
case, a new CLI option, and a dump file written only when source regions exist.
## 9. Solver suitability (verify only; `t7f_solver.{sh,py}`, `t7f_solver.csv`, `logs/solver*/`)

The cases are:

- multi-zone (MZ), nx = 32, N = 32;
- two sources (SRC), nx = 32, N = 32;
- natural-conv (NAT), nx = 32, N = 64, Pe_cell 0.14;
- natural-conv at moderate Pe (NATPE), ν = 0.02, Pe_cell 0.87.

Each is run with three solvers:

- **ref** = SCHUR_SWEEP + LU spatial solves;
- **sw** = SCHUR_SWEEP with its normal iterative spatial backend (GMRES + GAMG);
- **add** = ADDITIVE_GAMG (the distributed default, no `--pc` flag).

| | SCHUR_SWEEP + LU | SCHUR_SWEEP (iterative spatial) | ADDITIVE_GAMG BE | ADDITIVE_GAMG CN |
|---|---|---|---|---|
| MZ | 2 its, res 2.8e-15 / 2.1e-15 | 2 its; fields vs LU ≤ 8.7e-15 | 8 its, res 1.7e-11 | **reason −3**, 200 its, res = ‖rhs‖ (2.1e-3) |
| SRC | 2 its | 2 its; ≤ 5.6e-15 | 11 its | **reason −3**, 200 its (3.4e-2) |
| NAT | 2 its, res 1.7e-13 | 2 its; ≤ 7.6e-15 | 50 its, res 4.1e-9 | stopped by hand after 58 min (no convergence) |
| NATPE (Pe_cell 0.87) | 2 its, res 2.5e-13 | 2 its; ≤ 2.3e-15 | stopped by hand after 37 min (no convergence) | not run |

**SCHUR_SWEEP remains the appropriate solver** for all three new capabilities:

- 2 outer iterations everywhere, BE and CN, with LU or with the iterative spatial
  backend;
- the two backends give identical solutions (≤ 9e-15).

No structural change was needed or made.

**Finding, pre-existing, not caused by T7F: ADDITIVE_GAMG does not converge under CN
for distributed control.** It reaches 200 iterations with the true residual equal to
‖rhs‖, i.e. no reduction at all. Attribution (`logs/solver_attribution/`, nx = 16,
N = 32, w = (1, 0.5), natural BC):

| control / tracking | BE | CN | T7D binary (CN) |
|---|---|---|---|
| multi-zone | 8 its | −3 (res 4.245e-3) | **identical failure** (4.245e-3); BE identical too |
| one box, non-colocated | 8 its | −3 | identical failure (2.643e-3) |
| one box, colocated | 12 its | −3 | identical failure (3.964e-3) |
| full domain | 22 its | −3 | identical failure (9.903e-3) |
| colocated, convection **off** (heat), natural BC | 12 its | **−3** (3.961e-3) | |
| colocated, convection off (heat), Dirichlet BC | 12 its | **−3** (3.947e-3) | |
| colocated, convection on, Dirichlet BC | 12 its | **−3** (3.947e-3) | |

Consequences:

- The failure is independent of multi-zone, multi-source, natural BC **and of convection** (the heat equation fails too), and it is
  present in the T7D binary.
- It is never silent: the T6 acceptance guard rejects the solve (`EX21_SOLVE_REJECTED
  status=ksp_not_converged`, exit 8).
- The earlier audit row "ADDITIVE_GAMG VALIDATED (low Pe)" rested on BE runs. It is
  reclassified **VALIDATED WITH DOCUMENTED LIMITATION: BE only; use `--pc-schur-sweep`
  for CN**.
- Per the brief, the default was **not** changed and no tuning was attempted.
- It is not a 2D-closure blocker: SCHUR_SWEEP is validated for every distributed
  formulation, BE and CN.

## 10. Final 2D capability table (`feature_parity_final.csv`: 33 rows)

| category | feature | status |
|---|---|---|
| CONTROL | full-domain distributed | VALIDATED |
| | subdomain distributed | VALIDATED |
| | **multi-zone distributed** | **VALIDATED WITH DOCUMENTED LIMITATION** (state order 2.2–2.8 at control-box interfaces; no closed form) |
| | Neumann boundary | VALIDATED WITH DOCUMENTED LIMITATION (SCHUR_SWEEP only; CN terminal layer) |
| | Dirichlet boundary | VALIDATED WITH DOCUMENTED LIMITATION (reduced-solver memory; data compatibility) |
| TRACKING | full-domain | VALIDATED |
| | subdomain | VALIDATED |
| | **multi-zone** | **VALIDATED** |
| | non-colocated | VALIDATED |
| | time-dependent target | VALIDATED WITH DOCUMENTED LIMITATION (some target types preset-only) |
| DATA / BC | whole-domain source | VALIDATED |
| | single-region source | VALIDATED |
| | **multiple source regions** | **VALIDATED WITH DOCUMENTED LIMITATION** (one field on the union; source shapes preset-only) |
| | homogeneous Dirichlet state | VALIDATED |
| | inhomogeneous Dirichlet state | VALIDATED WITH DOCUMENTED LIMITATION (time-independent g, presets, distributed only) |
| | **natural state BC** | **VALIDATED** |
| TIME / REG | BE | VALIDATED |
| | CN | VALIDATED |
| | L2 control | VALIDATED |
| | Dirichlet H1-space | VALIDATED |
| | Dirichlet H1-time | VALIDATED WITH DOCUMENTED LIMITATION (no closed form) |
| | combined H1-space/H1-time | VALIDATED WITH DOCUMENTED LIMITATION (no closed form) |
| CONVECTION | constant wind | VALIDATED |
| | variable wind | VALIDATED WITH DOCUMENTED LIMITATION (2 named fields) |
| | exact transpose | VALIDATED |
| | Galerkin high-Pe behaviour | VALIDATED WITH DOCUMENTED LIMITATION |
| | LPS | VALIDATED WITH DOCUMENTED LIMITATION (P0 order loss; distributed only) |
| SOLVERS | SCHUR_SWEEP | VALIDATED |
| | ADDITIVE_GAMG | VALIDATED WITH DOCUMENTED LIMITATION (**BE only**; CN fails, pre-existing) |
| | SCHUR_MATCHING | VALIDATED WITH DOCUMENTED LIMITATION (low Pe) |
| | SCHUR_DIRICHLET_REDUCED | VALIDATED WITH DOCUMENTED LIMITATION (memory) |
| | SCHUR_BOUNDARY_REDUCED under convection | NOT VALIDATED (refused explicitly; not needed) |
| | MPI production path | VALIDATED |

## 11. Formulation → validated solver

| formulation | validated solver | notes |
|---|---|---|
| distributed control (full / subdomain / multi-zone; any tracking; any sources; Dirichlet or natural state BC), BE and CN | **SCHUR_SWEEP** (LU, MUMPS, or the iterative GMRES+GAMG spatial backend at low/moderate Pe; LU/ILU at high Pe) | ADDITIVE_GAMG is acceptable for **BE only** (8–50 its); MATCHING at low Pe |
| distributed + LPS | SCHUR_SWEEP (GAMG + LPS spatial) | T6 |
| Neumann boundary control + convection | **SCHUR_SWEEP** | boundary-reduced is refused under convection |
| Dirichlet boundary control + convection (L2 / H1-space / H1-time / combined) | **SCHUR_DIRICHLET_REDUCED** (default) | sweep / matching / boundary-reduced are refused |
| heat (no convection) | unchanged ex20 mapping | |

## 12. Blocking vs non-blocking; verdict

**Blocking: none.** All three remaining gaps are closed: multi-zone (§2), multiple
sources (§3) and natural BC (§4–§6). Each has gates, convergence (exact or
fine-reference), MPI and regression evidence.

**Non-blocking (documented, carried forward):**

- High-Pe Neumann/Dirichlet boundary-control campaigns.
- LPS combined with boundary control.
- A convection-capable Neumann boundary-reduced Schur.
- An alternative scalable Dirichlet solver (the reduced solver's memory grows as
  (Nt·|C|)²).
- The hypre AIR np = 4 crash.
- A Q2-order-preserving LPS.
- The CN J_reg constant endpoint reporting convention (Dirichlet).
- **New (T7F):** ADDITIVE_GAMG does not converge under CN. It is pre-existing, rejected
  explicitly, and SCHUR_SWEEP covers every case. A possible follow-up, outside the 2D
  closure: change the CN default to SCHUR_SWEEP, or warn.
- Multi-zone: the state's spatial order is reduced at control-box interfaces
  (a regularity effect).
- Per-region source amplitudes; non-constant source shapes from the CLI.
- Variable wind limited to the two named fields; no inflow Robin BC option.

**Every 2D physical/control capability intended for ex21 is now implemented and
validated** (some with documented, non-blocking limitations).

# 2D CONVECTION FEATURE SET: CLOSED

No 3D work was started.

## 13. Source changes (additive only) and files

`ex21/include/` (untracked directory):

- `ExactOcBenchmark.cpp`: case `natural-conv` (`NAT_CONV`). That is:
  - `IsNaturalCase`;
  - `NatR`, `NatPParts`, `NatYParts`;
  - the `ExactY`/`ExactP`, `Source` and `Target` branches;
  - the setup (NATURAL state BC);
  - the name in the case list and the error message.

  It refuses a wind field. Distributed full-domain control and tracking come from the
  common convective setup.
- `CommandLine.cpp`: `--source-box-add xmin xmax zmin zmax`, which pushes a box onto
  `ProblemConfig::source_regions`.
- `Verification.cpp`: `RunSpatialOperatorDump` also writes `<prefix>_M_source.coo`
  when source regions exist.

Nothing else changed. The new binary has sha 6278531a; the reference (T7D final) is
sha cf170d75.

**Files in `aux/investigation/`:**

- `T7F_FINAL_2D_FEATURE_CLOSURE.md` (this report);
- `feature_parity_final.csv`, `t7f_multizone.csv`, `t7f_multisource.csv`,
  `t7f_natural_bc.csv`, `t7f_solver.csv`;
- `EX21_FEATURE_PARITY_AUDIT.md`: a post-T7 closure note appended; the historical text
  is unchanged (verified by diff);
- `t7f/scripts/`: t7f_symbolic.py, t7f_exact_ladder.sh, t7f_ladder_table.py,
  t7f_gates.sh, t7f_gates_table.py, t7f_masks.{sh,py}, t7f_region_index.py,
  t7f_selfconv.{sh,py}, t7f_multisource.{sh,py}, t7f_mpi.{sh,py}, t7f_solver.{sh,py},
  t7f_tables.py, t7f_regress.sh, regress_suite.sh;
- `t7f/logs/`: exact/, gates/, selfconv/, multisource/, mpi/, solver/,
  solver_attribution/, symbolic.log, masks.log, region_index.log, regress.out, R1–R4
  logs;
- `t7f/results/`: gates.csv, selfconv_mz.csv, selfconv_natural.csv,
  multisource_runs.csv, mpi.csv, solver_attribution.csv.
