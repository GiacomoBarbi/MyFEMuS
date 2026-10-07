# T4 — unstabilised high-Péclet map of LP §5.2 (ex21, Q2, corrected tracking)

**Purpose.** To *separate*, not to fix:
- **A.** where standard Galerkin becomes spatially unreliable;
- **B.** where the GMRES+GAMG spatial solves stop being robust;
- **C.** how BE versus CN changes the solver behaviour.

No LPS was implemented and the KKT solver was not changed.

Problem: LP §5.2 with the T3b-corrected affine tracking, `T = 2`, Q2 with
`nx = 2^{l−1}` elements (the node set of LP's Q1 level l), and `τ = h_LP = 1/nx`
(Nt = 2nx), with production solver tolerances.

Files:
- data: `t4/results/*.csv` and snapshots, `t4/logs/campaign/*.log` (with outer
  residual histories), `t4/logs/regression/`;
- scripts: `t4/run_t4.py`, `t4/post_t4.py`;
- summaries: `t4_highpe_map.csv`, `t4_spatial_block_solver.csv`;
- figures: `figures/t4_y_profiles.png`, `figures/t4_y_profiles_beta.png`,
  `figures/t4_block_iterations.png`.

## 1. Method

**Two variants of every KKT solve:**
- **`gamg`**: production SCHUR_SWEEP. The spatial solves in the sweeps use
  GMRES(50), right-preconditioned, with GAMG, rtol 1e-10 and at most 500
  iterations.
- **`lu`**: the *same* KKT solve, but every sweep's spatial solve is exact,
  through `PETSC_OPTIONS -{sweep_right_,sweep_left_,sweep_ay_,fieldsplit_y_sweep_}ksp_type preonly … pc_type lu`.
  This does not change the code. It isolates the spatial preconditioner from
  the outer/Schur iteration, and it provides the discrete solution even where
  GAMG fails.

  Where both converge they agree to ≤ 1.3e-10 (max |Δy|).

**Péclet numbers.**
- `Pe_cell = max|w| h_e/(2ε)` with `max|w| = 2` and `h_e = 2/nx`.
- LP's patch Péclet number of the matching LP level, from LP's own definition
  (`t3/lp52_patch_peclet.py`), and whether LP's LPS would be active.

**Stop conditions.**
- The per-run timeout is 900 s. The runner now kills the whole process group;
  the first version left two timed-out solvers running as orphans, and they
  were stopped by PID.
- BE `gamg` at ε = 1/100 and 1/500 with nx ≥ 32 was **not run**. It is an
  established failure: θ = 1 blocks fail at all those Péclet numbers in §4,
  and the easier CN `gamg` runs timed out there.

**Discretisation indicators** (t = 1 snapshot plus all-time extrema):
- y range;
- wiggle rows: at least 3 consecutive sign alternations of `∂²y/∂x1²` along an
  x2 row;
- excess total variation: row TV minus row range, 0 for monotone rows;
- the x1 = 1 layer distances `d50` and `d10` (where y = ½ and y = 1/10 on
  x2 = 0);
- mesh-to-mesh `|Δy|` at shared nodes, everywhere and at distance > 0.25 from
  the corners.

## 2. Discretisation map (A), `lu` variant, β = 1e-2 (LP's Figs. 2–3 / Table 6–8 β)

CN shown; BE gives the same discrete picture to 2–3 digits.

| ε | nx (LP l) | Pe_cell | LP patch Pe / LPS | y(t=1) min | y min, all t | wiggle rows (t=1) | excess TV | layer d50 | |Δy| vs nx/2 (all / away from corners) |
|---|---|---|---|---|---|---|---|---|---|
| 1/20 | 8 (4) | 5 | 4.3 / ON | −1.5e-5 | −3.0e-2 | 0/17 | 1.7e-3 | 0.095 | – |
| 1/20 | 16 (5) | 2.5 | 2.3 / ON | −1.9e-7 | −2.1e-7 | 0/33 | 8e-8 | 0.090 | 2.6e-2 / 2.6e-2 |
| 1/20 | 32 (6) | 1.25 | 1.2 / ON | −2.1e-7 | −2.4e-7 | 0/65 | 6e-8 | 0.087 | 2.2e-2 / 2.0e-3 |
| 1/20 | 64 (7) | 0.63 | 0.62 / off | −2.4e-7 | −2.6e-7 | 0/129 | 8e-8 | 0.087 | 2.3e-2 / 1.8e-4 |
| 1/100 | 8 (4) | 25 | 21.6 / ON | **−3.5e-3** | **−8.1e-3** | **10/17** | **1.9e-2** | 0.084 | – |
| 1/100 | 16 (5) | 12.5 | 11.7 / ON | −3.8e-8 | −1.3e-5 | 0/33 | 6e-8 | 0.070 | 7.3e-2 / 7.1e-2 |
| 1/100 | 32 (6) | 6.25 | 6.1 / ON | −2.2e-8 | −4.1e-8 | 0/65 | 3e-9 | 0.069 | 4.1e-2 / 5.2e-3 |
| 1/100 | 64 (7) | 3.1 | 3.1 / ON | −2.3e-8 | −4.2e-8 | 0/129 | 3e-9 | 0.068 | 2.1e-2 / 8.7e-4 |
| 1/500 | 8 (4) | 125 | 108 / ON | **−4.8e-3** | **−1.4e-2** | **14/17** | **4.5e-2** | 0.081 | – |
| 1/500 | 16 (5) | 62.5 | 58 / ON | **−9.9e-4** | **−3.1e-3** | **18/33** | **1.1e-2** | 0.063 | 0.13 / 6.9e-2 |
| 1/500 | 32 (6) | 31 | 30 / ON | −3.2e-7 | −9.9e-6 | 0/65 | 8e-7 | 0.065 | 0.10 / 4.6e-2 |
| 1/500 | 64 (7) | 15.6 | 15.4 / ON | 0 | −4.0e-9 | 0/129 | ≈ 0 | 0.065 | 5.9e-2 / 7.8e-3 |

**Reading.**
- At β = 1e-2 the controlled state **follows the target**
  `y_d = e^{−10(1−x1)}` (figure `t4_y_profiles.png`). The x1 = 1 layer has a
  width (`d50` ≈ 0.065–0.095) set by the target, not by ε.
- Plain Galerkin therefore stays clean far beyond `Pe_cell = 1`: up to 12.5
  (ε = 1/100) and 31 (ε = 1/500).
- Visible alternating oscillations appear only on the two coarsest levels, at
  `Pe_cell ≥ 25`, and they are small: undershoot ≤ 4.8e-3 at t = 1 and
  ≤ 1.4e-2 over all t.
- The mesh-to-mesh differences away from the corners decrease under
  refinement. The corner differences at (1, ±1) stay near 2–6 %: these are the
  discontinuous boundary data, as in T3.

## 3. β dependence (A, continued): weaker control (CN, `lu`)

The Pe mechanism at β = 1e-2 is understood, so here the control is weakened:
β = 1e-1 is LP's largest β, and β = 1e2 is close to uncontrolled.

| β | ε | nx | Pe_cell | y(t=1) min | y min, all t | wiggle rows | excess TV | d50 / d10 |
|---|---|---|---|---|---|---|---|---|
| 1e-1 | 1/100 | 16 / 32 / 64 | 12.5 / 6.25 / 3.1 | ≈ −1.5e-5 | −6.1e-2 / −4.0e-2 / −9.8e-3 | 0 / 0 / 0 | ≈ 2e-6 | 0.062 / 0.17 |
| 1e-1 | 1/500 | 16 / 32 / 64 | 62.5 / 31 / 15.6 | −8.4e-3 / −4.9e-6 / −5.3e-6 | **−6.6e-2 / −5.0e-2 / −5.1e-2** | **26/33** / 1/65 / 0/129 | 9e-2 / 4e-7 / 4e-7 | 0.046–0.050 / 0.18 |
| 1e2 | 1/100 | 16 / 32 / 64 | 12.5 / 6.25 / 3.1 | −1.4e-3 / −8.2e-5 / −8.5e-5 | −7.8e-2 / −4.8e-2 / −1.3e-2 | **10/33** / 0 / 0 | 3.3e-3 / 2.8e-5 / 2.4e-5 | 0.057 / 0.14 |
| 1e2 | 1/500 | 16 / 32 / 64 | 62.5 / 31 / 15.6 | **−6.3e-2 / −3.1e-3** / −7.6e-9 | **−0.10 / −7.2e-2 / −6.0e-2** | **18/33 / 23/65** / 0/129 | 0.38 / 6.2e-3 / 6.7e-5 | **0.026–0.034** / 0.06 |

**Reading.**
- With weak control the x1 = 1 layer is ε-governed and sharper (`d50` 0.026 at
  ε = 1/500).
- **Alternating Galerkin oscillations** appear at t = 1:
  - ε = 1/500: at `Pe_cell` = 62.5 and 31;
  - ε = 1/100: at 12.5.
- More importantly, an **all-time undershoot of 5–10 %** at ε = 1/500 does
  **not** decrease up to nx = 64 (Pe 15.6), while at t = 1 the same runs are
  clean. So it lives at other time levels: most plausibly the early transient
  from LP's discontinuous initial/boundary data at x1 = 1, where the layer
  starts arbitrarily thin. That location is inferred from "not at t = 1"; it
  was not localised further.
- At ε = 1/100 the all-time undershoot decreases with refinement
  (−7.8e-2 → −1.3e-2).

## 4. Isolated spatial-block experiment (B, C): `--exact-oc-block-test`

**Blocks.** The three spatial blocks the sweeps solve:
- `E = M/dt + θL` (y-split);
- `right = E + √β N`;
- `left = (E + √β M_ω)ᵀ`.

**Solver.** The **identical** solver configuration (GMRES(50), right, GAMG,
rtol 1e-10, 500 its), on 3 seeded random right-hand sides and 1 smooth one, for
θ = 1 (BE) and θ = ½ (CN diagonal). `√β = 10` here.

Columns: max iterations over the 4 right-hand sides; 500 means every solve hit
the cap (KSP reason −3).

**τ = h_LP rule** (advective ratio `θ|w|dt/h_e` = θ):

| ε | nx | Pe_cell | E, θ=1 | right/left, θ=1 | E, θ=½ | right/left, θ=½ |
|---|---|---|---|---|---|---|
| 1/20 | 8 | 5 | **500** | 115 / 111 | 209 | 39 / 39 |
| 1/20 | 16 | 2.5 | **500** | 278 / 266 | 129 | 45 / 44 |
| 1/20 | 32 | 1.25 | 24 | 24 / 24 | 23 | 23 / 23 |
| 1/20 | 64 | 0.63 | 14 | 14 / 14 | 14 | 14 / 14 |
| 1/100 | 8 | 25 | **500** | **500 / 500** | **500** | 194 / 194 |
| 1/100 | 16, 32, 64 | 12.5, 6.25, 3.1 | **500** | **500** | **500** | **500** |
| 1/500 | 8 | 125 | 349 | **500** | **500** | 242 / 244 |
| 1/500 | 16, 32, 64 | 62.5, 31, 15.6 | **500** | **500** | **500** | **500** |

**dt sweep at nx = 32.**

| ε | Pe_cell | advective ratio θ\|w\|dt/h | max its (E / right / left) |
|---|---|---|---|
| 1/20 | 1.25 | 0.06 … 8 | 12–29, at every ratio |
| 1/100 | 6.25 | ≥ 0.5 | **500 / 500 / 500** |
| 1/100 | 6.25 | 0.25 | 172 / 135 / 134 |
| 1/100 | 6.25 | 0.125 | 30 / 29 / 28 |
| 1/100 | 6.25 | 0.0625 | 15 / 14 / 14 |

**Key identities and findings.**
1. **Same block for equal θ·dt.** `M/dt + θL = θ (M/(θ dt) + L)`: the blocks
   for the same θ·dt are the same matrix up to scale. The experiment shows it
   exactly. BE at dt = 1/256 and CN at dt = 1/128 give **identical**
   iterations (171/169/172/148). BE at 1/512 and CN at 1/256 also match
   (30/30/30/25).

   **The BE-versus-CN difference of T3 is therefore entirely the effective
   time step θ·dt of the diagonal blocks.** Under the τ = h rule, CN's blocks
   have half the advective ratio of BE's. It is not a property of the
   optimal-control system.
2. **GAMG robustness is governed jointly by `Pe_cell` and the advective ratio
   `θ|w|dt/h`**, i.e. by how mass-dominated the block is:
   - for `Pe_cell ≲ 1.25` it is robust at any tested ratio;
   - for `Pe_cell` 2.5–5 it fails at ratio 1 (BE) but converges at ratio ½ (CN);
   - for `Pe_cell ≥ 3` (ε ≤ 1/100) it fails at ratio ≥ ½ and needs ratio
     ≲ 0.125 to be efficient.
3. **The matching blocks are easier than E.** The extra `√β M` term
   (`√β = 10`) adds mass: at ε = 1/20, θ = 1, the right/left blocks converge
   (111–278 its) where E fails.

## 5. Solver map of the KKT runs (B)

| ε | nx | Pe_cell | CN `gamg` | BE `gamg` | CN / BE `lu` (outer its; Schur its per outer) |
|---|---|---|---|---|---|
| 1/20 | 8 | 5 | converged: 2 outer; spatial mean/max 94/206 | **timeout** (stagnation, as in T3) | 2; 8 / 2; 8 |
| 1/20 | 16 | 2.5 | converged: 2 outer; spatial 69/129 | **timeout** (T3 with dt = 0.125: reason −11) | 2; 8 / 2; 8 |
| 1/20 | 32 | 1.25 | converged: 2; spatial 20/24 | converged: 2; spatial 22/25 | 2; 8 / 2; 8 |
| 1/20 | 64 | 0.63 | converged: 1; spatial 13/13 | converged: 1; spatial 14/14 | 1; 8 / 1; 8 |
| 1/100 | 8 | 25 | **−11**: 560 failed spatial solves | **−11**: 3040 | 2; 11 / 2; 8.5 |
| 1/100 | 16 | 12.5 | **−11**: 3104 | **−11**: 6080 | 2; 9.5 / 2; 8 |
| 1/100 | 32, 64 | 6.25, 3.1 | **timeout** | not run (stop condition) | 2; 8 / 2; 8 |
| 1/500 | 8 | 125 | **−11**: 560 | **−11**: 1040 | 2; 13.5 / 2; 8.5 |
| 1/500 | 16 | 62.5 | **−11**: 3104 | **−11**: 3104 | 2; 14 / 2; 8.5 |
| 1/500 | 32, 64 | 31, 15.6 | **timeout** | not run | 2; 11 / 2; 8–8.5 |

With exact spatial solves, **every** run in the map converges in 1–2 outer
FGMRES iterations, with 5–14 Schur GMRES iterations per outer iteration and no
Schur cap hit. This holds for ε down to 1/500, `Pe_cell` up to 125, both
schemes, and β from 1e-2 to 1e2.

The outer/Schur strategy (matching Schur with exact sweeps) is therefore
**robust at high Péclet**. The only failing component is the spatial
GMRES+GAMG preconditioner. Failure modes are:
- KSP reason −11 (`DIVERGED_PC_FAILED`) with thousands of spatial solves at
  the 500 cap;
- outer stagnation or timeout;
- never a silent wrong answer.

## 6. Conclusions

1. **Onset of visible Galerkin oscillations.**
   - At LP's β = 1e-2: only at `Pe_cell ≥ 25` (LP levels l = 4 at ε = 1/100;
     l = 4, 5 at ε = 1/500), with amplitude ≤ 0.5 % at t = 1 and ≤ 1.4 % over
     all t.
   - With weak control (β ≥ 1e-1): from `Pe_cell` 12.5 (ε = 1/100) and 31–62
     (ε = 1/500) at t = 1. At ε = 1/500 there is a persistent **5–10 %
     all-time undershoot** that does not vanish by `Pe_cell` 15.6.
   - LP's LPS would be ON on every one of these levels: patch Pe 3–108.
2. **Spatial solver failure.** It sets in much earlier than the discretisation
   failure. GMRES+GAMG fails whenever `Pe_cell ≳ 2.5` *and* the block is not
   mass-dominated (`θ|w|dt/h ≳ 0.5`). Under LP's τ = h rule this means:
   - BE fails from `Pe_cell` 2.5;
   - CN fails from `Pe_cell` ≈ 3;
   - **every** ε = 1/100 and 1/500 level up to nx = 64 fails.
3. **BE versus CN** is fully explained by the diagonal block depending only on
   θ·dt (§4.1).
4. **LPS.** It is scientifically justified **as a discretisation extension**
   for weak control at ε = 1/500, and for coarse meshes at `Pe_cell ≥ 25`.
   - **It is not the binding constraint for LP's own β range on moderate
     meshes:** at β = 1e-2, unstabilised Q2 is clean from `Pe_cell ≤ 31`.
   - **It will not by itself fix the solver.** The blocks remain
     convection-dominated. The LPS term is symmetric PSD and may help GAMG
     somewhat, but that is untested.
5. **A separate spatial solver/preconditioner improvement is required**,
   independent of LPS. Without it, no ε ≤ 1/100 run of LP's Tables 6–8 is
   possible with the production path; with exact sweeps all of them converge.

## 7. Recommended next step

1. **Screen convection-robust spatial preconditioners.** Use the existing
   `--exact-oc-block-test` harness and the existing sweep prefixes via
   `PETSC_OPTIONS`, with no code change. Candidates:
   - GAMG options for nonsymmetric problems (smoothers, smoothing steps,
     graph symmetrisation);
   - hypre BoomerAMG with advection-oriented (AIR-type) settings, if the PETSc
     build has hypre;
   - ILU / ASM;
   - direct sparse solves for these modest 2D blocks (LU worked here at
     nx ≤ 64).

   The acceptance target is the τ = h_LP blocks at ε = 1/100 and 1/500,
   θ = 1 and ½.
2. **Then decide on LPS** with a working solver, by comparing unstabilised and
   stabilised Q2 against LP at weak control / ε = 1/500.

## 8. Cost

- Block experiment: 0.2–37 s per run.
- `lu` KKT runs: ≤ 56 s.
- Converged `gamg` runs: ≤ 107 s.
- Failed `gamg` runs: 15–308 s.
- Timeouts: 900 s each (6 runs).
- No run exceeded 15 min. Peak memory ≈ 4.5 GB at nx = 64.

## 9. Regression on the final binary (500a2b2e)

- ex20 goldens 61/61;
- constant-wind dumps, T1/T2 runs and gates byte-identical;
- Math2025 Ex3/Ex4/Ex5 byte-identical;
- EF 9.5 numerics identical (label-only difference, pre-T3);
- the T3b binary (825465c0) and the final binary give identical `EX21_`
  output on lift-track, lp52 and variable-wind runs.
