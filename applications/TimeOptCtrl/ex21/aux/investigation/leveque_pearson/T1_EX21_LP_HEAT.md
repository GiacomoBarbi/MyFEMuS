# T1 — Leveque–Pearson §5.1 heat benchmark through the ex21 production path

**Result: PASS.** The analytic heat-control benchmark of Leveque & Pearson
(NLAA 2022, §5.1), already reproduced by ex20 with a Q1 harness, also passes
through **ex21's production path**. That path is the Q2 `SolveOptimalControl`
with SCHUR_SWEEP and the ex21 binary.

Measured results:
- **CN, τ = h:** O(h²); adjoint order **2.00** on `t ≤ T/2`; state **2.12**.
- **BE, τ = h²:** O(h²); adjoint **2.02** on `t ≤ T/2`; state **2.04**.
- **Pure CN temporal errors** equal ex20's to **0.1–3.5 %** at `Nt = 8…32`. The
  two codes share the time discretisation, so this is the expected outcome.
- **`v = 0` with the convection (transposed-adjoint) path switched on** gives
  errors identical to the path switched off (difference 0).

This is a regression and anchoring test, not the main result (that is T2).

Code: the `lp-heat` case of `include/ExactOcBenchmark.cpp` (`--exact-oc lp-heat`).

Data:
- `results/t1_*.csv`
- `logs/campaign/t1_*.log`
- `logs/t2_symbolic_check.log` (the T1 part)

## 1. Data — taken from the ex20 benchmark, not reinvented

Source: `ex20/aux/EX20_LEVEQUE_PEARSON_2022_BENCHMARK.md`, §4 (analytic solution)
and §5 (shift).

- `Ω = (−1,1)²`, `T = 2`, `f = 0`, `ν = 1`, `v = 0`, `η = α_track = 1`.
- Control and tracking are full-domain, and **`α_u = β`** exactly (ex20 §5).
- With `s = cos(πx/2)cos(πz/2)`:

      y = 1 + [2/(π²β) e^T − 2/((2+π²)β) e^t] s,     p_LP = (e^T − e^t) s,     u = p_LP/β,
      ŷ = 1 + [(2/(π²β) + π²/2) e^T + (1 − 2/((2+π²)β) − π²/2) e^t] s.

- **The exact constant shift `w = y − 1`** (ex20 §5) is kept. `s` vanishes on
  `∂Ω`, so `g = 1`, and the shifted problem is the homogeneous-Dirichlet
  distributed problem with target `ŵ = ŷ − 1` and `w(0) = y(0) − 1`. Errors of
  `y` equal errors of `w`; relative state errors are quoted with respect to the
  **unshifted** `y = 1 + w`, as ex20 did.
- **ex21 sign:** `p_ex21 = −p_LP` (T2 §1).

Symbolic check (`t2_symbolic_check.py`, T1 part). LP's own optimality system,
the ex21-convention system with the shift, `u_ex21 = u_LP`, and the zero traces
of `w` and `ŵ` on all four sides all give **0**.

## 2. Configuration

- Production Q2, `nx = 2^l` elements per direction (`h = 2^{1−l}` as in LP; Q2
  has `(2nx+1)²` nodes, unlike LP's Q1).
- SCHUR_SWEEP; tolerances tightened in a driver-local policy (outer 1e-12,
  inner 1e-10, spatial 1e-12); no default changed.
- Paper rules: CN `τ = h` (`Nt = 2^l`, `l = 3…6`) and BE `τ = h²`
  (`Nt = 2^{2l−1}`, `l = 3…5`; BE `l = 6` would have `Nt = 2048` and was not
  needed).
- Main value `β = 1e-3`. Range check at `β = 1e-2` and `1e-4`.

**Norms** (as in T2):
- true `L²(0,T;L²)` relative error by element quadrature, state at `t_n`;
  adjoint and control at `t_{n−½}` (CN) or `t_n` (BE);
- the fixed window `t ≤ T/2`;
- the terminal slot (absolute);
- **ex20's own metric**, relative nodal `l²` over the interior space-time nodes
  (ex20 §12, `y_l2rel`, `p_l2rel`), for a direct comparison.

## 3. Paper rules, β = 1e-3

**CN, τ = h**

| l | Nt | y | ord | p | ord | p, t ≤ T/2 | ord | u | ord | abs. err. p_N | ord |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 8 | 1.231e-03 | – | 6.516e-03 | – | 2.122e-03 | – | 6.516e-03 | – | 6.52e-02 | – |
| 4 | 16 | 3.698e-04 | 1.74 | 1.895e-03 | 1.78 | 5.185e-04 | 2.03 | 1.895e-03 | 1.78 | 2.70e-02 | 1.27 |
| 5 | 32 | 1.000e-04 | 1.89 | 5.864e-04 | 1.69 | 1.289e-04 | 2.01 | 5.864e-04 | 1.69 | 1.23e-02 | 1.13 |
| 6 | 64 | 2.305e-05 | **2.12** | 1.709e-04 | 1.78 | 3.216e-05 | **2.00** | 1.709e-04 | 1.78 | 4.89e-03 | 1.34 |

**BE, τ = h²**

| l | Nt | y | ord | p | ord | p, t ≤ T/2 | ord | u | ord | abs. err. p_N | ord |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 32 | 1.001e-03 | – | 6.524e-03 | – | 1.643e-03 | – | 6.524e-03 | – | 1.26e-01 | – |
| 4 | 128 | 2.505e-04 | 2.00 | 2.215e-03 | 1.56 | 3.943e-04 | 2.06 | 2.215e-03 | 1.56 | 7.56e-02 | 0.74 |
| 5 | 512 | 6.099e-05 | **2.04** | 6.200e-04 | 1.84 | 9.752e-05 | **2.02** | 6.200e-04 | 1.84 | 2.58e-02 | 1.55 |

**Reading.**
- State and adjoint converge at **O(h²)** under both coupled rules, which is the
  order LP report for these rules.
- On the whole interval the adjoint order is lower: 1.7–1.8 (CN) and 1.56 → 1.84
  (BE). The `t ≤ T/2` window shows clean second order, and the terminal slot
  converges at about first order (CN 1.13–1.34). So the deficit is the DTO
  terminal adjoint layer. ex20 documented this layer for this very benchmark
  (ex20 §19, §27: CN adjoint terminal slot at order ≈ 1, interior slots at 2).
- It is larger here than in T2 because LP's adjoint has a large slope at `T`
  (`|p_t(T)| = e^T ‖s‖`).
- This is the exact adjoint of the state recursion (`λ_{N+1} = 0`), not LP's
  imposed `p(t_f) = 0` (T0 §5).

## 4. Consistency with ex20

**Pure temporal (CN, β = 1e-3).** ex21 Q2 at `l = 5`, spatially negligible, is
compared with ex20 Q1 at `l = 7` (ex20 §22). Both use ex20's nodal metric.

| Nt | ex21 y_l2rel | ex20 y_l2rel | ratio | ex21 p_l2rel | ex20 p_l2rel | ratio |
|---|---|---|---|---|---|---|
| 8 | 1.1314e-03 | 1.1320e-03 | **0.999** | 6.4928e-03 | 6.4620e-03 | **1.005** |
| 16 | 3.6480e-04 | 3.6500e-04 | **0.999** | 1.8941e-03 | 1.8680e-03 | **1.014** |
| 32 | 9.9761e-05 | 9.9920e-05 | **0.998** | 5.8628e-04 | 5.6670e-04 | **1.035** |
| 64 | 2.3033e-05 | 2.3240e-05 | 0.991 | 1.7085e-04 | 1.5990e-04 | 1.068 |

- ex20 and ex21 use **the same DTO Crank–Nicolson time discretisation**, so
  their temporal errors should agree once both spatial errors are small. They
  do: to 0.1 % for the state and 0.5–3.5 % for the adjoint at `Nt ≤ 32`.
- The gap grows to 6.8 % at `Nt = 64`. There the spatial error of at least one
  of the two codes (ex20 Q1 `l = 7`, or ex21 Q2 `l = 5`) is no longer
  negligible next to the temporal adjoint error. Which one was not determined.
- The ex21 temporal orders (adjoint 1.78 / 1.69 / 1.78 whole-interval,
  2.00 / 2.00 / 1.96 on `t ≤ T/2`) match ex20 §22 (1.79 / 1.72 / 1.82).

**Paper rules, ex20's metric.** ex21 Q2 is compared with ex20 Q1 (ex20 §15–17,
β = 1e-3).

| l | CN y: ex21 / ex20 | CN p: ex21 / ex20 | BE y: ex21 / ex20 | BE p: ex21 / ex20 |
|---|---|---|---|---|
| 3 | 1.133e-3 / 1.329e-3 (0.85) | 6.47e-3 / 1.082e-2 (0.60) | 8.68e-4 / 1.084e-3 (0.80) | 6.53e-3 / 1.732e-2 (0.38) |
| 4 | 3.649e-4 / 4.049e-4 (0.90) | 1.89e-3 / 2.857e-3 (0.66) | 2.43e-4 / 2.881e-4 (0.84) | 2.22e-3 / 4.571e-3 (0.48) |
| 5 | 9.976e-5 / 1.088e-4 (0.92) | 5.86e-4 / 7.915e-4 (0.74) | 6.05e-5 / 7.171e-5 (0.84) | 6.20e-4 / 1.175e-3 (0.53) |
| 6 | 2.303e-5 / 2.522e-5 (0.91) | 1.71e-4 / 2.167e-4 (0.79) | – | – |

- ex21's errors are smaller at every level, and the ratio is bounded (0.38–0.92).
- This is the expected effect of Q2: ex20's errors are the same temporal error
  **plus** a Q1 `O(h²)` spatial part, while ex21's Q2 spatial part is
  `O(h³)` and small.
- The CN adjoint ratio rises toward 1 (0.60 → 0.79) because ex21's error is
  temporal-dominated, and ex20's spatial part shrinks relative to it.

**LP Tables 1–3** are used only as context (DTO vs OTD, Q2 vs Q1, a different
solver and a pointwise metric). LP observe O(h²) convergence of both state and
adjoint for CN with `τ = h` and BE with `τ = h²`, and so does ex21 (adjoint on
`t ≤ T/2`; the terminal-layer caveat applies to the whole interval). No
numerical agreement with LP's digits is claimed or expected.

## 5. β range, v = 0 path, solver

CN, `l = 4`, `Nt = 16`:

| run | α_u = β | `--convection` | y | p | u | outer its | relative KKT residual |
|---|---|---|---|---|---|---|---|
| β = 1e-2 | 1e-2 | off | 1.500e-03 | 3.948e-03 | 3.948e-03 | 2 | ≤ 1.8e-13 |
| β = 1e-3 | 1e-3 | off | 3.698e-04 | 1.895e-03 | 1.895e-03 | 2 | ≤ 1.8e-13 |
| β = 1e-4 | 1e-4 | off | 7.535e-05 | 1.437e-03 | 1.437e-03 | 2 | ≤ 1.8e-13 |
| β = 1e-3, v = 0 | 1e-3 | **on** | 3.698e-04 | 1.895e-03 | 1.895e-03 | 2 | ≤ 1.8e-13 |

- **No regression at v = 0.** With `--convection on` and `v = 0`, every adjoint
  product goes through the transposed (ex21) path. Its errors equal the
  `--convection off` (ex20) path to all printed digits (relative change **0**).
- **Solver over all T1 runs:**
  - 2 outer FGMRES iterations everywhere, every solve converged;
  - relative KKT block residuals ≤ 4.2e-13;
  - the absolute `‖Kx − b‖` reaches 4e-10, but the heat data are `O(1/β)`;
  - independent forward LU march ≤ 3.6e-14;
  - `‖α_u u + ηp‖/‖α_u u‖` ≤ 6.7e-17.
- **Cost:** the largest runs are CN `l = 6` (219 s, 2.3 GB) and BE `l = 5`
  (222 s, 4.5 GB).

## 6. Acceptance (brief §5)

| criterion | status |
|---|---|
| ex21 production path converges to the LP analytic heat solution | **PASS** (O(h²) under both paper rules; adjoint 2.00 / 2.02 on `t ≤ T/2`) |
| compatible with the validated ex20 benchmark | **PASS** (temporal errors equal to 0.1–3.5 % at `Nt ≤ 32`; paper-rule errors bounded below ex20's by the Q1 → Q2 spatial gain) |
| no regression at v = 0 | **PASS** (convection-on and convection-off identical; ex20 goldens 61/61 identical on this binary — see `LEVEQUE_PEARSON_SOURCE_AUDIT.md` post-audit section) |
