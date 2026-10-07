# ErrorFeedback2026 Examples 9.5 and 9.6 — source audit and ex21 feasibility

**Source.** `~/Downloads/ErrorFeedback2026.pdf`, manuscript dated **3 October
2026**, Section 9, Example 9.5 (pp. 35–37, Figs. 28–31, Table 9) and Example 9.6
(pp. 37–39, Figs. 32–35, Table 10). Read from the PDF text layer and from the
vector paths (`pdftocairo -svg`); no raster reading is used as data.
Tags: **[A]** stated · **[B]** inferred / recoverable · **[C]** missing.

Scripts and raw data: `errorfeedback_95_96/scripts/extract_figs.py`,
`results/errorfeedback_fig{29,30,33,34}_raw.csv`,
`results/errorfeedback_95_96_extraction.json`,
`results/errorfeedback_95_96_geometry.json`.

---

## 1. Specification

| datum | 9.5 (colocated) | 9.6 (non-colocated) | tag |
|---|---|---|---|
| PDE | `z_t = A z + B_in u + B_d d`, `A = Δ − V·∇` | same plant ("we retain … plant … and numerical parameters") | [A] |
| `A*` | `Δ + V·∇` (continuous; not used by ex21, which uses `Eᵀ`) | idem | [A] |
| diffusion | normalised to 1 | idem | [A] |
| velocity | `V = (v1, v2) = (2, 1)`, constant | idem | [A] |
| domain | `Ω = (0, π) × (0, 1)` | idem | [A] |
| BC | homogeneous Dirichlet on `∂Ω` | idem | [A] |
| **IC** | **not stated** ("h … contains the contributions of the reference, disturbance, and plant initial condition") | idem | **[C]** (§4) |
| `B_in` | extension by zero `L²(Ω_B) → L²(Ω)` (infinite-dimensional input) | idem | [A] |
| `C` | restriction `L²(Ω) → L²(Ω_C)`, `Cz = z|_{Ω_C}` | idem | [A] |
| `B_d` | extension by zero `L²(Ω_d) → L²(Ω)` | idem | [A] |
| arrangement | `Ω_B = Ω_C`, `Ω_d` disjoint from `Ω_C` | `Ω_B ∩ Ω_C = ∅` | [A] |
| **region coordinates** | figure only (Fig. 28) | figure only (Fig. 32) | **[C]** (§3) |
| reference | `r = x exp(−(y−½)²/σ_y²) sin(2t)` on `Ω_C` (9.25) | idem | [A] |
| disturbance | `d = x exp(−(y−½)²/σ_y²) cos(3t)` on `Ω_d` (9.26) — **no switch-on**: `d(·,0) ≠ 0` | idem | [A] |
| `σ_y` | 0.15 | idem | [A] |
| controller `α`, `β`, order | **not given** ("numerical parameters used in the preceding example" — never valued) | idem | [C] (documentation only) |
| horizon `T` | figure axes end at `t = 20` (Figs. 29–30) | idem (Figs. 33–34) | [B] |
| tail window | "`E_j := lim ‖e_j‖_{L²(Ω_C)}`" (9.5); "`t_sup` is chosen beyond the initial transient" (9.6) — no value | | [C] — but immaterial, §5 |
| metric | `E_j = lim sup_t ‖e_j(t)‖_{L²(Ω_C)}`, `e_j` the iteration errors (9.27); by the Section-5 identity the plant error of a `j`-stage controller equals `e_j` | idem | [A] |
| discretisation | not stated | idem | [C] |

**Published values.**

| j | Table 9 (9.5) `E_j` | ratio | Table 10 (9.6) `E_j` | ratio |
|---|---|---|---|---|
| 1 | 1.25e-2 | – | 6.355e-2 | – |
| 2 | 3.93e-3 | 0.31 | 5.496e-2 | 0.8649 |
| 3 | 1.70e-3 | 0.43 | 5.338e-2 | 0.9713 |
| 4 | 8.76e-4 | 0.52 | 5.264e-2 | 0.9861 |
| 5 | 5.15e-4 | 0.59 | 5.220e-2 | 0.9917 |

**Figure axes (text layer).** Fig. 29: `t ∈ [2, 20]` ticks, `y ∈ [0, 1.4e-3]`;
Fig. 30: `t ∈ [5, 20]`, `y ∈ [0, 4e-3]`; Fig. 31: `t ∈ [0, 2]`, `y ∈ [0, 0.18]`;
Fig. 33: `[2, 20]`, `[0, 0.06]`; Fig. 34: `[5, 20]`, `[0, 0.07]`; Fig. 35:
`[0, 2]`, `[0, 0.7]`. Curves in Figs. 30/34 are plotted from `t = 5`, in
Figs. 29/33 from `t = 0.6`, in Figs. 31/35 from `t = 0.05`.

**Draft inconsistencies.** 9.5 text: "five iteration levels are displayed, so
the iteration errors are indexed as e1, e2, e3, e4", while Table 9 lists
`j = 1..5`; Fig. 29 caption compares `e` with `e₄`.

## 2. What the figures say about the tables (vector extraction)

Calibration residuals ≤ 2e-4 in t, ≤ 1.4e-6 in value.

* **9.5, Fig. 30 vs Table 9.** The four plotted curves (legend `e₁…e₄`) have
  tail sups 3.930e-3, 1.694e-3, 8.715e-4, 5.104e-4 = Table 9's `j = 2…5`
  (3.93e-3, 1.70e-3, 8.76e-4, 5.15e-4) to 0.0, −0.4, −0.5, −0.9 %. The figure
  legend is therefore **shifted by one level** relative to the table
  (consistent with the "indexed as e1..e4" remark). The plotted curves
  reproduce Table 9 to < 1 %.
* **9.6, Fig. 34 vs Table 10.** The five plotted curves have tail sups
  6.682e-2, 5.697e-2, 5.532e-2, 5.461e-2, 5.417e-2 — **3.6–5.1 % above**
  Table 10 at every level (no index shift fits). The table is not reproduced by
  its own figure to better than ≈ 4 %; the cause (different window, norm or
  sampling) is not determinable from the source.

## 3. Geometry (B2)

Both region figures are vector drawings: the frame and the region boxes are
exact rectangles in the PDF. But:

1. **The tick labels are vector glyphs, not text** — `pdftotext -bbox` returns
   no fraction labels for Figs. 28/32, so they cannot be read as data.
2. **The drawings are not to scale.** Fig. 32: the six bottom ticks (glyphs
   reading π/8 … 7π/8) sit at page x = 221.9, 259.3, 292.1, 329.8, 365.0,
   400.9, i.e. at positions 1, 2, **2.88, 3.89, 4.83, 5.79** in units of the
   first spacing instead of 1, 2, 3, 4, 6, 7. Fig. 28: ticks for 3π/8, 4π/8,
   6π/8, 7π/8 at 3, 4, **5.40, 6.45** instead of 3, 4, 6, 7. The `Ω_d` boxes do
   not even sit on their own ticks (Fig. 32: box 363.8–402.6 vs ticks
   365.0–400.9). Reading the frame as a linear `[0, π]` axis gives
   `Ω_B = Ω_C = [0.32, 0.47]π`, `Ω_d = [0.70, 0.86]π` (Fig. 28) — inconsistent
   with any reading of the labels.

**Verdict: exact region coordinates are NOT machine-recoverable → [C].** The
glyph labels, read by eye at high resolution, suggest
`Ω_B = Ω_C = [3π/8, π/2] × [1/3, 2/3]`, `Ω_d = [3π/4, 7π/8] × [1/3, 2/3]`
(9.5) and `Ω_B = [π/8, π/4]`, `Ω_C = [3π/8, π/2]`, `Ω_d = [3π/4, 7π/8]`
(× `[1/3, 2/3]`) (9.6) — recorded as an **unverified candidate** only (an
author confirmation would turn it into [A]).

## 4. Initial condition (B3)

| question | answer |
|---|---|
| IC stated? | **No** — not in 9.5, 9.6, the Section 9 preamble, or by reference. |
| IC inferable? | **No** — no earlier example shares this plant (9.1–9.4 are 1D). |
| IC numerically identifiable? | **No.** `e(0) = r(0) − C z₀ = −z₀|_{Ω_C}` would bound `‖z₀‖_{L²(Ω_C)}`, but the plotted curves start at `t = 0.05` (Figs. 31, 35), and the REFC acts from `t = 0` with unknown `α, β`, so the early transient cannot be inverted for `z₀`. |
| What the figures do say | At `t = 0.05`: `‖e‖ = 0.165` (9.5) and `0.566` (9.6), against `‖r(0.05)‖_{L²(Ω_C)} ≈ 0.04`. In 9.6 the disturbance region lies downstream of `Ω_C` and the control region upstream at distance ≈ 0.39; a zero IC would require the controller itself to create an O(0.5) error on `Ω_C` within 0.05 time units (diffusion factor `e^{−0.39²/0.2} ≈ 0.47` does not forbid it for a very large control). **A zero IC is therefore implausible, not excluded**; the IC is most likely nonzero and O(1) on `Ω_C`. Its shape is unknown. |
| Does the IC matter for the published metric? | Probably not for the tail (`t ≥ 5`): the plant is exponentially stable and the tail sups in Figs. 30/34 are window-independent from `t_sup = 2`. It matters for any comparison of curves before `t ≈ 2`. |

**Status: genuinely missing ([C]); a zero IC is implausible given Fig. 35, but not excluded.**

## 5. Feasibility (B4)

| question | 9.5 | 9.6 |
|---|---|---|
| reproduce Table 9/10 without tuning? | **No** — the tables are REFC tail levels at unpublished `α, β`; ex21 has no source `ε`. Only a transparent `α_u` match to `E_5` (ex20 methodology) is possible. | same |
| comparable ex21 quantity | tail `sup_t ‖y − r‖_{L²(Ω_C)}` of the AAO optimum on the same plant | same |
| tune `α_u` to a source level? | yes, one-parameter match to `E_5 = 5.15e-4` | yes, to `5.22e-2` — **but** 9.6 is non-colocated, so its tail may be a reachability floor (cf. Mathematics 2025 Ex. 3); then no `α_u` reproduces it and the comparison becomes a floor-vs-floor check |
| metric | `L²(Ω_C)` norm, sup over a tail in time | same |
| tail window known? | no, but immaterial (sups window-independent for `t_sup ∈ [2, 15]`) | same |
| `T` known? | [B] 20 (axes) | same |
| geometry | **[C]** | **[C]** |
| IC | [C] (tail-irrelevant, probably) | [C] |

**Classification.**

* **9.5: C now; B if the geometry is supplied.** Assumptions needed today:
  (1) region coordinates from the glyph labels; (2) an IC (tail-only comparison
  makes it inessential, to be verified by IC sensitivity); (3) `T = 20`;
  (4) `α_u` matched to `E_5` (not a source value). The geometry is an
  **essential plant datum** and is unconstrained by the vector data, so the
  benchmark would rest on a visual reading.
* **9.6: C now; B with the same assumptions** — plus the internal Table 10 vs
  Fig. 34 discrepancy (≈ 4 %), which limits any quantitative claim to ≈ 5 %.

Per the brief ("do not implement if the answer still depends on multiple
unconstrained guesses"): **not implemented.**

## 6. Implementation readiness (B5)

| item | status |
|---|---|
| 2D Q2 mesh on `(0, π) × (0, 1)` (`--domain 0 3.14159… 0 1`) | **ready now** |
| `ν = 1`, `--convection on --vx 2 --vy 1`, homogeneous Dirichlet | **ready now** (validated 2D convection path) |
| distributed control on a box, distributed `L²` tracking on a box, disturbance on a box (`source_regions`) | **ready now** |
| Gaussian `r(x,y,t)`, `d(x,y,t)` (data overrides) | ready with a small preset (≈ Mathematics 2025 preset pattern) |
| region coordinates | **blocked** ([C]; candidate from glyph labels only) |
| IC | **blocked** ([C]); tail-only comparison with IC sensitivity would make it inessential |
| `α_u` | ready with assumption: transparent match to `E_5` (9.5); floor check (9.6) |
| solver | SCHUR_SWEEP (2D CN/BE validated; GMRES spatial sweeps); ADDITIVE_GAMG fails at small `α_u` |

**Ready now:** none. **Ready with assumptions:** both, if the region
coordinates are accepted from the glyph labels. **Blocked:** exact geometry
(and, for transient comparisons, the IC). Asking the authors for the region
coordinates and the IC would move 9.5 to B (and close to A for the tail).

Exact configuration if unblocked (9.5):

    TimeOptCtrl_ex21 --distributed-solve --nx 64 --nz 30 --domain 0 π 0 1
      --state-bc dirichlet --convection on --vx 2 --vy 1 --nu 1
      --control-box 3π/8 π/2 1/3 2/3 --tracking-box 3π/8 π/2 1/3 2/3
      [source region 3π/4 7π/8 1/3 2/3 + Gaussian data via a preset]
      --T 20 --time-scheme cn --pc-schur-sweep --alpha-u <matched to E_5>

(`nx` must place the region edges on element faces: multiples of 8 in x and 3
in y.)
