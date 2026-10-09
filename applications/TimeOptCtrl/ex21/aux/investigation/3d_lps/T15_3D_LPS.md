# T15 — 3D Local Projection Stabilisation (LPS) for ex21 (HEX27 Q2)

*Baseline:* **3d59677b8** "Add validated 3D variable wind support to ex21" (T14, committed at the start of this task). T15 is frozen in a single commit on top of this baseline (not pushed).

*Goal:* port the validated 2D T6 LPS to 3D unchanged in philosophy. W is symmetric PSD, added to L once; the adjoint stays the exact transpose. Validate it for distributed control (constant and variable winds, BE/CN, SWEEP, MPI), then for natural / Neumann / Dirichlet / multi-face / H1.

## 1. Audit of the 2D T6 implementation (written before any code change)

Sources read: `leveque_pearson/T6_LPS_STABILISATION.md` (full); `Discretization.cpp` (`BuildLpsMatrix`, the hook in `BuildNativeSpatialOperators`, `ElementCentroid`, `EliminateConstrainedDofs`); `Discretization.hpp` (`LpsStats`, `W_lps`); `Config.hpp` (Stabilization, LpsProjection, the refusals in Validate3D / ValidateSimplex); `CommandLine.cpp` (`--convection-stabilization`, `--lps-projection`); `ConvectionVerification.cpp` (G1 LPS-free, G5, `natural_gate`); `ExactOcBenchmark.cpp` (W dump, ν). `Solver.cpp` has no LPS code: it consumes the assembled L.

**A. Bilinear form** (LP eq. (7), patchwise δ):
  S_h(v, z) = Σ_K δ_K (κ_K(w·∇v), κ_K(w·∇z))_K,  so W_ij = S_h(φ_j, φ_i).
- **Discrete form:** W_K = δ_K Σ_g ω_g (f_g − Πf_g)(f_g − Πf_g)ᵀ, f_g,i = w(x_g)·∇φ_i(x_g), assembled as δ(A − Cᵀ G⁻¹ C) with A = Σ ω f fᵀ, C = Σ ω q fᵀ, G = Σ ω q qᵀ (Gauss–Jordan solve). It is symmetrised entrywise to remove round-off.
- **Patch:** one Q2 element (= LP's 2×2 Q1 patch on the same node grid).

**B. Projection space.**
- **P0** (production, "LPS-P0 (LP)"): q = {1}.
- **P1** (labelled diagnostic): q = {1, (x−x_c)/H_x, (z−z_c)/H_z}, physical coordinates centred at the centroid and scaled by the extents, discontinuous per element.

**C. Fluctuation.** κ_K = id − π_K, with π_K the L2(K) projection onto the patch space, computed with the SAME element Gauss rule (so W_K is exactly PSD).

**D. δ_K.** δ_K = h_K/(2|w_K|)·(1 − 1/Pe_K) if Pe_K > 1, else 0 (element skipped). Pe_K = |w_K| h_K/(2ν), |w_K| the Euclidean norm.

**E. h_K.** The FULL chord through the centroid along w_K, for an axis-aligned element of extents H_axis (bounding box of its nodes): h_K = min over axes with w_K,axis ≠ 0 of H_axis·|w_K|/|w_K,axis|. If |w_K| = 0 then Pe_K = 0 and the element is inactive; zero components are skipped. This is the T3/T4 `lp52_patch_peclet.py` definition, which matched LP's switch 18/18.

**F. Where w is evaluated.** w_K at the element centroid (mean of the coordinate nodes) for h_K, Pe_K and δ_K; w(x_g) at every Gauss point inside the integrand.

**G. Pe ≤ 1.** δ_K = 0 and the element contributes nothing. With zero active elements nothing is added and the run is byte-identical to LPS off (T6 §5).

**H. Symmetry.** W = Wᵀ exactly (each local block is symmetrised), PSD, even in w: W_{−w} = W_w, since f flips sign and δ depends on |w|. W·1 = 0.

**I. State and adjoint.**
- W is ADDED to the spatial operator once, `MatAXPY(K, 1, W)`, BEFORE the Dirichlet lifting and elimination: L = νK + C + W. W is kept separately (eliminated like M, diagonal 0) for dumps.
- The adjoint applies the exact transpose Lᵀ = νK + Cᵀ + W. No adjoint stabilisation is assembled. BE and CN consume the same L.

**J. Order loss in 2D.**
- **Fixed patch Péclet:** P0 drops Q2 from O(h³) to **O(h²)** (orders 2.33 → 2.07). κ₀ annihilates only constants, so ‖κ₀ w·∇y‖ = O(h) and δ ~ h. P1 keeps O(h³).
- **Fixed ν:** hides the loss (elements switch off as h → 0).
- **Time:** orders unchanged.

**K. P0 and P1** both exist (`--lps-projection p0|p1`, default p0). P1 is a diagnostic.

**L. 2D-specific geometric assumptions** (to generalise):
- the wind via `EvaluateVelocity`/`field(x, z)`, 2 components;
- `ElementCentroid(x, z)` (2D);
- extents and the chord over two axes (`axis < 2u`);
- P1 basis with two coordinates;
- W preallocated at 64 nonzeros per row (3D needs 125);
- in the verification driver, `natural_gate` requires stabilisation NONE (the natural/wrong-BC gates are skipped with LPS).

Also: Validate3D and ValidateSimplex refuse LPS in 3D; the EX21_LPS diagnostic line is printed in 2D and its format must stay byte-identical.

## 2. Frozen 3D form (fixed before implementation)

Same LPS form as T6, on HEX27 Q2 with one element per patch:

    S_h(y, v) = Σ_K δ_K (κ_K(w·∇y), κ_K(w·∇v))_K,   κ_K = id − π_K,   W_ij = S_h(φ_j, φ_i).

| item | 3D definition |
|---|---|
| projection space | **P0** (production): q = {1}, a 1×1 Gram matrix. **P1** (diagnostic): q = {1, (x−c_x)/H_x, (y−c_y)/H_y, (z−c_z)/H_z}, a 4×4 Gram matrix, solved by Gauss–Jordan (SPD) |
| quadrature | the element rule ("seventh": 4×4×4 Gauss) for A, C, G alike, so W_K is exactly PSD |
| centroid c | mean of the 27 coordinate nodes (`ElementCentroid3D`) |
| extents H_i | bounding box of the element nodes, i = x, y, z |
| w_K | w(c): the constant (vx, vy, vz) or the `field3d` wind (`EvaluateVelocity3D`) |
| \|w_K\| | sqrt(w_x² + w_y² + w_z²) |
| h_K | full chord through c along w_K: min over i with w_K,i ≠ 0 of H_i\|w_K\|/\|w_K,i\| (zero components skipped; \|w_K\| = 0 gives Pe_K = 0, an inactive patch) |
| Pe_K | \|w_K\| h_K / (2ν) |
| δ_K | h_K/(2\|w_K\|) (1 − 1/Pe_K) if Pe_K > 1, else 0 (the patch is skipped) |
| integrand | f_g = w(x_g)·∇φ(x_g), with w at each quadrature point (variable winds) |
| local matrix | W_K = δ_K (A − Cᵀ G⁻¹ C), symmetrised entrywise |
| state operator | L = νK + C + W (W added once, before the Dirichlet lifting/elimination) |
| adjoint operator | Lᵀ = νK + Cᵀ + W (the exact transpose; nothing is assembled for the adjoint) |
| natural BC | L + Lᵀ = 2νK + (B_w − D_w) + 2W |
| Dirichlet control | B_D / B_Dᵀ consume the unhatted L (with W); the free block is L_FF |
| preallocation | 125 nonzeros per row (HEX27 vertex coupling) |

## 3. Implementation (baseline 3d59677b8)

- **`Discretization.cpp` `BuildLpsMatrix`.** A dimension-3 branch: the 3D centroid, three extents, `EvaluateVelocity3D` at the centroid and at the Gauss points, the 3-axis chord, a P1 basis of 4 functions, and 125-nonzero preallocation. The 1D/2D path is unchanged expression by expression: the same `ElementCentroid`, `hypot`, axis loops bounded by min(dim, 2), and the same P1 coordinates. `LpsStats` gains pe_min, h_min and h_max, which are reduced over ranks.
- **Hook.** In 3D only, a new `EX21_LPS3D` line reports the active fraction, Pe min/max, h_K min/max, δ min/max, ‖W‖_F and nnz(W). The 2D `EX21_LPS` line is unchanged.
- **`Config.hpp` Validate3D.** The "LPS is not ported" refusal is lifted. Simplices keep refusing LPS (ValidateSimplex).
- **`ConvectionVerification.cpp`:**
  - **G15 adjoint negatives.** `lps_CT_minus_W` (Sᵀ = νK + Cᵀ − W) and `lps_CT_only` (Sᵀ = νK + Cᵀ). Both feed the transpose, Hessian and FD gates; "C + W" is the existing `ex20_style_wrong`.
  - **3D natural gate under LPS.** C = L − νK − W, plus the `EX21_VC_G15_NATURAL_SPLIT` check L + Lᵀ − 2νK − 2W = C + Cᵀ (then G1N/G1W on C). The wrong-BC negative becomes νK + W − C. The 2D natural gate stays off with LPS, so 2D logs are unchanged.
  - **Dirichlet control.** `lps_BDT_W_omitted` and `lps_BDT_W_wrong_sign` build B_Dᵀ from L − W and L − 2W, as separate `DirichletControlOperators` owners. `lps_adj_CT_minus_W` is the interior adjoint on the free block. The `EX21_VC_G15D_BDT_NEGATIVE` transpose defects are printed.

## 4. Independent operator reference and projection checks (`t15_operator.py` → `t15_projection.csv`, `t15_operator_reference.csv`)

**Method.** The binary's W is recovered as W_dump = K(lps) − K(none) from `--dump-operators`, with the same mesh and wind, under the natural state (no elimination) plus three homogeneous-Dirichlet free-block cases. Python reassembles W independently on the structured HEX27 lattice, using tensor-product 1D Q2 bases and the same 4-point Gauss rule per axis ("seventh"). It uses the frozen form of §2: box centre, chord, Pe, δ, w(x_g) in the integrand, and the P0/P1 projections.

**Cases (17).**
- Constant wind A on the unit and non-unit boxes, P0 and P1.
- Pe_K = 1.51, where the "Pe > 2" negative becomes detectable.
- Pe_K = 0.60: no active patch, so W = 0.
- Winds with zero components: (1,0,0) and (0,0,−¾).
- harmonic-3d and affine-div-3d at a median-Pe ν (15 or 16 of 24 patches active) and at ν = 1e-3, P0 and P1.
- Dirichlet free blocks.

**Results: 173/173 PASS.**
- **Agreement.** W_dump = W_ref to ≤ 3.2e-14 (relative max entrywise) in every case. The active patch count matches exactly, including the partial sets 16/24, 15/24 and 7/8.
- **Structure.** Symmetric (‖W−Wᵀ‖ = 0), PSD (λ_min/λ_max ≥ −1e-13), W·1 = 0 to ≤ 1e-13.
- **Negatives.** Each differs wherever the form makes it detectable, and is verified to coincide otherwise:
  - h_K = max extent: 0.34 on the (1,0,0) wind;
  - centroid-frozen integrand: detected on the variable winds, and identical by construction for constant winds;
  - no projection (δA): always detected;
  - "Pe > 2": detected at Pe = 1.51;
  - "Pe > ½": detected at Pe = 0.60.
- **Projection algebra.** The P1 Gram matrix is |K| diag(1, 1/12, 1/12, 1/12) (condition number 12) and the P0 Gram is |K|; Π₁ reproduces P1 to 1e-14 on two element shapes. These Gram checks use a 4e-15 bar. The first bar, 1e-15, failed at 1.1e-15 in the final re-run because multithreaded OpenBLAS changed the summation order; BLAS is now pinned to one thread.
- **Binary-level exact projection** (constant wind):
  - W_P0·v_affine = 2.4e-16 and W_P1·v_affine = 8.0e-16;
  - W_P1·v_quadratic = 7.1e-16 (κ₁ annihilates w·∇v ∈ P1);
  - **NEG** W_P0·v_quadratic = 1.0e-2.

## 5. Operator gates G15.1–G15.7 (in every `--verify-convection` run, `t15_gates_*.csv`; `EX21_VC_G5_LPS*`)

| gate | quantity | worst over 84 runs |
|---|---|---|
| G15.1 symmetry | ‖W−Wᵀ‖/‖W‖ | 0 |
| G15.2 PSD | min over 20 random x of xᵀWx/(‖W‖‖x‖²) | ≥ +1.7e-2 |
| G15.3 constants | ‖W1‖∞/‖W‖ | ≤ 2.5e-16 |
| G15.4 reversal | ‖W_w − W_−w‖/‖W‖, same active set | 0 |
| G15.5 Pe switch | ν raised to max Pe_K = 1: active, ‖W‖ | 0, 0 (every run) |
| G15.6 exact projection | W_P0 v_aff, W_P1 v_quad (§4) | ≤ 8e-16 |
| G15.7 no wind / Pe ≤ 1 | `t15_refusals.sh` Z rows: zero wind; ν = 1 constant wind; natural-var-3d at ν = ½ | active = 0, ‖W‖ = 0, solution **byte-identical** to Galerkin |

## 6. Gate battery: state, adjoint, BCs, multi-face, H1 (`t15_gates.sh`, `t15_gates_table.py`)

**Coverage.** 84 `--verify-convection` runs with LPS on, every one with all patches active, Pe_K from 1.27 to 806. The parts of the battery:
- **Distributed, homogeneous Dirichlet state (20).** Winds A, harmonic and affine-div; ν = 0.1 (mod), ν = 1e-3 (high) and P1; BE and CN; plus the exact conv-sin3-3d case at ν = 1e-2.
- **Natural state (16).**
- **Neumann (18).** Single face, opposite union and adjacent union, ν = 1e-2.
- **Dirichlet control (30).** zmin, opposite union, adjacent union, and a non-unit box with zmin+xmin, ν = 1e-2; H1-space, H1-time and combined on the harmonic wind.

**Positives (worst values).**

| quantity | worst value |
|---|---|
| shell transpose | 4.3e-17 |
| Hessian symmetry (dist / Neumann / Dirichlet) | 2.9e-13 / 6.8e-13 / 2.1e-15 |
| FD gradient, exact and shell | ≤ 1.3e-11 |
| production true residual | ≤ 3.2e-14 (Dirichlet ≤ 6.8e-12) |
| forward march | ≤ 3.2e-11 |
| `G15_NATURAL_SPLIT` ‖L+Lᵀ−2νK−2W−(C+Cᵀ)‖/‖L+Lᵀ‖ | ≤ 5.5e-17 |
| natural identity C+Cᵀ = B_w (constant wind) | ≤ 3.0e-14 |
| natural identity C+Cᵀ = B_w − D_w (variable winds) | ≤ 3.0e-14 |
| G1W_DIRICHLET | ≤ 1.9e-14 |
| G13N face union | 0, true-union FD ≤ 8.5e-10 |
| B_Dᵀ transpose | ≤ 8.6e-18 |
| R symmetry | ≤ 1.4e-14 |
| Dirichlet KKT state residual | ≤ 6.7e-12 |
| Dirichlet \|grad J\| | ≤ 2.8e-9 |

**Negatives (minimum detection over the battery; bars: transpose > 1e-6, FD > 1e-4).**

| wrong operator | transpose defect | FD |
|---|---|---|
| adjoint νK + Cᵀ − W | ≥ 4.3e-4 | ≥ 4.8e-2 |
| adjoint νK + Cᵀ (W dropped) | ≥ 2.2e-4 | ≥ 5.2e-2 |
| C + W untransposed (ex20 style) | ≥ 5.8e-4 | ≥ 3.2e-2 |
| natural wrong BC νK + W − C | — | ≥ 0.13 |
| missing −(div w)p, affine-div | — | ≥ 1.0 |
| multi-face drop / duplicate face | — | ≥ 2.4e-2 |
| Dirichlet legacy B_Dᵀ | ≥ 1.2e-3 | ≥ 2.7e-4 |
| Dirichlet B_Dᵀ from L − W (W omitted) | ≥ 6.7e-4 | ≥ 1.3e-4 |
| Dirichlet B_Dᵀ from L − 2W (wrong sign) | ≥ 1.3e-3 | ≥ 2.6e-4 |
| Dirichlet interior adjoint Cᵀ − W | — | ≥ 0.99 |

The weakest FD detection, 1.3e-4 for "B_Dᵀ with W omitted", is the harmonic-wind H1-time run under CN: the time regularisation damps the gradient's response. Its transpose defect there is 1.6e-2.

**Result: 84/84 PASS.**

**Documented revision** (made after the first table; the script docstring records it):
- **The miss.** Three natural-state runs at ν = 1e-3 missed the round-off bars: Hessian symmetry 2.7e-12, 2.4e-11 and 1.5e-11 (bar 1e-12), and |grad J| = 1.19e-8 (bar 1e-8).
- **Evidence for a conditioning floor:**
  - the exact-LU adjoint shows the same floor (2.66e-12 / 2.41e-11);
  - the KKT true residuals are about 5e-16;
  - **Galerkin at the same ν is far worse:** its Hessian floor is 3.6e-10 / 7.8e-8, and its production solve stalls (|grad J| = 2e-2) or diverges (reason −3).
- **The revised bar.** For the natural-state rows at ν = 1e-3 only, the shell Hessian must be ≤ max(1e-12, 2 × the exact-LU value) with exact-LU ≤ 1e-10, and |grad J| ≤ 1e-7 provided the true residual is ≤ 1e-14. These rows are flagged `bar_revision`; all other bars are unchanged.

## 7. MPI (np = 1/2/4, `t15_mpi.sh`, `t15_mpi.py` → `t15_mpi.csv`)

**Cases** (n = 4, N = 8, LPS-P0 on unless noted):

| case | configuration | solver |
|---|---|---|
| A | distributed, wind A, BE | SWEEP |
| B | distributed, harmonic, CN | SWEEP |
| C | natural, affine-div | SWEEP |
| D | Neumann zmin+zmax, harmonic | SWEEP |
| E | Dirichlet zmin+xmin, affine-div | reduced |
| F | Dirichlet zmin+zmax, harmonic, combined H1 | reduced |
| G | P1, wind A | SWEEP |

The SWEEP cases use LU at np = 1 and MUMPS at np > 1; E and F use the reduced solver.

**Result: 21/21 PASS** (T14 criteria; the RSS column is informational, since concurrent ranks can interleave their `/usr/bin/time` lines in the shared log):
- fields ≤ 6.5e-15 and J ≤ 7.9e-16 relative to np = 1;
- identical outer counts, core counts (7 / 4) and residual histories;
- control traces equal the predicted face sets.
- **LPS partition invariance:** the `EX21_LPS3D` line (active fraction, Pe, h_K, δ, nnz(W) = 35937; ‖W‖_F to 1e-12) is identical at every np. Each patch is one element, assembled by its owner.

## 8. Refusals and acceptances (`t15_refusals.sh`, 62/62)

**Refused (15), each with an explicit message:**
- LPS on simplex meshes;
- LPS without `--convection on`;
- LPS with Neumann H1-space or H1-time;
- 2D wind presets in 3D;
- an unknown projection (p2) or an unknown method (supg);
- time- or state-dependent winds (no field type exists, so no CLI can request them; unknown `--wind` names are rejected);
- Neumann CN without SWEEP;
- Dirichlet control with SWEEP or MATCHING;
- Dirichlet control with a natural state;
- SCHUR_MATCHING for 3D distributed control;
- natural-var-3d with a non-potential wind.

**Accepted with LPS active (44):**
- distributed P0 and P1, natural, Neumann union, Dirichlet union, and combined Dirichlet H1, for all three winds under BE and CN;
- distributed ADDITIVE_GAMG under BE;
- the Dirichlet dense core with `--verify-dirichlet-core`;
- the exact cases conv-sin3-3d (constant and harmonic), neumann-conv-3d, dirichlet-conv-3d, dirichlet-h1-3d and dirichlet-opp-3d.

No exact-case + LPS combination is invalid. LPS perturbs the discretisation, and the exact error then measures that perturbation. The only exact-case refusals are the pre-existing wind-compatibility ones.

**Inactive (3).** Zero wind, Pe ≤ 1 with a constant wind, and natural-var-3d at ν = ½. Each is byte-identical to Galerkin (G15.7).

## 9. Spatial convergence (`t15_space.sh`, `t15_space_table.py` → `t15_space_convergence.csv`, `t15_p1_diagnostic.csv`, figure `t15_space_convergence.png`)

**Setup.** Exact conv-sin3-3d, CN N = 64, n = 2, 3, 4, 6, 8. Every LPS run has all patches active; the activity report per run is in the CSV.

State y orders, n = 3→4, 4→6, 6→8:

| ladder | Galerkin | LPS-P0 | LPS-P1 (diagnostic) |
|---|---|---|---|
| **fixed Pe_K = 10** (ν = 0.0906/n) | 0.16, 2.75, 2.68 | **1.91, 1.96, 2.10** | 1.57, 2.88, 2.78 |
| fixed ν = 1e-3, wind A (Pe_K 302 → 113) | −0.06, 2.60, 2.26 | 1.77, 1.59, 1.98 | 1.79, 2.48, 2.34 |
| fixed ν = 1e-3, harmonic-3d | −0.73, 2.81, 2.65 | 1.91, 1.72, 1.62 | — |

Galerkin is pre-asymptotic at n ≤ 4, where it shows odd/even oscillation.

- **P0 order loss reproduced in 3D.** At fixed Pe_K, LPS-P0 converges at O(h²), one Q2 order below Galerkin, as in 2D T6 (2.07). κ₀ annihilates only constants, so ‖κ₀(w·∇y)‖ = O(h), and δ ~ h.
- **P1 keeps the Galerkin rate.** It sits within 1.15× of the Galerkin error at n = 8 (Pe_K = 10).
- **The smooth exact solution has no layers,** so Galerkin is the more accurate method here. LPS's value lies in layers and solver robustness (§10, §13).
- **Adjoint and control** converge at about 3 for all methods at fixed Pe; p is smaller and less affected by the stabilisation of y.

## 10. High-Pe 3D benchmark with activity report (`t15_benchmark.sh`, `t15_benchmark.py` → `t15_lps_benchmark.csv`, figure `t15_lps_activity.png`)

**Setup.** Distributed control, homogeneous Dirichlet state, constant source f = 1, α_u = 1, ν = 1e-3, T = 1, CN N = 16. The state is transport-dominated, with outflow boundary layers. n = 4, 6, 8, 12, 16; winds A and harmonic-3d.

**Activity** (LPS-P0; 100 % of patches active in every LPS run):

| | n = 4 | n = 16 |
|---|---|---|
| Pe_K, constant wind | 227 | 57 |
| Pe_K range within a run, harmonic | 25–225 | 1.5–67 |
| δ_K, constant wind | 0.124 | 0.031 (∝ h) |
| max δ_K, harmonic (largest where \|w_K\| is small) | 0.64 | 0.31 |
| ‖W‖_F, constant wind | 1.19 | 0.61 |

nnz(W) = nnz(L) in every run.

**Results** (reference: LPS-P0 at n = 16, max y = 1.858):

| | Galerkin | LPS-P0 | LPS-P1 (diag.) |
|---|---|---|---|
| undershoot −min y / max y, wind A, n = 4/6/8/12 | 13.7 % / 7.4 % / 2.8 % / 0 | **0 / 0 / 0 / 0** | 0.4 % / 10.7 % / 15.6 % / 13.3 % |
| same, harmonic | 31 % / 28 % / 23 % / 13 % | **0 at every n** | 0–6 % |
| max y, wind A (ref 1.858) | 2.79 / 2.88 / 2.89 / 2.84: **~55 % overshoot, not decaying** | 1.70 / 1.80 / 1.84 / 1.865: monotone convergence | 1.80–1.93 |
| J deviation vs ref at n = 12 | 7.2e-2 | 3.2e-2 | 1.1e-2 |

LPS-P0 removes the layer wiggles at every resolution. P1 does not reliably do so, which confirms that it stays a diagnostic. These are the same conclusions as in 2D (T6: "P0 rescues; undershoot transient").

## 11. Time convergence with LPS active (`t15_time_runs.sh`, `t15_time_table.sh` → `t15_time_convergence.csv`, figure `t15_time_convergence.png`)

**Setup.** Fixed mesh n = 4, ν = 1e-2 (Pe_K ≈ 23, all patches active), self-convergence against nt = 1024.

**Orders at nt = 64 → 128:**

| case | p | y |
|---|---|---|
| distributed wind A, BE | 1.05 | 1.01 |
| distributed wind A, CN | 2.09 | 2.10 |
| distributed harmonic, CN | 2.10 | 2.09 |
| Galerkin twin, CN | 2.11 | 2.02 |
| Dirichlet zmin, BE | 1.09 | 1.03 |
| Dirichlet zmin, CN | 2.13 | 1.21 |

**The Dirichlet CN state is pre-existing behaviour.** In the CN Dirichlet case the state y converges at about 1.2 with LPS and equally with Galerkin (1.18). The control on t ≤ T/2 converges at **2.00 in both**, and p at 2. The reduced state order is the known terminal-time layer of the nodal Dirichlet control under CN (T11D), not LPS.

**Verdict:** BE1 / CN2 are preserved with LPS active.

## 12. Spectral diagnostics (optional; `t15_spectral.py` → `t15_spectral.csv`; dense, n = 4)

**Homogeneous Dirichlet state, wind A, ν = 1e-3:**

| | Galerkin | LPS-P0 | LPS-P1 |
|---|---|---|---|
| λ_min(M⁻¹ sym L) | 0.030 | 0.082 | 0.034 |
| ‖skew L‖/‖sym L‖ | 20.2 | 0.26 | 0.44 |
| cond₂(L) | 98 | 52 | 33 |

- LPS turns a skew-dominated operator into a symmetric-dominated one, about 80× less skew. This is what the AMG/ILU-preconditioned spatial solves see; see §13.
- Harmonic wind: skew/sym goes from 13.9 to 0.28.
- Natural state: L·1 = 0, so L is singular and cond₂ is meaningless (labelled in the CSV). λ_min(sym L) < 0 there for every method, because B_w has inflow faces. LPS raises it from −36.7 to −9.5.

## 13. Solver study (`t15_solver.sh`, `t15_solver_table.py` → `t15_solver.csv`, figure `t15_solver_iterations.png`)

**Setup.** n = 8, N = 32, production rtol 1e-8, `--solve-only`, no tuning, wall cap 600 s per run (exit 124 = not converged within the cap).
- **SCHUR_SWEEP cases at ν = 1e-3:** spatial LU, GMRES+GAMG and GMRES+ASM(4,1)+ILU(1), each with Galerkin and LPS-P0, under BE and CN.
  - dA: distributed, Dirichlet state, wind A;
  - dH: the same with harmonic-3d;
  - nA: natural state;
  - mA: Neumann zmin.
- **Reduced Dirichlet solver** (rA/rH: zmin+xmin, ν = 1e-2, wind A and harmonic): iterative CG core vs dense core.

**Converged runs:**

| case | Galerkin | LPS-P0 |
|---|---|---|
| dA distributed, wind A | 4/6 (GAMG: cap, BE and CN) | **6/6** |
| dH distributed, harmonic | 5/6 (BE GAMG: cap; CN GAMG 81 315 its, max 126 per solve) | **6/6** (GAMG max 18–19 per solve) |
| nA natural state | **0/6** (LU included: cap) | 2/6 (LU, 3 outer its; GAMG/ASM: cap) |
| mA Neumann zmin | **0/6** (LU: rejected, reason −11, sweep failures 32; GAMG/ASM: cap) | **6/6** (GAMG max 21–24, ASM max 6–7) |
| rA/rH Dirichlet, iterative and dense core | 4/8 (BE: cap with both cores; CN converges, spatial max 51–148 per solve) | **8/8** (spatial max 17–22, core CG 4 / 11–12) |
| **total** | **13/32** | **28/32** |

**Findings:**
- **Every LPS run that converges matches its LU / dense reference.** Distributed and Neumann fields agree to ≤ 2.4e-10 (J ≤ 3.3e-11). In the Dirichlet runs the dense and iterative cores agree to ≤ 1.5e-11, and the iterative core is 6–14× faster (6–16 s vs 80–94 s).
- **The core is unchanged.** The Dirichlet core CG count is the same with and without LPS (4 or 11–12). The Galerkin failures are in the inner spatial solves, which is consistent with §12: LPS cuts skew/sym from 20 to 0.26.
- **Natural state at ν = 1e-3 remains hard even with LPS.** Only the LU sweep converges, because the singular, inflow-dominated natural operator (§12) defeats GAMG and ASM. This is a limit to state, not a defect: Galerkin fails there with every backend.
- **Galerkin + LU sweeps "converging" in dA/dH does not mean Galerkin is usable.** At this resolution the Galerkin solution itself oscillates (§10).

## 14. Memory (`t15_memory.py` → `t15_memory.csv`, figure `t15_memory.png`)

**Setup.** N_t = 32, n = 4, 6, 8.
- Distributed: conv-sin3-3d data, ν = 1e-3, SWEEP + LU.
- Dirichlet control zmin: ν = 0.1, iterative core.

**Results.**
- **nnz(L) is unchanged** (35937 / 117649 / 274625). W is added with SUBSET_NONZERO_PATTERN inside the Q2 coupling pattern.
- **nnz(W) = nnz(L).** W is kept as a separate matrix for diagnostics and dumps, an estimated 3.1 MB at n = 8.
- **MaxRSS increase: +0.6 % to +1.3 %**, for example 1041.7 → 1048.6 MB in the distributed case at n = 8.
- **No space-time storage is added.** W enters only the spatial operator, so bytes per space-time unknown are unchanged to within 1 %.

**Why the Dirichlet kind uses ν = 0.1.** ν = 1e-2 was the first choice, but there the **Galerkin** Dirichlet runs stall from n = 6 on, with both the iterative and the dense core (§13). ν = 0.1 gives a like-for-like comparison; LPS is still active on every patch (Pe_K 1.13–2.3).

## 15. TET10 probe (optional): not done

FEMuS has no native TET10. SECOND on simplices means TET15 (memory note T12), and the T12 simplex path uses SERENDIPITY. A simplex LPS needs its own chord, centroid and Gram-matrix definitions, plus a validation of the P1 space on tetrahedra. Following the instruction not to delay T15, simplex + LPS stays **refused**, with an explicit message.

## 16. Regression (`t15_regress_all.sh`; OLD = private clean build of 3d59677b8, sha256 1c218b7d…; NEW sha256 50ee7b38…)

All suites exit 0.

**Suites:**

| suite | what it covers |
|---|---|
| 2D | ex20 goldens 61/61; Math2025 Ex3/Ex4/Ex5 byte-identical; T0–T8; the T6 LPS cases |
| T9, T10, T11N, T11D, T11H1, T12, T13 | old-vs-new suites |
| **new T14 suite** | variable-wind exact cases, gates, JSON and operator dumps |
| **2D T6 LPS byte-identity** | below |

**2D T6 LPS byte-identity.** P0 and P1 gates, distributed, natural, Dirichlet and q2-divfree, BE and CN; LP52; production JSON; W/L operator dumps for P0 and P1 with lp-recirculating. All IDENTICAL.

**The items flagged by the suites, each explained:**
- **EF 9.5 diff against the historic golden.** It is the `variant=source` metadata token, the same 4 lines as in T14. Comparing OLD vs NEW directly, the output is byte-identical once wall-time fields are stripped: 415 lines.
- **`v2d_gate_lp52` in T13.** A stderr interleave artifact, because the NEW side captures 2>&1; as in T14. The stdout of both binaries is byte-identical (45 lines).
- **`parity_neumann_*_w0_boundaryreduced` "JSON_DIFFERS".** Neither binary writes a JSON: both refuse with the identical message (exit 134), so the compare reads two missing files. Same as T14.
- **Intended acceptances:**
  - `t9_3d_lps_ACCEPTED_NOW_T15` — 3D LPS is now accepted;
  - `convection+dirichlet_ACCEPTED_NOW` and `t8f_..._inert` — pre-existing labels.

**Build hygiene.** `env -i` build; 0 miniforge tokens in `readelf -d` and in the CMake cache. ex20 untouched.

**Derived views** (`t15_subsets.py`):
- `t15_variable_wind.csv`: 56 gate rows with harmonic or affine-div winds, 56 PASS;
- `t15_multiface.csv`: 30 Neumann/Dirichlet face-union rows, 30 PASS.

## 17. Verdicts

### 3D LPS — DISTRIBUTED CONTROL: **VALIDATED**

| criterion | evidence |
|---|---|
| independent assembly of W, constant and variable winds, P0/P1 | §4: 173/173, W_dump = W_ref ≤ 3.2e-14, negatives detected |
| operator gates G15.1–G15.7 | §5: symmetry 0, PSD ≥ 1.7e-2, W1 ≤ 2.5e-16, reversal 0, Pe switch exact, exact projection ≤ 8e-16, inactive LPS byte-identical to Galerkin |
| state L = νK + C + W, adjoint = exact transpose | §6: 20/20 distributed (shell transpose ≤ 2e-17, Hessian ≤ 2.9e-13, FD ≤ 1e-12, production residual ≤ 2.1e-14) |
| adjoint negatives Cᵀ − W, Cᵀ only, C + W | §6: transpose ≥ 2.2e-4, FD ≥ 3.2e-2 in every run |
| BE/CN, moderate and high Pe, constant and variable winds | §6, §11 (BE1/CN2 preserved) |
| convergence and activity report | §9 (P0 O(h²) at fixed Pe_K, P1 O(h³) recovery), §10 benchmark |
| MPI | §7: 21/21, LPS activity partition-invariant |
| memory | §14: +0.6–1.3 % RSS, nnz(L) unchanged |
| regression | §16: no differences; 2D T6 LPS byte-identical |

### 3D LPS STABILISATION: **VALIDATED**

The distributed verdict above holds, and in addition:

| criterion | evidence |
|---|---|
| natural BC: L + Lᵀ = 2νK + (B_w − D_w) + 2W | §6: split ≤ 5.5e-17, identities ≤ 3e-14, 16/16 (documented conditioning-scaled bars for the ν = 1e-3 rows) |
| Neumann single / opposite / adjacent union | §6: 18/18, G13N union exact, drop/duplicate negatives detected |
| Dirichlet: B_D/B_Dᵀ consume L incl. W; W omitted / wrong sign detected; core SPD; dense = matrix-free | §6: 30/30; §13 solver (dense vs iterative) |
| Dirichlet H1 (space, time, combined) smoke | §6: 12 H1 runs PASS |
| multi-face | §6 (Neumann and Dirichlet unions, non-unit box), §7 (D, E, F) |
| refusals | §8: 62/62 |

**Limits stated, not hidden:**
- **P0 costs one Q2 order** in the asymptotic regime. This is by design (LP), the same as in 2D.
- **P1 is a diagnostic.** It keeps the order but does not remove layer undershoots (§10).
- **Simplex + LPS remains refused** (§15).
- **High-Pe natural-state problems at ν = 1e-3 stay ill-conditioned** (§6 revision; §13).

## 18. Stop, and the project-level recommendation

T15 closes here. No further feature is started; T15 is frozen in one commit on top of 3d59677b8, and nothing is pushed.

**Recommendation: a paper-readiness audit, before cluster scaling.**

**Why this order.** The numerical scope is complete and validated in 2D and 3D:
- distributed, Neumann and Dirichlet control;
- L2, H1-space and H1-time regularisation;
- multi-face boundaries, variable winds, simplices, and now LPS.

T15 also showed that the *solver* story is now the open scientific question. In 3D at high Pe:
- Galerkin is unsolvable in 19 of 32 configurations;
- LPS rescues 28 of 32, but natural-state problems at ν = 1e-3 still need LU sweeps;
- the GAMG/ASM iteration counts grow with Pe.

Scaling a solver whose robustness envelope is not yet mapped would spend cluster time on configurations that may not converge.

**The audit should:**
1. **Freeze the claims.** Write a claims → evidence table from T0–T15, with one canonical script per figure, and regenerate every paper figure from the committed CSVs.
2. **Map the robustness envelope.** Iterations vs (n, N, Pe, α_u) for SWEEP + GAMG/ASM with LPS, at the mesh-independent resolution nx = 16 (memory note). Identify where the natural-state failure begins.
3. **Close the documented gaps, or state them as limitations:**
   - the P0 order loss (offer P1 or a higher-order LPS as a remark);
   - the Dirichlet CN terminal layer;
   - simplex LPS (refused);
   - the conditioning floors at ν = 1e-3.
4. **Only then run cluster scaling** (np ≫ 4, nx ≥ 32 in 3D) on the configurations the envelope marks as robust: distributed and Neumann with LPS and SWEEP + GAMG, and Dirichlet with the iterative core. The MPI invariance already proven at np = 1/2/4 makes this a performance study, not a correctness study.
