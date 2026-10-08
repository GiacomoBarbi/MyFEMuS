# T10 — 3D phase 2: volume geometry, data and natural-BC closure (ex21, HEX27 Q2)

*Baselines: e0410efe3 (frozen 2D) and 8ed57d629 (T9, 3D phase 1). Scope: rectangular 3D
boxes; distributed control/tracking boxes, multi-zone and non-colocated; single and
multiple source regions; the natural homogeneous state BC; constant 3D wind; BE/CN;
L2 control; SCHUR_SWEEP; MPI. No boundary control, H1 boundary regularisation, LPS,
variable winds or non-hex meshes. No commit, push or staging.*

Scripts, logs and results are in `t10/`; the CSVs are in this directory.

## 1. Dependency map (written before any code change)

| file | symbol | T9 state | T10 change | 1D/2D path |
|---|---|---|---|---|
| Config.hpp | `Domain {xmin,xmax,zmin,zmax,ymin,ymax}` | 3D refused unless the unit cube | allow any box with xmin<xmax, ymin<ymax, zmin<zmax (validated in 3D) | untouched (y fields unread) |
| Config.hpp | `Box {xmin,xmax,zmin,zmax}` | x and storage axis 1 only | **add `ymin, ymax`** (defaults 0, 1; read only in 3D). 3D convention as for Domain: (x, y, z) = storage axes (0, 1, 2), so `ymin/ymax` = axis 1 and `zmin/zmax` = axis 2 in 3D | aggregate initialisers `{x0,x1,z0,z1}` unchanged; y never read |
| Config.hpp | `ValidateBox(box, domain, …)` | x, z only | dimension-aware: in 3D also ymin<ymax and y-containment | 2D/1D messages and checks identical |
| Config.hpp | `Validate3DPhase1` | refuses natural BC, non-unit cube, boxes, multi-zone, sources | becomes the T10 whitelist: these five are allowed; the rest stays refused (§14) | n/a |
| Config.hpp | `ControlRegionIsWholeDomain` | **x, z only** | 3D also compares y. *Latent bug otherwise:* a 3D box full in x and z but partial in y would select the full mass `M` as the control operator (`SelectedControlOperator`) and silently control the whole domain | 2D expression unchanged |
| Config.hpp | `OwnedDofIsControlled(x, z)` (diagnostic mask count `geometric_mask_dofs`) | x, z | optional `axis2`; 3D tests (x, y, z) against (x, y, z) extents | unchanged |
| Discretization.cpp | `ElementCentroid` | axes 0, 1 | also axis 2 in 3D (new 3D overload; the 2D one untouched) | unchanged |
| Discretization.cpp | `CentroidIsInBox` | strict, axes 0/1 | 3D variant: strict on all three axes | unchanged |
| Discretization.cpp | `TrackingRegionIndex`, `ControlRegionIndex` | 3D returns 1 | 3D: the same union / lowest-index-wins logic with the 3D centroid test | unchanged |
| Discretization.cpp | source-region union in `BuildNativeSpatialOperators` | not reached in 3D | 3D centroid test; same "assemble once, first match breaks" | unchanged |
| Discretization.cpp | natural BC | nothing to assemble: no row is constrained, the adjoint uses the exact transpose of L = νK + C, which carries ν∂ₙp + (w·n)p = 0 (T7N/T7F) | none | — |
| Discretization.cpp | face loop (M_Γ, K_Γ) | 2D only (1D-face arc-length stiffness) | stays 2D-only. **New** `BuildBoundaryConvectionMatrix`: Σ over all boundary faces of ∫_f (w·n_f) φᵢφⱼ with the JacobianSur outward normal; used only by the 3D G1N gate | not called |
| ConvectionVerification.cpp | G1N / G8b (natural gate) | `dimension == 2` (face masses from 4 Neumann-face M_Γ builds) | 3D branch: G1N from `BuildBoundaryConvectionMatrix` with a per-face (normal, w·n, area) report; G8b wrong-BC adjoint (νK − C) is dimension-free and is enabled in 3D | 2D branch byte-identical |
| ConvectionVerification.cpp | `without.convection.vx = vy = 0` | vz left as is | also vz = 0 (K-only operator) | no effect in 2D (vz = 0) |
| ExactOc3D.cpp | `conv-sin-3d`, `conv-sin3-3d` | Dirichlet, unit cube | **new `natural-conv-3d`**: separable R_k profiles (3D T7F); natural BC; boxes set over the full domain including y | — |
| CommandLine.cpp | `--domain`, `--control-box`, `--tracking-box`, `--*-box-add`, `--source-box-add` | 4-number (x, axis-1) syntax; 3D refuses `--domain` | **new 3D-only options** with 6 numbers: `--domain3d`, `--control-box3d`, `--control-box3d-add`, `--tracking-box3d`, `--tracking-box3d-add`, `--source-box3d-add` (x0 x1 y0 y1 z0 z1). The 4-number options stay 2D/1D; in 3D they are refused unless interpretation-free (§2) | 2D parsing untouched |
| CommandLine.cpp | `--control-full-domain` | copies the domain at parse time (x, z) | in 3D re-applied after parsing with all three axes | 2D unchanged |
| Output.cpp | `--write-region-fields` | per-element `TrackingRegionIndex`/`ControlRegionIndex` | generic once the index functions are | unchanged |
| Solver.cpp / Output.cpp | `EX20_PAR_SOLVE control=DISTRIBUTED_WHOLE/BOX`, `geometric_mask_dofs` | via the two predicates above | follow the predicate fixes | unchanged |
| Solver.cpp | source load `M I_h f` or `M_source I_h f` | dimension-free | none | — |

## 2. CLI decision

The historical 4-number syntax names (x, storage-axis-1) extents. In 3D that is ambiguous
(axis 1 is y and z is a third axis), so it is **not** reinterpreted. New, clearly 3D-only
options take six numbers in the storage-axis order (x, y, z):

    --domain3d           x0 x1 y0 y1 z0 z1
    --control-box3d      x0 x1 y0 y1 z0 z1      (primary distributed-control box)
    --control-box3d-add  x0 x1 y0 y1 z0 z1      (additional control region)
    --tracking-box3d     x0 x1 y0 y1 z0 z1      (primary tracking box)
    --tracking-box3d-add x0 x1 y0 y1 z0 z1
    --source-box3d-add   x0 x1 y0 y1 z0 z1

- They are refused unless the final dimension is 3.
- In 3D the 4-number box options are refused. The one exception keeps the T9 command lines valid: `--tracking-box a b c d` / `--control-box a b c d` that is the full x extent with (c, d) equal to **both** the y and z extents of the domain. That box means the full domain under either reading, so nothing is reinterpreted.
- `--domain`, `--*-box-add` and `--source-box-add` are always refused in 3D.
- In 3D the tracking and control regions must be explicit: `--tracking-box3d …` or the full-domain form above; `--control-full-domain` or `--control-box3d …`. The 2D default boxes [¼, ¾]² have no 3D meaning and are refused.
- The 3D exact cases own their regions and domain; region or domain options are refused with them.

## 3. Box and centroid convention

A 3D box is [x0, x1] × [y0, y1] × [z0, z1], with y = storage axis 1 and z = storage axis 2.

An element belongs to a box iff its centroid (the arithmetic mean of its 27 HEX27
coordinate nodes, which is the geometric centre of an affine hexahedron) lies **strictly**
inside it:

    x0 < x_c < x1,  y0 < y_c < y1,  z0 < z_c < z1.

A centroid on a box face is outside. This is the 2D rule extended by one axis.

A region list is a union. An element is assembled into M_track / N / M_source **once**,
however many boxes contain it. For region-field output the reported index is the
**lowest** matching one: 1 is the primary box, then the additional boxes in command-line
order.

## 4. Implementation summary

- **Config.hpp**
  - `Box` gains `ymin, ymax`, read only in 3D.
  - `ValidateBox(…, dimension)` checks the y extent in 3D.
  - `Validate3DPhase1` becomes `Validate3D`, the T10 whitelist (§14).
  - `ControlRegionIsWholeDomain` compares y in 3D (the latent bug of §1).
  - `OwnedDofIsControlled(…, axis2)` gains the third axis.
- **Discretization.cpp/.hpp**
  - New 3D helpers: `ElementCentroid3D`, `CentroidIsInBox3D` (strict), and `RegionIndex3D` (lowest index wins).
  - The tracking, control and source indices use them in 3D. `SourceRegionIndex3D` / `ElementCentroid3DOf` are exported, so assembly and diagnostics share one implementation.
  - **New** `BuildBoundaryConvectionMatrix`: Σ over all boundary faces of ∫(w·n)φᵢφⱼ with JacobianSur outward normals, plus per-face area, mean normal, mean w·n and max |w·n|.
- **ConvectionVerification.cpp:** the natural-BC gate is enabled in 3D. G1N uses the new matrix with a six-face report; G8b uses the wrong-BC adjoint νK − C. The 2D branch is unchanged.
- **ExactOc3D.cpp/.hpp:** the new `natural-conv-3d` lives in **separate** functions (`Nat*`). The T9 data functions are textually unchanged; a first version with `if (g_natural)` branches inside them changed the T9 exact runs at 1e-15 through compiler contraction/inlining (§13).
- **CommandLine.cpp:** the 3D-only 6-number region options, applied after the presets (`ApplyRegions3D`, §2); `natural-conv-3d` dispatch; the 3D refusal messages now read "3D: …".
- **Verification.cpp:** `--dump-operators` in 3D also writes `<prefix>_regions3d.txt` (per element: centroid and the tracking/control/source indices). 2D dumps are unchanged.

## 5. Geometry and region-mask gates (`t10_geometry.py`, `t10_geometry.csv`: 224/224 PASS)

Every case checks five things:
- the element centroids against the analytic grid;
- every element's tracking, control and source index against an **independent** Python evaluation of the strict centroid rule with lowest-index precedence;
- 1ᵀM1 = domain volume;
- 1ᵀM_region1 = n_sel·h_x h_y h_z;
- for mesh-aligned boxes, 1ᵀM_region1 = the geometric union volume.

Volumes are measured on natural-BC dumps; Dirichlet-eliminated masses are not volumes.

| case | result |
|---|---|
| one aligned control box [0,½]³ | volume 0.125 exact |
| two disjoint boxes | union volume 0.25; M(A∪B) − M(A) − M(B) = 0 entrywise |
| two overlapping boxes C = [0,¾]×[0,½]², D = [¼,1]×[0,½]² | vol(C∪D) = 0.25 < vol C + vol D = 0.375; M(C∪D) = M(explicit union [0,1]×[0,½]²) **bitwise**; precedence (index 1, index 2) = (12, 4) as predicted |
| full-domain box | `N_selected` = M bitwise (DISTRIBUTED_WHOLE) |
| box full in x, z but y ∈ [0,½] | `N_selected` = M_ctrl_box ≠ M (difference 2.4e-3): the 3D whole-domain fix works |
| non-aligned box [0.1,0.6]×[0.2,0.9]×[0.3,0.7] | element set = centroid rule |
| box x < 0.375 (centroids at x = 0.375 lie on the face) | excluded: only the 16-element x = 0.125 layer is selected (strict rule) |
| tracking: two disjoint (0.5), overlapping vs a disjoint partition of the same union | volumes exact; M_track identical bitwise |
| non-colocated 2 control + 2 tracking octants | **0 elements** in both; 57 shared DOFs, which are box-face nodes only |
| sources: one, two, overlapping vs explicit union, full-domain box | volumes 1/64, 2/64, 3/64 exact; overlap = union bitwise; M_source(full box) = M bitwise |
| natural BC with wind A | no constrained row; (K + C)1 = 0 to 1e-13 |
| Dirichlet BC with wind A | constrained rows = boundary nodes exactly |
| non-unit domains (0,2)×(−1,1)×(0,½) and (−1.5,½)×(2,3)×(−¼,1¾) | coordinate extents, DOFs = Π(2n+1), total volume 2 and 2, half-domain boxes (vol 1), Dirichlet boundary set |

## 6. Natural state BC (formulation)

- **State:** y_t − νΔy + w·∇y = f + ηu, with ν∂ₙy = 0 on all six faces (natural: nothing is constrained).
- **Adjoint:** the exact discrete transpose of L = νK + C, i.e. −p_t − νΔp − w·∇p = α_t(y − y_d) with ν∂ₙp + (w·n)p = 0. This is not replaced by homogeneous Neumann for p.
- **Why the Robin condition is automatic:** for constant (divergence-free) w, C + Cᵀ = B_w = Σ_f ∫_f (w·n)φᵢφⱼ. The transpose therefore carries exactly the boundary term that turns the natural state condition into the Robin adjoint condition.
- **No new assembly:** natural BC is the existing unconstrained path. T10 lifts the 3D refusal and verifies the identity.

## 7. 3D boundary identity G1N (`t10_natural_bc.csv`)

The gate checks ‖C + Cᵀ − B_w‖_F / ‖C + Cᵀ‖_F with B_w from the six-face JacobianSur assembly. It holds to **2.6–3.5e-14** for every natural run:
- winds A = (1, ½, −¾) and B = (−2, 1, ½);
- n = 2, 3, 4;
- the non-unit domain.

The per-face reports confirm the outward normals and the FEMuS face map:

| face | normal | w·n (A) | w·n (B) |
|---|---|---|---|
| 1 | (0,0,−1), z = 0 | +0.75 | −0.5 |
| 2 | (0,−1,0), y = 0 | −0.5 | −1 |
| 3 | (+1,0,0), x = 1 | +1 | −2 |
| 4 | (0,+1,0), y = 1 | +0.5 | +1 |
| 5 | (−1,0,0), x = 0 | −1 | +2 |
| 6 | (0,0,+1), z = 1 | −0.75 | +0.5 |

Face areas: 1 on the unit cube; 4, 1, 1, 1, 1, 4 on (0,2)×(−1,1)×(0,½). Both winds have w·n ≠ 0 on **every** face, with opposite sign patterns. With w = 0, B_w = 0 and C = 0, so G1N is identically 0.

## 8. Exact natural-BC OC (`natural-conv-3d`) and symbolic verification (`t10_symbolic.py`: 94/94 PASS)

    R_k(s) = e^(−ks)(2 + cos πs),  R_k' = −k R_k at s = 0, 1;   k_i = w_i/ν
    p = α_u B(t) R_kx(x) R_ky(y) R_kz(z),   B = sin(T − t)
    y = A(t) cos πx cos πy cos πz,           A = cos 2t
    u = −(η/α_u) p
    f = y_t − νΔy + w·∇y − ηu
    y_d = y + (p_t + νΔp + w·∇p)/α_t

The parameters are ν = ¼, α_u = 1e-2, α_t = η = 1, T = 1, on the unit cube. The case is a direct extension of the 2D T7F `natural-conv`.

sympy verifies exactly, for w = 0, wind A, wind B and a symbolic wind:
- the state and adjoint PDE residuals;
- the six state BCs ν∂ₙy = 0;
- the **six adjoint Robin BCs** ν∂ₙp + (w·n)p = 0;
- p(T) = 0 and stationarity.

The literal C++ port matches sympy to ≤ 6e-15 relative at 375 points; the tolerance is relative because k = −8 makes R ~ e⁸.

The boundary checks are N8 and N9. For both winds, max|(w·n)p| > 0 **on every face** (e.g. wind A: 1.52 on x = 0, 9.3e-3 on x = 1, 0.76 on y = 0, …), and so is the wrong-BC residual ν∂ₙp. A boundary-wrong adjoint is therefore detectable on all six faces.

## 9. Algebraic gates (`t10_gates.sh`, `t10_natural_bc.csv`: 24 runs)

The runs cover:
- natural-conv-3d at n = 2, 3, BE/CN, heat / w0 / wA / wB;
- the non-colocated multi-zone layout (2 + 2 boxes plus 2 separated source boxes, natural BC) with winds A and B, BE/CN, n = 4;
- the same layout with the Dirichlet state BC;
- the non-unit domain with boxes and a source.

The outer rtol is 1e-12.

| gate | correct adjoint (max over 24) | ex20-style adjoint (under wind) | wrong natural BC νK − C (under wind) |
|---|---|---|---|
| G1N boundary identity | ≤ 3.5e-14 | — | — |
| G1 orientation x/y/z | ≤ 3.4e-14 | — | — |
| G2 transpose | ≤ 3.5e-17 | 7.3e-4 – 3.3e-2 | 9.7e-6 – 1.2e-2 |
| G3 FD gradient (best) | ≤ 3.0e-11 | 1.6e-2 – 3.2 | 7.7e-3 – 5.7 |
| G4 reduced-Hessian symmetry | ≤ 9.3e-11 | 2.0e-4 – 1.96 | 5.9e-4 – 0.76 |
| G5 outer its / true KKT residual | 2–3 / ≤ 4.3e-11 | — | — |
| G7 gradient consistency | ≤ 2.8e-11 | — | — |
| G6 independent forward march | ≤ 1.8e-13 | — | — |

The largest correct-adjoint values are the wind-B runs, whose fields scale like e⁸ (absolute KKT size). Every other run is at 1e-13 to 1e-15. Stationarity in the exact driver is ≈ 5e-17. Both negative controls fail clearly under every nonzero wind and coincide with the exact adjoint for w = 0, as they must.

## 10. Spatial convergence (natural BC; `t10_convergence.csv`, `figures/t10_natural_space_convergence.png`)

**Exact errors** (CN, N = 64, LU; orders vs the previous n):

| n | wind A: y | p = u | heat: y | p = u | wind B (k = −8): y | p = u |
|---|---|---|---|---|---|---|
| 4 | 1.12e-3 (3.10) | 5.36e-4 (2.85) | 1.09e-3 (3.05) | 7.98e-5 (2.99) | (3.02) | (2.23) |
| 6 | 3.27e-4 (3.04) | 1.61e-4 (2.97) | 3.22e-4 (3.01) | 2.39e-5 (2.98) | (3.44) | (2.63) |
| 8 | 1.38e-4 (3.01) | 6.79e-5 (2.99) | 1.36e-4 (2.99) | 1.04e-5 (2.90) | (3.70) | (2.79) |
| 10 | 7.16e-5 (2.93) | 3.49e-5 (2.99) | 7.09e-5 (2.92) | 5.73e-6 (2.65) | (3.82) | (2.87) |
| 12 | 4.35e-5 (2.73) | 2.03e-5 (2.96) | | | | |

From n = 10 on, the raw y order meets the CN N = 64 **time floor**: the y time error of the same data is 1.69e-5 and the p time error 2.53e-6 (§11). Removing that measured time error in quadrature, e_s = √(e² − e_t²), gives for wind A:

| n | 4 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|
| y | 3.10 | 3.05 | 3.03 | 3.02 | **3.02** |
| p, u | 2.85 | 2.97 | 2.99 | 3.00 | **2.99** |

**Fixed-Δt self-convergence** (N = 16, n = 2, 4, 8 vs n = 16; CN and BE alike):

| field | 2 → 4 | 4 → 8 |
|---|---|---|
| y | 3.04–3.09 | 2.88–2.91 |
| p | 2.84–2.98 | 2.84–2.86 |

To calibrate this measure, `t10_selfconv_metric_check.py` applies **the same metric to the Q2 interpolants of the exact fields**:

| field | interpolant vs exact | interpolant in the self-convergence metric |
|---|---|---|
| y | 3.02, 3.01, 2.99 | 3.01, **2.88** |
| p | 2.79, 2.94, 2.97 | 2.78, **2.81** |

So the discrete OC solution reproduces optimal interpolation behaviour. The sub-3 self-convergence values come from a reference only 2× finer and from the steep e^{4x} profile of p; they are not a discretisation defect. The BE runs rule out the CN start-up layer as the cause.

**Wind B** (k_x = −8, R ~ e^{8x}) is still pre-asymptotic at n = 10: y overshoots (3.4 → 3.8) and p climbs (2.2 → 2.9). No order above 3 is claimed.

**Verdict: Q2 order 3 for y, p, u with the natural BC.**

## 11. Temporal convergence (natural BC; fixed mesh n = 6, reference N = 512 on the same mesh)

The corrected order removes the bias of a reference only 4× finer.

| N | BE y | BE p = u | CN y | CN p = u |
|---|---|---|---|---|
| 8 | 0.71 | 0.20 | 2.08 | 1.61 |
| 16 | 0.88 | 0.46 | 2.21 | 1.84 |
| 32 | 0.96 | 0.72 | 2.17 | 1.95 |
| 64 | 0.99 | 0.85 | 2.07 | 1.97 |
| 128 | **1.00** | **0.92** (still rising) | **2.00** | **1.94** (raw 2.01) |

- **BE y: order 1.** BE p, u approach 1 monotonically: 0.20, 0.46, 0.72, 0.85, 0.92. The adjoint has a large second-order pre-asymptotic component on this steep profile. J differences halve per level (ratios 2.02–2.14).
- **CN: order 2** for y, p, u and J (|ΔJ| ratios 4.0–4.2).
- **CN start-up layer** (§16 of T9): visible but weak here, because the initial data are smooth with homogeneous-Neumann-compatible modes. The first-level difference decays slowly (2.9e-5 → 1.7e-5), but y on [¼, T] converges at 2.13 and y(T) at 2.07. The frozen CN scheme is unchanged.

## 12. Multi-zone (`t10_multizone.csv`, `figures/t10_multizone_space_convergence.png`, `figures/t10_natural_time_convergence.png`)

**Layout:** control = octants (−,−,−) ∪ (+,+,+); tracking = (−,+,−) ∪ (+,−,+), with zero element overlap; sources (amplitude 1) = (−,+,+) ∪ (+,−,−). Natural BC, wind A, T = 1. All region faces lie on x, y, z = ½.

A first multi-zone time ladder was invalid and was repeated. Production runs keep Δt = 0.05 unless `--T` is given, so `--steps` changed the final time. The T10 production ladders use `--T 1`.

**Time** (n = 4, reference N = 512, corrected orders):

| N | BE y | BE p | BE u | CN y | CN p | CN u |
|---|---|---|---|---|---|---|
| 8 | 1.13 | 1.01 | 1.00 | 1.95 | 1.96 | 1.95 |
| 16 | 1.05 | 1.01 | 0.99 | 1.93 | 1.98 | 1.97 |
| 32 | 1.02 | 1.00 | 0.99 | 1.94 | 2.03 | 2.01 |
| 64 | 1.01 | 1.00 | 1.00 | 1.98 | 2.10 | 2.07 |
| 128 | **1.00** | **1.00** | **1.00** | **2.03** | 2.17 | 2.12 |

So BE is 1 and CN is 2. On this rough data the CN start-up layer is visible (first-level difference 3.8e-5 at N = 128), and y on [¼, T] converges at 2.04–2.08.

**Space** (fixed Δt, CN N = 8, n = 2, 4, 8 vs n = 16). The difference is split into a **band** (nodes within 1/8 of a region face, i.e. one coarse element at n = 8) and the **interior**:

| field | total order (4, 8) | band (4, 8) | interior (4, 8) | share of the n = 8 error in the band |
|---|---|---|---|---|
| y | 2.58, 2.32 | 2.54, 2.11 | 2.65, **3.22** | 1.40e-4 of 1.47e-4 |
| p | 2.59, 2.71 | 2.74, 2.63 | 2.41, 2.78 | — |
| u | 0.70, 0.93 | 0.68, 0.91 | — | **all** (interior 1.5e-4 vs total 5.4e-2) |

**Interface limitation, not an implementation error.**
- The exact control u = −(η/α_u)p·χ_C **jumps** across every control-box face. A continuous Q2 control smears that jump over one element layer, so u converges at O(h^{½–1}) in L², entirely inside the band.
- The state and adjoint inherit reduced regularity only near the region faces: χ_C u and χ_ω(y − y_d) are discontinuous sources. Away from them, y converges at 3.2.
- **Evidence it is regularity, not a bug:**
  - the same layouts pass every algebraic gate to round-off (§9);
  - the reduction is localised to the band;
  - the interior order is 3;
  - the natural exact case without masks converges at 3 (§10).

Global order 3 is not required (or expected) with discontinuous masks.

## 13. Multiple sources (`t10_sources.sh/.py`, `t10_sources.csv`: all PASS)

Setup: natural BC, wind A, non-colocated multi-zone control/tracking, two separated source octants, n = 4, N = 16, outer rtol 1e-12, BE and CN. With Z the zero-source solution and d(X) = X − Z (the optimality system is affine in the source):

| identity | BE | CN |
|---|---|---|
| superposition d(SAB) = d(SA) + d(SB) | 1.4e-12 | 1.1e-12 |
| remove-one SAB − SA = SB − S0, SAB − SB = SA − S0 | 7.5e-12, 1.7e-12 | 5.9e-12, 1.4e-12 |
| amplitude d(2·SAB) = 2 d(SAB) | 5.3e-13 | 1.9e-12 |
| overlap: two overlapping boxes = their explicit union | **0 (bitwise)** | **0 (bitwise)** |
| whole domain: no box = one full-domain box | **0 (bitwise)** | **0 (bitwise)** |
| each contributes: max \|d(SA)\|, \|d(SB)\| | 0.73, 0.26 | 0.68, 0.26 |
| distinguishable: \|d(SA) − d(SB)\| / \|d(SA)\| | 0.995 | 0.996 |
| centroid of \|d(SA)·y\| / \|d(SB)·y\| | (0.39, 0.70, 0.64) / (0.76, 0.32, 0.26) | same |

The response centroids lie on each source's side. The `--verify-convection` runs of SA, SB and SAB (N = 4) give a true KKT residual ≤ 4.8e-16, gradient consistency ≤ 3.1e-13 and an independent forward march ≤ 8.6e-16, under both schemes.

Semantics carried over from T7F: one source field over the union; overlaps counted once; an empty list means the whole domain. Per-region amplitudes are **not** a feature.

## 14. Final 3D whitelist and refusals (`t10_refusals.sh`, `logs/refusals.log`: 40/40)

| accepted in 3D | refused in 3D (explicit `invalid_argument`) |
|---|---|
| distributed control: full domain (`--control-full-domain`), box (`--control-box3d`), multi-zone (`--control-box3d-add`) | Neumann and Dirichlet boundary control |
| tracking: full domain, box (`--tracking-box3d`), multi-zone (`--tracking-box3d-add`) | H1-space / H1-time regularisation |
| source regions (`--source-box3d-add`; none = whole domain) | boundary-reduced, Dirichlet-reduced and matching solvers |
| homogeneous Dirichlet and **natural** state BC | variable / named winds; LPS |
| constant wind (vx, vy, vz) or none | 1D/2D presets and exact cases (Math2025, ErrorFeedback, lift-exp, conv-sin3, …), including combined with `--dimension 3` |
| rectangular HEX27 domains (`--domain3d`) | the 2D region syntax (`--domain`, `--*-box-add`, `--source-box-add`, ambiguous 4-number boxes) |
| BE / CN; L2 control | an implicit (2D-default) tracking or control region |
| SCHUR_SWEEP (BE and CN); ADDITIVE_GAMG (BE only) | CN without `--pc-schur-sweep` |
| exact cases conv-sin-3d, conv-sin3-3d, natural-conv-3d | region options together with a 3D exact case; boxes outside the domain or empty; invalid `--domain3d` |
| | 3D options in 2D; `--vz` / `--ny` in 2D; tetrahedra / wedges / unstructured meshes (no CLI path exists: the mesh is always the HEX27 box) |

The accepted lines (8, all converged): the T9 production command line, natural BC CN, multi-zone, sources, non-unit domain, ADDITIVE_GAMG BE, and natural-conv-3d / conv-sin3-3d.

## 15. Non-unit rectangular domain (§18 of the task)

The domain (0,2)×(−1,1)×(0,½) is checked at several levels:
- **HEX27 geometry, volume and selection** (§5): extents, DOFs, volume 2, half-domain boxes of volume 1, centroid rule.
- **Dirichlet boundary set** (§5).
- **Convection assembly:** the G1N identity 2.8e-14 with face areas 4/1/1/1/1/4, and orientation ≤ 2.6e-14.
- **Natural-BC gates** (§9): FD 3.5e-15, Hessian 6e-16, true residual ≤ 6.5e-15, march ≤ 5.6e-15.
- **Production solves:** Dirichlet (refusal battery, accepted) and natural with MPI np = 1/2/4 (§17).

The phase-1 exact benchmarks are intentionally not rescaled to the box.

## 16. Solver policy (`t10_backends.sh`)

All correctness and convergence studies use SCHUR_SWEEP with LU spatial solves (MUMPS LU in MPI).

Iterative-backend check on the multi-zone natural case at n = 8, N = 16 (Pe_cell ≈ 0.34):

| scheme | backend | outer its | spatial its/solve (max) | fields vs LU | J vs LU | wall |
|---|---|---|---|---|---|---|
| BE | LU | 2 | 1 | — | — | 6.8 s |
| BE | GMRES+GAMG | 2 | 13 | 2.5e-15 | 5e-16 | 19.5 s |
| BE | GMRES+ASM/ILU | 2 | 12 | 2.3e-15 | 4e-16 | 70.9 s |
| CN | LU | 2 | 1 | — | — | 8.2 s |
| CN | GMRES+GAMG | 2 | 11 | 5.9e-15 | 1.9e-15 | 18.9 s |
| CN | GMRES+ASM/ILU | 2 | 10 | 5.4e-15 | 1.5e-15 | 61.2 s |

There was no tuning, and ADDITIVE_GAMG was not re-investigated under CN.

## 17. MPI (`t10_mpi.sh/.py`, `t10_mpi.csv`: 15/15 PASS)

The runs use np = 1, 2, 4 (verified `nrank`) with fields matched by coordinates:
- A: multi-zone + 2 sources, natural BC, BE;
- B: full domain + 2 separated sources, Dirichlet BC, wind B, CN;
- C: natural-conv-3d data, BE and CN;
- D: non-unit domain, natural BC, BE.

Fields agree with np = 1 to ≤ 1.8e-15 relative and J to ≤ 7.9e-16. Outer iteration counts are identical (2), the FGMRES histories agree to ≤ 1.8e-16·r₀, and the true residuals match to the printed digits.

## 18. Memory (`t10_memory.py`, `t10_memory.csv`)

At N ≥ 64 the T10 cases follow the T9 law (MaxRSS ≈ 2.4 KB per space-time unknown) at **0.76–0.98×**:

| case | ratio to the T9 law |
|---|---|
| natural exact, n = 6–12 | 0.90–0.98 |
| natural time ladders | 0.81–0.89 |
| multi-zone time ladder (n = 4, N = 512) | 0.76 |

The fine-mesh / small-N rows (n = 16 with N = 8 or 16: 1.3–1.9×) are a property of **all** 3D runs at small N: the spatial factorisations and operators dominate there. A same-size **T9-type control** (Dirichlet, full domain, no regions) shows it:

| size | T9-type control | T10 cases |
|---|---|---|
| n = 16, N = 8 | 3753 MB | multi-zone 3768 MB |
| n = 16, N = 16 | 5491 MB | natural 5094–5666 MB |

**Boxes, sources and the natural BC do not change the memory law (≤ 3%).** The outer iteration counts are 2–3 everywhere.

## 19. Regression (2D and T9)

**2D** (`t10_regress_2d.sh`, OLD = the T9 build 8ed57d629, private clean build sha d9f96b42; NEW = the final shared build sha ef577253):

| block | result |
|---|---|
| ex20 goldens | **61/61** |
| Math2025 Ex3/Ex4/Ex5 | BYTE-IDENTICAL |
| EF 9.5 | only the known `variant=source` CONFIG label (as in T8D/T9) |
| R4 T0–T2 constant wind | REGRESSION PASS (rerun with python3.11, §20) |
| R5 T3/T4 | 9/9 IDENTICAL |
| old vs new EX20_/EX21_ lines | **55/55 IDENTICAL**, covering T6, T7N, T7D, T7F, the T8D iterative default and verify, Dirichlet heat (all regularisations), Neumann |
| zero-wind ex20/ex21 parity | **34/34 JSON_IDENTICAL**; the 2 boundary-reduced + convection cases are refused identically by both binaries |
| 2D refusals | unchanged |

**T9** (`t10_regress_t9.sh`): **31/31 identical**. That covers the 16 T9 gate runs, the exact driver (conv-sin3-3d, conv-sin-3d, GAMG), the production JSON (T9 command line, ADDITIVE_GAMG BE), and all T9 operator dumps byte-identical (the only new file is the 3D-only `_regions3d.txt`). MPI np = 2 differs from T9 by 3.6e-15, within the run-to-run noise of MUMPS np > 1.

The first T10 build changed the T9 exact runs at 1e-15. A private build of the T10 sources with only `ExactOc3D.cpp` reverted reproduced T9 bitwise, which localised the cause to the compilation of that file. Moving the natural case into separate functions restored bitwise T9 identity.

## 20. Notes on the environment

- Everything ran in the sanitised environment (`env -i … PATH=<petsc arch>/bin:/usr/bin:/bin`), so `mpirun` and `pkg-config` are the PETSc arch's, not miniforge's. The final shared binary has no miniforge in readelf or ldd and is byte-identical to the private clean build of the same sources.
- The sanitised `/usr/bin/python3` is Python 3.6:
  - `t10_geometry.py` uses `stdout=PIPE` instead of `capture_output`;
  - the R4 regression is run with `/usr/bin/python3.11`.
- The T10 campaign (`t10_campaign.sh`) was run on the final binary. Logs from before the ExactOc3D refactor were moved out of the tree.

## 21. Verdict

**3D VOLUME/DATA/NATURAL-BC FEATURE SET: VALIDATED**

| # | criterion | evidence |
|---|---|---|
| 1 | rectangular domains | §5, §15, MPI D |
| 2 | 3D box selection | 224/224, independent centroid rule |
| 3 | control boxes | §5, gates |
| 4 | tracking boxes | §5 |
| 5 | multi-zone union / overlap / precedence | §5 |
| 6 | non-colocated control / tracking | 0 shared elements, gates, BE/CN |
| 7 | source regions | §5 |
| 8 | multiple source regions | §13 |
| 9 | natural BC mathematically correct | §6–§9 |
| 10 | 3D convection boundary identity | ≤ 3.5e-14 |
| 11 | exact natural-BC OC | 94/94 symbolic |
| 12 | natural-BC gates incl. both negative controls | §9 |
| 13 | spatial order 3 | §10 |
| 14 | BE 1 / CN 2 | §11, §12 |
| 15 | MPI np = 1/2/4 | 15/15 |
| 16 | no 2D or T9 regression | §19 |

Documented limitations, none a blocker:
1. Multi-zone convergence is interface-limited: u is discontinuous across the control-box faces, and the reduction is localised to the band.
2. BE p, u reach 0.92 at N = 128 and are still rising (pre-asymptotic).
3. Wind B (k = −8) is pre-asymptotic in space at n ≤ 10.
4. The inherited CN start-up layer of the frozen time scheme is visible on rough data.

## 22. Remaining before 3D boundary control (T11, not started)

- **Face machinery for 2D faces:** QUAD9 face masses on a selected face set (generalising `BuildBoundaryConvectionMatrix` from Σ(w·n) to a per-face M_Γ), and a surface-gradient K_Γ. The current tangential stiffness assumes one-dimensional faces.
- **3D Neumann control:** B_N = M_Γ on a face, with the exact transpose already carrying the Robin adjoint condition.
- **3D Dirichlet control:** lifting, B_D, the reduced Woodbury core (T8D) on face DOFs.
- **H1-space surface regularisation** and H1-time (Dirichlet), the boundary-reduced / Dirichlet-reduced Schur solvers in 3D, and their memory.
- **Per-region source amplitudes**, if wanted (not a feature).
