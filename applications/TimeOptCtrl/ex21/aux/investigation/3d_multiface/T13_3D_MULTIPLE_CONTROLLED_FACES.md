# T13 — Multiple controlled faces in 3D (ex21, HEX27, Neumann and Dirichlet boundary control)

*Baseline:* **120c74442** "Add validated simplex mesh support to ex21" (T12, committed at the start of
this task). Nothing for T13 is committed, staged or pushed.

*Goal:* generalise the validated one-whole-face boundary control (T11N Neumann, T11D Dirichlet L2, T11H1
Dirichlet H1) to a UNION of whole box faces, without changing the validated algebra.

Scripts, logs and results are in `t13/`; the tables are in this directory.

## 1. Audit of the single-face architecture (written before any code change)

**A. Where a single integer face id is assumed.**
- `ProblemConfig::neumann_control_face` (unsigned) and `neumann_control_boundary` (name). They are set by
  `CommandLine.cpp` `ApplyRegions3D` from `--neumann-face3d` / `--dirichlet-face3d` (one name), and by the
  3D exact cases in `ExactOc3D.cpp` `Apply` (zmin).
- `Config.hpp` `Validate3D` (`LookupBoxFace3D(neumann_control_face)`).
- `OwnedDofIsControlled` (one plane).
- `Discretization.cpp`:
  - `g_free_face` (the Dirichlet BC callback leaves ONE face free);
  - the 3D face loop (`!= neumann_control_face`);
  - the area/normal assertion (one face; the normals are summed into one accumulator).
- `ex21.cpp` passes the one free face to `InitializeStateSpace`.
- `ExactOc3D.cpp`: the stationarity print lines (`face=%s(id %u)`), the expected C count (2n_x−1)(2n_y−1),
  `IntegrateBoundaryField3D` / `IntegrateBoundaryGradient3D` (one `face_id`).
- 2D paths (`Benchmarks.cpp`, `ExactOcBenchmark.cpp`, `ConvectionVerification.cpp` G1N face loop) are
  2D-only and untouched.

**B. Where M_Γ / K_Γ are assembled.** Only `BuildNativeSpatialOperators` (the 3D face loop): M_Γ (Neumann and
Dirichlet) and K_Γ (Dirichlet). Element-face contributions are ADDED (`add_matrix_blocked`), so a face union
gives Σ_f M_f, Σ_f K_f with shared nodes appearing once in the global numbering — no code change beyond the
membership test.

**C. Where the control mask is created.**
- Production: `ActiveControlDofs` (Neumann: rows of N = M_Γ) and the Dirichlet C = support diag(M_Γ) after
  the elimination (`DirichletControlOperators::BuildPartition`). Both are derived from the assembled
  operator, so they are generic.
- Diagnostic: `OwnedDofIsControlled` (geometric; one face).

**D. Where the edge/corner elimination is assumed.** The Dirichlet policy of T11D is implemented entirely by
the FEMuS BC callback (`StateBoundaryCondition`: every boundary face constrains its nodes, except the free
face) plus `GenerateBdc`'s set-only behaviour: a node shared with any constraining face stays constrained.
For a union, the callback must leave EVERY controlled face free. Then:
- a node on an internal edge between two controlled faces is touched only by free faces, so it is active;
- a node touched by any uncontrolled face is constrained.

That is exactly the requested policy, geometric and independent of processing order.

**E. Exact benchmarks tied to zmin.** `neumann-conv-3d`, `dirichlet-conv-3d`, `dirichlet-h1-3d` (face 1
hard-coded in `Apply`; data functions use the z = 0 normal).

**F. MPI / control-layout assumptions.**
- The Dirichlet reduced layout orders the owned C DOFs by global index (`BuildGlobalControlList`, the
  distributed reduced vector).
- Neumann N is a distributed matrix.

Neither depends on the number of faces.

## 2. Multi-face semantics (fixed before coding)

- ONE control function u on Γ_c = ∪_{f∈F_c} Γ_f (not one control per face). A shared edge or corner node is
  ONE DOF.
- M_Γ = Σ_f M_f and K_Γ = Σ_f K_f (facewise surface integrals added). The shared edge has zero surface
  measure and contributes nothing by itself; its DOFs receive contributions from both adjacent faces.
- ∫_Γ |∇_Γu|² = Σ_f ∫_{Γ_f} |∇_{Γ_f}u|². The surface gradient is facewise; the FE trace is continuous across
  internal edges (same DOFs). No extra edge term and no derivative continuity across the dihedral edge: this
  is the H¹ space of the polyhedral surface (continuous, piecewise H¹ on the faces).
- **Neumann:** ν∂_n y = ηu on Γ_c, ν∂_n y = 0 elsewhere; adjoint ν∂_n p + (w·n)p = 0 on ∂Ω; the control
  space is the continuous trace on the union (shared edges active, no essential conflict).
- **Dirichlet:** y = ηu on Γ_c, y = 0 on ∂Ω \ Γ_c, with u = 0 on ∂Γ_c (the EXTERNAL boundary of the union).
  Edge/corner nodes:
  - touched only by controlled faces: active C DOF (internal edges, corners of three controlled faces);
  - touched by ANY uncontrolled face: constrained to 0.
- **Face sets are canonical.** The list is sorted by FEMuS id and duplicates are refused, so operators and
  results cannot depend on the order the names were given.

## 3. Exact benchmarks (decided before coding)

Smooth manufactured solutions must satisfy the boundary-control formulation including its edge
compatibility conditions.
- **Neumann.** At a 90° edge between face A (controlled, flux g_A) and face B (flux g_B; 0 if uncontrolled),
  C² compatibility is ∂_t g_A = ∂_t g_B along the in-face directions normal to the edge. With
  u = −(η/α_u)p and the Robin adjoint, this requires:
  - the in-face derivative of p normal to every EXTERNAL edge vanishes;
  - at an INTERNAL edge between faces with normals −e_x and −e_z, p_x = p_z there, i.e.
    (w_x − w_z)p = 0 on the edge.

  The adjacent Neumann benchmark therefore uses heat or a wind with w_x = w_z, which keeps u nonzero on
  the shared edge. The case refuses w_x ≠ w_z.
- **Dirichlet, opposite faces** (zmin ∪ zmax): image heat kernels about z = 0 and z = 1, each multiplied by
  a septic cutoff over z ∈ [¼, ¾]; ω = (0,1)² × (¼,¾) is away from both faces.
- **Dirichlet, adjacent faces: no clean closed form.**
  - The adjoint must be homogeneous outside ω, and ω may not touch Γ_c (the discrete tracking acts on free
    DOFs only — T11D). Every separable construction (eigenfunction × image kernel × cutoff) leaves its
    cutoff residual in a slab that touches one of the two controlled faces.
  - A field vanishing on two adjacent faces plus the far faces, with zero final data and homogeneous
    near those faces, has no image-kernel closed form (a heat solution on [0,1] vanishing at both ends
    and at T must be driven from inside).
  - Moreover, for the L2 optimum with smooth p (p = 0 on both faces), u = ην∂_n p/α_u is 0 on the internal
    edge.

  Dirichlet adjacent is therefore validated by the algebra, by fine-reference spatial and temporal
  self-convergence, and by MPI/permutation tests — stated as such, not forced.

## 4. Implementation (what changed; nothing else touched)

Seven files, 2D paths unchanged by construction. Single-face runs are byte-identical to the T12 build (§24).

| file | change |
|---|---|
| `Config.hpp` | `control_face_mask` (bit `1<<id`; 0 = the legacy single `neumann_control_face`); `ControlFaceMask3D`, `IsControlFace3D`, `IsMultiFace3D`; `Validate3D` refuses bits outside 1..6; `OwnedDofIsControlled` loops over the face set (Dirichlet: on a controlled face and on no uncontrolled face) |
| `CommandLine.cpp` | `ParseFaceList3D` (comma list → canonical mask; refuses an empty list, empty names, unknown names, case variants, numeric ids, duplicates); `SetControlFaces3D` (mask, lowest id, canonical names in id order); repeated `--neumann-face3d` / `--dirichlet-face3d` refused; the T13 exact case names registered |
| `Discretization.cpp/.hpp` | Dirichlet BC callback: a face MASK left free (`g_free_face_mask`); 3D face loop: `IsControlFace3D`; area/normal assertion per controlled face; `IntegrateBoundaryField3DFaces` / `IntegrateBoundaryGradient3DFaces` (the single-face integrators forward to them with `1<<id`) |
| `ex21.cpp` | passes `ControlFaceMask3D(config)` to `InitializeStateSpace` (3D Dirichlet) |
| `ConvectionVerification.cpp` | G13N gate (union only): `N = Σ_f M_Γ({f})` from separate per-face assemblies; FD gradient with the true `N`, with one face dropped, with one face counted twice |
| `ExactOc3D.cpp/.hpp` | `neumann-opp-3d`, `neumann-adj-3d`, `dirichlet-opp-3d` (L2 + H1); stationarity / trace lines report the face set and the predicted count of the union; surface errors integrate over all the controlled faces |

The algebra is untouched:
- M_Γ and K_Γ are assembled by the same face loop with the membership test generalised.
- The Neumann N, the Dirichlet C (support of diag M_Γ after elimination), B_D, B_Dᵀ, R_H1t, the matrix-free core and SCHUR_SWEEP all consume the assembled operators.

## 5. CLI and face-set semantics (§3 of the request)

- `--neumann-face3d zmin,xmin`, `--dirichlet-face3d xmax,ymax,zmax`: one control on the union. A single name is the pre-T13 syntax and behaviour (backward compatible, byte-identical, §24).
- **Order independence.** The list is converted to a bitmask, so `xmin,zmin` ≡ `zmin,xmin`. The printed face set is the canonical id order. Permutation invariance is byte-identical (§15).
- **Refused** (§23):
  - an empty list, `,`, `zmin,`, `,zmin`, `zmin,,xmin`;
  - duplicates (adjacent or not, e.g. `zmin,xmin,zmin`), seven entries;
  - unknown names, `ZMIN`, numeric ids `1,3`, a space inside the list;
  - repeating the option, which would read as independent per-face controls.

## 6. Face-set classes and predicted topology (`t13_trace.csv`, 228/228)

Each set was run as Neumann and Dirichlet on the unit cube (3,3,3), and selected sets on the anisotropic (2,3,4) mesh and the non-unit box (0,2)×(−1,1)×(0,½) at (4,4,2). Predicted counts come from lattice enumeration; Dirichlet counts are also checked against hand closed forms (N_a = 2n_a).

| class | example | Neumann: union nodes (n = 3) | Dirichlet: C (n = 3) | Dirichlet closed form |
|---|---|---|---|---|
| one | zmin | 49 | 25 | (N_x−1)(N_y−1) |
| opposite | zmin,zmax | 98 | 50 | 2(N_x−1)(N_y−1) |
| adjacent | zmin,xmin | 91 (= 49 + 49 − 7 shared edge) | 55 (5 internal-edge nodes active) | (N_y−1)(N_x+N_z−1) |
| three mutually adjacent | zmin,ymin,xmin | 127 | 91 (16 = 3·5 internal-edge + 1 corner active) | N_xN_yN_z − (N_x−1)(N_y−1)(N_z−1) |
| U (two opposite + one) | zmin,xmin,xmax | 133 | 85 (10 internal) | — (lattice) |
| five | all but zmax | 193 | 169 | — (lattice) |
| all six | — | 218 | 218 (every boundary node, 68 internal edge/corner) | (N_x+1)(N_y+1)(N_z+1) − (N_x−1)(N_y−1)(N_z−1) |

Checks per case:
- **Neumann:** trace DOFs = lattice prediction; support(M_Γ) = union of the face planes; inclusion–exclusion (shared edges and corners counted once); N_selected = M_Γ.
- **Dirichlet:**
  - |C| = prediction (and the closed form); C = the geometric set node by node;
  - C = support diag(M_Γ); every internal edge/corner node of the union is in C;
  - no node of an uncontrolled face is in C; D (unhatted) = boundary \ C; C ∩ D = ∅;
  - hatted eliminated set = D ∪ C.

## 7. Union face operators (`t13_face_mass.csv`, 312/312)

Independent references: M_ref = Σ_f M1(a_f)⊗M1(b_f) and K_ref = Σ_f [K1⊗M1 + M1⊗K1], from 1D Q2 matrices placed on each face's nodes. Shared edge nodes receive both faces' contributions.

- **Assembled vs reference:** M_Γ = M_ref ≤ 7e-15 and K_Γ = K_ref ≤ 7.5e-15 (rel. max entrywise, all sets and meshes). R = M_ref[C,C] (Dirichlet) is SPD.
- **No double counting:**
  - M_Γ(union) = Σ_f M_Γ(single-face run) to ≤ 5.5e-17 (0 for Neumann);
  - 1ᵀM_Γ1 = Σ_f |Γ_f| exactly (2, 3, 6; non-unit areas);
  - **negative:** a face counted twice gives area 3 ≠ 2 (or 7 ≠ 6).
- **Polynomial integrals:**
  - vᵀM_Γv = Σ_f ∫_f (x²+yz−2z²+x)² dS to ≤ 1e-12 (facewise biquadratic, reproduced exactly);
  - vᵀRv = Σ_f ∫_f φ², φ = Π_{uncontrolled f} dist_f (vanishes exactly where the policy constrains, biquadratic on each controlled face);
  - vᵀK_Γv = Σ_f ∫_f |∇_f φ|² (FD-integrated reference, ≤ 1e-9).

## 8. Neumann gates (`t13_gates_neumann.csv`, 34/34)

`--verify-convection`, SCHUR_SWEEP + LU, outer rtol 1e-12.
- **Exact cases:** neumann-opp-3d (heat / wind A / wind B) and neumann-adj-3d (heat / wind E), n = 2, 3, BE/CN.
- **Production unions:** xmin,xmax; zmin,xmin; ymin,zmax; xmin,ymin,zmin; zmin,xmin,xmax; all six; non-unit zmin,ymax.

Positives (worst over the 34 runs):

| quantity | value |
|---|---|
| G1 orientation, G2 shell transpose, Hessian symmetry | ≤ 1e-12 |
| G1N boundary identity C+Cᵀ = Σ_f (w·n_f)M_f | ≤ 3.1e-14 |
| FD gradient (shell) | ≤ 1.8e-12 |
| G13N union identity N = Σ_f M_Γ(f) | ≤ 1.4e-17 |
| G13N FD with the true union | ≤ 1.2e-9 |
| production true residual | ≤ 3.8e-12 |

Negatives:
- **G13N, every run:**
  - **drop one face:** FD rel 7.4e-3 … 9.7;
  - **duplicate one face:** the same magnitude; the error is ±M_f(dt α_u U + ηΛ), hence symmetric.
- **With wind:**
  - **wrong adjoint BC** (ν K − C, i.e. ν∂_n p = 0): FD ≥ 0.125, Hessian asymmetry, transpose defect;
  - **ex20-style untransposed adjoint:** detected.
- **Heat:** the ex20-style adjoint coincides (L symmetric), as expected.

Criteria were fixed before the table (script docstring): direct > 1e-6, FD > 1e-4, Hessian > 1e-8.

## 9. Dirichlet B_D / B_Dᵀ (`t13_gates_dirichlet_bd.csv`, 150/150; `t13_gates_dirichlet.csv`, 104/104)

**Explicit dumps (`t13_bd.py`).** B_D and B_Dᵀ are dumped column by column from ApplyBD/ApplyBDT. They are compared with the BE/CN formulas rebuilt from the unhatted M, L and the GEOMETRIC C:
- dumped = formula and B_Dᵀ = formulaᵀ = (dumped B_D)ᵀ, all ≤ 1e-13;
- the dumped C equals the geometric C.

Faces tested: opposite, adjacent, three, U and all six, heat/wind, BE/CN.

Negatives, all detected:
- **wrong C, single-face edge policy** (internal edges constrained): 1.0 … 1.5;
- **wrong C, one face omitted:** 1.0 … 1.5;
- **L instead of Lᵀ in B_Dᵀ:** 0.35 under wind; it coincides under heat, as expected;
- **CN coupling p_{n−1} instead of p_{n+1}:** detected.

G1D on the union: C + Cᵀ = Σ_{f∈F}(w·n_f)M_f on I∪C ≤ 1e-13, built from single-face dumps. With one face dropped from the sum it fails (0.37 … 1.1).

**Gate battery (`--verify-convection`).**
- Exact dirichlet-opp-3d at n = 4, 8: L2, H1s, H1t and comb; heat, wind N, wind A.
- Eight production unions at n = 3/4 (opposite, two adjacent pairs, three, U, five, all six, non-unit), with all four regularisations, BE/CN.

Worst positives:

| quantity | value |
|---|---|
| G3D B_Dᵀ | 1.9e-17 |
| G5D Hessian symmetry | 3.7e-14 |
| G4D FD | 4.7e-13 |
| true residual | 8.1e-11 |
| kkt_state_rel | 8.1e-11 |
| \|grad J\| | 2.8e-9 |
| core failures | 0 |

Negatives: the legacy B_Dᵀ and the untransposed E are detected under every wind.

**One criterion revision, documented in the table script.** The inherited T11H1 bar for the DIRECT transpose detectors (> 1e-4) was calibrated on n ≤ 4.
- **dirichlet-opp-3d at n = 8:** the direct detectors read 4.0e-6 … 1.6e-3, while the positives at n = 8 are ≤ 1.3e-17, a separation of 11 orders. The derived FD negatives stay 3e-4 … 65.
- **Cause:** these detectors measure the skew part relative to the whole space-time operator on random vectors, and that share shrinks as the diffusion grows with n.
- **New bar:** T13 uses > 1e-6 (the T13 Neumann bar).

## 10. Reduced core: SPD, dense vs matrix-free (`t13_core_equivalence.csv`, 32/32)

Coverage: 8 cases (6 production unions incl. all six and non-unit, 2 exact dirichlet-opp-3d) × {L2, H1s, H1t, comb}.

| quantity | result |
|---|---|
| dense K = R_H1t + βH | symmetric (‖K−Kᵀ‖ = 0); λ_min > 0 (7e-4 … 0.14); cond(K) 12 … 1340 (H1t largest) |
| shell action vs dense | ≤ 1.3e-14 on 9 probes |
| sparse R_T vs dense | 0 |
| iterative core rtol 1e-12 vs dense | fields and J ≤ 1.6e-14 |
| iterative core rtol 1e-8 vs dense | ≤ 4.1e-10 |
| CG iterations | 4–13 |
| outer iterations | identical |

The union only changes nC = |C| (77 … 386). Dense stays the gold reference.

## 11. Exact benchmarks (§3; closed forms in `ExactOc3D.cpp`, symbolic check §12)

| case | Γ_c | construction | restriction |
|---|---|---|---|
| `neumann-opp-3d` | zmin ∪ zmax | T11N p (Robin on all faces), y with a corrector per face (H0 = z − z²/2, H1 = z²/2) | none (any constant wind) |
| `neumann-adj-3d` | zmin ∪ xmin | p = α_u B F_k(x) X(y) F_k(z), F_k = e^{−ks}(1−s)²(1+2s); y = A cos cos cos + κ X(y) G(x,z), G = F(x)H0(z) + H0(x)F(z) − F′(0)H0(x)H0(z) | w_x = w_z (refused otherwise) |
| `dirichlet-opp-3d` | zmin ∪ zmax | p = X X [(1−ψ)Qs0 + ψQs1], ψ septic on [¼,¾]; u_zmax = e^{−w_z/2ν} u_zmin; L2 AND H1-space/time | n_z multiple of 4 (ω resolved by whole elements; n_z = 2 puts ω's edges on centroids → empty region) |

- **Dirichlet adjacent has no closed form (§3).** Its validation is algebraic (§6–10), plus fine-reference self-convergence with attribution (§13), MPI and permutation.
- **neumann-adj-3d** keeps a NONZERO control on the internal edge (|u| = 1.29 there; M9 in §12), so the edge DOF is really exercised.

## 12. Symbolic verification (`t13_symbolic.csv`, 166/166; `t13_symbolic.py`)

**Neumann** (opp: symbolic wind, heat, wind A; adj: symbolic wind (a,b,a), heat, (0.6,−0.4,0.6)):
- **PDE residuals:** state and adjoint residuals vanish.
- **Boundary conditions:**
  - ν∂_n y = ηu on EVERY controlled face, 0 on every uncontrolled face;
  - the adjoint Robin condition holds on all six faces.
- **Time and stationarity:** p(T) = 0; stationarity on every controlled face.
- **Edge compatibility on all 12 edges** (∂_{n_B} g_A = ∂_{n_A} g_B).
- **Negatives:**
  - w_x ≠ w_z breaks the internal-edge compatibility (27, the reason for the refusal);
  - removing the zmax corrector breaks the zmax flux.

**Dirichlet** (heat, wind N, wind A):
- **Kernels:** Qs0 and Qs1 solve the homogeneous backward adjoint (residual/p = 0 on a grid); Qs0(0) = 0, Qs1(1) = 0.
- **Outside ω:** the cut-off adjoint residual vanishes for z ≤ ¼ and z ≥ ¾.
- **Face relation:** ∂_zQs1(1) = −e^{−a}∂_zQs0(0) exactly. Hence u_zmax = e^{−a}u_zmin satisfies the L2 stationarity on zmax.
- **Negative:** u_zmax = u_zmin fails under w_z ≠ 0.
- **H1:** S = sin sin is a facewise surface eigenfunction (−Δ_Γ S = 2π²S) vanishing on the perimeter of each face. Since the union is two DISJOINT faces, the T11H1 temporal closed form applies on each face up to the factor e^{−a}.
- **State and final time:** y = ηu on both faces and 0 on the four lateral faces; |Qs| at τ = 1e-4 ≤ 5e-172.

The C++ was validated by optimal convergence (§13): a transcription error in f, y_d, y₀ or u would stall it.

## 13. Spatial convergence (`t13_space_convergence.csv`, `figures/t13_space_convergence.png`)

Errors: y and p in L2(Ω)×L2(0,T); u in L2(Γ_c)×L2(0,T) with ‖·‖²_{Γ_c} = Σ_f ‖·‖²_{Γ_f}. All runs are exact-case CN.
- **Neumann:** N = 64.
- **Dirichlet:** N = 256 (time error ≲ 1e-4, below the spatial errors shown).

**Neumann exact** (orders between consecutive n = 2, 4, 6, 8, 10):

| ladder | y | p | u (Γ_c = Σ_f) |
|---|---|---|---|
| opp, wind A | 3.58, 3.02, 3.00, **3.00** | 4.19, 3.61, 3.37, 3.22 | 3.27, 3.11, 3.02, **2.99** |
| opp, heat | 3.71, 3.06, 3.03, **3.02** | 4.21, 3.70, 3.70, 3.62 | 4.08, 3.41, 3.32, **3.22** |
| adj, wind E (0.6,−0.4,0.6) | 3.69, 3.05, 3.02, **3.01** | 4.52, 3.99, 3.99, 4.00 | 4.37, 3.71, 3.64, **3.54** |
| adj, heat | 3.65, 3.03, 3.01, **3.00** | 4.23, 3.80, 3.86, 3.81 | 4.24, 3.57, 3.50, **3.40** |

The forward march is ≤ 3e-14 and stationarity ≤ 4e-17 at every n. **Verdict: Q2 order 3 for y and for u on the union.** u approaches 3 from above, as in T11N; p is superconvergent on these smooth cases, which is not claimed.

**Dirichlet exact, dirichlet-opp-3d** (n = 4, 8, 12, multiples of 4; orders 4→8, 8→12):

| ladder | y | p | u (Γ_c = Σ_f) | \|u\|_{H¹(Γ_c)} |
|---|---|---|---|---|
| L2, wind N | 2.09, **3.28** | 3.88, 2.67 | 2.15, **3.28** | — |
| L2, heat | 2.21, **3.36** | 3.95, 2.87 | 2.17, **3.36** | — |
| H1-space ℓ = 0.3, wind N | 2.01, **3.30** | 3.90, 2.98 | 2.03, **3.29** | 2.02, **3.22** |
| H1-time γ = 1e-2, wind N | 1.18, **3.27** | 3.81, 3.21 | 0.64, **3.27** | 0.70, **3.24** |
| combined, wind N | 1.43, **3.27** | 3.71, 3.30 | 0.71, **3.27** | 0.81, **3.21** |

- **The n = 4 → 8 step is pre-asymptotic** in every regularisation: the same ≈ 2 for L2, so it is a property of the benchmark data, not of the H1 terms. The order is ≥ 3 at the finest pair.
- **Relative errors** are 1.8 % (n = 8) and 0.5 % (n = 12). This benchmark is sharper than T11D's, because its two image kernels feed the blending window [¼,¾].
- **Combined H1:** the same plateau then recovery is what T11H1 documented for the single-face H1-time benchmark (memory note: u plateau at n = 6–8, then 3.2–3.6). Q2 H1 seminorm orders above 2 at n = 12 mean the asymptotic regime is not yet reached.
- **Settled independently below:** fixed-dt self-convergence on the SAME exact data against an n = 24 reference.

**Fixed-dt spatial self-convergence on dirichlet-opp-3d** (CN N = 16; time error cancels; n = 4, 8, 12 vs n_ref = 24):

| data | y (n = 4, 8, 12), orders | p, orders | u (Γ_c = Σ_f), orders |
|---|---|---|---|
| L2, wind N | 2.83e-2, 6.55e-3, 1.64e-3; **2.11, 3.41** | 2.17e-3, 1.52e-4, 5.15e-5; 3.84, 2.66 | 8.27e-2, 1.84e-2, 4.59e-3; **2.17, 3.42** |
| combined H1, wind N | 6.41e-3, 2.43e-3, 6.25e-4; **1.40, 3.35** | 2.58e-3, 2.00e-4, 5.38e-5; 3.69, 3.25 | 1.02e-2, 6.38e-3, 1.62e-3; **0.68, 3.38** |

- **Same shape as the exact errors:** the self-convergence errors reproduce the exact errors in size and shape (L2 y: 6.55e-3 vs exact 6.69e-3 at n = 8; 1.64e-3 vs 1.77e-3 at n = 12).
- **Pre-asymptotic, not a time or data artefact:** the n = 4 → 8 dip lies in the semi-discrete spatial error itself.
- **The closed form is the discrete limit,** which also cross-checks the C++ data.
- **Order 3.3–3.4 at the finest pair** for every field.

**Dirichlet adjacent (no closed form): self-convergence with attribution.**
- **Setup:** production data (wind A, target sin_x_minus_t, tracking (½,1)×(0,1)×(½,1)), CN N = 16, n = 2, 4, 6 vs n_ref = 12 (2× at n = 6, which reads ≈ 2.85 even for optimal Q2, see the memory note).
- **Controls:** the union zmin,xmin is compared with the single faces zmin and xmin on the SAME data.

| run | y (n = 6), order 4→6 | p (n = 6), order | u (n = 6), order |
|---|---|---|---|
| union zmin,xmin, L2 | 3.507e-4, 1.56 | 2.416e-5, 2.47 | 1.182e-4, 3.76 |
| single zmin, L2 | 6.788e-5, 2.06 | 2.417e-5, 2.47 | 5.373e-5, 4.92 |
| single xmin, L2 | 3.444e-4, 1.53 | 2.416e-5, 2.47 | 1.056e-4, 3.14 |
| union zmin,xmin, combined H1 | 3.766e-6, 2.51 | 2.417e-5, 2.47 | 6.814e-6, 3.04 |
| single zmin, combined H1 | 9.436e-7, 3.72 | 2.417e-5, 2.47 | 2.283e-6, 4.71 |
| single xmin, combined H1 | 3.654e-6, 2.37 | 2.417e-5, 2.47 | 6.284e-6, 2.49 |

- **The union's y error is the quadrature sum of the two single-face errors on the same data.** √(3.444² + 0.679²)·10⁻⁴ = 3.51e-4 vs 3.507e-4 measured; with H1, √(3.654² + 0.944²)·10⁻⁶ = 3.77e-6 vs 3.766e-6.
- **Where the error sits:** 90–94 % lies near the xmin face, and only 2 % in the corner region of the internal edge (x, z < ¼). xmin is the INFLOW face of wind A (w·n = −1). The single-face xmin run reads the same low 2×-reference order with the validated T11D path.
- **p:** identical in all six runs (2.416e-5). Its 2.1–2.5 reading belongs to the data (y_d ≠ 0 where the tracking box meets the homogeneous faces) and is unaffected by the control.
- **Verdict:** the union introduces no error of its own. The Dirichlet adjacent spatial behaviour equals that of its single faces, which T11D/T11H1 validated, and the internal edge carries no layer.

## 14. Temporal convergence (`t13_time_convergence.csv`, `figures/t13_time_convergence.png`)

Fixed mesh, reference N = 512 (Neumann, n = 6) or 1024 (Dirichlet, n = 4). Orders are reference-corrected where available; values at N = 64 → 128.

| case | BE y / p / u | CN y / p / u |
|---|---|---|
| neumann-opp-3d, wind A | 0.98 / 0.96 / 0.86 (raw 1.21 / 1.19 / 1.12) | y(T) **2.05**, p **1.99**, u **1.96**; global y 1.50 (start-up layer) |
| neumann-adj-3d, wind E | 0.99 / 0.99 / 0.98 | y(T) **2.03**, p **2.00**, u **1.95**; global y 1.51 |
| dirichlet-opp-3d L2, wind N | 0.95 / 0.96 / 0.91 | **1.82 / 2.11 / 1.96** |
| dirichlet-opp-3d combined H1 | 1.08 / 1.07 / 1.09 (raw) | **2.10 / 2.12 / 2.04** (raw) |
| Dirichlet adjacent production, combined H1 | 1.03 / 0.99 / 1.03 | **2.02 / 2.17 / 2.01** |
| single face zmin, same data, combined H1 | 1.03 / 0.99 / 1.03 | 2.02 / 2.17 / 2.01 |
| Dirichlet adjacent production, L2 | 0.85 / 0.99 / 0.90 | 0.81 / 1.85 / 0.69 |
| single face zmin, same data, L2 | 0.79 / 0.99 / 0.77 | 0.78 / 2.17 / 0.79 |

- **BE order 1 and CN order 2** hold for every union with compatible data: both exact Neumann cases, the exact Dirichlet opposite case (L2 and H1), and the production Dirichlet adjacent case with H1-time.
- **Neumann CN global y** reads 1.3–1.6. This is the inherited CN state start-up layer of boundary-flux control, documented in T11N §10 with the same numbers (1.79, 1.44, 1.26, 1.25, 1.49). y(T) and J converge at 2 (|ΔJ| ratios ≈ 4).
- **Production Dirichlet L2 under CN** is order-reduced IDENTICALLY on one face and on the union. With γ_u = 0 the optimal u jumps from u₀ = y₀/η = 0 at t = 0, which is a data incompatibility (memory note from T7D). H1-time removes it and restores order 2 on the union exactly as on one face.

## 15. Permutation invariance (`t13_permutation.csv`, 96/96 BYTE-IDENTICAL)

Every face set given in several orders was compared with its first order, byte for byte:
- every EX20_/EX21_ line of a `--verify-convection` run (gates);
- every line of a production solve;
- the JSON dump;
- every operator dump file (9 Neumann / 14 Dirichlet files: M_Γ, N, C, B_D, B_Dᵀ, K_Γ, R_T, …).

Timing fields and the JSON output path are stripped; they differ by construction. Sets covered:
- **Neumann:** opposite, adjacent, three (all 6 orders), all six (3 orders).
- **Dirichlet:**
  - opposite and adjacent (BE and CN), three (6 orders), U (3 orders), all six;
  - adjacent with combined H1 (CN), three with H1-space, adjacent with the dense core.

Byte identity holds because the list is canonicalised to a mask before anything is built.

## 16. MPI np = 1/2/4 (`t13_mpi.csv`, 27/27)

Required cases plus extras, at n = 4, N = 8.

| kind | cases |
|---|---|
| Neumann (SCHUR_SWEEP, LU / MUMPS LU, outer rtol 1e-10) | opposite (exact, BE), adjacent (exact, CN), production adjacent (BE), three faces (CN) |
| Dirichlet (reduced, core rtol 1e-12, outer rtol 1e-14) | adjacent CN, opposite BE (exact), three faces CN, adjacent combined H1 CN, all six BE |

Results:
- **Fields, J, outer:** fields agree to ≤ 3.2e-15 (y, p, u by sorted coordinates); J to ≤ 1.5e-15; the outer count is identical.
- **Trace set:** at every np it equals the PREDICTED set of the union (162, 153, 217 Neumann; 105, 98, 169, 386 Dirichlet). u = 0 elsewhere and y_C = ηu_C exactly.
- **Outer histories:** within the T11N / T11D criteria.

**One criterion revision (Dirichlet core count; documented in `t13_mpi.py`).** Dopp_be needed 14 core CG iterations at np = 2 vs 13 at np = 1. The core monitors (`logs/mpi_core_diag`) show:
- **first core solve** (the physical right-hand side): 12 iterations at both np, with matching residuals;
- **second core solve:** its right-hand side is the outer residual after step 1, i.e. round-off (r1/r0 = 6.3e-11 vs 6.1e-11). CG on that noise crosses rtol 1e-12 at 13 vs 14.

This is the same mechanism as the T11D outer-history note. The criterion is now "core count within ±1".

## 17. Memory (`t13_memory.csv`, `figures/t13_memory.png`)

Dirichlet setup: n = 4, CN, wind A. The union grows |C| from 49 (one face) to 386 (all six), and m = Nt|C| with it.

| face set | nC | dense RSS / excess (Nt = 32) | measured/predicted 12 B m² + 8 B Nt N\|C\| | iterative RSS Nt = 32 / 128 / 256 |
|---|---|---|---|---|
| zmin | 49 | 155 MB / 43 MB | — (small) | 112 / 305 / 563 MB |
| zmin,zmax | 98 | 250 / 137 | 1.05 | 113 / 309 / 568 |
| zmin,xmin | 105 | 270 / 157 | 1.06 | 113 / 310 / 570 |
| zmin,ymin,xmin | 169 | 490 / 377 | 1.03 | 113 / 312 / 578 |
| all six | 386 | 1969 / 1851 | 1.02 | 119 / 334 / — (Nt = 64: 190) |

- **Dense core:** it grows like m², so all six faces at Nt = 128 would need 28 GB.
- **Iterative core:** its RSS is independent of the union, within 6 %, and linear in Nt. Bytes per space-time DOF are 1580–2670 for every set, with no O(m²) object.
- **Neumann:** RSS is unchanged by the union: 165 → 167 MB (Nt = 32) and 285 → 289 MB (Nt = 64), from one face (81 trace nodes) to all six (386). N = M_Γ is a sparse spatial matrix; the union only adds nonzeros.

## 18. H1 on a union: semantics and surface operators (`t13_h1_surface.csv`, 286/286)

**Semantics.** ∫_Γ|∇_Γu|² = Σ_f ∫_{Γ_f}|∇_{Γ_f}u|². The FE trace is continuous across internal edges (one DOF), the gradient is facewise, and there is no edge term. H1-time is (γ_u/2)∫‖u_t‖²_{M_Γ} with the union mass. This is the H¹ of the polyhedral surface; no derivative continuity across the dihedral edge is imposed or implied.

Checks: dumps with ℓ = 0.3, γ_u = 1e-2.
- **Sets and meshes:** opposite, two adjacent pairs, three, U, five and all six. Meshes: unit (3,3,3) BE, anisotropic (4,3,5) CN, non-unit (4,4,4) CN.
- **Union operator:** K_union = Σ_f[K1⊗M1 + M1⊗K1] ≤ 1e-14, and K_union = Σ_f K(single-face run) ≤ 1e-15. It is symmetric and PSD with K1 = 0.
- **Kernel = number of connected components of the union:** 2 for an opposite pair (constants on each face), 1 for every set with two axes (the shared edge DOFs couple the faces).
- **Quadratic forms:** vᵀKv = Σ_f ∫_f|∇_f v|² for a linear and a facewise-biquadratic v, ≤ 1e-12.
- **Restriction to C:** K|_C = K_face|_C exactly, and K|_C is SPD (H¹₀ of the union).
- **Closed surface (all six):** K|_C keeps exactly the constants (dimension 1), while R = M + ℓ²K is SPD for ℓ ∈ {0, 0.3, 3}.
- **Sparse R_H1t:** equals Ŵ⊗R + (γ/(α dt²))T⊗M rebuilt here (≤ 1e-14) and is SPD.
- **Negatives:** a dropped face, the one-direction tangent, R instead of M in the time term, the wrong sign, a missing 1/dt, and T_NN = 2. All are detected.

## 19. H1 gates, cores, MPI, permutation

- **Gates (`t13_h1_gates.csv`, 76/76):** every Dirichlet gate run with H1s / H1t / comb, i.e. exact dirichlet-opp-3d n = 4, 8 and the eight production unions. Positives and negatives are as in §9.
- **Core equivalence:** 24/24 of §10 are H1 rows (cond(K) up to 1340 for H1t, CG ≤ 13).
- **MPI:** adjacent combined CN np 1/2/4, fields ≤ 2.6e-15 (§16).
- **Permutation:** byte-identical with H1 (adjacent combined CN, three faces H1-space; §15).
- **All six faces:** combined H1, BE/CN (§21).
- **Normal leak:** the surface-gradient normal leak in the exact runs is ≤ 4e-30 (the surface gradient is tangential on every face).

## 20. H1 convergence and sub-verdict

- **Exact H1 on the union — opposite faces, genuine closed form, no forcing.** The union is two DISJOINT faces. S = sin sin is a facewise eigenfunction of −Δ_Γ with u = 0 on each face's perimeter, so the T11H1 temporal factor applies on each face, with u_zmax = e^{−a}u_zmin (symbolic §12).
  - **Space:** ≥ 3.2 at the finest pair for H1-space, H1-time and combined (§13), plus the n = 24 self-convergence.
  - **Time:** dirichlet-opp-3d combined BE 1.08 / CN 2.04–2.12 (§14).
- **Adjacent faces — no closed form,** and none is forced (§3).
  - Validated by the union operator checks (§18), gates (§19), cores, MPI and permutation.
  - Spatial self-convergence: the union's error is the quadrature sum of its single faces' errors, also with H1 (§13).
  - Time: CN order 2.02 / 2.17 / 2.01 with H1-time, identical to the single-face control (§14).
- **What is NOT validated:** an exact H1 benchmark whose u crosses an internal edge with nonzero surface gradient. The adjacent H1 case is covered by algebra plus self-convergence, not by a closed form.

## 21. All six faces (`t13_allsix.csv`, 6/6)

Γ_c = ∂Ω, n = 4, N = 8, wind A, BE/CN: Neumann, Dirichlet L2, Dirichlet combined H1.
- **Topology:** all 386 boundary nodes are control DOFs, including 84/84 edge and 8/8 corner nodes. u = 0 at every interior node.
- **Dirichlet:** y_C = ηu_C exactly. There is no uncontrolled face, so D = ∅ and C is the whole boundary.
- **Solves:** reason 2; true residual ≤ 6.8e-12; KKT state/adjoint ≤ 6.8e-12.
- **Closed surface (H1):** the constants are in the kernel of K_Γ, yet R = M + ℓ²K is still SPD (§18).
- **Covered elsewhere:** the trace/mass checks (§6–7), gates (§8–9), core equivalence (§10) and MPI (§16).

## 22. Solver policy (unchanged)

- **Neumann:** SCHUR_SWEEP. CN without `--pc-schur-sweep` is refused (ADDITIVE_GAMG is not robust under CN, T7F/T8D).
- **Dirichlet:** SCHUR_DIRICHLET_REDUCED with the matrix-free iterative core (default); the dense core is the reference. Sweep / matching / boundary-reduced are refused.

No solver was tuned for T13.

## 23. Strict refusals (`logs/t13_refusals.log`, 86/86)

Refused, with explicit messages:
- **Malformed face lists** (12 variants × both options, §5).
- **Repeated face option** (independent per-face controls).
- **Partial faces** (`--control-box3d-add` / `--control-box3d` with boundary control).
- **Unsupported with a union:** LPS, variable winds, simplex meshes (both kinds); Neumann H1-space / H1-time; Neumann CN without SWEEP; Dirichlet sweep / matching / boundary-reduced; Dirichlet with natural state BC.
- **Mixed options:** mixed Neumann/Dirichlet face options.
- **Tracking:** a tracking region touching ANY face of the union (first or second face; operator-based check).
- **3D-only syntax in 2D.**
- **Exact cases:** given a face list or a region; neumann-adj-3d with w_x ≠ w_z; dirichlet-opp-3d with n_z = 2 (empty ω) or n_z = 6; dirichlet-opp-3d with sweep, or with H1-space under tangential wind.

Accepted (reason 2):
- every face-set class for both kinds (CN);
- Dirichlet unions with H1-space / H1-time / combined, the dense core, `--verify-dirichlet-core` (dense), the explicit reduced PC;
- non-unit domains;
- the three exact cases (incl. dirichlet-opp-3d with H1);
- the legacy single-face exact cases.

## 24. Regression (`t13/logs/t13_regress_*.log`)

OLD is the T12 build (120c74442, private clean build `t13_old/clean_build_t12.sh`, no miniforge); NEW is T13. All of it ran in the sanitised environment.

| suite | result |
|---|---|
| ex20 goldens | **61/61** identical |
| Math2025 Ex3/4/5, EF 9.5, T0–T4, constant-wind R4 | PASS; the EF diff is only the known `variant=source` label, exactly as in T12 |
| 2D old vs new (T6, T7N, T7D all four regularisations, T7F, T8D dense/iterative/verify, natural-BC, Dirichlet heat, zero-wind parity) | 102 IDENTICAL; JSON 28/28 identical. The 2 boundary-reduced + w0 parity cases are refused by both binaries (no JSON), as in T12 |
| 2D restrictions + T9 3D whitelist | text identical to the T12 run (numbers stripped) |
| T9 (HEX27 distributed exact) | **32/32** |
| T10 | **27/27** |
| T11N single face | **43/43** |
| T11D single face | **101/101** |
| T11H1 single face | **52/52** |
| T12 simplex (gates TRI/QUAD/TET/HEX BE/CN, exact drivers, JSON, operator dumps, simplex refusal texts) | **45/45** |

Single-face behaviour is byte-identical, which also covers operator dumps and JSON. The only user-visible change is the refusal text: "needs one whole face" became "needs whole faces: …[,...]".

The build is independent of Conda: no miniforge/conda token anywhere in the shared build tree (link.txt, flags.make, CMakeCache.txt) or in the RPATH / ldd of the binary. Every build used `env -i` with the PETSc-arch PATH.

## 25. Files

Tables in `aux/investigation/3d_multiface/`:

| file | content |
|---|---|
| `t13_trace.csv` | topology per face set and mesh (228) |
| `t13_face_mass.csv` | union M_Γ / K_Γ / R checks (312) |
| `t13_gates_neumann.csv` | per-run Neumann gates (34) |
| `t13_gates_dirichlet.csv` | per-run Dirichlet gates (104) |
| `t13_gates_dirichlet_bd.csv` | explicit B_D / B_Dᵀ / G1D (150) |
| `t13_core_equivalence.csv` | dense vs matrix-free (32) |
| `t13_space_convergence.csv` | exact ladders + self-convergence |
| `t13_time_convergence.csv` | fixed-mesh ladders |
| `t13_permutation.csv` | byte comparisons (96) |
| `t13_mpi.csv` | np 1/2/4 (27) |
| `t13_memory.csv` | dense vs iterative, Neumann |
| `t13_h1_surface.csv` | H1 surface operators (286) |
| `t13_h1_gates.csv` | H1 subset of the gates (76) |
| `t13_allsix.csv` | Γ_c = ∂Ω smoke (6) |

Scripts in `t13/scripts/`:
- trace, BD and symbolic: `t13_trace.py`, `t13_bd.py`, `t13_symbolic.py`;
- gates: `t13_gates.sh`, `t13_gates_table.py`;
- core: `t13_core_equivalence.py`;
- ladders and convergence: `t13_exact_ladder.sh`, `t13_runs.sh`, `t13_ladder_table.py`, `t13_selfconv.py`, `t13_time.py`, `t13_fields.py`, `t13_convergence.sh`;
- permutation: `t13_permutation.sh`, `t13_permutation.py`;
- MPI: `t13_mpi.sh`, `t13_mpi.py`;
- memory: `t13_memory.py`;
- H1 surface: `t13_h1_surface.py`;
- all six: `t13_allsix.sh`, `t13_allsix.py`;
- refusals: `t13_refusals.sh`;
- regression: `t13_regress_*.sh` (copies of the T12 scripts plus `t13_regress_t12.sh` for simplex invariance);
- figures: `t13_figures.py`.

Logs are in `t13/logs/`; the symbolic table is `t13/t13_symbolic.csv`.

## 26. Figures (`figures/`)

- `t13_multiface_geometry.png`: control-DOF topology at n = 3 for Neumann adjacent, Dirichlet adjacent (internal edge active), Dirichlet three faces (shared corner active) and Dirichlet opposite.
- `t13_space_convergence.png`: exact ladders, y / p / u.
- `t13_time_convergence.png`: BE / CN, u on Γ_c.
- `t13_memory.png`: RSS vs m = Nt|C|, dense vs iterative, with 12 B m².

## 27. Limitations and what was not done

- **Dirichlet adjacent:** no closed form (§3); validated by algebra plus self-convergence with single-face attribution.
- **Exact H1 crossing an internal edge:** not available; the H1 exact benchmark is the opposite pair (§20).
- **dirichlet-opp-3d spatial orders:** pre-asymptotic at n = 4 → 8 in every regularisation; n ≤ 12 is the memory limit at N = 256 (13–14.5 GB), and the self-convergence settles the order.
- **Production sin_x_minus_t data:**
  - Dirichlet L2 under CN is order-reduced (t = 0 incompatibility; identical on one face);
  - the inflow face reads low 2×-reference orders; neither is new in T13.
- **Not ported (refused):** independent per-face controls, partial faces, simplex boundary control, variable 3D wind, 3D LPS, Neumann H1. Variable wind and LPS were not started, as instructed.
- **Criterion revisions:** two, documented in the scripts — Dirichlet direct transpose detectors 1e-4 → 1e-6 (§9), MPI core count ±1 (§16). Both rest on a diagnostic showing the positives and negatives stay orders apart.

## 28. Verdicts

**3D MULTIPLE CONTROLLED FACES (L2): VALIDATED.** Scope: HEX27 Q2, constant wind, BE/CN, MPI. One Neumann or Dirichlet control on any union of whole box faces, from one face to all six:
- topology and union operators are checked node by node and entry by entry against independent references (540 checks);
- Neumann and Dirichlet gates pass, with negatives for a dropped face, a duplicated face, the wrong adjoint BC, a wrong C, L vs Lᵀ and the CN coupling;
- the reduced core is SPD, and dense = matrix-free;
- exact benchmarks: Neumann opposite and adjacent, Dirichlet opposite, all sympy-verified. Spatial order 3; BE order 1 / CN order 2;
- Dirichlet adjacent: validated by algebra and attributed self-convergence;
- permutation invariance is byte-identical; MPI np 1/2/4 agrees; memory is linear with the matrix-free core; the all-six smoke test passes; refusals are strict;
- single-face T9–T12 is byte-identical.

**3D DIRICHLET MULTI-FACE H1: VALIDATED** (H1-space, H1-time, combined). Coverage:
- the union surface operators (facewise gradient, continuous trace, kernel = connected components, R_H1t);
- 76/76 gates, 24/24 core equivalences, MPI and byte-identical permutation;
- an exact closed-form H1 benchmark on the opposite pair: spatial order ≥ 3.2 at the finest pair, confirmed by an n = 24 self-convergence; CN order 2;
- the adjacent pair by self-convergence with single-face attribution, and CN order 2 under H1-time.

Not covered: an exact H1 benchmark whose control crosses an internal edge with nonzero surface gradient.

Neumann H1 stays refused, as before. Variable 3D wind and 3D LPS were not started.
