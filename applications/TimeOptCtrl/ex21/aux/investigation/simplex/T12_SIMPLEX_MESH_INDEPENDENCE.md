# T12 — Simplex / mesh-independence validation (ex21, P2 triangles and P2 tetrahedra)

*Baseline:* **81f32b89e** "Add validated 3D Dirichlet H1 regularisation to ex21" (T11H1, committed at
the start of this task). Nothing for T12 is committed, staged or pushed.

*Question:* is the ex21 all-at-once optimal-control formulation and its solver machinery tied to the
structured tensor-product QUAD9 / HEX27 meshes? T12 solves the SAME exact PDEs on P2 triangles and P2
tetrahedra and checks geometry, operators, algebra, convergence, solvers, MPI and memory.

*Scope (first stage, strict):* distributed control, full-domain control and tracking, homogeneous
Dirichlet state, constant wind, BE/CN, L2 control, SCHUR_SWEEP. Everything else is refused on simplices.

Scripts, logs and results are in `t12/`; the tables are in this directory.

## 1. FEMuS simplex support (audit before any change)

Read: `src/mesh/input/MeshGeneration.cpp` (`BuildBox`), `src/mesh/input/GambitIO.cpp`,
`src/mesh/Elem.hpp` (`ig` face table), `src/fe/ElemType.cpp` and `src/fe/Basis.hpp` (basis families),
`src/quadrature/GaussPoints.cpp`, and in ex21 `Discretization.cpp` (`BuildMesh`, assembly, integrators)
and the call sites of `solution_type`.

| item | FEMuS | consequence for T12 |
|---|---|---|
| element identifiers | geometry types: 0 hex, 1 tet, 2 wedge, 3 quad, 4 tri, 5 line; mesh enums `TRI6`, `TET10` exist | names exist |
| **basis per FE order** | type 1 ("quadratic", `SERENDIPITY`) = `TriQuadratic` (6 nodes) / `TetQuadratic` (10 nodes) = **true P2**; type 2 ("biquadratic", `SECOND`) = `TriBiquadratic` (**7 nodes**: P2 + bubble) / `TetBiquadratic` (**15 nodes**: P2 + face and interior bubbles) | ex21 adds its fields with `LAGRANGE, SECOND` (type 2: QUAD9 / HEX27). On simplices type 2 would be bubble-enriched, NOT P2. **T12 uses type 1 on simplices** (the order is set in one place: `solution_type` is read from the "Ti" field in `ex21.cpp`); QUAD9/HEX27 stay type 2 |
| 2D generator | `BuildBox` supports `TRI6`: each square of the nx × ny grid is split along the same diagonal (2 nx ny triangles), boundary face ids 1 bottom, 2 right, 3 top, 4 left (as QUAD9) | **native**; P2 node set = the (2nx+1)(2ny+1) QUAD9 lattice |
| 3D generator | `BuildBox` supports `HEX27` only; every `TET10` branch is commented out | **no native TET generator** |
| mesh readers | Gambit `.neu` (TET10 = 10-node elements, boundary sets by element/face; vertex map `{0,4,1,6,5,2,7,8,9,3}`, face map identity); MED | a runtime-generated `.neu` file is read by the FEMuS-native reader; FEMuS adds the biquadratic nodes itself |
| tet faces (`ig`) | f0 {0,2,1}, f1 {0,1,3}, f2 {1,2,3}, f3 {2,0,3}; mid-edge 4..9 = (0,1),(1,2),(2,0),(0,3),(1,3),(2,3) | used by the generator for the boundary sets |
| quadrature | Gauss–Legendre tri and tet tables up to order 37; ex21's "seventh" exists for both | no change |
| Jacobian / P2 basis | `elem_type_2D/3D::Jacobian` is generic per geometry | ex21 assembly and integrators already take `GetElementType(element)` and `_finiteElement[geometry][solution_type]` — element-generic |
| boundary faces | tri faces = EDGE3 (type 1: 3 nodes), tet faces = TRI6 (type 1) / TRI7 (type 2) | needed only for boundary control (refused; §R) |
| partitioning, VTU | generic (METIS / FEMuS partition, writer per element type) | MPI in §P |
| coordinates | always stored at type 2 (`GetSolutionDof(node, el, 2)`) | the 10 coordinate look-ups in ex21 stay valid for the first 6/10 local nodes |
| preallocation | ex21 rows: 64 (2D), 125 (3D) | a P2 vertex couples ≤ 25 (TRI6) / ≤ 125 (Kuhn TET10, 5×5×5 lattice block) — sufficient |

**Dependency map (what ex21 touches):** `BuildMesh` (mesh source) → `InitializeStateSpace` (FE order) →
`BuildNativeSpatialOperators` (generic element loop: M, K, C, M_track, N) → space-time operators, KKT,
SCHUR_SWEEP (algebraic, mesh-independent) → exact drivers (`ExactOcBenchmark` 2D, `ExactOc3D` 3D:
generic integrators). Only the first two need a simplex branch. Boundary-control face loops, region
boxes (element centroids), LPS and variable winds are NOT reached in the T12 scope and are refused.

## 2. Mesh source and element choice (decided before coding)

- **2D TRI (P2, 6 nodes):** FEMuS-native `GenerateCoarseBoxMesh(nx, nz, 0, …, TRI6)` (option A).
- **3D TET (P2, 10 nodes):** option B — a deterministic **Kuhn (Freudenthal) subdivision** of the same
  nx × ny × nz box: every cube is split along its main diagonal into the 6 tetrahedra
  v000 → v000+e_a → v000+e_a+e_b → v111 (all permutations (a,b,c)). Every cube has the same
  orientation, so the mesh is conforming. All P2 nodes (edge midpoints of cube edges, face centres
  on the face diagonals, cube centres on the main diagonal) are points of the HEX27 lattice
  (2nx+1)(2ny+1)(2nz+1), and every lattice point is used. It is written by rank 0 as a Gambit
  `.neu` file in `output/` and read with the FEMuS reader (`ReadCoarseMesh`, "seventh"),
  with boundary sets 1 zmin, 2 ymin, 3 xmax, 4 ymax, 5 xmin, 6 zmax (the HEX27 face numbering).
  Deterministic, reproducible, refinement controlled by `--nx --ny --nz`, no external tool.
- **FE order on simplices:** type 1 (`SERENDIPITY` in FEMuS's enum) = true P2. QUAD9/HEX27 keep type 2.

**Consequence (useful for the comparison):** at equal (nx, ny, nz) the P2 simplex mesh and the Q2
tensor mesh have the **identical spatial DOF set** (same lattice); only the elements and the basis
differ (2D: 2× the elements; 3D: 6× the elements).

## 3. Scope, CLI and refusals

`--mesh-type quad|tri` (2D), `--mesh-type hex|tet` (3D); default unchanged (quad / hex); invalid
dimension/type combinations refused. Simplex runs refuse every feature outside §scope (Neumann and
Dirichlet boundary control, H1, region/source boxes, natural BC, variable wind, LPS, non-SWEEP KKT
preconditioners, exact cases other than the distributed smooth ones).

## 4. Implementation

| File | Change |
|---|---|
| `Config.hpp` | `MeshConfig::ElementType {DEFAULT, QUAD, TRI, HEX, TET}` (DEFAULT = the historical mesh); `IsSimplex()`; `ValidateSimplex()` (the scope of §3); dimension/type compatibility in `Validate`. |
| `CommandLine.cpp` | `--mesh-type quad\|tri\|hex\|tet`; the simplex run-mode refusals (presets, literature / manufactured / ex20 verification batteries, exact cases other than conv-sin3, conv-sin, conv-sin-3d, conv-sin3-3d, non-SWEEP KKT preconditioners). |
| `Discretization.cpp/.hpp` | `BuildMesh`: TRI via FEMuS `GenerateCoarseBoxMesh(…, TRI6)`; TET via `WriteKuhnTetMesh` (deterministic Gambit `.neu` by rank 0, `MPI_Barrier`, FEMuS `ReadCoarseMesh`). `InitializeStateSpace(…, simplex)`: FE order type 1 (`SERENDIPITY` = P2) on simplices. `EX21_MESH` diagnostics line and a dump-only hook for the pre-elimination volume M and L — both only with an explicit `--mesh-type`, so default runs print and dump exactly as before. |
| `ex21.cpp` | passes `config.IsSimplex()` to `InitializeStateSpace`. |
| `Verification.cpp` | with an explicit `--mesh-type`, the dump also writes `_M_full`, `_L_full` (pre-elimination). |

Nothing in the assembly, the KKT system, SCHUR_SWEEP, the gates or the exact drivers changed: they were
already element-generic (§1). QUAD9/HEX27 code paths are untouched (regression §S).

## 5. Geometry and DOFs (`t12_operators.py` → `t12_geometry.csv`, **133/133 PASS**)

Cases: QUAD and TRI at (2,2), (4,4), (3,5) and on the non-unit (0,2)×(−1,½) at (4,2); HEX and TET at
(2,2,2), (2,3,4) and on the non-unit (0,2)×(−1,1)×(0,½) at (3,2,2). Checks against independently derived
counts:
- elements = n_x n_z (quad), 2 n_x n_z (tri), n_x n_y n_z (hex), **6 n_x n_y n_z (tet)**;
- P2/Q2 DOFs = the lattice ∏(2n_a+1) (every P2 node is a lattice point and vice versa; no duplicates);
- FEMuS type-2 node counts = lattice + n_el (TRI7 bubble) and lattice + faces + n_el (TET15), with
  faces = (4 n_el + 4 Σ n_a n_b)/2 — confirming the type-2 enrichment that §1 avoids;
- coordinates within the box and on the lattice (max deviation 0);
- constrained DOFs = the boundary lattice points (∏(2n+1) − ∏(2n−1)), free = interior;
- |Ω| = 1ᵀM_full 1 to 1e-13 (unit and non-unit boxes);
- partition ownership: MPI §P (every rank combination sees each lattice point exactly once).

## 6. M, K, C operators (`t12_operators.csv`, **266/266 PASS**)

For all four families (worst over cases):

| check | quad | tri | hex | tet |
|---|---|---|---|---|
| M_full symmetric | 0 | 0 | 0 | 0 |
| M_FF SPD λ_min | 1.6e-2 | 2.9e-2 | 1.8e-3 | 3.7e-3 |
| vᵀMw = ∫fg, f,g ∈ {1, x_a, x_a x_b} | 1.6e-15 | 1.3e-14 | 4.9e-15 | 1.7e-14 |
| L_heat symmetric / L·1 = 0 | 0 / 5.9e-16 | 0 / 3.3e-16 | 0 / 9.8e-16 | 0 / 6.8e-16 |
| vᵀLw = ν∫∇f·∇g (quadratics) | 6.1e-15 | 1.6e-14 | 5.2e-14 | 4.2e-14 |
| L_FF SPD λ_min | 0.81 | 0.80 | 0.35 | 0.50 |
| C orientation C·I(x_a) = w_a M1 | 3.0e-14 | 6.0e-14 | 4.6e-14 | 5.7e-14 |
| **NEG** transposed orientation | 2.5 | 3.5 | 2.6 | 3.2 |
| vᵀCg = ∫v (w·∇g) (quadratics) | 1.6e-14 | 1.6e-14 | 2.2e-14 | 4.3e-14 |
| free-subspace skew \|(C+Cᵀ)_FF\|/max\|C\| | 5.0e-14 | 6.7e-14 | 2.8e-14 | 1.5e-13 |
| **NEG** (C−Cᵀ)_FF | 2.0 | 1.5 | 2.0 | 2.0 |

Winds 2D (1,0.5), (−2,1); 3D (1,0.5,−0.75), (−2,1,0.5). C = L(w) − L(0) is a difference of two assembled
operators about 10× larger than C, so the skew bar is 1e-12 (it was 1e-13 in the first run; the tet
values 1.4–1.5e-13 are that subtraction's round-off). The first run also divided moments that vanish
by symmetry on the non-unit boxes by 1e-300; both harness fixes are in the script.

## 7. Heat / zero-wind parity (`t12_parity.sh`, `t12/logs/parity/parity.log`)

TRI n = 6 and TET n = 3: M, K, M_track, M_ctrl_whole, M_ctrl_box, N_selected, M_full, L_full, coords
**18/18 byte-identical**; production solves (BE, CN; LU, rtol 1e-12) fields ≤ 8.0e-16, J identical to
≤ 1.8e-16, same outer counts.

## 8. Distributed-control algebra and gates (`t12_gates.sh`, `t12_gates_table.py` → `t12_gates.csv`, **60/60 PASS**)

`--verify-convection` (N = M, M_track = M) on QUAD/TRI and HEX/TET × BE/CN:
- exact cases: conv-sin3 (2D, n = 4; zero wind, (1,0.5), (−2,1) via `--exact-oc-velocity` — the 2D
  convective exact case always has convection on, so 2D heat is covered by the production rows), and
  conv-sin3-3d (3D, n = 2; heat, w = 0, (1,0.5,−0.75), (−2,1,0.5));
- production full-domain problems: 2D unit square n = 4; 3D non-unit box (0,2)×(−1,1)×(0,½), (3,2,2).

| gate | worst over 60 runs |
|---|---|
| G1 orientation / free skew | 1.7e-14 / 9.4e-14 |
| G2 shell transpose / assembled E | 4.4e-17 / 1.8e-16 |
| G2 reduced-Hessian symmetry (exact LU / production shell) | 1.4e-13 / 2.5e-13 |
| G3 FD gradient (exact / shell) | 4.9e-13 / 4.9e-13 |
| G4 production: reason 2, true residual, ‖∇J‖, independent forward march | all, 1.1e-14, 6.2e-13, 1.5e-15 |
| outer iterations | 2–3 |
| **NEG** (nonzero wind): transposed orientation / ex20-style adjoint transpose / Hessian / FD | ≥ 1.9 / 4.9e-4 / 2.4e-4 / 4.8e-2 |
| heat / w = 0: the ex20-style operators coincide | yes |

The first table mislabelled 8 rows: the 2D exact case ignores `--convection off`/`--vx` (it owns its
wind), so "heat"/"w0" there ran with w = (1,0.5); fixed by `--exact-oc-velocity` (QUAD and TRI alike).

## 9. Exact benchmarks (reused unchanged; §H/§12 of the request)

The SAME `--exact-oc conv-sin3` (2D: y = A(t) S³, p = α_u B(t) S³, unit square, ν = ¼, w = (1,0.5), homogeneous
Dirichlet, full-domain control/tracking) and `conv-sin3-3d` (3D, T9: sin³ product, w = (1,0.5,−0.75)) run on
both element families. Source, target, initial state, stationarity and boundary values are the C++ data
functions of ExactOcBenchmark.cpp / ExactOc3D.cpp, untouched by T12 (they take only coordinates and time and
were symbolically verified in T7/T9); the ONLY changed object is the spatial discretisation (`--mesh-type`).
Every run prints the same EX21_EXACT*_CONFIG (case, ν, wind, α_u, T) on both families.

## 10. Spatial convergence (`t12_tri_space.csv`, `t12_tet_space.csv`, figures)

Exact errors y, p, u in L²(0,T;L²(Ω)) (FEM-integrated with the element's own quadrature), SCHUR_SWEEP + LU.

**2D conv-sin3, CN N = 256** (orders vs h; identical vs DOFs since both families share the lattice):

| n | N_space | QUAD el | TRI el | QUAD y / p | TRI y / p | ratio TRI/QUAD y / p |
|---|---|---|---|---|---|---|
| 8 | 289 | 64 | 128 | 8.65e-4 / 4.64e-5 | 1.25e-3 / 5.43e-5 | 1.44 / 1.17 |
| 16 | 1089 | 256 | 512 | 1.06e-4 (3.03) / 5.16e-6 (3.17) | 1.48e-4 (3.08) / 5.99e-6 (3.18) | 1.40 / 1.16 |
| 32 | 4225 | 1024 | 2048 | 1.32e-5 (3.01) / 6.24e-7 (3.05) | 1.81e-5 (3.03) / 7.10e-7 (3.08) | 1.38 / 1.14 |
| 64 | 16641 | 4096 | 8192 | 1.65e-6 (3.00) / 8.78e-8 (2.83) | 2.26e-6 (3.00) / 9.67e-8 (2.87) | 1.37 / 1.10 |

u = p/α_u (stationarity) has the order of p. The n = 64 dip of p, u is the CN N = 256 time error (from the
n = 16 time study, ≈ 4.2e-6 in u): removed in quadrature, **QUAD 3.00 / 3.02 / 3.02 and TRI 3.00 / 3.03 / 3.03**
(rows "time error removed"; y is not corrected — its fixed-mesh CN time error is the h-dependent start-up
layer and does not transfer between meshes).

**3D conv-sin3-3d, CN N = 128:**

| n | N_space | HEX el | TET el | HEX y / p | TET y / p | ratio TET/HEX y / p |
|---|---|---|---|---|---|---|
| 4 | 729 | 64 | 384 | 5.15e-3 / 7.57e-5 | 9.23e-3 / 1.97e-4 | 1.79 / 2.61 |
| 6 | 2197 | 216 | 1296 | 1.43e-3 (3.16) / 1.84e-5 (3.49) | 2.76e-3 (2.98) / 5.82e-5 (3.01) | 1.93 / 3.17 |
| 8 | 4913 | 512 | 3072 | 5.90e-4 (3.08) / 6.77e-6 (3.47) | 1.12e-3 (3.13) / 2.20e-5 (3.38) | 1.90 / 3.25 |
| 10 | 9261 | 1000 | 6000 | 2.99e-4 (3.04) / 3.16e-6 (3.42) | 5.57e-4 (3.14) / 1.00e-5 (3.53) | 1.86 / 3.16 |
| 12 | 15625 | — | 10368 | — | 3.14e-4 (3.13) / 5.19e-6 (3.60) | — |

TET y converges at ~3.1 and p/u at a pre-asymptotically superconvergent 3.4–3.6 (as HEX, 3.42–3.49). Every
run: stationarity ≤ 5e-15, independent march ≤ 1e-13, 1–2 outer iterations.

## 11. Time convergence (`t12_time.csv`)

Fixed meshes TRI/QUAD n = 16 (conv-sin3) and TET/HEX n = 4 (conv-sin3-3d), production path, the mesh's own
FE L² norm (eᵀ M_full e); BE ref N = 1024, CN ref N = 512; reference-corrected orders at N = 256 (BE) /
128 (CN):

| | BE y_late / p / u | CN y_late / p / u | CN y (full interval) |
|---|---|---|---|
| TRI | 1.00 / 0.99 / 0.99 | 2.67 → 2.18 / 2.00 / 2.00 | layer-dominated (0.78 at N = 128) |
| QUAD | — | 2.09 / 2.00 / 2.00 | layer-dominated (1.54 at N = 128) |
| TET | 1.00 / 0.99 / 0.99 | 2.06 / 2.04 / 2.04 | 1.51 → 1.85 (layer) |
| HEX | — | 2.05 / 2.00 / 2.00 | 2.10 |

y_late = t ≥ T/4. The CN y start-up layer is **inherited from the time scheme and present on QUAD as well**
(full-interval y order collapses at fine Δt on QUAD9 too); BE order 1 and CN order 2 hold on every family
for p, u and for y outside the layer. The time scheme is unchanged. (QUAD/HEX BE rows are in the CSV as well.)

## 12. Mesh comparison (`t12_mesh_comparison.csv`, figures `t12_error_vs_dofs_{2d,3d}.png`)

At equal (n_x, n_y, n_z) the two families have the **identical spatial DOF set**, so the equal-h and
equal-DOF comparisons coincide; the element count differs (TRI 2×, TET 6×).

- **QUAD vs TRI:** same order 3; TRI error 1.37× (y) and 1.10× (p, u) QUAD's at equal DOFs; TRI uses 19%
  less memory (sparser P2 rows: 11.3 vs 15.6 nnz/row) and is equally fast or faster (n = 64: 139 s vs 159 s);
  same outer iterations (2).
- **HEX vs TET:** same order 3; TET error 1.9× (y) and 3.2× (p, u) HEX's at equal DOFs (the HEX27 basis is
  richer per DOF; TET10 rows have 25.8 vs 55.9 nnz/row); TET uses ~46% less memory and is faster at equal
  DOFs (n = 10: 165 s vs 213 s); same outer iterations (1–2). Against elements (right panels) TET needs 6×
  more elements for the same lattice — an unfair axis, reported for completeness.

The claim is not that one family is better; it is that the AAO formulation and SCHUR_SWEEP show the
expected accuracy (order 3) and algebraic correctness on both.

## 13. Solver policy (`t12_solver.sh` → `t12_solver.csv`)

Same exact PDEs, n = 16 (2D) / 6 (3D), N = 64, BE and CN: SCHUR_SWEEP with the LU spatial backend vs the
default CG/GMRES + GAMG backend. Every run reason 2; LU and GAMG errors identical to all printed digits;
outer iterations identical (2D 2, 3D 1–2).

| | GAMG inner (right/left/y-split max) BE / CN | wall GAMG / LU |
|---|---|---|
| QUAD | 13 / 12 | 9.7 s / 2.3 s |
| **TRI** | **19–20 / 15–16** | 12.2 s / 2.2 s |
| HEX | 13 / 13 | 59 s / 10 s |
| TET | 12 / 12 | 16 s / 7.9 s |

**GAMG on P2 TRI needs ~1.4× the inner iterations of QUAD9** (recorded, not tuned); on TET it behaves like
HEX. **MUMPS note (MPI, §P):** the parallel LU (MUMPS) of the P2 TET spatial operator at np = 4 stopped with
INFOG(1) = −9 (work array estimated too small) — an allocation estimate, not a correctness issue (the same
operators converge with GAMG at np = 4; HEX27 at np = 4 does not hit it); the standard remedy ICNTL(14) = 100
(+100% workspace) on the four spatial-factor prefixes removes it.

## 14. MPI np = 1/2/4 (`t12_mpi.sh`, `t12_mpi.py` → `t12_mpi.csv`, **9/9 PASS**)

A TRI convection CN (n = 8), B TET convection BE, C TET convection CN (n = 4), N = 8, LU (np 1) / MUMPS
(np > 1, ICNTL(14) = 100), outer rtol 1e-12: fields vs np = 1 ≤ 6.9e-15, J ≤ 3.9e-15, same outer counts (2),
r₀ identical and histories within 2e-17 r₀, every lattice point present exactly once at every np (no
duplicated / missing DOF across the partition), nrank = np.

## 15. Memory (`t12_memory.csv`, figure `t12_memory_vs_unknowns.png`)

- MaxRSS is linear in the space-time size on all four families: per time slot ≈ 1.7 kB (TRI), 2.1 kB (QUAD),
  2.8 kB (TET), 5.3 kB (HEX) per spatial DOF (fits for N_t ≥ 64), and 0.9 / 1.1 / 1.7 / 3.2 kB per space-time
  unknown on the ladders — the ~O(N_t N_space) AAO law holds.
- Simplex support adds **no** extra AAO storage; the difference is the spatial sparsity: nnz(L)/row
  TRI 11.3 vs QUAD 15.6, TET 25.8 vs HEX 55.9 (LU fill follows).

## R. Optional boundary-control feasibility (audit only; nothing implemented)

- **2D TRI:** the boundary faces of TRI6 are EDGE3 (type 1, 3 nodes); the existing 2D face loop (M_Γ, K_Γ)
  already reads `GetElementFaceDofNumber(…, solution_type)` and `_finiteElement[face_geometry][solution_type]`
  and uses the 1D arc-length tangent, which is exact for straight edges — it should work unchanged.
- **3D TET:** the boundary faces are TRI6 (type 1); the T11N/T11D/T11H1 face loop uses `JacobianSur`
  (generic) and `FaceSurfaceGradients` (metric tensor from `GetDPhiDXi/GetDPhiDEta` of the 2D face element —
  generic, also for triangles); the box-face area/normal assertion and the Kuhn mesh's face ids 1–6 are the
  HEX27 ones; C = support diag(M_Γ) and the perimeter elimination are generic.
- **Estimate:** L2 Neumann and Dirichlet control on simplices look like a refusal lift plus the full
  validation battery (face-mass/trace checks against TRI6 references instead of tensor products, gates,
  exact convergence). Not started: a smoke run would have required a rebuild while the regression was
  running, and the task forbids delaying the T12 closure for it.

## T. Whitelist / refusals (`t12_refusals.sh`, `t12/logs/refusals.log`)

**57/57.** Refused on simplices (simplex-specific messages):
- Neumann and Dirichlet boundary control (2D and 3D), the natural state BC, H1-space and H1-time;
- partial control or tracking boxes, multi-zone and source boxes;
- variable winds, LPS, non-SWEEP KKT preconditioners (additive, matching, boundary-reduced);
- the Math2025 / ErrorFeedback presets, the ex20 verification batteries;
- the exact cases lp52, lift-*, neumann-conv, natural-conv-3d, dirichlet-conv-3d, neumann-conv-3d;
- the invalid type/dimension combinations (tri/quad in 1D or 3D, tet/hex in 2D, unknown type).

Accepted:
- TRI and TET heat BE, wind CN, and non-unit boxes;
- the exact cases conv-sin3, conv-sin, conv-sin3-3d, conv-sin-3d;
- `--verify-convection` on TET;
- the explicit `--mesh-type quad` / `hex` default paths;
- the GAMG backend (TET BE, TRI CN).

## S. Regression (`t12_regress_all.sh`; logs `t12/logs/t12_regress_*.log`)

OLD = a private clean build of the **T11H1 commit 81f32b89e** (`git archive`, sanitised; recipe
`t12/scripts/clean_build_t11h1.sh`; sha256 521667fb…; no Miniforge).

| Suite | Result |
|---|---|
| ex20 goldens | **61/61** |
| Math2025 Ex3/Ex4/Ex5 | byte-identical |
| EF 9.5 | only the known `variant=source` label (as in every previous regression) |
| R4 T0–T2 | **PASS** |
| R5 T3/T4 | 9/9 identical |
| 2D QUAD old vs new (T6, T7N, T7D all four regularisations, T7F, T8D dense/iterative and verify, natural-BC, Dirichlet heat) | **55/55 IDENTICAL**, **34/34 JSON_IDENTICAL** (the 2 boundary-reduced + w0 cases refused by both binaries, as before) |
| 2D/3D restrictions | identical to the T11H1 log |
| T9 (HEX27 distributed, incl. conv-sin-3d / conv-sin3-3d exact) | **32/32** (MPI np 2 round-off 3.0e-15) |
| T10 | **27/27** |
| T11N | **43/43** |
| T11D | **101/101** |
| T11H1 (gates, exact driver incl. dense/verify core, JSON, dumps) | **52/52** |

Default runs (no `--mesh-type`) print and dump exactly as before.

Build hygiene: the NEW binary and its link line contain no Miniforge/Conda. All builds and runs used the
sanitised PETSc-arch environment. Miniforge Python is not used in T12 (no sympy step).

## U. Verdict

**SIMPLEX / MESH-INDEPENDENCE: VALIDATED.** P2 triangles and P2 tetrahedra run correctly in the
distributed full-domain scope, with correct geometry and DOF counts, operators, algebra and parity. They
converge at order 3 in space and BE 1 / CN 2 in time on the same exact PDEs as QUAD9/HEX27, with
consistent QUAD/TRI and HEX/TET comparisons, MPI agreement, the memory law kept, and no regression.

Notes, not blockers:
1. GAMG takes ~1.4× more inner iterations on P2 TRI than on QUAD9.
2. MUMPS needs ICNTL(14) for P2 TET at np = 4.
3. The CN y start-up layer is inherited and present on QUAD too.
4. TET errors at equal DOFs are 1.9× (y) / 3.2× (p) the HEX ones; TRI 1.37× / 1.10× the QUAD ones.

## V. Remaining work after T12 (planned separately; not started)

- **Multiple controlled faces in 3D.**
  - C = the union of interior face nodes, with an explicit edge/corner policy between two controlled faces (shared edge: control or eliminated?).
  - M_Γ, K_Γ assembled over several faces with a block surface metric.
  - `neumann_control_face` becomes a list; the CLI becomes `--dirichlet-face3d a,b`.
  - The face orientation/area assertion per face.
  - Trace, transpose, gates, an exact benchmark with two faces, H1 across a shared edge (H¹ continuity), and regression.
- **Spatially variable 3D wind.**
  - Port `convection.field` to 3D: a quadrature-point wind in the volume assembly, a divergence-free 3D test field, and the orientation gate for non-constant w.
  - The skew identity with ∇·w = 0.
  - The boundary term (w·n) for Neumann/natural faces with a varying w·n.
  - An exact benchmark with a variable wind; reuse of 2D T3 (q2-divfree) ideas; the GAMG/SWEEP behaviour.
- **3D LPS stabilisation.**
  - Port the T6 LPS (P0 projection onto element patches) to HEX27 (and TET10).
  - The 3D patch/fluctuation operator W, its symmetry and PSD.
  - The adjoint consistency (LPS added before elimination, transposed adjoint).
  - The order loss documented in 2D (P0 LPS loses an order on Q2), the high-Pe exact benchmarks, the gate battery, memory.

## W. Files

- Sources: `ex21/ex21.cpp`, `ex21/include/{Config.hpp, CommandLine.cpp, Discretization.cpp, Discretization.hpp, Verification.cpp}`.
- Report and tables: `aux/investigation/simplex/T12_SIMPLEX_MESH_INDEPENDENCE.md`, `t12_{geometry,operators,gates,tri_space,tet_space,time,mesh_comparison,solver,mpi,memory}.csv`.
- Figures: `figures/t12_{tri_space_convergence,tet_space_convergence,error_vs_dofs_2d,error_vs_dofs_3d,memory_vs_unknowns}.png`.
- Scripts: `t12/scripts/` (operators, gates, ladders, runs, time, compare, solver, parity, refusals, MPI, memory, figures, regression, clean build).
- Logs: `t12/logs/`.
