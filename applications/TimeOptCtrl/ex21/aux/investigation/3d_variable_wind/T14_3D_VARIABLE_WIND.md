# T14 — Spatially variable 3D convection fields (ex21, HEX27 Q2)

*Baseline:* **a28630779** "Add validated multi-face boundary control to ex21" (T13, committed at the start of this task). Nothing for T14 is committed, staged or pushed.

*Goal:* replace the constant 3D wind w with w(x,y,z), time-independent, without changing the discrete DTO structure. C is assembled at quadrature points and the adjoint stays the exact transpose Cᵀ. Scope:
- distributed control, natural state BC, and Neumann / Dirichlet / multi-face boundary control;
- the existing H1 Dirichlet machinery.

Scripts, logs and results are in `t14/`; the tables are in this directory.

## 1. Audit of the existing 2D variable-wind support (written before any code change)

Sources read: `Config.hpp` (ConvectionConfig, EvaluateVelocity, WindLpRecirculating, WindQ2Divfree, Validate, Validate3D, ValidateSimplex), `CommandLine.cpp` (`--wind`), `Discretization.cpp` (the convection term of BuildNativeSpatialOperators, BuildLpsMatrix, BuildBoundaryConvectionMatrix), `ConvectionVerification.cpp` (G1 orientation, G1N / G8b natural gates), `ExactOcBenchmark.cpp` (conv-sin / conv-sin3 with `--wind`, LP52), and the reports `leveque_pearson/T3_VARIABLE_WIND_DIRICHLET.md` and `T6_LPS_STABILISATION.md`.

**A. Representation.**
- `ProblemConfig::ConvectionConfig::field` is a function pointer `void(const ProblemConfig&, double x, double z, double w[2])` plus a `field_name` used for logs. Null means the constant pair (vx, vy).
- It is selected by `--wind lp-recirculating | q2-divfree` (CommandLine.cpp) and requires `--convection on` (Validate).
- Two presets, both divergence-free:
  - `q2-divfree` = (½ + x², ¼ − 2xz), biquadratic, exactly in Q2;
  - `lp-recirculating` = (2z(1 − x²), −2x(1 − z²)) (LP §5.2).
- There is no time dependence and no divergence function.

**B. Where it is evaluated.**
- **Galerkin term:** BuildNativeSpatialOperators evaluates w at the PHYSICAL QUADRATURE POINT, x_q = Σ_a φ_a(ξ_q) x_a (isoparametric), whenever `field` is set. The constant `velocity[3]` is used unchanged otherwise.
- **LPS:** BuildLpsMatrix uses w at the quadrature point in the integrand and w at the element centroid for δ_k (LP's definition). LPS is refused in 3D.
- **Gates and Pe:** the 2D G1 orientation gate and the exact driver's Pe report evaluate w at nodes.

**C. Divergence.**
- The assembled operator C_ij = ∫ φ_i (w·∇φ_j) does not assume div w = 0.
- Everything around it does:
  - both presets are divergence-free;
  - the 2D exact targets use y_d = y + (p_t + νΔp + w·∇p)/α_t, i.e. −w·∇p only, correct only for div w = 0;
  - the T3 gate "skew-symmetry of C on the free DOFs (div w = 0)".
- There is no D_w = ∫ (div w)φ_iφ_j anywhere.

**D. How C is assembled.** In the strong form, inside the same element loop as K: `L_ab += ∫ (w(x_q)·∇φ_b) φ_a`, row a = test, column b = trial. C is not stored separately: the gates recover it as C = L(with wind) − L(without) on the same mesh.

**E. Adjoint.** The exact transpose of the assembled space-time operator, built with Lᵀ (MatTranspose / MatMultTranspose). No adjoint convection term is assembled, so the −(div w)p term is carried automatically by Cᵀ. The continuous adjoint matters only for exact data and boundary-condition derivations.

**F. Dimension-specific assumptions** (each one to generalise or keep refused):
- **The field signature is 2D:** (x, z) → w[2] writes two components. In 3D the assembly would pass point_velocity[3] but call the 2D field with (x, z) = (x, y), reading only w[0], w[1]. 3D therefore refuses `field` (Validate3D: "only a constant wind (named/variable winds are not ported)"; ValidateSimplex and the 3D distributed whitelist likewise).
- **BuildBoundaryConvectionMatrix (3D B_w)** uses the constant w; w·n is constant per face.
- **The 3D G1 orientation gate** expects v·M1 (constant).
- **The 3D natural gate** (`natural_gate`) requires `!convection.field`.
- **The exact cases** (ExactOc3D.cpp) read vx, vy, vz directly.
- **The Pe report** (EX21_EXACT3D_CONFIG) uses the constant |w|.
- **Output.cpp JSON** records vx, vy only.

**G. CLI.** Reuse `--wind <name>` rather than inventing new syntax, adding 3D presets that are valid only with `--dimension 3`:
- the 2D names stay 2D;
- a 3D name in 2D is refused, and so is a 2D name in 3D (as today);
- a field with a nonzero --vx/--vy/--vz is refused (no silent conflict).

## 2. Formulation and choices (fixed before coding)

**Discrete structure (unchanged).**
- C_ij = ∫_Ω φ_i (w(x)·∇φ_j) dx, with w evaluated at every physical quadrature point.
- L = νK + C; the adjoint uses the exact transpose Lᵀ.
- The strong adjoint −p_t − νΔp − div(wp) = −p_t − νΔp − w·∇p − (div w)p is used ONLY for exact data and boundary-condition derivations.

**Identity (primary gate).** Integration by parts of C_ij + C_ji gives

  C + Cᵀ = B_w − D_w,  B_w,ij = ∫_∂Ω (w·n) φ_iφ_j dS,  D_w,ij = ∫_Ω (div w) φ_iφ_j dx.

Its consequences:
- div w = 0: C + Cᵀ = B_w;
- homogeneous Dirichlet state: (C + Cᵀ)_FF = −(D_w)_FF;
- both: C_FF is skew.

**Fields.** Presets are quadratic potentials, Φ = ½ xᵀHx, w = ∇Φ = Hx, div w = tr H, Hessian H constant. The field is AFFINE, hence exactly representable in Q2. The orientation gate C·x_k = M·I_h(w_k) is therefore exact, and Gauss quadrature integrates C exactly.

| preset (`--wind`) | H | w | div w |
|---|---|---|---|
| `harmonic-3d` | [[0, 1, −¾], [1, 0, ½], [−¾, ½, 0]] | (y − ¾z, x + ½z, ½y − ¾x) | 0 |
| `affine-div-3d` | the above + diag(½, ¼, ¼) | (½x + y − ¾z, x + ¼y + ½z, ½y − ¾x + ¼z) | 1 |

- All three components are nontrivial, w·n varies along every face, and |w| ≤ 1.6 on the unit cube.
- With ν = ¼ and h = ⅛, the cell Péclet number is ≤ 0.4. That is moderate: no layers, and Galerkin works without LPS.

**Exact benchmarks.**
- **Distributed** (`conv-sin3-3d --wind …`): the T9 shape S = sin³ sin³ sin³ in y and p. The source is f = y_t − νΔy + w·∇y − ηu. The target is y_d = y + (p_t + νΔp + w·∇p + (div w)p)/α_t, i.e. the FULL adjoint; the divergence term enters only with a field. The sin³ factors make every datum vanish on ∂Ω for any w.
- **Natural** (`natural-var-3d --wind <potential preset>`): p = α_u B(t) e^{−Φ/ν} Q, with Q = (2 + cos πx)(2 + cos πy)(2 + cos πz) and ∂_nQ = 0. This gives ν∂_np + (w·n)p = e^{−Φ/ν} ν∂_nQ = 0 on every face, EXACTLY, for any potential wind (w·n = ∂_nΦ), including div w ≠ 0. It generalises T10's R_k(s) = e^{−ks}(2 + cos πs), the constant-wind special case Φ = w·x. The state is y = A cos cos cos. ν defaults to ½ so that e^{−Φ/ν} stays within a factor 90 on the cube.
- **Neumann** (`neumann-var-3d` on zmin, `neumann-var-opp-3d` on zmin ∪ zmax): p = α_u B e^{−Φ/ν} X(x)X(y)(2 + cos πz), X = 1 − cos 2πs.
  - **Lateral faces:** p = ∂_np = 0.
  - **z faces:** Robin is exact by the potential argument.
  - **Control:** u = −(η/α_u) p on Γ_c.
  - **State:** y = A cos cos cos + κ₀G₀(x,y)H₀(z) [+ κ₁G₁(x,y)H₁(z)], with G_k = e^{−Φ(x,y,z_k)/ν} X(x)X(y)(2 + cos πz_k), κ₀ = η²B/ν and κ₁ = −η²B/ν. This makes ν∂_ny = ηu on each controlled face.
- **Dirichlet** (`dirichlet-var-3d`, zmin): owns the swirl wind
  w = (s(x)s′(y)g(z), −s′(x)s(y)g(z), γ), s = sin²(πs), g the septic step 0 → 1 on [½, 1], γ = `--vz`.
  - It is div-free and C³. The tangential components vanish on the lateral faces, so the source's lateral trace vanishes as in T11D's normal-wind runs.
  - Below ω = (0,1)²×(½,1) the wind is the constant (0, 0, γ). The T11D image-kernel adjoint (homogeneous there) is reused unchanged, and stationarity and u are the T11D ones.
  - The variable part enters the state everywhere it is nonzero, and the adjoint in ω, where its residual goes into y_d.
  - **Why this field:** with a variable wind reaching Γ_c, the adjoint must solve the homogeneous variable-coefficient equation in the slab between Γ_c and ω. No closed form exists, and tracking on Γ_c is not implemented (refused).
- **Neumann adjacent, Dirichlet multi-face, H1:** no new closed forms. These are covered by algebra, gates and MPI.

**Negatives built in.**
- The "missing divergence" adjoint: spatial adjoint Lᵀ + D_w, i.e. the strong −w·∇p without −(div w)p.
- C instead of Cᵀ.
- A wind frozen at element centroids (Python reference assembly).
- The div-free identity C + Cᵀ = B_w applied to the generic field.

## 3. Implementation (the constant-wind path is unchanged operation for operation)

| file | change |
|---|---|
| `Config.hpp` | `ConvectionConfig::field3d` (x, y, z → w[3]), `divergence3d`, `potential` + `hessian[9]` (quadratic presets), `field3d_reads_vz`; `EvaluateVelocity3D`, `EvaluateDivergence3D`, `WindPotential3D`, `DivPotential3D`, `PotentialPhi3D`, `InstallWindPreset3D`. Validate: a field needs `--convection on` and `--dimension 3`; Validate3D refuses field + constant vector, and a 2D field in 3D; ValidateSimplex refuses 3D fields |
| `CommandLine.cpp` | `--wind harmonic-3d \| affine-div-3d` (the 2D names unchanged; `--wind` given twice refused); the T14 exact case names |
| `Discretization.cpp/.hpp` | volume C: `field3d` at every 3D quadrature point (isoparametric x_q); B_w (`BuildBoundaryConvectionMatrix`): w(x_q)·n at every FACE quadrature point; new `BuildDivergenceMassMatrix` (D_w, verification) and `ComputeWindStats3D` (min/max \|w\|, max \|div w\|, Pe_h,max) |
| `ConvectionVerification.cpp` | G1 orientation with a 3D field: C x_k = M I_h(w_k). Natural state: `G1W_IDENTITY` (C + Cᵀ = B_w − D_w; the G1N line is the div-free identity, a negative for div w ≠ 0). Dirichlet state / Dirichlet control: `G1W_DIRICHLET` ((C + Cᵀ)_FF = (B_w)_FF − (D_w)_FF). The missing-divergence adjoint variant `missing_div_term` (Lᵀ + D_w) in the transpose (`G8D`), Hessian and FD gates. `EX21_VC_WIND3D` diagnostics. All new lines print only with a 3D field |
| `ExactOc3D.cpp/.hpp` | conv-sin3-3d reads the field (w·∇S; y_d with +(div w)p); `natural-var-3d`, `neumann-var-3d`, `neumann-var-opp-3d` (potential construction); `dirichlet-var-3d` (owned swirl); the multi-face print and expected-count branches keyed on `IsMultiFace3D` (equivalent for T13) |

The wind is evaluated on demand. The presets store a 3×3 matrix and nothing is stored per node or per element (§17).

## 4. Quadrature-point assembly and the identity C + Cᵀ = B_w − D_w (`t14_assembly_reference.csv` 18/18, `t14_operator_identity.csv` 39/39)

An INDEPENDENT Python assembly (`t14_assembly.py`) uses tensor 1D Q2 Lagrange bases and 4-point Gauss per axis, which is exact for the affine presets. It assembles C_ref, B_ref (face quadrature) and D_ref on the HEX27 lattice and compares them with the binary's dumps (C = L(wind) − L(off), same mesh). Cases: unit cube (2,2,2) and the NON-UNIT ANISOTROPIC box (0,2)×(−1,1)×(0,½) at (3,2,2) and (2,3,4); natural and Dirichlet state; both presets.

| check | worst |
|---|---|
| C_dump = C_ref (rel. max entrywise) | 1.7e-14 |
| NEG: wind frozen at element centroids | differs by 4.6e-2 … 9.5e-2 |
| C + Cᵀ = B_w − D_w before elimination | 1.8e-14 |
| (C + Cᵀ)_FF = −(D_w)_FF after elimination | 2.7e-14 |
| 1ᵀD_w1 = 1ᵀB_w1 = \|Ω\| div w | exact (0, 0; 2, 2 on the non-unit box) |
| D_w = 0 for harmonic-3d | exactly 0 |
| NEG: div-free identity on affine-div-3d | fails, 4.6e-2 … 1.4e-1 |

**In the binary** (`--verify-convection`, every gate run):
- `G1W_IDENTITY` ≤ 3e-14;
- `G1W_DIRICHLET` ≤ 2e-14 for distributed Dirichlet state and for single-face and union Dirichlet control.

**Dirichlet control:** the controlled face stays free, so the boundary term survives on Γ_c and the free-block identity is (C + Cᵀ)_FF = (B_w)_FF − (D_w)_FF. The first version of this gate omitted B_w and read 0.13–0.21. That was a gate bug, fixed before the battery.

**The swirl (trigonometric) wind** satisfies the identity only to quadrature error: 1.75e-3 (n = 2) and 6.4e-5 (n = 4), a factor 27 per halving. The discrete transpose stays exact because the adjoint uses the same assembled matrix (§7).

## 5. Distributed control algebra (`t14_gates_distributed.csv` 18/18, `t14_gates_natural.csv` 14/14)

Runs:
- exact conv-sin3-3d n = 2, 3 and production on the non-unit box;
- constant wind A as the baseline, harmonic-3d and affine-div-3d; BE/CN;
- homogeneous Dirichlet state; natural state on the non-unit box and natural-var-3d.

| gate | worst |
|---|---|
| G1 orientation (exact for affine w) | ≤ 3.9e-14 |
| shell transpose | ≤ 4e-17 |
| Hessian symmetry | ≤ 1.8e-13 |
| FD gradient | ≤ 1.9e-13 |
| production true residual | ≤ 1.4e-13 |
| forward march | ≤ 1e-14 |

Negatives:
- **ex20-style untransposed adjoint:** detected everywhere.
- **Natural wrong BC (νK − C):** detected (FD 0.035 … 3.4).
- **Missing-divergence adjoint (Lᵀ + D_w, the strong −w·∇p without −(div w)p) on affine-div-3d:** transpose defect 4.7e-5 … 1.5e-3, FD 8.9e-3 … 1.0. All detected.
- **Same on the div-free winds:** it COINCIDES with the exact adjoint (FD ≤ 1.9e-13, equal to the exact adjoint's own FD; G8D ≤ 4e-17), as it must.
- **Div-free identities on the generic field:** fail (G1N natural 0.157; skew-only Dirichlet 0.286).

## 6. Symbolic verification (`t14/t14_symbolic.csv`, 146/146)

- **Divergence identity:** −div(wp) = −w·∇p − (div w)p for a SYMBOLIC symmetric H and a symbolic p.
- **Presets:** w = ∇Φ; div w = 0 / 1; w·n non-constant on every face.
- **conv-sin3-3d** (both fields):
  - the state residual and the FULL adjoint residual vanish;
  - y, p, ∇y, ∇p vanish on all six faces; p(T) = 0; stationarity holds;
  - NEG: the target without (div w)p leaves an adjoint residual for affine-div (and none for harmonic).
- **natural-var-3d:**
  - the Robin condition ν∂_np + (w·n)p = 0 holds on all six faces for a SYMBOLIC symmetric H;
  - ν∂_ny = 0; the full adjoint residual vanishes; p(T) = 0;
  - NEG: Robin with the face-AVERAGED w·n fails, so w·n must vary along the face.
- **neumann-var-3d / -opp-3d** (both fields):
  - ν∂_ny = ηu on each controlled face and 0 elsewhere;
  - Robin on all faces; stationarity; residuals.
- **dirichlet-var-3d swirl:**
  - div w = 0 for any g(z); the tangential parts vanish on the lateral faces;
  - the septic step is flat at z = ½;
  - the T11D adjoint is homogeneous for the wind (0, 0, γ) below ω;
  - p = 0 on the boundary; y = ηu on zmin.
- **Literal ports:** the C++ `PotShape` gradient/Laplacian and the `VarG` Laplacian match sympy (≤ 1e-12).

## 7. Boundary control under variable wind

**Neumann** (`t14_gates_neumann.csv`, 32/32):
- **Coverage:** exact neumann-var-3d (zmin) and neumann-var-opp-3d (zmin ∪ zmax), n = 2, 3; production single face and the unions zmin,zmax, zmin,xmin and xmin,ymin,zmin; both fields; BE/CN.
- **Positives:** G1W ≤ 3.5e-14 (B_w at face quadrature points); FD ≤ 3.3e-12; union identity N = Σ_f M_Γ(f) = 0.
- **Negatives:** drop/duplicate face, wrong BC (FD 0.010 … 12.7), ex20-style, and missing divergence on affine-div (FD 0.059 … 17.6). All detected.

**Dirichlet** (`t14_gates_dirichlet.csv`, 64/64; explicit `t14/t14_bd.csv`, 132/132):
- **Coverage:** single face, opposite, adjacent and the non-unit adjacent union; constant / harmonic / affine-div; L2, H1-space, H1-time, combined; BE/CN; plus exact dirichlet-var-3d n = 2, 4.
- **Positives:** B_Dᵀ ≤ 1.8e-17; FD ≤ 7.9e-14; true residual ≤ 2.8e-11; kkt ≤ 2.8e-11; G1W_DIRICHLET ≤ 1.9e-14 (affine fields).
- **Explicit B_D / B_Dᵀ** equal the BE/CN formula from the dumped M, L and the geometric C to ≤ 1e-13, for both fields, single face and three unions.
- **G1D on I ∪ C** against the independent B_w, D_w: ≤ 1e-13.
- **Negatives:** L instead of Lᵀ, wrong C, and the identity without D_w are all detected.
- **Swirl:** judged by quadrature convergence (§4); a documented revision in `t14_gates_table.py`.

## 8. Reduced core (`t14_core_equivalence.csv`, 26/26)

Coverage: single face, opposite, adjacent and non-unit adjacent; harmonic / affine-div; L2 / H1s / H1t / comb; plus dirichlet-var-3d BE/CN.
- **Dense K:** symmetric and SPD.
- **Shell action vs dense:** ≤ 2.8e-15.
- **Iterative (1e-12) vs dense:** ≤ 8.6e-14; λ_min ≥ 8.8e-4. The wind enters only through L, Lᵀ in B_D, so the core is unchanged.

## 9. H1 interaction (algebraic)

H1-space, H1-time and combined run with both variable winds on single faces and unions:
- 36 of the 64 Dirichlet gate rows (all PASS);
- 18 of the 26 core equivalences;
- MPI: the opposite pair, harmonic, combined H1, np 1/2/4 (§12);
- byte-identical permutation (adjacent, combined, CN; §11).

M_Γ, K_Γ and R_H1t do not depend on the wind. No new H1 closed form with a variable wind was built. **H1 + variable wind is validated algebraically**; exact H1 convergence is covered by T11H1/T13, and exact variable-wind convergence by T14.

## 10. Multi-face interaction (`t14_multiface.csv`, 217/217)

- **Union gates (Neumann and Dirichlet):** pass.
- **Free-block identity:** (C + Cᵀ)_FF = (B_w − D_w)_FF on unions, checked in the binary and against the independent reference.
- **Permutation** (`t14_permutation.csv`): 44/44 byte-identical under the GENERIC wind (Neumann opp / adj / three; Dirichlet opp / adj BE/CN, three, adjacent combined H1).
- **Union MPI cases:** pass.

## 11. MPI (`t14_mpi.csv`, 24/24)

np = 1/2/4 for:
- distributed div-free (BE) and distributed GENERIC (CN);
- natural generic (CN);
- Neumann single face (BE) and the opposite union, generic (CN);
- Dirichlet: exact swirl CN, the production adjacent union generic CN, and the opposite union harmonic with combined H1 CN.

Fields ≤ 4.1e-15 vs np = 1, J ≤ 2.6e-15, trace sets as predicted, histories within the T11N/T13 criteria.

## 12. Spatial convergence (`t14_space_convergence.csv`, `figures/t14_space_convergence.png`)

Exact errors: y, p in L2(Ω)×L2(0,T); u in L2(Ω) (distributed / natural) or L2(Γ_c) = Σ_f (boundary control); CN.

| ladder | y orders (n = 2 → … → 10) | p | u |
|---|---|---|---|
| distributed, constant wind A (baseline), N = 64 | 3.45, 3.16, 3.08, **3.05** | 5.10, 3.49, 3.47, 3.41 | = p |
| distributed, harmonic-3d | 3.45, 3.16, 3.08, **3.05** | 5.14, 3.48, 3.48, 3.43 | = p |
| distributed, affine-div-3d (div w = 1) | 3.45, 3.19, 3.09, **3.06** | 5.19, 3.51, 3.51, 3.46 | = p |
| natural-var-3d harmonic, N = 64 | 3.08, 3.02, 2.98, 2.87 | 3.06, 2.99, 2.85, 2.43 | = p |
| natural-var-3d affine-div, N = 64 | 3.09, 3.03, 3.00, 2.91 | 3.04, 2.99, 2.87, 2.50 | = p |
| natural-var-3d affine-div, **N = 128**, n = 6 → 8 → 10 | 3.02, **3.01** | 3.00, **2.96** | 3.00, **2.96** |
| natural-var-3d harmonic, **N = 128** | 3.01, **3.00** | 3.00, **2.96** | 3.00, **2.96** |
| neumann-var-3d harmonic (zmin) | 3.46, 3.23, 3.12, **3.07** | 4.01, 4.02, 3.96, 3.88 | 3.78, 3.44, 3.26, **3.12** |
| neumann-var-opp-3d affine-div (zmin ∪ zmax) | 3.24, 3.21, 3.10, **3.06** | 4.28, 3.65, 3.44, 3.28 | 3.93, 3.43, 3.25, **3.13** |
| dirichlet-var-3d swirl (zmin), N = 256, n = 2 … 12 | 4.43, 3.36, 3.10, 3.05, **3.04** | 1.83, 4.05, 3.70, 2.66, 2.84 | 5.19, 4.64, 3.28, 2.99, **3.11** |

- **Q2 order 3 for every variable-wind case,** including the generic div w ≠ 0 field. The distributed errors are within 1–2 % of the constant-wind baseline at every n. Orders above 3 for p and u are pre-asymptotic or superconvergent and are not claimed.
- **natural-var at N = 64:** the CN time error (≈ 2.9e-6 in p, §13) floors p and u at n = 10. At N = 128 they read 2.96.
- **dirichlet-var-3d:** an exact variable-wind Dirichlet L2 benchmark. Its orders follow T11D's normal-wind case (the irregular p sequence is the same image-kernel pre-asymptotics). The swirl moves J by only ~1e-9 relative to the normal wind, because it acts inside ω where the solution is smooth. It is a weak PROBE of the variable wind, so:
- **Dirichlet with a variable wind REACHING Γ_c:** fixed-dt self-convergence (CN N = 16) on production data (zmin, sin_x data), affine-div-3d vs the constant winds on the SAME data.
  - **Reference n = 24, n = 2 … 12:**
    - y: generic 3.30, 1.90, 1.73, 1.78 vs constant 4.36, 2.59, 1.90, 1.81;
    - p identical (2.62);
    - u: generic 4.42, 3.71, 3.23, 2.98 vs constant 4.74, 4.47, 4.10, 3.83.
  - **Both winds share the asymptotic y ≈ 1.8.** That is the known Dirichlet-L2 Γ_c trace layer of this data (T7D; 96–97 % of the y error lies in z < 1/6 for both winds). It is not a variable-wind effect.
  - **Combined H1 (ref 12):** raises both (generic 2.63, constant 3.57 at the pessimistic 2× reference).

## 13. Temporal convergence (`t14_time_convergence.csv`, `figures/t14_time_convergence.png`)

Fixed mesh n = 4; reference N = 1024 (distributed, Dirichlet) or 512 (natural, Neumann). Reference-corrected orders, N = 64 → 128.

| case | BE y / p / u | CN y / p / u |
|---|---|---|
| distributed conv-sin3-3d harmonic | 0.98 / 0.99 / 0.99 | **2.10 / 2.03 / 2.03** |
| distributed conv-sin3-3d affine-div (generic) | — | **2.10 / 2.04 / 2.04** |
| natural-var-3d affine-div | 1.00 / 0.93 / 0.93 | 1.90 / **1.99 / 1.99** (y(T) 2.0) |
| neumann-var-3d harmonic | — | global y 1.70, y(T) **2.05**, p **2.02**, u **2.00** |
| dirichlet-var-3d swirl | 0.91 / 0.92 / 0.91 (rising) | **2.05 / 2.21 / 2.06** |

- **BE order 1, CN order 2, J included:** for distributed / natural / Neumann the |ΔJ| ratios are ≈ 2 / 4. dirichlet-var's J is irregular (ratios 10 and 63, the pre-asymptotic image kernel, as in T11D).
- **CN start-up layer, inherited (T11N):** y reads 1.3–1.7 at coarse N on the distributed and Neumann cases (the boundary-flux / source start-up layer), while y(T) and J converge at 2. It is not tuned away.
- **dirichlet-var BE** approaches 1 slowly, as T11D's dirichlet-conv-3d did (the sharp late-time image kernel).

## 14. Solver audit (`t14_solver.csv`, `figures/t14_solver_iterations.png`)

Production rtol 1e-8, n = 8, N = 32.

| case | SCHUR_SWEEP + LU | GMRES+GAMG max its / wall | GMRES+ASM(4,1)+ILU(1) max its / wall |
|---|---|---|---|
| natural, constant wind A (ν = ¼) BE / CN | 2 outer, 24 / 25 s | 11 / 11, 79 / 53 s | 10 / 8, 194 / 135 s |
| natural-var harmonic (ν = ½) BE / CN | 2 outer, 20 / 22 s | 13 / 10, 51 / 50 s | 12 / 9, 180 / 159 s |
| natural-var affine-div (ν = ½) BE / CN | 2 outer, 19 / 23 s | 13 / 10, 52 / 52 s | 12 / 9, 186 / 148 s |
| neumann-var harmonic BE / CN | 2 outer, 27 / 38 s | 13 / 10, 81 / 100 s | 11 / 9, 312 / 344 s |

- **SWEEP backends:** every backend gives 2 outer iterations, 0 sweep failures and fields vs LU ≤ 1.1e-14. Under BE, GAMG needs 13 instead of 11 spatial iterations with the variable winds. Note that ν differs (the potential cases default to ν = ½).
- **Dirichlet production zmin,xmin** (constant / harmonic / affine-div), iterative core vs dense: 1–2 outer iterations, 5–7 core CG iterations, fields vs dense ≤ 4.6e-10, J ≤ 4e-16. The iterative core takes 7–11 s and 0.61 GB; the dense one 100–107 s and 3.75 GB.
- **No tuning.**

## 15. Memory (`t14_memory.csv`, `figures/t14_memory.png`)

- **RSS ratio:** variable / constant wind is 0.999 … 1.001 in all 18 runs (natural SWEEP + LU and Dirichlet reduced; n = 4, 6; N_t = 32, 64).
- **Sparsity:** nnz(L) is identical for every wind.
- **Per space-time unknown:** 2098 … 2442 B (natural) and 1204 … 1650 B (Dirichlet); it falls with N_t.
- **Storage:** the wind is evaluated on demand from a function pointer and a 3×3 matrix, with no nodal or element array. The only new cost is assembly work.

## 16. Local Péclet diagnostic (diagnostic only)

`EX21_VC_WIND3D` (`ComputeWindStats3D`) reports:
- min/max |w| and max |div w| over all volume quadrature points;
- Pe_h,max = max_K h_K max_{x_q∈K} |w(x_q)| / (2ν), with h_K the largest bounding-box extent of K. One global |w| is not enough: |w| ranges 0.04 … 1.9 on the unit cube, and w → 0 at the origin for the potential presets.

| setting | min \|w\| | max \|w\| | max \|div w\| | Pe_h,max |
|---|---|---|---|---|
| unit cube n = 2 (h = ½), ν = ¼ | 0.05 / 0.07 | 1.75 / 1.89 | 0 / 1 | 1.75 / 1.89 |
| unit cube n = 3 | 0.04 | 1.78 / 1.92 | 0 / 1 | 1.19 / 1.28 |
| non-unit box (0,2)×(−1,1)×(0,½), (3,2,2) / (4,4,4) | 0.05 … 0.10 | 3.1 … 3.2 | 0 / 1 | 3.1 … 6.1 (h_K = 1 / ½, coarse) |
| ladders, n = 10, ν = ¼ | — | 1.92 | — | ≈ 0.38 |

Values are harmonic-3d / affine-div-3d. Nothing switches on Pe (no LPS, no upwinding). The coarse non-unit gate meshes reach Pe_h ≈ 6 and still pass every algebraic gate, as they must: the gates are exact algebra and independent of Pe.

## 17. Regression (`t14/logs/t14_regress_*.log`)

OLD is the T13 build (a28630779, private clean build `t14/scripts/clean_build_t13.sh`, no miniforge). Everything ran in the sanitised env.

| suite | result |
|---|---|
| ex20 goldens | **61/61** |
| Math2025 / EF / constant-wind R4 | PASS (EF: only the known `variant=source` label) |
| 2D old vs new, incl. the T3 2D variable winds (q2-divfree, lp52) | 102 IDENTICAL; JSON 28/28; the 2 boundary-reduced w0 parity cases refused by both (as T12/T13) |
| T9 / T10 / T11N | 32/32 / 27/27 / 43/43 |
| T11D | 100 identical + MPI np 2 at round-off (6.6e-16; MUMPS np 2 is not bitwise, T13's run happened to be JSON-identical) |
| T11H1 / T12 simplex | 52/52 / 45/45 |
| T13 (multi-face exact + gates + JSON + dumps, constant wind; 2D variable-wind gates) | 49/50 IDENTICAL; the 50th (2D LP52 gate) differed only because the harness merges stderr into the NEW stream: a rejected inner diagnostic solve prints to stderr mid-line. Rerun with stderr excluded on both sides: IDENTICAL (45 lines) |

**Constant-wind output is unchanged** (byte-identical stdout, JSON and dumps). The only user-visible text changes:
- the 3D refusal for a 2D field now names the 3D presets;
- the exact Dirichlet L2-only refusal names the actual case (dirichlet-conv-3d's text is unchanged).

## 18. Strict refusals (`t14/logs/t14_refusals.log`, 70/70)

Refused:
- **CLI:** a 3D field in 2D; a 2D field in 3D; field + `--vx/--vy/--vz`; a field without `--convection on`; `--wind` twice; unknown names.
- **Unsupported combinations:** LPS; simplex meshes; Neumann H1; Neumann CN without SWEEP; Dirichlet sweep / matching / natural state / tracking on Γ_c.
- **Exact cases:** every exact case whose closed form uses a constant wind, when given `--wind`; natural-var / neumann-var without a potential field; dirichlet-var with `--wind`, tangential `--vx`, no convection, H1 or sweep.
- **Time-dependent and state-dependent winds:** have no CLI or field type, so nothing can request them.

Accepted: all validated combinations, plus the legacy constant-wind runs.

## 19. Optional simplex interaction

Not run. Volume assembly on TET10 would need no new code, but ValidateSimplex refuses 3D fields, and lifting that refusal is feature work outside the HEX27 closure. Simplex + variable wind stays REFUSED (§18).

## 20. Verdicts

**3D VARIABLE WIND — DIVERGENCE-FREE: VALIDATED** (harmonic-3d; dirichlet-var-3d swirl):
- quadrature-point assembly checked against an independent reference, with the centroid-frozen negative;
- C + Cᵀ = B_w;
- distributed, natural, Neumann (single and union) and Dirichlet (single and union) algebraic gates;
- symbolically verified exact benchmarks: distributed, natural, Neumann single / opposite, Dirichlet;
- Q2 order 3; BE 1 / CN 2;
- variable w·n at face quadrature points;
- byte-identical permutation; MPI; memory; no regression.

**3D VARIABLE WIND — GENERIC: VALIDATED** (affine-div-3d, div w = 1):
- div w ≠ 0 is exercised, and D_w is assembled and independently reconstructed;
- C + Cᵀ = B_w − D_w before and after elimination;
- the strong adjoint includes −(div w)p (symbolic), and the missing-divergence adjoint is detected (transpose, Hessian, FD; the exact target without it leaves a residual);
- the exact distributed benchmark (order 3, BE 1 / CN 2), plus natural-var and neumann-var-opp exact benchmarks;
- natural, Neumann and Dirichlet algebra; no regression.

**3D SPATIALLY VARIABLE WIND: VALIDATED.**

Scope: HEX27 Q2, constant-in-time analytic preset winds (quadratic potentials) and the exact-case swirl.
- **Refused:** simplex + variable wind, LPS, and time- or state-dependent winds.
- **H1 + variable wind:** validated algebraically, with no new exact H1 benchmark.

## 21. Remaining work for T15 (3D LPS) — not started

- **Port BuildLpsMatrix to HEX27:** P0 and the diagnostic P1 local projection per element (3D Gram matrices 1×1 / 4×4); δ_k = h_k/(2|w_k|)(1 − 1/Pe_k), with the 3D chord h_k through the centroid along w_k (box geometry: min over axes of extent·|w|/|w_axis|); w_k at the centroid, as in 2D, so variable winds are handled by the T14 field3d.
- **Lift the 3D refusals:** Validate3D / the 3D distributed whitelist ("no convection stabilisation").
- **Re-run the T6 G5 gates in 3D:** symmetry, PSD, reversal invariance W_{−w} = W_w, constants W1 = 0, the Pe switch. Also the full DTO gates: the adjoint carries the same W because W is symmetric and even in w.
- **Boundary control with LPS:** W has no boundary terms, so B_w and the T13/T14 identities are unchanged, but natural / Neumann / Dirichlet / multi-face with LPS must be gated.
- **Convergence:** document the expected loss of one Q2 order with P0 LPS (2D T6 finding) on the T9/T14 exact cases; add a high-Pe 3D benchmark (e.g. an LP §5.2-type recirculating field) and a solver study (GAMG at high Pe).
- **Regression:** all T9–T14.

## 22. Files

- **Report and tables** in `aux/investigation/3d_variable_wind/`: this report and the CSVs `t14_operator_identity`, `t14_assembly_reference`, `t14_gates_{distributed,natural,neumann,dirichlet}`, `t14_core_equivalence`, `t14_space_convergence`, `t14_time_convergence`, `t14_multiface`, `t14_solver`, `t14_mpi`, `t14_memory`, `t14_permutation`.
- **Also:** `t14/t14_symbolic.csv` and `t14/t14_bd.csv`.
- **Scripts** in `t14/scripts/`:
  - assembly, symbolic and B_D: `t14_assembly.py`, `t14_symbolic.py`, `t14_bd.py`;
  - gates: `t14_gates.sh`, `t14_gates_table.py`;
  - core: `t14_core_equivalence.py`;
  - ladders and convergence: `t14_exact_ladder.sh`, `t14_runs.sh`, `t14_ladder_table.py`, `t14_selfconv.py`, `t14_time.py`, `t14_fields.py`, `t14_convergence.sh`;
  - permutation: `t14_permutation.sh`, `t14_permutation.py`;
  - MPI: `t14_mpi.sh`, `t14_mpi.py`;
  - solver: `t14_solver.sh`, `t14_solver_table.py`;
  - memory, multi-face and refusals: `t14_memory.py`, `t14_multiface.py`, `t14_refusals.sh`;
  - figures: `t14_figures.py`;
  - regression and baseline build: `t14_regress_*.sh`, `clean_build_t13.sh`.
- **Logs** in `t14/logs/`; **figures** in `figures/`.
