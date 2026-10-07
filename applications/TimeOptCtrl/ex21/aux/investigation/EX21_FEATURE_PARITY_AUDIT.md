# ex21 capability-parity audit (pre-3D)

Bounded audit, done on 2026-10-05:

- source read: `include/Config.hpp`, `CommandLine.cpp`, `Discretization.cpp`,
  `DirichletControl.{hpp,cpp}`, `Solver.cpp`, `ConvectionVerification.cpp`,
  and the existing validation reports;
- tiny smoke tests (nx = 8, Nt = 8; under 1 s each);
- a literature check, including the **full text** of Seymen's 2013 thesis.

There was no campaign, no source change, and no restriction was removed.

Machine-readable tables: `feature_parity.csv`, `literature_feature_map.csv`. Generator:
`feature_parity/make_tables.py`. Smoke tests: `feature_parity/smoke.sh`,
`feature_parity/logs/`.

**Four statuses are kept apart throughout:**

- *implemented*: code exists;
- *allowed*: Validate()/CLI accept it with `--convection on`;
- *convection-compatible*: mathematically and in the code as written;
- *validated*: evidence exists.

"SMOKE" means only tiny gate runs from this audit.

## 1. Residual processes

Four background shells were found, all Claude-launched watcher loops from the earlier
ErrorFeedback phase (not T5/T6):

| PID | loop | target | state of target |
|---|---|---|---|
| 25281, 25509 | `until grep -q DONE …/errorfeedback_95_96/logs/campaign_stage3.out; sleep 60` | EF stage-3 driver | last written 2026-10-04 02:04 ("stage-3 driver stopped … pruned"); no `DONE` will ever be written |
| 19361, 19749 | `until grep -q SOLVER_STATS …/solver_diag/logs/diag_ef96_a1e-10_N50.log; sleep 30` | EF96 α = 1e-10 diagnostic | last written 2026-10-04 15:52; no producer running |

- No ex20/ex21 solver, driver, `mpirun` or `prterun` process existed.
- The four loops (and their `sleep` children) were stopped by PID; the tasks ended with
  exit 144.
- Nothing else was touched.

## 2. Definitive capability table

| feature | implemented? | allowed with convection? | convection-compatible? | validated with convection? | evidence | known limitation | literature analogue | external status | 3D |
|---|---|---|---|---|---|---|---|---|---|
| full-domain distributed control | YES | YES | YES | VALIDATED | LP T1-T6 (lp-heat exact, lp52, LPS); exact-OC conv-sin3/lift-*; verify-convection G2-G4 | - | LP 2022 (STRONG); Seymen 2013 thesis / JCAM 2014 (PARTIAL: control-constrained); Burman 2011 (theory) | externally reproducible | EASY |
| subdomain (box) distributed control | YES | YES | YES | VALIDATED | M25 Ex3/Ex4 (1D), Ex5, EF 9.5/9.6 (2D); run_all_gates control box 0.25-0.75 (G2-G4, 3 PCs) | box selection by element centroid; axis-aligned boxes only | M25 Ex3 (STRONG, source-OC reproduction), Ex5 (STRONG periodic regime), EF 9.5/9.6 (PARTIAL, cross-method) | externally reproducible | EASY (Box needs a 3rd extent) |
| multiple distributed-control regions (--control-box-add) | YES | YES | YES | SMOKE | this audit: multizone_{dirichlet,natural}_{cn,be}, +LPS: all gates pass; ex20 goldens (no convection) | union of boxes, lowest index wins on overlap | NONE FOUND (bounded search) | internally validated only | EASY/MODERATE |
| Neumann boundary control | YES (ex20 stages 1-2, BE/CN) | NO (Validate: convection => DISTRIBUTED only) | YES mathematically (DTO exact transpose yields the adjoint BC nu dn p + (w.n) p = 0 automatically); generic PCs plausible; boundary-reduced Schur NOT (SPD Cholesky) | NO | never run with convection | single face (neumann_control_face); L2(Gamma_c) regularisation only; boundary-reduced solver SPD-only | Seymen 2013 (Robin eps dn y + gamma y = u, PARTIAL); parabolic Neumann-control AAO preconditioning (heat, BACKGROUND) | currently unvalidated | MODERATE (surface mass on a 3D face) |
| Dirichlet boundary control | YES (ex20 stage 3, BE/CN, H1(Gamma) + H1-time) | NO (Validate) | NOT as implemented: B_D and B_D^T both apply MatMult(stiffness) (ApplyStep), i.e. L not L^T; reduced Schur = dense LAPACK Cholesky (SPD); R-Cholesky fine | NO | never run with convection | needs a transposed ApplyStep for B_D^T and a nonsymmetric reduced solver | Seymen 2013 Ex 4.2.1 is Robin with gamma = 1e3 (penalty, BACKGROUND); steady HDG Dirichlet control of conv-diff (BACKGROUND) | currently unvalidated | NONTRIVIAL/DEFER |
| full-domain tracking | YES | YES | YES | VALIDATED | LP T1-T6; exact-OC cases | - | LP 2022 (STRONG); Seymen (PARTIAL) | externally reproducible | EASY |
| subdomain tracking | YES | YES | YES | VALIDATED | M25 Ex3/5, EF 9.5/9.6 (Omega_C); gates tracking box | centroid selection | M25/EF (STRONG/PARTIAL) | externally reproducible | EASY |
| multiple tracking regions (--tracking-box-add) | YES | YES | YES | SMOKE | this audit (multizone_*): all gates pass | - | NONE FOUND | internally validated only | EASY/MODERATE |
| non-colocated control/tracking | YES | YES | YES | VALIDATED | M25 Ex3 (RMS 6.2e-6 vs source OC), Ex5, EF 9.6; smoke multizone | - | M25 Ex3/Ex5 (STRONG) | externally reproducible | EASY |
| time-dependent target | YES (CLI: constant, sin_x_minus_t, one_minus_y_sin_x_minus_t; presets/overrides: more) | YES | YES | VALIDATED | gates (sin_x_minus_t), M25/EF signals, LP data | moving_gaussian / exp_minus_t_cos2pi are preset-only (CLI rejects them) | M25/EF/LP | externally reproducible | EASY (formulas are 2D) |
| source / disturbance (whole domain) | YES | YES | YES | VALIDATED | exact-OC manufactured sources (T2/T3), MMS | - | LP/Seymen manufactured | internally validated only | EASY |
| source regions (subdomain disturbance) | PARTIAL (presets only, no CLI) | YES | YES | VALIDATED (single region) | M25 Ex3-5, EF 9.5/9.6 (one Omega_d each) | not CLI-reachable; >1 region never exercised | M25/EF (STRONG/PARTIAL) | externally reproducible (single region) | EASY/MODERATE |
| multiple source regions | PARTIAL (std::vector, presets only) | YES | YES | IMPLEMENTED-UNVALIDATED | never used with >1 region | not CLI-reachable | NONE FOUND | currently unvalidated | EASY |
| homogeneous Dirichlet state BC | YES | YES | YES | VALIDATED | LP, exact-OC, T2 gates, CN trace fix (audit #21) | whole external boundary only | LP, Seymen, M25/EF | externally reproducible | EASY |
| inhomogeneous Dirichlet state BC (lifting) | PARTIAL (data_overrides; exact-OC presets only) | YES | YES | VALIDATED | T3/T3b: lift-const/exp/track, lp52; G3P physical gradient; 34/34 | time-independent g only; distributed control only; not CLI-reachable | LP 5.2 (STRONG, structural) | externally structurally supported | MODERATE |
| natural (homogeneous Neumann) state BC | YES | YES | YES (strong-form convection: nu dn y = 0 on dOmega; N not skew where w.n != 0; exact transpose still the DTO adjoint) | SMOKE | G1 orientation (pre-existing); this audit: multizone_natural_{cn,be}, +LPS, variable wind q2-divfree: all gates pass | no convective-flux (Robin) inflow/outflow BC | Seymen Robin adjoint BC (gamma + beta.n) (STRUCTURAL) | internally validated only | EASY |
| backward Euler | YES | YES | YES | VALIDATED | orders 1 (T2, T6), gates, LP | - | Seymen BE (PARTIAL), LP | externally reproducible | EASY |
| Crank-Nicolson | YES | YES | YES | VALIDATED | LP algebra 49/49 (T0), order 2 (T2, T6) | interval-constant controls (distributed) | LP 2022 (STRONG), Burman 2011 (theory), Seymen (PARTIAL) | externally reproducible | EASY |
| L2 control regularisation | YES | YES | YES | VALIDATED | everything above | - | all sources | externally reproducible | EASY |
| H1-in-space boundary-control regularisation | YES (Dirichlet control only) | NO (Dirichlet control rejected) | n/a until Dirichlet control is convection-ready | NO | ex20 Dirichlet stages (heat) | curve (1D) tangential derivative, arc-length Jacobian | NONE FOUND for convection | currently unvalidated (with convection) | NONTRIVIAL (surface gradient) |
| H1-in-time control regularisation | YES (Dirichlet control only; Validate rejects it otherwise) | NO | derivation exists only for nodal Dirichlet controls | NO | ex20 aux/dirichlet_h1_time derivation check (heat) | distributed/Neumann interval controls need a different discrete d/dt | NONE FOUND for convection | internally validated only (heat) | dimension-independent once the control operator exists; DEFER |
| combined H1(space) + H1(time) | YES (Dirichlet only) | NO | as above | NO | ex20 (heat) | Dirichlet only | NONE FOUND | currently unvalidated (with convection) | DEFER |
| SCHUR_SWEEP | YES | YES | YES (GMRES spatial, transposed backward sweep) | VALIDATED | gates, LP T4-T6, M25/EF; 2 outer its | default spatial GAMG fails at high Pe | LP matching Schur (STRONG) | externally reproducible | EASY (cost) |
| ADDITIVE_GAMG (distributed default) | YES | YES | YES (preconditioner only) | VALIDATED (low Pe) | run_all_gates; smoke (natural BC, variable wind: 20-74 outer its) | outer iterations grow with Pe | - | internally validated only | EASY |
| SCHUR_MATCHING | YES | YES | YES | VALIDATED (low Pe) | run_all_gates 3 PCs | - | LP | internally validated only | EASY |
| boundary-reduced Schur | YES | NO (Neumann only; SPD Cholesky) | NO as implemented | NO | - | SPD-specific | - | currently unvalidated | DEFER |
| Dirichlet-reduced Schur | YES | NO (Dirichlet only; dense potrf) | NO as implemented | NO | - | SPD-specific | - | currently unvalidated | DEFER |
| LU / MUMPS override (PETSC_OPTIONS sweep_*) | YES | YES | YES | VALIDATED | T4-T6; MPI np 1/2/4 (T5, T6) | memory in 3D | - | internally validated only | EASY (memory) |
| high-Pe spatial backends (ILU, ASM+ILU, hypre AIR, GAMG+LPS) | YES (options) | YES | YES | VALIDATED (2D) | T5 (ILU/ASM), T5b (AIR serial; np 4 crash), T6 (GAMG+LPS) | AIR np=4 SEGV; ILU serial | - | internally validated only | MODERATE |
| MPI production path | YES | YES | YES | VALIDATED | T5, T6 MPI checks; this audit np 1/2 (2e-16) | verification drivers serial-only | - | internally validated only | EASY |
| constant wind | YES | YES | YES | VALIDATED | G1-G4, T2 orders, M25/EF | - | LP/Seymen/M25/EF | externally reproducible | EASY (w[2] -> w[3]) |
| variable wind | YES (named fields lp-recirculating, q2-divfree) | YES | YES | VALIDATED | T3 MMS/OC orders, LP 5.2 | no user-defined field; time-independent; div-free assumed | LP 5.2 (STRONG) | externally structurally supported | MODERATE (3D fields) |
| exact transpose adjoint | YES | YES | YES | VALIDATED | G2 everywhere (<= 3e-17), negative control | - | LP (W_-w = W^T), SUPG adjoint inconsistency (Seymen) | externally structurally supported | EASY |
| LPS (--convection-stabilization lps) | YES (T6) | YES | YES (symmetric, even in w) | VALIDATED (with caveats) | T6: matrix gates 35/35, KKT gates, MPI; P0 order 2 | corner undershoots; P0 O(h^2); h_k uses 2 extents | LP 2022 (STRONG for Q1 definition), Burman 2011 (theory) | externally structurally supported | MODERATE |
| high-Pe Galerkin | YES | YES | YES | VALIDATED (behaviour) | T4 map; T6 undershoot location | oscillates on under-resolved meshes | LP figs (structural) | externally structurally supported | EASY |
| high-Pe stabilised path | YES | YES | YES | VALIDATED (with caveats) | T6 | see LPS | LP | externally structurally supported | MODERATE |
| 1D convection | YES | YES (distributed only) | YES | VALIDATED | M25 Ex3/4, 1D gates | - | M25 Ex3 (STRONG) | externally reproducible | n/a |

## 3. Explicit restrictions (every one found) and their nature

Classification: A = mathematically necessary, B = solver-related, C =
implementation-only, D = historical/conservative.

| # | where | restriction | stated reason | assessment |
|---|---|---|---|---|
| R1 | `Config::Validate` | **convection ⇒ DISTRIBUTED control only** | "the boundary-control paths contain reduced operators and factorizations derived for a SYMMETRIC stiffness (EX21_SOLVER_NONSYMMETRY_AUDIT.md); they are refused, not reinterpreted" | **Neumann:** D + B. The control enters only through the symmetric M_Γ; the DTO exact transpose already produces the convective adjoint BC. The generic paths (ADDITIVE / MATCHING / SWEEP) would need only an audit of the Neumann data side. The boundary-reduced Schur (SPD Cholesky) is B. **Dirichlet:** C + B. `DirichletControlOperators::ApplyBDT` reuses `ApplyStep` = `MatMult(stiffness)`, i.e. it applies L where Lᵀ is needed (silently wrong under convection); the reduced Schur is a dense LAPACK Cholesky (B). **Not A in either case.** |
| R2 | `ResolveProblemData` (CLI) | `--convection on` refused with the ex20 literature/manufactured benchmarks and `--verify-stage1/-sweeps/-boundary-reduced` | "they encode a symmetric diffusion operator" | C/D (test-harness scope). They are replaced by `--verify-convection` and `--convection-mms` |
| R3 | CLI | `--vx/--vy` require `--convection on` | silent-ignore guard | consistency check, not a capability limit |
| R4 | Validate | `--wind` and `--convection-stabilization lps` require `--convection on` | consistency | not a capability limit |
| R5 | Validate | `--control-h1-time-weight` requires **Dirichlet** control (hence unreachable with convection, via R1) | distributed and Neumann controls are piecewise-constant *interval* quantities, "whose discrete time derivative is a different object" | C (derivation and implementation not done for interval controls) |
| R6 | CLI | `--pc-schur-boundary-reduced` ⇒ Neumann; `--pc-schur-dirichlet-reduced` ⇒ Dirichlet (unreachable with convection) | small control support / Woodbury | B (both are SPD factorizations) |
| R7 | Validate | inhomogeneous Dirichlet state data ⇒ homogeneous-Dirichlet BC and distributed control | lifting design | C (not convection-specific). Also: time-independent g only, and not CLI-reachable (presets only) |
| R8 | drivers | `--verify-convection`, `--convection-mms`, `--exact-oc` (without `--solve-only`) and the 2D/M25 presets run **serially** | `RequireSerial` | C (harness). The production path is MPI-validated |
| R9 | Validate | homogeneous-Dirichlet state BC incompatible with Neumann/Dirichlet control | the constraint would remove the control (whole boundary) | A for the whole-boundary constraint; C as a design limit (no partial Dirichlet boundary) |
| R10 | Validate | dimension 1 or 2; 1D ⇒ distributed only | QUAD9/EDGE3 meshes | C (3D not implemented) |
| R11 | implicit | velocity: constant (vx, vy) or the two named fields only; time-independent; divergence-free assumed (not enforced) | — | C |
| R12 | CLI | target types `moving_gaussian`, `exp_minus_t_cos2pi` and the source regions are not CLI options (presets/overrides only) | — | C. An invalid `--target-type` aborts via an uncaught `std::invalid_argument` (exit 134) instead of a clean usage error — cosmetic |

## 4. Smoke tests (this audit) — `feature_parity/logs/smoke_*.log`, `smoke_summary.txt`

Settings: nx = 8, Nt = 8, LU sweeps, SCHUR_SWEEP, α_u = 1e-3, target `sin_x_minus_t`.
Multi-zone = two control boxes [0.1, 0.4]² and [0.6, 0.9]², plus two tracking boxes
[0.5, 0.9] × [0.1, 0.5] and [0.1, 0.5] × [0.5, 0.9] (non-colocated).

| test | G2 transpose | wrong-adjoint control | Hessian sym. (wrong) | FD gradient (wrong) | G4 reason / true res. / forward march | verdict |
|---|---|---|---|---|---|---|
| multi-zone, hom. Dirichlet, v = (2,1), ν = 0.5, CN | 3.0e-17 | 1.0e-3 | 3.4e-15 (3.6e-3) | 2.8e-15 (0.65) | 2 / 1.5e-15 / 9.0e-16 | PASS |
| same, BE | 2.1e-18 | 2.0e-4 | 1.8e-15 (2.8e-3) | 7.7e-15 (0.62) | 2 / 1.5e-15 / 2.0e-15 | PASS |
| multi-zone, **natural BC** (outflow, w·n ≠ 0), CN | 4.7e-18 | 6.9e-4 | 6.4e-16 (2.5e-3) | 1.4e-15 (1.59) | 2 / 1.5e-15 / 8.4e-16 | PASS |
| same, BE | 8.5e-18 | 2.1e-4 | 4.1e-15 (9.0e-3) | 3.7e-16 (1.86) | 2 / 1.0e-15 / 7.5e-15 | PASS |
| multi-zone, natural BC, **LPS** (ν = 0.05, 64/64 active), CN | 1.3e-17 | 3.1e-3 | 1.3e-15 (1.6e-2) | 2.2e-14 (0.90) | 2 / 2.4e-15 / 6.3e-16 | PASS |
| full domain, natural BC, **variable wind** q2-divfree, ν = 0.1, CN | 7.6e-18 | 2.1e-3 | 1.5e-14 | 1.1e-15 (0.41) | 2 / 9.1e-16 / 5.3e-16 | PASS |
| **production CLI** (not a verification driver), multi-zone, natural, CN, `mpirun -np 1` / `-np 2` | — | — | — | — | reason 2, 2 outer, 0 sweep failures, accepted; np 2 vs np 1: max\|Δy\| 2.1e-16, max\|Δp\| 7.8e-18, J_track equal to 16 digits | PASS |

Also observed with no LU override (default ADDITIVE_GAMG, natural BC, ν = 0.1):
- q2-divfree wind, control box: converged in 20 outer iterations;
- constant vx = 1, full-domain control and tracking: converged in 74 outer iterations;
- G4 consistent in both.

No sign or transposition issue appeared anywhere. Multi-zone control/tracking and the natural
state BC are now **smoke-validated** with convection. They are not convergence-validated.

## 5. Boundary-control machinery (2D), and what convection changes

| item | Neumann control (`--neumann-solve`) | Dirichlet control (`--dirichlet-solve`) |
|---|---|---|
| control space | traces on the face Γ_c (`neumann_control_face`, default the bottom side), P2 trace nodes; **interval-constant in time** (BE and CN) | nodal traces on C = supp M_Γ; **nodal (P1) in time**, u₀ = y₀\|_Γ/η as data |
| boundary mass | M_Γ (face quadrature, `JacobianSur`) | M_Γ |
| tangential stiffness | not used | K_Γ: 1D arc-length derivative (dφ/dξ)/J |
| regularisation | (α_u/2) Σ dt ‖u_n‖²_{M_Γ} (L2(Γ_c)) | r(u,v) = (u,v)_Γ + ℓ²(∂_s u, ∂_s v)_Γ: R_Γ = M_Γ + ℓ²K_Γ |
| temporal regularisation | none (rejected, R5) | (γ_u/2)∫‖∂_t u‖²_{M_Γ}: R_H1t = Ŵ_u ⊗ R_Γ + ρ T ⊗ M_Γ (dense Cholesky when γ_u > 0) |
| state/control coupling | weak flux: η M_Γ u in the state rows | strong elimination: B_D u = P_I[A Q_C u_n − (M/dt) Q_C u_{n−1}] (BE); CN with A_±; u₀ as data |
| adjoint coupling | M_Γᵀ = M_Γ (symmetric): unaffected by L ≠ Lᵀ | B_Dᵀ p = Q_Cᵀ[A p_n − (M/dt) p_{n+1}]: uses **A, not Aᵀ** |
| exact transpose | space-time adjoint applies Lᵀ (generic shells) | B_Dᵀ does **not** transpose L (`ApplyStep` = `MatMult(stiffness)`) |
| CN | interval controls, trapezoidal tracking (ex20 stage 2) | nodal controls, What_u = diag(1, …, 1, ½) (ex20 stage 3) |
| default solver | ADDITIVE_GAMG; SCHUR_BOUNDARY_REDUCED available (SPD Cholesky) | SCHUR_DIRICHLET_REDUCED (Woodbury + dense potrf) |

**What convection changes, mathematically:**

- **Neumann control.**
  - With the strong-form Galerkin convection, `N + Nᵀ = ∫_∂Ω (w·n) φ_i φ_j` ≠ 0 on any
    boundary with w·n ≠ 0 (Γ_c included).
  - The DTO adjoint (the exact transpose) therefore carries the boundary condition
    ν∂_n p + (w·n) p = 0. This is the Robin-type adjoint condition
    ε∂_n p + (γ + β·n) p = 0 that Seymen derives in the OTD setting (thesis eq. (2.64)).
  - ex21's exact transpose produces it automatically. The control coupling η M_Γ is
    symmetric and unaffected.
  - The flux condition remains ν∂_n y = η u (no convective flux): a modelling choice,
    in contrast to Robin control ν∂_n y + γy = u.
  - Only the SPD boundary-reduced Schur solver is genuinely incompatible. Lifting R1 for
    Neumann + generic PCs would need, at most, an audit of the Neumann data side and of the
    CN Neumann paths for any implicit use of Kᵀ = K.
- **Dirichlet control.**
  - The state rows involve L_IC, while the adjoint coupling must involve (Lᵀ)_CI = (L_IC)ᵀ.
  - The present B_Dᵀ applies L_CI instead, which is wrong whenever L ≠ Lᵀ. It needs a
    transposed `ApplyStep` (L in B_D, Lᵀ in B_Dᵀ).
  - The reduced Schur's dense Cholesky must be replaced by LU, or by the SWEEP/matching
    path.
  - The control-to-state map also feels the inflow/outflow character of Γ_c (w·n). Where
    w·n < 0 (inflow) the Dirichlet control acts like an inflow condition and the solution
    can develop boundary layers.
  - R_Γ, R_H1t and the temporal regularisation are unaffected: they act on the control
    space only.

**Status under convection:** both boundary controls are **implemented but not allowed**,
and therefore **unvalidated**. Neumann is close to convection-ready. Dirichlet needs a code
change (transposed B_Dᵀ) plus a nonsymmetric reduced solver.

## 6. Literature (bounded search; full text read for Seymen 2013 and Burman 2011)

| feature | closest source | dist./bdry | boundary type | time | FEM | AAO? | stabilisation | reproducible in ex21? | value | class |
|---|---|---|---|---|---|---|---|---|---|---|
| distributed control (full domain) | Leveque & Pearson, NLAA 29:e2419 (2022) §5.1-5.2 | distributed | Dirichlet state (inhomogeneous in 5.2) | BE, CN | Q1 | YES (OTD, MINRES + matching Schur) | LPS (P0 patches) | YES: 5.1 exact (heat), 5.2 structural (done T1-T6) | primary | STRONG MATCH |
| distributed control (full domain, constrained) | Seymen PhD thesis METU 2013, Ch. 4.1; 'Distributed optimal control of time-dependent diffusion-convection-reaction equations using space-time discretization', J. Comput. Appl. Math. (2014), doi via S0377042713006250 | distributed | homogeneous Dirichlet | BE, CN, semi-implicit | P1 + SUPG | YES (DTO, MINRES, idealised block-diag PC) | SUPG (adjoint-inconsistent DTO) | NO directly: all DTO examples are control-constrained (u >= 0, 0 <= u <= 0.5); unconstrained variants would be new manufactured problems | structural (orders O(h^3/2) for SUPG P1, not comparable to Q2) | PARTIAL MATCH |
| distributed control (theory, symmetric stabilisation) | Burman, Commun. Math. Sci. 9 (2011) 319-329 | distributed (space-time) | Dirichlet (Nitsche) | CN | P1 + CIP | theory only | CIP (adjoint-consistent) | NO (OC part has no numerics; only a forward transport example) | supports exact-transpose + symmetric stabilisation design | BACKGROUND ONLY |
| subdomain / non-colocated control & tracking | Aulisa-Chierici-Gilliam, Mathematics 13:3707 (2025) Ex 3-5; ErrorFeedback 2026 Ex 9.5-9.6 | distributed (subdomain) | homogeneous Dirichlet | source: continuous/cascade; ex21: BE/CN | - | source: no (cascade/feedback) | none (low Pe) | YES (done: Ex3 gold standard, Ex5 periodic, EF tail scale) | primary for subdomain/non-colocated | STRONG MATCH (Ex3, Ex5) / PARTIAL (EF) |
| distributed control, constrained, DG | Akman-Yucel-Karasozen, COAP (2014) upwind SIPG | distributed | Dirichlet | BE / dG(0) | SIPG upwind | - | upwind DG | NO (constrained, DG) | background | BACKGROUND ONLY |
| Neumann boundary control | Seymen 2013 Ch. 2.3 / 3.2 (Ex 3.2.1, 1D), Seymen & Karasozen IJMMNO 4 (2013) 282-300 | boundary | ROBIN eps dn y + gamma y = u on Sigma (1D: both ends, gamma = 0.1, two controls) | space-time one-shot (OTD, biharmonic in (x,t)) | COMSOL Lagrange + SUPG | space-time one-shot (not a time-stepping AAO) | SUPG | NO (Robin, OTD biharmonic, 1D both-end controls, COMSOL) | structural: adjoint BC eps dn p + (gamma + beta.n) p = 0 | PARTIAL MATCH (Robin, not Neumann) |
| Neumann boundary control, AAO preconditioning | arXiv 2408.04954 (2024) 'Some notes concerning preconditioning of linear parabolic optimal control problems' | boundary | Neumann | implicit | conforming FEM | YES | none (heat) | heat only | solver background | BACKGROUND ONLY |
| Dirichlet boundary control | Seymen 2013 Ex 4.2.1 | boundary | ROBIN with gamma = 1e3 on whole dOmega (penalised Dirichlet), Omega = (0,1)x(0,0.2) | CN | P1 + SUPG | YES (DTO, block-diag PC, lumped masses) | SUPG | NO (penalty Robin, y_d / T not stated, no exact solution; only cost-functional convergence) | background | BACKGROUND ONLY (not true Dirichlet control) |
| Dirichlet boundary control (convection, steady) | 'A new HDG method for Dirichlet boundary control of convection diffusion PDEs' I/II (arXiv 1801.01461, 1801.01056) | boundary | true Dirichlet, L2 regularisation | steady | HDG | n/a | HDG | NO (steady, HDG) | background | BACKGROUND ONLY |
| Dirichlet boundary control (parabolic, heat) | arXiv 1410.0136 (FE method for Dirichlet boundary control of parabolic PDEs) | boundary | true Dirichlet | implicit | FEM | - | none | heat only (ex20 territory) | background | BACKGROUND ONLY |
| time-dependent Dirichlet data (conv-diff, control design) | MPI Magdeburg preprint MPIMD15-03 'Time-dependent Dirichlet conditions in finite element discretizations' | boundary input g(x)u(t) | Dirichlet input | - | FEM | - | - | NO | background | BACKGROUND ONLY |
| H1-in-time regularisation | (none found in a bounded search for convection-diffusion) | - | - | - | - | - | - | - | internal (ex20 Dirichlet stage, heat) | NONE FOUND |
| multi-zone control / tracking | (none found; M25/EF use single Omega_B, Omega_C, Omega_d) | - | - | - | - | - | - | - | ex21 capability | NONE FOUND |
| LPS adjoint-consistent stabilisation | Leveque & Pearson 2022 §2.1; Becker & Vexler (2007, steady); Braack (2009, steady) | distributed | Dirichlet | BE/CN | Q1 LPS | YES (LP) | LPS | structurally (done T6) | definition source | STRONG MATCH (definition) / PARTIAL (Q1 vs Q2) |

**Seymen 2013 thesis (full text):**

- **State:** y_t − εΔy + β·∇y + σy = f + u, homogeneous Dirichlet (distributed);
  J = ½‖y − y_d‖²_Q + (α/2)‖u‖²_Q, box constraints.
- **Ch. 4 DTO "All-at-Once method":** saddle point [A Bᵀ; B 0] over all time steps;
  MINRES with an idealised block-diagonal preconditioner (exact Schur); P1 + SUPG
  (τ ~ h, or ~ k); θ-scheme (BE, CN) and semi-implicit.
- **Distributed examples 4.1.1–4.1.3:** Ω = (0,1)², ε = 1e-5, β = (1,0) or (0.5,0.5);
  exact solutions sin(2πx₁)sin(2πx₂)e^{−t}, …; **all control-constrained**
  (u = max{−p, 0}; 0 ≤ u ≤ 0.5); observed L2 orders ≈ h^{3/2} (CN), h^{4/3} (BE).
- **Boundary control (Ch. 2.3, 3.2, 4.2):** **Robin** control ε∇y·n + γy = u on the whole
  Σ, L2(Σ) regularisation.
  - 1D OTD Ex 3.2.1: Robin at both ends, γ = 0.1, two controls, COMSOL one-shot
    biharmonic in (x, t).
  - 2D DTO Ex 4.2.1: Ω = (0,1) × (0,0.2), ε = 1e-3, σ = 1, β = (1,0), γ = 10³
    (a penalty approximation of Dirichlet control), α = 1e-4, CN; no exact solution;
    only the cost-functional convergence is reported (Table 4.10: 4.4e-2 → 4.2e-5).
  - y_d and T are not stated in the text: generated from an elliptic OCP in its ref. [3].
- **The IJMMNO 2013 boundary paper** ("Optimal boundary control for time-dependent
  diffusion-convection-reaction equations", Seymen & Karasözen) is, per its abstract, the
  OTD space-time biharmonic + SUPG + COMSOL approach, i.e. thesis Ch. 3.2. Its boundary
  condition is therefore the Robin condition above. The paper itself was not fetched; the
  thesis is the same work in full.

**Answers to §8:**

- **A. Distributed control.** Beyond LP 2022 the closest AAO transient convection–diffusion
  sources are Seymen 2013 / JCAM 2014 (DTO all-at-once, P1-SUPG, BE/CN). Every DTO example
  there is control-constrained, which ex21 does not support, so they are
  structural / PARTIAL. Burman 2011 is theory only.
  - Most reproducible with ex21 (already done): LP 5.1 (heat, exact), LP 5.2 (structural),
    M25 Ex3 (gold standard, 1D) and Ex5 (2D periodic regime).
  - A *possible* addition: Seymen's exact solution of Ex 4.1.1 (β = (1,0)) with the
    constraint dropped (u = −p/α). That is a new manufactured problem, not a reproduction.
- **B. Neumann boundary control.** No transient convection–diffusion AAO benchmark with true
  Neumann control was found. Seymen uses Robin (γ > 0) on the whole boundary, and in 1D an
  OTD space-time one-shot. The structural support is the adjoint Robin condition
  (γ + β·n)p.
- **C. Dirichlet boundary control.** The only transient convection example found (Seymen
  4.2.1) is **Robin with γ = 10³, a penalty approximation, not true Dirichlet control**.
  True Dirichlet control with convection appears only in steady HDG work. Transient
  Dirichlet control appears only for the heat equation. Background only.
- **D. H1-in-time regularisation.** No convection–diffusion reference was found. It is
  **primarily internally validated functionality** (ex20 Dirichlet stage, heat equation,
  `aux/dirichlet_h1_time/derivation_check.py`), and is not available with convection.
- **E. Multi-zone control/tracking.** No benchmark was found. M25/EF use single control,
  observation and disturbance zones. Multi-zone is an **ex21 capability**: ex20 goldens
  without convection, smoke gates with convection (this audit).

## 7. External-status summary

| status | features |
|---|---|
| **externally reproducible** | full-domain distributed control and tracking (LP); subdomain + non-colocated control/tracking (M25 Ex3/Ex5, EF); BE, CN, L2 regularisation, homogeneous Dirichlet state, constant wind, SCHUR_SWEEP (LP matching Schur), 1D convection (M25 Ex3), single source region (M25/EF) |
| **externally structurally supported** | variable wind and inhomogeneous Dirichlet data (LP 5.2), exact-transpose adjoint (LP W₋w = Wᵀ; Seymen on SUPG inconsistency), LPS (LP definition, Q1 vs Q2), high-Pe Galerkin/stabilised behaviour (LP figures) |
| **internally validated only** | multi-zone control/tracking (smoke + ex20 goldens), natural state BC with convection (smoke), ADDITIVE_GAMG / SCHUR_MATCHING under convection, LU/ILU/ASM/AIR/GAMG+LPS backends, MPI production path, whole-domain manufactured sources |
| **currently unvalidated** (with convection) | Neumann control, Dirichlet control, H1-space and H1-time regularisation, boundary-/Dirichlet-reduced Schur, multiple source regions |

## 8. Pre-3D difficulty (per feature; column "3D" in §2)

- **EASY:**
  - distributed volume control and tracking;
  - full-domain regions;
  - homogeneous Dirichlet state; natural BC;
  - BE/CN, L2 regularisation;
  - constant wind (w[2] → w[3]);
  - exact transpose (dimension-free);
  - SCHUR_SWEEP, ADDITIVE, MATCHING;
  - the acceptance guard and MPI path.

  The KKT/space-time layer is dimension-free.
- **EASY/MODERATE:**
  - boxes and multi-zone: `Box`, the centroid tests and the layout carry only (x, z);
  - in 2D the code calls the *second* coordinate **z** (356 references). 3D needs
    (x, y, z) naming or a third member, which is a mechanical but wide change.
- **MODERATE:**
  - inhomogeneous lifting (data functions are (x, z));
  - variable winds (the named fields are 2D formulas);
  - LPS (h_k uses two extents; the patch chord generalises to 3D directly);
  - Neumann control (surface mass on a 3D face: `JacobianSur` exists for faces);
  - high-Pe backends (3D LU memory);
  - exact-OC/MMS data (2D formulas).
- **NONTRIVIAL:**
  - tangential H1(Γ) regularisation: the 2D arc-length derivative must become a surface
    gradient on faces;
  - Dirichlet control under convection, even in 2D (transposed B_Dᵀ; nonsymmetric reduced
    solver).
- **DEFER:**
  - Dirichlet control in 3D;
  - H1-in-time (dimension-independent once a control operator exists, but tied to Dirichlet
    control);
  - boundary-/Dirichlet-reduced Schur;
  - hypre AIR (np = 4 crash);
  - M25/EF/LP presets (2D benchmarks).

## 9. Recommended first 3D scope

**Include (phase 3D-1):**

1. Q2 **HEX27** box mesh (FEMuS `GenerateCoarseBoxMesh(nx, ny, nz, …, HEX27, "seventh")`),
   with a third coordinate wired through `Domain`, the layout and the data callbacks.
   Decide the (x, y, z) naming *first*, because the 2D code's "z" is the second coordinate.
2. **Distributed control** on the **whole domain**, with **full-domain tracking**.
3. **Homogeneous Dirichlet state** BC.
4. **v = 0** sanity case (heat, reproducing the dimension-independent ex20 algebra), then a
   **constant nonzero wind** (vx, vy, vz).
5. **BE and CN.**
6. **Exact OC:** a 3D analogue of `conv-sin3` (product of sines, the ex21 manufactured
   construction) at low/moderate Pe; orders 3 / 2 / 1.
7. Gates: `--verify-convection` G1–G4 (orientation, transpose, Hessian, FD gradient,
   production solve) on nx = 4; the acceptance guard; MPI np = 1/2/4 on the production path.
8. Solvers: SCHUR_SWEEP with **MUMPS LU** (reference) and **ASM+ILU** (MPI baseline) on the
   sweep prefixes; default GAMG only at low Pe.

**Optionally, at the end of phase 1 (easy once the coordinate change is done):** box
subdomain control/tracking and multi-zone. These are EASY/MODERATE and smoke-validated in
2D, but they add geometry code to the first change.

**Do NOT include in the first 3D phase:**

- Neumann and Dirichlet boundary control;
- H1(Γ) and H1-in-time regularisation;
- boundary-/Dirichlet-reduced Schur;
- inhomogeneous Dirichlet lifting;
- variable/named winds;
- LPS;
- high-Pe runs;
- hypre/AIR;
- the M25/EF/LP presets;
- the 1D path;
- source regions.

None of these is needed to validate the 3D discretisation and the KKT machinery, and each
adds its own validation burden. LPS in particular showed Q2-specific order and corner
issues in 2D (T6).

## 10. Should 3D remain in ex21?

**Yes.**

- ex21 already carries the dimension switch (1D/2D) and a dimension-free space-time/KKT and
  solver layer, the convection-aware adjoint, the acceptance guard and the gates.
- A separate application would duplicate all of it.
- The 3D extension should be behind `--dimension 3` with an explicit Validate() whitelist
  (distributed + homogeneous Dirichlet/natural + constant wind, no LPS / boundary control /
  presets). Every other combination should be rejected until it is validated: the same
  "refuse, don't reinterpret" policy ex21 already uses.
- The default 2D path must stay byte-identical (goldens).

---

## Post-T7 closure note (appended after T7N, T7D, T7F; the audit above is historical and unchanged)

The rows above reflect the state **before** T7N/T7D/T7F. The current status is in
`feature_parity_final.csv`, and the closure evidence is in
`T7F_FINAL_2D_FEATURE_CLOSURE.md`. In summary:

- **Neumann boundary control + convection:** VALIDATED (T7N,
  `boundary_convection/T7N_NEUMANN_CONVECTION.md`), with SCHUR_SWEEP. R1 was lifted for
  Neumann control; boundary-reduced Schur is still refused under convection.
- **Dirichlet boundary control + convection:** VALIDATED (T7D,
  `boundary_convection/T7D_DIRICHLET_CONVECTION.md`):
  - B_Dᵀ is now the exact transpose;
  - the Dirichlet-reduced solver is convection-aware;
  - L2, H1-space, H1-time and combined regularisation are validated, the last two by
    gates and self-convergence.
- **Multi-zone control/tracking:** SMOKE → VALIDATED (tracking), and VALIDATED WITH
  DOCUMENTED LIMITATION (control: the state's order drops at control-box interfaces)
  (T7F).
- **Multiple source regions:** IMPLEMENTED-UNVALIDATED → VALIDATED WITH DOCUMENTED
  LIMITATION (T7F). They are now CLI-reachable through `--source-box-add`; R12 is
  lifted for source regions.
- **Natural state BC + convection:** SMOKE → VALIDATED (T7F, exact OC `natural-conv`
  exercising ν∂ₙp + (w·n)p = 0).

**Verdict (T7F §12): 2D CONVECTION FEATURE SET: CLOSED.**
