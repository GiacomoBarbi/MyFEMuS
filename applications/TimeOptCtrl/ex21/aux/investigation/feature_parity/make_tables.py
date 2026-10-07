#!/usr/bin/env python3
"""Feature-parity audit tables (single source of truth) -> ../feature_parity.csv,
../literature_feature_map.csv.  Status words:
  implemented: YES / PARTIAL (code exists but not reachable from the CLI, presets only)
  allowed_conv: YES / NO (rejected by Validate/CLI under --convection on) / N/A
  validated_conv: VALIDATED / SMOKE (tiny gates only, this audit) / IMPLEMENTED-UNVALIDATED / NO
  ext_class: externally reproducible / externally structurally supported /
             internally validated only / currently unvalidated"""
import csv, os
HERE = os.path.dirname(os.path.abspath(__file__))
INV = os.path.dirname(HERE)
F = [
# group, feature, implemented, allowed_conv, conv_compatible, validated_conv, evidence, limitation, literature, ext_class, d3
("A control", "full-domain distributed control", "YES", "YES", "YES", "VALIDATED",
 "LP T1-T6 (lp-heat exact, lp52, LPS); exact-OC conv-sin3/lift-*; verify-convection G2-G4", "-",
 "LP 2022 (STRONG); Seymen 2013 thesis / JCAM 2014 (PARTIAL: control-constrained); Burman 2011 (theory)", "externally reproducible", "EASY"),
("A control", "subdomain (box) distributed control", "YES", "YES", "YES", "VALIDATED",
 "M25 Ex3/Ex4 (1D), Ex5, EF 9.5/9.6 (2D); run_all_gates control box 0.25-0.75 (G2-G4, 3 PCs)", "box selection by element centroid; axis-aligned boxes only",
 "M25 Ex3 (STRONG, source-OC reproduction), Ex5 (STRONG periodic regime), EF 9.5/9.6 (PARTIAL, cross-method)", "externally reproducible", "EASY (Box needs a 3rd extent)"),
("A control", "multiple distributed-control regions (--control-box-add)", "YES", "YES", "YES", "SMOKE",
 "this audit: multizone_{dirichlet,natural}_{cn,be}, +LPS: all gates pass; ex20 goldens (no convection)", "union of boxes, lowest index wins on overlap",
 "NONE FOUND (bounded search)", "internally validated only", "EASY/MODERATE"),
("A control", "Neumann boundary control", "YES (ex20 stages 1-2, BE/CN)", "NO (Validate: convection => DISTRIBUTED only)",
 "YES mathematically (DTO exact transpose yields the adjoint BC nu dn p + (w.n) p = 0 automatically); generic PCs plausible; boundary-reduced Schur NOT (SPD Cholesky)",
 "NO", "never run with convection", "single face (neumann_control_face); L2(Gamma_c) regularisation only; boundary-reduced solver SPD-only",
 "Seymen 2013 (Robin eps dn y + gamma y = u, PARTIAL); parabolic Neumann-control AAO preconditioning (heat, BACKGROUND)", "currently unvalidated", "MODERATE (surface mass on a 3D face)"),
("A control", "Dirichlet boundary control", "YES (ex20 stage 3, BE/CN, H1(Gamma) + H1-time)", "NO (Validate)",
 "NOT as implemented: B_D and B_D^T both apply MatMult(stiffness) (ApplyStep), i.e. L not L^T; reduced Schur = dense LAPACK Cholesky (SPD); R-Cholesky fine",
 "NO", "never run with convection", "needs a transposed ApplyStep for B_D^T and a nonsymmetric reduced solver",
 "Seymen 2013 Ex 4.2.1 is Robin with gamma = 1e3 (penalty, BACKGROUND); steady HDG Dirichlet control of conv-diff (BACKGROUND)", "currently unvalidated", "NONTRIVIAL/DEFER"),
("B tracking", "full-domain tracking", "YES", "YES", "YES", "VALIDATED", "LP T1-T6; exact-OC cases", "-", "LP 2022 (STRONG); Seymen (PARTIAL)", "externally reproducible", "EASY"),
("B tracking", "subdomain tracking", "YES", "YES", "YES", "VALIDATED", "M25 Ex3/5, EF 9.5/9.6 (Omega_C); gates tracking box", "centroid selection", "M25/EF (STRONG/PARTIAL)", "externally reproducible", "EASY"),
("B tracking", "multiple tracking regions (--tracking-box-add)", "YES", "YES", "YES", "SMOKE", "this audit (multizone_*): all gates pass", "-", "NONE FOUND", "internally validated only", "EASY/MODERATE"),
("B tracking", "non-colocated control/tracking", "YES", "YES", "YES", "VALIDATED", "M25 Ex3 (RMS 6.2e-6 vs source OC), Ex5, EF 9.6; smoke multizone", "-", "M25 Ex3/Ex5 (STRONG)", "externally reproducible", "EASY"),
("B tracking", "time-dependent target", "YES (CLI: constant, sin_x_minus_t, one_minus_y_sin_x_minus_t; presets/overrides: more)", "YES", "YES", "VALIDATED",
 "gates (sin_x_minus_t), M25/EF signals, LP data", "moving_gaussian / exp_minus_t_cos2pi are preset-only (CLI rejects them)", "M25/EF/LP", "externally reproducible", "EASY (formulas are 2D)"),
("C data/state", "source / disturbance (whole domain)", "YES", "YES", "YES", "VALIDATED", "exact-OC manufactured sources (T2/T3), MMS", "-", "LP/Seymen manufactured", "internally validated only", "EASY"),
("C data/state", "source regions (subdomain disturbance)", "PARTIAL (presets only, no CLI)", "YES", "YES", "VALIDATED (single region)",
 "M25 Ex3-5, EF 9.5/9.6 (one Omega_d each)", "not CLI-reachable; >1 region never exercised", "M25/EF (STRONG/PARTIAL)", "externally reproducible (single region)", "EASY/MODERATE"),
("C data/state", "multiple source regions", "PARTIAL (std::vector, presets only)", "YES", "YES", "IMPLEMENTED-UNVALIDATED", "never used with >1 region", "not CLI-reachable", "NONE FOUND", "currently unvalidated", "EASY"),
("C data/state", "homogeneous Dirichlet state BC", "YES", "YES", "YES", "VALIDATED", "LP, exact-OC, T2 gates, CN trace fix (audit #21)", "whole external boundary only", "LP, Seymen, M25/EF", "externally reproducible", "EASY"),
("C data/state", "inhomogeneous Dirichlet state BC (lifting)", "PARTIAL (data_overrides; exact-OC presets only)", "YES", "YES", "VALIDATED",
 "T3/T3b: lift-const/exp/track, lp52; G3P physical gradient; 34/34", "time-independent g only; distributed control only; not CLI-reachable", "LP 5.2 (STRONG, structural)", "externally structurally supported", "MODERATE"),
("C data/state", "natural (homogeneous Neumann) state BC", "YES", "YES",
 "YES (strong-form convection: nu dn y = 0 on dOmega; N not skew where w.n != 0; exact transpose still the DTO adjoint)", "SMOKE",
 "G1 orientation (pre-existing); this audit: multizone_natural_{cn,be}, +LPS, variable wind q2-divfree: all gates pass", "no convective-flux (Robin) inflow/outflow BC", "Seymen Robin adjoint BC (gamma + beta.n) (STRUCTURAL)", "internally validated only", "EASY"),
("D time/reg", "backward Euler", "YES", "YES", "YES", "VALIDATED", "orders 1 (T2, T6), gates, LP", "-", "Seymen BE (PARTIAL), LP", "externally reproducible", "EASY"),
("D time/reg", "Crank-Nicolson", "YES", "YES", "YES", "VALIDATED", "LP algebra 49/49 (T0), order 2 (T2, T6)", "interval-constant controls (distributed)", "LP 2022 (STRONG), Burman 2011 (theory), Seymen (PARTIAL)", "externally reproducible", "EASY"),
("D time/reg", "L2 control regularisation", "YES", "YES", "YES", "VALIDATED", "everything above", "-", "all sources", "externally reproducible", "EASY"),
("D time/reg", "H1-in-space boundary-control regularisation", "YES (Dirichlet control only)", "NO (Dirichlet control rejected)", "n/a until Dirichlet control is convection-ready", "NO",
 "ex20 Dirichlet stages (heat)", "curve (1D) tangential derivative, arc-length Jacobian", "NONE FOUND for convection", "currently unvalidated (with convection)", "NONTRIVIAL (surface gradient)"),
("D time/reg", "H1-in-time control regularisation", "YES (Dirichlet control only; Validate rejects it otherwise)", "NO", "derivation exists only for nodal Dirichlet controls", "NO",
 "ex20 aux/dirichlet_h1_time derivation check (heat)", "distributed/Neumann interval controls need a different discrete d/dt", "NONE FOUND for convection", "internally validated only (heat)", "dimension-independent once the control operator exists; DEFER"),
("D time/reg", "combined H1(space) + H1(time)", "YES (Dirichlet only)", "NO", "as above", "NO", "ex20 (heat)", "Dirichlet only", "NONE FOUND", "currently unvalidated (with convection)", "DEFER"),
("E solvers", "SCHUR_SWEEP", "YES", "YES", "YES (GMRES spatial, transposed backward sweep)", "VALIDATED", "gates, LP T4-T6, M25/EF; 2 outer its", "default spatial GAMG fails at high Pe", "LP matching Schur (STRONG)", "externally reproducible", "EASY (cost)"),
("E solvers", "ADDITIVE_GAMG (distributed default)", "YES", "YES", "YES (preconditioner only)", "VALIDATED (low Pe)", "run_all_gates; smoke (natural BC, variable wind: 20-74 outer its)", "outer iterations grow with Pe", "-", "internally validated only", "EASY"),
("E solvers", "SCHUR_MATCHING", "YES", "YES", "YES", "VALIDATED (low Pe)", "run_all_gates 3 PCs", "-", "LP", "internally validated only", "EASY"),
("E solvers", "boundary-reduced Schur", "YES", "NO (Neumann only; SPD Cholesky)", "NO as implemented", "NO", "-", "SPD-specific", "-", "currently unvalidated", "DEFER"),
("E solvers", "Dirichlet-reduced Schur", "YES", "NO (Dirichlet only; dense potrf)", "NO as implemented", "NO", "-", "SPD-specific", "-", "currently unvalidated", "DEFER"),
("E solvers", "LU / MUMPS override (PETSC_OPTIONS sweep_*)", "YES", "YES", "YES", "VALIDATED", "T4-T6; MPI np 1/2/4 (T5, T6)", "memory in 3D", "-", "internally validated only", "EASY (memory)"),
("E solvers", "high-Pe spatial backends (ILU, ASM+ILU, hypre AIR, GAMG+LPS)", "YES (options)", "YES", "YES", "VALIDATED (2D)", "T5 (ILU/ASM), T5b (AIR serial; np 4 crash), T6 (GAMG+LPS)", "AIR np=4 SEGV; ILU serial", "-", "internally validated only", "MODERATE"),
("E solvers", "MPI production path", "YES", "YES", "YES", "VALIDATED", "T5, T6 MPI checks; this audit np 1/2 (2e-16)", "verification drivers serial-only", "-", "internally validated only", "EASY"),
("F convection", "constant wind", "YES", "YES", "YES", "VALIDATED", "G1-G4, T2 orders, M25/EF", "-", "LP/Seymen/M25/EF", "externally reproducible", "EASY (w[2] -> w[3])"),
("F convection", "variable wind", "YES (named fields lp-recirculating, q2-divfree)", "YES", "YES", "VALIDATED", "T3 MMS/OC orders, LP 5.2", "no user-defined field; time-independent; div-free assumed", "LP 5.2 (STRONG)", "externally structurally supported", "MODERATE (3D fields)"),
("F convection", "exact transpose adjoint", "YES", "YES", "YES", "VALIDATED", "G2 everywhere (<= 3e-17), negative control", "-", "LP (W_-w = W^T), SUPG adjoint inconsistency (Seymen)", "externally structurally supported", "EASY"),
("F convection", "LPS (--convection-stabilization lps)", "YES (T6)", "YES", "YES (symmetric, even in w)", "VALIDATED (with caveats)", "T6: matrix gates 35/35, KKT gates, MPI; P0 order 2", "corner undershoots; P0 O(h^2); h_k uses 2 extents", "LP 2022 (STRONG for Q1 definition), Burman 2011 (theory)", "externally structurally supported", "MODERATE"),
("F convection", "high-Pe Galerkin", "YES", "YES", "YES", "VALIDATED (behaviour)", "T4 map; T6 undershoot location", "oscillates on under-resolved meshes", "LP figs (structural)", "externally structurally supported", "EASY"),
("F convection", "high-Pe stabilised path", "YES", "YES", "YES", "VALIDATED (with caveats)", "T6", "see LPS", "LP", "externally structurally supported", "MODERATE"),
("F convection", "1D convection", "YES", "YES (distributed only)", "YES", "VALIDATED", "M25 Ex3/4, 1D gates", "-", "M25 Ex3 (STRONG)", "externally reproducible", "n/a"),
]
L = [
# feature, closest source, dist/boundary, boundary type, time, FEM, AAO, stabilisation, reproducible in ex21, value, class
("distributed control (full domain)", "Leveque & Pearson, NLAA 29:e2419 (2022) §5.1-5.2", "distributed", "Dirichlet state (inhomogeneous in 5.2)", "BE, CN", "Q1", "YES (OTD, MINRES + matching Schur)", "LPS (P0 patches)", "YES: 5.1 exact (heat), 5.2 structural (done T1-T6)", "primary", "STRONG MATCH"),
("distributed control (full domain, constrained)", "Seymen PhD thesis METU 2013, Ch. 4.1; 'Distributed optimal control of time-dependent diffusion-convection-reaction equations using space-time discretization', J. Comput. Appl. Math. (2014), doi via S0377042713006250", "distributed", "homogeneous Dirichlet", "BE, CN, semi-implicit", "P1 + SUPG", "YES (DTO, MINRES, idealised block-diag PC)", "SUPG (adjoint-inconsistent DTO)", "NO directly: all DTO examples are control-constrained (u >= 0, 0 <= u <= 0.5); unconstrained variants would be new manufactured problems", "structural (orders O(h^3/2) for SUPG P1, not comparable to Q2)", "PARTIAL MATCH"),
("distributed control (theory, symmetric stabilisation)", "Burman, Commun. Math. Sci. 9 (2011) 319-329", "distributed (space-time)", "Dirichlet (Nitsche)", "CN", "P1 + CIP", "theory only", "CIP (adjoint-consistent)", "NO (OC part has no numerics; only a forward transport example)", "supports exact-transpose + symmetric stabilisation design", "BACKGROUND ONLY"),
("subdomain / non-colocated control & tracking", "Aulisa-Chierici-Gilliam, Mathematics 13:3707 (2025) Ex 3-5; ErrorFeedback 2026 Ex 9.5-9.6", "distributed (subdomain)", "homogeneous Dirichlet", "source: continuous/cascade; ex21: BE/CN", "-", "source: no (cascade/feedback)", "none (low Pe)", "YES (done: Ex3 gold standard, Ex5 periodic, EF tail scale)", "primary for subdomain/non-colocated", "STRONG MATCH (Ex3, Ex5) / PARTIAL (EF)"),
("distributed control, constrained, DG", "Akman-Yucel-Karasozen, COAP (2014) upwind SIPG", "distributed", "Dirichlet", "BE / dG(0)", "SIPG upwind", "-", "upwind DG", "NO (constrained, DG)", "background", "BACKGROUND ONLY"),
("Neumann boundary control", "Seymen 2013 Ch. 2.3 / 3.2 (Ex 3.2.1, 1D), Seymen & Karasozen IJMMNO 4 (2013) 282-300", "boundary", "ROBIN eps dn y + gamma y = u on Sigma (1D: both ends, gamma = 0.1, two controls)", "space-time one-shot (OTD, biharmonic in (x,t))", "COMSOL Lagrange + SUPG", "space-time one-shot (not a time-stepping AAO)", "SUPG", "NO (Robin, OTD biharmonic, 1D both-end controls, COMSOL)", "structural: adjoint BC eps dn p + (gamma + beta.n) p = 0", "PARTIAL MATCH (Robin, not Neumann)"),
("Neumann boundary control, AAO preconditioning", "arXiv 2408.04954 (2024) 'Some notes concerning preconditioning of linear parabolic optimal control problems'", "boundary", "Neumann", "implicit", "conforming FEM", "YES", "none (heat)", "heat only", "solver background", "BACKGROUND ONLY"),
("Dirichlet boundary control", "Seymen 2013 Ex 4.2.1", "boundary", "ROBIN with gamma = 1e3 on whole dOmega (penalised Dirichlet), Omega = (0,1)x(0,0.2)", "CN", "P1 + SUPG", "YES (DTO, block-diag PC, lumped masses)", "SUPG", "NO (penalty Robin, y_d / T not stated, no exact solution; only cost-functional convergence)", "background", "BACKGROUND ONLY (not true Dirichlet control)"),
("Dirichlet boundary control (convection, steady)", "'A new HDG method for Dirichlet boundary control of convection diffusion PDEs' I/II (arXiv 1801.01461, 1801.01056)", "boundary", "true Dirichlet, L2 regularisation", "steady", "HDG", "n/a", "HDG", "NO (steady, HDG)", "background", "BACKGROUND ONLY"),
("Dirichlet boundary control (parabolic, heat)", "arXiv 1410.0136 (FE method for Dirichlet boundary control of parabolic PDEs)", "boundary", "true Dirichlet", "implicit", "FEM", "-", "none", "heat only (ex20 territory)", "background", "BACKGROUND ONLY"),
("time-dependent Dirichlet data (conv-diff, control design)", "MPI Magdeburg preprint MPIMD15-03 'Time-dependent Dirichlet conditions in finite element discretizations'", "boundary input g(x)u(t)", "Dirichlet input", "-", "FEM", "-", "-", "NO", "background", "BACKGROUND ONLY"),
("H1-in-time regularisation", "(none found in a bounded search for convection-diffusion)", "-", "-", "-", "-", "-", "-", "-", "internal (ex20 Dirichlet stage, heat)", "NONE FOUND"),
("multi-zone control / tracking", "(none found; M25/EF use single Omega_B, Omega_C, Omega_d)", "-", "-", "-", "-", "-", "-", "-", "ex21 capability", "NONE FOUND"),
("LPS adjoint-consistent stabilisation", "Leveque & Pearson 2022 §2.1; Becker & Vexler (2007, steady); Braack (2009, steady)", "distributed", "Dirichlet", "BE/CN", "Q1 LPS", "YES (LP)", "LPS", "structurally (done T6)", "definition source", "STRONG MATCH (definition) / PARTIAL (Q1 vs Q2)"),
]
with open(os.path.join(INV, "feature_parity.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["group", "feature", "implemented", "allowed_with_convection", "convection_compatible", "validated_with_convection",
                "validation_evidence", "known_limitation", "literature_analogue", "external_status", "3D_difficulty"])
    w.writerows(F)
with open(os.path.join(INV, "literature_feature_map.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["feature", "closest_source", "distributed_or_boundary", "boundary_type", "time_scheme", "FEM", "all_at_once",
                "stabilisation", "benchmark_reproducible_in_ex21", "value_for_validation", "match_class"])
    w.writerows(L)
def md(rows, idx, hdr):
    out = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in rows:
        out.append("| " + " | ".join(str(r[i]).replace("|", "/") for i in idx) + " |")
    return "\n".join(out)
FEATURE_MD = (md(F, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    ["feature", "implemented?", "allowed with convection?", "convection-compatible?", "validated with convection?", "evidence", "known limitation", "literature analogue", "external status", "3D"]))
LIT_MD = (md(L, list(range(11)),
    ["feature", "closest source", "dist./bdry", "boundary type", "time", "FEM", "AAO?", "stabilisation", "reproducible in ex21?", "value", "class"]))
# refresh the two tables inside ../EX21_FEATURE_PARITY_AUDIT.md (between the section headings)
rep = os.path.join(INV, "EX21_FEATURE_PARITY_AUDIT.md")
if os.path.exists(rep):
    import re
    t = open(rep).read()
    t = re.sub(r"(## 2\. Definitive capability table\n\n).*?(\n\n## 3\.)", lambda m: m.group(1) + FEATURE_MD + m.group(2), t, flags=re.S)
    t = re.sub(r"(## 6\. Literature[^\n]*\n\n).*?(\n\n\*\*Seymen 2013)", lambda m: m.group(1) + LIT_MD + m.group(2), t, flags=re.S)
    open(rep, "w").write(t)
print(len(F), "features,", len(L), "literature rows")
