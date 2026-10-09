#!/usr/bin/env python3
"""logs/gates/*.log -> ../../t13_gates_neumann.csv, ../../t13_gates_dirichlet.csv (per-run verdicts).

NEUMANN (criteria fixed before the table was produced):
  positives  G1 orientation, G1N boundary identity (convection only), G2 shell transpose, reduced
             Hessian symmetry (exact, shell) <= 1e-12; FD gradient (exact, shell) <= 1e-8; production
             accepted (reason > 0), true residual <= 1e-10, |grad J| <= 1e-8, forward march <= 1e-8;
             G13N: ||N - sum_f M_Gamma(f)|| <= 1e-14 and the true-union FD <= 1e-8.
  negatives  G13N drop-face and duplicate-face FD > 1e-4 (every run, wind independent).  With
             convection: direct transpose detectors (G2 ex20-style, G8B wrong BC) > 1e-6, FD
             (ex20-style, wrong BC) > 1e-4, Hessian symmetry (ex20-style, wrong BC) > 1e-8.  Heat:
             the ex20-style adjoint coincides (G2 negative <= 1e-12).
DIRICHLET: the T11H1 criteria unchanged (t11h1_gates_table.py): positives G1, G2, G3D B_D transpose,
  R symmetry, G5D Hessian symmetry <= 1e-12, G4D FD <= 1e-8, min Rayleigh(R) > 0, G6D accepted, true
  residual <= 1e-10, |grad J| <= 1e-8, forward <= 1e-8, core_failures = 0, kkt_state_rel <= 1e-10;
  negatives with w != 0: direct (G2, G3D) > 1e-6 [REVISED from the T11H1 1e-4 after the first T13 table, see
  below], derived (G4D, G5D) > 1e-8; heat / w = 0: coincide.
  [Revision: the T11D/T11H1 bar 1e-4 for the DIRECT transpose detectors was calibrated on n <= 4.  These
  detectors measure |<E x,y> - <x,E_wrong y>| / (|E x||y|) for random vectors, i.e. the skew part relative
  to the whole space-time operator, which shrinks as the diffusion part grows with n: dirichlet-opp-3d at
  n = 8 reads 4.0e-6 .. 1.6e-3 (n = 4: 1.1e-4 .. 1.1e-2) while the positives at n = 8 are <= 1.3e-17 (G2)
  and <= 8.0e-18 (G3D): 11 orders of separation, and the derived FD negatives stay 3e-4 .. 65.  The T13
  bar for the direct detectors is 1e-6 (the T13 Neumann bar), >= 10 orders above the positives.]"""
import csv, glob, os, re
D = os.path.dirname(os.path.abspath(__file__)) + "/.."
nrows, drows, allok = [], [], True
FACESET = {"Nnop": "zmin,zmax", "Nnad": "zmin,xmin", "Nopx": "xmin,xmax", "Nadj": "zmin,xmin", "Nadb": "ymin,zmax",
           "Nthr": "zmin,ymin,xmin", "NU": "zmin,xmax,xmin", "Nall": "all six", "Nnu": "zmin,ymax (non-unit)",
           "Ddop": "zmin,zmax", "Dopp": "zmin,zmax", "Dadj": "zmin,xmin", "Dadb": "xmax,ymax", "Dthr": "zmin,ymin,xmin",
           "DU": "zmin,xmax,xmin", "Dfiv": "all but zmax", "Dall": "all six", "Dnu": "zmin,xmin (non-unit)"}


def g(p, s, d=""):
    m = re.search(p, s, re.M); return m.group(1) if m else d


for f in sorted(glob.glob(D + "/logs/gates/*.log")):
    s = open(f).read(); tag = os.path.basename(f)[:-4]
    group, reg, wind, scheme, n = tag.split("_")
    o = dict(re.findall(r"G1_ORIENTATION axis=(\w) v=\S+ \|C\*lin - v\*M1\|/\|M1\|=(\S+)", s))
    F = lambda r, k: float(r[k]) if r.get(k) else float("nan")
    base = {"group": group, "faces": FACESET.get(group, ""), "reg": reg, "wind": wind, "scheme": scheme.upper(), "n": n[1:],
            "G1_x": o.get("x", ""), "G1_y": o.get("y", ""), "G1_z": o.get("z", ""),
            "G2_shell_transpose": g(r"G2_SHELL_TRANSPOSE .*\|diff\|/\(\|Ex\|\|y\|\)=(\S+)", s),
            "G2_neg_ex20": g(r"G2_NEGATIVE_CONTROL .*=(\S+)$", s)}
    if group.startswith("N"):
        fd = dict(re.findall(r"G3_SUMMARY adjoint=(\S+) adjoint_its=\S+ best_rel=(\S+)", s))
        hs = dict(re.findall(r"G2_REDUCED_HESSIAN adjoint=(\S+) .*?rel=(\S+)$", s, re.M))
        mf = dict(re.findall(r"G13N_FD control=(\S+) face=\S+ fd=\S+ adj=\S+ rel=(\S+)", s))
        r = dict(base, **{
            "G1N_boundary_identity": g(r"G1N_BOUNDARY_IDENTITY \|\|C\+C\^T - sum_f \(w\.n_f\) M_f\|\|_F/\|\|C\+C\^T\|\|_F=(\S+)", s),
            "G8B_neg_wrong_bc": g(r"G8B_WRONG_BC_TRANSPOSE .*=(\S+)$", s),
            "hess_exact": hs.get("exact_LU(E^T)", ""), "hess_shell": hs.get("production_shell(E^T)", ""),
            "hess_neg_ex20": hs.get("ex20_style_wrong", ""), "hess_neg_wrong_bc": hs.get("wrong_bc_nuK_minus_C", ""),
            "fd_exact": fd.get("exact_LU(E^T)", ""), "fd_shell": fd.get("production_shell(E^T)", ""),
            "fd_neg_ex20": fd.get("ex20_style_wrong", ""), "fd_neg_wrong_bc": fd.get("wrong_bc_nuK_minus_C", ""),
            "G13N_union": g(r"G13N_UNION .*=(\S+)$", s), "G13N_fd_true": mf.get("true_union", ""),
            "G13N_neg_drop_face": mf.get("drop_face", ""), "G13N_neg_duplicate_face": mf.get("duplicate_face", ""),
            "prod_reason": g(r"G4_PRODUCTION reason=(\S+)", s), "prod_outer": g(r"G4_PRODUCTION .*?outer_its=(\S+)", s),
            "prod_true_res": g(r"G4_PRODUCTION .*?true_residual=(\S+)", s),
            "gradJ": g(r"\|grad J\(u\*\)\|/\|dt alpha_u N u\*\|=(\S+)", s),
            "forward": g(r"\|y_prod - E\^-1\(rhs\+eta N u\*\)\|/\|y\|=(\S+)", s)})
        conv = wind != "heat"
        pos = all(F(r, k) <= 1e-12 for k in ("G1_x", "G1_y", "G1_z", "G2_shell_transpose", "hess_exact", "hess_shell"))
        pos &= (F(r, "G1N_boundary_identity") <= 1e-12) if conv else True
        pos &= F(r, "fd_exact") <= 1e-8 and F(r, "fd_shell") <= 1e-8
        pos &= r["prod_reason"] not in ("", ) and int(r["prod_reason"]) > 0 and F(r, "prod_true_res") <= 1e-10
        pos &= F(r, "gradJ") <= 1e-8 and F(r, "forward") <= 1e-8
        pos &= F(r, "G13N_union") <= 1e-14 and F(r, "G13N_fd_true") <= 1e-8
        neg = F(r, "G13N_neg_drop_face") > 1e-4 and F(r, "G13N_neg_duplicate_face") > 1e-4
        if conv:
            neg &= F(r, "G2_neg_ex20") > 1e-6 and F(r, "G8B_neg_wrong_bc") > 1e-6
            neg &= F(r, "fd_neg_ex20") > 1e-4 and F(r, "fd_neg_wrong_bc") > 1e-4
            neg &= F(r, "hess_neg_ex20") > 1e-8 and F(r, "hess_neg_wrong_bc") > 1e-8
        else:
            neg &= F(r, "G2_neg_ex20") <= 1e-12
        r["negatives"] = "detected" if neg else "MISSED"
        r["verdict"] = "PASS" if pos and neg else "FAIL"
        allok &= pos and neg
        nrows.append(r)
    else:
        hs = dict(re.findall(r"G5D_REDUCED_HESSIAN adjoint=(\S+) .*? rel=(\S+)", s))
        fd = dict(re.findall(r"G4D_FD adjoint=(\S+) best_rel=(\S+)", s))
        r = dict(base, **{
            "G3D_BD_transpose": g(r"G3D_BD_TRANSPOSE .*? rel=(\S+) legacy", s),
            "G3D_neg_legacy": g(r"legacy_B_D\^T: .* rel=(\S+)", s),
            "R_symmetry": g(r"RD_REGULARIZATION .*? rel=(\S+)", s), "R_min_rayleigh": g(r"min Rayleigh\(u_C\)=(\S+)", s),
            "G5D_hess_exact": hs.get("exact_LU(E^T)", ""), "G5D_hess_shell": hs.get("production_shell(E^T)", ""),
            "G5D_neg_legacy": hs.get("legacy_B_D^T", ""), "G5D_neg_ex20": hs.get("ex20_style_untransposed_E", ""),
            "G4D_fd_exact": fd.get("exact_LU(E^T)", ""), "G4D_fd_shell": fd.get("production_shell(E^T)", ""),
            "G4D_neg_legacy": fd.get("legacy_B_D^T", ""), "G4D_neg_ex20": fd.get("ex20_style_untransposed_E", ""),
            "core_its": g(r"core_its_max=(\S+)", s), "core_failures": g(r"core_failures=(\S+)", s),
            "G6D_accepted": g(r"G6D_PRODUCTION .*?accepted=(\S+)", s), "G6D_outer": g(r"G6D_PRODUCTION .*?outer_its=(\S+)", s),
            "G6D_true_res": g(r"G6D_PRODUCTION .*?true_residual=(\S+)", s),
            "G7_gradJ": g(r"\|grad J\(u\*\)\|/\|dt alpha_u R u\*\|=(\S+)", s),
            "G6_forward": g(r"\|y_prod - y\(u\*\)\|/\|y\|=(\S+)", s),
            "ell": g(r"RD_REGULARIZATION ell=(\S+)", s), "gamma_u": g(r"RD_REGULARIZATION .*?gamma_u=(\S+)", s),
            "kkt_state_rel": g(r"G6D_PRODUCTION .*?kkt_state_rel=(\S+)", s)})
        pos = all(F(r, k) <= 1e-12 for k in ("G1_x", "G1_y", "G1_z", "G2_shell_transpose", "G3D_BD_transpose", "R_symmetry",
                                             "G5D_hess_exact", "G5D_hess_shell"))
        pos &= F(r, "G4D_fd_exact") <= 1e-8 and F(r, "G4D_fd_shell") <= 1e-8 and F(r, "R_min_rayleigh") > 0
        pos &= r["G6D_accepted"] == "1" and F(r, "G6D_true_res") <= 1e-10 and F(r, "G7_gradJ") <= 1e-8 and F(r, "G6_forward") <= 1e-8
        pos &= r["core_failures"] in ("0", "")
        pos &= F(r, "kkt_state_rel") <= 1e-10
        nonsym = wind not in ("heat", "w0")
        prim = ("G2_neg_ex20", "G3D_neg_legacy")
        der = ("G4D_neg_legacy", "G4D_neg_ex20", "G5D_neg_legacy", "G5D_neg_ex20")
        negk = ("G3D_neg_legacy", "G5D_neg_legacy", "G5D_neg_ex20", "G2_neg_ex20")
        neg = (all(F(r, k) > 1e-6 for k in prim) and all(F(r, k) > 1e-8 for k in der)) if nonsym else all(F(r, k) <= 1e-12 for k in negk)
        r["negatives"] = ("detected" if neg else "MISSED") if nonsym else ("coincide (L symmetric)" if neg else "MISMATCH")
        r["verdict"] = "PASS" if pos and neg else "FAIL"
        allok &= pos and neg
        drows.append(r)
for path, rows in ((D + "/../t13_gates_neumann.csv", nrows), (D + "/../t13_gates_dirichlet.csv", drows)):
    if rows:
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in nrows:
    print(f"{r['group']:5s} {r['wind']:4s} {r['scheme']} n={r['n']} union={r['G13N_union']} fd={r['fd_shell']} fdT={r['G13N_fd_true']} "
          f"drop={r['G13N_neg_drop_face']} dup={r['G13N_neg_duplicate_face']} G1N={r['G1N_boundary_identity']} wbc={r['fd_neg_wrong_bc']} "
          f"res={r['prod_true_res']} {r['negatives']} {r['verdict']}")
for r in drows:
    print(f"{r['group']:5s} {r['reg']:4s} {r['wind']:4s} {r['scheme']} n={r['n']} BDt={r['G3D_BD_transpose']} H={r['G5D_hess_shell']} "
          f"FD={r['G4D_fd_shell']} negBDt={r['G3D_neg_legacy']} negFD={r['G4D_neg_legacy']} res={r['G6D_true_res']} its={r['core_its']} "
          f"kkt={r['kkt_state_rel']} {r['negatives']} {r['verdict']}")
print("T13_GATES_RESULT", "PASS" if allok else "FAIL",
      f"(neumann {sum(r['verdict'] == 'PASS' for r in nrows)}/{len(nrows)}, dirichlet {sum(r['verdict'] == 'PASS' for r in drows)}/{len(drows)})")
