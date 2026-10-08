#!/usr/bin/env python3
"""logs/gates/*.log -> ../../t11d_gates.csv with a per-run verdict.
Pass criteria: G1 orientation <= 1e-12; G2 shell transpose, G3D B_D transpose, R
symmetry, G5D Hessian symmetry (exact, shell) <= 1e-12; G4D FD (exact, shell) <= 1e-8;
min Rayleigh(R) > 0; G6D production accepted, true residual <= 1e-10, |grad J| <= 1e-8,
|y_prod - y(u*)| <= 1e-8, core_failures = 0.  Negative controls (legacy forward-L B_D^T,
ex20-style untransposed E) must be detected whenever L is nonsymmetric (convection with
w != 0) and coincide (<= 1e-12) for heat / w = 0.  Detection: the PRIMARY detectors (G2
transpose, G3D B_D transpose, G4D FD gradient) > 1e-4; the G5D Hessian-SYMMETRY
negatives > 1e-8 (>= 4 orders above the 1e-12 positive threshold).  [Revised after the
first table: a single 1e-4 bar flagged 4 runs whose G5D symmetry negatives were 6e-6..7e-5
while their primary negatives were 2.4e-3..5e-1 -- a wrong B_D^T perturbs H only through
E^-1 (L - L^T) and is damped when omega is far from Gamma_c; T11N saw the same.]"""
import csv, glob, os, re
D = os.path.dirname(os.path.abspath(__file__)) + "/.."
rows, allok = [], True
def g(p, s, d=""):
    m = re.search(p, s, re.M); return m.group(1) if m else d
for f in sorted(glob.glob(D + "/logs/gates/*.log")):
    s = open(f).read(); tag = os.path.basename(f)[:-4]
    group, wind, scheme, n = tag.split("_")
    o = dict(re.findall(r"G1_ORIENTATION axis=(\w) v=\S+ \|C\*lin - v\*M1\|/\|M1\|=(\S+)", s))
    hs = dict(re.findall(r"G5D_REDUCED_HESSIAN adjoint=(\S+) .*? rel=(\S+)", s))
    fd = dict(re.findall(r"G4D_FD adjoint=(\S+) best_rel=(\S+)", s))
    r = {"group": group, "wind": wind, "scheme": scheme.upper(), "n": n[1:],
         "G1_x": o.get("x", ""), "G1_y": o.get("y", ""), "G1_z": o.get("z", ""),
         "G1_skew_info": g(r"G1_SKEW_DIRICHLET \|\|C\+C\^T\|\|_F/\|\|C\|\|_F=(\S+)", s),
         "G2_shell_transpose": g(r"G2_SHELL_TRANSPOSE .*\|diff\|/\(\|Ex\|\|y\|\)=(\S+)", s),
         "G2_neg_ex20": g(r"G2_NEGATIVE_CONTROL .*=(\S+)$", s),
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
         "G6_forward": g(r"\|y_prod - y\(u\*\)\|/\|y\|=(\S+)", s)}
    F = lambda k: float(r[k]) if r[k] else float("nan")
    pos = all(F(k) <= 1e-12 for k in ("G1_x", "G1_y", "G1_z", "G2_shell_transpose", "G3D_BD_transpose", "R_symmetry",
                                      "G5D_hess_exact", "G5D_hess_shell"))
    pos &= F("G4D_fd_exact") <= 1e-8 and F("G4D_fd_shell") <= 1e-8 and F("R_min_rayleigh") > 0
    pos &= r["G6D_accepted"] == "1" and F("G6D_true_res") <= 1e-10 and F("G7_gradJ") <= 1e-8 and F("G6_forward") <= 1e-8
    pos &= r["core_failures"] == "0"
    nonsym = wind not in ("heat", "w0")
    negk = ("G3D_neg_legacy", "G5D_neg_legacy", "G5D_neg_ex20", "G4D_neg_legacy", "G4D_neg_ex20", "G2_neg_ex20")
    prim = ("G2_neg_ex20", "G3D_neg_legacy", "G4D_neg_legacy", "G4D_neg_ex20")
    neg = (all(F(k) > 1e-4 for k in prim) and all(F(k) > 1e-8 for k in ("G5D_neg_legacy", "G5D_neg_ex20"))) if nonsym else all(F(k) <= 1e-12 for k in negk if k != "G4D_neg_ex20" and k != "G4D_neg_legacy")
    r["negatives"] = ("detected" if neg else "MISSED") if nonsym else ("coincide (L symmetric)" if neg else "MISMATCH")
    r["verdict"] = "PASS" if pos and neg else "FAIL"
    allok &= pos and neg
    rows.append(r)
with open(D + "/../t11d_gates.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows:
    print(f"{r['group']:4s} {r['wind']:5s} {r['scheme']} n={r['n']} BDt={r['G3D_BD_transpose']} H={r['G5D_hess_shell']} "
          f"FD={r['G4D_fd_shell']} negBDt={r['G3D_neg_legacy']} negH={r['G5D_neg_legacy']}/{r['G5D_neg_ex20']} "
          f"res={r['G6D_true_res']} gJ={r['G7_gradJ']} its={r['core_its']} {r['negatives']} {r['verdict']}")
print("T11D_GATES_RESULT", "PASS" if allok else "FAIL", f"({sum(r['verdict'] == 'PASS' for r in rows)}/{len(rows)})")
