#!/usr/bin/env python3
"""logs/gates/*.log -> ../../t10_natural_bc.csv (all gate columns, T9 layout + natural-BC columns)."""
import csv, glob, os, re
D = os.path.dirname(os.path.abspath(__file__)) + "/.."
rows = []
def g(p, s):
    m = re.search(p, s, re.M); return m.group(1) if m else ""
for f in sorted(glob.glob(D + "/logs/gates/*.log")):
    s = open(f).read(); tag = os.path.basename(f)[:-4]
    group, wind, scheme, n = tag.split("_")
    o = dict(re.findall(r"G1_ORIENTATION axis=(\w) v=\S+ \|C\*lin - v\*M1\|/\|M1\|=(\S+)", s))
    fd = dict(re.findall(r"G3_SUMMARY adjoint=(\S+) adjoint_its=\S+ best_rel=(\S+)", s))
    hs = dict(re.findall(r"G2_REDUCED_HESSIAN adjoint=(\S+) .*?rel=(\S+)$", s, re.M))
    rows.append({"group": group, "wind": wind, "scheme": scheme.upper(), "n": n[1:],
        "G1N_boundary_identity": g(r"G1N_BOUNDARY_IDENTITY \|\|C\+C\^T - sum_f \(w\.n_f\) M_f\|\|_F/\|\|C\+C\^T\|\|_F=(\S+)", s),
        "G1_x": o.get("x", ""), "G1_y": o.get("y", ""), "G1_z": o.get("z", ""),
        "G2_shell_transpose": g(r"G2_SHELL_TRANSPOSE .*\|diff\|/\(\|Ex\|\|y\|\)=(\S+)", s),
        "G2_neg_ex20": g(r"G2_NEGATIVE_CONTROL .*=(\S+)$", s),
        "G2_neg_wrong_bc": g(r"G8B_WRONG_BC_TRANSPOSE .*=(\S+)$", s),
        "G3_fd_exact": fd.get("exact_LU(E^T)", ""), "G3_fd_shell": fd.get("production_shell(E^T)", ""),
        "G3_fd_ex20(neg)": fd.get("ex20_style_wrong", ""), "G3_fd_wrong_bc(neg)": fd.get("wrong_bc_nuK_minus_C", ""),
        "G4_hess_shell": hs.get("production_shell(E^T)", ""), "G4_hess_ex20(neg)": hs.get("ex20_style_wrong", ""),
        "G4_hess_wrong_bc(neg)": hs.get("wrong_bc_nuK_minus_C", ""),
        "G5_outer": g(r"G4_PRODUCTION .*?outer_its=(\S+)", s), "G5_true_res": g(r"G4_PRODUCTION .*?true_residual=(\S+)", s),
        "G7_gradJ": g(r"\|grad J\(u\*\)\|/\|dt alpha_u N u\*\|=(\S+)", s),
        "G6_forward_march": g(r"\|y_prod - E\^-1\(rhs\+eta N u\*\)\|/\|y\|=(\S+)", s),
        "faces": g(r"G1N_BOUNDARY_IDENTITY .* faces=(\S+)", s)})
with open(D + "/../t10_natural_bc.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows:
    print(" ".join(f"{k}={v}" for k, v in r.items() if k != "faces"))
