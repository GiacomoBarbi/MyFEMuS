#!/usr/bin/env python3
import csv, glob, os, re
D = os.path.dirname(os.path.abspath(__file__)) + "/.."
rows = []
def g(p, s):
    m = re.search(p, s, re.M); return m.group(1) if m else ""
for f in sorted(glob.glob(D + "/logs/gates/*.log")):
    s = open(f).read(); tag = os.path.basename(f)[:-4]
    wind, scheme, n = tag.split("_")
    o = dict(re.findall(r"G1_ORIENTATION axis=(\w) v=\S+ \|C\*lin - v\*M1\|/\|M1\|=(\S+)", s))
    ot = dict(re.findall(r"G1_ORIENTATION axis=(\w) v=\S+ .*?\|C\^T\*lin - v\*M1\|/\|M1\|=(\S+)", s))
    fd = dict(re.findall(r"G3_SUMMARY adjoint=(\S+) adjoint_its=\S+ best_rel=(\S+)", s))
    hs = dict(re.findall(r"G2_REDUCED_HESSIAN adjoint=(\S+) .*?rel=(\S+)$", s, re.M))
    rows.append({"wind": wind, "scheme": scheme.upper(), "n": n[1:],
        "G1_x": o.get("x", ""), "G1_y": o.get("y", ""), "G1_z": o.get("z", ""),
        "G1_transposed_x(neg)": ot.get("x", ""),
        "G1_skew_free": g(r"G1_SKEW_DIRICHLET \|\|C\+C\^T\|\|_F/\|\|C\|\|_F=(\S+)", s),
        "G2_shell_transpose": g(r"G2_SHELL_TRANSPOSE .*\|diff\|/\(\|Ex\|\|y\|\)=(\S+)", s),
        "G2_neg_ex20": g(r"G2_NEGATIVE_CONTROL .*=(\S+)$", s),
        "G3_fd_exact": fd.get("exact_LU(E^T)", ""), "G3_fd_shell": fd.get("production_shell(E^T)", ""),
        "G3_fd_ex20(neg)": fd.get("ex20_style_wrong", ""),
        "G4_hess_shell": hs.get("production_shell(E^T)", ""), "G4_hess_ex20(neg)": hs.get("ex20_style_wrong", ""),
        "G5_outer": g(r"G4_PRODUCTION .*?outer_its=(\S+)", s), "G5_true_res": g(r"G4_PRODUCTION .*?true_residual=(\S+)", s),
        "G7_gradJ": g(r"\|grad J\(u\*\)\|/\|dt alpha_u N u\*\|=(\S+)", s),
        "G6_forward_march": g(r"\|y_prod - E\^-1\(rhs\+eta N u\*\)\|/\|y\|=(\S+)", s)})
with open(D + "/../t9_3d_gates.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows:
    print(" ".join(f"{k}={v}" for k, v in r.items()))
