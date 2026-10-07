#!/usr/bin/env python3
"""Collect t7d/logs/gates/*.log into boundary_convection/t7d_matrix_gates.csv."""
import csv, glob, os, re
D = os.path.dirname(os.path.abspath(__file__)) + "/.."
rows = []
def g(pat, s, k=1):
    m = re.search(pat, s)
    return m.group(k) if m else ""
for f in sorted(glob.glob(D + "/logs/gates/*.log")):
    s = open(f).read()
    name, scheme = os.path.basename(f)[:-4].rsplit("_", 1)
    fd = dict(re.findall(r"EX21_VC_G4D_FD adjoint=(\S+) best_rel=(\S+)", s))
    hs = dict(re.findall(r"EX21_VC_G5D_REDUCED_HESSIAN adjoint=(\S+) .*?rel=(\S+)", s))
    rows.append({
        "case": name, "scheme": scheme.upper(),
        "vx": g(r"vx=(\S+)", s), "vy": g(r"vy=(\S+)", s), "convection": g(r"convection=(\S+)", s),
        "ell": g(r"RD_REGULARIZATION ell=(\S+)", s), "gamma_u": g(r"gamma_u=(\S+)", s),
        "G1_C_orient": g(r"axis=x v=\S+ \|C\*lin - v\*M1\|/\|M1\|=(\S+)", s),
        "G2_E_transpose": g(r"G2_SHELL_TRANSPOSE .*=(\S+)$", s.replace("\n", "\n")) or g(r"G2_SHELL_TRANSPOSE .*\|diff\|/\(\|Ex\|\|y\|\)=(\S+)", s),
        "G2neg_ex20_untransposed": g(r"G2_NEGATIVE_CONTROL .*=(\S+)", s),
        "G3D_BD_transpose": g(r"G3D_BD_TRANSPOSE .*? rel=(\S+)", s),
        "G9D_legacy_BDT_transpose": g(r"legacy_B_D\^T: .*? rel=(\S+)", s),
        "R_symmetry": g(r"<R u1,u2>-<u1,R u2> rel=(\S+)", s),
        "G5D_hess_exact": hs.get("exact_LU(E^T)", ""), "G5D_hess_shell": hs.get("production_shell(E^T)", ""),
        "G5D_hess_legacy": hs.get("legacy_B_D^T", ""), "G5D_hess_untransposedE": hs.get("ex20_style_untransposed_E", ""),
        "G4D_fd_exact": fd.get("exact_LU(E^T)", ""), "G4D_fd_shell": fd.get("production_shell(E^T)", ""),
        "G9D_fd_legacy": fd.get("legacy_B_D^T", ""), "G9D_fd_untransposedE": fd.get("ex20_style_untransposed_E", ""),
        "G6D_outer": g(r"G6D_PRODUCTION .*?outer_its=(\S+)", s), "G6D_true_res": g(r"G6D_PRODUCTION .*?true_residual=(\S+)", s),
        "G6D_accepted": g(r"G6D_PRODUCTION .*?accepted=(\S+)", s),
        "G6D_gradJ_rel": g(r"\|grad J\(u\*\)\|/\|dt alpha_u R u\*\|=(\S+)", s),
        "G7D_forward_march": g(r"\|y_prod - y\(u\*\)\|/\|y\|=(\S+)", s),
        "G8D_stationarity": g(r"kkt_stationarity_rel=(\S+)", s),
        "kkt_state": g(r"kkt_state_rel=(\S+)", s), "kkt_adjoint": g(r"kkt_adjoint_rel=(\S+)", s),
    })
out = D + "/../t7d_matrix_gates.csv"
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("wrote", out, len(rows))
for r in rows:
    print(f"{r['case']:<14}{r['scheme']:<3} G3D={r['G3D_BD_transpose']:<10} legacy={r['G9D_legacy_BDT_transpose']:<10} FD={r['G4D_fd_shell']:<10} FDleg={r['G9D_fd_legacy']:<10} FDunE={r['G9D_fd_untransposedE']:<10} H={r['G5D_hess_shell']:<10} out={r['G6D_outer']} res={r['G6D_true_res']:<10} grad={r['G6D_gradJ_rel']:<10} march={r['G7D_forward_march']:<10} stat={r['G8D_stationarity']}")
