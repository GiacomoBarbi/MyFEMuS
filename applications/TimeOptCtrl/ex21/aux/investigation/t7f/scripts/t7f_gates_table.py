#!/usr/bin/env python3
"""t7f/logs/gates/*.log -> t7f/results/gates.csv (+ console summary)."""
import csv, glob, os, re
D = os.path.dirname(os.path.abspath(__file__)) + "/.."
def g(p, s):
    m = re.search(p, s); return m.group(1) if m else ""
rows = []
for f in sorted(glob.glob(D + "/logs/gates/*.log")):
    s = open(f).read(); name, scheme = os.path.basename(f)[:-4].rsplit("_", 1)
    fd = dict(re.findall(r"EX21_VC_G3_SUMMARY adjoint=(\S+) adjoint_its=\S+ best_rel=(\S+)", s))
    hs = dict(re.findall(r"EX21_VC_G2_REDUCED_HESSIAN adjoint=(\S+) .*?rel=(\S+)", s))
    rows.append({"case": name, "scheme": scheme.upper(), "state_bc": g(r"state_bc=(\S+)", s),
        "vx": g(r" vx=(\S+)", s), "vy": g(r" vy=(\S+)", s),
        "G1_C_orient": g(r"axis=x v=\S+ \|C\*lin - v\*M1\|/\|M1\|=(\S+)", s),
        "G1N_boundary_identity": g(r"G1N_BOUNDARY_IDENTITY .*?M_f\|\|_F/\|\|C\+C\^T\|\|_F=(\S+)", s),
        "G2_transpose": g(r"G2_SHELL_TRANSPOSE .*\|diff\|/\(\|Ex\|\|y\|\)=(\S+)", s),
        "G2_neg_ex20": g(r"G2_NEGATIVE_CONTROL .*=(\S+)", s),
        "G8b_wrongbc_transpose": g(r"G8B_WRONG_BC_TRANSPOSE .*=(\S+)", s),
        "hess_exact": hs.get("exact_LU(E^T)", ""), "hess_shell": hs.get("production_shell(E^T)", ""),
        "hess_ex20_wrong": hs.get("ex20_style_wrong", ""), "hess_wrongbc": hs.get("wrong_bc_nuK_minus_C", ""),
        "fd_exact": fd.get("exact_LU(E^T)", ""), "fd_shell": fd.get("production_shell(E^T)", ""),
        "fd_ex20_wrong": fd.get("ex20_style_wrong", ""), "fd_wrongbc": fd.get("wrong_bc_nuK_minus_C", ""),
        "G4_reason": g(r"G4_PRODUCTION reason=(\S+)", s), "G4_outer": g(r"G4_PRODUCTION .*?outer_its=(\S+)", s),
        "G4_true_res": g(r"G4_PRODUCTION .*?true_residual=(\S+)", s),
        "G4_gradJ": g(r"\|grad J\(u\*\)\|/\|dt alpha_u N u\*\|=(\S+)", s),
        "G4_forward_march": g(r"\|y_prod - E\^-1\(rhs\+eta N u\*\)\|/\|y\|=(\S+)", s),
        "J_track": g(r"G4_PRODUCTION .*?J_track=(\S+)", s), "J_reg": g(r"G4_PRODUCTION .*?J_reg=(\S+)", s)})
os.makedirs(D + "/results", exist_ok=True)
with open(D + "/results/gates.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows:
    print(f"{r['case']:<18}{r['scheme']:<3}{r['state_bc']:<10} G1N={r['G1N_boundary_identity']:<9} G2={r['G2_transpose']:<9} neg={r['G2_neg_ex20']:<9} G8b={r['G8b_wrongbc_transpose']:<9} H={r['hess_shell']:<9} FD={r['fd_shell']:<9} FDneg={r['fd_ex20_wrong']:<9} FDwbc={r['fd_wrongbc']:<9} out={r['G4_outer']} res={r['G4_true_res']:<9} grad={r['G4_gradJ']:<9} march={r['G4_forward_march']} Jr={r['J_reg']}")
