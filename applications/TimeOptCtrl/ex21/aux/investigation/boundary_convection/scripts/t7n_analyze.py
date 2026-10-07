#!/usr/bin/env python3
"""T7N tables -> ../t7n_matrix_gates.csv, ../t7n_convergence.csv, ../t7n_solver.csv"""
import csv, glob, math, os, re
H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(H, "results")
def load(n):
    p = os.path.join(R, n); return list(csv.DictReader(open(p))) if os.path.exists(p) else []
def g(t, p):
    m = re.search(p, t); return m.group(1) if m else ""
# ---------------------------------------------------------------- gates
rows = []
for f in sorted(glob.glob(os.path.join(H, "logs", "gates", "vc_neumann_*.log"))):
    t = open(f).read(); _, _, wind, sch = os.path.basename(f)[:-4].split("_")
    rec = dict(run=os.path.basename(f)[:-4], wind=wind, scheme=sch.upper(),
               w=g(t, r"vx=(\S+) vy=") + "," + g(t, r"vy=(\S+) state_bc"), faces=g(t, r"faces=(\S+)"),
               G1_orientation_x=g(t, r"axis=x v=\S+ \|C\*lin - v\*M1\|/\|M1\|=(\S+)"),
               G1_orientation_z=g(t, r"axis=z v=\S+ \|C\*lin - v\*M1\|/\|M1\|=(\S+)"),
               G1N_boundary_identity_rel=g(t, r"M_f\|\|_F/\|\|C\+C\^T\|\|_F=(\S+)"),
               G2_spacetime_transpose=g(t, r"SHELL_TRANSPOSE.*\|diff\|/\(\|Ex\|\|y\|\)=(\S+)"),
               G2_assembled_vs_shell=g(t, r"\|E_aij\^T y - ET y\|/\|ET y\|=(\S+)"),
               G4_hessian_sym_exactLU=g(t, r"adjoint=exact_LU\(E\^T\) <H.*rel=(\S+)"),
               G4_hessian_sym_shell=g(t, r"adjoint=production_shell\(E\^T\) <H.*rel=(\S+)"),
               G3_fd_gradient_exactLU=g(t, r"G3_SUMMARY adjoint=exact_LU\(E\^T\).*best_rel=(\S+)"),
               G3_fd_gradient_shell=g(t, r"G3_SUMMARY adjoint=production_shell\(E\^T\).*best_rel=(\S+)"),
               G5_true_kkt_residual=g(t, r"G4_PRODUCTION.*true_residual=(\S+)"),
               G6_forward_march_rel=g(t, r"\|y_prod - E\^-1\(rhs\+eta N u\*\)\|/\|y\|=(\S+)"),
               G7_reduced_gradient_at_optimum=g(t, r"\|grad J\(u\*\)\|/\|dt alpha_u N u\*\|=(\S+)"),
               G8_ex20_untransposed_transpose=g(t, r"NEGATIVE_CONTROL.*=(\S+)"),
               G8_ex20_untransposed_hessian=g(t, r"ex20_style_wrong <H.*rel=(\S+)"),
               G8_ex20_untransposed_fd=g(t, r"G3_SUMMARY adjoint=ex20_style_wrong.*best_rel=(\S+)"),
               G8b_wrong_bc_transpose=g(t, r"WRONG_BC_TRANSPOSE.*=(\S+)"),
               G8b_wrong_bc_hessian=g(t, r"wrong_bc_nuK_minus_C <H.*rel=(\S+)"),
               G8b_wrong_bc_fd=g(t, r"G3_SUMMARY adjoint=wrong_bc_nuK_minus_C.*best_rel=(\S+)"))
    f2 = lambda k: float(rec[k])
    ok = (f2("G1_orientation_x") < 1e-12 and f2("G1N_boundary_identity_rel") < 1e-12 and f2("G2_spacetime_transpose") < 1e-14
          and f2("G4_hessian_sym_exactLU") < 1e-11 and f2("G3_fd_gradient_exactLU") < 1e-10 and f2("G5_true_kkt_residual") < 1e-9
          and f2("G6_forward_march_rel") < 1e-12 and f2("G8_ex20_untransposed_fd") > 1e-2 and f2("G8b_wrong_bc_fd") > 1e-2
          and f2("G8b_wrong_bc_hessian") > 1e-4)
    rec["verdict"] = "PASS" if ok else "FAIL"
    rows.append(rec)
# G7 Neumann stationarity on Gamma_c from the exact-OC runs
for r in load("space.csv") + load("time.csv"):
    rows.append(dict(run=r["id"], wind=r["wind"], scheme=r["scheme"], w=r["vx"] + "," + r["vy"],
                     G7_neumann_stationarity_gamma_c=r.get("neumann_stationarity_rel", ""),
                     max_u_off_gamma_c=r.get("max_u_off_Gamma_c", ""), control_nodes=r.get("control_nodes", ""),
                     G5_true_kkt_residual=r["true_residual"], G6_forward_march_rel=r["march_rel"],
                     verdict="PASS" if float(r.get("neumann_stationarity_rel") or 1) < 1e-12 and float(r["march_rel"]) < 1e-12
                     and float(r.get("max_u_off_Gamma_c") or 1) == 0.0 else "FAIL"))
keys = []
for r in rows: keys += [k for k in r if k not in keys]
with open(os.path.join(H, "t7n_matrix_gates.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
print("gates:", sum(r["verdict"] == "PASS" for r in rows), "/", len(rows), "PASS")
# ---------------------------------------------------------------- convergence
out = []
for grp, key in (("space", lambda r: (r["wind"], r["scheme"], r["steps"])), ("time", lambda r: (r["wind"], r["scheme"], r["nx"])),
                 ("cnlayer", lambda r: (r["wind"], r["scheme"], r["nx"]))):
    prev = {}
    for r in load(grp + ".csv"):
        k = key(r)
        rec = dict(group=grp, id=r["id"], wind=r["wind"], w=r["vx"] + "," + r["vy"], scheme=r["scheme"], nx=r["nx"], steps=r["steps"],
                   nu=r["nu"], Pe_cell=r["Pe_cell"], reason=r["reason"], accepted=r.get("accepted", ""), true_residual=r["true_residual"],
                   march_rel=r["march_rel"], neumann_stationarity_rel=r.get("neumann_stationarity_rel", ""), J_err_rel=r.get("J_err_rel", ""))
        fields = ("y_L2L2", "y_half", "p_L2L2", "p_half", "p_int", "u_L2L2", "u_half", "u_int", "u_last", "p_nodetime_L2L2")
        for fld in fields:
            rec[fld] = r[fld]
            if k in prev:
                rec["order_" + fld] = f"{math.log2(float(prev[k][fld]) / float(r[fld])):.3f}"
        prev[k] = r
        out.append(rec)
keys = []
for r in out: keys += [k for k in r if k not in keys]
with open(os.path.join(H, "t7n_convergence.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(out)
print("convergence rows:", len(out))
# ---------------------------------------------------------------- solver
srows = []
for r in load("solver.csv"):
    srows.append(dict(source="manufactured neumann-conv (wind A, CN, production tol)", variant=r["variant"], nx=r["nx"],
                      steps=r.get("steps", ""), status=r["status"], accepted=r.get("accepted", ""), reason=r.get("reason", ""),
                      outer_its=r.get("outer_its", ""), true_residual=r.get("true_residual", ""),
                      schur_its=r.get("schur_ksp_its", ""), schur_max=r.get("schur_ksp_max", ""),
                      right_its=r.get("right_its", ""), right_max=r.get("right_max", ""), left_its=r.get("left_its", ""),
                      left_max=r.get("left_max", ""), ysplit_its=r.get("ysplit_its", ""), ysplit_max=r.get("ysplit_max", ""),
                      sweep_failures=r.get("sweep_failures", ""), J_track=r.get("J_track", ""), J_reg=r.get("J_reg", ""),
                      solve_wall=r.get("solve_wall", ""), outer_history=r.get("outer_history", "")))
for line in open(os.path.join(H, "logs", "solver_attribution.log")):
    if ":" not in line or line.startswith("---"): continue
    name, res = line.split(":", 1)
    srows.append(dict(source="attribution (nx 8, Nt 32)", variant=name.strip(), status=res.strip()[:240]))
for r in load("mpi.csv"):
    srows.append(dict(source="MPI production (neumann-conv, wind A, CN, nx 16, Nt 32, MUMPS)", variant=f"np={r['np']}",
                      reason=r["reason"], outer_its=r["outer_its"], true_residual=r["true_residual"], J_track=r["J_track"],
                      J_reg=r["J_reg"], outer_history=r["outer_history"],
                      status=f"max|dy| {r.get('max_dy_vs_np1','-')} max|dp| {r.get('max_dp_vs_np1','-')} max|du| {r.get('max_du_vs_np1','-')} {r.get('verdict','ref')}"))
keys = []
for r in srows: keys += [k for k in r if k not in keys]
with open(os.path.join(H, "t7n_solver.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(srows)
print("solver rows:", len(srows))
