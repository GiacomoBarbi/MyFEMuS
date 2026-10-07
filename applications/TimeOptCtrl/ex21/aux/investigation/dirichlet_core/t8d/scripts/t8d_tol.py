#!/usr/bin/env python3
import csv, json, os, re, sys
import numpy as np
D = os.path.dirname(os.path.abspath(__file__)) + "/.."; J = sys.argv[1]
def L(t):
    d = json.load(open(f"{J}/{t}.json"))
    return d, {k: np.array([s["values"] for s in d[k]]) for k in ("state", "adjoint", "control_full")}
rows = []
for r in ("l2", "h1st"):
    for a in ("1e-2", "1e-4"):
        dd, fd = L(f"{r}_a{a}_dense")
        for tol in ("1e-2", "1e-4", "1e-6", "1e-8", "1e-10", "1e-12"):
            t = f"{r}_a{a}_tol{tol}"; d, f = L(t); s = open(f"{D}/logs/tol/{t}.log").read()
            g = lambda p: (re.search(p, s).group(1) if re.search(p, s) else "")
            row = {"reg": r, "alpha_u": a, "core_rtol": tol, "core_solves": g(r"core_solves=(\d+)"),
                   "core_its_total": g(r"core_its_total=(\d+)"), "core_its_max": g(r"core_its_max=(\d+)"),
                   "outer": d["solve"]["iterations"], "outer_dense": dd["solve"]["iterations"], "reason": d["solve"]["reason"],
                   "true_residual": f"{d['solve']['true_residual']:.2e}", "true_residual_dense": f"{dd['solve']['true_residual']:.2e}",
                   "max_field_rel_vs_dense": f"{max(np.abs(f[k]-fd[k]).max()/np.abs(fd[k]).max() for k in fd):.2e}",
                   "J_rel_vs_dense": f"{abs(d['functional']['J_total']-dd['functional']['J_total'])/abs(dd['functional']['J_total']):.2e}",
                   "wall": g(r"wall=(\S+)"), "wall_dense": re.search(r"wall=(\S+)", open(f"{D}/logs/tol/{r}_a{a}_dense.log").read()).group(1),
                   "core_seconds": g(r"core_seconds=(\S+)")}
            rows.append(row); print(" ".join(f"{k}={v}" for k, v in row.items()))
with open(D + "/results/tolerance.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
