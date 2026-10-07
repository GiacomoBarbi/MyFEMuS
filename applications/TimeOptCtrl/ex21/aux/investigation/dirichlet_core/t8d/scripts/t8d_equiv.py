#!/usr/bin/env python3
"""dense vs iterative fields -> dirichlet_core/t8d_solution_equivalence.csv"""
import csv, glob, json, os, re, sys
import numpy as np
D = os.path.dirname(os.path.abspath(__file__)) + "/.."
J = sys.argv[1]
rows = []
def L(t):
    d = json.load(open(f"{J}/{t}.json"))
    xy = np.round(np.array(d["spatial_coordinates"]), 12); o = np.lexsort((xy[:, 0], xy[:, 1]))
    return d, {k: np.array([s["values"] for s in d[k]])[:, o] for k in ("state", "adjoint", "control_full")}
for f in sorted(glob.glob(f"{J}/*_dense.json")):
    base = os.path.basename(f)[:-11]
    if not os.path.exists(f"{J}/{base}_iterative.json"): continue
    dd, fd = L(base + "_dense"); di, fi = L(base + "_iterative")
    log = open(f"{D}/logs/equiv/{base}_iterative.log").read()
    g = lambda p: (re.search(p, log).group(1) if re.search(p, log) else "")
    r = {"case": base}
    for k, lab in (("state", "y"), ("adjoint", "p"), ("control_full", "u")):
        r[f"{lab}_maxrel"] = f"{np.abs(fi[k] - fd[k]).max() / np.abs(fd[k]).max():.3e}"
    for k in ("J_track", "J_reg"):
        r[k + "_rel"] = f"{abs(di['functional'][k] - dd['functional'][k]) / abs(dd['functional'][k]):.3e}"
    r["true_res_dense"] = f"{dd['solve']['true_residual']:.3e}"; r["true_res_iter"] = f"{di['solve']['true_residual']:.3e}"
    r["outer_dense"] = dd["solve"]["iterations"]; r["outer_iter"] = di["solve"]["iterations"]
    r["stationarity_iter"] = f"{di['kkt']['stationarity_consistency_residual_rel']:.2e}"
    r["core_its_mean"] = g(r"core_its_mean=(\S+)"); r["core_its_max"] = g(r"core_its_max=(\S+)")
    rows.append(r)
    print(" ".join(f"{k}={v}" for k, v in r.items()))
with open(D + "/../t8d_solution_equivalence.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
