#!/usr/bin/env python3
import csv, json, os, re, sys
import numpy as np
D = os.path.dirname(os.path.abspath(__file__)) + "/.."; J = sys.argv[1]
def L(t):
    d = json.load(open(f"{J}/{t}.json"))
    return d, {k: np.array([s["values"] for s in d[k]]) for k in ("state", "adjoint", "control_full")}
rows = []
for s in ("be", "cn"):
    for r in ("l2", "h1s", "h1t", "h1st"):
        d0, f0 = L(f"ex20_{s}_{r}")
        for v in ("ex21dense", "ex21iter"):
            d, f = L(f"{v}_{s}_{r}")
            log = open(f"{D}/logs/ex20/{v}_{s}_{r}.log").read()
            m = re.search(r"core_its_max=(\d+)", log)
            row = {"scheme": s.upper(), "reg": r, "vs_ex20": v,
                   "max_field_rel": f"{max(np.abs(f[k]-f0[k]).max()/np.abs(f0[k]).max() for k in f0):.2e}",
                   "bitwise_fields": all(np.array_equal(f[k], f0[k]) for k in f0),
                   "J_rel": f"{abs(d['functional']['J_total']-d0['functional']['J_total'])/abs(d0['functional']['J_total']):.2e}",
                   "outer": d["solve"]["iterations"], "outer_ex20": d0["solve"]["iterations"],
                   "true_res": f"{d['solve']['true_residual']:.2e}", "true_res_ex20": f"{d0['solve']['true_residual']:.2e}",
                   "core_its_max": m.group(1) if m else ""}
            rows.append(row); print(" ".join(f"{k}={x}" for k, x in row.items()))
with open(D + "/results/ex20_heat_comparison.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
