#!/usr/bin/env python3
"""T13 all-six smoke: logs/allsix + json -> ../../t13_allsix.csv.  usage: t13_allsix.py <work-dir>"""
import csv, json, os, re, sys
import numpy as np
W, D = sys.argv[1], os.path.dirname(os.path.abspath(__file__)) + "/.."
rows, ok = [], True
for t in ("N_be", "N_cn", "D_L2_be", "D_L2_cn", "D_comb_be", "D_comb_cn"):
    d = json.load(open(f"{W}/{t}.json")); log = open(f"{D}/logs/allsix/{t}.log").read()
    xyz = np.round(np.array(d["spatial_coordinates"]), 12)
    u = np.array([s["values"] for s in d["control_full"]]); y = np.array([s["values"] for s in d["state"]])
    bnd = np.any((xyz == 0.0) | (xyz == 1.0), axis=1)
    nb = (np.isclose(xyz, 0) | np.isclose(xyz, 1)).sum(1)
    active = np.abs(u).max(0) > 0
    r = {"case": t, "boundary_nodes": int(bnd.sum()), "active_nodes": int(active.sum()),
         "active_eq_boundary": bool(np.array_equal(active, bnd)),
         "edge_nodes_active": f"{int((active & (nb == 2)).sum())}/{int((nb == 2).sum())}",
         "corner_nodes_active": f"{int((active & (nb == 3)).sum())}/{int((nb == 3).sum())}",
         "max|u|_interior": f"{np.abs(u[:, ~bnd]).max():.1e}",
         "max|y_C - eta u_C|/max|u|": f"{np.abs(y[:, bnd] - u[:, bnd]).max() / np.abs(u).max():.1e}" if t.startswith("D") else "",
         "reason": re.search(r"EX20_PAR_SOLVE .*? reason=(\S+)", log).group(1),
         "outer": d["solve"]["iterations"], "true_res": f"{d['solve']['true_residual']:.2e}",
         "kkt_state_rel": re.search(r"state_kkt_residual_rel=(\S+)", log).group(1),
         "kkt_adjoint_rel": re.search(r"adjoint_kkt_residual_rel=(\S+)", log).group(1),
         "core_its": (re.search(r"core_its_max=(\d+)", log) or [None, ""])[1] if t.startswith("D") else "",
         "J": f"{d['functional']['J_total']:.10e}", "rss_MB": int(re.search(r"MAXRSS=(\d+)", log).group(1)) // 1024}
    good = (r["active_eq_boundary"] and float(r["max|u|_interior"]) == 0.0 and int(r["reason"]) > 0
            and float(r["true_res"]) < 1e-8 and float(r["kkt_state_rel"]) < 1e-8 and float(r["kkt_adjoint_rel"]) < 1e-8
            and r["edge_nodes_active"].split("/")[0] == r["edge_nodes_active"].split("/")[1]
            and r["corner_nodes_active"] == "8/8"
            and (not t.startswith("D") or float(r["max|y_C - eta u_C|/max|u|"]) < 1e-13))
    r["verdict"] = "PASS" if good else "FAIL"; ok &= good; rows.append(r)
    print(" ".join(f"{k}={v}" for k, v in r.items()))
with open(D + "/../t13_allsix.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("T13_ALLSIX_RESULT", "PASS" if ok else "FAIL", f"({sum(r['verdict'] == 'PASS' for r in rows)}/{len(rows)})")
