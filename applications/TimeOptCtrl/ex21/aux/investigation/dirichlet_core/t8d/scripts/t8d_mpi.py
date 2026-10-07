#!/usr/bin/env python3
import csv, json, os, re, sys
import numpy as np
D = os.path.dirname(os.path.abspath(__file__)) + "/.."; J = sys.argv[1]
def L(t):
    d = json.load(open(f"{J}/{t}.json"))
    xy = np.round(np.array(d["spatial_coordinates"]), 12); o = np.lexsort((xy[:, 0], xy[:, 1]))
    return d, {k: np.array([s["values"] for s in d[k]])[:, o] for k in ("state", "adjoint", "control_full")}
rows = []
for case in ("iterative_nx32_nt128", "dense_nx32_nt128", "iterative_nx48_nt256"):
    d1, f1 = L(case + "_np1")
    for np_ in (1, 2, 4):
        t = f"{case}_np{np_}"; d, f = L(t); s = open(f"{D}/logs/mpi/{t}.log").read()
        rss = [int(v) / 1024 for v in re.findall(r"RANK_MAXRSS=(\d+)", s)]
        g = lambda p: (re.search(p, s).group(1) if re.search(p, s) else "")
        r = {"case": case, "np": np_, "max_field_rel_vs_np1": f"{max(np.abs(f[k]-f1[k]).max()/np.abs(f1[k]).max() for k in f):.2e}",
             "J_rel_vs_np1": f"{abs(d['functional']['J_total']-d1['functional']['J_total'])/abs(d1['functional']['J_total']):.2e}",
             "outer": d["solve"]["iterations"], "true_residual": f"{d['solve']['true_residual']:.2e}",
             "core_its_total": g(r"core_its_total=(\d+)"), "core_its_max": g(r"core_its_max=(\d+)"),
             "rt_backend": g(r"rt_backend=(\S+)"), "spatial_backend": g(r"forward_backend=(\S+)"),
             "rank_maxrss_MB_max": f"{max(rss):.0f}", "rank_maxrss_MB_min": f"{min(rss):.0f}", "sum_maxrss_MB": f"{sum(rss):.0f}",
             "wall": g(r"wall=(\S+)")}
        rows.append(r); print(" ".join(f"{k}={v}" for k, v in r.items()))
with open(D + "/../t8d_mpi.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
