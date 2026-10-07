#!/usr/bin/env python3
import csv, json, os, re, sys
import numpy as np
D = os.path.dirname(os.path.abspath(__file__)) + "/.."; J = sys.argv[1]
def L(t):
    d = json.load(open(f"{J}/{t}.json"))
    return d, {k: np.array([s["values"] for s in d[k]]) for k in ("state", "adjoint", "control_full")}
def strip(path):
    return [re.sub(r" (solve_wall|total_wall|t_operators|t_pcsetup|t_kspsolve|t_total|setup_seconds|json)=\S*", "", l)
            for l in open(path) if l.startswith(("EX20_", "EX21_"))]
rows = []
for conv in ("off", "on"):
    for s in ("be", "cn"):
        for r in ("l2", "h1s", "h1t", "h1st"):
            t = f"conv{conv}_{s}_{r}"
            do, fo = L(t + "_old_default"); dn, fn = L(t + "_new_dense"); dd, fd = L(t + "_new_default")
            jo = open(f"{J}/{t}_old_default.json").read(); jn = open(f"{J}/{t}_new_dense.json").read()
            log = open(f"{D}/logs/default/{t}_new_default.log").read()
            m = re.search(r"EX21_T8D_CORE_STATS mode=(\S+) pc=(\S+) rtol=(\S+) .*?core_its_max=(\d+)", log)
            row = {"case": t,
                   "new_default_core": f"{m.group(1)}/{m.group(2)}/{m.group(3)}" if m else "dense?",
                   "new_dense_vs_old_json_bytewise": jo == jn,
                   "new_dense_vs_old_log_lines_identical": strip(f"{D}/logs/default/{t}_old_default.log") == strip(f"{D}/logs/default/{t}_new_dense.log"),
                   "default_vs_old_max_field_rel": f"{max(np.abs(fd[k]-fo[k]).max()/np.abs(fo[k]).max() for k in fo):.2e}",
                   "default_vs_old_J_track_rel": f"{abs(dd['functional']['J_track']-do['functional']['J_track'])/abs(do['functional']['J_track']):.2e}",
                   "default_vs_old_J_reg_rel": f"{abs(dd['functional']['J_reg']-do['functional']['J_reg'])/abs(do['functional']['J_reg']):.2e}",
                   "outer_old": do["solve"]["iterations"], "outer_default": dd["solve"]["iterations"],
                   "true_res_old": f"{do['solve']['true_residual']:.2e}", "true_res_default": f"{dd['solve']['true_residual']:.2e}",
                   "core_its_max": m.group(4) if m else ""}
            rows.append(row); print(" ".join(f"{k}={v}" for k, v in row.items()))
with open(D + "/results/default_change.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
