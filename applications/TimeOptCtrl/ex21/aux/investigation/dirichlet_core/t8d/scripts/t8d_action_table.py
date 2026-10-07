#!/usr/bin/env python3
"""logs/action/*.log -> dirichlet_core/t8d_core_action.csv"""
import csv, glob, os, re
D = os.path.dirname(os.path.abspath(__file__)) + "/.."
rows = []
for f in sorted(glob.glob(D + "/logs/action/*.log")):
    s = open(f).read(); tag = os.path.basename(f)[:-4]
    conv, scheme, reg, nx, nt = re.match(r"conv(\w+)_(\w+)_(\w+)_nx(\d+)_nt(\d+)", tag).groups()
    spd = re.search(r"EX21_T8D_CORE_SPD (.*)", s).group(1)
    kv = dict(re.findall(r"(\S+?)=(\S+)", spd))
    act = dict(re.findall(r"EX21_T8D_CORE_ACTION vector=(\S+) .*=(\S+)$", s, re.M))
    r = {"convection": conv, "scheme": scheme.upper(), "regularisation": reg, "nx": nx, "steps": nt, "m": kv["m"],
         "sym_rel": kv["||K-K^T||_F/||K||_F"], "lambda_min": kv["lambda_min"], "lambda_max": kv["lambda_max"],
         "cond_K": kv["cond_K"], "min_rayleigh_random": kv["min_rayleigh_random"], "cond_R": kv["cond_R"],
         "RinvK_lambda_min": kv["RinvK_lambda_min"], "RinvK_lambda_max": kv["RinvK_lambda_max"], "cond_RinvK": kv["cond_RinvK"],
         "shell_symmetry": re.search(r"SHELL_SYMMETRY .*=(\S+)$", s, re.M).group(1),
         "rt_sparse_vs_dense": re.search(r"RT_SPARSE_VS_DENSE rel=(\S+)", s).group(1)}
    for k, v in act.items(): r["action_" + k] = v
    r["action_max"] = f"{max(float(v) for v in act.values()):.3e}"
    rows.append(r)
with open(D + "/../t8d_core_action.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows:
    print(f"{r['convection']:<4}{r['scheme']:<3}{r['regularisation']:<5} nx={r['nx']:<3} m={r['m']:<5} sym={r['sym_rel']} lmin={float(r['lambda_min']):.2e} condK={float(r['cond_K']):.1e} condRinvK={float(r['cond_RinvK']):.3f} action_max={r['action_max']} shellsym={r['shell_symmetry']} RT={r['rt_sparse_vs_dense']}")
