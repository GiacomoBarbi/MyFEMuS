#!/usr/bin/env python3
# (T13 copy of t11d_ladder_table.py: the multi-face exact cases; u in L2(Gamma_c) = sum over the controlled faces)
"""t9/logs/exact/<tag>_<scheme>_n*_nt*.log -> orders.  usage: t9_ladder_table.py <out.csv> <tag> [...]"""
import csv, glob, math, os, re, sys
D = os.path.dirname(os.path.abspath(__file__)) + "/../logs/exact/"   # T10 copy of t9_ladder_table.py
rows = []
for tag in sys.argv[2:]:
    runs = []
    for f in glob.glob(f"{D}{tag}_*_n*_nt*.log"):
        s = open(f).read(); m = re.search(r"EX21_EXACT3D_RESULT .*", s)
        if not m: continue
        L = m.group(0); c = re.search(r"EX21_EXACT3D_CONFIG .*", s).group(0)
        g = lambda k, t=L: re.search(rf" {k}=(\S+)", t).group(1)
        r = {"ladder": tag, "case": g("case", c), "scheme": g("scheme"), "n": int(g("nx")), "nt": int(g("steps")),
             "convection": g("convection", c), "vx": g("vx", c), "vy": g("vy", c), "vz": g("vz", c),
             "dofs": int(g("dofs", c)), "spacetime_unknowns": int(g("spacetime_unknowns", c))}
        for k in ("y_L2L2", "p_L2L2", "u_L2L2", "y_L2L2_rel", "p_L2L2_rel", "u_L2L2_rel"):
            r[k] = float(g(k))
        for k in ("outer_its", "true_residual", "stationarity_rel", "march_rel"):
            r[k] = g(k)
        mr = re.search(r"MAXRSS=(\d+) wall=(\S+)", s)
        r["maxrss_MB"], r["wall_s"] = (int(mr.group(1)) // 1024, mr.group(2)) if mr else ("", "")
        mc = re.search(r"core_its_max=(\d+)", s); r["core_its_max"] = mc.group(1) if mc else ""
        runs.append(r)
    vary = "n" if len({r["n"] for r in runs}) > 1 else "nt"
    runs.sort(key=lambda r: r[vary])
    for i, r in enumerate(runs):
        for k in ("y_L2L2", "p_L2L2", "u_L2L2"):
            r["ord_" + k[0]] = "" if i == 0 else f"{math.log(runs[i-1][k] / r[k]) / math.log(r[vary] / runs[i-1][vary]):.2f}"
        rows.append(r)
cols = list(rows[0].keys())
with open(sys.argv[1], "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
for r in rows:
    print(f"{r['ladder']:<10}{r['case']:<13}{r['scheme']:<3} n={r['n']:<3} nt={r['nt']:<4} y={r['y_L2L2']:.3e}({r['ord_y']:>5}) p={r['p_L2L2']:.3e}({r['ord_p']:>5}) u={r['u_L2L2']:.3e}({r['ord_u']:>5}) out={r['outer_its']} march={r['march_rel']} stat={r['stationarity_rel']} core={r['core_its_max']} MB={r['maxrss_MB']} wall={r['wall_s']}")
