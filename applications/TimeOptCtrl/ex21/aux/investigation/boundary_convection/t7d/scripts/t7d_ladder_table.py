#!/usr/bin/env python3
"""Exact-OC ladders (t7d/logs/<tag>_<scheme>_nx*_nt*.log) -> CSV with observed orders.
usage: t7d_ladder_table.py <out.csv> <tag> [<tag> ...]"""
import csv, glob, math, os, re, sys
D = os.path.dirname(os.path.abspath(__file__)) + "/../logs/"
keys = ["y_L2L2", "y_half", "y_last", "p_L2L2", "p_half", "u_L2L2", "u_half", "u_Linf_L2",
        "y_L2L2_rel", "p_L2L2_rel", "u_L2L2_rel", "J_err_rel"]
rows = []
for tag in sys.argv[2:]:
    runs = []
    for f in glob.glob(f"{D}{tag}_*_nx*_nt*.log"):
        s = open(f).read()
        m = re.search(r"EX21_EXACT_RESULT .*", s)
        if not m:
            continue
        line = m.group(0)
        r = {"ladder": tag}
        r["scheme"] = re.search(r"scheme=(\S+)", line).group(1)
        r["nx"] = int(re.search(r" nx=(\d+)", line).group(1))
        r["nt"] = int(re.search(r"steps=(\d+)", line).group(1))
        for k in keys:
            mm = re.search(rf" {k}=(\S+)", line) or re.search(rf" {k}=(\S+)", s)
            r[k] = float(mm.group(1)) if mm else float("nan")
        st = re.search(r"EX21_DIRICHLET_STATIONARITY .*?kkt_stationarity_rel=(\S+) .*?max\|y_C-eta u\|=(\S+) ell=(\S+)", s)
        r["stationarity"], r["yC_minus_eta_u"], r["ell"] = st.groups() if st else ("", "", "")
        sv = re.search(r"EX21_EXACT_SOLVER_STATS outer_its=(\d+) .*?true_residual=(\S+)", s)
        r["outer"], r["true_res"] = sv.groups() if sv else ("", "")
        r["march_rel"] = re.search(r"march_rel=(\S+)", line).group(1)
        mr = re.search(r"MAXRSS=(\d+) wall=(\S+)", s)
        r["maxrss_MB"], r["wall_s"] = (int(mr.group(1)) // 1024, mr.group(2)) if mr else ("", "")
        runs.append(r)
    vary = "nx" if len({r["nx"] for r in runs}) > 1 else "nt"
    runs.sort(key=lambda r: r[vary])
    for i, r in enumerate(runs):
        for k in ("y_L2L2", "y_half", "p_L2L2", "u_L2L2", "u_half"):
            r["ord_" + k] = "" if i == 0 else f"{math.log(runs[i-1][k] / r[k]) / math.log(r[vary] / runs[i-1][vary]):.2f}"
        rows.append(r)
cols = ["ladder", "scheme", "nx", "nt", "ell"] + keys + [f"ord_{k}" for k in ("y_L2L2", "y_half", "p_L2L2", "u_L2L2", "u_half")] + \
       ["stationarity", "yC_minus_eta_u", "march_rel", "outer", "true_res", "maxrss_MB", "wall_s"]
with open(sys.argv[1], "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
for r in rows:
    print(f"{r['ladder']:<10}{r['scheme']:<3} nx={r['nx']:<3} nt={r['nt']:<4} y={r['y_L2L2']:.3e}({r['ord_y_L2L2']:>5}) yh={r['y_half']:.3e}({r['ord_y_half']:>5}) "
          f"p={r['p_L2L2']:.3e}({r['ord_p_L2L2']:>5}) u={r['u_L2L2']:.3e}({r['ord_u_L2L2']:>5}) uh={r['u_half']:.3e}({r['ord_u_half']:>5}) Jrel={r['J_err_rel']:.2e} st={r['stationarity']} out={r['outer']} MB={r['maxrss_MB']}")
