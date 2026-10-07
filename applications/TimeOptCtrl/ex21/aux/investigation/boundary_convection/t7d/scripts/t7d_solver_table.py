#!/usr/bin/env python3
"""t7d/logs/solver/*.log -> boundary_convection/t7d_solver.csv"""
import csv, glob, os, re
D = os.path.dirname(os.path.abspath(__file__)) + "/.."
rows = []
def g(p, s):
    m = re.search(p, s); return m.group(1) if m else ""
for f in sorted(glob.glob(D + "/logs/solver/*.log")):
    s = open(f).read(); tag = os.path.basename(f)[:-4]
    rows.append({
        "tag": tag, "nrank": g(r"nrank=(\d+)", s), "scheme": g(r"time_scheme=(\S+)", s), "mesh": g(r"mesh=(\S+)", s), "N": g(r" N=(\d+)", s),
        "convection": g(r"convection=(\S+)", s), "h1_length": g(r"h1_length=(\S+)", s), "pc": g(r"EX20_PAR_SOLVE .*? pc=(\S+)", s),
        "reason": g(r"EX20_PAR_SOLVE .*? reason=(\S+)", s), "outer": g(r"outer_iterations=(\d+)", s),
        "true_res": g(r"true_residual=(\S+)", s),
        "backend": g(r"forward_backend=(\S+)", s), "dense_dim": g(r"EX20_REDUCED_SCHUR .*?dimension=(\d+)", s),
        "J_total": g(r"J_total=(\S+)", s) or g(r"J_track=(\S+)", s),
        "t_total": g(r"t_total=(\S+)", s), "maxrss_MB": str(int(g(r"MAXRSS=(\d+)", s) or 0) // 1024), "wall": g(r"wall=(\S+)", s),
        "refusal": g(r"what\(\):\s+(.*)", s)[:120],
    })
out = D + "/../t7d_solver.csv"
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows:
    print(f"{r['tag']:<36} np={r['nrank']:<2} out={r['outer']:<3} reason={r['reason']:<3} J={r['J_total']:<22} be={r['backend']:<16} dim={r['dense_dim']:<6} t={r['t_total']:<9} MB={r['maxrss_MB']:<6} {r['refusal'][:70]}")
