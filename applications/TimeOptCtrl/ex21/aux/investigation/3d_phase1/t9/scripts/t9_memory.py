#!/usr/bin/env python3
"""T9 memory audit -> ../t9_3d_memory.csv and ../figures/3d_memory_vs_unknowns.png.
Sources: the serial exact ladders (logs/exact, tight policy, SCHUR_SWEEP), the
solver baseline (logs/solver, production rtol 1e-8, n=8 N=32) and the time
ladders (logs/time).  MaxRSS from /usr/bin/time (KB).  Fits RSS = b*U (through the origin) over
the LU exact-ladder runs with n >= 4 (U = space-time unknowns = 3 * dofs * N) and
extrapolates."""
import csv, glob, os, re
import numpy as np
D = os.path.dirname(os.path.abspath(__file__)) + "/.."
def g(pat, s, cast=str, d=""):
    m = re.search(pat, s); return cast(m.group(1)) if m else d
rows = []
for p in sorted(glob.glob(f"{D}/logs/exact/*.log")):
    L = open(p).read(); tag = os.path.basename(p)[:-4]
    if "MAXRSS" not in L or "EX21_EXACT3D_CONFIG" not in L: continue
    n = g(r"EX21_EXACT3D_CONFIG .*? nx=(\d+)", L, int); N = g(r"EX21_EXACT3D_CONFIG .*? steps=(\d+)", L, int)
    dofs = g(r" dofs=(\d+)", L, int); U = g(r"spacetime_unknowns=(\d+)", L, int)
    rs, ri = g(r"right_solves=(\d+)", L, int, 0), g(r"right_its=(\d+)", L, int, 0)
    rows.append({"source": "exact_ladder", "tag": tag, "backend": "gamg" if ri > rs else "lu", "tolerance": "tight 1e-12",
                 "n": n, "N": N, "spatial_dofs": dofs, "spacetime_unknowns": U,
                 "maxrss_MB": round(g(r"MAXRSS=(\d+)", L, int) / 1024), "bytes_per_unknown": round(g(r"MAXRSS=(\d+)", L, int) * 1024 / U),
                 "outer_its": g(r"EX21_EXACT3D_SOLVER_STATS outer_its=(\d+)", L), "spatial_its_per_solve_max": g(r"right_max=(\d+)", L),
                 "spatial_solves": rs + g(r"left_solves=(\d+)", L, int, 0) + g(r"ysplit_solves=(\d+)", L, int, 0),
                 "solve_wall_s": g(r"solve_wall=(\S+)", L), "wall_s": g(r"wall=(\S+)\s*$", L)})
for p in sorted(glob.glob(f"{D}/logs/solver/*_nu*.log")):
    L = open(p).read(); tag = os.path.basename(p)[:-4]
    if "MAXRSS" not in L: continue
    dofs, N = 17 ** 3, 32; U = 3 * dofs * N
    rows.append({"source": "solver_baseline", "tag": tag, "backend": tag.split("_")[-1], "tolerance": "production 1e-8",
                 "n": 8, "N": N, "spatial_dofs": dofs, "spacetime_unknowns": U,
                 "maxrss_MB": round(g(r"MAXRSS=(\d+)", L, int) / 1024), "bytes_per_unknown": round(g(r"MAXRSS=(\d+)", L, int) * 1024 / U),
                 "outer_its": g(r"outer_iterations=(\d+)", L), "spatial_its_per_solve_max": g(r" right=\d+/\d+/\S+/(\d+)", L),
                 "spatial_solves": "", "solve_wall_s": g(r"t_total=(\S+)", L), "wall_s": g(r"wall=(\S+)\s*$", L)})
lu = [r for r in rows if r["source"] == "exact_ladder" and r["backend"] == "lu" and r["N"] == 64 and r["n"] >= 4]
U = np.array([r["spacetime_unknowns"] for r in lu], float); M = np.array([r["maxrss_MB"] for r in lu], float)
a, b = 0.0, float(np.sum(U * M) / np.sum(U * U))   # proportional fit (n >= 4; n = 2 is dominated by the fixed ~90 MB start-up)
for n, N in ((16, 64), (20, 64), (24, 64), (32, 64), (16, 128)):
    u = 3 * (2 * n + 1) ** 3 * N
    rows.append({"source": "extrapolation (proportional fit of the LU exact ladder n>=4)", "tag": f"n{n}_N{N}", "backend": "lu",
                 "n": n, "N": N, "spatial_dofs": (2 * n + 1) ** 3, "spacetime_unknowns": u, "maxrss_MB": round(a + b * u)})
keys = list(dict.fromkeys(k for r in rows for k in r))
with open(D + "/../t9_3d_memory.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
for r in rows: print(" ".join(f"{k}={r.get(k,'')}" for k in ("source", "tag", "backend", "n", "N", "spacetime_unknowns", "maxrss_MB", "bytes_per_unknown", "outer_its", "spatial_its_per_solve_max", "wall_s")))
print(f"fit: RSS_MB = {a:.1f} + {b*1e6:.1f} MB per 1e6 unknowns  (~{b*1024*1024:.0f} B/unknown)")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
os.makedirs(D + "/../figures", exist_ok=True)
fig, ax = plt.subplots(figsize=(6.4, 4.4))
for bk, mk, c in (("lu", "o", "#2a6fdb"), ("gamg", "s", "#d9822b")):
    ex = [r for r in rows if r["source"] == "exact_ladder" and r["backend"] == bk]
    if ex: ax.loglog([r["spacetime_unknowns"] for r in ex], [r["maxrss_MB"] for r in ex], mk, color=c, label=f"exact ladders, {bk.upper()} spatial solves")
uu = np.logspace(np.log10(1e5), np.log10(3 * 65 ** 3 * 64), 50)
ax.loglog(uu, a + b * uu, "--", color="#777", lw=1.5, label=f"proportional fit ({b*1024*1024:.0f} B/unknown)")
ax.set_xlabel("space-time unknowns 3·(2n+1)³·N"); ax.set_ylabel("MaxRSS [MB]"); ax.grid(True, which="both", alpha=0.25)
ax.set_title("T9 3D: serial memory vs unknowns (SCHUR_SWEEP)"); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(D + "/../figures/3d_memory_vs_unknowns.png", dpi=150)
