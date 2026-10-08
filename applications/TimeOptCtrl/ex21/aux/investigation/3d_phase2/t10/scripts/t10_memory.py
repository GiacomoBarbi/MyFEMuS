#!/usr/bin/env python3
"""T10 memory audit -> ../../t10_memory.csv.

Does adding boxes / sources / the natural BC change the T9 serial memory law
(MaxRSS ~ 2.4 KB per space-time unknown, T9 report section 14)?  Sources:
  logs/exact/*.log   natural-conv-3d exact ladders (natural BC, full domain)
  logs/runs/*.log    production runs: multi-zone (mzs/mzt), sources (S*),
                     natural time ladders (natt)
MaxRSS from /usr/bin/time; U = 3 * dofs * N; outer iterations from the logs.
For each row the ratio to the T9 law (2385 B/unknown) is reported; small
problems carry the ~70-90 MB start-up and are flagged (U < 1e5)."""
import csv, glob, os, re
D = os.path.dirname(os.path.abspath(__file__)) + "/.."
T9_BYTES = 2385.0
def g(p, s, cast=str, d=""):
    m = re.search(p, s); return cast(m.group(1)) if m else d
rows = []
for p in sorted(glob.glob(D + "/logs/exact/*.log")):
    s = open(p).read()
    if "MAXRSS" not in s or "EX21_EXACT3D_CONFIG" not in s: continue
    U = g(r"spacetime_unknowns=(\d+)", s, int); rss = g(r"MAXRSS=(\d+)", s, int)
    rows.append({"source": "exact ladder (natural-conv-3d)", "tag": os.path.basename(p)[:-4],
                 "features": "natural BC, full domain", "n": g(r"EX21_EXACT3D_CONFIG .*? nx=(\d+)", s, int),
                 "N": g(r"EX21_EXACT3D_CONFIG .*? steps=(\d+)", s, int), "spatial_dofs": g(r" dofs=(\d+)", s, int),
                 "spacetime_unknowns": U, "maxrss_MB": round(rss / 1024), "bytes_per_unknown": round(rss * 1024 / U),
                 "ratio_to_T9_law": f"{rss * 1024 / U / T9_BYTES:.2f}",
                 "outer_its": g(r"EX21_EXACT3D_SOLVER_STATS outer_its=(\d+)", s)})
feat = {"mzs": "octant multi-zone 2+2 boxes, 2 sources, natural", "mzt": "octant multi-zone 2+2 boxes, 2 sources, natural",
        "natt": "natural BC exact data (solve-only)", "S": "multi-zone + source boxes (sources study)",
        "natsA": "natural BC exact data, wind A (fixed-dt self-conv)", "natsH": "natural BC exact data, heat (fixed-dt self-conv)",
        "t9ctl": "CONTROL: T9-type Dirichlet, full domain, no regions",
        "bk_lu": "octant multi-zone, LU spatial", "bk_gamg": "octant multi-zone, GMRES+GAMG spatial",
        "bk_asm": "octant multi-zone, GMRES+ASM/ILU spatial"}
for p in sorted(glob.glob(D + "/logs/runs/*.log")):
    s = open(p).read(); tag = os.path.basename(p)[:-4]
    if "MAXRSS" not in s: continue
    m = re.match(r"(\w+?)_(be|cn)_n(\d+)_nt(\d+)", tag)
    if not m: continue
    n, N = int(m.group(3)), int(m.group(4)); dofs = (2 * n + 1) ** 3; U = 3 * dofs * N
    rss = g(r"MAXRSS=(\d+)", s, int)
    key = m.group(1) if m.group(1) in feat else "S"
    rows.append({"source": "production run", "tag": tag, "features": feat[key], "n": n, "N": N, "spatial_dofs": dofs,
                 "spacetime_unknowns": U, "maxrss_MB": round(rss / 1024), "bytes_per_unknown": round(rss * 1024 / U),
                 "ratio_to_T9_law": f"{rss * 1024 / U / T9_BYTES:.2f}",
                 "outer_its": g(r"outer_iterations=(\d+)", s)})
for r in rows:
    r["note"] = "start-up dominated (U < 1e5)" if r["spacetime_unknowns"] < 1e5 else ""
with open(D + "/../t10_memory.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
big = [r for r in rows if r["spacetime_unknowns"] >= 4e5]
for r in sorted(big, key=lambda r: r["spacetime_unknowns"]):
    print(f"{r['tag']:<28s} {r['features']:<48s} U={r['spacetime_unknowns']:>9d} MaxRSS={r['maxrss_MB']:>6d} MB "
          f"{r['bytes_per_unknown']:>5d} B/unknown ratio={r['ratio_to_T9_law']} outer={r['outer_its']}")
