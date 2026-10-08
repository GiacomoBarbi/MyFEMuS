#!/usr/bin/env python3
"""T11N memory audit -> ../../t11n_memory.csv and ../../figures/t11n_memory.png.
Serial MaxRSS of the 3D Neumann runs (exact ladders, solver audit) with spatial
DOFs, trace DOFs (2n+1)^2, space-time unknowns U = 3 (2n+1)^3 N, compared with the
T9 law (2385 B/unknown) and with SAME-SIZE distributed runs from T9 / T10
(3d_phase1/t9_3d_memory.csv, 3d_phase2/t10_memory.csv): a dense surface object
would show up as an excess over them."""
import csv, glob, os, re
D = os.path.dirname(os.path.abspath(__file__)) + "/.."
I = os.path.abspath(D + "/../..")
T9_BYTES = 2385.0
def g(p, s, cast=str, d=""):
    m = re.search(p, s); return cast(m.group(1)) if m else d
rows = []
for p in sorted(glob.glob(D + "/logs/exact/*.log")):
    s = open(p).read()
    if "MAXRSS" not in s or "EX21_EXACT3D_CONFIG" not in s: continue
    n = g(r"EX21_EXACT3D_CONFIG .*? nx=(\d+)", s, int); N = g(r"EX21_EXACT3D_CONFIG .*? steps=(\d+)", s, int)
    U = g(r"spacetime_unknowns=(\d+)", s, int); rss = g(r"MAXRSS=(\d+)", s, int)
    rows.append({"source": "Neumann exact ladder", "tag": os.path.basename(p)[:-4], "n": n, "N": N,
                 "spatial_dofs": (2 * n + 1) ** 3, "trace_dofs": (2 * n + 1) ** 2, "spacetime_unknowns": U,
                 "maxrss_MB": round(rss / 1024), "bytes_per_unknown": round(rss * 1024 / U),
                 "ratio_to_T9_law": f"{rss * 1024 / U / T9_BYTES:.2f}", "outer_its": g(r"EX21_EXACT3D_SOLVER_STATS outer_its=(\d+)", s)})
for p in sorted(glob.glob(D + "/logs/solver/*_sweep_*.log")) + glob.glob(D + "/logs/solver/be_additive.log"):
    s = open(p).read()
    if "MAXRSS" not in s: continue
    n, N = 8, 32; U = 3 * (2 * n + 1) ** 3 * N; rss = g(r"MAXRSS=(\d+)", s, int)
    rows.append({"source": "Neumann solver audit", "tag": os.path.basename(p)[:-4], "n": n, "N": N,
                 "spatial_dofs": (2 * n + 1) ** 3, "trace_dofs": (2 * n + 1) ** 2, "spacetime_unknowns": U,
                 "maxrss_MB": round(rss / 1024), "bytes_per_unknown": round(rss * 1024 / U),
                 "ratio_to_T9_law": f"{rss * 1024 / U / T9_BYTES:.2f}", "outer_its": g(r"outer_iterations=(\d+)", s)})
ref = {}
for path, lab in ((I + "/3d_phase1/t9_3d_memory.csv", "T9 distributed (Dirichlet)"), (I + "/3d_phase2/t10_memory.csv", "T10 natural distributed")):
    for r in csv.DictReader(open(path)):
        try:
            key = (int(r["n"]), int(r["N"]))
        except (ValueError, KeyError):
            continue
        if "extrapolation" in r.get("source", ""): continue
        if lab.startswith("T10") and not r["tag"].startswith("natA"): continue
        if lab.startswith("T9") and not r["tag"].startswith(("spaceA", "spaceB")): continue
        ref.setdefault(key, {})[lab] = int(float(r["maxrss_MB"]))
for r in rows:
    m = ref.get((r["n"], r["N"]), {})
    r["T9_same_size_MB"] = m.get("T9 distributed (Dirichlet)", ""); r["T10_same_size_MB"] = m.get("T10 natural distributed", "")
with open(D + "/../t11n_memory.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in sorted(rows, key=lambda r: r["spacetime_unknowns"]):
    print(f"{r['tag']:<22s} n={r['n']:>2} N={r['N']:>3} trace={r['trace_dofs']:>4} U={r['spacetime_unknowns']:>8d} "
          f"MaxRSS={r['maxrss_MB']:>5d} MB ratio={r['ratio_to_T9_law']} T9={r['T9_same_size_MB']} T10={r['T10_same_size_MB']}")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
os.makedirs(D + "/../figures", exist_ok=True)
fig, ax = plt.subplots(figsize=(6.4, 4.4))
ex = sorted((r for r in rows if r["source"].endswith("ladder")), key=lambda r: r["spacetime_unknowns"])
ax.loglog([r["spacetime_unknowns"] for r in ex], [r["maxrss_MB"] for r in ex], "o", color="#2a6fdb", ms=7, label="3D Neumann (exact ladders, LU)")
t9 = [(r["spacetime_unknowns"], r["T9_same_size_MB"]) for r in ex if r["T9_same_size_MB"] != ""]
t10 = [(r["spacetime_unknowns"], r["T10_same_size_MB"]) for r in ex if r["T10_same_size_MB"] != ""]
if t9: ax.loglog(*zip(*t9), "s", mfc="none", color="#d9822b", ms=10, label="T9 distributed, same size")
if t10: ax.loglog(*zip(*t10), "^", mfc="none", color="#3a9e5f", ms=10, label="T10 natural distributed, same size")
uu = [1e5, 1e7]; ax.loglog(uu, [T9_BYTES * u / 1048576 for u in uu], "--", color="#888", label="T9 law 2385 B/unknown")
ax.set_xlabel("space-time unknowns 3(2n+1)³N"); ax.set_ylabel("MaxRSS [MB]"); ax.grid(True, which="both", alpha=0.25)
ax.set_title("T11N 3D Neumann control: serial memory (CN, N = 64)"); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(D + "/../figures/t11n_memory.png", dpi=150)
