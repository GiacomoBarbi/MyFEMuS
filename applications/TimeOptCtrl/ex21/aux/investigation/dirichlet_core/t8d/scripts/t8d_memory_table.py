#!/usr/bin/env python3
"""logs/memory/*.log -> dirichlet_core/t8d_memory.csv, fits, figures."""
import csv, glob, os, re
import numpy as np
D = os.path.dirname(os.path.abspath(__file__)) + "/.."
OUT = D + "/.."
rows = []
def g(p, s, cast=float):
    m = re.search(p, s); return cast(m.group(1)) if m else None
for f in glob.glob(D + "/logs/memory/*.log"):
    s = open(f).read(); mode, nx, nt = re.match(r"(\w+)_nx(\d+)_nt(\d+)", os.path.basename(f)).groups()
    nx, nt = int(nx), int(nt); n = (2 * nx + 1) ** 2; C = 2 * nx + 1; m = nt * C
    if g(r"MAXRSS=(\d+)", s) is None: continue
    r = {"mode": mode, "nx": nx, "steps": nt, "n_spatial": n, "C": C, "m": m, "Nt_n": nt * n,
         "maxrss_MB": g(r"MAXRSS=(\d+)", s) / 1024, "wall": g(r"wall=(\S+)", s),
         "t_operators": g(r"t_operators=(\S+)", s), "t_pcsetup": g(r"t_pcsetup=(\S+)", s),
         "t_kspsolve": g(r"t_kspsolve=(\S+)", s), "t_total": g(r"t_total=(\S+)", s),
         "reduced_setup": g(r"EX20_REDUCED_SCHUR .*?setup_seconds=(\S+)", s),
         "sweep_solves": g(r"sweep_solves=(\d+)", s, int), "outer": g(r"EX20_SOLVERSTATS .*? outer=(\d+)", s, int),
         "core_its_total": g(r"core_its_total=(\d+)", s, int), "core_seconds": g(r"core_seconds=(\S+)", s),
         "true_residual": g(r"true_residual=(\S+)", s),
         # dense-object model: K 8 m^2 + Gram 8 Nt(Nt+1)/2 |C|^2 + responses 8 Nt n |C|  (bytes)
         "dense_objects_model_MB": (8 * m * m + 8 * nt * (nt + 1) / 2 * C * C + 8 * nt * n * C) / 2**20 if mode == "dense" else 0.0}
    rows.append(r)
rows.sort(key=lambda r: (r["mode"], r["nx"], r["steps"]))
# fits
dense = [r for r in rows if r["mode"] == "dense"]; it = [r for r in rows if r["mode"] == "iterative"]
A = np.array([[1, r["m"] ** 2] for r in dense]); y = np.array([r["maxrss_MB"] for r in dense])
cd, *_ = np.linalg.lstsq(A, y, rcond=None)
Ai = np.array([[1, r["Nt_n"]] for r in it]); yi = np.array([r["maxrss_MB"] for r in it])
ci, *_ = np.linalg.lstsq(Ai, yi, rcond=None)
# log-log slopes vs m at fixed nx (nt ladder)
def slope(rs):
    x = np.log([r["m"] for r in rs]); yy = np.log([r["maxrss_MB"] for r in rs]); return np.polyfit(x, yy, 1)[0]
for r in rows:
    r["fit_MB"] = cd[0] + cd[1] * r["m"] ** 2 if r["mode"] == "dense" else ci[0] + ci[1] * r["Nt_n"]
    # model validated on the 12 dense runs: dense = iterative + 12 m^2 + 8 Nt n |C| bytes
    r["dense_predicted_MB"] = (r["maxrss_MB"] + (12 * r["m"] ** 2 + 8 * r["Nt_n"] * r["C"]) / 2**20) if r["mode"] == "iterative" else r["maxrss_MB"]
with open(OUT + "/t8d_memory.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"dense fit: MaxRSS_MB = {cd[0]:.1f} + {cd[1]:.4e} m^2   (8 bytes/entry x 1.5 objects of m^2 -> {8*1.5/2**20:.4e})")
print(f"iterative fit: MaxRSS_MB = {ci[0]:.1f} + {ci[1]:.4e} (Nt n)   -> {ci[1]*2**20/8:.1f} doubles per space-time DOF")
for nx in (16, 32, 48):
    for mode in ("dense", "iterative"):
        rs = [r for r in rows if r["mode"] == mode and r["nx"] == nx and r["steps"] >= 64]
        if len(rs) >= 2: print(f"nx={nx} {mode:<9} log-log slope of MaxRSS vs m (Nt>=64): {slope(rs):.2f}")
print(f"{'mode':<10}{'nx':>4}{'Nt':>6}{'m':>7}{'MB':>9}{'fit':>9}{'wall':>8}{'t_pcsetup':>10}{'t_ksp':>8}{'core_s':>8}{'its':>5}{'outer':>6}")
for r in rows:
    print(f"{r['mode']:<10}{r['nx']:>4}{r['steps']:>6}{r['m']:>7}{r['maxrss_MB']:>9.0f}{r['fit_MB']:>9.0f}{r['wall']:>8.1f}{r['t_pcsetup']:>10.2f}{r['t_kspsolve']:>8.2f}{(r['core_seconds'] or 0):>8.2f}{(r['core_its_total'] or 0):>5}{r['outer']:>6}")
# figures
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6.4, 4.4))
for mode, mk in (("dense", "o"), ("iterative", "s")):
    for nx, c in ((16, "#2a6f97"), (32, "#c8553d"), (48, "#588157"), (64, "#7b2cbf"), (96, "#333333")):
        rs = [r for r in rows if r["mode"] == mode and r["nx"] == nx]
        if rs: ax.loglog([r["m"] for r in rs], [r["maxrss_MB"] for r in rs], mk + ("-" if mode == "dense" else "--"), color=c, label=f"{mode} nx={nx}")
# dense prediction where it was NOT run: iterative MaxRSS + the measured dense-object
# excess 12 m^2 + 8 Nt n |C| bytes (K, Gram, responses; fitted to 1.00 on the 12 dense runs)
ran = {(r["nx"], r["steps"]) for r in rows if r["mode"] == "dense"}
pred = [r for r in rows if r["mode"] == "iterative" and (r["nx"], r["steps"]) not in ran]
ax.loglog([r["m"] for r in pred], [r["maxrss_MB"] + (12 * r["m"] ** 2 + 8 * r["Nt_n"] * r["C"]) / 2**20 for r in pred],
          "o", mfc="none", color="gray", label="dense, predicted (not run)")
ax.set_xlabel("core size m = Nt |C|"); ax.set_ylabel("MaxRSS (MB)"); ax.legend(fontsize=7, ncol=2); ax.grid(alpha=.3, which="both")
ax.set_title("Dirichlet-reduced core: memory"); fig.tight_layout(); fig.savefig(OUT + "/t8d/results/memory_vs_core_size.png", dpi=130)
fig, ax = plt.subplots(figsize=(6.4, 4.4))
for mode, mk in (("dense", "o-"), ("iterative", "s--")):
    rs = [r for r in rows if r["mode"] == mode and r["nx"] in (16, 32, 48)]
    ax.loglog([r["m"] for r in rs], [r["wall"] for r in rs], mk[0], label=mode)
ax.set_xlabel("core size m = Nt |C|"); ax.set_ylabel("wall (s)"); ax.legend(); ax.grid(alpha=.3, which="both")
ax.set_title("Dirichlet-reduced core: wall time (serial)"); fig.tight_layout(); fig.savefig(OUT + "/t8d/results/dense_vs_matrixfree_time.png", dpi=130)
