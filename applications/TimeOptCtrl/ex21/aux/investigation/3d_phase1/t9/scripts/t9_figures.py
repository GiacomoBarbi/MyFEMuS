#!/usr/bin/env python3
"""T9 figures: ../figures/3d_space_convergence.png (from ../t9_3d_space_convergence.csv)
and ../figures/3d_time_convergence.png (from ../t9_3d_time_convergence.csv)."""
import csv, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = os.path.dirname(os.path.abspath(__file__)) + "/../.."
os.makedirs(R + "/figures", exist_ok=True)
C = {"y": "#2a6fdb", "p": "#d9822b", "u": "#3a9e5f"}
sp = list(csv.DictReader(open(R + "/t9_3d_space_convergence.csv")))
fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
for ax, (lad, title) in zip(axs, (("spaceA", "sin data, wind (1,0.5,−0.75)"), ("spaceB", "sin³ data, wind (1,0.5,−0.75)"), ("spaceH", "sin³ data, heat (w off)"))):
    rs = sorted((r for r in sp if r["ladder"] == lad), key=lambda r: int(r["n"]))
    if not rs: continue
    h = [1 / int(r["n"]) for r in rs]
    ax.loglog(h, [float(r["y_L2L2_rel"]) for r in rs], "o-", color=C["y"], lw=2, ms=6, label="y (rel. L²L²)")
    ax.loglog(h, [float(r["p_L2L2_rel"]) for r in rs], "s-", color=C["p"], lw=2, ms=6, label="p = u (rel. L²L²; u = −p/α_u)")
    e0 = float(rs[-1]["y_L2L2_rel"]); h0 = h[-1]
    ax.loglog(h, [e0 * (x / h0) ** 3 for x in h], ":", color="#777", label="h³")
    if lad == "spaceA": ax.loglog(h, [e0 * (x / h0) ** 2.5 for x in h], "--", color="#aaa", label="h^2.5")
    ax.set_title(title, fontsize=10); ax.set_xlabel("h = 1/n (HEX27 Q2)"); ax.grid(True, which="both", alpha=0.25)
axs[0].set_ylabel("relative space-time L² error"); axs[0].legend(frameon=False, fontsize=8)
fig.suptitle("T9 3D spatial convergence, CN, N = 64 (exact OC)"); fig.tight_layout(); fig.savefig(R + "/figures/3d_space_convergence.png", dpi=150)
tm = list(csv.DictReader(open(R + "/t9_3d_time_convergence.csv")))
fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
for ax, s in zip(axs, ("BE", "CN")):
    rs = sorted((r for r in tm if r["scheme"] == s), key=lambda r: int(r["nt"]))
    if not rs: continue
    dt = [1 / int(r["nt"]) for r in rs]
    for k in ("y", "p", "u"):
        ax.loglog(dt, [float(r[k]) for r in rs], "o-", color=C[k], lw=2, ms=6, label=k)
    q = 1 if s == "BE" else 2; e0 = float(rs[-1]["y" if s == "BE" else "p"]) * (1 if s == "BE" else 0.6); d0 = dt[-1]
    ax.loglog(dt, [e0 * (x / d0) ** q for x in dt], ":", color="#777", label=f"Δt^{q}")
    if s == "CN":
        ax.annotate("y: CN initial layer\n(t ≥ 0.25 and y(T): order 2)", xy=(dt[-1], float(rs[-1]["y"])), xytext=(dt[-1] * 1.3, float(rs[-1]["y"]) * 0.3),
                    fontsize=8, arrowprops=dict(arrowstyle="->", color="#555"))
    ax.set_title(f"{s}, n = {rs[0]['n']}, ref N = {rs[0]['ref']}", fontsize=10); ax.set_xlabel("Δt"); ax.grid(True, which="both", alpha=0.25)
axs[0].set_ylabel("space-time L² difference to reference (same mesh)")
for ax in axs: ax.legend(frameon=False, fontsize=8, loc="lower right")
fig.suptitle("T9 3D temporal self-convergence, wind (1,0.5,−0.75), sin³ data"); fig.tight_layout(); fig.savefig(R + "/figures/3d_time_convergence.png", dpi=150)
