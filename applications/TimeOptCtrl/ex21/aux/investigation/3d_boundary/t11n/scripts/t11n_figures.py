#!/usr/bin/env python3
"""T11N figures: ../../figures/t11n_space_convergence.png, t11n_time_convergence.png
(from ../../t11n_space_convergence.csv and ../../t11n_time_convergence.csv)."""
import csv, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = os.path.dirname(os.path.abspath(__file__)) + "/../.."
C = {"y": "#2a6fdb", "p": "#d9822b", "u": "#3a9e5f"}
sp = list(csv.DictReader(open(R + "/t11n_space_convergence.csv")))
fig, axs = plt.subplots(1, 3, figsize=(14, 4.3))
for ax, lad, title in zip(axs, ("neuA", "neuB", "neuH"), ("wind (1,0.5,−0.75): w·n = +0.75 on Γ_c", "wind (−2,1,0.5): w·n = −0.5 on Γ_c", "heat (w off)")):
    rs = sorted((r for r in sp if r.get("ladder") == lad), key=lambda r: int(r["n"]))
    h = [1 / int(r["n"]) for r in rs]
    for k, lab in (("y", "y  L²(Ω)"), ("p", "p  L²(Ω)"), ("u", "u  L²(Γ_c)")):
        ax.loglog(h, [float(r[f"{k}_L2L2_rel"]) for r in rs], "o-", color=C[k], lw=2, ms=6, label=lab + " (rel.)")
    e0 = float(rs[-1]["y_L2L2_rel"]); ax.loglog(h, [e0 * (x / h[-1]) ** 3 for x in h], ":", color="#777", label="h³")
    e0 = float(rs[-1]["p_L2L2_rel"]); ax.loglog(h, [e0 * (x / h[-1]) ** 4 for x in h], "-.", color="#bbb", label="h⁴")
    ax.set_title(title, fontsize=9); ax.set_xlabel("h = 1/n"); ax.grid(True, which="both", alpha=0.25)
axs[0].set_ylabel("relative space-time L² error (CN, N = 64)"); axs[0].legend(frameon=False, fontsize=8)
fig.suptitle("T11N 3D Neumann control (Γ_c = {z = 0}, QUAD9 face): spatial convergence"); fig.tight_layout()
fig.savefig(R + "/figures/t11n_space_convergence.png", dpi=150)
tm = list(csv.DictReader(open(R + "/t11n_time_convergence.csv")))
fig, axs = plt.subplots(1, 3, figsize=(14, 4.3))
for ax, st, q, title in zip(axs[:2], ("BE n=6 ref N=512", "CN n=6 ref N=512"), (1, 2), ("BE", "CN")):
    rs = sorted((r for r in tm if r["study"] == st), key=lambda r: int(r["nt"]))
    dt = [1 / int(r["nt"]) for r in rs]
    for k, ls in (("y", "o-"), ("p", "o-"), ("u", "o-")):
        ax.loglog(dt, [float(r[k]) for r in rs], ls, color=C[k], lw=2, ms=5, label=k + (" (Γ_c)" if k == "u" else ""))
    ax.loglog(dt, [float(r["u_last"]) for r in rs], "x--", color=C["u"], lw=1, label="u last slot")
    e0 = float(rs[-1]["p"]); ax.loglog(dt, [e0 * 0.6 * (x / dt[-1]) ** q for x in dt], ":", color="#777", label=f"Δt^{q}")
    ax.set_title(f"{title}, n = 6, reference N = 512", fontsize=10); ax.set_xlabel("Δt (T = 1)"); ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
axs[0].set_ylabel("space-time L² difference to reference")
ax = axs[2]
for st, mk, lab in (("CN y layer split, 3D n=6", "o", "3D"), ("CN y layer split, 2D T7N neumann-conv nx=8 (attribution)", "s", "2D T7N")):
    rs = sorted((r for r in tm if r["study"] == st), key=lambda r: int(r["nt"]))
    dt = [1 / int(r["nt"]) for r in rs]
    ax.loglog(dt, [float(r["y_L2L2"]) for r in rs], mk + "-", color=C["y"], lw=1.6, label=f"y L²(0,T), {lab}")
    ax.loglog(dt, [float(r["y_L2_t_ge_0.25"]) for r in rs], mk + "--", color="#555", lw=1.6, label=f"y on [¼,T], {lab}")
e0 = float(rs[-1]["y_L2_t_ge_0.25"]); ax.loglog(dt, [e0 * (x / dt[-1]) ** 2 for x in dt], ":", color="#999", label="Δt²")
ax.set_title("CN state start-up layer: 3D Neumann vs 2D T7N", fontsize=10); ax.set_xlabel("Δt"); ax.grid(True, which="both", alpha=0.25)
ax.legend(frameon=False, fontsize=7)
fig.suptitle("T11N temporal self-convergence (wind (1,0.5,−0.75))"); fig.tight_layout()
fig.savefig(R + "/figures/t11n_time_convergence.png", dpi=150)
