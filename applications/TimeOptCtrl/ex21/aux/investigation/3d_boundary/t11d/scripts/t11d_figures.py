#!/usr/bin/env python3
"""T11D figures (../../figures/): t11d_space_convergence.png, t11d_time_convergence.png,
t11d_core_iterations.png, t11d_memory.png, from the t11d_*.csv tables."""
import csv, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = os.path.dirname(os.path.abspath(__file__)) + "/../.."
C = {"y": "#2a6fdb", "p": "#d9822b", "u": "#3a9e5f"}
rd = lambda f: list(csv.DictReader(open(f"{R}/{f}")))
# ---- space
sp = rd("t11d_space_convergence.csv")
fig, axs = plt.subplots(1, 3, figsize=(14, 4.3))
for ax, lad, title in zip(axs, ("dirN", "dirH", "dirA"), ("normal wind (0,0,−0.75)", "heat (w off)",
                                                           "wind (1,0.5,−0.75) (tangential part ≠ 0)")):
    rs = sorted((r for r in sp if r.get("ladder") == lad), key=lambda r: int(r["n"]))
    h = [1 / int(r["n"]) for r in rs]
    for k, lab in (("y", "y  L²(Ω)"), ("p", "p  L²(Ω)"), ("u", "u  L²(Γ_c)")):
        ax.loglog(h, [float(r[f"{k}_L2L2"]) for r in rs], "o-", color=C[k], lw=2, ms=6, label=lab)
    e0 = float(rs[-1]["y_L2L2"]); ax.loglog(h, [e0 * (x / h[-1]) ** 3 for x in h], ":", color="#777", label="h³")
    ax.set_title(title, fontsize=9); ax.set_xlabel("h = 1/n"); ax.grid(True, which="both", alpha=0.25)
axs[0].set_ylabel("space-time L² error (CN, N = 256)"); axs[0].legend(frameon=False, fontsize=8)
fig.suptitle("T11D 3D Dirichlet control (Γ_c = {z = 0}, int(Γ_c) trace, HEX27/QUAD9): spatial convergence"); fig.tight_layout()
fig.savefig(R + "/figures/t11d_space_convergence.png", dpi=150)
# ---- time
tm = rd("t11d_time_convergence.csv")
fig, axs = plt.subplots(1, 2, figsize=(10, 4.3))
for ax, st, q, title in zip(axs, ("BE n=6 ref N=1024, wind (0,0,-0.75)", "CN n=6 ref N=512, wind (0,0,-0.75)"), (1, 2),
                            ("BE (reference N = 1024)", "CN (reference N = 512)")):
    rs = sorted((r for r in tm if r["study"] == st), key=lambda r: int(r["nt"]))
    dt = [1 / int(r["nt"]) for r in rs]
    for k in ("y", "p", "u"):
        ax.loglog(dt, [float(r[k]) for r in rs], "o-", color=C[k], lw=2, ms=5, label=k + (" (Γ_c, nodal)" if k == "u" else ""))
    ax.loglog(dt, [float(r["u_half"]) for r in rs], "s--", color=C["u"], lw=1, ms=4, label="u on t ≤ T/2")
    ax.loglog(dt, [float(r["u_last"]) for r in rs], "x:", color=C["u"], lw=1, label="u last slot")
    e0 = float(rs[-1]["y"]); ax.loglog(dt, [e0 * (x / dt[-1]) ** q for x in dt], ":", color="#777", label=f"Δt^{q}")
    ax.set_title(f"{title}, n = 6", fontsize=10); ax.set_xlabel("Δt (T = 1)"); ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
axs[0].set_ylabel("space-time L² difference to reference")
fig.suptitle("T11D temporal self-convergence"); fig.tight_layout()
fig.savefig(R + "/figures/t11d_time_convergence.png", dpi=150)
# ---- core iterations
cs = rd("t11d_core_scaling.csv")
fig, axs = plt.subplots(1, 3, figsize=(14, 4.3))
W = {"heat": ("#555", "heat"), "wA": ("#2a6fdb", "wind (1,0.5,−0.75)"), "wN": ("#3a9e5f", "wind (0,0,−0.75)")}
for w, (col, lab) in W.items():
    rs = sorted((r for r in cs if r["study"] == "mesh" and r["wind"] == w), key=lambda r: int(r["n"]))
    if rs: axs[0].plot([int(r["n"]) for r in rs], [int(r["core_its_max"]) for r in rs], "o-", color=col, label=lab)
    rs = sorted((r for r in cs if r["study"] == "time" and r["wind"] == w), key=lambda r: int(r["nt"]))
    if rs: axs[1].semilogx([int(r["nt"]) for r in rs], [int(r["core_its_max"]) for r in rs], "o-", color=col, label=lab)
    for s, mk in (("BE", "s--"), ("CN", "o-")):
        rs = sorted((r for r in cs if r["study"] == "alpha" and r["wind"] == w and r["scheme"] == s), key=lambda r: float(r["alpha_u"]))
        if rs: axs[2].loglog([float(r["alpha_u"]) for r in rs], [int(r["core_its_max"]) for r in rs], mk, color=col, label=f"{lab}, {s}")
axs[0].set(xlabel="n (Nt = 32, α_u = 1e-2)", ylabel="CG iterations (core)", ylim=(0, 12), title="mesh")
axs[1].set(xlabel="Nt (n = 4)", ylim=(0, 12), title="time steps")
axs[2].set(xlabel="α_u (n = 4, Nt = 32)", title="regularisation"); axs[2].invert_xaxis()
for ax in axs: ax.grid(True, which="both", alpha=0.25); ax.legend(frameon=False, fontsize=7)
fig.suptitle("T11D matrix-free reduced core K = R + βH: CG (R_T-preconditioned) iterations, ω adjacent to Γ_c"); fig.tight_layout()
fig.savefig(R + "/figures/t11d_core_iterations.png", dpi=150)
# ---- memory
mm = rd("t11d_memory.csv")
fig, ax = plt.subplots(figsize=(6.5, 4.5))
for core, col in (("dense", "#d9822b"), ("iterative", "#2a6fdb")):
    rs = sorted((r for r in mm if r["core"] == core), key=lambda r: int(r["m"]))
    ax.loglog([int(r["m"]) for r in rs], [float(r["rss_MB"]) for r in rs], "o", color=col, label=f"{core} core: peak RSS")
rs = sorted((r for r in mm if r["core"] == "dense"), key=lambda r: int(r["m"]))
ax.loglog([int(r["m"]) for r in rs], [float(r["dense_excess_MB"]) for r in rs], "D", mfc="none", color="#d9822b", label="measured dense excess RSS_dense − RSS_iter")
ax.loglog([int(r["m"]) for r in rs], [float(r["predicted_excess_MB"]) for r in rs], "x", color="#555", label="predicted 12 B m² + 8 B Nt N |C|")
ax.set(xlabel="m = Nt |C|", ylabel="MB"); ax.grid(True, which="both", alpha=0.25); ax.legend(frameon=False, fontsize=8)
ax.set_title("T11D memory: dense vs matrix-free core (n = 4, 6, 8; CN, wind A)", fontsize=9); fig.tight_layout()
fig.savefig(R + "/figures/t11d_memory.png", dpi=150)
