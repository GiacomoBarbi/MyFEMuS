#!/usr/bin/env python3
"""T11H1 figures (../../figures/): t11h1_surface_operator_check.png, t11h1_space_convergence.png,
t11h1_time_convergence.png, t11h1_core_iterations.png, t11h1_memory.png, from the t11h1_*.csv tables."""
import csv, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = os.path.dirname(os.path.abspath(__file__)) + "/../.."
C = {"y": "#2a6fdb", "p": "#d9822b", "u": "#3a9e5f", "uH1s": "#7a3fb0", "u_t": "#b03f6a"}
rd = lambda f: list(csv.DictReader(open(f"{R}/{f}")))
# ---- surface operator check: the worst positive and the weakest negative per check family
ks = rd("t11h1_surface_stiffness.csv")
fam = {}
for r in ks:
    try: v = abs(float(r["value"]))
    except ValueError: continue
    if "lambda" in r["check"] or "v^T" in r["check"] or "count" in r["check"] or "support" in r["check"] or "nnz" in r["check"] or "null space" in r["check"]: continue   # counts, not errors
    key = r["check"].split(":")[0][:58]
    neg = r["check"].startswith("NEG")
    cur = fam.get(key, (neg, None))
    fam[key] = (neg, v if cur[1] is None else (min(cur[1], v) if neg else max(cur[1], v)))
fig, ax = plt.subplots(figsize=(10, 6))
items = sorted(fam.items(), key=lambda kv: (kv[1][0], kv[0]))
ys = range(len(items))
ax.barh(list(ys), [max(v, 1e-18) for _, (neg, v) in items], color=["#d9822b" if neg else "#2a6fdb" for _, (neg, v) in items])
ax.set_xscale("log"); ax.set_yticks(list(ys)); ax.set_yticklabels([k for k, _ in items], fontsize=7)
ax.axvline(1e-14, color="#2a6fdb", ls=":", lw=1); ax.axvline(1e-6, color="#d9822b", ls=":", lw=1)
ax.set_xlabel("blue: worst positive (small); orange: weakest negative control (large); exact 0 drawn at 1e-18", fontsize=8)
ax.set_title("T11H1 K_Gamma / R_Gamma / R_H1t battery (393 checks, 6 faces, anisotropic and non-unit faces)", fontsize=9)
fig.tight_layout(); fig.savefig(R + "/figures/t11h1_surface_operator_check.png", dpi=150)
# ---- space
sp = rd("t11h1_space_convergence.csv")
lads = [("sN", "H1-space ell=0.3, wind (0,0,-0.75), CN N=256"), ("sH", "H1-space ell=0.3, heat, CN N=256"),
        ("tH", "H1-time gamma=1e-2, heat, CN N=64"), ("kS", "combined, wind (0,0,-0.75), CN N=64")]
fig, axs = plt.subplots(1, 4, figsize=(18, 4.3))
for ax, (lad, title) in zip(axs, lads):
    rs = sorted((r for r in sp if r.get("ladder") == lad), key=lambda r: int(r["n"]))
    if not rs: continue
    h = [1 / int(r["n"]) for r in rs]
    for k, col, lab in (("y_L2L2", "y", "y L2(Ω)"), ("p_L2L2", "p", "p L2(Ω)"), ("u_L2L2", "u", "u L2(Γ_c)"), ("u_H1s_L2L2", "uH1s", "u H1-seminorm(Γ_c)")):
        ax.loglog(h, [float(r[k]) for r in rs], "o-", color=C[col], lw=2, ms=5, label=lab)
    e0 = float(rs[-1]["y_L2L2"]); ax.loglog(h, [e0 * (x / h[-1]) ** 3 for x in h], ":", color="#777", label="h³")
    e0 = float(rs[-1]["u_H1s_L2L2"]); ax.loglog(h, [e0 * (x / h[-1]) ** 2 for x in h], "--", color="#bbb", label="h²")
    ax.set_title(title, fontsize=8); ax.set_xlabel("h = 1/n"); ax.grid(True, which="both", alpha=0.25)
axs[0].set_ylabel("space-time error vs dirichlet-h1-3d"); axs[0].legend(frameon=False, fontsize=7)
fig.suptitle("T11H1 3D Dirichlet H1 control: spatial convergence (u nodal in time)"); fig.tight_layout()
fig.savefig(R + "/figures/t11h1_space_convergence.png", dpi=150)
# ---- time
tm = rd("t11h1_time_convergence.csv")
studies = sorted({r["study"] for r in tm})
fig, axs = plt.subplots(1, len(studies), figsize=(4.6 * len(studies), 4.3))
for ax, st in zip(axs, studies):
    rs = sorted((r for r in tm if r["study"] == st), key=lambda r: int(r["nt"]))
    dt = [1 / int(r["nt"]) for r in rs]; q = 1 if " BE " in st else 2
    for k, lab in (("y", "y"), ("y_late", "y, t ≥ T/4"), ("p", "p"), ("u", "u (Γ_c)"), ("u_t", "u_t (Γ_c)")):
        ax.loglog(dt, [float(r[k]) for r in rs], "o-" if k != "y_late" else "s--", color=C.get(k, C["y"]), lw=1.8, ms=4, label=lab)
    e0 = float(rs[-1]["u"]); ax.loglog(dt, [e0 * (x / dt[-1]) ** q for x in dt], ":", color="#777", label=f"Δt^{q}")
    ax.set_title(st, fontsize=7); ax.set_xlabel("Δt"); ax.grid(True, which="both", alpha=0.25); ax.legend(frameon=False, fontsize=6)
axs[0].set_ylabel("L2 difference to reference (n = 6)")
fig.suptitle("T11H1 temporal self-convergence (H1-time and combined)"); fig.tight_layout()
fig.savefig(R + "/figures/t11h1_time_convergence.png", dpi=150)
# ---- core iterations
cs = rd("t11h1_core_scaling.csv"); el = rd("t11h1_ell_scaling.csv"); ga = rd("t11h1_gamma_scaling.csv")
fig, axs = plt.subplots(1, 5, figsize=(20, 4.2))
cols = {"H1s": "#2a6fdb", "H1t": "#3a9e5f", "comb": "#d9822b"}
for reg, col in cols.items():
    for w, ls in (("heat", "--"), ("wA", "-")):
        rs = sorted((r for r in cs if r["study"] == "mesh" and r["reg"] == reg and r["wind"] == w), key=lambda r: int(r["n"]))
        axs[0].plot([int(r["n"]) for r in rs], [int(r["core_its_max"]) for r in rs], "o" + ls, color=col, label=f"{reg} {w}")
        rs = sorted((r for r in cs if r["study"] == "time" and r["reg"] == reg and r["wind"] == w), key=lambda r: int(r["nt"]))
        axs[1].semilogx([int(r["nt"]) for r in rs], [int(r["core_its_max"]) for r in rs], "o" + ls, color=col, label=f"{reg} {w}")
        rs = sorted((r for r in cs if r["study"] == "alpha" and r["reg"] == reg and r["wind"] == w and r["scheme"] == "CN"), key=lambda r: float(r["alpha_u"]))
        axs[2].loglog([float(r["alpha_u"]) for r in rs], [int(r["core_its_max"]) for r in rs], "o" + ls, color=col, label=f"{reg} {w} CN")
for w, ls in (("heat", "--"), ("wA", "-")):
    rs = [r for r in el if r["study"] == "ell" and r["wind"] == w]
    axs[3].semilogx([max(float(r["ell"]), 0.03) for r in rs], [int(r["core_its_max"]) for r in rs], "o" + ls, color="#2a6fdb", label=f"{w} its")
    rs = [r for r in ga if r["wind"] == w and r["scheme"] == "CN"]
    axs[4].semilogx([max(float(r["gamma_u"]), 3e-5) for r in rs], [int(r["core_its_max"]) for r in rs], "o" + ls, color="#3a9e5f", label=f"{w} its")
axs[0].set(xlabel="n (Nt = 32)", ylabel="core CG iterations", ylim=(0, 10), title="mesh")
axs[1].set(xlabel="Nt (n = 4)", ylim=(0, 10), title="time steps")
axs[2].set(xlabel="α_u (n = 4, Nt = 32)", title="regularisation weight"); axs[2].invert_xaxis()
axs[3].set(xlabel="ℓ (0 plotted at 0.03)", ylim=(0, 10), title="H1-space length")
axs[4].set(xlabel="γ_u (0 plotted at 3e-5)", ylim=(0, 10), title="H1-time weight")
for ax in axs: ax.grid(True, which="both", alpha=0.25); ax.legend(frameon=False, fontsize=6)
fig.suptitle("T11H1 matrix-free core K = R_H1t + βH: CG iterations with the sparse R_H1t Cholesky preconditioner"); fig.tight_layout()
fig.savefig(R + "/figures/t11h1_core_iterations.png", dpi=150)
# ---- memory
mm = rd("t11h1_memory.csv")
fig, ax = plt.subplots(figsize=(7, 4.6))
for reg, col in (("L2", "#555"), ("H1s", "#2a6fdb"), ("H1t", "#3a9e5f"), ("comb", "#d9822b")):
    for core, mk in (("iterative", "o"), ("dense", "^")):
        rs = sorted((r for r in mm if r["reg"] == reg and r["core"] == core), key=lambda r: int(r["m"]))
        ax.loglog([int(r["m"]) for r in rs], [float(r["rss_MB"]) for r in rs], mk, color=col, mfc="none" if core == "dense" else col,
                  label=f"{reg} {core}")
ax.set(xlabel="m = Nt |C|", ylabel="peak RSS [MB]"); ax.grid(True, which="both", alpha=0.25); ax.legend(frameon=False, fontsize=7, ncol=2)
ax.set_title("T11H1 memory: iterative (filled) is linear in Nt; dense gold mode (open) carries O(m²)", fontsize=9)
fig.tight_layout(); fig.savefig(R + "/figures/t11h1_memory.png", dpi=150)
