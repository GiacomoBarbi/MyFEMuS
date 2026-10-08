#!/usr/bin/env python3
"""T10 figures (from ../../t10_convergence.csv and ../../t10_multizone.csv):
  ../../figures/t10_natural_space_convergence.png
  ../../figures/t10_natural_time_convergence.png
  ../../figures/t10_multizone_space_convergence.png"""
import csv, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = os.path.dirname(os.path.abspath(__file__)) + "/../.."
os.makedirs(R + "/figures", exist_ok=True)
C = {"y": "#2a6fdb", "p": "#d9822b", "u": "#3a9e5f"}
conv = list(csv.DictReader(open(R + "/t10_convergence.csv")))
mz = list(csv.DictReader(open(R + "/t10_multizone.csv")))

# ---- natural space: exact errors (winds A, heat, B) + fixed-dt self-convergence (A, heat)
fig, axs = plt.subplots(1, 4, figsize=(17, 4.2), sharey=False)
for ax, lad, title in zip(axs[:3], ("natA", "natH", "natB"),
                          ("exact error, wind (1,0.5,−0.75)", "exact error, heat (w off)", "exact error, wind (−2,1,0.5) (k=−8: pre-asymptotic)")):
    rs = sorted((r for r in conv if r.get("ladder") == lad), key=lambda r: int(r["n"]))
    h = [1 / int(r["n"]) for r in rs]
    ax.loglog(h, [float(r["y_L2L2_rel"]) for r in rs], "o-", color=C["y"], lw=2, ms=6, label="y (rel. L²L²)")
    ax.loglog(h, [float(r["p_L2L2_rel"]) for r in rs], "s-", color=C["p"], lw=2, ms=6, label="p = u (rel.; u = −p/α_u)")
    e0 = float(rs[-1]["p_L2L2_rel"]); h0 = h[-1]
    ax.loglog(h, [e0 * (x / h0) ** 3 for x in h], ":", color="#777", label="h³")
    ax.set_title(title, fontsize=9); ax.set_xlabel("h = 1/n"); ax.grid(True, which="both", alpha=0.25)
axs[0].set_ylabel("relative space-time L² error (CN, N = 64)"); axs[0].legend(frameon=False, fontsize=8)
ax = axs[3]
for study, mk, lab in (("natural fixed-dt space self-conv wind A (CN N=16, ref n=16)", "o", "wind A"),
                       ("natural fixed-dt space self-conv heat (CN N=16, ref n=16)", "^", "heat")):
    rs = sorted((r for r in conv if r["study"] == study), key=lambda r: int(r["n"]))
    h = [1 / int(r["n"]) for r in rs]
    for k in ("y", "p"):
        ax.loglog(h, [float(r[k]) for r in rs], mk + "-", color=C[k], lw=1.6, ms=6, label=f"{k}, {lab}")
rs = sorted((r for r in conv if r["study"].startswith("natural fixed-dt space self-conv wind A")), key=lambda r: int(r["n"]))
h = [1 / int(r["n"]) for r in rs]; e0 = float(rs[-1]["y"])
ax.loglog(h, [e0 * (x / h[-1]) ** 3 for x in h], ":", color="#777", label="h³")
ax.set_title("fixed-Δt self-convergence (vs n = 16)", fontsize=9); ax.set_xlabel("h = 1/n"); ax.grid(True, which="both", alpha=0.25)
ax.set_ylabel("space-time L² difference"); ax.legend(frameon=False, fontsize=7)
fig.suptitle("T10 natural-BC exact OC (natural-conv-3d): Q2 HEX27 spatial convergence"); fig.tight_layout()
fig.savefig(R + "/figures/t10_natural_space_convergence.png", dpi=150)

# ---- natural + multi-zone time (BE, CN)
fig, axs = plt.subplots(1, 4, figsize=(17, 4.2))
for ax, study, q, title in zip(axs, ("natural time BE (n=6, ref N=512)", "natural time CN (n=6, ref N=512)",
                                     "multizone time BE (n=4, ref N=512)", "multizone time CN (n=4, ref N=512)"),
                               (1, 2, 1, 2), ("natural, BE", "natural, CN", "multi-zone, BE", "multi-zone, CN")):
    rs = sorted((r for r in (conv + mz) if r["study"] == study), key=lambda r: int(r["nt"]))
    if not rs: continue
    dt = [1 / int(r["nt"]) for r in rs]
    for k in ("y", "p", "u"):
        ax.loglog(dt, [float(r[k]) for r in rs], "o-", color=C[k], lw=2, ms=5, label=k)
    e0 = float(rs[-1]["p"]); ax.loglog(dt, [e0 * 0.6 * (x / dt[-1]) ** q for x in dt], ":", color="#777", label=f"Δt^{q}")
    ax.set_title(title, fontsize=10); ax.set_xlabel("Δt (T = 1)"); ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
axs[0].set_ylabel("space-time L² difference to reference (same mesh)")
fig.suptitle("T10 temporal self-convergence (wind (1,0.5,−0.75), natural BC)"); fig.tight_layout()
fig.savefig(R + "/figures/t10_natural_time_convergence.png", dpi=150)

# ---- multi-zone space: total / band / interior
rs = sorted((r for r in mz if r["study"].startswith("multizone space")), key=lambda r: int(r["n"]))
fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
h = [1 / int(r["n"]) for r in rs]
for ax, k in zip(axs, ("y", "p", "u")):
    for suf, ls, lab in (("", "o-", "total"), ("_band", "s--", "band (≤ 1/8 from a region face)"), ("_interior", "^:", "interior")):
        ax.loglog(h, [float(r[k + suf]) for r in rs], ls, color=C[k], lw=2, ms=6, label=lab)
    e0 = float(rs[-1][k + "_interior"])
    ax.loglog(h, [e0 * (x / h[-1]) ** 3 for x in h], "-", color="#bbb", lw=1, label="h³")
    q = 1 if k == "u" else 2   # u: the control jumps across the box faces (interface-limited)
    e0 = float(rs[-1][k]); ax.loglog(h, [e0 * (x / h[-1]) ** q for x in h], "-.", color="#999", lw=1, label=f"h^{q}")
    ax.set_title(k, fontsize=10); ax.set_xlabel("h = 1/n"); ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=7)
axs[0].set_ylabel("space-time L² difference to n = 16 (CN, N = 8)")
fig.suptitle("T10 multi-zone (2 control + 2 tracking octants, 2 sources): fixed-Δt spatial self-convergence")
fig.tight_layout(); fig.savefig(R + "/figures/t10_multizone_space_convergence.png", dpi=150)
