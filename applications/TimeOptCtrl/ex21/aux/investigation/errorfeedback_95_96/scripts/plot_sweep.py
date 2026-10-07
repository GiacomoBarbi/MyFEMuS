#!/usr/bin/env python3
"""ErrorFeedback 9.5 / 9.6: AAO tail sup on [5,20] vs alpha_u, per mesh level
(finest highlighted, coarser muted, all direct-labelled), with the published
cascade E5 (Tables 9/10) as a reference line.  One measure per panel; log-log.
Reads ../../results/errorfeedback_{95,96}_aao_runs.csv (summarize.py)."""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "results"); OUT = os.path.join(HERE, "..")
INK, MUTED, GRID, SURF = "#1f1f1e", "#6b6a63", "#e4e3dc", "#fcfcfb"
S1, S2 = "#2a78d6", "#eb6834"
E5 = {"95": 5.15e-4, "96": 5.220e-2}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), facecolor=SURF)
for ax, ex in zip(axes, ("95", "96")):
    ax.set_facecolor(SURF); ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    rows = [r for r in csv.DictReader(open(os.path.join(RES, f"errorfeedback_{ex}_aao_runs.csv")))
            if r["run"].startswith("sweep_") and r["scheme"] == "CN"]
    meshes = sorted({int(r["nx"]) for r in rows})
    finest = max(n for n in meshes if sum(int(r["nx"]) == n for r in rows) >= 2)
    for n in meshes:
        d = {}
        for r in rows:
            if int(r["nx"]) == n:      # per alpha keep the largest nt (time-converged)
                a = float(r["alpha_u"])
                if a not in d or int(r["steps"]) > d[a][1]:
                    d[a] = (float(r["sup_5_20"]), int(r["steps"]))
        xs = sorted(d); ys = [d[a][0] for a in xs]
        if len(xs) < 1: continue
        hl = n == 32 and ex == "96" or n == finest and ex == "95"
        ax.plot(xs, ys, marker="o", color=S1 if hl else MUTED, lw=2 if hl else 1,
                ms=6 if hl else 4, mec=SURF, mew=1.2, ls="-" if hl else (0, (3, 2)))
        ax.annotate(f"nx={n}", (xs[0], ys[0]), xytext=(5, 0), textcoords="offset points", fontsize=7,
                    color=INK, va="center")
    ax.axhline(E5[ex], color=S2, lw=1.5, ls=(0, (6, 3)))
    ax.text(0.02, E5[ex] * 1.08, f"published REFC E₅ = {E5[ex]:.3g}", fontsize=8, color=INK, ha="left",
            va="bottom", transform=matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData))
    if ex == "96":
        ax.set_ylim(0.04, 0.45)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis()
    ax.set_xlabel("α_u (ex21 AAO regularisation; decreasing →)")
    ax.set_ylabel("tail sup ‖y − r‖ in L²(Ω_C), t ∈ [5, 20]")
    ax.set_title({"95": "EF 9.5 (colocated)", "96": "EF 9.6 (non-colocated)"}[ex], loc="left", fontsize=10)
fig.text(0.01, 0.005, "CN, z₀ = sin x sin πy (tail IC-insensitive). Highlighted: the mesh carrying the conclusions; "
         "muted: other levels.", fontsize=7, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 1, 1))
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"ef_alpha_sweep.{ext}"), dpi=160)
print("ok")
