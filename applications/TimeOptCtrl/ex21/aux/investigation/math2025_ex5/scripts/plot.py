#!/usr/bin/env python3
"""Example 5 figures.
 ex5_fig46_comparison.{png,pdf}: top -- source E_oc (Fig. 46, orange), source
   cascade e3 (Fig. 46, blue) and the ex21 AAO optimum ((W,H) = (pi,1), z0 = 0,
   eps = 1e-7); bottom -- pointwise differences ex21 - source.
 ex5_fig45_free_response.{png,pdf}: Fig. 45 plant error (IC xy) for t <= 0.5
   against the ex21 free response ||chi_c S(t) xy|| for the four geometries
   (log scale; (pi,1) highlighted, alternatives muted, all direct-labelled).
Palette: validated categorical slots 1-3 (light surface #fcfcfb)."""
import glob, math, os, re, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compare as C

LOG = os.path.join(HERE, "..", "logs"); OUT = os.path.join(HERE, "..")
INK, MUTED, GRID, SURF = "#1f1f1e", "#6b6a63", "#e4e3dc", "#fcfcfb"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})


def style(ax):
    ax.set_facecolor(SURF); ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)


cur = C.source()
primary = os.path.join(LOG, sys.argv[1] if len(sys.argv) > 1 else "ref_cn_nx48_nz16_nt400.log")
_, _, ser, _ = C.parse(primary)
oc, e3 = cur["Eoc_c"], cur["e3_cascade_c"]
fig, (a, b) = plt.subplots(2, 1, figsize=(8, 5.6), sharex=True, facecolor=SURF,
                           gridspec_kw=dict(height_ratios=[2.2, 1]))
style(a); style(b)
a.plot([t for t, _ in oc], [v for _, v in oc], color=S2, lw=2, label="source E_oc (Fig. 46)")
a.plot([t for t, _ in e3], [v for _, v in e3], color=S1, lw=2, label="source cascade e₃ (Fig. 46)")
a.plot([t for t, _ in ser], [v for _, v in ser], color=S3, lw=2, ls=(0, (4, 2)), label="ex21 AAO, (π,1), z₀=0")
for (t, v), txt, col in (((2.25, 0.0400), "ex21", S3), ((0.1, 0.0125), "source E_oc / e₃", MUTED)):
    a.annotate(txt, (t, v), xytext=(8, 0), textcoords="offset points", fontsize=8, color=INK, va="center")
a.set_ylabel("‖e(t)‖ in L²(Ω_c)"); a.set_title("Example 5, ε = 10⁻⁷: tracking error", loc="left", fontsize=10)
a.legend(frameon=False, fontsize=8, loc="upper right")
a.axvspan(0, 3, color=GRID, alpha=0.5, lw=0); a.text(0.1, 0.047, "switch-on, F(t) < 1", fontsize=7, color=MUTED)
b.plot([t for t, _ in oc], [C.interp(ser, t) - v for t, v in oc], color=S2, lw=2, label="ex21 − source E_oc")
b.plot([t for t, _ in e3], [C.interp(ser, t) - v for t, v in e3], color=S1, lw=2, label="ex21 − source e₃")
b.axhline(0, color=MUTED, lw=0.8)
b.set_ylabel("difference"); b.set_xlabel("t"); b.legend(frameon=False, fontsize=8, loc="upper right")
b.axvspan(0, 3, color=GRID, alpha=0.5, lw=0)
fig.text(0.01, 0.005, f"ex21 run: {os.path.basename(primary)[:-4]}. Source curves: vector extraction of p. 31.",
         fontsize=7, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 1))
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"ex5_fig46_comparison.{ext}"), dpi=160)

# ---- Fig. 45 free response
import csv
rows = [r for r in csv.DictReader(open(os.path.join(C.RES, "math2025_ex5_fig45_raw.csv"))) if r["curve"] == "e_plant_c"]
src = [(round(float(r["t"]) * 20) / 20, float(r["value"])) for r in rows if float(r["t"]) <= 0.51]
fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=SURF); style(ax)
labels = {"3.142_H1": "(π, 1)", "1.000_H1": "(1, 1)", "3.142_H2": "(π, 2)", "6.283_H1": "(2π, 1)"}
for key, lab in labels.items():
    fs = sorted(glob.glob(os.path.join(LOG, f"free_W{key}_be_*_T1_nt400.log")))
    if not fs: continue
    s = open(fs[-1]).read()
    fr = [(float(t), float(v)) for t, v in re.findall(r"^EX21_EX5_FREE t=(\S+) free_c=(\S+)$", s, re.M) if float(t) <= 0.52]
    hl = key == "3.142_H1"
    ax.plot([t for t, _ in fr], [v for _, v in fr], color=S1 if hl else MUTED, lw=2 if hl else 1.2,
            ls="-" if hl else (0, (3, 2)))
    ax.annotate(f"ex21 free response {lab}", (fr[-1][0], fr[-1][1]), xytext=(4, 0), textcoords="offset points",
                fontsize=7.5, color=INK, va="center")
ax.plot([t for t, _ in src], [v for _, v in src], "o", color=S2, ms=8, mec=SURF, mew=1.5,
        label="source plant error, IC xy (Fig. 45)")
ax.set_yscale("log"); ax.set_xlim(0.05, 0.68); ax.set_xlabel("t"); ax.set_ylabel("‖·‖ in L²(Ω_c)")
ax.set_title("Example 5: early plant error vs free response of xy, by geometry (W, H)", loc="left", fontsize=10)
ax.legend(frameon=False, fontsize=8, loc="lower left")
fig.text(0.01, 0.005, "ex21: backward Euler, dt = 0.0025, converged in space. For t >= 0.4 the source curve also "
         "contains the cascade error e3, which the free response does not.", fontsize=7, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 1))
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"ex5_fig45_free_response.{ext}"), dpi=160)
print("ok")
