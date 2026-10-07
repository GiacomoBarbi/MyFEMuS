#!/usr/bin/env python3
"""alpha_u sweeps of Mathematics 2025 Examples 3 (non-colocated) and 4
(colocated) on their converged discretisations -> CSV + two-panel figure
(tail error vs alpha_u; control effort vs alpha_u).  Separate panels, one
measure each (no dual axis).  Palette: validated categorical slots 1-2."""
import csv, glob, os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
E4 = os.path.dirname(HERE)
RES = os.path.join(os.path.dirname(E4), "results")
# converged runs: Ex.4 CN nx=640 N=1200 (nx=320 N=2400 checks below 0.5 %);
# Ex.3 BE nx=160 N=800 (identical to nx=40)
PICK = {"4": "sweep_cn_a{a}_nx640_nt1200", "3": "ex3sweep_be_a{a}_nx160_nt800"}
ALPHAS = ["1e-3", "1e-5", "1e-7", "1e-9", "1e-11"]
rows = []
for ex, pat in PICK.items():
    for a in ALPHAS:
        s = open(os.path.join(E4, "logs", pat.format(a=a) + ".log")).read()
        r = dict(re.findall(r"(\w+)=(\S+)", re.search(rf"^EX21_EX{ex}_RESULT (.*)$", s, re.M).group(1)))
        tail = dict(re.findall(r"(\w+)=(\S+)", re.search(r"^EX21_EX4_TAIL (.*)$", s, re.M).group(1))) if ex == "4" else {}
        rows.append(dict(example=ex, alpha_u=float(a), run=pat.format(a=a),
                         tail_window="[5,15]" if ex == "4" else "[10,20]",
                         tail_sup=float(tail.get("sup_e_5_15", r.get("sup_e_10_20"))),
                         tail_rms=float(tail.get("rms_e_5_15", r.get("rms_e_10_20"))),
                         J_track=float(r["J_track"]), J_reg=float(r["J_reg"]), J_total=float(r["J_total"]),
                         u_L2=float(r["u_L2"]), true_residual=float(r["true_residual"]),
                         outer_its=int(r["outer_its"]), sweep_failures=int(r["sweep_failures"]),
                         wall_s=float(re.search(r"WALL (\S+)", s).group(1))))
with open(os.path.join(RES, "math2025_ex3_ex4_alpha_sweep.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dc"
COL = {"3": "#2a78d6", "4": "#eb6834"}
NAME = {"3": "Ex. 3 non-colocated", "4": "Ex. 4 colocated"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), facecolor="#fcfcfb")
for ax, key, ylabel, title in (
        (axes[0], "tail_sup", "tail sup of ‖e(t)‖ on Ω_c", "Attainable tail tracking error"),
        (axes[1], "u_L2", "‖u‖ in L²(0,T; L²(Ω_b))", "Control effort")):
    ax.set_facecolor("#fcfcfb")
    for ex in ("3", "4"):
        d = [r for r in rows if r["example"] == ex]
        x = [r["alpha_u"] for r in d]; y = [r[key] for r in d]
        ax.plot(x, y, "-o", color=COL[ex], lw=2, ms=6, mec="#fcfcfb", mew=1.5, label=NAME[ex])
        ax.annotate(NAME[ex], (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis()
    ax.set_xlabel("α_u (ex21 regularisation; decreasing →)")
    ax.set_ylabel(ylabel); ax.set_title(title, loc="left", fontsize=10)
    ax.grid(True, which="major", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.margins(x=0.25)
axes[0].legend(frameon=False, fontsize=8, loc="lower left")
fig.text(0.01, 0.005, "Converged discretisations: Ex. 4 CN nx=640 N=1200 (T=15, tail [5,15]); "
         "Ex. 3 BE nx=160 N=800 (T=20, tail [10,20]). Same plant family a=v=2.",
         fontsize=7, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 1, 1))
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(E4, f"ex3_ex4_alpha_sweep.{ext}"), dpi=160)
for r in rows:
    print(r["example"], f"{r['alpha_u']:.0e}", f"{r['tail_sup']:.5e}", f"{r['tail_rms']:.5e}",
          f"{r['u_L2']:.4f}", f"{r['J_track']:.4e}", f"{r['J_reg']:.4e}", f"{r['J_total']:.4e}",
          r["outer_its"], f"{r['true_residual']:.1e}", r["wall_s"])
