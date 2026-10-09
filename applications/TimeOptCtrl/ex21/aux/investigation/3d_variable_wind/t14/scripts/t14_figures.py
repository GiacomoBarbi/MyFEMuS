#!/usr/bin/env python3
"""T14 figures -> ../../figures/: t14_variable_wind_field.png, t14_space_convergence.png, t14_time_convergence.png,
t14_solver_iterations.png, t14_memory.png (from the T14 CSVs)."""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = os.path.dirname(os.path.abspath(__file__)) + "/../.."
F = D + "/figures"; os.makedirs(F, exist_ok=True)
INK, MUTED, GRID = "#1f2328", "#6e7781", "#d0d7de"
COL = ["#2f6fdf", "#d4572a", "#1a9e77", "#8a5cd1", "#b08800", "#6e7781"]
HARM = np.array([[0, 1, -0.75], [1, 0, 0.5], [-0.75, 0.5, 0]])
FIELDS = {"harmonic-3d (div w = 0)": HARM, "affine-div-3d (div w = 1)": HARM + np.diag([0.5, 0.25, 0.25])}
rd = lambda f: list(csv.DictReader(open(f"{D}/{f}")))


def field():
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    s = np.linspace(0, 1, 11)
    for ax, (lab, H) in zip(axs[:2], FIELDS.items()):
        Xg, Yg = np.meshgrid(s, s)
        for zc, col in ((0.0, COL[0]), (1.0, COL[1])):
            P = np.stack([Xg, Yg, np.full_like(Xg, zc)], -1)
            Wv = P @ H.T
            ax.quiver(Xg, Yg, Wv[..., 0], Wv[..., 1], color=col, alpha=0.8, label=f"(w_x, w_y) on z = {zc:g}")
        ax.set_title(lab, fontsize=9); ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal")
        ax.legend(fontsize=7, loc="upper left", frameon=False)
    ax = axs[2]
    for (lab, H), col in zip(FIELDS.items(), COL):
        wn = -(H[0, 1] * 0.5 + H[0, 2] * s)   # xmin: n = -e_x, w.n = -w_x(0, 1/2, z)
        ax.plot(s, wn, color=col, lw=2, label=f"{lab.split()[0]}: w.n on xmin along z (y = 1/2)")
    ax.axhline(0, color=GRID); ax.set_xlabel("z"); ax.set_ylabel("w.n"); ax.legend(fontsize=7, frameon=False)
    ax.set_title("normal velocity varies along a face (inflow / outflow)", fontsize=9)
    fig.suptitle("T14 3D variable winds: w = grad Phi, Phi = x^T H x / 2 (affine, exact in Q2)", fontsize=10)
    fig.tight_layout(); fig.savefig(F + "/t14_variable_wind_field.png", dpi=150); plt.close(fig)


def space():
    rows = rd("t14_space_convergence.csv")
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    series = [("dA", "distributed, constant wind A"), ("dH", "distributed, harmonic"), ("dF", "distributed, affine-div"),
              ("nF128", "natural-var, affine-div (N=128)"), ("oF", "neumann-var-opp, affine-div"), ("rS", "dirichlet-var swirl")]
    for ax, fld in zip(axs, ("y", "p", "u")):
        for (tag, lab), col in zip(series, COL):
            rr = [r for r in rows if r["tag"] == tag]
            if not rr: continue
            n = np.array([float(r["n"]) for r in rr]); e = np.array([float(r[fld]) for r in rr])
            ax.loglog(1 / n, e, "o-", color=col, lw=2, ms=5, label=lab)
        h = np.array([1 / 2, 1 / 12]); ax.loglog(h, (2e-2 if fld != "p" else 2e-3) * (h / h[0]) ** 3, "--", color=MUTED, lw=1, label="h^3")
        ax.set_xlabel("h = 1/n"); ax.set_title(f"{fld} (exact error, CN)", fontsize=9); ax.grid(True, which="both", color=GRID, lw=0.5)
    axs[2].legend(fontsize=6, frameon=False)
    fig.suptitle("T14 spatial convergence with variable 3D winds (HEX27 Q2)", fontsize=10)
    fig.tight_layout(); fig.savefig(F + "/t14_space_convergence.png", dpi=150); plt.close(fig)


def time():
    rows = rd("t14_time_convergence.csv")
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, s in zip(axs, ("BE", "CN")):
        for (tag, lab), col in zip((("tdH", "distributed harmonic"), ("tdF", "distributed affine-div"), ("tnF", "natural affine-div"),
                                    ("tmH", "neumann harmonic"), ("trS", "dirichlet swirl")), COL):
            rr = [r for r in rows if r["tag"] == tag and r["scheme"] == s]
            if not rr: continue
            nt = np.array([float(r["nt"]) for r in rr]); e = np.array([float(r["p"]) for r in rr])
            ax.loglog(1 / nt, e, "o-", color=col, lw=2, ms=5, label=lab)
        dt = np.array([1 / 4, 1 / 128]); q = 1 if s == "BE" else 2
        ax.loglog(dt, 1e-2 * (dt / dt[0]) ** q, "--", color=MUTED, lw=1, label=f"dt^{q}")
        ax.set_xlabel("dt"); ax.set_title(f"{s}: adjoint p, fixed mesh n = 4, fine-N reference", fontsize=9)
        ax.grid(True, which="both", color=GRID, lw=0.5)
    axs[1].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(F + "/t14_time_convergence.png", dpi=150); plt.close(fig)


def solver():
    rows = [r for r in rd("t14_solver.csv") if not r["case"].startswith("dir")]
    fig, ax = plt.subplots(figsize=(10, 4))
    labs = []
    for i, r in enumerate(rows):
        v = float(r["y_its_max"] or 0) if r["y_its_max"] else 0
        ax.bar(i, v, color={"lu": COL[0], "gamg": COL[1], "asmilu": COL[2]}[r["backend"]])
        labs.append(f"{r['case']}\n{r['scheme']} {r['backend']}")
    ax.set_xticks(range(len(rows))); ax.set_xticklabels(labs, fontsize=6, rotation=90)
    ax.set_ylabel("max spatial its per y-split solve"); ax.grid(True, axis="y", color=GRID, lw=0.5)
    ax.set_title("T14 solver audit (n = 8, N = 32, SCHUR_SWEEP): constant (natA) vs variable winds", fontsize=9)
    fig.tight_layout(); fig.savefig(F + "/t14_solver_iterations.png", dpi=150); plt.close(fig)


def memory():
    rows = rd("t14_memory.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    labs = []
    for i, r in enumerate(rows):
        ax.bar(i, float(r["rss_vs_const"]), color={"const": COL[5], "harmonic-3d": COL[0], "affine-div-3d": COL[1]}[r["wind"]])
        labs.append(f"{r['kind']} n={r['n']} Nt={r['N_t']}\n{r['wind']}")
    ax.axhline(1.0, color=INK, lw=0.8); ax.set_ylim(0.9, 1.1)
    ax.set_xticks(range(len(rows))); ax.set_xticklabels(labs, fontsize=6, rotation=90)
    ax.set_ylabel("MaxRSS / constant-wind MaxRSS"); ax.set_title("T14 memory: the variable wind adds no storage", fontsize=9)
    fig.tight_layout(); fig.savefig(F + "/t14_memory.png", dpi=150); plt.close(fig)


for fn in (field, space, time, solver, memory):
    try:
        fn()
    except FileNotFoundError as e:
        print("skip", fn.__name__, e)
print("figures written to", F)
