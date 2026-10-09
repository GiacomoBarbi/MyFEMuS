#!/usr/bin/env python3
"""T15 figures -> ../../figures/: t15_lps_activity.png, t15_space_convergence.png, t15_time_convergence.png,
t15_solver_iterations.png, t15_memory.png (from the T15 CSVs)."""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = os.path.dirname(os.path.abspath(__file__)) + "/../.."
F = D + "/figures"; os.makedirs(F, exist_ok=True)
INK, MUTED, GRID = "#1f2328", "#6e7781", "#d0d7de"
# fixed method order: Galerkin, LPS-P0 (production), LPS-P1 (diagnostic)
MCOL = {"Galerkin": "#6e7781", "LPS-P0": "#2f6fdf", "LPS-P1": "#d4572a", "LPS-P1 (diagnostic)": "#d4572a"}
MARK = {"Galerkin": "s", "LPS-P0": "o", "LPS-P1": "^", "LPS-P1 (diagnostic)": "^"}
rd = lambda f: list(csv.DictReader(open(f"{D}/{f}")))
fl = lambda v: float(v) if v not in ("", None) else np.nan


def style(ax):
    ax.grid(True, which="both", color=GRID, lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def activity():
    sp = [r for r in rd("t15_space_convergence.csv") if r["method"] != "Galerkin"]
    bm = rd("t15_lps_benchmark.csv")
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    ax = axs[0]
    for (study, lab), col in zip((("fixed nu=1e-3, wind A", "nu = 1e-3, wind A"), ("fixed Pe_K=10, wind A", "Pe_K = 10 ladder"),
                                  ("fixed nu=1e-3, harmonic-3d", "nu = 1e-3, harmonic-3d")), ("#2f6fdf", "#1a9e77", "#8a5cd1")):
        rr = sorted({int(r["n"]): r for r in sp if r["study"] == study and r["method"] == "LPS-P0"}.values(), key=lambda r: int(r["n"]))
        n = np.array([int(r["n"]) for r in rr])
        lo = np.array([fl(r["lps_pe_min"]) for r in rr]); hi = np.array([fl(r["lps_pe_max"]) for r in rr])
        ax.fill_between(1 / n, lo, hi, color=col, alpha=0.18, lw=0)
        ax.loglog(1 / n, hi, "o-", color=col, lw=2, ms=6, label=f"{lab}: max Pe_K")
        ax.loglog(1 / n, lo, "o--", color=col, lw=1, ms=4)
    ax.axhline(1.0, color=INK, lw=0.8); ax.text(0.13, 1.15, "Pe_K = 1 (switch)", fontsize=7, color=INK)
    ax.set_xlabel("h = 1/n"); ax.set_ylabel("patch Peclet Pe_K (band: min..max)"); ax.set_title("LPS activity: patch Peclet", fontsize=9)
    ax.legend(fontsize=6, frameon=False); style(ax)
    ax = axs[1]
    for (study, lab), col in zip((("fixed nu=1e-3, wind A", "nu = 1e-3, wind A"), ("fixed Pe_K=10, wind A", "Pe_K = 10 ladder"),
                                  ("fixed nu=1e-3, harmonic-3d", "nu = 1e-3, harmonic-3d")), ("#2f6fdf", "#1a9e77", "#8a5cd1")):
        rr = sorted({int(r["n"]): r for r in sp if r["study"] == study and r["method"] == "LPS-P0"}.values(), key=lambda r: int(r["n"]))
        n = np.array([int(r["n"]) for r in rr])
        ax.loglog(1 / n, [fl(r["lps_delta_max"]) for r in rr], "o-", color=col, lw=2, ms=6, label=f"{lab}: max delta_K")
        ax.loglog(1 / n, [fl(r["lps_delta_min"]) for r in rr], "o--", color=col, lw=1, ms=4)
    h = np.array([1 / 2, 1 / 8]); ax.loglog(h, 0.5 * h, ":", color=MUTED, lw=1, label="~ h")
    ax.set_xlabel("h = 1/n"); ax.set_ylabel("delta_K = h_K/(2|w_K|)(1 - 1/Pe_K)"); ax.set_title("LPS activity: delta_K (all patches active)", fontsize=9)
    ax.legend(fontsize=6, frameon=False); style(ax)
    ax = axs[2]
    for m in ("Galerkin", "LPS-P0", "LPS-P1"):
        for w, ls in (("constant A", "-"), ("harmonic-3d", "--")):
            rr = sorted([r for r in bm if r["method"] == m and r["wind"] == w], key=lambda r: int(r["n"]))
            if not rr: continue
            n = np.array([int(r["n"]) for r in rr]); u = np.array([fl(r["undershoot_rel"]) for r in rr])
            ax.semilogy(n, np.maximum(u, 1e-6), ls, marker=MARK[m], color=MCOL[m], lw=2, ms=6, label=f"{m}, {w}")
    ax.axhline(1e-6, color=GRID, lw=0.8); ax.text(4, 1.4e-6, "no undershoot (plotted at 1e-6)", fontsize=7, color=MUTED)
    ax.set_xlabel("n (elements per axis)"); ax.set_ylabel("undershoot  max(0, -min y) / max y")
    ax.set_title("high-Pe layer benchmark (nu = 1e-3, f = 1)", fontsize=9); ax.legend(fontsize=6, frameon=False); style(ax)
    fig.suptitle("T15 3D LPS (HEX27 Q2): activity and effect", fontsize=10)
    fig.tight_layout(); fig.savefig(F + "/t15_lps_activity.png", dpi=150); plt.close(fig)


def space():
    rows = rd("t15_space_convergence.csv")
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for ax, study, ttl in zip(axs, ("fixed Pe_K=10, wind A", "fixed nu=1e-3, wind A", "fixed nu=1e-3, harmonic-3d"),
                              ("fixed patch Peclet Pe_K = 10 (nu ~ h)", "fixed nu = 1e-3, wind A", "fixed nu = 1e-3, harmonic-3d")):
        for m in ("Galerkin", "LPS-P0", "LPS-P1 (diagnostic)"):
            rr = sorted([r for r in rows if r["study"] == study and r["method"] == m], key=lambda r: int(r["n"]))
            if not rr: continue
            n = np.array([int(r["n"]) for r in rr]); e = np.array([fl(r["y"]) for r in rr])
            ax.loglog(1 / n, e, "-", marker=MARK[m], color=MCOL[m], lw=2, ms=6, label=m)
        h = np.array([1 / 3, 1 / 8])
        ax.loglog(h, 2.5e-2 * (h / h[0]) ** 2, ":", color=MUTED, lw=1, label="h^2")
        ax.loglog(h, 6e-3 * (h / h[0]) ** 3, "--", color=MUTED, lw=1, label="h^3")
        ax.set_xlabel("h = 1/n"); ax.set_ylabel("state error ||y - y_ex|| L2(L2)"); ax.set_title(ttl, fontsize=9)
        ax.legend(fontsize=7, frameon=False); style(ax)
    fig.suptitle("T15 spatial convergence, exact conv-sin3-3d (CN N = 64): LPS-P0 loses one Q2 order at fixed Pe_K; P1 keeps it", fontsize=10)
    fig.tight_layout(); fig.savefig(F + "/t15_space_convergence.png", dpi=150); plt.close(fig)


def time():
    rows = rd("t15_time_convergence.csv")
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    series = (("tA0", "distributed, wind A, LPS-P0", "#2f6fdf", "o"), ("tAg", "distributed, wind A, Galerkin", "#6e7781", "s"),
              ("tH0", "distributed, harmonic-3d, LPS-P0", "#8a5cd1", "D"), ("tD0", "Dirichlet zmin, wind A, LPS-P0", "#1a9e77", "^"),
              ("tDg", "Dirichlet zmin, wind A, Galerkin", "#b08800", "v"))
    for ax, s in zip(axs, ("BE", "CN")):
        for tag, lab, col, mk in series:
            rr = [r for r in rows if r["tag"] == tag and r["scheme"] == s]
            if not rr: continue
            nt = np.array([fl(r["nt"]) for r in rr]); e = np.array([fl(r["p"]) for r in rr])
            ax.loglog(1 / nt, e, "-", marker=mk, color=col, lw=2, ms=5, label=lab)
        dt = np.array([1 / 4, 1 / 128]); q = 1 if s == "BE" else 2
        ax.loglog(dt, 2e-3 * (dt / dt[0]) ** q, "--", color=MUTED, lw=1, label=f"dt^{q}")
        ax.set_xlabel("dt"); ax.set_ylabel("adjoint error ||p - p_ref|| (ref N = 1024)")
        ax.set_title(f"{s}: fixed mesh n = 4, LPS active (Pe_K ~ 23)", fontsize=9); ax.legend(fontsize=6, frameon=False); style(ax)
    fig.tight_layout(); fig.savefig(F + "/t15_time_convergence.png", dpi=150); plt.close(fig)


def solver():
    rows = [r for r in rd("t15_solver.csv") if not r["case"].startswith("r")]
    cases = [c for c in ("dA", "dH", "nA", "mA") if any(r["case"] == c for r in rows)]
    conv = [fl(r["y_its"]) for r in rows if r["converged"] == "yes" and r["y_its"]]
    top = 10 ** np.ceil(np.log10(max(conv)) + 0.5)          # fixed "not converged" band above every converged bar
    fig, axs = plt.subplots(1, len(cases), figsize=(3.6 * len(cases), 4.2), sharey=True)
    axs = np.atleast_1d(axs)
    from matplotlib.patches import Patch
    for ax, c in zip(axs, cases):
        x = 0; ticks, labs = [], []
        for s in ("BE", "CN"):
            for bk in ("lu", "gamg", "asmilu"):
                for k, st in enumerate(("Galerkin", "LPS-P0")):
                    r = next((r for r in rows if r["case"] == c and r["scheme"] == s and r["backend"] == bk and r["stabilisation"] == st), None)
                    if r is None: continue
                    if r["converged"] == "yes":
                        ax.bar(x + 0.42 * k, fl(r["y_its"]), width=0.38, color=MCOL[st])
                    else:
                        ax.bar(x + 0.42 * k, top, width=0.38, color="white", edgecolor=MCOL[st], hatch="///")
                ticks.append(x + 0.21); labs.append(f"{s}\n{bk}"); x += 1.2
        ax.axhline(top / 3, color=GRID, lw=0.8)
        ax.set_yscale("log"); ax.set_ylim(1e2, top * 1.5); ax.set_xticks(ticks); ax.set_xticklabels(labs, fontsize=7)
        ax.set_title({"dA": "distributed, wind A", "dH": "distributed, harmonic-3d", "nA": "natural state, wind A",
                      "mA": "Neumann zmin, wind A"}[c] + "  (nu = 1e-3)", fontsize=8); style(ax)
    axs[0].set_ylabel("total spatial y-split iterations (log)")
    axs[0].text(0.0, top * 0.55, "hatched (full height) = not converged:\nrejected or 600 s cap", fontsize=6, color=INK, va="top")
    handles = [Patch(color=MCOL["Galerkin"], label="Galerkin, converged"), Patch(color=MCOL["LPS-P0"], label="LPS-P0, converged"),
               Patch(facecolor="white", edgecolor=MCOL["Galerkin"], hatch="///", label="Galerkin, NOT converged"),
               Patch(facecolor="white", edgecolor=MCOL["LPS-P0"], hatch="///", label="LPS-P0, NOT converged")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7, frameon=False)
    fig.suptitle("T15 solver audit (n = 8, N = 32, SCHUR_SWEEP): Galerkin vs LPS-P0", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(F + "/t15_solver_iterations.png", dpi=150); plt.close(fig)


def memory():
    rows = rd("t15_memory.csv")
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    ax = axs[0]
    for kind, ls in (("distributed", "-"), ("dirichlet", "--")):
        for m, lab in (("galerkin", "Galerkin"), ("lps-p0", "LPS-P0"), ("lps-p1", "LPS-P1")):
            rr = [r for r in rows if r["kind"] == kind and r["mode"] == m]
            ax.plot([int(r["n"]) for r in rr], [fl(r["rss_vs_galerkin"]) for r in rr], ls, marker=MARK[lab], color=MCOL[lab], lw=2, ms=6, label=f"{lab}, {kind}")
    ax.axhline(1.0, color=INK, lw=0.8); ax.set_ylim(0.95, 1.05)
    ax.set_xlabel("n"); ax.set_ylabel("MaxRSS / Galerkin MaxRSS"); ax.set_title("memory overhead of LPS (N_t = 32)", fontsize=9)
    ax.legend(fontsize=6, frameon=False); style(ax)
    ax = axs[1]
    rr = [r for r in rows if r["kind"] == "distributed" and r["mode"] == "lps-p0"]
    n = [int(r["n"]) for r in rr]
    ax.semilogy(n, [fl(r["nnz_L"]) for r in rr], "s-", color=MCOL["Galerkin"], lw=2, ms=6, label="nnz(L) (unchanged by LPS)")
    ax.semilogy(n, [fl(r["nnz_W"]) for r in rr], "o--", color=MCOL["LPS-P0"], lw=2, ms=6, label="nnz(W)")
    ax.set_xlabel("n"); ax.set_ylabel("nonzeros"); ax.set_title("W shares the Q2 pattern of L (nnz(W) = nnz(L))", fontsize=9)
    ax.legend(fontsize=7, frameon=False); style(ax)
    fig.tight_layout(); fig.savefig(F + "/t15_memory.png", dpi=150); plt.close(fig)


for fn in (activity, space, time, solver, memory):
    try:
        fn()
    except FileNotFoundError as e:
        print("skip", fn.__name__, e)
print("figures written to", F)
