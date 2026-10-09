#!/usr/bin/env python3
"""T13 figures -> ../../figures/: t13_multiface_geometry.png, t13_space_convergence.png,
t13_time_convergence.png, t13_memory.png (from the T13 CSVs; the geometry panel from the lattice policy,
which t13_trace.py checked node by node against the dumped C)."""
import csv, itertools, math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = os.path.dirname(os.path.abspath(__file__)) + "/../.."
F = D + "/figures"; os.makedirs(F, exist_ok=True)
FACES = {"zmin": (2, 0), "ymin": (1, 0), "xmax": (0, 1), "ymax": (1, 1), "xmin": (0, 0), "zmax": (2, 1)}
INK, MUTED, GRID = "#1f2328", "#6e7781", "#d0d7de"
C1, C2, C3, C4 = "#2f6fdf", "#d4572a", "#1a9e77", "#8a5cd1"


def lattice_sets(n, faces, kind):
    pts, act, con = [], [], []
    for i, j, k in itertools.product(range(2 * n + 1), repeat=3):
        idx = (i, j, k)
        on = {f: idx[a] == (2 * n if m else 0) for f, (a, m) in FACES.items()}
        if not any(on.values()):
            continue
        p = np.array(idx) / (2 * n)
        if any(on[f] for f in faces) and (kind == "N" or not any(on[f] for f in FACES if f not in faces)):
            act.append(p)
        else:
            con.append(p)
    return np.array(act), np.array(con)


def geometry():
    fig = plt.figure(figsize=(13, 3.6))
    panels = [("Neumann zmin+xmin: trace = union (shared edge once)", ["zmin", "xmin"], "N"),
              ("Dirichlet zmin+xmin: internal edge ACTIVE", ["zmin", "xmin"], "D"),
              ("Dirichlet zmin+ymin+xmin: shared corner ACTIVE", ["zmin", "ymin", "xmin"], "D"),
              ("Dirichlet zmin+zmax (opposite)", ["zmin", "zmax"], "D")]
    for k, (title, faces, kind) in enumerate(panels):
        ax = fig.add_subplot(1, 4, k + 1, projection="3d")
        act, con = lattice_sets(3, faces, kind)
        if len(con):
            ax.scatter(con[:, 0], con[:, 1], con[:, 2], s=4, c=GRID, depthshade=False, label="boundary, constrained / natural")
        ax.scatter(act[:, 0], act[:, 1], act[:, 2], s=9, c=C1 if kind == "N" else C2, depthshade=False,
                   label=f"control DOF ({len(act)})")
        ax.set_title(title, fontsize=8, color=INK)
        ax.set_xlabel("x", fontsize=7); ax.set_ylabel("y", fontsize=7); ax.set_zlabel("z", fontsize=7)
        ax.tick_params(labelsize=6); ax.view_init(22, -60)
        ax.legend(fontsize=6, loc="upper left", frameon=False)
    fig.suptitle("T13 control-DOF topology on face unions (Q2, n = 3): one control on the union; "
                 "Dirichlet: any uncontrolled face wins with zero", fontsize=9, color=INK)
    fig.tight_layout(); fig.savefig(F + "/t13_multiface_geometry.png", dpi=150); plt.close(fig)


def rd(p):
    return list(csv.DictReader(open(p)))


def space():
    rows = rd(D + "/t13_space_convergence.csv")
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    series = [("nopA", "N opp wind A", C1, "o"), ("nadW", "N adj wind E", C3, "s"), ("dopN", "D opp L2 wind N", C2, "^"),
              ("dopC", "D opp H1 comb wind N", C4, "v")]
    for ax, fld in zip(axs, ("y", "p", "u")):
        for tag, lab, col, mk in series:
            rr = [r for r in rows if r["tag"] == tag]
            if not rr: continue
            n = np.array([float(r["n"]) for r in rr]); e = np.array([float(r[fld]) for r in rr])
            ax.loglog(1 / n, e, marker=mk, color=col, lw=2, ms=6, label=lab)
        h = np.array([1 / 2, 1 / 12]); ax.loglog(h, 2e-2 * (h / h[0]) ** 3 * (1 if fld != "p" else 0.05), "--", color=MUTED, lw=1, label="h^3")
        ax.set_xlabel("h = 1/n"); ax.set_title(f"{fld}: {'L2(Omega)' if fld != 'u' else 'L2(Gamma_c) = sum_f'} x L2(0,T)", fontsize=9)
        ax.grid(True, which="both", color=GRID, lw=0.5)
    axs[0].set_ylabel("exact error (CN; N = 64 Neumann, 256 Dirichlet)")
    axs[2].legend(fontsize=7, frameon=False)
    fig.suptitle("T13 spatial convergence, exact multi-face benchmarks (HEX27 Q2)", fontsize=10)
    fig.tight_layout(); fig.savefig(F + "/t13_space_convergence.png", dpi=150); plt.close(fig)


def time():
    rows = rd(D + "/t13_time_convergence.csv")
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, s in zip(axs, ("BE", "CN")):
        for tag, lab, col, mk in (("notA", "N opp", C1, "o"), ("natW", "N adj", C3, "s"), ("dotN", "D opp L2", C2, "^"),
                                  ("dotC", "D opp H1 comb", C4, "v"), ("datH", "D adj H1 comb (self-conv)", "#b08800", "D")):
            rr = [r for r in rows if r["tag"] == tag and r["scheme"] == s]
            if not rr: continue
            nt = np.array([float(r["nt"]) for r in rr]); e = np.array([float(r["u"]) for r in rr])
            ax.loglog(1 / nt, e, marker=mk, color=col, lw=2, ms=6, label=lab)
        dt = np.array([1 / 4, 1 / 128]); q = 1 if s == "BE" else 2
        ax.loglog(dt, 3e-1 * (dt / dt[0]) ** q * (1 if s == "BE" else 0.3), "--", color=MUTED, lw=1, label=f"dt^{q}")
        ax.set_xlabel("dt = T/N"); ax.set_title(f"{s}: u in L2(Gamma_c) x L2(0,T), fixed mesh, fine-N reference", fontsize=9)
        ax.grid(True, which="both", color=GRID, lw=0.5)
    axs[0].set_ylabel("self-convergence error"); axs[1].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(F + "/t13_time_convergence.png", dpi=150); plt.close(fig)


def memory():
    rows = [r for r in rd(D + "/t13_memory.csv") if r["kind"] == "Dirichlet"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for core, col, mk in (("dense", C2, "o"), ("iterative", C1, "s")):
        rr = [r for r in rows if r["core"] == core]
        m = np.array([float(r["m"]) for r in rr]); rss = np.array([float(r["rss_MB"]) for r in rr])
        o = np.argsort(m); ax.loglog(m[o], rss[o], mk, color=col, ms=6, label=f"{core} core (face sets 1..6, Nt 16..256)")
    m = np.logspace(3, 5, 20); ax.loglog(m, 12 * m ** 2 / 2 ** 20, "--", color=MUTED, lw=1, label="12 B m^2 (dense K)")
    ax.set_xlabel("m = Nt |C|  (reduced core size; |C| grows with the union)"); ax.set_ylabel("peak RSS (MB)")
    ax.grid(True, which="both", color=GRID, lw=0.5); ax.legend(fontsize=7, frameon=False)
    ax.set_title("T13 Dirichlet reduced core memory on face unions (n = 4, CN)", fontsize=9)
    fig.tight_layout(); fig.savefig(F + "/t13_memory.png", dpi=150); plt.close(fig)


geometry()
for f in (space, time, memory):
    try:
        f()
    except FileNotFoundError as e:
        print("skip", f.__name__, e)
print("figures written to", F)
