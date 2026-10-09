#!/usr/bin/env python3
"""T12 figures (../../figures/): t12_tri_space_convergence.png, t12_tet_space_convergence.png,
t12_error_vs_dofs_2d.png, t12_error_vs_dofs_3d.png, t12_memory_vs_unknowns.png."""
import csv, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = os.path.dirname(os.path.abspath(__file__)) + "/../.."
rd = lambda f: list(csv.DictReader(open(f"{R}/{f}")))
C = {"quad": "#2a6fdb", "tri": "#d9822b", "hex": "#2a6fdb", "tet": "#d9822b"}
LS = {"y": "o-", "p": "s--", "u": "^:"}
def raw(rows, m): return sorted((r for r in rows if r["mesh"] == m and "corrected" not in r["ladder"]), key=lambda r: int(r["n"]))
for f, pair, title, out in (("t12_tri_space.csv", ("quad", "tri"), "2D conv-sin3: QUAD9 vs P2 TRI (CN, N = 256)", "t12_tri_space_convergence.png"),
                            ("t12_tet_space.csv", ("hex", "tet"), "3D conv-sin3-3d, w = (1,0.5,-0.75): HEX27 vs P2 TET (CN, N = 128)", "t12_tet_space_convergence.png")):
    sp = rd(f)
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in pair:
        rs = raw(sp, m); h = [1 / int(r["n"]) for r in rs]
        for k in ("y", "p"):
            ax.loglog(h, [float(r[k + "_L2L2"]) for r in rs], LS[k], color=C[m], lw=1.8, ms=5, label=f"{m}: {k} (u = p/alpha)")
    rs = raw(sp, pair[0]); h = [1 / int(r["n"]) for r in rs]; e0 = float(rs[-1]["y_L2L2"])
    ax.loglog(h, [e0 * (x / h[-1]) ** 3 for x in h], "k:", lw=1, label="h^3")
    ax.set(xlabel="h = 1/n", ylabel="L2(0,T;L2) error vs exact", title=title); ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(f"{R}/figures/{out}", dpi=150)
for f, pair, out, title in (("t12_tri_space.csv", ("quad", "tri"), "t12_error_vs_dofs_2d.png", "2D: error vs spatial DOFs (left) and elements (right)"),
                            ("t12_tet_space.csv", ("hex", "tet"), "t12_error_vs_dofs_3d.png", "3D: error vs spatial DOFs (left) and elements (right)")):
    sp = rd(f)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    for m in pair:
        rs = raw(sp, m)
        for ax, key in zip(axs, ("dofs", "elements")):
            x = [int(r[key]) for r in rs]
            for k in ("y", "p"):
                ax.loglog(x, [float(r[k + "_L2L2"]) for r in rs], LS[k], color=C[m], lw=1.8, ms=5, label=f"{m}: {k}")
    axs[0].set(xlabel="spatial DOFs N_space (identical lattice for both)", ylabel="L2 error"); axs[1].set(xlabel="elements")
    for ax in axs: ax.grid(True, which="both", alpha=0.25); ax.legend(frameon=False, fontsize=8)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(f"{R}/figures/{out}", dpi=150)
mm = [r for r in rd("t12_memory.csv") if r["study"].startswith("space")]
fig, ax = plt.subplots(figsize=(7, 5))
for m, mk in (("quad", "o"), ("tri", "s"), ("hex", "^"), ("tet", "D")):
    rs = sorted((r for r in mm if r["mesh"] == m), key=lambda r: int(r["spacetime_unknowns"]))
    ax.loglog([int(r["spacetime_unknowns"]) for r in rs], [float(r["maxrss_MB"]) for r in rs], mk + "-", label=m,
              color="#2a6fdb" if m in ("quad", "hex") else "#d9822b", mfc="none" if m in ("quad", "tri") else None)
x = [1e4, 1e7]; ax.loglog(x, [v * 1500 / 2**20 for v in x], "k:", lw=1, label="1.5 kB / unknown")
ax.set(xlabel="space-time unknowns 2 N_t N_space", ylabel="peak RSS [MB]", title="T12 memory (CN ladders, LU spatial factors)")
ax.grid(True, which="both", alpha=0.25); ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(f"{R}/figures/t12_memory_vs_unknowns.png", dpi=150)
