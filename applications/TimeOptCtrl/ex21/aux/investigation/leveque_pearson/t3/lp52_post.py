#!/usr/bin/env python3
"""LP section 5.2 post-processing of the t = 1 snapshots written by
`--exact-oc lp52 --exact-oc-snapshot` (columns x, z, y, p_LP = -p_ex21, u).

  * mesh sensitivity: snapshot differences at the nodes shared by two levels
    (Q2 nx = 32 nodes are a subset of Q2 nx = 64 nodes);
  * oscillation indicators: values outside [0, 1] for y (y is NOT bounded by
    [0, 1] a priori -- the control acts on the whole domain -- so this is
    reported, not gated), and node-to-node sign alternations of the second
    difference of y along x1 on every grid row (an odd-even "wiggle" count:
    a row is flagged when >= 3 consecutive second differences alternate in
    sign with magnitude above 1e-3 of the row range);
  * figures: lp52_y_t1.png, lp52_p_t1.png (surface + contours, beta = 1e-2).
Writes ../lp52_lowpe_results.csv (one row per run, merged with the solver CSVs).
"""
import csv, glob, os, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LP = os.path.dirname(HERE)
R = os.path.join(HERE, "results")


def snap(path):
    d = np.genfromtxt(path, delimiter=",", names=True)
    xs, zs = np.unique(np.round(d["x"], 12)), np.unique(np.round(d["z"], 12))
    ix = {v: i for i, v in enumerate(xs)}
    iz = {v: i for i, v in enumerate(zs)}
    out = {}
    for k in ("y", "p_LP", "u"):
        G = np.full((len(zs), len(xs)), np.nan)
        for x, z, v in zip(np.round(d["x"], 12), np.round(d["z"], 12), d[k]):
            G[iz[z], ix[x]] = v
        out[k] = G
    return xs, zs, out


def wiggles(G, xs, frac=1e-3):
    """rows (fixed x2) with >= 3 consecutive sign alternations of d2y/dx1^2."""
    flagged = 0
    for row in G:
        rng = row.max() - row.min()
        d2 = row[2:] - 2 * row[1:-1] + row[:-2]
        s = np.sign(np.where(np.abs(d2) > frac * max(rng, 1e-300), d2, 0))
        run = best = 0
        for a, b in zip(s, s[1:]):
            run = run + 1 if (a * b < 0) else 0
            best = max(best, run)
        flagged += best >= 3
    return flagged


def main():
    rows = []
    for grp in ("lp52_mesh", "lp52_beta"):
        p = os.path.join(R, grp + ".csv")
        if os.path.exists(p):
            rows += list(csv.DictReader(open(p)))
    snaps = {}
    for path in sorted(glob.glob(os.path.join(R, "lp52_snapshot_*.csv"))):
        m = re.search(r"_b([0-9e.-]+)_nx(\d+)\.csv", path)
        snaps[(m.group(1), int(m.group(2)))] = snap(path)
    print("| beta | nx | y(t=1) range | p_LP(t=1) range | u(t=1) range | y wiggle rows | p wiggle rows |")
    print("|---|---|---|---|---|---|---|")
    extra = {}
    for (b, nx), (xs, zs, F) in sorted(snaps.items(), key=lambda t: (float(t[0][0]), t[0][1])):
        wy, wp = wiggles(F["y"], xs), wiggles(F["p_LP"], xs)
        print(f"| {b} | {nx} | [{np.nanmin(F['y']):.4f}, {np.nanmax(F['y']):.4f}] | "
              f"[{np.nanmin(F['p_LP']):.3e}, {np.nanmax(F['p_LP']):.3e}] | "
              f"[{np.nanmin(F['u']):.3e}, {np.nanmax(F['u']):.3e}] | {wy}/{len(zs)} | {wp}/{len(zs)} |")
        extra[(b, nx)] = {"y_wiggle_rows": wy, "p_wiggle_rows": wp, "grid_rows": len(zs)}
    print("\nmesh sensitivity at shared nodes (max |coarse - fine| / max|fine|):")
    for b in sorted({k[0] for k in snaps}, key=float):
        levels = sorted(nx for (bb, nx) in snaps if bb == b)
        for c, fn in zip(levels, levels[1:]):
            xc, zc, Fc = snaps[(b, c)]
            xf, zf, Ff = snaps[(b, fn)]
            step = (len(xf) - 1) // (len(xc) - 1)
            for k in ("y", "p_LP"):
                sub = Ff[k][::step, ::step]
                d = np.max(np.abs(Fc[k] - sub)) / np.max(np.abs(sub))
                # interior-only (excluding the two boundary-adjacent node layers)
                di = np.max(np.abs(Fc[k][2:-2, 2:-2] - sub[2:-2, 2:-2])) / np.max(np.abs(sub))
                print(f"  beta={b} nx {c} -> {fn}: {k}: {d:.3e} (excluding 2 boundary layers: {di:.3e})")
                extra.setdefault((b, fn), {})[f"{k}_vs_nx{c}"] = f"{d:.3e}"
                extra.setdefault((b, fn), {})[f"{k}_vs_nx{c}_inner"] = f"{di:.3e}"
    # merged CSV
    keep = ["id", "scheme", "nx", "steps", "dt", "beta", "eps", "reason", "outer_its", "true_residual",
            "state_rel", "adjoint_rel", "march_rel", "finite", "max|y-g|", "max|p|", "max|u|",
            "y_min", "y_max", "y_int_min", "y_int_max", "y_t1_min", "y_t1_max", "pLP_t1_min", "pLP_t1_max",
            "u_t1_min", "u_t1_max", "J_track", "J_reg", "Pe_cell", "Pe_node", "max_w",
            "schur_ksp_solves", "schur_ksp_its", "schur_ksp_max", "right_solves", "right_its", "right_max",
            "left_solves", "left_its", "left_max", "ysplit_solves", "ysplit_its", "ysplit_max",
            "sweep_failures", "solve_wall", "maxrss_kb", "wall_s"]
    allkeys = keep + ["y_wiggle_rows", "p_wiggle_rows", "grid_rows", "y_vs_nx32", "y_vs_nx32_inner",
                      "p_LP_vs_nx32", "p_LP_vs_nx32_inner", "y_vs_nx16", "y_vs_nx16_inner",
                      "p_LP_vs_nx16", "p_LP_vs_nx16_inner"]
    out = []
    for r in rows:
        b = r["id"].split("_")[1][1:]
        rec = {k: r.get(k, "") for k in keep}
        rec.update(extra.get((b, int(r["nx"])), {}))
        out.append(rec)
    with open(os.path.join(LP, "lp52_lowpe_results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=allkeys, extrasaction="ignore")
        w.writeheader()
        w.writerows(out)
    # figures (beta = 1e-2, finest level)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lv = max(nx for (b, nx) in snaps if b == "1e-2")
    xs, zs, F = snaps[("1e-2", lv)]
    X, Z = np.meshgrid(xs, zs)
    for key, fname, title in (("y", "lp52_y_t1.png", "state y(x, t = 1)"),
                              ("p_LP", "lp52_p_t1.png", "adjoint p (LP sign, = -p_ex21) at t = 1")):
        fig = plt.figure(figsize=(9, 3.8))
        ax = fig.add_subplot(1, 2, 1, projection="3d")
        ax.plot_surface(X, Z, F[key], cmap="viridis", linewidth=0, antialiased=True, rstride=2, cstride=2)
        ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.view_init(30, -120)
        ax2 = fig.add_subplot(1, 2, 2)
        cs = ax2.contourf(X, Z, F[key], 30, cmap="viridis")
        fig.colorbar(cs, ax=ax2)
        ax2.set_aspect("equal"); ax2.set_xlabel("x1"); ax2.set_ylabel("x2")
        fig.suptitle(f"ex21 LP 5.2, eps = 1/20, beta = 1e-2, Q2 nx = {lv} (LP l = {int(np.log2(lv)) + 1} node set): {title}",
                     fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(LP, "figures", fname), dpi=120)
    print("\nwritten lp52_lowpe_results.csv, figures/lp52_y_t1.png, figures/lp52_p_t1.png")


if __name__ == "__main__":
    main()
