#!/usr/bin/env python3
"""T4 post-processing: high-Peclet map of LP 5.2 (beta = 1e-2, tau = 1/nx).

Discretisation (from the `lu` runs = the exact discrete KKT solution with exact
spatial sweeps; the `gamg` solution is compared with it where both converge):
  * y range over all time levels (EX21_LP52_RESULT) and at t = 1, overshoot
    max(y) - 1 and undershoot -min(y) (the data and the target lie in [0, 1];
    the control may push y outside, so these are indicators, not gates);
  * wiggle rows (lp52_post.wiggles: >= 3 consecutive sign alternations of the
    x1 second difference above 1e-3 of the row range);
  * excess total variation per x2-row, TV(row) - (max - min)(row), summed and
    normalised by the number of rows: 0 for monotone rows, > 0 for any
    non-monotone row (a smooth physical bump also contributes, so it is read
    together with the wiggle count and the mesh comparison);
  * the x1 = 1 layer: on the row x2 = 0, the distance from x1 = 1 at which y
    falls to 1/2 and to 1/10 (linear interpolation), against eps/|w| scales;
  * mesh-to-mesh differences of y at shared nodes (nx -> 2 nx), all nodes and
    excluding dist < 0.25 from the corners (1, +-1).
Solver: outer FGMRES its / reason / history, Schur its per outer and cap hits
(schur_ksp_max == 100), spatial solves / mean / max its and failures
(sweep_failures counts every spatial solve that did not converge, including
max_it), wall time.  LP patch Peclet of the matching LP level l = log2(nx) + 1.
Writes ../../t4_highpe_map.csv, ../../t4_spatial_block_solver.csv,
../../figures/t4_y_profiles.png, ../../figures/t4_block_iterations.png."""
import csv, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LP = os.path.dirname(HERE)
R = os.path.join(HERE, "results")
sys.path.insert(0, os.path.join(LP, "t3"))
from lp52_post import snap, wiggles

EPS = {"20": 1 / 20, "100": 1 / 100, "500": 1 / 500}


def lp_patch(eps, l):
    h = 2.0 ** (1 - l)
    n = 2 ** (l - 1)
    c = -1 + (2 * np.arange(n) + 1) * h
    X, Y = np.meshgrid(c, c)
    wx, wy = 2 * Y * (1 - X ** 2), -2 * X * (1 - Y ** 2)
    sp = np.hypot(wx, wy)
    m = np.maximum(np.abs(wx), np.abs(wy)) / np.where(sp > 0, sp, 1)
    hk = np.where(sp > 0, 2 * h / np.where(m > 0, m, 1), 2 * h)
    pe = sp * hk / (2 * eps)
    return pe.max(), int((pe > 1).sum()), n * n


def layer(xs, zs, Y):
    j = int(np.argmin(np.abs(zs)))
    row = Y[j]
    out = []
    for lev in (0.5, 0.1):
        d = np.nan
        for i in range(len(xs) - 1, 0, -1):
            if (row[i] - lev) * (row[i - 1] - lev) <= 0 and row[i] != row[i - 1]:
                x = xs[i - 1] + (lev - row[i - 1]) * (xs[i] - xs[i - 1]) / (row[i] - row[i - 1])
                d = 1 - x
                break
        out.append(d)
    return out


def excess_tv(Y):
    tv = np.abs(np.diff(Y, axis=1)).sum(axis=1)
    rng = Y.max(axis=1) - Y.min(axis=1)
    return float(np.mean(tv - rng))


def main():
    rows = []
    snaps = {}
    for group in ("cn_map", "be_map", "beta_map"):
        p = os.path.join(R, f"{group}.csv")
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p)):
            parts = r["id"].split("_")
            scheme = parts[1]
            beta = parts[2][1:] if parts[2].startswith("b") else "1e-2"
            if beta != "1e-2":
                parts = parts[:2] + parts[3:]
            e = parts[2][3:]
            nx = int(parts[3][2:])
            var = parts[4]
            l = int(round(math.log2(nx))) + 1
            pmax, act, tot = lp_patch(EPS[e], l)
            rec = dict(scheme=scheme.upper(), beta=beta, eps=f"1/{e}", nx=nx, LP_level=l, variant=var, status=r.get("status"),
                       Pe_cell=f"{2 * (2 / nx) / (2 * EPS[e]):.3g}", Pe_node=f"{2 * (1 / nx) / (2 * EPS[e]):.3g}",
                       LP_patch_Pe_max=f"{pmax:.3g}", LP_LPS_active_patches=f"{act}/{tot}",
                       LP_LPS="ON" if act else "off", reason=r.get("reason", ""), outer_its=r.get("outer_its", ""),
                       outer_history=r.get("outer_history", ""), true_residual=r.get("true_residual", ""),
                       adjoint_rel=r.get("adjoint_rel", ""), march_rel=r.get("march_rel", ""),
                       schur_its_per_outer=(f"{int(r['schur_ksp_its']) / max(1, int(r['schur_ksp_solves'])):.1f}"
                                            if r.get("schur_ksp_its") else ""),
                       schur_cap_hit=(int(r["schur_ksp_max"]) >= 100) if r.get("schur_ksp_max") else "",
                       spatial_solves=sum(int(r.get(k + "_solves", 0) or 0) for k in ("right", "left", "ay", "ysplit")),
                       spatial_its_mean=(f"{sum(int(r.get(k + '_its', 0) or 0) for k in ('right', 'left', 'ay', 'ysplit')) / max(1, sum(int(r.get(k + '_solves', 0) or 0) for k in ('right', 'left', 'ay', 'ysplit'))):.1f}"
                                         if r.get("right_solves") else ""),
                       spatial_its_max=max((int(r.get(k + "_max", 0) or 0) for k in ("right", "left", "ay", "ysplit")), default=0),
                       spatial_failures=r.get("sweep_failures", ""), finite=r.get("finite", ""),
                       y_min_all=r.get("y_min", ""), y_max_all=r.get("y_max", ""), solve_wall=r.get("solve_wall", ""),
                       wall_s=r.get("wall_s", ""), maxrss_kb=r.get("maxrss_kb", ""))
            sp = os.path.join(R, f"snap_{scheme}_eps{e}_nx{nx}_{var}.csv" if beta == "1e-2"
                              else f"snap_{scheme}_b{beta}_eps{e}_nx{nx}_{var}.csv")
            if r.get("reason") and int(r["reason"]) > 0 and r.get("finite") == "1" and os.path.exists(sp):
                xs, zs, F = snap(sp)
                snaps[(scheme, e, nx, var, beta)] = (xs, zs, F)
                Y = F["y"]
                l5, l1 = layer(xs, zs, Y)
                rec.update(y_t1_min=f"{Y.min():.4e}", y_t1_max=f"{Y.max():.4f}", overshoot=f"{max(0, Y.max() - 1):.3e}",
                           undershoot=f"{max(0, -Y.min()):.3e}", wiggle_rows_y=f"{wiggles(Y, xs)}/{len(zs)}",
                           wiggle_rows_p=f"{wiggles(F['p_LP'], xs)}/{len(zs)}", excess_TV_y=f"{excess_tv(Y):.3e}",
                           layer_d50=f"{l5:.4f}", layer_d10=f"{l1:.4f}",
                           pLP_t1_min=f"{F['p_LP'].min():.3e}", pLP_t1_max=f"{F['p_LP'].max():.3e}")
            rows.append(rec)
    # mesh-to-mesh and gamg-vs-lu
    for rec in rows:
        key = (rec["scheme"].lower(), rec["eps"][2:], rec["nx"], rec["variant"], rec["beta"])
        coarse = (key[0], key[1], key[2] // 2, key[3], key[4])
        if key in snaps and coarse in snaps:
            xc, zc, Fc = snaps[coarse]
            xf, zf, Ff = snaps[key]
            sub = Ff["y"][::2, ::2]
            X, Z = np.meshgrid(xc, zc)
            dist = np.minimum(np.hypot(X - 1, Z - 1), np.hypot(X - 1, Z + 1))
            d = np.abs(Fc["y"] - sub)
            rec["y_change_vs_coarse"] = f"{d.max():.3e}"
            rec["y_change_vs_coarse_away"] = f"{d[dist > 0.25].max():.3e}"
        if rec["variant"] == "gamg" and key in snaps and (key[0], key[1], key[2], "lu", key[4]) in snaps:
            a = snaps[key][2]["y"]
            b = snaps[(key[0], key[1], key[2], "lu", key[4])][2]["y"]
            rec["gamg_vs_lu_y"] = f"{np.abs(a - b).max():.2e}"
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(os.path.join(LP, "t4_highpe_map.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("| scheme | beta | eps | nx (LP l) | Pe cell | LP patch Pe / LPS | variant | reason | outer | Schur/outer | spatial mean/max | spatial failures |"
          " y range (all t) | y(t=1) range | wiggle rows y | excess TV | layer d50 / d10 | y change vs nx/2 (all / away) | gamg vs lu |")
    print("|" + "---|" * 19)
    for r in sorted(rows, key=lambda r: (r["scheme"], float(r["beta"]), -EPS[r["eps"][2:]], r["nx"], r["variant"])):
        print(f"| {r['scheme']} | {r['beta']} | {r['eps']} | {r['nx']} ({r['LP_level']}) | {r['Pe_cell']} | {r['LP_patch_Pe_max']} / {r['LP_LPS']} |"
              f" {r['variant']} | {r['reason'] or r['status']} | {r['outer_its']} | {r['schur_its_per_outer']} |"
              f" {r['spatial_its_mean']}/{r['spatial_its_max']} | {r['spatial_failures']} |"
              f" [{r['y_min_all']}, {r['y_max_all']}] | [{r.get('y_t1_min', '')}, {r.get('y_t1_max', '')}] |"
              f" {r.get('wiggle_rows_y', '')} | {r.get('excess_TV_y', '')} | {r.get('layer_d50', '')} / {r.get('layer_d10', '')} |"
              f" {r.get('y_change_vs_coarse', '')} / {r.get('y_change_vs_coarse_away', '')} | {r.get('gamg_vs_lu_y', '')} |")
    # block CSV
    p = os.path.join(R, "blocks.csv")
    if os.path.exists(p):
        blk = list(csv.DictReader(open(p)))
        with open(os.path.join(LP, "t4_spatial_block_solver.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(blk[0]))
            w.writeheader()
            w.writerows(blk)
    # figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for j, e in enumerate(("20", "100", "500")):
        for nx in (8, 16, 32, 64):
            k = ("cn", e, nx, "lu", "1e-2")
            if k in snaps:
                xs, zs, F = snaps[k]
                jj = int(np.argmin(np.abs(zs)))
                ax[j].plot(xs, F["y"][jj], marker="." if nx <= 16 else None, lw=1, label=f"nx={nx} (Pe_cell {2*(2/nx)/(2*EPS[e]):.3g})")
        ax[j].set_title(f"CN, eps = 1/{e}: y(x1, x2 = 0, t = 1)", fontsize=8)
        ax[j].set_xlim(0.0, 1.0); ax[j].legend(fontsize=6); ax[j].grid(True, lw=0.3); ax[j].set_xlabel("x1")
    fig.tight_layout()
    fig.savefig(os.path.join(LP, "figures", "t4_y_profiles.png"), dpi=110)
    fig, ax = plt.subplots(2, 2, figsize=(10, 6.5))
    for i, b in enumerate(("1e-1", "1e2")):
        for j, e in enumerate(("100", "500")):
            for nx in (16, 32, 64):
                k = ("cn", e, nx, "lu", b)
                if k in snaps:
                    xs, zs, F = snaps[k]
                    jj = int(np.argmin(np.abs(zs)))
                    ax[i, j].plot(xs, F["y"][jj], marker="." if nx <= 16 else None, lw=1,
                                  label=f"nx={nx} (Pe_cell {2*(2/nx)/(2*EPS[e]):.3g})")
            ax[i, j].set_title(f"CN, beta = {b}, eps = 1/{e}: y(x1, x2 = 0, t = 1)", fontsize=8)
            ax[i, j].set_xlim(0.5, 1.0); ax[i, j].legend(fontsize=6); ax[i, j].grid(True, lw=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(LP, "figures", "t4_y_profiles_beta.png"), dpi=110)
    if os.path.exists(p):
        fig, ax = plt.subplots(figsize=(6, 4))
        for th, mk in (("1", "o"), ("0.5", "s")):
            for blkname, c in (("E", "C0"), ("right", "C1")):
                pts = [(float(b["adv_ratio"]), int(b["max_its"]), float(b["Pe_cell"])) for b in blk
                       if b["theta"] == th and b["block"] == blkname]
                for pe_sel, alpha in ((lambda pe: pe <= 1.3, 0.35), (lambda pe: pe > 1.3, 1.0)):
                    q = [(a, i) for a, i, pe in pts if pe_sel(pe)]
                    if q:
                        ax.scatter([a for a, _ in q], [i for _, i in q], marker=mk, color=c, alpha=alpha, s=18,
                                   label=f"theta={th} {blkname} " + ("Pe<=1.25" if alpha < 1 else "Pe>1.25"))
        ax.set_xscale("log"); ax.set_xlabel("theta |w|max dt / h"); ax.set_ylabel("max GMRES its (cap 500)")
        ax.axhline(500, color="k", lw=0.6, ls="--"); ax.legend(fontsize=6); ax.grid(True, lw=0.3)
        ax.set_title("spatial blocks M/dt + theta L (+ sqrt(beta) M): GMRES(50)+GAMG", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(LP, "figures", "t4_block_iterations.png"), dpi=110)


if __name__ == "__main__":
    main()
