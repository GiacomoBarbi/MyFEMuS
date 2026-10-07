#!/usr/bin/env python3
"""LP 5.2 (eps = 1/20, CN, Q2 nx = 32 / 64) OLD (T3, affine tracking term
dropped) vs NEW (T3b, corrected) for beta = 1e-2, 1e-4, 1e-6.
Writes ../lp52_smallbeta_corrected.csv and figures/lp52_corner_old_new.png."""
import csv, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
LP = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(LP, "t3"))
from lp52_post import snap, wiggles

OLD = {r["id"].split("_")[1][1:] + "_" + r["nx"]: r
       for g in ("lp52_mesh", "lp52_beta") for r in csv.DictReader(open(os.path.join(LP, "t3", "results", g + ".csv")))}
NEW = {r["id"].split("_")[1][1:] + "_" + r["nx"]: r
       for r in csv.DictReader(open(os.path.join(HERE, "results", "lp52_corr.csv")))}


def snapfile(which, b, nx):
    return (os.path.join(LP, "t3", "results", f"lp52_snapshot_b{b}_nx{nx}.csv") if which == "old"
            else os.path.join(HERE, "results", f"lp52corr_snapshot_b{b}_nx{nx}.csv"))


def metrics(which, b, nx):
    xs, zs, F = snap(snapfile(which, b, nx))
    X, Z = np.meshgrid(xs, zs)
    P = F["p_LP"]
    s = np.abs(P).max()
    dist = np.minimum.reduce([1 - X, X + 1, 1 - Z, Z + 1])
    i = np.unravel_index(np.abs(P).argmax(), P.shape)
    odd = np.zeros_like(P)
    odd[:, 1:-1] = np.abs(P[:, 1:-1] - 0.5 * (P[:, :-2] + P[:, 2:]))
    odd[1:-1, :] = np.maximum(odd[1:-1, :], np.abs(P[1:-1, :] - 0.5 * (P[:-2, :] + P[2:, :])))
    band = dist < 0.1
    return dict(pmin=P.min(), pmax=P.max(), peak=(X[i], Z[i]), peak_dist=dist[i],
                odd_band=odd[band].max() / s, odd_int=odd[~band].max() / s,
                wig_y=wiggles(F["y"], xs), wig_p=wiggles(P, xs), rows=len(zs),
                ymax=F["y"].max(), ymin=F["y"].min(), F=F, xs=xs, zs=zs)


def change(which, b):
    a, c = metrics(which, b, 32), metrics(which, b, 64)
    sub = c["F"]["p_LP"][::2, ::2]
    full = np.max(np.abs(a["F"]["p_LP"] - sub)) / np.max(np.abs(sub))
    X, Z = np.meshgrid(a["xs"], a["zs"])
    dist = np.minimum.reduce([1 - X, X + 1, 1 - Z, Z + 1])
    away = np.max(np.abs(a["F"]["p_LP"] - sub)[dist >= 0.1]) / np.max(np.abs(sub))
    return full, away


rows = []
print("| beta | nx | version | J_track | J_reg | p_LP(t=1) range | peak (x1,x2), dist | odd-even band / interior |"
      " wiggle rows y / p | y range | p change nx32->64 (all / dist>=0.1) | outer | rel. KKT adj | Schur its/outer | max spatial its |")
print("|" + "---|" * 15)
for b in ("1e-2", "1e-4", "1e-6"):
    for nx in (32, 64):
        for which, tab in (("old", OLD), ("new", NEW)):
            r = tab[f"{b}_{nx}"]
            m = metrics(which, b, nx)
            ch = change(which, b) if nx == 64 else (float("nan"), float("nan"))
            sp = max(int(r["right_max"]), int(r["left_max"]), int(r["ysplit_max"]))
            schur = int(r["schur_ksp_its"]) / max(1, int(r["schur_ksp_solves"]))
            print(f"| {b} | {nx} | {which} | {float(r['J_track']):.6e} | {float(r['J_reg']):.6e} |"
                  f" [{m['pmin']:.3e}, {m['pmax']:.3e}] | ({m['peak'][0]:+.3f},{m['peak'][1]:+.3f}), {m['peak_dist']:.3f} |"
                  f" {m['odd_band']:.2e} / {m['odd_int']:.2e} | {m['wig_y']}/{m['rows']} / {m['wig_p']}/{m['rows']} |"
                  f" [{m['ymin']:.2e}, {m['ymax']:.4f}] | {ch[0]:.2e} / {ch[1]:.2e} | {r['outer_its']} |"
                  f" {r['adjoint_rel']} | {schur:.1f} | {sp} |")
            rows.append(dict(beta=b, nx=nx, version=which, J_track=r["J_track"], J_reg=r["J_reg"],
                             pLP_t1_min=f"{m['pmin']:.6e}", pLP_t1_max=f"{m['pmax']:.6e}",
                             peak_x1=f"{m['peak'][0]:.4f}", peak_x2=f"{m['peak'][1]:.4f}",
                             peak_dist=f"{m['peak_dist']:.4f}", oddeven_band=f"{m['odd_band']:.3e}",
                             oddeven_interior=f"{m['odd_int']:.3e}", wiggle_rows_y=m["wig_y"],
                             wiggle_rows_p=m["wig_p"], grid_rows=m["rows"], y_min=f"{m['ymin']:.6e}",
                             y_max=f"{m['ymax']:.6e}", p_change_32_64=f"{ch[0]:.3e}",
                             p_change_32_64_away=f"{ch[1]:.3e}", outer_its=r["outer_its"],
                             state_rel=r["state_rel"], adjoint_rel=r["adjoint_rel"],
                             march_rel=r["march_rel"], schur_its_per_outer=f"{schur:.2f}",
                             max_spatial_its=sp, solve_wall=r["solve_wall"]))
with open(os.path.join(LP, "lp52_smallbeta_corrected.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 3, figsize=(12, 6.5))
for j, b in enumerate(("1e-2", "1e-4", "1e-6")):
    for i, which in enumerate(("old", "new")):
        m = metrics(which, b, 64)
        X, Z = np.meshgrid(m["xs"], m["zs"])
        sel = (X >= 0.5) & (Z >= 0.5)
        n = int(np.sqrt(sel.sum()))
        cs = ax[i, j].contourf(X[sel].reshape(n, -1), Z[sel].reshape(n, -1), m["F"]["p_LP"][sel].reshape(n, -1), 30)
        fig.colorbar(cs, ax=ax[i, j])
        ax[i, j].set_title(f"{which.upper()}  beta = {b}: p_LP(t=1) near (1,1)", fontsize=8)
        ax[i, j].set_aspect("equal")
fig.suptitle("LP 5.2, eps = 1/20, Q2 nx = 64: adjoint near the (1,1) corner, before/after T3b", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(LP, "figures", "lp52_corner_old_new.png"), dpi=110)
print("\nwritten lp52_smallbeta_corrected.csv, figures/lp52_corner_old_new.png")
