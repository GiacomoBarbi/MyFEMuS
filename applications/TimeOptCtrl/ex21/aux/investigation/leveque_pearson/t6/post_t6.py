#!/usr/bin/env python3
"""T6 post-processing -> ../t6_lp52_stabilised.csv, ../t6_lps_mms.csv, ../t6_solver_effect.csv,
../figures/t6_*.png.  The T4 discretisation diagnostics are imported, not re-implemented:
  wiggles (t3/lp52_post), excess_tv, layer d50/d10 (t4/post_t4), mesh change vs nx/2 at
  shared nodes (all, and away = dist > 0.25 from the corners (1, +-1)), as in post_t4.
New (T6): where/when the all-time minimum of y occurs (--exact-oc-series): t_min, (x, z),
classified by location (corner = within 0.25 of (1, +-1) [the data discontinuities];
boundary = within 2 elements of the boundary; interior) and time (first step / early
t <= 4 dt / final t = T / mid)."""
import csv, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
LP = os.path.dirname(HERE)
R = os.path.join(HERE, "results")
sys.path.insert(0, os.path.join(LP, "t3")); sys.path.insert(0, os.path.join(LP, "t4"))
from lp52_post import snap, wiggles
from post_t4 import layer, excess_tv, lp_patch
EPS = {"1/20": 1 / 20, "1/100": 1 / 100, "1/500": 1 / 500}
STABS = ("none", "p0", "p1")
LABEL = {"none": "Galerkin Q2", "p0": "LPS-P0 (LP)", "p1": "LPS-P1"}


def load(name):
    p = os.path.join(R, name)
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def classify(x, z, t, nx, dt, T):
    dc = min(math.hypot(x - 1, z - 1), math.hypot(x - 1, z + 1))
    db = min(1 - abs(x), 1 - abs(z))
    loc = "corner" if dc <= 0.25 else ("boundary" if db <= 2 * (2 / nx) + 1e-12 else "interior")
    if abs(t - dt) < 1e-12: tm = "first step"
    elif t <= 4 * dt + 1e-12: tm = "early"
    elif abs(t - T) < 1e-12: tm = "final"
    else: tm = "mid"
    return loc, tm, dc


def lp_rows():
    rows, snaps = [], {}
    for grp in ("lp52", "weak", "kkt_gamg", "undershoot", "lowpe"):
        for r in load(grp + ".csv"):
            # timed-out runs print nothing: nx from the run id
            nx = int(r["nx"] or __import__("re").search(r"_nx(\d+)", r["id"]).group(1)); e = r["eps"]; beta = r["beta"]
            rec = dict(group=grp, id=r["id"], stab=r["stab"], eps=e, beta=beta, nx=nx, scheme=r.get("scheme", ""),
                       steps=r.get("steps", ""), LP_level=int(round(math.log2(nx))) + 1,
                       Pe_cell=f"{2 * (2 / nx) / (2 * EPS[e]):.3g}", exit=r["exit"], accepted=r.get("accepted", ""),
                       status=r.get("status", ""), reason=r.get("reason", ""), outer_its=r.get("outer_its", ""),
                       true_residual=r.get("true_residual", ""), adjoint_rel=r.get("adjoint_rel", ""),
                       march_rel=r.get("march_rel", ""), sweep_failures=r.get("sweep_failures", ""),
                       spatial_max_its=max((int(r.get(k + "_max", 0) or 0) for k in ("right", "left", "ay", "ysplit")), default=0),
                       lps_active=r.get("lps_active", ""), lps_patches=r.get("lps_patches", ""),
                       lps_delta_min=r.get("lps_delta_min", ""), lps_delta_max=r.get("lps_delta_max", ""),
                       lps_pe_max=r.get("lps_pe_max", ""), W_over_sym_L0=r.get("lps_W/sym", ""),
                       W_over_skew_L0=r.get("lps_W/skew", ""),
                       y_min_all=r.get("y_min", ""), y_max_all=r.get("y_max", ""), J_track=r.get("J_track", ""),
                       J_reg=r.get("J_reg", ""), solve_wall=r.get("solve_wall", ""), maxrss_kb=r.get("maxrss_kb", ""))
            sp = os.path.join(R, "snap", r["id"] + ".csv")
            se = os.path.join(R, "series", r["id"] + ".csv")
            if r.get("accepted") == "1" and os.path.exists(sp):
                xs, zs, F = snap(sp)
                snaps[(grp, r["stab"], e, beta, nx)] = (xs, zs, F)
                Y = F["y"]
                d50, d10 = layer(xs, zs, Y)
                rec.update(y_t1_min=f"{Y.min():.4e}", y_t1_max=f"{Y.max():.4f}", undershoot_t1=f"{max(0, -Y.min()):.3e}",
                           overshoot_t1=f"{max(0, Y.max() - 1):.3e}", wiggle_rows_y=f"{wiggles(Y, xs)}/{len(zs)}",
                           wiggle_rows_p=f"{wiggles(F['p_LP'], xs)}/{len(zs)}", excess_TV_y=f"{excess_tv(Y):.3e}",
                           excess_TV_p=f"{excess_tv(F['p_LP']):.3e}", layer_d50=f"{d50:.4f}", layer_d10=f"{d10:.4f}",
                           pLP_t1_min=f"{F['p_LP'].min():.3e}", pLP_t1_max=f"{F['p_LP'].max():.3e}")
            if r.get("accepted") == "1" and os.path.exists(se):
                s = np.genfromtxt(se, delimiter=",", names=True)
                k = int(np.argmin(s["y_min"]))
                dt = float(s["t"][0]); T = float(s["t"][-1])
                loc, tm, dc = classify(float(s["x_min"][k]), float(s["z_min"][k]), float(s["t"][k]), nx, dt, T)
                late = s["t"] > 0.25
                kl = int(np.argmin(np.where(late, s["y_min"], np.inf)))
                rec.update(t_min=f"{s['t'][k]:.4f}", x_min=f"{s['x_min'][k]:.4f}", z_min=f"{s['z_min'][k]:.4f}",
                           undershoot_location=loc, undershoot_time=tm, dist_to_corner=f"{dc:.3f}",
                           y_min_after_t025=f"{s['y_min'][kl]:.4e}", t_min_after_t025=f"{s['t'][kl]:.4f}",
                           xz_min_after_t025=f"({s['x_min'][kl]:.3f},{s['z_min'][kl]:.3f})")
            rows.append(rec)
    for rec in rows:
        key = (rec["group"], rec["stab"], rec["eps"], rec["beta"], rec["nx"])
        coarse = key[:4] + (rec["nx"] // 2,)
        if rec["group"] in ("undershoot", "lowpe"):
            continue
        if key in snaps and coarse in snaps:
            xc, zc, Fc = snaps[coarse]; xf, zf, Ff = snaps[key]
            X, Z = np.meshgrid(xc, zc)
            dist = np.minimum(np.hypot(X - 1, Z - 1), np.hypot(X - 1, Z + 1))
            for k, nm in (("y", "y"), ("p_LP", "p")):
                d = np.abs(Fc[k] - Ff[k][::2, ::2])
                rec[f"{nm}_change_vs_coarse"] = f"{d.max():.3e}"
                rec[f"{nm}_change_vs_coarse_away"] = f"{d[dist > 0.25].max():.3e}"
        if rec["stab"] != "none" and key in snaps and key[:1] + ("none",) + key[2:] in snaps:
            a = snaps[key][2]; b = snaps[key[:1] + ("none",) + key[2:]][2]
            rec["max_dy_vs_galerkin_t1"] = f"{np.abs(a['y'] - b['y']).max():.3e}"
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(os.path.join(LP, "t6_lp52_stabilised.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    return rows, snaps


def mms_rows():
    out = []
    for grp, err in (("mms_space", ["err_L2_T", "err_L2_max"]), ("mms_fixedpe", ["err_L2_T", "err_L2_max"]),
                     ("mms_time", ["err_L2_T", "err_L2_max"]), ("oc_space", ["y_L2L2", "p_L2L2", "u_L2L2"]),
                     ("oc_time", ["y_L2L2", "p_L2L2", "u_L2L2"])):
        rr = load(grp + ".csv")
        prev = {}
        for r in rr:
            series = (r["stab"], r["scheme"]) + ((r["nx"],) if "time" in grp else (r["steps"],))
            rec = dict(group=grp, id=r["id"], stab=r["stab"], scheme=r["scheme"], nx=r["nx"], steps=r["steps"],
                       nu=r["nu"], lps_active=r.get("lps_active", ""), lps_patches=r.get("lps_patches", ""),
                       lps_pe_max=r.get("lps_pe_max", ""), reason=r.get("reason", ""), exit=r["exit"])
            for k in err:
                v = float(r[k]); rec[k] = f"{v:.4e}"
                if series in prev:
                    rec["order_" + k] = f"{math.log2(prev[series][k] / v):.2f}"
            prev[series] = {k: float(r[k]) for k in err}
            out.append(rec)
    keys = []
    for r in out:
        keys += [k for k in r if k not in keys]
    with open(os.path.join(LP, "t6_lps_mms.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(out)
    return out


def figures(rows, snaps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"none": "k", "p0": "C3", "p1": "C0"}
    # beta = 1e-2 profiles y(x1, x2 = 0, t = 1) and p
    fig, ax = plt.subplots(2, 3, figsize=(14, 7))
    for j, e in enumerate(("1/20", "1/100", "1/500")):
        for nx, ls in ((16, ":"), (64, "-")):
            for st in STABS:
                k = ("lp52", st, e, "1e-2", nx)
                if k in snaps:
                    xs, zs, F = snaps[k]; jj = int(np.argmin(np.abs(zs)))
                    ax[0, j].plot(xs, F["y"][jj], ls, color=col[st], lw=1, label=f"{LABEL[st]} nx={nx}")
                    ax[1, j].plot(xs, F["p_LP"][jj], ls, color=col[st], lw=1, label=f"{LABEL[st]} nx={nx}")
        ax[0, j].set_title(f"beta=1e-2, eps={e}: y(x1, x2=0, t=1)", fontsize=8); ax[0, j].set_xlim(0, 1)
        ax[1, j].set_title(f"beta=1e-2, eps={e}: p_LP(x1, x2=0, t=1)", fontsize=8)
        for a in ax[:, j]:
            a.grid(True, lw=0.3); a.legend(fontsize=5); a.set_xlabel("x1")
    fig.tight_layout(); fig.savefig(os.path.join(LP, "figures", "t6_profiles_beta1e-2.png"), dpi=110)
    # weak control profiles
    fig, ax = plt.subplots(2, 3, figsize=(14, 7))
    for i, b in enumerate(("1e-1", "1e2")):
        for j, st in enumerate(STABS):
            for nx in (8, 16, 32, 64):
                k = ("weak", st, "1/500", b, nx)
                if k in snaps:
                    xs, zs, F = snaps[k]; jj = int(np.argmin(np.abs(zs)))
                    ax[i, j].plot(xs, F["y"][jj], marker="." if nx <= 16 else None, lw=1, label=f"nx={nx}")
            ax[i, j].set_title(f"{LABEL[st]}: eps=1/500, beta={b}: y(x1, x2=0, t=1)", fontsize=8)
            ax[i, j].set_xlim(0.5, 1.0); ax[i, j].grid(True, lw=0.3); ax[i, j].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(LP, "figures", "t6_profiles_weak_control.png"), dpi=110)
    # undershoot time series
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    for j, (grp, b) in enumerate((("lp52", "1e-2"), ("weak", "1e-1"), ("weak", "1e2"))):
        for st in STABS:
            for nx, ls in ((16, ":"), (32, "--"), (64, "-")):
                rid = f"{grp}_{st}_eps500_b{b}_nx{nx}"
                p = os.path.join(R, "series", rid + ".csv")
                if os.path.exists(p):
                    s = np.genfromtxt(p, delimiter=",", names=True)
                    ax[j].plot(s["t"], s["y_min"], ls, color=col[st], lw=1, label=f"{LABEL[st]} nx={nx}")
        ax[j].axhline(0, color="gray", lw=0.5); ax[j].set_xlabel("t"); ax[j].set_ylabel("min_x y(x, t)")
        ax[j].set_title(f"eps=1/500, beta={b}: spatial minimum of y vs t", fontsize=8)
        ax[j].grid(True, lw=0.3); ax[j].legend(fontsize=5)
    fig.tight_layout(); fig.savefig(os.path.join(LP, "figures", "t6_undershoot_time.png"), dpi=110)
    # unstabilised vs LPS fields at the hardest weak case
    fig, ax = plt.subplots(2, 3, figsize=(13, 8))
    for i, nx in enumerate((16, 64)):
        for j, st in enumerate(STABS):
            k = ("weak", st, "1/500", "1e2", nx)
            if k in snaps:
                xs, zs, F = snaps[k]
                X, Z = np.meshgrid(xs, zs)
                cs = ax[i, j].contourf(X, Z, F["y"], levels=np.linspace(-0.15, 1.15, 27), cmap="RdBu_r", extend="both")
                ax[i, j].contour(X, Z, F["y"], levels=[0.0], colors="k", linewidths=0.5)
                ax[i, j].set_aspect("equal")
                ax[i, j].set_title(f"{LABEL[st]} nx={nx}: y(t=1), eps=1/500, beta=1e2\n"
                                   f"min {F['y'].min():+.3f}  max {F['y'].max():.3f}  wiggle rows {wiggles(F['y'], xs)}/{len(zs)}",
                                   fontsize=7)
    fig.colorbar(cs, ax=ax, shrink=0.6)
    fig.savefig(os.path.join(LP, "figures", "t6_unstabilised_vs_lps.png"), dpi=110)


def lp_fig23(snaps):
    """Side-by-side structure at LP's Figs. 2-3 setting (l = 5 <-> Q2 nx = 16, beta = 1e-2,
    eps = 1/20, 1/100): Galerkin Q2 vs LPS-P0.  Structure only (LP: Q1, OTD, raster)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 4, figsize=(16, 7.5))
    for i, e in enumerate(("1/20", "1/100")):
        for j, (st, key) in enumerate((("none", "y"), ("p0", "y"), ("none", "p_LP"), ("p0", "p_LP"))):
            k = ("lp52", st, e, "1e-2", 16)
            if k not in snaps:
                continue
            xs, zs, F = snaps[k]
            X, Z = np.meshgrid(xs, zs)
            cs = ax[i, j].contourf(X, Z, F[key], 30, cmap="viridis")
            fig.colorbar(cs, ax=ax[i, j], shrink=0.8)
            ax[i, j].set_aspect("equal")
            ax[i, j].set_title(f"{LABEL[st]} nx=16 (LP l=5), eps={e}: {key}(t=1)\n"
                               f"range [{F[key].min():.4f}, {F[key].max():.4f}]", fontsize=7)
    fig.suptitle("ex21 Q2 DTO vs LP Figs. 2-3 setting (beta = 1e-2): structure only, no digit-level comparison", fontsize=8)
    fig.savefig(os.path.join(LP, "figures", "t6_lp_figs23_l5.png"), dpi=110)


def solver_rows():
    rr = load("solver_effect.csv")
    if rr:
        with open(os.path.join(LP, "t6_solver_effect.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rr[0])); w.writeheader(); w.writerows(rr)


if __name__ == "__main__":
    rows, snaps = lp_rows()
    mms_rows()
    solver_rows()
    figures(rows, snaps)
    lp_fig23(snaps)
    hdr = ["group", "stab", "eps", "beta", "nx", "accepted", "lps_active", "y_min_all", "t_min", "x_min", "z_min",
           "undershoot_location", "undershoot_time", "y_min_after_t025", "y_t1_min", "y_t1_max", "wiggle_rows_y",
           "wiggle_rows_p", "excess_TV_y", "layer_d50", "layer_d10", "y_change_vs_coarse", "y_change_vs_coarse_away",
           "max_dy_vs_galerkin_t1", "J_track", "J_reg"]
    print("| " + " | ".join(hdr) + " |")
    for r in rows:
        print("| " + " | ".join(str(r.get(k, "")) for k in hdr) + " |")
