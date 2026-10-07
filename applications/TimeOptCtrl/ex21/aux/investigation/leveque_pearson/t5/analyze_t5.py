#!/usr/bin/env python3
"""T5 analysis: ../t5_block_solver_screen.csv, ../t5_kkt_solver_shortlist.csv,
../figures/t5_iterations_vs_peclet.png, ../figures/t5_time_vs_solver.png."""
import csv, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LP = os.path.dirname(HERE)
R = os.path.join(HERE, "results")
sys.path.insert(0, os.path.join(LP, "t3"))
from lp52_post import snap


def load(name):
    p = os.path.join(R, name)
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def main():
    blk = load("screen.csv") + load("scale.csv")
    keys = ["cfg", "point", "eps", "nx", "theta", "block", "Pe_cell", "adv_ratio", "its", "max_its", "cap_hits",
            "reasons", "setup_s", "solve_s", "maxrss_kb", "petsc_options"]
    with open(os.path.join(LP, "t5_block_solver_screen.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in blk:
            r = dict(r)
            r["solve_s"] = r["solve_s"]  # total over the 4 RHS
            w.writerow(r)
    # per-config verdict over all block points
    print("| config | points | all converged | max its (worst block) | worst point | setup E theta=1 (ms), D128 | solve/RHS E theta=1 (ms), D128 |")
    print("|---|---|---|---|---|---|---|")
    cfgs = []
    for r in blk:
        if r["cfg"] not in cfgs:
            cfgs.append(r["cfg"])
    for c in cfgs:
        rr = [r for r in blk if r["cfg"] == c]
        ok = all(int(v) >= 0 for r in rr for v in r["reasons"].split("/"))
        worst = max(rr, key=lambda r: int(r["max_its"]))
        d = [r for r in rr if r["point"] == "D128" and r["theta"] == "1" and r["block"] == "E"]
        st = (f"{1e3 * float(d[0]['setup_s']):.0f}", f"{1e3 * float(d[0]['solve_s']) / 4:.1f}") if d else ("–", "–")
        print(f"| {c} | {len({r['point'] for r in rr})} | {'yes' if ok else 'NO'} | {worst['max_its']} |"
              f" {worst['point']} theta={worst['theta']} {worst['block']} | {st[0]} | {st[1]} |")
    # KKT shortlist
    kkt = []
    for c in ("lu_petsc", "ilu1", "asm4o1_ilu1"):
        kkt += load(f"kkt_{c}.csv")
    ref = {(r["scheme"], r["eps"], r["nx"]): r for r in kkt if r["cfg"] == "lu_petsc"}
    out = []
    print("\n| config | scheme | eps | nx | status | outer | Schur its/outer | spatial solves | mean/max spatial its |"
          " failures | KSP s | solve wall s | total wall s | rel. KKT (state/adj) | march | max|dy| vs LU | max|dp| vs LU | J_track / J_reg rel. diff vs LU |")
    print("|" + "---|" * 18)
    for r in kkt:
        key = (r["scheme"], r["eps"], r["nx"])
        solves = sum(int(r.get(k + "_solves", 0) or 0) for k in ("right", "left", "ay", "ysplit"))
        its = sum(int(r.get(k + "_its", 0) or 0) for k in ("right", "left", "ay", "ysplit"))
        mx = max(int(r.get(k + "_max", 0) or 0) for k in ("right", "left", "ay", "ysplit")) if r.get("right_max") else ""
        schur = (int(r["schur_ksp_its"]) / max(1, int(r["schur_ksp_solves"]))) if r.get("schur_ksp_its") else float("nan")
        dy = dp = dj = ""
        if key in ref and r["cfg"] != "lu_petsc" and r.get("reason") and int(r["reason"]) > 0:
            a = snap(os.path.join(R, f"snap_{r['cfg']}_{r['scheme'].lower()}_eps{r['eps'][2:]}_nx{r['nx']}.csv"))[2]
            b = snap(os.path.join(R, f"snap_lu_petsc_{r['scheme'].lower()}_eps{r['eps'][2:]}_nx{r['nx']}.csv"))[2]
            dy = f"{np.nanmax(np.abs(a['y'] - b['y'])):.1e}"
            dp = f"{np.nanmax(np.abs(a['p_LP'] - b['p_LP'])):.1e}"
            rf = ref[key]
            dj = (f"{abs(float(r['J_track']) - float(rf['J_track'])) / abs(float(rf['J_track'])):.1e} / "
                  f"{abs(float(r['J_reg']) - float(rf['J_reg'])) / abs(float(rf['J_reg'])):.1e}")
        rec = dict(cfg=r["cfg"], scheme=r["scheme"], eps=r["eps"], nx=r["nx"], status=r["status"],
                   reason=r.get("reason", ""), outer_its=r.get("outer_its", ""), outer_history=r.get("outer_history", ""),
                   schur_its_per_outer=f"{schur:.1f}", spatial_solves=solves,
                   spatial_its_mean=f"{its / max(1, solves):.2f}", spatial_its_max=mx,
                   sweep_failures=r.get("sweep_failures", ""), ksp_solve_s=r.get("ksp_solve_s", ""),
                   solve_wall=r.get("solve_wall", ""), wall_s=r.get("wall_s", ""), maxrss_kb=r.get("maxrss_kb", ""),
                   state_rel=r.get("state_rel", ""), adjoint_rel=r.get("adjoint_rel", ""), march_rel=r.get("march_rel", ""),
                   true_residual=r.get("true_residual", ""), J_track=r.get("J_track", ""), J_reg=r.get("J_reg", ""),
                   max_dy_vs_lu=dy, max_dp_vs_lu=dp, J_rel_diff_vs_lu=dj, petsc_options=r.get("petsc_options", ""))
        out.append(rec)
        print(f"| {rec['cfg']} | {rec['scheme']} | {rec['eps']} | {rec['nx']} | {rec['status']}/{rec['reason']} |"
              f" {rec['outer_its']} | {rec['schur_its_per_outer']} | {solves} | {rec['spatial_its_mean']}/{mx} |"
              f" {rec['sweep_failures']} | {rec['ksp_solve_s']} | {rec['solve_wall']} | {rec['wall_s']} |"
              f" {rec['state_rel']}/{rec['adjoint_rel']} | {rec['march_rel']} | {dy} | {dp} | {dj} |")
    if out:
        with open(os.path.join(LP, "t5_kkt_solver_shortlist.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out[0]))
            w.writeheader()
            w.writerows(out)
    # figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for c, mk in (("gamg_base", "x"), ("gamg_sor_unsm4", "+"), ("ilu0", "v"), ("ilu1", "o"), ("ilu2", "^"),
                  ("asm4o1_ilu1", "s"), ("bjac4_ilu1", "D"), ("lu_petsc", "*")):
        pts = sorted({(float(r["Pe_cell"]), max(int(x["max_its"]) for x in blk if x["cfg"] == c and x["point"] == r["point"]))
                      for r in blk if r["cfg"] == c})
        if pts:
            ax.plot([p for p, _ in pts], [i for _, i in pts], marker=mk, lw=0.8, label=c)
    ax.axhline(500, color="k", ls="--", lw=0.6)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Pe_cell"); ax.set_ylabel("max GMRES its over E/right/left, both theta")
    ax.legend(fontsize=7); ax.grid(True, which="both", lw=0.3)
    ax.set_title("T5 spatial block screen (cap 500 = failure)", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(LP, "figures", "t5_iterations_vs_peclet.png"), dpi=110)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for c, mk in (("ilu0", "v"), ("ilu1", "o"), ("ilu2", "^"), ("asm4o1_ilu1", "s"), ("bjac4_ilu1", "D"),
                  ("lu_petsc", "*"), ("lu_umfpack", "P"), ("lu_mumps", "X")):
        pts = sorted((int(r["nx"]), float(r["setup_s"]), float(r["solve_s"]) / 4) for r in blk
                     if r["cfg"] == c and r["eps"] == "1/500" and r["theta"] == "1" and r["block"] == "E")
        if pts:
            n = [(2 * p[0] + 1) ** 2 for p in pts]
            ax.plot(n, [p[2] * 1e3 for p in pts], marker=mk, lw=0.8, label=f"{c} solve/RHS")
            ax.plot(n, [p[1] * 1e3 for p in pts], marker=mk, lw=0.8, ls=":", alpha=0.6)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("spatial DOFs (eps = 1/500, E block, theta = 1)")
    ax.set_ylabel("ms (solid: solve per RHS; dotted: setup)"); ax.legend(fontsize=6); ax.grid(True, which="both", lw=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(LP, "figures", "t5_time_vs_solver.png"), dpi=110)


if __name__ == "__main__":
    main()
