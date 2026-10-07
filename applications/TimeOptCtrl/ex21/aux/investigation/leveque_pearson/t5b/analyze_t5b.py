#!/usr/bin/env python3
"""T5b analysis: ../t5b_hypre_block_screen.csv, ../t5b_hypre_kkt.csv,
../figures/t5b_iterations_vs_peclet.png, ../figures/t5b_time_vs_backend.png.
The KKT table compares the hypre runs with the T5 LU (lu_petsc) and ASM+ILU
(asm4o1_ilu1) runs of the same cases (T5 snapshots in ../t5/results)."""
import csv, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LP = os.path.dirname(HERE)
R = os.path.join(HERE, "results")
R5 = os.path.join(LP, "t5", "results")
sys.path.insert(0, os.path.join(LP, "t3"))
from lp52_post import snap

NAMES = {("1", "E"): "t1_E", ("1", "right"): "t1_right", ("1", "left"): "t1_left",
         ("0.5", "E"): "th_E", ("0.5", "right"): "th_right", ("0.5", "left"): "th_left"}
HCFG = ("hypre_default", "hypre_ns", "hypre_air1", "hypre_air2", "hypre_air1_gs", "hypre_air1_mrs1")


def load(path):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []


def block_table():
    hier = {(h["cfg"], h["point"], h["block"]): h for h in load(os.path.join(R, "hierarchy.csv"))}
    probe = {(p["cfg"], p["point"], p["theta"], p["block"]): p["levels"]
             for p in load(os.path.join(R, "rowsum_probe.csv")) if p["max_row_sum"] == "1.0"}
    rows = []
    for src, recs in (("screen", load(os.path.join(R, "screen.csv"))),
                      ("screen", load(os.path.join(R, "screen_hypre_air1_mrs1.csv"))),
                      ("t5_baseline_on_hypre_binary", load(os.path.join(R, "baseline.csv")))):
        for r in recs:
            r = dict(r, source=src)
            th = "1" if r["theta"] in ("1", "1.0") else "0.5"
            h = hier.get((r["cfg"], r["point"], NAMES[(th, r["block"])]))
            r["levels"] = h["levels"] if h else probe.get((r["cfg"], r["point"], r["theta"], r["block"]), "")
            r["operator_complexity"] = h["operator_complexity"] if h else ""
            r["converged_all_rhs"] = "yes" if all(int(v) > 0 for v in r["reasons"].split("/")) else "NO"
            rows.append(r)
    for r in load(os.path.join(R, "rowsum_probe.csv")):
        rows.append(dict(r, source=f"rowsum_probe(max_row_sum={r['max_row_sum']})", its="",
                         converged_all_rhs="yes" if all(int(v) > 0 for v in r["reasons"].split("/")) else "NO"))
    keys = ["source", "cfg", "point", "eps", "nx", "theta", "block", "Pe_cell", "adv_ratio", "levels",
            "operator_complexity", "its", "max_its", "cap_hits", "reasons", "converged_all_rhs", "setup_s",
            "solve_s", "maxrss_kb", "petsc_options"]
    with open(os.path.join(LP, "t5b_hypre_block_screen.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print("| config | points | all blocks/RHS converged | max its | worst | 1-level hierarchies |")
    print("|---|---|---|---|---|---|")
    for c in HCFG:
        rr = [r for r in rows if r["cfg"] == c and r["source"] == "screen"]
        if not rr:
            continue
        ok = all(r["converged_all_rhs"] == "yes" for r in rr)
        worst = max(rr, key=lambda r: int(r["max_its"]))
        one = sum(1 for r in rr if r["levels"] == "1" or r["levels"] == 1)
        print(f"| {c} | {' '.join(dict.fromkeys(r['point'] for r in rr))} | {'yes' if ok else 'NO'} |"
              f" {worst['max_its']} | {worst['point']} theta={worst['theta']} {worst['block']} | {one}/{len(rr)} |")
    return rows


def kkt_table():
    kkt = []
    for c in ("hypre_default", "hypre_air1", "hypre_air1_mrs1"):
        kkt += [dict(r, source="t5b") for r in load(os.path.join(R, f"kkt_{c}.csv"))]
    for c in ("lu_petsc", "asm4o1_ilu1"):
        kkt += [dict(r, source="t5") for r in load(os.path.join(R5, f"kkt_{c}.csv"))]
    ref = {(r["scheme"], r["eps"], r["nx"]): r for r in kkt if r["cfg"] == "lu_petsc"}
    out = []
    print("\n| config | scheme | eps | nx | reason | outer | sweep failures | spatial its mean/max (right,left) |"
          " true res | solve wall s | max RSS MB | max|dy| vs LU | max|dp| vs LU | J_track rel | J_reg rel |")
    print("|" + "---|" * 15)
    for r in kkt:
        key = (r["scheme"], r["eps"], r["nx"])
        d = os.path.join(R if r["source"] == "t5b" else R5,
                         f"snap_{r['cfg']}_{r['scheme'].lower()}_eps{r['eps'][2:]}_nx{r['nx']}.csv")
        b = os.path.join(R5, f"snap_lu_petsc_{r['scheme'].lower()}_eps{r['eps'][2:]}_nx{r['nx']}.csv")
        dy = dp = djt = djr = ""
        if r["cfg"] != "lu_petsc" and key in ref:
            a, bb = snap(d)[2], snap(b)[2]
            dy = f"{np.nanmax(np.abs(a['y'] - bb['y'])):.1e}"
            dp = f"{np.nanmax(np.abs(a['p_LP'] - bb['p_LP'])):.1e}"
            rf = ref[key]
            djt = f"{abs(float(r['J_track']) - float(rf['J_track'])) / abs(float(rf['J_track'])):.1e}"
            djr = (f"{abs(float(r['J_reg']) - float(rf['J_reg'])) / abs(float(rf['J_reg'])):.1e}"
                   if float(r["J_reg"]) else "J_reg=0")
        solves = int(r["right_solves"]) + int(r["left_solves"])
        its = int(r["right_its"]) + int(r["left_its"])
        mx = max(int(r["right_max"]), int(r["left_max"]))
        genuine = int(r["reason"]) > 0 and int(r["sweep_failures"]) == 0 and float(r["true_residual"]) < 1e-8
        rec = dict(source=r["source"], cfg=r["cfg"], scheme=r["scheme"], eps=r["eps"], nx=r["nx"],
                   reason=r["reason"], outer_its=r["outer_its"], genuinely_converged="yes" if genuine else "NO",
                   true_residual=r["true_residual"], state_rel=r["state_rel"], adjoint_rel=r["adjoint_rel"],
                   march_rel=r.get("march_rel", ""), sweep_failures=r["sweep_failures"],
                   right_solves=r["right_solves"], right_its=r["right_its"], right_max=r["right_max"],
                   left_solves=r["left_solves"], left_its=r["left_its"], left_max=r["left_max"],
                   ysplit_solves=r["ysplit_solves"], ysplit_its=r["ysplit_its"], ysplit_max=r["ysplit_max"],
                   schur_ksp_its=r["schur_ksp_its"], spatial_its_mean_right_left=f"{its / max(1, solves):.2f}",
                   solve_wall=r["solve_wall"], wall_s=r["wall_s"], maxrss_kb=r["maxrss_kb"],
                   J_track=r["J_track"], J_reg=r["J_reg"], max_dy_vs_lu=dy, max_dp_vs_lu=dp,
                   J_track_rel_vs_lu=djt, J_reg_rel_vs_lu=djr, outer_history=r.get("outer_history", ""),
                   petsc_options=r["petsc_options"])
        out.append(rec)
        print(f"| {rec['cfg']} | {rec['scheme']} | {rec['eps']} | {rec['nx']} | {rec['reason']} | {rec['outer_its']} |"
              f" {rec['sweep_failures']} | {rec['spatial_its_mean_right_left']}/{mx} | {rec['true_residual']} |"
              f" {rec['solve_wall']} | {int(rec['maxrss_kb']) / 1024:.0f} | {dy} | {dp} | {djt} | {djr} |")
    with open(os.path.join(LP, "t5b_hypre_kkt.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    return out


def figures(blk, kkt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t5 = load(os.path.join(R5, "screen.csv")) + load(os.path.join(R5, "scale.csv"))
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    styles = (("hypre_default", "x", "#7f7f7f"), ("hypre_ns", "+", "#bcbd22"), ("hypre_air1", "o", "#ff7f0e"),
              ("hypre_air1_mrs1", "s", "#1f77b4"))
    for ax, th, lab in ((axs[0], "1", "theta = 1 (BE)"), (axs[1], "0.5", "theta = 1/2 (CN)")):
        for c, mk, col in styles:
            pts = {}
            for r in blk:
                if r["cfg"] == c and r["source"] == "screen" and r["theta"] == th:
                    its = int(r["max_its"]) if r["converged_all_rhs"] == "yes" else 500
                    k = float(r["Pe_cell"]), r["point"]
                    pts[k] = max(pts.get(k, 0), its)
            if pts:
                ks = sorted(pts)
                ax.plot([k[0] for k in ks], [pts[k] for k in ks], marker=mk, ls="none", ms=7, color=col, label=c)
        for c, mk, col in (("asm4o1_ilu1", "^", "#2ca02c"), ("lu_petsc", "*", "k")):
            pts = {}
            for r in t5:
                if r["cfg"] == c and r["theta"] == th:
                    k = float(r["Pe_cell"]), r["point"]
                    pts[k] = max(pts.get(k, 0), int(r["max_its"]))
            ks = sorted(pts)
            ax.plot([k[0] for k in ks], [pts[k] for k in ks], marker=mk, ls="none", ms=6, color=col,
                    mfc="none", label=f"{c} (T5)")
        ax.axhline(500, color="k", ls="--", lw=0.6)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Pe_cell"); ax.set_title(lab, fontsize=9)
        ax.grid(True, which="both", lw=0.3)
    axs[0].set_ylabel("max GMRES its (E/right/left, 4 RHS)")
    axs[1].legend(fontsize=7)
    fig.suptitle("T5b spatial block screen: hypre vs T5 references (AIR breakdown plotted at 500)", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(LP, "figures", "t5b_iterations_vs_peclet.png"), dpi=110)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for c, mk, col in (("lu_petsc", "*", "k"), ("asm4o1_ilu1", "^", "#2ca02c"), ("hypre_default", "x", "#7f7f7f"),
                       ("hypre_air1_mrs1", "s", "#1f77b4")):
        for e, ls in (("1/100", "-"), ("1/500", ":")):
            pts = sorted((int(r["nx"]), float(r["solve_wall"])) for r in kkt
                         if r["cfg"] == c and r["eps"] == e and r["scheme"] == "CN"
                         and r["genuinely_converged"] == "yes")
            if pts:
                ax.plot([(2 * p[0] + 1) ** 2 for p in pts], [p[1] for p in pts], marker=mk, ls=ls, color=col,
                        lw=0.9, label=f"{c} eps {e}")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("spatial DOFs (CN, Nt = 2 nx)")
    ax.set_ylabel("KKT solve wall s (np = 1)"); ax.grid(True, which="both", lw=0.3); ax.legend(fontsize=6)
    ax.set_title("T5b KKT solve time by spatial backend (converged runs only)", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(LP, "figures", "t5b_time_vs_backend.png"), dpi=110)


if __name__ == "__main__":
    blk = block_table()
    kkt = kkt_table()
    figures(blk, kkt)
