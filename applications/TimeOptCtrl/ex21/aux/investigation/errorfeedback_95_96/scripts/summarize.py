#!/usr/bin/env python3
"""ErrorFeedback 9.5 / 9.6 AAO campaign summary.

Reads ../logs/{ic,sweep}_ef{95,96}_*.log and writes
  ../../results/errorfeedback_{95,96}_aao_runs.csv   (one row per run)
and prints: IC sensitivity of the tail, the alpha_u sweep on every mesh,
and (9.5) the log-log interpolated alpha_u at which the converged tail sup on
[5,20] equals the published E5 = 5.15e-4 -- quoted as "AAO regularisation
yielding the same published tail scale", never as an identification of the
source's alpha/beta.  9.6: the tail floor vs the published E5 = 5.220e-2."""
import csv, glob, math, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "..", "logs")
RES = os.path.join(HERE, "..", "..", "results")
E5 = {"95": 5.15e-4, "96": 5.220e-2}
TABLE = {"95": [1.25e-2, 3.93e-3, 1.70e-3, 8.76e-4, 5.15e-4],
         "96": [6.355e-2, 5.496e-2, 5.338e-2, 5.264e-2, 5.220e-2]}


def kv(line):
    return dict(re.findall(r"(\w+)=(\S+)", line))


def load():
    rows = []
    for path in sorted(glob.glob(os.path.join(LOG, "*_ef9[56]_*.log"))):
        s = open(path).read()
        m = re.search(r"^EX21_EF(9[56])_RESULT (.*)$", s, re.M)
        if not m:
            continue
        ex = m.group(1)
        cfg = kv(re.search(rf"^EX21_EF{ex}_CONFIG (.*)$", s, re.M).group(1))
        res = kv(m.group(2))
        tail = kv(re.search(rf"^EX21_EF{ex}_TAIL (.*)$", s, re.M).group(1))
        st = re.search(r"max\|dy\|/max\|y\|=(\S+) max\|y\| on the Dirichlet boundary=(\S+)", s)
        wall = re.search(r"WALL (\S+)", s)
        rows.append(dict(run=os.path.basename(path)[:-4], example=ex, alpha_u=float(cfg["alpha_u"]),
                         initial=cfg["initial"], nx=int(cfg["nx"]), nz=int(cfg["nz"]), steps=int(cfg["steps"]),
                         scheme=cfg["scheme"], Pe_cell=float(cfg["Pe_cell"]),
                         sup_2_20=float(tail["sup_2_20"]), sup_5_20=float(tail["sup_5_20"]),
                         sup_10_20=float(tail["sup_10_20"]), rms_5_20=float(tail["rms_5_20"]),
                         J_track=float(res["J_track"]), J_reg=float(res["J_reg"]), u_L2=float(res["u_L2"]),
                         outer_its=int(res["outer_its"]), true_residual=float(res["true_residual"]),
                         sweep_failures=int(res["sweep_failures"]), reason=int(res["reason"]),
                         march=float(st.group(1)), trace=float(st.group(2)),
                         wall_s=float(wall.group(1)) if wall else float("nan")))
    return rows


def interp_alpha(pts, target):
    """log-log linear interpolation of alpha(target) on a monotone sweep."""
    pts = sorted(pts, reverse=True)       # decreasing alpha
    for (a0, e0), (a1, e1) in zip(pts, pts[1:]):
        if (e0 - target) * (e1 - target) <= 0 and e0 != e1:
            w = (math.log(target) - math.log(e0)) / (math.log(e1) - math.log(e0))
            return math.exp(math.log(a0) + w * (math.log(a1) - math.log(a0))), (a0, a1)
    return None, None


def main():
    rows = load()
    for ex in ("95", "96"):
        R = [r for r in rows if r["example"] == ex]
        if not R:
            continue
        keys = list(R[0].keys())
        with open(os.path.join(RES, f"errorfeedback_{ex}_aao_runs.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(R)
        print(f"===== ErrorFeedback 9.{ex[1]}   published E1..E5 = {TABLE[ex]}")
        print("-- IC sensitivity (tail sup on [5,20], [10,20]; transient sup on [2,20])")
        for a in sorted({r["alpha_u"] for r in R if r["run"].startswith("ic_")}, reverse=True):
            g = [r for r in R if r["run"].startswith("ic_") and r["alpha_u"] == a]
            base = [r for r in g if r["initial"] == "zero"]
            for r in sorted(g, key=lambda r: r["initial"]):
                d = abs(r["sup_5_20"] - base[0]["sup_5_20"]) / base[0]["sup_5_20"] if base else float("nan")
                print(f"  a={a:.0e} {r['initial']:7s} sup5={r['sup_5_20']:.10e} sup10={r['sup_10_20']:.10e}"
                      f" sup2={r['sup_2_20']:.6e} rel_vs_zero={d:.2e}")
        print("-- alpha_u sweep (z0 = sinsin)")
        meshes = sorted({(r["nx"], r["nz"], r["steps"], r["scheme"]) for r in R if r["run"].startswith("sweep_")})
        for m in meshes:
            g = sorted([r for r in R if r["run"].startswith("sweep_") and (r["nx"], r["nz"], r["steps"], r["scheme"]) == m],
                       key=lambda r: -r["alpha_u"])
            print(f"  mesh nx={m[0]} nz={m[1]} nt={m[2]} {m[3]} Pe_cell={g[0]['Pe_cell']:.3f}")
            for r in g:
                print(f"    a={r['alpha_u']:.0e} sup5={r['sup_5_20']:.6e} rms5={r['rms_5_20']:.6e} u={r['u_L2']:.4e}"
                      f" Jt={r['J_track']:.4e} Jr={r['J_reg']:.4e} its={r['outer_its']} res={r['true_residual']:.1e}"
                      f" fail={r['sweep_failures']} march={r['march']:.1e} trace={r['trace']:.0e} wall={r['wall_s']:.0f}")
            a, br = interp_alpha([(r["alpha_u"], r["sup_5_20"]) for r in g], E5[ex])
            if a:
                print(f"    -> alpha_u with sup5 = E5 = {E5[ex]:.3e}: {a:.3e} (bracket {br[0]:.0e}..{br[1]:.0e})")
            else:
                print(f"    -> E5 = {E5[ex]:.3e} not bracketed: min sup5 = {min(r['sup_5_20'] for r in g):.6e}")


if __name__ == "__main__":
    main()
