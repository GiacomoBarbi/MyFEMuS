#!/usr/bin/env python3
"""T9 solver baseline table -> ../t9_3d_solver.csv.  usage: t9_solver_table.py <json-dir>
Per run (logs/solver/<tag>.log): outer FGMRES its, reason, true residual, sweep
failures, spatial KSP its (y solves / right / left: total and max per solve),
wall, MaxRSS; fields and J against the LU run of the same scheme and nu."""
import csv, json, os, re, sys
import numpy as np
J, D = sys.argv[1], os.path.dirname(os.path.abspath(__file__)) + "/.."
def g(pat, s, cast=str):
    m = re.search(pat, s); return cast(m.group(1)) if m else ""
rows = []
for s in ("be", "cn"):
    for nu in ("0.25", "0.02"):
        ref = None
        for bk in ("lu", "gamg", "asmilu"):
            tag = f"{s}_nu{nu}_{bk}"; p = f"{D}/logs/solver/{tag}.log"
            if not os.path.exists(p): continue
            L = open(p).read(); r = {"scheme": s.upper(), "nu": nu, "Pe_cell": f"{1.346 * (1/8) / (2*float(nu)):.2f}", "backend": bk}
            r["reason"] = g(r"EX20_PAR_SOLVE .*? reason=(-?\d+)", L); r["outer_its"] = g(r"outer_iterations=(\d+)", L)
            r["true_residual"] = g(r"EX20_PAR_SOLVE .*? true_residual=(\S+)", L); r["sweep_failures"] = g(r"sweep_failures=(\d+)", L)
            for k in ("y", "right", "left"):
                m = re.search(rf" {k}=(\d+)/(\d+)/(\S+)/(\d+)", L)
                r[f"{k}_solves"], r[f"{k}_its"], r[f"{k}_its_max"] = (m.group(1), m.group(2), m.group(4)) if m else ("", "", "")
            if "T9_KILLED" in L:
                r["reason"] = "KILLED: no outer result in 22 min (LU 11.8 s); every fieldsplit_y_sweep GMRES+GAMG solve hits its 500-it cap (logs/solver/be_nu0.02_gamg_diag.log)"
            r["wall_s"] = g(r"wall=(\S+)", L); r["maxrss_MB"] = f"{int(g(r'MAXRSS=(\d+)', L) or 0)/1024:.0f}"
            try:
                d = json.load(open(f"{J}/{tag}.json"))
                f = {k: np.array([x["values"] for x in d[k]]) for k in ("state", "adjoint", "control_full")}
                if bk == "lu": ref = (d, f)
                if ref:
                    r["fields_maxrel_vs_lu"] = f"{max(np.abs(f[k]-ref[1][k]).max()/np.abs(ref[1][k]).max() for k in f):.2e}"
                    r["J_rel_vs_lu"] = f"{abs(d['functional']['J_total']-ref[0]['functional']['J_total'])/abs(ref[0]['functional']['J_total']):.2e}"
                r["J_total"] = f"{d['functional']['J_total']:.10e}"
            except Exception as e:
                r["J_total"] = f"no json ({e.__class__.__name__})"
            rows.append(r); print(" ".join(f"{k}={v}" for k, v in r.items()))
p = f"{D}/logs/solver/additive_be_n4.log"
if os.path.exists(p):
    L = open(p).read()
    rows.append({"scheme": "BE", "nu": "0.25", "Pe_cell": f"{1.346 * (1/4) / 0.5:.2f}", "backend": "ADDITIVE_GAMG (default PC, n=4, N=16)",
                 "reason": g(r"EX20_PAR_SOLVE .*? reason=(-?\d+)", L), "outer_its": g(r"outer_iterations=(\d+)", L),
                 "true_residual": g(r"EX20_PAR_SOLVE .*? true_residual=(\S+)", L), "wall_s": g(r"wall=(\S+)", L),
                 "maxrss_MB": f"{int(g(r'MAXRSS=(\d+)', L) or 0)/1024:.0f}"})
p = f"{D}/logs/solver/additive_cn_refused.log"
if os.path.exists(p):
    rows.append({"scheme": "CN", "backend": "ADDITIVE_GAMG (default PC)", "reason": "REFUSED: " + g(r"what\(\):\s+(.*)", open(p).read())})
keys = list(dict.fromkeys(k for r in rows for k in r))
with open(D + "/../t9_3d_solver.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
