#!/usr/bin/env python3
"""T11N solver table -> ../../t11n_solver.csv.  usage: t11n_solver_table.py <json-dir>
Per run (logs/solver/<tag>.log): reason, outer its, true residual, sweep failures,
spatial KSP its per solve (y split / right / left: total and max), wall, MaxRSS;
fields and J against the LU run of the same scheme; refusal messages."""
import csv, json, os, re, sys
import numpy as np
J, D = sys.argv[1], os.path.dirname(os.path.abspath(__file__)) + "/.."
def g(pat, s, cast=str):
    m = re.search(pat, s); return cast(m.group(1)) if m else ""
rows = []
for s in ("be", "cn"):
    ref = None
    for bk in ("sweep_lu", "sweep_gamg", "sweep_asmilu", "additive"):
        if bk == "additive" and s == "cn": continue   # refused in 3D (reported below)
        tag = f"{s}_{bk}"; p = f"{D}/logs/solver/{tag}.log"
        if not os.path.exists(p): continue
        L = open(p).read(); r = {"scheme": s.upper(), "solver": bk}
        r["reason"] = g(r"EX20_PAR_SOLVE .*? reason=(-?\d+)", L); r["outer_its"] = g(r"outer_iterations=(\d+)", L)
        r["true_residual"] = g(r"EX20_PAR_SOLVE .*? true_residual=(\S+)", L); r["sweep_failures"] = g(r"sweep_failures=(\d+)", L)
        r["rejected"] = "yes" if "EX21_SOLVE_REJECTED" in L else ""
        for k in ("y", "right", "left"):
            m = re.search(rf" {k}=(\d+)/(\d+)/(\S+)/(\d+)", L)
            r[f"{k}_solves"], r[f"{k}_its"], r[f"{k}_its_max"] = (m.group(1), m.group(2), m.group(4)) if m else ("", "", "")
        r["wall_s"] = g(r"wall=(\S+)", L); r["maxrss_MB"] = f"{int(g(r'MAXRSS=(\d+)', L) or 0) / 1024:.0f}"
        try:
            d = json.load(open(f"{J}/{tag}.json"))
            f = {k: np.array([x["values"] for x in d[k]]) for k in ("state", "adjoint", "control_full")}
            if bk == "sweep_lu": ref = (d, f)
            if ref:
                r["fields_maxrel_vs_lu"] = f"{max(np.abs(f[k] - ref[1][k]).max() / np.abs(ref[1][k]).max() for k in f):.2e}"
                r["J_rel_vs_lu"] = f"{abs(d['functional']['J_total'] - ref[0]['functional']['J_total']) / abs(ref[0]['functional']['J_total']):.2e}"
            r["J_total"] = f"{d['functional']['J_total']:.10e}"
        except Exception as e:
            r["J_total"] = f"no json ({e.__class__.__name__})"
        rows.append(r); print(" ".join(f"{k}={v}" for k, v in r.items()))
for tag in ("cn_additive", "be_matching", "be_boundary_reduced", "heat_boundary_reduced"):
    p = f"{D}/logs/solver/{tag}.log"
    if os.path.exists(p):
        r = {"scheme": tag.split("_")[0].upper(), "solver": tag, "reason": "REFUSED: " + g(r"what\(\):\s+(.*)", open(p).read())}
        rows.append(r); print(" ".join(f"{k}={v}" for k, v in r.items()))
keys = list(dict.fromkeys(k for r in rows for k in r))
with open(D + "/../t11n_solver.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
