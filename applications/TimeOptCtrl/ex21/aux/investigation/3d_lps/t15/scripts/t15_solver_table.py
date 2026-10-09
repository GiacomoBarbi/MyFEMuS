#!/usr/bin/env python3
"""T15 solver table -> ../../t15_solver.csv.  usage: t15_solver_table.py <json-dir>
Per run (t15_solver.sh): exit (124 = wall cap 600 s reached), reason, outer its, true residual, sweep failures,
spatial KSP its (y split / right / left: total, max), Dirichlet core its, wall, MaxRSS; fields (max rel.) and J
against the reference of the same case, scheme AND stabilisation (LU sweep; Dirichlet: the dense core)."""
import csv, json, os, re, sys
import numpy as np
J, D = sys.argv[1], os.path.dirname(os.path.abspath(__file__)) + "/.."
g = lambda p, s: (re.search(p, s).group(1) if re.search(p, s) else "")
rows = []
runs = open(D + "/logs/t15_solver_runs.log").read()
for case in ("dA", "dH", "nA", "mA", "rA", "rH"):
    for st in ("gal", "lps"):
        for s in ("be", "cn"):
            bks = ("lu", "gamg", "asmilu") if not case.startswith("r") else ("dense", "iterative")
            ref = None
            for bk in bks:
                tag = f"{case}_{st}_{s}_{bk}"; p = f"{D}/logs/solver/{tag}.log"
                if not os.path.exists(p):
                    continue
                L = open(p).read()
                r = {"case": case, "stabilisation": "Galerkin" if st == "gal" else "LPS-P0", "scheme": s.upper(), "backend": bk,
                     "exit": g(rf"{tag} exit=(\d+)", runs), "reason": g(r"EX20_PAR_SOLVE .*? reason=(-?\d+)", L),
                     "outer_its": g(r"outer_iterations=(\d+)", L), "true_residual": g(r"EX20_PAR_SOLVE .*? true_residual=(\S+)", L),
                     "sweep_failures": g(r"sweep_failures=(\d+)", L), "core_its_max": g(r"core_its_max=(\d+)", L)}
                for k in ("y", "right", "left"):
                    m = re.search(rf" {k}=(\d+)/(\d+)/(\S+)/(\d+)", L)
                    r[f"{k}_its"], r[f"{k}_its_max"] = (m.group(2), m.group(4)) if m else ("", "")
                r["wall_s"] = g(r"wall=(\S+)", L) or ("600 (cap)" if r["exit"] == "124" else "")
                r["maxrss_MB"] = f"{int(g(r'MAXRSS=(\d+)', L) or 0) / 1024:.0f}"
                r["lps_active_fraction"] = g(r"EX21_LPS3D active_fraction=(\S+)", L)
                try:
                    d = json.load(open(f"{J}/{tag}.json"))
                    f = {k: np.array([x["values"] for x in d[k]]) for k in ("state", "adjoint", "control_full")}
                    if ref is None:
                        ref = (d, f)
                    r["fields_vs_ref"] = f"{max(np.abs(f[k] - ref[1][k]).max() / max(np.abs(ref[1][k]).max(), 1e-300) for k in f):.1e}"
                    r["J_vs_ref"] = f"{abs(d['functional']['J_total'] - ref[0]['functional']['J_total']) / abs(ref[0]['functional']['J_total']):.1e}"
                except Exception as e:
                    r["fields_vs_ref"] = r["J_vs_ref"] = f"n/a ({type(e).__name__})"
                ok = r["exit"] == "0" and r["reason"] not in ("", None) and int(r["reason"]) > 0 and r["sweep_failures"] in ("0", "")
                r["converged"] = "yes" if ok else "NO"
                rows.append(r)
                print(" ".join(f"{k}={v}" for k, v in r.items()))
with open(D + "/../t15_solver.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
