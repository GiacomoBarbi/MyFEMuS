#!/usr/bin/env python3
import csv, json, os, re, sys
import numpy as np
J, D = sys.argv[1], os.path.dirname(os.path.abspath(__file__)) + "/.."
rows = []
def L(tag):
    d = json.load(open(f"{J}/{tag}.json"))
    return d, {k: np.array([s["values"] for s in d[k]]) for k in ("state", "adjoint", "control_full")}
for f in ("MZ", "SRC", "NAT", "NATPE"):
    for s in ("be", "cn"):
        dr, fr = L(f"{f}_{s}_ref")
        for v in ("ref", "sw", "add"):
            path = f"{D}/logs/solver/{f}_{s}_{v}.log"
            if not os.path.exists(path) or not os.path.exists(f"{J}/{f}_{s}_{v}.json"):
                m = re.search(r"T7F_NOTE (.*)", open(path).read()) if os.path.exists(path) else None
                note = m.group(1).split(":")[0] if m else "not run (BE twin stopped by hand, not repeated)"
                rows.append({"feature": f, "scheme": s.upper(), "solver": "ADDITIVE_GAMG", "reason": "", "outer": "", "true_residual": "",
                             "sweep_failures": "", "accepted": "", "J_rel_vs_ref": "", "max_field_rel_vs_ref": "", "wall": "", "Pe_cell": "", "note": note})
                print(f"{f:<6}{s.upper():<3}{'ADDITIVE_GAMG':<28} {note}")
                continue
            log = open(path).read()
            d, fv = L(f"{f}_{s}_{v}")
            g = lambda p: (re.search(p, log).group(1) if re.search(p, log) else "")
            r = {"feature": f, "scheme": s.upper(), "solver": {"ref": "SWEEP+LU", "sw": "SWEEP (GMRES+GAMG spatial)", "add": "ADDITIVE_GAMG"}[v],
                 "reason": d["solve"]["reason"], "outer": d["solve"]["iterations"], "true_residual": d["solve"]["true_residual"],
                 "sweep_failures": g(r"sweep_failures=(\d+)"), "accepted": g(r"accepted=(\d)"),
                 "J_rel_vs_ref": abs(d["functional"]["J_total"] - dr["functional"]["J_total"]) / abs(dr["functional"]["J_total"]),
                 "max_field_rel_vs_ref": max(np.abs(fv[k] - fr[k]).max() / np.abs(fr[k]).max() for k in fr),
                 "wall": g(r"wall=(\S+)"), "Pe_cell": g(r"Pe_cell=(\S+)"), "note": ""}
            rows.append(r)
            print(f"{f:<6}{s.upper():<3}{r['solver']:<28} reason={r['reason']:<3} outer={r['outer']:<4} res={r['true_residual']:.1e} fail={r['sweep_failures']:<2} J_rel={r['J_rel_vs_ref']:.1e} fields={r['max_field_rel_vs_ref']:.1e} wall={r['wall']}")
with open(D + "/../t7f_solver.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
