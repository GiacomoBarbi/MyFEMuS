#!/usr/bin/env python3
"""logs/solver/*.log -> ../../t12_solver.csv (mesh_backend_scheme)."""
import csv, glob, os, re
D = os.path.dirname(os.path.abspath(__file__)) + "/.."
rows = []
for f in sorted(glob.glob(D + "/logs/solver/*.log")):
    s = open(f).read(); mesh, backend, scheme = os.path.basename(f)[:-4].split("_")
    R = re.search(r"EX21_EXACT3?D?_RESULT .*", s).group(0)
    S = (re.search(r"EX21_EXACT3?D?_SOLVER_STATS .*", s) or re.search(r"EX20_SOLVERSTATS .*", s)).group(0)
    g = lambda p, t: (re.search(p, t).group(1) if re.search(p, t) else "")
    r = {"mesh": mesh, "backend": backend, "scheme": scheme.upper(), "reason": g(r" reason=(\S+)", R),
         "outer_its": g(r" outer_its=(\S+)", R), "true_residual": g(r" true_residual=(\S+)", R),
         "y_L2L2": g(r" y_L2L2=(\S+)", R), "p_L2L2": g(r" p_L2L2=(\S+)", R), "u_L2L2": g(r" u_L2L2=(\S+)", R)}
    for k in ("right", "left", "ay", "ysplit", "schur_ksp"):
        r[f"{k}_its"] = g(rf" {k}_its=(\d+)", S); r[f"{k}_max"] = g(rf" {k}_max=(\d+)", S)
    r["sweep_failures"] = g(r"sweep_failures=(\d+)", s)
    mr = re.search(r"MAXRSS=(\d+) wall=(\S+)", s); r["maxrss_MB"], r["wall_s"] = int(mr.group(1)) // 1024, mr.group(2)
    rows.append(r)
lu = {(r["mesh"], r["scheme"]): r for r in rows if r["backend"] == "lu"}
for r in rows:
    q = lu[(r["mesh"], r["scheme"])]
    r["rel_diff_err_y_vs_lu"] = f"{abs(float(r['y_L2L2']) - float(q['y_L2L2'])) / float(q['y_L2L2']):.1e}"
with open(D + "/../t12_solver.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows:
    print(" ".join(f"{k}={r[k]}" for k in ("mesh", "backend", "scheme", "reason", "outer_its", "true_residual", "y_L2L2",
                                          "right_max", "left_max", "ysplit_max", "sweep_failures", "maxrss_MB", "wall_s", "rel_diff_err_y_vs_lu")))
