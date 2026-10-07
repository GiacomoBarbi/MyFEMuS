#!/usr/bin/env python3
"""logs/iters/*.log -> dirichlet_core/t8d_core_iterations.csv"""
import csv, glob, os, re
D = os.path.dirname(os.path.abspath(__file__)) + "/.."
rows = []
def g(p, s):
    m = re.search(p, s); return m.group(1) if m else ""
for f in glob.glob(D + "/logs/iters/*.log"):
    s = open(f).read(); tag = os.path.basename(f)[:-4]
    ladder, conv, pc, val = tag.split("_", 3)
    rows.append({"ladder": ladder, "convection": conv, "pc": pc, "value": val,
                 "scheme": g(r"time_scheme=(\S+)", s), "mesh": g(r"mesh=(\S+)", s), "N": g(r" N=(\d+)", s),
                 "m": g(r"EX21_T8D_CORE_STATS .*? m=(\d+)", s), "core_solves": g(r"core_solves=(\d+)", s),
                 "core_its_mean": g(r"core_its_mean=(\S+)", s), "core_its_max": g(r"core_its_max=(\d+)", s),
                 "core_failures": g(r"core_failures=(\d+)", s), "outer": g(r"outer_iterations=(\d+)", s),
                 "reason": g(r"EX20_PAR_SOLVE .*? reason=(\S+)", s), "true_residual": g(r"true_residual=(\S+)", s),
                 "core_seconds": g(r"core_seconds=(\S+)", s)})
order = {"nx": 0, "nt": 1, "alpha": 2, "ell": 3, "gamma": 4, "be": 5}
rows.sort(key=lambda r: (order[r["ladder"]], r["convection"], r["pc"], float(r["value"]) if r["value"] != "base" else 0))
with open(D + "/../t8d_core_iterations.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
cur = None
for r in rows:
    k = (r["ladder"], r["convection"], r["pc"])
    if k != cur:
        print(f"\n{r['ladder']:<6} conv={r['convection']:<4} pc={r['pc']:<5}:", end=""); cur = k
    print(f"  {r['value']}:{r['core_its_max']}/{r['outer']}" + ("!" if r["core_failures"] not in ("0", "") or r["reason"].startswith("-") else ""), end="")
print()
