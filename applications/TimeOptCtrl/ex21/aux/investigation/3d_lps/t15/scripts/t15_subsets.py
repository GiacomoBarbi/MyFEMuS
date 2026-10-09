#!/usr/bin/env python3
"""T15 views on the gate tables: ../../t15_variable_wind.csv (every gate row with harmonic-3d / affine-div-3d, all
kinds, with the variable-wind identities and the LPS activity) and ../../t15_multiface.csv (every Neumann / Dirichlet
row on a face UNION: opp, adj, nu = non-unit box zmin+xmin).  Columns common to all kinds plus the kind-specific ones."""
import csv, os
D = os.path.dirname(os.path.abspath(__file__)) + "/../.."
rows = []
for k in ("distributed", "natural", "neumann", "dirichlet"):
    rows += list(csv.DictReader(open(f"{D}/t15_gates_{k}.csv")))
cols = []
for r in rows:
    cols += [c for c in r if c not in cols]
for name, keep in (("t15_variable_wind.csv", lambda r: r["wind"] in ("harm", "aff")),
                   ("t15_multiface.csv", lambda r: r["kind"] in ("neu", "dir") and r["group"] in ("opp", "adj", "nu"))):
    sel = [r for r in rows if keep(r)]
    with open(f"{D}/{name}", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, restval=""); w.writeheader(); w.writerows(sel)
    print(name, len(sel), "rows,", sum(r["verdict"] == "PASS" for r in sel), "PASS")
