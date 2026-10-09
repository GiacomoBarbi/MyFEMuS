#!/usr/bin/env python3
"""logs/t13_permutation.log -> ../../t13_permutation.csv; PASS iff every comparison is IDENTICAL."""
import csv, os, re, sys
D = os.path.dirname(os.path.abspath(__file__)) + "/.."
rows = []
for line in open(D + "/logs/t13_permutation.log"):
    if not line.startswith("PERM "): continue
    p = line.split()
    kv = dict(x.split("=", 1) for x in p[3:-1])
    rows.append({"kind": "Neumann" if p[1] == "N" else "Dirichlet", "set": p[2], "order": kv["order"], "reference_order": kv["vs"],
                 "compared": kv["what"], "size": kv.get("lines", kv.get("files", "")), "differing_files": kv.get("differing", ""),
                 "verdict": "PASS" if p[-1] == "IDENTICAL" else "FAIL", "result": p[-1]})
with open(D + "/../t13_permutation.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
ok = all(r["verdict"] == "PASS" for r in rows)
print("T13_PERMUTATION_RESULT", "PASS" if ok else "FAIL", f"({sum(r['verdict'] == 'PASS' for r in rows)}/{len(rows)} byte-identical)")
sys.exit(0 if ok else 1)
