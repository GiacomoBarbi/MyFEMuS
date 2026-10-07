#!/usr/bin/env python3
"""Build investigation/t7f_multizone.csv and t7f_multisource.csv (long format)
from the T7F logs/results.  t7f_natural_bc.csv is written by t7f_ladder_table.py."""
import csv, os, re
D = os.path.dirname(os.path.abspath(__file__)) + "/.."
OUT = D + "/.."
def gates(prefix):
    rows = []
    for r in csv.DictReader(open(D + "/results/gates.csv")):
        if not r["case"].startswith(prefix): continue
        for k, crit in (("G2_transpose", "<= 1e-15"), ("G2_neg_ex20", "> 1e-5 (must fail)"), ("G8b_wrongbc_transpose", "> 1e-5 (must fail)"),
                        ("G1N_boundary_identity", "<= 1e-12"), ("hess_shell", "<= 1e-12"), ("fd_shell", "<= 1e-10"),
                        ("fd_ex20_wrong", "> 1e-3 (must fail)"), ("fd_wrongbc", "> 1e-3 (must fail)"), ("G4_true_res", "<= 1e-10"),
                        ("G4_gradJ", "<= 1e-10"), ("G4_forward_march", "<= 1e-12")):
            v = r[k]
            if not v: continue
            x = float(v); thr = float(re.search(r"[\d.e+-]+", crit.split()[1]).group(0))
            ok = x > thr if crit.startswith(">") else x <= thr
            rows.append({"section": "gates", "case": r["case"], "scheme": r["scheme"], "check": k, "value": v, "criterion": crit, "verdict": "PASS" if ok else "FAIL"})
    return rows
def loglines(path, tag, section):
    rows = []
    for l in open(path):
        d = re.match(rf"{tag} (.*?)\s+max\|diff\|=(\S+) expected (\S+) (PASS|FAIL)", l.strip())
        if d:
            rows.append({"section": section, "case": "", "scheme": "", "check": d.group(1).strip(), "value": d.group(2),
                         "criterion": "expected " + d.group(3), "verdict": d.group(4)})
            continue
        m = re.match(rf"{tag} (.*?)\s+(\S+)\s+(?:ref \S+ )?(PASS|FAIL)(?: \((.*)\))?\s*$", l.strip())
        if m:
            rows.append({"section": section, "case": "", "scheme": "", "check": m.group(1).strip(), "value": m.group(2), "criterion": m.group(4) or "", "verdict": m.group(3)})
    return rows
def selfconv(path, section):
    rows = []
    for r in csv.DictReader(open(path)):
        for k in ("y", "u", "p"):
            rows.append({"section": section, "case": f"{r['mode']} nx={r['nx']} nt={r['nt']} ref={r['ref']}", "scheme": r["scheme"],
                         "check": f"{k} error vs fine reference (order)", "value": r[k], "criterion": r[k + "_order"], "verdict": ""})
    return rows
def write(name, rows):
    with open(f"{OUT}/{name}", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["section", "case", "scheme", "check", "value", "criterion", "verdict"]); w.writeheader(); w.writerows(rows)
    print(name, len(rows), "rows,", sum(r["verdict"] == "FAIL" for r in rows), "FAIL")
mz = loglines(D + "/logs/masks.log", "T7F_MASK", "masks")
for l in open(D + "/logs/region_index.log"):
    if l.startswith("T7F_REGION ") :
        mz.append({"section": "lowest-index rule", "case": "", "scheme": "", "check": l.split(":")[0][11:], "value": l.split(":", 1)[1].strip(), "criterion": "overlap cells carry index 1", "verdict": "PASS" if "matches_rule=True" in l else "FAIL"})
mz += gates("mz") + gates("src2_mz") + selfconv(D + "/results/selfconv_mz.csv", "self-convergence")
write("t7f_multizone.csv", mz)
ms = loglines(D + "/logs/multisource.log", "T7F_SRC", "superposition/overlap/MPI") + gates("src2")
ms += [r for r in loglines(D + "/logs/masks.log", "T7F_MASK", "masks") if "source" in r["check"]]
write("t7f_multisource.csv", ms)
