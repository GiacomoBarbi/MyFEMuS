#!/usr/bin/env python3
"""Compare np = 1/2/4 (fields matched by coordinates; J; residuals; outer history)."""
import csv, json, os, re, sys
import numpy as np
J, D = sys.argv[1], os.path.dirname(os.path.abspath(__file__)) + "/.."
ok_all, rows = True, []
def L(tag):
    d = json.load(open(f"{J}/{tag}.json"))
    xy = np.round(np.array(d["spatial_coordinates"]), 12); o = np.lexsort((xy[:, 0], xy[:, 1]))
    return d, {k: np.array([s["values"] for s in d[k]])[:, o] for k in ("state", "adjoint", "control_full")}
def hist(tag):
    return np.array([float(v) for v in re.findall(r"^\s*\d+ KSP Residual norm (\S+)", open(f"{D}/logs/mpi/{tag}.log").read(), re.M)])
for case in ("nat_be", "nat_cn", "mz_be", "mz_cn"):
    d1, f1 = L(f"{case}_np1"); h1 = hist(f"{case}_np1")
    for np_ in (2, 4):
        d, f = L(f"{case}_np{np_}"); h = hist(f"{case}_np{np_}")
        r = {"case": case, "np": np_}
        for k, lab in (("state", "y"), ("adjoint", "p"), ("control_full", "u")):
            r[f"{lab}_maxrel"] = np.abs(f[k] - f1[k]).max() / np.abs(f1[k]).max()
        r["J_rel"] = abs(d["functional"]["J_total"] - d1["functional"]["J_total"]) / abs(d1["functional"]["J_total"])
        r["true_res_np1"], r["true_res"] = d1["solve"]["true_residual"], d["solve"]["true_residual"]
        r["outer_np1"], r["outer"] = d1["solve"]["iterations"], d["solve"]["iterations"]
        r["hist_len_equal"] = len(h) == len(h1)
        # history difference relative to the INITIAL residual r_0 (the last iterates are at round-off level)
        r["hist_maxrel"] = float(np.max(np.abs(h - h1)) / h1[0]) if len(h) == len(h1) and len(h) else float("nan")
        good = max(r["y_maxrel"], r["p_maxrel"], r["u_maxrel"]) < 1e-12 and r["J_rel"] < 1e-13 and r["outer"] == r["outer_np1"] \
               and r["hist_len_equal"] and r["hist_maxrel"] < 1e-12 and r["true_res"] < 1e-10
        r["verdict"] = "PASS" if good else "FAIL"; ok_all &= good; rows.append(r)
        print(f"T7F_MPI {case:<7} np={np_} y={r['y_maxrel']:.1e} p={r['p_maxrel']:.1e} u={r['u_maxrel']:.1e} J={r['J_rel']:.1e} "
              f"res={r['true_res']:.1e}/{r['true_res_np1']:.1e} outer={r['outer']}/{r['outer_np1']} history={len(h)} its, max |dr_k|/r_0 {r['hist_maxrel']:.1e} {r['verdict']}")
with open(D + "/results/mpi.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("T7F_MPI_RESULT", "PASS" if ok_all else "FAIL")
