#!/usr/bin/env python3
"""Analyse t7f_multisource.sh: superposition / removal / overlap / MPI."""
import json, os, sys
import numpy as np
J = sys.argv[1]
ok_all = True
def L(tag):
    """fields reordered by (z, x): the JSON node order follows the MPI partition."""
    d = json.load(open(f"{J}/{tag}.json"))
    xy = np.round(np.array(d["spatial_coordinates"]), 12)
    order = np.lexsort((xy[:, 0], xy[:, 1]))
    f = {k: np.array([s["values"] for s in d[k]])[:, order] for k in ("state", "adjoint", "control_full")}
    return d, f
def rel(a, b):
    return np.abs(a - b).max() / max(np.abs(b).max(), 1e-300)
def rep(name, v, tol, larger=False):
    global ok_all
    ok = v > tol if larger else v <= tol; ok_all &= ok
    print(f"T7F_SRC {name:<70s} {v:.3e} {'PASS' if ok else 'FAIL'} ({'>' if larger else '<='} {tol:.0e})")
rows = []
for s in ("be", "cn"):
    z = {t: L(f"{t}_{s}") for t in ("z0", "z0box", "z1", "z2", "z12", "z12x", "zwhole", "ovAB", "ovU")}
    for np_ in (2, 4):
        z[f"z12_np{np_}"] = L(f"z12_{s}_np{np_}")
    S = s.upper()
    for k, lab in (("state", "y"), ("adjoint", "p"), ("control_full", "u")):
        f0, f1, f2, f12, f12x = (z[t][1][k] for t in ("z0", "z1", "z2", "z12", "z12x"))
        d1, d2, d12 = f1 - f0, f2 - f0, f12 - f0
        rep(f"{S} {lab}: zero-amplitude baseline independent of the boxes", rel(z["z0box"][1][k], f0), 1e-14)
        rep(f"{S} {lab}: region 1 contributes  max|d1|/max|d12|", np.abs(d1).max() / np.abs(d12).max(), 1e-2, larger=True)
        rep(f"{S} {lab}: region 2 contributes  max|d2|/max|d12|", np.abs(d2).max() / np.abs(d12).max(), 1e-2, larger=True)
        rep(f"{S} {lab}: contributions are distinct  max|d1-d2|/max|d12|", np.abs(d1 - d2).max() / np.abs(d12).max(), 1e-2, larger=True)
        rep(f"{S} {lab}: superposition z12 - z0 = (z1 - z0) + (z2 - z0)", rel(d1 + d2, d12), 1e-10)
        rep(f"{S} {lab}: removing region 1: z12 - z2 = z1 - z0", rel(f12 - f2, d1), 1e-10)
        rep(f"{S} {lab}: removing region 2: z12 - z1 = z2 - z0", rel(f12 - f1, d2), 1e-10)
        rep(f"{S} {lab}: amplitude 2.5: z12x - z0 = 2.5 (z12 - z0)", rel(f12x - f0, 2.5 * d12), 1e-10)
        rep(f"{S} {lab}: two regions differ from whole-domain source", rel(z["zwhole"][1][k], f12), 1e-2, larger=True)
        rep(f"{S} {lab}: overlapping boxes == single union box (bitwise)", np.abs(z["ovAB"][1][k] - z["ovU"][1][k]).max(), 0.0)
        for np_ in (2, 4):
            rep(f"{S} {lab}: MPI np={np_} vs np=1", rel(z[f"z12_np{np_}"][1][k], f12), 1e-12)
    for t in ("z0", "z1", "z2", "z12", "z12x", "ovAB", "ovU", "z12_np2", "z12_np4"):
        d = z[t][0]
        rows.append({"scheme": S, "run": t, "reason": d["solve"]["reason"], "outer": d["solve"]["iterations"],
                     "true_residual": d["solve"]["true_residual"], "state_kkt_rel": d["kkt"]["state_kkt_residual_rel"],
                     "adjoint_kkt_rel": d["kkt"]["adjoint_kkt_residual_rel"],
                     "stationarity_rel": d["kkt"]["stationarity_consistency_residual_rel"],
                     "J_total": d["functional"]["J_total"]})
    for t in ("z12", "z12_np2", "z12_np4"):
        rep(f"{S} true KKT residual {t}", z[t][0]["solve"]["true_residual"], 1e-10)
    js = [z[t][0]["functional"]["J_total"] for t in ("z12", "z12_np2", "z12_np4")]
    rep(f"{S} J np=1/2/4 max relative spread", max(abs(j - js[0]) for j in js) / abs(js[0]), 1e-13)
    its = [z[t][0]["solve"]["iterations"] for t in ("z12", "z12_np2", "z12_np4")]
    rep(f"{S} outer iterations np=1/2/4 identical ({its})", float(len(set(its)) - 1), 0)
import csv
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "multisource_runs.csv")
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("T7F_SRC_RESULT", "PASS" if ok_all else "FAIL")
