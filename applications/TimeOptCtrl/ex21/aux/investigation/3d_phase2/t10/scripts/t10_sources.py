#!/usr/bin/env python3
"""T10 source-region identities -> ../../t10_sources.csv.  usage: t10_sources.py <json-dir>
The discrete optimality system is affine in the source amplitude field, so with
Z = S0 (zero source) and d(X) = X - Z:
  superposition   d(SAB) = d(SA) + d(SB)
  remove-one      SAB - SA = SB - S0     (and SAB - SB = SA - S0)
  amplitude       d(SAB2) = 2 d(SAB)
  overlap         SOV = SUN   (overlapping boxes counted once = explicit union)
  whole domain    SWH = SFB   (empty list = M I_h f = one full-domain box)
  distinguishable |d(SA)|, |d(SB)| > 0, d(SA) != d(SB), and the centroid of
                  |d(SA).y| lies in box A's octant half-space (x < 1/2), of
                  |d(SB).y| in box B's (x > 1/2).
Fields y, p, u: max |difference| / max |reference| over all slots and DOFs.
Also logs/sources_verify: true KKT residual, gradient consistency, forward march."""
import csv, glob, json, os, re, sys
import numpy as np
J = sys.argv[1]; D = os.path.dirname(os.path.abspath(__file__)) + "/.."
K = ("state", "adjoint", "control_full")
def L(tag, s):
    d = json.load(open(f"{J}/{tag}_{s}_n4_nt16.json"))
    return d, {k: np.array([x["values"] for x in d[k]]) for k in K}, np.array(d["spatial_coordinates"])
rows, ok = [], True
def rel(a, b, ref):
    return max(np.abs(a[k] - b[k]).max() / max(np.abs(ref[k]).max(), 1e-300) for k in K)
def rec(s, check, v, tol, note=""):
    global ok
    good = v <= tol if tol is not None else True
    ok &= good
    rows.append({"scheme": s.upper(), "check": check, "value": f"{v:.3e}" if isinstance(v, float) else v,
                 "tolerance": tol, "verdict": "PASS" if good else "FAIL", "note": note})
    print(f"T10_SRC {s.upper()} {check:<55s} {rows[-1]['value']:<12} {'PASS' if good else 'FAIL'} {note}")
for s in ("be", "cn"):
    V = {t: L(t, s) for t in ("S0", "SA", "SB", "SAB", "SAB2", "SOV", "SUN", "SWH", "SFB")}
    F = {t: v[1] for t, v in V.items()}
    d = {t: {k: F[t][k] - F["S0"][k] for k in K} for t in F}
    zero = {k: np.zeros_like(F["S0"][k]) for k in K}
    sumAB = {k: d["SA"][k] + d["SB"][k] for k in K}
    tol = 1e-9
    rec(s, "superposition d(SAB) = d(SA) + d(SB)", rel(d["SAB"], sumAB, d["SAB"]), tol)
    rec(s, "remove-one SAB - SA = SB - S0", rel({k: F["SAB"][k] - F["SA"][k] for k in K}, d["SB"], d["SB"]), tol)
    rec(s, "remove-one SAB - SB = SA - S0", rel({k: F["SAB"][k] - F["SB"][k] for k in K}, d["SA"], d["SA"]), tol)
    rec(s, "amplitude d(SAB2) = 2 d(SAB)", rel(d["SAB2"], {k: 2 * d["SAB"][k] for k in K}, d["SAB2"]), tol)
    rec(s, "overlap: SOV = SUN (union counted once)", rel(F["SOV"], F["SUN"], F["SUN"]), tol)
    rec(s, "whole domain: SWH = SFB", rel(F["SWH"], F["SFB"], F["SFB"]), tol)
    nA, nB = (max(np.abs(d[t][k]).max() for k in K) for t in ("SA", "SB"))
    rec(s, "|d(SA)| > 0 and |d(SB)| > 0 (each contributes)", f"{nA:.3e}/{nB:.3e}", None)
    diff = rel(d["SA"], d["SB"], d["SA"])
    rec(s, "d(SA) != d(SB) (distinguishable): rel diff", f"{diff:.3e}", None)
    ok &= nA > 1e-6 and nB > 1e-6 and diff > 0.1
    xyz = V["S0"][2]
    for t, side in (("SA", "x<0.5"), ("SB", "x>0.5")):
        w = np.abs(d[t]["state"]).sum(0)
        c = (w[:, None] * xyz).sum(0) / w.sum()
        good = c[0] < 0.5 if t == "SA" else c[0] > 0.5
        ok &= good
        rows.append({"scheme": s.upper(), "check": f"centroid of |d({t}).y| ({side})", "value": np.round(c, 3).tolist(),
                     "tolerance": side, "verdict": "PASS" if good else "FAIL", "note": ""})
        print(f"T10_SRC {s.upper()} centroid of |d({t}).y| = {np.round(c,3)} {'PASS' if good else 'FAIL'}")
    for t in V:
        rows.append({"scheme": s.upper(), "check": f"{t} production true residual / outer its",
                     "value": f"{V[t][0]['solve']['true_residual']:.3e}/{V[t][0]['solve']['iterations']}",
                     "tolerance": "", "verdict": "INFO", "note": f"J={V[t][0]['functional']['J_total']:.12e}"})
def g(p, s):
    m = re.search(p, s, re.M); return m.group(1) if m else ""
for f in sorted(glob.glob(D + "/logs/sources_verify/*.log")):
    s_ = open(f).read(); tag = os.path.basename(f)[:-4]
    tr = g(r"G4_PRODUCTION .*?true_residual=(\S+)", s_); gr = g(r"\|grad J\(u\*\)\|/\|dt alpha_u N u\*\|=(\S+)", s_)
    mr = g(r"\|y_prod - E\^-1\(rhs\+eta N u\*\)\|/\|y\|=(\S+)", s_)
    good = tr != "" and float(tr) < 1e-10 and float(gr) < 1e-9 and float(mr) < 1e-12
    ok &= good
    rows.append({"scheme": tag.split("_")[1].upper(), "check": f"verify {tag}: true_res / gradJ / march",
                 "value": f"{tr}/{gr}/{mr}", "tolerance": "1e-10/1e-9/1e-12", "verdict": "PASS" if good else "FAIL", "note": ""})
    print(f"T10_SRC verify {tag}: {tr} {gr} {mr} {'PASS' if good else 'FAIL'}")
with open(D + "/../t10_sources.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("T10_SRC_RESULT", "PASS" if ok else "FAIL")
