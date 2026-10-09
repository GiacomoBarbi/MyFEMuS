#!/usr/bin/env python3
"""T12 mesh comparison -> ../../t12_mesh_comparison.csv, and the 2D time-corrected rows appended to
../../t12_tri_space.csv.
(a) for every n run on BOTH families of a pair (QUAD/TRI, HEX/TET) -- the SAME exact PDE, the SAME spatial
    DOF set (the P2 / Q2 lattice) -- error ratios simplex/tensor for y, p, u, element ratio, nnz-free
    cost measures (wall, MaxRSS), outer iterations.  Equal-DOF comparison is exact here (identical N_space);
    the equal-h comparison is the same rows.
(b) 2D: the CN N = 256 time error (true error ~ e_meas(256) / (1 - 1/4) vs the N = 512 reference, from the
    n = 16 time study of each family) removed in quadrature from p and u ONLY.  The fixed-mesh CN y time
    error is dominated by the h-dependent start-up layer (it exceeds the total n = 64 y error), so it does
    not transfer between meshes; y is left uncorrected (its order is already 3.00)."""
import csv, math, os
D = os.path.dirname(os.path.abspath(__file__)) + "/.."
rd = lambda f: list(csv.DictReader(open(f"{D}/../{f}")))
rows = []
for f, a, b in (("t12_tri_space.csv", "quad", "tri"), ("t12_tet_space.csv", "hex", "tet")):
    sp = [r for r in rd(f) if r.get("ladder", "").find("corrected") < 0]
    A = {int(r["n"]): r for r in sp if r["mesh"] == a}; B = {int(r["n"]): r for r in sp if r["mesh"] == b}
    for n in sorted(set(A) & set(B)):
        ra, rb = A[n], B[n]
        row = {"pair": f"{a}/{b}", "n": n, "h": 1.0 / n, "N_space_both": ra["dofs"], "elements_" + a: ra["elements"],
               "elements_" + b: rb["elements"], "element_ratio": f"{int(rb['elements']) / int(ra['elements']):.0f}"}
        for k in ("y", "p", "u"):
            row[f"{k}_{a}"] = ra[f"{k}_L2L2"]; row[f"{k}_{b}"] = rb[f"{k}_L2L2"]
            row[f"{k}_ratio_{b}/{a}"] = f"{float(rb[k + '_L2L2']) / float(ra[k + '_L2L2']):.3f}"
        for k in ("wall_s", "maxrss_MB", "outer_its", "ysplit_its_max"):
            row[f"{k}_{a}"] = ra[k]; row[f"{k}_{b}"] = rb[k]
        rows.append(row)
keys = list(dict.fromkeys(k for r in rows for k in r))
with open(f"{D}/../t12_mesh_comparison.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
for r in rows: print(r["pair"], r["n"], {k: v for k, v in r.items() if "ratio" in k})
# (b) time-error-corrected 2D rows
tm = rd("t12_time.csv")
sp = [r for r in rd("t12_tri_space.csv") if r.get("ladder", "").find("corrected") < 0]
extra = []
for mesh, tag in (("quad", "qT"), ("tri", "tT")):
    t = [r for r in tm if r["tag"] == tag and r["scheme"] == "CN" and int(r["nt"]) == 256][0]
    et = {k: float(t[k]) / (1 - 0.25) for k in ("p", "u")}
    et["y"] = 0.0
    prev = None
    for r in sorted((r for r in sp if r["mesh"] == mesh), key=lambda r: int(r["n"])):
        n = int(r["n"]); e = {k: math.sqrt(max(float(r[k + "_L2L2"]) ** 2 - et[k] ** 2, 0.0)) for k in ("y", "p", "u")}
        row = dict(r); row["ladder"] = f"{r['ladder']} (CN N=256 time error removed)"
        for k in e:
            row[k + "_L2L2"] = f"{e[k]:.6e}"
            row[f"ord_{k}_h"] = row[f"ord_{k}_dofs"] = "" if prev is None else f"{math.log(prev[k] / e[k]) / math.log(n / prev['n']):.2f}"
        row["time_error_removed_y_p_u"] = f"{et['y']:.2e}/{et['p']:.2e}/{et['u']:.2e}"
        extra.append(row); prev = dict(e, n=n)
allrows = sp + extra
keys = list(dict.fromkeys(k for r in allrows for k in r))
with open(f"{D}/../t12_tri_space.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(allrows)
for r in extra: print(r["mesh"], r["n"], "y", r["ord_y_h"], "p", r["ord_p_h"], "u", r["ord_u_h"])
