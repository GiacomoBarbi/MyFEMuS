#!/usr/bin/env python3
"""T12 fixed-mesh temporal self-convergence on simplex meshes (production path JSON).
usage: t12_time.py <json-dir> <tag> <scheme> <n> "<nt list>" <nt_ref>
Norm: the mesh's own FE L2 norm, ||e||^2 = dt sum_n e_n^T M_full e_n (M_full from the operator dump of the
same mesh; JSON and dump share the DOF order).  BE: y, p, u at t_n.  CN: y at t_n; p and u (interval
quantities of the distributed problem) at the coarse midpoint, against the mean of the two fine slots
adjacent to it (bias O(dt_fine^2)).  y_late: t >= T/4 (outside the inherited CN start-up layer).
*_refcorr removes the bias of a reference only a few times finer (e = C (dt^q - dt_ref^q))."""
import json, math, sys
import numpy as np
import scipy.sparse as sps
jd, tag, s, n, nts, ref = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), [int(v) for v in sys.argv[5].split()], int(sys.argv[6])
with open(f"{jd}/{tag}_ops_M_full.coo") as fh:
    fh.readline(); N = int(fh.readline().split()[0]); fh.readline(); d = np.loadtxt(fh, ndmin=2)
M = sps.csr_matrix((d[:, 2], (d[:, 0].astype(int), d[:, 1].astype(int))), shape=(N, N))
def L(nt):
    dd = json.load(open(f"{jd}/{tag}_{s}_n{n}_nt{nt}.json"))
    return dd, {k: np.array([x["values"] for x in dd[k]]) for k in ("state", "adjoint", "control_full")}
dF, fF = L(ref); cn = s == "cn"
def qcorr(e1, e2, d1, d2, dr):
    f = lambda q: (d1 ** q - dr ** q) / (d2 ** q - dr ** q) - e1 / e2
    lo, hi = 0.05, 8.0
    if f(lo) * f(hi) > 0: return ""
    for _ in range(200):
        mid = 0.5 * (lo + hi); lo, hi = (mid, hi) if f(lo) * f(mid) > 0 else (lo, mid)
    return f"{0.5 * (lo + hi):.2f}"
keys = ["y", "y_late", "p", "u"]
print("tag,scheme,n,nt,ref," + ",".join(f"{k},{k}_order,{k}_order_refcorr" for k in keys) + ",J,dJ")
prev = None
for nt in nts:
    d, f = L(nt); r = ref // nt; dt = d["config"]["dt"]; T = dt * nt; e = {k: 0.0 for k in keys}
    for k, lab in (("state", "y"), ("adjoint", "p"), ("control_full", "u")):
        for m in range(nt):
            if lab == "y" or not cn: g = fF[k][r * (m + 1) - 1]
            else: g = 0.5 * (fF[k][r * m + r // 2 - 1] + fF[k][r * m + r // 2])
            v = f[k][m] - g
            sq = dt * float(v @ (M @ v))
            e[lab] += sq
            if lab == "y" and (m + 1) * dt > T / 4 + 1e-12: e["y_late"] += sq
    e = {k: math.sqrt(v) for k, v in e.items()}
    dJ = abs(d["functional"]["J_total"] - dF["functional"]["J_total"])
    cols = []
    for k in keys:
        o = "" if prev is None else f"{math.log(prev[k] / e[k]) / math.log(nt / prev['nt']):.2f}"
        oc = "" if prev is None else qcorr(prev[k], e[k], 1 / prev["nt"], 1 / nt, 1 / ref)
        cols += [f"{e[k]:.4e}", o, oc]
    print(f"{tag},{s.upper()},{n},{nt},{ref}," + ",".join(cols) + f",{d['functional']['J_total']:.10e},{dJ:.3e}")
    prev = dict(e, nt=nt)
