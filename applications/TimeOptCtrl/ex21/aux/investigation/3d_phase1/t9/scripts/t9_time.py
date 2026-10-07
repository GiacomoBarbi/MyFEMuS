#!/usr/bin/env python3
"""Fixed-mesh temporal self-convergence (3D).  usage: t9_time.py <json-dir> <wind> <scheme> <n> "<nt list>" <nt_ref>
Space-time L2 norm (element Simpson weights on the HEX27 grid, as in
t9_selfconv.py) on the SAME mesh (the spatial error cancels): y at t_n; p and u (interval quantities under CN) against
the mean of the two fine midpoints inside the coarse interval; BE: at t_n.
<k>_order is the raw log2 ratio; <k>_order_refcorr solves
e1/e2 = (dt1^q - dtr^q)/(dt2^q - dtr^q), i.e. removes the bias of a reference
that is only a few times finer than the finest run (e = C (dt^q - dt_ref^q))."""
import json, math, sys
import numpy as np
from t9_selfconv import grid, simpson_1d
jd, wt, s, n, nts, ref = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), [int(v) for v in sys.argv[5].split()], int(sys.argv[6])
def L(nt):
    d = json.load(open(f"{jd}/{wt}_{s}_n{n}_nt{nt}.json"))
    xyz = np.round(np.array(d["spatial_coordinates"]), 12)
    return d, {k: grid(xyz, np.array([x["values"] for x in d[k]])) for k in ("state", "adjoint", "control_full")}
dF, fF = L(ref); cn = s == "cn"
w1 = simpson_1d(n); W3 = np.einsum("i,j,k->ijk", w1, w1, w1)
def qcorr(e1, e2, d1, d2, dr):
    f = lambda q: (d1 ** q - dr ** q) / (d2 ** q - dr ** q) - e1 / e2
    lo, hi = 0.05, 8.0
    if f(lo) * f(hi) > 0: return ""
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(lo) * f(mid) > 0 else (lo, mid)
    return f"{0.5 * (lo + hi):.2f}"
print("wind,scheme,n,nt,ref,y,y_order,y_order_refcorr,p,p_order,p_order_refcorr,u,u_order,u_order_refcorr,J,dJ")
prev = None
for nt in nts:
    d, f = L(nt); r = ref // nt; dt = d["config"]["dt"]; e = {}
    for k, lab in (("state", "y"), ("adjoint", "p"), ("control_full", "u")):
        acc = 0.0
        for m in range(nt):
            if lab == "y" or not cn:
                g = fF[k][r * (m + 1) - 1]
            else:
                g = 0.5 * (fF[k][r * m] + fF[k][r * m + r - 1])
            acc += dt * np.sum(W3 * (f[k][m] - g) ** 2)
        e[lab] = math.sqrt(acc)
    dJ = abs(d["functional"]["J_total"] - dF["functional"]["J_total"])
    o = {k: ("" if prev is None else f"{math.log(prev[k] / e[k]) / math.log(nt / prev['nt']):.2f}") for k in e}
    oc = {k: ("" if prev is None else qcorr(prev[k], e[k], 1 / prev["nt"], 1 / nt, 1 / ref)) for k in e}
    print(f"{wt},{s.upper()},{n},{nt},{ref},{e['y']:.4e},{o['y']},{oc['y']},{e['p']:.4e},{o['p']},{oc['p']},{e['u']:.4e},{o['u']},{oc['u']},{d['functional']['J_total']:.10e},{dJ:.3e}")
    prev = dict(e, nt=nt)
