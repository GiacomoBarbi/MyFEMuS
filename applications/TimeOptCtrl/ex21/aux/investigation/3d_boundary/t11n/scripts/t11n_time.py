#!/usr/bin/env python3
"""T11N fixed-mesh temporal self-convergence for 3D Neumann control.
usage: t11n_time.py <json-dir> <tag> <scheme> <n> "<nt list>" <nt_ref>
(T10 t10_time.py with the Neumann control measured where it lives.)
  y, p : space-time L2 over Omega (HEX27 element-Simpson weights), same mesh;
  u    : space-time L2 over Gamma_c = {z = 0} (QUAD9 face Simpson weights on the
         z = 0 node slice) -- the surface norm of the control;
  BE: y, p, u at t_n.  CN: y at t_n; p and u (interval quantities) against the
  mean of the two fine midpoints symmetric about the coarse midpoint.
  u is also reported on the window t <= T/2 and on the LAST slot alone (the
  T7N CN terminal-layer diagnostics).  *_refcorr removes the bias of a
  reference only a few times finer (e = C (dt^q - dt_ref^q))."""
import json, math, sys
import numpy as np
from t11n_selfconv import grid, simpson_1d
jd, wt, s, n, nts, ref = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), [int(v) for v in sys.argv[5].split()], int(sys.argv[6])
def L(nt):
    d = json.load(open(f"{jd}/{wt}_{s}_n{n}_nt{nt}.json"))
    xyz = np.round(np.array(d["spatial_coordinates"]), 12)
    return d, {k: grid(xyz, np.array([x["values"] for x in d[k]])) for k in ("state", "adjoint", "control_full")}
dF, fF = L(ref); cn = s == "cn"
w1 = simpson_1d(n); W3 = np.einsum("i,j,k->ijk", w1, w1, w1); W2 = np.einsum("i,j->ij", w1, w1)   # grid layout [z, y, x]
def qcorr(e1, e2, d1, d2, dr):
    f = lambda q: (d1 ** q - dr ** q) / (d2 ** q - dr ** q) - e1 / e2
    lo, hi = 0.05, 8.0
    if f(lo) * f(hi) > 0: return ""
    for _ in range(200):
        mid = 0.5 * (lo + hi); lo, hi = (mid, hi) if f(lo) * f(mid) > 0 else (lo, mid)
    return f"{0.5 * (lo + hi):.2f}"
keys = ["y", "p", "u", "u_half", "u_last"]
print("tag,scheme,n,nt,ref," + ",".join(f"{k},{k}_order,{k}_order_refcorr" for k in keys) + ",J,dJ")
prev = None
for nt in nts:
    d, f = L(nt); r = ref // nt; dt = d["config"]["dt"]; T = dt * nt; e = {k: 0.0 for k in keys}
    for k, lab in (("state", "y"), ("adjoint", "p"), ("control_full", "u")):
        for m in range(nt):
            g = fF[k][r * (m + 1) - 1] if (lab == "y" or not cn) else 0.5 * (fF[k][r * m] + fF[k][r * m + r - 1])
            if lab == "u":
                sq = np.sum(W2 * (f[k][m][0] - g[0]) ** 2)        # z = 0 slice = Gamma_c
                e["u"] += dt * sq
                if (m + 1) * dt <= T / 2 + 1e-12: e["u_half"] += dt * sq
                if m == nt - 1: e["u_last"] = sq
            else:
                e[lab] += dt * np.sum(W3 * (f[k][m] - g) ** 2)
    e = {k: math.sqrt(v) for k, v in e.items()}
    dJ = abs(d["functional"]["J_total"] - dF["functional"]["J_total"])
    cols = []
    for k in keys:
        o = "" if prev is None else f"{math.log(prev[k] / e[k]) / math.log(nt / prev['nt']):.2f}"
        oc = "" if prev is None or k == "u_last" else qcorr(prev[k], e[k], 1 / prev["nt"], 1 / nt, 1 / ref)
        cols += [f"{e[k]:.4e}", o, oc]
    print(f"{wt},{s.upper()},{n},{nt},{ref}," + ",".join(cols) + f",{d['functional']['J_total']:.10e},{dJ:.3e}")
    prev = dict(e, nt=nt)
