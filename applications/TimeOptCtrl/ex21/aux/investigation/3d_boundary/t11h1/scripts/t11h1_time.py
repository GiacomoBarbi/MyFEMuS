#!/usr/bin/env python3
"""T11H1 fixed-mesh temporal self-convergence for 3D Dirichlet control with H1 regularisation
(copy of t11d_time.py, plus u_t: the coarse first difference (u_n - u_{n-1})/dt against the same
difference of the reference sampled at the coarse nodes, u_0 shared DATA, surface L2 on Gamma_c).
usage: t11h1_time.py <json-dir> <tag> <scheme> <n> "<nt list>" <nt_ref>
(t11n_time.py with the Dirichlet control's time placement.)
  y, p : space-time L2 over Omega (HEX27 element-Simpson weights), same mesh;
  u    : space-time L2 over Gamma_c = {z = 0} (QUAD9 face Simpson weights on the
         z = 0 node slice; u = 0 on dGamma_c).
  The Dirichlet control is NODAL in time (u_n = y_n|_C / eta), for BE AND CN:
  y and u are compared at t_n.  p: BE at t_n; CN (interval quantity) against the
  mean of the two fine midpoints symmetric about the coarse midpoint.
  u is also reported on the window t <= T/2 and on the LAST slot alone.
  *_refcorr removes the bias of a reference only a few times finer."""
import json, math, sys
import numpy as np
from t11h1_selfconv import grid, simpson_1d
jd, wt, s, n, nts, ref = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), [int(v) for v in sys.argv[5].split()], int(sys.argv[6])
def L(nt):
    d = json.load(open(f"{jd}/{wt}_{s}_n{n}_nt{nt}.json"))
    xyz = np.round(np.array(d["spatial_coordinates"]), 12)
    return d, {k: grid(xyz, np.array([x["values"] for x in d[k]])) for k in ("state", "adjoint", "control_full")}
dF, fF = L(ref); cn = s == "cn"
# u_t: at m = 0 the shared datum u_0 cancels: (u_1 - u_0) - (u_ref - u_0) = u_1 - u_ref.
w1 = simpson_1d(n); W3 = np.einsum("i,j,k->ijk", w1, w1, w1); W2 = np.einsum("i,j->ij", w1, w1)   # grid layout [z, y, x]
def qcorr(e1, e2, d1, d2, dr):
    f = lambda q: (d1 ** q - dr ** q) / (d2 ** q - dr ** q) - e1 / e2
    lo, hi = 0.05, 8.0
    if f(lo) * f(hi) > 0: return ""
    for _ in range(200):
        mid = 0.5 * (lo + hi); lo, hi = (mid, hi) if f(lo) * f(mid) > 0 else (lo, mid)
    return f"{0.5 * (lo + hi):.2f}"
keys = ["y", "y_late", "p", "u", "u_t", "u_half", "u_last"]   # y_late: t >= T/4 (outside the CN start-up layer)
print("tag,scheme,n,nt,ref," + ",".join(f"{k},{k}_order,{k}_order_refcorr" for k in keys) + ",J,dJ")
prev = None
for nt in nts:
    d, f = L(nt); r = ref // nt; dt = d["config"]["dt"]; T = dt * nt; e = {k: 0.0 for k in keys}
    for k, lab in (("state", "y"), ("adjoint", "p"), ("control_full", "u")):
        for m in range(nt):
            g = fF[k][r * (m + 1) - 1] if (lab != "p" or not cn) else 0.5 * (fF[k][r * m] + fF[k][r * m + r - 1])
            if lab == "u":
                sq = np.sum(W2 * (f[k][m][0] - g[0]) ** 2)        # z = 0 slice = Gamma_c
                e["u"] += dt * sq
                if (m + 1) * dt <= T / 2 + 1e-12: e["u_half"] += dt * sq
                if m == nt - 1: e["u_last"] = sq
                prev_c = f[k][m - 1][0] if m > 0 else 0.0
                prev_r = fF[k][r * m - 1][0] if m > 0 else 0.0
                dq = ((f[k][m][0] - prev_c) - (g[0] - prev_r)) / dt
                e["u_t"] += dt * np.sum(W2 * dq ** 2)
            else:
                sqv = dt * np.sum(W3 * (f[k][m] - g) ** 2)
                e[lab] += sqv
                if lab == "y" and (m + 1) * dt > T / 4 + 1e-12: e["y_late"] += sqv
    e = {k: math.sqrt(v) for k, v in e.items()}
    dJ = abs(d["functional"]["J_total"] - dF["functional"]["J_total"])
    cols = []
    for k in keys:
        o = "" if prev is None else f"{math.log(prev[k] / e[k]) / math.log(nt / prev['nt']):.2f}"
        oc = "" if prev is None or k == "u_last" else qcorr(prev[k], e[k], 1 / prev["nt"], 1 / nt, 1 / ref)
        cols += [f"{e[k]:.4e}", o, oc]
    print(f"{wt},{s.upper()},{n},{nt},{ref}," + ",".join(cols) + f",{d['functional']['J_total']:.10e},{dJ:.3e}")
    prev = dict(e, nt=nt)
