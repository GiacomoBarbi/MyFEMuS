#!/usr/bin/env python3
"""T14 (copy of t13_time.py, plus kind = distributed: u in L2(Omega), placed in time like p).
usage: t13_time.py <json-dir> <tag> <scheme> <n> "<nt list>" <nt_ref> <kind: neumann|dirichlet|distributed> <faces a,b,...>
  y, p : space-time L2 over Omega; u : space-time L2 over Gamma_c = sum_f int_{Gamma_f} (face slices).
  Time placement (T11N / T11D): y at t_n; Neumann u and CN p are interval quantities (compared with the
  mean of the two fine midpoints symmetric about the coarse midpoint); the Dirichlet u is NODAL (t_n).
  u is also reported on t <= T/2 and on the last slot alone; y on the window t >= T/4 and at t = T (the CN
  state start-up layer diagnostic of T11N).  *_refcorr removes the bias of a reference
  only a few times finer (e = C (dt^q - dt_ref^q))."""
import math, sys
from t14_fields import load, surf_sq, vol_sq
jd, wt, s, n, nts, ref, kind, faces = (sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]),
                                       [int(v) for v in sys.argv[5].split()], int(sys.argv[6]), sys.argv[7], (sys.argv[8] if len(sys.argv) > 8 else "-").split(","))
dF, fF = load(f"{jd}/{wt}_{s}_n{n}_nt{ref}.json"); cn = s == "cn"
def qcorr(e1, e2, d1, d2, dr):
    f = lambda q: (d1 ** q - dr ** q) / (d2 ** q - dr ** q) - e1 / e2
    lo, hi = 0.05, 8.0
    if f(lo) * f(hi) > 0: return ""
    for _ in range(200):
        mid = 0.5 * (lo + hi); lo, hi = (mid, hi) if f(lo) * f(mid) > 0 else (lo, mid)
    return f"{0.5 * (lo + hi):.2f}"
keys = ["y", "p", "u", "u_half", "u_last", "y_late", "y_T"]
print("tag,scheme,n,nt,ref,faces," + ",".join(f"{k},{k}_order,{k}_order_refcorr" for k in keys) + ",J,dJ")
prev = None
for nt in nts:
    d, f = load(f"{jd}/{wt}_{s}_n{n}_nt{nt}.json"); r = ref // nt; dt = d["config"]["dt"]; T = dt * nt; e = {k: 0.0 for k in keys}
    for k, lab in (("state", "y"), ("adjoint", "p"), ("control_full", "u")):
        interval = cn and (lab == "p" or (lab == "u" and kind in ("neumann", "distributed")))
        for m in range(nt):
            g = 0.5 * (fF[k][r * m] + fF[k][r * m + r - 1]) if interval else fF[k][r * (m + 1) - 1]
            if lab == "u":
                sq = vol_sq(n, f[k][m] - g) if kind == "distributed" else surf_sq(n, f[k][m] - g, faces)
                e["u"] += dt * sq
                if (m + 1) * dt <= T / 2 + 1e-12: e["u_half"] += dt * sq
                if m == nt - 1: e["u_last"] = sq
            else:
                sq = vol_sq(n, f[k][m] - g)
                e[lab] += dt * sq
                if lab == "y":
                    if (m + 1) * dt >= T / 4 - 1e-12: e["y_late"] += dt * sq
                    if m == nt - 1: e["y_T"] = sq
    e = {k: math.sqrt(v) for k, v in e.items()}
    dJ = abs(d["functional"]["J_total"] - dF["functional"]["J_total"])
    cols = []
    for k in keys:
        o = "" if prev is None else f"{math.log(prev[k] / e[k]) / math.log(nt / prev['nt']):.2f}"
        oc = "" if prev is None or k in ("u_last", "y_T") else qcorr(prev[k], e[k], 1 / prev["nt"], 1 / nt, 1 / ref)
        cols += [f"{e[k]:.4e}", o, oc]
    print(f"{wt},{s.upper()},{n},{nt},{ref},{'+'.join(faces)}," + ",".join(cols) + f",{d['functional']['J_total']:.10e},{dJ:.3e}")
    prev = dict(e, nt=nt)
