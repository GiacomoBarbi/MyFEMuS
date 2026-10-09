#!/usr/bin/env python3
"""T14 (copy of t13_selfconv.py) fixed-dt spatial self-convergence (fine reference, exact Q2 prolongation) on a union of faces.
usage: t13_selfconv.py <json-dir> <tag> <scheme> <nt> "<n list>" <n_ref> <faces a,b,...>
y, p: L2(Omega); u: L2(Gamma_c) = sum_f int_{Gamma_f} (face slices of the prolongated difference);
same time levels (fixed dt), so the time error cancels."""
import math, sys
import numpy as np
from t14_fields import load, prolong_1d, surf_sq, vol_sq
jd, wt, s, nt, ns, nref, faces = (sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]),
                                  [int(v) for v in sys.argv[5].split()], int(sys.argv[6]), sys.argv[7].split(","))
dF, fF = load(f"{jd}/{wt}_{s}_n{nref}_nt{nt}.json")
dt = dF["config"]["dt"]
print("tag,scheme,nt,n,ref,faces,y,y_order,p,p_order,u,u_order,J,dJ")
prev = None
for n in ns:
    dC, fC = load(f"{jd}/{wt}_{s}_n{n}_nt{nt}.json")
    P = prolong_1d(n, nref // n)
    err = {}
    for key, lab in (("state", "y"), ("adjoint", "p"), ("control_full", "u")):
        Cf = np.einsum("ia,jb,kc,tabc->tijk", P, P, P, fC[key])
        D = Cf - fF[key]
        err[lab] = math.sqrt(dt * sum((surf_sq(nref, D[m], faces) if lab == "u" else vol_sq(nref, D[m])) for m in range(D.shape[0])))
    dJ = abs(dC["functional"]["J_total"] - dF["functional"]["J_total"])
    o = {k: ("" if prev is None else f"{math.log(prev[1][k] / err[k]) / math.log(n / prev[0]):.2f}") for k in err}
    print(f"{wt},{s.upper()},{nt},{n},{nref},{'+'.join(faces)},{err['y']:.4e},{o['y']},{err['p']:.4e},{o['p']},{err['u']:.4e},{o['u']},"
          f"{dC['functional']['J_total']:.10e},{dJ:.3e}")
    prev = (n, err)
