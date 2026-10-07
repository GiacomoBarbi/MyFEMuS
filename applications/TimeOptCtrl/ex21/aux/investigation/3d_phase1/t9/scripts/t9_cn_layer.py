#!/usr/bin/env python3
"""CN initial-layer diagnostic for the state (fixed mesh, reference N_ref).
usage: t9_cn_layer.py <json-pattern with {nt}> <n_elements or 0 (2D nodal RMS)> "<nt list>" <nt_ref>
Prints, per N: y space-time L2 difference to the reference over (0,T], over
[0.25,T], at t = T and at the first level t = dt.  3D uses the HEX27 Simpson
weights of t9_selfconv.py; 2D (n = 0) uses the nodal RMS."""
import json, math, sys
import numpy as np
from t9_selfconv import grid, simpson_1d
pat, n, nts, ref = sys.argv[1], int(sys.argv[2]), [int(v) for v in sys.argv[3].split()], int(sys.argv[4])
if n:
    w = simpson_1d(n); W = np.einsum("i,j,k->ijk", w, w, w)
def L(nt):
    d = json.load(open(pat.format(nt=nt))); v = np.array([x["values"] for x in d["state"]])
    return grid(np.round(np.array(d["spatial_coordinates"]), 12), v) if n else v
nrm = (lambda a: np.sum(W * a * a)) if n else (lambda a: np.mean(a * a))
F = L(ref); prev = None
print("nt,y_L2L2,order,y_L2_t_ge_0.25,order,y_T,order,y_first_level")
for nt in nts:
    C = L(nt); r = ref // nt; dt = 1 / nt; e = [nrm(C[m] - F[r * (m + 1) - 1]) for m in range(nt)]
    v = (math.sqrt(dt * sum(e)), math.sqrt(dt * sum(e[m] for m in range(nt) if (m + 1) * dt >= 0.25 - 1e-12)), math.sqrt(e[-1]))
    o = ["" if prev is None else f"{math.log(prev[i] / v[i]) / math.log(nt / prev[3]):.2f}" for i in range(3)]
    print(f"{nt},{v[0]:.4e},{o[0]},{v[1]:.4e},{o[1]},{v[2]:.4e},{o[2]},{math.sqrt(e[0]):.3e}")
    prev = (*v, nt)
