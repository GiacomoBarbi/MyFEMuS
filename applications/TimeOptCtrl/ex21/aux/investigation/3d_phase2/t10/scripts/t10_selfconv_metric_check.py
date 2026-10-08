#!/usr/bin/env python3
"""T10: calibration of the fixed-dt self-convergence metric (t10_selfconv.py).

Applies the SAME metric (Q2 prolongation to the n = 16 grid, element-Simpson
weights, difference to the n = 16 field) to pure Q2 INTERPOLANTS of the exact
natural-conv-3d shapes y = cos(pi x) cos(pi y) cos(pi z) and
p ~ R_kx(x) R_ky(y) R_kz(z) (wind (1, 0.5, -0.75), nu = 1/4), and compares with
the interpolation error against the exact function (Simpson on n = 64).  If the
solver's self-convergence orders match the interpolants' in the same metric, the
discretisation is optimal and the < 3 values are the metric (reference only 2x
finer) and the pre-asymptotic steep p profile, not a defect."""
import math
import numpy as np
from t10_selfconv import prolong_1d, simpson_1d
nu = 0.25; k = (1 / nu, 0.5 / nu, -0.75 / nu)
R = lambda kk, s: np.exp(-kk * s) * (2 + np.cos(np.pi * s))
fy = lambda X, Y, Z: np.cos(np.pi * X) * np.cos(np.pi * Y) * np.cos(np.pi * Z)
fp = lambda X, Y, Z: R(k[0], X) * R(k[1], Y) * R(k[2], Z)
def nodes(n):
    g = np.linspace(0, 1, 2 * n + 1); Z, Y, X = np.meshgrid(g, g, g, indexing="ij"); return X, Y, Z
def ladder(f, ns, nref, ref_field):
    w = simpson_1d(nref); W = np.einsum("i,j,k->ijk", w, w, w); prev = None; out = []
    for n in ns:
        P = prolong_1d(n, nref // n)
        e = math.sqrt(np.einsum("ijk,ijk->", W, (np.einsum("ia,jb,kc,abc->ijk", P, P, P, f(*nodes(n))) - ref_field) ** 2))
        out.append((n, e, "" if prev is None else f"{math.log2(prev / e):.2f}")); prev = e
    return "  ".join(f"n={a}: {b:.3e} ({c})" for a, b, c in out)
for nm, f in (("y", fy), ("p", fp)):
    print(nm, "interpolant vs I_16 (self-convergence metric):", ladder(f, (2, 4, 8), 16, f(*nodes(16))))
    print(nm, "interpolant vs exact (Simpson n = 64):          ", ladder(f, (2, 4, 8, 16), 64, f(*nodes(64))))
