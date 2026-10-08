#!/usr/bin/env python3
"""T10 multi-zone fixed-dt spatial self-convergence with interface localisation.

usage: t10_mz_space.py <json-dir> <tag> <scheme> <nt> "<n list>" <n_ref> <out.csv>
Files <json-dir>/<tag>_<scheme>_n<n>_nt<nt>.json (unit cube, same time levels).
The coarse Q2 fields are prolongated exactly to the reference HEX27 grid
(t10_selfconv.prolong_1d) and the space-time L2 difference is integrated with
element Simpson weights.  The difference is also split into
  band      reference nodes within distance 1/8 (= h at n = 8) of a region face
            interior to the domain (the planes x, y, z = 1/2 of the octant
            layout) -- where u = -(eta/alpha_u) p chi_C jumps,
  interior  all other nodes,
so an order reduction can be attributed to the mask discontinuity.
"""
import csv
import math
import sys

import numpy as np

from t10_selfconv import grid, load, prolong_1d, simpson_1d

jd, tag, s, nt, ns, nref, out = (sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]),
                                 [int(v) for v in sys.argv[5].split()], int(sys.argv[6]), sys.argv[7])
dF, xF, fF = load(f"{jd}/{tag}_{s}_n{nref}_nt{nt}.json")
w = simpson_1d(nref)
W3 = np.einsum("i,j,k->ijk", w, w, w)
dt = dF["config"]["dt"]
g1 = np.linspace(0, 1, 2 * nref + 1)
Z, Y, Xg = np.meshgrid(g1, g1, g1, indexing="ij")   # grid() layout is [z, y, x]
dist = np.minimum.reduce([np.abs(Xg - .5), np.abs(Y - .5), np.abs(Z - .5)])
band = dist <= 0.125 + 1e-12
rows, prev = [], None
for n in ns:
    dC, xC, fC = load(f"{jd}/{tag}_{s}_n{n}_nt{nt}.json")
    P = prolong_1d(n, nref // n)
    r = {"tag": tag, "scheme": s.upper(), "nt": nt, "n": n, "n_ref": nref}
    for key, lab in (("state", "y"), ("adjoint", "p"), ("control_full", "u")):
        C = grid(xC, fC[key]); F = grid(xF, fF[key])
        D2 = (np.einsum("ia,jb,kc,tabc->tijk", P, P, P, C) - F) ** 2
        e = math.sqrt(dt * np.einsum("ijk,tijk->", W3, D2))
        eb = math.sqrt(dt * np.einsum("ijk,tijk->", W3 * band, D2))
        ei = math.sqrt(dt * np.einsum("ijk,tijk->", W3 * (~band), D2))
        r[lab] = f"{e:.4e}"; r[lab + "_band"] = f"{eb:.4e}"; r[lab + "_interior"] = f"{ei:.4e}"
        if prev:
            for suf in ("", "_band", "_interior"):
                r[lab + suf + "_order"] = f"{math.log(float(prev[lab + suf]) / float(r[lab + suf])) / math.log(n / prev['n']):.2f}"
    rows.append(r); prev = r
    print(" ".join(f"{k}={v}" for k, v in r.items()))
keys = list(dict.fromkeys(k for r in rows for k in r))
with open(out, "a", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=keys)
    if fh.tell() == 0:
        wr.writeheader()
    wr.writerows(rows)
