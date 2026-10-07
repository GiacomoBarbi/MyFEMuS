#!/usr/bin/env python3
"""T9 fixed-dt 3D spatial self-convergence (true L2(Omega) norms).

usage: t9_selfconv.py <json-dir> <tag-prefix> <nt> "<n list>" <n_ref>
Files: <json-dir>/<tag-prefix>_n<n>_nt<nt>.json from the production path
(--exact-oc conv-sin-3d --solve-only --dump-json).  The coarse Q2 field is
prolongated EXACTLY to the reference HEX27 mesh (tensor 1D Q2 prolongation in
x, y, z) and integrated with element Simpson weights (exact for the triquadratic
products here up to quadrature of the squared difference); same time levels
(fixed dt), so the time error cancels.  Fields are matched by coordinates.
"""
import json
import math
import sys

import numpy as np


def load(path):
    d = json.load(open(path))
    xyz = np.round(np.array(d["spatial_coordinates"]), 12)
    f = {k: np.array([s["values"] for s in d[k]]) for k in ("state", "adjoint", "control_full")}
    return d, xyz, f


def grid(xyz, field):
    xs = np.unique(xyz[:, 0]); ys = np.unique(xyz[:, 1]); zs = np.unique(xyz[:, 2])
    ix = np.searchsorted(xs, xyz[:, 0]); iy = np.searchsorted(ys, xyz[:, 1]); iz = np.searchsorted(zs, xyz[:, 2])
    out = np.zeros(field.shape[:-1] + (len(zs), len(ys), len(xs)))
    out[..., iz, iy, ix] = field
    return out


def prolong_1d(nc, r):
    nf = 2 * nc * r + 1
    P = np.zeros((nf, 2 * nc + 1))
    for j in range(nf):
        s = j / (2 * r); e = min(int(s), nc - 1); xi = s - e
        l = [2 * (xi - 0.5) * (xi - 1), -4 * xi * (xi - 1), 2 * xi * (xi - 0.5)]
        for k in range(3):
            P[j, 2 * e + k] += l[k]
    return P


def simpson_1d(n):
    w = np.zeros(2 * n + 1)
    for e in range(n):
        w[2 * e:2 * e + 3] += np.array([1, 4, 1]) / (6.0 * n)
    return w


def main():
    jdir, prefix, nt, ns, nref = sys.argv[1], sys.argv[2], int(sys.argv[3]), [int(v) for v in sys.argv[4].split()], int(sys.argv[5])
    dF, xF, fF = load(f"{jdir}/{prefix}_n{nref}_nt{nt}.json")
    w = simpson_1d(nref)
    W3 = np.einsum("i,j,k->ijk", w, w, w)
    dt = dF["config"]["dt"]
    print("n,nt,ref,y,y_order,u,u_order,p,p_order,J,dJ")
    prev = None
    for n in ns:
        dC, xC, fC = load(f"{jdir}/{prefix}_n{n}_nt{nt}.json")
        P = prolong_1d(n, nref // n)
        err = {}
        for key, lab in (("state", "y"), ("control_full", "u"), ("adjoint", "p")):
            C = grid(xC, fC[key]); F = grid(xF, fF[key])
            Cf = np.einsum("ia,jb,kc,tabc->tijk", P, P, P, C)
            e2 = dt * np.einsum("ijk,tijk->", W3, (Cf - F) ** 2)
            err[lab] = math.sqrt(e2)
        dJ = abs(dC["functional"]["J_total"] - dF["functional"]["J_total"])
        o = {k: ("" if prev is None else f"{math.log(prev[1][k] / err[k]) / math.log(n / prev[0]):.2f}") for k in err}
        od = "" if prev is None or dJ == 0 else f"{math.log(prev[2] / dJ) / math.log(n / prev[0]):.2f}"
        print(f"{n},{nt},{nref},{err['y']:.4e},{o['y']},{err['u']:.4e},{o['u']},{err['p']:.4e},{o['p']},{dC['functional']['J_total']:.10e},{dJ:.3e}")
        prev = (n, err, dJ)


if __name__ == "__main__":
    main()
