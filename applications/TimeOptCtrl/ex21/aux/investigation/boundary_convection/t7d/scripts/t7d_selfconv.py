#!/usr/bin/env python3
"""T7D fine-reference self-convergence from production JSON fields.

usage: t7d_selfconv.py time  <case> <scheme> <nx> "<nt list>" <nt_ref>
       t7d_selfconv.py space <case> <scheme> <nt> "<nx list>" <nx_ref>
Errors are discrete space-time L2 norms on the COARSE grid against the fine
reference restricted to it (nested meshes and time grids):
  u: Dirichlet control, nodal in time (t_n), on the Gamma_c nodes z = 0
     (1D trapezoid weights);
  y: state at t_n on all nodes (2D trapezoid weights);
  p: adjoint -- BE: at t_n; CN: at interval midpoints, the fine reference
     averaged over the two fine midpoints in the coarse interval (an O(dt_f^2)
     interpolation, adequate for an order estimate only).
SPACE mode instead uses TRUE L2 norms (compare_l2: exact Q2 prolongation of the
coarse field to the reference mesh + element Simpson quadrature); a coarse-node
norm would see Q2 nodal superconvergence, not the L2 order.
Also prints |J_h - J_ref|.  Appends rows to stdout as CSV.
"""
import json
import math
import os
import sys

import numpy as np

# the JSON fields (~1.2 GB for the T7D ladders) live OUTSIDE the repository tree
D = os.environ.get("T7D_JSON", "/tmp/t7d_selfconv_json").rstrip("/") + "/"


def load(case, scheme, nx, nt):
    d = json.load(open(f"{D}{case}_{scheme}_nx{nx}_nt{nt}.json"))
    xy = np.array(d["spatial_coordinates"])
    f = {k: np.array([s["values"] for s in d[k]]) for k in ("state", "adjoint", "control_full")}
    return d, xy, f


def trap_weights(coords):
    """tensor trapezoid weights for the (possibly 1D) node set."""
    w = np.ones(len(coords))
    for ax in range(coords.shape[1]):
        c = coords[:, ax]
        vals = np.unique(np.round(c, 12))
        if len(vals) == 1:
            continue
        h = vals[1] - vals[0]
        wa = np.where(np.isclose(c, vals[0]) | np.isclose(c, vals[-1]), h / 2, h)
        w *= wa
    return w


def keyed(xy):
    return {(round(a, 10), round(b, 10)): i for i, (a, b) in enumerate(xy)}


def grid(xy, f):
    """reshape nodal vectors on the uniform Q2 node grid -> [..., nz_nodes, nx_nodes]."""
    xs, zs = np.unique(np.round(xy[:, 0], 10)), np.unique(np.round(xy[:, 1], 10))
    ix = np.searchsorted(xs, np.round(xy[:, 0], 10)); iz = np.searchsorted(zs, np.round(xy[:, 1], 10))
    out = np.zeros(f.shape[:-1] + (len(zs), len(xs)))
    out[..., iz, ix] = f
    return xs, zs, out


def q2_prolong_1d(nc, r):
    """matrix evaluating a 1D Q2 (nc elements, 2nc+1 nodes) function at the
    2(nc r)+1 nodes of the r-times refined Q2 mesh (exact, piecewise quadratic)."""
    nf = 2 * nc * r + 1
    P = np.zeros((nf, 2 * nc + 1))
    for j in range(nf):
        s = j / (2 * r)                      # coordinate in coarse-element units
        e = min(int(s), nc - 1); xi = s - e  # local coordinate in [0, 1]
        l = [2 * (xi - 0.5) * (xi - 1), -4 * xi * (xi - 1), 2 * xi * (xi - 0.5)]
        for k in range(3):
            P[j, 2 * e + k] += l[k]
    return P


def simpson_1d(n_el):
    """Q2 nodal Simpson weights on [0, 1] with n_el elements (2 n_el + 1 nodes)."""
    h = 1.0 / n_el
    w = np.zeros(2 * n_el + 1)
    for e in range(n_el):
        w[2 * e:2 * e + 3] += h * np.array([1, 4, 1]) / 6
    return w


def compare_l2(coarse, fine, cn):
    """true L2(Omega) / L2(Gamma_c) errors: the coarse Q2 field prolongated
    exactly to the reference Q2 mesh, integrated with element Simpson weights
    (same time levels: spatial ladder at fixed N)."""
    dC, xyC, fC = coarse
    dF, xyF, fF = fine
    nc, nf = dC["config"]["mesh"][0], dF["config"]["mesh"][0]
    r = nf // nc
    P = q2_prolong_1d(nc, r)
    w1 = simpson_1d(nf)
    W2 = np.outer(w1, w1)
    dt, nt = dC["config"]["dt"], dC["config"]["N"]
    out = {}
    for name, key in (("y", "state"), ("u", "control_full"), ("p", "adjoint")):
        _, _, C = grid(xyC, fC[key]); _, _, F = grid(xyF, fF[key])
        Cf = np.einsum("ia,nab,jb->nij", P, C, P)        # [n, z, x]
        if name == "u":
            e2 = dt * np.einsum("j,nj->", w1, (Cf[:, 0, :] - F[:, 0, :]) ** 2)
            x2 = dt * np.einsum("j,nj->", w1, F[:, 0, :] ** 2)
        else:
            e2 = dt * np.einsum("ij,nij->", W2, (Cf - F) ** 2)
            x2 = dt * np.einsum("ij,nij->", W2, F ** 2)
        out[name], out[name + "_rel"] = math.sqrt(e2), math.sqrt(e2 / x2)
    out["J"], out["dJ"] = dC["functional"]["J_total"], abs(dC["functional"]["J_total"] - dF["functional"]["J_total"])
    out["outer"], out["true_res"] = dC["solve"]["iterations"], dC["solve"]["true_residual"]
    out["stat"] = dC["kkt"]["stationarity_consistency_residual_rel"]
    return out


def compare(coarse, fine, r_t, cn):
    dC, xyC, fC = coarse
    dF, xyF, fF = fine
    kF = keyed(xyF)
    idx = np.array([kF[(round(a, 10), round(b, 10))] for a, b in xyC])
    dt = dC["config"]["dt"]
    nt = dC["config"]["N"]
    wy = trap_weights(xyC)
    g = np.isclose(xyC[:, 1], 0.0)
    wu = trap_weights(xyC[g][:, :1])
    out = {}
    for name, key, mask, w in (("y", "state", None, wy), ("u", "control_full", g, wu), ("p", "adjoint", None, wy)):
        e2 = x2 = 0.0
        for n in range(nt):
            c = fC[key][n]
            if name == "p" and cn:
                f = 0.5 * (fF[key][r_t * n][idx] + fF[key][r_t * n + r_t - 1][idx]) if r_t > 1 else fF[key][n][idx]
            else:
                f = fF[key][r_t * (n + 1) - 1][idx]
            if mask is not None:
                c, f = c[mask], f[mask]
            e2 += dt * np.dot(w, (c - f) ** 2)
            x2 += dt * np.dot(w, f**2)
        out[name] = math.sqrt(e2)
        out[name + "_rel"] = math.sqrt(e2 / x2)
    jc = dC["functional"]["J_total"]
    jf = dF["functional"]["J_total"]
    out["J"] = jc
    out["dJ"] = abs(jc - jf)
    out["outer"] = dC["solve"]["iterations"]
    out["true_res"] = dC["solve"]["true_residual"]
    out["stat"] = dC["kkt"]["stationarity_consistency_residual_rel"]
    return out


def main():
    mode, case, scheme = sys.argv[1:4]
    fixed, ladder, ref = int(sys.argv[4]), [int(v) for v in sys.argv[5].split()], int(sys.argv[6])
    cn = scheme == "cn"
    if mode == "time":
        fine = load(case, scheme, fixed, ref)
        get = lambda n: (load(case, scheme, fixed, n), ref // n)
    else:
        fine = load(case, scheme, ref, fixed)
        get = lambda n: (load(case, scheme, n, fixed), 1)
    print("mode,case,scheme,nx,nt,ref,y,y_rel,y_order,u,u_rel,u_order,p,p_rel,p_order,J,dJ,dJ_order,outer,true_res,stationarity")
    prev = None
    for n in ladder:
        c, r = get(n)
        o = compare_l2(c, fine, cn) if mode == "space" else compare(c, fine, r, cn)
        orders = {}
        for k in ("y", "u", "p", "dJ"):
            orders[k] = "" if prev is None or o[k] == 0 or prev[1][k] == 0 else \
                f"{math.log(prev[1][k] / o[k]) / math.log(n / prev[0]):.2f}"
        nx_, nt_ = (fixed, n) if mode == "time" else (n, fixed)
        print(f"{mode},{case},{scheme.upper()},{nx_},{nt_},{ref},{o['y']:.4e},{o['y_rel']:.3e},{orders['y']},"
              f"{o['u']:.4e},{o['u_rel']:.3e},{orders['u']},{o['p']:.4e},{o['p_rel']:.3e},{orders['p']},"
              f"{o['J']:.10e},{o['dJ']:.3e},{orders['dJ']},{o['outer']},{o['true_res']:.2e},{o['stat']:.2e}")
        prev = (n, o)


if __name__ == "__main__":
    main()
