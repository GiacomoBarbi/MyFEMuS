#!/usr/bin/env python3
"""T9 HEX27 mesh / operator checks from --dump-operators (3D, homogeneous
Dirichlet).  usage: t9_mesh.py <dump-prefix-heat> <dump-prefix-conv> n"""
import sys
import numpy as np
ph, pc, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
ok = True
def load(prefix, op):
    L = [l.split() for l in open(f"{prefix}_{op}.coo") if not l.startswith("#")]
    N = int(L[0][0]); M = np.zeros((N, N))
    for r, c, v in L[1:]: M[int(r), int(c)] += float(v)
    return M
def rep(name, val, ref, tol=1e-12):
    global ok
    good = abs(val - ref) <= tol * max(1.0, abs(ref)); ok &= good
    print(f"T9_MESH n={n} {name:<58s} {val:.15g} ref {ref:.15g} {'PASS' if good else 'FAIL'}")
X = np.loadtxt(f"{ph}_coords.txt")
M, K = load(ph, "M"), load(ph, "K")
rep("spatial DOFs = (2n+1)^3", len(X), (2 * n + 1) ** 3, 0)
rep("coordinates are 3D (3 columns)", X.shape[1], 3, 0)
on_bnd = np.any((np.abs(X) < 1e-12) | (np.abs(X - 1) < 1e-12), axis=1)
constrained = np.all(M == 0, axis=1)        # eliminated rows (mass diagonal 0)
rep("constrained DOFs = (2n+1)^3 - (2n-1)^3", constrained.sum(), (2 * n + 1) ** 3 - (2 * n - 1) ** 3, 0)
rep("constrained set == boundary nodes (mismatches)", float((constrained != on_bnd).sum()), 0.0, 0)
kid = np.all(np.isclose(K[constrained][:, constrained], np.eye(constrained.sum())))
rep("K on constrained set = identity", float(kid), 1.0, 0)
q = X[:, 0] * (1 - X[:, 0]) * X[:, 1] * (1 - X[:, 1]) * X[:, 2] * (1 - X[:, 2])
rep("q^T M q = int q^2 = 1/27000 (Q2 bubble, exact)", q @ M @ q, 1 / 27000.0)
rep("q^T K q = nu int |grad q|^2 = 1/900 (nu = 1)", q @ K @ q, 1 / 900.0)
rep("M symmetric (max |M - M^T|)", np.abs(M - M.T).max(), 0.0, 1e-15)
rep("K symmetric (heat)", np.abs(K - K.T).max(), 0.0, 1e-15)
# total volume from the free-row mass times a function that is 1 inside:
Kc = load(pc, "K"); C = Kc - K
rep("C = L_conv - K_heat, q^T C q = int q w.grad q = 0", q @ C @ q, 0.0, 1e-14)
Cfree = C[np.ix_(~constrained, ~constrained)]
rep("C skew on free DOFs: |C+C^T|/|C|", np.abs(Cfree + Cfree.T).max() / max(np.abs(Cfree).max(), 1e-300), 0.0, 1e-13)
rep("M, K of conv run identical to heat run (M diff)", np.abs(load(pc, "M") - M).max(), 0.0, 0)
print("T9_MESH_RESULT", "PASS" if ok else "FAIL")
