#!/usr/bin/env python3
"""T5b diagnostic: fraction of free rows of A = M/dt + theta L for which BoomerAMG's
max_row_sum test (|sum_j a_ij| > max_row_sum |a_ii|, default 0.9) marks ALL couplings
weak.  Input: the spatial M and L dumps (text: "n n" header, then "i j a_ij"), in
DUMP (default: the T5b scratch dump directory).  Output: logs/rowsum_structure.log"""
import os, sys
import numpy as np
import scipy.sparse as sp
D = os.environ.get("DUMP", sys.argv[1] if len(sys.argv) > 1 else ".")
HERE = os.path.dirname(os.path.abspath(__file__))


def load(f):
    with open(f) as fh:
        n = int(fh.readline().split()[0])
    a = np.loadtxt(f, skiprows=1)
    return sp.csr_matrix((a[:, 2], (a[:, 0].astype(int), a[:, 1].astype(int))), shape=(n, n))


out = ["max_row_sum test of hypre BoomerAMG on A = M/dt + theta L (LP 5.2, tau = 1/nx, dt = 1/nx)",
       "ratio_i = |sum_j a_ij| / |a_ii|; row i is left with NO strong couplings when ratio_i > 0.9"]
for tag, eps, nx in (("C", "1/100", 32), ("D", "1/500", 32), ("D128", "1/500", 128)):
    M, L = load(f"{D}/{tag}_M.txt"), load(f"{D}/{tag}_L.txt")
    dt = 1.0 / nx
    for th in (1.0, 0.5):
        A = (M / dt + th * L).tocsr()
        d = A.diagonal()
        offnnz = np.diff(A.indptr) - 1
        free = offnnz > 0
        r = np.abs(np.asarray(A.sum(axis=1)).ravel()) / np.abs(d)
        r = r[free]
        out.append(f"eps={eps} nx={nx} theta={th}: free rows {free.sum()}; ratio>0.9 in {np.mean(r > 0.9) * 100:.1f}%"
                   f" (>0.99: {np.mean(r > 0.99) * 100:.1f}%); median ratio {np.median(r):.3f}, min {r.min():.3f}")
open(os.path.join(HERE, "logs", "rowsum_structure.log"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
