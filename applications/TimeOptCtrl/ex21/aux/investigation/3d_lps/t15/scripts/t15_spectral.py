#!/usr/bin/env python3
"""T15 optional spectral diagnostics -> ../../t15_spectral.csv (dense, n = 4, 729 DOFs; operator dumps).
For L = nu K + C (+ W) on the free DOFs (homogeneous Dirichlet state: the nonzero rows of the eliminated M; natural:
all), Galerkin vs LPS-P0 vs LPS-P1, nu = 1e-3 and 1e-2, wind A and harmonic-3d:
  lambda_min(M^-1 sym(L))        discrete coercivity (natural state: may be < 0 where w.n < 0 / div w > 0)
  ||skew(L)|| / ||sym(L)||        skewness ratio (Frobenius)
  cond_2(L)                       2-norm condition number (natural state: L 1 = 0, singular; labelled in the CSV)
  ||L L^T - L^T L|| / ||L||^2     departure from normality (Frobenius)
  lambda_max(M^-1 sym(L)) / lambda_min  (Dirichlet state only)"""
import csv, os, subprocess
import numpy as np, scipy.linalg as sla, scipy.sparse as sps
X = os.environ.get("EX21_BIN", "/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21")
W = os.environ.get("T15_WORK", "/tmp/t15_spec"); os.makedirs(W + "/output", exist_ok=True)
ENV = {"HOME": os.environ["HOME"], "PATH": "/home/giacomo/software/petsc/arch-linux-c-openblas-opt/bin:/usr/bin:/bin",
       "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
D = os.path.dirname(os.path.abspath(__file__)) + "/.."


def coo(path):
    with open(path) as fh:
        fh.readline(); r, c = map(int, fh.readline().split()); fh.readline()
        d = np.loadtxt(fh, ndmin=2)
    return sps.csr_matrix((d[:, 2], (d[:, 0].astype(int), d[:, 1].astype(int))), shape=(r, c)).toarray()


rows = []
WINDS = {"A": ["--convection", "on", "--vx", "1", "--vy", "0.5", "--vz", "-0.75"], "harmonic-3d": ["--convection", "on", "--wind", "harmonic-3d"]}
MODES = {"Galerkin": [], "LPS-P0": ["--convection-stabilization", "lps"], "LPS-P1": ["--convection-stabilization", "lps", "--lps-projection", "p1"]}
for state in ("dirichlet", "natural"):
    for wind, wa in WINDS.items():
        for nu in ("1e-3", "1e-2"):
            for mode, ma in MODES.items():
                pre = f"{W}/{state}_{wind}_{nu}_{mode}"
                subprocess.run([X, "--dimension", "3", "--distributed-solve", "--control-full-domain", "--tracking-box3d", "0", "1", "0", "1", "0", "1",
                                "--state-bc", state, "--nu", nu, "--nx", "4", "--ny", "4", "--nz", "4", "--steps", "2", "--pc-schur-sweep",
                                "--no-paraview", "--dump-operators", pre] + wa + ma, cwd=W, env=ENV, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                L, M = coo(pre + "_K.coo"), coo(pre + "_M.coo")
                F = np.where(np.abs(M).sum(1) > 0)[0]
                L, M = L[np.ix_(F, F)], M[np.ix_(F, F)]
                S, K = 0.5 * (L + L.T), 0.5 * (L - L.T)
                ev = sla.eigh(S, M, eigvals_only=True)
                sv = np.linalg.svd(L, compute_uv=False)
                r = {"state": state, "wind": wind, "nu": nu, "method": mode, "free_dofs": len(F),
                     "lambda_min_sym": f"{ev[0]:.4e}", "lambda_max_sym": f"{ev[-1]:.4e}",
                     "skew_over_sym": f"{np.linalg.norm(K) / np.linalg.norm(S):.4e}", "cond2_L": f"{sv[0] / sv[-1]:.4e}",
                     "nonnormality": f"{np.linalg.norm(L @ L.T - L.T @ L) / np.linalg.norm(L) ** 2:.4e}"}
                rows.append(r); print(" ".join(f"{k}={v}" for k, v in r.items()), flush=True)
with open(D + "/../t15_spectral.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
