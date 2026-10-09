#!/usr/bin/env python3
"""T14 memory audit, constant vs variable wind -> ../../t14_memory.csv.

The variable wind is EVALUATED ON DEMAND at the quadrature points during assembly (Config.hpp field3d, a
function pointer; the presets store only the 3x3 Hessian): no nodal or element wind array exists, so no new
persistent storage is expected.  Per run: N_space, N_t, nnz(L) (the dumped spatial operator; the wind does not
change the Q2 sparsity), MaxRSS, bytes per space-time unknown (3 N_t N_space for the KKT), and the RSS ratio
to the constant-wind run of the same configuration.
  natural state, distributed control (SCHUR_SWEEP + LU): natural-conv-3d wind A | natural-var-3d harmonic | affine-div
  Dirichlet control zmin (reduced, iterative core): wind A | harmonic | affine-div"""
import csv, os, re, subprocess
import numpy as np
X = os.environ.get("EX21_BIN", "/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21")
W = os.environ.get("T14_WORK", "/tmp/t14_mem"); os.makedirs(W + "/output", exist_ok=True)
D = os.path.dirname(os.path.abspath(__file__)) + "/.."
os.makedirs(D + "/logs/memory", exist_ok=True)
ENV = {"HOME": os.environ["HOME"], "PATH": "/home/giacomo/software/petsc/arch-linux-c-openblas-opt/bin:/usr/bin:/bin",
       "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
LU = "-sweep_right_pc_type lu -sweep_left_pc_type lu -sweep_ay_pc_type lu -fieldsplit_y_sweep_pc_type lu"
WINDS = {"const": ["--convection", "on", "--vx", "1", "--vy", "0.5", "--vz", "-0.75"],
         "harmonic-3d": ["--convection", "on", "--wind", "harmonic-3d"], "affine-div-3d": ["--convection", "on", "--wind", "affine-div-3d"]}
rows = []


def args(kind, wind, n, nt):
    if kind == "natural":
        case = ["--exact-oc", "natural-conv-3d"] if wind == "const" else ["--exact-oc", "natural-var-3d"]
        return case + ["--solve-only", "--pc-schur-sweep", "--nx", str(n), "--ny", str(n), "--nz", str(n), "--steps", str(nt)] + WINDS[wind]
    return ["--dimension", "3", "--dirichlet-solve", "--state-bc", "dirichlet", "--dirichlet-face3d", "zmin", "--nu", "0.25",
            "--alpha-u", "1e-2", "--target-type", "sin_x_minus_t", "--T", "1", "--tracking-box3d", "0.25", "0.75", "0.25", "0.75", "0.5", "1",
            "--nx", str(n), "--ny", str(n), "--nz", str(n), "--steps", str(nt)] + WINDS[wind]


def nnz(kind, wind, n):
    pre = f"{W}/nnz_{kind}_{wind}_{n}"
    # the exact driver does not dump operators: the same mesh / state BC / wind as a production run (the
    # sparsity of L depends only on the mesh)
    a = (["--dimension", "3", "--distributed-solve", "--control-full-domain", "--tracking-box3d", "0", "1", "0", "1", "0", "1",
          "--state-bc", "natural", "--pc-schur-sweep", "--nx", str(n), "--ny", str(n), "--nz", str(n), "--steps", "2"] + WINDS[wind]
         if kind == "natural" else args(kind, wind, n, 2))
    subprocess.run([X] + a + ["--no-paraview", "--dump-operators", pre], cwd=W, env=dict(ENV, PETSC_OPTIONS=LU),
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open(pre + "_K.coo") as fh:
        return sum(1 for _ in fh) - 3


for kind in ("natural", "dirichlet"):
    for n, nt in ((4, 32), (6, 32), (6, 64)):
        base = None
        for wind in WINDS:
            tag = f"{kind}_{wind}_n{n}_nt{nt}"
            env = dict(ENV, PETSC_OPTIONS=LU if kind == "natural" else "")
            r = subprocess.run(["/usr/bin/time", "-f", "MAXRSS=%M wall=%e", X] + args(kind, wind, n, nt) + ["--no-paraview"], cwd=W, env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
            open(f"{D}/logs/memory/{tag}.log", "w").write(r.stdout)
            rss = int(re.search(r"MAXRSS=(\d+)", r.stdout).group(1)) / 1024.0
            N = (2 * n + 1) ** 3
            row = {"kind": kind, "wind": wind, "n": n, "N_space": N, "N_t": nt, "nnz_L": nnz(kind, wind, n),
                   "exit": r.returncode, "rss_MB": f"{rss:.1f}", "bytes_per_spacetime_unknown": f"{rss * 2 ** 20 / (3 * nt * N):.0f}",
                   "wall_s": re.search(r"wall=(\S+)", r.stdout).group(1)}
            if base is None:
                base = rss
            row["rss_vs_const"] = f"{rss / base:.3f}"
            rows.append(row)
            print(" ".join(f"{k}={v}" for k, v in row.items()), flush=True)
with open(D + "/../t14_memory.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
