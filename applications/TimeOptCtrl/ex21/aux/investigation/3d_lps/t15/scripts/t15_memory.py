#!/usr/bin/env python3
"""T15 memory audit, Galerkin vs LPS-P0 vs LPS-P1 -> ../../t15_memory.csv.

W is assembled once into its own matrix (kept for diagnostics; W_lps in NativeSpatialOperators) and ADDED to L with
SUBSET_NONZERO_PATTERN (the Q2 element couplings: W lives inside the pattern of L), so nnz(L) must not change and
the only new persistent storage is W itself (one extra spatial matrix of nnz(L) entries).  Per run: N_space, N_t,
nnz(L) (operator dump), nnz(W) (EX21_LPS3D), MaxRSS, bytes per space-time unknown (3 N_t N_space) and the RSS
ratio to the Galerkin run of the same configuration.
  distributed (conv-sin3-3d data, nu = 1e-3, wind A; SCHUR_SWEEP + LU)
  Dirichlet control zmin (production, nu = 0.1, wind A; reduced solver, iterative core).  [nu = 1e-2 was the first
  choice: there the GALERKIN Dirichlet runs stall from n = 6 on (iterative AND dense core; inner spatial solves at
  Pe_K ~ 11), so no like-for-like RSS exists; nu = 0.1 keeps Galerkin convergent with LPS partly active.  The stall
  itself is a solver-study result (t15_solver.csv, rA/rH).]  Each run is capped at 600 s."""
import csv, os, re, subprocess
X = os.environ.get("EX21_BIN", "/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21")
W = os.environ.get("T15_WORK", "/tmp/t15_mem"); os.makedirs(W + "/output", exist_ok=True)
D = os.path.dirname(os.path.abspath(__file__)) + "/.."
os.makedirs(D + "/logs/memory", exist_ok=True)
ENV = {"HOME": os.environ["HOME"], "PATH": "/home/giacomo/software/petsc/arch-linux-c-openblas-opt/bin:/usr/bin:/bin",
       "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
LU = "-sweep_right_pc_type lu -sweep_left_pc_type lu -sweep_ay_pc_type lu -fieldsplit_y_sweep_pc_type lu"
A = ["--convection", "on", "--vx", "1", "--vy", "0.5", "--vz", "-0.75"]
MODES = {"galerkin": [], "lps-p0": ["--convection-stabilization", "lps"], "lps-p1": ["--convection-stabilization", "lps", "--lps-projection", "p1"]}
rows = []


def prod(kind, n, nt):
    if kind == "distributed":
        return ["--dimension", "3", "--distributed-solve", "--control-full-domain", "--tracking-box3d", "0", "1", "0", "1", "0", "1",
                "--state-bc", "dirichlet", "--nu", "1e-3", "--alpha-u", "1e-2", "--target-type", "sin_x_minus_t", "--T", "1",
                "--pc-schur-sweep", "--nx", str(n), "--ny", str(n), "--nz", str(n), "--steps", str(nt)] + A
    return ["--dimension", "3", "--dirichlet-solve", "--state-bc", "dirichlet", "--dirichlet-face3d", "zmin", "--nu", "0.1",
            "--alpha-u", "1e-2", "--target-type", "sin_x_minus_t", "--T", "1", "--tracking-box3d", "0.25", "0.75", "0.25", "0.75", "0.5", "1",
            "--nx", str(n), "--ny", str(n), "--nz", str(n), "--steps", str(nt)] + A


def nnz_L(kind, mode, n):
    pre = f"{W}/nnz_{kind}_{mode}_{n}"
    subprocess.run([X] + prod(kind, n, 2) + MODES[mode] + ["--no-paraview", "--dump-operators", pre], cwd=W,
                   env=dict(ENV, PETSC_OPTIONS=LU if kind == "distributed" else ""), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open(pre + "_K.coo") as fh:
        return sum(1 for _ in fh) - 3


for kind in ("distributed", "dirichlet"):
    for n, nt in ((4, 32), (6, 32), (8, 32)):
        base = None
        for mode in MODES:
            tag = f"{kind}_{mode}_n{n}_nt{nt}"
            r = subprocess.run(["timeout", "600", "/usr/bin/time", "-f", "MAXRSS=%M wall=%e", X] + prod(kind, n, nt) + MODES[mode] + ["--solve-only", "--no-paraview"],
                               cwd=W, env=dict(ENV, PETSC_OPTIONS=LU if kind == "distributed" else ""),
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
            open(f"{D}/logs/memory/{tag}.log", "w").write(r.stdout)
            mr = re.search(r"MAXRSS=(\d+)", r.stdout)
            rss = int(mr.group(1)) / 1024.0 if mr else float("nan")
            N = (2 * n + 1) ** 3
            m = re.search(r"nnz_W=(\S+)", r.stdout)
            row = {"kind": kind, "mode": mode, "n": n, "N_space": N, "N_t": nt, "nnz_L": nnz_L(kind, mode, n),
                   "nnz_W": m.group(1) if m else "0", "exit": r.returncode, "rss_MB": f"{rss:.1f}",
                   "bytes_per_spacetime_unknown": f"{rss * 2 ** 20 / (3 * nt * N):.0f}", "wall_s": (re.search(r"wall=(\S+)", r.stdout).group(1) if mr else "600 (cap)"),
                   "lps_active_fraction": (re.search(r"active_fraction=(\S+)", r.stdout).group(1) if "active_fraction=" in r.stdout else "")}
            if base is None:
                base = rss
            row["rss_vs_galerkin"] = f"{rss / base:.3f}"
            row["W_bytes_est_MB"] = f"{float(row['nnz_W']) * 12 / 2 ** 20:.2f}"   # AIJ: 8 B value + 4 B column index
            rows.append(row)
            print(" ".join(f"{k}={v}" for k, v in row.items()), flush=True)
with open(D + "/../t15_memory.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
