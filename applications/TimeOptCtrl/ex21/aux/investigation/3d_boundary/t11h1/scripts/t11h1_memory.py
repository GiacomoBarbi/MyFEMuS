#!/usr/bin/env python3
"""T11H1 memory audit -> ../../t11h1_memory.csv.  Production path, face zmin, wind A, CN, T = 1, tracking
(0,1)^2 x (0.3,1).  Peak RSS (/usr/bin/time) for L2, H1s (ell = 0.3), H1t (gamma = 1e-2), comb, with the
default iterative core over n = 4 (Nt = 32..512), n = 6 (Nt = 32, 64), n = 8 (Nt = 32, 128); the dense core for
the smaller sizes (its excess is O(m^2), T11D: 12 B m^2 + 8 B Nt N |C|).  nnz(K_Gamma|_C), nnz(M_Gamma|_C)
from one operator dump per mesh; nnz(R_H1t) = Nt nnz(R_C) + 2 (Nt-1) nnz(M_C) (gamma > 0; Nt nnz(R_C) otherwise),
R_C = M_C + ell^2 K_C (same pattern) -- exactly the sparse matrix the core factors (validated entry by entry by
t11h1_surface_stiffness.py).  m = Nt |C|; space-time unknowns 2 Nt N."""
import csv, os, re, subprocess
import numpy as np

X = os.environ.get("EX21_BIN", "/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21")
W = os.environ.get("T11H1_WORK", "/tmp/t11h1_mem"); os.makedirs(W + "/output", exist_ok=True)
D = os.path.dirname(os.path.abspath(__file__)) + "/.."
os.makedirs(D + "/logs/memory", exist_ok=True)
ENV = {"HOME": os.environ["HOME"], "PATH": "/home/giacomo/software/petsc/arch-linux-c-openblas-opt/bin:/usr/bin:/bin",
       "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
REGS = {"L2": [], "H1s": ["--control-h1-length", "0.3"], "H1t": ["--control-h1-time-weight", "1e-2"],
        "comb": ["--control-h1-length", "0.3", "--control-h1-time-weight", "1e-2"]}
nnz_cache = {}


def base(n, nt):
    return ["--dimension", "3", "--dirichlet-solve", "--dirichlet-face3d", "zmin", "--state-bc", "dirichlet",
            "--tracking-box3d", "0", "1", "0", "1", "0.3", "1", "--target-type", "sin_x_minus_t", "--T", "1",
            "--nx", str(n), "--ny", str(n), "--nz", str(n), "--steps", str(nt), "--time-scheme", "cn",
            "--convection", "on", "--vx", "1", "--vy", "0.5", "--vz", "-0.75", "--no-paraview"]


def nnz(n):
    if n not in nnz_cache:
        p = f"{W}/ops_n{n}"
        subprocess.run([X] + base(n, 2) + ["--dump-operators", p], cwd=W, env=ENV, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, check=True)
        cnt = lambda f: int(np.loadtxt(f, skiprows=3, ndmin=2).shape[0])
        nnz_cache[n] = (cnt(p + "_K_Gamma.coo"), cnt(p + "_M_Gamma.coo"))
    return nnz_cache[n]


rows = []


def run(n, nt, reg, core):
    tag = f"{reg}_{core}_n{n}_nt{nt}"
    r = subprocess.run(["/usr/bin/time", "-f", "MAXRSS=%M wall=%e", X] + base(n, nt) + REGS[reg] + ["--dirichlet-core-solver", core],
                       cwd=W, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    s = r.stdout
    open(f"{D}/logs/memory/{tag}.log", "w").write(s)
    g = lambda p: re.search(p, s).group(1) if re.search(p, s) else ""
    N, C = (2 * n + 1) ** 3, (2 * n - 1) ** 2
    kC, mC = nnz(n)
    gam = reg in ("H1t", "comb")
    row = {"reg": reg, "core": core, "n": n, "nt": nt, "N_spatial": N, "face_dofs_C": C, "m": nt * C,
           "spacetime_unknowns": 2 * nt * N, "nnz_K_Gamma_C": kC, "nnz_M_Gamma_C": mC,
           "nnz_R_H1t": nt * mC + (2 * (nt - 1) * mC if gam else 0), "exit": r.returncode,
           "reason": g(r"EX20_PAR_SOLVE .*? reason=(\S+)"), "core_its": g(r"core_its_max=(\d+)"),
           "rss_MB": int(g(r"MAXRSS=(\d+)") or 0) / 1024.0, "wall_s": g(r"wall=(\S+)"), "J_track": g(r"J_track=(\S+)")}
    row["bytes_per_spacetime_unknown"] = f"{row['rss_MB'] * 2 ** 20 / row['spacetime_unknowns']:.0f}"
    row["dense_predicted_excess_MB"] = f"{(12.0 * row['m'] ** 2 + 8.0 * nt * N * C) / 2 ** 20:.1f}"
    rows.append(row)
    print(" ".join(f"{k}={v}" for k, v in row.items()), flush=True)


for reg in REGS:
    for n, nts in ((4, (32, 64, 128, 256, 512)), (6, (32, 64)), (8, (32, 128))):
        for nt in nts:
            run(n, nt, reg, "iterative")
            if nt * (2 * n - 1) ** 2 <= 8000:
                run(n, nt, reg, "dense")
with open(f"{D}/../t11h1_memory.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
