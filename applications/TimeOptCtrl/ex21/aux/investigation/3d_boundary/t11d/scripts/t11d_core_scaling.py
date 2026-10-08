#!/usr/bin/env python3
"""T11D: iteration scaling of the matrix-free reduced Dirichlet core -> ../../t11d_core_scaling.csv.

Production path, face zmin, target sin_x_minus_t, T = 1, tracking omega = (0,1)^2 x (0.3,1)
(adjacent to Gamma_c, i.e. the strongest tracking/control coupling the free-DOF tracking
allows), default core: CG + sparse R_T Cholesky preconditioner, rtol 1e-8.
Scans: mesh n = 2..10 (Nt = 32), Nt = 8..512 (n = 4), alpha_u = 1e-1..1e-6 (n = 4, Nt = 32),
each for heat and wind A = (1, 0.5, -0.75) (and the normal wind (0,0,-0.75) on the mesh scan),
BE and CN.  For the alpha_u scan at n = 4 the dense gold core (--verify-dirichlet-core) also
reports cond(K) and cond(R^-1 K), the quantity that bounds CG."""
import csv, os, re, subprocess, sys
X = os.environ.get("EX21_BIN", "/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21")
W = os.environ.get("T11D_WORK", "/tmp/t11d_scal"); os.makedirs(W + "/output", exist_ok=True)
D = os.path.dirname(os.path.abspath(__file__)) + "/.."
os.makedirs(D + "/logs/scaling", exist_ok=True)
ENV = {"HOME": os.environ["HOME"], "PATH": "/home/giacomo/software/petsc/arch-linux-c-openblas-opt/bin:/usr/bin:/bin",
       "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
WINDS = {"heat": ["--convection", "off"], "wA": ["--convection", "on", "--vx", "1", "--vy", "0.5", "--vz", "-0.75"],
         "wN": ["--convection", "on", "--vx", "0", "--vy", "0", "--vz", "-0.75"]}
rows = []


def g(p, s):
    m = re.search(p, s, re.M)
    return m.group(1) if m else ""


def run(study, n, nt, au, wind, scheme, verify=False):
    tag = f"{study}_{wind}_{scheme}_n{n}_nt{nt}_au{au}"
    args = [X, "--dimension", "3", "--dirichlet-solve", "--dirichlet-face3d", "zmin", "--state-bc", "dirichlet",
            "--tracking-box3d", "0", "1", "0", "1", "0.3", "1", "--target-type", "sin_x_minus_t", "--nu", "0.25",
            "--alpha-u", repr(au), "--T", "1", "--nx", str(n), "--ny", str(n), "--nz", str(n), "--steps", str(nt),
            "--time-scheme", scheme, "--no-paraview"] + WINDS[wind]
    r = subprocess.run(["/usr/bin/time", "-f", "MAXRSS=%M wall=%e"] + args, cwd=W, env=ENV, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, universal_newlines=True)
    s = r.stdout
    open(f"{D}/logs/scaling/{tag}.log", "w").write(s)
    row = {"study": study, "wind": wind, "scheme": scheme.upper(), "n": n, "nt": nt, "alpha_u": au,
           "nC": (2 * n - 1) ** 2, "m_core": nt * (2 * n - 1) ** 2, "exit": r.returncode,
           "reason": g(r"EX20_PAR_SOLVE .*? reason=(\S+)", s), "outer": g(r"outer_iterations=(\d+)", s),
           "core_its_max": g(r"core_its_max=(\d+)", s), "core_its_mean": g(r"core_its_mean=(\S+)", s),
           "core_failures": g(r"core_failures=(\d+)", s), "core_s": g(r"core_seconds=(\S+)", s),
           "true_residual": g(r"EX20_PAR_SOLVE .*? true_residual=(\S+)", s), "J_track": g(r"J_track=(\S+)", s),
           "maxrss_MB": int(g(r"MAXRSS=(\d+)", s) or 0) // 1024, "wall_s": g(r"wall=(\S+)", s),
           "cond_K": "", "cond_RinvK": ""}
    if verify:
        rv = subprocess.run(args + ["--dirichlet-core-solver", "dense", "--verify-dirichlet-core"], cwd=W, env=ENV,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        row["cond_K"], row["cond_RinvK"] = g(r"cond_K=(\S+)", rv.stdout), g(r"cond_RinvK=(\S+)", rv.stdout)
    rows.append(row)
    print(" ".join(f"{k}={v}" for k, v in row.items()), flush=True)


for wind in ("heat", "wA", "wN"):
    for n in (2, 4, 6, 8, 10):
        run("mesh", n, 32, 1e-2, wind, "cn")
for wind in ("heat", "wA"):
    for nt in (8, 32, 128, 512):
        run("time", 4, nt, 1e-2, wind, "cn")
    for au in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
        for scheme in ("be", "cn"):
            run("alpha", 4, 32, au, wind, scheme, verify=True)
with open(D + "/../t11d_core_scaling.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
