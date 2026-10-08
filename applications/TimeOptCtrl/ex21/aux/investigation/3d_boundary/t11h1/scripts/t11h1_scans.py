#!/usr/bin/env python3
"""T11H1 production scans (face zmin, target sin_x_minus_t, T = 1, alpha_u = 1e-2 unless scanned, tracking
omega = (0,1)^2 x (0.3,1) adjacent to Gamma_c, default solver: CG + sparse R_H1t Cholesky, rtol 1e-8):
  ell scan   ell in {0, 0.1, 0.3, 1, 3} (gamma = 0), n = 4 (and n = 6, 8 for ell = 0.3, 3), Nt = 32, CN, heat / wind A
  gamma scan gamma_u in {0, 1e-4, 1e-3, 1e-2, 1e-1} (ell = 0), n = 4, Nt = 32, BE / CN, heat / wind A
  core       iterations vs mesh n = 2..10 (Nt = 32), Nt = 8..512 (n = 4), alpha_u = 1e-1..1e-6 (n = 4, Nt = 32)
             for H1s (ell = 0.3), H1t (gamma = 1e-2), comb, heat / wind A, CN (alpha scan: BE too)
Control norms are EXACT discrete quadratic forms with the dumped face operators of the same mesh
(JSON and dump share the DOF order): ||u||^2 = dt sum_n u_n^T M_G u_n, ||grad_G u||^2 = dt sum_n u_n^T K_G u_n,
||u_t||^2 = sum_n (u_n - u_{n-1})^T M_G (u_n - u_{n-1}) / dt (u_0 = y_0/eta = 0 here), plus the max nodal
slot jump max|u_n - u_{n-1}| (temporal smoothness).  cond(R^-1 K) by the dense gold core (--verify-dirichlet-core)
where m = Nt |C| <= 3200.
usage: t11h1_scans.py ell|gamma|core"""
import csv, json, os, re, subprocess, sys
import numpy as np
import scipy.sparse as sps

X = os.environ.get("EX21_BIN", "/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21")
W = os.environ.get("T11H1_WORK", "/tmp/t11h1_scan"); os.makedirs(W + "/output", exist_ok=True)
D = os.path.dirname(os.path.abspath(__file__)) + "/.."
os.makedirs(D + "/logs/scans", exist_ok=True)
ENV = {"HOME": os.environ["HOME"], "PATH": "/home/giacomo/software/petsc/arch-linux-c-openblas-opt/bin:/usr/bin:/bin",
       "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
WINDS = {"heat": ["--convection", "off"], "wA": ["--convection", "on", "--vx", "1", "--vy", "0.5", "--vz", "-0.75"]}
face_ops = {}


def g(p, s):
    m = re.search(p, s, re.M)
    return m.group(1) if m else ""


def coo(path):
    with open(path) as fh:
        fh.readline(); r, c = map(int, fh.readline().split()); fh.readline()
        d = np.loadtxt(fh, ndmin=2)
    return sps.csr_matrix((d[:, 2], (d[:, 0].astype(int), d[:, 1].astype(int))), shape=(r, c))


def base(n, nt, au, scheme, wind):
    return ["--dimension", "3", "--dirichlet-solve", "--dirichlet-face3d", "zmin", "--state-bc", "dirichlet",
            "--tracking-box3d", "0", "1", "0", "1", "0.3", "1", "--target-type", "sin_x_minus_t", "--nu", "0.25",
            "--alpha-u", repr(au), "--T", "1", "--nx", str(n), "--ny", str(n), "--nz", str(n), "--steps", str(nt),
            "--time-scheme", scheme, "--no-paraview"] + WINDS[wind]


def ops(n):
    if n not in face_ops:
        p = f"{W}/ops_n{n}"
        subprocess.run([X] + base(n, 2, 1e-2, "be", "heat") + ["--dump-operators", p], cwd=W, env=ENV,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        face_ops[n] = (coo(p + "_M_Gamma.coo"), coo(p + "_K_Gamma.coo"))
    return face_ops[n]


def run(study, n, nt, au, scheme, wind, ell, gam, verify=False):
    tag = f"{study}_{wind}_{scheme}_n{n}_nt{nt}_au{au:g}_ell{ell:g}_g{gam:g}"
    jf = f"{W}/{tag}.json"
    args = base(n, nt, au, scheme, wind) + ["--control-h1-length", repr(ell), "--control-h1-time-weight", repr(gam)]
    r = subprocess.run(["/usr/bin/time", "-f", "MAXRSS=%M wall=%e", X] + args + ["--dump-json", jf], cwd=W, env=ENV,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    s = r.stdout
    open(f"{D}/logs/scans/{tag}.log", "w").write(s)
    row = {"study": study, "wind": wind, "scheme": scheme.upper(), "n": n, "nt": nt, "alpha_u": au, "ell": ell,
           "gamma_u": gam, "nC": (2 * n - 1) ** 2, "m_core": nt * (2 * n - 1) ** 2, "exit": r.returncode,
           "reason": g(r"EX20_PAR_SOLVE .*? reason=(\S+)", s), "outer": g(r"outer_iterations=(\d+)", s),
           "core_its_max": g(r"core_its_max=(\d+)", s), "core_failures": g(r"core_failures=(\d+)", s),
           "core_s": g(r"core_seconds=(\S+)", s), "true_residual": g(r"EX20_PAR_SOLVE .*? true_residual=(\S+)", s),
           "maxrss_MB": int(g(r"MAXRSS=(\d+)", s) or 0) // 1024, "wall_s": g(r"wall=(\S+)", s)}
    d = json.load(open(jf))
    M, K = ops(n)
    U = np.array([e["values"] for e in d["control_full"]]); dt = d["config"]["dt"]
    row["J_total"] = d["functional"]["J_total"]; row["J_track"] = d["functional"]["J_track"]
    row["u_L2L2"] = float(np.sqrt(dt * sum(u @ (M @ u) for u in U)))
    row["gradG_u_L2L2"] = float(np.sqrt(dt * sum(u @ (K @ u) for u in U)))
    dU = np.diff(np.vstack([np.zeros(U.shape[1]), U]), axis=0)
    row["u_t_L2L2"] = float(np.sqrt(sum(v @ (M @ v) for v in dU) / dt))
    row["max_slot_jump"] = float(np.abs(dU).max())
    row["gradG_over_u"] = row["gradG_u_L2L2"] / row["u_L2L2"] if row["u_L2L2"] > 0 else ""
    os.remove(jf)
    row["cond_K"] = row["cond_RinvK"] = ""
    if verify and nt * (2 * n - 1) ** 2 <= 3200:
        rv = subprocess.run([X] + args + ["--dirichlet-core-solver", "dense", "--verify-dirichlet-core"], cwd=W, env=ENV,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        row["cond_K"], row["cond_RinvK"] = g(r"cond_K=(\S+)", rv.stdout), g(r"cond_RinvK=(\S+)", rv.stdout)
    print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
    return row


what = sys.argv[1]
rows = []
if what == "ell":
    for wind in ("heat", "wA"):
        for ell in (0.0, 0.1, 0.3, 1.0, 3.0):
            rows.append(run("ell", 4, 32, 1e-2, "cn", wind, ell, 0.0, verify=True))
        for n in (6, 8):
            for ell in (0.3, 3.0):
                rows.append(run("ell_mesh", n, 32, 1e-2, "cn", wind, ell, 0.0))
    out = "t11h1_ell_scaling.csv"
elif what == "gamma":
    for wind in ("heat", "wA"):
        for scheme in ("be", "cn"):
            for gam in (0.0, 1e-4, 1e-3, 1e-2, 1e-1):
                rows.append(run("gamma", 4, 32, 1e-2, scheme, wind, 0.0, gam, verify=True))
    out = "t11h1_gamma_scaling.csv"
else:
    REG = {"H1s": (0.3, 0.0), "H1t": (0.0, 1e-2), "comb": (0.3, 1e-2)}
    for reg, (ell, gam) in REG.items():
        for wind in ("heat", "wA"):
            for n in (2, 4, 6, 8, 10):
                r = run("mesh", n, 32, 1e-2, "cn", wind, ell, gam); r["reg"] = reg; rows.append(r)
            for nt in (8, 32, 128, 512):
                r = run("time", 4, nt, 1e-2, "cn", wind, ell, gam); r["reg"] = reg; rows.append(r)
            for au in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
                for scheme in ("be", "cn"):
                    r = run("alpha", 4, 32, au, scheme, wind, ell, gam, verify=(wind == "wA")); r["reg"] = reg; rows.append(r)
    out = "t11h1_core_scaling.csv"
keys = list(dict.fromkeys(k for r in rows for k in r))
with open(f"{D}/../{out}", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
