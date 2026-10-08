#!/usr/bin/env python3
"""T11D memory audit: dense vs matrix-free reduced Dirichlet core -> ../../t11d_memory.csv.

Production path, face zmin, wind A, CN, T = 1, tracking (0,1)^2 x (0.3,1).  Peak RSS
(/usr/bin/time MAXRSS) for the dense core (assembled K = R + beta H, m x m, Cholesky:
T8D accounting 12 B m^2 + 8 B Nt N |C|) and the iterative core (CG on the shell, sparse
R_T factor: no m x m object).  m = Nt |C|, |C| = (2n-1)^2.
Checks: (1) the dense excess RSS_dense - RSS_iter grows like m^2 and matches the
predicted 12 B m^2 + 8 B Nt N |C| (ratio reported); (2) the iterative RSS is linear in
Nt (per-slot increment constant: RSS_iter(Nt) - RSS_iter(Nt/2) ~ doubles each step,
i.e. bytes per space-time DOF constant) -- no O(m^2) storage."""
import csv, os, re, subprocess
X = os.environ.get("EX21_BIN", "/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21")
W = os.environ.get("T11D_WORK", "/tmp/t11d_mem"); os.makedirs(W + "/output", exist_ok=True)
D = os.path.dirname(os.path.abspath(__file__)) + "/.."
os.makedirs(D + "/logs/memory", exist_ok=True)
ENV = {"HOME": os.environ["HOME"], "PATH": "/home/giacomo/software/petsc/arch-linux-c-openblas-opt/bin:/usr/bin:/bin",
       "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
rows = []


def run(n, nt, core):
    tag = f"{core}_n{n}_nt{nt}"
    args = [X, "--dimension", "3", "--dirichlet-solve", "--dirichlet-face3d", "zmin", "--state-bc", "dirichlet",
            "--tracking-box3d", "0", "1", "0", "1", "0.3", "1", "--target-type", "sin_x_minus_t", "--T", "1",
            "--nx", str(n), "--ny", str(n), "--nz", str(n), "--steps", str(nt), "--time-scheme", "cn",
            "--convection", "on", "--vx", "1", "--vy", "0.5", "--vz", "-0.75", "--dirichlet-core-solver", core, "--no-paraview"]
    r = subprocess.run(["/usr/bin/time", "-f", "MAXRSS=%M wall=%e"] + args, cwd=W, env=ENV, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, universal_newlines=True)
    s = r.stdout
    open(f"{D}/logs/memory/{tag}.log", "w").write(s)
    g = lambda p: (re.search(p, s) or [None, ""])[1] if re.search(p, s) is None else re.search(p, s).group(1)
    N, C = (2 * n + 1) ** 3, (2 * n - 1) ** 2
    return {"n": n, "nt": nt, "core": core, "N_spatial": N, "nC": C, "m": nt * C, "spacetime_dofs": 2 * nt * N,
            "exit": r.returncode, "reason": g(r"EX20_PAR_SOLVE .*? reason=(\S+)"), "core_its": g(r"core_its_max=(\d+)"),
            "rss_MB": int(g(r"MAXRSS=(\d+)") or 0) / 1024.0, "wall_s": float(g(r"wall=(\S+)") or 0),
            "J_track": g(r"J_track=(\S+)")}


plan = [(4, nt) for nt in (32, 64, 128, 256)] + [(6, nt) for nt in (16, 32, 64)] + [(8, nt) for nt in (8, 16, 32)]
for n, nt in plan:
    dense, it = run(n, nt, "dense"), run(n, nt, "iterative")
    pred = (12.0 * dense["m"] ** 2 + 8.0 * nt * dense["N_spatial"] * dense["nC"]) / 2 ** 20
    for r in (dense, it):
        r["dense_excess_MB"] = f"{dense['rss_MB'] - it['rss_MB']:.1f}"
        r["predicted_excess_MB"] = f"{pred:.1f}"
        r["measured/predicted"] = f"{(dense['rss_MB'] - it['rss_MB']) / pred:.3f}" if pred > 50 else ""
        r["bytes_per_spacetime_dof"] = f"{r['rss_MB'] * 2 ** 20 / r['spacetime_dofs']:.0f}"
        r["J_rel_diff_dense"] = f"{abs(float(it['J_track']) - float(dense['J_track'])) / abs(float(dense['J_track'])):.1e}"
        rows.append(r)
        print(" ".join(f"{k}={v}" for k, v in r.items()), flush=True)
# iterative-only extension (the dense core would need >= 7 GB here)
for n, nt in ((4, 512), (4, 1024), (8, 64), (8, 128)):
    it = run(n, nt, "iterative")
    it["predicted_excess_MB"] = f"{(12.0 * it['m'] ** 2 + 8.0 * nt * it['N_spatial'] * it['nC']) / 2 ** 20:.1f}"
    it["dense_excess_MB"] = "not run (dense would need the predicted excess)"
    it["measured/predicted"] = ""
    it["bytes_per_spacetime_dof"] = f"{it['rss_MB'] * 2 ** 20 / it['spacetime_dofs']:.0f}"
    it["J_rel_diff_dense"] = ""
    rows.append(it)
    print(" ".join(f"{k}={v}" for k, v in it.items()), flush=True)
with open(D + "/../t11d_memory.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
