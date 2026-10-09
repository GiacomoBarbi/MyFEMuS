#!/usr/bin/env python3
"""T13 memory audit on face unions -> ../../t13_memory.csv.

DIRICHLET (production path, wind A, CN, T = 1, tracking (0.2,0.8)^3, n = 4): the reduced core size grows with
the union, m = Nt |C|, |C| = the geometric controlled set (lattice count).  Dense core (assembled K, m x m,
Cholesky; T8D accounting 12 B m^2 + 8 B Nt N |C|) vs iterative core (CG on the shell, sparse R_T factor).
Checks: (1) dense excess RSS_dense - RSS_iter matches the predicted 12 B m^2 + 8 B Nt N |C| (ratio); (2) the
iterative RSS per space-time DOF stays bounded as |C| grows from one face to all six (no O(m^2) object),
and is linear in Nt.
NEUMANN (SCHUR_SWEEP + LU spatial solves): RSS vs the number of faces at fixed n, Nt (N = M_Gamma is a sparse
spatial matrix; the union only adds its nonzeros)."""
import csv, itertools, os, re, subprocess
X = os.environ.get("EX21_BIN", "/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21")
W = os.environ.get("T13_WORK", "/tmp/t13_mem"); os.makedirs(W + "/output", exist_ok=True)
D = os.path.dirname(os.path.abspath(__file__)) + "/.."
os.makedirs(D + "/logs/memory", exist_ok=True)
ENV = {"HOME": os.environ["HOME"], "PATH": "/home/giacomo/software/petsc/arch-linux-c-openblas-opt/bin:/usr/bin:/bin",
       "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
LU = "-sweep_right_pc_type lu -sweep_left_pc_type lu -sweep_ay_pc_type lu -fieldsplit_y_sweep_pc_type lu"
ALL = ["zmin", "ymin", "xmax", "ymax", "xmin", "zmax"]
rows = []


def ncontrol(n, faces, kind):
    cnt = 0
    for i, j, k in itertools.product(range(2 * n + 1), repeat=3):
        on = {"xmin": i == 0, "xmax": i == 2 * n, "ymin": j == 0, "ymax": j == 2 * n, "zmin": k == 0, "zmax": k == 2 * n}
        if any(on[f] for f in faces) and (kind == "N" or not any(on[f] for f in ALL if f not in faces)):
            cnt += 1
    return cnt


def g(p, s):
    m = re.search(p, s)
    return m.group(1) if m else ""


def run(kind, faces, n, nt, core=""):
    tag = f"{kind}_{'+'.join(faces) if len(faces) < 6 else 'all6'}_{core or 'sweep'}_n{n}_nt{nt}"
    args = [X, "--dimension", "3", "--nu", "0.25", "--alpha-u", "1e-2", "--target-type", "sin_x_minus_t", "--T", "1",
            "--tracking-box3d", "0.2", "0.8", "0.2", "0.8", "0.2", "0.8", "--nx", str(n), "--ny", str(n), "--nz", str(n),
            "--steps", str(nt), "--time-scheme", "cn", "--convection", "on", "--vx", "1", "--vy", "0.5", "--vz", "-0.75", "--no-paraview"]
    env = dict(ENV)
    if kind == "D":
        args += ["--dirichlet-solve", "--state-bc", "dirichlet", "--dirichlet-face3d", ",".join(faces), "--dirichlet-core-solver", core]
    else:
        args += ["--neumann-solve", "--state-bc", "natural", "--neumann-face3d", ",".join(faces), "--pc-schur-sweep"]
        env["PETSC_OPTIONS"] = LU
    r = subprocess.run(["/usr/bin/time", "-f", "MAXRSS=%M wall=%e"] + args, cwd=W, env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, universal_newlines=True)
    s = r.stdout
    open(f"{D}/logs/memory/{tag}.log", "w").write(s)
    N = (2 * n + 1) ** 3
    C = ncontrol(n, faces, kind)
    return {"kind": "Dirichlet" if kind == "D" else "Neumann", "faces": "+".join(faces) if len(faces) < 6 else "all six",
            "n_faces": len(faces), "n": n, "nt": nt, "core": core or "SCHUR_SWEEP", "N_spatial": N, "nC": C, "m": nt * C,
            "spacetime_dofs": 2 * nt * N, "exit": r.returncode, "reason": g(r"EX20_PAR_SOLVE .*? reason=(\S+)", s),
            "core_its": g(r"core_its_max=(\d+)", s), "rss_MB": int(g(r"MAXRSS=(\d+)", s) or 0) / 1024.0,
            "wall_s": float(g(r"wall=(\S+)", s) or 0), "J_track": g(r"J_track=(\S+)", s)}


SETS = [["zmin"], ["zmin", "zmax"], ["zmin", "xmin"], ["zmin", "ymin", "xmin"], ALL]
for faces in SETS:
    for nt in (16, 32):
        dense, it = run("D", faces, 4, nt, "dense"), run("D", faces, 4, nt, "iterative")
        pred = (12.0 * dense["m"] ** 2 + 8.0 * nt * dense["N_spatial"] * dense["nC"]) / 2 ** 20
        for r in (dense, it):
            r["dense_excess_MB"] = f"{dense['rss_MB'] - it['rss_MB']:.1f}"
            r["predicted_excess_MB"] = f"{pred:.1f}"
            r["measured/predicted"] = f"{(dense['rss_MB'] - it['rss_MB']) / pred:.3f}" if pred > 50 else ""
            r["bytes_per_spacetime_dof"] = f"{r['rss_MB'] * 2 ** 20 / r['spacetime_dofs']:.0f}"
            r["J_rel_diff_dense"] = f"{abs(float(it['J_track']) - float(dense['J_track'])) / abs(float(dense['J_track'])):.1e}"
            rows.append(r)
            print(" ".join(f"{k}={v}" for k, v in r.items()), flush=True)
    for nt in (64, 128, 256):   # iterative only (the dense core would need the predicted excess)
        it = run("D", faces, 4, nt, "iterative")
        it["dense_excess_MB"] = "not run"
        it["predicted_excess_MB"] = f"{(12.0 * it['m'] ** 2 + 8.0 * nt * it['N_spatial'] * it['nC']) / 2 ** 20:.1f}"
        it["measured/predicted"] = ""
        it["bytes_per_spacetime_dof"] = f"{it['rss_MB'] * 2 ** 20 / it['spacetime_dofs']:.0f}"
        it["J_rel_diff_dense"] = ""
        rows.append(it)
        print(" ".join(f"{k}={v}" for k, v in it.items()), flush=True)
for faces in SETS:
    for nt in (32, 64):
        r = run("N", faces, 4, nt)
        r.update({"dense_excess_MB": "", "predicted_excess_MB": "", "measured/predicted": "",
                  "bytes_per_spacetime_dof": f"{r['rss_MB'] * 2 ** 20 / r['spacetime_dofs']:.0f}", "J_rel_diff_dense": ""})
        rows.append(r)
        print(" ".join(f"{k}={v}" for k, v in r.items()), flush=True)
with open(D + "/../t13_memory.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
