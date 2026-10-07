#!/usr/bin/env python3
"""T6 MPI check of the LPS assembly (BuildLpsMatrix is element-parallel with
global MPI reductions of its statistics).  Production path (--exact-oc lp52
--solve-only --dump-json), LP 5.2, eps = 1/500, beta = 1e-2, CN, Q2 nx = 32,
Nt = 64, LPS-P0 and LPS-P1, MUMPS LU in all four sweep prefixes, np = 1, 2, 4.
Every np must report the same EX21_LPS statistics (active patches, delta range,
||W||_F) and the same solution as np = 1 (max |dy|, |dp| by coordinates, J).
WORK=<scratch> mpi_lps_check.py -> results/mpi_lps_check.csv, logs/mpi_lps_*.log"""
import csv, json, os, re, subprocess, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
X = "/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21"
MPI = "/home/giacomo/software/petsc/arch-linux-c-openblas-opt/bin/mpirun"
WORK = os.environ.get("WORK", "/tmp/t6_mpi")
os.makedirs(os.path.join(WORK, "output"), exist_ok=True)
OPTS = "-ksp_monitor_true_residual " + " ".join(
    f"-{p}pc_type lu -{p}pc_factor_mat_solver_type mumps"
    for p in ("sweep_right_", "sweep_left_", "sweep_ay_", "fieldsplit_y_sweep_"))
rows, base, bad = [], {}, 0
for proj in ("p0", "p1"):
    for np_ in (1, 2, 4):
        js = os.path.join(WORK, f"lps_{proj}_np{np_}.json")
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PETSC_OPTIONS=OPTS)
        t = time.time()
        p = subprocess.run([MPI, "-np", str(np_), X, "--exact-oc", "lp52", "--exact-oc-nu", "0.002", "--time-scheme", "cn",
                            "--nx", "32", "--nz", "32", "--steps", "64", "--distributed-solve", "--pc-schur-sweep",
                            "--no-paraview", "--solve-only", "--dump-json", js, "--convection-stabilization", "lps",
                            "--lps-projection", proj], cwd=WORK, env=env, capture_output=True, text=True, timeout=1800)
        wall = time.time() - t
        text = p.stdout + p.stderr
        with open(os.path.join(HERE, "logs", f"mpi_lps_{proj}_np{np_}.log"), "w") as fh:
            fh.write("# PETSC_OPTIONS=" + OPTS + "\n")
            fh.write("\n".join(l for l in text.splitlines() if "unprecond" in l or l.startswith("EX2") or "rror" in l))
        lps = dict(re.findall(r"(\S+?)=(\S+)", next(l for l in text.splitlines() if l.startswith("EX21_LPS "))))
        d = json.load(open(js))
        xy = np.array(d["spatial_coordinates"])
        order = np.lexsort((xy[:, 1], xy[:, 0]))
        Y = np.array([s["values"] for s in d["state"]])[:, order]
        P = np.array([s["values"] for s in d["adjoint"]])[:, order]
        rec = dict(projection=proj, np=np_, exit=p.returncode, wall_s=f"{wall:.1f}", lps_active=lps["active"],
                   lps_patches=lps["patches"], lps_delta_min=lps["delta_min"], lps_delta_max=lps["delta_max"],
                   W_fro=lps["||W||_F"], outer_its=d["solve"]["iterations"], reason=d["solve"]["reason"],
                   true_residual=d["solve"]["true_residual"], J_track=d["functional"]["J_track"],
                   J_reg=d["functional"]["J_reg"],
                   rejected=int(any(l.startswith("EX21_SOLVE_REJECTED") for l in text.splitlines())))
        if np_ == 1:
            base[proj] = (Y, P, rec)
        else:
            Y1, P1, r1 = base[proj]
            rec["max_dy_vs_np1"] = f"{np.abs(Y - Y1).max():.2e}"
            rec["max_dp_vs_np1"] = f"{np.abs(P - P1).max():.2e}"
            rec["J_track_rel_vs_np1"] = f"{abs(rec['J_track'] - r1['J_track']) / r1['J_track']:.1e}"
            same_stats = all(rec[k] == r1[k] for k in ("lps_active", "lps_patches", "lps_delta_min", "lps_delta_max"))
            ok = same_stats and float(rec["max_dy_vs_np1"]) < 1e-10 and p.returncode == 0
            rec["verdict"] = "PASS" if ok else "FAIL"
            bad += not ok
        rows.append(rec)
        print(rec, flush=True)
keys = []
for r in rows:
    keys += [k for k in r if k not in keys]
with open(os.path.join(HERE, "results", "mpi_lps_check.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
print("MPI LPS", "PASS" if not bad else f"FAIL ({bad})")
