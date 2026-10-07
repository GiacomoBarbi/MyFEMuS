#!/usr/bin/env python3
"""T7N MPI consistency: production path (--exact-oc neumann-conv --solve-only
--dump-json), wind A (1, 0.5), CN, nx = 16, Nt = 32, MUMPS LU sweeps, np = 1, 2, 4.
Compares y, p, u (all slots, matched by coordinates), J_track, J_reg and the
outer residual history against np = 1.   WORK=<scratch> t7n_mpi.py"""
import csv, json, os, re, subprocess, time
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
X = "/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21"
MPI = "/home/giacomo/software/petsc/arch-linux-c-openblas-opt/bin/mpirun"
WORK = os.environ.get("WORK", "/tmp/t7n_mpi"); os.makedirs(os.path.join(WORK, "output"), exist_ok=True)
OPTS = "-ksp_monitor_true_residual " + " ".join(f"-{p}pc_type lu -{p}pc_factor_mat_solver_type mumps"
                                               for p in ("sweep_right_", "sweep_left_", "sweep_ay_", "fieldsplit_y_sweep_"))
rows, base, bad = [], None, 0
for np_ in (1, 2, 4):
    js = os.path.join(WORK, f"np{np_}.json")
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PETSC_OPTIONS=OPTS)
    t = time.time()
    p = subprocess.run([MPI, "-np", str(np_), X, "--exact-oc", "neumann-conv", "--exact-oc-velocity", "1", "0.5",
                        "--time-scheme", "cn", "--nx", "16", "--nz", "16", "--steps", "32", "--distributed-solve",
                        "--pc-schur-sweep", "--no-paraview", "--solve-only", "--dump-json", js],
                       cwd=WORK, env=env, capture_output=True, text=True, timeout=1800)
    text = p.stdout + p.stderr
    with open(os.path.join(HERE, "logs", f"mpi_np{np_}.log"), "w") as fh:
        fh.write("# PETSC_OPTIONS=" + OPTS + "\n" + "\n".join(l for l in text.splitlines()
                 if "unprecond" in l or l.startswith("EX2") or "rror" in l))
    d = json.load(open(js))
    xy = np.array(d["spatial_coordinates"]); o = np.lexsort((xy[:, 1], xy[:, 0]))
    F = {k: np.array([s["values"] for s in d[k]])[:, o] for k in ("state", "adjoint", "control_full")}
    hist = [float(h) for h in re.findall(r"\|\|r\(i\)\|\|/\|\|b\|\| (\S+)", text)]
    rec = dict(np=np_, exit=p.returncode, wall_s=f"{time.time() - t:.1f}", control=d["config"].get("control", ""),
               outer_its=d["solve"]["iterations"], reason=d["solve"]["reason"], true_residual=d["solve"]["true_residual"],
               J_track=d["functional"]["J_track"], J_reg=d["functional"]["J_reg"],
               rejected=int("EX21_SOLVE_REJECTED" in text), outer_history=" ".join(f"{h:.4e}" for h in hist))
    if base is None:
        base = (F, rec, hist)
    else:
        F1, r1, h1 = base
        for k, nm in (("state", "y"), ("adjoint", "p"), ("control_full", "u")):
            rec[f"max_d{nm}_vs_np1"] = f"{np.abs(F[k] - F1[k]).max():.2e}"
        rec["J_track_rel_vs_np1"] = f"{abs(rec['J_track'] - r1['J_track']) / r1['J_track']:.1e}"
        rec["J_reg_rel_vs_np1"] = f"{abs(rec['J_reg'] - r1['J_reg']) / r1['J_reg']:.1e}"
        rec["history_max_rel_vs_np1"] = (f"{max(abs(a - b) / max(abs(b), 1e-300) for a, b in zip(hist, h1)):.1e}"
                                         if len(hist) == len(h1) else "length differs")
        ok = (p.returncode == 0 and max(float(rec[f"max_d{n}_vs_np1"]) for n in "ypu") < 1e-10
              and len(hist) == len(h1))
        rec["verdict"] = "PASS" if ok else "FAIL"; bad += not ok
    rows.append(rec); print(rec, flush=True)
keys = []
for r in rows: keys += [k for k in r if k not in keys]
with open(os.path.join(HERE, "results", "mpi.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
print("MPI", "PASS" if not bad else f"FAIL ({bad})")
