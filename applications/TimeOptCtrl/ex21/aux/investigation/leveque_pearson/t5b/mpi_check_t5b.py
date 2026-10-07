#!/usr/bin/env python3
"""T5b MPI check (copy of t5/mpi_check.py run on the hypre binary).  T5 optional MPI consistency check of the MPI-capable spatial backends on the
PRODUCTION path (--exact-oc lp52 --solve-only: preset data, distributed solve,
JSON output).  LP 5.2, eps = 1/100, CN, Q2 nx = 32, Nt = 64, beta = 1e-2.

    mpi_check.py   -> results/mpi_check.csv, printed table

Compares every np run with np = 1 of the same backend: outer residual history,
J_track / J_reg, and max |dy|, |dp| over all slots (matched by coordinates).
Global ILU (PCILU) is sequential-only in PETSc and is therefore not run here."""
import csv, json, os, re, subprocess, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_t5b as r
r5 = r.r5
MPI = "/home/giacomo/software/petsc/arch-linux-c-openblas-opt/bin/mpirun"
WORK = os.environ.get("WORK", "/tmp/t5bmpi")
CFGS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["hypre_air1_mrs1"]
NPS = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1, 2, 4]
TAG = sys.argv[3] if len(sys.argv) > 3 else ""
TIMEOUT = int(os.environ.get("MPI_TIMEOUT", "300"))   # np = 2 needs ~25 s; a hang is recorded, not waited on
os.makedirs(os.path.join(WORK, "output"), exist_ok=True)
rows = []
base = {}
for cfg in CFGS:
    for np_ in NPS:
        js = os.path.join(WORK, f"{cfg}_np{np_}.json")
        opts = "-ksp_monitor_true_residual " + r5.expand(cfg, r5.SWEEP_PREFIXES)
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PETSC_OPTIONS=opts)
        t = time.time()
        p = subprocess.Popen([MPI, "-np", str(np_), r5.EXE, "--exact-oc", "lp52", "--exact-oc-nu", "0.01",
                            "--time-scheme", "cn", "--nx", "32", "--nz", "32", "--steps", "64",
                            "--distributed-solve", "--pc-schur-sweep", "--no-paraview", "--solve-only",
                            "--dump-json", js], cwd=WORK, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, start_new_session=True)
        try:
            text, _ = p.communicate(timeout=TIMEOUT)
            status = "done"
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, 9)
            text, _ = p.communicate()
            status = f"HANG(>{TIMEOUT}s, killed)"
        wall = time.time() - t
        with open(os.path.join(HERE, "logs", f"mpi_{cfg}_np{np_}{TAG}.log"), "w") as fh:
            fh.write("# PETSC_OPTIONS=" + opts + "\n")
            fh.write("\n".join(l for l in text.splitlines() if "unprecond" in l or l.startswith("EX2") or "rror" in l or "ignal" in l or "SEGV" in l))
        if status != "done" or not os.path.exists(js) or os.path.getmtime(js) < t:
            rec = dict(cfg=cfg, np=np_, exit=p.returncode, status=status, wall_s=f"{wall:.1f}",
                       note=" | ".join(l for l in text.splitlines() if "ignal" in l or "SEGV" in l or "rror" in l)[:300])
            rows.append(rec)
            print(rec, flush=True)
            continue
        hist = re.findall(r"\|\|r\(i\)\|\|/\|\|b\|\| (\S+)", text)
        d = json.load(open(js))
        xy = np.array(d["spatial_coordinates"])
        order = np.lexsort((xy[:, 1], xy[:, 0])) if xy.ndim == 2 else np.argsort(xy)
        Y = np.array([s["values"] for s in d["state"]])[:, order]
        P = np.array([s["values"] for s in d["adjoint"]])[:, order]
        rec = dict(cfg=cfg, np=np_, exit=p.returncode, status=status, wall_s=f"{wall:.1f}", outer_its=d["solve"]["iterations"],
                   reason=d["solve"]["reason"], true_residual=d["solve"]["true_residual"],
                   adjoint_rel=d["kkt"]["adjoint_kkt_residual_rel"], J_track=d["functional"]["J_track"],
                   J_reg=d["functional"]["J_reg"], outer_history=" ".join(f"{float(h):.3e}" for h in hist))
        if np_ == 1 or cfg not in base:
            base[cfg] = (Y, P, rec)
        else:
            Y1, P1, r1 = base[cfg]
            rec["max_dy_vs_np1"] = f"{np.abs(Y - Y1).max():.2e}"
            rec["max_dp_vs_np1"] = f"{np.abs(P - P1).max():.2e}"
            rec["J_track_rel_vs_np1"] = f"{abs(rec['J_track'] - r1['J_track']) / r1['J_track']:.1e}"
        rows.append(rec)
        print(rec, flush=True)
keys = []
for r in rows:
    keys += [k for k in r if k not in keys]
with open(os.path.join(HERE, "results", f"mpi_check_t5b{TAG}.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)
