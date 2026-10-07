#!/usr/bin/env python3
"""T6 step 0: negative/positive tests of the KKT acceptance guard
(DistributedOptimalControlSolve::accepted).  LP 5.2, eps = 1/500, CN, nx 16, Nt 32.
  neg_air_breakdown : hypre AIR without max_row_sum 1.0 (T5b H2) -- the T5b
                      false-convergence case (outer reason 3, true residual 0.18)
  neg_maxit1        : sweep spatial solves starved (Jacobi, ksp_max_it 1)
  neg_default_gamg  : production default spatial GAMG (fails at this Pe, T5)
  pos_lu            : LU on all four sweep prefixes (must be accepted, exit 0)
Expected: every neg_* exits 8 with an EX21_SOLVE_REJECTED line; pos_lu exits 0
with no such line.   WORK=<scratch> guard_test.py  -> logs/guard_*.log, results/guard.csv"""
import csv, os, re, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "t5"))
import run_t5 as r5
X = "/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21"
HX = "/home/giacomo/software/FEMuS/femusbin-control-hypre/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21"
WORK = os.environ.get("WORK", "/tmp/t6_guard")
os.makedirs(os.path.join(WORK, "output"), exist_ok=True)
r5.CONFIGS.update({
    "air_h2": "-pc_type hypre -pc_hypre_type boomeramg -pc_hypre_boomeramg_restriction_type 1"
              " -pc_hypre_boomeramg_grid_sweeps_down 0 -pc_hypre_boomeramg_grid_sweeps_up 3"
              " -pc_hypre_boomeramg_postrelax F,F,C -pc_hypre_boomeramg_relax_type_all l1scaled-Jacobi",
    "maxit1": "-pc_type jacobi -ksp_max_it 1",
})
CASES = [("neg_air_breakdown", HX, r5.expand("air_h2", r5.SWEEP_PREFIXES), False),
         ("neg_maxit1", X, r5.expand("maxit1", r5.SWEEP_PREFIXES), False),
         ("neg_default_gamg", X, "", False),
         ("pos_lu", X, r5.expand("lu_petsc", r5.SWEEP_PREFIXES), True)]
ARGS = ["--exact-oc", "lp52", "--exact-oc-nu", "0.002", "--time-scheme", "cn", "--nx", "16", "--nz", "16",
        "--steps", "32", "--distributed-solve", "--pc-schur-sweep", "--no-paraview", "--exact-oc-production-tol"]
rows, bad = [], 0
for name, exe, opts, expect_ok in CASES:
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PETSC_OPTIONS=opts)
    p = subprocess.run([exe, *ARGS], cwd=WORK, env=env, capture_output=True, text=True, timeout=1800)
    keep = [l for l in p.stdout.splitlines() if l.startswith(("EX21_SOLVE_REJECTED", "EX21_EXACT_SOLVER_STATS", "EX21_LP52_RESULT"))]
    errs = [l for l in p.stderr.splitlines() if l.startswith("ERROR: ex21")]
    with open(os.path.join(HERE, "logs", f"guard_{name}.log"), "w") as fh:
        fh.write(f"# PETSC_OPTIONS={opts}\n# {os.path.basename(exe)} {' '.join(ARGS)}\nEXIT={p.returncode}\n")
        fh.write("\n".join(keep + errs) + "\n")
    st = dict(re.findall(r"(\S+?)=(\S+)", next((l for l in keep if l.startswith("EX21_EXACT_SOLVER_STATS")), "")))
    rejected = any(l.startswith("EX21_SOLVE_REJECTED") for l in keep)
    ok = (p.returncode == 0 and not rejected) if expect_ok else (p.returncode == 8 and rejected and bool(errs))
    bad += not ok
    rows.append(dict(case=name, expected="accepted" if expect_ok else "rejected", exit=p.returncode,
                     rejected_line=int(rejected), stderr_error=int(bool(errs)), reason=st.get("reason"),
                     outer_its=st.get("outer_its"), sweep_failures=st.get("sweep_failures"),
                     true_residual=st.get("true_residual"), status=st.get("status"),
                     verdict="PASS" if ok else "FAIL", petsc_options=opts))
    print(rows[-1]["case"], rows[-1]["verdict"], {k: rows[-1][k] for k in ("exit", "reason", "outer_its", "sweep_failures", "true_residual", "status")}, flush=True)
with open(os.path.join(HERE, "results", "guard.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("GUARD", "PASS" if not bad else f"FAIL ({bad})")
