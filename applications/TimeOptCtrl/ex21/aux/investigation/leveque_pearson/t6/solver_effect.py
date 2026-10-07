#!/usr/bin/env python3
"""T6 step 14/15 (secondary): does W_LPS change the spatial sweep blocks'
conditioning?  T5 block harness (--exact-oc-block-test: blocks E, right, left for
theta = 1 and 1/2; 3 random + 1 smooth RHS; GMRES(50), right PC, rtol 1e-10,
maxit 500), the SAME points as T5/T5b, unstabilised vs LPS (p0):
  gamg_base        production default spatial GAMG (production binary)
  hypre_air1_mrs1  T5b H2c, AIR + max_row_sum 1.0 (hypre binary; serial only),
                   with the BoomerAMG level count per block (print_statistics)
No LPS tuning.   WORK=<scratch> solver_effect.py -> results/solver_effect.csv"""
import csv, os, re, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
LP = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(LP, "t5b"))
import run_t5b as r                      # imports t5/run_t5 and registers the hypre configs
r5 = r.r5
X = "/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21"
HX = "/home/giacomo/software/FEMuS/femusbin-control-hypre/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21"
WORK = os.environ.get("WORK", "/tmp/t6_solver")
os.makedirs(os.path.join(WORK, "output"), exist_ok=True)
POINTS = dict(r5.POINTS)
POINTS.update({"E16": ("500", 16), "D64": ("500", 64)})
NAMES = ["t1_E", "t1_right", "t1_left", "th_E", "th_right", "th_left"]
rows = []
for cfg, exe in (("gamg_base", X), ("hypre_air1_mrs1", HX)):
    for pt, (e, nx) in POINTS.items():
        for stab in ("none", "p0"):
            opts = r5.expand(cfg, ("block_test_",))
            if cfg.startswith("hypre"):
                opts += " -block_test_pc_hypre_boomeramg_print_statistics"
            args = ["--exact-oc", "lp52", "--exact-oc-nu", repr(r5.EPS[e]), "--nx", str(nx), "--nz", str(nx),
                    "--steps", str(2 * nx), "--distributed-solve", "--no-paraview", "--exact-oc-block-test"]
            if stab == "p0":
                args += ["--convection-stabilization", "lps"]
            env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PETSC_OPTIONS=opts)
            p = subprocess.run(["/usr/bin/time", "-f", "MAXRSS_KB=%M", exe, *args], cwd=WORK, env=env,
                               capture_output=True, text=True, timeout=3600)
            text = p.stdout + p.stderr
            tables = re.split(r"Operator Matrix Information:", text)[1:]
            levels = [len(re.findall(r"^\s*\d+\s+\d+\s+\d+\s+[\d.]+\s+\d+\s+\d+",
                                     t.split("Interpolation Matrix Information")[0], re.M)) for t in tables[:6]]
            lps = dict(re.findall(r"(\S+?)=(\S+)", next((l for l in text.splitlines() if l.startswith("EX21_LPS ")), "")))
            blocks = [dict(re.findall(r"(\S+?)=(\S+)", l)) for l in text.splitlines() if l.startswith("EX21_BLOCK ")]
            for k, b in enumerate(blocks):
                rows.append(dict(cfg=cfg, stab=stab, point=pt, eps=f"1/{e}", nx=nx, theta=b.get("theta"),
                                 block=b.get("block"), Pe_cell=b.get("Pe_cell"), its=b.get("its"),
                                 max_its=b.get("max_its"), reasons=b.get("reasons"),
                                 converged_all=int(all(int(v) > 0 for v in b.get("reasons", "-1").split("/"))),
                                 levels=levels[k] if k < len(levels) else "", setup_s=b.get("setup_s"),
                                 solve_s=b.get("solve_s"), lps_active=lps.get("active", ""),
                                 lps_patches=lps.get("patches", ""), petsc_options=opts))
            print(cfg, stab, pt, [f"{b.get('max_its')}[{b.get('reasons')}]" for b in blocks],
                  "levels", levels if levels else "-", flush=True)
with open(os.path.join(HERE, "results", "solver_effect.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
