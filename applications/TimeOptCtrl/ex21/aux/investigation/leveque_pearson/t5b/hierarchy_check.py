#!/usr/bin/env python3
"""T5b diagnostic: is the hypre preconditioner a genuine multilevel AMG at each
screen point?  Runs the block test with -pc_hypre_boomeramg_print_statistics and
parses, for each of the six blocks (theta = 1: E, right, left; theta = 1/2: E,
right, left), the number of levels, the grid and operator complexities.
A 1-level "hierarchy" means BoomerAMG found no strong couplings to coarsen
(it then degenerates to its coarse-level relaxation only).
    hierarchy_check.py  -> results/hierarchy.csv"""
import csv, os, re, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_t5b as r
WORK = os.environ.get("WORK", "/tmp/t5b_hier")
os.makedirs(os.path.join(WORK, "output"), exist_ok=True)
pts = dict(r.r5.POINTS)
pts["E16"] = ("500", 16)
rows = []
for cfg in r.HYPRE:
    for pt, (e, nx) in pts.items():
        opts = r.r5.expand(cfg, ("block_test_",)) + " -block_test_pc_hypre_boomeramg_print_statistics"
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PETSC_OPTIONS=opts)
        p = subprocess.run([r.r5.EXE, "--exact-oc", "lp52", "--exact-oc-nu", repr(r.EPS[e]), "--nx", str(nx),
                            "--nz", str(nx), "--steps", str(2 * nx), "--distributed-solve", "--no-paraview",
                            "--exact-oc-block-test"], cwd=WORK, env=env, capture_output=True, text=True, timeout=900)
        text = p.stdout
        # one "Operator Matrix Information" table per setup (6 blocks)
        tables = re.split(r"Operator Matrix Information:", text)[1:]
        comps = re.findall(r"Complexity:\s+grid = (\S+)\s+operator = (\S+)", text)
        names = ["t1_E", "t1_right", "t1_left", "th_E", "th_right", "th_left"]
        for k, t in enumerate(tables[:6]):
            body = t.split("Interpolation Matrix Information")[0]
            levels = len(re.findall(r"^\s*\d+\s+\d+\s+\d+\s+[\d.]+\s+\d+\s+\d+", body, re.M))
            g, o = comps[k] if k < len(comps) else ("", "")
            rows.append(dict(cfg=cfg, point=pt, eps=f"1/{e}", nx=nx, block=names[k], levels=levels,
                             grid_complexity=g, operator_complexity=o))
        print(cfg, pt, [x["levels"] for x in rows if x["cfg"] == cfg and x["point"] == pt], flush=True)
with open(os.path.join(HERE, "results", "hierarchy.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
