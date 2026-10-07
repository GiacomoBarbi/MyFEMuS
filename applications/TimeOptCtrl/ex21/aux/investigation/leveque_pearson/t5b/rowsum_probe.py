#!/usr/bin/env python3
"""T5b diagnostic probe: BoomerAMG's max_row_sum (PETSc default 0.9) declares every
coupling of row i weak when |sum_j a_ij| > max_row_sum * |a_ii|.  Mass-dominated
blocks M/dt + theta L have row sums close to the diagonal, so no point is coarsened.
max_row_sum = 1.0 disables the test.  Runs the block test at the failing points with
max_row_sum 0.9 (default) and 1.0 and prints levels + iterations per block.
    rowsum_probe.py  -> logs/rowsum_probe.log, results/rowsum_probe.csv"""
import csv, os, re, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_t5b as r
WORK = os.environ.get("WORK", "/tmp/t5b_rowsum")
os.makedirs(os.path.join(WORK, "output"), exist_ok=True)
POINTS = {"C": ("100", 32), "D": ("500", 32), "E16": ("500", 16), "C16": ("100", 16), "D64": ("500", 64)}
NAMES = ["t1_E", "t1_right", "t1_left", "th_E", "th_right", "th_left"]
rows = []
log = open(os.path.join(HERE, "logs", "rowsum_probe.log"), "w")
for cfg in ("hypre_default", "hypre_air1"):
    for mrs in ("0.9", "1.0"):
        for pt, (e, nx) in POINTS.items():
            opts = (r.r5.expand(cfg, ("block_test_",)) + f" -block_test_pc_hypre_boomeramg_max_row_sum {mrs}"
                    " -block_test_pc_hypre_boomeramg_print_statistics")
            env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PETSC_OPTIONS=opts)
            p = subprocess.run([r.r5.EXE, "--exact-oc", "lp52", "--exact-oc-nu", repr(r.EPS[e]), "--nx", str(nx),
                                "--nz", str(nx), "--steps", str(2 * nx), "--distributed-solve", "--no-paraview",
                                "--exact-oc-block-test"], cwd=WORK, env=env, capture_output=True, text=True, timeout=1800)
            text = p.stdout
            tables = re.split(r"Operator Matrix Information:", text)[1:]
            lev = [len(re.findall(r"^\s*\d+\s+\d+\s+\d+\s+[\d.]+\s+\d+\s+\d+",
                                  t.split("Interpolation Matrix Information")[0], re.M)) for t in tables[:6]]
            blocks = [dict(re.findall(r"(\S+?)=(\S+)", l)) for l in text.splitlines() if l.startswith("EX21_BLOCK ")]
            for k, b in enumerate(blocks):
                rows.append(dict(cfg=cfg, max_row_sum=mrs, point=pt, eps=f"1/{e}", nx=nx, theta=b.get("theta"),
                                 block=b.get("block"), Pe_cell=b.get("Pe_cell"), levels=lev[k] if k < len(lev) else "",
                                 max_its=b.get("max_its"), reasons=b.get("reasons"), setup_s=b.get("setup_s"),
                                 solve_s=b.get("solve_s"), petsc_options=opts))
            line = (f"{cfg} max_row_sum={mrs} {pt} (eps 1/{e} nx {nx}) levels={lev} "
                    + " ".join(f"{b.get('max_its')}[{b.get('reasons')}]" for b in blocks))
            print(line, flush=True)
            log.write(line + "\n")
with open(os.path.join(HERE, "results", "rowsum_probe.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
