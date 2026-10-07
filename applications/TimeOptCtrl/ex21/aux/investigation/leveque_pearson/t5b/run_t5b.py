#!/usr/bin/env python3
"""T5b: hypre / BoomerAMG / AIR screening of the SCHUR_SWEEP spatial blocks.
Same harness as t5/run_t5.py (block test, prefixes, Krylov settings), run with
the ex21 binary linked against PETSc 3.23.5 + hypre 2.32.0
(PETSC_ARCH=arch-linux-c-openblas-hypre, build dir femusbin-control-hypre).

    run_t5b.py baseline [--jobs J]     T5 baselines (gamg_base, ilu1, lu_petsc) on the hypre
                                       binary at A-D: results/baseline.csv (must match T5)
    run_t5b.py screen   [--jobs J]     hypre configurations at A, B, C, D, E16: results/screen.csv
    run_t5b.py kkt <cfg>[,<cfg>] [--jobs J]   LP 5.2 KKT runs: results/kkt_<cfgs>.csv
    run_t5b.py show <cfg>              print the options of a configuration

hypre configurations (options WITHOUT prefix, verified against -help of this build):
  H0  hypre_default : BoomerAMG with PETSc's defaults (Falgout coarsening, classical
                      interpolation, strong threshold 0.25, 1 sweep, symmetric-SOR/Jacobi)
  H1  hypre_ns      : nonsymmetric-oriented: HMIS coarsening, ext+i interpolation,
                      forward l1-Gauss-Seidel relaxation (nonsymmetric), strong threshold 0.5
  H2  hypre_air1    : AIR, distance-1 approximate ideal restriction (restriction_type 1;
                      PETSc then forces one-point interpolation), no pre-smoothing,
                      post-relaxation F,F,C, l1-scaled Jacobi
  H2b hypre_air2    : as H2 with distance-2 AIR (restriction_type 2)
  H3  hypre_air1_gs : as H2 but forward l1-Gauss-Seidel relaxation
  H2c hypre_air1_mrs1: as H2 plus max_row_sum 1.0 (disables BoomerAMG's "row sum
                      close to the diagonal => all couplings weak" test, which
                      leaves the mass-dominated blocks uncoarsened; added after
                      the rowsum_probe.py diagnostic, then screened like the others)

    run_t5b.py screen1 <cfg> [--jobs J]   one configuration at A, B, C, D, E16 and the
                                          T5 scale points C64, D64, D128: results/screen_<cfg>.csv
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("EX21_EXE",
    "/home/giacomo/software/FEMuS/femusbin-control-hypre/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21")
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "t5"))
import run_t5 as r5

r5.HERE = HERE            # results and logs go to t5b/
r5.EXE = os.environ["EX21_EXE"]
B = "-pc_type hypre -pc_hypre_type boomeramg"
AIR = (" -pc_hypre_boomeramg_grid_sweeps_down 0 -pc_hypre_boomeramg_grid_sweeps_up 3"
       " -pc_hypre_boomeramg_postrelax F,F,C")
HYPRE = {
    "hypre_default": B,
    "hypre_ns": B + " -pc_hypre_boomeramg_coarsen_type HMIS -pc_hypre_boomeramg_interp_type ext+i"
                    " -pc_hypre_boomeramg_relax_type_all l1-Gauss-Seidel -pc_hypre_boomeramg_strong_threshold 0.5",
    "hypre_air1": B + " -pc_hypre_boomeramg_restriction_type 1" + AIR
                    + " -pc_hypre_boomeramg_relax_type_all l1scaled-Jacobi",
    "hypre_air2": B + " -pc_hypre_boomeramg_restriction_type 2" + AIR
                    + " -pc_hypre_boomeramg_relax_type_all l1scaled-Jacobi",
    "hypre_air1_gs": B + " -pc_hypre_boomeramg_restriction_type 1" + AIR
                    + " -pc_hypre_boomeramg_relax_type_all l1-Gauss-Seidel",
}
HYPRE_LATE = {
    "hypre_air1_mrs1": HYPRE["hypre_air1"] + " -pc_hypre_boomeramg_max_row_sum 1.0",
}
r5.CONFIGS.update(HYPRE)
r5.CONFIGS.update(HYPRE_LATE)
EPS = r5.EPS


def block_runs(cfgs, points):
    R = []
    for cfg in cfgs:
        for pt, (e, nx) in points.items():
            R.append((f"blk_{cfg}_{pt}", ["--exact-oc", "lp52", "--exact-oc-nu", repr(EPS[e]),
                      "--nx", str(nx), "--nz", str(nx), "--steps", str(2 * nx),
                      "--distributed-solve", "--no-paraview", "--exact-oc-block-test"],
                      r5.expand(cfg, ("block_test_",)), dict(cfg=cfg, point=pt, eps=f"1/{e}", nx=nx)))
    return R


def main():
    mode = sys.argv[1]
    if mode == "show":
        print(r5.CONFIGS[sys.argv[2]])
        print("sweep prefixes:", r5.expand(sys.argv[2], r5.SWEEP_PREFIXES))
        return
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 1
    for d in ("logs/campaign", "results"):
        os.makedirs(os.path.join(HERE, d), exist_ok=True)
    if mode == "baseline":
        items = block_runs(("gamg_base", "ilu1", "lu_petsc"), r5.POINTS)
        name = "baseline"
    elif mode == "screen":
        pts = dict(r5.POINTS)
        pts["E16"] = ("500", 16)
        items = block_runs(tuple(HYPRE), pts)
        name = "screen"
    elif mode == "screen1":
        pts = dict(r5.POINTS)
        pts.update({"E16": ("500", 16), "C64": ("100", 64), "D64": ("500", 64), "D128": ("500", 128)})
        items = block_runs((sys.argv[2],), pts)
        name = "screen_" + sys.argv[2]
    elif mode == "kkt":
        cfgs = sys.argv[2].split(",")
        items = r5.runs("kkt", cfgs)
        name = "kkt_" + "_".join(cfgs)
    from concurrent.futures import ThreadPoolExecutor
    import csv
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        out = [r for recs in pool.map(r5.one, items) for r in recs]
    keys = []
    for r in out:
        keys += [k for k in r if k not in keys]
    with open(os.path.join(HERE, "results", name + ".csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(out)
    print(f"{name}: DONE ({len(out)} rows)")


if __name__ == "__main__":
    main()
