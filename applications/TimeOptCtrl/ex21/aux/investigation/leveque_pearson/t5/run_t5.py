#!/usr/bin/env python3
"""T5: screening of convection-robust spatial preconditioners for the SCHUR_SWEEP
diagonal blocks (no source change: PETSc options only).

    run_t5.py screen [--jobs J]          block screen  -> results/screen.csv
    run_t5.py scale [--jobs J]           survivors at larger/harder points -> results/scale.csv
    run_t5.py kkt <cfg>[,<cfg>...] [--jobs J]
                                          LP 5.2 KKT runs -> results/kkt_<cfgs>.csv
    run_t5.py show <cfg>                  print the PETSC_OPTIONS of a configuration

Every configuration is ONE logical spatial-solver setting.  It is expanded onto
  * the block-test KSP prefix  block_test_          (screen), or
  * all four sweep prefixes    sweep_right_ sweep_left_ sweep_ay_ fieldsplit_y_sweep_   (kkt).
The Krylov method is NOT changed: the code sets GMRES(50), right preconditioning,
unpreconditioned residual norm, rtol = 1e-10 (--spatial-tol default), maxits = 500,
zero initial guess (Preconditioners.cpp CreateSweepSpatialSolver, and the same in
the block test).  Only the preconditioner differs.

Block-screen matrix points (LP 5.2, tau = h_LP = 1/nx, beta = 1e-2):
  A  eps = 1/20,  nx = 32  (Pe_cell 1.25)   easy control
  B  eps = 1/20,  nx = 16  (Pe_cell 2.5)    first failure of the baseline
  C  eps = 1/100, nx = 32  (Pe_cell 6.25)   moderate convection
  D  eps = 1/500, nx = 32  (Pe_cell 31)     hard
  (each run solves E, right, left for theta = 1 and 1/2, 3 seeded random + 1 smooth RHS)
"""
import csv, os, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
EXE = os.environ.get("EX21_EXE",
    "/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21")
WORK = os.environ.get("WORK", "/tmp")
TIMEOUT = int(os.environ.get("T5_TIMEOUT", "900"))
SWEEP_PREFIXES = ("sweep_right_", "sweep_left_", "sweep_ay_", "fieldsplit_y_sweep_")

# logical configurations: options WITHOUT prefix ("" = production default GAMG)
CONFIGS = {
    # A. GAMG baseline and small, defensible changes for a nonsymmetric operator
    "gamg_base":      "",
    "gamg_sor":       "-pc_type gamg -mg_levels_ksp_type richardson -mg_levels_pc_type sor",
    "gamg_sor_unsm":  "-pc_type gamg -mg_levels_ksp_type richardson -mg_levels_pc_type sor -pc_gamg_agg_nsmooths 0",
    "gamg_sor_unsm4": "-pc_type gamg -mg_levels_ksp_type richardson -mg_levels_pc_type sor -pc_gamg_agg_nsmooths 0"
                      " -mg_levels_ksp_max_it 4 -pc_gamg_aggressive_coarsening 0",
    # E. whole-domain (serial) ILU
    "ilu0":           "-pc_type ilu -pc_factor_levels 0",
    "ilu1":           "-pc_type ilu -pc_factor_levels 1",
    "ilu2":           "-pc_type ilu -pc_factor_levels 2",
    # C. overlapping additive Schwarz + local ILU (4 subdomains emulated in serial)
    "asm4o1_ilu1":    "-pc_type asm -pc_asm_blocks 4 -pc_asm_overlap 1 -sub_pc_type ilu -sub_pc_factor_levels 1",
    "asm4o0_ilu1":    "-pc_type asm -pc_asm_blocks 4 -pc_asm_overlap 0 -sub_pc_type ilu -sub_pc_factor_levels 1",
    # D. block Jacobi + local ILU (4 blocks emulated in serial)
    "bjac4_ilu1":     "-pc_type bjacobi -pc_bjacobi_blocks 4 -sub_pc_type ilu -sub_pc_factor_levels 1",
    # F. direct LU (reference): PETSc native, MUMPS, UMFPACK
    "lu_petsc":       "-pc_type lu",
    "lu_mumps":       "-pc_type lu -pc_factor_mat_solver_type mumps",
    "lu_umfpack":     "-pc_type lu -pc_factor_mat_solver_type umfpack",
}
POINTS = {"A": ("20", 32), "B": ("20", 16), "C": ("100", 32), "D": ("500", 32)}
EPS = {"20": 1 / 20, "100": 1 / 100, "500": 1 / 500}


def expand(cfg, prefixes):
    opts = CONFIGS[cfg].split()
    out = []
    for p in prefixes:
        for o in opts:
            out.append("-" + p + o[1:] if o.startswith("-") else o)
    return " ".join(out)


def runs(group, cfgs):
    R = []
    if group == "screen":
        for cfg in CONFIGS:
            for pt, (e, nx) in POINTS.items():
                R.append((f"blk_{cfg}_{pt}", ["--exact-oc", "lp52", "--exact-oc-nu", repr(EPS[e]),
                          "--nx", str(nx), "--nz", str(nx), "--steps", str(2 * nx),
                          "--distributed-solve", "--no-paraview", "--exact-oc-block-test"],
                          expand(cfg, ("block_test_",)), dict(cfg=cfg, point=pt, eps=f"1/{e}", nx=nx)))
    elif group == "scale":
        # survivors only, larger / harder points: E16 = eps 1/500 nx 16 (Pe 62.5),
        # C64 = eps 1/100 nx 64, D64 / D128 = eps 1/500 nx 64 / 128
        keep = ("ilu0", "ilu1", "ilu2", "asm4o1_ilu1", "bjac4_ilu1", "lu_petsc", "lu_mumps", "lu_umfpack")
        for cfg in keep:
            for pt, (e, nx) in {"E16": ("500", 16), "C64": ("100", 64), "D64": ("500", 64),
                                "D128": ("500", 128)}.items():
                R.append((f"blk_{cfg}_{pt}", ["--exact-oc", "lp52", "--exact-oc-nu", repr(EPS[e]),
                          "--nx", str(nx), "--nz", str(nx), "--steps", str(2 * nx),
                          "--distributed-solve", "--no-paraview", "--exact-oc-block-test"],
                          expand(cfg, ("block_test_",)), dict(cfg=cfg, point=pt, eps=f"1/{e}", nx=nx)))
    elif group == "kkt":
        for cfg in cfgs:
            for scheme, e, nx in (("cn", "100", 16), ("cn", "100", 32), ("cn", "500", 16), ("cn", "500", 32),
                                  ("be", "100", 32), ("cn", "100", 64), ("cn", "500", 64)):
                rid = f"kkt_{cfg}_{scheme}_eps{e}_nx{nx}"
                snap = os.path.join(HERE, "results", f"snap_{cfg}_{scheme}_eps{e}_nx{nx}.csv")
                R.append((rid, ["--exact-oc", "lp52", "--exact-oc-nu", repr(EPS[e]), "--time-scheme", scheme,
                                "--nx", str(nx), "--nz", str(nx), "--steps", str(2 * nx),
                                "--distributed-solve", "--pc-schur-sweep", "--no-paraview",
                                "--exact-oc-production-tol", "--exact-oc-snapshot", snap],
                          "-ksp_monitor_true_residual " + expand(cfg, SWEEP_PREFIXES),
                          dict(cfg=cfg, scheme=scheme.upper(), eps=f"1/{e}", nx=nx)))
    return R


def one(item):
    rid, args, opts, meta = item
    os.makedirs(os.path.join(WORK, "output"), exist_ok=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PETSC_OPTIONS=opts)
    t = time.time()
    p = subprocess.Popen(["/usr/bin/time", "-f", "MAXRSS_KB=%M", EXE, *args], cwd=WORK, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        text, _ = p.communicate(timeout=TIMEOUT)
        status = "done"
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, 9)
        text, _ = p.communicate()
        status = "timeout"
    wall = time.time() - t
    with open(os.path.join(HERE, "logs", "campaign", rid + ".log"), "w") as fh:
        fh.write("# PETSC_OPTIONS=" + opts + "\n# " + " ".join([os.path.basename(EXE), *args]) + "\n")
        for line in text.splitlines():
            if (line.startswith("EX21_") or "rror" in line or line.startswith("MAXRSS")
                    or "unpreconditioned" in line):
                fh.write(line + "\n")
    m = re.search(r"MAXRSS_KB=(\d+)", text)
    rss = m.group(1) if m else ""
    recs = []
    blocks = [dict(re.findall(r"(\S+?)=(\S+)", l)) for l in text.splitlines() if l.startswith("EX21_BLOCK ")]
    if blocks:
        for b in blocks:
            recs.append({**meta, "status": status, "maxrss_kb": rss, "petsc_options": opts, **b})
    else:
        rec = {**meta, "id": rid, "status": status, "exit": p.returncode, "wall_s": f"{wall:.1f}",
               "maxrss_kb": rss, "petsc_options": opts}
        for line in text.splitlines():
            for tag in ("EX21_EXACT_CONFIG", "EX21_EXACT_BOUNDARY", "EX21_EXACT_SOLVER_STATS", "EX21_LP52_RESULT"):
                if line.startswith(tag + " "):
                    for k, v in re.findall(r"(\S+?)=(\S+)", line):
                        rec.setdefault(k, v)
        hist = re.findall(r"KSP unpreconditioned resid norm \S+ true resid norm \S+ \|\|r\(i\)\|\|/\|\|b\|\| (\S+)", text)
        rec["outer_history"] = " ".join(f"{float(h):.2e}" for h in hist)
        recs.append(rec)
    print(f"{rid}: {status} wall={wall:.1f}s rows={len(recs)}", flush=True)
    return recs


def main():
    mode = sys.argv[1]
    if mode == "show":
        print(CONFIGS[sys.argv[2]] or "(production default)")
        print("sweep prefixes:", expand(sys.argv[2], SWEEP_PREFIXES))
        return
    cfgs = sys.argv[2].split(",") if mode == "kkt" else []
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 1
    for d in ("logs/campaign", "results"):
        os.makedirs(os.path.join(HERE, d), exist_ok=True)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        out = [r for recs in pool.map(one, runs(mode, cfgs)) for r in recs]
    keys = []
    for r in out:
        keys += [k for k in r if k not in keys]
    name = mode if mode in ("screen", "scale") else "kkt_" + "_".join(cfgs)
    with open(os.path.join(HERE, "results", name + ".csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(out)
    print(f"{name}: DONE ({len(out)} rows)")


if __name__ == "__main__":
    main()
