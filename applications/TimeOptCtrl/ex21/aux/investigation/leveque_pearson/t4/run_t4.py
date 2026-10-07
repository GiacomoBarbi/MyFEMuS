#!/usr/bin/env python3
"""T4: unstabilised high-Peclet map of LP 5.2 (corrected tracking), ex21 Q2.

    run_t4.py <group> [--jobs J]   -> results/<group>.csv, logs/campaign/<id>.log

Every KKT run uses SCHUR_SWEEP with the production tolerances, beta = 1e-2,
tau = h_LP = 1/nx (Nt = 2 nx, T = 2; Q2 nx elements = LP Q1 level log2(nx)+1),
in two variants:
  gamg  the production spatial solver in the sweeps (GMRES(50)+GAMG, 500 its)
  lu    the SAME KKT solve with exact (LU) spatial solves in every sweep
        (PETSC_OPTIONS -sweep_{right,left,ay}_ / -fieldsplit_y_sweep_ ksp_type
        preonly pc_type lu): isolates the spatial preconditioner from the
        outer/Schur iteration and provides the reference discrete solution.
Outer residual histories: -ksp_monitor_true_residual (outer KSP has no prefix).
Per-run timeout: 900 s (status "timeout").

Groups:
  cn_map    CN, eps = 1/20, 1/100, 1/500, nx = 8, 16, 32, 64, gamg and lu
  be_map    BE, same ladder, except gamg at eps = 1/100, 1/500 with nx >= 32
            (stop condition: established failure, see runs())
  beta_map  CN, lu, beta = 1e-1 and 1e2, eps = 1/100, 1/500, nx = 16, 32, 64
  blocks    isolated spatial-block experiment (--exact-oc-block-test):
            eps x nx at dt = 1/nx; and a dt sweep at nx = 32, eps = 1/100, 1/20
"""
import csv, os, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
EXE = os.environ.get("EX21_EXE",
    "/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21")
WORK = os.environ.get("WORK", "/tmp")
TIMEOUT = 900
EPS = {"20": 1 / 20, "100": 1 / 100, "500": 1 / 500}
LU = " ".join(f"-{p}ksp_type preonly -{p}pc_type lu"
              for p in ("sweep_right_", "sweep_left_", "sweep_ay_", "fieldsplit_y_sweep_"))


def runs(group):
    R = []
    if group in ("cn_map", "be_map"):
        scheme = "cn" if group == "cn_map" else "be"
        for e in ("20", "100", "500"):
            for nx in (8, 16, 32, 64):
                for var in ("lu", "gamg"):
                    # stop condition: BE gamg at eps = 1/100, 1/500, nx >= 32 is an
                    # established failure (theta = 1 blocks fail at every such Pe in
                    # the block experiment; the easier CN gamg runs timed out there)
                    if scheme == "be" and var == "gamg" and e != "20" and nx >= 32:
                        continue
                    rid = f"t4_{scheme}_eps{e}_nx{nx}_{var}"
                    snap = os.path.join(HERE, "results", f"snap_{scheme}_eps{e}_nx{nx}_{var}.csv")
                    args = ["--exact-oc", "lp52", "--exact-oc-nu", repr(EPS[e]), "--time-scheme", scheme,
                            "--nx", str(nx), "--nz", str(nx), "--steps", str(2 * nx),
                            "--distributed-solve", "--pc-schur-sweep", "--no-paraview",
                            "--exact-oc-production-tol", "--exact-oc-snapshot", snap]
                    opts = "-ksp_monitor_true_residual" + (" " + LU if var == "lu" else "")
                    R.append((rid, args, opts))
    elif group == "beta_map":
        # after the Pe mechanism at beta = 1e-2: weaker control (LP's largest
        # beta = 1e-1, and beta = 1e2 ~ uncontrolled) with exact (LU) sweeps
        for b in ("1e-1", "1e2"):
            for e in ("100", "500"):
                for nx in (16, 32, 64):
                    rid = f"t4_cn_b{b}_eps{e}_nx{nx}_lu"
                    snap = os.path.join(HERE, "results", f"snap_cn_b{b}_eps{e}_nx{nx}_lu.csv")
                    R.append((rid, ["--exact-oc", "lp52", "--exact-oc-nu", repr(EPS[e]), "--exact-oc-alpha-u", b,
                                    "--time-scheme", "cn", "--nx", str(nx), "--nz", str(nx), "--steps", str(2 * nx),
                                    "--distributed-solve", "--pc-schur-sweep", "--no-paraview",
                                    "--exact-oc-production-tol", "--exact-oc-snapshot", snap],
                              "-ksp_monitor_true_residual " + LU))
    elif group == "blocks":
        for e in ("20", "100", "500"):
            for nx in (8, 16, 32, 64):
                R.append((f"blk_eps{e}_nx{nx}_dt{2*nx}", ["--exact-oc", "lp52", "--exact-oc-nu", repr(EPS[e]),
                          "--nx", str(nx), "--nz", str(nx), "--steps", str(2 * nx), "--distributed-solve",
                          "--no-paraview", "--exact-oc-block-test"], ""))
        for e in ("20", "100"):
            for nt in (8, 16, 32, 64, 128, 256, 512):
                if nt == 64:
                    continue  # = the tau = h run above
                R.append((f"blk_eps{e}_nx32_dt{nt}", ["--exact-oc", "lp52", "--exact-oc-nu", repr(EPS[e]),
                          "--nx", "32", "--nz", "32", "--steps", str(nt), "--distributed-solve",
                          "--no-paraview", "--exact-oc-block-test"], ""))
    else:
        raise SystemExit(f"unknown group {group}")
    return R


def parse(text):
    rec = {}
    for line in text.splitlines():
        for tag in ("EX21_EXACT_CONFIG", "EX21_EXACT_BOUNDARY", "EX21_EXACT_SOLVER_STATS", "EX21_LP52_RESULT"):
            if line.startswith(tag + " "):
                for k, v in re.findall(r"(\S+?)=(\S+)", line):
                    rec.setdefault(k, v)
    hist = [float(m) for m in re.findall(r"KSP unpreconditioned resid norm \S+ true resid norm \S+ \|\|r\(i\)\|\|/\|\|b\|\| (\S+)", text)
            if True]
    rec["outer_history"] = " ".join(f"{h:.2e}" for h in hist)
    rec["outer_monitored_its"] = max(0, len(hist) - 1)
    rec["blocks"] = [dict(re.findall(r"(\S+?)=(\S+)", l)) for l in text.splitlines() if l.startswith("EX21_BLOCK ")]
    return rec


def one(item):
    rid, args, opts = item
    os.makedirs(os.path.join(WORK, "output"), exist_ok=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PETSC_OPTIONS=opts)
    t = time.time()
    # own process group, so a timeout kills the solver too (not only /usr/bin/time)
    p = subprocess.Popen(["/usr/bin/time", "-f", "MAXRSS_KB=%M", EXE, *args], cwd=WORK, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         start_new_session=True)
    try:
        text, _ = p.communicate(timeout=TIMEOUT)
        code, status = p.returncode, "done"
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, 9)
        text, _ = p.communicate()
        code, status = -9, "timeout"
    with open(os.path.join(HERE, "logs", "campaign", rid + ".log"), "w") as fh:
        fh.write("# PETSC_OPTIONS=" + opts + "\n# " + " ".join([os.path.basename(EXE), *args]) + "\n")
        for line in text.splitlines():
            if line.startswith("EX21_") or "rror" in line or line.startswith("MAXRSS") or "unpreconditioned" in line:
                fh.write(line + "\n")
    rec = parse(text)
    rec.update(id=rid, exit=code, status=status, wall_s=f"{time.time() - t:.1f}")
    m = re.search(r"MAXRSS_KB=(\d+)", text)
    rec["maxrss_kb"] = m.group(1) if m else ""
    print(f"{rid}: {status} exit={code} wall={rec['wall_s']}s outer={rec.get('outer_its')} reason={rec.get('reason')}"
          f" sweep_failures={rec.get('sweep_failures')} blocks={len(rec['blocks'])}", flush=True)
    return rec


def main():
    group = sys.argv[1]
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 1
    for d in ("logs/campaign", "results"):
        os.makedirs(os.path.join(HERE, d), exist_ok=True)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        recs = list(pool.map(one, runs(group)))
    if group == "blocks":
        rows = []
        for r in recs:
            for b in r["blocks"]:
                rows.append({"run": r["id"], **b})
    else:
        rows = [{k: v for k, v in r.items() if k != "blocks"} for r in recs]
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(os.path.join(HERE, "results", group + ".csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"{group}: DONE ({len(recs)} runs)")


if __name__ == "__main__":
    main()
