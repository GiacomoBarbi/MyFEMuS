#!/usr/bin/env python3
"""T7N campaign runner (manufactured convection + Neumann boundary control,
--exact-oc neumann-conv).  t7n_run.py <group> [--jobs J] -> results/<group>.csv,
logs/campaign/<id>.log (every EX21_ line except per-slot lines).
Groups
  space   CN, Nt = 512, nx = 4, 8, 16, 32; winds A (1, 0.5) and B (1, -0.5); LU sweeps
  time    nx = 32; BE Nt = 8..64, CN Nt = 4..32; winds A, B; LU sweeps
  solver  wind A, CN, nx = 16 (Nt 64) and 32 (Nt 128), production tolerance:
          sweep_lu (SCHUR_SWEEP + LU, reference), sweep_gamg (SCHUR_SWEEP, default
          spatial GMRES+GAMG), matching (SCHUR_MATCHING), additive (default ADDITIVE_GAMG)
  cnlayer CN terminal-layer study: wind A nx 48 Nt 32..128; w = 0 (heat) CN/BE nx 32
  (the space/time groups use the tight exact-OC tolerance)"""
import csv, os, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = os.environ.get("EX21_EXE", "/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21")
WORK = os.environ.get("WORK", "/tmp/t7n_work")
LU = " ".join(f"-{p}pc_type lu" for p in ("sweep_right_", "sweep_left_", "sweep_ay_", "fieldsplit_y_sweep_"))
WIND = {"A": ("1", "0.5"), "B": ("1", "-0.5"), "C": ("1", "0"), "0": ("0", "0")}
BASE = ["--exact-oc", "neumann-conv", "--distributed-solve", "--no-paraview"]


def runs(group):
    R = []
    def r(rid, wind, scheme, nx, nt, pc=("--pc-schur-sweep",), opts=LU, extra=(), meta=None):
        R.append((rid, BASE + ["--exact-oc-velocity", *WIND[wind], "--time-scheme", scheme, "--nx", str(nx),
                               "--nz", str(nx), "--steps", str(nt), *pc, *extra], opts,
                  dict(wind=wind, **(meta or {}))))
    if group == "space":
        for w in ("A", "B"):
            for nx in (4, 8, 16, 32): r(f"space_{w}_cn_nx{nx}", w, "cn", nx, 512)
    elif group == "time":
        for w in ("A", "B"):
            for nt in (8, 16, 32, 64): r(f"time_{w}_be_nt{nt}", w, "be", 32, nt)
            for nt in (4, 8, 16, 32): r(f"time_{w}_cn_nt{nt}", w, "cn", 32, nt)
    elif group == "cnlayer":
        # CN terminal-layer study: wind A on a finer mesh (spatial floor of u ~1.5e-5)
        # and the w = 0 (heat) control experiment with the same manufactured structure
        for nt in (32, 64, 128): r(f"cnlayer_A_cn_nx48_nt{nt}", "A", "cn", 48, nt)
        for nt in (16, 32, 64): r(f"cnlayer_0_cn_nx32_nt{nt}", "0", "cn", 32, nt)
        for nt in (16, 32, 64): r(f"cnlayer_0_be_nx32_nt{nt}", "0", "be", 32, nt)
    elif group == "solver":
        for nx, nt in ((16, 64), (32, 128)):
            for v, pc, opts in (("sweep_lu", ("--pc-schur-sweep",), LU), ("sweep_gamg", ("--pc-schur-sweep",), ""),
                                ("matching", ("--pc-schur-matching",), ""), ("additive", (), "")):
                r(f"solver_{v}_A_cn_nx{nx}", "A", "cn", nx, nt, pc, "-ksp_monitor_true_residual " + opts,
                  ("--exact-oc-production-tol",), dict(variant=v))
    else:
        raise SystemExit(f"unknown group {group}")
    return R


def one(item):
    rid, args, opts, meta = item
    wd = os.path.join(WORK, rid); os.makedirs(os.path.join(wd, "output"), exist_ok=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PETSC_OPTIONS=opts)
    t = time.time()
    p = subprocess.Popen(["/usr/bin/time", "-f", "MAXRSS_KB=%M", EXE, *args], cwd=wd, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        text, _ = p.communicate(timeout=3600); status = "done"
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, 9); text, _ = p.communicate(); status = "timeout"
    keep = [l for l in text.splitlines() if (l.startswith(("EX21_", "MAXRSS")) and not l.startswith("EX21_EXACT_SLOT"))
            or "rror" in l or "unpreconditioned" in l]
    with open(os.path.join(HERE, "logs", "campaign", rid + ".log"), "w") as fh:
        fh.write("# PETSC_OPTIONS=" + opts + "\n# " + " ".join([os.path.basename(EXE), *args]) + "\n" + "\n".join(keep) + "\n")
    rec = {**meta, "id": rid, "status": status, "exit": p.returncode, "wall_s": f"{time.time() - t:.1f}"}
    for line in keep:
        if line.startswith("EX21_NEUMANN_STATIONARITY"):
            m = re.search(r"\|M_G\(alpha_u u \+ eta p\)\|/\|alpha_u M_G u\|=(\S+)", line)
            rec["neumann_stationarity_rel"] = m.group(1) if m else ""
            m = re.search(r"max\|u\|_off_Gamma_c=(\S+)", line); rec["max_u_off_Gamma_c"] = m.group(1) if m else ""
            m = re.search(r"control_nodes=(\d+)", line); rec["control_nodes"] = m.group(1) if m else ""
            continue
        if line.startswith("EX21_"):
            for k, v in re.findall(r"(\S+?)=(\S+)", line):
                rec.setdefault(k, v)
    hist = re.findall(r"KSP unpreconditioned resid norm \S+ true resid norm \S+ \|\|r\(i\)\|\|/\|\|b\|\| (\S+)", text)
    rec["outer_history"] = " ".join(f"{float(h):.2e}" for h in hist)
    print(f"{rid}: {status} exit={p.returncode}", flush=True)
    return rec


if __name__ == "__main__":
    group = sys.argv[1]
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 1
    os.makedirs(os.path.join(HERE, "logs", "campaign"), exist_ok=True)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        out = list(pool.map(one, runs(group)))
    keys = []
    for r in out:
        keys += [k for k in r if k not in keys]
    with open(os.path.join(HERE, "results", group + ".csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(out)
    print(f"{group}: DONE ({len(out)})")
