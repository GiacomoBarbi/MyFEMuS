#!/usr/bin/env python3
"""T6 campaign runner (LPS).  run_t6.py <group> [--jobs J]
   -> results/<group>.csv (every key=value of every EX21_ line, first occurrence
      wins, plus the stabilisation label), logs/campaign/<id>.log

Stabilisation variants: none | p0 (LP's patch mean: the T6 LPS) | p1 (diagnostic).
Groups
  mms_space    forward MMS sin3, q2-divfree wind, nu = 0.01, hom. Dirichlet, T = 1,
               CN Nt = 1024, nx = 4..64                        (none, p0, p1)
  mms_fixedpe  as mms_space with nu = 0.16/nx: FIXED patch Peclet (max Pe_k ~ 9), so delta_k ~ h
               on every level and the stabilised order is asymptotic   (none, p0, p1)
  mms_time     same as mms_space, nx = 32, BE and CN, Nt = 4..64  (none, p0)
  oc_space     exact OC conv-sin3 + q2-divfree, nu = 0.01, CN Nt = 256, nx = 4..32 (none, p0, p1)
  oc_time      same, nx = 32, CN Nt = 4..32, BE Nt = 8..64      (none, p0)
  lowpe        LP 5.2 eps = 1/20, nx = 64 / Nt 128 (zero active patches), CN and BE:
               none vs lps, LU sweeps, snapshots + series     (exact-equality regression)
  lp52         LP 5.2 primary comparison: beta 1e-2, eps 1/20, 1/100, 1/500,
               nx 8/16/32/64 (Nt = 2 nx; the T4 ladder), CN, LU sweeps, snapshots + series
  weak         eps = 1/500, beta 1e-1 and 1e2, nx 8/16/32/64, CN, LU sweeps
  undershoot   eps 1/500, beta 1e2, nx 32, CN and BE, Nt = 64, 256, 1024 (none, p0)
  kkt_gamg     production default spatial GAMG (no options), beta 1e-2, eps 1/100, 1/500,
               nx 32/64, CN (secondary solver-effect check; T6_TIMEOUT applies)
"""
import csv, os, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__))
EXE = os.environ.get("EX21_EXE",
    "/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21")
WORK = os.environ.get("WORK", "/tmp/t6_work")
TIMEOUT = int(os.environ.get("T6_TIMEOUT", "7200"))
LU = " ".join(f"-{p}pc_type lu" for p in ("sweep_right_", "sweep_left_", "sweep_ay_", "fieldsplit_y_sweep_"))
STAB = {"none": [], "p0": ["--convection-stabilization", "lps"],
        "p1": ["--convection-stabilization", "lps", "--lps-projection", "p1"]}
MMS = ["--convection-mms", "sin3", "--distributed-solve", "--state-bc", "dirichlet", "--nu", "0.01",
       "--T", "1", "--convection", "on", "--wind", "q2-divfree", "--no-paraview"]
SOLVE = ["--distributed-solve", "--pc-schur-sweep", "--no-paraview"]
EPS = {"20": 0.05, "100": 0.01, "500": 0.002}


def runs(group):
    R = []
    def mms(rid, st, scheme, nx, nt, nu=None):
        args = MMS + ["--time-scheme", scheme, "--nx", str(nx), "--nz", str(nx), "--steps", str(nt)] + STAB[st]
        if nu is not None:   # a later --nu overrides the 0.01 of MMS
            args += ["--nu", repr(nu)]
        R.append((rid, args, "", dict(stab=st)))
    def oc(rid, st, scheme, nx, nt):
        R.append((rid, ["--exact-oc", "conv-sin3", "--wind", "q2-divfree", "--exact-oc-nu", "0.01",
                        "--time-scheme", scheme, "--nx", str(nx), "--nz", str(nx), "--steps", str(nt)]
                  + SOLVE + STAB[st], LU, dict(stab=st)))
    def lp(rid, st, e, nx, beta, scheme="cn", extra=(), opts=LU, steps=None):
        snap = os.path.join(HERE, "results", "snap", f"{rid}.csv")
        series = os.path.join(HERE, "results", "series", f"{rid}.csv")
        R.append((rid, ["--exact-oc", "lp52", "--exact-oc-nu", repr(EPS[e]), "--exact-oc-alpha-u", beta,
                        "--time-scheme", scheme, "--nx", str(nx), "--nz", str(nx), "--steps", str(steps or 2 * nx)]
                  + SOLVE + ["--exact-oc-production-tol", "--exact-oc-snapshot", snap,
                             "--exact-oc-series", series, *extra] + STAB[st],
                  opts, dict(stab=st, eps=f"1/{e}", beta=beta)))
    if group == "mms_space":
        for st in ("none", "p0", "p1"):
            for nx in (4, 8, 16, 32, 64): mms(f"mms_{st}_cn_nx{nx}", st, "cn", nx, 1024)
    elif group == "mms_fixedpe":
        for st in ("none", "p0", "p1"):
            for nx in (4, 8, 16, 32, 64): mms(f"mmsfpe_{st}_cn_nx{nx}", st, "cn", nx, 1024, 0.16 / nx)
    elif group == "mms_time":
        for st in ("none", "p0"):
            for s in ("be", "cn"):
                for nt in (4, 8, 16, 32, 64): mms(f"mmst_{st}_{s}_nx32_nt{nt}", st, s, 32, nt)
    elif group == "oc_space":
        for st in ("none", "p0", "p1"):
            for nx in (4, 8, 16, 32): oc(f"oc_{st}_cn_nx{nx}", st, "cn", nx, 256)
    elif group == "oc_time":
        for st in ("none", "p0"):
            for nt in (4, 8, 16, 32): oc(f"oct_{st}_cn_nx32_nt{nt}", st, "cn", 32, nt)
            for nt in (8, 16, 32, 64): oc(f"oct_{st}_be_nx32_nt{nt}", st, "be", 32, nt)
    elif group == "lowpe":
        for st in ("none", "p0"):
            for sch in ("cn", "be"): lp(f"lowpe_{st}_eps20_nx64_{sch}", st, "20", 64, "1e-2", sch)
    elif group == "lp52":
        for e in ("20", "100", "500"):
            for nx in (8, 16, 32, 64):
                for st in ("none", "p0", "p1"): lp(f"lp52_{st}_eps{e}_b1e-2_nx{nx}", st, e, nx, "1e-2")
    elif group == "weak":
        for b in ("1e-1", "1e2"):
            for nx in (8, 16, 32, 64):
                for st in ("none", "p0", "p1"): lp(f"weak_{st}_eps500_b{b}_nx{nx}", st, "500", nx, b)
    elif group == "undershoot":
        # step 12: is the first-step corner undershoot a time-discretisation effect
        # of the discontinuous initial/boundary data?  eps 1/500, beta 1e2, nx 32
        for st in ("none", "p0"):
            for sch in ("cn", "be"):
                for nt in (64, 256, 1024):
                    lp(f"under_{st}_{sch}_eps500_b1e2_nx32_nt{nt}", st, "500", 32, "1e2", sch, steps=nt)
    elif group == "kkt_gamg":
        # secondary (step 14): the PRODUCTION default spatial solver (GMRES+GAMG,
        # no PETSC_OPTIONS) end to end; unstabilised runs are expected to be
        # rejected by the T6 acceptance guard
        for e in ("100", "500"):
            for nx in (32, 64):
                for st in ("none", "p0", "p1"):
                    lp(f"kkt_gamg_{st}_eps{e}_b1e-2_nx{nx}", st, e, nx, "1e-2", opts="")
    else:
        raise SystemExit(f"unknown group {group}")
    return R


def one(item):
    rid, args, opts, meta = item
    os.makedirs(os.path.join(WORK, rid, "output"), exist_ok=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PETSC_OPTIONS=opts)
    t = time.time()
    p = subprocess.Popen(["/usr/bin/time", "-f", "MAXRSS_KB=%M", EXE, *args], cwd=os.path.join(WORK, rid),
                         env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         start_new_session=True)
    try:
        text, _ = p.communicate(timeout=TIMEOUT); status = "done"
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, 9); text, _ = p.communicate(); status = "timeout"
    wall = time.time() - t
    keep = [l for l in text.splitlines() if l.startswith(("EX21_", "MAXRSS")) or "rror" in l]
    with open(os.path.join(HERE, "logs", "campaign", rid + ".log"), "w") as fh:
        fh.write("# PETSC_OPTIONS=" + opts + "\n# " + " ".join([os.path.basename(EXE), *args]) + "\n")
        fh.write("\n".join(l for l in keep if not l.startswith("EX21_EXACT_SLOT")) + "\n")
    rec = {**meta, "id": rid, "status": status, "exit": p.returncode, "wall_s": f"{wall:.1f}"}
    for line in keep:
        if line.startswith(("EX21_EXACT_SLOT", "EX21_SERIES")):
            continue
        tag = line.split()[0]
        for k, v in re.findall(r"(\S+?)=(\S+)", line):
            rec.setdefault(k if tag != "EX21_LPS" else "lps_" + k, v)
    print(f"{rid}: {status} exit={p.returncode} wall={wall:.1f}s", flush=True)
    return rec


def main():
    group = sys.argv[1]
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 1
    for d in ("logs/campaign", "results/snap", "results/series"):
        os.makedirs(os.path.join(HERE, d), exist_ok=True)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        out = list(pool.map(one, runs(group)))
    keys = []
    for r in out:
        keys += [k for k in r if k not in keys]
    with open(os.path.join(HERE, "results", group + ".csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(out)
    print(f"{group}: DONE ({len(out)} runs)")


if __name__ == "__main__":
    main()
