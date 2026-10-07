#!/usr/bin/env python3
"""T3 campaign: variable wind and inhomogeneous Dirichlet state data (ex21).

    run_t3.py <group> [--jobs J]      -> results/<group>.csv, logs/campaign/<id>.log

Forward MMS (--convection-mms sin3, homogeneous Dirichlet, T = 1, nu = 1/4,
wind q2-divfree w = (1/2 + x^2, 1/4 - 2xz)):
    vw_mms_space   CN, Nt = 512, nx = 4..32
    vw_mms_time    BE and CN, nx = 64, Nt = 4..64
Exact optimal control (--exact-oc, ExactOcBenchmark.cpp):
    vw_oc_space    conv-sin3 + q2-divfree, CN, Nt = 128, nx = 4..32
    vw_oc_time     conv-sin3 + q2-divfree, CN nx = 32 Nt = 4..32; BE nx = 32 Nt = 8..64
    dir_space      lift-exp (constant v = (1, 1/2)), CN, Nt = 128, nx = 4..32
    dir_time       lift-exp, CN nx = 32 Nt = 4..32; BE nx = 32 Nt = 8..64
    dirvw_space    lift-const + q2-divfree, CN, Nt = 128, nx = 4..32
    dirvw_time     lift-const + q2-divfree, CN nx = 32 Nt = 4..32
Leveque-Pearson 5.2 (lp52, eps = 1/20, CN, tau = h_LP = 2^(1-l), T = 2):
    lp52_mesh      beta = 1e-2: Q2 nx = 16 (Nt 32), 32 (Nt 64), 64 (Nt 128)
                   (Q2 nx = 2^(l-1) has the node set of LP's Q1 level l)
    lp52_beta      beta = 1e-4, 1e-6 at Q2 nx = 64 / Nt 128; 1e-4, 1e-6 at nx = 32 / Nt 64
"""
import csv, os, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
EXE = os.environ.get("EX21_EXE",
    "/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21")
WORK = os.environ.get("WORK", "/tmp")
SOLVE = ["--distributed-solve", "--pc-schur-sweep", "--no-paraview"]
MMS = ["--distributed-solve", "--state-bc", "dirichlet", "--nu", "0.25", "--T", "1",
       "--convection", "on", "--wind", "q2-divfree", "--no-paraview"]


def runs(group):
    R = []
    def oc(rid, case, scheme, nx, nt, *extra):
        R.append((rid, ["--exact-oc", case, "--time-scheme", scheme, "--nx", str(nx), "--nz", str(nx),
                        "--steps", str(nt), *SOLVE, *extra]))
    def mms(rid, scheme, nx, nt):
        R.append((rid, ["--convection-mms", "sin3", "--time-scheme", scheme, "--nx", str(nx),
                        "--nz", str(nx), "--steps", str(nt), *MMS]))
    W = ["--wind", "q2-divfree"]
    if group == "vw_mms_space":
        for nx in (4, 8, 16, 32): mms(f"vw_mms_cn_nx{nx}_nt512", "cn", nx, 512)
    elif group == "vw_mms_time":
        for s in ("be", "cn"):
            for nt in (4, 8, 16, 32, 64): mms(f"vw_mms_{s}_nx64_nt{nt}", s, 64, nt)
    elif group == "vw_oc_space":
        for nx in (4, 8, 16, 32): oc(f"vw_oc_cn_nx{nx}_nt128", "conv-sin3", "cn", nx, 128, *W)
    elif group == "vw_oc_time":
        for nt in (4, 8, 16, 32): oc(f"vw_oc_cn_nx32_nt{nt}", "conv-sin3", "cn", 32, nt, *W)
        for nt in (8, 16, 32, 64): oc(f"vw_oc_be_nx32_nt{nt}", "conv-sin3", "be", 32, nt, *W)
    elif group == "dir_space":
        for nx in (4, 8, 16, 32): oc(f"dir_cn_nx{nx}_nt128", "lift-exp", "cn", nx, 128)
    elif group == "dir_time":
        for nt in (4, 8, 16, 32): oc(f"dir_cn_nx32_nt{nt}", "lift-exp", "cn", 32, nt)
        for nt in (8, 16, 32, 64): oc(f"dir_be_nx32_nt{nt}", "lift-exp", "be", 32, nt)
    elif group == "dirvw_space":
        for nx in (4, 8, 16, 32): oc(f"dirvw_cn_nx{nx}_nt128", "lift-const", "cn", nx, 128, *W)
    elif group == "dirvw_time":
        for nt in (4, 8, 16, 32): oc(f"dirvw_cn_nx32_nt{nt}", "lift-const", "cn", 32, nt, *W)
    elif group == "lp52_mesh":
        for nx, nt in ((16, 32), (32, 64), (64, 128)):
            snap = os.path.join(HERE, "results", f"lp52_snapshot_b1e-2_nx{nx}.csv")
            oc(f"lp52_b1e-2_nx{nx}_nt{nt}", "lp52", "cn", nx, nt, "--exact-oc-production-tol",
               "--exact-oc-snapshot", snap)
    elif group == "lp52_beta":
        for b in ("1e-4", "1e-6"):
            for nx, nt in ((32, 64), (64, 128)):
                snap = os.path.join(HERE, "results", f"lp52_snapshot_b{b}_nx{nx}.csv")
                oc(f"lp52_b{b}_nx{nx}_nt{nt}", "lp52", "cn", nx, nt, "--exact-oc-production-tol",
                   "--exact-oc-alpha-u", b, "--exact-oc-snapshot", snap)
    else:
        raise SystemExit(f"unknown group {group}")
    return R


def parse(text):
    rec = {}
    for line in text.splitlines():
        for tag in ("EX21_MMS", "EX21_EXACT_CONFIG", "EX21_EXACT_RESULT", "EX21_EXACT_BOUNDARY",
                    "EX21_EXACT_SOLVER_STATS", "EX21_LP52_RESULT"):
            if line.startswith(tag + " "):
                for k, v in re.findall(r"(\S+?)=(\S+)", line):
                    rec.setdefault(k, v)
    return rec


def one(item):
    rid, args = item
    os.makedirs(os.path.join(WORK, "output"), exist_ok=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    t = time.time()
    p = subprocess.run(["/usr/bin/time", "-f", "MAXRSS_KB=%M", EXE, *args], cwd=WORK, env=env,
                       capture_output=True, text=True)
    text = p.stdout + p.stderr
    with open(os.path.join(HERE, "logs", "campaign", rid + ".log"), "w") as fh:
        fh.write("# " + " ".join([os.path.basename(EXE), *args]) + "\n")
        for line in text.splitlines():
            if line.startswith("EX21_") or "rror" in line or line.startswith("MAXRSS"):
                fh.write(line + "\n")
    rec = parse(text)
    rec["id"] = rid
    rec["exit"] = p.returncode
    m = re.search(r"MAXRSS_KB=(\d+)", text)
    rec["maxrss_kb"] = m.group(1) if m else ""
    rec["wall_s"] = f"{time.time() - t:.1f}"
    print(f"{rid}: exit={p.returncode} wall={rec['wall_s']}s outer={rec.get('outer_its')}"
          f" y={rec.get('y_L2L2_rel', rec.get('rel_err_L2_T', rec.get('y_t1_max')))}", flush=True)
    return rec


def main():
    group = sys.argv[1]
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 1
    os.makedirs(os.path.join(HERE, "logs", "campaign"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        recs = list(pool.map(one, runs(group)))
    keys = []
    for r in recs:
        keys += [k for k in r if k not in keys]
    with open(os.path.join(HERE, "results", group + ".csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(recs)
    print(f"{group}: DONE ({len(recs)} runs)")
    return 0 if all(r["exit"] == 0 for r in recs) else 1


if __name__ == "__main__":
    sys.exit(main())
