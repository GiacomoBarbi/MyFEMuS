#!/usr/bin/env python3
"""T3b / T4 campaign runner (same machinery as t3/run_t3.py).

    run_t3b.py <group> [--jobs J]   -> results/<group>.csv, logs/campaign/<id>.log

T3b, exact optimal control `lift-track` (g = 1/2 + exp((v.x - |v|_1)/nu),
y_d != g on dOmega, v = (1, 1/2), nu = 1/4, alpha_u = 1e-2); every ladder is run
with the affine tracking term (aff) and without it (noaff, diagnostic
--lift-no-affine-tracking, the pre-T3b behaviour):
    lt_space     CN Nt = 128, nx = 4..32
    lt_time      CN nx = 32 Nt = 4..32; BE nx = 32 Nt = 8..64
    lt_regress   lift-exp / lift-const+q2 reference runs (compare with t3/)
LP 5.2 (corrected tracking), CN, production tolerances:
    lp52_corr    eps = 1/20, beta = 1e-2, 1e-4, 1e-6, Q2 nx = 32 (Nt 64), 64 (Nt 128)
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
    NO = ["--lift-no-affine-tracking"]
    if group == "lt_space":
        for nx in (4, 8, 16, 32):
            oc(f"lt_aff_cn_nx{nx}_nt128", "lift-track", "cn", nx, 128)
            oc(f"lt_noaff_cn_nx{nx}_nt128", "lift-track", "cn", nx, 128, *NO)
    elif group == "lt_time":
        for nt in (4, 8, 16, 32):
            oc(f"lt_aff_cn_nx32_nt{nt}", "lift-track", "cn", 32, nt)
            oc(f"lt_noaff_cn_nx32_nt{nt}", "lift-track", "cn", 32, nt, *NO)
        for nt in (8, 16, 32, 64):
            oc(f"lt_aff_be_nx32_nt{nt}", "lift-track", "be", 32, nt)
            oc(f"lt_noaff_be_nx32_nt{nt}", "lift-track", "be", 32, nt, *NO)
    elif group == "lt_regress":
        oc("dir_cn_nx16_nt128", "lift-exp", "cn", 16, 128)
        oc("dir_be_nx32_nt16", "lift-exp", "be", 32, 16)
        oc("dirvw_cn_nx16_nt128", "lift-const", "cn", 16, 128, "--wind", "q2-divfree")
    elif group == "lp52_corr":
        for b in ("1e-2", "1e-4", "1e-6"):
            for nx, nt in ((32, 64), (64, 128)):
                snap = os.path.join(HERE, "results", f"lp52corr_snapshot_b{b}_nx{nx}.csv")
                oc(f"lp52corr_b{b}_nx{nx}_nt{nt}", "lp52", "cn", nx, nt, "--exact-oc-production-tol",
                   "--exact-oc-alpha-u", b, "--exact-oc-snapshot", snap)
    else:
        raise SystemExit(f"unknown group {group}")
    return R


def parse(text):
    rec = {}
    for line in text.splitlines():
        for tag in ("EX21_MMS", "EX21_EXACT_CONFIG", "EX21_EXACT_RESULT", "EX21_EXACT_BOUNDARY",
                    "EX21_EXACT_SOLVER_STATS", "EX21_LP52_RESULT", "EX21_EXACT_OBJECTIVE"):
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
