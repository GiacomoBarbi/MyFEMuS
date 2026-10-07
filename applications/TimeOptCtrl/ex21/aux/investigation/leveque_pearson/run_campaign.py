#!/usr/bin/env python3
"""Runs the T1/T2 exact optimal-control campaign of the Leveque-Pearson
validation phase (one serial TimeOptCtrl_ex21 --exact-oc process per run) and
collects the EX21_EXACT_RESULT records into results/<group>.csv.

    run_campaign.py <group> [--jobs J]

Groups (see T2_EXACT_CONVECTION_OC.md / T1_EX21_LP_HEAT.md for the rationale):
    t2_space      CN, Nt = 256, conv-sin3 nx = 4..64, conv-sin nx = 4..32
    t2_space_chk  temporal-contamination checks of t2_space (nx = 32, Nt = 128)
    t2_cn_time    CN, nx = 64, Nt = 4..32, conv-sin3
    t2_cn_term    CN, nx = 16, Nt = 16..512, conv-sin3 (terminal adjoint slot)
    t2_be_time    BE, nx = 32, Nt = 8..64, conv-sin3
    t1_rule       lp-heat, paper rules: CN tau = h (l = 3..6), BE tau = h^2 (l = 3..5), beta = 1e-3
    t1_time       lp-heat, CN, l = 5 (nx = 32), Nt = 8..64, beta = 1e-3 (ex20 section 22 analogue)
    t1_beta       lp-heat, CN, l = 4, Nt = 16, beta = 1e-2, 1e-4 (beta range of LP Tables 1-3)
    t1_conv_on    lp-heat, CN, l = 4, Nt = 16, --convection on with v = 0 (ex21 transposed path)
Environment: EX21_EXE (default: the femusbin-control build), WORK (scratch cwd).
"""
import csv
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
EXE = os.environ.get(
    "EX21_EXE",
    "/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21")
WORK = os.environ.get("WORK", "/tmp")
COMMON = ["--distributed-solve", "--pc-schur-sweep", "--no-paraview"]


def runs(group):
    R = []
    def add(rid, case, scheme, nx, steps, *extra):
        R.append((rid, ["--exact-oc", case, "--time-scheme", scheme, "--nx", str(nx),
                        "--nz", str(nx), "--steps", str(steps), *COMMON, *extra]))
    if group == "t2_space":
        for nx in (4, 8, 16, 32, 64):
            add(f"t2_space_sin3_cn_nx{nx}_nt256", "conv-sin3", "cn", nx, 256)
        for nx in (4, 8, 16, 32):
            add(f"t2_space_sin_cn_nx{nx}_nt256", "conv-sin", "cn", nx, 256)
    elif group == "t2_space_chk":
        add("t2_space_sin3_cn_nx32_nt128", "conv-sin3", "cn", 32, 128)
        add("t2_space_sin_cn_nx32_nt128", "conv-sin", "cn", 32, 128)
    elif group == "t2_cn_time":
        for nt in (4, 8, 16, 32):
            add(f"t2_cntime_sin3_nx64_nt{nt}", "conv-sin3", "cn", 64, nt)
    elif group == "t2_cn_term":
        for nt in (16, 32, 64, 128, 256, 512):
            add(f"t2_cnterm_sin3_nx16_nt{nt}", "conv-sin3", "cn", 16, nt)
    elif group == "t2_be_time":
        for nt in (8, 16, 32, 64):
            add(f"t2_betime_sin3_nx32_nt{nt}", "conv-sin3", "be", 32, nt)
    elif group == "t1_rule":
        for l in (3, 4, 5, 6):
            add(f"t1_rule_cn_l{l}", "lp-heat", "cn", 2**l, 2**l)          # tau = h = 2^(1-l), T = 2
        for l in (3, 4, 5):
            add(f"t1_rule_be_l{l}", "lp-heat", "be", 2**l, 2**(2 * l - 1))  # tau = h^2
    elif group == "t1_time":
        for nt in (8, 16, 32, 64):
            add(f"t1_time_cn_l5_nt{nt}", "lp-heat", "cn", 32, nt)
    elif group == "t1_beta":
        for b in ("1e-2", "1e-4"):
            add(f"t1_beta{b}_cn_l4", "lp-heat", "cn", 16, 16, "--exact-oc-alpha-u", b)
    elif group == "t1_conv_on":
        add("t1_convon_cn_l4", "lp-heat", "cn", 16, 16, "--convection", "on")
        add("t1_convoff_cn_l4", "lp-heat", "cn", 16, 16)
    else:
        raise SystemExit(f"unknown group {group}")
    return R


def parse(text):
    rec = {}
    for line in text.splitlines():
        if line.startswith("EX21_EXACT_RESULT") or line.startswith("EX21_EXACT_CONFIG"):
            for k, v in re.findall(r"(\S+?)=(\S+)", line):
                rec.setdefault(k, v)
    return rec


def one(item):
    rid, args = item
    os.makedirs(os.path.join(WORK, "output"), exist_ok=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    t = time.time()
    p = subprocess.run(["/usr/bin/time", "-f", "MAXRSS_KB=%M", EXE, *args], cwd=WORK,
                       env=env, capture_output=True, text=True)
    text = p.stdout + p.stderr
    log = os.path.join(HERE, "logs", "campaign", rid + ".log")
    with open(log, "w") as fh:
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
          f" res={rec.get('true_residual')} y={rec.get('y_L2L2_rel')} p={rec.get('p_L2L2_rel')}",
          flush=True)
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
