#!/usr/bin/env python3
"""Constant-wind / homogeneous-Dirichlet regression of the T3 build against the
archived T0/T1/T2 outputs (pre-T3 binary 8ff3fe15):
  F  operator dumps (M, L, N, M_track, E, E^T) byte-identical to t0_dump/
  B  archived --exact-oc runs: every key=value of EX21_EXACT_RESULT (except wall
     times) and every EX21_EXACT_SLOT line identical
  G  archived --verify-convection T2 gate logs identical
Usage: WORK=<scratch> regress_constant_wind.py"""
import os, re, subprocess, sys, filecmp
HERE = os.path.dirname(os.path.abspath(__file__))
LP = os.path.dirname(HERE)
EXE = os.environ.get("EX21_EXE", "/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21")
WORK = os.environ["WORK"]
os.makedirs(os.path.join(WORK, "output"), exist_ok=True)
ENV = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
fails = 0
def run(args):
    return subprocess.run([EXE, *args], cwd=WORK, env=ENV, capture_output=True, text=True).stdout
def kv(line):
    return {k: v for k, v in re.findall(r"(\S+?)=(\S+)", line) if "wall" not in k}
# F
for sch in ("cn", "be"):
    pre = os.path.join(WORK, f"dump_{sch}")
    run(["--exact-oc", "conv-sin3", "--nx", "2", "--nz", "2", "--steps", "3", "--time-scheme", sch,
         "--pc-schur-sweep", "--distributed-solve", "--no-paraview", "--exact-oc-dump", pre])
    for tag in ("M", "L", "N", "Mtrack", "E", "ET"):
        same = filecmp.cmp(f"{pre}_{tag}.txt", os.path.join(LP, "t0_dump", f"ex21_nx2_nt3_{sch}_{tag}.txt"), shallow=False)
        fails += not same
        print(f"F  dump {sch} {tag:6s}: {'BYTE-IDENTICAL' if same else 'DIFFERS'}")
# B
for rid in ("t2_space_sin3_cn_nx8_nt256", "t2_space_sin_cn_nx8_nt256", "t2_cntime_sin3_nx64_nt4",
            "t2_betime_sin3_nx32_nt8", "t2_cnterm_sin3_nx16_nt32", "t1_rule_cn_l4", "t1_rule_be_l3",
            "t1_convon_cn_l4"):
    log = open(os.path.join(LP, "logs", "campaign", rid + ".log")).read().splitlines()
    args = log[0].split()[2:]
    out = run(args).splitlines()
    old_res = kv(next(l for l in log if l.startswith("EX21_EXACT_RESULT")))
    new_res = kv(next(l for l in out if l.startswith("EX21_EXACT_RESULT")))
    diff = [k for k in old_res if old_res[k] != new_res.get(k)]
    old_slots = [l for l in log if l.startswith("EX21_EXACT_SLOT")]
    new_slots = [l for l in out if l.startswith("EX21_EXACT_SLOT")]
    ok = not diff and old_slots == new_slots
    fails += not ok
    print(f"B  {rid:30s}: {len(old_res)} result keys, {len(old_slots)} slot lines -> "
          f"{'IDENTICAL' if ok else 'DIFFERS ' + str(diff[:6])}")
# G
for sch in ("cn", "be"):
    path = os.path.join(LP, "logs", "gates", f"verify_convection_t2_sin3_nx8_nt16_{sch}.log")
    old = open(path).read().splitlines()
    args = old[0].split()[2:] + ["--no-paraview"]
    new = [l for l in run(args).splitlines() if l.startswith("EX21_") or "rror" in l]
    ok = old[1:] == new
    fails += not ok
    print(f"G  verify-convection {sch}: {len(new)} lines {'IDENTICAL' if ok else 'DIFFERS'}")
    if not ok:
        for a, b in zip(old[1:], new):
            if a != b: print("   old:", a[:200]); print("   new:", b[:200])
print("REGRESSION", "PASS" if fails == 0 else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
