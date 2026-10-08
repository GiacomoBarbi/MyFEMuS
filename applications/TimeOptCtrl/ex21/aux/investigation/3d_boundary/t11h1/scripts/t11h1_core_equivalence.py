#!/usr/bin/env python3
"""T11H1 (copy of t11d_core_equivalence.py with the regularisation as a case dimension and the outer /
core iteration counts recorded): dense (gold) vs matrix-free (iterative CG) reduced Dirichlet core with
H1-space / H1-time / combined -> ../../t11h1_core_equivalence.csv.

Per case:
  (a) --dirichlet-core-solver dense --verify-dirichlet-core: the dense K = R + beta H is
      assembled column by column and checked: ||K - K^T||/||K||, lambda_min(K) > 0, cond(K),
      cond(R^-1 K), max over 9 probe vectors of ||K_dense v - K_shell v||/||K_dense v||,
      shell symmetry, sparse vs dense R_T;
  (b) solves with the dense core, the iterative core at the default rtol 1e-8 and at
      rtol 1e-12; production cases compare the full space-time fields y, p, u node by
      node (output/ex20_fields.json, max-norm relative) and J; the exact cases compare
      J_track and the y/p/u error norms of EX21_EXACT3D_RESULT.
Pass: SPD and action/symmetry <= 1e-13; iterative(1e-12) vs dense <= 1e-10;
iterative(1e-8) vs dense <= 1e-6 (the CG tolerance, preconditioned residual)."""
import csv, json, os, re, subprocess, sys
import numpy as np

X = os.environ.get("EX21_BIN", "/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21")
W = os.environ.get("T11H1_WORK", "/tmp/t11h1_core"); os.makedirs(W + "/output", exist_ok=True)
D = os.path.dirname(os.path.abspath(__file__)) + "/.."
os.makedirs(D + "/logs/core", exist_ok=True)
ENV = {"HOME": os.environ["HOME"], "PATH": "/home/giacomo/software/petsc/arch-linux-c-openblas-opt/bin:/usr/bin:/bin",
       "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
WA = ["--convection", "on", "--vx", "1", "--vy", "0.5", "--vz", "-0.75"]
WB = ["--convection", "on", "--vx", "-2", "--vy", "1", "--vz", "0.5"]
WN = ["--convection", "on", "--vx", "0", "--vy", "0", "--vz", "-0.75"]
HEAT = ["--convection", "off"]
P = ["--dimension", "3", "--dirichlet-solve", "--state-bc", "dirichlet", "--nu", "0.25", "--alpha-u", "1e-2",
     "--target-type", "sin_x_minus_t", "--T", "1"]
REGS = {"L2": [], "H1s": ["--control-h1-length", "0.3"], "H1t": ["--control-h1-time-weight", "1e-2"],
        "comb": ["--control-h1-length", "0.3", "--control-h1-time-weight", "1e-2"]}
BASE = [
    ("prod_xmax_wA_BE_n4", P + ["--dirichlet-face3d", "xmax", "--tracking-box3d", "0.3", "0.7", "0.2", "0.8", "0.2", "0.8",
                                "--nx", "4", "--ny", "4", "--nz", "4", "--steps", "8", "--time-scheme", "be"] + WA),
    ("prod_xmax_wA_CN_n4", P + ["--dirichlet-face3d", "xmax", "--tracking-box3d", "0.3", "0.7", "0.2", "0.8", "0.2", "0.8",
                                "--nx", "4", "--ny", "4", "--nz", "4", "--steps", "8", "--time-scheme", "cn"] + WA),
    ("prod_zmax_heat_CN_n4", P + ["--dirichlet-face3d", "zmax", "--tracking-box3d", "0.2", "0.8", "0.2", "0.8", "0", "0.7",
                                  "--nx", "4", "--ny", "4", "--nz", "4", "--steps", "8", "--time-scheme", "cn"] + HEAT),
    ("prod_nonunit_ymax_wA_BE", P + ["--dirichlet-face3d", "ymax", "--domain3d", "0", "2", "-1", "1", "0", "0.5",
                                     "--tracking-box3d", "0.5", "1.5", "-1", "0.4", "0", "0.5",
                                     "--nx", "4", "--ny", "4", "--nz", "2", "--steps", "8", "--time-scheme", "be"] + WA),
    ("exact_heat_BE_n3", ["--exact-oc", "dirichlet-h1-3d", "--nx", "3", "--ny", "3", "--nz", "3", "--steps", "8",
                          "--time-scheme", "be"] + HEAT),
    ("exact_wN_CN_n4", ["--exact-oc", "dirichlet-h1-3d", "--nx", "4", "--ny", "4", "--nz", "4", "--steps", "8",
                        "--time-scheme", "cn"] + WN),
]
CASES = [(f"{t}_{rg}", a + REGS[rg]) for rg in ("H1s", "H1t", "comb") for t, a in BASE] + \
        [(f"{t}_L2", a) for t, a in BASE[:2]]
rows, ok = [], True


def run(tag, args):
    jf = W + "/output/ex20_fields.json"
    if os.path.exists(jf):
        os.remove(jf)
    r = subprocess.run([X, "--no-paraview"] + args, cwd=W, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       universal_newlines=True)
    open(f"{D}/logs/core/{tag}.log", "w").write(r.stdout)
    if r.returncode != 0:
        raise RuntimeError(f"{tag}: exit {r.returncode}: {r.stdout[-800:]}")
    js = json.load(open(jf)) if os.path.exists(jf) else None
    return r.stdout, js


def g(p, s):
    m = re.search(p, s, re.M)
    return float(m.group(1)) if m else float("nan")


def fields(js):
    return {k: np.concatenate([np.array(e["values"], float) for e in js[k]]) for k in ("state", "adjoint", "control_full")}


def rd(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


for tag, args in CASES:
    s, _ = run(tag + "_verify", args + ["--dirichlet-core-solver", "dense", "--verify-dirichlet-core"])
    spd = {"m": int(g(r"CORE_SPD m=(\d+)", s)), "nC": int(g(r"CORE_SPD .*?nC=(\d+)", s)),
           "K_asym": g(r"\|\|K-K\^T\|\|_F/\|\|K\|\|_F=(\S+)", s), "lambda_min": g(r"CORE_SPD .*? lambda_min=(\S+)", s),
           "cond_K": g(r"cond_K=(\S+)", s), "cond_RinvK": g(r"cond_RinvK=(\S+)", s),
           "action_max": max(float(v) for v in re.findall(r"CORE_ACTION .*?=(\S+)$", s, re.M)),
           "shell_sym": g(r"CORE_SHELL_SYMMETRY .*?=(\S+)$", s), "RT_sparse_vs_dense": g(r"RT_SPARSE_VS_DENSE rel=(\S+)", s)}
    sd, jd = run(tag + "_dense", args + ["--dirichlet-core-solver", "dense"])
    res, outer = {}, {}
    outer["dense"] = int(g(r"outer_iterations=(\d+)", sd)) if "outer_iterations=" in sd else int(g(r"outer_its=(\d+)", sd))
    for lab, extra in (("it1e-8", []), ("it1e-12", ["--dirichlet-core-rtol", "1e-12"])):
        si, ji = run(tag + "_" + lab, args + ["--dirichlet-core-solver", "iterative"] + extra)
        its = int(g(r"core_its_max=(\d+)", si))
        outer[lab] = int(g(r"outer_iterations=(\d+)", si)) if "outer_iterations=" in si else int(g(r"outer_its=(\d+)", si))
        if jd is not None:
            fd, fi = fields(jd), fields(ji)
            dif = {f"d_{k}": rd(fi[k], fd[k]) for k in fd}
            dif["d_J"] = abs(ji["functional"]["J_total"] - jd["functional"]["J_total"]) / abs(jd["functional"]["J_total"])
        else:
            dif = {}
            for k in ("J_track", "y_L2L2", "p_L2L2", "u_L2L2"):
                a, b = g(rf"EXACT3D_RESULT .*? {k}=(\S+)", si), g(rf"EXACT3D_RESULT .*? {k}=(\S+)", sd)
                dif["d_" + k] = abs(a - b) / abs(b)
        res[lab] = (its, dif)
    good = spd["K_asym"] <= 1e-13 and spd["lambda_min"] > 0 and spd["action_max"] <= 1e-13 and spd["shell_sym"] <= 1e-13
    good &= spd["RT_sparse_vs_dense"] <= 1e-13
    good &= max(res["it1e-12"][1].values()) <= 1e-10 and max(res["it1e-8"][1].values()) <= 1e-6
    ok &= good
    row = {"case": tag, **spd, "its_1e-8": res["it1e-8"][0], "maxdiff_1e-8": f"{max(res['it1e-8'][1].values()):.2e}",
           "its_1e-12": res["it1e-12"][0], "maxdiff_1e-12": f"{max(res['it1e-12'][1].values()):.2e}",
           "outer_dense/1e-8/1e-12": f"{outer['dense']}/{outer['it1e-8']}/{outer['it1e-12']}",
           "compared": "fields y,p,u nodal + J" if jd is not None else "J_track + y/p/u error norms",
           "detail_1e-12": " ".join(f"{k}={v:.1e}" for k, v in res["it1e-12"][1].items()),
           "verdict": "PASS" if good else "FAIL"}
    rows.append(row)
    print(" ".join(f"{k}={v:.3e}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
with open(D + "/../t11h1_core_equivalence.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("T11H1_CORE_EQUIV_RESULT", "PASS" if ok else "FAIL", f"({sum(r['verdict'] == 'PASS' for r in rows)}/{len(rows)})")
sys.exit(0 if ok else 1)
