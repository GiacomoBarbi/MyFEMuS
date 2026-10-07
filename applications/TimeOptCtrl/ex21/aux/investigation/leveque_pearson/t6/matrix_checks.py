#!/usr/bin/env python3
"""T6 matrix-level LPS checks -> ../t6_lps_matrix_checks.csv
  PE   active-patch count and max Pe_k of the ex21 assembly (EX21_LPS line) against
       the INDEPENDENT LP evaluation t4/post_t4.lp_patch (= t3/lp52_patch_peclet.py),
       eps = 1/20, 1/100, 1/500, Q2 nx = 4..128 (LP l = log2(nx)+1); --steps 2
       (only the spatial operator matters)
  DUMP independent dense checks from --exact-oc-dump on tiny meshes:
       W = L_lps - L_none exactly (the dumped _W is what was added),
       ||W - W^T||, eigenvalues of W (min / max, number of zero eigenvalues),
       for p0 and p1, LP wind (eps 1/500, 1/20) and q2-divfree (nu 0.01)
  G5   the in-code gate lines of logs/gates/*.log (reversal, PSD, Pe switch)
WORK=<scratch> matrix_checks.py"""
import csv, glob, math, os, re, subprocess, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
LP = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(LP, "t4")); sys.path.insert(0, os.path.join(LP, "t3"))
from post_t4 import lp_patch
X = os.environ.get("EX21_EXE", "/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21")
WORK = os.environ.get("WORK", "/tmp/t6_mat")
os.makedirs(os.path.join(WORK, "output"), exist_ok=True)
ENV = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
           PETSC_OPTIONS=" ".join(f"-{p}pc_type lu" for p in ("sweep_right_", "sweep_left_", "sweep_ay_", "fieldsplit_y_sweep_")))
rows, bad = [], 0
def kv(text, tag):
    l = next((l for l in text.splitlines() if l.startswith(tag + " ")), "")
    return dict(re.findall(r"(\S+?)=(\S+)", l))
# PE
for e in (20, 100, 500):
    for nx in (4, 8, 16, 32, 64, 128):
        t = subprocess.run([X, "--exact-oc", "lp52", "--exact-oc-nu", repr(1 / e), "--time-scheme", "cn", "--nx", str(nx),
                            "--nz", str(nx), "--steps", "2", "--distributed-solve", "--pc-schur-sweep", "--no-paraview",
                            "--convection-stabilization", "lps"], cwd=WORK, env=ENV, capture_output=True, text=True).stdout
        d = kv(t, "EX21_LPS")
        pmax, act, tot = lp_patch(1 / e, int(round(math.log2(nx))) + 1)
        ok = int(d["active"]) == act and int(d["patches"]) == tot and abs(float(d["pe_max"]) - pmax) <= 1e-5 * pmax
        bad += not ok
        rows.append(dict(check="PE", case=f"lp52 eps=1/{e} nx={nx} (LP l={int(round(math.log2(nx))) + 1})",
                         ex21_active=d["active"], ex21_patches=d["patches"], ex21_pe_max=d["pe_max"],
                         independent_active=act, independent_patches=tot, independent_pe_max=f"{pmax:.6e}",
                         delta_min=d["delta_min"], delta_max=d["delta_max"], W_fro=d["||W||_F"],
                         W_over_sym_L0=d["W/sym"], W_over_skew_L0=d["W/skew"], verdict="PASS" if ok else "FAIL"))
        print(rows[-1]["case"], rows[-1]["verdict"], d["active"], act, flush=True)
# DUMP
def load(f):
    with open(f) as fh:
        n = int(fh.readline().split()[0])
    a = np.loadtxt(f, skiprows=1, ndmin=2)
    A = np.zeros((n, n)); A[a[:, 0].astype(int), a[:, 1].astype(int)] = a[:, 2]
    return A
for name, args in (("lp52 eps=1/500 nx=4", ["--exact-oc", "lp52", "--exact-oc-nu", "0.002", "--nx", "4", "--nz", "4"]),
                   ("lp52 eps=1/500 nx=8", ["--exact-oc", "lp52", "--exact-oc-nu", "0.002", "--nx", "8", "--nz", "8"]),
                   ("lp52 eps=1/20 nx=8", ["--exact-oc", "lp52", "--nx", "8", "--nz", "8"]),
                   ("conv-sin3 q2-divfree nu=0.01 nx=6", ["--exact-oc", "conv-sin3", "--wind", "q2-divfree",
                                                          "--exact-oc-nu", "0.01", "--nx", "6", "--nz", "6"])):
    base = args + ["--steps", "2", "--time-scheme", "cn", "--distributed-solve", "--pc-schur-sweep", "--no-paraview"]
    pre0 = os.path.join(WORK, "d_none")
    subprocess.run([X, *base, "--exact-oc-dump", pre0], cwd=WORK, env=ENV, capture_output=True)
    L0 = load(pre0 + "_L.txt")
    for proj in ("p0", "p1"):
        pre = os.path.join(WORK, f"d_{proj}")
        subprocess.run([X, *base, "--convection-stabilization", "lps", "--lps-projection", proj, "--exact-oc-dump", pre],
                       cwd=WORK, env=ENV, capture_output=True)
        L, W = load(pre + "_L.txt"), load(pre + "_W.txt")
        nw = np.linalg.norm(W)
        # constrained (Dirichlet) rows: eliminated -> zero rows of W; restrict to free rows
        free = np.abs(L0).sum(axis=1) != np.abs(np.diag(L0))
        Wf = W[np.ix_(free, free)]
        ev = np.linalg.eigvalsh(0.5 * (Wf + Wf.T))
        diffL = np.abs((L - L0)[np.ix_(free, free)] - Wf).max() / max(np.abs(Wf).max(), 1e-300)
        sym = np.linalg.norm(W - W.T) / nw
        ok = diffL < 1e-13 and sym < 1e-14 and ev.min() > -1e-12 * ev.max()
        bad += not ok
        rows.append(dict(check="DUMP", case=f"{name} {proj}", W_fro=f"{nw:.6e}", L_minus_L0_vs_W=f"{diffL:.2e}",
                         sym_rel=f"{sym:.2e}", eig_min=f"{ev.min():.3e}", eig_max=f"{ev.max():.3e}",
                         eig_min_over_max=f"{ev.min() / ev.max():.2e}",
                         zero_eigs=int((np.abs(ev) < 1e-12 * ev.max()).sum()), free_dofs=int(free.sum()),
                         verdict="PASS" if ok else "FAIL"))
        print(rows[-1], flush=True)
# G5
for f in sorted(glob.glob(os.path.join(HERE, "logs", "gates", "*.log"))):
    t = open(f).read()
    g, s = kv(t, "EX21_VC_G5_LPS"), kv(t, "EX21_VC_G5_LPS_PE_SWITCH")
    if not g:
        continue
    ok = (float(g["sym_rel"]) == 0 and float(g["reversal_rel"]) == 0 and float(g["psd_min_rel"]) >= 0
          and g["reversal_active"] == g["active"] and float(g["W1_rel"]) < 1e-14
          and s["active"] == "0" and float(s["||W||_F"]) == 0)
    bad += not ok
    rows.append(dict(check="G5", case=os.path.basename(f)[:-4], ex21_active=g["active"], ex21_patches=g["patches"],
                     ex21_pe_max=g["pe_max"], W_fro=g["||W||_F"], sym_rel=g["sym_rel"], psd_min_rel=g["psd_min_rel"],
                     reversal_rel=g["reversal_rel"], W1_rel=g["W1_rel"], pe_switch_active=s["active"],
                     pe_switch_W_fro=s["||W||_F"], verdict="PASS" if ok else "FAIL"))
keys = []
for r in rows:
    keys += [k for k in r if k not in keys]
with open(os.path.join(LP, "t6_lps_matrix_checks.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
print("MATRIX CHECKS", "PASS" if not bad else f"FAIL ({bad})", len(rows), "rows")
