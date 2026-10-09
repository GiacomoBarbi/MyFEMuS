#!/usr/bin/env python3
"""T12 memory -> ../../t12_memory.csv.
(a) spatial ladders (CN, LU spatial factors): N_elem, N_space, nnz(M_full), nnz(L_full) (from an operator
    dump of the same mesh), space-time unknowns 2 N_t N_space, MaxRSS, bytes per space-time unknown;
(b) N_t scaling on one fixed mesh per family (the time-study runs, LU): MaxRSS vs N_t -> the per-slot
    increment (MB per time slot) and bytes per space-time unknown at the largest N_t (the ~O(N_t N_space)
    AAO law)."""
import csv, glob, os, re, subprocess
import numpy as np
X = os.environ.get("EX21_BIN", "/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21")
W = os.environ.get("T12_WORK", "/tmp/t12_mem"); os.makedirs(W + "/output", exist_ok=True)
D = os.path.dirname(os.path.abspath(__file__)) + "/.."
ENV = {"HOME": os.environ["HOME"], "PATH": "/home/giacomo/software/petsc/arch-linux-c-openblas-opt/bin:/usr/bin:/bin",
       "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
def nnz(m, n):
    p = f"{W}/{m}_{n}"
    if not os.path.exists(p + "_L_full.coo"):
        if m in ("hex", "tet"):
            a = ["--dimension", "3", "--domain3d", "0", "1", "0", "1", "0", "1", "--tracking-box3d", "0", "1", "0", "1", "0", "1",
                 "--nx", str(n), "--ny", str(n), "--nz", str(n), "--convection", "on", "--vx", "1", "--vy", "0.5", "--vz", "-0.75"]
        else:
            a = ["--domain", "0", "1", "0", "1", "--tracking-box", "0", "1", "0", "1", "--nx", str(n), "--nz", str(n),
                 "--convection", "on", "--vx", "1", "--vy", "0.5"]
        subprocess.run([X, "--mesh-type", m] + a + ["--distributed-solve", "--control-full-domain", "--state-bc", "dirichlet",
                        "--steps", "2", "--pc-schur-sweep", "--no-paraview", "--dump-operators", p], cwd=W, env=ENV,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    cnt = lambda f: sum(1 for _ in open(f)) - 3
    return cnt(p + "_M_full.coo"), cnt(p + "_L_full.coo")
rows = []
for f in ("t12_tri_space.csv", "t12_tet_space.csv"):
    for r in csv.DictReader(open(f"{D}/../{f}")):
        n, m = int(r["n"]), r["mesh"]
        nm, nl = nnz(m, n) if (m in ("quad", "tri") and n <= 32) or (m in ("hex", "tet") and n <= 8) else ("", "")
        st = 2 * int(r["nt"]) * int(r["dofs"])
        rows.append({"study": "space ladder (CN, LU)", "mesh": m, "n": n, "nt": r["nt"], "N_elem": r["elements"], "N_space": r["dofs"],
                     "nnz_M": nm, "nnz_L": nl, "nnz_L_per_row": f"{nl / int(r['dofs']):.1f}" if nl != "" else "",
                     "spacetime_unknowns": st, "maxrss_MB": r["maxrss_MB"], "bytes_per_spacetime_unknown": f"{int(r['maxrss_MB']) * 2**20 / st:.0f}",
                     "wall_s": r["wall_s"]})
for tag, m in (("tT", "tri"), ("qT", "quad"), ("tK", "tet"), ("hK", "hex")):
    pts = []
    for f in glob.glob(f"{D}/logs/runs/{tag}_be_n*_nt*.log"):
        s = open(f).read(); mr = re.search(r"MAXRSS=(\d+)", s)
        nt = int(re.search(r"_nt(\d+)\.log", f).group(1)); n = int(re.search(r"_n(\d+)_", f).group(1))
        if mr: pts.append((nt, int(mr.group(1)) / 1024.0, n))
    pts.sort()
    nts = np.array([p[0] for p in pts], float); mb = np.array([p[1] for p in pts])
    slope, icpt = np.polyfit(nts[nts >= 64], mb[nts >= 64], 1)
    n = pts[0][2]; N = (2 * n + 1) ** (3 if m in ("hex", "tet") else 2)
    for nt, rss, _ in pts:
        rows.append({"study": "N_t scaling (BE, LU, fixed mesh)", "mesh": m, "n": n, "nt": nt, "N_elem": "", "N_space": N, "nnz_M": "",
                     "nnz_L": "", "nnz_L_per_row": "", "spacetime_unknowns": 2 * nt * N, "maxrss_MB": f"{rss:.0f}",
                     "bytes_per_spacetime_unknown": f"{rss * 2**20 / (2 * nt * N):.0f}", "wall_s": "",
                     "fit_MB_per_slot(Nt>=64)": f"{slope:.3f}", "fit_bytes_per_slot_per_dof": f"{slope * 2**20 / N:.0f}"})
keys = list(dict.fromkeys(k for r in rows for k in r))
with open(f"{D}/../t12_memory.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
for r in rows: print(" ".join(f"{k}={r.get(k, '')}" for k in ("study", "mesh", "n", "nt", "N_elem", "N_space", "nnz_M", "nnz_L", "nnz_L_per_row", "spacetime_unknowns", "maxrss_MB", "bytes_per_spacetime_unknown", "fit_bytes_per_slot_per_dof")))
