#!/usr/bin/env python3
"""T15 spatial ladders (logs/exact/<tag>_cn_n*_nt64.log) -> ../../t15_space_convergence.csv (+ the LPS activity per run).
Ladders (t15_space.sh): gF/l0F/l1F fixed nu = 1e-3 (Galerkin / LPS-P0 / LPS-P1), gP/l0P/l1P fixed patch Peclet
Pe_K = 10 (nu = 0.0906/n), gH/l0H harmonic-3d fixed nu = 1e-3.  Orders between consecutive n (h = 1/n)."""
import csv, glob, math, os, re
D = os.path.dirname(os.path.abspath(__file__)) + "/.."
STUDY = {"gF": ("fixed nu=1e-3, wind A", "Galerkin"), "l0F": ("fixed nu=1e-3, wind A", "LPS-P0"),
         "l1F": ("fixed nu=1e-3, wind A", "LPS-P1 (diagnostic)"), "gP": ("fixed Pe_K=10, wind A", "Galerkin"),
         "l0P": ("fixed Pe_K=10, wind A", "LPS-P0"), "l1P": ("fixed Pe_K=10, wind A", "LPS-P1 (diagnostic)"),
         "gH": ("fixed nu=1e-3, harmonic-3d", "Galerkin"), "l0H": ("fixed nu=1e-3, harmonic-3d", "LPS-P0")}
rows = []
for tag, (study, method) in STUDY.items():
    runs = []
    for f in glob.glob(f"{D}/logs/exact/{tag}_cn_n*_nt64.log"):
        s = open(f).read()
        m = re.search(r"EX21_EXACT3D_RESULT .*", s)
        if not m:
            continue
        g = lambda k, t=m.group(0): re.search(rf" {k}=(\S+)", t).group(1)
        c = re.search(r"EX21_EXACT3D_CONFIG .*", s).group(0)
        a = re.search(r"EX21_LPS3D (.*)", s)
        a = dict(kv.split("=") for kv in a.group(1).split()) if a else {}
        r = {"study": study, "method": method, "tag": tag, "n": int(g("nx")), "nu": g("nu", c), "Pe_cell": g("Pe_cell", c),
             "y": float(g("y_L2L2")), "p": float(g("p_L2L2")), "u": float(g("u_L2L2")),
             "outer_its": g("outer_its"), "true_residual": g("true_residual"), "stationarity_rel": g("stationarity_rel"),
             "march_rel": g("march_rel"), "lps_active_fraction": a.get("active_fraction", ""), "lps_pe_min": a.get("pe_min", ""),
             "lps_pe_max": a.get("pe_max", ""), "lps_h_min": a.get("h_min", ""), "lps_h_max": a.get("h_max", ""),
             "lps_delta_min": a.get("delta_min", ""), "lps_delta_max": a.get("delta_max", ""), "lps_W_norm": a.get("||W||_F", ""),
             "lps_nnz_W": a.get("nnz_W", "")}
        mr = re.search(r"MAXRSS=(\d+) wall=(\S+)", s)
        r["maxrss_MB"], r["wall_s"] = (int(mr.group(1)) // 1024, mr.group(2)) if mr else ("", "")
        runs.append(r)
    runs.sort(key=lambda r: r["n"])
    for i, r in enumerate(runs):
        for k in ("y", "p", "u"):
            r["ord_" + k] = "" if i == 0 else f"{math.log(runs[i - 1][k] / r[k]) / math.log(r['n'] / runs[i - 1]['n']):.2f}"
        rows.append(r)
cols = list(rows[0].keys())
cols = cols[:7] + ["ord_y"] + cols[7:8] + ["ord_p"] + cols[8:9] + ["ord_u"] + [c for c in cols[9:] if not c.startswith("ord_")]
with open(D + "/../t15_space_convergence.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
for r in rows:
    print(f"{r['study']:<28}{r['method']:<22} n={r['n']:<2} nu={r['nu']:<9} y={r['y']:.3e}({r['ord_y']:>5}) p={r['p']:.3e}({r['ord_p']:>5}) "
          f"u={r['u']:.3e}({r['ord_u']:>5}) act={r['lps_active_fraction']:<8} Pe=[{r['lps_pe_min']},{r['lps_pe_max']}] its={r['outer_its']} res={r['true_residual']}")
