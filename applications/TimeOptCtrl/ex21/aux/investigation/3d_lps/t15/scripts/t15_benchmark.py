#!/usr/bin/env python3
"""T15 high-Pe layer benchmark table (t15_benchmark.sh) -> ../../t15_lps_benchmark.csv.
Per run: the LPS activity report (EX21_LPS3D: active fraction, Pe_K min/max, h_K min/max, delta_K min/max, ||W||_F,
nnz(W)), the solve (reason, outer its, true residual), the state extrema over space-time (the source-driven state is
>= 0 up to the control's effect; min y < 0 and an overshoot of max y are the Galerkin wiggles), the adjoint
extrema, J, and the y / p / J deviation from the finest LPS-P0 run of the same wind (n = 16, nodal values on the
coarse lattice: the n = 16 lattice contains every coarse node when 16 % n == 0)."""
import csv, json, os, re, sys
import numpy as np
D = os.path.dirname(os.path.abspath(__file__)) + "/.."
J = sys.argv[1] if len(sys.argv) > 1 else "/tmp/t15_bench_json"
rows = []


def load(tag):
    d = json.load(open(f"{J}/{tag}.json"))
    xyz = np.round(np.array(d["spatial_coordinates"]), 12)
    return d, xyz, np.array([s["values"] for s in d["state"]]), np.array([s["values"] for s in d["adjoint"]])


for w in ("A", "H"):
    dref, xr, yr, pr = load(f"{w}_p0_n16")
    key = {tuple(x): i for i, x in enumerate(xr)}
    for n in (4, 6, 8, 12, 16):
        for m in ("gal", "p0", "p1"):
            tag = f"{w}_{m}_n{n}"
            p = f"{D}/logs/benchmark/{tag}.log"
            if not os.path.exists(p) or not os.path.exists(f"{J}/{tag}.json"):
                continue
            L = open(p).read()
            a = re.search(r"EX21_LPS3D (.*)", L)
            a = dict(kv.split("=") for kv in a.group(1).split()) if a else {}
            d, xyz, y, pa = load(tag)
            r = {"wind": {"A": "constant A", "H": "harmonic-3d"}[w], "method": {"gal": "Galerkin", "p0": "LPS-P0", "p1": "LPS-P1"}[m],
                 "n": n, "h": f"{1 / n:.4f}", "active_fraction": a.get("active_fraction", "0"), "Pe_min": a.get("pe_min", ""),
                 "Pe_max": a.get("pe_max", ""), "h_K_min": a.get("h_min", ""), "h_K_max": a.get("h_max", ""),
                 "delta_min": a.get("delta_min", ""), "delta_max": a.get("delta_max", ""), "W_norm": a.get("||W||_F", ""),
                 "nnz_W": a.get("nnz_W", ""),
                 "reason": re.search(r"EX20_PAR_SOLVE .*? reason=(-?\d+)", L).group(1),
                 "outer_its": re.search(r"outer_iterations=(\d+)", L).group(1),
                 "true_residual": re.search(r"EX20_PAR_SOLVE .*? true_residual=(\S+)", L).group(1),
                 "y_min": f"{y.min():.4e}", "y_max": f"{y.max():.4e}", "undershoot_rel": f"{max(0.0, -y.min()) / y.max():.3e}",
                 "p_min": f"{pa.min():.4e}", "p_max": f"{pa.max():.4e}", "J": f"{d['functional']['J_total']:.10e}"}
            if 16 % n == 0:
                idx = [key[tuple(x)] for x in xyz]
                r["y_dev_vs_p0n16"] = f"{np.abs(y - yr[:, idx]).max() / np.abs(yr).max():.3e}"
                r["p_dev_vs_p0n16"] = f"{np.abs(pa - pr[:, idx]).max() / np.abs(pr).max():.3e}"
            else:
                r["y_dev_vs_p0n16"] = r["p_dev_vs_p0n16"] = ""
            r["J_dev_vs_p0n16"] = f"{abs(d['functional']['J_total'] - dref['functional']['J_total']) / abs(dref['functional']['J_total']):.3e}"
            mr = re.search(r"MAXRSS=(\d+) wall=(\S+)", L)
            r["maxrss_MB"], r["wall_s"] = (int(mr.group(1)) // 1024, mr.group(2)) if mr else ("", "")
            rows.append(r)
            print(" ".join(f"{k}={v}" for k, v in r.items() if k not in ("h_K_min", "h_K_max", "nnz_W")))
with open(D + "/../t15_lps_benchmark.csv", "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
