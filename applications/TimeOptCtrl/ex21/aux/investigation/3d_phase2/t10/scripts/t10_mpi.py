#!/usr/bin/env python3
"""T10 MPI comparison by coordinates -> ../../t10_mpi.csv.  usage: t10_mpi.py <json-dir>
Per case and np: max |field - field_np1| / max |field_np1| for y, p, u (DOFs matched
by sorted coordinates), |J - J_np1| / |J_np1|, true residual, outer iterations and
the outer FGMRES history (-ksp_monitor) against np = 1.  np > 1 uses MUMPS LU and
is not bitwise reproducible (T9: ~2e-15 run to run), so PASS means round-off:
fields < 1e-11, J < 1e-12, same outer count, history within 1e-12 of r0."""
import csv, json, os, re, sys
import numpy as np
J, D = sys.argv[1], os.path.dirname(os.path.abspath(__file__)) + "/.."
def L(t):
    d = json.load(open(f"{J}/{t}.json"))
    xyz = np.round(np.array(d["spatial_coordinates"]), 12); o = np.lexsort((xyz[:, 0], xyz[:, 1], xyz[:, 2]))
    return d, {k: np.array([s["values"] for s in d[k]])[:, o] for k in ("state", "adjoint", "control_full")}
def hist(t):
    return np.array([float(v) for v in re.findall(r"^\s*\d+ KSP Residual norm (\S+)", open(f"{D}/logs/mpi/{t}.log").read(), re.M)])
rows, ok = [], True
for case in ("mz_be", "src_cn", "nat_be", "nat_cn", "nu_be"):
    d1, f1 = L(f"{case}_np1"); h1 = hist(f"{case}_np1")
    for np_ in (1, 2, 4):
        t = f"{case}_np{np_}"; d, f = L(t); h = hist(t); log = open(f"{D}/logs/mpi/{t}.log").read()
        rss = [int(v) / 1024 for v in re.findall(r"RANK_MAXRSS=(\d+)", log)]
        r = {"case": case, "np": np_, "nrank": re.search(r"nrank=(\d+)", log).group(1), "dofs": f["state"].shape[1],
             "steps": d["config"]["N"]}
        for k, lab in (("state", "y"), ("adjoint", "p"), ("control_full", "u")):
            r[lab + "_maxrel"] = f"{np.abs(f[k] - f1[k]).max() / np.abs(f1[k]).max():.2e}"
        r["J_rel"] = f"{abs(d['functional']['J_total'] - d1['functional']['J_total']) / abs(d1['functional']['J_total']):.2e}"
        r["true_res"] = f"{d['solve']['true_residual']:.3e}"; r["outer"] = d["solve"]["iterations"]
        r["hist_len"] = len(h)
        r["hist_max_dr_over_r0"] = f"{(np.abs(h - h1).max() / h1[0]) if len(h) == len(h1) and len(h) else float('nan'):.2e}"
        r["rank_maxrss_MB_max"] = f"{max(rss):.0f}"; r["sum_maxrss_MB"] = f"{sum(rss):.0f}"
        good = (max(float(r[k]) for k in ("y_maxrel", "p_maxrel", "u_maxrel")) < 1e-11 and float(r["J_rel"]) < 1e-12
                and r["outer"] == d1["solve"]["iterations"] and len(h) == len(h1) and float(r["hist_max_dr_over_r0"]) < 1e-12
                and int(r["nrank"]) == np_)
        r["verdict"] = "PASS" if good else "FAIL"; ok &= good; rows.append(r)
        print(" ".join(f"{k}={v}" for k, v in r.items()))
with open(D + "/../t10_mpi.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("T10_MPI_RESULT", "PASS" if ok else "FAIL")
