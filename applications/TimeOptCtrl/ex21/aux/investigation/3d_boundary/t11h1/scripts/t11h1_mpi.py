#!/usr/bin/env python3
"""T11H1 MPI comparison (copy of t11d_mpi.py; adds the R_H1t factor backend) -> ../../t11h1_mpi.csv.
usage: t11d_mpi.py <json-dir>
Per case and np: max |field - field_np1| / max |field_np1| for y, p, u (DOFs matched by
sorted coordinates); the control TRACE node by node: the set of nodes where u != 0 must be
the same int(Gamma_c) set at every np, count (2n_a-1)(2n_b-1), u = 0 elsewhere (incl. the
perimeter dGamma_c), and y = eta u on it (eta = 1); |J - J_np1|/|J_np1|; outer iterations
and the outer -ksp_monitor history vs np = 1; core CG iterations (distributed reduced layout)
vs np = 1.  PASS: fields < 1e-11, J < 1e-12, same outer and core counts, trace set identical,
y_C = eta u_C to 1e-13, outer history of the same length with r0 identical (1e-14 rel.) and every
later residual <= 1e-8 r0.  [The T11N criterion "history within 1e-12 of r0" is not used: the
exact Woodbury preconditioner converges in one outer step, so r1/r0 = 1e-11..1e-9 is ITSELF
round-off amplified through the space-time sweeps' direct factors (factor package and ordering
depend on np) and differs between np by its own size; outer rtol 1e-14 then takes a 2nd step
to ~1e-17 r0 at every np.]"""
import csv, json, os, re, sys
import numpy as np
J, D = sys.argv[1], os.path.dirname(os.path.abspath(__file__)) + "/.."
EXPECT = {"s_cn": 49, "t_be": 49, "k_cn": 49, "kh_be": 49, "knu_cn": 21}
def L(t):
    d = json.load(open(f"{J}/{t}.json"))
    xyz = np.round(np.array(d["spatial_coordinates"]), 12); o = np.lexsort((xyz[:, 0], xyz[:, 1], xyz[:, 2]))
    return d, {k: np.array([s["values"] for s in d[k]])[:, o] for k in ("state", "adjoint", "control_full")}, xyz[o]
def hist(t):
    return np.array([float(v) for v in re.findall(r"^\s*\d+ KSP Residual norm (\S+)", open(f"{D}/logs/mpi/{t}.log").read(), re.M)])
rows, ok = [], True
for case in EXPECT:
    d1, f1, x1 = L(f"{case}_np1"); h1 = hist(f"{case}_np1")
    face1 = np.abs(f1["control_full"]).max(0) > 0
    c1 = re.search(r"core_its_max=(\d+)", open(f"{D}/logs/mpi/{case}_np1.log").read()).group(1)
    for np_ in (1, 2, 4):
        t = f"{case}_np{np_}"; d, f, x = L(t); h = hist(t); log = open(f"{D}/logs/mpi/{t}.log").read()
        rss = [int(v) / 1024 for v in re.findall(r"RANK_MAXRSS=(\d+)", log)]
        r = {"case": case, "np": np_, "nrank": re.search(r"nrank=(\d+)", log).group(1), "dofs": f["state"].shape[1],
             "steps": d["config"]["N"], "coords_match": bool(np.array_equal(x, x1))}
        for k, lab in (("state", "y"), ("adjoint", "p"), ("control_full", "u")):
            r[lab + "_maxrel"] = f"{np.abs(f[k] - f1[k]).max() / np.abs(f1[k]).max():.2e}"
        face = np.abs(f["control_full"]).max(0) > 0
        r["trace_nodes"] = int(face.sum()); r["trace_expected"] = EXPECT[case]
        r["trace_set_equal_np1"] = bool(np.array_equal(face, face1))
        r["u_trace_maxrel"] = f"{np.abs(f['control_full'][:, face1] - f1['control_full'][:, face1]).max() / np.abs(f1['control_full']).max():.2e}"
        r["u_off_trace_max"] = f"{np.abs(f['control_full'][:, ~face1]).max():.1e}"
        r["max|y_C - eta u_C|/max|u|"] = f"{np.abs(f['state'][:, face1] - f['control_full'][:, face1]).max() / np.abs(f['control_full']).max():.1e}"
        r["J_rel"] = f"{abs(d['functional']['J_total'] - d1['functional']['J_total']) / abs(d1['functional']['J_total']):.2e}"
        r["true_res"] = f"{d['solve']['true_residual']:.3e}"; r["outer"] = d["solve"]["iterations"]
        r["core_its"] = re.search(r"core_its_max=(\d+)", log).group(1)
        r["rt_backend"] = re.search(r"rt_backend=(\S+)", log).group(1) if "rt_backend=" in log else ""
        r["hist_len"] = len(h)
        r["hist_max_dr_over_r0"] = f"{(np.abs(h - h1).max() / h1[0]) if len(h) == len(h1) and len(h) else float('nan'):.2e}"
        r["r0_rel_diff"] = f"{abs(h[0] - h1[0]) / h1[0]:.1e}"
        r["r_i/r0 (i>=1)"] = " ".join(f"{v / h[0]:.1e}" for v in h[1:])
        r["rank_maxrss_MB_max"] = f"{max(rss):.0f}"; r["sum_maxrss_MB"] = f"{sum(rss):.0f}"
        good = (max(float(r[k]) for k in ("y_maxrel", "p_maxrel", "u_maxrel")) < 1e-11 and float(r["J_rel"]) < 1e-12
                and r["outer"] == d1["solve"]["iterations"] and len(h) == len(h1) and float(r["r0_rel_diff"]) < 1e-14
                and all(v <= 1e-8 * h[0] for v in h[1:])
                and int(r["nrank"]) == np_ and float(r["u_off_trace_max"]) == 0.0 and r["trace_set_equal_np1"]
                and r["trace_nodes"] == EXPECT[case] and r["core_its"] == c1 and r["coords_match"]
                and float(r["max|y_C - eta u_C|/max|u|"]) < 1e-13)
        r["verdict"] = "PASS" if good else "FAIL"; ok &= good; rows.append(r)
        print(" ".join(f"{k}={v}" for k, v in r.items()))
with open(D + "/../t11h1_mpi.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("T11H1_MPI_RESULT", "PASS" if ok else "FAIL", f"({sum(r['verdict'] == 'PASS' for r in rows)}/{len(rows)})")
