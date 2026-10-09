#!/usr/bin/env python3
"""T14 MPI comparison by coordinates -> ../../t14_mpi.csv (t13_mpi.py with variable winds; kind V = distributed /
natural: no control trace, u compared as a volume field; SWEEP criterion).
usage: t13_mpi.py <json-dir>
Per case and np: max |field - field_np1| / max |field_np1| (y, p, u; DOFs matched by sorted coordinates),
|J - J_np1| / |J_np1|, outer iterations and the -ksp_monitor history vs np = 1, the control TRACE: the
node set where u != 0 must equal the predicted set of the union (lattice count: Neumann = nodes on the
union, Dirichlet = union minus the nodes of uncontrolled faces) at every np, u = 0 elsewhere.
PASS, Neumann (T11N criterion): fields < 1e-11, J < 1e-12, same outer count, history within 1e-12 of r0.
PASS, Dirichlet (T11D criterion): fields < 1e-11, J < 1e-12, same outer and core counts, history of the same
length with r0 identical (1e-14 rel.) and every later residual <= 1e-8 r0, y_C = eta u_C to 1e-13, core CG
count (max over the core solves) within +-1 of np = 1.
[REVISED after the first table (T11D required the same core count): Dopp_be np = 2 needed 14 core
iterations vs 13 at np = 1.  The core residual histories (logs/mpi_core_diag, -dirichlet_core_ksp_monitor_
true_residual): the FIRST core solve (the physical right-hand side) stops at 12 at both np with matching
residuals (4.24e-14 / 4.02e-14 at it 12); the SECOND core solve's right-hand side is the outer residual after
step 1, i.e. round-off (r1/r0 = 6.31e-11 at np = 1, 6.08e-11 at np = 2), and CG on that noise crosses
rtol 1e-12 at 13 vs 14.  Same mechanism as the T11D outer-history note; the fields agree to 1.4e-15.]"""
import csv, itertools, json, os, re, sys
import numpy as np
J, D = sys.argv[1], os.path.dirname(os.path.abspath(__file__)) + "/.."
CASES = {"Vdh": ("V", ""), "Vdf": ("V", ""), "Vnf": ("V", ""), "Nmh": ("N", "zmin"), "Nof": ("N", "zmin,zmax"),
         "Drs": ("D", "zmin"), "Daf": ("D", "zmin,xmin"), "Dh1": ("D", "zmin,zmax")}
AX = {"x": 0, "y": 1, "z": 2}


def expected_set(xyz, kind, faces):
    on = {f: np.isclose(xyz[:, AX[f[0]]], 1.0 if f.endswith("max") else 0.0, atol=1e-12) for f in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")}
    u = np.zeros(len(xyz), bool); unc = np.zeros(len(xyz), bool)
    for f, m in on.items():
        if f in faces: u |= m
        else: unc |= m
    return u if kind == "N" else (u & ~unc)


def L(t):
    d = json.load(open(f"{J}/{t}.json"))
    xyz = np.round(np.array(d["spatial_coordinates"]), 12); o = np.lexsort((xyz[:, 0], xyz[:, 1], xyz[:, 2]))
    return d, {k: np.array([s["values"] for s in d[k]])[:, o] for k in ("state", "adjoint", "control_full")}, xyz[o]


def hist(t):
    return np.array([float(v) for v in re.findall(r"^\s*\d+ KSP Residual norm (\S+)", open(f"{D}/logs/mpi/{t}.log").read(), re.M)])


rows, ok = [], True
for case, (kind, faces) in CASES.items():
    faces = faces.split(",")
    d1, f1, x1 = L(f"{case}_np1"); h1 = hist(f"{case}_np1")
    want = expected_set(x1, kind, faces) if kind != "V" else np.ones(len(x1), bool)
    log1 = open(f"{D}/logs/mpi/{case}_np1.log").read()
    c1 = re.search(r"core_its_max=(\d+)", log1).group(1) if kind == "D" else ""
    for np_ in (1, 2, 4):
        t = f"{case}_np{np_}"; d, f, x = L(t); h = hist(t); log = open(f"{D}/logs/mpi/{t}.log").read()
        rss = [int(v) / 1024 for v in re.findall(r"RANK_MAXRSS=(\d+)", log)]
        r = {"case": case, "kind": {"N": "Neumann", "D": "Dirichlet", "V": "distributed/natural"}[kind], "faces": "+".join(faces), "np": np_,
             "nrank": re.search(r"nrank=(\d+)", log).group(1), "dofs": f["state"].shape[1], "steps": d["config"]["N"],
             "coords_match": bool(np.array_equal(x, x1))}
        for k, lab in (("state", "y"), ("adjoint", "p"), ("control_full", "u")):
            r[lab + "_maxrel"] = f"{np.abs(f[k] - f1[k]).max() / np.abs(f1[k]).max():.2e}"
        face = np.abs(f["control_full"]).max(0) > 0
        r["trace_nodes"] = int(face.sum()); r["trace_expected"] = int(want.sum())
        r["trace_set_equal_predicted"] = bool(np.array_equal(face, want))
        r["u_off_trace_max"] = f"{np.abs(f['control_full'][:, ~want]).max():.1e}" if (~want).any() else "0.0e+00"
        if kind == "D":
            r["max|y_C - eta u_C|/max|u|"] = f"{np.abs(f['state'][:, want] - f['control_full'][:, want]).max() / np.abs(f['control_full']).max():.1e}"
        else:
            r["max|y_C - eta u_C|/max|u|"] = ""
        r["J_rel"] = f"{abs(d['functional']['J_total'] - d1['functional']['J_total']) / abs(d1['functional']['J_total']):.2e}"
        r["true_res"] = f"{d['solve']['true_residual']:.3e}"; r["outer"] = d["solve"]["iterations"]
        r["core_its"] = re.search(r"core_its_max=(\d+)", log).group(1) if kind == "D" else ""
        r["hist_len"] = len(h)
        r["hist_max_dr_over_r0"] = f"{(np.abs(h - h1).max() / h1[0]) if len(h) == len(h1) and len(h) else float('nan'):.2e}"
        r["r0_rel_diff"] = f"{abs(h[0] - h1[0]) / h1[0]:.1e}" if len(h) else ""
        r["rank_maxrss_MB_max"] = f"{max(rss):.0f}"; r["sum_maxrss_MB"] = f"{sum(rss):.0f}"
        good = (max(float(r[k]) for k in ("y_maxrel", "p_maxrel", "u_maxrel")) < 1e-11 and float(r["J_rel"]) < 1e-12
                and r["outer"] == d1["solve"]["iterations"] and len(h) == len(h1) and int(r["nrank"]) == np_
                and (kind == "V" or (float(r["u_off_trace_max"]) == 0.0 and r["trace_set_equal_predicted"])) and r["coords_match"])
        if kind in ("N", "V"):
            good &= float(r["hist_max_dr_over_r0"]) < 1e-12
        else:
            good &= (float(r["r0_rel_diff"]) < 1e-14 and all(v <= 1e-8 * h[0] for v in h[1:]) and abs(int(r["core_its"]) - int(c1)) <= 1
                     and float(r["max|y_C - eta u_C|/max|u|"]) < 1e-13)
        r["verdict"] = "PASS" if good else "FAIL"; ok &= good; rows.append(r)
        print(" ".join(f"{k}={v}" for k, v in r.items()))
with open(D + "/../t14_mpi.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("T14_MPI_RESULT", "PASS" if ok else "FAIL", f"({sum(r['verdict'] == 'PASS' for r in rows)}/{len(rows)})")
sys.exit(0 if ok else 1)
