#!/usr/bin/env python3
"""T15 MPI comparison by coordinates -> ../../t15_mpi.csv (copy of t14_mpi.py; cases A-G of t15_mpi.sh, LPS on).
Same criteria as T14 (SWEEP: fields < 1e-11, J < 1e-12, same outer count, history within 1e-12 of r0;
Dirichlet: + r0 identical, later residuals <= 1e-8 r0, core count +-1 (T14 revision), y_C = eta u_C), PLUS the LPS
partition invariance: the EX21_LPS3D activity line (active fraction, Pe min/max, h_K, delta, ||W||_F to 1e-12 rel.,
nnz(W)) identical at every np.
usage: t15_mpi.py <json-dir>"""
import csv, itertools, json, os, re, sys
import numpy as np
J, D = sys.argv[1], os.path.dirname(os.path.abspath(__file__)) + "/.."
CASES = {"A": ("V", ""), "B": ("V", ""), "C": ("V", ""), "D": ("N", "zmin,zmax"),
         "E": ("D", "zmin,xmin"), "F": ("D", "zmin,zmax"), "G": ("V", "")}


def lps(log):
    m = re.search(r"EX21_LPS3D (.*)", log)
    return dict(kv.split("=") for kv in m.group(1).split()) if m else {}
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
        # concurrent ranks can interleave their /usr/bin/time lines in the shared log: RSS is informational only
        r["rank_maxrss_MB_max"] = f"{max(rss):.0f}" if rss else "n/a (interleaved)"; r["sum_maxrss_MB"] = f"{sum(rss):.0f}" if len(rss) == np_ else "n/a (interleaved)"
        good = (max(float(r[k]) for k in ("y_maxrel", "p_maxrel", "u_maxrel")) < 1e-11 and float(r["J_rel"]) < 1e-12
                and r["outer"] == d1["solve"]["iterations"] and len(h) == len(h1) and int(r["nrank"]) == np_
                and (kind == "V" or (float(r["u_off_trace_max"]) == 0.0 and r["trace_set_equal_predicted"])) and r["coords_match"])
        if kind in ("N", "V"):
            good &= float(r["hist_max_dr_over_r0"]) < 1e-12
        else:
            good &= (float(r["r0_rel_diff"]) < 1e-14 and all(v <= 1e-8 * h[0] for v in h[1:]) and abs(int(r["core_its"]) - int(c1)) <= 1
                     and float(r["max|y_C - eta u_C|/max|u|"]) < 1e-13)
        a1, a = lps(log1), lps(log)
        r["lps_active_fraction"] = a.get("active_fraction", ""); r["lps_W_norm"] = a.get("||W||_F", ""); r["lps_nnz_W"] = a.get("nnz_W", "")
        lps_same = bool(a) and all(a[k] == a1[k] for k in a1 if k != "||W||_F") and \
            abs(float(a["||W||_F"]) - float(a1["||W||_F"])) <= 1e-12 * float(a1["||W||_F"])
        r["lps_identical_to_np1"] = lps_same; good &= lps_same
        r["verdict"] = "PASS" if good else "FAIL"; ok &= good; rows.append(r)
        print(" ".join(f"{k}={v}" for k, v in r.items()))
with open(D + "/../t15_mpi.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("T15_MPI_RESULT", "PASS" if ok else "FAIL", f"({sum(r['verdict'] == 'PASS' for r in rows)}/{len(rows)})")
sys.exit(0 if ok else 1)
