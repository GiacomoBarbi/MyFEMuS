#!/usr/bin/env python3
"""Collect every Example 3 run: solver statistics, objective terms, error
metrics and the pointwise comparison with Figures 32/33.  Writes
results/math2025_ex3_runs.csv and prints a markdown table."""
import csv, glob, io, json, os, re, sys, contextlib

HERE = os.path.dirname(os.path.abspath(__file__))
E = os.path.dirname(HERE)
RES = os.path.join(os.path.dirname(E), "results")
sys.path.insert(0, HERE)
import compare  # noqa: E402

rows = []
for log in sorted(glob.glob(os.path.join(E, "logs", "*.log"))):
    name = os.path.basename(log)[:-4]
    text = open(log).read()
    m = re.search(r"^EX21_EX3_RESULT (.*)$", text, re.M)
    if not m:
        continue
    r = dict(re.findall(r"(\w+)=(\S+)", m.group(1)))
    cfg = dict(re.findall(r"(\w+)=(\S+)", re.search(r"^EX21_EX3_CONFIG (.*)$", text, re.M).group(1)))
    cmd = re.search(r"^CMD: (.*)$", text, re.M)
    wall = re.search(r"WALL (\S+) s MAXRSS (\S+) kB", text)
    with contextlib.redirect_stdout(io.StringIO()):
        compare.main(log, name)
    c = json.load(open(os.path.join(RES, f"math2025_ex3_compare_{name}.json")))
    row = {"run": name, "scheme": cfg["scheme"], "nx": cfg["nx"], "steps": cfg["steps"],
           "T": cfg["T"], "dt": cfg["dt"], "nu": cfg["nu"], "vx": cfg["vx"],
           "initial": cfg["initial"], "pc": cfg["pc"], **r,
           "wall_s": wall.group(1) if wall else "", "maxrss_kB": wall.group(2) if wall else "",
           "command": cmd.group(1) if cmd else ""}
    for key, lab in (("fig32_Eoc_c_0_19.7", "Eoc32_0_19.7"), ("fig32_Eoc_c_0_20", "Eoc32_0_20"),
                     ("fig32_Eoc_c_19.7_20", "Eoc32_19.7_20"), ("fig33_Eoc_c_0_2", "Eoc33_0_2"),
                     ("fig32_e2_cascade_c_0_19.7", "e2_32_0_19.7"), ("fig33_e2_cascade_c_0_2", "e2_33_0_2")):
        s = c[key]
        row[f"{lab}_linf"], row[f"{lab}_rms"], row[f"{lab}_rmsrel"] = s["linf"], s["rms"], s["rms_rel"]
    pa, pb = c["peaks_ex21"], c["peaks_Eoc"]
    n = min(len(pa), len(pb))
    row["peak_shift_max"] = max(abs(pa[k][0] - pb[k][0]) for k in range(n))
    row["peak_amp_diff_max"] = max(abs(pa[k][1] - pb[k][1]) for k in range(n))
    rows.append(row)

keys = list(rows[0].keys())
with open(os.path.join(RES, "math2025_ex3_runs.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)

print("| run | its | true res | J_track | J_reg | ‖u‖_L2 | sup e [10,20] | RMS e [0,20] | vs E_oc RMS [0,19.7] | L∞ [0,19.7] | Fig.33 RMS | peak Δt / Δamp | vs e₂ RMS [0,19.7] | wall s |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r['run']} | {r['outer_its']} | {float(r['true_residual']):.1e} | {float(r['J_track']):.6e} |"
          f" {float(r['J_reg']):.4e} | {float(r['u_L2']):.3f} | {float(r['sup_e_10_20']):.6f} |"
          f" {float(r['rms_e_0_20']):.6f} | {r['Eoc32_0_19.7_rms']:.2e} | {r['Eoc32_0_19.7_linf']:.2e} |"
          f" {r['Eoc33_0_2_rms']:.2e} | {r['peak_shift_max']:.0e} / {r['peak_amp_diff_max']:.1e} |"
          f" {r['e2_32_0_19.7_rms']:.2e} | {r['wall_s']} |")
