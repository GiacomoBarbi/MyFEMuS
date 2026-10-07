#!/usr/bin/env python3
"""Phase 1: ex21 with v = 0 must reproduce ex20.

For every case the ex20 binary and two ex21 variants are run:
  ex21_default  : no ex21 option at all (the ex20 code path)
  ex21_conv_v0  : --convection on --vx 0 --vy 0  (convection machinery active:
                  adjoint products through MatMultTranspose, transposed sweep
                  blocks, GMRES spatial sweeps -- with a zero velocity)
and the JSON fields (state, adjoint, control, objective) and the solver
statistics are compared with ex20.  Writes a CSV and prints a table.
"""
import csv, json, os, re, subprocess, sys

B = "/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl"
EX20, EX21 = f"{B}/ex20/TimeOptCtrl_ex20", f"{B}/ex21/TimeOptCtrl_ex21"
WORK = sys.argv[1]
OUT = sys.argv[2]
COMMON = ["--nx", "8", "--nz", "8", "--steps", "16", "--no-paraview",
          "--target-type", "sin_x_minus_t", "--alpha-u", "1e-3"]
CASES = [
    ("dist_BE_additive", ["--distributed-solve"], True),
    ("dist_CN_additive", ["--distributed-solve", "--time-scheme", "cn"], True),
    ("dist_BE_matching", ["--distributed-solve", "--pc-schur-matching"], True),
    ("dist_BE_sweep", ["--distributed-solve", "--pc-schur-sweep"], True),
    ("dist_CN_sweep", ["--distributed-solve", "--time-scheme", "cn", "--pc-schur-sweep"], True),
    ("dist_CN_matching", ["--distributed-solve", "--time-scheme", "cn", "--pc-schur-matching"], True),
    ("neumann_BE_additive", ["--neumann-solve"], False),
    ("neumann_CN_boundary_reduced", ["--neumann-solve", "--time-scheme", "cn",
                                     "--pc-schur-boundary-reduced"], False),
    ("dirichlet_CN_reduced", ["--dirichlet-solve", "--time-scheme", "cn"], False),
]
ENV = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")


def run(exe, args, tag):
    d = os.path.join(WORK, tag)
    os.makedirs(os.path.join(d, "output"), exist_ok=True)
    cmd = [exe] + COMMON + args + ["--dump-json", "out.json"]
    p = subprocess.run(cmd, cwd=d, env=ENV, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, timeout=3600)
    out = p.stdout.decode("utf-8", "replace")
    if p.returncode != 0:
        raise SystemExit(f"{tag} failed rc={p.returncode}\n{out[-2000:]}")
    j = json.load(open(os.path.join(d, "out.json")))
    stats = dict(re.findall(r"(\w+)=([^\s\[]+)", next(
        l for l in out.splitlines() if l.startswith("EX20_SOLVERSTATS"))))
    return j, stats, " ".join(os.path.basename(cmd[0]) if i == 0 else c
                              for i, c in enumerate(cmd))


def flat(x):
    if isinstance(x, dict):           # {"slot", "time", "values"}
        yield from flat(x["values"])
    elif isinstance(x, list):
        for e in x:
            yield from flat(e)
    else:
        yield float(x)


def maxrel(a, b):
    fa, fb = list(flat(a)), list(flat(b))
    assert len(fa) == len(fb)
    scale = max(1e-300, max(abs(v) for v in fb))
    return max(abs(x - y) for x, y in zip(fa, fb)) / scale


rows = []
for name, args, conv_ok in CASES:
    ref, sref, cref = run(EX20, args, f"{name}_ex20")
    variants = [("ex21_default", [])]
    if conv_ok:
        variants.append(("ex21_conv_v0", ["--convection", "on", "--vx", "0", "--vy", "0"]))
    for vname, extra in variants:
        got, sgot, cgot = run(EX21, args + extra, f"{name}_{vname}")
        row = {"case": name, "variant": vname, "command": cgot,
               "ex20_command": cref}
        for f in ("state", "adjoint", "control_full"):
            row[f"{f}_maxrel"] = maxrel(got[f], ref[f])
        for f in ("J_track", "J_reg", "J_total"):
            a, b = got["functional"][f], ref["functional"][f]
            row[f"{f}_rel"] = abs(a - b) / abs(b)
            row[f"{f}_ex21"] = a
        row["outer_ex20"], row["outer_ex21"] = ref["solve"]["iterations"], got["solve"]["iterations"]
        row["true_res_ex20"], row["true_res_ex21"] = ref["solve"]["true_residual"], got["solve"]["true_residual"]
        row["reason_ex21"] = got["solve"]["reason"]
        for k in ("y", "p"):
            row[f"inner_{k}_its_ex20"] = ref["solve"][f"inner_{k}_iterations"]
            row[f"inner_{k}_its_ex21"] = got["solve"][f"inner_{k}_iterations"]
        row["bitwise"] = (row["state_maxrel"] == 0 and row["adjoint_maxrel"] == 0
                          and row["control_full_maxrel"] == 0 and row["J_total_rel"] == 0)
        rows.append(row)
        print(f"{name:30s} {vname:13s} state={row['state_maxrel']:.2e} adj={row['adjoint_maxrel']:.2e}"
              f" u={row['control_full_maxrel']:.2e} J={row['J_total_rel']:.2e}"
              f" outer {row['outer_ex20']}->{row['outer_ex21']}"
              f" inner_y {row['inner_y_its_ex20']}->{row['inner_y_its_ex21']}"
              f" inner_p {row['inner_p_its_ex20']}->{row['inner_p_its_ex21']}"
              f" res {row['true_res_ex20']:.2e}->{row['true_res_ex21']:.2e}"
              f" bitwise={row['bitwise']}", flush=True)

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
