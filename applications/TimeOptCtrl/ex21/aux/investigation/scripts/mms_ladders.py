#!/usr/bin/env python3
"""Phase 4: manufactured convection-diffusion ladders with ex21 --convection-mms.

y = cos(2t) s(xi) [s(zeta)], s = sin (shape "sin") or sin^3 (shape "sin3"),
f = y_t - nu Lap y + v . grad y.  "sin" has f != 0 on the boundary once v != 0,
"sin3" has f = 0 there for every v (see EX21_CONVECTION_VALIDATION.md).
homogeneous Dirichlet data, T = 1.  Each ladder refines ONE parameter with the
other fixed fine enough that its own error is negligible; observed orders are
log2 of successive error ratios of the L2 error at t = T (and of the maximum
over the time levels).  Writes CSV + prints tables.
"""
import csv, math, os, re, subprocess, sys

EXE = "/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21"
WORK, OUT = sys.argv[1], sys.argv[2]
ENV = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")

D2 = ["--distributed-solve", "--state-bc", "dirichlet", "--nu", "0.5", "--T", "1"]
D1 = ["--distributed-solve", "--state-bc", "dirichlet", "--dimension", "1",
      "--domain", "0", "2", "0", "1", "--nu", "2", "--T", "1"]
CONV2 = ["--convection", "on", "--vx", "2", "--vy", "1"]
CONV1 = ["--convection", "on", "--vx", "2"]

LADDERS = []
SHAPE = {}
for scheme in ("be", "cn"):
    for conv, tag in ((CONV2, "v=(2,1)"), ([], "v=0")):
        LADDERS.append((f"2D time {scheme.upper()} {tag} nx=64", "steps",
                        [(D2 + conv + ["--time-scheme", scheme, "--nx", "64", "--nz", "64",
                                       "--steps", str(n)]) for n in (4, 8, 16, 32, 64)]))
    LADDERS.append((f"1D time {scheme.upper()} v=2 nx=256", "steps",
                    [(D1 + CONV1 + ["--time-scheme", scheme, "--nx", "256", "--steps", str(n)])
                     for n in (4, 8, 16, 32, 64, 128, 256)]))
for shape in ("sin", "sin3"):
    for conv, tag in ((CONV2, "v=(2,1)"), ([], "v=0")):
        t = f"2D space CN {tag} shape={shape} steps=512"
        SHAPE[t] = shape
        LADDERS.append((t, "nx", [(D2 + conv + ["--time-scheme", "cn", "--nx", str(n), "--nz",
                                                str(n), "--steps", "512"]) for n in (2, 4, 8, 16)]))
    for conv, tag in ((CONV1, "v=2"), ([], "v=0")):
        t = f"1D space CN {tag} shape={shape} steps=4096"
        SHAPE[t] = shape
        LADDERS.append((t, "nx", [(D1 + conv + ["--time-scheme", "cn", "--nx", str(n),
                                                "--steps", "4096"]) for n in (2, 4, 8, 16, 32)]))

rows = []
for title, key, runs in LADDERS:
    print(f"\n== {title}")
    prev = None
    for args in runs:
        os.makedirs(os.path.join(WORK, "output"), exist_ok=True)
        cmd = [EXE, "--convection-mms", SHAPE.get(title, "sin")] + args
        p = subprocess.run(cmd, cwd=WORK, env=ENV, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, timeout=7200)
        out = p.stdout.decode("utf-8", "replace")
        line = next((l for l in out.splitlines() if l.startswith("EX21_MMS")), None)
        if p.returncode != 0 or line is None:
            raise SystemExit(f"failed: {' '.join(cmd)}\n{out[-2000:]}")
        kv = dict(re.findall(r"(\w+)=(\S+)", line))
        r = {"ladder": title, "refined": key, **kv, "command": " ".join(["TimeOptCtrl_ex21"] + cmd[1:])}
        e, em = float(kv["err_L2_T"]), float(kv["err_L2_max"])
        r["order_T"] = "" if prev is None else f"{math.log2(prev[0] / e):.3f}"
        r["order_max"] = "" if prev is None else f"{math.log2(prev[1] / em):.3f}"
        prev = (e, em)
        rows.append(r)
        print(f"  {key}={kv['steps'] if key == 'steps' else kv['nx']:>5} dt={float(kv['dt']):.4e}"
              f" err_T={e:.4e} order={r['order_T']:>6}  err_max={em:.4e} order={r['order_max']:>6}"
              f"  rel_T={float(kv['rel_err_L2_T']):.3e} aao_res={kv['aao_residual']}", flush=True)

with open(OUT, "w", newline="") as f:
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "ladder", k))
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)
