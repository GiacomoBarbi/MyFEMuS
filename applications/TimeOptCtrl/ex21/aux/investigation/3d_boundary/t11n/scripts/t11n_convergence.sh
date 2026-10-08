#!/bin/bash
# T11N convergence tables:
#   ../../t11n_space_convergence.csv   neumann-conv-3d exact errors (CN, N = 64; y, p in
#        L2(Omega), u in L2(Gamma_c)), winds A, B, heat; plus wind A with the measured
#        CN N = 64 time error (n = 6 self-convergence, same data) removed in quadrature
#   ../../t11n_time_convergence.csv    fixed mesh n = 6, ref N = 512, BE/CN (u on Gamma_c,
#        windows full / t <= T/2 / last slot), the CN y-layer split (3D) and the same
#        split for the 2D T7N neumann-conv case (attribution)
D=$(cd "$(dirname "$0")" && pwd); R=$D/../..; J=${T11N_JSON:-/tmp/t11n/json}; T=$(mktemp -d)
cd $D
python3 t11n_ladder_table.py $T/space.csv neuA neuB neuH > /dev/null
python3 t11n_time.py $J neut be 6 "4 8 16 32 64 128" 512 > $T/be.csv
python3 t11n_time.py $J neut cn 6 "4 8 16 32 64 128" 512 > $T/cn.csv
python3 t11n_cn_layer.py "$J/neut_cn_n6_nt{nt}.json" 6 "4 8 16 32 64 128" 512 > $T/layer3d.csv
grep -v "^==" $D/../logs/cn_layer_2d.log > $T/layer2d.csv
python3 - $T $R <<'PY'
import csv, math, sys
T, R = sys.argv[1], sys.argv[2]
rd = lambda p: list(csv.DictReader(open(p)))
sp = rd(f"{T}/space.csv")
tc = {int(r["nt"]): r for r in rd(f"{T}/cn.csv")}[64]
et = {"y": float(tc["y"]), "p": float(tc["p"]), "u": float(tc["u"])}
corr, prev = [], None
for r in sorted((r for r in sp if r["ladder"] == "neuA"), key=lambda r: int(r["n"])):
    n = int(r["n"]); e = {k: float(r[f"{k}_L2L2"]) for k in ("y", "p", "u")}
    es = {k: math.sqrt(max(e[k] ** 2 - et[k] ** 2, 0.0)) for k in e}
    row = {"ladder": "neuA (time error removed)", "n": n, "nt": 64}
    for k in e:
        row[f"{k}_L2L2"] = f"{es[k]:.4e}"; row[f"{k}_time_N64"] = f"{et[k]:.3e}"
        row[f"ord_{k}"] = "" if prev is None else f"{math.log(prev[k] / es[k]) / math.log(n / prev['n']):.2f}"
    corr.append(row); prev = dict(es, n=n)
def write(path, blocks):
    keys = list(dict.fromkeys(k for _, rows in blocks for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["study"] + keys); w.writeheader()
        for st, rows in blocks:
            for r in rows: w.writerow(dict(r, study=st))
write(f"{R}/t11n_space_convergence.csv", [("exact errors CN N=64 (u on Gamma_c)", sp),
                                          ("wind A, CN N=64 time error removed in quadrature", corr)])
write(f"{R}/t11n_time_convergence.csv", [("BE n=6 ref N=512", rd(f"{T}/be.csv")), ("CN n=6 ref N=512", rd(f"{T}/cn.csv")),
                                         ("CN y layer split, 3D n=6", rd(f"{T}/layer3d.csv")),
                                         ("CN y layer split, 2D T7N neumann-conv nx=8 (attribution)", rd(f"{T}/layer2d.csv"))])
for r in corr: print(" ".join(f"{k}={v}" for k, v in r.items()))
PY
rm -rf $T
