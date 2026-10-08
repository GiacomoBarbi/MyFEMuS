#!/bin/bash
# T11D convergence tables:
#   ../../t11d_space_convergence.csv  dirichlet-conv-3d exact errors (CN, N = 256; y, p in
#        L2(Omega), u in L2(Gamma_c), u nodal in time): dirN = normal wind (0,0,-0.75) (the
#        clean-order ladder), dirH = heat, dirA = wind A (1,0.5,-0.75) (documents the general
#        wind); plus dirN / dirH with the CN N = 256 time error removed in quadrature (e_t(256)
#        extrapolated from the n = 6 time ladder: e_t(128)^2 / e_t(64))
#   ../../t11d_time_convergence.csv   fixed mesh n = 6, wind N: BE (ref N = 1024), CN (ref N = 512)
D=$(cd "$(dirname "$0")" && pwd); R=$D/../..; J=${T11D_JSON:-/tmp/t11d_json}; T=$(mktemp -d)
cd $D
/usr/bin/python3.11 t11d_ladder_table.py $T/space.csv dirN dirH dirA > /dev/null
/usr/bin/python3.11 t11d_time.py $J dirt be 6 "4 8 16 32 64 128 256" 1024 > $T/be.csv
/usr/bin/python3.11 t11d_time.py $J dirt cn 6 "4 8 16 32 64 128" 512 > $T/cn.csv
/usr/bin/python3.11 - $T $R <<'PY'
import csv, math, sys
T, R = sys.argv[1], sys.argv[2]
rd = lambda p: list(csv.DictReader(open(p)))
sp = rd(f"{T}/space.csv")
tc = {int(r["nt"]): r for r in rd(f"{T}/cn.csv")}
et = {k: float(tc[128][k]) ** 2 / float(tc[64][k]) for k in ("y", "p", "u")}
corr = []
for lad in ("dirN", "dirH"):
    prev = None
    for r in sorted((r for r in sp if r["ladder"] == lad), key=lambda r: int(r["n"])):
        n = int(r["n"]); e = {k: float(r[f"{k}_L2L2"]) for k in ("y", "p", "u")}
        es = {k: math.sqrt(max(e[k] ** 2 - et[k] ** 2, 0.0)) for k in e}
        row = {"ladder": f"{lad} (time error removed)", "n": n, "nt": 256}
        for k in e:
            row[f"{k}_L2L2"] = f"{es[k]:.4e}"; row[f"{k}_time_N256_est"] = f"{et[k]:.3e}"
            row[f"ord_{k}"] = "" if prev is None else f"{math.log(prev[k] / es[k]) / math.log(n / prev['n']):.2f}"
        corr.append(row); prev = dict(es, n=n)
def write(path, blocks):
    keys = list(dict.fromkeys(k for _, rows in blocks for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["study"] + keys); w.writeheader()
        for st, rows in blocks:
            for r in rows: w.writerow(dict(r, study=st))
write(f"{R}/t11d_space_convergence.csv", [("exact errors CN N=256 (u on Gamma_c, nodal in time)", sp),
                                          ("CN N=256 time error (extrapolated, n=6 ladder) removed in quadrature", corr)])
write(f"{R}/t11d_time_convergence.csv", [("BE n=6 ref N=1024, wind (0,0,-0.75)", rd(f"{T}/be.csv")),
                                         ("CN n=6 ref N=512, wind (0,0,-0.75)", rd(f"{T}/cn.csv"))])
for r in corr: print(" ".join(f"{k}={v}" for k, v in r.items()))
PY
rm -rf $T
