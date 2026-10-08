#!/bin/bash
# T10 convergence tables (after t10_campaign.sh); the natural and multi-zone time
# ladders go to N = 128 against N = 512 (one level beyond N = 64, where the BE
# adjoint was still pre-asymptotic):
#   ../../t10_convergence.csv  natural-conv-3d exact spatial ladders (winds A, B,
#                              heat; CN, N = 64) + natural fixed-mesh time ladders
#                              (n = 6, ref N = 256, BE/CN, with the CN-layer split)
#   ../../t10_multizone.csv    multi-zone fixed-dt space self-convergence (n = 2,4,8
#                              vs 16, band/interior split) + fixed-mesh time ladders
#                              (n = 4, ref N = 256, BE/CN, CN-layer split)
D=$(cd "$(dirname "$0")" && pwd); R=$D/../..; J=${T10_JSON:-/tmp/t10/json}; T=$(mktemp -d)
cd $D
python3 t10_ladder_table.py $T/space.csv natA natB natH > $T/space.out
python3 t10_time.py $J natt be 6 "4 8 16 32 64 128" 512 > $T/nat_be.csv
python3 t10_time.py $J natt cn 6 "4 8 16 32 64 128" 512 > $T/nat_cn.csv
python3 t10_cn_layer.py "$J/natt_cn_n6_nt{nt}.json" 6 "4 8 16 32 64 128" 512 > $T/nat_layer.csv
python3 t10_mz_space.py $J mzs cn 8 "2 4 8" 16 $T/mz_space.csv > $T/mz_space.out
python3 t10_selfconv.py $J natsA_cn 16 "2 4 8" 16 > $T/nat_selfA.csv
python3 t10_selfconv.py $J natsH_cn 16 "2 4 8" 16 > $T/nat_selfH.csv
python3 t10_selfconv.py $J natsA_be 16 "2 4 8" 16 > $T/nat_selfA_be.csv
python3 t10_selfconv.py $J natsH_be 16 "2 4 8" 16 > $T/nat_selfH_be.csv
# exact-error ladder (wind A, CN N=64) with the N=64 time error of the same data
# (natt CN, n=6, vs N=512) removed in quadrature: e_s = sqrt(e^2 - e_t^2)
python3 - $T > $T/nat_corr.csv <<'PYC'
import csv, math, sys
T = sys.argv[1]
sp = [r for r in csv.DictReader(open(f"{T}/space.csv")) if r["ladder"] == "natA"]
tc = {int(r["nt"]): r for r in csv.DictReader(open(f"{T}/nat_cn.csv"))}[64]
et = {"y": float(tc["y"]), "p": float(tc["p"]), "u": float(tc["u"])}
print("n,y_exact,y_time_N64,y_space,y_space_order,p_exact,p_time_N64,p_space,p_space_order,u_space,u_space_order")
prev = None
for r in sorted(sp, key=lambda r: int(r["n"])):
    n = int(r["n"]); e = {k: float(r[f"{k}_L2L2"]) for k in ("y", "p", "u")}
    es = {k: math.sqrt(max(e[k] ** 2 - et[k] ** 2, 0.0)) for k in e}
    o = {k: ("" if prev is None else f"{math.log(prev[1][k] / es[k]) / math.log(n / prev[0]):.2f}") for k in e}
    print(f"{n},{e['y']:.4e},{et['y']:.3e},{es['y']:.4e},{o['y']},{e['p']:.4e},{et['p']:.3e},{es['p']:.4e},{o['p']},{es['u']:.4e},{o['u']}")
    prev = (n, es)
PYC
python3 t10_time.py $J mzt be 4 "4 8 16 32 64 128" 512 > $T/mz_be.csv
python3 t10_time.py $J mzt cn 4 "4 8 16 32 64 128" 512 > $T/mz_cn.csv
python3 t10_cn_layer.py "$J/mzt_cn_n4_nt{nt}.json" 4 "4 8 16 32 64 128" 512 > $T/mz_layer.csv
python3 - $T $R <<'PY'
import csv, sys
T, R = sys.argv[1], sys.argv[2]
def rd(p): return list(csv.DictReader(open(p)))
def write(path, blocks):
    keys = list(dict.fromkeys(k for _, rows in blocks for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["study"] + keys); w.writeheader()
        for study, rows in blocks:
            for r in rows: w.writerow(dict(r, study=study))
write(f"{R}/t10_convergence.csv", [
    ("natural exact space (CN N=64)", rd(f"{T}/space.csv")),
    ("natural time BE (n=6, ref N=512)", rd(f"{T}/nat_be.csv")),
    ("natural time CN (n=6, ref N=512)", rd(f"{T}/nat_cn.csv")),
    ("natural CN y layer split (n=6)", rd(f"{T}/nat_layer.csv")),
    ("natural fixed-dt space self-conv wind A (CN N=16, ref n=16)", rd(f"{T}/nat_selfA.csv")),
    ("natural fixed-dt space self-conv heat (CN N=16, ref n=16)", rd(f"{T}/nat_selfH.csv")),
    ("natural fixed-dt space self-conv wind A (BE N=16, ref n=16)", rd(f"{T}/nat_selfA_be.csv")),
    ("natural fixed-dt space self-conv heat (BE N=16, ref n=16)", rd(f"{T}/nat_selfH_be.csv")),
    ("natural exact space wind A, CN N=64 time error removed in quadrature", rd(f"{T}/nat_corr.csv"))])
write(f"{R}/t10_multizone.csv", [
    ("multizone space self-conv (CN N=8, ref n=16)", rd(f"{T}/mz_space.csv")),
    ("multizone time BE (n=4, ref N=512)", rd(f"{T}/mz_be.csv")),
    ("multizone time CN (n=4, ref N=512)", rd(f"{T}/mz_cn.csv")),
    ("multizone CN y layer split (n=4)", rd(f"{T}/mz_layer.csv"))])
PY
cat $T/space.out | cut -c1-170; for f in nat_be nat_cn nat_layer nat_selfA nat_selfH nat_selfA_be nat_selfH_be nat_corr mz_space.out mz_be mz_cn mz_layer; do echo "== $f"; column -s, -t < $T/$f* | cut -c1-230; done
rm -rf $T
