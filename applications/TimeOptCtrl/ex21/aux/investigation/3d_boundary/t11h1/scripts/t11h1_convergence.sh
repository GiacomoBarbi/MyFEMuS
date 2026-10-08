#!/bin/bash
# T11H1 convergence tables:
#   ../../t11h1_space_convergence.csv  dirichlet-h1-3d exact errors, CN N = 256: sN = H1-space ell = 0.3 with the
#        normal wind (0,0,-0.75), sH = ell = 0.3 heat, cN = combined ell = 0.3 + gamma = 1e-2 normal wind;
#        tH = H1-time gamma = 1e-2 heat (N = 64, n <= 14), kS = combined normal wind (N = 64, n <= 14) -- with
#        gamma > 0 the u error has NO time floor at N = 64 (n = 6 N-scan) and a pre-asymptotic plateau at n = 6..8;
#        y, p in L2(Omega), u in L2(Gamma_c) and the surface H1 seminorm |u - u_h|_{H1(Gamma_c)} (nodal in time)
#   ../../t11h1_time_convergence.csv   fixed mesh n = 6, normal wind: tN = H1-time gamma = 1e-2, kN = combined;
#        BE ref N = 1024, CN ref N = 512; y, y on t >= T/4, p, u, u_t, J
D=$(cd "$(dirname "$0")" && pwd); R=$D/../..; J=${T11H1_JSON:-/tmp/t11h1_json}; T=$(mktemp -d)
cd $D
/usr/bin/python3.11 t11h1_ladder_table.py $T/space.csv sN sH cN tH kS > /dev/null
for t in tN kN; do
  /usr/bin/python3.11 t11h1_time.py $J $t be 6 "4 8 16 32 64 128 256" 1024 > $T/${t}_be.csv
  /usr/bin/python3.11 t11h1_time.py $J $t cn 6 "4 8 16 32 64 128" 512 > $T/${t}_cn.csv
done
/usr/bin/python3.11 - $T $R <<'PY'
import csv, sys
T, R = sys.argv[1], sys.argv[2]
rd = lambda p: list(csv.DictReader(open(p)))
def write(path, blocks):
    keys = list(dict.fromkeys(k for _, rows in blocks for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["study"] + keys); w.writeheader()
        for st, rows in blocks:
            for r in rows: w.writerow(dict(r, study=st))
write(f"{R}/t11h1_space_convergence.csv", [("dirichlet-h1-3d exact errors CN N=256 (u: surface L2 and H1 seminorm)", rd(f"{T}/space.csv"))])
write(f"{R}/t11h1_time_convergence.csv",
      [(f"{lab} {s.upper()} n=6 ref N={1024 if s == 'be' else 512}, wind (0,0,-0.75)", rd(f"{T}/{t}_{s}.csv"))
       for t, lab in (("tN", "H1-time gamma=1e-2"), ("kN", "combined ell=0.3 gamma=1e-2")) for s in ("be", "cn")])
PY
rm -rf $T
