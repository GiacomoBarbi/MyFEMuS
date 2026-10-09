#!/bin/bash
# T14 convergence tables:
#   ../../t14_space_convergence.csv  (a) exact errors: conv-sin3-3d (distributed; constant wind A / harmonic-3d /
#        affine-div-3d) and natural-var-3d (harmonic / affine-div), CN N = 64 (natural also N = 128, n = 6..10:
#        the N = 64 time error ~3e-6 floors p, u at n = 10); neumann-var-3d (harmonic), neumann-var-opp-3d
#        (affine-div), CN N = 64; dirichlet-var-3d swirl, CN N = 256;  (b) Dirichlet production (zmin, sin_x data),
#        fixed-dt (CN N = 16) self-convergence: affine-div-3d vs the constant winds on the SAME data, ref n = 12
#        (L2 and combined H1) and ref n = 24 (L2).
#   ../../t14_time_convergence.csv   fixed-mesh (n = 4) self-convergence: conv-sin3-3d harmonic BE/CN and
#        affine-div CN (ref 1024), natural-var-3d affine-div BE/CN (512), neumann-var-3d harmonic CN (512),
#        dirichlet-var-3d BE/CN (1024).
D=$(cd "$(dirname "$0")" && pwd); R=$D/../..; J=${T14_JSON:?}; T=$(mktemp -d)
cd $D
python3 t14_ladder_table.py $T/space.csv dA dH dF nH nF nH128 nF128 mH oF rS > $D/../logs/t14_space_ladders.log
{ for t in scF scA scN scFH scAH; do python3 t14_selfconv.py $J $t cn 16 "2 4 6" 12 zmin; done
  for t in scF scA; do python3 t14_selfconv.py $J $t cn 16 "2 4 6 8 12" 24 zmin; done; } > $T/self.csv
{ for t in tdH:be:1024:distributed:- tdH:cn:1024:distributed:- tdF:cn:1024:distributed:- tnF:be:512:distributed:- tnF:cn:512:distributed:- \
           tmH:cn:512:neumann:zmin trS:be:1024:dirichlet:zmin trS:cn:1024:dirichlet:zmin; do
    IFS=: read tag s ref kind faces <<< "$t"
    python3 t14_time.py $J $tag $s 4 "4 8 16 32 64 128" $ref $kind $faces; done; } > $T/time.csv
python3 - $T $R <<'PY'
import csv, sys
T, R = sys.argv[1], sys.argv[2]
STUDY = {"dA": "exact conv-sin3-3d constant wind A (baseline)", "dH": "exact conv-sin3-3d harmonic-3d", "dF": "exact conv-sin3-3d affine-div-3d",
         "nH": "exact natural-var-3d harmonic-3d N=64", "nF": "exact natural-var-3d affine-div-3d N=64",
         "nH128": "exact natural-var-3d harmonic-3d N=128", "nF128": "exact natural-var-3d affine-div-3d N=128",
         "mH": "exact neumann-var-3d harmonic-3d", "oF": "exact neumann-var-opp-3d affine-div-3d", "rS": "exact dirichlet-var-3d swirl"}
rows = []
for r in csv.DictReader(open(f"{T}/space.csv")):
    rows.append({"study": STUDY[r["ladder"]], "tag": r["ladder"], "kind": "exact", "scheme": r["scheme"], "n": r["n"], "nt": r["nt"],
                 "ref": "exact", "y": r["y_L2L2"], "ord_y": r["ord_y"], "p": r["p_L2L2"], "ord_p": r["ord_p"], "u": r["u_L2L2"],
                 "ord_u": r["ord_u"], "maxrss_MB": r["maxrss_MB"], "wall_s": r["wall_s"], "stationarity_rel": r["stationarity_rel"],
                 "march_rel": r["march_rel"]})
SS = {"scF": "self-conv Dirichlet zmin affine-div-3d L2", "scA": "self-conv Dirichlet zmin constant wind A L2",
      "scN": "self-conv Dirichlet zmin constant normal wind L2", "scFH": "self-conv Dirichlet zmin affine-div-3d H1 comb",
      "scAH": "self-conv Dirichlet zmin constant wind A H1 comb"}
for l in open(f"{T}/self.csv"):
    if l.startswith("tag,"): continue
    t, s, nt, n, ref, faces, y, oy, p, op, u, ou, J, dJ = l.strip().split(",")
    rows.append({"study": SS[t], "tag": t, "kind": "self-convergence", "scheme": s, "n": n, "nt": nt, "ref": f"n={ref}", "y": y, "ord_y": oy,
                 "p": p, "ord_p": op, "u": u, "ord_u": ou, "maxrss_MB": "", "wall_s": "", "stationarity_rel": "", "march_rel": ""})
with open(f"{R}/t14_space_convergence.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
tl = list(open(f"{T}/time.csv"))
with open(f"{R}/t14_time_convergence.csv", "w") as fh:
    fh.write(tl[0]); fh.writelines(l for l in tl if not l.startswith("tag,"))
print("space rows", len(rows), "time rows", sum(1 for l in tl if not l.startswith("tag,")))
PY
