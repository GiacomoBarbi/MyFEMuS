#!/bin/bash
# T13 convergence tables:
#   ../../t13_space_convergence.csv  (a) exact errors (u in L2(Gamma_c) = sum over the controlled faces):
#        neumann-opp-3d (wind A, heat) and neumann-adj-3d (wind E, heat), CN N = 64, n = 2..10;
#        dirichlet-opp-3d (wind N, heat; L2, and H1s / H1t / comb with wind N), CN N = 256, n = 4, 8, 12;
#        (b) Dirichlet adjacent (no closed form): fixed-dt (CN N = 16) self-convergence vs n_ref = 12,
#        production data (wind A, sin_x_minus_t, tracking (0.5,1) x (0,1) x (0.5,1)), union zmin,xmin vs the
#        single faces zmin and xmin on the SAME data, L2 and combined H1; (c) dirichlet-opp-3d data, fixed-dt
#        (CN N = 16) self-convergence n = 4, 8, 12 vs n_ref = 24, L2 and combined H1.
#   ../../t13_time_convergence.csv   fixed-mesh self-convergence, BE / CN: neumann-opp-3d, neumann-adj-3d
#        (n = 6, ref 512), dirichlet-opp-3d L2 and comb (n = 4, ref 1024), Dirichlet adjacent production
#        L2 and comb with the single-face zmin control (n = 4, ref 1024).
D=$(cd "$(dirname "$0")" && pwd); R=$D/../..; J=${T13_JSON:?}; T=$(mktemp -d)
cd $D
python3 t13_ladder_table.py $T/space.csv nopA nopH nadW nadH dopN dopH dopC dopS dopT > $D/../logs/t13_space_ladders.log
{ for t in dsA:zmin,xmin dssA:zmin dsxA:xmin dsH:zmin,xmin dssH:zmin dsxH:xmin; do python3 t13_selfconv.py $J ${t%%:*} cn 16 "2 4 6" 12 ${t#*:}; done
  for t in dssN dssC; do python3 t13_selfconv.py $J $t cn 16 "4 8 12" 24 zmin,zmax; done; } > $T/self.csv
{ for t in notA:neumann:zmin,zmax:6:512 natW:neumann:zmin,xmin:6:512 dotN:dirichlet:zmin,zmax:4:1024 dotC:dirichlet:zmin,zmax:4:1024 \
           datA:dirichlet:zmin,xmin:4:1024 dasA:dirichlet:zmin:4:1024 datH:dirichlet:zmin,xmin:4:1024 dasH:dirichlet:zmin:4:1024; do
    IFS=: read tag kind faces n ref <<< "$t"
    for s in be cn; do python3 t13_time.py $J $tag $s $n "4 8 16 32 64 128" $ref $kind $faces; done; done; } > $T/time.csv
python3 - $T $R <<'PY'
import csv, sys
T, R = sys.argv[1], sys.argv[2]
sp = list(csv.DictReader(open(f"{T}/space.csv")))
STUDY = {"nopA": "exact neumann-opp-3d wind A", "nopH": "exact neumann-opp-3d heat", "nadW": "exact neumann-adj-3d wind E",
         "nadH": "exact neumann-adj-3d heat", "dopN": "exact dirichlet-opp-3d L2 wind N", "dopH": "exact dirichlet-opp-3d L2 heat",
         "dopC": "exact dirichlet-opp-3d H1 comb wind N", "dopS": "exact dirichlet-opp-3d H1-space wind N", "dopT": "exact dirichlet-opp-3d H1-time wind N"}
rows = []
for r in sp:
    rows.append({"study": STUDY[r["ladder"]], "tag": r["ladder"], "kind": "exact", "scheme": r["scheme"], "n": r["n"], "nt": r["nt"],
                 "ref": "exact", "y": r["y_L2L2"], "ord_y": r["ord_y"], "p": r["p_L2L2"], "ord_p": r["ord_p"], "u": r["u_L2L2"], "ord_u": r["ord_u"],
                 "maxrss_MB": r["maxrss_MB"], "wall_s": r["wall_s"], "stationarity_rel": r["stationarity_rel"], "march_rel": r["march_rel"]})
lines = [l for l in open(f"{T}/self.csv") if not l.startswith("tag,")]
SS = {"dsA": "self-conv Dirichlet adjacent zmin,xmin L2", "dssA": "self-conv single face zmin L2 (same data)",
      "dsxA": "self-conv single face xmin L2 (same data)", "dsH": "self-conv Dirichlet adjacent zmin,xmin H1 comb",
      "dssH": "self-conv single face zmin H1 comb", "dsxH": "self-conv single face xmin H1 comb",
      "dssN": "self-conv dirichlet-opp-3d data L2 wind N (ref n=24)", "dssC": "self-conv dirichlet-opp-3d data H1 comb wind N (ref n=24)"}
for l in lines:
    t, s, nt, n, ref, faces, y, oy, p, op, u, ou, J, dJ = l.strip().split(",")
    rows.append({"study": SS[t], "tag": t, "kind": "self-convergence", "scheme": s, "n": n, "nt": nt, "ref": f"n={ref}", "y": y, "ord_y": oy,
                 "p": p, "ord_p": op, "u": u, "ord_u": ou, "maxrss_MB": "", "wall_s": "", "stationarity_rel": "", "march_rel": ""})
with open(f"{R}/t13_space_convergence.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
tl = [l for l in open(f"{T}/time.csv")]
hdr = tl[0]
with open(f"{R}/t13_time_convergence.csv", "w") as fh:
    fh.write(hdr); fh.writelines(l for l in tl if not l.startswith("tag,"))
print("space rows", len(rows), "time rows", sum(1 for l in tl if not l.startswith("tag,")))
PY
