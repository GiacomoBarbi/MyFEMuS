#!/bin/bash
# T12 time table -> ../../t12_time.csv: fixed-mesh self-convergence (t12_time.py) for
#   tT  P2 TRI n = 16 (conv-sin3)        qT  QUAD9 n = 16 (same PDE)
#   tK  P2 TET n = 4  (conv-sin3-3d, wind (1,0.5,-0.75))   hK  HEX27 n = 4 (same PDE)
# BE ref N = 1024, CN ref N = 512; FE L2 norm with the mesh's own M_full.
D=$(cd "$(dirname "$0")" && pwd); J=${T12_JSON:-/tmp/t12_json}; out=$D/../../t12_time.csv; T=$(mktemp -d)
cd $D
for t in "tT 16 tri" "qT 16 quad" "tK 4 tet" "hK 4 hex"; do set -- $t
  for s in be cn; do r=1024; [ $s = cn ] && r=512
    /usr/bin/python3.11 t12_time.py $J $1 $s $2 "4 8 16 32 64 128 256" $r | sed "1!s/^/$3,/; 1s/^/mesh,/" > $T/$1_$s.csv
  done
done
head -1 $T/tT_be.csv > $out; for f in $T/*.csv; do tail -n +2 $f >> $out; done
rm -rf $T; wc -l $out
