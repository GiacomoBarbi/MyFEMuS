#!/bin/bash
# T8D ex20 heat relevance: identical CLI (ex20 defaults: heat, target constant),
# ex20 binary (its dense core) vs ex21 dense vs ex21 iterative; 4 regularisations x BE/CN.
E=${EX20_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex20/TimeOptCtrl_ex20}
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
D=$(cd "$(dirname "$0")/.." && pwd)
W=${T8D_WORK:-/tmp/t8d_ex20}; J=${T8D_JSON:-/tmp/t8d_ex20_json}; mkdir -p $W/output $D/logs/ex20 $J; cd $W
for s in be cn; do
for reg in "l2:" "h1s:--control-h1-length 0.3" "h1t:--control-h1-time-weight 1e-3" "h1st:--control-h1-length 0.3 --control-h1-time-weight 1e-3"; do
  r=${reg%%:*}; ra=${reg#*:}
  base="--dirichlet-solve --nx 16 --nz 16 --steps 32 --alpha-u 1e-2 --time-scheme $s $ra --no-paraview"
  OMP_NUM_THREADS=1 $E $base --dump-json $J/ex20_${s}_$r.json > $D/logs/ex20/ex20_${s}_$r.log 2>&1
  OMP_NUM_THREADS=1 $X $base --dirichlet-core-solver dense --dump-json $J/ex21dense_${s}_$r.json > $D/logs/ex20/ex21dense_${s}_$r.log 2>&1
  OMP_NUM_THREADS=1 $X $base --dirichlet-core-solver iterative --dump-json $J/ex21iter_${s}_$r.json > $D/logs/ex20/ex21iter_${s}_$r.log 2>&1
  echo "$s $r done"
done; done
