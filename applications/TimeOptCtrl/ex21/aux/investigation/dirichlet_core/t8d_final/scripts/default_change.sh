#!/bin/bash
# T8D finalisation: default-change regression.  For heat/convection x BE/CN x
# L2/H1-space/H1-time/combined (nx = 16, N = 32, production path):
#   new_default : new binary, no core flag (iterative + rt + 1e-8)
#   new_dense   : new binary, --dirichlet-core-solver dense
#   old_default : T8D binary (sha a2e6f680), no core flag (= dense)
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
OLD=${OLD:?set OLD}
D=$(cd "$(dirname "$0")/.." && pwd)
W=${WORK:-/tmp/t8f_default}; J=${JSON:-/tmp/t8f_default_json}; mkdir -p $W/output $D/logs/default $J; cd $W
for conv in "off" "on --vx 1 --vy 0.5"; do c=${conv%% *}
for s in be cn; do
for reg in "l2:" "h1s:--control-h1-length 0.3" "h1t:--control-h1-time-weight 1e-3" "h1st:--control-h1-length 0.3 --control-h1-time-weight 1e-3"; do
  r=${reg%%:*}; ra=${reg#*:}
  base="--dirichlet-solve --nx 16 --nz 16 --steps 32 --nu 0.25 --alpha-u 1e-2 --target-type sin_x_minus_t --convection $conv --time-scheme $s $ra --no-paraview"
  t=conv${c}_${s}_${r}
  OMP_NUM_THREADS=1 $X $base --dump-json $J/${t}_new_default.json > $D/logs/default/${t}_new_default.log 2>&1
  OMP_NUM_THREADS=1 $X $base --dirichlet-core-solver dense --dump-json $J/${t}_new_dense.json > $D/logs/default/${t}_new_dense.log 2>&1
  OMP_NUM_THREADS=1 $OLD $base --dump-json $J/${t}_old_default.json > $D/logs/default/${t}_old_default.log 2>&1
  echo "$t done"
done; done; done
