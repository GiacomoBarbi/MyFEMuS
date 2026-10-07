#!/bin/bash
# T8D solution equivalence: dense core vs iterative core (CG + R_H1t, rtol 1e-10),
# heat/convection x BE/CN x L2/H1-space/H1-time/combined, nx = 16, N = 32 (production path).
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
D=$(cd "$(dirname "$0")/.." && pwd)
W=${T8D_WORK:-/tmp/t8d_equiv}; J=${T8D_JSON:-/tmp/t8d_equiv_json}; mkdir -p $W/output $D/logs/equiv $J; cd $W
for conv in "off" "on --vx 1 --vy 0.5"; do c=${conv%% *}
for s in be cn; do
for reg in "l2:" "h1s:--control-h1-length 0.3" "h1t:--control-h1-time-weight 1e-3" "h1st:--control-h1-length 0.3 --control-h1-time-weight 1e-3"; do
  r=${reg%%:*}; ra=${reg#*:}
  for mode in dense iterative; do
    tag=conv${c}_${s}_${r}_$mode
    OMP_NUM_THREADS=1 /usr/bin/time -f "MAXRSS=%M wall=%e" $X --dirichlet-solve --nx 16 --nz 16 --steps 32 --nu 0.25 --alpha-u 1e-2 \
      --target-type sin_x_minus_t --convection $conv --time-scheme $s $ra --dirichlet-core-solver $mode \
      --no-paraview --dump-json $J/$tag.json > $D/logs/equiv/$tag.log 2>&1
    echo "$tag exit=$?"
  done
done; done; done
