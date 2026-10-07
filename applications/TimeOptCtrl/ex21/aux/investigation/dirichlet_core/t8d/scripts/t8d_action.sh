#!/bin/bash
# T8D gold matrix-action + SPD checks (--verify-dirichlet-core, dense mode):
# heat / convection x BE / CN x L2 / H1-space / H1-time / combined.
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
D=$(cd "$(dirname "$0")/.." && pwd)
W=${T8D_WORK:-/tmp/t8d_action}; mkdir -p $W/output $D/logs/action; cd $W
for size in "8 16" "16 32"; do set -- $size; nx=$1; nt=$2
for conv in "off" "on --vx 1 --vy 0.5"; do c=${conv%% *}
for s in be cn; do
for reg in "l2:" "h1s:--control-h1-length 0.3" "h1t:--control-h1-time-weight 1e-3" "h1st:--control-h1-length 0.3 --control-h1-time-weight 1e-3"; do
  r=${reg%%:*}; ra=${reg#*:}
  tag=conv${c}_${s}_${r}_nx${nx}_nt${nt}
  OMP_NUM_THREADS=1 $X --dirichlet-solve --nx $nx --nz $nx --steps $nt --nu 0.25 --alpha-u 1e-2 \
    --target-type sin_x_minus_t --convection $conv --time-scheme $s $ra --verify-dirichlet-core \
    --no-paraview > $D/logs/action/$tag.log 2>&1
  echo "$tag exit=$?"
done; done; done; done
