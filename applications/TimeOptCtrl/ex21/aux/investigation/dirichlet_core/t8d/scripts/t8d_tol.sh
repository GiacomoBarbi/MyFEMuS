#!/bin/bash
# T8D inner-core tolerance ladder vs the dense reference (production data,
# convection w = (1, 0.5), CN, nx = 32, N = 64, T = 1): L2 and combined H1,
# alpha_u = 1e-2 and 1e-4.
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
D=$(cd "$(dirname "$0")/.." && pwd)
W=${T8D_WORK:-/tmp/t8d_tol}; J=${T8D_JSON:-/tmp/t8d_tol_json}; mkdir -p $W/output $D/logs/tol $J; cd $W
for reg in "l2:" "h1st:--control-h1-length 0.3 --control-h1-time-weight 1e-3"; do r=${reg%%:*}; ra=${reg#*:}
for a in 1e-2 1e-4; do
  base="--dirichlet-solve --nu 0.25 --target-type sin_x_minus_t --T 1 --convection on --vx 1 --vy 0.5 --nx 32 --nz 32 --steps 64 --time-scheme cn --alpha-u $a $ra --no-paraview"
  tag=${r}_a${a}_dense
  OMP_NUM_THREADS=1 /usr/bin/time -f "MAXRSS=%M wall=%e" $X $base --dirichlet-core-solver dense --dump-json $J/$tag.json > $D/logs/tol/$tag.log 2>&1; echo "$tag exit=$?"
  for tol in 1e-2 1e-4 1e-6 1e-8 1e-10 1e-12; do
    tag=${r}_a${a}_tol$tol
    OMP_NUM_THREADS=1 /usr/bin/time -f "MAXRSS=%M wall=%e" $X $base --dirichlet-core-solver iterative --dirichlet-core-rtol $tol \
      --dump-json $J/$tag.json > $D/logs/tol/$tag.log 2>&1; echo "$tag exit=$?"
  done
done; done
