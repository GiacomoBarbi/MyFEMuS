#!/bin/bash
# T7D self-convergence (no closed form for gamma_u > 0): production solves on
# nested grids, fields kept as JSON in $T7D_JSON/<tag>.json (outside the repo).
# usage: t7d_selfconv.sh <case> <scheme> "<nx list>" "<nt list>"
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
D=$(cd "$(dirname "$0")/.." && pwd)
case_=$1; s=$2; nxs=$3; nts=$4
case $case_ in
  l2)   reg="" ;;
  h1t)  reg="--control-h1-time-weight 1e-3" ;;
  h1s)  reg="--control-h1-length 0.3" ;;
  h1st) reg="--control-h1-length 0.3 --control-h1-time-weight 1e-3" ;;
  exact|exacth1s) reg="" ;;
esac
# default data: target sin_x_minus_t, wind (1, 0.5);  exact*: the dirichlet-conv
# closed-form data through the production path (--solve-only), wind (0, 0.5)
data="--dirichlet-solve --nu 0.25 --alpha-u 1e-2 --target-type sin_x_minus_t --convection on --vx 1 --vy 0.5 --T 1"
[ $case_ = exact ] && data="--exact-oc dirichlet-conv --solve-only"
[ $case_ = exacth1s ] && data="--exact-oc dirichlet-conv --solve-only --control-h1-length 0.3"
W=${T7D_WORK:-/tmp/claude-1000/t7d_sc_$case_$s}; J=${T7D_JSON:-/tmp/t7d_selfconv_json}   # JSON fields kept outside the repository (~1.2 GB)
mkdir -p $W/output $D/logs/selfconv $J; cd $W
for nx in $nxs; do for nt in $nts; do
  tag=${case_}_${s}_nx${nx}_nt${nt}
  OMP_NUM_THREADS=1 /usr/bin/time -f "MAXRSS=%M wall=%e" $X $data --steps $nt --nx $nx --nz $nx \
    --time-scheme $s $reg --no-paraview --dump-json $J/$tag.json > $D/logs/selfconv/$tag.log 2>&1
  echo "$tag exit=$? $(grep -o 'MAXRSS.*' $D/logs/selfconv/$tag.log)"
done; done
