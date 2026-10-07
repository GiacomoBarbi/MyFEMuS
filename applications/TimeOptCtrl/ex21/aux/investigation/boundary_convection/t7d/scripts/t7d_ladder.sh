#!/bin/bash
# T7D L2 Dirichlet + convection convergence ladders (exact OC dirichlet-conv).
# usage: t7d_ladder.sh <tag> <scheme> "<nx list>" "<nt list>" [extra args]
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
D=$(cd "$(dirname "$0")/.." && pwd)
tag=$1; scheme=$2; nxs=$3; nts=$4; shift 4
W=${T7D_WORK:-/tmp/t7d_work_$tag}; mkdir -p $W/output $D/logs
cd $W
for nx in $nxs; do for nt in $nts; do
  log=$D/logs/${tag}_${scheme}_nx${nx}_nt${nt}.log
  /usr/bin/time -f "MAXRSS=%M wall=%e" $X --exact-oc dirichlet-conv --nx $nx --nz $nx \
     --steps $nt --time-scheme $scheme --no-paraview "$@" > $log 2>&1
  echo "$tag $scheme nx=$nx nt=$nt exit=$? $(grep -o 'MAXRSS=.*' $log)"
done; done
