#!/bin/bash
# T11H1 exact-OC ladders (dirichlet-h1-3d; ell, gamma_u as extra args) with the DEFAULT Dirichlet solver
# (SCHUR_DIRICHLET_REDUCED, iterative CG core).  Errors: y, p in L2(Omega), u in
# L2(Gamma_c) (nodal in time), from EX21_EXACT3D_RESULT.
# usage: t11d_exact_ladder.sh <tag> <scheme> "<n list>" "<nt list>" [extra args, e.g. wind]
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
D=$(cd "$(dirname "$0")/.." && pwd)
tag=$1; s=$2; ns=$3; nts=$4; shift 4
W=${T11H1_WORK:-/tmp/t11h1_ladder_$tag}; mkdir -p $W/output $D/logs/exact; cd $W
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for n in $ns; do for nt in $nts; do
  log=$D/logs/exact/${tag}_${s}_n${n}_nt${nt}.log
  /usr/bin/time -f "MAXRSS=%M wall=%e" $X --exact-oc dirichlet-h1-3d \
    --nx $n --ny $n --nz $n --steps $nt --time-scheme $s --no-paraview "$@" > $log 2>&1
  echo "$tag $s n=$n nt=$nt exit=$? $(grep -o 'MAXRSS=.*' $log)"
done; done
