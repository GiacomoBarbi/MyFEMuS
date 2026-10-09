#!/bin/bash
# T14 exact-OC ladders (variable-wind and constant-wind baseline cases): SCHUR_SWEEP + LU for
# distributed / natural / Neumann; the default reduced Dirichlet solver for dirichlet-var-3d.
# Errors: y, p in L2(Omega); u in L2(Omega) (distributed / natural) or L2(Gamma_c) (boundary control).
# usage: t14_exact_ladder.sh <tag> <case> <scheme> "<n list>" "<nt list>" [extra args, e.g. wind]
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
D=$(cd "$(dirname "$0")/.." && pwd)
tag=$1; c=$2; s=$3; ns=$4; nts=$5; shift 5
LU="-sweep_right_pc_type lu -sweep_left_pc_type lu -sweep_ay_pc_type lu -fieldsplit_y_sweep_pc_type lu"
W=${T14_WORK:-/tmp/t14_ladder_$tag}; mkdir -p $W/output $D/logs/exact; cd $W
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
case $c in dirichlet-*) PC=""; PO="";; *) PC="--pc-schur-sweep"; PO="$LU";; esac
for n in $ns; do for nt in $nts; do
  log=$D/logs/exact/${tag}_${s}_n${n}_nt${nt}.log
  PETSC_OPTIONS="$PO" /usr/bin/time -f "MAXRSS=%M wall=%e" $X --exact-oc $c \
    --nx $n --ny $n --nz $n --steps $nt --time-scheme $s $PC --no-paraview "$@" > $log 2>&1
  echo "$tag $s n=$n nt=$nt exit=$? $(grep -o 'MAXRSS=.*' $log)"
done; done
