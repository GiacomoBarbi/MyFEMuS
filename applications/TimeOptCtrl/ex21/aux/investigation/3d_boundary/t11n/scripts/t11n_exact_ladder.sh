#!/bin/bash
# (T11N copy of 3d_phase2/t10/scripts/t10_exact_ladder.sh)
# T10 3D exact-OC ladders (T9 t9_exact_ladder.sh with the case as a parameter),
# SCHUR_SWEEP + LU spatial solves (serial; T9_PETSC="" selects GMRES+GAMG).
# usage: t11n_exact_ladder.sh <tag> <case> <scheme> "<n list>" "<nt list>" [extra args, e.g. wind]
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
D=$(cd "$(dirname "$0")/.." && pwd)
tag=$1; c=$2; s=$3; ns=$4; nts=$5; shift 5
LU="-sweep_right_pc_type lu -sweep_left_pc_type lu -sweep_ay_pc_type lu -fieldsplit_y_sweep_pc_type lu"
W=${T11N_WORK:-/tmp/t11n_ladder_$tag}; mkdir -p $W/output $D/logs/exact; cd $W
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for n in $ns; do for nt in $nts; do
  log=$D/logs/exact/${tag}_${s}_n${n}_nt${nt}.log
  PETSC_OPTIONS="${T9_PETSC-$LU}" /usr/bin/time -f "MAXRSS=%M wall=%e" $X --exact-oc $c \
    --nx $n --ny $n --nz $n --steps $nt --time-scheme $s --pc-schur-sweep --no-paraview "$@" > $log 2>&1
  echo "$tag $s n=$n nt=$nt exit=$? $(grep -o 'MAXRSS=.*' $log)"
done; done
