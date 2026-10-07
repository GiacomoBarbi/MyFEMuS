#!/bin/bash
# T9 3D exact-OC ladders (conv-sin-3d), SCHUR_SWEEP + LU spatial solves (serial).
# usage: t9_exact_ladder.sh <tag> <scheme> "<n list>" "<nt list>" [extra args, e.g. wind]
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
D=$(cd "$(dirname "$0")/.." && pwd)
tag=$1; s=$2; ns=$3; nts=$4; shift 4
LU="-sweep_right_pc_type lu -sweep_left_pc_type lu -sweep_ay_pc_type lu -fieldsplit_y_sweep_pc_type lu"
W=${T9_WORK:-/tmp/t9_ladder_$tag}; mkdir -p $W/output $D/logs/exact; cd $W
for n in $ns; do for nt in $nts; do
  log=$D/logs/exact/${tag}_${s}_n${n}_nt${nt}.log
  PETSC_OPTIONS="${T9_PETSC-$LU}" OMP_NUM_THREADS=1 /usr/bin/time -f "MAXRSS=%M wall=%e" $X --exact-oc conv-sin-3d \
    --nx $n --ny $n --nz $n --steps $nt --time-scheme $s --pc-schur-sweep --no-paraview "$@" > $log 2>&1
  echo "$tag $s n=$n nt=$nt exit=$? $(grep -o 'MAXRSS=.*' $log)"
done; done
