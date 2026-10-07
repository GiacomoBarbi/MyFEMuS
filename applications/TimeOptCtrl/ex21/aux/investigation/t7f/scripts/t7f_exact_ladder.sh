#!/bin/bash
# T7F exact-OC ladders (SCHUR_SWEEP with LU spatial solves = correctness reference).
# usage: t7f_exact_ladder.sh <tag> <case> <scheme> "<nx list>" "<nt list>" [extra args]
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
D=$(cd "$(dirname "$0")/.." && pwd)
tag=$1; case_=$2; s=$3; nxs=$4; nts=$5; shift 5
LU="-sweep_right_pc_type lu -sweep_left_pc_type lu -sweep_ay_pc_type lu -fieldsplit_y_sweep_pc_type lu"
W=${T7F_WORK:-/tmp/t7f_work_$tag}; mkdir -p $W/output $D/logs/exact; cd $W
for nx in $nxs; do for nt in $nts; do
  log=$D/logs/exact/${tag}_${s}_nx${nx}_nt${nt}.log
  PETSC_OPTIONS="${T7F_PETSC-$LU}" OMP_NUM_THREADS=1 /usr/bin/time -f "MAXRSS=%M wall=%e" $X --exact-oc $case_ \
    --nx $nx --nz $nx --steps $nt --time-scheme $s --pc-schur-sweep --no-paraview "$@" > $log 2>&1
  echo "$tag $s nx=$nx nt=$nt exit=$? $(grep -o 'MAXRSS=.*' $log)"
done; done
