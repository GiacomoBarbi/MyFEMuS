#!/bin/bash
# T15 exact-OC run (conv-sin3-3d family): SCHUR_SWEEP + LU.  One run per call (the fixed-Pe ladder changes nu
# with n).  usage: t15_exact.sh <tag> <scheme> <n> <nt> [args: --exact-oc ... --exact-oc-nu ... wind, lps]
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
D=$(cd "$(dirname "$0")/.." && pwd)
tag=$1; s=$2; n=$3; nt=$4; shift 4
LU="-sweep_right_pc_type lu -sweep_left_pc_type lu -sweep_ay_pc_type lu -fieldsplit_y_sweep_pc_type lu"
W=${T15_WORK:-/tmp/t15_exact_$tag}; mkdir -p $W/output $D/logs/exact; cd $W
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
log=$D/logs/exact/${tag}_${s}_n${n}_nt${nt}.log
PETSC_OPTIONS="$LU" /usr/bin/time -f "MAXRSS=%M wall=%e" $X --nx $n --ny $n --nz $n --steps $nt --time-scheme $s \
  --pc-schur-sweep --no-paraview "$@" > $log 2>&1
echo "$tag $s n=$n nt=$nt exit=$? $(grep -o 'MAXRSS=.*' $log)"
