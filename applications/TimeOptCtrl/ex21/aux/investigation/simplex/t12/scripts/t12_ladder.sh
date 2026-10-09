#!/bin/bash
# T12 exact-OC ladders on the SAME PDE for two element families, SCHUR_SWEEP + LU (serial).
# usage: t12_ladder.sh <tag> <case> <mesh-type> <scheme> "<n list>" "<nt list>" [extra args]
#   2D cases: --nx n --nz n;  3D cases: --nx n --ny n --nz n.  Logs: logs/exact/<tag>_<scheme>_n<n>_nt<nt>.log
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
D=$(cd "$(dirname "$0")/.." && pwd)
tag=$1; c=$2; m=$3; s=$4; ns=$5; nts=$6; shift 6
LU="-sweep_right_pc_type lu -sweep_left_pc_type lu -sweep_ay_pc_type lu -fieldsplit_y_sweep_pc_type lu"
W=${T12_WORK:-/tmp/t12_ladder_$tag}; mkdir -p $W/output $D/logs/exact; cd $W
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for n in $ns; do for nt in $nts; do
  case $m in hex|tet) N="--nx $n --ny $n --nz $n";; *) N="--nx $n --nz $n";; esac
  log=$D/logs/exact/${tag}_${s}_n${n}_nt${nt}.log
  PETSC_OPTIONS="${T12_PETSC-$LU}" /usr/bin/time -f "MAXRSS=%M wall=%e" $X --exact-oc $c --mesh-type $m $N --steps $nt \
    --time-scheme $s --pc-schur-sweep --no-paraview "$@" > $log 2>&1
  echo "$tag $s n=$n nt=$nt exit=$? $(grep -o 'MAXRSS=.*' $log)"
done; done
