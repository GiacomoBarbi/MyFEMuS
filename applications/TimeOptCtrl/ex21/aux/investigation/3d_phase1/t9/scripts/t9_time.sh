#!/bin/bash
# T9 temporal self-convergence on a FIXED 3D mesh (spatial error cancels):
# production path, conv-sin3-3d data, SCHUR_SWEEP + LU.  usage: t9_time.sh <scheme> <n> "<nt list>" <wind tag>
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
D=$(cd "$(dirname "$0")/.." && pwd)
s=$1; n=$2; nts=$3; wt=$4
case $wt in w0) wind="--convection off";; wA) wind="--convection on --vx 1 --vy 0.5 --vz -0.75";; esac
LU="-sweep_right_pc_type lu -sweep_left_pc_type lu -sweep_ay_pc_type lu -fieldsplit_y_sweep_pc_type lu -ksp_rtol 1e-12"
W=${T9_WORK:-/tmp/t9_time}; J=${T9_JSON:-/tmp/t9_time_json}; mkdir -p $W/output $D/logs/time $J; cd $W
for nt in $nts; do
  tag=${wt}_${s}_n${n}_nt${nt}
  PETSC_OPTIONS="$LU" OMP_NUM_THREADS=1 $X --exact-oc conv-sin3-3d --solve-only --nx $n --ny $n --nz $n --steps $nt \
    --time-scheme $s --pc-schur-sweep --no-paraview $wind --dump-json $J/$tag.json > $D/logs/time/$tag.log 2>&1
  echo "$tag exit=$?"
done
