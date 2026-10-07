#!/bin/bash
# T7F multiple source regions (production path, SCHUR_SWEEP + LU), natural
# state BC, w = (1, 0.5), control box [1/4,3/4]^2, tracking box [1/8,7/8]x[1/2,7/8].
# S1 = [0,1/4]x[3/4,1], S2 = [3/4,1]x[0,1/4]; f = --source amplitude on the union.
#   z0: --source 0 (an EMPTY region list means the WHOLE domain, so the source-
#       free baseline is a zero amplitude);  z1: S1;  z2: S2;  z12: S1 u S2;
#   z12x: S1 u S2 with amplitude 2.5;  ovAB: overlapping boxes A, B;  ovU: one box = A u B.
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
D=$(cd "$(dirname "$0")/.." && pwd)
LU="-sweep_right_pc_type lu -sweep_left_pc_type lu -sweep_ay_pc_type lu -fieldsplit_y_sweep_pc_type lu"
W=${T7F_WORK:-/tmp/t7f_ms}; J=${T7F_JSON:-/tmp/t7f_ms_json}; mkdir -p $W/output $D/logs/multisource $J; cd $W
base="--distributed-solve --nu 0.25 --alpha-u 1e-2 --target-type sin_x_minus_t --state-bc natural
      --convection on --vx 1 --vy 0.5 --T 1 --nx 16 --nz 16 --steps 32 --pc-schur-sweep
      --control-box 0.25 0.75 0.25 0.75 --tracking-box 0.125 0.875 0.5 0.875 --no-paraview"
S1="--source-box-add 0 0.25 0.75 1"; S2="--source-box-add 0.75 1 0 0.25"
run() { local tag=$1 np=$2; shift 2
  local cmd="$X $base --dump-json $J/$tag.json $*"
  if [ $np = 1 ]; then PETSC_OPTIONS="$LU" OMP_NUM_THREADS=1 $cmd > $D/logs/multisource/$tag.log 2>&1
  else PETSC_OPTIONS="$LU -sweep_right_pc_factor_mat_solver_type mumps -sweep_left_pc_factor_mat_solver_type mumps -sweep_ay_pc_factor_mat_solver_type mumps -fieldsplit_y_sweep_pc_factor_mat_solver_type mumps" \
       OMP_NUM_THREADS=1 mpirun -np $np $cmd > $D/logs/multisource/$tag.log 2>&1; fi
  echo "$tag exit=$?"; }
for s in be cn; do
  run z0_$s 1 --time-scheme $s --source 0
  run z0box_$s 1 --time-scheme $s --source 0 $S1 $S2
  run z1_$s 1 --time-scheme $s --source 1 $S1
  run z2_$s 1 --time-scheme $s --source 1 $S2
  run z12_$s 1 --time-scheme $s --source 1 $S1 $S2
  run z12x_$s 1 --time-scheme $s --source 2.5 $S1 $S2
  run zwhole_$s 1 --time-scheme $s --source 1
  run ovAB_$s 1 --time-scheme $s --source 1 --source-box-add 0.125 0.5 0.125 0.375 --source-box-add 0.375 0.875 0.125 0.375
  run ovU_$s 1 --time-scheme $s --source 1 --source-box-add 0.125 0.875 0.125 0.375
  for np in 2 4; do run z12_${s}_np$np $np --time-scheme $s --source 1 $S1 $S2; done
done
