#!/bin/bash
# T10 multiple-source production study (natural BC, wind A, non-colocated
# multi-zone control/tracking as in the gate battery), BE and CN, n = 4.
#  (a) production solves with --dump-json (N = 16, outer rtol 1e-12) for the
#      affine-map identities: S0 (zero source), SA, SB, SAB, SAB2 (amplitude 2),
#      SOV (two overlapping boxes) vs SUN (their explicit union box), SWH (no
#      box = whole domain) vs SFB (one full-domain box);
#  (b) --verify-convection (N = 4) of SA, SB, SAB: true KKT residual, gradient
#      consistency and the independent forward march with each source layout.
D=$(cd "$(dirname "$0")" && pwd)
BASE="--distributed-solve --dimension 3 --nu 0.25 --alpha-u 1e-2 --target-type sin_x_minus_t --state-bc natural
      --convection on --vx 1 --vy 0.5 --vz -0.75
      --tracking-box3d 0 0.5 0.5 1 0 0.5 --tracking-box3d-add 0.5 1 0 0.5 0.5 1
      --control-box3d 0 0.5 0 0.5 0 0.5 --control-box3d-add 0.5 1 0.5 1 0.5 1"
A="--source-box3d-add 0 0.5 0.5 1 0.5 1"          # octant (x<1/2, y>1/2, z>1/2)
B="--source-box3d-add 0.5 1 0 0.5 0 0.5"          # octant (x>1/2, y<1/2, z<1/2)
OV="--source-box3d-add 0 0.5 0.5 1 0.5 1 --source-box3d-add 0.25 0.75 0.5 1 0.5 1"
UN="--source-box3d-add 0 0.75 0.5 1 0.5 1"
FB="--source-box3d-add 0 1 0 1 0 1"
for s in be cn; do
  bash $D/t10_runs.sh S0 $s 4 16 $BASE --source 0 $A $B
  bash $D/t10_runs.sh SA $s 4 16 $BASE --source 1 $A
  bash $D/t10_runs.sh SB $s 4 16 $BASE --source 1 $B
  bash $D/t10_runs.sh SAB $s 4 16 $BASE --source 1 $A $B
  bash $D/t10_runs.sh SAB2 $s 4 16 $BASE --source 2 $A $B
  bash $D/t10_runs.sh SOV $s 4 16 $BASE --source 1 $OV
  bash $D/t10_runs.sh SUN $s 4 16 $BASE --source 1 $UN
  bash $D/t10_runs.sh SWH $s 4 16 $BASE --source 1
  bash $D/t10_runs.sh SFB $s 4 16 $BASE --source 1 $FB
done
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
L=$(cd "$D/.." && pwd)/logs/sources_verify; mkdir -p $L; W=${T10_WORK:-/tmp/t10_srcv}; mkdir -p $W/output; cd $W
LU="-sweep_right_pc_type lu -sweep_left_pc_type lu -sweep_ay_pc_type lu -fieldsplit_y_sweep_pc_type lu -ksp_rtol 1e-12"
for s in be cn; do for v in "SA:$A" "SB:$B" "SAB:$A $B"; do
  PETSC_OPTIONS="$LU" OMP_NUM_THREADS=1 $X --verify-convection --pc-schur-sweep --no-paraview $BASE --nx 4 --ny 4 --nz 4 \
    --steps 4 --time-scheme $s --source 1 ${v#*:} > $L/${v%%:*}_$s.log 2>&1; echo "verify ${v%%:*}_$s exit=$?"
done; done
