#!/bin/bash
# T10 3D gate battery (--verify-convection, SCHUR_SWEEP + LU spatial solves, outer
# FGMRES rtol 1e-12 as in T9: the production 1e-8 stops after one outer iteration
# and the G7 consistency then measures the tolerance, not the algebra).
#   nat  natural-conv-3d exact OC (natural BC), n = 2, 3, BE/CN,
#        winds heat (off), w0, wA = (1, 0.5, -0.75), wB = (-2, 1, 0.5)
#   mz   non-colocated multi-zone (2 control + 2 tracking boxes, zero element
#        overlap) + two separated source boxes, natural BC, n = 4, winds A/B, BE/CN
#   mzd  the same with the homogeneous Dirichlet state BC (wind A)
#   nu   non-unit domain (0,2)x(-1,1)x(0,0.5), (4,4,2), boxes + source, natural, wind A
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
D=$(cd "$(dirname "$0")/.." && pwd)
W=${T10_WORK:-/tmp/t10_gates}; mkdir -p $W/output $D/logs/gates; cd $W
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
LU="-sweep_right_pc_type lu -sweep_left_pc_type lu -sweep_ay_pc_type lu -fieldsplit_y_sweep_pc_type lu -ksp_rtol 1e-12"
run() { tag=$1; shift; PETSC_OPTIONS="$LU" $X --verify-convection --pc-schur-sweep --no-paraview "$@" > $D/logs/gates/$tag.log 2>&1; echo "$tag exit=$?"; }
for n in 2 3; do for s in be cn; do
  for w in "heat:--convection off" "w0:--convection on --vx 0 --vy 0 --vz 0" \
           "wA:--convection on --vx 1 --vy 0.5 --vz -0.75" "wB:--convection on --vx -2 --vy 1 --vz 0.5"; do
    run nat_${w%%:*}_${s}_n$n --exact-oc natural-conv-3d --nx $n --ny $n --nz $n --steps 4 --time-scheme $s ${w#*:}
  done
done; done
MZ="--distributed-solve --dimension 3 --nx 4 --ny 4 --nz 4 --steps 4 --nu 0.25 --alpha-u 1e-2 --target-type sin_x_minus_t
    --tracking-box3d 0 0.5 0.5 1 0 0.5 --tracking-box3d-add 0.5 1 0 0.5 0.5 1
    --control-box3d 0 0.5 0 0.5 0 0.5 --control-box3d-add 0.5 1 0.5 1 0.5 1
    --source 1 --source-box3d-add 0 0.25 0.75 1 0.75 1 --source-box3d-add 0.75 1 0 0.25 0 0.25"
for s in be cn; do
  run mz_wA_${s}_n4 $MZ --state-bc natural --time-scheme $s --convection on --vx 1 --vy 0.5 --vz -0.75
  run mz_wB_${s}_n4 $MZ --state-bc natural --time-scheme $s --convection on --vx -2 --vy 1 --vz 0.5
  run mzd_wA_${s}_n4 $MZ --state-bc dirichlet --time-scheme $s --convection on --vx 1 --vy 0.5 --vz -0.75
  run nu_wA_${s}_n4 --distributed-solve --dimension 3 --domain3d 0 2 -1 1 0 0.5 --nx 4 --ny 4 --nz 2 --steps 4 \
      --nu 0.25 --alpha-u 1e-2 --target-type sin_x_minus_t --state-bc natural --time-scheme $s \
      --tracking-box3d 1 2 -1 1 0 0.5 --control-box3d 0 1 -1 0 0 0.5 --source 1 --source-box3d-add 0 0.5 0 1 0 0.25 \
      --convection on --vx 1 --vy 0.5 --vz -0.75
done
