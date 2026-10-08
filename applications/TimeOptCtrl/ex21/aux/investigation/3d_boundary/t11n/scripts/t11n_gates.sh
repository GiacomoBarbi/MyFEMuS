#!/bin/bash
# T11N gate battery (--verify-convection, SCHUR_SWEEP + LU, outer rtol 1e-12):
#   neu   neumann-conv-3d exact OC (Gamma_c = zmin), n = 2, 3, BE/CN, winds
#         heat (off), w0, wA = (1,0.5,-0.75), wB = (-2,1,0.5), wC = (1,0.5,0) (w.n = 0 on Gamma_c)
#   pxm   production, face xmax, wind A, n = 3        pym  face ymin, wind B, n = 3
#   pnu   production, non-unit (0,2)x(-1,1)x(0,0.5), face ymax, tracking y > 0 (adjacent
#         to the face), T = 1, wind A, (4,4,2)
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
D=$(cd "$(dirname "$0")/.." && pwd)
W=${T11N_WORK:-/tmp/t11n_gates}; mkdir -p $W/output $D/logs/gates; cd $W
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
LU="-sweep_right_pc_type lu -sweep_left_pc_type lu -sweep_ay_pc_type lu -fieldsplit_y_sweep_pc_type lu -ksp_rtol 1e-12"
run() { tag=$1; shift; PETSC_OPTIONS="$LU" $X --verify-convection --pc-schur-sweep --no-paraview "$@" > $D/logs/gates/$tag.log 2>&1; echo "$tag exit=$?"; }
for n in 2 3; do for s in be cn; do
  for w in "heat:--convection off" "w0:--convection on --vx 0 --vy 0 --vz 0" "wA:--convection on --vx 1 --vy 0.5 --vz -0.75" \
           "wB:--convection on --vx -2 --vy 1 --vz 0.5" "wC:--convection on --vx 1 --vy 0.5 --vz 0"; do
    run neu_${w%%:*}_${s}_n$n --exact-oc neumann-conv-3d --nx $n --ny $n --nz $n --steps 4 --time-scheme $s ${w#*:}
  done
done; done
P="--dimension 3 --neumann-solve --state-bc natural --nu 0.25 --alpha-u 1e-2 --target-type sin_x_minus_t --steps 4"
for s in be cn; do
  run pxm_wA_${s}_n3 $P --neumann-face3d xmax --tracking-box3d 0 1 0 1 0 1 --nx 3 --ny 3 --nz 3 --time-scheme $s --convection on --vx 1 --vy 0.5 --vz -0.75
  run pym_wB_${s}_n3 $P --neumann-face3d ymin --tracking-box3d 0 1 0 1 0 1 --nx 3 --ny 3 --nz 3 --time-scheme $s --convection on --vx -2 --vy 1 --vz 0.5
  run pnu_wA_${s}_n4 $P --T 1 --neumann-face3d ymax --domain3d 0 2 -1 1 0 0.5 --tracking-box3d 0 2 0 1 0 0.5 --nx 4 --ny 4 --nz 2 --time-scheme $s --convection on --vx 1 --vy 0.5 --vz -0.75
done
