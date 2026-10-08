#!/bin/bash
# T11D gate battery (--verify-convection, reduced Dirichlet solver = the default
# SCHUR_DIRICHLET_REDUCED with the iterative CG core):
#   dir   dirichlet-conv-3d exact OC (Gamma_c = zmin), n = 2, 3, BE/CN, winds
#         heat (off), w0, wA = (1,0.5,-0.75), wB = (-2,1,0.5), wN = (0,0,-0.75) (normal),
#         wC = (1,0.5,0) (tangential on Gamma_c)
#   pxm   production, face xmax, wind A, n = 3       pym  face ymin, wind B, n = 3
#   pzx   production, face zmax, heat, n = 3
#   pnu   production, non-unit (0,2)x(-1,1)x(0,0.5), face ymax, wind A, (4,4,2)
# Production tracking boxes stay away from Gamma_c (the KKT tracks free DOFs only).
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
D=$(cd "$(dirname "$0")/.." && pwd)
W=${T11D_WORK:-/tmp/t11d_gates}; mkdir -p $W/output $D/logs/gates; cd $W
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
run() { tag=$1; shift; $X --verify-convection --no-paraview "$@" > $D/logs/gates/$tag.log 2>&1; echo "$tag exit=$?"; }
for n in 2 3; do for s in be cn; do
  for w in "heat:--convection off" "w0:--convection on --vx 0 --vy 0 --vz 0" "wA:--convection on --vx 1 --vy 0.5 --vz -0.75" \
           "wB:--convection on --vx -2 --vy 1 --vz 0.5" "wN:--convection on --vx 0 --vy 0 --vz -0.75" \
           "wC:--convection on --vx 1 --vy 0.5 --vz 0"; do
    run dir_${w%%:*}_${s}_n$n --exact-oc dirichlet-conv-3d --nx $n --ny $n --nz $n --steps 4 --time-scheme $s ${w#*:}
  done
done; done
P="--dimension 3 --dirichlet-solve --state-bc dirichlet --nu 0.25 --alpha-u 1e-2 --target-type sin_x_minus_t --steps 4 --T 1"
for s in be cn; do
  run pxm_wA_${s}_n3 $P --dirichlet-face3d xmax --tracking-box3d 0 0.6 0.2 0.8 0.2 0.8 --nx 3 --ny 3 --nz 3 --time-scheme $s --convection on --vx 1 --vy 0.5 --vz -0.75
  run pym_wB_${s}_n3 $P --dirichlet-face3d ymin --tracking-box3d 0.2 0.8 0.4 1 0.2 0.8 --nx 3 --ny 3 --nz 3 --time-scheme $s --convection on --vx -2 --vy 1 --vz 0.5
  run pzx_heat_${s}_n3 $P --dirichlet-face3d zmax --tracking-box3d 0.2 0.8 0.2 0.8 0 0.6 --nx 3 --ny 3 --nz 3 --time-scheme $s --convection off
  run pnu_wA_${s}_n4 $P --dirichlet-face3d ymax --domain3d 0 2 -1 1 0 0.5 --tracking-box3d 0.5 1.5 -1 0 0 0.5 --nx 4 --ny 4 --nz 2 --time-scheme $s --convection on --vx 1 --vy 0.5 --vz -0.75
done
