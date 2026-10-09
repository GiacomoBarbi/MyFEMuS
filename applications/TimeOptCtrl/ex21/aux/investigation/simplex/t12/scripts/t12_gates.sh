#!/bin/bash
# T12 gate battery (--verify-convection, SCHUR_SWEEP + LU, outer rtol 1e-12) on QUAD9 vs P2 TRI and
# HEX27 vs P2 TET, BE / CN, winds heat (off), w0 (on, w = 0), wA, wB:
#   ex   exact cases: conv-sin3 (2D, n = 4; wind by --exact-oc-velocity, convection always on) and
#        conv-sin3-3d (3D, n = 2; wind by --convection/--vx..)
#   pr   production full-domain distributed control, target sin_x_minus_t: 2D n = 4 on the unit square;
#        3D n = (3,2,2) on the non-unit box (0,2) x (-1,1) x (0,0.5)
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
D=$(cd "$(dirname "$0")/.." && pwd)
W=${T12_WORK:-/tmp/t12_gates}; mkdir -p $W/output $D/logs/gates; cd $W
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
LU="-sweep_right_pc_type lu -sweep_left_pc_type lu -sweep_ay_pc_type lu -fieldsplit_y_sweep_pc_type lu -ksp_rtol 1e-12"
run() { tag=$1; shift; PETSC_OPTIONS="$LU" $X --verify-convection --pc-schur-sweep --no-paraview "$@" > $D/logs/gates/$tag.log 2>&1; echo "$tag exit=$?"; }
for s in be cn; do
  for m in quad tri; do
    # the 2D convective exact case always has convection on; its wind is --exact-oc-velocity (no heat
    # variant there -- the production rows cover 2D heat)
    for w in "w0:--exact-oc-velocity 0 0" "wA:--exact-oc-velocity 1 0.5" "wB:--exact-oc-velocity -2 1"; do
      run ex_${m}_${w%%:*}_$s --exact-oc conv-sin3 --mesh-type $m --nx 4 --nz 4 --steps 4 --time-scheme $s ${w#*:}
    done
    for w in "heat:--convection off" "w0:--convection on --vx 0 --vy 0" "wA:--convection on --vx 1 --vy 0.5" "wB:--convection on --vx -2 --vy 1"; do
      run pr_${m}_${w%%:*}_$s --mesh-type $m --domain 0 1 0 1 --distributed-solve --control-full-domain --state-bc dirichlet \
          --tracking-box 0 1 0 1 --target-type sin_x_minus_t --nu 0.25 --alpha-u 1e-2 --nx 4 --nz 4 --steps 4 --T 1 --time-scheme $s ${w#*:}
    done
  done
  for m in hex tet; do
    for w in "heat:--convection off" "w0:--convection on --vx 0 --vy 0 --vz 0" "wA:--convection on --vx 1 --vy 0.5 --vz -0.75" "wB:--convection on --vx -2 --vy 1 --vz 0.5"; do
      run ex_${m}_${w%%:*}_$s --exact-oc conv-sin3-3d --mesh-type $m --nx 2 --ny 2 --nz 2 --steps 4 --time-scheme $s ${w#*:}
      run pr_${m}_${w%%:*}_$s --dimension 3 --mesh-type $m --domain3d 0 2 -1 1 0 0.5 --distributed-solve --control-full-domain --state-bc dirichlet \
          --tracking-box3d 0 2 -1 1 0 0.5 --target-type sin_x_minus_t --nu 0.25 --alpha-u 1e-2 --nx 3 --ny 2 --nz 2 --steps 4 --T 1 --time-scheme $s ${w#*:}
    done
  done
done
