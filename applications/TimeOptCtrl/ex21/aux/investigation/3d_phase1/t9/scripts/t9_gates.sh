#!/bin/bash
# T9 3D gate battery: --verify-convection on the conv-sin-3d data (3D, distributed
# full domain, homogeneous Dirichlet).  SCHUR_SWEEP + LU spatial solves.  The
# outer FGMRES is tightened to 1e-12 (-ksp_rtol: the outer KSP has no prefix):
# at the production rtol 1e-8 a 3D n=2 solve stops after ONE outer iteration and
# leaves the G4 consistency at ~1e-7, which is tolerance, not algebra.
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
D=$(cd "$(dirname "$0")/.." && pwd)
W=${T9_WORK:-/tmp/t9_gates}; mkdir -p $W/output $D/logs/gates; cd $W
LU="-sweep_right_pc_type lu -sweep_left_pc_type lu -sweep_ay_pc_type lu -fieldsplit_y_sweep_pc_type lu -ksp_rtol 1e-12"
for n in 2 3; do for s in be cn; do
  for w in "heat:--convection off" "w0:--convection on --vx 0 --vy 0 --vz 0" \
           "wA:--convection on --vx 1 --vy 0.5 --vz -0.75" "wB:--convection on --vx -2 --vy 1 --vz 0.5"; do
    tag=${w%%:*}_${s}_n$n; a=${w#*:}
    PETSC_OPTIONS="$LU" OMP_NUM_THREADS=1 $X --verify-convection --exact-oc conv-sin-3d --nx $n --ny $n --nz $n \
      --steps 4 --time-scheme $s --pc-schur-sweep --no-paraview $a > $D/logs/gates/$tag.log 2>&1
    echo "$tag exit=$?"
  done
done; done
