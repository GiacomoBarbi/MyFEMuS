#!/bin/bash
# T12 solver policy: SCHUR_SWEEP with the LU spatial backend vs the default GMRES/CG + GAMG backend
# (no PETSC_OPTIONS), same exact PDEs on all four element families:
#   2D conv-sin3, n = 16, N = 64 (Pe_cell = |w| h/(2 nu) = 0.14)   quad / tri
#   3D conv-sin3-3d wind (1,0.5,-0.75), n = 6, N = 64 (Pe_cell = 0.45)   hex / tet
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
D=$(cd "$(dirname "$0")/.." && pwd)
W=${T12_WORK:-/tmp/t12_solver}; mkdir -p $W/output $D/logs/solver; cd $W
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
LU="-sweep_right_pc_type lu -sweep_left_pc_type lu -sweep_ay_pc_type lu -fieldsplit_y_sweep_pc_type lu"
for s in be cn; do for b in lu gamg; do opt=""; [ $b = lu ] && opt="$LU"
  for m in quad tri; do
    PETSC_OPTIONS="$opt" /usr/bin/time -f "MAXRSS=%M wall=%e" $X --exact-oc conv-sin3 --mesh-type $m --nx 16 --nz 16 --steps 64 \
      --time-scheme $s --pc-schur-sweep --no-paraview > $D/logs/solver/${m}_${b}_$s.log 2>&1; echo "${m}_${b}_$s exit=$?"
  done
  for m in hex tet; do
    PETSC_OPTIONS="$opt" /usr/bin/time -f "MAXRSS=%M wall=%e" $X --exact-oc conv-sin3-3d --mesh-type $m --nx 6 --ny 6 --nz 6 --steps 64 \
      --convection on --vx 1 --vy 0.5 --vz -0.75 --time-scheme $s --pc-schur-sweep --no-paraview > $D/logs/solver/${m}_${b}_$s.log 2>&1; echo "${m}_${b}_$s exit=$?"
  done
done; done
