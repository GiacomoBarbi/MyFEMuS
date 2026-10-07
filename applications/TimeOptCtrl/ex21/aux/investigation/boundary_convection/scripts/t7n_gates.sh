#!/bin/bash
# T7N algebraic gates: --verify-convection on the manufactured Neumann case
# (neumann-conv, natural state BC, Gamma_c = {z = 0}), BE and CN, three winds:
#   A w = (1, 0.5)   w.n = -0.5 on Gamma_c (inflow), +0.5 on z = 1
#   B w = (1, -0.5)  w.n = +0.5 on Gamma_c (outflow)
#   C w = (1, 0)     tangential on Gamma_c (w.n = 0 on z faces, +-1 on x faces)
# LU sweeps.  WORK=<scratch> t7n_gates.sh
set -u
H=$(cd "$(dirname "$0")/.." && pwd)
X=${EX21_EXE:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
W=${WORK:?set WORK}; mkdir -p $W/output $H/logs/gates; cd $W
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PETSC_OPTIONS="-sweep_right_pc_type lu -sweep_left_pc_type lu -sweep_ay_pc_type lu -fieldsplit_y_sweep_pc_type lu"
for wind in "A 1 0.5" "B 1 -0.5" "C 1 0"; do
  set -- $wind; tag=$1; vx=$2; vy=$3
  for s in be cn; do
    name=vc_neumann_${tag}_${s}
    { echo "# $(basename $X) --verify-convection --exact-oc neumann-conv --exact-oc-velocity $vx $vy --nx 8 --nz 8 --steps 16 --time-scheme $s --distributed-solve --pc-schur-sweep"
      echo "# PETSC_OPTIONS=$PETSC_OPTIONS"
      $X --verify-convection --exact-oc neumann-conv --exact-oc-velocity $vx $vy --nx 8 --nz 8 --steps 16 \
         --time-scheme $s --distributed-solve --pc-schur-sweep --no-paraview 2>&1 | grep -E "^EX21_|rror"; } > $H/logs/gates/$name.log
    echo "$name: $(grep -c ^EX21_ $H/logs/gates/$name.log) lines"
  done
done
