#!/bin/bash
# T7F gate battery: --verify-convection (G1 orientation, G1N boundary identity C+C^T,
# G2 exact transpose + ex20 negative control, G8b wrong natural-BC adjoint,
# G2 reduced-Hessian symmetry, G3 FD gradient, G4 production: true residual,
# |grad J(u*)|, independent forward march), SCHUR_SWEEP + LU spatial solves.
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
D=$(cd "$(dirname "$0")/.." && pwd)
W=${T7F_WORK:-/tmp/t7f_gates}; mkdir -p $W/output $D/logs/gates; cd $W
LU="-sweep_right_pc_type lu -sweep_left_pc_type lu -sweep_ay_pc_type lu -fieldsplit_y_sweep_pc_type lu"
# boxes on the 1/8 lattice: element-aligned for every nx = 8k
MZ="--control-box 0.125 0.375 0.125 0.375 --control-box-add 0.625 0.875 0.625 0.875
    --tracking-box 0.5 0.875 0.125 0.375 --tracking-box-add 0.125 0.5 0.5 0.875"
SRC="--source 1 --source-box-add 0 0.25 0.75 1 --source-box-add 0.75 1 0 0.25"
DATA="--distributed-solve --nu 0.25 --alpha-u 1e-2 --target-type sin_x_minus_t"
run() {  # name, args
  local name=$1; shift
  for s in be cn; do
    PETSC_OPTIONS="$LU" OMP_NUM_THREADS=1 $X --verify-convection --pc-schur-sweep --nx 8 --nz 8 --steps 8 \
      --time-scheme $s --no-paraview "$@" > $D/logs/gates/${name}_${s}.log 2>&1
    echo "$name $s exit=$?"
  done
}
run nat_exact_w1      --exact-oc natural-conv
run nat_exact_w2      --exact-oc natural-conv --exact-oc-velocity -1 -0.5
run nat_exact_w3      --exact-oc natural-conv --exact-oc-velocity 2 -1
run mz_natural        $DATA --state-bc natural --convection on --vx 1 --vy 0.5 $MZ
run mz_dirichlet      $DATA --state-bc dirichlet --convection on --vx 1 --vy 0.5 $MZ
run mz_overlap        $DATA --state-bc natural --convection on --vx 1 --vy 0.5 \
                      --control-box 0.125 0.5 0.125 0.375 --control-box-add 0.375 0.875 0.125 0.375 \
                      --tracking-box 0.125 0.875 0.5 0.875
run src2_natural      $DATA --state-bc natural --convection on --vx 1 --vy 0.5 $SRC \
                      --control-box 0.25 0.75 0.25 0.75 --tracking-box 0 1 0 1
run src2_mz_dirichlet $DATA --state-bc dirichlet --convection on --vx -1 --vy 0.5 $SRC $MZ
