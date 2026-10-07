#!/bin/bash
# T8D: the T7D gate battery (copied from boundary_convection/t7d/scripts/t7d_gates.sh) with the ITERATIVE core.
# T7D gate battery: --verify-convection --dirichlet-solve (G1, G3D, RD, G4D FD,
# G5D Hessian, G6D production KKT) for BE/CN x winds x regularisations.
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
D=$(cd "$(dirname "$0")/.." && pwd)
W=${T7D_WORK:-/tmp/claude-1000/t7d_gates}; mkdir -p $W/output $D/logs/gates; cd $W
run() {  # name, extra args
  local name=$1; shift
  for s in be cn; do
    log=$D/logs/gates/${name}_${s}.log
    OMP_NUM_THREADS=1 $X --verify-convection --dirichlet-solve --nx 8 --nz 8 --steps 8 \
      --time-scheme $s --nu 0.25 --alpha-u 1e-2 --target-type sin_x_minus_t --no-paraview --dirichlet-core-solver iterative "$@" > $log 2>&1
    echo "$name $s exit=$?"
  done
}
run heat_l2          --convection off
run w1_0.5_l2        --convection on --vx 1 --vy 0.5
run w0_0.5_l2        --convection on --vx 0 --vy 0.5
run wm2_1_l2         --convection on --vx -2 --vy 1
run w0_m4_l2         --convection on --vx 0 --vy -4
run w1_0.5_h1s       --convection on --vx 1 --vy 0.5 --control-h1-length 0.3
run w1_0.5_h1t       --convection on --vx 1 --vy 0.5 --control-h1-time-weight 1e-3
run w1_0.5_h1st      --convection on --vx 1 --vy 0.5 --control-h1-length 0.3 --control-h1-time-weight 1e-3
run heat_h1st        --convection off --control-h1-length 0.3 --control-h1-time-weight 1e-3
