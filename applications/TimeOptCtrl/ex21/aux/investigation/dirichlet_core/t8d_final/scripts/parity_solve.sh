#!/bin/bash
# Zero-velocity parity, solutions: ex20 vs ex21 (convection off) vs ex21
# (convection on, w = (0, 0)); identical CLI (ex20 defaults otherwise), nx = 8,
# N = 16; every solver available for the control type; BE and CN.
E=${EX20_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex20/TimeOptCtrl_ex20}
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
D=$(cd "$(dirname "$0")/.." && pwd)
W=${WORK:-/tmp/t8f_parity}; J=${JSON:-/tmp/t8f_parity_json}; mkdir -p $W/output $D/logs/parity $J; cd $W
LU="-sweep_right_pc_type lu -sweep_left_pc_type lu -sweep_ay_pc_type lu -fieldsplit_y_sweep_pc_type lu"
run() {   # tag binary-label petsc-options args...
  local tag=$1 bin=$2 popt=$3; shift 3
  local exe=$X extra=""
  [ $bin = ex20 ] && exe=$E
  [ $bin = ex21w0 ] && extra="--convection on --vx 0 --vy 0"
  PETSC_OPTIONS="$popt" OMP_NUM_THREADS=1 $exe "$@" $extra --nx 8 --nz 8 --steps 16 --no-paraview \
    --dump-json $J/${tag}_$bin.json > $D/logs/parity/${tag}_$bin.log 2>&1
  echo "${tag}_$bin exit=$?"; }
for s in be cn; do
  for bin in ex20 ex21off ex21w0; do
    for ctl in distributed neumann; do
      run ${ctl}_${s}_default   $bin ""    --$ctl-solve --time-scheme $s --alpha-u 1e-2
      run ${ctl}_${s}_sweep     $bin ""    --$ctl-solve --time-scheme $s --alpha-u 1e-2 --pc-schur-sweep
      run ${ctl}_${s}_sweeplu   $bin "$LU" --$ctl-solve --time-scheme $s --alpha-u 1e-2 --pc-schur-sweep
      run ${ctl}_${s}_matching  $bin ""    --$ctl-solve --time-scheme $s --alpha-u 1e-2 --pc-schur-matching
    done
    run neumann_${s}_boundaryreduced $bin "" --neumann-solve --time-scheme $s --alpha-u 1e-2 --pc-schur-boundary-reduced
    for reg in "l2:" "h1s:--control-h1-length 0.3" "h1t:--control-h1-time-weight 1e-3" "h1st:--control-h1-length 0.3 --control-h1-time-weight 1e-3"; do
      r=${reg%%:*}; ra=${reg#*:}
      run dirichlet${r}_${s}_default $bin "" --dirichlet-solve --time-scheme $s --alpha-u 1e-2 $ra
      [ $bin = ex20 ] && continue
      run dirichlet${r}_${s}_dense     $bin "" --dirichlet-solve --time-scheme $s --alpha-u 1e-2 $ra --dirichlet-core-solver dense
      run dirichlet${r}_${s}_iterative $bin "" --dirichlet-solve --time-scheme $s --alpha-u 1e-2 $ra --dirichlet-core-solver iterative
    done
  done
done
