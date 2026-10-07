#!/bin/bash
# T7F solver suitability (verify only, no tuning): each feature with
#   ref  = SCHUR_SWEEP + LU spatial solves (correctness reference)
#   sw   = SCHUR_SWEEP with its default iterative spatial backend (GMRES + GAMG)
#   add  = ADDITIVE_GAMG (the distributed default, no --pc flag)
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
D=$(cd "$(dirname "$0")/.." && pwd)
W=${T7F_WORK:-/tmp/t7f_solver}; J=${T7F_JSON:-/tmp/t7f_solver_json}; mkdir -p $W/output $D/logs/solver $J; cd $W
LU="-sweep_right_pc_type lu -sweep_left_pc_type lu -sweep_ay_pc_type lu -fieldsplit_y_sweep_pc_type lu"
MZ="--distributed-solve --nu 0.25 --alpha-u 1e-2 --target-type sin_x_minus_t --state-bc natural --convection on --vx 1 --vy 0.5 --T 1
    --control-box 0.125 0.375 0.125 0.375 --control-box-add 0.625 0.875 0.625 0.875
    --tracking-box 0.5 0.875 0.125 0.375 --tracking-box-add 0.125 0.5 0.5 0.875 --nx 32 --nz 32 --steps 32"
SRC="--distributed-solve --nu 0.25 --alpha-u 1e-2 --target-type sin_x_minus_t --state-bc natural --convection on --vx 1 --vy 0.5 --T 1
     --control-box 0.25 0.75 0.25 0.75 --tracking-box 0.125 0.875 0.5 0.875 --source 1
     --source-box-add 0 0.25 0.75 1 --source-box-add 0.75 1 0 0.25 --nx 32 --nz 32 --steps 32"
NAT="--exact-oc natural-conv --solve-only --nx 32 --nz 32 --steps 64"
NATPE="--exact-oc natural-conv --solve-only --exact-oc-nu 0.02 --nx 32 --nz 32 --steps 64"
run() { local tag=$1 popt=$2; shift 2
  PETSC_OPTIONS="$popt" OMP_NUM_THREADS=1 /usr/bin/time -f "MAXRSS=%M wall=%e" $X "$@" --no-paraview \
     --dump-json $J/$tag.json > $D/logs/solver/$tag.log 2>&1; echo "$tag exit=$?"; }
# NATPE + ADDITIVE_GAMG: the BE run was stopped by hand after 37 min wall
# without convergence (logs/solver/NATPE_be_add.log); it is not repeated for CN.
for s in ${T7F_SCHEMES:-be cn}; do
  for f in MZ SRC NAT NATPE; do
    args=${!f}
    run ${f}_${s}_ref "$LU" $args --time-scheme $s --pc-schur-sweep
    run ${f}_${s}_sw "" $args --time-scheme $s --pc-schur-sweep
    [ $f = NATPE ] && continue
    run ${f}_${s}_add "" $args --time-scheme $s
  done
done
