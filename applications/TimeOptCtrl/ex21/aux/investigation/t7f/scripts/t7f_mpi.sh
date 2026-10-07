#!/bin/bash
# T7F MPI consistency np = 1/2/4: natural-conv (exact data through the production
# path, --solve-only) and the multi-zone problem; SCHUR_SWEEP, LU (np=1) / MUMPS LU
# (np>1) spatial solves; -ksp_monitor gives the OUTER FGMRES history.
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
D=$(cd "$(dirname "$0")/.." && pwd)
W=${T7F_WORK:-/tmp/t7f_mpi}; J=${T7F_JSON:-/tmp/t7f_mpi_json}; mkdir -p $W/output $D/logs/mpi $J; cd $W
LU="-sweep_right_pc_type lu -sweep_left_pc_type lu -sweep_ay_pc_type lu -fieldsplit_y_sweep_pc_type lu"
MUMPS="-sweep_right_pc_factor_mat_solver_type mumps -sweep_left_pc_factor_mat_solver_type mumps -sweep_ay_pc_factor_mat_solver_type mumps -fieldsplit_y_sweep_pc_factor_mat_solver_type mumps"
MZ="--distributed-solve --nu 0.25 --alpha-u 1e-2 --target-type sin_x_minus_t --state-bc natural --convection on --vx 1 --vy 0.5 --T 1
    --control-box 0.125 0.375 0.125 0.375 --control-box-add 0.625 0.875 0.625 0.875
    --tracking-box 0.5 0.875 0.125 0.375 --tracking-box-add 0.125 0.5 0.5 0.875"
run() { local tag=$1 np=$2; shift 2
  local popt="$LU -ksp_monitor"; [ $np -gt 1 ] && popt="$LU $MUMPS -ksp_monitor"
  PETSC_OPTIONS="$popt" OMP_NUM_THREADS=1 mpirun -np $np $X "$@" --pc-schur-sweep --nx 16 --nz 16 --steps 32 \
     --no-paraview --dump-json $J/${tag}_np$np.json > $D/logs/mpi/${tag}_np$np.log 2>&1
  echo "${tag}_np$np exit=$?"; }
for s in be cn; do for np in 1 2 4; do
  run nat_$s $np --exact-oc natural-conv --solve-only --time-scheme $s
  run mz_$s $np $MZ --time-scheme $s
done; done
