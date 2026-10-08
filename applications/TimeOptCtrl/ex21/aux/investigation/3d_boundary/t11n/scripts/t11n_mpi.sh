#!/bin/bash
# T11N MPI consistency, np = 1/2/4 (production path, --dump-json, -ksp_monitor = outer
# FGMRES history), SCHUR_SWEEP, LU (np = 1) / MUMPS LU (np > 1), outer rtol 1e-10.
# Sanitised environment (PETSc-arch mpirun).
#   neu_be / neu_cn   neumann-conv-3d data (Gamma_c = zmin), wind A, n = 4, N = 8
#   heat_cn           neumann-conv-3d data, convection off, CN
#   pxm_be            production, face xmax, wind B = (-2, 1, 0.5), tracking full, BE
#   pnu_cn            production, non-unit (0,2)x(-1,1)x(0,0.5), face ymax, tracking y > 0, T = 1, CN
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
D=$(cd "$(dirname "$0")/.." && pwd)
W=${T11N_WORK:-/tmp/t11n_mpi}; J=${T11N_JSON:-/tmp/t11n_mpi_json}; mkdir -p $W/output $D/logs/mpi $J; cd $W
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
LU="-sweep_right_pc_type lu -sweep_left_pc_type lu -sweep_ay_pc_type lu -fieldsplit_y_sweep_pc_type lu -ksp_rtol 1e-10 -ksp_monitor"
MUMPS="-sweep_right_pc_factor_mat_solver_type mumps -sweep_left_pc_factor_mat_solver_type mumps -sweep_ay_pc_factor_mat_solver_type mumps -fieldsplit_y_sweep_pc_factor_mat_solver_type mumps"
EX="--exact-oc neumann-conv-3d --solve-only --nx 4 --ny 4 --nz 4 --steps 8"
PR="--dimension 3 --neumann-solve --state-bc natural --nu 0.25 --alpha-u 1e-2 --target-type sin_x_minus_t --steps 8"
for c in "neu_be:be:$EX --convection on --vx 1 --vy 0.5 --vz -0.75" "neu_cn:cn:$EX --convection on --vx 1 --vy 0.5 --vz -0.75" \
         "heat_cn:cn:$EX --convection off" \
         "pxm_be:be:$PR --neumann-face3d xmax --tracking-box3d 0 1 0 1 0 1 --nx 4 --ny 4 --nz 4 --convection on --vx -2 --vy 1 --vz 0.5" \
         "pnu_cn:cn:$PR --T 1 --neumann-face3d ymax --domain3d 0 2 -1 1 0 0.5 --tracking-box3d 0 2 0 1 0 0.5 --nx 4 --ny 4 --nz 2 --convection on --vx 1 --vy 0.5 --vz -0.75"; do
  tag=${c%%:*}; rest=${c#*:}; s=${rest%%:*}; args=${rest#*:}
  for np in 1 2 4; do
    popt="$LU"; [ $np -gt 1 ] && popt="$LU $MUMPS"
    PETSC_OPTIONS="$popt" mpirun -np $np /usr/bin/time -f "RANK_MAXRSS=%M wall=%e" $X $args --time-scheme $s \
      --pc-schur-sweep --no-paraview --dump-json $J/${tag}_np$np.json > $D/logs/mpi/${tag}_np$np.log 2>&1
    echo "${tag}_np$np exit=$? $(grep -oE 'nrank=\S+' $D/logs/mpi/${tag}_np$np.log | head -1)"
  done
done
