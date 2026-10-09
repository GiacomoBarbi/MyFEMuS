#!/bin/bash
# T12 MPI consistency, np = 1/2/4 (production path of the exact cases, --dump-json, -ksp_monitor),
# SCHUR_SWEEP, LU (np = 1) / MUMPS LU (np > 1), outer rtol 1e-12.  Sanitised env.
#   A tri_cn   conv-sin3 (wind (1,0.5)), P2 TRI n = 8, N = 8, CN
#   B tet_be   conv-sin3-3d, wind (1,0.5,-0.75), P2 TET n = 4, N = 8, BE
#   C tet_cn   same, CN
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
D=$(cd "$(dirname "$0")/.." && pwd)
W=${T12_WORK:-/tmp/t12_mpi}; J=${T12_JSON:-/tmp/t12_mpi_json}; mkdir -p $W/output $D/logs/mpi $J; cd $W
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
LU="-sweep_right_pc_type lu -sweep_left_pc_type lu -sweep_ay_pc_type lu -fieldsplit_y_sweep_pc_type lu -ksp_rtol 1e-12 -ksp_monitor"
MUMPS="-sweep_right_pc_factor_mat_solver_type mumps -sweep_left_pc_factor_mat_solver_type mumps -sweep_ay_pc_factor_mat_solver_type mumps -fieldsplit_y_sweep_pc_factor_mat_solver_type mumps"
# MUMPS ICNTL(14) = 100 (extra workspace, %): without it the P2 TET factor at np = 4 stops with
# INFOG(1) = -9 ("work array too small", an allocation estimate -- the same operators converge with
# GAMG at np = 4, and HEX27 at np = 4 does not need it).  Standard MUMPS remedy, not a tuning.
MUMPS="$MUMPS -sweep_right_mat_mumps_icntl_14 100 -sweep_left_mat_mumps_icntl_14 100 -sweep_ay_mat_mumps_icntl_14 100 -fieldsplit_y_sweep_mat_mumps_icntl_14 100"
for c in "tri_cn:cn:--exact-oc conv-sin3 --mesh-type tri --nx 8 --nz 8" \
         "tet_be:be:--exact-oc conv-sin3-3d --mesh-type tet --nx 4 --ny 4 --nz 4 --convection on --vx 1 --vy 0.5 --vz -0.75" \
         "tet_cn:cn:--exact-oc conv-sin3-3d --mesh-type tet --nx 4 --ny 4 --nz 4 --convection on --vx 1 --vy 0.5 --vz -0.75"; do
  tag=${c%%:*}; rest=${c#*:}; s=${rest%%:*}; args=${rest#*:}
  for np in 1 2 4; do
    popt="$LU"; [ $np -gt 1 ] && popt="$LU $MUMPS"
    PETSC_OPTIONS="$popt" mpirun -np $np /usr/bin/time -f "RANK_MAXRSS=%M wall=%e" $X $args --solve-only --steps 8 --time-scheme $s \
      --pc-schur-sweep --no-paraview --dump-json $J/${tag}_np$np.json > $D/logs/mpi/${tag}_np$np.log 2>&1
    echo "${tag}_np$np exit=$? $(grep -oE 'nrank=\S+' $D/logs/mpi/${tag}_np$np.log | head -1)"
  done
done
