#!/bin/bash
# T9 3D MPI consistency np = 1/2/4 (production path, conv-sin3-3d data, n = 6,
# N = 16): w = 0 and w = (1, 0.5, -0.75), BE and CN; SCHUR_SWEEP with LU (np=1)
# or MUMPS LU (np>1) spatial solves; -ksp_monitor = OUTER FGMRES history.
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
D=$(cd "$(dirname "$0")/.." && pwd)
W=${T9_WORK:-/tmp/t9_mpi}; J=${T9_JSON:-/tmp/t9_mpi_json}; mkdir -p $W/output $D/logs/mpi $J; cd $W
LU="-sweep_right_pc_type lu -sweep_left_pc_type lu -sweep_ay_pc_type lu -fieldsplit_y_sweep_pc_type lu"
MUMPS="-sweep_right_pc_factor_mat_solver_type mumps -sweep_left_pc_factor_mat_solver_type mumps -sweep_ay_pc_factor_mat_solver_type mumps -fieldsplit_y_sweep_pc_factor_mat_solver_type mumps"
for s in be cn; do for w in "w0:--convection on --vx 0 --vy 0 --vz 0" "wA:--convection on --vx 1 --vy 0.5 --vz -0.75"; do
  for np in 1 2 4; do
    tag=${w%%:*}_${s}_np$np; popt="$LU -ksp_monitor"; [ $np -gt 1 ] && popt="$LU $MUMPS -ksp_monitor"
    PETSC_OPTIONS="$popt" OMP_NUM_THREADS=1 mpirun -np $np /usr/bin/time -f "RANK_MAXRSS=%M wall=%e" $X \
      --exact-oc conv-sin3-3d --solve-only --nx 6 --ny 6 --nz 6 --steps 16 --time-scheme $s --pc-schur-sweep \
      --no-paraview ${w#*:} --dump-json $J/$tag.json > $D/logs/mpi/$tag.log 2>&1
    echo "$tag exit=$? $(grep -o 'RANK_MAXRSS=[0-9]*' $D/logs/mpi/$tag.log | tr '\n' ' ')"
  done
done; done
