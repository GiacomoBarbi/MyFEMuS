#!/bin/bash
# T10 MPI consistency, np = 1/2/4 (production path, --dump-json, -ksp_monitor =
# outer FGMRES history).  SCHUR_SWEEP, LU (np = 1) / MUMPS LU (np > 1), outer
# rtol 1e-10.  Run in the sanitised environment (PETSc-arch mpirun).
#   A mz_be     octant multi-zone + 2 sources, natural BC, wind A, BE, n=4, N=8
#   B src_cn    full-domain control/tracking, 2 separated sources, Dirichlet BC,
#               wind B = (-2, 1, 0.5), CN, n=4, N=8
#   C nat_be/nat_cn  natural-conv-3d exact data (--solve-only), wind A, n=4, N=8
#   D nu_be     non-unit domain (0,2)x(-1,1)x(0,0.5), (4,4,2), boxes + source, natural, BE
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
D=$(cd "$(dirname "$0")/.." && pwd)
W=${T10_WORK:-/tmp/t10_mpi}; J=${T10_JSON:-/tmp/t10_mpi_json}; mkdir -p $W/output $D/logs/mpi $J; cd $W
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
LU="-sweep_right_pc_type lu -sweep_left_pc_type lu -sweep_ay_pc_type lu -fieldsplit_y_sweep_pc_type lu -ksp_rtol 1e-10 -ksp_monitor"
MUMPS="-sweep_right_pc_factor_mat_solver_type mumps -sweep_left_pc_factor_mat_solver_type mumps -sweep_ay_pc_factor_mat_solver_type mumps -fieldsplit_y_sweep_pc_factor_mat_solver_type mumps"
COMMON="--distributed-solve --dimension 3 --nu 0.25 --alpha-u 1e-2 --target-type sin_x_minus_t"
MZ="$COMMON --state-bc natural --convection on --vx 1 --vy 0.5 --vz -0.75 --nx 4 --ny 4 --nz 4 --steps 8
    --control-box3d 0 0.5 0 0.5 0 0.5 --control-box3d-add 0.5 1 0.5 1 0.5 1
    --tracking-box3d 0 0.5 0.5 1 0 0.5 --tracking-box3d-add 0.5 1 0 0.5 0.5 1
    --source 1 --source-box3d-add 0 0.5 0.5 1 0.5 1 --source-box3d-add 0.5 1 0 0.5 0 0.5"
SRC="$COMMON --state-bc dirichlet --convection on --vx -2 --vy 1 --vz 0.5 --nx 4 --ny 4 --nz 4 --steps 8
    --control-full-domain --tracking-box3d 0 1 0 1 0 1
    --source 1 --source-box3d-add 0 0.25 0 0.25 0 0.25 --source-box3d-add 0.75 1 0.75 1 0.75 1"
NAT="--exact-oc natural-conv-3d --solve-only --convection on --vx 1 --vy 0.5 --vz -0.75 --nx 4 --ny 4 --nz 4 --steps 8"
NU="$COMMON --state-bc natural --convection on --vx 1 --vy 0.5 --vz -0.75 --domain3d 0 2 -1 1 0 0.5 --nx 4 --ny 4 --nz 2 --steps 8
    --tracking-box3d 1 2 -1 1 0 0.5 --control-box3d 0 1 -1 0 0 0.5 --source 1 --source-box3d-add 0 0.5 0 1 0 0.25"
for c in "mz_be:be:$MZ" "src_cn:cn:$SRC" "nat_be:be:$NAT" "nat_cn:cn:$NAT" "nu_be:be:$NU"; do
  tag=${c%%:*}; rest=${c#*:}; s=${rest%%:*}; args=${rest#*:}
  for np in 1 2 4; do
    popt="$LU"; [ $np -gt 1 ] && popt="$LU $MUMPS"
    PETSC_OPTIONS="$popt" mpirun -np $np /usr/bin/time -f "RANK_MAXRSS=%M wall=%e" $X $args --time-scheme $s \
      --pc-schur-sweep --no-paraview --dump-json $J/${tag}_np$np.json > $D/logs/mpi/${tag}_np$np.log 2>&1
    echo "${tag}_np$np exit=$? $(grep -oE 'nrank=\S+' $D/logs/mpi/${tag}_np$np.log | head -1)"
  done
done
