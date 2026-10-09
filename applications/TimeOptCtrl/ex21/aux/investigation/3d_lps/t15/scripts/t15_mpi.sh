#!/bin/bash
# T15 MPI consistency with LPS, np = 1/2/4 (production path, --dump-json, -ksp_monitor), sanitised env, n = 4, N = 8.
# SWEEP (LU np = 1 / MUMPS LU np > 1, outer rtol 1e-10):
#   A  distributed, constant wind A, LPS-P0, BE      (conv-sin3-3d data, nu = 1e-2: every patch active)
#   B  distributed, harmonic-3d, LPS-P0, CN           (conv-sin3-3d data, nu = 1e-2)
#   C  natural state, affine-div-3d, LPS-P0, CN       (production, nu = 1e-2)
#   D  Neumann union zmin+zmax, harmonic-3d, LPS-P0, CN (production, nu = 1e-2)
# Dirichlet reduced (core rtol 1e-12, outer rtol 1e-14):
#   E  Dirichlet zmin+xmin, affine-div-3d, LPS-P0, CN (production, nu = 1e-2)
#   F  Dirichlet zmin+zmax, harmonic-3d, LPS-P0, combined H1, CN
#   G  (P1 diagnostic) distributed, constant wind A, LPS-P1, CN
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
D=$(cd "$(dirname "$0")/.." && pwd)
W=${T15_WORK:-/tmp/t15_mpi}; J=${T15_JSON:-/tmp/t15_mpi_json}; mkdir -p $W/output $D/logs/mpi $J; cd $W
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
LU="-sweep_right_pc_type lu -sweep_left_pc_type lu -sweep_ay_pc_type lu -fieldsplit_y_sweep_pc_type lu -ksp_rtol 1e-10 -ksp_monitor"
MUMPS="-sweep_right_pc_factor_mat_solver_type mumps -sweep_left_pc_factor_mat_solver_type mumps -sweep_ay_pc_factor_mat_solver_type mumps -fieldsplit_y_sweep_pc_factor_mat_solver_type mumps"
N4="--nx 4 --ny 4 --nz 4 --steps 8"; L="--convection-stabilization lps"
A="--convection on --vx 1 --vy 0.5 --vz -0.75"
E="--exact-oc conv-sin3-3d --exact-oc-nu 1e-2 --solve-only $N4"
PR="--dimension 3 --nu 1e-2 --alpha-u 1e-2 --target-type sin_x_minus_t --T 1 $N4"
PD="$PR --dirichlet-solve --state-bc dirichlet --tracking-box3d 0.2 0.8 0.2 0.8 0.2 0.8"
for c in "A:be:$E $A $L" \
         "B:cn:$E --convection on --wind harmonic-3d $L" \
         "C:cn:$PR --distributed-solve --control-full-domain --tracking-box3d 0 1 0 1 0 1 --state-bc natural --convection on --wind affine-div-3d $L" \
         "D:cn:$PR --neumann-solve --state-bc natural --tracking-box3d 0.2 0.8 0.2 0.8 0.2 0.8 --neumann-face3d zmin,zmax --convection on --wind harmonic-3d $L" \
         "E:cn:$PD --dirichlet-face3d zmin,xmin --convection on --wind affine-div-3d $L" \
         "F:cn:$PD --dirichlet-face3d zmin,zmax --convection on --wind harmonic-3d --control-h1-length 0.3 --control-h1-time-weight 1e-2 $L" \
         "G:cn:$E $A $L --lps-projection p1"; do
  tag=${c%%:*}; rest=${c#*:}; s=${rest%%:*}; args=${rest#*:}
  for np in 1 2 4; do
    case $tag in
      E|F) popt="-ksp_monitor -ksp_rtol 1e-14"; extra="--dirichlet-core-rtol 1e-12";;
      *)  popt="$LU"; [ $np -gt 1 ] && popt="$LU $MUMPS"; extra="--pc-schur-sweep";;
    esac
    PETSC_OPTIONS="$popt" mpirun -np $np /usr/bin/time -f "RANK_MAXRSS=%M wall=%e" $X $args --time-scheme $s \
      $extra --no-paraview --dump-json $J/${tag}_np$np.json > $D/logs/mpi/${tag}_np$np.log 2>&1
    echo "${tag}_np$np exit=$? $(grep -oE 'nrank=\S+' $D/logs/mpi/${tag}_np$np.log | head -1)"
  done
done
echo T15_MPI_DONE
