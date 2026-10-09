#!/bin/bash
# T14 MPI consistency, np = 1/2/4 (production path, --dump-json, -ksp_monitor), sanitised env, n = 4, N = 8.
# SWEEP kinds (LU np = 1 / MUMPS LU np > 1, outer rtol 1e-10): Vdh distributed div-free (conv-sin3-3d harmonic-3d, BE);
#   Vdf distributed GENERIC (conv-sin3-3d affine-div-3d, CN); Vnf natural (natural-var-3d affine-div-3d, CN);
#   Nmh Neumann single face (neumann-var-3d harmonic-3d, BE); Nof Neumann opposite union (neumann-var-opp-3d affine-div-3d, CN).
# Dirichlet reduced (core rtol 1e-12, outer rtol 1e-14): Drs dirichlet-var-3d swirl CN; Daf production zmin,xmin affine-div-3d CN
#   (multi-face); Dh1 production zmin,zmax harmonic-3d, combined H1, CN.
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
D=$(cd "$(dirname "$0")/.." && pwd)
W=${T14_WORK:-/tmp/t14_mpi}; J=${T14_JSON:-/tmp/t14_mpi_json}; mkdir -p $W/output $D/logs/mpi $J; cd $W
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
LU="-sweep_right_pc_type lu -sweep_left_pc_type lu -sweep_ay_pc_type lu -fieldsplit_y_sweep_pc_type lu -ksp_rtol 1e-10 -ksp_monitor"
MUMPS="-sweep_right_pc_factor_mat_solver_type mumps -sweep_left_pc_factor_mat_solver_type mumps -sweep_ay_pc_factor_mat_solver_type mumps -fieldsplit_y_sweep_pc_factor_mat_solver_type mumps"
N4="--nx 4 --ny 4 --nz 4 --steps 8"
PD="--dimension 3 --dirichlet-solve --state-bc dirichlet --nu 0.25 --alpha-u 1e-2 --target-type sin_x_minus_t --T 1 $N4 --tracking-box3d 0.2 0.8 0.2 0.8 0.2 0.8"
for c in "Vdh:be:--exact-oc conv-sin3-3d --solve-only $N4 --convection on --wind harmonic-3d" \
         "Vdf:cn:--exact-oc conv-sin3-3d --solve-only $N4 --convection on --wind affine-div-3d" \
         "Vnf:cn:--exact-oc natural-var-3d --solve-only $N4 --convection on --wind affine-div-3d" \
         "Nmh:be:--exact-oc neumann-var-3d --solve-only $N4 --convection on --wind harmonic-3d" \
         "Nof:cn:--exact-oc neumann-var-opp-3d --solve-only $N4 --convection on --wind affine-div-3d" \
         "Drs:cn:--exact-oc dirichlet-var-3d --solve-only $N4 --convection on --vz -0.75" \
         "Daf:cn:$PD --dirichlet-face3d zmin,xmin --convection on --wind affine-div-3d" \
         "Dh1:cn:$PD --dirichlet-face3d zmin,zmax --convection on --wind harmonic-3d --control-h1-length 0.3 --control-h1-time-weight 1e-2"; do
  tag=${c%%:*}; rest=${c#*:}; s=${rest%%:*}; args=${rest#*:}
  for np in 1 2 4; do
    case $tag in
      D*) popt="-ksp_monitor -ksp_rtol 1e-14"; extra="--dirichlet-core-rtol 1e-12";;
      *)  popt="$LU"; [ $np -gt 1 ] && popt="$LU $MUMPS"; extra="--pc-schur-sweep";;
    esac
    PETSC_OPTIONS="$popt" mpirun -np $np /usr/bin/time -f "RANK_MAXRSS=%M wall=%e" $X $args --time-scheme $s \
      $extra --no-paraview --dump-json $J/${tag}_np$np.json > $D/logs/mpi/${tag}_np$np.log 2>&1
    echo "${tag}_np$np exit=$? $(grep -oE 'nrank=\S+' $D/logs/mpi/${tag}_np$np.log | head -1)"
  done
done
