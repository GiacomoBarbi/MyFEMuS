#!/bin/bash
# T13 MPI consistency, np = 1/2/4 (production path, --dump-json, -ksp_monitor = outer history).  Sanitised env.
# NEUMANN (T11N policy): SCHUR_SWEEP, LU (np = 1) / MUMPS LU (np > 1), outer rtol 1e-10
#   Nopp_be   neumann-opp-3d data (zmin u zmax), wind A, n = 4, N = 8, BE          [required: opposite]
#   Nadj_cn   neumann-adj-3d data (zmin u xmin), wind E = (0.6,-0.4,0.6), CN      [required: adjacent]
#   Npadj_be  production zmin,xmin, wind B, BE;   Npthr_cn production xmin,ymin,zmin, wind A, CN
# DIRICHLET (T11D policy): default reduced solver, iterative CG core rtol 1e-12, outer rtol 1e-14
#   Dadj_cn   production zmin,xmin, wind A, CN                                    [required: adjacent CN]
#   Dopp_be   dirichlet-opp-3d data (zmin u zmax), wind N = (0,0,-0.75), BE        [required: opposite BE]
#   Dthr_cn   production zmin,ymin,xmin, wind B, CN;  DadjH1_cn production zmin,xmin, ell = 0.3, gamma_u = 1e-2, CN
#   Dall_be   production all six faces, wind A, BE
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
D=$(cd "$(dirname "$0")/.." && pwd)
W=${T13_WORK:-/tmp/t13_mpi}; J=${T13_JSON:-/tmp/t13_mpi_json}; mkdir -p $W/output $D/logs/mpi $J; cd $W
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
LU="-sweep_right_pc_type lu -sweep_left_pc_type lu -sweep_ay_pc_type lu -fieldsplit_y_sweep_pc_type lu -ksp_rtol 1e-10 -ksp_monitor"
MUMPS="-sweep_right_pc_factor_mat_solver_type mumps -sweep_left_pc_factor_mat_solver_type mumps -sweep_ay_pc_factor_mat_solver_type mumps -fieldsplit_y_sweep_pc_factor_mat_solver_type mumps"
wA="--convection on --vx 1 --vy 0.5 --vz -0.75"; wB="--convection on --vx -2 --vy 1 --vz 0.5"; wE="--convection on --vx 0.6 --vy -0.4 --vz 0.6"; wN="--convection on --vx 0 --vy 0 --vz -0.75"
N4="--nx 4 --ny 4 --nz 4 --steps 8"; TB="--tracking-box3d 0.2 0.8 0.2 0.8 0.2 0.8"
PN="--dimension 3 --neumann-solve --state-bc natural --nu 0.25 --alpha-u 1e-2 --target-type sin_x_minus_t --T 1 $N4 $TB"
PD="--dimension 3 --dirichlet-solve --state-bc dirichlet --nu 0.25 --alpha-u 1e-2 --target-type sin_x_minus_t --T 1 $N4 $TB"
for c in "Nopp_be:be:--exact-oc neumann-opp-3d --solve-only $N4 $wA" "Nadj_cn:cn:--exact-oc neumann-adj-3d --solve-only $N4 $wE" \
         "Npadj_be:be:$PN --neumann-face3d zmin,xmin $wB" "Npthr_cn:cn:$PN --neumann-face3d xmin,ymin,zmin $wA" \
         "Dadj_cn:cn:$PD --dirichlet-face3d zmin,xmin $wA" "Dopp_be:be:--exact-oc dirichlet-opp-3d --solve-only $N4 $wN" \
         "Dthr_cn:cn:$PD --dirichlet-face3d zmin,ymin,xmin $wB" \
         "DadjH1_cn:cn:$PD --dirichlet-face3d zmin,xmin $wA --control-h1-length 0.3 --control-h1-time-weight 1e-2" \
         "Dall_be:be:$PD --dirichlet-face3d zmin,ymin,xmax,ymax,xmin,zmax $wA"; do
  tag=${c%%:*}; rest=${c#*:}; s=${rest%%:*}; args=${rest#*:}
  for np in 1 2 4; do
    case $tag in
      N*) popt="$LU"; [ $np -gt 1 ] && popt="$LU $MUMPS"; extra="--pc-schur-sweep";;
      *)  popt="-ksp_monitor -ksp_rtol 1e-14"; extra="--dirichlet-core-rtol 1e-12";;
    esac
    PETSC_OPTIONS="$popt" mpirun -np $np /usr/bin/time -f "RANK_MAXRSS=%M wall=%e" $X $args --time-scheme $s \
      $extra --no-paraview --dump-json $J/${tag}_np$np.json > $D/logs/mpi/${tag}_np$np.log 2>&1
    echo "${tag}_np$np exit=$? $(grep -oE 'nrank=\S+' $D/logs/mpi/${tag}_np$np.log | head -1)"
  done
done
