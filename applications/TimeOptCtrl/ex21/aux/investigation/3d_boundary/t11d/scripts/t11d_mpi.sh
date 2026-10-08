#!/bin/bash
# T11D MPI consistency, np = 1/2/4 (production path, --dump-json, -ksp_monitor = outer
# history, outer rtol 1e-14: the default 1e-8 stops after ONE outer iteration at a ~1e-11
# residual, and the np-to-np field differences then sit at that level -- see
# logs/mpi_table_default_outer_rtol.out), default Dirichlet solver (SCHUR_DIRICHLET_REDUCED, iterative CG core with the
# distributed reduced layout and the sparse R_T factor), core rtol 1e-12.  Sanitised env.
#   heat_be   dirichlet-conv-3d data (Gamma_c = zmin), convection off, BE, n = 4, N = 8
#   conv_be   dirichlet-conv-3d data, wind A, BE        conv_cn   same, CN
#   pxm_cn    production, face xmax, wind B = (-2, 1, 0.5), tracking away from the face, CN
#   pnu_be    production, non-unit (0,2)x(-1,1)x(0,0.5), face ymax, wind A, (4,4,2), BE
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
D=$(cd "$(dirname "$0")/.." && pwd)
W=${T11D_WORK:-/tmp/t11d_mpi}; J=${T11D_JSON:-/tmp/t11d_mpi_json}; mkdir -p $W/output $D/logs/mpi $J; cd $W
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
EX="--exact-oc dirichlet-conv-3d --solve-only --nx 4 --ny 4 --nz 4 --steps 8"
PR="--dimension 3 --dirichlet-solve --state-bc dirichlet --nu 0.25 --alpha-u 1e-2 --target-type sin_x_minus_t --steps 8 --T 1"
for c in "heat_be:be:$EX --convection off" "conv_be:be:$EX --convection on --vx 1 --vy 0.5 --vz -0.75" \
         "conv_cn:cn:$EX --convection on --vx 1 --vy 0.5 --vz -0.75" \
         "pxm_cn:cn:$PR --dirichlet-face3d xmax --tracking-box3d 0 0.7 0.2 0.8 0.2 0.8 --nx 4 --ny 4 --nz 4 --convection on --vx -2 --vy 1 --vz 0.5" \
         "pnu_be:be:$PR --dirichlet-face3d ymax --domain3d 0 2 -1 1 0 0.5 --tracking-box3d 0.4 1.6 -1 0.4 0 0.5 --nx 4 --ny 4 --nz 2 --convection on --vx 1 --vy 0.5 --vz -0.75"; do
  tag=${c%%:*}; rest=${c#*:}; s=${rest%%:*}; args=${rest#*:}
  for np in 1 2 4; do
    PETSC_OPTIONS="-ksp_monitor -ksp_rtol 1e-14" mpirun -np $np /usr/bin/time -f "RANK_MAXRSS=%M wall=%e" $X $args --time-scheme $s \
      --dirichlet-core-rtol 1e-12 --no-paraview --dump-json $J/${tag}_np$np.json > $D/logs/mpi/${tag}_np$np.log 2>&1
    echo "${tag}_np$np exit=$? $(grep -oE 'nrank=\S+' $D/logs/mpi/${tag}_np$np.log | head -1)"
  done
done
