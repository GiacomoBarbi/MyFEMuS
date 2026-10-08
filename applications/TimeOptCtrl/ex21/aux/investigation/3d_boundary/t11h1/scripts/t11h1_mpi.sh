#!/bin/bash
# T11H1 MPI consistency, np = 1/2/4 (production path, --dump-json, -ksp_monitor, outer rtol 1e-14 -- T11D:
# the default 1e-8 stops after one Woodbury step at ~1e-11), default solver, core rtol 1e-12.  Sanitised env.
#   s_cn   dirichlet-h1-3d (ell = 0.3), wind (0,0,-0.75), CN, n = 4, N = 8          H1-space convection CN
#   t_be   dirichlet-h1-3d (gamma = 1e-2), wind A (1,0.5,-0.75), BE                 H1-time convection BE
#   k_cn   production face xmax, wind B, ell = 0.3 + gamma = 1e-2, CN               combined convection CN
#   kh_be  dirichlet-h1-3d combined, heat, BE                                        combined heat
#   knu_cn production non-unit box face ymax, wind A, combined, CN
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
D=$(cd "$(dirname "$0")/.." && pwd)
W=${T11H1_WORK:-/tmp/t11h1_mpi}; J=${T11H1_JSON:-/tmp/t11h1_mpi_json}; mkdir -p $W/output $D/logs/mpi $J; cd $W
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
EX="--exact-oc dirichlet-h1-3d --solve-only --nx 4 --ny 4 --nz 4 --steps 8"
PR="--dimension 3 --dirichlet-solve --state-bc dirichlet --nu 0.25 --alpha-u 1e-2 --target-type sin_x_minus_t --steps 8 --T 1"
COMB="--control-h1-length 0.3 --control-h1-time-weight 1e-2"
for c in "s_cn:cn:$EX --control-h1-length 0.3 --convection on --vx 0 --vy 0 --vz -0.75" \
         "t_be:be:$EX --control-h1-time-weight 1e-2 --convection on --vx 1 --vy 0.5 --vz -0.75" \
         "k_cn:cn:$PR $COMB --dirichlet-face3d xmax --tracking-box3d 0 0.7 0.2 0.8 0.2 0.8 --nx 4 --ny 4 --nz 4 --convection on --vx -2 --vy 1 --vz 0.5" \
         "kh_be:be:$EX $COMB --convection off" \
         "knu_cn:cn:$PR $COMB --dirichlet-face3d ymax --domain3d 0 2 -1 1 0 0.5 --tracking-box3d 0.4 1.6 -1 0.4 0 0.5 --nx 4 --ny 4 --nz 2 --convection on --vx 1 --vy 0.5 --vz -0.75"; do
  tag=${c%%:*}; rest=${c#*:}; s=${rest%%:*}; args=${rest#*:}
  for np in 1 2 4; do
    PETSC_OPTIONS="-ksp_monitor -ksp_rtol 1e-14" mpirun -np $np /usr/bin/time -f "RANK_MAXRSS=%M wall=%e" $X $args --time-scheme $s \
      --dirichlet-core-rtol 1e-12 --no-paraview --dump-json $J/${tag}_np$np.json > $D/logs/mpi/${tag}_np$np.log 2>&1
    echo "${tag}_np$np exit=$? $(grep -oE 'nrank=\S+' $D/logs/mpi/${tag}_np$np.log | head -1) $(grep -oE 'rt_backend=\S+' $D/logs/mpi/${tag}_np$np.log | head -1)"
  done
done
