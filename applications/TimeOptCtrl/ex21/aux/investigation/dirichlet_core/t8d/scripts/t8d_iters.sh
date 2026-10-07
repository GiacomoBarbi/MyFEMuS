#!/bin/bash
# T8D inner CG iteration scaling (iterative core, rtol 1e-10): production
# data sin_x_minus_t, nu = 0.25, T = 1, CN unless stated; base nx = 16, N = 32,
# alpha_u = 1e-2, ell = 0, gamma_u = 0; heat and convection w = (1, 0.5).
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
D=$(cd "$(dirname "$0")/.." && pwd)
W=${T8D_WORK:-/tmp/t8d_iters}; mkdir -p $W/output $D/logs/iters; cd $W
run() { local tag=$1; shift
  OMP_NUM_THREADS=1 $X --dirichlet-solve --nu 0.25 --target-type sin_x_minus_t --T 1 --no-paraview \
    --dirichlet-core-solver iterative "$@" > $D/logs/iters/$tag.log 2>&1; echo "$tag exit=$?"; }
for conv in "off" "on --vx 1 --vy 0.5"; do c=${conv%% *}
  for pc in rt none; do
    for nx in 8 16 32 64; do run nx_${c}_${pc}_$nx --convection $conv --dirichlet-core-pc $pc --nx $nx --nz $nx --steps 32 --alpha-u 1e-2 --time-scheme cn; done
    for nt in 16 32 64 128 256; do run nt_${c}_${pc}_$nt --convection $conv --dirichlet-core-pc $pc --nx 16 --nz 16 --steps $nt --alpha-u 1e-2 --time-scheme cn; done
    for a in 1e-1 1e-2 1e-3 1e-4 1e-5; do run alpha_${c}_${pc}_$a --convection $conv --dirichlet-core-pc $pc --nx 16 --nz 16 --steps 32 --alpha-u $a --time-scheme cn; done
  done
  for l in 0 0.1 0.3 1 3; do run ell_${c}_rt_$l --convection $conv --nx 16 --nz 16 --steps 32 --alpha-u 1e-2 --control-h1-length $l --time-scheme cn; done
  for g in 0 1e-4 1e-3 1e-2 1e-1; do run gamma_${c}_rt_$g --convection $conv --nx 16 --nz 16 --steps 32 --alpha-u 1e-2 --control-h1-time-weight $g --time-scheme cn; done
  for s in be; do run be_${c}_rt_base --convection $conv --nx 16 --nz 16 --steps 32 --alpha-u 1e-2 --time-scheme be; done
done
