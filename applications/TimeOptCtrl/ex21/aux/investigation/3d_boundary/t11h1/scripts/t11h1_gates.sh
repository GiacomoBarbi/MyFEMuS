#!/bin/bash
# T11H1 gate battery (--verify-convection, default reduced Dirichlet solver, iterative CG core).
# Regularisations: L2, H1s (ell = 0.3), H1t (gamma_u = 1e-2), comb (both).  BE / CN.
#   ex   dirichlet-h1-3d exact OC (Gamma_c = zmin), n = 2, 3: heat and the normal wind wN = (0,0,-0.75)
#        for all four; the tangential wind wA = (1,0.5,-0.75) for L2 and H1t (ell > 0 needs w_x = w_y = 0)
#   pxm  production face xmax, wind A, n = 3;  pym  face ymin, wind B = (-2,1,0.5), n = 3;
#   pnu  production non-unit (0,2)x(-1,1)x(0,0.5), face ymax, wind A, (4,4,2)
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
D=$(cd "$(dirname "$0")/.." && pwd)
W=${T11H1_WORK:-/tmp/t11h1_gates}; mkdir -p $W/output $D/logs/gates; cd $W
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
run() { tag=$1; shift; $X --verify-convection --no-paraview "$@" > $D/logs/gates/$tag.log 2>&1; echo "$tag exit=$?"; }
REGS=("L2:" "H1s:--control-h1-length 0.3" "H1t:--control-h1-time-weight 1e-2" "comb:--control-h1-length 0.3 --control-h1-time-weight 1e-2")
for n in 2 3; do for s in be cn; do for r in "${REGS[@]}"; do
  for w in "heat:--convection off" "wN:--convection on --vx 0 --vy 0 --vz -0.75" "wA:--convection on --vx 1 --vy 0.5 --vz -0.75"; do
    [ ${w%%:*} = wA ] && { [ ${r%%:*} = L2 ] || [ ${r%%:*} = H1t ]; } || [ ${w%%:*} != wA ] || continue
    run ex_${r%%:*}_${w%%:*}_${s}_n$n --exact-oc dirichlet-h1-3d --nx $n --ny $n --nz $n --steps 4 --time-scheme $s ${w#*:} ${r#*:}
  done
done; done; done
P="--dimension 3 --dirichlet-solve --state-bc dirichlet --nu 0.25 --alpha-u 1e-2 --target-type sin_x_minus_t --steps 4 --T 1"
for s in be cn; do for r in "${REGS[@]}"; do
  run pxm_${r%%:*}_wA_${s}_n3 $P --dirichlet-face3d xmax --tracking-box3d 0 0.6 0.2 0.8 0.2 0.8 --nx 3 --ny 3 --nz 3 --time-scheme $s --convection on --vx 1 --vy 0.5 --vz -0.75 ${r#*:}
  run pym_${r%%:*}_wB_${s}_n3 $P --dirichlet-face3d ymin --tracking-box3d 0.2 0.8 0.4 1 0.2 0.8 --nx 3 --ny 3 --nz 3 --time-scheme $s --convection on --vx -2 --vy 1 --vz 0.5 ${r#*:}
  run pnu_${r%%:*}_wA_${s}_n4 $P --dirichlet-face3d ymax --domain3d 0 2 -1 1 0 0.5 --tracking-box3d 0.5 1.5 -1 0 0 0.5 --nx 4 --ny 4 --nz 2 --time-scheme $s --convection on --vx 1 --vy 0.5 --vz -0.75 ${r#*:}
done; done
