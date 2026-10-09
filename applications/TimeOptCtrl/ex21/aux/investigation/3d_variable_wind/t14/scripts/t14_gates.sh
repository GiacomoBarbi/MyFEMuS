#!/bin/bash
# T14 gate battery (--verify-convection).  Tags: <kind>_<group>_<wind>_<reg>_<scheme>_n<n>,
#   kind  dist (distributed, homogeneous Dirichlet state) | nat (distributed, natural state) |
#         neu (Neumann boundary control) | dir (Dirichlet boundary control);
#   wind  wA (constant (1,0.5,-0.75), the baseline) | harm (harmonic-3d) | aff (affine-div-3d) | swirl.
# SCHUR_SWEEP + LU (outer rtol 1e-12) except Dirichlet control (default reduced solver, CG core).
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
D=$(cd "$(dirname "$0")/.." && pwd)
W=${T14_WORK:-/tmp/t14_gates}; mkdir -p $W/output $D/logs/gates; cd $W
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
LU="-sweep_right_pc_type lu -sweep_left_pc_type lu -sweep_ay_pc_type lu -fieldsplit_y_sweep_pc_type lu -ksp_rtol 1e-12"
runS() { tag=$1; shift; PETSC_OPTIONS="$LU" $X --verify-convection --pc-schur-sweep --no-paraview "$@" > $D/logs/gates/$tag.log 2>&1; echo "$tag exit=$?"; }
runD() { tag=$1; shift; $X --verify-convection --no-paraview "$@" > $D/logs/gates/$tag.log 2>&1; echo "$tag exit=$?"; }
declare -A WIND=( [wA]="--convection on --vx 1 --vy 0.5 --vz -0.75" [harm]="--convection on --wind harmonic-3d" [aff]="--convection on --wind affine-div-3d" )
P="--dimension 3 --nu 0.25 --alpha-u 1e-2 --target-type sin_x_minus_t --steps 4 --T 1"
NU="--domain3d 0 2 -1 1 0 0.5"
for s in be cn; do
  for w in wA harm aff; do
    for n in 2 3; do
      runS dist_ex_${w}_L2_${s}_n$n --exact-oc conv-sin3-3d --nx $n --ny $n --nz $n --steps 4 --time-scheme $s ${WIND[$w]}
    done
    runS dist_pr_${w}_L2_${s}_n3 $P $NU --distributed-solve --control-full-domain --tracking-box3d 0 2 -1 1 0 0.5 --state-bc dirichlet --nx 3 --ny 2 --nz 2 --time-scheme $s ${WIND[$w]}
    runS nat_pr_${w}_L2_${s}_n3 $P $NU --distributed-solve --control-full-domain --tracking-box3d 0 2 -1 1 0 0.5 --state-bc natural --nx 3 --ny 2 --nz 2 --time-scheme $s ${WIND[$w]}
  done
  for w in harm aff; do
    for n in 2 3; do
      runS nat_ex_${w}_L2_${s}_n$n --exact-oc natural-var-3d --nx $n --ny $n --nz $n --steps 4 --time-scheme $s ${WIND[$w]}
      runS neu_ex1_${w}_L2_${s}_n$n --exact-oc neumann-var-3d --nx $n --ny $n --nz $n --steps 4 --time-scheme $s ${WIND[$w]}
      runS neu_ex2_${w}_L2_${s}_n$n --exact-oc neumann-var-opp-3d --nx $n --ny $n --nz $n --steps 4 --time-scheme $s ${WIND[$w]}
    done
    PN="$P --neumann-solve --state-bc natural --nx 3 --ny 3 --nz 3 --tracking-box3d 0.2 0.8 0.2 0.8 0.2 0.8 --time-scheme $s ${WIND[$w]}"
    runS neu_one_${w}_L2_${s}_n3 $PN --neumann-face3d zmin
    runS neu_opp_${w}_L2_${s}_n3 $PN --neumann-face3d zmin,zmax
    runS neu_adj_${w}_L2_${s}_n3 $PN --neumann-face3d zmin,xmin
    runS neu_thr_${w}_L2_${s}_n3 $PN --neumann-face3d xmin,ymin,zmin
  done
  for n in 2 4; do
    runD dir_ex_swirl_L2_${s}_n$n --exact-oc dirichlet-var-3d --nx $n --ny $n --nz $n --steps 4 --time-scheme $s --convection on --vz -0.75
  done
  for w in wA harm aff; do
    for r in "L2:" "H1s:--control-h1-length 0.3" "H1t:--control-h1-time-weight 1e-2" "comb:--control-h1-length 0.3 --control-h1-time-weight 1e-2"; do
      [ $w = wA ] && [ ${r%%:*} != L2 ] && continue
      PD="$P --dirichlet-solve --state-bc dirichlet --nx 3 --ny 3 --nz 3 --tracking-box3d 0.2 0.8 0.2 0.8 0.2 0.8 --time-scheme $s ${WIND[$w]} ${r#*:}"
      runD dir_one_${w}_${r%%:*}_${s}_n3 $PD --dirichlet-face3d zmin
      runD dir_opp_${w}_${r%%:*}_${s}_n3 $PD --dirichlet-face3d zmin,zmax
      runD dir_adj_${w}_${r%%:*}_${s}_n3 $PD --dirichlet-face3d zmin,xmin
    done
    runD dir_nu_${w}_L2_${s}_n4 $P $NU --dirichlet-solve --state-bc dirichlet --nx 4 --ny 4 --nz 4 --tracking-box3d 0.5 1.5 -0.5 0.5 0.125 0.375 --dirichlet-face3d zmin,xmin --time-scheme $s ${WIND[$w]}
  done
done
