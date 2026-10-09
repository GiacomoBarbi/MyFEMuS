#!/bin/bash
# T15 gate battery (--verify-convection with --convection-stabilization lps).
# Tags: <kind>_<group>_<wind>_<var>_<scheme>_n<n>
#   kind  dist (distributed, homogeneous Dirichlet state) | nat (distributed, natural state) |
#         neu (Neumann boundary control, natural state) | dir (Dirichlet boundary control);
#   wind  wA (constant (1,0.5,-0.75)) | harm (harmonic-3d) | aff (affine-div-3d);
#   var   mod (nu = 0.1: moderate patch Pe) | high (nu = 1e-3) | p1 (high, P1 diagnostic projection) |
#         L2 / H1s / H1t / comb (Dirichlet regularisations, nu = 1e-2).
# SCHUR_SWEEP + LU (outer rtol 1e-12) except Dirichlet control (default reduced solver, CG core).
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
D=$(cd "$(dirname "$0")/.." && pwd)
W=${T15_WORK:-/tmp/t15_gates}; mkdir -p $W/output $D/logs/gates; cd $W
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
LU="-sweep_right_pc_type lu -sweep_left_pc_type lu -sweep_ay_pc_type lu -fieldsplit_y_sweep_pc_type lu -ksp_rtol 1e-12"
runS() { tag=$1; shift; PETSC_OPTIONS="$LU" $X --verify-convection --pc-schur-sweep --no-paraview --convection-stabilization lps "$@" > $D/logs/gates/$tag.log 2>&1; echo "$tag exit=$?"; }
runD() { tag=$1; shift; $X --verify-convection --no-paraview --convection-stabilization lps "$@" > $D/logs/gates/$tag.log 2>&1; echo "$tag exit=$?"; }
declare -A WIND=( [wA]="--convection on --vx 1 --vy 0.5 --vz -0.75" [harm]="--convection on --wind harmonic-3d" [aff]="--convection on --wind affine-div-3d" )
declare -A NUV=( [mod]="--nu 0.1" [high]="--nu 1e-3" [p1]="--nu 1e-3 --lps-projection p1" )
P="--dimension 3 --alpha-u 1e-2 --target-type sin_x_minus_t --steps 4 --T 1"
NU="--domain3d 0 2 -1 1 0 0.5"
for s in be cn; do
  for w in wA harm aff; do
    for v in mod high p1; do
      [ $v = p1 ] && [ $w = aff ] && continue
      runS dist_pr_${w}_${v}_${s}_n3 $P $NU ${NUV[$v]} --distributed-solve --control-full-domain --tracking-box3d 0 2 -1 1 0 0.5 --state-bc dirichlet --nx 3 --ny 2 --nz 2 --time-scheme $s ${WIND[$w]}
      runS nat_pr_${w}_${v}_${s}_n3 $P $NU ${NUV[$v]} --distributed-solve --control-full-domain --tracking-box3d 0 2 -1 1 0 0.5 --state-bc natural --nx 3 --ny 2 --nz 2 --time-scheme $s ${WIND[$w]}
    done
  done
  for w in wA harm; do
    runS dist_ex_${w}_high_${s}_n3 --exact-oc conv-sin3-3d --exact-oc-nu 1e-2 --nx 3 --ny 3 --nz 3 --steps 4 --time-scheme $s ${WIND[$w]}
  done
  for w in wA harm aff; do
    PN="$P --nu 1e-2 --neumann-solve --state-bc natural --nx 3 --ny 3 --nz 3 --tracking-box3d 0.2 0.8 0.2 0.8 0.2 0.8 --time-scheme $s ${WIND[$w]}"
    runS neu_one_${w}_high_${s}_n3 $PN --neumann-face3d zmin
    runS neu_opp_${w}_high_${s}_n3 $PN --neumann-face3d zmin,zmax
    runS neu_adj_${w}_high_${s}_n3 $PN --neumann-face3d zmin,xmin
  done
  for w in wA harm aff; do
    for r in "L2:" "H1s:--control-h1-length 0.3" "H1t:--control-h1-time-weight 1e-2" "comb:--control-h1-length 0.3 --control-h1-time-weight 1e-2"; do
      [ $w != harm ] && [ ${r%%:*} != L2 ] && continue
      PD="$P --nu 1e-2 --dirichlet-solve --state-bc dirichlet --nx 3 --ny 3 --nz 3 --tracking-box3d 0.2 0.8 0.2 0.8 0.2 0.8 --time-scheme $s ${WIND[$w]} ${r#*:}"
      runD dir_one_${w}_${r%%:*}_${s}_n3 $PD --dirichlet-face3d zmin
      [ ${r%%:*} != L2 ] && continue
      runD dir_opp_${w}_${r%%:*}_${s}_n3 $PD --dirichlet-face3d zmin,zmax
      runD dir_adj_${w}_${r%%:*}_${s}_n3 $PD --dirichlet-face3d zmin,xmin
    done
    runD dir_nu_${w}_L2_${s}_n4 $P $NU --nu 1e-2 --dirichlet-solve --state-bc dirichlet --nx 4 --ny 4 --nz 4 --tracking-box3d 0.5 1.5 -0.5 0.5 0.125 0.375 --dirichlet-face3d zmin,xmin --time-scheme $s ${WIND[$w]}
  done
done
echo T15_GATES_DONE
