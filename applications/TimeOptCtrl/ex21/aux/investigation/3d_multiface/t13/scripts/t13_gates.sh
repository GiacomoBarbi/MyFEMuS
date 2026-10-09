#!/bin/bash
# T13 gate battery (--verify-convection).  Tags: <kind><group>_<reg>_<wind>_<scheme>_n<n>.
# NEUMANN (L2 only; SCHUR_SWEEP + LU spatial solves, outer rtol 1e-12):
#   Nnop  neumann-opp-3d exact (zmin u zmax), n = 2, 3, heat / wA / wB
#   Nnad  neumann-adj-3d exact (zmin u xmin), n = 2, 3, heat / wE = (0.6,-0.4,0.6) (w_x = w_z)
#   production n = 3: Nopx xmin,xmax wA; Nadj zmin,xmin wA; Nadb ymin,zmax wB; Nthr xmin,ymin,zmin wA;
#   NU zmin,xmin,xmax wB; Nall all six wA; Nnu non-unit (0,2)x(-1,1)x(0,0.5) zmin,ymax (4,4,2) wA
#   (the G13N union / drop-face / duplicate-face lines appear for every union)
# DIRICHLET (default reduced solver, iterative CG core), regularisations L2 / H1s (ell = 0.3) /
#   H1t (gamma_u = 1e-2) / comb:
#   Ddop  dirichlet-opp-3d exact (zmin u zmax), n = 4, 8: heat and wN for all four; wA for L2 / H1t
#   production n = 3, tracking box (0.2,0.8)^3 (interior, away from every face): Dopp zmin,zmax wA;
#   Dadj zmin,xmin wA; Dadb xmax,ymax wB; Dthr zmin,ymin,xmin wA; DU zmin,xmin,xmax wB;
#   Dfiv all but zmax wA; Dall all six wA; Dnu non-unit zmin,xmin (4,4,2) wA
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
D=$(cd "$(dirname "$0")/.." && pwd)
W=${T13_WORK:-/tmp/t13_gates}; mkdir -p $W/output $D/logs/gates; cd $W
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
LU="-sweep_right_pc_type lu -sweep_left_pc_type lu -sweep_ay_pc_type lu -fieldsplit_y_sweep_pc_type lu -ksp_rtol 1e-12"
runN() { tag=$1; shift; PETSC_OPTIONS="$LU" $X --verify-convection --pc-schur-sweep --no-paraview "$@" > $D/logs/gates/$tag.log 2>&1; echo "$tag exit=$?"; }
runD() { tag=$1; shift; $X --verify-convection --no-paraview "$@" > $D/logs/gates/$tag.log 2>&1; echo "$tag exit=$?"; }
wA="--convection on --vx 1 --vy 0.5 --vz -0.75"; wB="--convection on --vx -2 --vy 1 --vz 0.5"; wN="--convection on --vx 0 --vy 0 --vz -0.75"
wE="--convection on --vx 0.6 --vy -0.4 --vz 0.6"; heat="--convection off"
for n in 2 3; do for s in be cn; do
  for w in heat wA wB; do runN Nnop_L2_${w}_${s}_n$n --exact-oc neumann-opp-3d --nx $n --ny $n --nz $n --steps 4 --time-scheme $s ${!w}; done
  for w in heat wE; do runN Nnad_L2_${w}_${s}_n$n --exact-oc neumann-adj-3d --nx $n --ny $n --nz $n --steps 4 --time-scheme $s ${!w}; done
done; done
PN="--dimension 3 --neumann-solve --state-bc natural --nu 0.25 --alpha-u 1e-2 --target-type sin_x_minus_t --steps 4 --T 1 --nx 3 --ny 3 --nz 3 --tracking-box3d 0.2 0.8 0.2 0.8 0.2 0.8"
for s in be cn; do
  runN Nopx_L2_wA_${s}_n3 $PN --neumann-face3d xmin,xmax --time-scheme $s $wA
  runN Nadj_L2_wA_${s}_n3 $PN --neumann-face3d zmin,xmin --time-scheme $s $wA
  runN Nadb_L2_wB_${s}_n3 $PN --neumann-face3d ymin,zmax --time-scheme $s $wB
  runN Nthr_L2_wA_${s}_n3 $PN --neumann-face3d xmin,ymin,zmin --time-scheme $s $wA
  runN NU_L2_wB_${s}_n3 $PN --neumann-face3d zmin,xmin,xmax --time-scheme $s $wB
  runN Nall_L2_wA_${s}_n3 $PN --neumann-face3d zmin,ymin,xmax,ymax,xmin,zmax --time-scheme $s $wA
  runN Nnu_L2_wA_${s}_n4 --dimension 3 --neumann-solve --state-bc natural --nu 0.25 --alpha-u 1e-2 --target-type sin_x_minus_t --steps 4 --T 1 \
       --domain3d 0 2 -1 1 0 0.5 --nx 4 --ny 4 --nz 2 --tracking-box3d 0.5 1.5 -0.5 0.5 0 0.5 --neumann-face3d zmin,ymax --time-scheme $s $wA
done
REGS=("L2:" "H1s:--control-h1-length 0.3" "H1t:--control-h1-time-weight 1e-2" "comb:--control-h1-length 0.3 --control-h1-time-weight 1e-2")
for n in 4 8; do for s in be cn; do for r in "${REGS[@]}"; do   # n_z multiple of 4 (omega resolved; n = 2 is refused)
  for w in heat wN wA; do
    [ $w = wA ] && { [ ${r%%:*} = L2 ] || [ ${r%%:*} = H1t ]; } || [ $w != wA ] || continue
    runD Ddop_${r%%:*}_${w}_${s}_n$n --exact-oc dirichlet-opp-3d --nx $n --ny $n --nz $n --steps 4 --time-scheme $s ${!w} ${r#*:}
  done
done; done; done
PD="--dimension 3 --dirichlet-solve --state-bc dirichlet --nu 0.25 --alpha-u 1e-2 --target-type sin_x_minus_t --steps 4 --T 1 --nx 3 --ny 3 --nz 3 --tracking-box3d 0.2 0.8 0.2 0.8 0.2 0.8"
for s in be cn; do for r in "${REGS[@]}"; do
  R=${r#*:}; G=${r%%:*}
  runD Dopp_${G}_wA_${s}_n3 $PD --dirichlet-face3d zmin,zmax --time-scheme $s $wA $R
  runD Dadj_${G}_wA_${s}_n3 $PD --dirichlet-face3d zmin,xmin --time-scheme $s $wA $R
  runD Dadb_${G}_wB_${s}_n3 $PD --dirichlet-face3d xmax,ymax --time-scheme $s $wB $R
  runD Dthr_${G}_wA_${s}_n3 $PD --dirichlet-face3d zmin,ymin,xmin --time-scheme $s $wA $R
  runD DU_${G}_wB_${s}_n3 $PD --dirichlet-face3d zmin,xmin,xmax --time-scheme $s $wB $R
  runD Dfiv_${G}_wA_${s}_n3 $PD --dirichlet-face3d zmin,ymin,xmax,ymax,xmin --time-scheme $s $wA $R
  runD Dall_${G}_wA_${s}_n3 $PD --dirichlet-face3d zmin,ymin,xmax,ymax,xmin,zmax --time-scheme $s $wA $R
  runD Dnu_${G}_wA_${s}_n4 --dimension 3 --dirichlet-solve --state-bc dirichlet --nu 0.25 --alpha-u 1e-2 --target-type sin_x_minus_t --steps 4 --T 1 \
       --domain3d 0 2 -1 1 0 0.5 --nx 4 --ny 4 --nz 2 --tracking-box3d 1 2 -1 1 0.25 0.5 --dirichlet-face3d zmin,xmin --time-scheme $s $wA $R
done; done
