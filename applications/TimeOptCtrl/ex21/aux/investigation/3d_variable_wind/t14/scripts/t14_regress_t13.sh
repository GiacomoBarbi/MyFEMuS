#!/bin/bash
# T14: T13 (multi-face, constant wind) and 2D variable-wind invariance.  OLD = the T13 build (a28630779, private
# clean build) vs NEW (T14): every EX20_/EX21_ line (timings stripped), JSON and operator dumps byte-compared.
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
W=${WORK:?}; OLD=${OLD:?}; mkdir -p $W/output; cd $W
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
LU="-sweep_right_pc_type lu -sweep_left_pc_type lu -sweep_ay_pc_type lu -fieldsplit_y_sweep_pc_type lu -ksp_rtol 1e-12"
strip() { grep -E "^EX2[01]_" | sed -E 's/ (solve_wall|total_wall|setup_s|solve_s|t_operators|t_pcsetup|t_kspsolve|t_total|setup_seconds|core_seconds|shell_seconds|json)=[^ ]*//g'; }
cmpx() { name=$1; po=$2; shift 2
  PETSC_OPTIONS="$po" $OLD "$@" --no-paraview 2>/dev/null | strip > $name.old
  PETSC_OPTIONS="$po" $X "$@" --no-paraview > $name.raw 2>&1; e=$?; strip < $name.raw > $name.new
  if diff -q $name.old $name.new > /dev/null; then v=IDENTICAL; else v="DIFFERS ($(diff $name.old $name.new | grep -c '^[<>]') lines)"; fi
  echo "$name: $v ($(wc -l < $name.new) lines, exit $e)"; }
jcmp() { name=$1; po=$2; shift 2
  PETSC_OPTIONS="$po" $OLD "$@" --no-paraview --dump-json $W/$name.old.json > /dev/null 2>&1
  PETSC_OPTIONS="$po" $X "$@" --no-paraview --dump-json $W/$name.new.json > /dev/null 2>&1
  sed -i -E 's/"json":"[^"]*"//' $W/$name.old.json $W/$name.new.json
  cmp -s $W/$name.old.json $W/$name.new.json && echo "$name: JSON_IDENTICAL" || echo "$name: JSON_DIFFERS"; }
wA="--convection on --vx 1 --vy 0.5 --vz -0.75"
echo "== T13 exact multi-face cases (constant wind)"
for s in be cn; do
  cmpx ex_nopp_$s "$LU" --exact-oc neumann-opp-3d --pc-schur-sweep --nx 3 --ny 3 --nz 3 --steps 8 --time-scheme $s $wA
  cmpx ex_nadj_$s "$LU" --exact-oc neumann-adj-3d --pc-schur-sweep --nx 3 --ny 3 --nz 3 --steps 8 --time-scheme $s --convection on --vx 0.6 --vy -0.4 --vz 0.6
  cmpx ex_dopp_$s "" --exact-oc dirichlet-opp-3d --nx 4 --ny 4 --nz 4 --steps 8 --time-scheme $s --convection on --vx 0 --vy 0 --vz -0.75
  cmpx ex_doppH1_$s "" --exact-oc dirichlet-opp-3d --nx 4 --ny 4 --nz 4 --steps 8 --time-scheme $s --convection on --vx 0 --vy 0 --vz -0.75 --control-h1-length 0.3 --control-h1-time-weight 1e-2
done
echo "== T13 gates (unions, constant wind)"
PN="--dimension 3 --neumann-solve --state-bc natural --nu 0.25 --alpha-u 1e-2 --target-type sin_x_minus_t --steps 4 --T 1 --nx 3 --ny 3 --nz 3 --tracking-box3d 0.2 0.8 0.2 0.8 0.2 0.8 $wA"
PD="--dimension 3 --dirichlet-solve --state-bc dirichlet --nu 0.25 --alpha-u 1e-2 --target-type sin_x_minus_t --steps 4 --T 1 --nx 3 --ny 3 --nz 3 --tracking-box3d 0.2 0.8 0.2 0.8 0.2 0.8 $wA"
for s in be cn; do
  cmpx gate_N_adj_$s "$LU" $PN --neumann-face3d zmin,xmin --pc-schur-sweep --time-scheme $s --verify-convection
  cmpx gate_N_thr_$s "$LU" $PN --neumann-face3d xmin,ymin,zmin --pc-schur-sweep --time-scheme $s --verify-convection
  cmpx gate_D_adj_$s "" $PD --dirichlet-face3d zmin,xmin --time-scheme $s --verify-convection
  cmpx gate_D_all_comb_$s "" $PD --dirichlet-face3d zmin,ymin,xmax,ymax,xmin,zmax --time-scheme $s --verify-convection --control-h1-length 0.3 --control-h1-time-weight 1e-2
done
echo "== T13 JSON and dumps"
jcmp json_D_thr "" $PD --dirichlet-face3d zmin,ymin,xmin --time-scheme cn
jcmp json_N_opp "$LU" $PN --neumann-face3d zmin,zmax --pc-schur-sweep --time-scheme be
for k in N D; do
  if [ $k = N ]; then b="$PN --neumann-face3d zmin,xmin --pc-schur-sweep"; else b="$PD --dirichlet-face3d zmin,xmin --control-h1-length 0.3"; fi
  $OLD $b --steps 2 --no-paraview --dump-operators old_$k > /dev/null 2>&1; $X $b --steps 2 --no-paraview --dump-operators new_$k > /dev/null 2>&1
  for g in old_${k}_*; do t=${g#old_}; cmp -s $g new_$t && echo "dump $t: BYTE-IDENTICAL" || echo "dump $t: DIFFERS"; done
done
echo "== 2D variable-wind gates and exact cases (q2-divfree, lp-recirculating)"
for s in be cn; do
  cmpx v2d_gate_q2_$s "$LU" --verify-convection --pc-schur-sweep --exact-oc conv-sin3 --wind q2-divfree --nx 6 --nz 6 --steps 8 --time-scheme $s --distributed-solve
  cmpx v2d_exact_q2_$s "$LU" --exact-oc conv-sin3 --wind q2-divfree --nx 8 --nz 8 --steps 16 --time-scheme $s --distributed-solve --pc-schur-sweep
done
cmpx v2d_gate_lp52 "" --verify-convection --pc-schur-sweep --exact-oc lp52 --nx 8 --nz 8 --steps 8 --time-scheme cn --distributed-solve
echo T14_REGRESS_T13_DONE
