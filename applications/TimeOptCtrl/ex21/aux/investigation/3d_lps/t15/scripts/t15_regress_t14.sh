#!/bin/bash
# T15: T14 (3D variable wind, Galerkin) invariance and 2D T6 LPS byte-identity.  OLD = the T14 build (3d59677b8,
# private clean build) vs NEW (T15): every EX20_/EX21_ line (timings stripped), JSON and operator dumps byte-compared.
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
W=${WORK:?}; OLD=${OLD:?}; mkdir -p $W/output; cd $W
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
LU="-sweep_right_pc_type lu -sweep_left_pc_type lu -sweep_ay_pc_type lu -fieldsplit_y_sweep_pc_type lu -ksp_rtol 1e-12"
strip() { grep -E "^EX2[01]_" | sed -E 's/ (solve_wall|total_wall|setup_s|solve_s|t_operators|t_pcsetup|t_kspsolve|t_total|setup_seconds|core_seconds|shell_seconds|json)=[^ ]*//g'; }
cmpx() { name=$1; po=$2; shift 2
  PETSC_OPTIONS="$po" $OLD "$@" --no-paraview > $name.oraw 2>&1; strip < $name.oraw > $name.old
  PETSC_OPTIONS="$po" $X "$@" --no-paraview > $name.raw 2>&1; e=$?; strip < $name.raw > $name.new
  if diff -q $name.old $name.new > /dev/null; then v=IDENTICAL; else v="DIFFERS ($(diff $name.old $name.new | grep -c '^[<>]') lines)"; fi
  echo "$name: $v ($(wc -l < $name.new) lines, exit $e)"; }
jcmp() { name=$1; po=$2; shift 2
  PETSC_OPTIONS="$po" $OLD "$@" --no-paraview --dump-json $W/$name.old.json > /dev/null 2>&1
  PETSC_OPTIONS="$po" $X "$@" --no-paraview --dump-json $W/$name.new.json > /dev/null 2>&1
  sed -i -E 's/"json":"[^"]*"//' $W/$name.old.json $W/$name.new.json
  cmp -s $W/$name.old.json $W/$name.new.json && echo "$name: JSON_IDENTICAL" || echo "$name: JSON_DIFFERS"; }
dcmp() { name=$1; shift
  $OLD "$@" --no-paraview --dump-operators $W/old_$name > /dev/null 2>&1; $X "$@" --no-paraview --dump-operators $W/new_$name > /dev/null 2>&1
  for g in $W/old_${name}_*; do t=${g#$W/old_}; cmp -s $g $W/new_$t && echo "dump $t: BYTE-IDENTICAL" || echo "dump $t: DIFFERS"; done; }
H="--convection on --wind harmonic-3d"; F="--convection on --wind affine-div-3d"
echo "== T14 exact variable-wind cases"
for s in be cn; do
  for w in "$H" "$F"; do
    cmpx ex_dist_${w##* }_$s "$LU" --exact-oc conv-sin3-3d --pc-schur-sweep --nx 3 --ny 3 --nz 3 --steps 8 --time-scheme $s $w
    cmpx ex_nat_${w##* }_$s "$LU" --exact-oc natural-var-3d --pc-schur-sweep --nx 3 --ny 3 --nz 3 --steps 8 --time-scheme $s $w
    cmpx ex_neu_${w##* }_$s "$LU" --exact-oc neumann-var-3d --pc-schur-sweep --nx 3 --ny 3 --nz 3 --steps 8 --time-scheme $s $w
    cmpx ex_neuopp_${w##* }_$s "$LU" --exact-oc neumann-var-opp-3d --pc-schur-sweep --nx 3 --ny 3 --nz 3 --steps 8 --time-scheme $s $w
  done
  cmpx ex_dirswirl_$s "" --exact-oc dirichlet-var-3d --nx 4 --ny 4 --nz 4 --steps 8 --time-scheme $s --convection on --vz -0.75
done
echo "== T14 gates (Galerkin, variable winds)"
P="--dimension 3 --nu 0.25 --alpha-u 1e-2 --target-type sin_x_minus_t --steps 4 --T 1"
for s in be cn; do
  for w in "$H" "$F"; do
    cmpx g_dist_${w##* }_$s "$LU" --verify-convection --pc-schur-sweep $P --distributed-solve --control-full-domain --tracking-box3d 0 1 0 1 0 1 --state-bc dirichlet --nx 3 --ny 3 --nz 3 --time-scheme $s $w
    cmpx g_nat_${w##* }_$s "$LU" --verify-convection --pc-schur-sweep $P --distributed-solve --control-full-domain --tracking-box3d 0 1 0 1 0 1 --state-bc natural --nx 3 --ny 3 --nz 3 --time-scheme $s $w
    cmpx g_neu_${w##* }_$s "$LU" --verify-convection --pc-schur-sweep $P --neumann-solve --state-bc natural --nx 3 --ny 3 --nz 3 --tracking-box3d 0.2 0.8 0.2 0.8 0.2 0.8 --neumann-face3d zmin,xmin --time-scheme $s $w
    cmpx g_dir_${w##* }_$s "" --verify-convection $P --dirichlet-solve --state-bc dirichlet --nx 3 --ny 3 --nz 3 --tracking-box3d 0.2 0.8 0.2 0.8 0.2 0.8 --dirichlet-face3d zmin,zmax --time-scheme $s $w --control-h1-length 0.3 --control-h1-time-weight 1e-2
  done
done
echo "== T14 production JSON and operator dumps"
for w in "$H" "$F"; do
  jcmp j_dist_${w##* } "$LU" $P --solve-only --pc-schur-sweep --distributed-solve --control-full-domain --tracking-box3d 0 1 0 1 0 1 --state-bc dirichlet --nx 4 --ny 4 --nz 4 --time-scheme cn $w
  jcmp j_dir_${w##* } "" $P --solve-only --dirichlet-solve --state-bc dirichlet --nx 4 --ny 4 --nz 4 --tracking-box3d 0.2 0.8 0.2 0.8 0.2 0.8 --dirichlet-face3d zmin,xmin --time-scheme cn $w
  dcmp d_${w##* } $P --distributed-solve --control-full-domain --tracking-box3d 0 1 0 1 0 1 --state-bc natural --nx 3 --ny 2 --nz 2 --pc-schur-sweep $w
done
echo "== 2D T6 LPS byte-identity (gates, production, dumps; P0 and P1; constant and variable 2D winds)"
L2="--convection-stabilization lps"
for s in be cn; do
  for pj in p0 p1; do
    cmpx t6_gate_dist_${pj}_$s "$LU" --verify-convection --pc-schur-sweep --distributed-solve --nx 8 --nz 8 --steps 4 --nu 1e-3 --time-scheme $s --convection on --vx 1 --vy 0.5 $L2 --lps-projection $pj
    cmpx t6_gate_nat_${pj}_$s "$LU" --verify-convection --pc-schur-sweep --distributed-solve --state-bc natural --nx 8 --nz 8 --steps 4 --nu 1e-3 --time-scheme $s --convection on --vx 1 --vy 0.5 $L2 --lps-projection $pj
  done
  cmpx t6_gate_dir_$s "" --verify-convection --dirichlet-solve --nx 8 --nz 8 --steps 4 --nu 1e-2 --time-scheme $s --convection on --vx 1 --vy 0.5 $L2 --dirichlet-core-solver dense
  cmpx t6_gate_wind_q2_$s "$LU" --verify-convection --pc-schur-sweep --distributed-solve --nx 8 --nz 8 --steps 4 --nu 1e-3 --time-scheme $s --convection on --wind q2-divfree $L2
  cmpx t6_lp52_$s "$LU" --exact-oc lp52 --exact-oc-nu 0.002 --nx 8 --nz 8 --steps 16 --time-scheme $s --distributed-solve --pc-schur-sweep --convection-stabilization lps
  jcmp t6_json_lps_$s "$LU" --solve-only --pc-schur-sweep --distributed-solve --nx 8 --nz 8 --steps 8 --nu 1e-3 --time-scheme $s --convection on --vx 1 --vy 0.5 $L2
  jcmp t6_json_lps_neu_$s "$LU" --solve-only --pc-schur-sweep --neumann-solve --nx 8 --nz 8 --steps 8 --nu 1e-3 --time-scheme $s --convection on --vx 1 --vy 0.5 $L2
done
dcmp t6_lps_p0 --distributed-solve --nx 4 --nz 4 --steps 2 --nu 1e-3 --pc-schur-sweep --convection on --vx 1 --vy 0.5 $L2
dcmp t6_lps_p1 --distributed-solve --nx 4 --nz 4 --steps 2 --nu 1e-3 --pc-schur-sweep --convection on --wind lp-recirculating $L2 --lps-projection p1
echo T15_REGRESS_T14_DONE
