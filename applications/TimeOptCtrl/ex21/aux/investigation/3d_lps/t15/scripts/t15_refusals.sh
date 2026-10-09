#!/bin/bash
# T15 strict refusals / acceptances for 3D LPS: "R" must be refused, "A" accepted (exit 0, reason 2).
# Time-dependent and state-dependent winds have no CLI and no field type (nothing can request them); an
# unknown --wind name is refused by the parser.  "Z" = accepted AND LPS inactive (active = 0, W = 0) AND the
# solution byte-identical to the same run without LPS (the Pe <= 1 / zero-wind switch at production level).
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
W=${T15_WORK:-/tmp/t15_refusals}; mkdir -p $W/output; cd $W
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
pass=0; fail=0
R() { name=$1; shift; out=$($X "$@" --no-paraview 2>&1 | grep -oE "what\(\): .*" | head -1)
  if [ -n "$out" ]; then pass=$((pass+1)); v=REFUSED; else fail=$((fail+1)); v="NOT REFUSED (FAIL)"; fi
  echo "R $name: $v ${out#what(): }"; }
A() { name=$1; shift; out=$($X "$@" --no-paraview 2>&1); e=$?
  line=$(echo "$out" | grep -oE "EX20_PAR_SOLVE .*reason=\S+|EX21_EXACT3D_RESULT .*reason=\S+" | grep -oE "control=\S+|pc=\S+|reason=\S+" | tr '\n' ' ')
  act=$(echo "$out" | grep -oE "EX21_LPS3D active_fraction=\S+")
  if [ $e = 0 ] && echo "$line" | grep -q "reason=2" && [ -n "$act" ]; then pass=$((pass+1)); v=ACCEPTED; else fail=$((fail+1)); v="FAILED exit=$e $(echo "$out" | grep -oE 'what\(\): .*')"; fi
  echo "A $name: $v $line ${act#EX21_LPS3D }"; }
Z() { name=$1; shift; $X "$@" --no-paraview --dump-json $W/z_on.json > $W/z_on.log 2>&1; e1=$?
  off=(); skip=0; for a in "$@"; do [ $skip = 1 ] && { skip=0; continue; }; [ "$a" = --convection-stabilization ] && { skip=1; continue; }; off+=("$a"); done
  $X "${off[@]}" --no-paraview --dump-json $W/z_off.json > /dev/null 2>&1
  act=$(grep -oE "EX21_LPS projection=\S+ patches=\S+ active=\S+" $W/z_on.log); wn=$(grep -oE "\|\|W\|\|_F=\S+" $W/z_on.log | head -1)
  same=$(python3 -c "
import json,sys
a=json.load(open('$W/z_on.json')); b=json.load(open('$W/z_off.json'))
print(all(a[k]==b[k] for k in ('state','adjoint','control_full')) and a['functional']==b['functional'])")
  if [ $e1 = 0 ] && echo "$act" | grep -q "active=0 *$\|active=0$" && [ "$wn" = "||W||_F=0.000000e+00" ] && [ "$same" = True ]; then pass=$((pass+1)); v=INACTIVE_IDENTICAL; else fail=$((fail+1)); v="FAIL exit=$e1 same=$same"; fi
  echo "Z $name: $v $act $wn identical_to_galerkin=$same"; }
B3="--dimension 3 --nx 3 --ny 3 --nz 3 --steps 2 --nu 1e-2 --alpha-u 1e-2 --target-type sin_x_minus_t --T 1"
DIST="$B3 --distributed-solve --control-full-domain --tracking-box3d 0 1 0 1 0 1 --pc-schur-sweep"
NEU="$B3 --neumann-solve --state-bc natural --tracking-box3d 0.2 0.8 0.2 0.8 0.2 0.8 --pc-schur-sweep"
DIR="$B3 --dirichlet-solve --state-bc dirichlet --tracking-box3d 0.2 0.8 0.2 0.8 0.2 0.8"
C="--convection on --vx 1 --vy 0.5 --vz -0.75"; H="--convection on --wind harmonic-3d"; F="--convection on --wind affine-div-3d"
L="--convection-stabilization lps"
# ---- refused
R lps_simplex_const --dimension 3 --mesh-type tet --nx 2 --ny 2 --nz 2 --steps 2 --distributed-solve --control-full-domain --tracking-box3d 0 1 0 1 0 1 --state-bc dirichlet $C $L
R lps_without_convection $DIST --state-bc dirichlet $L
R lps_neumann_h1 $NEU --neumann-face3d zmin $H $L --control-h1-length 0.3
R lps_neumann_h1t $NEU --neumann-face3d zmin $H $L --control-h1-time-weight 1e-2
R lps_wind2d_in_3d $DIST --state-bc dirichlet --convection on --wind q2-divfree $L
R lps_unknown_projection $DIST --state-bc dirichlet $C $L --lps-projection p2
R lps_unknown_method $DIST --state-bc dirichlet $C --convection-stabilization supg
R lps_time_dependent_wind $DIST --state-bc dirichlet --convection on --wind time-dependent $L
R lps_state_dependent_wind $DIST --state-bc dirichlet --convection on --wind burgers $L
R lps_neumann_cn_without_sweep $B3 --neumann-solve --state-bc natural --tracking-box3d 0.2 0.8 0.2 0.8 0.2 0.8 --neumann-face3d zmin $H $L --time-scheme cn
R lps_dirichlet_sweep $DIR --dirichlet-face3d zmin $F $L --pc-schur-sweep
R lps_dirichlet_matching $DIR --dirichlet-face3d zmin $F $L --pc-schur-matching
R lps_dirichlet_natural_state $B3 --dirichlet-solve --state-bc natural --tracking-box3d 0.2 0.8 0.2 0.8 0.2 0.8 --dirichlet-face3d zmin $F $L
R lps_dist_matching_3d $B3 --distributed-solve --control-full-domain --tracking-box3d 0 1 0 1 0 1 --state-bc dirichlet $C $L --pc-schur-matching
R lps_exact_wind_mismatch --exact-oc natural-var-3d --nx 2 --ny 2 --nz 2 --steps 2 --pc-schur-sweep --convection on --vx 1 $L
# ---- accepted (LPS active)
for w in "c:$C" "h:$H" "f:$F"; do
  for s in be cn; do
    A dist_${w%%:*}_$s $DIST --state-bc dirichlet ${w#*:} $L --time-scheme $s
    A dist_p1_${w%%:*}_$s $DIST --state-bc dirichlet ${w#*:} $L --lps-projection p1 --time-scheme $s
    A nat_${w%%:*}_$s $DIST --state-bc natural ${w#*:} $L --time-scheme $s
    A neu_union_${w%%:*}_$s $NEU --neumann-face3d zmin,xmin ${w#*:} $L --time-scheme $s
    A dir_union_${w%%:*}_$s $DIR --dirichlet-face3d zmin,xmin ${w#*:} $L --time-scheme $s
    A dir_comb_${w%%:*}_$s $DIR --dirichlet-face3d zmin,zmax ${w#*:} $L --time-scheme $s --control-h1-length 0.3 --control-h1-time-weight 1e-2
  done
done
A dist_additive_gamg_be $B3 --distributed-solve --control-full-domain --tracking-box3d 0 1 0 1 0 1 --state-bc dirichlet $C $L
A dir_dense_core $DIR --dirichlet-face3d zmin,xmin $F $L --dirichlet-core-solver dense --verify-dirichlet-core
E="--nx 2 --ny 2 --nz 2 --steps 2"
A exact_conv_sin3_const --exact-oc conv-sin3-3d --exact-oc-nu 1e-2 $E --pc-schur-sweep $C $L
A exact_conv_sin3_harm --exact-oc conv-sin3-3d --exact-oc-nu 1e-2 $E --pc-schur-sweep $H $L
A exact_neumann_conv --exact-oc neumann-conv-3d --exact-oc-nu 2e-2 $E --pc-schur-sweep --convection on --vz 1 $L
A exact_dirichlet_conv --exact-oc dirichlet-conv-3d --exact-oc-nu 2e-2 $E --convection on --vz -0.75 $L
A exact_dirichlet_h1 --exact-oc dirichlet-h1-3d --exact-oc-nu 2e-2 $E --convection on --vz -0.75 --control-h1-length 0.3 $L
A exact_dirichlet_opp --exact-oc dirichlet-opp-3d --exact-oc-nu 2e-2 --nx 4 --ny 4 --nz 4 --steps 2 --convection on --vz -0.75 $L
# ---- accepted, LPS inactive, identical to Galerkin
Z zero_wind $DIST --state-bc dirichlet --convection on --vx 0 --vy 0 --vz 0 $L
Z low_pe_const $DIST --state-bc dirichlet --nu 1 $C $L
Z low_pe_natural_var --exact-oc natural-var-3d $E --pc-schur-sweep $H $L
echo "T15_REFUSALS pass=$pass fail=$fail"
