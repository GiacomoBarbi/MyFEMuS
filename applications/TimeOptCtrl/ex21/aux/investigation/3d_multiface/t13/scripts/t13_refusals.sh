#!/bin/bash
# T13 strict refusals and acceptances for face unions: every "R" line must be refused (invalid_argument),
# every "A" line accepted (exit 0, reason 2).
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
W=${T13_WORK:-/tmp/t13_refusals}; mkdir -p $W/output; cd $W
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
B="--dimension 3 --nx 2 --ny 2 --nz 2 --steps 2"
TBI="--tracking-box3d 0.2 0.8 0.2 0.8 0.2 0.8"   # n = 2: no centroid inside (0.25, 0.75 on the edges) -> use the n = 3 base below
B3="--dimension 3 --nx 3 --ny 3 --nz 3 --steps 2"
NEU="$B3 --neumann-solve --state-bc natural $TBI"
DIR="$B3 --dirichlet-solve --state-bc dirichlet $TBI"
wA="--convection on --vx 1 --vy 0.5 --vz -0.75"
pass=0; fail=0
R() { name=$1; shift; out=$($X "$@" --no-paraview 2>&1 | grep -oE "what\(\): .*" | head -1)
  if [ -n "$out" ]; then pass=$((pass+1)); v=REFUSED; else fail=$((fail+1)); v="NOT REFUSED (FAIL)"; fi
  echo "R $name: $v ${out#what(): }"; }
A() { name=$1; shift; out=$($X "$@" --no-paraview 2>&1); e=$?
  line=$(echo "$out" | grep -oE "EX20_PAR_SOLVE .*reason=\S+|EX21_EXACT3D_RESULT .*reason=\S+" | grep -oE "control=\S+|pc=\S+|reason=\S+" | tr '\n' ' ')
  if [ $e = 0 ] && echo "$line" | grep -q "reason=2"; then pass=$((pass+1)); v=ACCEPTED; else fail=$((fail+1)); v="FAILED exit=$e $(echo "$out" | grep -oE 'what\(\): .*')"; fi
  echo "A $name: $v $line"; }
# ---- malformed face lists (both options)
for opt in neumann dirichlet; do
  base=$NEU; [ $opt = dirichlet ] && base=$DIR
  R ${opt}_list_empty $base --$opt-face3d ""
  R ${opt}_list_comma_only $base --$opt-face3d ","
  R ${opt}_list_trailing_comma $base --$opt-face3d "zmin,"
  R ${opt}_list_leading_comma $base --$opt-face3d ",zmin"
  R ${opt}_list_double_comma $base --$opt-face3d "zmin,,xmin"
  R ${opt}_list_duplicate $base --$opt-face3d "zmin,zmin"
  R ${opt}_list_duplicate_nonadjacent $base --$opt-face3d "zmin,xmin,zmin"
  R ${opt}_list_unknown_name $base --$opt-face3d "zmin,top"
  R ${opt}_list_uppercase $base --$opt-face3d "ZMIN"
  R ${opt}_list_numeric_ids $base --$opt-face3d "1,3"
  R ${opt}_list_space $base --$opt-face3d "zmin, xmin"
  R ${opt}_list_seven_entries $base --$opt-face3d "zmin,ymin,xmax,ymax,xmin,zmax,zmin"
  R ${opt}_repeated_option_independent_controls $base --$opt-face3d zmin --$opt-face3d xmin
  R ${opt}_partial_face_box $base --$opt-face3d zmin,xmin --control-box3d-add 0 0.5 0 0.5 0 0.1
  R ${opt}_control_box_with_union $base --$opt-face3d zmin,xmin --control-box3d 0 1 0 1 0 0.5
  R ${opt}_lps_union $base --$opt-face3d zmin,xmin $wA --convection-stabilization lps
  R ${opt}_variable_wind_union $base --$opt-face3d zmin,xmin --convection on --wind q2-divfree
  R ${opt}_simplex_union --dimension 3 --mesh-type tet --nx 2 --ny 2 --nz 2 --steps 2 --$opt-solve $TBI --$opt-face3d zmin,xmin \
      $([ $opt = dirichlet ] && echo "--state-bc dirichlet" || echo "--state-bc natural")
done
R mixed_dirichlet_with_neumann_list $DIR --dirichlet-face3d zmin --neumann-face3d xmin,xmax
R mixed_neumann_with_dirichlet_list $NEU --neumann-face3d zmin --dirichlet-face3d xmin,xmax
R neumann_union_h1_space $NEU --neumann-face3d zmin,xmin --control-h1-length 0.3
R neumann_union_h1_time $NEU --neumann-face3d zmin,xmin --control-h1-time-weight 1e-2
R neumann_union_cn_without_sweep $NEU --neumann-face3d zmin,xmin --time-scheme cn $wA
R dirichlet_union_sweep $DIR --dirichlet-face3d zmin,xmin --pc-schur-sweep
R dirichlet_union_matching $DIR --dirichlet-face3d zmin,xmin --pc-schur-matching
R dirichlet_union_boundary_reduced $DIR --dirichlet-face3d zmin,xmin --pc-schur-boundary-reduced
R dirichlet_union_natural_state $B3 --dirichlet-solve --state-bc natural $TBI --dirichlet-face3d zmin,xmin
R dirichlet_union_verify_core_iterative $DIR --dirichlet-face3d zmin,xmin --dirichlet-core-solver iterative --verify-dirichlet-core
R dirichlet_union_tracking_touches_second_face $B3 --dirichlet-solve --state-bc dirichlet --dirichlet-face3d zmin,xmax --tracking-box3d 0.3 1 0.2 0.8 0.3 0.8
R dirichlet_union_tracking_touches_first_face $B3 --dirichlet-solve --state-bc dirichlet --dirichlet-face3d zmin,xmax --tracking-box3d 0.2 0.7 0.2 0.8 0 0.8
R neumann_union_face3d_in_2d --neumann-solve --nx 2 --nz 2 --steps 2 --neumann-face3d zmin,xmin
R exact_neumann_opp_with_face_list --exact-oc neumann-opp-3d --nx 2 --ny 2 --nz 2 --steps 2 --neumann-face3d zmin,zmax
R exact_dirichlet_opp_with_face_list --exact-oc dirichlet-opp-3d --nx 4 --ny 4 --nz 4 --steps 2 --dirichlet-face3d zmin,zmax
R exact_dirichlet_opp_with_region --exact-oc dirichlet-opp-3d --nx 4 --ny 4 --nz 4 --steps 2 --tracking-box3d 0 1 0 1 0 1
R exact_neumann_adj_wx_ne_wz --exact-oc neumann-adj-3d --nx 2 --ny 2 --nz 2 --steps 2 --convection on --vx 1 --vy 0 --vz 0.5
R exact_dirichlet_opp_nz2_empty_omega --exact-oc dirichlet-opp-3d --nx 2 --ny 2 --nz 2 --steps 2
R exact_dirichlet_opp_nz6 --exact-oc dirichlet-opp-3d --nx 6 --ny 6 --nz 6 --steps 2
R exact_dirichlet_opp_sweep --exact-oc dirichlet-opp-3d --nx 4 --ny 4 --nz 4 --steps 2 --pc-schur-sweep
R exact_dirichlet_opp_h1_tangential_wind --exact-oc dirichlet-opp-3d --nx 4 --ny 4 --nz 4 --steps 2 $wA --control-h1-length 0.3
R exact_unknown_multiface_name --exact-oc dirichlet-adj-3d --nx 4 --ny 4 --nz 4 --steps 2
# ---- accepted
LU="-sweep_right_pc_type lu -sweep_left_pc_type lu -sweep_ay_pc_type lu -fieldsplit_y_sweep_pc_type lu"
for fs in zmin zmax,zmin xmin,zmin zmin,ymin,xmin zmin,xmin,xmax zmin,ymin,xmax,ymax,xmin zmin,ymin,xmax,ymax,xmin,zmax; do
  A neumann_${fs//,/+}_cn $NEU --neumann-face3d $fs --pc-schur-sweep --time-scheme cn $wA
  A dirichlet_${fs//,/+}_cn $DIR --dirichlet-face3d $fs --time-scheme cn $wA
done
A dirichlet_union_h1_space $DIR --dirichlet-face3d zmin,xmin --control-h1-length 0.3 $wA
A dirichlet_union_h1_time $DIR --dirichlet-face3d zmin,xmin --control-h1-time-weight 1e-2 $wA --time-scheme cn
A dirichlet_union_h1_combined $DIR --dirichlet-face3d zmin,ymin,xmin --control-h1-length 0.3 --control-h1-time-weight 1e-2 $wA
A dirichlet_union_dense_core $DIR --dirichlet-face3d zmin,xmin --dirichlet-core-solver dense
A dirichlet_union_verify_core_dense $DIR --dirichlet-face3d zmin,xmin --dirichlet-core-solver dense --verify-dirichlet-core
A dirichlet_union_explicit_reduced_pc $DIR --dirichlet-face3d zmin,xmin --pc-schur-dirichlet-reduced
A neumann_union_nonunit $B3 --neumann-solve --state-bc natural --neumann-face3d zmin,ymax --domain3d 0 2 -1 1 0 0.5 --tracking-box3d 0.4 1.6 -0.6 0.6 0 0.5 --pc-schur-sweep $wA
A dirichlet_union_nonunit $B3 --dirichlet-solve --state-bc dirichlet --dirichlet-face3d zmin,xmin --domain3d 0 2 -1 1 0 0.5 --tracking-box3d 1 2 -1 1 0.2 0.5 $wA
A exact_neumann_opp --exact-oc neumann-opp-3d --nx 2 --ny 2 --nz 2 --steps 2 --pc-schur-sweep $wA
A exact_neumann_adj --exact-oc neumann-adj-3d --nx 2 --ny 2 --nz 2 --steps 2 --pc-schur-sweep --convection on --vx 0.6 --vy -0.4 --vz 0.6
A exact_dirichlet_opp --exact-oc dirichlet-opp-3d --nx 4 --ny 4 --nz 4 --steps 2 --convection on --vx 0 --vy 0 --vz -0.75
A exact_dirichlet_opp_h1 --exact-oc dirichlet-opp-3d --nx 4 --ny 4 --nz 4 --steps 2 --convection on --vx 0 --vy 0 --vz -0.75 --control-h1-length 0.3 --control-h1-time-weight 1e-2
A legacy_single_face_dirichlet_conv --exact-oc dirichlet-conv-3d --nx 2 --ny 2 --nz 2 --steps 2 $wA
A legacy_single_face_neumann_conv --exact-oc neumann-conv-3d --nx 2 --ny 2 --nz 2 --steps 2 --pc-schur-sweep $wA
echo "T13_REFUSALS pass=$pass fail=$fail"
