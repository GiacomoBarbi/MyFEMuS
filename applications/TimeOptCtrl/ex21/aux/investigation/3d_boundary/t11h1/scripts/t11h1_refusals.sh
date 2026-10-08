#!/bin/bash
# T11H1 3D whitelist (copy of t11d_refusals.sh): every "R" line must be refused (invalid_argument),
# every "A" line accepted (exit 0, converged).  H1-space / H1-time / combined are now ACCEPTED for 3D
# Dirichlet control and still REFUSED for 3D Neumann and distributed control.
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
W=${T11H1_WORK:-/tmp/t11h1_refusals}; mkdir -p $W/output; cd $W
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
B="--dimension 3 --nx 2 --ny 2 --nz 2 --steps 2"
TB="--tracking-box3d 0.2 0.8 0.2 0.8 0.4 1"   # element-centroid selection: n = 2 centroids 0.25, 0.75 must lie strictly inside
DIR="$B --dirichlet-solve --dirichlet-face3d zmin --state-bc dirichlet $TB"
NEU="$B --neumann-solve --neumann-face3d zmin --tracking-box3d 0 1 0 1 0 1 --state-bc natural"
OK="$B --distributed-solve --control-full-domain --tracking-box3d 0 1 0 1 0 1 --state-bc natural"
pass=0; fail=0
R() { name=$1; shift; out=$($X "$@" --no-paraview 2>&1 | grep -oE "what\(\): .*" | head -1)
  if [ -n "$out" ]; then pass=$((pass+1)); v=REFUSED; else fail=$((fail+1)); v="NOT REFUSED (FAIL)"; fi
  echo "R $name: $v ${out#what(): }"; }
A() { name=$1; shift; out=$($X "$@" --no-paraview 2>&1); e=$?
  line=$(echo "$out" | grep -oE "EX20_PAR_SOLVE .*reason=\S+|EX21_EXACT3D_RESULT .*reason=\S+" | grep -oE "control=\S+|pc=\S+|reason=\S+" | tr '\n' ' ')
  if [ $e = 0 ] && echo "$line" | grep -q "reason=2"; then pass=$((pass+1)); v=ACCEPTED; else fail=$((fail+1)); v="FAILED exit=$e $(echo "$out" | grep -oE 'what\(\): .*')"; fi
  echo "A $name: $v $line"; }
# ---- T11D: Dirichlet-specific refusals
R dirichlet_without_face $B --dirichlet-solve --state-bc dirichlet $TB
R dirichlet_bad_face_name $B --dirichlet-solve --dirichlet-face3d bottom --state-bc dirichlet $TB
R dirichlet_natural_state_2d_variant $B --dirichlet-solve --dirichlet-face3d zmin --state-bc natural $TB
R dirichlet_face3d_with_distributed $OK --dirichlet-face3d zmin
R dirichlet_face3d_with_neumann $NEU --dirichlet-face3d zmin
R dirichlet_face3d_in_2d --dirichlet-solve --nx 2 --nz 2 --steps 2 --dirichlet-face3d zmin
R dirichlet_with_neumann_face $DIR --neumann-face3d xmax
R dirichlet_control_box $DIR --control-box3d 0 1 0 1 0 0.5
R dirichlet_control_full_domain $DIR --control-full-domain
R dirichlet_partial_face_or_union $DIR --control-box3d-add 0 0.5 0 0.5 0 0.1
R dirichlet_negative_ell $DIR --control-h1-length -0.3
R dirichlet_negative_gamma $DIR --control-h1-time-weight -1e-3
R exact_l2_case_with_h1_space --exact-oc dirichlet-conv-3d --nx 2 --ny 2 --nz 2 --steps 2 --control-h1-length 0.3
R exact_l2_case_with_h1_time --exact-oc dirichlet-conv-3d --nx 2 --ny 2 --nz 2 --steps 2 --control-h1-time-weight 1e-2
R exact_h1_case_ell_tangential_wind --exact-oc dirichlet-h1-3d --nx 2 --ny 2 --nz 2 --steps 2 --control-h1-length 0.3 --convection on --vx 1 --vy 0.5 --vz -0.75
R exact_h1_case_gamma_too_small --exact-oc dirichlet-h1-3d --nx 2 --ny 2 --nz 2 --steps 2 --control-h1-time-weight 1e-5
R exact_h1_case_with_face --exact-oc dirichlet-h1-3d --nx 2 --ny 2 --nz 2 --steps 2 --dirichlet-face3d xmax
R exact_h1_case_with_sweep --exact-oc dirichlet-h1-3d --nx 2 --ny 2 --nz 2 --steps 2 --control-h1-length 0.3 --pc-schur-sweep
R dirichlet_h1_partial_face --dimension 3 --nx 2 --ny 2 --nz 2 --steps 2 --dirichlet-solve --dirichlet-face3d zmin --state-bc dirichlet --tracking-box3d 0.2 0.8 0.2 0.8 0.4 1 --control-h1-length 0.3 --control-box3d-add 0 0.5 0 0.5 0 0.1
R dirichlet_h1_variable_wind $DIR --control-h1-length 0.3 --convection on --wind q2-divfree
R dirichlet_h1_lps $DIR --control-h1-length 0.3 --convection on --vx 1 --convection-stabilization lps
R dirichlet_h1_sweep $DIR --control-h1-time-weight 1e-2 --pc-schur-sweep
R dirichlet_h1_boundary_reduced $DIR --control-h1-length 0.3 --pc-schur-boundary-reduced
R neumann_h1_time $NEU --control-h1-time-weight 1e-3 --pc-schur-sweep
R neumann_h1_combined $NEU --control-h1-length 0.3 --control-h1-time-weight 1e-3 --pc-schur-sweep
R distributed_h1_space $OK --control-h1-length 0.3 --pc-schur-sweep
R distributed_h1_time $OK --control-h1-time-weight 1e-3 --pc-schur-sweep
R dirichlet_sweep $DIR --pc-schur-sweep
R dirichlet_matching $DIR --pc-schur-matching
R dirichlet_boundary_reduced $DIR --pc-schur-boundary-reduced
R dirichlet_variable_wind $DIR --convection on --wind q2-divfree
R dirichlet_lps $DIR --convection on --vx 1 --convection-stabilization lps
R dirichlet_tracking_touches_face $B --dirichlet-solve --dirichlet-face3d xmax --state-bc dirichlet --tracking-box3d 0.3 0.9 0.2 0.8 0.2 0.8
R dirichlet_verify_core_iterative $DIR --dirichlet-core-solver iterative --verify-dirichlet-core
R exact_dir_case_with_face --exact-oc dirichlet-conv-3d --nx 2 --ny 2 --nz 2 --steps 2 --dirichlet-face3d xmax
R exact_dir_case_with_region --exact-oc dirichlet-conv-3d --nx 2 --ny 2 --nz 2 --steps 2 --tracking-box3d 0 1 0 1 0 1
R exact_dir_case_with_sweep --exact-oc dirichlet-conv-3d --nx 2 --ny 2 --nz 2 --steps 2 --pc-schur-sweep
# ---- carried over from T11N / T10
R neumann_without_face $B --neumann-solve --tracking-box3d 0 1 0 1 0 1 --state-bc natural
R neumann_dirichlet_state $B --neumann-solve --neumann-face3d zmin --tracking-box3d 0 1 0 1 0 1 --state-bc dirichlet
R neumann_h1_space $NEU --control-h1-length 0.3 --pc-schur-sweep
R neumann_cn_without_sweep $NEU --time-scheme cn
R neumann_lps $NEU --convection on --vx 1 --convection-stabilization lps
R exact_case_with_face --exact-oc neumann-conv-3d --nx 2 --ny 2 --nz 2 --steps 2 --neumann-face3d xmax
R lps_distributed $OK --convection on --vx 1 --convection-stabilization lps
R variable_wind_distributed $OK --convection on --wind q2-divfree
R math2025_preset $B --math2025-ex3
R domain_2d_syntax $OK --domain 0 2 0 1
R implicit_2d_tracking_default $B --distributed-solve --control-full-domain --state-bc natural
R bad_dimension --dimension 4 --distributed-solve
# ---- accepted
TBF() { case $1 in xmin) echo "0.4 1 0.2 0.8 0.2 0.8";; xmax) echo "0 0.6 0.2 0.8 0.2 0.8";; ymin) echo "0.2 0.8 0.4 1 0.2 0.8";;
  ymax) echo "0.2 0.8 0 0.6 0.2 0.8";; zmin) echo "0.2 0.8 0.2 0.8 0.4 1";; zmax) echo "0.2 0.8 0.2 0.8 0 0.6";; esac; }
for f in xmin xmax ymin ymax zmin zmax; do
  A dirichlet_face_${f}_cn_wA $B --dirichlet-solve --dirichlet-face3d $f --tracking-box3d $(TBF $f) --state-bc dirichlet --convection on --vx 1 --vy 0.5 --vz -0.75 --time-scheme cn
done
R dirichlet_face_by_numeric_id $B --dirichlet-solve --dirichlet-face3d 3 --tracking-box3d $(TBF xmax) --state-bc dirichlet   # names only, as --neumann-face3d
A dirichlet_explicit_reduced_pc $DIR --pc-schur-dirichlet-reduced
A dirichlet_dense_core $DIR --dirichlet-core-solver dense
A dirichlet_core_pc_none $DIR --dirichlet-core-pc none
A dirichlet_verify_core_dense $DIR --dirichlet-core-solver dense --verify-dirichlet-core
A dirichlet_heat_be $DIR
A dirichlet_nonunit $B --dirichlet-solve --dirichlet-face3d ymax --domain3d 0 2 -1 1 0 0.5 --tracking-box3d 0.4 1.6 -1 0.1 0 0.5 --state-bc dirichlet --time-scheme cn --convection on --vx 1 --vy 0.5 --vz -0.75
A dirichlet_h1_space_cn_wA $DIR --control-h1-length 0.3 --time-scheme cn --convection on --vx 1 --vy 0.5 --vz -0.75
A dirichlet_h1_time_be_wA $DIR --control-h1-time-weight 1e-2 --convection on --vx 1 --vy 0.5 --vz -0.75
A dirichlet_h1_combined_cn_heat $DIR --control-h1-length 0.3 --control-h1-time-weight 1e-2 --time-scheme cn
A dirichlet_h1_combined_dense_core $DIR --control-h1-length 0.3 --control-h1-time-weight 1e-2 --dirichlet-core-solver dense
A dirichlet_h1_combined_verify_core $DIR --control-h1-length 0.3 --control-h1-time-weight 1e-2 --dirichlet-core-solver dense --verify-dirichlet-core
A dirichlet_h1_face_xmax $B --dirichlet-solve --dirichlet-face3d xmax --tracking-box3d $(TBF xmax) --state-bc dirichlet --control-h1-length 1 --control-h1-time-weight 1e-3 --time-scheme cn
A dirichlet_h1_nonunit $B --dirichlet-solve --dirichlet-face3d ymax --domain3d 0 2 -1 1 0 0.5 --tracking-box3d 0.4 1.6 -1 0.1 0 0.5 --state-bc dirichlet --control-h1-length 0.3 --control-h1-time-weight 1e-2 --time-scheme cn --convection on --vx 1 --vy 0.5 --vz -0.75
A exact_dirichlet_h1_3d_combined --exact-oc dirichlet-h1-3d --nx 2 --ny 2 --nz 2 --steps 2 --control-h1-length 0.3 --control-h1-time-weight 1e-2 --convection on --vx 0 --vy 0 --vz -0.75 --time-scheme cn
A exact_dirichlet_h1_3d_time_tangential_wind --exact-oc dirichlet-h1-3d --nx 2 --ny 2 --nz 2 --steps 2 --control-h1-time-weight 1e-2 --convection on --vx 1 --vy 0.5 --vz -0.75
A exact_dirichlet_conv_3d --exact-oc dirichlet-conv-3d --nx 2 --ny 2 --nz 2 --steps 2 --convection on --vx 1 --vy 0.5 --vz -0.75 --time-scheme cn
LU="-sweep_right_pc_type lu -sweep_left_pc_type lu -sweep_ay_pc_type lu -fieldsplit_y_sweep_pc_type lu"
A t11n_neumann_cn $NEU --convection on --vx 1 --vy 0.5 --vz -0.75 --pc-schur-sweep --time-scheme cn
A t10_distributed_natural $OK --convection on --vx 1 --vy 0.5 --vz -0.75 --pc-schur-sweep --time-scheme cn
A t9_production_cmdline $B --distributed-solve --control-full-domain --tracking-box 0 1 0 1 --state-bc dirichlet --pc-schur-sweep
echo "T11H1_REFUSALS pass=$pass fail=$fail"
