#!/bin/bash
# T11N 3D whitelist: every "R" line must be refused (invalid_argument), every "A"
# line accepted (exit 0, converged).  Extends the T10 battery with Neumann control.
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
W=${T11N_WORK:-/tmp/t11n_refusals}; mkdir -p $W/output; cd $W
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
B="--dimension 3 --nx 2 --ny 2 --nz 2 --steps 2"
NEU="$B --neumann-solve --neumann-face3d zmin --tracking-box3d 0 1 0 1 0 1 --state-bc natural"
OK="$B --distributed-solve --control-full-domain --tracking-box3d 0 1 0 1 0 1 --state-bc natural"
pass=0; fail=0
R() { name=$1; shift; out=$($X "$@" --no-paraview 2>&1 | grep -oE "what\(\): .*" | head -1)
  if [ -n "$out" ]; then pass=$((pass+1)); v=REFUSED; else fail=$((fail+1)); v="NOT REFUSED (FAIL)"; fi
  echo "R $name: $v ${out#what(): }"; }
A() { name=$1; shift; out=$($X "$@" --no-paraview 2>&1); e=$?
  line=$(echo "$out" | grep -oE "EX20_PAR_SOLVE .*reason=\S+|EX21_EXACT3D_RESULT .*reason=\S+" | grep -oE "control=\S+|reason=\S+" | tr '\n' ' ')
  if [ $e = 0 ] && echo "$line" | grep -q "reason=2"; then pass=$((pass+1)); v=ACCEPTED; else fail=$((fail+1)); v="FAILED exit=$e $(echo "$out" | grep -oE 'what\(\): .*')"; fi
  echo "A $name: $v $line"; }
# ---- T11N: Neumann-specific refusals
R dirichlet_control $B --dirichlet-solve --control-full-domain --tracking-box3d 0 1 0 1 0 1 --state-bc natural
R neumann_without_face $B --neumann-solve --tracking-box3d 0 1 0 1 0 1 --state-bc natural
R neumann_bad_face_name $B --neumann-solve --neumann-face3d bottom --tracking-box3d 0 1 0 1 0 1 --state-bc natural
R implicit_control_type $B --neumann-face3d zmin --tracking-box3d 0 1 0 1 0 1 --state-bc natural
R neumann_face_with_distributed $OK --neumann-face3d zmin
R neumann_dirichlet_state $B --neumann-solve --neumann-face3d zmin --tracking-box3d 0 1 0 1 0 1 --state-bc dirichlet
R neumann_control_box $NEU --control-box3d 0 1 0 1 0 0.5
R neumann_control_full_domain $NEU --control-full-domain
R neumann_partial_face_or_union $NEU --control-box3d-add 0 0.5 0 0.5 0 0.1
R neumann_h1_space $NEU --control-h1-length 0.3
R neumann_h1_time $NEU --control-h1-time-weight 1e-3
R neumann_h1_combined $NEU --control-h1-length 0.3 --control-h1-time-weight 1e-3
R neumann_boundary_reduced_conv $NEU --convection on --vx 1 --pc-schur-boundary-reduced
R neumann_boundary_reduced_heat $NEU --pc-schur-boundary-reduced
R neumann_matching $NEU --pc-schur-matching
R neumann_cn_without_sweep $NEU --time-scheme cn
R neumann_variable_wind $NEU --convection on --wind q2-divfree
R neumann_lps $NEU --convection on --vx 1 --convection-stabilization lps
R neumann_face3d_in_2d --neumann-solve --nx 2 --nz 2 --steps 2 --neumann-face3d zmin
R exact_case_with_face --exact-oc neumann-conv-3d --nx 2 --ny 2 --nz 2 --steps 2 --neumann-face3d xmax
R exact_case_with_region --exact-oc neumann-conv-3d --nx 2 --ny 2 --nz 2 --steps 2 --tracking-box3d 0 1 0 1 0 1
# ---- carried over from T10
R lps_distributed $OK --convection on --vx 1 --convection-stabilization lps
R variable_wind_distributed $OK --convection on --wind q2-divfree
R math2025_preset $B --math2025-ex3
R domain_2d_syntax $OK --domain 0 2 0 1
R implicit_2d_tracking_default $B --distributed-solve --control-full-domain --state-bc natural
R bad_dimension --dimension 4 --distributed-solve
# ---- accepted
LU="-sweep_right_pc_type lu -sweep_left_pc_type lu -sweep_ay_pc_type lu -fieldsplit_y_sweep_pc_type lu"
export PETSC_OPTIONS="$LU"
for f in xmin xmax ymin ymax zmin zmax; do
  A neumann_face_$f $B --neumann-solve --neumann-face3d $f --tracking-box3d 0 1 0 1 0 1 --state-bc natural --convection on --vx 1 --vy 0.5 --vz -0.75 --pc-schur-sweep --time-scheme cn
done
A neumann_tracking_box_nonunit $B --neumann-solve --neumann-face3d ymax --domain3d 0 2 -1 1 0 0.5 --tracking-box3d 0 2 0 1 0 0.5 --state-bc natural --pc-schur-sweep
A neumann_heat_be $NEU --pc-schur-sweep
A exact_neumann_conv_3d --exact-oc neumann-conv-3d --nx 2 --ny 2 --nz 2 --steps 2 --convection on --vx 1 --vy 0.5 --vz -0.75 --pc-schur-sweep
A t10_distributed_natural $OK --convection on --vx 1 --vy 0.5 --vz -0.75 --pc-schur-sweep --time-scheme cn
A t9_production_cmdline $B --distributed-solve --control-full-domain --tracking-box 0 1 0 1 --state-bc dirichlet --pc-schur-sweep
echo "T11N_REFUSALS pass=$pass fail=$fail"
