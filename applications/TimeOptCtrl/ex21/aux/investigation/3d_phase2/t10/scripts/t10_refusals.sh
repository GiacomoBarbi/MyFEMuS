#!/bin/bash
# T10 3D whitelist: every "R" line must be refused (invalid_argument), every "A"
# line accepted (exit 0, a converged EX20_PAR_SOLVE / exact-driver line).
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
W=${T10_WORK:-/tmp/t10_refusals}; mkdir -p $W/output; cd $W
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
B="--dimension 3 --nx 2 --ny 2 --nz 2 --steps 2"
OK="$B --distributed-solve --control-full-domain --tracking-box3d 0 1 0 1 0 1 --state-bc natural"
pass=0; fail=0
R() { name=$1; shift; out=$($X "$@" --no-paraview 2>&1 | grep -oE "what\(\): .*" | head -1)
  if [ -n "$out" ]; then pass=$((pass+1)); v=REFUSED; else fail=$((fail+1)); v="NOT REFUSED (FAIL)"; fi
  echo "R $name: $v ${out#what(): }"; }
A() { name=$1; shift; out=$($X "$@" --no-paraview 2>&1); e=$?
  line=$(echo "$out" | grep -oE "EX20_PAR_SOLVE .*reason=\S+|EX21_EXACT3D_RESULT .*reason=\S+" | grep -oE "control=\S+|reason=\S+" | tr '\n' ' ')
  if [ $e = 0 ] && echo "$line" | grep -q "reason=2"; then pass=$((pass+1)); v=ACCEPTED; else fail=$((fail+1)); v="FAILED exit=$e $(echo "$out" | grep -oE 'what\(\): .*')"; fi
  echo "A $name: $v $line"; }
# ---- still refused in 3D
R neumann_control $B --neumann-solve --control-full-domain --tracking-box3d 0 1 0 1 0 1 --state-bc natural
R dirichlet_control $B --dirichlet-solve --control-full-domain --tracking-box3d 0 1 0 1 0 1 --state-bc natural
R h1_space $OK --control-h1-length 0.3
R h1_time $OK --control-h1-time-weight 1e-3
R boundary_reduced $OK --pc-schur-boundary-reduced
R dirichlet_reduced $OK --pc-schur-dirichlet-reduced
R matching $OK --pc-schur-matching
R variable_wind $OK --convection on --wind q2-divfree
R lps $OK --convection on --vx 1 --convection-stabilization lps
R lifting_exact_case $B --exact-oc lift-exp
R math2025_preset $B --math2025-ex3
R errorfeedback_preset $B --errorfeedback-95
R literature_battery $OK --verify-stage1
R cn_without_sweep $OK --time-scheme cn
R domain_2d_syntax $OK --domain 0 2 0 1
R tracking_box_2d_ambiguous $B --distributed-solve --control-full-domain --tracking-box 0 1 0 0.5 --state-bc natural
R control_box_2d_ambiguous $B --distributed-solve --control-box 0 1 0 0.5 --tracking-box3d 0 1 0 1 0 1 --state-bc natural
R tracking_add_2d $OK --tracking-box-add 0 0.5 0 0.5
R control_add_2d $OK --control-box-add 0 0.5 0 0.5
R source_add_2d $OK --source 1 --source-box-add 0 0.5 0 0.5
R implicit_2d_tracking_default $B --distributed-solve --control-full-domain --state-bc natural
R implicit_2d_control_default $B --distributed-solve --tracking-box3d 0 1 0 1 0 1 --state-bc natural
R box3d_outside_domain $OK --control-box3d-add 0.5 1.5 0 1 0 1
R box3d_empty_y $OK --control-box3d-add 0 1 0.5 0.5 0 1
R domain3d_invalid $OK --domain3d 0 1 1 0 0 1
R box3d_in_2d --distributed-solve --nx 2 --nz 2 --steps 2 --tracking-box3d 0 1 0 1 0 1
R domain3d_in_2d --distributed-solve --nx 2 --nz 2 --steps 2 --domain3d 0 1 0 1 0 1
R exact_case_with_regions --exact-oc natural-conv-3d --nx 2 --ny 2 --nz 2 --steps 2 --control-box3d 0 0.5 0 1 0 1
R vz_in_2d --distributed-solve --nx 2 --nz 2 --steps 2 --convection on --vz 1
R ny_in_2d --distributed-solve --nx 2 --ny 2 --nz 2 --steps 2
R velocity_without_convection $OK --vx 1
R bad_dimension --dimension 4 --distributed-solve
# ---- accepted (T9 + T10 whitelist), small production solves
LU="-sweep_right_pc_type lu -sweep_left_pc_type lu -sweep_ay_pc_type lu -fieldsplit_y_sweep_pc_type lu"
export PETSC_OPTIONS="$LU"
A t9_production_cmdline $B --distributed-solve --control-full-domain --tracking-box 0 1 0 1 --state-bc dirichlet --convection on --vx 1 --vy 0.5 --vz -0.75 --pc-schur-sweep
A natural_bc $OK --convection on --vx 1 --vy 0.5 --vz -0.75 --pc-schur-sweep --time-scheme cn
A control_box_multizone $B --distributed-solve --control-box3d 0 0.5 0 0.5 0 0.5 --control-box3d-add 0.5 1 0.5 1 0.5 1 --tracking-box3d 0 0.5 0.5 1 0 0.5 --tracking-box3d-add 0.5 1 0 0.5 0.5 1 --state-bc natural --pc-schur-sweep
A sources $OK --source 1 --source-box3d-add 0 0.5 0 0.5 0 0.5 --source-box3d-add 0.5 1 0.5 1 0.5 1 --pc-schur-sweep
A nonunit_domain $B --distributed-solve --domain3d 0 2 -1 1 0 0.5 --control-full-domain --tracking-box3d 0 2 -1 1 0 0.5 --state-bc dirichlet --pc-schur-sweep
A additive_gamg_be $OK --convection on --vx 1 --vy 0.5 --vz -0.75
A exact_natural_conv_3d --exact-oc natural-conv-3d --nx 2 --ny 2 --nz 2 --steps 2 --convection on --vx 1 --vy 0.5 --vz -0.75 --pc-schur-sweep
A exact_conv_sin3_3d --exact-oc conv-sin3-3d --nx 2 --ny 2 --nz 2 --steps 2 --pc-schur-sweep
echo "T10_REFUSALS pass=$pass fail=$fail"
