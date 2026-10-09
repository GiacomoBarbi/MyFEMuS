#!/bin/bash
# T12 whitelist: every "R" line must be refused (invalid_argument), every "A" line accepted (exit 0,
# converged reason 2).  Simplex meshes accept ONLY the distributed full-domain problem with SCHUR_SWEEP.
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
W=${T12_WORK:-/tmp/t12_refusals}; mkdir -p $W/output; cd $W
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
T2="--mesh-type tri --domain 0 1 0 1 --nx 2 --nz 2 --steps 2"
T3="--dimension 3 --mesh-type tet --nx 2 --ny 2 --nz 2 --steps 2"
D2="$T2 --distributed-solve --control-full-domain --state-bc dirichlet --tracking-box 0 1 0 1 --pc-schur-sweep"
D3="$T3 --distributed-solve --control-full-domain --state-bc dirichlet --tracking-box3d 0 1 0 1 0 1 --pc-schur-sweep"
pass=0; fail=0
R() { name=$1; shift; out=$($X "$@" --no-paraview 2>&1 | grep -oE "what\(\): .*" | head -1)
  if [ -n "$out" ]; then pass=$((pass+1)); v=REFUSED; else fail=$((fail+1)); v="NOT REFUSED (FAIL)"; fi
  echo "R $name: $v ${out#what(): }"; }
A() { name=$1; shift; out=$($X "$@" --no-paraview 2>&1); e=$?
  line=$(echo "$out" | grep -oE "EX20_PAR_SOLVE .*reason=\S+|EX21_EXACT3?D?_RESULT .*reason=\S+|EX21_VC_G4_PRODUCTION reason=\S+" | grep -oE "pc=\S+|reason=\S+" | tr '\n' ' ')
  if [ $e = 0 ] && echo "$line" | grep -q "reason=2"; then pass=$((pass+1)); v=ACCEPTED; else fail=$((fail+1)); v="FAILED exit=$e $(echo "$out" | grep -oE 'what\(\): .*')"; fi
  echo "A $name: $v $line"; }
# ---- dimension / type
R tri_in_3d --dimension 3 --mesh-type tri --distributed-solve --control-full-domain --state-bc dirichlet --tracking-box3d 0 1 0 1 0 1 --pc-schur-sweep
R tet_in_2d --mesh-type tet --distributed-solve --control-full-domain --state-bc dirichlet --pc-schur-sweep
R quad_in_3d --dimension 3 --mesh-type quad --distributed-solve --control-full-domain --state-bc dirichlet --tracking-box3d 0 1 0 1 0 1
R hex_in_2d --mesh-type hex --distributed-solve --control-full-domain --state-bc dirichlet
R bad_mesh_type --mesh-type prism --distributed-solve
R tri_in_1d --dimension 1 --mesh-type tri --distributed-solve
# ---- features not validated on simplices
for M in 2 3; do eval B=\$D$M; eval T=\$T$M
  R neumann_control_$M $T --neumann-solve $( [ $M = 3 ] && echo "--neumann-face3d zmin --tracking-box3d 0 1 0 1 0 1" || echo "--tracking-box 0 1 0 1") --state-bc natural --pc-schur-sweep
  R dirichlet_control_$M $T --dirichlet-solve $( [ $M = 3 ] && echo "--dirichlet-face3d zmin --tracking-box3d 0.2 0.8 0.2 0.8 0.4 1" || echo "--tracking-box 0.2 0.8 0.4 1") --state-bc dirichlet
  R natural_bc_$M $T --distributed-solve --control-full-domain --state-bc natural $( [ $M = 3 ] && echo "--tracking-box3d 0 1 0 1 0 1" || echo "--tracking-box 0 1 0 1") --pc-schur-sweep
  R h1_space_$M $B --control-h1-length 0.3
  R h1_time_$M $B --control-h1-time-weight 1e-2
  R partial_control_box_$M $T --distributed-solve --state-bc dirichlet --pc-schur-sweep $( [ $M = 3 ] && echo "--tracking-box3d 0 1 0 1 0 1 --control-box3d 0 0.5 0 1 0 1" || echo "--tracking-box 0 1 0 1 --control-box 0 0.5 0 1")
  R partial_tracking_box_$M $T --distributed-solve --control-full-domain --state-bc dirichlet --pc-schur-sweep $( [ $M = 3 ] && echo "--tracking-box3d 0 0.5 0 1 0 1" || echo "--tracking-box 0 0.5 0 1")
  R multizone_$M $B $( [ $M = 3 ] && echo "--control-box3d-add 0 0.5 0 0.5 0 0.5" || echo "--control-box-add 0 0.5 0 0.5")
  R source_box_$M $B --source 1 $( [ $M = 3 ] && echo "--source-box3d-add 0 0.5 0 0.5 0 0.5" || echo "--source-box-add 0 0.5 0 0.5")
  R variable_wind_$M $B --convection on --wind q2-divfree
  R lps_$M $B --convection on --vx 1 --convection-stabilization lps
  R no_sweep_additive_$M $(echo $B | sed 's/--pc-schur-sweep//')
  R matching_$M $B --pc-schur-matching
done
R boundary_reduced_2 $D2 --pc-schur-boundary-reduced
R exact_lp52_tri --exact-oc lp52 --mesh-type tri --nx 2 --nz 2 --steps 2 --pc-schur-sweep
R exact_lift_tri --exact-oc lift-const --mesh-type tri --nx 2 --nz 2 --steps 2 --pc-schur-sweep
R exact_neumann_conv_tri --exact-oc neumann-conv --mesh-type tri --nx 2 --nz 2 --steps 2 --pc-schur-sweep
R exact_natural_conv_3d_tet --exact-oc natural-conv-3d --mesh-type tet --nx 2 --ny 2 --nz 2 --steps 2 --pc-schur-sweep
R exact_dirichlet_conv_3d_tet --exact-oc dirichlet-conv-3d --mesh-type tet --nx 2 --ny 2 --nz 2 --steps 2
R exact_neumann_conv_3d_tet --exact-oc neumann-conv-3d --mesh-type tet --nx 2 --ny 2 --nz 2 --steps 2 --pc-schur-sweep
R math2025_ex3_tri --math2025-ex3 --mesh-type tri
R errorfeedback_tri --errorfeedback-95 --mesh-type tri
R verify_stage1_tri $D2 --verify-stage1
# ---- accepted
LU="-sweep_right_pc_type lu -sweep_left_pc_type lu -sweep_ay_pc_type lu -fieldsplit_y_sweep_pc_type lu"
export PETSC_OPTIONS="$LU"
A tri_heat_be $D2
A tri_wind_cn $D2 --convection on --vx 1 --vy 0.5 --time-scheme cn
A tri_nonunit_domain --mesh-type tri --domain 0 2 -1 0.5 --nx 4 --nz 2 --steps 2 --distributed-solve --control-full-domain --state-bc dirichlet --tracking-box 0 2 -1 0.5 --pc-schur-sweep
A tet_heat_be $D3
A tet_wind_cn $D3 --convection on --vx 1 --vy 0.5 --vz -0.75 --time-scheme cn
A tet_nonunit_domain --dimension 3 --mesh-type tet --domain3d 0 2 -1 1 0 0.5 --nx 3 --ny 2 --nz 2 --steps 2 --distributed-solve --control-full-domain --state-bc dirichlet --tracking-box3d 0 2 -1 1 0 0.5 --pc-schur-sweep
A exact_conv_sin3_tri --exact-oc conv-sin3 --mesh-type tri --nx 2 --nz 2 --steps 2 --pc-schur-sweep
A exact_conv_sin_tri --exact-oc conv-sin --mesh-type tri --nx 2 --nz 2 --steps 2 --pc-schur-sweep
A exact_conv_sin3_3d_tet --exact-oc conv-sin3-3d --mesh-type tet --nx 2 --ny 2 --nz 2 --steps 2 --pc-schur-sweep
A exact_conv_sin_3d_tet --exact-oc conv-sin-3d --mesh-type tet --nx 2 --ny 2 --nz 2 --steps 2 --pc-schur-sweep
A verify_convection_tet --verify-convection --exact-oc conv-sin3-3d --mesh-type tet --nx 2 --ny 2 --nz 2 --steps 2 --pc-schur-sweep
A explicit_quad_default_path --mesh-type quad --domain 0 1 0 1 --nx 2 --nz 2 --steps 2 --distributed-solve --control-full-domain --state-bc dirichlet --tracking-box 0 1 0 1 --pc-schur-sweep
A explicit_hex_default_path --dimension 3 --mesh-type hex --nx 2 --ny 2 --nz 2 --steps 2 --distributed-solve --control-full-domain --state-bc dirichlet --tracking-box3d 0 1 0 1 0 1 --pc-schur-sweep
unset PETSC_OPTIONS
A tet_gamg_be $D3 --convection on --vx 1 --vy 0.5 --vz -0.75
A tri_gamg_cn $D2 --convection on --vx 1 --vy 0.5 --time-scheme cn
echo "T12_REFUSALS pass=$pass fail=$fail"
