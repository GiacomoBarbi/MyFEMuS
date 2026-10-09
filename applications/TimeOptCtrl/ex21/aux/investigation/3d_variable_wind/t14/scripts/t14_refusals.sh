#!/bin/bash
# T14 strict refusals / acceptances for 3D variable winds: "R" must be refused, "A" accepted (exit 0, reason 2).
# Time-dependent and state-dependent winds have no CLI and no field type (nothing can request them).
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
W=${T14_WORK:-/tmp/t14_refusals}; mkdir -p $W/output; cd $W
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
pass=0; fail=0
R() { name=$1; shift; out=$($X "$@" --no-paraview 2>&1 | grep -oE "what\(\): .*" | head -1)
  if [ -n "$out" ]; then pass=$((pass+1)); v=REFUSED; else fail=$((fail+1)); v="NOT REFUSED (FAIL)"; fi
  echo "R $name: $v ${out#what(): }"; }
A() { name=$1; shift; out=$($X "$@" --no-paraview 2>&1); e=$?
  line=$(echo "$out" | grep -oE "EX20_PAR_SOLVE .*reason=\S+|EX21_EXACT3D_RESULT .*reason=\S+" | grep -oE "control=\S+|pc=\S+|reason=\S+" | tr '\n' ' ')
  if [ $e = 0 ] && echo "$line" | grep -q "reason=2"; then pass=$((pass+1)); v=ACCEPTED; else fail=$((fail+1)); v="FAILED exit=$e $(echo "$out" | grep -oE 'what\(\): .*')"; fi
  echo "A $name: $v $line"; }
B3="--dimension 3 --nx 3 --ny 3 --nz 3 --steps 2 --nu 0.25 --alpha-u 1e-2 --target-type sin_x_minus_t --T 1"
DIST="$B3 --distributed-solve --control-full-domain --tracking-box3d 0 1 0 1 0 1 --pc-schur-sweep"
NEU="$B3 --neumann-solve --state-bc natural --tracking-box3d 0.2 0.8 0.2 0.8 0.2 0.8 --pc-schur-sweep"
DIR="$B3 --dirichlet-solve --state-bc dirichlet --tracking-box3d 0.2 0.8 0.2 0.8 0.2 0.8"
H="--convection on --wind harmonic-3d"; F="--convection on --wind affine-div-3d"
# ---- CLI / configuration
R wind3d_in_2d --nx 4 --nz 4 --steps 2 --distributed-solve --convection on --wind harmonic-3d
R wind2d_in_3d $DIST --state-bc dirichlet --convection on --wind q2-divfree
R wind2d_lp_in_3d $DIST --state-bc dirichlet --convection on --wind lp-recirculating
R wind_and_constant_vector $DIST --state-bc dirichlet $H --vx 1
R wind_and_constant_vz $DIST --state-bc dirichlet $F --vz -0.75
R wind_without_convection_on $DIST --state-bc dirichlet --wind harmonic-3d
R wind_twice $DIST --state-bc dirichlet --convection on --wind harmonic-3d --wind affine-div-3d
R wind_unknown_name $DIST --state-bc dirichlet --convection on --wind swirl-3d
R wind_unknown_name_2 $DIST --state-bc dirichlet --convection on --wind time-dependent
R wind_lps $DIST --state-bc dirichlet $H --convection-stabilization lps
R wind_simplex --dimension 3 --mesh-type tet --nx 2 --ny 2 --nz 2 --steps 2 --distributed-solve --control-full-domain --tracking-box3d 0 1 0 1 0 1 --state-bc dirichlet $H
R wind_neumann_h1 $NEU --neumann-face3d zmin $H --control-h1-length 0.3
R wind_neumann_cn_without_sweep $B3 --neumann-solve --state-bc natural --tracking-box3d 0.2 0.8 0.2 0.8 0.2 0.8 --neumann-face3d zmin $H --time-scheme cn
R wind_dirichlet_sweep $DIR --dirichlet-face3d zmin $F --pc-schur-sweep
R wind_dirichlet_matching $DIR --dirichlet-face3d zmin $F --pc-schur-matching
R wind_dirichlet_natural_state $B3 --dirichlet-solve --state-bc natural --tracking-box3d 0.2 0.8 0.2 0.8 0.2 0.8 --dirichlet-face3d zmin $F
R wind_dirichlet_tracking_touches_face $B3 --dirichlet-solve --state-bc dirichlet --tracking-box3d 0 1 0 1 0 1 --dirichlet-face3d zmin $F
# ---- exact cases
for c in conv-sin-3d natural-conv-3d neumann-conv-3d neumann-opp-3d neumann-adj-3d; do
  R exact_${c}_with_wind --exact-oc $c --nx 2 --ny 2 --nz 2 --steps 2 --pc-schur-sweep $H
done
for c in dirichlet-conv-3d dirichlet-h1-3d dirichlet-var-3d; do R exact_${c}_with_wind --exact-oc $c --nx 2 --ny 2 --nz 2 --steps 2 $F; done
R exact_dirichlet_opp_with_wind --exact-oc dirichlet-opp-3d --nx 4 --ny 4 --nz 4 --steps 2 $F
R exact_natural_var_constant_wind --exact-oc natural-var-3d --nx 2 --ny 2 --nz 2 --steps 2 --pc-schur-sweep --convection on --vx 1
R exact_natural_var_no_convection --exact-oc natural-var-3d --nx 2 --ny 2 --nz 2 --steps 2 --pc-schur-sweep
R exact_neumann_var_constant_wind --exact-oc neumann-var-3d --nx 2 --ny 2 --nz 2 --steps 2 --pc-schur-sweep --convection on --vz 1
R exact_neumann_var_opp_face_list --exact-oc neumann-var-opp-3d --nx 2 --ny 2 --nz 2 --steps 2 --pc-schur-sweep $H --neumann-face3d zmin,zmax
R exact_dirichlet_var_tangential --exact-oc dirichlet-var-3d --nx 2 --ny 2 --nz 2 --steps 2 --convection on --vx 1 --vz -0.75
R exact_dirichlet_var_no_convection --exact-oc dirichlet-var-3d --nx 2 --ny 2 --nz 2 --steps 2
R exact_dirichlet_var_h1 --exact-oc dirichlet-var-3d --nx 2 --ny 2 --nz 2 --steps 2 --convection on --vz -0.75 --control-h1-length 0.3
R exact_dirichlet_var_sweep --exact-oc dirichlet-var-3d --nx 2 --ny 2 --nz 2 --steps 2 --convection on --vz -0.75 --pc-schur-sweep
# ---- accepted
for w in harmonic-3d affine-div-3d; do
  for s in be cn; do
    A dist_${w}_$s $DIST --state-bc dirichlet --convection on --wind $w --time-scheme $s
    A nat_${w}_$s $DIST --state-bc natural --convection on --wind $w --time-scheme $s
    A neu_one_${w}_$s $NEU --neumann-face3d zmin --convection on --wind $w --time-scheme $s
    A neu_union_${w}_$s $NEU --neumann-face3d zmin,xmin --convection on --wind $w --time-scheme $s
    A dir_one_${w}_$s $DIR --dirichlet-face3d zmin --convection on --wind $w --time-scheme $s
    A dir_union_comb_${w}_$s $DIR --dirichlet-face3d zmin,zmax --convection on --wind $w --time-scheme $s --control-h1-length 0.3 --control-h1-time-weight 1e-2
  done
  A exact_conv_sin3_${w} --exact-oc conv-sin3-3d --nx 2 --ny 2 --nz 2 --steps 2 --pc-schur-sweep --convection on --wind $w
  A exact_natural_var_${w} --exact-oc natural-var-3d --nx 2 --ny 2 --nz 2 --steps 2 --pc-schur-sweep --convection on --wind $w
  A exact_neumann_var_${w} --exact-oc neumann-var-3d --nx 2 --ny 2 --nz 2 --steps 2 --pc-schur-sweep --convection on --wind $w
  A exact_neumann_var_opp_${w} --exact-oc neumann-var-opp-3d --nx 2 --ny 2 --nz 2 --steps 2 --pc-schur-sweep --convection on --wind $w
done
A exact_dirichlet_var --exact-oc dirichlet-var-3d --nx 2 --ny 2 --nz 2 --steps 2 --convection on --vz -0.75
A dir_dense_core_variable $DIR --dirichlet-face3d zmin,xmin $F --dirichlet-core-solver dense --verify-dirichlet-core
A legacy_constant_wind_dist $DIST --state-bc dirichlet --convection on --vx 1 --vy 0.5 --vz -0.75
A legacy_exact_conv_sin3_constant --exact-oc conv-sin3-3d --nx 2 --ny 2 --nz 2 --steps 2 --pc-schur-sweep --convection on --vx 1 --vy 0.5 --vz -0.75
echo "T14_REFUSALS pass=$pass fail=$fail"
