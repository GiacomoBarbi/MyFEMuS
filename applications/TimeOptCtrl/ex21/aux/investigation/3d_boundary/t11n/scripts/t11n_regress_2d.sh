#!/bin/bash
# (T11N copy of 3d_phase2/t10/scripts/t10_regress_2d.sh: logs into 3d_boundary/t11n/logs.)
# (T10 copy of 3d_phase1/t9/scripts/t9_regress_*.sh: logs into 3d_phase2/t10/logs; OLD = the T9 build, 8ed57d629.)
# T9 mandatory 2D regression: OLD = the frozen 2D binary (commit e0410efe3,
# sha256 8eae9680) vs the current (3D-enabled) binary.  Built on
# dirichlet_core/t8d_final/scripts/regress_final.sh:
# (1) the T6 suite (ex20 goldens 61, Math2025 Ex3/Ex4/Ex5, EF 9.5, T0-T4, T3/T4 runs);
# (2) old vs new on every EX20_/EX21_ line (timings stripped, incl. the T8D
#     core_seconds/shell_seconds): Dirichlet heat (all regularisations, dense AND the
#     iterative default), Neumann, T7N, T7D, T7F, T6 LPS, natural-BC gates, the
#     T8D --verify-dirichlet-core, and the zero-wind ex20/ex21 parity
#     configurations (whose --dump-json files must be byte-identical);
# (3) the 2D restrictions, plus the T9 3D whitelist refusals.   WORK=<scratch> OLD=<binary>
set -u
H=$(cd "$(dirname "$0")/.." && pwd)
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
W=${WORK:?set WORK}; OLD=${OLD:?set OLD}; mkdir -p $W/x/output
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
echo "== (1) T6 suite (2D)"; EX21_BIN=$X WORK=$W/t6 OLD=$OLD bash $H/scripts/t11n_regress_suite.sh 2>&1 | sed -n '1,200p'
echo "== (2) 2D old (frozen) vs new"
strip() { grep -E "^EX2[01]_" | grep -v "^EX21_SOLVE_REJECTED" | sed -E 's/ (solve_wall|total_wall|setup_s|solve_s|t_operators|t_pcsetup|t_kspsolve|t_total|setup_seconds|core_seconds|shell_seconds)=[^ ]*//g; s/ accepted=[01] status=[a-z_]*//'; }
cmpx() { name=$1; opts=$2; shift 2; cd $W/x
  PETSC_OPTIONS="$opts" $OLD "$@" --no-paraview 2>/dev/null | strip > $name.old; oe=${PIPESTATUS[0]}
  PETSC_OPTIONS="$opts" $X "$@" --no-paraview > $name.raw 2>/dev/null; ne=$?; strip < $name.raw > $name.new
  if diff -q $name.old $name.new > /dev/null; then v=IDENTICAL; else v="DIFFERS ($(diff $name.old $name.new | grep -c '^[<>]') lines)"; fi
  echo "$name: $v ($(wc -l < $name.new) lines) exit old=$oe new=$ne"; }
LU="-sweep_right_pc_type lu -sweep_left_pc_type lu -sweep_ay_pc_type lu -fieldsplit_y_sweep_pc_type lu"
for s in be cn; do
  cmpx dirichlet_heat_l2_$s "" --dirichlet-solve --nx 8 --nz 8 --steps 16 --time-scheme $s --dirichlet-core-solver dense
  cmpx dirichlet_heat_h1s_$s "" --dirichlet-solve --nx 8 --nz 8 --steps 16 --time-scheme $s --control-h1-length 0.2 --dirichlet-core-solver dense
  cmpx dirichlet_heat_h1t_$s "" --dirichlet-solve --nx 8 --nz 8 --steps 16 --time-scheme $s --control-h1-time-weight 1e-3 --dirichlet-core-solver dense
  cmpx dirichlet_heat_h1st_$s "" --dirichlet-solve --nx 8 --nz 8 --steps 16 --time-scheme $s --control-h1-length 0.2 --control-h1-time-weight 1e-3 --dirichlet-core-solver dense
  cmpx dirichlet_heat_sinx_nx16_$s "" --dirichlet-solve --nx 16 --nz 16 --steps 32 --time-scheme $s --nu 0.25 --alpha-u 1e-2 --target-type sin_x_minus_t --dirichlet-core-solver dense
  cmpx neumann_heat_additive_$s "" --neumann-solve --nx 8 --nz 8 --steps 16 --time-scheme $s --alpha-u 1e-2
  cmpx neumann_heat_sweep_$s "" --neumann-solve --nx 8 --nz 8 --steps 16 --time-scheme $s --pc-schur-sweep
  cmpx neumann_heat_reduced_$s "" --neumann-solve --nx 8 --nz 8 --steps 16 --time-scheme $s --pc-schur-boundary-reduced
  cmpx t7n_neumann_conv_exact_$s "" --exact-oc neumann-conv --nx 16 --nz 16 --steps 32 --time-scheme $s --pc-schur-sweep
  cmpx t7n_neumann_conv_prod_$s "" --neumann-solve --nx 8 --nz 8 --steps 16 --time-scheme $s --convection on --vx 1 --vy 0.5 --pc-schur-sweep --alpha-u 1e-2
done
for s in be cn; do
  # T7D: Dirichlet control + convection (exact OC, production, gates)
  cmpx t7d_dirichlet_conv_exact_$s "" --exact-oc dirichlet-conv --nx 16 --nz 16 --steps 32 --time-scheme $s
  cmpx t7d_dirichlet_conv_h1s_exact_$s "" --exact-oc dirichlet-conv --nx 16 --nz 16 --steps 32 --time-scheme $s --control-h1-length 0.3
  cmpx t7d_dirichlet_conv_prod_h1st_$s "" --dirichlet-solve --nx 16 --nz 16 --steps 32 --time-scheme $s --nu 0.25 --alpha-u 1e-2 --target-type sin_x_minus_t --convection on --vx 1 --vy 0.5 --control-h1-length 0.3 --control-h1-time-weight 1e-3 --dirichlet-core-solver dense
  cmpx t7d_dirichlet_gates_$s "" --verify-convection --dirichlet-solve --nx 8 --nz 8 --steps 8 --time-scheme $s --convection on --vx 1 --vy 0.5 --nu 0.25 --alpha-u 1e-2 --target-type sin_x_minus_t --dirichlet-core-solver dense
  # pre-existing multi-zone / natural-BC smoke configurations (feature audit)
  cmpx audit_multizone_natural_$s "$LU" --verify-convection --distributed-solve --pc-schur-sweep --nx 8 --nz 8 --steps 8 --time-scheme $s --state-bc natural --convection on --vx 2 --vy 1 --nu 0.5 --target-type sin_x_minus_t --alpha-u 1e-3 --control-box 0.1 0.4 0.1 0.4 --control-box-add 0.6 0.9 0.6 0.9 --tracking-box 0.5 0.9 0.1 0.5 --tracking-box-add 0.1 0.5 0.5 0.9
  cmpx audit_multizone_dirichlet_$s "$LU" --verify-convection --distributed-solve --pc-schur-sweep --nx 8 --nz 8 --steps 8 --time-scheme $s --state-bc dirichlet --convection on --vx 2 --vy 1 --nu 0.5 --target-type sin_x_minus_t --alpha-u 1e-3 --control-box 0.1 0.4 0.1 0.4 --control-box-add 0.6 0.9 0.6 0.9 --tracking-box 0.5 0.9 0.1 0.5 --tracking-box-add 0.1 0.5 0.5 0.9
  # T5/T6 exact-OC cases (distributed, constant wind)
  cmpx conv_sin3_exact_$s "$LU" --exact-oc conv-sin3 --nx 16 --nz 16 --steps 32 --time-scheme $s --pc-schur-sweep
  cmpx lift_exp_exact_$s "$LU" --exact-oc lift-exp --nx 16 --nz 16 --steps 32 --time-scheme $s --pc-schur-sweep
done
for s in be cn; do
  # T7F: natural-conv exact OC, multi-zone + two sources production
  cmpx t7f_natural_conv_exact_$s "$LU" --exact-oc natural-conv --nx 16 --nz 16 --steps 32 --time-scheme $s --pc-schur-sweep
  cmpx t7f_mz_src_prod_$s "$LU" --distributed-solve --pc-schur-sweep --nx 16 --nz 16 --steps 16 --time-scheme $s --nu 0.25 --alpha-u 1e-2 --target-type sin_x_minus_t --state-bc natural --convection on --vx 1 --vy 0.5 --control-box 0.125 0.375 0.125 0.375 --control-box-add 0.625 0.875 0.625 0.875 --tracking-box 0.5 0.875 0.125 0.375 --tracking-box-add 0.125 0.5 0.5 0.875 --source 1 --source-box-add 0 0.25 0.75 1 --source-box-add 0.75 1 0 0.25
  # Dirichlet dense core, larger reduced dimension, all regularisations (convection)
  cmpx t7d_dense_core_h1s_nx32_$s "" --dirichlet-solve --nx 32 --nz 32 --steps 32 --time-scheme $s --nu 0.25 --alpha-u 1e-2 --target-type sin_x_minus_t --convection on --vx 1 --vy 0.5 --control-h1-length 0.3 --dirichlet-core-solver dense
  cmpx t7d_dense_core_h1t_nx32_$s "" --dirichlet-solve --nx 32 --nz 32 --steps 32 --time-scheme $s --nu 0.25 --alpha-u 1e-2 --target-type sin_x_minus_t --convection on --vx 1 --vy 0.5 --control-h1-time-weight 1e-3 --dirichlet-core-solver dense
  cmpx t7d_dense_core_heat_h1st_nx32_$s "" --dirichlet-solve --nx 32 --nz 32 --steps 32 --time-scheme $s --control-h1-length 0.3 --control-h1-time-weight 1e-3 --dirichlet-core-solver dense
done
cmpx t6_lps_lp52_eps500_nx16 "$LU" --exact-oc lp52 --exact-oc-nu 0.002 --nx 16 --nz 16 --steps 32 --time-scheme cn --distributed-solve --pc-schur-sweep --exact-oc-production-tol --convection-stabilization lps
cmpx distributed_natural_verify_cn "$LU" --verify-convection --distributed-solve --pc-schur-sweep --nx 8 --nz 8 --steps 8 --convection on --vx 2 --vy 1 --target-type sin_x_minus_t --alpha-u 1e-3 --state-bc natural --nu 0.5 --time-scheme cn
cmpx neumann_natural_verify_cn "$LU" --verify-convection --neumann-solve --pc-schur-sweep --nx 8 --nz 8 --steps 8 --convection on --vx 1 --vy 0.5 --target-type sin_x_minus_t --alpha-u 1e-2 --nu 0.25 --time-scheme cn

# T8D iterative default + verify (2D)
for s in be cn; do
  cmpx t8d_iter_default_h1st_$s "" --dirichlet-solve --nx 16 --nz 16 --steps 16 --time-scheme $s --nu 0.25 --alpha-u 1e-2 --target-type sin_x_minus_t --convection on --vx 1 --vy 0.5 --control-h1-length 0.3 --control-h1-time-weight 1e-3
  cmpx t8d_iter_default_heat_l2_$s "" --dirichlet-solve --nx 16 --nz 16 --steps 16 --time-scheme $s
  cmpx t8d_verify_core_$s "" --dirichlet-solve --verify-dirichlet-core --nx 8 --nz 8 --steps 8 --time-scheme $s --convection on --vx 1 --vy 0.5 --control-h1-length 0.3
done
# zero-wind ex20/ex21 parity configurations (T8D finalisation): JSON byte-compare old vs new
jcmp() { name=$1; opts=$2; shift 2; cd $W/x
  PETSC_OPTIONS="$opts" $OLD "$@" --nx 8 --nz 8 --steps 16 --no-paraview --dump-json $W/x/$name.old.json > /dev/null 2>&1
  PETSC_OPTIONS="$opts" $X "$@" --nx 8 --nz 8 --steps 16 --no-paraview --dump-json $W/x/$name.new.json > /dev/null 2>&1
  sed -i -E 's/"json":"[^"]*"//' $W/x/$name.old.json $W/x/$name.new.json
  if cmp -s $W/x/$name.old.json $W/x/$name.new.json; then v=JSON_IDENTICAL; else v=JSON_DIFFERS; fi; echo "$name: $v"; }
for s in be cn; do for wz in "off:" "w0:--convection on --vx 0 --vy 0"; do w=${wz%%:*}; wa=${wz#*:}
  for ctl in distributed neumann; do
    jcmp parity_${ctl}_${s}_${w}_default "" --$ctl-solve --time-scheme $s --alpha-u 1e-2 $wa
    jcmp parity_${ctl}_${s}_${w}_sweeplu "$LU" --$ctl-solve --time-scheme $s --alpha-u 1e-2 --pc-schur-sweep $wa
    jcmp parity_${ctl}_${s}_${w}_matching "" --$ctl-solve --time-scheme $s --alpha-u 1e-2 --pc-schur-matching $wa
  done
  jcmp parity_neumann_${s}_${w}_boundaryreduced "" --neumann-solve --time-scheme $s --alpha-u 1e-2 --pc-schur-boundary-reduced $wa
  jcmp parity_dirichlet_${s}_${w}_default "" --dirichlet-solve --time-scheme $s --alpha-u 1e-2 $wa
  jcmp parity_dirichlet_h1st_${s}_${w}_dense "" --dirichlet-solve --time-scheme $s --alpha-u 1e-2 --control-h1-length 0.3 --control-h1-time-weight 1e-3 --dirichlet-core-solver dense $wa
done; done
echo "== (3) restrictions (new binary)"
ref() { name=$1; shift; out=$($X "$@" --no-paraview 2>&1 | grep -oE "what\(\): .*|invalid_argument.*" | head -1); echo "$name: ${out:-NOT REFUSED}"; }
cd $W/x
ref convection+dirichlet_ACCEPTED_NOW --dirichlet-solve --nx 4 --nz 4 --steps 4 --convection on --vx 1 --vy 0.5
ref convection+dirichlet+sweep --dirichlet-solve --nx 4 --nz 4 --steps 4 --convection on --vx 1 --vy 0.5 --pc-schur-sweep
ref convection+dirichlet+matching --dirichlet-solve --nx 4 --nz 4 --steps 4 --convection on --vx 1 --vy 0.5 --pc-schur-matching
ref convection+dirichlet+boundary_reduced --dirichlet-solve --nx 4 --nz 4 --steps 4 --convection on --vx 1 --vy 0.5 --pc-schur-boundary-reduced
ref convection+boundary_reduced --neumann-solve --nx 4 --nz 4 --steps 4 --convection on --vx 1 --vy 0.5 --pc-schur-boundary-reduced
ref convection+h1_time_neumann --neumann-solve --nx 4 --nz 4 --steps 4 --convection on --vx 1 --vy 0.5 --control-h1-time-weight 1e-3
ref convection+verify_boundary_reduced --neumann-solve --nx 4 --nz 4 --steps 4 --convection on --vx 1 --vy 0.5 --verify-boundary-reduced
ref dirichlet_conv_exact_vx_nonzero --exact-oc dirichlet-conv --exact-oc-velocity 1 0.5 --nx 4 --nz 4 --steps 4
ref natural_conv_exact_wind_field --exact-oc natural-conv --wind q2-divfree --nx 4 --nz 4 --steps 4
ref source_box_outside_domain --distributed-solve --nx 4 --nz 4 --steps 4 --source 1 --source-box-add 0.5 1.5 0 1
ref t8d_iterative_core_needs_dirichlet --distributed-solve --nx 4 --nz 4 --steps 4 --dirichlet-core-solver iterative
ref t8d_verify_core_needs_dirichlet --neumann-solve --nx 4 --nz 4 --steps 4 --verify-dirichlet-core
ref t8d_bad_core_solver --dirichlet-solve --nx 4 --nz 4 --steps 4 --dirichlet-core-solver lu
ref t8d_bad_core_rtol --dirichlet-solve --nx 4 --nz 4 --steps 4 --dirichlet-core-solver iterative --dirichlet-core-rtol 0
ref t8f_verify_with_explicit_iterative --dirichlet-solve --nx 4 --nz 4 --steps 4 --verify-dirichlet-core --dirichlet-core-solver iterative
ref t8f_default_core_on_distributed_is_inert --distributed-solve --nx 4 --nz 4 --steps 4
ref dirichlet_conv_exact_gamma --exact-oc dirichlet-conv --control-h1-time-weight 1e-3 --nx 4 --nz 4 --steps 4
# T9: 2D must still refuse vz; 3D whitelist refusals ("refuse, don't reinterpret")
ref t9_2d_vz_nonzero --distributed-solve --nx 4 --nz 4 --steps 4 --convection on --vx 1 --vz 1
D3="--dimension 3 --nx 2 --ny 2 --nz 2 --steps 2"
ref t9_3d_neumann_control --neumann-solve $D3
ref t9_3d_dirichlet_control --dirichlet-solve $D3
ref t9_3d_natural_state_bc --distributed-solve $D3 --state-bc natural
ref t9_3d_h1_space --distributed-solve $D3 --control-h1-length 0.3
ref t9_3d_h1_time --distributed-solve $D3 --control-h1-time-weight 1e-3
ref t9_3d_control_box --distributed-solve $D3 --control-box 0 0.5 0 0.5
ref t9_3d_multizone --distributed-solve $D3 --control-box-add 0 0.5 0 0.5
ref t9_3d_source_box --distributed-solve $D3 --source 1 --source-box-add 0 0.5 0 0.5
ref t9_3d_wind_field --distributed-solve $D3 --convection on --wind q2-divfree
ref t9_3d_lps --distributed-solve $D3 --convection on --vx 1 --convection-stabilization lps
ref t9_3d_domain --distributed-solve $D3 --domain 0 2 0 1
ref t9_3d_cn_without_sweep --distributed-solve $D3 --time-scheme cn
ref t9_3d_matching --distributed-solve $D3 --pc-schur-matching
ref t9_3d_2d_exact_case --exact-oc conv-sin3 $D3
ref t9_3d_math2025 --dimension 3 --math2025-ex3
ref t9_3d_velocity_without_convection --distributed-solve $D3 --vx 1
ref t9_bad_dimension --distributed-solve --dimension 4 --nx 2 --nz 2 --steps 2
echo T11N_REGRESS_2D_DONE
