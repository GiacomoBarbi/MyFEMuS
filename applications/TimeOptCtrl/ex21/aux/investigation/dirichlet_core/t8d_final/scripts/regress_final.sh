#!/bin/bash
# T8D finalisation regression, OLD = the T8D binary (sha a2e6f680, dense default).
# Dirichlet old-vs-new comparisons pass --dirichlet-core-solver dense to BOTH binaries
# (explicit dense must reproduce the old default); the new default (iterative) is
# compared separately by default_change.{sh,py}.
# (1) the full T6 suite (ex20 goldens 61, Math2025 Ex3/Ex4/Ex5, EF 9.5, T0-T4);
# (2) old-vs-new on every EX20_/EX21_ line (wall times stripped): Dirichlet heat
#     plain / H1-space / H1-time / combined (BE, CN) -- must be IDENTICAL, the
#     heat path of B_D^T and of the reduced solver is untouched; Neumann heat;
#     T7N neumann-conv exact OC; T6 LPS; distributed natural-BC gate run;
# (3) restrictions: what must still be refused, and the one intended change
#     (convection + Dirichlet control now accepted).      WORK=<scratch> OLD=<binary>
set -u
H=$(cd "$(dirname "$0")/.." && pwd)
X=/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21
W=${WORK:?set WORK}; OLD=${OLD:?set OLD}; mkdir -p $W/x/output
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
echo "== (1) T6 suite"; WORK=$W/t6 OLD=$OLD bash $H/scripts/regress_suite.sh 2>&1 | sed -n '1,200p'
echo "== (2) T8D-final old vs new"
strip() { grep -E "^EX2[01]_" | grep -v "^EX21_SOLVE_REJECTED" | sed -E 's/ (solve_wall|total_wall|setup_s|solve_s|t_operators|t_pcsetup|t_kspsolve|t_total|setup_seconds)=[^ ]*//g; s/ accepted=[01] status=[a-z_]*//'; }
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
echo T8D_FINAL_REGRESS_DONE
