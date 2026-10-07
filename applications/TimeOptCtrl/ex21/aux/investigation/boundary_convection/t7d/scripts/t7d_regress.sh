#!/bin/bash
# T7D regression, OLD = the T7N final binary (sha 1d772913).
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
echo "== (2) T7D old vs new"
strip() { grep -E "^EX2[01]_" | grep -v "^EX21_SOLVE_REJECTED" | sed -E 's/ (solve_wall|total_wall|setup_s|solve_s|t_operators|t_pcsetup|t_kspsolve|t_total|setup_seconds)=[^ ]*//g; s/ accepted=[01] status=[a-z_]*//'; }
cmpx() { name=$1; opts=$2; shift 2; cd $W/x
  PETSC_OPTIONS="$opts" $OLD "$@" --no-paraview 2>/dev/null | strip > $name.old; oe=${PIPESTATUS[0]}
  PETSC_OPTIONS="$opts" $X "$@" --no-paraview > $name.raw 2>/dev/null; ne=$?; strip < $name.raw > $name.new
  if diff -q $name.old $name.new > /dev/null; then v=IDENTICAL; else v="DIFFERS ($(diff $name.old $name.new | grep -c '^[<>]') lines)"; fi
  echo "$name: $v ($(wc -l < $name.new) lines) exit old=$oe new=$ne"; }
LU="-sweep_right_pc_type lu -sweep_left_pc_type lu -sweep_ay_pc_type lu -fieldsplit_y_sweep_pc_type lu"
for s in be cn; do
  cmpx dirichlet_heat_l2_$s "" --dirichlet-solve --nx 8 --nz 8 --steps 16 --time-scheme $s
  cmpx dirichlet_heat_h1s_$s "" --dirichlet-solve --nx 8 --nz 8 --steps 16 --time-scheme $s --control-h1-length 0.2
  cmpx dirichlet_heat_h1t_$s "" --dirichlet-solve --nx 8 --nz 8 --steps 16 --time-scheme $s --control-h1-time-weight 1e-3
  cmpx dirichlet_heat_h1st_$s "" --dirichlet-solve --nx 8 --nz 8 --steps 16 --time-scheme $s --control-h1-length 0.2 --control-h1-time-weight 1e-3
  cmpx dirichlet_heat_sinx_nx16_$s "" --dirichlet-solve --nx 16 --nz 16 --steps 32 --time-scheme $s --nu 0.25 --alpha-u 1e-2 --target-type sin_x_minus_t
  cmpx neumann_heat_additive_$s "" --neumann-solve --nx 8 --nz 8 --steps 16 --time-scheme $s --alpha-u 1e-2
  cmpx neumann_heat_sweep_$s "" --neumann-solve --nx 8 --nz 8 --steps 16 --time-scheme $s --pc-schur-sweep
  cmpx neumann_heat_reduced_$s "" --neumann-solve --nx 8 --nz 8 --steps 16 --time-scheme $s --pc-schur-boundary-reduced
  cmpx t7n_neumann_conv_exact_$s "" --exact-oc neumann-conv --nx 16 --nz 16 --steps 32 --time-scheme $s --pc-schur-sweep
  cmpx t7n_neumann_conv_prod_$s "" --neumann-solve --nx 8 --nz 8 --steps 16 --time-scheme $s --convection on --vx 1 --vy 0.5 --pc-schur-sweep --alpha-u 1e-2
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
ref dirichlet_conv_exact_gamma --exact-oc dirichlet-conv --control-h1-time-weight 1e-3 --nx 4 --nz 4 --steps 4
echo T7D_REGRESS_DONE
