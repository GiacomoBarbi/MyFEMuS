#!/bin/bash
# T9 build sanitisation: minimal regression of the REBUILT shared-tree ex21 (X)
# against the frozen 2D binary (OLD, e0410efe3) and the private clean T9 build
# (P, t9_clean_build.sh = the scientific reference).   WORK=<dir> OLD=<bin> P=<bin>
set -u
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
W=${WORK:?}; OLD=${OLD:?}; P=${P:?}; mkdir -p $W/output; cd $W
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
LU="-sweep_right_pc_type lu -sweep_left_pc_type lu -sweep_ay_pc_type lu -fieldsplit_y_sweep_pc_type lu"
strip() { grep -E "^EX2[01]_" | grep -v "^EX21_SOLVE_REJECTED" | sed -E 's/ (solve_wall|total_wall|setup_s|solve_s|t_operators|t_pcsetup|t_kspsolve|t_total|setup_seconds|core_seconds|shell_seconds)=[^ ]*//g'; }
cmp2() { name=$1 ref=$2 opts=$3; shift 3    # stdout EX2x_ lines, ref binary vs X
  PETSC_OPTIONS="$opts" $ref "$@" --no-paraview 2>/dev/null | strip > $name.ref; PETSC_OPTIONS="$opts" $X "$@" --no-paraview > $name.raw 2>&1; e=$?
  strip < $name.raw > $name.new
  if diff -q $name.ref $name.new >/dev/null; then v=IDENTICAL; else v="DIFFERS ($(diff $name.ref $name.new | grep -c '^[<>]') lines)"; fi
  echo "$name: $v ($(wc -l < $name.new) lines, exit $e)"; }
jcmp() { name=$1 ref=$2 opts=$3; shift 3   # --dump-json byte compare
  PETSC_OPTIONS="$opts" $ref "$@" --no-paraview --dump-json $W/$name.ref.json >/dev/null 2>&1
  PETSC_OPTIONS="$opts" $X "$@" --no-paraview --dump-json $W/$name.new.json >/dev/null 2>&1
  sed -i -E 's/"json":"[^"]*"//' $W/$name.ref.json $W/$name.new.json
  cmp -s $W/$name.ref.json $W/$name.new.json && echo "$name: JSON_IDENTICAL" || echo "$name: JSON_DIFFERS"; }
echo "== B. 2D zero-wind parity cases vs frozen 2D (JSON)"
jcmp B_parity_distributed_cn_w0_sweeplu $OLD "$LU" --distributed-solve --time-scheme cn --alpha-u 1e-2 --pc-schur-sweep --convection on --vx 0 --vy 0 --nx 8 --nz 8 --steps 16
jcmp B_parity_neumann_be_off_default $OLD "" --neumann-solve --time-scheme be --alpha-u 1e-2 --nx 8 --nz 8 --steps 16
echo "== C. 2D Dirichlet, iterative core default, vs frozen 2D"
cmp2 C_dirichlet_iter_default_h1st_cn $OLD "" --dirichlet-solve --nx 16 --nz 16 --steps 16 --time-scheme cn --nu 0.25 --alpha-u 1e-2 --target-type sin_x_minus_t --convection on --vx 1 --vy 0.5 --control-h1-length 0.3 --control-h1-time-weight 1e-3
grep -oE "EX21_T8D_CORE_STATS mode=\S+|EX20_PAR_SOLVE .*" C_dirichlet_iter_default_h1st_cn.raw | grep -oE "mode=\S+|reason=\S+|outer_iterations=\S+|true_residual=\S+" | tr '\n' ' '; echo
echo "== D/E. 3D gates (heat n=2 BE; wind A n=3 CN) vs private clean build"
G="-ksp_rtol 1e-12"
cmp2 D_gate_heat_be_n2 $P "$LU $G" --verify-convection --exact-oc conv-sin-3d --nx 2 --ny 2 --nz 2 --steps 4 --time-scheme be --pc-schur-sweep --convection off
cmp2 E_gate_wA_cn_n3 $P "$LU $G" --verify-convection --exact-oc conv-sin-3d --nx 3 --ny 3 --nz 3 --steps 4 --time-scheme cn --pc-schur-sweep --convection on --vx 1 --vy 0.5 --vz -0.75
for t in D_gate_heat_be_n2 E_gate_wA_cn_n3; do echo "$t key gates:"; grep -E "G2|G3|G4|G5|G7|G6|transpose|true_residual|PASS|FAIL" $t.new | cut -c1-160 | head -12; done
echo "== G. exact 3D case (sin3, wind A, CN n=4 N=16) and operator dump vs private clean build"
cmp2 G_exact3d_sin3_wA_cn_n4 $P "$LU" --exact-oc conv-sin3-3d --nx 4 --ny 4 --nz 4 --steps 16 --time-scheme cn --pc-schur-sweep --convection on --vx 1 --vy 0.5 --vz -0.75
b="--dimension 3 --distributed-solve --control-full-domain --tracking-box 0 1 0 1 --state-bc dirichlet --nx 3 --ny 3 --nz 3 --steps 2 --convection on --vx 1 --vy 0.5 --vz -0.75 --no-paraview"
$P $b --dump-operators ref >/dev/null 2>&1; $X $b --dump-operators new >/dev/null 2>&1
for f in ref_*; do t=${f#ref_}; cmp -s $f new_$t && echo "G_dump $t: BYTE-IDENTICAL" || echo "G_dump $t: DIFFERS"; done
echo "== G. 2D case vs private clean build"
cmp2 G_2d_t7d_dirichlet_conv_exact_cn $P "" --exact-oc dirichlet-conv --nx 16 --nz 16 --steps 32 --time-scheme cn
echo "== F. MPI np=2, 3D distributed production (MUMPS LU), X vs P"
M="$LU -sweep_right_pc_factor_mat_solver_type mumps -sweep_left_pc_factor_mat_solver_type mumps -sweep_ay_pc_factor_mat_solver_type mumps -fieldsplit_y_sweep_pc_factor_mat_solver_type mumps"
for b in P X; do e=${!b}; PETSC_OPTIONS="$M" mpirun -np 2 $e --exact-oc conv-sin3-3d --solve-only --nx 6 --ny 6 --nz 6 --steps 16 --time-scheme cn --pc-schur-sweep --convection on --vx 1 --vy 0.5 --vz -0.75 --no-paraview --dump-json $W/F_np2_$b.json > F_np2_$b.log 2>&1; echo "F_np2 [$b] exit=$? $(grep -oE 'nrank=\S+|ownership=\S+|reason=\S+|outer_iterations=\S+|true_residual=\S+' F_np2_$b.log | head -5 | tr '\n' ' ')"; done
python3 - $W <<'PY'
import json, sys, numpy as np
W = sys.argv[1]; a, b = (json.load(open(f"{W}/F_np2_{t}.json")) for t in ("P", "X"))
d = max(np.abs(np.array([s["values"] for s in a[k]]) - np.array([s["values"] for s in b[k]])).max() for k in ("state", "adjoint", "control_full"))
print(f"F_np2 X vs P: max field diff {d:.2e}, J {a['functional']['J_total']!r} / {b['functional']['J_total']!r}")
PY
echo T9_SHARED_BUILD_CHECK_DONE
