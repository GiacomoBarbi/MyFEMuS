#!/bin/bash
# (T11N copy of 3d_phase2/t10/scripts/t10_regress_t9.sh: logs into 3d_boundary/t11n/logs.)
# T10: T9 (3D phase 1) invariance, OLD = the T9 build (8ed57d629, private clean
# build) vs NEW (T10).  Every EX21_/EX20_ line with timings stripped, JSON and
# operator dumps byte-compared.  Run in the sanitised environment.  WORK= OLD=
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
W=${WORK:?}; OLD=${OLD:?}; mkdir -p $W/output; cd $W
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
LU="-sweep_right_pc_type lu -sweep_left_pc_type lu -sweep_ay_pc_type lu -fieldsplit_y_sweep_pc_type lu"
strip() { grep -E "^EX2[01]_" | sed -E 's/ (solve_wall|total_wall|setup_s|solve_s|t_operators|t_pcsetup|t_kspsolve|t_total|setup_seconds|core_seconds|shell_seconds)=[^ ]*//g'; }
cmpx() { name=$1 opts=$2; shift 2
  PETSC_OPTIONS="$opts" $OLD "$@" --no-paraview 2>/dev/null | strip > $name.old
  PETSC_OPTIONS="$opts" $X "$@" --no-paraview > $name.raw 2>&1; e=$?; strip < $name.raw > $name.new
  if diff -q $name.old $name.new > /dev/null; then v=IDENTICAL; else v="DIFFERS ($(diff $name.old $name.new | grep -c '^[<>]') lines)"; fi
  echo "$name: $v ($(wc -l < $name.new) lines, exit $e)"; }
jcmp() { name=$1 opts=$2; shift 2
  PETSC_OPTIONS="$opts" $OLD "$@" --no-paraview --dump-json $W/$name.old.json > /dev/null 2>&1
  PETSC_OPTIONS="$opts" $X "$@" --no-paraview --dump-json $W/$name.new.json > /dev/null 2>&1
  sed -i -E 's/"json":"[^"]*"//' $W/$name.old.json $W/$name.new.json
  cmp -s $W/$name.old.json $W/$name.new.json && echo "$name: JSON_IDENTICAL" || echo "$name: JSON_DIFFERS"; }
echo "== T9 gate battery (verify-convection conv-sin-3d, outer rtol 1e-12)"
for n in 2 3; do for s in be cn; do
  for w in "heat:--convection off" "w0:--convection on --vx 0 --vy 0 --vz 0" \
           "wA:--convection on --vx 1 --vy 0.5 --vz -0.75" "wB:--convection on --vx -2 --vy 1 --vz 0.5"; do
    cmpx t9gate_${w%%:*}_${s}_n$n "$LU -ksp_rtol 1e-12" --verify-convection --exact-oc conv-sin-3d --nx $n --ny $n --nz $n \
      --steps 4 --time-scheme $s --pc-schur-sweep ${w#*:}
  done
done; done
echo "== T9 exact driver"
cmpx t9exact_sin3_wA_cn_n4 "$LU" --exact-oc conv-sin3-3d --nx 4 --ny 4 --nz 4 --steps 64 --time-scheme cn --pc-schur-sweep --convection on --vx 1 --vy 0.5 --vz -0.75
cmpx t9exact_sin_wA_be_n4 "$LU" --exact-oc conv-sin-3d --nx 4 --ny 4 --nz 4 --steps 32 --time-scheme be --pc-schur-sweep --convection on --vx 1 --vy 0.5 --vz -0.75
cmpx t9exact_sin3_heat_cn_n6_gamg "" --exact-oc conv-sin3-3d --nx 6 --ny 6 --nz 6 --steps 16 --time-scheme cn --pc-schur-sweep --convection off
echo "== T9 production path (JSON)"
jcmp t9solveonly_sin3_wA_cn_n4 "$LU" --exact-oc conv-sin3-3d --solve-only --nx 4 --ny 4 --nz 4 --steps 16 --time-scheme cn --pc-schur-sweep --convection on --vx 1 --vy 0.5 --vz -0.75
jcmp t9prod_cmdline_dirichlet_wA_be "$LU" --dimension 3 --distributed-solve --control-full-domain --tracking-box 0 1 0 1 --state-bc dirichlet --nx 4 --ny 4 --nz 4 --steps 8 --convection on --vx 1 --vy 0.5 --vz -0.75 --pc-schur-sweep --target-type sin_x_minus_t --nu 0.25 --alpha-u 1e-2
jcmp t9prod_additive_be "" --dimension 3 --distributed-solve --control-full-domain --tracking-box 0 1 0 1 --state-bc dirichlet --nx 3 --ny 3 --nz 3 --steps 8 --convection on --vx 1 --vy 0.5 --vz -0.75
cmpx t9prod_cmdline_lines "$LU" --dimension 3 --distributed-solve --control-full-domain --tracking-box 0 1 0 1 --state-bc dirichlet --nx 4 --ny 4 --nz 4 --steps 8 --convection on --vx 1 --vy 0.5 --vz -0.75 --pc-schur-sweep
echo "== T9 operator dumps (n = 3, wind A)"
b="--dimension 3 --distributed-solve --control-full-domain --tracking-box 0 1 0 1 --state-bc dirichlet --nx 3 --ny 3 --nz 3 --steps 2 --convection on --vx 1 --vy 0.5 --vz -0.75 --no-paraview"
$OLD $b --dump-operators old > /dev/null 2>&1; $X $b --dump-operators new > /dev/null 2>&1
for f in old_*; do t=${f#old_}; cmp -s $f new_$t && echo "dump $t: BYTE-IDENTICAL" || echo "dump $t: DIFFERS"; done
ls new_* | sed 's/^new_//' | while read t; do [ -f old_$t ] || echo "dump $t: NEW FILE (T10 3D-only)"; done
echo "== T9 MPI np = 2 (MUMPS), production solve-only, old vs new JSON (round-off: np>1 is not bitwise reproducible)"
M="$LU -sweep_right_pc_factor_mat_solver_type mumps -sweep_left_pc_factor_mat_solver_type mumps -sweep_ay_pc_factor_mat_solver_type mumps -fieldsplit_y_sweep_pc_factor_mat_solver_type mumps"
for b in OLD X; do e=${!b}; PETSC_OPTIONS="$M" mpirun -np 2 $e --exact-oc conv-sin3-3d --solve-only --nx 6 --ny 6 --nz 6 --steps 16 --time-scheme cn --pc-schur-sweep --convection on --vx 1 --vy 0.5 --vz -0.75 --no-paraview --dump-json $W/mpi_$b.json > $W/mpi_$b.log 2>&1; done
python3 - $W <<'PY'
import json, sys, numpy as np
W = sys.argv[1]; a, b = (json.load(open(f"{W}/mpi_{t}.json")) for t in ("OLD", "X"))
d = max(np.abs(np.array([s["values"] for s in a[k]]) - np.array([s["values"] for s in b[k]])).max() / np.abs(np.array([s["values"] for s in a[k]])).max() for k in ("state", "adjoint", "control_full"))
print(f"t9_mpi_np2 old vs new: max field rel diff {d:.2e}, outer {a['solve']['iterations']}/{b['solve']['iterations']}, {'ROUND-OFF' if d < 1e-12 else 'DIFFERS'}")
PY
echo T11N_REGRESS_T9_DONE
