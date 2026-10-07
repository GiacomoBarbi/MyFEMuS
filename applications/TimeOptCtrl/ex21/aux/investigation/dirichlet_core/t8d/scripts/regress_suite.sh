#!/bin/bash
# (T8D copy of t7f/scripts/regress_suite.sh) writing into dirichlet_core/t8d/logs.
#   R1 ex20 goldens (61 cases)               ex20/aux/cn_stage1/be_nonregression.py
#   R2 Mathematics 2025 Ex. 3 / Ex. 4        ../../scripts/regress_ex3_ex4.sh
#   R3 Mathematics 2025 Ex. 5, EF 9.5        ../t3/regress_2d.sh
#   R4 T0/T1/T2 constant wind (dumps, archived runs, gates)   ../t3/regress_constant_wind.py
#   R5 T3/T3b/T4 representative runs: the T4 binary (500a2b2e, copy in $OLD) vs the
#      current one, every EX21_ line, wall times removed; the T6 status fields
#      (accepted=, status= on EX21_EXACT_SOLVER_STATS) and EX21_SOLVE_REJECTED lines
#      are reported separately (new diagnostics, expected only on failed solves).
# WORK=<scratch> OLD=<old binary> regress_t6.sh
set -u
H=$(cd "$(dirname "$0")/.." && pwd); I=$(dirname $(dirname $H)); LP=$I/leveque_pearson
X=/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21
W=${WORK:?set WORK}; OLD=${OLD:?set OLD}; mkdir -p $W/r5/output
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
sha256sum $X $OLD
EX20=$(cd $I/../../../ex20 && pwd)
echo "== R1"; ( cd $EX20 && env -i HOME=$HOME PATH=/home/giacomo/software/petsc/arch-linux-c-openblas-opt/bin:/usr/bin:/bin \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3.11 aux/cn_stage1/be_nonregression.py \
    --golden aux/cn_stage1/baseline/refactor_golden_openblas --golden aux/cn_stage1/baseline/cli_golden_openblas \
    --exe $X --work $W/benr ) > $H/logs/regress_ex20_goldens.log 2>&1; tail -n 2 $H/logs/regress_ex20_goldens.log
echo "== R2"; WORK=$W/r2 bash $I/scripts/regress_ex3_ex4.sh > $H/logs/regress_ex3_ex4.log 2>&1; cat $H/logs/regress_ex3_ex4.log
echo "== R3"; WORK=$W/r3 bash $LP/t3/regress_2d.sh > $H/logs/regress_2d.log 2>&1; cat $H/logs/regress_2d.log
echo "== R4"; WORK=$W/r4 EX21_EXE=$X python3 $LP/t3/regress_constant_wind.py > $H/logs/regress_constant_wind.log 2>&1; tail -n 1 $H/logs/regress_constant_wind.log
echo "== R5"
LU="-sweep_right_pc_type lu -sweep_left_pc_type lu -sweep_ay_pc_type lu -fieldsplit_y_sweep_pc_type lu"
strip() { grep -E "^EX21_" | grep -v "^EX21_SOLVE_REJECTED" | sed -E 's/ (solve_wall|total_wall|setup_s|solve_s)=[^ ]*//g; s/ accepted=[01] status=[a-z_]*//'; }
r5() { name=$1; opts=$2; shift 2
  cd $W/r5
  PETSC_OPTIONS="$opts" $OLD "$@" --no-paraview 2>/dev/null | strip > $W/r5/$name.old; oe=${PIPESTATUS[0]}
  PETSC_OPTIONS="$opts" $X "$@" --no-paraview > $W/r5/$name.raw 2>/dev/null; ne=$?
  strip < $W/r5/$name.raw > $W/r5/$name.new
  rej=$(grep -c "^EX21_SOLVE_REJECTED" $W/r5/$name.raw)
  if diff -q $W/r5/$name.old $W/r5/$name.new > /dev/null; then v=IDENTICAL; else v="DIFFERS ($(diff $W/r5/$name.old $W/r5/$name.new | grep -c '^[<>]') lines)"; fi
  echo "$name: $v ($(wc -l < $W/r5/$name.new) EX21_ lines) exit old=$oe new=$ne rejected_lines=$rej"; }
r5 t3b_lift_track_cn_nx8 "" --exact-oc lift-track --nx 8 --nz 8 --steps 16 --time-scheme cn --distributed-solve --pc-schur-sweep
r5 t3_lift_const_q2divfree_be_nx8 "" --exact-oc lift-const --wind q2-divfree --nx 8 --nz 8 --steps 16 --time-scheme be --distributed-solve --pc-schur-sweep
r5 t3_vw_oc_cn_nx8 "" --exact-oc conv-sin3 --wind q2-divfree --nx 8 --nz 8 --steps 32 --time-scheme cn --distributed-solve --pc-schur-sweep
r5 t3_lp52_eps20_nx16_gamg "" --exact-oc lp52 --nx 16 --nz 16 --steps 32 --time-scheme cn --distributed-solve --pc-schur-sweep --exact-oc-production-tol
r5 t4_lp52_eps500_nx16_lu "$LU" --exact-oc lp52 --exact-oc-nu 0.002 --nx 16 --nz 16 --steps 32 --time-scheme cn --distributed-solve --pc-schur-sweep --exact-oc-production-tol
r5 t4_lp52_eps100_nx32_be_lu "$LU" --exact-oc lp52 --exact-oc-nu 0.01 --nx 32 --nz 32 --steps 64 --time-scheme be --distributed-solve --pc-schur-sweep --exact-oc-production-tol
r5 t4_lp52_eps500_nx16_gamg_FAILS "" --exact-oc lp52 --exact-oc-nu 0.002 --nx 16 --nz 16 --steps 32 --time-scheme cn --distributed-solve --pc-schur-sweep --exact-oc-production-tol
r5 t4_block_test_D "" --exact-oc lp52 --exact-oc-nu 0.002 --nx 32 --nz 32 --steps 64 --distributed-solve --exact-oc-block-test
r5 t2_verify_convection_sin3_cn "" --verify-convection --exact-oc conv-sin3 --nx 8 --nz 8 --steps 16 --time-scheme cn --distributed-solve --pc-schur-sweep
echo REGRESS_T6_DONE
