#!/bin/bash
# (T12 copy of t11h1_regress_t10.sh: logs into simplex/t12/logs; OLD = the T11H1 build, 81f32b89e, private clean build)
# (T11H1 copy of t11d_regress_t10.sh: logs into 3d_boundary/t11h1/logs; OLD = the T11D-L2 build, 7530c58f8, private clean build)
# (T11D copy of t11n_regress_t10.sh: logs into 3d_boundary/t11d/logs; OLD = the T11N build, ec825f26d, private clean build)
# T11N: T10 (3D phase 2) invariance, OLD = the T10 build (577805e36, private clean
# build) vs NEW (T11N).  Every EX20_/EX21_ line with timings stripped; JSON and
# operator dumps byte-compared.  Sanitised environment.   WORK= OLD=
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
G="$LU -ksp_rtol 1e-12"
echo "== T10 natural-BC gates (natural-conv-3d)"
for s in be cn; do for w in "wA:--convection on --vx 1 --vy 0.5 --vz -0.75" "wB:--convection on --vx -2 --vy 1 --vz 0.5" "heat:--convection off"; do
  cmpx t10gate_nat_${w%%:*}_$s "$G" --verify-convection --exact-oc natural-conv-3d --nx 3 --ny 3 --nz 3 --steps 4 --time-scheme $s --pc-schur-sweep ${w#*:}
done; done
MZ="--distributed-solve --dimension 3 --nx 4 --ny 4 --nz 4 --steps 4 --nu 0.25 --alpha-u 1e-2 --target-type sin_x_minus_t
    --tracking-box3d 0 0.5 0.5 1 0 0.5 --tracking-box3d-add 0.5 1 0 0.5 0.5 1
    --control-box3d 0 0.5 0 0.5 0 0.5 --control-box3d-add 0.5 1 0.5 1 0.5 1
    --source 1 --source-box3d-add 0 0.25 0.75 1 0.75 1 --source-box3d-add 0.75 1 0 0.25 0 0.25"
echo "== T10 multi-zone / sources / non-unit gates"
for s in be cn; do
  cmpx t10gate_mz_wA_$s "$G" --verify-convection --pc-schur-sweep $MZ --state-bc natural --time-scheme $s --convection on --vx 1 --vy 0.5 --vz -0.75
  cmpx t10gate_mzd_wA_$s "$G" --verify-convection --pc-schur-sweep $MZ --state-bc dirichlet --time-scheme $s --convection on --vx 1 --vy 0.5 --vz -0.75
  cmpx t10gate_nu_wA_$s "$G" --verify-convection --pc-schur-sweep --distributed-solve --dimension 3 --domain3d 0 2 -1 1 0 0.5 --nx 4 --ny 4 --nz 2 --steps 4 \
      --nu 0.25 --alpha-u 1e-2 --target-type sin_x_minus_t --state-bc natural --time-scheme $s \
      --tracking-box3d 1 2 -1 1 0 0.5 --control-box3d 0 1 -1 0 0 0.5 --source 1 --source-box3d-add 0 0.5 0 1 0 0.25 --convection on --vx 1 --vy 0.5 --vz -0.75
done
echo "== T10 exact driver and production JSON"
cmpx t10exact_nat_wA_cn_n4 "$LU" --exact-oc natural-conv-3d --nx 4 --ny 4 --nz 4 --steps 64 --time-scheme cn --pc-schur-sweep --convection on --vx 1 --vy 0.5 --vz -0.75
cmpx t10exact_nat_wB_be_n4 "$LU" --exact-oc natural-conv-3d --nx 4 --ny 4 --nz 4 --steps 32 --time-scheme be --pc-schur-sweep --convection on --vx -2 --vy 1 --vz 0.5
jcmp t10json_mz_T1_cn "$LU -ksp_rtol 1e-12" $MZ --T 1 --state-bc natural --time-scheme cn --pc-schur-sweep --convection on --vx 1 --vy 0.5 --vz -0.75
jcmp t10json_nat_solveonly_be "$LU -ksp_rtol 1e-12" --exact-oc natural-conv-3d --solve-only --nx 4 --ny 4 --nz 4 --steps 16 --time-scheme be --pc-schur-sweep --convection on --vx 1 --vy 0.5 --vz -0.75
jcmp t10json_sources_gamg "" $MZ --T 1 --state-bc natural --time-scheme be --pc-schur-sweep --convection on --vx 1 --vy 0.5 --vz -0.75
echo "== T10 operator dumps (multi-zone + sources, natural, wind A, n = 4)"
b="$MZ --state-bc natural --convection on --vx 1 --vy 0.5 --vz -0.75 --no-paraview"
$OLD $b --dump-operators old > /dev/null 2>&1; $X $b --dump-operators new > /dev/null 2>&1
for f in old_*; do t=${f#old_}; cmp -s $f new_$t && echo "dump $t: BYTE-IDENTICAL" || echo "dump $t: DIFFERS"; done
echo T12_REGRESS_T10_DONE
