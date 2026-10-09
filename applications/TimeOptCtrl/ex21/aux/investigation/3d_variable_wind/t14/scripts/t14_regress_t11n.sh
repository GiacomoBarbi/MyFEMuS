#!/bin/bash
# (T14 copy of t13_regress_t11n.sh: logs into 3d_variable_wind/t14/logs; OLD = the T13 build, a28630779, private clean build)
# (T13 copy of t12_regress_t11n.sh: logs into 3d_multiface/t13/logs; OLD = the T12 build, 120c74442, private clean build)
# (T12 copy of t11h1_regress_t11n.sh: logs into simplex/t12/logs; OLD = the T11H1 build, 81f32b89e, private clean build)
# (T11H1 copy of t11d_regress_t11n.sh: logs into 3d_boundary/t11h1/logs; OLD = the T11D-L2 build, 7530c58f8, private clean build)
# T11D: T11N (3D Neumann boundary control) invariance, OLD = the T11N build (ec825f26d,
# private clean build) vs NEW (T11D).  Every EX20_/EX21_ line with timings stripped; JSON
# and operator dumps byte-compared.  Sanitised environment.   WORK= OLD=
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
echo "== T11N Neumann gates (neumann-conv-3d, n = 3) and production gates"
for s in be cn; do for w in "heat:--convection off" "wA:--convection on --vx 1 --vy 0.5 --vz -0.75" "wB:--convection on --vx -2 --vy 1 --vz 0.5" "wC:--convection on --vx 1 --vy 0.5 --vz 0"; do
  cmpx t11ngate_neu_${w%%:*}_$s "$G" --verify-convection --exact-oc neumann-conv-3d --nx 3 --ny 3 --nz 3 --steps 4 --time-scheme $s --pc-schur-sweep ${w#*:}
done
P="--dimension 3 --neumann-solve --state-bc natural --nu 0.25 --alpha-u 1e-2 --target-type sin_x_minus_t --steps 4"
cmpx t11ngate_pxm_$s "$G" --verify-convection --pc-schur-sweep $P --neumann-face3d xmax --tracking-box3d 0 1 0 1 0 1 --nx 3 --ny 3 --nz 3 --time-scheme $s --convection on --vx 1 --vy 0.5 --vz -0.75
cmpx t11ngate_pnu_$s "$G" --verify-convection --pc-schur-sweep $P --T 1 --neumann-face3d ymax --domain3d 0 2 -1 1 0 0.5 --tracking-box3d 0 2 0 1 0 0.5 --nx 4 --ny 4 --nz 2 --time-scheme $s --convection on --vx 1 --vy 0.5 --vz -0.75
done
echo "== T11N exact driver and production JSON"
cmpx t11nexact_neu_wA_cn_n4 "$LU" --exact-oc neumann-conv-3d --nx 4 --ny 4 --nz 4 --steps 64 --time-scheme cn --pc-schur-sweep --convection on --vx 1 --vy 0.5 --vz -0.75
cmpx t11nexact_neu_heat_be_n4 "$LU" --exact-oc neumann-conv-3d --nx 4 --ny 4 --nz 4 --steps 32 --time-scheme be --pc-schur-sweep --convection off
jcmp t11njson_neu_solveonly_cn "$LU -ksp_rtol 1e-12" --exact-oc neumann-conv-3d --solve-only --nx 4 --ny 4 --nz 4 --steps 16 --time-scheme cn --pc-schur-sweep --convection on --vx 1 --vy 0.5 --vz -0.75
jcmp t11njson_pym_gamg_be "" --dimension 3 --neumann-solve --state-bc natural --neumann-face3d ymin --tracking-box3d 0 1 0 1 0 1 --nx 4 --ny 4 --nz 4 --steps 8 --T 1 --time-scheme be --pc-schur-sweep --convection on --vx -2 --vy 1 --vz 0.5
echo "== T11N operator dumps (Neumann, faces zmin / xmax / ymax non-unit, wind A)"
for c in "zmin:--nx 3 --ny 3 --nz 3" "xmax:--nx 2 --ny 3 --nz 4" "ymax:--nx 4 --ny 4 --nz 2 --domain3d 0 2 -1 1 0 0.5"; do f=${c%%:*}
  b="--dimension 3 --neumann-solve --neumann-face3d $f --state-bc natural --steps 2 --convection on --vx 1 --vy 0.5 --vz -0.75 --tracking-box3d 0 1 0 1 0 1 --no-paraview ${c#*:}"
  [ $f = ymax ] && b="${b/--tracking-box3d 0 1 0 1 0 1/--tracking-box3d 0 2 0 1 0 0.5}"
  $OLD $b --dump-operators old_$f > /dev/null 2>&1; $X $b --dump-operators new_$f > /dev/null 2>&1
  for g in old_${f}_*; do t=${g#old_}; cmp -s $g new_$t && echo "dump $t: BYTE-IDENTICAL" || echo "dump $t: DIFFERS"; done
  echo "dump $f: new-only files: $(cd $W; for g in new_${f}_*; do [ -e old_${g#new_} ] || echo -n "${g#new_} "; done)"
done
echo T14_REGRESS_T11N_DONE
