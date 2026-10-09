#!/bin/bash
# (T15 copy of t14_regress_t11h1.sh: logs into 3d_lps/t15/logs; OLD = the T14 build, 3d59677b8, private clean build)
# (T14 copy of t13_regress_t11h1.sh: logs into 3d_variable_wind/t14/logs; OLD = the T13 build, a28630779, private clean build)
# (T13 copy of t12_regress_t11h1.sh: logs into 3d_multiface/t13/logs; OLD = the T12 build, 120c74442, private clean build)
# T12: T11H1 (3D Dirichlet H1) invariance, OLD = the T11H1 build (81f32b89e, private clean build) vs NEW
# (T12).  Every EX20_/EX21_ line with timings stripped; JSON and operator dumps byte-compared.  WORK= OLD=
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
W=${WORK:?}; OLD=${OLD:?}; mkdir -p $W/output; cd $W
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
strip() { grep -E "^EX2[01]_" | sed -E 's/ (solve_wall|total_wall|setup_s|solve_s|t_operators|t_pcsetup|t_kspsolve|t_total|setup_seconds|core_seconds|shell_seconds)=[^ ]*//g'; }
cmpx() { name=$1; shift
  $OLD "$@" --no-paraview 2>/dev/null | strip > $name.old
  $X "$@" --no-paraview > $name.raw 2>&1; e=$?; strip < $name.raw > $name.new
  if diff -q $name.old $name.new > /dev/null; then v=IDENTICAL; else v="DIFFERS ($(diff $name.old $name.new | grep -c '^[<>]') lines)"; fi
  echo "$name: $v ($(wc -l < $name.new) lines, exit $e)"; }
jcmp() { name=$1; shift
  $OLD "$@" --no-paraview --dump-json $W/$name.old.json > /dev/null 2>&1
  $X "$@" --no-paraview --dump-json $W/$name.new.json > /dev/null 2>&1
  sed -i -E 's/"json":"[^"]*"//' $W/$name.old.json $W/$name.new.json
  cmp -s $W/$name.old.json $W/$name.new.json && echo "$name: JSON_IDENTICAL" || echo "$name: JSON_DIFFERS"; }
REGS=("H1s:--control-h1-length 0.3" "H1t:--control-h1-time-weight 1e-2" "comb:--control-h1-length 0.3 --control-h1-time-weight 1e-2")
echo "== T11H1 gates (dirichlet-h1-3d exact n = 2; production xmax, BE / CN)"
for s in be cn; do for r in "${REGS[@]}"; do
  cmpx t11h1gate_ex_${r%%:*}_wN_$s --verify-convection --exact-oc dirichlet-h1-3d --nx 2 --ny 2 --nz 2 --steps 4 --time-scheme $s --convection on --vx 0 --vy 0 --vz -0.75 ${r#*:}
  cmpx t11h1gate_pxm_${r%%:*}_$s --verify-convection --dimension 3 --dirichlet-solve --state-bc dirichlet --nu 0.25 --alpha-u 1e-2 --target-type sin_x_minus_t --steps 4 --T 1 \
       --dirichlet-face3d xmax --tracking-box3d 0 0.6 0.2 0.8 0.2 0.8 --nx 3 --ny 3 --nz 3 --time-scheme $s --convection on --vx 1 --vy 0.5 --vz -0.75 ${r#*:}
done; done
echo "== T11H1 exact driver (incl. dense / verify core) and JSON"
cmpx t11h1exact_comb_cn_n4 --exact-oc dirichlet-h1-3d --nx 4 --ny 4 --nz 4 --steps 16 --time-scheme cn --convection on --vx 0 --vy 0 --vz -0.75 --control-h1-length 0.3 --control-h1-time-weight 1e-2
cmpx t11h1exact_h1t_be_dense --exact-oc dirichlet-h1-3d --nx 3 --ny 3 --nz 3 --steps 8 --time-scheme be --convection off --control-h1-time-weight 1e-2 --dirichlet-core-solver dense --verify-dirichlet-core
jcmp t11h1json_comb_cn --exact-oc dirichlet-h1-3d --solve-only --nx 4 --ny 4 --nz 4 --steps 16 --time-scheme cn --convection on --vx 0 --vy 0 --vz -0.75 --control-h1-length 0.3 --control-h1-time-weight 1e-2 --dirichlet-core-rtol 1e-12
jcmp t11h1json_prod_nonunit --dimension 3 --dirichlet-solve --state-bc dirichlet --nu 0.25 --alpha-u 1e-2 --target-type sin_x_minus_t --steps 8 --T 1 --dirichlet-face3d ymax --domain3d 0 2 -1 1 0 0.5 --tracking-box3d 0.5 1.5 -1 0.4 0 0.5 --nx 4 --ny 4 --nz 2 --time-scheme be --convection on --vx 1 --vy 0.5 --vz -0.75 --control-h1-length 0.3 --control-h1-time-weight 1e-2
echo "== T11H1 operator dumps (combined, faces zmin / xmax)"
for f in zmin xmax; do
  b="--dimension 3 --dirichlet-solve --dirichlet-face3d $f --state-bc dirichlet --nx 3 --ny 3 --nz 3 --steps 3 --T 0.3 --time-scheme cn --convection on --vx 1 --vy 0.5 --vz -0.75 --control-h1-length 0.3 --control-h1-time-weight 1e-2 --no-paraview"
  [ $f = zmin ] && b="$b --tracking-box3d 0.2 0.8 0.2 0.8 0.4 1" || b="$b --tracking-box3d 0 0.6 0.2 0.8 0.2 0.8"
  $OLD $b --dump-operators old_$f > /dev/null 2>&1; $X $b --dump-operators new_$f > /dev/null 2>&1
  for g in old_${f}_*; do t=${g#old_}; cmp -s $g new_$t && echo "dump $t: BYTE-IDENTICAL" || echo "dump $t: DIFFERS"; done
  echo "dump $f: new-only files: $(for g in new_${f}_*; do [ -e old_${g#new_} ] || echo -n "${g#new_} "; done)"
done
echo T15_REGRESS_T11H1_DONE
