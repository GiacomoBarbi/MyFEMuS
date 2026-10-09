#!/bin/bash
# (T15 copy of t14_regress_t12.sh: logs into 3d_lps/t15/logs; OLD = the T14 build, 3d59677b8, private clean build)
# (T14 copy of t13_regress_t12.sh: logs into 3d_variable_wind/t14/logs; OLD = the T13 build, a28630779, private clean build)
# T13: T12 (simplex P2 TRI / TET, distributed control) invariance.  OLD = the T12 build (120c74442, private
# clean build) vs NEW (T13).  (a) --verify-convection gates (exact conv-sin3 / conv-sin3-3d and production,
# BE / CN, wind A): every EX20_/EX21_ line (timings stripped) compared; (b) production JSON byte-compared;
# (c) operator dumps (TRI, TET) byte-compared; (d) the simplex refusals (message text) compared.  WORK= OLD=
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
W=${WORK:?}; OLD=${OLD:?}; mkdir -p $W/output; cd $W
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
LU="-sweep_right_pc_type lu -sweep_left_pc_type lu -sweep_ay_pc_type lu -fieldsplit_y_sweep_pc_type lu -ksp_rtol 1e-12"
strip() { grep -E "^EX2[01]_" | sed -E 's/ (solve_wall|total_wall|setup_s|solve_s|t_operators|t_pcsetup|t_kspsolve|t_total|setup_seconds|core_seconds|shell_seconds|json)=[^ ]*//g'; }
cmpx() { name=$1; shift
  PETSC_OPTIONS="$LU" $OLD "$@" --no-paraview 2>/dev/null | strip > $name.old
  PETSC_OPTIONS="$LU" $X "$@" --no-paraview > $name.raw 2>&1; e=$?; strip < $name.raw > $name.new
  if diff -q $name.old $name.new > /dev/null; then v=IDENTICAL; else v="DIFFERS ($(diff $name.old $name.new | grep -c '^[<>]') lines)"; fi
  echo "$name: $v ($(wc -l < $name.new) lines, exit $e)"; }
jcmp() { name=$1; shift
  PETSC_OPTIONS="$LU" $OLD "$@" --no-paraview --dump-json $W/$name.old.json > /dev/null 2>&1
  PETSC_OPTIONS="$LU" $X "$@" --no-paraview --dump-json $W/$name.new.json > /dev/null 2>&1
  sed -i -E 's/"json":"[^"]*"//' $W/$name.old.json $W/$name.new.json
  cmp -s $W/$name.old.json $W/$name.new.json && echo "$name: JSON_IDENTICAL" || echo "$name: JSON_DIFFERS"; }
wA2="--exact-oc-velocity 1 0.5"; wA3="--convection on --vx 1 --vy 0.5 --vz -0.75"
echo "== T12 gates (simplex)"
for s in be cn; do
  for m in tri quad; do
    cmpx gate_ex_${m}_$s --verify-convection --pc-schur-sweep --exact-oc conv-sin3 --mesh-type $m --nx 4 --nz 4 --steps 4 --time-scheme $s $wA2
    cmpx gate_pr_${m}_$s --verify-convection --pc-schur-sweep --mesh-type $m --domain 0 1 0 1 --distributed-solve --control-full-domain --state-bc dirichlet \
         --tracking-box 0 1 0 1 --target-type sin_x_minus_t --nu 0.25 --alpha-u 1e-2 --nx 4 --nz 4 --steps 4 --T 1 --time-scheme $s --convection on --vx 1 --vy 0.5
  done
  for m in tet hex; do
    cmpx gate_ex_${m}_$s --verify-convection --pc-schur-sweep --exact-oc conv-sin3-3d --mesh-type $m --nx 2 --ny 2 --nz 2 --steps 4 --time-scheme $s $wA3
    cmpx gate_pr_${m}_$s --verify-convection --pc-schur-sweep --dimension 3 --mesh-type $m --domain3d 0 2 -1 1 0 0.5 --distributed-solve --control-full-domain \
         --state-bc dirichlet --tracking-box3d 0 2 -1 1 0 0.5 --target-type sin_x_minus_t --nu 0.25 --alpha-u 1e-2 --nx 3 --ny 2 --nz 2 --steps 4 --T 1 --time-scheme $s $wA3
  done
done
echo "== T12 exact drivers and JSON"
cmpx exact_tri_cn --pc-schur-sweep --exact-oc conv-sin3 --mesh-type tri --nx 6 --nz 6 --steps 16 --time-scheme cn $wA2
cmpx exact_tet_cn --pc-schur-sweep --exact-oc conv-sin3-3d --mesh-type tet --nx 3 --ny 3 --nz 3 --steps 8 --time-scheme cn $wA3
jcmp json_tet_be --pc-schur-sweep --dimension 3 --mesh-type tet --domain3d 0 1 0 1 0 1 --distributed-solve --control-full-domain --state-bc dirichlet \
     --tracking-box3d 0 1 0 1 0 1 --target-type sin_x_minus_t --nu 0.25 --alpha-u 1e-2 --nx 3 --ny 3 --nz 3 --steps 8 --T 1 --time-scheme be $wA3
jcmp json_tri_cn --pc-schur-sweep --mesh-type tri --domain 0 1 0 1 --distributed-solve --control-full-domain --state-bc dirichlet --tracking-box 0 1 0 1 \
     --target-type sin_x_minus_t --nu 0.25 --alpha-u 1e-2 --nx 6 --nz 6 --steps 8 --T 1 --time-scheme cn --convection on --vx 1 --vy 0.5
echo "== T12 operator dumps"
for m in tri tet; do
  if [ $m = tri ]; then b="--mesh-type tri --domain 0 1 0 1 --distributed-solve --control-full-domain --state-bc dirichlet --tracking-box 0 1 0 1 --nx 4 --nz 4 --convection on --vx 1 --vy 0.5"
  else b="--dimension 3 --mesh-type tet --domain3d 0 1 0 1 0 1 --distributed-solve --control-full-domain --state-bc dirichlet --tracking-box3d 0 1 0 1 0 1 --nx 2 --ny 2 --nz 2 $wA3"; fi
  $OLD $b --steps 2 --pc-schur-sweep --no-paraview --dump-operators old_$m > /dev/null 2>&1; $X $b --steps 2 --pc-schur-sweep --no-paraview --dump-operators new_$m > /dev/null 2>&1
  for g in old_${m}_*; do t=${g#old_}; cmp -s $g new_$t && echo "dump $t: BYTE-IDENTICAL" || echo "dump $t: DIFFERS"; done
done
echo "== T12 simplex refusals (message text)"
for c in "--dimension 3 --mesh-type tet --nx 2 --ny 2 --nz 2 --steps 2 --neumann-solve --neumann-face3d zmin --state-bc natural --tracking-box3d 0 1 0 1 0 1" \
         "--dimension 3 --mesh-type tet --nx 2 --ny 2 --nz 2 --steps 2 --dirichlet-solve --dirichlet-face3d zmin --state-bc dirichlet --tracking-box3d 0 1 0 1 0.5 1" \
         "--mesh-type tri --nx 4 --nz 4 --steps 2 --neumann-solve" "--mesh-type tri --nx 4 --nz 4 --steps 2 --distributed-solve --control-full-domain --convection on --vx 1 --convection-stabilization lps"; do
  a=$($OLD $c --no-paraview 2>&1 | grep -oE "what\(\): .*"); b=$($X $c --no-paraview 2>&1 | grep -oE "what\(\): .*")
  [ "$a" = "$b" ] && echo "refusal: IDENTICAL ${b#what(): }" || echo "refusal: DIFFERS old='$a' new='$b'"
done
echo T15_REGRESS_T12_DONE
