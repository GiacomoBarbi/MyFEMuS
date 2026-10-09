#!/bin/bash
# (T14 copy of t13_regress_t11d.sh: logs into 3d_variable_wind/t14/logs; OLD = the T13 build, a28630779, private clean build)
# (T13 copy of t12_regress_t11d.sh: logs into 3d_multiface/t13/logs; OLD = the T12 build, 120c74442, private clean build)
# (T12 copy of t11h1_regress_t11d.sh: logs into simplex/t12/logs; OLD = the T11H1 build, 81f32b89e, private clean build)
# T11H1: T11D-L2 (3D Dirichlet control, L2) invariance, OLD = the T11D-L2 build (7530c58f8, private
# clean build) vs NEW (T11H1).  Every EX20_/EX21_ line with timings stripped; JSON and operator dumps
# byte-compared (NEW writes extra dump files -- K_Gamma, RT, *_face -- listed separately).  WORK= OLD=
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
W=${WORK:?}; OLD=${OLD:?}; mkdir -p $W/output; cd $W
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
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
echo "== T11D gates (dirichlet-conv-3d exact, n = 2, 3; production xmax / ymin / zmax heat / non-unit)"
for n in 2 3; do for s in be cn; do for w in "heat:--convection off" "wA:--convection on --vx 1 --vy 0.5 --vz -0.75" "wN:--convection on --vx 0 --vy 0 --vz -0.75" "wC:--convection on --vx 1 --vy 0.5 --vz 0"; do
  cmpx t11dgate_dir_${w%%:*}_${s}_n$n "" --verify-convection --exact-oc dirichlet-conv-3d --nx $n --ny $n --nz $n --steps 4 --time-scheme $s ${w#*:}
done; done; done
P="--dimension 3 --dirichlet-solve --state-bc dirichlet --nu 0.25 --alpha-u 1e-2 --target-type sin_x_minus_t --steps 4 --T 1"
for s in be cn; do
  cmpx t11dgate_pxm_$s "" --verify-convection $P --dirichlet-face3d xmax --tracking-box3d 0 0.6 0.2 0.8 0.2 0.8 --nx 3 --ny 3 --nz 3 --time-scheme $s --convection on --vx 1 --vy 0.5 --vz -0.75
  cmpx t11dgate_pzx_heat_$s "" --verify-convection $P --dirichlet-face3d zmax --tracking-box3d 0.2 0.8 0.2 0.8 0 0.6 --nx 3 --ny 3 --nz 3 --time-scheme $s --convection off
  cmpx t11dgate_pnu_$s "" --verify-convection $P --dirichlet-face3d ymax --domain3d 0 2 -1 1 0 0.5 --tracking-box3d 0.5 1.5 -1 0 0 0.5 --nx 4 --ny 4 --nz 2 --time-scheme $s --convection on --vx 1 --vy 0.5 --vz -0.75
done
echo "== T11D exact driver (both cores, verify core), production JSON"
cmpx t11dexact_wN_cn_n4 "" --exact-oc dirichlet-conv-3d --nx 4 --ny 4 --nz 4 --steps 64 --time-scheme cn --convection on --vx 0 --vy 0 --vz -0.75
cmpx t11dexact_heat_be_n4_dense "" --exact-oc dirichlet-conv-3d --nx 4 --ny 4 --nz 4 --steps 32 --time-scheme be --convection off --dirichlet-core-solver dense
cmpx t11dexact_wA_cn_verifycore "" --exact-oc dirichlet-conv-3d --nx 3 --ny 3 --nz 3 --steps 8 --time-scheme cn --convection on --vx 1 --vy 0.5 --vz -0.75 --dirichlet-core-solver dense --verify-dirichlet-core
jcmp t11djson_solveonly_cn "" --exact-oc dirichlet-conv-3d --solve-only --nx 4 --ny 4 --nz 4 --steps 16 --time-scheme cn --convection on --vx 0 --vy 0 --vz -0.75 --dirichlet-core-rtol 1e-12
jcmp t11djson_prod_xmax_cn "" $P --dirichlet-face3d xmax --tracking-box3d 0.3 0.7 0.2 0.8 0.2 0.8 --nx 4 --ny 4 --nz 4 --steps 8 --time-scheme cn --convection on --vx 1 --vy 0.5 --vz -0.75
jcmp t11djson_prod_nonunit_be "" $P --dirichlet-face3d ymax --domain3d 0 2 -1 1 0 0.5 --tracking-box3d 0.5 1.5 -1 0.4 0 0.5 --nx 4 --ny 4 --nz 2 --steps 8 --time-scheme be --convection on --vx 1 --vy 0.5 --vz -0.75
echo "== T11D operator dumps (L2: faces zmin / xmax, BE / CN)"
for f in zmin xmax; do for s in be cn; do
  b="--dimension 3 --dirichlet-solve --dirichlet-face3d $f --state-bc dirichlet --nx 2 --ny 2 --nz 2 --steps 3 --T 0.3 --time-scheme $s --convection on --vx 1 --vy 0.5 --vz -0.75 --tracking-box3d 0.25 0.75 0.25 0.75 0.25 0.75 --no-paraview"
  $OLD $b --dump-operators old_${f}_$s > /dev/null 2>&1; $X $b --dump-operators new_${f}_$s > /dev/null 2>&1
  for g in old_${f}_${s}_*; do t=${g#old_}; cmp -s $g new_$t && echo "dump $t: BYTE-IDENTICAL" || echo "dump $t: DIFFERS"; done
  echo "dump ${f}_$s: new-only files: $(for g in new_${f}_${s}_*; do [ -e old_${g#new_} ] || echo -n "${g#new_} "; done)"
done; done
echo "== T11D MPI np 2 (exact conv CN, JSON vs OLD, outer rtol 1e-14)"
for b in old new; do bin=$OLD; [ $b = new ] && bin=$X
  PETSC_OPTIONS="-ksp_rtol 1e-14" mpirun -np 2 $bin --exact-oc dirichlet-conv-3d --solve-only --nx 4 --ny 4 --nz 4 --steps 8 --time-scheme cn --convection on --vx 1 --vy 0.5 --vz -0.75 --dirichlet-core-rtol 1e-12 --no-paraview --dump-json $W/mpi_$b.json > /dev/null 2>&1
done
sed -i -E 's/"json":"[^"]*"//' $W/mpi_old.json $W/mpi_new.json
cmp -s $W/mpi_old.json $W/mpi_new.json && echo "t11d_mpi_np2: JSON_IDENTICAL" || /usr/bin/python3.11 -c "
import json, numpy as np
a, b = (json.load(open('$W/mpi_%s.json' % t)) for t in ('old', 'new'))
f = lambda d, k: np.array([s['values'] for s in d[k]])
print('t11d_mpi_np2 old vs new: max field rel diff %.2e (MUMPS np 2 is not bitwise run to run)' % max(np.abs(f(a, k) - f(b, k)).max() / np.abs(f(a, k)).max() for k in ('state', 'adjoint', 'control_full')))"
echo T14_REGRESS_T11D_DONE
