#!/bin/bash
# (T11N copy of 3d_phase2/t10/scripts/t10_runs.sh)
# T10 production-path runs with --dump-json (fixed-mesh time ladders, fixed-dt
# space ladders, multi-zone / source studies).  SCHUR_SWEEP; spatial backend LU
# unless T11N_PETSC is set.  Outer rtol 1e-12 (fields compared to ~1e-10).
# usage: t11n_runs.sh <tag> <scheme> "<n list>" "<nt list>" <args...>
#   writes $T11N_JSON/<tag>_<scheme>_n<n>_nt<nt>.json and logs/runs/<same>.log
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
D=$(cd "$(dirname "$0")/.." && pwd)
tag=$1; s=$2; ns=$3; nts=$4; shift 4
LU="-sweep_right_pc_type lu -sweep_left_pc_type lu -sweep_ay_pc_type lu -fieldsplit_y_sweep_pc_type lu -ksp_rtol 1e-12"
W=${T11N_WORK:-/tmp/t11n_runs_$tag}; J=${T11N_JSON:-/tmp/t11n_json}; mkdir -p $W/output $D/logs/runs $J; cd $W
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for n in $ns; do for nt in $nts; do
  t=${tag}_${s}_n${n}_nt${nt}
  PETSC_OPTIONS="${T11N_PETSC-$LU}" /usr/bin/time -f "MAXRSS=%M wall=%e" $X --nx $n --ny $n --nz $n --steps $nt \
    --time-scheme $s --pc-schur-sweep --no-paraview --dump-json $J/$t.json "$@" > $D/logs/runs/$t.log 2>&1
  echo "$t exit=$? $(grep -o 'MAXRSS=.*' $D/logs/runs/$t.log)"
done; done
