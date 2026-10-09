#!/bin/bash
# T15 (copy of t14_runs.sh) production-path runs (--solve-only --dump-json) for the fixed-mesh time ladders.
# SCHUR_SWEEP + LU spatial solves (distributed / natural / Neumann); Dirichlet: the default reduced solver,
# core rtol 1e-12.  The case/production arguments (incl. --convection-stabilization lps) are passed through.
# usage: t15_runs.sh <tag> <scheme> "<n list>" "<nt list>" <args...>
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
D=$(cd "$(dirname "$0")/.." && pwd)
tag=$1; s=$2; ns=$3; nts=$4; shift 4
W=${T15_WORK:-/tmp/t15_runs_$tag}; J=${T15_JSON:-/tmp/t15_json}; mkdir -p $W/output $D/logs/runs $J; cd $W
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
LU="-sweep_right_pc_type lu -sweep_left_pc_type lu -sweep_ay_pc_type lu -fieldsplit_y_sweep_pc_type lu"
case "$*" in *dirichlet*) PC="--dirichlet-core-rtol 1e-12"; PO="";; *) PC="--pc-schur-sweep"; PO="$LU -ksp_rtol 1e-12";; esac
for n in $ns; do for nt in $nts; do
  t=${tag}_${s}_n${n}_nt${nt}
  PETSC_OPTIONS="$PO" /usr/bin/time -f "MAXRSS=%M wall=%e" $X --solve-only --nx $n --ny $n --nz $n --steps $nt \
    --time-scheme $s --no-paraview $PC --dump-json $J/$t.json "$@" > $D/logs/runs/$t.log 2>&1
  echo "$t exit=$? $(grep -o 'MAXRSS=.*' $D/logs/runs/$t.log)"
done; done
