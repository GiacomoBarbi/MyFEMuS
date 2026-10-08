#!/bin/bash
# T11D production-path runs of the exact case (--solve-only --dump-json) for the
# fixed-mesh time ladders.  Default Dirichlet solver; core rtol 1e-12 so the fields
# are compared well below the time errors.
# usage: t11d_runs.sh <tag> <scheme> "<n list>" "<nt list>" <args...>
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
D=$(cd "$(dirname "$0")/.." && pwd)
tag=$1; s=$2; ns=$3; nts=$4; shift 4
W=${T11D_WORK:-/tmp/t11d_runs_$tag}; J=${T11D_JSON:-/tmp/t11d_json}; mkdir -p $W/output $D/logs/runs $J; cd $W
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for n in $ns; do for nt in $nts; do
  t=${tag}_${s}_n${n}_nt${nt}
  /usr/bin/time -f "MAXRSS=%M wall=%e" $X --exact-oc dirichlet-conv-3d --solve-only --nx $n --ny $n --nz $n --steps $nt \
    --time-scheme $s --no-paraview --dirichlet-core-rtol 1e-12 --dump-json $J/$t.json "$@" > $D/logs/runs/$t.log 2>&1
  echo "$t exit=$? $(grep -o 'MAXRSS=.*' $D/logs/runs/$t.log)"
done; done
