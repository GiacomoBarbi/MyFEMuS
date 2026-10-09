#!/bin/bash
# T12 production-path runs of an exact case (--solve-only --dump-json) for the fixed-mesh time ladders,
# SCHUR_SWEEP + LU, outer rtol 1e-12; plus one operator dump of the same mesh (M_full for the FE L2 norm).
# usage: t12_runs.sh <tag> <case> <mesh-type> <scheme> <n> "<nt list>" [extra]
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
D=$(cd "$(dirname "$0")/.." && pwd)
tag=$1; c=$2; m=$3; s=$4; n=$5; nts=$6; shift 6
LU="-sweep_right_pc_type lu -sweep_left_pc_type lu -sweep_ay_pc_type lu -fieldsplit_y_sweep_pc_type lu -ksp_rtol 1e-12"
W=${T12_WORK:-/tmp/t12_runs_$tag}; J=${T12_JSON:-/tmp/t12_json}; mkdir -p $W/output $D/logs/runs $J; cd $W
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
case $m in hex|tet) N="--nx $n --ny $n --nz $n"; P="--dimension 3 --domain3d 0 1 0 1 0 1 --tracking-box3d 0 1 0 1 0 1";;
           *) N="--nx $n --nz $n"; P="--domain 0 1 0 1 --tracking-box 0 1 0 1";; esac
[ -e $J/${tag}_ops_M_full.coo ] || $X --mesh-type $m $P --distributed-solve --control-full-domain --state-bc dirichlet $N --steps 2 \
   --pc-schur-sweep --no-paraview --dump-operators $J/${tag}_ops > /dev/null 2>&1
for nt in $nts; do
  t=${tag}_${s}_n${n}_nt${nt}
  PETSC_OPTIONS="$LU" /usr/bin/time -f "MAXRSS=%M wall=%e" $X --exact-oc $c --solve-only --mesh-type $m $N --steps $nt \
    --time-scheme $s --pc-schur-sweep --no-paraview --dump-json $J/$t.json "$@" > $D/logs/runs/$t.log 2>&1
  echo "$t exit=$? $(grep -o 'MAXRSS=.*' $D/logs/runs/$t.log)"
done
