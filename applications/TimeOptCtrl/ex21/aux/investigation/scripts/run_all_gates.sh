#!/bin/bash
# ex21 final campaign on ONE binary: ex20 goldens, zero-convection baseline,
# convection gates (1D/2D x BE/CN x 3 PCs), manufactured ladders.
set -e
I=$(cd "$(dirname "$0")/.." && pwd)
EX20DIR=$(cd "$I/../../../ex20" && pwd)
X=/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21
W=${WORK:?set WORK}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
sha256sum $X > $I/logs/final_binary.sha256
mkdir -p $W/benr $W/base $W/vc/output $W/mms
( cd $EX20DIR && env -i HOME=$HOME PATH=/home/giacomo/software/petsc/arch-linux-c-openblas-opt/bin:/usr/bin:/bin \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 $PY aux/cn_stage1/be_nonregression.py \
    --golden aux/cn_stage1/baseline/refactor_golden_openblas --golden aux/cn_stage1/baseline/cli_golden_openblas \
    --exe $X --work $W/benr ) > $I/logs/ex21_default_vs_ex20_goldens.log 2>&1
$PY $I/scripts/zero_convection_baseline.py $W/base $I/results/zero_convection_baseline.csv > $I/logs/zero_convection_baseline.log 2>&1
cd $W/vc
for sch in be cn; do
  for pc in additive matching sweep; do
    opt=""; [ $pc != additive ] && opt="--pc-schur-$pc"
    $X --verify-convection --distributed-solve --nx 8 --nz 8 --steps 8 --time-scheme $sch --state-bc dirichlet \
       --convection on --vx 2 --vy 1 --nu 0.5 --target-type sin_x_minus_t --alpha-u 1e-3 \
       --control-box 0.25 0.75 0.25 0.75 $opt 2>&1 | grep -E "^EX21_|rror" > $I/logs/verify_convection_2d_${sch}_${pc}.log || true
    $X --verify-convection --dimension 1 --domain 0 2 0 1 --distributed-solve --nx 32 --steps 16 --time-scheme $sch \
       --state-bc dirichlet --convection on --vx 2 --nu 2 --target-type sin_x_minus_t --alpha-u 1e-3 \
       --control-box 0.25 0.95 0 1 --tracking-box 0.55 1.35 0 1 $opt 2>&1 | grep -E "^EX21_|rror" > $I/logs/verify_convection_1d_${sch}_${pc}.log || true
  done
done
$PY $I/scripts/mms_ladders.py $W/mms $I/results/mms_ladders.csv > $I/logs/mms_ladders.log 2>&1
echo ALL_DONE
