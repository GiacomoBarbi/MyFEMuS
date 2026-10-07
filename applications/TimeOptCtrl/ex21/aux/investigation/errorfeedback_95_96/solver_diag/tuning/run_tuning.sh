#!/bin/bash
# EF 9.6 small-alpha solver tuning (cheap proxy).  Usage:
#   run_tuning.sh <label> <alpha_u> <N> <outer_max_it> <timeout_s> "<extra PETSc options>"
# Real EF 9.6 spatial problem (nx 32 x nz 12, CN, SCHUR_SWEEP, np = 1); only
# PETSc runtime options change (outer: no prefix; Schur split: fieldsplit_schur_).
set -u
BIN=/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21
HERE=$(cd "$(dirname "$0")" && pwd); RUN=${RUN:-/tmp/ex21_tuning}
mkdir -p $RUN/output $HERE/logs; cd $RUN
label=$1 a=$2 n=$3 maxit=$4 cap=$5 extra=${6:-}
log=$HERE/logs/${label}_a${a}_N${n}.log
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
PETSC_OPTIONS="-ksp_max_it $maxit -ksp_monitor -fieldsplit_schur_ksp_converged_reason -options_left $extra" \
  /usr/bin/time -f "WALL %e" timeout -s TERM $cap $BIN --errorfeedback-96 --ef-initial sinsin --alpha-u $a \
  --nx 32 --nz 12 --steps $n --T 20 --time-scheme cn --pc-schur-sweep --no-paraview > $log 2>&1
echo "$label a=$a N=$n exit=$?"
