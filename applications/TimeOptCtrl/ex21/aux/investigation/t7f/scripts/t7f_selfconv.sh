#!/bin/bash
# T7F fine-reference self-convergence runs (production path, JSON fields kept
# OUTSIDE the repository in $T7F_JSON).  SCHUR_SWEEP + LU spatial solves.
# usage: t7f_selfconv.sh <case> <scheme> "<nx list>" "<nt list>"
#   case nat1 / nat2: natural-conv exact data through the production path
#   case mz: two control boxes + two non-colocated tracking boxes (1/8 lattice),
#            natural state BC, w = (1, 0.5), target sin_x_minus_t, T = 1
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
D=$(cd "$(dirname "$0")/.." && pwd)
case_=$1; s=$2; nxs=$3; nts=$4
LU="-sweep_right_pc_type lu -sweep_left_pc_type lu -sweep_ay_pc_type lu -fieldsplit_y_sweep_pc_type lu"
MZ="--control-box 0.125 0.375 0.125 0.375 --control-box-add 0.625 0.875 0.625 0.875
    --tracking-box 0.5 0.875 0.125 0.375 --tracking-box-add 0.125 0.5 0.5 0.875"
case $case_ in
  mz) data="--distributed-solve --nu 0.25 --alpha-u 1e-2 --target-type sin_x_minus_t --state-bc natural
            --convection on --vx 1 --vy 0.5 --T 1 $MZ" ;;
  nat1) data="--exact-oc natural-conv --solve-only" ;;                            # w = (1, 0.5)
  nat2) data="--exact-oc natural-conv --solve-only --exact-oc-velocity -1 -0.5" ;;  # reversed
  *) echo "unknown case"; exit 1 ;;
esac
W=${T7F_WORK:-/tmp/t7f_sc_$case_$s}; J=${T7F_JSON:-/tmp/t7f_selfconv_json}
mkdir -p $W/output $D/logs/selfconv $J; cd $W
for nx in $nxs; do for nt in $nts; do
  tag=${case_}_${s}_nx${nx}_nt${nt}
  PETSC_OPTIONS="$LU" OMP_NUM_THREADS=1 /usr/bin/time -f "MAXRSS=%M wall=%e" $X $data --pc-schur-sweep \
    --steps $nt --nx $nx --nz $nx --time-scheme $s --no-paraview --dump-json $J/$tag.json > $D/logs/selfconv/$tag.log 2>&1
  echo "$tag exit=$? $(grep -o 'MAXRSS.*' $D/logs/selfconv/$tag.log)"
done; done
