#!/bin/bash
# T7F region-mask checks from --dump-operators (serial): control / tracking /
# source unions, overlap, centroid selection.  Analysis: t7f_masks.py.
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
D=$(cd "$(dirname "$0")/.." && pwd)
W=${T7F_WORK:-/tmp/t7f_masks}; mkdir -p $W/output; cd $W
d() { local tag=$1 nx=$2; shift 2
  $X --distributed-solve --convection on --vx 1 --vy 0.5 --nx $nx --nz $nx --steps 2 --no-paraview \
     --dump-operators $W/$tag "$@" > $W/$tag.log 2>&1 || echo "FAIL $tag"; }
for nx in 8 16; do
  d c1_$nx $nx --control-box 0.125 0.375 0.125 0.375 --tracking-box 0.5 0.875 0.125 0.375
  d c2_$nx $nx --control-box 0.625 0.875 0.625 0.875 --tracking-box 0.125 0.5 0.5 0.875
  d mz_$nx $nx --control-box 0.125 0.375 0.125 0.375 --control-box-add 0.625 0.875 0.625 0.875 \
               --tracking-box 0.5 0.875 0.125 0.375 --tracking-box-add 0.125 0.5 0.5 0.875
  # overlap: A = [1/8,1/2]x[1/8,3/8], B = [3/8,7/8]x[1/8,3/8]; union = one box
  d ovA_$nx $nx --control-box 0.125 0.5 0.125 0.375 --tracking-box 0.125 0.5 0.125 0.375
  d ovB_$nx $nx --control-box 0.375 0.875 0.125 0.375 --tracking-box 0.375 0.875 0.125 0.375
  d ovAB_$nx $nx --control-box 0.125 0.5 0.125 0.375 --control-box-add 0.375 0.875 0.125 0.375 \
                 --tracking-box 0.375 0.875 0.125 0.375 --tracking-box-add 0.125 0.5 0.125 0.375 \
                 --source 1 --source-box-add 0.125 0.5 0.125 0.375 --source-box-add 0.375 0.875 0.125 0.375
  d ovU_$nx $nx --control-box 0.125 0.875 0.125 0.375 --tracking-box 0.125 0.875 0.125 0.375 \
                --source 1 --source-box-add 0.125 0.875 0.125 0.375
  # sources: two disjoint boxes, each alone
  d s1_$nx $nx --source 1 --source-box-add 0 0.25 0.75 1
  d s2_$nx $nx --source 1 --source-box-add 0.75 1 0 0.25
  d s12_$nx $nx --source 1 --source-box-add 0 0.25 0.75 1 --source-box-add 0.75 1 0 0.25
  # centroid selection on a non-aligned box (0.3, 0.7)^2
  d na_$nx $nx --control-box 0.3 0.7 0.3 0.7 --tracking-box 0.3 0.7 0.3 0.7
done
echo masks-done
