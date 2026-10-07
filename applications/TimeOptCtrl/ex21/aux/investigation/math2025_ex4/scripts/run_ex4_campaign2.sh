#!/bin/bash
# Example 4, stage 2: spatial refinement across the alpha_u sweep, and the
# matching alpha_u sweep on Example 3 (BE) for the colocated/noncolocated contrast.
E=$(cd "$(dirname "$0")/.." && pwd)
X=/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
J=$WORK/jobs2.txt; : > $J
add(){ echo "$1|$2|$3" >> $J; }      # preset | name | args
for a in 1e-3 1e-5 1e-7 1e-9 1e-11; do
  for nx in 80 160 320; do
    add ex4 sweep_cn_a${a}_nx${nx}_nt1200 "--time-scheme cn --nx $nx --steps 1200 --alpha-u $a --pc-schur-sweep --ex4-initial x1mx"
  done
  add ex4 sweep_cn_a${a}_nx640_nt1200 "--time-scheme cn --nx 640 --steps 1200 --alpha-u $a --pc-schur-sweep --ex4-initial x1mx"
done
# Example 3 contrast: same alpha_u sweep (alpha_u override is not allowed by the
# Ex.3 preset, so it is passed through the sensitivity path below)
for a in 1e-3 1e-5 1e-7 1e-9 1e-11; do
  for nx in 40 160; do
    add ex3 ex3sweep_be_a${a}_nx${nx}_nt800 "--time-scheme be --nx $nx --steps 800 --pc-schur-sweep --ex3-initial zero --ex3-alpha-u $a"
  done
done
run(){ IFS='|' read -r preset name args <<< "$1"; d=$WORK/$name; mkdir -p $d/output; cd $d
  echo "CMD: TimeOptCtrl_ex21 --math2025-$preset $args --no-paraview" > $E/logs/$name.log
  /usr/bin/time -f "WALL %e s MAXRSS %M kB" $X --math2025-$preset $args --no-paraview >> $E/logs/$name.log 2>&1
  echo "EXIT $?" >> $E/logs/$name.log; }
export -f run; export X E WORK
tr '\n' '\0' < $J | xargs -0 -P5 -I{} bash -c 'run "$@"' _ {}
echo CAMPAIGN2_DONE
