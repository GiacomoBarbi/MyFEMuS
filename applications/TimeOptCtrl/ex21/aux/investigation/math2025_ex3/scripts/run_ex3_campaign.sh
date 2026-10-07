#!/bin/bash
# Mathematics 2025 Example 3 campaign.  WORK must be set; logs go to ../logs.
E=$(cd "$(dirname "$0")/.." && pwd)
X=/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
sha256sum $X > $E/logs/binary.sha256
J=$WORK/jobs.txt; : > $J
add(){ echo "$1|$2" >> $J; }      # name | arguments
for nt in 400 800 1600 3200; do
  add cn_sweep_nx40_nt$nt "--time-scheme cn --pc-schur-sweep --nx 40 --steps $nt --ex3-initial zero"
  add be_sweep_nx40_nt$nt "--time-scheme be --pc-schur-sweep --nx 40 --steps $nt --ex3-initial zero"
done
for nx in 10 20 80; do
  add cn_sweep_nx${nx}_nt1600 "--time-scheme cn --pc-schur-sweep --nx $nx --steps 1600 --ex3-initial zero"
done
add cn_additive_nx40_nt800 "--time-scheme cn --nx 40 --steps 800 --ex3-initial zero"
add be_additive_nx40_nt800 "--time-scheme be --nx 40 --steps 800 --ex3-initial zero"
add be_matching_nx40_nt800 "--time-scheme be --pc-schur-matching --nx 40 --steps 800 --ex3-initial zero"
add cn_sweep_nx40_nt1600_icsource "--time-scheme cn --pc-schur-sweep --nx 40 --steps 1600 --ex3-initial source"
for av in "1 1" "2 0" "2 1" "4 2"; do
  set -- $av
  add cn_sweep_nx40_nt800_a$1_v$2 "--time-scheme cn --pc-schur-sweep --nx 40 --steps 800 --ex3-initial zero --ex3-coefficients $1 $2"
done
add cn_sweep_nx40_T25_nt2000 "--time-scheme cn --pc-schur-sweep --nx 40 --steps 2000 --T 25 --ex3-initial zero"
add verify_cn "--verify-convection --time-scheme cn --pc-schur-sweep --nx 10 --steps 40 --T 2 --ex3-initial zero"
add verify_be "--verify-convection --time-scheme be --pc-schur-sweep --nx 10 --steps 40 --T 2 --ex3-initial zero"
run(){ name=${1%%|*}; args=${1#*|}; d=$WORK/$name; mkdir -p $d/output; cd $d
  echo "CMD: TimeOptCtrl_ex21 --math2025-ex3 $args --no-paraview" > $E/logs/$name.log
  /usr/bin/time -f "WALL %e s MAXRSS %M kB" $X --math2025-ex3 $args --no-paraview >> $E/logs/$name.log 2>&1
  echo "EXIT $?" >> $E/logs/$name.log; }
export -f run; export X E WORK
tr '\n' '\0' < $J | xargs -0 -P4 -I{} bash -c 'run "$@"' _ {}
echo CAMPAIGN_DONE
