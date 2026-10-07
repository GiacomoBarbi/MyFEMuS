#!/bin/bash
# Mathematics 2025 Example 4 campaign.  WORK must be set; logs go to ../logs.
E=$(cd "$(dirname "$0")/.." && pwd)
X=/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
sha256sum $X > $E/logs/binary.sha256
J=$WORK/jobs.txt; : > $J
add(){ echo "$1|$2" >> $J; }
B="--pc-schur-sweep --ex4-initial x1mx"
# refinement at alpha_u = 1e-9 (CN): space at dt = 0.025, time at nx = 40
for nx in 20 40 80; do add ref_cn_a1e-9_nx${nx}_nt600 "--time-scheme cn --nx $nx --steps 600 --alpha-u 1e-9 $B"; done
for nt in 300 1200 2400; do add ref_cn_a1e-9_nx40_nt$nt "--time-scheme cn --nx 40 --steps $nt --alpha-u 1e-9 $B"; done
for nt in 1200 2400; do add ref_be_a1e-9_nx40_nt$nt "--time-scheme be --nx 40 --steps $nt --alpha-u 1e-9 $B"; done
# alpha_u sweep at the reference resolution nx = 40, N = 1200 (dt = 0.0125)
for a in 1e-3 1e-5 1e-7 1e-11; do add sweep_cn_a${a}_nx40_nt1200 "--time-scheme cn --nx 40 --steps 1200 --alpha-u $a $B"; done
add ref_cn_a1e-11_nx40_nt2400 "--time-scheme cn --nx 40 --steps 2400 --alpha-u 1e-11 $B"
add ref_cn_a1e-11_nx80_nt1200 "--time-scheme cn --nx 80 --steps 1200 --alpha-u 1e-11 $B"
# IC sensitivity
for a in 1e-9 1e-11; do add ic0_cn_a${a}_nx40_nt1200 "--time-scheme cn --nx 40 --steps 1200 --alpha-u $a --pc-schur-sweep --ex4-initial zero"; done
# gates on this configuration
add verify_cn "--verify-convection --time-scheme cn --nx 20 --steps 40 --T 2 --alpha-u 1e-9 $B"
add verify_be "--verify-convection --time-scheme be --nx 20 --steps 40 --T 2 --alpha-u 1e-9 $B"
run(){ name=${1%%|*}; args=${1#*|}; d=$WORK/$name; mkdir -p $d/output; cd $d
  echo "CMD: TimeOptCtrl_ex21 --math2025-ex4 $args --no-paraview" > $E/logs/$name.log
  /usr/bin/time -f "WALL %e s MAXRSS %M kB" $X --math2025-ex4 $args --no-paraview >> $E/logs/$name.log 2>&1
  echo "EXIT $?" >> $E/logs/$name.log; }
export -f run; export X E WORK
tr '\n' '\0' < $J | xargs -0 -P5 -I{} bash -c 'run "$@"' _ {}
echo CAMPAIGN_DONE
