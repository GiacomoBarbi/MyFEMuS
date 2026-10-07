#!/bin/bash
# T8D memory / time ladder: dense vs iterative core, production path on the
# T7D exact data (--exact-oc dirichlet-conv --solve-only; CN, w = (0, 0.5)),
# serial, one run at a time.  usage: t8d_memory.sh "<mode:nx:nt> ..."
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
D=$(cd "$(dirname "$0")/.." && pwd)
W=${T8D_WORK:-/tmp/t8d_memory}; mkdir -p $W/output $D/logs/memory; cd $W
for spec in $1; do
  IFS=: read mode nx nt <<< "$spec"
  tag=${mode}_nx${nx}_nt${nt}
  OMP_NUM_THREADS=1 /usr/bin/time -f "MAXRSS=%M wall=%e" $X --exact-oc dirichlet-conv --solve-only --nx $nx --nz $nx \
     --steps $nt --time-scheme cn --dirichlet-core-solver $mode --no-paraview $T8D_EXTRA > $D/logs/memory/$tag.log 2>&1
  echo "$tag exit=$? $(grep -o 'MAXRSS=.*' $D/logs/memory/$tag.log)"
done
