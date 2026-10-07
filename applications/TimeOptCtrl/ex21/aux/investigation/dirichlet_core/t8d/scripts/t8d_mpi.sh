#!/bin/bash
# T8D MPI np = 1/2/4: iterative core (and dense, to show its per-rank
# replication) on the T7D exact data through the production path, CN.
# Per-rank MaxRSS: every rank runs under /usr/bin/time.
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
D=$(cd "$(dirname "$0")/.." && pwd)
W=${T8D_WORK:-/tmp/t8d_mpi}; J=${T8D_JSON:-/tmp/t8d_mpi_json}; mkdir -p $W/output $D/logs/mpi $J; cd $W
SPECS=${T8D_MPI_SPECS:-iterative:32:128 dense:32:128 iterative:48:256}
for spec in $SPECS; do
  IFS=: read mode nx nt <<< "$spec"
  for np in 1 2 4; do
    tag=${mode}_nx${nx}_nt${nt}_np$np
    OMP_NUM_THREADS=1 mpirun -np $np /usr/bin/time -f "RANK_MAXRSS=%M wall=%e" $X --exact-oc dirichlet-conv --solve-only \
      --nx $nx --nz $nx --steps $nt --time-scheme cn --dirichlet-core-solver $mode --no-paraview \
      --dump-json $J/$tag.json $T8D_EXTRA > $D/logs/mpi/$tag.log 2>&1
    echo "$tag exit=$? $(grep -o 'RANK_MAXRSS=[0-9]*' $D/logs/mpi/$tag.log | tr '\n' ' ')"
  done
done
