#!/bin/bash
# Small MPI check of the PRODUCTION path (--solve-only) on EF 9.6 data:
# alpha_u = 1e-7, nx 32 x nz 12, N = 50, CN, SCHUR_SWEEP; np = 1, 2, 4 run one
# after another.  Separate from the scientific benchmark.
set -u
BIN=/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21
ARCH=/home/giacomo/software/petsc/arch-linux-c-openblas-opt/bin
HERE=$(cd "$(dirname "$0")" && pwd); RUN=${RUN:-/tmp/ex21_mpi}
sha256sum $BIN > $HERE/binary.sha256
for np in 1 2 4; do
  mkdir -p $RUN/np$np/output; cd $RUN/np$np
  env -i HOME=$HOME PATH=$ARCH:/usr/bin:/bin OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    /usr/bin/time -f "WALL %e" $ARCH/mpiexec -n $np $BIN --errorfeedback-96 --solve-only --ef-initial sinsin \
    --alpha-u 1e-7 --nx 32 --nz 12 --steps 50 --T 20 --time-scheme cn --pc-schur-sweep --no-paraview \
    > $HERE/mpi_np$np.log 2>&1
  cp output/ex20_fields.json $HERE/mpi_np$np.json 2>/dev/null
done
echo DONE
