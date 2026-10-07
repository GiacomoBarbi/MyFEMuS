#!/bin/bash
# T5b reproduction (hypre / BoomerAMG / AIR for the SCHUR_SWEEP spatial blocks).
# 1. separate PETSc arch with hypre (does not touch arch-linux-c-openblas-opt):
#      PKGS=<dir with the tarballs> ./build_petsc_hypre.sh
# 2. separate FEMuS build of ex21 against it (the production build dir is untouched):
#      B=/home/giacomo/software/FEMuS/femusbin-control-hypre; mkdir -p $B; cd $B
#      env -i HOME=$HOME PATH=/usr/bin:/bin PETSC_DIR=/home/giacomo/software/petsc \
#        PETSC_ARCH=arch-linux-c-openblas-hypre cmake /home/giacomo/software/MyFEMuS-control \
#        -DCMAKE_BUILD_TYPE=Release -DPETSC_DIR=/home/giacomo/software/petsc \
#        -DPETSC_ARCH=arch-linux-c-openblas-hypre -DBUILD_TIME_OPTIMAL_CONTROL=ON
#      make -j10 b64 jsoncpp adept && make -j10 TimeOptCtrl_ex21
# 3. campaign (scratch run directory in $WORK; results/ and logs/ next to this script):
set -e
cd "$(dirname "$0")"
W=${WORK_ROOT:-/tmp/t5b_work}
WORK=$W/base   python3 run_t5b.py baseline --jobs 2          # T5 baselines on the hypre binary (must match T5)
WORK=$W/screen python3 run_t5b.py screen --jobs 2            # H0 H1 H2 H2b H3 at A B C D E16
WORK=$W/hier   python3 hierarchy_check.py                    # AMG levels per block
WORK=$W/rs     python3 rowsum_probe.py                       # max_row_sum 0.9 vs 1.0 at the failing points
python3 rowsum_structure.py "${DUMP:?set DUMP to the M/L dump directory}"  # optional: needs matrix dumps
WORK=$W/s1     python3 run_t5b.py screen1 hypre_air1_mrs1 --jobs 2   # H2c at A..E16 + C64 D64 D128
for c in hypre_default hypre_air1 hypre_air1_mrs1; do
  WORK=$W/kkt_$c python3 run_t5b.py kkt $c --jobs 2          # LP 5.2 KKT, CN/BE, eps 1/100, 1/500, nx 16/32/64
done
WORK=$W/mpi python3 mpi_check_t5b.py hypre_air1_mrs1 1,2,4
WORK=$W/mpi python3 mpi_check_t5b.py hypre_air1_mrs1,hypre_air1,hypre_default 4,3 _isolate
python3 analyze_t5b.py                                       # ../t5b_hypre_*.csv, ../figures/t5b_*.png
