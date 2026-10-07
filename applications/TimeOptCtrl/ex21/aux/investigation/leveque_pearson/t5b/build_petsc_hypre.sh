#!/bin/bash
# T5b: separate PETSC_ARCH with hypre.  Identical to arch-linux-c-openblas-opt
# (lib/petsc/conf/reconfigure-arch-linux-c-openblas-opt.py) except:
#   + --download-hypre                      (hypre 2.32.0, the version PETSc 3.23.5 pins)
#   + --with-mpi-dir=<existing arch>        (reuse its OpenMPI 5.0.7 instead of --download-openmpi)
#   package sources from PKGS (pre-downloaded; bitbucket.org is not reachable from the build
#   sandbox, so METIS/ParMETIS tarballs were exported with `git archive` from the existing
#   arch's git.metis / git.parmetis clones at the pinned tags v5.1.0-p12 / v4.0.3-p9).
# The existing arch is not modified.   Usage: PKGS=<dir with tarballs> build_petsc_hypre.sh
set -e
PKGS=${PKGS:?set PKGS}
cd /home/giacomo/software/petsc
env -i HOME=$HOME PATH=/usr/bin:/bin /usr/bin/python3.11 ./configure \
  PETSC_ARCH=arch-linux-c-openblas-hypre \
  --with-mpi-dir=/home/giacomo/software/petsc/arch-linux-c-openblas-opt \
  --download-hypre=$PKGS/v2.32.0.tar.gz \
  --download-blacs=1 \
  --download-hdf5=$PKGS/hdf5-1.14.3-p1.tar.bz2 \
  --download-metis=$PKGS/pkg-metis-v5.1.0-p12.tar.gz \
  --download-parmetis=$PKGS/pkg-parmetis-v4.0.3-p9.tar.gz \
  --download-mumps=$PKGS/MUMPS_5.7.3.tar.gz \
  --download-scalapack=$PKGS/0e8767285b7a201c7b1ff34d2c2bb009534145df.tar.gz \
  --download-suitesparse=$PKGS/v7.10.1.tar.gz \
  '--with-blaslapack-lib=-L/usr/lib64 -lopenblas' \
  --with-debugging=0 --with-shared-libraries=1 --with-x=1 \
  'COPTFLAGS=-O3 -march=native -mtune=native' 'CXXOPTFLAGS=-O3 -march=native -mtune=native' \
  'FOPTFLAGS=-O3 -march=native -mtune=native'
env -i HOME=$HOME PATH=/home/giacomo/software/petsc/arch-linux-c-openblas-opt/bin:/usr/bin:/bin \
  make PETSC_DIR=/home/giacomo/software/petsc PETSC_ARCH=arch-linux-c-openblas-hypre all
echo PETSC_HYPRE_BUILD_DONE
