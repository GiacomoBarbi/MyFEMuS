#!/bin/bash
# T9: private, uncontaminated build of ex21 OUTSIDE the shared build tree.
# The shared femusbin-control CMake cache was re-configured (12:03, automatic
# re-run after CMakeLists.txt gained ExactOc3D.cpp) from a conda-activated shell:
# pkg-config found miniforge's SLEPc 3.24, whose static link list put a second
# PETSc (3.24), HYPRE, MUMPS, LAPACK, BLAS ... on the ex21 link line (and
# -isystem miniforge3/include on the compile line).  This script recompiles the
# CURRENT ex21 sources with the same flags MINUS every miniforge token and links
# with the shared tree's library list MINUS every miniforge token, RUNPATH =
# the frozen binary's.  Nothing in the build tree is written.   WORK=<dir>
# The shared libfemus.so was rebuilt (12:04) under the same cache and now has
# undefined SLEPc references (SVD*, never called by ex21); the frozen binary
# runs against that same libfemus.so through lazy binding, so the clean binary is
# linked with --allow-shlib-undefined to be in exactly the same runtime situation.
set -eu
BT=/home/giacomo/software/FEMuS/femusbin-control; BD=$BT/applications/TimeOptCtrl/ex21
SRC=/home/giacomo/software/MyFEMuS-control/applications/TimeOptCtrl/ex21
W=${WORK:?set WORK}; mkdir -p $W/obj
F=$BD/CMakeFiles/TimeOptCtrl_ex21.dir/flags.make
get() { grep -E "^$1 = " $F | sed "s/^$1 = //"; }
INC=$(get CXX_INCLUDES | sed -E 's#-isystem /home/giacomo/miniforge3/include##g; s#-I/home/giacomo/miniforge3[^ ]*##g')
FLAGS="$(get CXX_FLAGS) $(get CXX_DEFINES)"
echo "$INC $FLAGS" | grep -q miniforge && { echo "miniforge left in compile flags"; exit 1; }
export PATH=/home/giacomo/software/petsc/arch-linux-c-openblas-opt/bin:/usr/bin:/bin
objs=()
for o in $(tr ' ' '\n' < $BD/CMakeFiles/TimeOptCtrl_ex21.dir/link.txt | grep '\.cpp\.o$'); do
  s=${o#CMakeFiles/TimeOptCtrl_ex21.dir/}; s=$SRC/${s%.o}; out=$W/obj/$(basename $s).o
  eval /usr/bin/c++ $FLAGS $INC -o $out -c $s &
  objs+=($out)
done; wait
LIBS=$(tr ' ' '\n' < $BD/CMakeFiles/TimeOptCtrl_ex21.dir/link.txt | sed -n '/^-o$/,$p' | tail -n +3 \
  | grep -v miniforge | grep -v '^-Wl,-rpath' | grep -v '^$' | sed "s#^\.\./\.\./\.\./#$BT/#" | tr '\n' ' ')
/usr/bin/c++ -O3 -DNDEBUG -march=native -DADEPT_RECORDING_PAUSABLE "${objs[@]}" -o $W/TimeOptCtrl_ex21_clean \
  -Wl,--allow-shlib-undefined -Wl,-rpath,$BT/lib64:/home/giacomo/software/petsc/arch-linux-c-openblas-opt/lib $LIBS
readelf -d $W/TimeOptCtrl_ex21_clean | grep -c miniforge && { echo "miniforge in clean binary"; exit 1; } || true
readelf -d $W/TimeOptCtrl_ex21_clean | grep -E "NEEDED|RUNPATH" | sed -E 's/.*\[(.*)\]/\1/' > $W/clean_needed.txt
sha256sum $W/TimeOptCtrl_ex21_clean
