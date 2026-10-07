#!/bin/bash
# Feature-parity audit: TINY smoke tests (nx = 8, Nt = 8) of convection features
# that are allowed by Validate() but had no explicit convection validation.
# --verify-convection runs G1 (Galerkin orientation), G2 (space-time transpose,
# wrong-adjoint negative control, reduced-Hessian symmetry), G3 (FD reduced
# gradient), G4 (production KKT solve + independent forward march) [+ G5 LPS].
# LU sweeps.  WORK=<scratch> smoke.sh
set -u
H=$(cd "$(dirname "$0")" && pwd)
X=/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21
MPI=/home/giacomo/software/petsc/arch-linux-c-openblas-opt/bin/mpirun
W=${WORK:?set WORK}; mkdir -p $W/output; cd $W
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PETSC_OPTIONS="-sweep_right_pc_type lu -sweep_left_pc_type lu -sweep_ay_pc_type lu -fieldsplit_y_sweep_pc_type lu"
MZ="--control-box 0.1 0.4 0.1 0.4 --control-box-add 0.6 0.9 0.6 0.9 --tracking-box 0.5 0.9 0.1 0.5 --tracking-box-add 0.1 0.5 0.5 0.9"
COMMON="--distributed-solve --pc-schur-sweep --nx 8 --nz 8 --steps 8 --convection on --vx 2 --vy 1 --target-type sin_x_minus_t --alpha-u 1e-3 --no-paraview"
g() { name=$1; shift
  { echo "# $(basename $X) --verify-convection $*"; timeout 300 $X --verify-convection "$@" 2>&1 | grep -E "^EX21_|rror"; echo "EXIT=${PIPESTATUS[0]}"; } > $H/logs/smoke_$name.log
  echo "$name done"; }
for s in cn be; do
  g multizone_dirichlet_$s $COMMON $MZ --state-bc dirichlet --nu 0.5 --time-scheme $s
  g multizone_natural_$s   $COMMON $MZ --state-bc natural --nu 0.5 --time-scheme $s
done
g multizone_natural_lps_cn $COMMON $MZ --state-bc natural --nu 0.05 --time-scheme cn --convection-stabilization lps
g full_natural_variablewind_cn --distributed-solve --pc-schur-sweep --nx 8 --nz 8 --steps 8 --convection on --wind q2-divfree --control-full-domain --tracking-box 0 1 0 1 --target-type sin_x_minus_t --alpha-u 1e-3 --state-bc natural --nu 0.1 --time-scheme cn --no-paraview
# production CLI path (not a verification driver), MPI np = 1 and 2
for np_ in 1 2; do
  { echo "# mpirun -np $np_ $(basename $X) $COMMON $MZ --state-bc natural --nu 0.5 --time-scheme cn --dump-json $W/prod_np$np_.json"
    timeout 300 $MPI -np $np_ $X $COMMON $MZ --state-bc natural --nu 0.5 --time-scheme cn --dump-json $W/prod_np$np_.json 2>&1 | grep -E "^EX2|rror"; echo "EXIT=${PIPESTATUS[0]}"; } > $H/logs/smoke_production_multizone_natural_cn_np$np_.log
  echo "production np$np_ done"
done
