#!/bin/bash
# T6 KKT/matrix gates with LPS on: --verify-convection (G1 Galerkin orientation,
# G2 space-time transpose + wrong-adjoint negative control + reduced-Hessian
# symmetry, G3 FD reduced gradient, G3P physical objective (lifted data),
# G4 production KKT solve + independent forward march, G5 LPS matrix gates).
# Sweep solves: LU (all four prefixes).   WORK=<scratch> gates_t6.sh
set -u
H=$(cd "$(dirname "$0")" && pwd); X=${EX21_EXE:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
W=${WORK:?set WORK}; mkdir -p $W/output $H/logs/gates; cd $W
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PETSC_OPTIONS="-sweep_right_pc_type lu -sweep_left_pc_type lu -sweep_ay_pc_type lu -fieldsplit_y_sweep_pc_type lu"
g() { name=$1; shift
  { echo "# $(basename $X) $*"; echo "# PETSC_OPTIONS=$PETSC_OPTIONS"; $X --verify-convection "$@" --distributed-solve --pc-schur-sweep --no-paraview 2>&1 | grep -E "^EX21_|rror"; } > $H/logs/gates/$name.log
  echo "$name: $(grep -c ^EX21_ $H/logs/gates/$name.log) lines"; }
for s in cn be; do
  g vc_lp52_eps20_nx8_${s}_lps   --exact-oc lp52 --nx 8 --nz 8 --steps 16 --time-scheme $s --convection-stabilization lps
  g vc_lp52_eps500_nx8_${s}_lps  --exact-oc lp52 --exact-oc-nu 0.002 --nx 8 --nz 8 --steps 16 --time-scheme $s --convection-stabilization lps
  g vc_q2divfree_nu0.01_nx8_${s}_lps --exact-oc conv-sin3 --wind q2-divfree --exact-oc-nu 0.01 --nx 8 --nz 8 --steps 16 --time-scheme $s --convection-stabilization lps
  g vc_liftconst_q2divfree_nu0.01_nx8_${s}_lps --exact-oc lift-const --wind q2-divfree --exact-oc-nu 0.01 --nx 8 --nz 8 --steps 16 --time-scheme $s --convection-stabilization lps
done
g vc_lp52_eps500_nx8_cn_lps_p1 --exact-oc lp52 --exact-oc-nu 0.002 --nx 8 --nz 8 --steps 16 --time-scheme cn --convection-stabilization lps --lps-projection p1
g vc_lp52_eps500_nx8_cn_none   --exact-oc lp52 --exact-oc-nu 0.002 --nx 8 --nz 8 --steps 16 --time-scheme cn
