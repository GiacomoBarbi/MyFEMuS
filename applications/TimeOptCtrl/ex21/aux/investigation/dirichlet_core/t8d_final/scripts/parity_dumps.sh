#!/bin/bash
# Zero-velocity parity, spatial operators: --dump-operators from ex20, ex21
# (convection off) and ex21 (convection on, w = (0, 0)) for distributed /
# Neumann / Dirichlet control.  Comparison: parity_dumps.py.
E=${EX20_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex20/TimeOptCtrl_ex20}
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
D=$(cd "$(dirname "$0")/.." && pwd)
W=${WORK:-/tmp/t8f_dumps}; mkdir -p $W/output; cd $W
for ctl in distributed neumann dirichlet; do
  base="--$ctl-solve --nx 8 --nz 8 --steps 4 --no-paraview"
  $E $base --dump-operators $W/ex20_$ctl > $W/ex20_$ctl.log 2>&1 || echo "ex20 $ctl FAIL"
  $X $base --dump-operators $W/ex21off_$ctl > $W/ex21off_$ctl.log 2>&1 || echo "ex21off $ctl FAIL"
  $X $base --convection on --vx 0 --vy 0 --dump-operators $W/ex21w0_$ctl > $W/ex21w0_$ctl.log 2>&1 || echo "ex21w0 $ctl FAIL"
done
