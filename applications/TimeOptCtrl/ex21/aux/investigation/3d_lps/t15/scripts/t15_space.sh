#!/bin/bash
# T15 spatial ladders, exact conv-sin3-3d (distributed, homogeneous Dirichlet state), CN N = 64.
#   gF / l0F / l1F : Galerkin / LPS-P0 / LPS-P1, FIXED nu = 1e-3, wind A = (1, 0.5, -0.75)   (all patches active)
#   gP / l0P / l1P : FIXED patch Peclet Pe_K = 10: nu = 0.0906 / n (unit cube, chord 1.346 h, |w| = 1.346)
#   gH / l0H       : Galerkin / LPS-P0, harmonic-3d wind, fixed nu = 1e-3
# usage: t15_space.sh "<n list>"
S=$(cd "$(dirname "$0")" && pwd); ns=${1:-"2 3 4 6 8"}
A="--convection on --vx 1 --vy 0.5 --vz -0.75"; H="--convection on --wind harmonic-3d"
L0="--convection-stabilization lps"; L1="--convection-stabilization lps --lps-projection p1"
for n in $ns; do
  nuP=$(python3 -c "print(f'{0.0906/$n:.6g}')")
  E="--exact-oc conv-sin3-3d"
  bash $S/t15_exact.sh gF  cn $n 64 $E --exact-oc-nu 1e-3 $A
  bash $S/t15_exact.sh l0F cn $n 64 $E --exact-oc-nu 1e-3 $A $L0
  bash $S/t15_exact.sh l1F cn $n 64 $E --exact-oc-nu 1e-3 $A $L1
  bash $S/t15_exact.sh gP  cn $n 64 $E --exact-oc-nu $nuP $A
  bash $S/t15_exact.sh l0P cn $n 64 $E --exact-oc-nu $nuP $A $L0
  bash $S/t15_exact.sh l1P cn $n 64 $E --exact-oc-nu $nuP $A $L1
  bash $S/t15_exact.sh gH  cn $n 64 $E --exact-oc-nu 1e-3 $H
  bash $S/t15_exact.sh l0H cn $n 64 $E --exact-oc-nu 1e-3 $H $L0
done
echo T15_SPACE_DONE
