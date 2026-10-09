#!/bin/bash
# T15 fixed-mesh (n = 4) time ladders with LPS ACTIVE (nu = 1e-2: Pe_K ~ 23 on every patch), production path
# with JSON dumps (t15_runs.sh), self-convergence against nt = 1024.
#   tA0 : exact-case data conv-sin3-3d, wind A, LPS-P0, BE and CN      tAg : the same, Galerkin, CN
#   tH0 : conv-sin3-3d, harmonic-3d, LPS-P0, CN
#   tD0 : Dirichlet boundary control zmin (production sin_x data), wind A, LPS-P0, BE and CN
S=$(cd "$(dirname "$0")" && pwd); NT="4 8 16 32 64 128 1024"
A="--convection on --vx 1 --vy 0.5 --vz -0.75"; L="--convection-stabilization lps"
E="--exact-oc conv-sin3-3d --exact-oc-nu 1e-2"
DIR="--dimension 3 --dirichlet-solve --state-bc dirichlet --dirichlet-face3d zmin --nu 1e-2 --alpha-u 1e-2 --target-type sin_x_minus_t --T 1 --tracking-box3d 0.25 0.75 0.25 0.75 0.5 1"
bash $S/t15_runs.sh tA0 be 4 "$NT" $E $A $L
bash $S/t15_runs.sh tA0 cn 4 "$NT" $E $A $L
bash $S/t15_runs.sh tAg cn 4 "$NT" $E $A
bash $S/t15_runs.sh tH0 cn 4 "$NT" $E --convection on --wind harmonic-3d $L
bash $S/t15_runs.sh tD0 be 4 "$NT" $DIR $A $L
bash $S/t15_runs.sh tD0 cn 4 "$NT" $DIR $A $L
echo T15_TIME_RUNS_DONE
