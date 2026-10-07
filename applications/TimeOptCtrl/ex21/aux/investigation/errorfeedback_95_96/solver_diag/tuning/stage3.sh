#!/bin/bash
# Stage 3: baseline vs best Stage-1 candidates at N = 50, outer max_it 8, 600 s cap each, sequential.
cd "$(dirname "$0")"
for a in 1e-11 1e-10; do
  ./run_tuning.sh C_rtol1e-3_max30  $a 50 8 600 "-fieldsplit_schur_ksp_rtol 1e-3 -fieldsplit_schur_ksp_max_it 30"
  ./run_tuning.sh B_rtol1e-4_max50  $a 50 8 600 "-fieldsplit_schur_ksp_rtol 1e-4 -fieldsplit_schur_ksp_max_it 50"
  ./run_tuning.sh A_rtol1e-6_max100 $a 50 8 600 "-fieldsplit_schur_ksp_rtol 1e-6 -fieldsplit_schur_ksp_max_it 100"
done
echo STAGE3_DONE
