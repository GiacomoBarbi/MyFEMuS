#!/bin/bash
# Stage 1: Schur-split tolerance / cap variants, N = 25, outer max_it 6, 300 s cap each, sequential.
cd "$(dirname "$0")"
for a in 1e-10 1e-11; do
  ./run_tuning.sh A_rtol1e-6_max100 $a 25 6 300 "-fieldsplit_schur_ksp_rtol 1e-6 -fieldsplit_schur_ksp_max_it 100"
  ./run_tuning.sh B_rtol1e-4_max50  $a 25 6 300 "-fieldsplit_schur_ksp_rtol 1e-4 -fieldsplit_schur_ksp_max_it 50"
  ./run_tuning.sh C_rtol1e-3_max30  $a 25 6 300 "-fieldsplit_schur_ksp_rtol 1e-3 -fieldsplit_schur_ksp_max_it 30"
  ./run_tuning.sh D_rtol1e-6_max50  $a 25 6 300 "-fieldsplit_schur_ksp_rtol 1e-6 -fieldsplit_schur_ksp_max_it 50"
  ./run_tuning.sh E_rtol1e-3_max100 $a 25 6 300 "-fieldsplit_schur_ksp_rtol 1e-3 -fieldsplit_schur_ksp_max_it 100"
done
echo STAGE1_DONE
