#!/bin/bash
# T10 spatial-backend check (SCHUR_SWEEP, outer rtol 1e-12): the octant multi-zone
# case (natural BC, wind A, 2 sources, T = 1), n = 8, N = 16, BE and CN, with LU,
# GMRES+GAMG (default) and GMRES+ASM(4, overlap 1)/ILU(1) spatial solves.
D=$(cd "$(dirname "$0")" && pwd)
MZ="--distributed-solve --dimension 3 --T 1 --nu 0.25 --alpha-u 1e-2 --target-type sin_x_minus_t --state-bc natural
    --convection on --vx 1 --vy 0.5 --vz -0.75
    --control-box3d 0 0.5 0 0.5 0 0.5 --control-box3d-add 0.5 1 0.5 1 0.5 1
    --tracking-box3d 0 0.5 0.5 1 0 0.5 --tracking-box3d-add 0.5 1 0 0.5 0.5 1
    --source 1 --source-box3d-add 0 0.5 0.5 1 0.5 1 --source-box3d-add 0.5 1 0 0.5 0 0.5"
ASM="-ksp_rtol 1e-12"
for p in sweep_right sweep_left sweep_ay fieldsplit_y_sweep; do
  ASM="$ASM -${p}_pc_type asm -${p}_pc_asm_blocks 4 -${p}_pc_asm_overlap 1 -${p}_sub_pc_type ilu -${p}_sub_pc_factor_levels 1"
done
for s in be cn; do
  bash $D/t10_runs.sh bk_lu $s 8 16 $MZ
  T10_PETSC="-ksp_rtol 1e-12" bash $D/t10_runs.sh bk_gamg $s 8 16 $MZ
  T10_PETSC="$ASM" bash $D/t10_runs.sh bk_asm $s 8 16 $MZ
done
