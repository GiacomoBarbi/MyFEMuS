#!/bin/bash
# T9 3D solver baseline (verify only): SCHUR_SWEEP with three spatial backends
# on the production path (--exact-oc conv-sin3-3d --solve-only), n = 8, N = 32,
# wind (1, 0.5, -0.75): nu = 0.25 (Pe_cell ~0.34) and nu = 0.02 (Pe_cell ~4.2).
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
D=$(cd "$(dirname "$0")/.." && pwd)
W=${T9_WORK:-/tmp/t9_solver}; J=${T9_JSON:-/tmp/t9_solver_json}; mkdir -p $W/output $D/logs/solver $J; cd $W
LU="-sweep_right_pc_type lu -sweep_left_pc_type lu -sweep_ay_pc_type lu -fieldsplit_y_sweep_pc_type lu"
ASM=""
for p in sweep_right sweep_left sweep_ay fieldsplit_y_sweep; do
  ASM="$ASM -${p}_pc_type asm -${p}_pc_asm_blocks 4 -${p}_pc_asm_overlap 1 -${p}_sub_pc_type ilu -${p}_sub_pc_factor_levels 1"
done
for s in be cn; do for nu in 0.25 0.02; do for bk in "lu" "gamg" "asmilu"; do
  case $bk in lu) o="$LU";; gamg) o="";; asmilu) o="$ASM";; esac
  tag=${s}_nu${nu}_$bk
  PETSC_OPTIONS="$o" OMP_NUM_THREADS=1 /usr/bin/time -f "MAXRSS=%M wall=%e" $X --exact-oc conv-sin3-3d --exact-oc-nu $nu \
    --solve-only --nx 8 --ny 8 --nz 8 --steps 32 --time-scheme $s --pc-schur-sweep --convection on --vx 1 --vy 0.5 --vz -0.75 \
    --no-paraview --dump-json $J/$tag.json > $D/logs/solver/$tag.log 2>&1
  echo "$tag exit=$?"
done; done; done
# one tiny ADDITIVE_GAMG BE smoke (the default distributed preconditioner; CN is refused in 3D)
OMP_NUM_THREADS=1 /usr/bin/time -f "MAXRSS=%M wall=%e" $X --exact-oc conv-sin3-3d --solve-only --nx 4 --ny 4 --nz 4 --steps 16 \
  --time-scheme be --convection on --vx 1 --vy 0.5 --vz -0.75 --no-paraview --dump-json $J/additive_be_n4.json > $D/logs/solver/additive_be_n4.log 2>&1
echo "additive_be_n4 exit=$?"
OMP_NUM_THREADS=1 $X --exact-oc conv-sin3-3d --solve-only --nx 4 --ny 4 --nz 4 --steps 16 --time-scheme cn \
  --convection on --vx 1 --vy 0.5 --vz -0.75 --no-paraview > $D/logs/solver/additive_cn_refused.log 2>&1
echo "additive_cn_refused exit=$? $(grep -o 'what.*' $D/logs/solver/additive_cn_refused.log)"
