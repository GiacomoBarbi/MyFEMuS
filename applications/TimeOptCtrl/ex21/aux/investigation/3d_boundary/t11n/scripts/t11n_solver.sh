#!/bin/bash
# T11N solver audit, 3D Neumann control (production path, production outer rtol
# 1e-8), neumann-conv-3d data (--solve-only), Gamma_c = zmin, n = 8, N = 32,
# wind (1, 0.5, -0.75) (Pe_cell ~0.34): SCHUR_SWEEP with LU / GMRES+GAMG /
# GMRES+ASM(4,1)+ILU(1); ADDITIVE_GAMG (BE: the default distributed PC); and the
# refusal messages of SCHUR_MATCHING and SCHUR_BOUNDARY_REDUCED (with and
# without convection) in 3D.
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
D=$(cd "$(dirname "$0")/.." && pwd)
W=${T11N_WORK:-/tmp/t11n_solver}; J=${T11N_JSON:-/tmp/t11n_solver_json}; mkdir -p $W/output $D/logs/solver $J; cd $W
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
LU="-sweep_right_pc_type lu -sweep_left_pc_type lu -sweep_ay_pc_type lu -fieldsplit_y_sweep_pc_type lu"
ASM=""
for p in sweep_right sweep_left sweep_ay fieldsplit_y_sweep; do
  ASM="$ASM -${p}_pc_type asm -${p}_pc_asm_blocks 4 -${p}_pc_asm_overlap 1 -${p}_sub_pc_type ilu -${p}_sub_pc_factor_levels 1"
done
C="--exact-oc neumann-conv-3d --solve-only --nx 8 --ny 8 --nz 8 --steps 32 --convection on --vx 1 --vy 0.5 --vz -0.75 --no-paraview"
for s in be cn; do for bk in lu gamg asmilu; do
  case $bk in lu) o="$LU";; gamg) o="";; asmilu) o="$ASM";; esac
  tag=${s}_sweep_$bk
  PETSC_OPTIONS="$o" timeout 1800 /usr/bin/time -f "MAXRSS=%M wall=%e" $X $C --time-scheme $s --pc-schur-sweep --dump-json $J/$tag.json > $D/logs/solver/$tag.log 2>&1
  echo "$tag exit=$?"
done; done
PETSC_OPTIONS="" timeout 1800 /usr/bin/time -f "MAXRSS=%M wall=%e" $X $C --time-scheme be --dump-json $J/be_additive.json > $D/logs/solver/be_additive.log 2>&1
echo "be_additive exit=$?"
for r in "cn_additive:--time-scheme cn" "be_matching:--time-scheme be --pc-schur-matching" \
         "be_boundary_reduced:--time-scheme be --pc-schur-boundary-reduced"; do
  $X $C ${r#*:} > $D/logs/solver/${r%%:*}.log 2>&1; echo "${r%%:*} exit=$? $(grep -o 'what.*' $D/logs/solver/${r%%:*}.log)"
done
$X --exact-oc neumann-conv-3d --solve-only --nx 4 --ny 4 --nz 4 --steps 8 --convection off --no-paraview --time-scheme be --pc-schur-boundary-reduced > $D/logs/solver/heat_boundary_reduced.log 2>&1
echo "heat_boundary_reduced exit=$? $(grep -o 'what.*' $D/logs/solver/heat_boundary_reduced.log)"
