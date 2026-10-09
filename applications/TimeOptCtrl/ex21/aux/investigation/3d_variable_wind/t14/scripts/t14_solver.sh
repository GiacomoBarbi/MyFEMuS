#!/bin/bash
# T14 solver audit (production rtol 1e-8, --solve-only), n = 8, N = 32, moderate Pe:
#   natural state (distributed control): natural-conv-3d wind A (constant baseline), natural-var-3d harmonic-3d /
#   affine-div-3d; Neumann: neumann-var-3d harmonic-3d.  SCHUR_SWEEP with LU / GMRES+GAMG / GMRES+ASM(4,1)+ILU(1).
#   Dirichlet control (production zmin,xmin, wind A / harmonic / affine-div): SCHUR_DIRICHLET_REDUCED, iterative
#   CG core (default) and dense core.  No tuning.
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
D=$(cd "$(dirname "$0")/.." && pwd)
W=${T14_WORK:-/tmp/t14_solver}; J=${T14_JSON:-/tmp/t14_solver_json}; mkdir -p $W/output $D/logs/solver $J; cd $W
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
LU="-sweep_right_pc_type lu -sweep_left_pc_type lu -sweep_ay_pc_type lu -fieldsplit_y_sweep_pc_type lu"
ASM=""
for p in sweep_right sweep_left sweep_ay fieldsplit_y_sweep; do
  ASM="$ASM -${p}_pc_type asm -${p}_pc_asm_blocks 4 -${p}_pc_asm_overlap 1 -${p}_sub_pc_type ilu -${p}_sub_pc_factor_levels 1"
done
N8="--solve-only --nx 8 --ny 8 --nz 8 --steps 32 --no-paraview"
for c in "natA:--exact-oc natural-conv-3d --convection on --vx 1 --vy 0.5 --vz -0.75" \
         "natH:--exact-oc natural-var-3d --convection on --wind harmonic-3d" \
         "natF:--exact-oc natural-var-3d --convection on --wind affine-div-3d" \
         "neuH:--exact-oc neumann-var-3d --convection on --wind harmonic-3d"; do
  for s in be cn; do for bk in lu gamg asmilu; do
    case $bk in lu) o="$LU";; gamg) o="";; asmilu) o="$ASM";; esac
    tag=${c%%:*}_${s}_$bk
    PETSC_OPTIONS="$o" timeout 2400 /usr/bin/time -f "MAXRSS=%M wall=%e" $X ${c#*:} $N8 --time-scheme $s --pc-schur-sweep --dump-json $J/$tag.json > $D/logs/solver/$tag.log 2>&1
    echo "$tag exit=$?"
  done; done
done
PD="--dimension 3 --dirichlet-solve --state-bc dirichlet --dirichlet-face3d zmin,xmin --nu 0.25 --alpha-u 1e-2 --target-type sin_x_minus_t --T 1 --tracking-box3d 0.25 0.75 0.25 0.75 0.25 0.75 --nx 8 --ny 8 --nz 8 --steps 32 --no-paraview"
for w in "dirA:--convection on --vx 1 --vy 0.5 --vz -0.75" "dirH:--convection on --wind harmonic-3d" "dirF:--convection on --wind affine-div-3d"; do
  for s in be cn; do for core in iterative dense; do
    tag=${w%%:*}_${s}_$core
    timeout 2400 /usr/bin/time -f "MAXRSS=%M wall=%e" $X $PD ${w#*:} --time-scheme $s --dirichlet-core-solver $core --dump-json $J/$tag.json > $D/logs/solver/$tag.log 2>&1
    echo "$tag exit=$?"
  done; done
done
