#!/bin/bash
# T15 solver audit (production rtol 1e-8, --solve-only), n = 8, N = 32, HIGH Pe, Galerkin (gal) vs LPS-P0 (lps):
#   dA  distributed, homogeneous Dirichlet state, conv-sin3-3d data, nu = 1e-3, wind A
#   dH  the same, harmonic-3d
#   nA  distributed, natural state (production, unit cube, sin_x target), nu = 1e-3, wind A
#   mA  Neumann control zmin, natural state (production), nu = 1e-3, wind A
#   SCHUR_SWEEP with LU / GMRES+GAMG / GMRES+ASM(4,1)+ILU(1) spatial solves.  No tuning.  Wall cap 600 s per run
#   (exit 124 = not converged within the cap; recorded as such).
#   rA / rH  Dirichlet control zmin,xmin (production), nu = 1e-2, wind A / harmonic-3d: SCHUR_DIRICHLET_REDUCED,
#   iterative CG core (default) and dense core.
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
D=$(cd "$(dirname "$0")/.." && pwd)
W=${T15_WORK:-/tmp/t15_solver}; J=${T15_JSON:-/tmp/t15_solver_json}; mkdir -p $W/output $D/logs/solver $J; cd $W
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
LU="-sweep_right_pc_type lu -sweep_left_pc_type lu -sweep_ay_pc_type lu -fieldsplit_y_sweep_pc_type lu"
ASM=""
for p in sweep_right sweep_left sweep_ay fieldsplit_y_sweep; do
  ASM="$ASM -${p}_pc_type asm -${p}_pc_asm_blocks 4 -${p}_pc_asm_overlap 1 -${p}_sub_pc_type ilu -${p}_sub_pc_factor_levels 1"
done
N8="--solve-only --nx 8 --ny 8 --nz 8 --steps 32 --no-paraview"
A="--convection on --vx 1 --vy 0.5 --vz -0.75"
PR="--dimension 3 --nu 1e-3 --alpha-u 1e-2 --target-type sin_x_minus_t --T 1"
for c in "dA:--exact-oc conv-sin3-3d --exact-oc-nu 1e-3 $A" \
         "dH:--exact-oc conv-sin3-3d --exact-oc-nu 1e-3 --convection on --wind harmonic-3d" \
         "nA:$PR --distributed-solve --control-full-domain --tracking-box3d 0 1 0 1 0 1 --state-bc natural $A" \
         "mA:$PR --neumann-solve --state-bc natural --tracking-box3d 0.2 0.8 0.2 0.8 0.2 0.8 --neumann-face3d zmin $A"; do
  for st in gal lps; do
    stab=""; [ $st = lps ] && stab="--convection-stabilization lps"
    for s in be cn; do for bk in lu gamg asmilu; do
      case $bk in lu) o="$LU";; gamg) o="";; asmilu) o="$ASM";; esac
      tag=${c%%:*}_${st}_${s}_$bk
      PETSC_OPTIONS="$o" timeout 600 /usr/bin/time -f "MAXRSS=%M wall=%e" $X ${c#*:} $stab $N8 --time-scheme $s --pc-schur-sweep --dump-json $J/$tag.json > $D/logs/solver/$tag.log 2>&1
      echo "$tag exit=$?"
    done; done
  done
done
PD="--dimension 3 --dirichlet-solve --state-bc dirichlet --dirichlet-face3d zmin,xmin --nu 1e-2 --alpha-u 1e-2 --target-type sin_x_minus_t --T 1 --tracking-box3d 0.25 0.75 0.25 0.75 0.25 0.75 --nx 8 --ny 8 --nz 8 --steps 32 --no-paraview"
for w in "rA:$A" "rH:--convection on --wind harmonic-3d"; do
  for st in gal lps; do
    stab=""; [ $st = lps ] && stab="--convection-stabilization lps"
    for s in be cn; do for core in iterative dense; do
      tag=${w%%:*}_${st}_${s}_$core
      timeout 600 /usr/bin/time -f "MAXRSS=%M wall=%e" $X $PD ${w#*:} $stab --time-scheme $s --dirichlet-core-solver $core --dump-json $J/$tag.json > $D/logs/solver/$tag.log 2>&1
      echo "$tag exit=$?"
    done; done
  done
done
echo T15_SOLVER_DONE
