#!/bin/bash
# T15 high-Pe 3D layer benchmark: distributed control, homogeneous Dirichlet state, constant source f = 1, constant
# target, alpha_u = 1, nu = 1e-3, T = 1, CN N = 16: transport-dominated state with outflow boundary layers.
# Galerkin (gal) / LPS-P0 (p0) / LPS-P1 (p1), wind A (1, 0.5, -0.75) and harmonic-3d, n = 4 6 8 12 (+ 16 for P0).
# SCHUR_SWEEP + LU.  JSON dumps for the extrema (t15_benchmark.py).
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
D=$(cd "$(dirname "$0")/.." && pwd)
W=${T15_WORK:-/tmp/t15_bench}; J=${T15_JSON:-/tmp/t15_bench_json}; mkdir -p $W/output $D/logs/benchmark $J; cd $W
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
LU="-sweep_right_pc_type lu -sweep_left_pc_type lu -sweep_ay_pc_type lu -fieldsplit_y_sweep_pc_type lu -ksp_rtol 1e-10"
B="--dimension 3 --distributed-solve --control-full-domain --tracking-box3d 0 1 0 1 0 1 --state-bc dirichlet --nu 1e-3 --alpha-u 1 --source 1 --steps 16 --T 1 --time-scheme cn --pc-schur-sweep --no-paraview"
for w in "A:--convection on --vx 1 --vy 0.5 --vz -0.75" "H:--convection on --wind harmonic-3d"; do
  for n in 4 6 8 12 16; do
    for m in gal p0 p1; do
      [ $n = 16 ] && [ $m != p0 ] && continue
      case $m in gal) st="";; p0) st="--convection-stabilization lps";; p1) st="--convection-stabilization lps --lps-projection p1";; esac
      tag=${w%%:*}_${m}_n$n
      PETSC_OPTIONS="$LU" /usr/bin/time -f "MAXRSS=%M wall=%e" $X $B ${w#*:} $st --nx $n --ny $n --nz $n --dump-json $J/$tag.json > $D/logs/benchmark/$tag.log 2>&1
      echo "$tag exit=$?"
    done
  done
done
echo T15_BENCH_DONE
