#!/bin/bash
# T13 all-six-faces smoke test (Gamma_c = the whole boundary; Dirichlet: no uncontrolled face, so C = every
# boundary node incl. all 12 edges and 8 corners, y = eta u on the whole boundary; Neumann: every boundary
# node carries the control).  n = 4, N = 8, wind A, BE / CN; Dirichlet L2 and combined H1; --dump-json, then
# t13_allsix.py checks the trace set, u = 0 inside, y_C = eta u_C (Dirichlet), edge/corner DOFs active.
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
D=$(cd "$(dirname "$0")/.." && pwd)
W=${T13_WORK:-/tmp/t13_allsix}; mkdir -p $W/output $D/logs/allsix; cd $W
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
LU="-sweep_right_pc_type lu -sweep_left_pc_type lu -sweep_ay_pc_type lu -fieldsplit_y_sweep_pc_type lu -ksp_rtol 1e-10"
ALL=zmin,ymin,xmax,ymax,xmin,zmax
C="--dimension 3 --nu 0.25 --alpha-u 1e-2 --target-type sin_x_minus_t --T 1 --nx 4 --ny 4 --nz 4 --steps 8 --tracking-box3d 0.2 0.8 0.2 0.8 0.2 0.8 --convection on --vx 1 --vy 0.5 --vz -0.75 --no-paraview"
for s in be cn; do
  PETSC_OPTIONS="$LU" /usr/bin/time -f "MAXRSS=%M wall=%e" $X $C --neumann-solve --state-bc natural --neumann-face3d $ALL --pc-schur-sweep --time-scheme $s \
     --dump-json $W/N_$s.json > $D/logs/allsix/N_$s.log 2>&1; echo "N_$s exit=$?"
  for r in "L2:" "comb:--control-h1-length 0.3 --control-h1-time-weight 1e-2"; do
    /usr/bin/time -f "MAXRSS=%M wall=%e" $X $C --dirichlet-solve --state-bc dirichlet --dirichlet-face3d $ALL --time-scheme $s ${r#*:} \
       --dirichlet-core-rtol 1e-12 --dump-json $W/D_${r%%:*}_$s.json > $D/logs/allsix/D_${r%%:*}_$s.log 2>&1; echo "D_${r%%:*}_$s exit=$?"
  done
done
