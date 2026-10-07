#!/bin/bash
# T7D solver audit / comparison / MPI for Dirichlet control + convection
# (production path, target sin_x_minus_t, nu = 0.25, alpha_u = 1e-2).
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
D=$(cd "$(dirname "$0")/.." && pwd)
W=${T7D_WORK:-/tmp/claude-1000/t7d_solver}; mkdir -p $W/output $D/logs/solver; cd $W
common="--dirichlet-solve --nu 0.25 --alpha-u 1e-2 --target-type sin_x_minus_t --no-paraview"
run() {  # tag np petsc_options args...
  local tag=$1 np=$2 popt=$3; shift 3
  local log=$D/logs/solver/$tag.log
  if [ "$np" = 1 ]; then
    PETSC_OPTIONS="$popt" OMP_NUM_THREADS=1 /usr/bin/time -f "MAXRSS=%M wall=%e" $X $common "$@" > $log 2>&1
  else
    PETSC_OPTIONS="$popt" OMP_NUM_THREADS=1 /usr/bin/time -f "MAXRSS=%M wall=%e" mpirun -np $np $X $common "$@" > $log 2>&1
  fi
  echo "$tag exit=$?"
}
# (a) explicit refusals
for s in be cn; do
  for pc in sweep matching boundary-reduced; do
    run refuse_${pc}_$s 1 "" --nx 8 --nz 8 --steps 8 --time-scheme $s --convection on --vx 1 --vy 0.5 --pc-schur-$pc
  done
done
# (b) default reduced solver: winds x schemes x sizes
for w in "off 0 0" "on 1 0.5" "on 0 0.5" "on -2 1" "on 0 -4" "on 4 2"; do
  set -- $w
  for s in be cn; do for nx in 16 32; do for nt in 32 64; do
    run red_c$1_vx$2_vy$3_${s}_nx${nx}_nt${nt} 1 "" --nx $nx --nz $nx --steps $nt --time-scheme $s --convection $1 --vx $2 --vy $3
  done; done; done
done
# (c) H1 variants under convection
for s in be cn; do
  run red_h1s_$s 1 "" --nx 16 --nz 16 --steps 32 --time-scheme $s --convection on --vx 1 --vy 0.5 --control-h1-length 0.3
  run red_h1t_$s 1 "" --nx 16 --nz 16 --steps 32 --time-scheme $s --convection on --vx 1 --vy 0.5 --control-h1-time-weight 1e-3
  run red_h1st_$s 1 "" --nx 16 --nz 16 --steps 32 --time-scheme $s --convection on --vx 1 --vy 0.5 --control-h1-length 0.3 --control-h1-time-weight 1e-3
done
# (d) factor backend overrides (PETSc options database), and the inexact
#     time-block Woodbury prototype (pre-existing, also 17 its for heat)
for s in be cn; do
  run lu_mumps_$s 1 "-dirichlet_reduced_a_pc_factor_mat_solver_type mumps -dirichlet_reduced_at_pc_factor_mat_solver_type mumps" \
      --nx 16 --nz 16 --steps 32 --time-scheme $s --convection on --vx 1 --vy 0.5
  run timeblocks4_$s 1 "" --nx 16 --nz 16 --steps 32 --time-scheme $s --convection on --vx 1 --vy 0.5 --pc-time-blocks 4
done
# (e) MPI np = 1/2/4
for s in be cn; do for np in 1 2 4; do
  for v in "on 1 0.5" "off 0 0"; do set -- $v
    run mpi_c$1_${s}_np$np $np "" --nx 32 --nz 32 --steps 32 --time-scheme $s --convection $1 --vx $2 --vy $3
  done
  run mpi_h1st_${s}_np$np $np "" --nx 32 --nz 32 --steps 32 --time-scheme $s --convection on --vx 1 --vy 0.5 --control-h1-length 0.3 --control-h1-time-weight 1e-3
done; done
