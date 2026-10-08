#!/bin/bash
# T10 full campaign on the final binary (run in the sanitised environment:
# env -i HOME=$HOME PATH=<petsc arch>/bin:/usr/bin:/bin).  Scratch under $T10_SCRATCH.
#   geometry, gates, sources, refusals, MPI, natural-BC spatial ladders (winds A,
#   B, heat), natural-BC time ladders, multi-zone space/time ladders.
D=$(cd "$(dirname "$0")" && pwd); L=$D/../logs; R=$D/../..; S=${T10_SCRATCH:-/tmp/t10}
mkdir -p $L $S
echo "== geometry";  T10_WORK=$S/geom python3 $D/t10_geometry.py $R/t10_geometry.csv > $L/geometry.log 2>&1; tail -n1 $L/geometry.log
echo "== gates";     T10_WORK=$S/gates bash $D/t10_gates.sh > $L/gates_driver.out 2>&1; python3 $D/t10_gates_table.py > $L/gates_table.out 2>&1; grep -c exit=0 $L/gates_driver.out
echo "== sources";   T10_JSON=$S/json T10_WORK=$S/src bash $D/t10_sources.sh > $L/sources_driver.out 2>&1; (cd $D && python3 t10_sources.py $S/json > $L/sources.log 2>&1); tail -n1 $L/sources.log
echo "== refusals";  T10_WORK=$S/ref bash $D/t10_refusals.sh > $L/refusals.log 2>&1; tail -n1 $L/refusals.log
echo "== mpi";       T10_WORK=$S/mpi T10_JSON=$S/mpi_json bash $D/t10_mpi.sh > $L/mpi_driver.out 2>&1; (cd $D && python3 t10_mpi.py $S/mpi_json > $L/mpi.log 2>&1); tail -n1 $L/mpi.log
echo "== ladders"
( T10_WORK=$S/lA bash $D/t10_exact_ladder.sh natA natural-conv-3d cn "2 4 6 8 10 12" 64 --convection on --vx 1 --vy 0.5 --vz -0.75 > $L/ladder_natA.out 2>&1 &
  T10_WORK=$S/lB bash $D/t10_exact_ladder.sh natB natural-conv-3d cn "2 4 6 8 10" 64 --convection on --vx -2 --vy 1 --vz 0.5 > $L/ladder_natB.out 2>&1 &
  T10_WORK=$S/lH bash $D/t10_exact_ladder.sh natH natural-conv-3d cn "2 4 6 8 10" 64 --convection off > $L/ladder_natH.out 2>&1 &
  for s in be cn; do T10_JSON=$S/json T10_WORK=$S/natt bash $D/t10_runs.sh natt $s 6 "4 8 16 32 64 128 512" --exact-oc natural-conv-3d --solve-only --convection on --vx 1 --vy 0.5 --vz -0.75; done > $L/nat_time_driver.out 2>&1 &
  wait )
# fixed-dt spatial self-convergence of the natural exact case (the exact-error
# ladder reaches the N = 64 CN time floor for y at n >= 10): N = 16, n = 2 4 8 vs 16
T10_JSON=$S/json T10_WORK=$S/nats bash $D/t10_runs.sh natsA cn "2 4 8 16" 16 --exact-oc natural-conv-3d --solve-only --convection on --vx 1 --vy 0.5 --vz -0.75 > $L/nat_selfconv_driver.out 2>&1
T10_JSON=$S/json T10_WORK=$S/nats bash $D/t10_runs.sh natsH cn "2 4 8 16" 16 --exact-oc natural-conv-3d --solve-only --convection off >> $L/nat_selfconv_driver.out 2>&1
T10_JSON=$S/json T10_WORK=$S/mzs bash $D/t10_mz_runs.sh space > $L/mz_space_driver.out 2>&1
T10_JSON=$S/json T10_WORK=$S/mzt bash $D/t10_mz_runs.sh time > $L/mz_time_driver.out 2>&1
echo T10_CAMPAIGN_DONE
