#!/bin/bash
# Mathematics 2025 Example 5 campaign: gates, refinement, IC and geometry
# discrimination.  Serial runs, single-thread BLAS, up to JOBS in parallel.
set -u
BIN=/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21
HERE=$(cd "$(dirname "$0")/.." && pwd); LOG=$HERE/logs; RUN=${RUN:-/tmp/ex21_ex5_run}
JOBS=${JOBS:-5}
mkdir -p "$LOG" "$RUN/output"; cd "$RUN"
sha256sum "$BIN" > "$LOG/binary.sha256"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
run() {  # name args...
  local name=$1; shift
  [ -s "$LOG/$name.log" ] && grep -q RESULT "$LOG/$name.log" && return
  { /usr/bin/time -f "WALL %e" "$BIN" "$@" --no-paraview; } > "$LOG/$name.log" 2>&1
}
throttle() { while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done; }
PI=3.14159265358979323846
# gates (convection G1-G4) on the preset, both schemes
for s in cn be; do
  [ -s "$LOG/gates_$s.log" ] || "$BIN" --math2025-ex5 --ex5-initial xy --verify-convection --nx 6 --nz 2 --steps 10 --T 1 \
     --time-scheme $s --pc-schur-sweep --no-paraview > "$LOG/gates_$s.log" 2>&1
done
# refinement, (pi,1), z0 = 0, CN
for c in "12 4 100" "24 8 200" "48 16 400" "24 8 400" "48 16 200" "24 8 100"; do
  set -- $c; throttle
  run ref_cn_nx$1_nz$2_nt$3 --math2025-ex5 --ex5-initial zero --nx $1 --nz $2 --steps $3 --time-scheme cn --pc-schur-sweep &
done
throttle; run ref_be_nx24_nz8_nt400 --math2025-ex5 --ex5-initial zero --nx 24 --nz 8 --steps 400 --time-scheme be --pc-schur-sweep &
# IC xy
for c in "24 8 200" "48 16 400"; do
  set -- $c; throttle
  run ic_xy_cn_nx$1_nz$2_nt$3 --math2025-ex5 --ex5-initial xy --nx $1 --nz $2 --steps $3 --time-scheme cn --pc-schur-sweep &
done
# geometry discrimination (z0 = 0 and xy), h ~ 0.13
for g in "1 1 9 8" "$PI 2 24 16" "6.28318530717958647692 1 48 8"; do
  set -- $g
  for ic in zero xy; do throttle
    run geo_W$(printf %.3f $1)_H$2_${ic}_cn_nx$3_nz$4_nt200 --math2025-ex5 --ex5-geometry $1 $2 --ex5-initial $ic \
        --nx $3 --nz $4 --steps 200 --time-scheme cn --pc-schur-sweep &
  done
done
wait
echo DONE
