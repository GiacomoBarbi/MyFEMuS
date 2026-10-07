#!/bin/bash
# Example 5, stage 2 (binary with the free-response output and the
# SENSITIVITY-ONLY --ex5-variant): Fig. 45 free response per geometry,
# transient diagnostics, BE run from xy, and a bit-identity check of the EF
# path against the stage-1 binary.
set -u
BIN=/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21
HERE=$(cd "$(dirname "$0")/.." && pwd); LOG=$HERE/logs; RUN=${RUN:-/tmp/ex21_ex5_run2}
JOBS=${JOBS:-4}
mkdir -p "$LOG" "$RUN/output"; cd "$RUN"
sha256sum "$BIN" > "$LOG/binary_stage2.sha256"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
run() {
  local log=$1; shift
  [ -s "$log" ] && grep -q _RESULT "$log" && return
  { /usr/bin/time -f "WALL %e" "$BIN" "$@" --no-paraview; } > "$log" 2>&1
}
throttle() { while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done; }
PI=3.14159265358979323846; TWOPI=6.28318530717958647692
# free response of xy on Omega_c, T = 1, dt = 0.0025 (BE and CN), per geometry
for g in "$PI 1 24 8" "$PI 1 48 16" "1 1 12 12" "$PI 2 24 16" "$TWOPI 1 48 8"; do
  set -- $g
  for s in be cn; do throttle
    run $LOG/free_W$(printf %.3f $1)_H$2_${s}_nx$3_nz$4_T1_nt400.log --math2025-ex5 --ex5-geometry $1 $2 --ex5-initial zero \
        --nx $3 --nz $4 --steps 400 --T 1 --time-scheme $s --pc-schur-sweep &
  done
done
# transient diagnostics (SENSITIVITY ONLY), (pi,1), z0 = 0
for v in d-zero d-noswitch; do throttle
  run $LOG/diag_${v}_cn_nx24_nz8_nt200.log --math2025-ex5 --ex5-variant $v --nx 24 --nz 8 --steps 200 --time-scheme cn --pc-schur-sweep &
done
# OC from xy with an L-stable scheme (CN rings on the restricted xy)
throttle; run $LOG/ic_xy_be_nx24_nz8_nt800.log --math2025-ex5 --ex5-initial xy --nx 24 --nz 8 --steps 800 --time-scheme be --pc-schur-sweep &
# bit-identity of the EF path, stage-1 binary vs this one
throttle; run $HERE/../errorfeedback_95_96/logs/bitcheck_ef95_a1e-7_sinsin_cn_nx32_nz12_nt400_stage2bin.log \
    --errorfeedback-95 --ef-initial sinsin --alpha-u 1e-7 --nx 32 --nz 12 --steps 400 --time-scheme cn --pc-schur-sweep &
wait
echo DONE
