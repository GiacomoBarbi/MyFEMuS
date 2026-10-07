#!/bin/bash
# ErrorFeedback 9.5 / 9.6 AAO campaign: gates, IC sensitivity, alpha_u sweep,
# space/time refinement.  Serial runs, single-thread BLAS, JOBS in parallel.
# nx multiple of 8 (x-edges k pi/8), nz multiple of 3 (y-edges 1/3, 2/3).
set -u
BIN=/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21
HERE=$(cd "$(dirname "$0")/.." && pwd); LOG=$HERE/logs; RUN=${RUN:-/tmp/ex21_ef_run}
JOBS=${JOBS:-5}; STAGE=${STAGE:-1}
mkdir -p "$LOG" "$RUN/output"; cd "$RUN"
sha256sum "$BIN" > "$LOG/binary_stage$STAGE.sha256"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
run() {
  local name=$1; shift
  [ -s "$LOG/$name.log" ] && grep -q _RESULT "$LOG/$name.log" && return
  { /usr/bin/time -f "WALL %e" "$BIN" "$@" --no-paraview; } > "$LOG/$name.log" 2>&1
}
throttle() { while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done; }
if [ "$STAGE" = 1 ]; then
  for ex in 95 96; do for s in cn be; do
    [ -s "$LOG/gates_ef${ex}_$s.log" ] || "$BIN" --errorfeedback-$ex --ef-initial sinsin --alpha-u 1e-7 --verify-convection \
       --nx 8 --nz 3 --steps 10 --T 1 --time-scheme $s --pc-schur-sweep --no-paraview > "$LOG/gates_ef${ex}_$s.log" 2>&1
  done; done
  # IC sensitivity
  for ex in 95 96; do for a in 1e-5 1e-7 1e-9; do for ic in zero sinsin poly; do throttle
    run ic_ef${ex}_a${a}_${ic}_cn_nx32_nz12_nt400 --errorfeedback-$ex --ef-initial $ic --alpha-u $a \
        --nx 32 --nz 12 --steps 400 --time-scheme cn --pc-schur-sweep &
  done; done; done
  # alpha_u sweep (z0 = sinsin), every decade
  for ex in 95 96; do for a in 1e-3 1e-4 1e-5 1e-6 1e-7 1e-8 1e-9 1e-10 1e-11; do throttle
    run sweep_ef${ex}_a${a}_cn_nx32_nz12_nt400 --errorfeedback-$ex --ef-initial sinsin --alpha-u $a \
        --nx 32 --nz 12 --steps 400 --time-scheme cn --pc-schur-sweep &
  done; done
fi
if [ "$STAGE" = 2 ]; then
  # space refinement (nx 16/32/64/128, nz 6/12/24/48) and time refinement
  # (nt 200/400/800 at nx 64) at the alpha_u that carry the conclusions:
  # 9.5 -- the bracket of the E5 crossing (1e-9, 1e-10) and the smallest 1e-11;
  # 9.6 -- the floor (1e-7, 1e-9, 1e-11).
  for job in "95 1e-9" "95 1e-10" "95 1e-11" "96 1e-7" "96 1e-9" "96 1e-11"; do
    set -- $job; ex=$1; a=$2
    for m in "16 6" "64 24"; do set -- $m; throttle
      run sweep_ef${ex}_a${a}_cn_nx$1_nz$2_nt400 --errorfeedback-$ex --ef-initial sinsin --alpha-u $a \
          --nx $1 --nz $2 --steps 400 --time-scheme cn --pc-schur-sweep &
    done
    for nt in 200 800; do throttle
      run sweep_ef${ex}_a${a}_cn_nx64_nz24_nt$nt --errorfeedback-$ex --ef-initial sinsin --alpha-u $a \
          --nx 64 --nz 24 --steps $nt --time-scheme cn --pc-schur-sweep &
    done
    throttle
    run sweep_ef${ex}_a${a}_be_nx64_nz24_nt800 --errorfeedback-$ex --ef-initial sinsin --alpha-u $a \
        --nx 64 --nz 24 --steps 800 --time-scheme be --pc-schur-sweep &
  done
fi
if [ "$STAGE" = 3 ]; then
  # Pruned refinement after stage 2 (memory: nx128 x nt400 was OOM-killed when
  # run in parallel; the nx 128 / 96 levels therefore use nt = 200, whose time
  # error is 0.6 % at nx 64).  9.5: the E5 bracket; 9.6: the floor (alpha 1e-7).
  for job in "95 1e-9 96 36 200" "95 1e-10 96 36 200" "95 1e-9 128 48 200" "95 1e-10 128 48 200" \
             "96 1e-7 16 6 400" "96 1e-7 64 24 400" "96 1e-7 64 24 200" "96 1e-7 64 24 800" "96 1e-7 128 48 200"; do
    set -- $job; throttle
    run sweep_ef$1_a$2_cn_nx$3_nz$4_nt$5 --errorfeedback-$1 --ef-initial sinsin --alpha-u $2 \
        --nx $3 --nz $4 --steps $5 --time-scheme cn --pc-schur-sweep &
  done
  throttle
  run sweep_ef96_a1e-7_be_nx64_nz24_nt800 --errorfeedback-96 --ef-initial sinsin --alpha-u 1e-7 \
      --nx 64 --nz 24 --steps 800 --time-scheme be --pc-schur-sweep &
fi
if [ "$STAGE" = 4 ]; then
  # 9.5 spatial convergence at moderate alpha_u (where the edge layer of the
  # colocated optimum is still resolved): nx 16 / 64 / 96 next to stage 1's 32.
  for job in "1e-5 16 6" "1e-7 16 6" "1e-5 64 24" "1e-7 64 24" "1e-5 96 36" "1e-7 96 36"; do
    set -- $job; throttle
    nt=400; [ "$2" = 96 ] && nt=200
    run sweep_ef95_a$1_cn_nx$2_nz$3_nt$nt --errorfeedback-95 --ef-initial sinsin --alpha-u $1 \
        --nx $2 --nz $3 --steps $nt --time-scheme cn --pc-schur-sweep &
  done
fi
if [ "$STAGE" = 5 ]; then
  # 9.5 on the finest affordable mesh: bracket E5 at nx 128 (alpha 1e-10 there
  # is not yet mesh-converged, so this bounds alpha_u(E5) from above).
  run sweep_ef95_a1e-7_cn_nx128_nz48_nt200 --errorfeedback-95 --ef-initial sinsin --alpha-u 1e-7 \
      --nx 128 --nz 48 --steps 200 --time-scheme cn --pc-schur-sweep &
  wait
  run sweep_ef95_a1e-11_cn_nx128_nz48_nt200 --errorfeedback-95 --ef-initial sinsin --alpha-u 1e-11 \
      --nx 128 --nz 48 --steps 200 --time-scheme cn --pc-schur-sweep &
fi
wait
echo DONE
