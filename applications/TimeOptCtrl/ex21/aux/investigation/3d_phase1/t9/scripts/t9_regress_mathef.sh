#!/bin/bash
# Re-run one archived Mathematics 2025 Ex. 5 run and one ErrorFeedback 9.5 run on
# the CURRENT binary and diff every EX21_ line present in the archived log
# (wall-clock fields and the later-added SOLVER_STATS line stripped).  WORK=<dir>
set -u
I=$(cd "$(dirname "$0")/../../.." && pwd)   # T9 copy of leveque_pearson/t3/regress_2d.sh (binary overridable)
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
W=${WORK:?set WORK}; mkdir -p $W/output; cd $W
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
strip() { grep -E "^EX21_" "$1" | grep -v SOLVER_STATS | sed -E 's/ (solve_wall|total_wall)=[^ ]*//g'; }
check() {  # name reference-log args...
  local name=$1 ref=$2; shift 2
  "$X" "$@" --no-paraview > $W/$name.log 2>&1
  if diff <(strip "$ref") <(strip $W/$name.log) > $W/$name.diff; then
    echo "$name: BYTE-IDENTICAL ($(strip "$ref" | wc -l) lines)"
  else
    echo "$name: DIFFERS ($(wc -l < $W/$name.diff) diff lines, see $W/$name.diff)"
  fi
}
sha256sum $X
check ex5_dzero_cn_nx24_nz8_nt200 $I/math2025_ex5/logs/diag_d-zero_cn_nx24_nz8_nt200.log \
      --math2025-ex5 --ex5-variant d-zero --nx 24 --nz 8 --steps 200 --time-scheme cn --pc-schur-sweep
check ef95_a1e-7_zero_cn_nx32_nz12_nt400 $I/errorfeedback_95_96/logs/ic_ef95_a1e-7_zero_cn_nx32_nz12_nt400.log \
      --errorfeedback-95 --ef-initial zero --alpha-u 1e-7 --nx 32 --nz 12 --steps 400 --time-scheme cn --pc-schur-sweep
