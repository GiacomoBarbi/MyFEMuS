#!/bin/bash
# Re-run the Mathematics 2025 Example 3 probe and primary runs and one
# Example 4 run on the CURRENT binary and diff every EX21_ output line against
# the archived logs (wall-clock fields stripped).  Usage: WORK=<dir> $0
set -u
I=$(cd "$(dirname "$0")/.." && pwd)
X=/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21
W=${WORK:?set WORK}; mkdir -p $W/output; cd $W
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
strip() { grep -E "^EX21_" "$1" | sed -E 's/ (solve_wall|total_wall)=[^ ]*//g'; }
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
check ex3_probe_cn_nx20_nt400_iczero $I/math2025_ex3/logs/probe_cn_nx20_nt400_iczero.log \
      --math2025-ex3 --ex3-initial zero --nx 20 --steps 400 --time-scheme cn --pc-schur-sweep
check ex3_cn_sweep_nx40_nt1600 $I/math2025_ex3/logs/cn_sweep_nx40_nt1600.log \
      --math2025-ex3 --ex3-initial zero --nx 40 --steps 1600 --time-scheme cn --pc-schur-sweep
check ex4_ref_cn_a1e-9_nx20_nt600 $I/math2025_ex4/logs/ref_cn_a1e-9_nx20_nt600.log \
      --math2025-ex4 --ex4-initial x1mx --alpha-u 1e-9 --nx 20 --steps 600 --time-scheme cn --pc-schur-sweep
