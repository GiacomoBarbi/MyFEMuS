#!/bin/bash
# EF 9.6 SCHUR_SWEEP cost diagnosis: alpha_u in {1e-10, 1e-11}, N in {50,100,200},
# nx 32 x nz 12, CN, benchmark driver (np = 1).  The algorithm is unchanged;
# instrumentation is (i) the existing sweep / Schur-KSP counters printed by the
# driver (EX21_EF96_SOLVER_STATS), (ii) one PETSc converged-reason line per
# spatial solve, streamed through awk into per-slot statistics (no raw log is
# kept), (iii) PETSc nested -log_view (flamegraph) for the timing breakdown,
# (iv) the outer FGMRES residual history (-ksp_monitor: the outer KSP is the
# only unprefixed one) and one line per Schur-split GMRES solve. Each run is
# capped at CAP seconds (default 3 h); a capped run keeps its partial history.
set -u
BIN=/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21
HERE=$(cd "$(dirname "$0")" && pwd); LOG=$HERE/logs; RUN=${RUN:-/tmp/ex21_diag}
mkdir -p $LOG $RUN/output; cd $RUN; sha256sum $BIN > $LOG/binary.sha256
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
one() {
  local a=$1 n=$2 name=diag_ef96_a$1_N$2
  [ -s $LOG/$name.log ] && grep -q _RESULT $LOG/$name.log && return
  PETSC_OPTIONS="-ksp_monitor -fieldsplit_schur_ksp_converged_reason -sweep_right_ksp_converged_reason -sweep_left_ksp_converged_reason \
-fieldsplit_y_sweep_ksp_converged_reason -log_view :$LOG/$name.flame:ascii_flamegraph" \
  /usr/bin/time -f "WALL %e" timeout -s TERM ${CAP:-10800} $BIN --errorfeedback-96 --ef-initial sinsin --alpha-u $a \
      --nx 32 --nz 12 --steps $n --T 20 --time-scheme cn --pc-schur-sweep --no-paraview 2>&1 |
  awk -v N=$n -v slots=$LOG/$name.slots.csv '
    /^ *Linear (sweep_right_|sweep_left_|fieldsplit_y_sweep_)/ {
      p=$2; k=$NF+0; s=c[p]%N; c[p]++; if (p=="sweep_left_") s=N-1-s
      key=p SUBSEP s; n[key]++; sum[key]+=k
      if (!(key in mn) || k<mn[key]) mn[key]=k; if (k>mx[key]) mx[key]=k
      hist[p SUBSEP k]++; if ($0 ~ /DIVERGED|did not converge/) fail[p]++
      next }
    { print }
    END {
      print "prefix,slot,solves,mean_its,min_its,max_its" > slots
      for (key in n) { split(key,q,SUBSEP); printf "%s,%d,%d,%.4f,%d,%d\n", q[1], q[2], n[key], sum[key]/n[key], mn[key], mx[key] >> slots }
      for (p in c) printf "EX21_DIAG_PREFIX %s solves=%d failures=%d\n", p, c[p], fail[p]+0
      for (h in hist) { split(h,q,SUBSEP); printf "EX21_DIAG_HIST %s its=%d count=%d\n", q[1], q[2], hist[h] }
    }' > $LOG/$name.log
}
for a in 1e-10 1e-11; do for n in 50 100 200; do one $a $n & done; done
wait; echo DONE
