#!/bin/bash
# T12 heat limit on simplices: --convection off vs --convection on with w = 0, P2 TRI and P2 TET.
# (a) --dump-operators: M, K (eliminated), M_track, M_ctrl_whole, M_ctrl_box (= N), N_selected, M_full,
#     L_full, coords -- sha256;  (b) production full-domain solves (BE / CN, LU, outer rtol 1e-12):
#     fields, J, outer iterations.
B=$(cd "$(dirname "$0")/.." && pwd)
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
W=${T12_WORK:-/tmp/t12_par}; mkdir -p $W/output $B/logs/parity; cd $W
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
LU="-sweep_right_pc_type lu -sweep_left_pc_type lu -sweep_ay_pc_type lu -fieldsplit_y_sweep_pc_type lu -ksp_rtol 1e-12"
args() { case $1 in
  tri) echo "--mesh-type tri --domain 0 1 0 1 --distributed-solve --control-full-domain --state-bc dirichlet --tracking-box 0 1 0 1 --nx 6 --nz 6";;
  tet) echo "--dimension 3 --mesh-type tet --domain3d 0 1 0 1 0 1 --distributed-solve --control-full-domain --state-bc dirichlet --tracking-box3d 0 1 0 1 0 1 --nx 3 --ny 3 --nz 3";;
  esac; }
w0() { [ $1 = tet ] && echo "--convection on --vx 0 --vy 0 --vz 0" || echo "--convection on --vx 0 --vy 0"; }
for m in tri tet; do
  b="$(args $m) --steps 2 --pc-schur-sweep --no-paraview --nu 0.25"
  $X $b --convection off --dump-operators off_$m > /dev/null 2>&1; $X $b $(w0 $m) --dump-operators w0_$m > /dev/null 2>&1
  for op in M K M_track M_ctrl_whole M_ctrl_box N_selected M_full L_full coords; do e=coo; [ $op = coords ] && e=txt
    a=$(sha256sum < off_${m}_$op.$e | cut -c1-12); c=$(sha256sum < w0_${m}_$op.$e | cut -c1-12)
    echo "dump $m $op off=$a w0=$c $( [ $a = $c ] && echo BYTE-IDENTICAL || echo DIFFER)"
  done
done | tee $B/logs/parity/parity.log
for m in tri tet; do for s in be cn; do
  b="$(args $m) --steps 8 --T 1 --time-scheme $s --pc-schur-sweep --no-paraview --nu 0.25 --alpha-u 1e-2 --target-type sin_x_minus_t"
  PETSC_OPTIONS="$LU" $X $b --convection off --dump-json $W/${m}_${s}_off.json > $B/logs/parity/${m}_${s}_off.log 2>&1
  PETSC_OPTIONS="$LU" $X $b $(w0 $m) --dump-json $W/${m}_${s}_w0.json > $B/logs/parity/${m}_${s}_w0.log 2>&1
  /usr/bin/python3.11 - <<PY
import json, numpy as np, re
L=lambda t: json.load(open(f"$W/{t}.json"))
a,b=L("${m}_${s}_off"),L("${m}_${s}_w0")
f=lambda d,k: np.array([s["values"] for s in d[k]])
mx=max(np.abs(f(a,k)-f(b,k)).max()/np.abs(f(a,k)).max() for k in ("state","adjoint","control_full"))
la=open("$B/logs/parity/${m}_${s}_off.log").read(); lb=open("$B/logs/parity/${m}_${s}_w0.log").read()
g=lambda p,s: re.search(p,s).group(1) if re.search(p,s) else "-"
print("${m} ${s}: fields maxrel=%.2e J off/w0 = %.15e / %.15e (rel %.1e) outer %s/%s"%(mx,a['functional']['J_total'],b['functional']['J_total'],abs(a['functional']['J_total']-b['functional']['J_total'])/abs(a['functional']['J_total']),g(r'outer_iterations=(\d+)',la),g(r'outer_iterations=(\d+)',lb)))
PY
done; done | tee -a $B/logs/parity/parity.log
