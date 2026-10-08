#!/bin/bash
# T11N heat limit: --convection off vs --convection on with w = (0,0,0), 3D Neumann control.
# (a) --dump-operators (n = 3, face zmin and xmax): M, K, M_Gamma, N_selected, M_track, coords (sha256);
# (b) production solves (neumann-conv-3d data, n = 4, N = 16), BE/CN, LU and the default
#     iterative spatial backend: fields, J, iterations, solver path (pc= / inner= labels).
B=$(cd "$(dirname "$0")/.." && pwd)
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
W=${T11N_WORK:-/tmp/t11n_par}; mkdir -p $W/output $B/logs/parity; cd $W
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for face in zmin xmax; do
  b="--dimension 3 --neumann-solve --neumann-face3d $face --tracking-box3d 0 1 0 1 0 1 --state-bc natural --nx 3 --ny 3 --nz 3 --steps 2 --no-paraview"
  $X $b --dump-operators off_$face > /dev/null 2>&1; $X $b --convection on --vx 0 --vy 0 --vz 0 --dump-operators w0_$face > /dev/null 2>&1
  for op in M K M_Gamma N_selected M_track coords; do e=coo; [ $op = coords ] && e=txt
    a=$(sha256sum < off_${face}_$op.$e | cut -c1-12); c=$(sha256sum < w0_${face}_$op.$e | cut -c1-12)
    echo "dump $face $op off=$a w0=$c $( [ $a = $c ] && echo BYTE-IDENTICAL || echo DIFFER)"
  done
done | tee $B/logs/parity/parity.log
LU="-sweep_right_pc_type lu -sweep_left_pc_type lu -sweep_ay_pc_type lu -fieldsplit_y_sweep_pc_type lu"
for s in be cn; do for bk in "lu:$LU" "it:"; do k=${bk%%:*}; o=${bk#*:}
  for m in "off:--convection off" "w0:--convection on --vx 0 --vy 0 --vz 0"; do t=${m%%:*}
    PETSC_OPTIONS="$o" $X --exact-oc neumann-conv-3d --solve-only --nx 4 --ny 4 --nz 4 --steps 16 --time-scheme $s \
      --pc-schur-sweep --no-paraview ${m#*:} --dump-json $W/${s}_${k}_$t.json > $B/logs/parity/${s}_${k}_$t.log 2>&1
  done
  python3 - <<EOF
import json, numpy as np, re
L=lambda t: json.load(open(f"$W/{t}.json"))
a,b=L("${s}_${k}_off"),L("${s}_${k}_w0")
f=lambda d,k: np.array([s["values"] for s in d[k]])
mx=max(np.abs(f(a,k)-f(b,k)).max()/np.abs(f(a,k)).max() for k in ("state","adjoint","control_full"))
la=open("$B/logs/parity/${s}_${k}_off.log").read(); lb=open("$B/logs/parity/${s}_${k}_w0.log").read()
g=lambda p,s: re.search(p,s).group(1)
print("${s} ${k}: fields maxrel=%.2e J off/w0 = %.15e / %.15e outer %s/%s ysplit %s/%s path off=[%s] w0=[%s]"%(mx,a['functional']['J_total'],b['functional']['J_total'],g(r'outer_iterations=(\d+)',la),g(r'outer_iterations=(\d+)',lb),g(r' y=(\d+/\d+)',la),g(r' y=(\d+/\d+)',lb),g(r'(pc=\S+ inner=\S+)',la),g(r'(pc=\S+ inner=\S+)',lb)))
EOF
done; done | tee -a $B/logs/parity/parity.log
