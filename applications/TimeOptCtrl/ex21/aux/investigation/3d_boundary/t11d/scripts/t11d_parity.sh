#!/bin/bash
# T11D heat limit: --convection off vs --convection on with w = (0,0,0), 3D Dirichlet control.
# (a) --dump-operators (n = 3, faces zmin and xmax, BE and CN): hatted M, K, M_Gamma, M_track,
#     coords, unhatted D_Mfull, D_Lfull, the C list, and the explicit B_D, B_D^T (sha256);
# (b) exact-data production solves (dirichlet-conv-3d --solve-only, n = 4, N = 16), BE/CN,
#     dense and iterative core: fields, J, core iterations, solver path.
B=$(cd "$(dirname "$0")/.." && pwd)
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
W=${T11D_WORK:-/tmp/t11d_par}; mkdir -p $W/output $B/logs/parity; cd $W
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for face in zmin xmax; do for s in be cn; do
  b="--dimension 3 --dirichlet-solve --dirichlet-face3d $face --state-bc dirichlet --tracking-box3d 0.25 0.75 0.25 0.75 0.25 0.75 --nx 3 --ny 3 --nz 3 --steps 3 --T 0.3 --time-scheme $s --no-paraview"
  $X $b --convection off --dump-operators off_${face}_$s > /dev/null 2>&1; $X $b --convection on --vx 0 --vy 0 --vz 0 --dump-operators w0_${face}_$s > /dev/null 2>&1
  for op in M K M_Gamma M_track coords D_Mfull D_Lfull D_C BD BDT; do e=coo; [ $op = coords -o $op = D_C ] && e=txt
    a=$(sha256sum < off_${face}_${s}_$op.$e | cut -c1-12); c=$(sha256sum < w0_${face}_${s}_$op.$e | cut -c1-12)
    if [ $a = $c ]; then v=BYTE-IDENTICAL
    elif [ $op = D_C ]; then   # header line 2 carries the transposes_L path flag (0 heat / 1 convection on)
      a=$(tail -n +3 off_${face}_${s}_D_C.txt | sha256sum | cut -c1-12); c=$(tail -n +3 w0_${face}_${s}_D_C.txt | sha256sum | cut -c1-12)
      v="C-list $( [ $a = $c ] && echo BYTE-IDENTICAL || echo DIFFER); header off=[$(sed -n 2p off_${face}_${s}_D_C.txt)] w0=[$(sed -n 2p w0_${face}_${s}_D_C.txt)] (transposes_L path flag)"
    else
      v="DIFFER $(python3 -c "
import numpy as np
def L(p):
    with open(p) as fh:
        fh.readline(); fh.readline(); fh.readline(); return np.loadtxt(fh, ndmin=2)
a, b = L('off_${face}_${s}_$op.$e'), L('w0_${face}_${s}_$op.$e')
if a.shape != b.shape: print('pattern/shape', a.shape, b.shape)
else: print('maxrel=%.1e same_pattern=%s' % (abs(a[:, 2] - b[:, 2]).max() / abs(a[:, 2]).max(), bool((a[:, :2] == b[:, :2]).all())))")"; fi
    echo "dump $face $s $op off=$a w0=$c $v"
  done
done; done | tee $B/logs/parity/parity.log
for s in be cn; do for core in dense iterative; do
  for m in "off:--convection off" "w0:--convection on --vx 0 --vy 0 --vz 0"; do t=${m%%:*}
    $X --exact-oc dirichlet-conv-3d --solve-only --nx 4 --ny 4 --nz 4 --steps 16 --time-scheme $s --dirichlet-core-solver $core \
      --no-paraview ${m#*:} --dump-json $W/${s}_${core}_$t.json > $B/logs/parity/${s}_${core}_$t.log 2>&1
  done
  python3 - <<PY
import json, numpy as np, re
L=lambda t: json.load(open(f"$W/{t}.json"))
a,b=L("${s}_${core}_off"),L("${s}_${core}_w0")
f=lambda d,k: np.array([s["values"] for s in d[k]])
mx=max(np.abs(f(a,k)-f(b,k)).max()/np.abs(f(a,k)).max() for k in ("state","adjoint","control_full"))
la=open("$B/logs/parity/${s}_${core}_off.log").read(); lb=open("$B/logs/parity/${s}_${core}_w0.log").read()
g=lambda p,s: (re.search(p,s) or [None,"-"])[1] if re.search(p,s) is None else re.search(p,s).group(1)
print("${s} ${core}: fields maxrel=%.2e J off/w0 = %.15e / %.15e outer %s/%s core_its %s/%s path off=[%s] w0=[%s]"%(mx,a['functional']['J_total'],b['functional']['J_total'],g(r'outer_iterations=(\d+)',la),g(r'outer_iterations=(\d+)',lb),g(r'core_its_max=(\d+)',la),g(r'core_its_max=(\d+)',lb),g(r'(pc=SCHUR\S+ inner=\S+)',la),g(r'(pc=SCHUR\S+ inner=\S+)',lb)))
PY
done; done | tee -a $B/logs/parity/parity.log
