#!/bin/bash
# T11H1 heat limit: --convection off vs --convection on with w = (0,0,0), 3D Dirichlet control with
# H1-space (ell = 0.3), H1-time (gamma_u = 1e-2) and combined.
# (a) --dump-operators (n = 3, faces zmin and xmax, BE and CN): M_Gamma, K_Gamma (eliminated and whole
#     face), R_H1t (sparse reduced), the unhatted M and L, the C list, B_D, B_D^T (sha256);
# (b) exact-data production solves (dirichlet-h1-3d --solve-only, n = 4, N = 16), BE/CN, dense and
#     iterative core, outer rtol 1e-14 (T11D: the default stops after one step at ~1e-11): fields, J,
#     core iterations, solver path.
B=$(cd "$(dirname "$0")/.." && pwd)
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
W=${T11H1_WORK:-/tmp/t11h1_par}; mkdir -p $W/output $B/logs/parity; cd $W
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
REGS=("H1s:--control-h1-length 0.3" "H1t:--control-h1-time-weight 1e-2" "comb:--control-h1-length 0.3 --control-h1-time-weight 1e-2")
for r in "${REGS[@]}"; do for face in zmin xmax; do for s in be cn; do
  t=${r%%:*}_${face}_$s
  b="--dimension 3 --dirichlet-solve --dirichlet-face3d $face --state-bc dirichlet --tracking-box3d 0.2 0.8 0.2 0.8 0.4 1 --nx 3 --ny 3 --nz 3 --steps 3 --T 0.3 --time-scheme $s --no-paraview ${r#*:}"
  [ $face = xmax ] && b="${b/--tracking-box3d 0.2 0.8 0.2 0.8 0.4 1/--tracking-box3d 0 0.6 0.2 0.8 0.2 0.8}"
  $X $b --convection off --dump-operators off_$t > /dev/null 2>&1; $X $b --convection on --vx 0 --vy 0 --vz 0 --dump-operators w0_$t > /dev/null 2>&1
  for op in M_Gamma K_Gamma M_Gamma_face K_Gamma_face RT D_Mfull D_Lfull BD BDT coords D_C; do e=coo; [ $op = coords -o $op = D_C ] && e=txt
    a=$(sha256sum < off_${t}_$op.$e | cut -c1-12); c=$(sha256sum < w0_${t}_$op.$e | cut -c1-12)
    if [ $a = $c ]; then v=BYTE-IDENTICAL
    elif [ $op = D_C ]; then   # header line 2 carries the transposes_L path flag (0 heat / 1 convection on)
      a=$(grep -v "^#" off_${t}_D_C.txt | tail -n +2 | sha256sum | cut -c1-12); c=$(grep -v "^#" w0_${t}_D_C.txt | tail -n +2 | sha256sum | cut -c1-12)
      v="C-list $( [ $a = $c ] && echo BYTE-IDENTICAL || echo DIFFER); header off=[$(sed -n 2p off_${t}_D_C.txt)] w0=[$(sed -n 2p w0_${t}_D_C.txt)] (transposes_L path flag)"
    else v=DIFFER; fi
    echo "dump $t $op off=$a w0=$c $v"
  done
done; done; done | tee $B/logs/parity/parity.log
for r in "${REGS[@]}"; do for s in be cn; do for core in dense iterative; do
  rr=${r%%:*}
  for m in "off:--convection off" "w0:--convection on --vx 0 --vy 0 --vz 0"; do t=${m%%:*}
    PETSC_OPTIONS="-ksp_rtol 1e-14" $X --exact-oc dirichlet-h1-3d --solve-only --nx 4 --ny 4 --nz 4 --steps 16 --time-scheme $s --dirichlet-core-solver $core \
      --dirichlet-core-rtol 1e-14 --no-paraview ${m#*:} ${r#*:} --dump-json $W/${rr}_${s}_${core}_$t.json > $B/logs/parity/${rr}_${s}_${core}_$t.log 2>&1
  done
  /usr/bin/python3.11 - <<PY
import json, numpy as np, re
L=lambda t: json.load(open(f"$W/{t}.json"))
a,b=L("${rr}_${s}_${core}_off"),L("${rr}_${s}_${core}_w0")
f=lambda d,k: np.array([s["values"] for s in d[k]])
mx=max(np.abs(f(a,k)-f(b,k)).max()/np.abs(f(a,k)).max() for k in ("state","adjoint","control_full"))
la=open("$B/logs/parity/${rr}_${s}_${core}_off.log").read(); lb=open("$B/logs/parity/${rr}_${s}_${core}_w0.log").read()
g=lambda p,s: re.search(p,s).group(1) if re.search(p,s) else "-"
print("${rr} ${s} ${core}: fields maxrel=%.2e J off/w0 = %.15e / %.15e (rel %.1e) outer %s/%s core_its %s/%s path off=[%s] w0=[%s]"%(mx,a['functional']['J_total'],b['functional']['J_total'],abs(a['functional']['J_total']-b['functional']['J_total'])/abs(a['functional']['J_total']),g(r'outer_iterations=(\d+)',la),g(r'outer_iterations=(\d+)',lb),g(r'core_its_max=(\d+)',la),g(r'core_its_max=(\d+)',lb),g(r'(pc=SCHUR\S+ inner=\S+)',la),g(r'(pc=SCHUR\S+ inner=\S+)',lb)))
PY
done; done; done | tee -a $B/logs/parity/parity.log
