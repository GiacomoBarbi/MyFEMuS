#!/bin/bash
# CN initial layer: 3D (time-ladder JSONs of t9_time.sh, n = 8, wind A) and the
# same diagnostic on 2D conv-sin3 with the FROZEN 2D binary (nx = 16), to show it
# is a property of the inherited CN time discretisation, not of the 3D path.
# T9_JSON=<dir of t9_time.sh JSONs> OLD=<frozen 2D binary> WORK=<scratch>
H=$(cd "$(dirname "$0")" && pwd)
LU="-sweep_right_pc_type lu -sweep_left_pc_type lu -sweep_ay_pc_type lu -fieldsplit_y_sweep_pc_type lu -ksp_rtol 1e-12"
mkdir -p $WORK/output; cd $WORK
for nt in 4 8 16 32 64 256; do PETSC_OPTIONS="$LU" OMP_NUM_THREADS=1 $OLD --exact-oc conv-sin3 --solve-only --nx 16 --nz 16 \
  --steps $nt --time-scheme cn --pc-schur-sweep --no-paraview --dump-json $WORK/cn2d_nt$nt.json > /dev/null 2>&1; done
echo "== 3D conv-sin3-3d, n = 8, wind (1, 0.5, -0.75), CN, ref N = 256"
python3 $H/t9_cn_layer.py "$T9_JSON/wA_cn_n8_nt{nt}.json" 8 "4 8 16 32 64" 256
echo "== 2D conv-sin3 (frozen 2D binary), nx = 16, CN, ref N = 256"
python3 $H/t9_cn_layer.py "$WORK/cn2d_nt{nt}.json" 0 "4 8 16 32 64" 256
