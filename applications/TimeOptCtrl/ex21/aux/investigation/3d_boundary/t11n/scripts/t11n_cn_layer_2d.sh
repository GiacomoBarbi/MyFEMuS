#!/bin/bash
# T11N: the same CN state-layer diagnostic on the 2D T7N neumann-conv exact case
# (wind (1, 0.5), nx = 8, N = 4..128 vs 512) with a 2D-unchanged binary (OLD = the
# T10 build), to attribute the 3D CN y start-up layer.   WORK= OLD=
H=$(cd "$(dirname "$0")" && pwd)
LU="-sweep_right_pc_type lu -sweep_left_pc_type lu -sweep_ay_pc_type lu -fieldsplit_y_sweep_pc_type lu -ksp_rtol 1e-12"
mkdir -p $WORK/output; cd $WORK
for nt in 4 8 16 32 64 128 512; do
  PETSC_OPTIONS="$LU" OMP_NUM_THREADS=1 $OLD --exact-oc neumann-conv --solve-only --nx 8 --nz 8 --steps $nt --time-scheme cn \
    --pc-schur-sweep --no-paraview --dump-json $WORK/n2d_nt$nt.json > /dev/null 2>&1; done
echo "== 2D neumann-conv (T7N), nx = 8, CN, ref N = 512 (nodal RMS)"
python3 $H/t11n_cn_layer.py "$WORK/n2d_nt{nt}.json" 0 "4 8 16 32 64 128" 512
