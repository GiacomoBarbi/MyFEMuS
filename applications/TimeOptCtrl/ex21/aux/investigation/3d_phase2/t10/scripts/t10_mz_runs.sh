#!/bin/bash
# T10 multi-zone convergence runs (production path, --dump-json), octant layout,
# all region faces on the planes x, y, z = 1/2 (exactly resolved for n even):
#   control  (x<1/2,y<1/2,z<1/2) U (x>1/2,y>1/2,z>1/2)
#   tracking (x<1/2,y>1/2,z<1/2) U (x>1/2,y<1/2,z>1/2)      (zero element overlap)
#   sources  (x<1/2,y>1/2,z>1/2) U (x>1/2,y<1/2,z<1/2), amplitude 1
# natural BC, wind (1, 0.5, -0.75), nu = 0.25, alpha_u = 1e-2, target sin(x - t),
# final time T = 1 (--T: dt = T/N; without it production keeps dt = 0.05 and N
# would change T).
#   space: CN, N = 8 fixed, n = 2 4 8 16       time: n = 4 fixed, N = 4..128, 512; BE, CN
D=$(cd "$(dirname "$0")" && pwd)
MZ="--distributed-solve --dimension 3 --T 1 --nu 0.25 --alpha-u 1e-2 --target-type sin_x_minus_t --state-bc natural
    --convection on --vx 1 --vy 0.5 --vz -0.75
    --control-box3d 0 0.5 0 0.5 0 0.5 --control-box3d-add 0.5 1 0.5 1 0.5 1
    --tracking-box3d 0 0.5 0.5 1 0 0.5 --tracking-box3d-add 0.5 1 0 0.5 0.5 1
    --source 1 --source-box3d-add 0 0.5 0.5 1 0.5 1 --source-box3d-add 0.5 1 0 0.5 0 0.5"
case $1 in
  space) bash $D/t10_runs.sh mzs cn "2 4 8 16" 8 $MZ ;;
  time)  for s in be cn; do bash $D/t10_runs.sh mzt $s 4 "4 8 16 32 64 128 512" $MZ; done ;;
esac
