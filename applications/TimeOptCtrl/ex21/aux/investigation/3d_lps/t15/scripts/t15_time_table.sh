#!/bin/bash
# T15 time self-convergence table -> ../../t15_time_convergence.csv (fixed mesh n = 4, ref nt = 1024; t15_time.py).
#   tA0 BE/CN (conv-sin3-3d data, wind A, LPS-P0 active, nu = 1e-2), tAg CN (Galerkin twin), tH0 CN (harmonic-3d, LPS),
#   tD0 BE/CN (Dirichlet zmin production, wind A, LPS), tDg CN (Galerkin twin: the CN terminal-time layer of the
#   nodal Dirichlet control is pre-existing).
S=$(cd "$(dirname "$0")" && pwd); J=${T15_JSON:-/tmp/t15_json}; cd $S
{ for t in tA0:be:distributed:- tA0:cn:distributed:- tAg:cn:distributed:- tH0:cn:distributed:- tD0:be:dirichlet:zmin tD0:cn:dirichlet:zmin tDg:cn:dirichlet:zmin; do
    IFS=: read tag s kind faces <<< "$t"; python3 t15_time.py $J $tag $s 4 "4 8 16 32 64 128" 1024 $kind $faces; done; } | awk 'NR==1 || !/^tag,/' > $S/../../t15_time_convergence.csv
