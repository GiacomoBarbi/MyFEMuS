#!/bin/bash
# T14 (t13_permutation.sh under the GENERIC variable wind affine-div-3d) permutation invariance: the same face SET given in different orders must give byte-identical
# results.  For each set, every listed order is run and compared with the first: (a) every EX20_/EX21_
# line (timings stripped), (b) the JSON dump (path field stripped), (c) every operator dump file
# (M_Gamma, N_selected / D_C, B_D, B_D^T, K_Gamma, R_T, ...).  Neumann: SCHUR_SWEEP + LU; Dirichlet: default.
# Output: one line per comparison (IDENTICAL / DIFFERS) -> t13_permutation.py -> ../../t13_permutation.csv
X=${EX21_BIN:-/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21}
W=${T13_WORK:?}; mkdir -p $W/output; cd $W
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
LU="-sweep_right_pc_type lu -sweep_left_pc_type lu -sweep_ay_pc_type lu -fieldsplit_y_sweep_pc_type lu"
strip() { grep -E "^EX2[01]_" | sed -E 's/ (solve_wall|total_wall|setup_s|solve_s|t_operators|t_pcsetup|t_kspsolve|t_total|setup_seconds|core_seconds|shell_seconds|wall|json)=[^ ]*//g'; }   # json= is the dump path (differs by construction)
wA="--convection on --wind affine-div-3d"
PN="--dimension 3 --neumann-solve --state-bc natural --nu 0.25 --alpha-u 1e-2 --target-type sin_x_minus_t --steps 4 --T 1 --nx 3 --ny 3 --nz 3 --tracking-box3d 0.2 0.8 0.2 0.8 0.2 0.8 $wA"
PD="--dimension 3 --dirichlet-solve --state-bc dirichlet --nu 0.25 --alpha-u 1e-2 --target-type sin_x_minus_t --steps 4 --T 1 --nx 3 --ny 3 --nz 3 --tracking-box3d 0.2 0.8 0.2 0.8 0.2 0.8 $wA"
one() {   # kind tag order extra...   (COMPARE_ONLY=1: re-strip the saved raw output, no runs)
  k=$1; t=$2; o=$3; shift 3
  if [ -n "$COMPARE_ONLY" ]; then strip < $t.run > $t.out; strip < $t.vcraw > $t.vc; return; fi
  if [ $k = N ]; then base="$PN --neumann-face3d $o --pc-schur-sweep"; po="$LU"; else base="$PD --dirichlet-face3d $o"; po=""; fi
  PETSC_OPTIONS="$po" $X $base "$@" --no-paraview --verify-convection > $t.vcraw 2>/dev/null; strip < $t.vcraw > $t.vc
  PETSC_OPTIONS="$po" $X $base "$@" --no-paraview --dump-json $W/$t.json > $t.run 2>&1; strip < $t.run > $t.out
  sed -i -E 's/"json":"[^"]*"//' $W/$t.json
  PETSC_OPTIONS="$po" $X $base "$@" --no-paraview --dump-operators $W/ops_$t > /dev/null 2>&1
}
cmpset() {   # kind set-tag "order1 order2 ..." extra...
  k=$1; st=$2; orders=$3; shift 3; i=0
  for o in $orders; do one $k ${st}_$i $o "$@"; i=$((i+1)); done
  i=0
  for o in $orders; do
    if [ $i -gt 0 ]; then
      for ext in vc out json; do
        cmp -s ${st}_0.$ext ${st}_$i.$ext && v=IDENTICAL || v=DIFFERS
        echo "PERM $k $st order=$o vs=$(echo $orders | cut -d' ' -f1) what=$ext lines=$(wc -l < ${st}_$i.$ext) $v"
      done
      nd=0; nf=0
      for f in ops_${st}_0_*; do g=ops_${st}_${i}_${f#ops_${st}_0_}; nf=$((nf+1)); cmp -s $f $g || nd=$((nd+1)); done
      [ $nd -eq 0 ] && v=IDENTICAL || v=DIFFERS
      echo "PERM $k $st order=$o vs=$(echo $orders | cut -d' ' -f1) what=operator_dumps files=$nf differing=$nd $v"
    fi
    i=$((i+1))
  done
}
cmpset N opp "zmin,zmax zmax,zmin"
cmpset N adj "zmin,xmin xmin,zmin"
cmpset N thr "zmin,ymin,xmin xmin,ymin,zmin ymin,zmin,xmin"
for s in be cn; do
  cmpset D opp_$s "zmin,zmax zmax,zmin" --time-scheme $s
  cmpset D adj_$s "zmin,xmin xmin,zmin" --time-scheme $s
done
cmpset D thr "zmin,ymin,xmin xmin,ymin,zmin ymin,zmin,xmin"
cmpset D adjH1 "zmin,xmin xmin,zmin" --control-h1-length 0.3 --control-h1-time-weight 1e-2 --time-scheme cn
echo T14_PERMUTATION_DONE
