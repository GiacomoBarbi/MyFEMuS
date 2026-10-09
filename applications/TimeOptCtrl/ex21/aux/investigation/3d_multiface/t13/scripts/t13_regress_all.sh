#!/bin/bash
# T13 mandatory regression driver: OLD = the T12 build (120c74442, private clean build, sanitised; recipe
# t13_old/clean_build_t12.sh).  2D (ex20 goldens 61, Math2025, EF 9.5, T0-T8 incl. 2D Neumann / Dirichlet /
# natural, zero-wind parity, restrictions), T9, T10, T11N, T11D, T11H1 (single face), T12 (simplex).
H=$(cd "$(dirname "$0")/.." && pwd); W=${WORK:?}; OLD=${OLD:?}
WORK=$W/r2d OLD=$OLD bash $H/scripts/t13_regress_2d.sh > $H/logs/t13_regress_2d.log 2>&1; echo "2d exit=$?"
for t in t9 t10 t11n t11d t11h1 t12; do WORK=$W/$t OLD=$OLD bash $H/scripts/t13_regress_$t.sh > $H/logs/t13_regress_$t.log 2>&1; echo "$t exit=$?"; done
