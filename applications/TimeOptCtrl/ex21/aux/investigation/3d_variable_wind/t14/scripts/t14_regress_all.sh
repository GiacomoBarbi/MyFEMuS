#!/bin/bash
# T14 mandatory regression driver: OLD = the T13 build (a28630779, private clean build, sanitised; recipe
# t14/scripts/clean_build_t13.sh).  2D (ex20 goldens 61, Math2025, EF 9.5, T0-T8 incl. 2D variable wind),
# T9, T10, T11N, T11D, T11H1 (single face), T12 (simplex), T13 (multi-face + 2D variable-wind gates).
H=$(cd "$(dirname "$0")/.." && pwd); W=${WORK:?}; OLD=${OLD:?}
WORK=$W/r2d OLD=$OLD bash $H/scripts/t14_regress_2d.sh > $H/logs/t14_regress_2d.log 2>&1; echo "2d exit=$?"
for t in t9 t10 t11n t11d t11h1 t12 t13; do WORK=$W/$t OLD=$OLD bash $H/scripts/t14_regress_$t.sh > $H/logs/t14_regress_$t.log 2>&1; echo "$t exit=$?"; done
echo T14_REGRESS_ALL_DONE
