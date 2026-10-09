#!/bin/bash
# T12 mandatory regression driver: OLD = the T11H1 build (81f32b89e, private clean build, sanitised;
# recipe t12/scripts/clean_build_t11h1.sh).  2D QUAD paths (ex20 goldens, Math2025 Ex3/4/5, EF 9.5,
# T0-T8 incl. T7D all regularisations and the T8D cores, zero-wind parity, restrictions), T9 (HEX27
# distributed exact cases), T10, T11N, T11D, T11H1.
H=$(cd "$(dirname "$0")/.." && pwd); W=${WORK:?}; OLD=${OLD:?}
bash $H/scripts/t12_regress_2d.sh > $H/logs/t12_regress_2d.log 2>&1; echo "2d exit=$?"
for t in t9 t10 t11n t11d t11h1; do WORK=$W/$t bash $H/scripts/t12_regress_$t.sh > $H/logs/t12_regress_$t.log 2>&1; echo "$t exit=$?"; done
