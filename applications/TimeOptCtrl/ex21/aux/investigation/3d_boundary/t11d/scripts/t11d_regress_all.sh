#!/bin/bash
# T11D mandatory regression driver: OLD = the T11N build (ec825f26d, private clean build,
# sanitised; recipe t11d/scripts/clean_build_t11n.sh).  2D (ex20 goldens, Math2025, EF 9.5,
# T0-T4, 55 old-vs-new 2D configurations, zero-wind parity, 2D/3D restrictions), T9, T10, T11N.
H=$(cd "$(dirname "$0")/.." && pwd); W=${WORK:?}; OLD=${OLD:?}
bash $H/scripts/t11d_regress_2d.sh > $H/logs/t11d_regress_2d.log 2>&1; echo "2d exit=$?"
WORK=$W/t9 bash $H/scripts/t11d_regress_t9.sh > $H/logs/t11d_regress_t9.log 2>&1; echo "t9 exit=$?"
WORK=$W/t10 bash $H/scripts/t11d_regress_t10.sh > $H/logs/t11d_regress_t10.log 2>&1; echo "t10 exit=$?"
WORK=$W/t11n bash $H/scripts/t11d_regress_t11n.sh > $H/logs/t11d_regress_t11n.log 2>&1; echo "t11n exit=$?"
