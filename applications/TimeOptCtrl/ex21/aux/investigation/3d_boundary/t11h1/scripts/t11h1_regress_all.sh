#!/bin/bash
# T11H1 mandatory regression driver: OLD = the T11D-L2 build (7530c58f8, private clean build, sanitised;
# recipe t11h1/scripts/clean_build_t11d.sh).  2D (ex20 goldens, Math2025 Ex3/4/5, EF 9.5, T0-T4, the 55
# old-vs-new 2D configurations incl. T7D all four regularisations and the T8D dense/iterative core,
# zero-wind parity, 2D/3D restrictions), T9, T10, T11N, T11D-L2.
H=$(cd "$(dirname "$0")/.." && pwd); W=${WORK:?}; OLD=${OLD:?}
bash $H/scripts/t11h1_regress_2d.sh > $H/logs/t11h1_regress_2d.log 2>&1; echo "2d exit=$?"
WORK=$W/t9 bash $H/scripts/t11h1_regress_t9.sh > $H/logs/t11h1_regress_t9.log 2>&1; echo "t9 exit=$?"
WORK=$W/t10 bash $H/scripts/t11h1_regress_t10.sh > $H/logs/t11h1_regress_t10.log 2>&1; echo "t10 exit=$?"
WORK=$W/t11n bash $H/scripts/t11h1_regress_t11n.sh > $H/logs/t11h1_regress_t11n.log 2>&1; echo "t11n exit=$?"
WORK=$W/t11d bash $H/scripts/t11h1_regress_t11d.sh > $H/logs/t11h1_regress_t11d.log 2>&1; echo "t11d exit=$?"
