#!/usr/bin/env python3
"""Where do the LP 5.2 level-to-level differences live?  RMS and max of
|coarse - fine| at shared nodes, over the whole domain and excluding disks of
radius r around the two discontinuity corners (1, +-1) of the boundary data."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lp52_post import snap
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
S = {nx: snap(os.path.join(R, f"lp52_snapshot_b1e-2_nx{nx}.csv")) for nx in (16, 32, 64)}
for c, f in ((16, 32), (32, 64)):
    xc, zc, Fc = S[c]; xf, zf, Ff = S[f]
    st = (len(xf) - 1) // (len(xc) - 1)
    X, Z = np.meshgrid(xc, zc)
    dist = np.minimum(np.hypot(X - 1, Z - 1), np.hypot(X - 1, Z + 1))
    for k in ("y", "p_LP"):
        d = np.abs(Fc[k] - Ff[k][::st, ::st]); sc = np.max(np.abs(Ff[k]))
        line = f"nx {c:2d}->{f:2d} {k:5s}: rms {np.sqrt(np.mean(d**2))/sc:.2e}  max {d.max()/sc:.2e}"
        for r in (0.125, 0.25, 0.5):
            line += f"  max(dist>{r}) {d[dist > r].max()/sc:.2e}"
        print(line)
