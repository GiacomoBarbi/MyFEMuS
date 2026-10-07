#!/usr/bin/env python3
"""Where are the small-beta adjoint features?  For each snapshot: location of
max|p_LP|; fraction of max|p| carried within distance d of the boundary; the
odd-even (node-to-node) oscillation amplitude of p, max |p_i - (p_{i-1}+p_{i+1})/2|,
split into boundary band (dist < 0.1) and interior; and the target/boundary-data
mismatch |y_d - g| on the boundary (the source of tracking-induced layers)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lp52_post import snap
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
for b in ("1e-2", "1e-4", "1e-6"):
    for nx in (32, 64):
        p = os.path.join(R, f"lp52_snapshot_b{b}_nx{nx}.csv")
        if not os.path.exists(p):
            continue
        xs, zs, F = snap(p)
        X, Z = np.meshgrid(xs, zs)
        P = F["p_LP"]; s = np.abs(P).max()
        dist = np.minimum.reduce([1 - X, X + 1, 1 - Z, Z + 1])
        i = np.unravel_index(np.abs(P).argmax(), P.shape)
        odd = np.zeros_like(P)
        odd[:, 1:-1] = np.abs(P[:, 1:-1] - 0.5 * (P[:, :-2] + P[:, 2:]))
        odd[1:-1, :] = np.maximum(odd[1:-1, :], np.abs(P[1:-1, :] - 0.5 * (P[:-2, :] + P[2:, :])))
        band = dist < 0.1
        print(f"beta={b:5s} nx={nx}: max|p|={s:.3e} at (x1,x2)=({X[i]:+.3f},{Z[i]:+.3f}) dist={dist[i]:.3f};"
              f" max|p| in dist>=0.1: {np.abs(P[~band]).max()/s:.2f} of max;"
              f" odd-even amplitude/max|p|: band {odd[band].max()/s:.2e}, interior {odd[~band].max()/s:.2e}")
x = np.linspace(-1, 1, 9)
print("boundary mismatch |y_d - g| on x2 = +-1 at x1 =", " ".join(f"{v:+.2f}:{np.exp(-10*(1-v)):.3f}" for v in x))
