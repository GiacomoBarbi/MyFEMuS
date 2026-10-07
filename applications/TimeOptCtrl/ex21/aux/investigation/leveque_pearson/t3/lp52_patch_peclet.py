#!/usr/bin/env python3
"""Leveque-Pearson (NLAA 2022) section 2.1 LPS switch, evaluated exactly as defined:
   patches = 2 x 2 Q1 elements (side 2h, h = 2^(1-l)), w_k = wind at the patch centroid,
   h_k = patch length in the direction of the wind (chord of the square patch through its
   centroid along w_k: 2h / max(|cos|, |sin|), in [2h, 2 sqrt(2) h]),
   Pe_k = ||w_k|| h_k / (2 eps);  delta_k = 0  iff  Pe_k <= 1.
Also reported: the bound with h_k = 2 sqrt(2) h (diagonal, the most pessimistic reading)
and the element (h) and Q2 node-spacing (h/2) Peclet numbers used by ex21."""
import numpy as np
def wind(x, y):
    return 2 * y * (1 - x**2), -2 * x * (1 - y**2)
print(" eps    l   h        max Pe_k(exact chord)  max Pe_k(h_k=2sqrt2 h)  #patches Pe_k>1  LPS   max|w|h/(2eps)")
for eps in (1/20, 1/100, 1/500):
    for l in range(3, 9):
        h = 2.0 ** (1 - l)
        n = 2 ** (l - 1)                     # patches per direction (2 elements each)
        c = -1 + (2 * np.arange(n) + 1) * h  # patch centroids
        X, Y = np.meshgrid(c, c)
        wx, wy = wind(X, Y)
        speed = np.hypot(wx, wy)
        with np.errstate(invalid="ignore", divide="ignore"):
            m = np.maximum(np.abs(wx), np.abs(wy)) / np.where(speed > 0, speed, 1)
        hk = np.where(speed > 0, 2 * h / np.where(m > 0, m, 1), 2 * h)
        pe = speed * hk / (2 * eps)
        pe_diag = speed * 2 * np.sqrt(2) * h / (2 * eps)
        nact = int((pe > 1).sum())
        print(f" 1/{round(1/eps):<4d} {l}  {h:.5f}  {pe.max():10.4f}            {pe_diag.max():10.4f}          {nact:6d}/{n*n:<6d}  {'ON ' if nact else 'off'}  {2*h/(2*eps):.4f}")
