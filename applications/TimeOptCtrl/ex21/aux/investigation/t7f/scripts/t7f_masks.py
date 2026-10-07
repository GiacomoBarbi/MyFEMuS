#!/usr/bin/env python3
"""Analyse t7f_masks.sh dumps.  1^T M 1 = measure of the selected elements."""
import os, sys
import numpy as np
W = sys.argv[1] if len(sys.argv) > 1 else "/tmp/t7f_masks"
ok_all = True
def load(tag, op):
    p = f"{W}/{tag}_{op}.coo"
    L = [l.split() for l in open(p) if not l.startswith("#")]
    n = int(L[0][0]); M = np.zeros((n, n))
    for r, c, v in L[1:]:
        M[int(r), int(c)] += float(v)
    return M
def rep(name, val, ref, tol=1e-13):
    global ok_all
    ok = abs(val - ref) <= tol * max(1, abs(ref)); ok_all &= ok
    print(f"T7F_MASK {name:<64s} {val:.15f} ref {ref:.15f} {'PASS' if ok else 'FAIL'}")
def same(name, A, B, expect=True):
    global ok_all
    d = np.abs(A - B).max(); ok = (d <= 1e-15) == expect; ok_all &= ok
    print(f"T7F_MASK {name:<64s} max|diff|={d:.2e} expected {'identical' if expect else 'DIFFERENT'} {'PASS' if ok else 'FAIL'}")
one = lambda M: M.sum()
for nx in (8, 16):
    C1, C2, CM = (load(f"{t}_{nx}", "M_ctrl_box") for t in ("c1", "c2", "mz"))
    T1, T2, TM = (load(f"{t}_{nx}", "M_track") for t in ("c1", "c2", "mz"))
    rep(f"nx={nx} control union area (2 x 0.25^2)", one(CM), 0.125)
    rep(f"nx={nx} tracking union area (0.375*0.25 + 0.375^2)", one(TM), 0.234375)
    same(f"nx={nx} M_ctrl(C1 u C2) == M_ctrl(C1) + M_ctrl(C2)", CM, C1 + C2)
    same(f"nx={nx} M_track(T1 u T2) == M_track(T1) + M_track(T2)", TM, T1 + T2)
    rep(f"nx={nx} N_selected == M_ctrl_box (multi-zone)", np.abs(load(f'mz_{nx}', 'N_selected') - CM).max(), 0.0)
    # control/tracking non-colocated: disjoint supports
    supp = lambda M: set(np.nonzero(np.abs(M).sum(1))[0])
    inter_el = (np.abs(CM) > 0) & (np.abs(TM) > 0)
    rep(f"nx={nx} control and tracking share no element-mass entries", float(inter_el.sum()), 0.0)
    for op, lab in (("M_ctrl_box", "control"), ("M_track", "tracking"), ("M_source", "source")):
        A, Bx, AB, U = (load(f"{t}_{nx}", op) if not (op == "M_source" and t in ("ovA", "ovB")) else None
                        for t in ("ovA", "ovB", "ovAB", "ovU"))
        same(f"nx={nx} overlap {lab}: M(A u B) == M(single union box)", AB, U)
        if A is not None:
            same(f"nx={nx} overlap {lab}: M(A u B) != M(A) + M(B) (no double count)", AB, A + Bx, expect=False)
        rep(f"nx={nx} overlap {lab}: area of A u B (0.75*0.25)", one(AB), 0.1875)
    S1, S2, S12 = (load(f"{t}_{nx}", "M_source") for t in ("s1", "s2", "s12"))
    rep(f"nx={nx} source union area (2 x 0.25^2)", one(S12), 0.125)
    same(f"nx={nx} M_source(S1 u S2) == M_source(S1) + M_source(S2)", S12, S1 + S2)
    # centroid rule on (0.3, 0.7): elements with centroid strictly inside
    h = 1.0 / nx; cen = (np.arange(nx) + 0.5) * h; k = ((cen > 0.3) & (cen < 0.7)).sum()
    rep(f"nx={nx} non-aligned box (0.3,0.7)^2: area = (#centroids inside * h)^2", one(load(f'na_{nx}', 'M_ctrl_box')), (k * h) ** 2)
print("T7F_MASK_RESULT", "PASS" if ok_all else "FAIL")
