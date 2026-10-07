#!/usr/bin/env python3
"""Pointwise comparison of ex21 ||e(t)||_{L2(Omega_c)} with the curves
extracted from Mathematics 2025 Figures 32 (t in [0,20]) and 33 (t in [0,2]).

Primary target: the source's OWN optimal-control error E_oc (red).
Secondary: the cascade iteration error e_2 (blue).

usage: compare.py <ex21_log> [<label>]   (log must contain EX21_EX3_SERIES lines)
Missing red vertices (collinear points dropped by the plot export) are filled
by linear interpolation between neighbouring vertices, which is exact for the
dropped points to the export tolerance.
"""
import csv, json, math, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(os.path.dirname(HERE)), "results")


def load_fig(name):
    rows = list(csv.DictReader(open(os.path.join(RES, f"math2025_ex3_{name}.csv"))))
    t = [float(r["t"]) for r in rows]
    out = {}
    for key in ("Eoc_c", "e2_cascade_c"):
        known = [(ti, float(r[key])) for ti, r in zip(t, rows) if r[key] != ""]
        vals = []
        for ti in t:
            # linear interpolation on the known vertices
            j = max(k for k in range(len(known)) if known[k][0] <= ti + 1e-12)
            if abs(known[j][0] - ti) < 1e-12 or j + 1 == len(known):
                vals.append(known[j][1])
            else:
                (t0, v0), (t1, v1) = known[j], known[j + 1]
                vals.append(v0 + (v1 - v0) * (ti - t0) / (t1 - t0))
        out[key] = vals
    return t, out


def load_ex21(path):
    t, e = [], []
    for line in open(path):
        m = re.match(r"EX21_EX3_SERIES t=(\S+) e_c=(\S+)", line)
        if m:
            t.append(float(m.group(1)))
            e.append(float(m.group(2)))
    return t, e


def sample(t, e, ti):
    """Linear interpolation of the ex21 series (exact when ti is a node)."""
    lo, hi = 0, len(t) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if t[mid] <= ti:
            lo = mid
        else:
            hi = mid
    if abs(t[lo] - ti) < 1e-9:
        return e[lo]
    if abs(t[hi] - ti) < 1e-9:
        return e[hi]
    return e[lo] + (e[hi] - e[lo]) * (ti - t[lo]) / (t[hi] - t[lo])


def peaks(t, v, lo, hi):
    """Local maxima (time, value) on [lo, hi], parabolic refinement."""
    out = []
    for k in range(1, len(t) - 1):
        if lo <= t[k] <= hi and v[k] > v[k - 1] and v[k] >= v[k + 1]:
            a = v[k - 1] - 2 * v[k] + v[k + 1]
            off = 0.5 * (v[k - 1] - v[k + 1]) / a if a != 0 else 0.0
            h = t[k + 1] - t[k]
            out.append((t[k] + off * h, v[k] - 0.25 * (v[k - 1] - v[k + 1]) * off))
    return out


def stats(a, b, t, lo, hi):
    idx = [k for k, ti in enumerate(t) if lo - 1e-9 <= ti <= hi + 1e-9]
    d = [a[k] - b[k] for k in idx]
    linf = max(abs(x) for x in d)
    rms = math.sqrt(sum(x * x for x in d) / len(d))
    ref = math.sqrt(sum(b[k] ** 2 for k in idx) / len(idx))
    kmax = max(idx, key=lambda k: abs(a[k] - b[k]))
    return dict(linf=linf, rms=rms, rms_rel=rms / ref, ref_rms=ref,
                t_of_linf=t[kmax], sup_a=max(a[k] for k in idx),
                sup_b=max(b[k] for k in idx))


def main(log, label="ex21"):
    te, ee = load_ex21(log)
    summary = {"label": label, "log": os.path.basename(log)}
    # [0,20] full; [0,19.7] excludes the terminal layer of the lambda(T) = 0
    # optimum (unresolved by any time step at eps = 1e-9, see the report);
    # [19.7,20] is reported separately.
    for fig, windows in (("fig32", [(0, 20), (0, 19.7), (10, 19.7), (19.7, 20)]),
                         ("fig33", [(0, 2), (0.5, 2)])):
        t, src = load_fig(fig)
        ours = [sample(te, ee, ti) for ti in t]
        for key in ("Eoc_c", "e2_cascade_c"):
            for lo, hi in windows:
                s = stats(ours, src[key], t, lo, hi)
                summary[f"{fig}_{key}_{lo}_{hi}"] = s
                print(f"{label:28s} {fig} vs {key:13s} [{lo:>4},{hi:>3}]"
                      f" Linf={s['linf']:.3e} (t={s['t_of_linf']:.2f}) RMS={s['rms']:.3e}"
                      f" RMSrel={s['rms_rel']:.3e} sup ex21={s['sup_a']:.5f} src={s['sup_b']:.5f}")
        if fig == "fig32":
            pa = peaks(t, ours, 3, 20)
            pb = peaks(t, src["Eoc_c"], 3, 20)
            n = min(len(pa), len(pb))
            shifts = [pa[k][0] - pb[k][0] for k in range(n)]
            amps = [pa[k][1] - pb[k][1] for k in range(n)]
            summary["peaks_ex21"] = pa
            summary["peaks_Eoc"] = pb
            print(f"{label:28s} peaks on [3,20]: ex21 {len(pa)}, E_oc {len(pb)};"
                  f" time shift mean={sum(shifts)/n:+.4f} max|.|={max(abs(x) for x in shifts):.4f};"
                  f" amplitude diff mean={sum(amps)/n:+.3e} max|.|={max(abs(x) for x in amps):.3e}")
            for (ta, va), (tb, vb) in zip(pa, pb):
                print(f"      peak ex21 t={ta:7.3f} {va:.5f}   E_oc t={tb:7.3f} {vb:.5f}")
    out = os.path.join(RES, f"math2025_ex3_compare_{label}.json")
    json.dump(summary, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "ex21")
