#!/usr/bin/env python3
"""Pointwise comparison of ex21 Example 5 runs with the vector-extracted
Fig. 46 curves (primary: E_oc, orange; secondary: cascade e3, blue).

For every log in ../logs matching the run-name patterns: interpolate the ex21
series ||y - r||_{L2(Omega_c)} (linear in t, the ex21 grid is finer than the
source's 0.1) at the source sample times and report
  * L_inf and RMS differences on [0.1, 9.5] (main), [0.1, 3] (transient),
    [3, 9.5] (periodic regime), [9.5, 9.9] (terminal window, separately);
  * peaks (local maxima) and troughs (local minima): value and time;
  * phase: the shift tau in [-0.3, 0.3] minimising the RMS on [3, 9.5];
  * the analytic ||r(t)||_{L2(Omega_c)} of the geometry, to separate what the
    controller does from what the reference alone fixes.
Writes ../../results/math2025_ex5_comparison.{csv,json}."""
import csv, glob, json, math, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "..", "logs")
RES = os.path.join(HERE, "..", "..", "results")


def F(s, eps=1.5):
    if s <= -eps: return 0.0
    if s >= eps: return 1.0
    t = (s + eps) / (2 * eps)
    return t ** 5 * (126 + t * (-420 + t * (540 + t * (-315 + t * 70))))


def r_norm(t, W, H):
    # ||x y(1-y)(y-1/2)||_{L2([W/3,2W/3] x [0,H])} |sin t F(t-1.5)|
    ix = ((2 * W / 3) ** 3 - (W / 3) ** 3) / 3
    n = 4000; h = H / n
    iy = sum(((k + .5) * h * (1 - (k + .5) * h) * ((k + .5) * h - .5)) ** 2 for k in range(n)) * h
    return math.sqrt(ix * iy) * abs(math.sin(t) * F(t - 1.5))


def source():
    rows = list(csv.DictReader(open(os.path.join(RES, "math2025_ex5_fig46_raw.csv"))))
    cur = {}
    for r in rows:
        cur.setdefault(r["curve"], []).append((float(r["t"]), float(r["value"])))
    # E_oc is sampled at t = 0.1 k: snap (calibration residual 1.4e-4)
    cur["Eoc_c"] = [(round(t * 10) / 10, v) for t, v in cur["Eoc_c"]]
    cur["e3_cascade_c"] = [(round(t * 20) / 20, v) for t, v in cur["e3_cascade_c"]]
    return cur


def parse(path):
    s = open(path).read()
    cfg = dict(re.findall(r"(\w+)=(\S+)", re.search(r"^EX21_EX5_CONFIG (.*)$", s, re.M).group(1)))
    res = dict(re.findall(r"(\w+)=(\S+)", re.search(r"^EX21_EX5_RESULT (.*)$", s, re.M).group(1)))
    ser = [(float(a), float(b)) for a, b in re.findall(r"^EX21_EX5_SERIES t=(\S+) e_c=(\S+)$", s, re.M)]
    gates = re.findall(r"^EX21_EX5_GATE_STATE .*$", s, re.M)
    return cfg, res, ser, gates


def interp(ser, t):
    lo, hi = 0, len(ser) - 1
    if t <= ser[0][0]: return ser[0][1]
    if t >= ser[-1][0]: return ser[-1][1]
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if ser[mid][0] <= t: lo = mid
        else: hi = mid
    (t0, v0), (t1, v1) = ser[lo], ser[hi]
    return v0 + (v1 - v0) * (t - t0) / (t1 - t0)


def extrema(pts, kind):
    out = []
    for i in range(1, len(pts) - 1):
        a, b, c = pts[i - 1][1], pts[i][1], pts[i + 1][1]
        if (kind == "max" and b >= a and b > c) or (kind == "min" and b <= a and b < c):
            out.append(pts[i])
    return out


def metrics(src, ser, lo, hi, shift=0.0):
    d = [interp(ser, t + shift) - v for t, v in src if lo - 1e-9 <= t <= hi + 1e-9]
    if not d: return None, None
    return max(abs(x) for x in d), math.sqrt(sum(x * x for x in d) / len(d))


def main():
    cur = source()
    out = []
    for path in sorted(glob.glob(os.path.join(LOG, "*.log"))):
        name = os.path.basename(path)[:-4]
        if name.startswith(("gates", "campaign")): continue
        try:
            cfg, res, ser, gates = parse(path)
        except AttributeError:
            continue
        W, H = float(cfg["W"]), float(cfg["H"])
        row = dict(run=name, W=W, H=H, nx=int(cfg["nx"]), nz=int(cfg["nz"]), steps=int(cfg["steps"]),
                   scheme=cfg["scheme"], initial=cfg["initial"], Pe_cell=float(cfg["Pe_cell"]),
                   outer_its=int(res["outer_its"]), true_residual=float(res["true_residual"]),
                   J_track=float(res["J_track"]), J_reg=float(res["J_reg"]), u_L2=float(res["u_L2"]))
        for key, src in (("oc", cur["Eoc_c"]), ("casc", cur["e3_cascade_c"])):
            for wname, lo, hi in (("main", 0.1, 9.5), ("trans", 0.1, 3.0), ("per", 3.0, 9.5), ("term", 9.5, 9.9 if key == "oc" else 10.0)):
                linf, rms = metrics(src, ser, lo, hi)
                row[f"{key}_{wname}_Linf"], row[f"{key}_{wname}_rms"] = linf, rms
        # phase vs E_oc on [3, 9.5]
        best = min((metrics(cur["Eoc_c"], ser, 3.0, 9.5, s / 1000)[1], s / 1000) for s in range(-300, 301, 5))
        row["oc_phase_tau"], row["oc_phase_rms"] = best[1], best[0]
        # extrema of ex21 (sampled on the source grid) vs E_oc
        model = [(t, interp(ser, t)) for t, _ in cur["Eoc_c"]]
        for kind in ("max", "min"):
            ms, ss = extrema(model, kind), extrema(cur["Eoc_c"], kind)
            row[f"oc_{kind}"] = [(s[0], s[1], m[0], m[1]) for s in ss for m in [min(ms, key=lambda q: abs(q[0] - s[0]))]] if ms else []
        row["r_norm_max"] = max(r_norm(t, W, H) for t, _ in cur["Eoc_c"])
        row["sup_e"] = max(v for _, v in ser if v == v)
        out.append(row)
    flat = [{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()} for r in out]
    keys = list(flat[0].keys())
    with open(os.path.join(RES, "math2025_ex5_comparison.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(flat)
    json.dump(out, open(os.path.join(RES, "math2025_ex5_comparison.json"), "w"), indent=1)
    src_max = [(round(t, 2), round(v, 5)) for t, v in extrema(cur["Eoc_c"], "max")]
    src_min = [(round(t, 2), round(v, 5)) for t, v in extrema(cur["Eoc_c"], "min")]
    print("source E_oc maxima", src_max, "minima", src_min)
    print(f"{'run':44s} {'W':>6s} {'H':>3s} ic   main_Linf  main_rms  trans_Linf per_Linf  term_Linf  tau    | casc_main_Linf  sup_e    ||r||max  its")
    for r in out:
        print(f"{r['run']:44s} {r['W']:6.3f} {r['H']:3.0f} {r['initial'][:4]:4s} {r['oc_main_Linf']:.3e} {r['oc_main_rms']:.3e} "
              f"{r['oc_trans_Linf']:.3e} {r['oc_per_Linf']:.3e} {r['oc_term_Linf']:.3e} {r['oc_phase_tau']:+.3f} | "
              f"{r['casc_main_Linf']:.3e} {r['sup_e']:.4e} {r['r_norm_max']:.4e} {r['outer_its']}")
        print("     max:", [(s[0], round(s[1], 5), m[0], round(m[3], 5)) for s, m in [(q[:2], q) for q in r["oc_max"]]],
              " min:", [(q[0], round(q[1], 5), round(q[3], 5)) for q in r["oc_min"]])


if __name__ == "__main__":
    main()
