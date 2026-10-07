#!/usr/bin/env python3
"""Vector extraction of Mathematics 2025 (13, 3707), page 26, Figures 30-33.

Source: ~/Downloads/mathematics-13-03707.pdf.  Pipeline:
  pdftocairo -svg -f 26 -l 26  ->  every stroked path with its accumulated
  transform (svg_paths.py)  ->  axis map from the light-grey GRID LINES, whose
  values are the tick labels read from the PDF TEXT LAYER (pdftotext -bbox),
  never from a raster.  Curves are identified by stroke colour, as the caption
  states: Fig. 32/33 "||e2(t)||_c (in blue) ... E_oc (in red)"; Fig. 30/31
  "||E0||_c, ||e||_c" (blue, red).

Writes ../results/math2025_ex3_fig{30,31,32,33}.csv with columns
t, value for each curve, plus an uncertainty summary (JSON).
"""
import csv, json, os, subprocess, sys
from svg_paths import stroked_paths

PDF = os.path.expanduser("~/Downloads/mathematics-13-03707.pdf")
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(os.path.dirname(HERE)), "results")   # investigation/results
GRID = "rgb(86.914062%, 86.914062%, 86.914062%)"
BLUE, RED = "rgb(0%, 0%, 100%)", "rgb(100%, 0%, 0%)"
# page-space vertical band of each panel, x ticks, y ticks (from the text layer)
FIGS = {
    "fig30": dict(band=(110, 215), xt=[0, 5, 10, 15, 20], yt=None, blue="E0_c", red="e_plant_c"),
    "fig31": dict(band=(240, 345), xt=[0, 0.5, 1, 1.5, 2], yt=None, blue="E0_c", red="e_plant_c"),
    "fig32": dict(band=(440, 545), xt=[0, 5, 10, 15, 20],
                  yt=[0, .01, .02, .03, .04, .05, .06], blue="e2_cascade_c", red="Eoc_c"),
    "fig33": dict(band=(570, 675), xt=[0, 0.5, 1, 1.5, 2],
                  yt=[0, .01, .02, .03, .04, .05], blue="e2_cascade_c", red="Eoc_c"),
}


def linfit(px, vals):
    n = len(px)
    mx, my = sum(px) / n, sum(vals) / n
    a = sum((p - mx) * (v - my) for p, v in zip(px, vals)) / sum((p - mx) ** 2 for p in px)
    b = my - a * mx
    resid = max(abs(a * p + b - v) for p, v in zip(px, vals))
    return a, b, resid


def main(svg):
    P = stroked_paths(svg)
    summary = {}
    for name, f in FIGS.items():
        y0, y1 = f["band"]
        inband = [p for p in P if all(y0 <= q[1] <= y1 for q in p["pts"])]
        grid = [p for p in inband if p["stroke"] == GRID]
        vx = sorted({round(p["pts"][0][0], 4) for p in grid
                     if abs(p["pts"][0][0] - p["pts"][-1][0]) < 1e-6})
        hy = sorted({round(p["pts"][0][1], 4) for p in grid
                     if abs(p["pts"][0][1] - p["pts"][-1][1]) < 1e-6}, reverse=True)
        yt = f["yt"] or [round(0.02 * k, 2) for k in range(len(hy))]
        if len(vx) != len(f["xt"]) or len(hy) != len(yt):
            sys.exit(f"{name}: grid {len(vx)}x{len(hy)} does not match the labels")
        ax, bx, rx = linfit(vx, f["xt"])
        ay, by, ry = linfit(hy, yt)
        curves = {}
        for col, key in ((BLUE, f["blue"]), (RED, f["red"])):
            segs = [p for p in inband if p["stroke"] == col]
            pts = [q for s in segs for q in s["pts"]]
            data = sorted({(round(ax * x + bx, 9), ay * y + by) for x, y in pts})
            curves[key] = data
        # The vertices are the plotted samples on a uniform output grid of step
        # 0.05 (401 vertices on [0,20], 41 on [0,2]); their page coordinates are
        # quantized (steps 0.0499/0.0506 in Fig. 32), so t is SNAPPED to the
        # grid and the snap distance is reported -- it must stay << h/2.
        h = 0.05
        allt = sorted({t for c in curves.values() for t, _ in c})
        dev = max(abs(t / h - round(t / h)) * h for t in allt)
        if dev > 0.1 * h:
            sys.exit(f"{name}: vertex {dev} off the 0.05 grid")
        # value quantum: smallest nonzero difference between distinct ordinates
        ys = sorted({round(v, 15) for c in curves.values() for _, v in c})
        dq = [b - a for a, b in zip(ys, ys[1:]) if b - a > 1e-12]
        quantum = min(dq) if dq else 0.0
        out = os.path.join(RES, f"math2025_ex3_{name}.csv")
        keys = list(curves)
        tset = sorted({round(t / h) for c in curves.values() for t, _ in c})
        maps = {k: {round(t / h): v for t, v in curves[k]} for k in keys}
        with open(out, "w", newline="") as fo:
            w = csv.writer(fo)
            w.writerow(["t"] + keys)
            for k in tset:
                w.writerow([f"{k * h:.6f}"] + [("%.8e" % maps[c][k]) if k in maps[c] else "" for c in keys])
        both = [k for k in tset if all(k in maps[c] for c in keys)]
        diff = max(abs(maps[keys[0]][k] - maps[keys[1]][k]) for k in both)
        summary[name] = dict(
            x_ticks_page=vx, y_ticks_page=hy, x_labels=f["xt"], y_labels=yt,
            x_map=[ax, bx], y_map=[ay, by], x_fit_resid=rx, y_fit_resid=ry,
            output_step=h, max_vertex_offgrid=dev, value_quantum=quantum,
            n_points={k: len(curves[k]) for k in keys},
            max_abs_blue_minus_red_common_t=diff,
            y_units_per_page_unit=abs(ay), csv=os.path.basename(out))
        print(f"{name}: step={h:.6g} snap={dev:.2e} quantum={quantum:.2e} xres={rx:.1e} yres={ry:.1e}"
              f" n={summary[name]['n_points']} |blue-red|max={diff:.3e}"
              f" dy/dpage={abs(ay):.3e}")
    json.dump(summary, open(os.path.join(RES, "math2025_ex3_extraction.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
