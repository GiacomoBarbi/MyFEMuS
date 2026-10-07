#!/usr/bin/env python3
"""Vector extraction of Mathematics 2025, Example 4, Figures 35-40 (pages 28-29).

Same pipeline as Example 3 (../../math2025_ex3/scripts/svg_paths.py):
pdftocairo -svg -> stroked paths with accumulated transforms -> axis map from
the light-grey grid lines, whose values are the tick labels of the PDF TEXT
LAYER (pdftotext -bbox; listed below with the x10^k multiplier printed on the
axis) -> curves identified by stroke colour per the source's colour code:
  Figs 35-38: ||e1|| blue, ||e2|| red, ||e3|| magenta, ||e4|| green, ||e5|| black
  Figs 39-40: ||e5||_c blue, plant ||e(t)||_c red.
RAW output: every vertex, mapped, no snapping/cleanup:
  ../../results/math2025_ex4_fig{35..40}_raw.csv  (curve, t, value)
plus an extraction record (calibration residuals, value quantum, ranges).
usage: extract_figs.py <scratch dir holding p28.svg, p29.svg>
"""
import csv, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "math2025_ex3", "scripts"))
from svg_paths import stroked_paths  # noqa: E402

RES = os.path.join(HERE, "..", "..", "results")
GRID = "rgb(86.914062%, 86.914062%, 86.914062%)"
COL5 = [("rgb(0%, 0%, 100%)", "e1"), ("rgb(100%, 0%, 0%)", "e2"),
        ("rgb(100%, 0%, 100%)", "e3"), ("rgb(0%, 100%, 0%)", "e4"),
        ("rgb(0%, 0%, 0%)", "e5")]
COL2 = [("rgb(0%, 0%, 100%)", "e5_cascade"), ("rgb(100%, 0%, 0%)", "e_plant")]
lin = lambda a, b, n: [a + (b - a) * k / (n - 1) for k in range(n)]
FIGS = {  # page, page-y band, x tick values, y tick values (scaled), curves, alpha
    "fig35": (28, (60, 172), [0, 5, 10, 15], lin(0, 0.14, 8), COL5, 1e-3),
    "fig36": (28, (195, 312), [0, 5, 10, 15], lin(0, 8e-3, 5), COL5, 1e-5),
    "fig37": (28, (340, 452), [0, 5, 10, 15], lin(0, 2.5e-3, 6), COL5, 1e-7),
    "fig38": (28, (480, 594), [0, 5, 10, 15], lin(0, 5e-4, 6), COL5, 1e-9),
    "fig39": (29, (65, 162), [0, 5, 10, 15], lin(0, 3.5e-4, 8), COL2, 1e-9),
    "fig40": (29, (185, 280), lin(0, 3, 7), lin(0, 1.5e-3, 4), COL2, 1e-9),
}


def linfit(px, vals):
    n = len(px)
    mx, my = sum(px) / n, sum(vals) / n
    a = sum((p - mx) * (v - my) for p, v in zip(px, vals)) / sum((p - mx) ** 2 for p in px)
    b = my - a * mx
    return a, b, max(abs(a * p + b - v) for p, v in zip(px, vals))


def main(scratch):
    pages = {p: stroked_paths(os.path.join(scratch, f"p{p}.svg")) for p in (28, 29)}
    record = {}
    for name, (page, (y0, y1), xt, yt, cols, alpha) in FIGS.items():
        P = [p for p in pages[page] if all(y0 <= q[1] <= y1 for q in p["pts"])]
        grid = [p for p in P if p["stroke"] == GRID]
        vx = sorted({round(p["pts"][0][0], 4) for p in grid if abs(p["pts"][0][0] - p["pts"][-1][0]) < 1e-6})
        hy = sorted({round(p["pts"][0][1], 4) for p in grid if abs(p["pts"][0][1] - p["pts"][-1][1]) < 1e-6}, reverse=True)
        if len(vx) != len(xt) or len(hy) != len(yt):
            sys.exit(f"{name}: grid {len(vx)}x{len(hy)} vs labels {len(xt)}x{len(yt)}")
        ax, bx, rx = linfit(vx, xt)
        ay, by, ry = linfit(hy, yt)
        rows, info = [], {}
        for col, key in cols:
            segs = [p for p in P if p["stroke"] == col and len(p["pts"]) > 15]
            pts = [(ax * x + bx, ay * y + by) for s in segs for x, y in s["pts"]]
            rows += [(key, t, v) for t, v in pts]
            vals = sorted({round(v, 14) for _, v in pts})
            dq = [b - a for a, b in zip(vals, vals[1:]) if b - a > 1e-15]
            info[key] = dict(n=len(pts), t_range=[min(t for t, _ in pts), max(t for t, _ in pts)],
                             v_range=[min(v for _, v in pts), max(v for _, v in pts)],
                             value_quantum=min(dq) if dq else 0.0)
        with open(os.path.join(RES, f"math2025_ex4_{name}_raw.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["curve", "t", "value"])
            w.writerows([(k, f"{t:.6f}", f"{v:.8e}") for k, t, v in rows])
        record[name] = dict(page=page, alpha=alpha, x_grid_page=vx, y_grid_page=hy, x_labels=xt,
                            y_labels=yt, x_fit_resid=rx, y_fit_resid=ry,
                            y_axis_top=yt[-1], curves=info)
        print(name, f"alpha={alpha:g} xres={rx:.1e} yres={ry:.1e}",
              {k: (v["n"], [round(x, 4) for x in v["t_range"]], [f"{x:.4e}" for x in v["v_range"]],
                   f"q={v['value_quantum']:.1e}") for k, v in info.items()})
    json.dump(record, open(os.path.join(RES, "math2025_ex4_extraction.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
