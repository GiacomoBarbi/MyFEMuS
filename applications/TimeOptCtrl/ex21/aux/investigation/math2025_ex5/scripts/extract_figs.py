#!/usr/bin/env python3
"""Vector extraction of Mathematics 2025 (13, 3707), page 31, Figure 46
("||e3||_c, ||E_oc||_c", Example 5, alpha = eps = 1e-7) and Figure 45
("||e3||_c, ||e||_c, alpha = 1e-9": cascade error and the PLANT error with
initial condition phi = xy; blue / red as in Example 3's Figs. 30-31).

Pipeline as Examples 3-4: pdftocairo -svg -f 31 -l 31 -> stroked paths with
accumulated transforms (svg_paths.py).  This panel has NO grid lines; the axis
map is fitted to the dark-grey TICK MARKS, whose values are the tick labels of
the PDF TEXT LAYER (pdftotext -bbox): x 0,2,...,10; y 0,0.01,...,0.06.

Curve identity [B]: the caption gives no colours.  Pure blue (0,0,1) is the
cascade colour of every other panel of Example 5 (Figs. 43-45, 197 vertices on
the cascade grid dt = 0.05); the second curve is the plotting-default second
colour (0.85,0.33,0.10), 99 vertices on its own grid dt = 0.1, t = 0.1..9.9,
the signature of a separate (optimal-control) run.  Caption order e3, E_oc ->
blue = e3 (cascade), orange = E_oc.

Writes ../../results/math2025_ex5_fig{45,46}_raw.csv (curve, t, value) and
../../results/math2025_ex5_fig{45,46}_extraction.json.
"""
import csv, json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "math2025_ex3", "scripts"))
from svg_paths import stroked_paths

PDF = os.path.expanduser("~/Downloads/mathematics-13-03707.pdf")
RES = os.path.join(HERE, "..", "..", "results")
TICK = "rgb(12.988281%, 12.988281%, 12.988281%)"
BLUE, ORANGE, RED = ("rgb(0%, 0%, 100%)", "rgb(84.959412%, 33.006287%, 10.009766%)",
                     "rgb(100%, 0%, 0%)")
XT = [0, 2, 4, 6, 8, 10]
FIGS = {
    46: dict(band=(527, 618), yt=[round(.01 * k, 2) for k in range(7)],
             curves=((BLUE, "e3_cascade_c"), (ORANGE, "Eoc_c")),
             identification="[B] blue=e3 cascade (Ex.5 cascade colour/grid), orange=E_oc (own run, dt=0.1)"),
    45: dict(band=(415, 505), yt=[round(.01 * k, 2) for k in range(10)],
             curves=((BLUE, "e3_cascade_c"), (RED, "e_plant_c")),
             identification="[B] blue=e3 cascade, red=plant error with phi=xy (Ex.3 Figs.30-31 convention; "
                            "red starts at 0.089 and decays like a free response)"),
}


def linfit(px, vals):
    n = len(px); mx, my = sum(px) / n, sum(vals) / n
    a = sum((p - mx) * (v - my) for p, v in zip(px, vals)) / sum((p - mx) ** 2 for p in px)
    b = my - a * mx
    return a, b, max(abs(a * p + b - v) for p, v in zip(px, vals))


def main(svg, fig):
    BAND, YT = FIGS[fig]["band"], FIGS[fig]["yt"]
    P = [p for p in stroked_paths(svg) if all(BAND[0] <= q[1] <= BAND[1] for q in p["pts"])]
    ticks = [p for p in P if p["stroke"] == TICK and len(p["pts"]) == 2]
    # short vertical ticks on the bottom axis -> x; short horizontal ticks on the left axis -> y
    vx = sorted({round(p["pts"][0][0], 4) for p in ticks
                 if abs(p["pts"][0][0] - p["pts"][1][0]) < 1e-6 and abs(p["pts"][0][1] - p["pts"][1][1]) < 2})
    left = min(min(q[0] for q in p["pts"]) for p in ticks)
    hy = sorted({round(p["pts"][0][1], 4) for p in ticks
                 if abs(p["pts"][0][1] - p["pts"][1][1]) < 1e-6 and abs(p["pts"][0][0] - p["pts"][1][0]) < 2
                 and abs(min(q[0] for q in p["pts"]) - left) < 1e-3}, reverse=True)
    if len(vx) != len(XT) or len(hy) != len(YT):
        sys.exit(f"ticks {len(vx)}x{len(hy)} do not match the labels")
    ax, bx, rx = linfit(vx, XT); ay, by, ry = linfit(hy, YT)
    rows, info = [], {}
    for col, key in FIGS[fig]["curves"]:
        cs = [p for p in P if p["stroke"] == col]
        if len(cs) != 1:
            sys.exit(f"{key}: {len(cs)} paths")
        pts = [(ax * x + bx, ay * y + by) for x, y in cs[0]["pts"]]
        for t, v in pts:
            rows.append((key, t, v))
        info[key] = dict(stroke=col, vertices=len(pts), t_first=pts[0][0], t_last=pts[-1][0],
                         max_value=max(v for _, v in pts),
                         t_at_max=max(pts, key=lambda q: q[1])[0])
    with open(os.path.join(RES, f"math2025_ex5_fig{fig}_raw.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["curve", "t", "value"])
        for k, t, v in rows:
            w.writerow([k, f"{t:.6f}", f"{v:.8e}"])
    # value quantum: smallest nonzero |dy| between vertices in page units, mapped to data
    dys = sorted({abs(round(b[1] - a[1], 6)) for p in P if p["stroke"] in [c for c, _ in FIGS[fig]["curves"]]
                  for a, b in zip(p["pts"], p["pts"][1:])} - {0.0})
    out = dict(source=PDF, page=31, figure=fig, svg=svg, tick_x_page=vx, tick_y_page=hy,
               x_labels=XT, y_labels=YT, x_map=[ax, bx], y_map=[ay, by],
               calibration_residual_t=rx, calibration_residual_value=ry,
               value_quantum=abs(ay) * dys[0], stroke_width_value=abs(ay) * 0.5,
               t_quantum_page_to_data=abs(ax) * 1e-3, curves=info,
               identification=FIGS[fig]["identification"])
    json.dump(out, open(os.path.join(RES, f"math2025_ex5_fig{fig}_extraction.json"), "w"), indent=1)
    print(fig, json.dumps({k: out[k] for k in ("calibration_residual_t", "calibration_residual_value",
                                              "value_quantum", "curves")}, indent=1))


if __name__ == "__main__":
    svg = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "logs", "page31.svg")
    if not os.path.exists(svg):
        subprocess.check_call(["pdftocairo", "-svg", "-f", "31", "-l", "31", PDF, svg])
    for fig in (45, 46):
        main(svg, fig)
