#!/usr/bin/env python3
"""Vector extraction of ErrorFeedback2026 (3 Oct 2026) Figures 29, 30, 33, 34
(pages 37, 39): ||e(t)||, ||e_j(t)|| in L2(Omega_C).  Same pipeline as the
Mathematics 2025 extractions (svg_paths.py): grid lines -> axis map with the
tick values of the PDF text layer; curves by stroke colour (Figs. 29/33: red =
plant e, blue = last iteration error; Figs. 30/34: plotting-default colour
order = e1, e2, e3, e4[, e5], matching the legends).  Writes RAW CSVs
../../results/errorfeedback_fig{29,30,33,34}_raw.csv and a tail-window scan
against Tables 9 and 10.  usage: extract_figs.py <scratch with ef37.svg, ef39.svg>"""
import csv, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "math2025_ex3", "scripts"))
from svg_paths import stroked_paths  # noqa: E402
RES = os.path.join(HERE, "..", "..", "results")
GRID = "rgb(86.914062%, 86.914062%, 86.914062%)"
ORDER = ["rgb(7.006836%, 43.992615%, 75%)", "rgb(86.914062%, 33.006287%, 0%)",
         "rgb(92.96875%, 68.943787%, 12.988281%)", "rgb(51.951599%, 9.008789%, 82.03125%)",
         "rgb(22.996521%, 66.992188%, 19.993591%)"]
lin = lambda a, b, n: [a + (b - a) * k / (n - 1) for k in range(n)]
PANELS = {
    "fig29": (37, (95, 300), (250, 385), lin(2, 20, 10), lin(0, 1.4e-3, 8),
              [("rgb(100%, 0%, 0%)", "e_plant"), ("rgb(0%, 0%, 100%)", "e_last")]),
    "fig30": (37, (325, 535), (250, 385), [5, 10, 15, 20], lin(0, 4e-3, 9),
              [(ORDER[k], f"e{k+1}") for k in range(4)]),
    "fig33": (39, (95, 300), (30, 160), lin(2, 20, 10), lin(0, 0.06, 7),
              [("rgb(100%, 0%, 0%)", "e_plant"), ("rgb(0%, 0%, 100%)", "e5")]),
    "fig34": (39, (325, 535), (30, 160), [5, 10, 15, 20], lin(0, 0.07, 8),
              [(ORDER[k], f"e{k+1}") for k in range(5)]),
}
TABLE9 = [1.25e-2, 3.93e-3, 1.70e-3, 8.76e-4, 5.15e-4]
TABLE10 = [6.355e-2, 5.496e-2, 5.338e-2, 5.264e-2, 5.220e-2]


def fit(px, v):
    n = len(px); mx = sum(px) / n; my = sum(v) / n
    a = sum((p - mx) * (q - my) for p, q in zip(px, v)) / sum((p - mx) ** 2 for p in px)
    return a, my - a * mx, max(abs(a * p + my - a * mx - q) for p, q in zip(px, v))


def main(scratch):
    pages = {p: stroked_paths(os.path.join(scratch, f"ef{p}.svg")) for p in (37, 39)}
    data, record = {}, {}
    for name, (page, xb, yb, xt, yt, cols) in PANELS.items():
        P = [q for q in pages[page] if all(xb[0] <= a <= xb[1] and yb[0] <= b <= yb[1] for a, b in q["pts"])]
        g = [q for q in P if q["stroke"] == GRID]
        vx = sorted({round(q["pts"][0][0], 4) for q in g if abs(q["pts"][0][0] - q["pts"][-1][0]) < 1e-6})
        hy = sorted({round(q["pts"][0][1], 4) for q in g if abs(q["pts"][0][1] - q["pts"][-1][1]) < 1e-6}, reverse=True)
        assert len(vx) == len(xt) and len(hy) == len(yt), (name, len(vx), len(hy))
        ax, bx, rx = fit(vx, xt); ay, by, ry = fit(hy, yt)
        rows = []
        for col, key in cols:
            pts = [(ax * x + bx, ay * y + by) for q in P if q["stroke"] == col and len(q["pts"]) > 20 for x, y in q["pts"]]
            data[(name, key)] = pts
            rows += [(key, t, v) for t, v in pts]
        with open(os.path.join(RES, f"errorfeedback_{name}_raw.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["curve", "t", "value"])
            w.writerows([(k, f"{t:.6f}", f"{v:.8e}") for k, t, v in rows])
        record[name] = dict(x_fit_resid=rx, y_fit_resid=ry, n={k: len(data[(name, k)]) for _, k in cols},
                            t_range=[min(t for _, k in cols for t, _ in data[(name, k)]),
                                     max(t for _, k in cols for t, _ in data[(name, k)])])
        print(name, f"xres={rx:.1e} yres={ry:.1e}", record[name]["n"], [round(x, 3) for x in record[name]["t_range"]])
    # tail-window scan: sup_{[tsup, 20]} of each e_j against the tables
    scan = {}
    for fig, table, nj in (("fig30", TABLE9, 4), ("fig34", TABLE10, 5)):
        print(f"\n{fig}: sup over [tsup,20] vs Table {'9' if fig == 'fig30' else '10'}")
        for tsup in (2, 4, 5, 6, 8, 10, 12, 15):
            sups = [max(v for t, v in data[(fig, f"e{j}")] if t >= tsup) for j in range(1, nj + 1)]
            rel = [s / table[j] - 1 for j, s in enumerate(sups)]
            scan[f"{fig}_tsup{tsup}"] = dict(sups=sups, rel_vs_table=rel)
            print(f"  tsup={tsup:>2}: " + " ".join(f"{s:.4e}({r:+.3f})" for s, r in zip(sups, rel)))
    record["tail_scan"] = scan
    json.dump(record, open(os.path.join(RES, "errorfeedback_95_96_extraction.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
