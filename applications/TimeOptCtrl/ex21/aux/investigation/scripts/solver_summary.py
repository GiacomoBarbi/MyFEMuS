#!/usr/bin/env python3
"""Cross-benchmark SCHUR_SWEEP solver summary for the Aulisa/Gilliam report.

Reads the existing production logs of Mathematics 2025 Ex. 3/4/5 and
ErrorFeedback 9.5/9.6 (no new runs) and writes
../results/aulisa_gilliam_solver_summary.csv, one row per configuration.

Columns from the driver's CONFIG / RESULT lines.  The Schur-split GMRES
(restart 30, rtol 1e-6, cap 100) is solved once per outer FGMRES iteration
(PCFIELDSPLIT Schur, FULL factorisation), so schur_its/outer_its = 100 means
every Schur solve stopped at its cap.  Spatial KSP iterations were not
recorded by these older runs (see solver_diag/ for the instrumented runs)."""
import csv, glob, os, re

I = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETS = [  # (benchmark, colocated, dim, glob)
    ("Math2025 Ex3", "no", 1, "math2025_ex3/logs/cn_sweep_nx40_nt1600.log"),
    ("Math2025 Ex3 (alpha sweep)", "no", 1, "math2025_ex4/logs/ex3sweep_be_a*_nx160_nt800.log"),
    ("Math2025 Ex4", "yes", 1, "math2025_ex4/logs/sweep_cn_a*_nx640_nt1200.log"),
    ("Math2025 Ex5", "no", 2, "math2025_ex5/logs/ref_cn_nx48_nz16_nt400.log"),
    ("EF 9.5", "yes", 2, "errorfeedback_95_96/logs/sweep_ef95_a*_cn_nx32_nz12_nt400.log"),
    ("EF 9.5", "yes", 2, "errorfeedback_95_96/logs/sweep_ef95_a*_cn_nx128_nz48_nt200.log"),
    ("EF 9.6", "no", 2, "errorfeedback_95_96/logs/sweep_ef96_a*_cn_nx32_nz12_nt400.log"),
    ("EF 9.6", "no", 2, "errorfeedback_95_96/logs/sweep_ef96_a*_cn_nx64_nz24_nt*.log"),
]
kv = lambda s: dict(re.findall(r"(\w+)=(\S+)", s))
rows = []
for name, coloc, dim, pat in SETS:
    for path in sorted(glob.glob(os.path.join(I, pat))):
        s = open(path).read()
        cfg = re.search(r"^EX21_\w+_CONFIG (.*)$", s, re.M)
        res = re.search(r"^EX21_\w+_RESULT (.*)$", s, re.M)
        if not (cfg and res): continue
        c, r = kv(cfg.group(1)), kv(res.group(1))
        nu, vx, vy = float(c["nu"]), float(c["vx"]), float(c.get("vy", 0))
        if dim == 1:
            pe = abs(vx) * (1.0 / int(c["nx"])) / (2 * nu)
        else:
            pe = float(c["Pe_cell"])
        outer, schur = int(r["outer_its"]), int(r["inner_p_its"])
        wall = re.search(r"WALL (\S+)", s)
        rows.append(dict(
            benchmark=name, dim=dim, colocated=coloc, nu=nu, velocity=f"({vx:g},{vy:g})" if dim == 2 else f"{vx:g}",
            Pe_cell=f"{pe:.3g}", alpha_u=float(c["alpha_u"]),
            mesh=c["nx"] + ("x" + c["nz"] if "nz" in c else ""), steps=int(c["steps"]), scheme=c["scheme"],
            outer_its=outer, schur_its=schur, schur_per_outer=f"{schur / outer:.1f}",
            schur_cap_hit="yes (every solve)" if schur >= 100 * outer else "no",
            spatial_its="n/r", true_residual=float(r["true_residual"]), reason=int(r["reason"]),
            sweep_failures=int(r["sweep_failures"]), wall_s=float(wall.group(1)) if wall else "",
            log=os.path.relpath(path, I)))
rows.append(dict(benchmark="EF 9.6 (ABORTED production)", dim=2, colocated="no", nu=1.0, velocity="(2,1)", Pe_cell="0.11",
                 alpha_u=1e-11, mesh="32x12", steps=400, scheme="CN", outer_its="n/a (killed)", schur_its="n/a",
                 schur_per_outer="", schur_cap_hit="(yes, per diagnostics)", spatial_its="n/r", true_residual="n/a",
                 reason="killed by user request", sweep_failures="", wall_s=37515.24,
                 log="errorfeedback_95_96/logs/ABORTED_sweep_ef96_a1e-11_cn_nx32_nz12_nt400.log"))
out = os.path.join(I, "results", "aulisa_gilliam_solver_summary.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
for r in rows:
    print(f"{r['benchmark']:28s} col={r['colocated']:3s} a={r['alpha_u']:.0e} {r['mesh']:>6s} N={r['steps']:<5} {r['scheme']} "
          f"Pe={r['Pe_cell']:>6s} outer={r['outer_its']!s:>4} schur={r['schur_its']!s:>5} s/o={r['schur_per_outer']:>5} "
          f"res={r['true_residual']!s:>9} wall={r['wall_s']}")
