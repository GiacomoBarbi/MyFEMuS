#!/usr/bin/env python3
"""Summarise the EF 9.6 SCHUR_SWEEP diagnostics (run_diag.sh) into
diag_summary.csv: outer progress, Schur cap hits, spatial iteration
statistics (overall and per time slot) and cost per outer iteration."""
import csv, os, re
HERE = os.path.dirname(os.path.abspath(__file__)); L = os.path.join(HERE, "logs")
out = []
for a in ("1e-10", "1e-11"):
    for n in (50, 100, 200):
        f = os.path.join(L, f"diag_ef96_a{a}_N{n}")
        s = open(f + ".log").read()
        res = [float(x) for x in re.findall(r"^\s*\d+ KSP Residual norm (\S+)", s, re.M)]
        schur = re.findall(r"Linear fieldsplit_schur_ solve (.*) iterations (\d+)", s)
        wall = float(re.search(r"WALL (\S+)", s).group(1))
        rows = list(csv.DictReader(open(f + ".slots.csv")))
        solves = sum(int(r["solves"]) for r in rows)
        its = sum(float(r["mean_its"]) * int(r["solves"]) for r in rows)
        slot_means = [float(r["mean_its"]) for r in rows]
        outer = len(res) - 1
        at = lambda k: res[min(k, outer)]
        out.append(dict(alpha_u=a, N=n, status="capped at 3 h (not converged)", outer_its_done=outer,
                        outer_res0=res[0], outer_res_10=at(10), outer_res_36=at(36), outer_res_last=res[-1],
                        outer_reduction=res[0] / res[-1], target_reduction=1e8,
                        schur_solves=len(schur), schur_capped=sum("DIVERGED_ITS" in r for r, _ in schur),
                        spatial_solves=solves, spatial_its=round(its), spatial_mean=its / solves,
                        spatial_min=min(int(r["min_its"]) for r in rows), spatial_max=max(int(r["max_its"]) for r in rows),
                        slot_mean_min=min(slot_means), slot_mean_max=max(slot_means),
                        spatial_failures=sum(int(x) for x in re.findall(r"failures=(\d+)", s)),
                        wall_s=wall, s_per_outer=wall / outer, spatial_solves_per_outer=solves / outer))
with open(os.path.join(HERE, "diag_summary.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
for r in out:
    print(" | ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))
