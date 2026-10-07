#!/usr/bin/env python3
"""Summarise the EF 9.6 tuning runs (logs/<label>_a<alpha>_N<N>.log) into
../tuning_summary.csv.  Efficiency = log10(r0 / r_final) / ksp_solve seconds
(outer residual from -ksp_monitor; ksp_solve time from the driver's
SOLVER_STATS line, or the process wall time if the run was cut by timeout)."""
import csv, glob, math, os, re
HERE = os.path.dirname(os.path.abspath(__file__))
rows = []
for f in sorted(glob.glob(os.path.join(HERE, "logs", "*.log"))):
    name = os.path.basename(f)[:-4]
    if name.startswith(("smoke", "schurmon")): continue  # option smoke test; Schur-monitor probe
    s = open(f).read()
    m = re.match(r"(.+)_a(1e-\d+)_N(\d+)$", name)
    label, a, n = m.group(1), m.group(2), int(m.group(3))
    res = [float(x) for x in re.findall(r"^\s*\d+ KSP Residual norm (\S+)", s, re.M)]
    schur = [(r, int(k)) for r, k in re.findall(r"Linear fieldsplit_schur_ solve (.*?) iterations (\d+)", s)]
    st = re.search(r"SOLVER_STATS (.*)$", s, re.M)
    kv = dict(re.findall(r"(\w+)=(\S+)", st.group(1))) if st else {}
    wall = float(re.search(r"WALL (\S+)", s).group(1)) if re.search(r"WALL (\S+)", s) else float("nan")
    t = float(kv.get("ksp_solve_s", wall))
    outer = len(res) - 1
    dec = math.log10(res[0] / res[-1]) if outer > 0 else 0.0
    sp_s = sum(int(kv.get(k + "_solves", 0)) for k in ("right", "left", "ysplit"))
    sp_i = sum(int(kv.get(k + "_its", 0)) for k in ("right", "left", "ysplit"))
    rows.append(dict(label=label, alpha_u=a, N=n, outer_its=outer, completed=bool(st),
                     r0=res[0], r_final=res[-1], decades=dec, ksp_solve_s=t, wall_s=wall,
                     efficiency_dec_per_s=dec / t if t > 0 else float("nan"),
                     s_per_decade=t / dec if dec > 0 else float("inf"), s_per_outer=t / outer if outer else float("nan"),
                     schur_its_total=sum(k for _, k in schur), schur_per_outer=(sum(k for _, k in schur) / len(schur)) if schur else 0,
                     schur_capped=sum("DIVERGED_ITS" in r for r, _ in schur), schur_solves=len(schur),
                     spatial_solves=sp_s, spatial_mean_its=(sp_i / sp_s) if sp_s else float("nan"),
                     sweep_failures=kv.get("sweep_failures", ""),
                     residual_history=" ".join(f"{x:.3e}" for x in res)))
out = os.path.join(os.path.dirname(HERE), "tuning_summary.csv")
if rows:
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in sorted(rows, key=lambda r: (r["N"], r["alpha_u"], r["label"])):
    print(f"{r['label']:22s} a={r['alpha_u']} N={r['N']:3d} outer={r['outer_its']} dec={r['decades']:.3f} t={r['ksp_solve_s']:.1f}s "
          f"eff={r['efficiency_dec_per_s']:.4f} s/dec={r['s_per_decade']:.1f} s/outer={r['s_per_outer']:.1f} "
          f"schur/outer={r['schur_per_outer']:.1f} capped={r['schur_capped']}/{r['schur_solves']} sp_mean={r['spatial_mean_its']:.2f} fail={r['sweep_failures']}")
    print("   hist:", r["residual_history"]) if False else None
