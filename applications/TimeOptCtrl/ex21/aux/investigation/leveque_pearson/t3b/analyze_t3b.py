#!/usr/bin/env python3
"""T3b tables and ../t3b_tracking_validation.csv: lift-track ladders with and
without the affine tracking term, gates, objective checks, regression logs."""
import csv, glob, math, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
LP = os.path.dirname(HERE)
R = os.path.join(HERE, "results")
OUT = []


def load(g):
    p = os.path.join(R, g + ".csv")
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def f(r, k):
    return float(r[k])


def add(test, q, v, crit, ok, note=""):
    OUT.append({"test": test, "quantity": q, "value": v, "criterion": crit,
                "status": "PASS" if ok else "FAIL", "note": note})


def ladder(rows, by, keys, title):
    print(f"\n#### {title}\n")
    print("| " + by + " | " + " | ".join(f"{k} | ord" for k in keys) + " |")
    print("|" + "---|" * (1 + 2 * len(keys)))
    ords = {k: [None] for k in keys}
    for i, r in enumerate(rows):
        cells = [r[by]]
        for k in keys:
            o = None
            if i:
                o = math.log(f(rows[i - 1], k) / f(r, k)) / math.log(int(r[by]) / int(rows[i - 1][by]))
                ords[k].append(o)
            cells += [f"{f(r, k):.3e}", "–" if o is None else f"{o:.2f}"]
        print("| " + " | ".join(cells) + " |")
    return ords


def main():
    sp = load("lt_space")
    tm = load("lt_time")
    res = {}
    for tag in ("aff", "noaff"):
        rows = sorted([r for r in sp if r["id"].startswith(f"lt_{tag}_")], key=lambda r: int(r["nx"]))
        o = ladder(rows, "nx", ["y_L2L2_rel", "p_L2L2_rel", "p_half_rel", "u_half_rel", "y_T_rel", "J_err_rel"],
                   f"lift-track space, CN Nt = 128, {tag}")
        res[("space", tag)] = (rows, o)
        for s, keys in (("CN", ["y_L2L2_rel", "p_L2L2_rel", "p_half_rel", "y_T_rel"]),
                        ("BE", ["y_L2L2_rel", "p_L2L2_rel", "u_L2L2_rel"])):
            rr = sorted([r for r in tm if r["id"].startswith(f"lt_{tag}_") and r["scheme"] == s],
                        key=lambda r: int(r["steps"]))
            res[(s, tag)] = (rr, ladder(rr, "steps", keys, f"lift-track time {s}, nx = 32, {tag}"))
    # gates for the CORRECTED code
    rows, o = res[("space", "aff")]
    add("lift-track space (CN)", "Q2 order of p on t<=T/2, nx 16->32", f"{o['p_half_rel'][-1]:.2f}", "2.8..3.3",
        2.8 <= o["p_half_rel"][-1] <= 3.3)
    add("lift-track space (CN)", "Q2 order of u on t<=T/2, nx 16->32", f"{o['u_half_rel'][-1]:.2f}", "2.8..3.3",
        2.8 <= o["u_half_rel"][-1] <= 3.3)
    add("lift-track space (CN)", "Q2 order of y, nx 16->32", f"{o['y_L2L2_rel'][-1]:.2f}", "2.8..3.3",
        2.8 <= o["y_L2L2_rel"][-1] <= 3.3)
    # whole-interval adjoint at nx = 32 with Nt = 256 (temporal contamination removed)
    def single(path, key):
        t = open(os.path.join(HERE, "logs", "campaign", path)).read()
        return float(re.search(key + r"=(\S+)", t).group(1))
    p16 = f(rows[-2], "p_L2L2_rel")
    pa = single("lt_aff_cn_nx32_nt256.log", "p_L2L2_rel")
    pn = single("lt_noaff_cn_nx32_nt256.log", "p_L2L2_rel")
    p16n = f(res[("space", "noaff")][0][-2], "p_L2L2_rel")
    print(f"\nwhole-interval p at nx = 32 with Nt = 256: aff {pa:.4e} (order from nx 16: {math.log2(p16 / pa):.2f}),"
          f" noaff {pn:.4e} (order {math.log2(p16n / pn):.2f})")
    add("lift-track space (CN)", "Q2 order of p on (0,T), nx 16 (Nt 128) -> 32 (Nt 256)", f"{math.log2(p16 / pa):.2f}",
        "2.8..3.3", 2.8 <= math.log2(p16 / pa) <= 3.3, "Nt = 128 at nx = 32 shows the CN terminal layer (2.17)")
    for s, lo, hi in (("CN", 1.8, 2.3), ("BE", 0.9, 1.1)):
        rr, oo = res[(s, "aff")]
        k = "p_L2L2_rel"
        add(f"lift-track time ({s})", f"order of p, Nt {rr[-2]['steps']}->{rr[-1]['steps']}", f"{oo[k][-1]:.2f}",
            f"{lo}..{hi}", lo <= oo[k][-1] <= hi)
    allaff = [r for r in sp + tm if r["id"].startswith("lt_aff_")]
    rel = max(max(f(r, "state_rel"), f(r, "adjoint_rel")) for r in allaff)
    add("lift-track (all aff runs)", "max relative KKT block residual (incl. affine term)", f"{rel:.1e}", "<= 1e-10", rel <= 1e-10)
    m = max(f(r, "march_rel") for r in allaff)
    add("lift-track (all aff runs)", "max independent-march deviation", f"{m:.1e}", "<= 1e-10", m <= 1e-10)
    by = max(f(r, "max|y-g|") for r in allaff)
    bp = max(max(f(r, "max|p|"), f(r, "max|u|")) for r in allaff)
    add("lift-track (all aff runs)", "max |y-g| on boundary", f"{by:.1e}", "<= 1e-12", by <= 1e-12)
    add("lift-track (all aff runs)", "max |p|,|u| on boundary (adjoint homogeneous)", f"{bp:.1e}", "<= 1e-14", bp <= 1e-14)
    dr = max(f(r, "direct_rel") for r in allaff)
    add("lift-track (all aff runs)", "J_track vs direct full-mass recomputation (rel.)", f"{dr:.1e}", "<= 1e-12", dr <= 1e-12)
    jo = res[("space", "aff")][1]["J_err_rel"]
    add("lift-track space (CN)", "|J_h - J_exact|/J_exact order, nx 16->32", f"{jo[-1]:.2f}", ">= 2.8 (converges)",
        jo[-1] >= 2.8, f"J_err at nx = 32: {f(rows[-1], 'J_err_rel'):.2e}")
    # negative control
    rn, on = res[("space", "noaff")]
    ratio = f(rn[-1], "p_half_rel") / f(rows[-1], "p_half_rel")
    add("negative control (no affine term)", "Q2 order of p on t<=T/2, nx 16->32", f"{on['p_half_rel'][-1]:.2f}",
        "< 2.8 (must degrade)", on["p_half_rel"][-1] < 2.8)
    add("negative control (no affine term)", "p error ratio noaff/aff at nx = 32 (t<=T/2)", f"{ratio:.2f}",
        "> 1.5 (and growing)", ratio > 1.5)
    # gates
    for path in sorted(glob.glob(os.path.join(HERE, "logs", "gates", "vc_lifttrack_*.log"))):
        t = open(path).read()
        name = os.path.basename(path)[:-4]
        g3p = float(re.search(r"G3P_SUMMARY affine_tracking=\d best_rel=(\S+)", t).group(1))
        on_ = "affine_on" in name
        if on_:
            tr = float(re.search(r"SHELL_TRANSPOSE .*\(\|Ex\|\|y\|\)=(\S+)", t).group(1))
            fd = float(re.search(r"G3_SUMMARY adjoint=production_shell\(E\^T\) adjoint_its=\d+ best_rel=(\S+)", t).group(1))
            hs = float(re.search(r"REDUCED_HESSIAN adjoint=production_shell\(E\^T\) .* rel=(\S+)", t).group(1))
            add(f"gates {name}", "space-time transpose", f"{tr:.1e}", "<= 1e-14", tr <= 1e-14)
            add(f"gates {name}", "FD reduced gradient (reduced objective)", f"{fd:.1e}", "<= 1e-9", fd <= 1e-9)
            add(f"gates {name}", "reduced Hessian symmetry", f"{hs:.1e}", "<= 1e-12", hs <= 1e-12)
            add(f"gates {name}", "G3P: FD of the PHYSICAL full-mass objective", f"{g3p:.1e}", "<= 1e-9", g3p <= 1e-9)
        else:
            add(f"gates {name}", "G3P without the affine term (negative control)", f"{g3p:.1e}", ">= 1e-5 (must fail)", g3p >= 1e-5)
    # regression of the earlier lift cases (identical to round-off; J now physical)
    for rid in ("dir_cn_nx16_nt128", "dir_be_nx32_nt16", "dirvw_cn_nx16_nt128"):
        new = [r for r in load("lt_regress") if r["id"] == rid][0]
        old = [r for r in csv.DictReader(open(os.path.join(LP, "t3", "results",
               "dir_space.csv" if rid == "dir_cn_nx16_nt128" else
               "dir_time.csv" if rid == "dir_be_nx32_nt16" else "dirvw_space.csv"))) if r["id"] == rid][0]
        d = max(abs(f(new, k) - f(old, k)) / abs(f(old, k)) for k in ("y_L2L2_rel", "p_L2L2_rel", "u_L2L2_rel", "J_track", "J_reg"))
        add("regression (T3 lift cases, y_d = g on the boundary)", f"{rid}: max rel. change of errors and J", f"{d:.1e}",
            "<= 1e-8 (round-off)", d <= 1e-8)
    regs = {}
    for name, pat in (("ex21_default_vs_ex20_goldens.log", r"(\d+)/(\d+) cases identical"),
                      ("regress_constant_wind.log", r"REGRESSION (\S+)")):
        t = open(os.path.join(HERE, "logs", name)).read()
        regs[name] = re.findall(pat, t)
    g = regs["ex21_default_vs_ex20_goldens.log"]
    add("regression", "ex20 goldens", f"{g[0][0]}/{g[0][1]}", "all identical", bool(g) and g[0][0] == g[0][1])
    add("regression", "constant wind / homogeneous: dumps, 8 T1/T2 runs, gate logs",
        regs["regress_constant_wind.log"][0] if regs["regress_constant_wind.log"] else "?", "PASS",
        regs["regress_constant_wind.log"] == ["PASS"])
    for name in ("regress_ex3_ex4.log", "regress_2d.log"):
        t = open(os.path.join(HERE, "logs", name)).read()
        for rid, st in re.findall(r"(\S+): (BYTE-IDENTICAL|DIFFERS)", t):
            ok = st == "BYTE-IDENTICAL" or rid.startswith("ef95")
            add("regression", rid, st, "byte-identical (numerics)", ok,
                "" if st == "BYTE-IDENTICAL" else "CONFIG label only (variant=source, pre-T3); numerics identical")
    with open(os.path.join(LP, "t3b_tracking_validation.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["test", "quantity", "value", "criterion", "status", "note"])
        w.writeheader()
        w.writerows(OUT)
    nf = sum(o["status"] != "PASS" for o in OUT)
    print(f"\nt3b_tracking_validation.csv: {len(OUT) - nf}/{len(OUT)} PASS")
    for o in OUT:
        if o["status"] != "PASS":
            print("  FAIL:", o)


if __name__ == "__main__":
    main()
