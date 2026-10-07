#!/usr/bin/env python3
"""Tables, observed orders, validation_summary.csv and compact plots for the
Leveque-Pearson validation phase (T0, T1, T2).  Reads results/*.csv written by
run_campaign.py and logs/t0_cn_matrix_check.log, logs/gates/*.log.

    analyze.py            -> prints markdown tables (logs/analyze.log) and writes
                             validation_summary.csv, figures/t2_*.png
"""
import csv
import glob
import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
summary = []


def load(group):
    path = os.path.join(R, group + ".csv")
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []


def f(r, k):
    return float(r[k])


def orders(rows, key, by):
    out = [None]
    for a, b in zip(rows, rows[1:]):
        out.append(math.log(f(a, key) / f(b, key)) / math.log(f(a, by) / f(b, by)))
    return out


def table(title, rows, cols, by, by_label):
    """cols: list of (key, label); orders w.r.t. the column `by` (h or dt)."""
    print(f"\n#### {title}\n")
    head = [by_label] + [c for _, c in cols for c in (c, "ord")]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    ords = {k: orders(rows, k, by) for k, _ in cols}
    for i, r in enumerate(rows):
        cells = [r["_label"]]
        for k, _ in cols:
            o = ords[k][i]
            cells += [f"{f(r, k):.3e}", "–" if o is None else f"{o:.2f}"]
        print("| " + " | ".join(cells) + " |")
    return ords


def add(test, quantity, value, criterion, ok, note=""):
    summary.append({"test": test, "quantity": quantity, "value": value,
                    "criterion": criterion, "status": "PASS" if ok else "FAIL", "note": note})


def prep(rows, label_fn, by_fn):
    for r in rows:
        r["_label"] = label_fn(r)
        r["_by"] = by_fn(r)
    return rows


def solver_rows(test, rows):
    # true_residual is the ABSOLUTE ||K x - b||_2 (Solver.cpp); the acceptance
    # quantity is the relative state/adjoint block residual (ComputeKktResiduals)
    worst = max(float(r["true_residual"]) for r in rows)
    relres = max(max(float(r["state_rel"]), float(r["adjoint_rel"])) for r in rows)
    march = max(float(r["march_rel"]) for r in rows)
    stat = max(float(r["alpha_u_u_plus_eta_p_rel"]) for r in rows)
    reasons = all(int(r["reason"]) > 0 for r in rows)
    its = sorted({int(r["outer_its"]) for r in rows})
    add(test, "max relative KKT block residual (state, adjoint)", f"{relres:.2e}", "<= 1e-10", relres <= 1e-10,
        f"absolute ||Kx-b|| max {worst:.1e}")
    add(test, "max ||y_prod - y_march|| / ||y|| (independent LU march)", f"{march:.2e}", "<= 1e-10", march <= 1e-10)
    add(test, "max ||alpha_u u + eta p|| / ||alpha_u u||", f"{stat:.2e}", "<= 1e-12", stat <= 1e-12)
    add(test, "all solves converged; outer FGMRES its", f"{its}", "reason > 0", reasons)
    return worst, march, its


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)

    # ------------------------------------------------------------------ T0
    t0 = open(os.path.join(HERE, "logs", "t0_cn_matrix_check.log")).read()
    m = re.search(r"T0 SUMMARY: (\d+)/(\d+) PASS", t0)
    add("T0", "algebraic LP-vs-ex21 CN checks", f"{m.group(1)}/{m.group(2)}", "all PASS", m.group(1) == m.group(2))

    # ------------------------------------------------------------------ gates
    for path in sorted(glob.glob(os.path.join(HERE, "logs", "gates", "verify_convection_t2_*.log"))):
        txt = open(path).read()
        sch = "CN" if "_cn.log" in path else "BE"
        tr = float(re.search(r"G2_SHELL_TRANSPOSE .*\|diff\|/\(\|Ex\|\|y\|\)=(\S+)", txt).group(1))
        fd = float(re.search(r"G3_SUMMARY adjoint=production_shell\(E\^T\) adjoint_its=\d+ best_rel=(\S+)", txt).group(1))
        hs = float(re.search(r"REDUCED_HESSIAN adjoint=production_shell\(E\^T\) .* rel=(\S+)", txt).group(1))
        wrong = float(re.search(r"G3_SUMMARY adjoint=ex20_style_wrong adjoint_its=\d+ best_rel=(\S+)", txt).group(1))
        add(f"T2 gates {sch}", "space-time transpose <Ex,y> - <x,E^T y>", f"{tr:.1e}", "<= 1e-14", tr <= 1e-14)
        add(f"T2 gates {sch}", "FD reduced gradient (production adjoint)", f"{fd:.1e}", "<= 1e-9", fd <= 1e-9)
        add(f"T2 gates {sch}", "reduced Hessian symmetry", f"{hs:.1e}", "<= 1e-12", hs <= 1e-12)
        add(f"T2 gates {sch}", "negative control: FD gradient with untransposed L", f"{wrong:.1e}", ">= 1e-2 (must fail)", wrong >= 1e-2)

    # ------------------------------------------------------------------ T2 space
    print("### T2 spatial convergence (CN, Nt = 256, dt = 1/256)")
    figs = {}
    for shape in ("sin3", "sin"):
        rows = sorted([r for r in load("t2_space") if r["case"] == f"conv-{shape}"], key=lambda r: int(r["nx"]))
        if not rows:
            continue
        prep(rows, lambda r: r["nx"], lambda r: 1.0 / int(r["nx"]))
        for r in rows:
            r["h"] = 1.0 / int(r["nx"])
        o = table(f"conv-{shape}", rows, [("y_L2L2_rel", "y"), ("p_L2L2_rel", "p"), ("p_int_rel", "p int"),
                                          ("p_half_rel", "p t<=T/2"), ("u_L2L2_rel", "u"), ("u_half_rel", "u t<=T/2"),
                                          ("y_T_rel", "y(T)"), ("interp_y_T", "|I_h y(T) - y(T)|")],
                  "h", "nx")
        figs[shape] = rows
        last = {k: o[k][-1] for k in o}
        if shape == "sin3":
            for k, lab in (("y_L2L2_rel", "y on (0,T)"), ("p_half_rel", "p on (0,T/2)"), ("u_half_rel", "u on (0,T/2)")):
                add("T2 space (sin3, compatible)", f"Q2 L2(L2) order of {lab}, nx {rows[-2]['nx']}->{rows[-1]['nx']}",
                    f"{last[k]:.2f}", "2.8 .. 3.3", 2.8 <= last[k] <= 3.3)
            for k, lab in (("p_L2L2_rel", "p on (0,T)"), ("u_L2L2_rel", "u on (0,T)")):
                add("T2 space (sin3, compatible)", f"Q2 L2(L2) order of {lab}, nx {rows[-3]['nx']}->{rows[-2]['nx']}",
                    f"{o[k][-2]:.2f}", "2.8 .. 3.3", 2.8 <= o[k][-2] <= 3.3)
                add("T2 space (sin3, compatible)", f"Q2 L2(L2) order of {lab}, nx {rows[-2]['nx']}->{rows[-1]['nx']}",
                    f"{last[k]:.2f}", "reported", True,
                    "temporal contamination: CN terminal adjoint layer at Nt = 256 (window t<=T/2 gives 3.0)")
            solver_rows("T2 space", rows)
        else:
            add("T2 space (sin, f/y_d != 0 on boundary)", f"Q2 L2(L2) order of y, nx {rows[-2]['nx']}->{rows[-1]['nx']}",
                f"{last['y_L2L2_rel']:.2f}", "~2.5 expected (inherited M_elim I_h f load)", 2.3 <= last["y_L2L2_rel"] <= 2.7)
    chk = {r["case"]: r for r in load("t2_space_chk")}
    sp = {(r["case"], r["nx"]): r for r in load("t2_space")}
    if chk:
        print("\n#### temporal contamination check (nx = 32): Nt = 128 vs Nt = 256\n")
        print("| case | quantity | Nt = 128 | Nt = 256 | rel. change |")
        print("|---|---|---|---|---|")
        for case, r in chk.items():
            s = sp.get((case, "32"))
            if not s:
                continue
            for k in ("y_L2L2_rel", "p_int_rel", "p_L2L2_rel"):
                ch = abs(f(r, k) - f(s, k)) / f(s, k)
                print(f"| {case} | {k} | {f(r, k):.4e} | {f(s, k):.4e} | {ch:.1e} |")
                if case == "conv-sin3" and k != "p_L2L2_rel":
                    add("T2 space", f"temporal contamination of {k} at nx=32 (Nt 128 vs 256)", f"{ch:.1e}", "<= 2e-2", ch <= 2e-2)

    # ------------------------------------------------------------------ T2 CN time
    rows = sorted(load("t2_cn_time"), key=lambda r: int(r["steps"]))
    if rows:
        prep(rows, lambda r: r["steps"], lambda r: 1.0 / int(r["steps"]))
        for r in rows:
            r["dt"] = r["dt"]
        print("\n### T2 CN temporal convergence (conv-sin3, nx = 64)")
        o = table("CN, nx = 64", rows, [("y_L2L2_rel", "y"), ("y_int_rel", "y int"), ("y_T_rel", "y(T)"),
                                        ("p_L2L2_rel", "p"), ("p_int_rel", "p int"), ("p_half_rel", "p t<=T/2"),
                                        ("u_L2L2_rel", "u")], "dt", "Nt")
        table("CN, nx = 64: diagnostics (adjoint sampled at the node t_n; wrong sign)", rows,
              [("p_nodetime_L2L2_rel", "p vs p(t_n)"), ("p_wrongsign_L2L2_rel", "p vs -p_exact")], "dt", "Nt")
        for k, lab in (("y_L2L2_rel", "y"), ("p_L2L2_rel", "p (all slots)"), ("p_int_rel", "p (interior)"),
                       ("u_L2L2_rel", "u"), ("y_T_rel", "y(T)")):
            add("T2 CN time", f"order of {lab}, Nt {rows[-2]['steps']}->{rows[-1]['steps']}", f"{o[k][-1]:.2f}",
                "1.8 .. 2.4", 1.8 <= o[k][-1] <= 2.4)
        nodet = orders(rows, "p_nodetime_L2L2_rel", "dt")[-1]
        add("T2 CN time", "order of p sampled at t_n instead of t_{n-1/2}", f"{nodet:.2f}", "~1 (adjoint is an interval quantity)", 0.8 <= nodet <= 1.2)
        ws = f(rows[-1], "p_wrongsign_L2L2_rel")
        add("T2 CN time", "relative error of p against -p_exact (wrong sign)", f"{ws:.3f}", "~2 (sign confirmed)", ws > 1.5)
        solver_rows("T2 CN time", rows)
        figs["cn"] = rows
    term = sorted(load("t2_cn_term"), key=lambda r: int(r["steps"]))
    if term:
        prep(term, lambda r: r["steps"], lambda r: 1.0 / int(r["steps"]))
        print("\n### T2 CN terminal adjoint slot (conv-sin3, nx = 16; the terminal slot is temporal-error dominated)")
        o = table("CN terminal slot", term, [("p_last", "abs. err. p_N"), ("p_last_rel", "rel. err. p_N"),
                                             ("p_int_rel", "p interior (spatial floor)")], "dt", "Nt")
        add("T2 CN terminal slot", f"order of |p_N - p(t_(N-1/2))| (abs), Nt {term[-2]['steps']}->{term[-1]['steps']}",
            f"{o['p_last'][-1]:.2f}", "reported (no first-order stagnation: > 1.2)", o["p_last"][-1] > 1.2)
        add("T2 CN terminal slot", "relative terminal error at Nt = 512", f"{f(term[-1], 'p_last_rel'):.3f}", "reported", True,
            "|p(t_(N-1/2))| = O(dt): relative error O(dt)")
        solver_rows("T2 CN terminal", term)
        figs["term"] = term

    # ------------------------------------------------------------------ T2 BE time
    rows = sorted(load("t2_be_time"), key=lambda r: int(r["steps"]))
    if rows:
        prep(rows, lambda r: r["steps"], lambda r: 1.0 / int(r["steps"]))
        print("\n### T2 BE temporal convergence (conv-sin3, nx = 32)")
        o = table("BE, nx = 32", rows, [("y_L2L2_rel", "y"), ("y_T_rel", "y(T)"), ("p_L2L2_rel", "p"),
                                        ("p_int_rel", "p int"), ("u_L2L2_rel", "u"), ("p_last", "abs. err. p_N")], "dt", "Nt")
        for k, lab in (("y_L2L2_rel", "y"), ("p_L2L2_rel", "p"), ("u_L2L2_rel", "u")):
            add("T2 BE time", f"order of {lab}, Nt {rows[-2]['steps']}->{rows[-1]['steps']}", f"{o[k][-1]:.2f}",
                "0.9 .. 1.1", 0.9 <= o[k][-1] <= 1.1)
        solver_rows("T2 BE time", rows)
        figs["be"] = rows

    # ------------------------------------------------------------------ T1
    ex20_rule = {("cn", 3): (1.329e-3, 1.082e-2), ("cn", 4): (4.049e-4, 2.857e-3), ("cn", 5): (1.088e-4, 7.915e-4),
                 ("cn", 6): (2.522e-5, 2.167e-4), ("be", 3): (1.084e-3, 1.732e-2), ("be", 4): (2.881e-4, 4.571e-3),
                 ("be", 5): (7.171e-5, 1.175e-3), ("be", 6): (1.787e-5, 2.962e-4)}
    ex20_time = {8: (1.132e-3, 6.462e-3), 16: (3.650e-4, 1.868e-3), 32: (9.992e-5, 5.667e-4), 64: (2.324e-5, 1.599e-4)}
    rule = load("t1_rule")
    if rule:
        print("\n### T1 Leveque-Pearson 5.1 through the ex21 production path (Q2), paper rules, beta = 1e-3")
        for sch in ("cn", "be"):
            rows = sorted([r for r in rule if r["scheme"] == sch.upper()], key=lambda r: int(r["nx"]))
            if not rows:
                continue
            prep(rows, lambda r: str(int(round(math.log2(int(r["nx"]))))), lambda r: 1.0 / int(r["nx"]))
            for r in rows:
                r["h"] = 2.0 / int(r["nx"])
            o = table(f"{sch.upper()} ({'tau = h' if sch == 'cn' else 'tau = h^2'}), quadrature L2(L2) relative", rows,
                      [("y_L2L2_rel_unshifted", "y"), ("y_half_rel", "w t<=T/2"), ("p_L2L2_rel", "p"), ("p_int_rel", "p int"),
                       ("p_half_rel", "p t<=T/2"), ("u_L2L2_rel", "u"), ("p_last", "abs. err. p_N")], "h", "l")
            print(f"\n{sch.upper()}: ex20-comparable nodal metric (relative l2 over interior space-time nodes) vs ex20 (Q1) §15-17\n")
            print("| l | ex21 Q2 y_l2rel | ex20 Q1 y_l2rel | ratio | ex21 Q2 p_l2rel | ex20 Q1 p_l2rel | ratio |")
            print("|---|---|---|---|---|---|---|")
            for r in rows:
                l = int(r["_label"])
                e = ex20_rule.get((sch, l))
                if e:
                    print(f"| {l} | {f(r, 'ex20_y_l2rel'):.3e} | {e[0]:.3e} | {f(r, 'ex20_y_l2rel') / e[0]:.2f} |"
                          f" {f(r, 'ex20_p_l2rel'):.3e} | {e[1]:.3e} | {f(r, 'ex20_p_l2rel') / e[1]:.2f} |")
            exp = 2.0
            for k, lab in (("y_L2L2_rel_unshifted", "y on (0,T)"), ("p_half_rel", "p on (0,T/2)")):
                add(f"T1 {sch.upper()} paper rule", f"h-order of {lab}, l {rows[-2]['_label']}->{rows[-1]['_label']}",
                    f"{o[k][-1]:.2f}", "1.8 .. 2.4 (O(h^2) via the tau rule)", 1.8 <= o[k][-1] <= 2.4)
            for k, lab in (("p_L2L2_rel", "p on (0,T)"), ("u_L2L2_rel", "u on (0,T)")):
                add(f"T1 {sch.upper()} paper rule", f"h-order of {lab}, l {rows[-2]['_label']}->{rows[-1]['_label']}",
                    f"{o[k][-1]:.2f}", "reported", True, "includes the DTO terminal adjoint layer (ex20 section 27)")
            solver_rows(f"T1 {sch.upper()} paper rule", rows)
    rows = sorted(load("t1_time"), key=lambda r: int(r["steps"]))
    if rows:
        prep(rows, lambda r: r["steps"], lambda r: 1.0 / int(r["steps"]))
        print("\n### T1 pure temporal (CN, l = 5 / nx = 32 Q2, beta = 1e-3) vs ex20 §22 (CN, l = 7 Q1)\n")
        print("| Nt | ex21 y_l2rel | ex20 y_l2rel | ratio | ex21 p_l2rel | ex20 p_l2rel | ratio |")
        print("|---|---|---|---|---|---|---|")
        worst = 0.0
        for r in rows:
            e = ex20_time[int(r["steps"])]
            ry, rp = f(r, "ex20_y_l2rel") / e[0], f(r, "ex20_p_l2rel") / e[1]
            print(f"| {r['steps']} | {f(r, 'ex20_y_l2rel'):.4e} | {e[0]:.4e} | {ry:.3f} | {f(r, 'ex20_p_l2rel'):.4e} | {e[1]:.4e} | {rp:.3f} |")
            if int(r["steps"]) <= 32:
                worst = max(worst, abs(ry - 1), abs(rp - 1))
        add("T1 temporal vs ex20", "max |ex21/ex20 - 1| of y_l2rel, p_l2rel at Nt = 8..32", f"{worst:.3f}",
            "<= 0.1 (both temporal-dominated: identical time discretisation)", worst <= 0.1)
        o = table("CN l = 5 temporal, quadrature L2(L2)", rows, [("y_L2L2_rel_unshifted", "y"), ("p_L2L2_rel", "p"),
                                                               ("p_int_rel", "p int"), ("p_half_rel", "p t<=T/2"),
                                                               ("u_L2L2_rel", "u"), ("p_last", "abs. err. p_N")], "dt", "Nt")
        add("T1 temporal", f"dt-order of p on (0,T/2), Nt {rows[-2]['steps']}->{rows[-1]['steps']}", f"{o['p_half_rel'][-1]:.2f}",
            "1.8 .. 2.3", 1.8 <= o["p_half_rel"][-1] <= 2.3)
        add("T1 temporal", f"dt-order of p on (0,T), Nt {rows[-2]['steps']}->{rows[-1]['steps']}", f"{o['p_L2L2_rel'][-1]:.2f}",
            "reported", True, "terminal adjoint layer; ex20 section 22 measured the same (1.72-1.82)")
        solver_rows("T1 temporal", rows)
    rows = load("t1_beta") + load("t1_conv_on")
    if rows:
        print("\n### T1 beta range and transposed path (CN, l = 4, Nt = 16)\n")
        print("| run | alpha_u = beta | convection flag | y | p | u | outer | true residual |")
        print("|---|---|---|---|---|---|---|---|")
        for r in rows:
            print(f"| {r['id']} | {r['alpha_u']} | {r['convection']} | {f(r, 'y_L2L2_rel_unshifted'):.4e} |"
                  f" {f(r, 'p_L2L2_rel'):.4e} | {f(r, 'u_L2L2_rel'):.4e} | {r['outer_its']} | {r['true_residual']} |")
        on = [r for r in rows if r["id"] == "t1_convon_cn_l4"]
        off = [r for r in rows if r["id"] == "t1_convoff_cn_l4"]
        if on and off:
            d = max(abs(f(on[0], k) - f(off[0], k)) / f(off[0], k) for k in ("y_L2L2_rel", "p_L2L2_rel", "u_L2L2_rel"))
            add("T1 v = 0", "convection on (v = 0, transposed adjoint path) vs off: max rel. change of errors", f"{d:.1e}",
                "<= 1e-6", d <= 1e-6)
        solver_rows("T1 beta / v=0", rows)

    # ------------------------------------------------------------------ plots
    def loglog(ax, xs, ys, label, marker):
        ax.loglog(xs, ys, marker=marker, label=label, lw=1.5, ms=5)
    if "sin3" in figs:
        fig, ax = plt.subplots(figsize=(5, 4))
        for shape, mk in (("sin3", "o"), ("sin", "s")):
            if shape not in figs:
                continue
            rows = figs[shape]
            h = [r["h"] for r in rows]
            loglog(ax, h, [f(r, "y_L2L2_rel") for r in rows], f"y ({shape})", mk)
            loglog(ax, h, [f(r, "p_half_rel") for r in rows], f"p, t<=T/2 ({shape})", mk)
        h = [r["h"] for r in figs["sin3"]]
        ax.loglog(h, [f(figs["sin3"][-1], "y_L2L2_rel") * (x / h[-1]) ** 3 for x in h], "k--", lw=0.8, label="h^3")
        if "sin" in figs:
            hs = [r["h"] for r in figs["sin"]]
            ax.loglog(hs, [f(figs["sin"][-1], "y_L2L2_rel") * (x / hs[-1]) ** 2.5 for x in hs],
                      "k:", lw=0.8, label="h^2.5")
        ax.set_xlabel("h = 1/nx"); ax.set_ylabel("relative L2(0,T;L2) error")
        ax.set_title("T2 spatial (Q2, CN, Nt = 256)"); ax.legend(fontsize=7); ax.grid(True, which="both", lw=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(HERE, "figures", "t2_space.png"), dpi=130)
    for key, title, ref, fname in (("cn", "T2 CN temporal (nx = 64)", 2, "t2_cn_time.png"),
                                   ("be", "T2 BE temporal (nx = 32)", 1, "t2_be_time.png")):
        if key not in figs:
            continue
        rows = figs[key]
        dt = [1.0 / int(r["steps"]) for r in rows]
        fig, ax = plt.subplots(figsize=(5, 4))
        loglog(ax, dt, [f(r, "y_L2L2_rel") for r in rows], "y", "o")
        loglog(ax, dt, [f(r, "p_L2L2_rel") for r in rows], "p (all slots)", "s")
        loglog(ax, dt, [f(r, "p_int_rel") for r in rows], "p (interior)", "^")
        loglog(ax, dt, [f(r, "u_L2L2_rel") for r in rows], "u", "x")
        ax.loglog(dt, [f(rows[-1], "p_L2L2_rel") * (x / dt[-1]) ** ref for x in dt], "k--", lw=0.8, label=f"dt^{ref}")
        ax.set_xlabel("dt"); ax.set_ylabel("relative L2(0,T;L2) error"); ax.set_title(title)
        ax.legend(fontsize=7); ax.grid(True, which="both", lw=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(HERE, "figures", fname), dpi=130)

    with open(os.path.join(HERE, "validation_summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["test", "quantity", "value", "criterion", "status", "note"])
        w.writeheader()
        w.writerows(summary)
    nf = sum(1 for s in summary if s["status"] != "PASS")
    print(f"\nvalidation_summary.csv: {len(summary) - nf}/{len(summary)} PASS")
    for s in summary:
        if s["status"] != "PASS":
            print("  FAIL:", s)


if __name__ == "__main__":
    main()
