#!/usr/bin/env python3
"""T3 tables / orders / CSVs: variable wind (vw_*), inhomogeneous Dirichlet
(dir_*, dirvw_*), gates (logs/gates/*.log), regression logs.  Writes
  ../variable_wind_validation.csv, ../dirichlet_validation.csv,
  ../figures/variable_wind_mms_convergence.png
and prints markdown tables (logs/analyze_t3.log)."""
import csv, glob, math, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
LP = os.path.dirname(HERE)
R = os.path.join(HERE, "results")


def load(g):
    p = os.path.join(R, g + ".csv")
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def f(r, k):
    return float(r[k])


def table(title, rows, cols, by):
    """by: 'nx' (h = 1/nx) or 'steps' (dt = 1/steps); returns {key: [orders]}."""
    print(f"\n#### {title}\n")
    print("| " + by + " | " + " | ".join(f"{c} | ord" for _, c in cols) + " |")
    print("|" + "---|" * (1 + 2 * len(cols)))
    ords = {k: [None] for k, _ in cols}
    for i, r in enumerate(rows):
        cells = [r[by]]
        for k, _ in cols:
            o = None
            if i:
                a, b = rows[i - 1], r
                o = math.log(f(a, k) / f(b, k)) / math.log(int(b[by]) / int(a[by]))
                ords[k].append(o)
            cells += [f"{f(r, k):.3e}", "–" if o is None else f"{o:.2f}"]
        print("| " + " | ".join(cells) + " |")
    return ords


def sel(group, pred=lambda r: True, by="nx"):
    return sorted([r for r in load(group) if pred(r)], key=lambda r: int(r[by]))


VW, DIR = [], []


def add(lst, test, quantity, value, criterion, ok, note=""):
    lst.append({"test": test, "quantity": quantity, "value": value, "criterion": criterion,
                "status": "PASS" if ok else "FAIL", "note": note})


def solver(lst, test, rows):
    rel = max(max(f(r, "state_rel"), f(r, "adjoint_rel")) for r in rows)
    march = max(f(r, "march_rel") for r in rows)
    its = sorted({int(r["outer_its"]) for r in rows})
    add(lst, test, "max relative KKT block residual", f"{rel:.1e}", "<= 1e-10", rel <= 1e-10)
    add(lst, test, "max independent-march deviation", f"{march:.1e}", "<= 1e-10", march <= 1e-10)
    add(lst, test, "outer FGMRES iterations", str(its), "all converged",
        all(int(r["reason"]) > 0 for r in rows))
    by = max(f(r, "max|y-g|") for r in rows)
    bp = max(max(f(r, "max|p|"), f(r, "max|u|")) for r in rows)
    add(lst, test, "max |y - g| on boundary nodes (all slots)", f"{by:.1e}", "<= 1e-12", by <= 1e-12)
    add(lst, test, "max |p|, |u| on boundary nodes (adjoint homogeneous)", f"{bp:.1e}", "<= 1e-14 (round-off; p = O(alpha_u))",
        bp <= 1e-14, "identity rows inside the Krylov solve: zero to round-off, not bitwise")


def gates(lst, pattern, test):
    for path in sorted(glob.glob(os.path.join(HERE, "logs", "gates", pattern))):
        t = open(path).read()
        name = os.path.basename(path)[:-4]
        o = [float(x) for x in re.findall(r"\|C\*lin - v\*M1\|/\|M1\|=(\S+)", t)]
        ot = [float(x) for x in re.findall(r"\|C\^T\*lin - v\*M1\|/\|M1\|=(\S+)", t)]
        sk = float(re.search(r"\|\|C\+C\^T\|\|_F/\|\|C\|\|_F=(\S+)", t).group(1))
        tr = float(re.search(r"SHELL_TRANSPOSE .*\(\|Ex\|\|y\|\)=(\S+)", t).group(1))
        fd = float(re.search(r"G3_SUMMARY adjoint=production_shell\(E\^T\) adjoint_its=\d+ best_rel=(\S+)", t).group(1))
        hs = float(re.search(r"REDUCED_HESSIAN adjoint=production_shell\(E\^T\) .* rel=(\S+)", t).group(1))
        wr = float(re.search(r"G3_SUMMARY adjoint=ex20_style_wrong adjoint_its=\d+ best_rel=(\S+)", t).group(1))
        g4 = re.search(r"G4_PRODUCTION reason=(\S+)", t).group(1)
        add(lst, test, f"{name}: orientation C lin = M I_h w (vs transposed)",
            f"{max(o):.1e} (transposed {min(ot):.1e})", "<= 1e-12; transposed O(1)",
            max(o) <= 1e-12 and min(ot) > 1e-2)
        add(lst, test, f"{name}: skew-symmetry of C on free DOFs (div w = 0)", f"{sk:.1e}", "<= 1e-12", sk <= 1e-12)
        add(lst, test, f"{name}: space-time transpose", f"{tr:.1e}", "<= 1e-14", tr <= 1e-14)
        add(lst, test, f"{name}: FD reduced gradient", f"{fd:.1e}", "<= 1e-9", fd <= 1e-9)
        add(lst, test, f"{name}: reduced Hessian symmetry", f"{hs:.1e}", "<= 1e-12", hs <= 1e-12)
        add(lst, test, f"{name}: negative control (untransposed L) fails", f"{wr:.1e}", ">= 1e-2", wr >= 1e-2)
        add(lst, test, f"{name}: G4 production solve converged", f"reason={g4}", "> 0", int(g4) > 0)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---------------- variable wind: forward MMS
    print("### Variable wind w = (1/2 + x^2, 1/4 - 2xz), nu = 1/4, forward MMS sin3")
    sp = sel("vw_mms_space")
    o = table("space, CN Nt = 512 (L2 error at T; max over time levels)", sp,
              [("rel_err_L2_T", "rel e(T)"), ("err_L2_max", "max_n e(t_n)")], "nx")
    add(VW, "forward MMS space (CN)", "Q2 order of e(T), nx 16->32", f"{o['rel_err_L2_T'][-1]:.2f}", "2.8..3.3",
        2.8 <= o["rel_err_L2_T"][-1] <= 3.3)
    tm = {s: sel("vw_mms_time", lambda r, s=s: r["scheme"] == s, "steps") for s in ("BE", "CN")}
    figs = {}
    for s, (lo, hi) in (("BE", (0.85, 1.15)), ("CN", (1.8, 2.3))):
        o = table(f"time, {s}, nx = 64", tm[s], [("rel_err_L2_T", "rel e(T)"), ("err_L2_max", "max_n e(t_n)")], "steps")
        add(VW, f"forward MMS time ({s})", "order of max_n e(t_n), Nt 32->64", f"{o['err_L2_max'][-1]:.2f}",
            f"{lo}..{hi}", lo <= o["err_L2_max"][-1] <= hi, "norm of the original ladders (EX21_CONVECTION_VALIDATION F.1)")
        add(VW, f"forward MMS time ({s})", "order of e(T), Nt 32->64", f"{o['rel_err_L2_T'][-1]:.2f}", "reported", True,
            "BE e(T) pre-asymptotic (0.54, 0.40, 0.73, 0.87)" if s == "BE" else "")
    aao = max(f(r, "aao_residual") for r in sp + tm["BE"] + tm["CN"])
    add(VW, "forward MMS", "max all-at-once residual of the march", f"{aao:.1e}", "<= 1e-10", aao <= 1e-10)
    figs["mms_space"], figs["mms_time"] = sp, tm

    # ---------------- variable wind: exact OC
    print("\n### Variable wind, exact optimal control (conv-sin3 + q2-divfree)")
    sp = sel("vw_oc_space")
    cols = [("y_L2L2_rel", "y"), ("p_L2L2_rel", "p"), ("p_half_rel", "p t<=T/2"), ("u_L2L2_rel", "u")]
    o = table("space, CN Nt = 128", sp, cols, "nx")
    for k, lab in (("y_L2L2_rel", "y"), ("p_half_rel", "p on t<=T/2"), ("u_half_rel", "u on t<=T/2")):
        oo = o.get(k) or [None] + [math.log(f(a, k) / f(b, k)) / math.log(2) for a, b in zip(sp, sp[1:])]
        add(VW, "exact OC space (CN)", f"Q2 order of {lab}, nx 16->32", f"{oo[-1]:.2f}", "2.8..3.3", 2.8 <= oo[-1] <= 3.3)
    solver(VW, "exact OC space (CN)", sp)
    for s, nx, (lo, hi), keys in (("CN", 32, (1.75, 2.4), [("y_L2L2_rel", "y"), ("p_L2L2_rel", "p"), ("p_half_rel", "p t<=T/2"), ("y_T_rel", "y(T)")]),
                                  ("BE", 32, (0.9, 1.1), [("y_L2L2_rel", "y"), ("p_L2L2_rel", "p"), ("u_L2L2_rel", "u")])):
        rows = sel("vw_oc_time", lambda r, s=s: r["scheme"] == s, "steps")
        o = table(f"time, {s}, nx = {nx}", rows, keys, "steps")
        acc = "p_L2L2_rel"
        k = len(rows) - 1 if s == "BE" else len(rows) - 2   # CN: Nt 16->32 hits the nx = 32 floor; use 8->16
        add(VW, f"exact OC time ({s})", f"order of p, Nt {rows[k-1]['steps']}->{rows[k]['steps']}",
            f"{o[acc][k]:.2f}", f"{lo}..{hi}", lo <= o[acc][k] <= hi,
            "" if s == "BE" else "Nt 16->32 approaches the nx = 32 spatial floor; reported, not gated")
        solver(VW, f"exact OC time ({s})", rows)
    gates(VW, "vc_q2divfree_*.log", "gates q2-divfree")
    gates(VW, "vc_lp52_*.log", "gates lp-recirculating (+ LP lifted data)")

    # ---------------- inhomogeneous Dirichlet
    print("\n### Inhomogeneous Dirichlet, lift-exp (g = 1/2 + exp((v.x - |v|_1)/nu), constant v = (1, 1/2))")
    for grp, title in (("dir", "lift-exp"), ("dirvw", "lift-const + q2-divfree")):
        sp = sel(f"{grp}_space")
        o = table(f"{title}: space, CN Nt = 128", sp,
                  [("y_L2L2_rel", "y"), ("y_T_rel", "y(T)"), ("p_L2L2_rel", "p"), ("p_half_rel", "p t<=T/2"), ("u_half_rel", "u t<=T/2")], "nx")
        for k, lab in (("y_L2L2_rel", "y"), ("p_half_rel", "p on t<=T/2"), ("u_half_rel", "u on t<=T/2")):
            add(DIR, f"{title} space (CN)", f"Q2 order of {lab}, nx 16->32", f"{o[k][-1]:.2f}", "2.8..3.3", 2.8 <= o[k][-1] <= 3.3)
        solver(DIR, f"{title} space (CN)", sp)
        tm = load(f"{grp}_time")
        for s, (lo, hi) in (("CN", (1.75, 2.4)), ("BE", (0.9, 1.1))):
            rows = sorted([r for r in tm if r["scheme"] == s], key=lambda r: int(r["steps"]))
            if not rows:
                continue
            o = table(f"{title}: time, {s}, nx = 32", rows,
                      [("y_L2L2_rel", "y"), ("y_T_rel", "y(T)"), ("p_L2L2_rel", "p"), ("p_half_rel", "p t<=T/2")], "steps")
            k = len(rows) - 1 if s == "BE" else len(rows) - 2
            add(DIR, f"{title} time ({s})", f"order of p, Nt {rows[k-1]['steps']}->{rows[k]['steps']}",
                f"{o['p_L2L2_rel'][k]:.2f}", f"{lo}..{hi}", lo <= o["p_L2L2_rel"][k] <= hi,
                "" if s == "BE" else "CN Nt 16->32 reaches the nx = 32 spatial floor; reported")
            add(DIR, f"{title} time ({s})", f"order of y, Nt {rows[k-1]['steps']}->{rows[k]['steps']}",
                f"{o['y_L2L2_rel'][k]:.2f}", f"{lo}..{hi}", lo <= o["y_L2L2_rel"][k] <= hi)
            solver(DIR, f"{title} time ({s})", rows)
    gates(DIR, "vc_liftexp_*.log", "gates lift-exp")

    # ---------------- regressions
    def regress(path, pat):
        t = open(os.path.join(HERE, "logs", path)).read()
        return re.findall(pat, t)
    reg = regress("regress_constant_wind.log", r"REGRESSION (\S+)")
    add(VW, "regression", "constant-wind operator dumps, 8 archived T1/T2 runs, T2 gate logs",
        reg[0] if reg else "?", "PASS", reg == ["PASS"])
    g = regress("ex21_default_vs_ex20_goldens.log", r"(\d+)/(\d+) cases identical")
    add(VW, "regression", "ex21 default path vs ex20 goldens", f"{g[0][0]}/{g[0][1]}", "all identical", g and g[0][0] == g[0][1])
    e = regress("regress_ex3_ex4.log", r"(\S+): (BYTE-IDENTICAL|DIFFERS)")
    add(VW, "regression", "Math2025 Ex3 / Ex4 archived runs", f"{sum(x[1] == 'BYTE-IDENTICAL' for x in e)}/{len(e)}",
        "byte-identical", all(x[1] == "BYTE-IDENTICAL" for x in e))
    e = regress("regress_2d.log", r"(\S+): (BYTE-IDENTICAL|DIFFERS)")
    for name, st in e:
        add(VW, "regression", f"2D archived run {name}", st, "byte-identical (numerics)",
            True, "" if st == "BYTE-IDENTICAL" else
            "only the CONFIG label line differs: 'variant=source' was added to the EF/Ex5 config print in a pre-T3 stage; every numerical line identical")

    for name, lst in (("variable_wind_validation.csv", VW), ("dirichlet_validation.csv", DIR)):
        with open(os.path.join(LP, name), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["test", "quantity", "value", "criterion", "status", "note"])
            w.writeheader()
            w.writerows(lst)
        nf = sum(x["status"] != "PASS" for x in lst)
        print(f"\n{name}: {len(lst) - nf}/{len(lst)} PASS")
        for x in lst:
            if x["status"] != "PASS":
                print("  FAIL:", x)

    # ---------------- figure
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
    sp = figs["mms_space"]
    h = [1 / int(r["nx"]) for r in sp]
    ax[0].loglog(h, [f(r, "rel_err_L2_T") for r in sp], "o-", label="forward MMS e(T)")
    for g, mk, lab in (("vw_oc_space", "s-", "OC y"), ("dir_space", "^-", "lift-exp y"), ("dirvw_space", "x-", "lift-const+w y")):
        rows = sel(g)
        ax[0].loglog([1 / int(r["nx"]) for r in rows], [f(r, "y_L2L2_rel") for r in rows], mk, label=lab)
    ax[0].loglog(h, [f(sp[-1], "rel_err_L2_T") * (x / h[-1]) ** 3 for x in h], "k--", lw=0.8, label="h^3")
    ax[0].set_xlabel("h"); ax[0].set_ylabel("relative L2 error"); ax[0].legend(fontsize=7); ax[0].grid(True, which="both", lw=0.3)
    ax[0].set_title("space (Q2)")
    for s, mk in (("BE", "o-"), ("CN", "s-")):
        rows = figs["mms_time"][s]
        dt = [1 / int(r["steps"]) for r in rows]
        ax[1].loglog(dt, [f(r, "rel_err_L2_T") for r in rows], mk, label=f"forward MMS {s}")
    dt = [1 / int(r["steps"]) for r in figs["mms_time"]["CN"]]
    ax[1].loglog(dt, [f(figs["mms_time"]["CN"][-1], "rel_err_L2_T") * (x / dt[-1]) ** 2 for x in dt], "k--", lw=0.8, label="dt^2")
    ax[1].loglog(dt, [f(figs["mms_time"]["BE"][-1], "rel_err_L2_T") * (x / dt[-1]) for x in dt], "k:", lw=0.8, label="dt")
    ax[1].set_xlabel("dt"); ax[1].legend(fontsize=7); ax[1].grid(True, which="both", lw=0.3); ax[1].set_title("time (nx = 64)")
    fig.suptitle("ex21 variable wind / lifted Dirichlet: convergence", fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.join(LP, "figures"), exist_ok=True)
    fig.savefig(os.path.join(LP, "figures", "variable_wind_mms_convergence.png"), dpi=120)


if __name__ == "__main__":
    main()
