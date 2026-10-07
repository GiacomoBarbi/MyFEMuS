#!/usr/bin/env python3
"""Zero-velocity solution parity: ex21 (off / w=0) vs ex20, same solver flags."""
import csv, glob, json, os, re, sys
import numpy as np
D = os.path.dirname(os.path.abspath(__file__)) + "/.."; J = sys.argv[1]
def load(tag):
    p = f"{J}/{tag}.json"
    if not os.path.exists(p): return None, None, None
    raw = open(p).read()
    # non-converged runs can print C-style nan/inf, which json rejects
    d = json.loads(re.sub(r'(?<![\w"])-?(nan|inf)(?![\w"])', "null", raw).replace("null", "NaN"))
    f = {k: np.array([s["values"] for s in d[k]]) for k in ("state", "adjoint", "control_full")}
    canon = re.sub(r',"convection":\{[^}]*\}', "", raw)      # ex21's extra config block (convection on)
    canon = re.sub(r'"json":"[^"]*"', "", canon)
    return d, f, canon
def logi(tag):
    p = f"{D}/logs/parity/{tag}.log"
    s = open(p).read() if os.path.exists(p) else ""
    g = lambda pat: (re.search(pat, s).group(1) if re.search(pat, s) else "")
    return {"pc": g(r"EX20_PAR_SOLVE .*? pc=(\S+)"), "reason": g(r"EX20_PAR_SOLVE .*? reason=(\S+)"),
            "outer": g(r"outer_iterations=(\d+)"), "refused": g(r"what\(\):\s+(.{0,90})"),
            "core": g(r"EX21_T8D_CORE_STATS mode=(\S+)"), "rejected": "EX21_SOLVE_REJECTED" in s,
            "lines": [re.sub(r" (t_operators|t_pcsetup|t_kspsolve|t_total|setup_seconds|json)=\S*", "", l)
                      for l in s.splitlines() if l.startswith(("EX20_PAR_SOLVE", "EX20_SOLVERSTATS", "EX20_REDUCED_SCHUR"))]}
tags = sorted({os.path.basename(p)[:-len("_ex21off.log")] for p in glob.glob(f"{D}/logs/parity/*_ex21off.log")})
rows = []
for t in tags:
    ref_tag = t if "_dense" not in t and "_iterative" not in t else re.sub(r"_(dense|iterative)$", "_default", t)
    d0, f0, c0 = load(ref_tag + "_ex20"); L0 = logi(ref_tag + "_ex20")
    for b in ("ex21off", "ex21w0"):
        d, f, c = load(f"{t}_{b}"); L = logi(f"{t}_{b}")
        r = {"case": t, "variant": b, "ex20_reference": ref_tag + "_ex20", "pc_ex20": L0["pc"], "pc_ex21": L["pc"],
             "core_ex21": L["core"] or ("dense" if "dirichlet" in t and L["pc"] else ""),
             "reason_ex20": L0["reason"] or ("REFUSED" if L0["refused"] else ""), "reason_ex21": L["reason"] or ("REFUSED" if L["refused"] else ""),
             "outer_ex20": L0["outer"], "outer_ex21": L["outer"], "refusal_ex21": L["refused"]}
        if d is not None and d0 is not None:
            r["fields_bitwise"] = all(np.array_equal(f[k], f0[k]) for k in f0)
            r["max_field_rel"] = f"{max(np.abs(f[k]-f0[k]).max()/max(np.abs(f0[k]).max(),1e-300) for k in f0):.2e}"
            for k in ("J_track", "J_reg"):
                r[k + "_rel"] = f"{abs(d['functional'][k]-d0['functional'][k])/max(abs(d0['functional'][k]),1e-300):.2e}"
            r["true_res_ex20"] = f"{d0['solve']['true_residual']:.2e}"; r["true_res_ex21"] = f"{d['solve']['true_residual']:.2e}"
            r["kkt_state_ex21"] = f"{d['kkt']['state_kkt_residual_rel']:.1e}"; r["kkt_adjoint_ex21"] = f"{d['kkt']['adjoint_kkt_residual_rel']:.1e}"
            r["json_byte_identical"] = c == c0
            r["log_lines_identical"] = L["lines"] == L0["lines"]
        rows.append(r)
keys = []
for r in rows:
    for k in r:
        if k not in keys: keys.append(k)
with open(D + "/results/parity_solutions.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
for r in rows:
    print(f"{r['case']:<34}{r['variant']:<8} pc20={r['pc_ex20']:<24} pc21={r['pc_ex21']:<24} {r.get('core_ex21',''):<9} reason {r['reason_ex20']:>7}/{r['reason_ex21']:<7} outer {r['outer_ex20']:>3}/{r['outer_ex21']:<3} bitwise={str(r.get('fields_bitwise','-')):<5} maxrel={r.get('max_field_rel','-'):<9} json={str(r.get('json_byte_identical','-')):<5} log={str(r.get('log_lines_identical','-')):<5} {r['refusal_ex21'][:50]}")
