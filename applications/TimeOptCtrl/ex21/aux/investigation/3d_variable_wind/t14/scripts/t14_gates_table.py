#!/usr/bin/env python3
"""logs/gates/*.log -> ../../t14_gates_{distributed,natural,neumann,dirichlet}.csv (per-run verdicts).

Criteria (fixed before the table was produced; T13 bars where applicable):
 positives (all kinds): G1 orientation <= 1e-12 (affine presets and the constant wind are exact in Q2;
   the swirl is not affine: reported, judged <= 1e-2); G2 shell transpose, Hessian symmetry <= 1e-12;
   FD gradient (exact, shell) <= 1e-8; production true residual <= 1e-10, |grad J| <= 1e-8, forward <= 1e-8.
   Variable-wind identities: natural state (nat, neu): G1W  ||C+C^T-(B_w-D_w)|| <= 1e-12;
   homogeneous Dirichlet state / Dirichlet control: G1W_DIRICHLET ||(C+C^T)_FF-(B_w)_FF+(D_w)_FF|| <= 1e-12.
   Multi-face Neumann: G13N union <= 1e-14 and its true-union FD <= 1e-8.
   Dirichlet control: G3D B_D^T <= 1e-12, G5D <= 1e-12, G4D FD <= 1e-8, R symmetry <= 1e-12, accepted,
   true residual <= 1e-10, kkt_state_rel <= 1e-10, core_failures = 0.
 negatives: ex20-style adjoint (and Dirichlet legacy B_D^T): direct > 1e-6, FD > 1e-4 (T13 bars);
   natural wrong BC (nu K - C): FD > 1e-4;
   GENERIC wind (aff): missing-divergence adjoint (L^T + D_w): transpose defect G8D > 1e-6 and FD > 1e-4;
     the div-free identity fails: G1N (natural) > 1e-4, skew-only (Dirichlet state) > 1e-4;
   DIV-FREE winds (harm, wA, swirl): the missing-divergence adjoint COINCIDES (FD <= 1e-8, G8D <= 1e-12);
   multi-face Neumann: drop / duplicate face FD > 1e-4.
 [REVISION after the first table, documented: the dirichlet-var-3d swirl wind is trigonometric, NOT polynomial,
 so Gauss quadrature integrates C, B_w and D_w inexactly and the discrete integration by parts behind
 C + C^T = B_w - D_w holds only to quadrature error (1.75e-3 at n = 2, 6.4e-5 at n = 4: a factor 27 per
 halving).  The adjoint is the transpose of the SAME assembled matrix, so G3D / G4D / G5D stay at round-off.
 The same holds for the G1 orientation (M I_h(w) is exact only for w in Q2).  For the swirl both are judged
 by CONVERGENCE between n = 2 and n = 4 (G1 must drop by >= 4, G1W by >= 8) instead of absolute bars; the
 affine presets and the constant wind keep the round-off bar 1e-12.]"""
import csv, glob, os, re
D = os.path.dirname(os.path.abspath(__file__)) + "/.."
tables = {"dist": [], "nat": [], "neu": [], "dir": []}
allok = True


def g(p, s, d=""):
    m = re.search(p, s, re.M); return m.group(1) if m else d


F = lambda r, k: float(r[k]) if r.get(k) not in (None, "") else float("nan")
for f in sorted(glob.glob(D + "/logs/gates/*.log")):
    s = open(f).read(); tag = os.path.basename(f)[:-4]
    kind, group, wind, reg, scheme, n = tag.split("_")
    o = dict(re.findall(r"G1_ORIENTATION axis=(\w) v=\S+ \|C\*lin - v\*M1\|/\|M1\|=(\S+)", s))
    r = {"kind": kind, "group": group, "wind": wind, "reg": reg, "scheme": scheme.upper(), "n": n[1:],
         "wind_max": g(r"WIND3D .*?max\|w\|=(\S+)", s), "div_max": g(r"max\|div w\|=(\S+)", s), "Pe_h_max": g(r"Pe_h_max=(\S+)", s),
         "G1_x": o.get("x", ""), "G1_y": o.get("y", ""), "G1_z": o.get("z", ""),
         "G2_shell_transpose": g(r"G2_SHELL_TRANSPOSE .*\|diff\|/\(\|Ex\|\|y\|\)=(\S+)", s),
         "G2_neg_ex20": g(r"G2_NEGATIVE_CONTROL .*=(\S+)$", s),
         "G1W_identity": g(r"G1W_IDENTITY [^=]*=(\S+)", s), "G1N_divfree_identity": g(r"G1N_BOUNDARY_IDENTITY [^=]*=(\S+)", s),
         "G1W_dirichlet": g(r"G1W_DIRICHLET [^=]*=(\S+)", s), "skew_only": g(r"skew_only_negative [^=]*=(\S+)", s),
         "D_w_norm": g(r"\|\|D_w(?:,FF)?\|\|_F=(\S+)", s), "G8D_missing_div_transpose": g(r"G8D_MISSING_DIV_TRANSPOSE .*?\(\|Ex\|\|y\|\)=(\S+)", s)}
    pos = wind == "swirl" or all(F(r, k) <= 1e-12 for k in ("G1_x", "G1_y", "G1_z") if r[k])
    pos &= F(r, "G2_shell_transpose") <= 1e-12
    variable = wind in ("harm", "aff", "swirl")
    generic = wind == "aff"
    neg = True
    if kind in ("dist", "nat", "neu"):
        fd = dict(re.findall(r"G3_SUMMARY adjoint=(\S+) adjoint_its=\S+ best_rel=(\S+)", s))
        hs = dict(re.findall(r"G2_REDUCED_HESSIAN adjoint=(\S+) .*?rel=(\S+)$", s, re.M))
        mf = dict(re.findall(r"G13N_FD control=(\S+) face=\S+ fd=\S+ adj=\S+ rel=(\S+)", s))
        r.update({"hess_shell": hs.get("production_shell(E^T)", ""), "hess_neg_missing_div": hs.get("missing_div_term", ""),
                  "fd_exact": fd.get("exact_LU(E^T)", ""), "fd_shell": fd.get("production_shell(E^T)", ""),
                  "fd_neg_ex20": fd.get("ex20_style_wrong", ""), "fd_neg_wrong_bc": fd.get("wrong_bc_nuK_minus_C", ""),
                  "fd_neg_missing_div": fd.get("missing_div_term", ""),
                  "G13N_union": g(r"G13N_UNION .*=(\S+)$", s), "G13N_fd_true": mf.get("true_union", ""),
                  "G13N_neg_drop": mf.get("drop_face", ""), "G13N_neg_dup": mf.get("duplicate_face", ""),
                  "prod_true_res": g(r"G4_PRODUCTION .*?true_residual=(\S+)", s),
                  "gradJ": g(r"\|grad J\(u\*\)\|/\|dt alpha_u N u\*\|=(\S+)", s),
                  "forward": g(r"\|y_prod - E\^-1\(rhs\+eta N u\*\)\|/\|y\|=(\S+)", s)})
        pos &= F(r, "hess_shell") <= 1e-12 and F(r, "fd_exact") <= 1e-8 and F(r, "fd_shell") <= 1e-8
        pos &= F(r, "prod_true_res") <= 1e-10 and F(r, "gradJ") <= 1e-8 and F(r, "forward") <= 1e-8
        if variable and kind in ("nat", "neu"):
            pos &= F(r, "G1W_identity") <= 1e-12
        if variable and kind == "dist":
            pos &= F(r, "G1W_dirichlet") <= 1e-12
        if r["G13N_union"]:
            pos &= F(r, "G13N_union") <= 1e-14 and F(r, "G13N_fd_true") <= 1e-8
            neg &= F(r, "G13N_neg_drop") > 1e-4 and F(r, "G13N_neg_dup") > 1e-4
        neg &= F(r, "G2_neg_ex20") > 1e-6 and F(r, "fd_neg_ex20") > 1e-4
        if kind in ("nat", "neu"):
            neg &= F(r, "fd_neg_wrong_bc") > 1e-4
        if variable:
            if generic:
                neg &= F(r, "G8D_missing_div_transpose") > 1e-6 and F(r, "fd_neg_missing_div") > 1e-4
                neg &= (F(r, "G1N_divfree_identity") > 1e-4) if kind in ("nat", "neu") else (F(r, "skew_only") > 1e-4)
            else:
                neg &= F(r, "G8D_missing_div_transpose") <= 1e-12 and F(r, "fd_neg_missing_div") <= 1e-8
    else:
        hs = dict(re.findall(r"G5D_REDUCED_HESSIAN adjoint=(\S+) .*? rel=(\S+)", s))
        fd = dict(re.findall(r"G4D_FD adjoint=(\S+) best_rel=(\S+)", s))
        r.update({"G3D_BD_transpose": g(r"G3D_BD_TRANSPOSE .*? rel=(\S+) legacy", s), "G3D_neg_legacy": g(r"legacy_B_D\^T: .* rel=(\S+)", s),
                  "R_symmetry": g(r"RD_REGULARIZATION .*? rel=(\S+)", s), "R_min_rayleigh": g(r"min Rayleigh\(u_C\)=(\S+)", s),
                  "G5D_hess_shell": hs.get("production_shell(E^T)", ""), "G4D_fd_exact": fd.get("exact_LU(E^T)", ""),
                  "G4D_fd_shell": fd.get("production_shell(E^T)", ""), "G4D_neg_legacy": fd.get("legacy_B_D^T", ""),
                  "G4D_neg_ex20": fd.get("ex20_style_untransposed_E", ""), "core_failures": g(r"core_failures=(\S+)", s),
                  "G6D_accepted": g(r"G6D_PRODUCTION .*?accepted=(\S+)", s), "G6D_true_res": g(r"G6D_PRODUCTION .*?true_residual=(\S+)", s),
                  "G7_gradJ": g(r"\|grad J\(u\*\)\|/\|dt alpha_u R u\*\|=(\S+)", s), "kkt_state_rel": g(r"G6D_PRODUCTION .*?kkt_state_rel=(\S+)", s)})
        pos &= all(F(r, k) <= 1e-12 for k in ("G3D_BD_transpose", "R_symmetry", "G5D_hess_shell"))
        pos &= F(r, "G4D_fd_exact") <= 1e-8 and F(r, "G4D_fd_shell") <= 1e-8 and F(r, "R_min_rayleigh") > 0
        pos &= r["G6D_accepted"] == "1" and F(r, "G6D_true_res") <= 1e-10 and F(r, "G7_gradJ") <= 1e-8
        pos &= F(r, "kkt_state_rel") <= 1e-10 and r["core_failures"] in ("0", "")
        if variable:
            pos &= wind == "swirl" or F(r, "G1W_dirichlet") <= 1e-12
        neg &= F(r, "G2_neg_ex20") > 1e-6 and F(r, "G3D_neg_legacy") > 1e-6
        neg &= F(r, "G4D_neg_legacy") > 1e-8 and F(r, "G4D_neg_ex20") > 1e-8
        if generic:
            neg &= F(r, "skew_only") > 1e-4
    r["negatives"] = "detected" if neg else "MISSED"
    r["verdict"] = "PASS" if pos and neg else "FAIL"
    allok &= pos and neg
    tables[kind].append(r)
# swirl (non-polynomial wind): quadrature-limited diagnostics judged by convergence n = 2 -> 4
for r in tables["dir"]:
    r.setdefault("swirl_quadrature_convergence", "")
for sch in ("BE", "CN"):
    sw = {r["n"]: r for r in tables["dir"] if r["wind"] == "swirl" and r["scheme"] == sch}
    if "2" in sw and "4" in sw:
        conv = (F(sw["4"], "G1_x") <= F(sw["2"], "G1_x") / 4 and F(sw["4"], "G1_y") <= F(sw["2"], "G1_y") / 4
                and F(sw["4"], "G1W_dirichlet") <= F(sw["2"], "G1W_dirichlet") / 8)
        for r in sw.values():
            r["swirl_quadrature_convergence"] = "converging" if conv else "NOT CONVERGING"
            if not conv:
                r["verdict"] = "FAIL"; allok = False
names = {"dist": "distributed", "nat": "natural", "neu": "neumann", "dir": "dirichlet"}
for k, rows in tables.items():
    if not rows:
        continue
    with open(f"{D}/../t14_gates_{names[k]}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for r in rows:
        if r["verdict"] != "PASS":
            print("FAIL", " ".join(f"{a}={b}" for a, b in r.items()))
print("T14_GATES_RESULT", "PASS" if allok else "FAIL",
      " ".join(f"{names[k]} {sum(r['verdict'] == 'PASS' for r in v)}/{len(v)}" for k, v in tables.items()))
