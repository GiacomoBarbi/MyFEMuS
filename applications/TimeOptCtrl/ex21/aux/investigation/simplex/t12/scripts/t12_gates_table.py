#!/usr/bin/env python3
"""logs/gates/*.log -> ../../t12_gates.csv (tags group_mesh_wind_scheme).
Pass: G1 orientation <= 1e-12 and free-subspace skew <= 1e-12 (wind on); G2 shell transpose, assembled
E, reduced-Hessian symmetry (exact LU and production shell) <= 1e-12; G3 FD best <= 1e-8 (exact and
shell); G4 production reason 2, true residual <= 1e-10, |grad J| <= 1e-8, forward march <= 1e-10.
Negatives (ex20-style untransposed adjoint): w != 0 -> G2 transpose and G3 FD > 1e-4 (direct /
gradient), G2 Hessian symmetry > 1e-8 (derived, T11H1 rule); heat / w = 0 -> they coincide (<= 1e-12)."""
import csv, glob, os, re
D = os.path.dirname(os.path.abspath(__file__)) + "/.."
rows, allok = [], True
def g(p, s, d=""):
    m = re.search(p, s, re.M); return m.group(1) if m else d
for f in sorted(glob.glob(D + "/logs/gates/*.log")):
    s = open(f).read(); tag = os.path.basename(f)[:-4]
    group, mesh, wind, scheme = tag.split("_")
    o = re.findall(r"G1_ORIENTATION axis=\w v=\S+ \|C\*lin - v\*M1\|/\|M1\|=(\S+) \|C\^T\*lin - v\*M1\|/\|M1\|=(\S+)", s)
    hs = dict(re.findall(r"G2_REDUCED_HESSIAN adjoint=(\S+) .*? rel=(\S+)", s))
    fd = dict(re.findall(r"G3_SUMMARY adjoint=(\S+) adjoint_its=\S+ best_rel=(\S+)", s))
    r = {"group": group, "mesh": mesh, "wind": wind, "scheme": scheme.upper(),
         "G1_orient": max((float(a) for a, b in o), default=float("nan")) if o else "",
         "G1_neg_transposed": min((float(b) for a, b in o), default=float("nan")) if o else "",
         "G1_skew_free": g(r"G1_SKEW_DIRICHLET \|\|C\+C\^T\|\|_F/\|\|C\|\|_F=(\S+)", s),
         "G2_shell_transpose": g(r"G2_SHELL_TRANSPOSE .*\|diff\|/\(\|Ex\|\|y\|\)=(\S+)", s),
         "G2_assembled": max((float(v) for v in re.findall(r"G2_ASSEMBLED .*?=(\S+) .*?=(\S+) .*?=(\S+)$", s, re.M)[0]), default=float("nan")) if re.search("G2_ASSEMBLED", s) else "",
         "G2_neg_ex20": g(r"G2_NEGATIVE_CONTROL .*=(\S+)$", s),
         "G2_hess_exact": hs.get("exact_LU(E^T)", ""), "G2_hess_shell": hs.get("production_shell(E^T)", ""),
         "G2_hess_neg_ex20": hs.get("ex20_style_wrong", ""),
         "G3_fd_exact": fd.get("exact_LU(E^T)", ""), "G3_fd_shell": fd.get("production_shell(E^T)", ""),
         "G3_fd_neg_ex20": fd.get("ex20_style_wrong", ""),
         "G4_reason": g(r"G4_PRODUCTION reason=(\S+)", s), "G4_outer": g(r"G4_PRODUCTION .*?outer_its=(\S+)", s),
         "G4_true_res": g(r"G4_PRODUCTION .*?true_residual=(\S+)", s),
         "G4_gradJ": g(r"\|grad J\(u\*\)\|/\|dt alpha_u N u\*\|=(\S+)", s),
         "G4_march": g(r"\|y_prod - E\^-1\(rhs\+eta N u\*\)\|/\|y\|=(\S+)", s)}
    F = lambda k: float(r[k]) if r[k] != "" else float("nan")
    on = wind != "heat"
    nonsym = wind in ("wA", "wB")
    pos = all(F(k) <= 1e-12 for k in ("G2_shell_transpose", "G2_assembled", "G2_hess_exact", "G2_hess_shell"))
    pos &= F("G3_fd_exact") <= 1e-8 and F("G3_fd_shell") <= 1e-8
    pos &= r["G4_reason"] == "2" and F("G4_true_res") <= 1e-10 and F("G4_gradJ") <= 1e-8 and F("G4_march") <= 1e-10
    if nonsym:
        pos &= F("G1_orient") <= 1e-12 and F("G1_skew_free") <= 1e-12 and F("G1_neg_transposed") > 1e-3
        neg = F("G2_neg_ex20") > 1e-4 and F("G3_fd_neg_ex20") > 1e-4 and F("G2_hess_neg_ex20") > 1e-8
        r["negatives"] = "detected" if neg else "MISSED"
    else:
        neg = all(F(k) <= 1e-12 for k in ("G2_neg_ex20", "G2_hess_neg_ex20")) if r["G2_neg_ex20"] != "" else True
        neg &= abs(F("G3_fd_neg_ex20") - F("G3_fd_exact")) <= 1e-9 if r["G3_fd_neg_ex20"] != "" else True
        r["negatives"] = "coincide (L symmetric)" if neg else "MISMATCH"
    r["verdict"] = "PASS" if pos and neg else "FAIL"; allok &= pos and neg
    rows.append(r)
with open(D + "/../t12_gates.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows:
    if r["verdict"] != "PASS": print({k: r[k] for k in r})
print("T12_GATES_RESULT", "PASS" if allok else "FAIL", f"({sum(r['verdict'] == 'PASS' for r in rows)}/{len(rows)})")
