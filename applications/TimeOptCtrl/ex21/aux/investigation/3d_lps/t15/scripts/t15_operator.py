#!/usr/bin/env python3
"""T15: independent reference assembly of the 3D LPS matrix W (dump based, serial) + projection checks.

The binary's W is recovered from the operator dump as W_dump = K(--convection-stabilization lps) - K(none)
(same mesh, same wind, natural state BC: no elimination; one homogeneous-Dirichlet case compares the free
block).  Python assembles, on the structured HEX27 lattice of the box, with tensor-product 1D Q2 Lagrange
bases and the SAME 4-point Gauss rule per axis as FEMuS ("seventh"):
  per element K:  c = box centre, H = extents, w_K = w(c), |w_K|,
                  h_K = min_{i: w_K,i != 0} H_i |w_K| / |w_K,i|,  Pe_K = |w_K| h_K / (2 nu),
                  delta_K = h_K / (2 |w_K|) (1 - 1/Pe_K) if Pe_K > 1 else 0,
                  f_g = w(x_g) . grad phi(x_g),  q = {1} (P0) | {1, (x_i - c_i)/H_i} (P1),
                  W_K = delta_K (A - C^T G^-1 C),  A = sum w f f^T, C = sum w q f^T, G = sum w q q^T.
NEGATIVES (each must differ from the dump): h_K = max extent; integrand wind frozen at the centroid;
no projection (delta A); wrong Pe threshold (Pe > 2, and Pe > 1/2 with the same delta formula).
[Gram bars 4e-15 (a few ulps): the first bar 1e-15 failed at 1.1e-15 under multithreaded OpenBLAS (summation
order); BLAS is now pinned to one thread.]
Projection checks (Python, per element): G is diagonal |K| diag(1, 1/12, 1/12, 1/12); Pi q = q; kappa q = 0.
Binary-level exact-projection gates: W_P0 v = 0 for v affine (constant wind: w.grad v constant);
W_P1 v = 0 for v quadratic (constant wind: w.grad v affine); NEG W_P0 v_quad != 0.
usage: t15_operator.py <projection.csv> <reference.csv>"""
import csv, itertools, os, subprocess, sys
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("OMP_NUM_THREADS", "1")   # deterministic sums
import numpy as np
import scipy.sparse as sps

X = os.environ.get("EX21_BIN", "/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21")
W = os.environ.get("T15_WORK", "/tmp/t15_op"); os.makedirs(W + "/output", exist_ok=True)
ENV = {"HOME": os.environ["HOME"], "PATH": "/home/giacomo/software/petsc/arch-linux-c-openblas-opt/bin:/usr/bin:/bin",
       "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
HARM = np.array([[0, 1, -0.75], [1, 0, 0.5], [-0.75, 0.5, 0]], float)
FIELDS = {"harmonic-3d": HARM, "affine-div-3d": HARM + np.diag([0.5, 0.25, 0.25])}
GL, GW = np.polynomial.legendre.leggauss(4)
pr_rows, ref_rows, ok = [], [], True


def coo(path):
    with open(path) as fh:
        fh.readline(); r, c = map(int, fh.readline().split()); fh.readline()
        d = np.loadtxt(fh, ndmin=2)
    return sps.csr_matrix((d[:, 2], (d[:, 0].astype(int), d[:, 1].astype(int))), shape=(r, c)) if d.size else sps.csr_matrix((r, c))


def rec(table, case, check, value, ref, good):
    global ok
    ok &= bool(good)
    table.append({"case": case, "check": check, "value": value, "reference": ref, "verdict": "PASS" if good else "FAIL"})
    print(f"{'PASS' if good else 'FAIL'} {case}: {check} = {value} ({ref})", flush=True)


def lag(xi):   # 1D Q2 on [0,1], nodes 0, 1/2, 1: values and derivatives
    return (np.array([2 * (xi - .5) * (xi - 1), -4 * xi * (xi - 1), 2 * xi * (xi - .5)]),
            np.array([4 * xi - 3, -8 * xi + 4, 4 * xi - 1]))


Q = (GL + 1) / 2; QW = GW / 2
V = [lag(s) for s in Q]


def element_W(wfun, x0, h, nu, proj, variant="exact", stats=None):
    """local 27x27 W_K (lattice-local order a,b,c), the variants of the module docstring"""
    cen = x0 + h / 2
    wk = wfun(cen)
    sp = np.linalg.norm(wk)
    if variant == "hmax":
        hk = h.max()
    else:
        hk = min((h[i] * sp / abs(wk[i]) for i in range(3) if wk[i] != 0), default=np.inf)
    pe = sp * hk / (2 * nu) if sp > 0 else 0.0
    thr = {"pe2": 2.0, "pehalf": 0.5}.get(variant, 1.0)
    if stats is not None:
        stats.append((pe, hk))
    if not pe > thr:
        return None
    delta = hk / (2 * sp) * (1 - 1 / pe)
    nq = 1 if proj == "p0" else 4
    A, C, G = np.zeros((27, 27)), np.zeros((nq, 27)), np.zeros((nq, nq))
    for (ia, va), (ib, vb), (ic, vc) in itertools.product(enumerate(V), repeat=3):
        g = np.stack([np.einsum("a,b,c->abc", va[1], vb[0], vc[0]).ravel() / h[0],
                      np.einsum("a,b,c->abc", va[0], vb[1], vc[0]).ravel() / h[1],
                      np.einsum("a,b,c->abc", va[0], vb[0], vc[1]).ravel() / h[2]], 1)
        p = x0 + np.array([Q[ia], Q[ib], Q[ic]]) * h
        wt = QW[ia] * QW[ib] * QW[ic] * np.prod(h)
        f = g @ (wfun(cen) if variant == "frozen" else wfun(p))
        q = np.concatenate([[1.0], (p - cen) / h])[:nq]
        A += wt * np.outer(f, f); C += wt * np.outer(q, f); G += wt * np.outer(q, q)
    if variant == "noproj":
        return delta * A
    Wl = delta * (A - C.T @ np.linalg.solve(G, C))
    return 0.5 * (Wl + Wl.T)


def reference(wfun, lo, hi, n, gidx, nu, proj, variant="exact", stats=None):
    N = len(gidx)
    h = (hi - lo) / n
    rows, cols, vals = [], [], []
    for e in itertools.product(range(n[0]), range(n[1]), range(n[2])):
        nodes = [gidx[(2 * e[0] + a, 2 * e[1] + b, 2 * e[2] + c)] for a in range(3) for b in range(3) for c in range(3)]
        Wl = element_W(wfun, lo + np.array(e) * h, h, nu, proj, variant, stats)
        if Wl is None:
            continue
        rr, cc = np.meshgrid(nodes, nodes, indexing="ij")
        rows += list(rr.ravel()); cols += list(cc.ravel()); vals += list(Wl.ravel())
    return sps.csr_matrix((vals, (rows, cols)), shape=(N, N))


def run(args, prefix):
    r = subprocess.run([X] + args + ["--dump-operators", prefix, "--no-paraview"], cwd=W, env=ENV,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    if r.returncode != 0:
        raise RuntimeError(f"{prefix}: {r.stdout[-1500:]}")
    return r.stdout


def rel(A, B):
    d = (A - B).tocsr(); d.eliminate_zeros()
    return (abs(d).max() if d.nnz else 0.0) / max(abs(B).max() if B.nnz else 0.0, 1e-300)


def case(wind, dom, n, nu, proj, state="natural", negs=True):
    unit = dom == (0, 1, 0, 1, 0, 1)
    wname = wind if isinstance(wind, str) else "c(" + ",".join(f"{v:g}" for v in wind) + ")"
    tag = f"{wname}_{'x'.join(map(str, n))}{'' if unit else '_nonunit'}_nu{nu:g}_{proj}_{state}"
    lo, hi = np.array(dom[0::2], float), np.array(dom[1::2], float)
    if isinstance(wind, str):
        H = FIELDS[wind]; wfun = lambda p: H @ p
        wargs = ["--convection", "on", "--wind", wind]
    else:
        wv = np.array(wind, float); wfun = lambda p: wv
        wargs = ["--convection", "on", "--vx", repr(wind[0]), "--vy", repr(wind[1]), "--vz", repr(wind[2])]
    base = ["--dimension", "3", "--distributed-solve", "--control-full-domain", "--tracking-box3d"] + [repr(v) for v in dom] + \
           ["--domain3d"] + [repr(v) for v in dom] + ["--state-bc", state, "--nx", str(n[0]), "--ny", str(n[1]), "--nz", str(n[2]),
                                                    "--steps", "2", "--nu", repr(nu), "--pc-schur-sweep"] + wargs
    out = run(base + ["--convection-stabilization", "lps", "--lps-projection", proj], f"{W}/{tag}_on")
    run(base, f"{W}/{tag}_off")
    Wd = (coo(f"{W}/{tag}_on_K.coo") - coo(f"{W}/{tag}_off_K.coo")).tocsr()
    xyz = np.loadtxt(f"{W}/{tag}_on_coords.txt", ndmin=2)
    lat = np.round((xyz - lo) / (hi - lo) * 2 * np.array(n)).astype(int)
    gidx = {tuple(l): i for i, l in enumerate(lat)}
    stats = []
    Wr = reference(wfun, lo, hi, n, gidx, nu, proj, stats=stats)
    pe = np.array([s[0] for s in stats])
    act = int((pe > 1).sum())
    line = [l for l in out.splitlines() if l.startswith("EX21_LPS ")]
    b_act = int(line[0].split("active=")[1].split()[0]) if line else 0
    rec(ref_rows, tag, "active patches binary = reference", f"{b_act}/{len(pe)}", f"{act}/{len(pe)} (Pe in [{pe.min():.3g},{pe.max():.3g}])", b_act == act)
    if state == "natural":
        d = rel(Wd, Wr)
        rec(ref_rows, tag, "W_dump = W_ref (rel. max entrywise)", f"{d:.1e}", "<=1e-12", d <= 1e-12)
    else:
        M = coo(f"{W}/{tag}_on_M.coo")
        F = np.where(abs(M).sum(1).A1 > 0)[0]
        d = rel(Wd[F][:, F], Wr[F][:, F])
        rec(ref_rows, tag, "W_FF (eliminated) = W_ref[F,F]", f"{d:.1e}", "<=1e-12", d <= 1e-12)
        return
    if Wr.nnz == 0:
        rec(ref_rows, tag, "W = 0 (no active patch)", f"{abs(Wd).max() if Wd.nnz else 0:.1e}", "0", Wd.nnz == 0 or abs(Wd).max() == 0)
    else:
        S = Wd.toarray()
        rec(ref_rows, tag, "symmetry ||W-W^T||/||W||", f"{np.linalg.norm(S - S.T) / np.linalg.norm(S):.1e}", "<=1e-14",
            np.linalg.norm(S - S.T) <= 1e-14 * np.linalg.norm(S))
        ev = np.linalg.eigvalsh(0.5 * (S + S.T))
        rec(ref_rows, tag, "PSD lambda_min/lambda_max", f"{ev[0] / ev[-1]:.1e}", ">= -1e-13", ev[0] >= -1e-13 * ev[-1])
        one = np.ones(len(xyz))
        rec(ref_rows, tag, "constants ||W 1||_inf/||W||_max", f"{abs(S @ one).max() / abs(S).max():.1e}", "<=1e-13",
            abs(S @ one).max() <= 1e-13 * abs(S).max())
    if negs and Wr.nnz:
        for var, lab in (("hmax", "h_K = max extent"), ("frozen", "integrand wind frozen at the centroid"),
                         ("noproj", "no projection (delta A)"), ("pe2", "wrong Pe threshold Pe > 2"),
                         ("pehalf", "wrong Pe threshold Pe > 1/2")):
            Wn = reference(wfun, lo, hi, n, gidx, nu, proj, variant=var)
            dn = rel(Wd, Wn) if Wn.nnz else 1.0
            undetectable = (var == "frozen" and not isinstance(wind, str)) or \
                           (var == "hmax" and np.count_nonzero(wfun((lo + hi) / 2)) == 1 and np.argmax(abs(wfun((lo + hi) / 2))) == np.argmax((hi - lo) / n)) or \
                           (var == "pe2" and not ((pe > 1) & (pe <= 2)).any()) or (var == "pehalf" and not ((pe > 0.5) & (pe <= 1)).any())
            if undetectable:
                rec(ref_rows, tag, f"NEG {lab}: coincides here by construction", f"{dn:.1e}", "<=1e-12 (identical form)", dn <= 1e-12)
            else:
                rec(ref_rows, tag, f"NEG {lab}: differs", f"{dn:.1e}", "> 1e-6", dn > 1e-6)
    return Wd, xyz


def projection_checks():
    """element-level Gram / projection identities on two element shapes (Python, the frozen algebra)"""
    for h in (np.array([0.5, 0.5, 0.5]), np.array([2 / 3, 1.0, 0.25])):
        tag = f"elem_H=({h[0]:.4g},{h[1]:.4g},{h[2]:.4g})"
        G = np.zeros((4, 4)); vol = np.prod(h)
        Pq = []
        for ia, ib, ic in itertools.product(range(4), repeat=3):
            p = np.array([Q[ia], Q[ib], Q[ic]]) * h
            q = np.concatenate([[1.0], (p - h / 2) / h])
            wt = QW[ia] * QW[ib] * QW[ic] * vol
            G += wt * np.outer(q, q); Pq.append((wt, q))
        Gex = vol * np.diag([1, 1 / 12, 1 / 12, 1 / 12])
        rec(pr_rows, tag, "P1 Gram G = |K| diag(1, 1/12, 1/12, 1/12)", f"{abs(G - Gex).max() / vol:.1e}", "<=4e-15", abs(G - Gex).max() <= 4e-15 * vol)
        rec(pr_rows, tag, "P0 Gram G = |K| (1x1)", f"{abs(G[0, 0] - vol) / vol:.1e}", "<=4e-15", abs(G[0, 0] - vol) <= 4e-15 * vol)
        rec(pr_rows, tag, "P1 Gram condition number", f"{np.linalg.cond(G):.4g}", "12 (scaled basis)", abs(np.linalg.cond(G) - 12) < 1e-10)
        # Pi q = q for every basis function, i.e. kappa q = 0 at every quadrature point
        Cq = sum(wt * np.outer(q, q) for wt, q in Pq)
        coef = np.linalg.solve(G, Cq)   # projection coefficients of each q_m
        rec(pr_rows, tag, "Pi_1 q_m = q_m (kappa_1 annihilates P1)", f"{abs(coef - np.eye(4)).max():.1e}", "<=1e-14", abs(coef - np.eye(4)).max() <= 1e-14)
        # P0 of a linear function: the mean; kappa_0 (x - c) != 0 (NEG: P0 does not annihilate P1)
        m = sum(wt * q[1] for wt, q in Pq) / vol
        rec(pr_rows, tag, "Pi_0 (x-c)/H = mean = 0 (so kappa_0 leaves the linear part)", f"{abs(m):.1e}", "<=1e-15", abs(m) <= 1e-15)


def binary_projection_gates(Wd0, Wd1, xyz, w):
    """W_P0 v = 0 for affine v, W_P1 v = 0 for quadratic v (constant wind); NEG W_P0 v_quad != 0"""
    tag = "binary_exact_projection"
    x, y, z = xyz.T
    va = 0.3 + 1.7 * x - 0.9 * y + 2.2 * z
    vq = va + x * x - 0.5 * x * y + 1.3 * y * z + 0.7 * z * z - y * y
    for Wd, v, lab, zero in ((Wd0, va, "W_P0 v_affine = 0", True), (Wd1, va, "W_P1 v_affine = 0", True),
                             (Wd1, vq, "W_P1 v_quadratic = 0", True), (Wd0, vq, "NEG W_P0 v_quadratic != 0", False)):
        r = abs(Wd @ v).max() / (abs(Wd).max() * abs(v).max())
        rec(pr_rows, tag, lab, f"{r:.1e}", "<=1e-12" if zero else "> 1e-4", r <= 1e-12 if zero else r > 1e-4)


def main():
    projection_checks()
    A = (1.0, 0.5, -0.75)
    unit, nonunit = (0, 1, 0, 1, 0, 1), (0, 2, -1, 1, 0, 0.5)
    # constant wind, high Pe (all active), P0 and P1
    Wd0, xyz = case(A, unit, (2, 2, 2), 1e-2, "p0")
    Wd1, _ = case(A, unit, (2, 2, 2), 1e-2, "p1")
    binary_projection_gates(Wd0, Wd1, xyz, A)
    case(A, nonunit, (3, 2, 2), 1e-2, "p0")
    case(A, nonunit, (3, 2, 2), 1e-2, "p1")
    # constant wind near the switch: Pe = 1.51 (Pe > 2 negative detectable), Pe = 0.60 (inactive; Pe > 1/2 negative)
    case(A, nonunit, (3, 2, 2), 0.2, "p0")
    case(A, nonunit, (3, 2, 2), 0.5, "p0")
    # zero components of w_K (chord = the one extent), mixed Dirichlet free block
    case((1.0, 0.0, 0.0), nonunit, (3, 2, 2), 1e-2, "p0")
    case((0.0, 0.0, -0.75), nonunit, (2, 3, 4), 1e-2, "p0")
    case(A, nonunit, (3, 2, 2), 1e-2, "p0", state="dirichlet")
    # variable winds: nu at the median element Pe of nu = 1 (half the patches active); and high Pe
    for field in FIELDS:
        lo, hi, n = np.array(nonunit[0::2], float), np.array(nonunit[1::2], float), np.array((2, 3, 4))
        st = []
        H = FIELDS[field]
        for e in itertools.product(*(range(k) for k in n)):
            element_W(lambda p: H @ p, lo + np.array(e) * (hi - lo) / n, (hi - lo) / n, 1.0, "p0", stats=st)
        nu_mid = float(f"{0.8 * np.median([s[0] for s in st]):.3g}")   # median patch Pe = 1.25
        for nu in (nu_mid, 1e-3):
            for proj in ("p0", "p1"):
                case(field, nonunit, (2, 3, 4), nu, proj)
        case(field, unit, (2, 2, 2), nu_mid, "p0", state="dirichlet")
    for path, rows in ((sys.argv[1], pr_rows), (sys.argv[2], ref_rows)):
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("T15_OPERATOR_RESULT", "PASS" if ok else "FAIL", f"(projection {sum(r['verdict'] == 'PASS' for r in pr_rows)}/{len(pr_rows)}, "
          f"reference {sum(r['verdict'] == 'PASS' for r in ref_rows)}/{len(ref_rows)})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
