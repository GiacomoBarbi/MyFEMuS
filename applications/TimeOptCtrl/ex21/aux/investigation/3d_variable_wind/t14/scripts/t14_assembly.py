#!/usr/bin/env python3
"""T14: independent quadrature-point reference assembly of the variable-wind operators (dump based, serial).

Python assembles, on the structured HEX27 lattice of the box, with tensor-product 1D Q2 Lagrange bases and a
4-point Gauss rule per axis (exact for the affine presets: integrand degree <= 5 per axis):
  C_ref[i,j] = int phi_i (w(x) . grad phi_j)      w evaluated at the quadrature points
  B_ref[i,j] = sum_faces int (w(x).n) phi_i phi_j  (face quadrature points)
  D_ref[i,j] = int (div w) phi_i phi_j
  C_cent     = C with w FROZEN at the element centroid (negative control)
and compares with the binary: C_dump = K(--convection on --wind W) - K(--convection off) (same mesh), natural
state BC (no elimination) and homogeneous Dirichlet state (eliminated: the free-free block).  Checks:
  C_dump = C_ref (rel. max entrywise); NEG C_dump vs C_cent differs;
  C_dump + C_dump^T = B_ref - D_ref (natural, before elimination);  D_ref = 0 (div-free) / int div = |Omega| div;
  (C + C^T)_FF = -(D_ref)_FF (Dirichlet state, after elimination);  NEG the div-free identity on the generic field.
Unit cube (2,2,2) and the non-unit anisotropic box (0,2)x(-1,1)x(0,1/2) at (3,2,2) / (2,3,4).
usage: t14_assembly.py <identity.csv> <assembly.csv>"""
import csv, itertools, os, subprocess, sys
import numpy as np
import scipy.sparse as sps

X = os.environ.get("EX21_BIN", "/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21")
W = os.environ.get("T14_WORK", "/tmp/t14_asm"); os.makedirs(W + "/output", exist_ok=True)
ENV = {"HOME": os.environ["HOME"], "PATH": "/home/giacomo/software/petsc/arch-linux-c-openblas-opt/bin:/usr/bin:/bin",
       "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
HARM = np.array([[0, 1, -0.75], [1, 0, 0.5], [-0.75, 0.5, 0]], float)
FIELDS = {"harmonic-3d": HARM, "affine-div-3d": HARM + np.diag([0.5, 0.25, 0.25])}
GL, GW = np.polynomial.legendre.leggauss(4)
id_rows, as_rows, ok = [], [], True


def coo(path):
    with open(path) as fh:
        fh.readline(); r, c = map(int, fh.readline().split()); fh.readline()
        d = np.loadtxt(fh, ndmin=2)
    return sps.csr_matrix((d[:, 2], (d[:, 0].astype(int), d[:, 1].astype(int))), shape=(r, c)) if d.size else sps.csr_matrix((r, c))


def rec(table, case, check, value, ref, good):
    global ok
    ok &= bool(good)
    table.append({"case": case, "check": check, "value": value, "reference": ref, "verdict": "PASS" if good else "FAIL"})
    print(f"T14_ASM {case:<40s} {check:<62s} {str(value):<12s} ref {str(ref):<12s} {'PASS' if good else 'FAIL'}", flush=True)


def lag(xi):   # 1D Q2 on [0,1], nodes 0, 1/2, 1: values and derivatives
    return (np.array([2 * (xi - .5) * (xi - 1), -4 * xi * (xi - 1), 2 * xi * (xi - .5)]),
            np.array([4 * xi - 3, -8 * xi + 4, 4 * xi - 1]))


def reference(H, lo, hi, n, gidx):
    """C_ref, C_cent, B_ref, D_ref on the global numbering gidx[(i,j,k)] (lattice indices)"""
    N = len(gidx)
    h = (hi - lo) / n
    w_of = lambda p: H @ p
    div = np.trace(H)
    C, Cc, D, Bm = (sps.lil_matrix((N, N)) for _ in range(4))
    q = (GL + 1) / 2; qw = GW / 2
    V = [lag(s) for s in q]
    for e in itertools.product(range(n[0]), range(n[1]), range(n[2])):
        nodes = [gidx[(2 * e[0] + a, 2 * e[1] + b, 2 * e[2] + c)] for a in range(3) for b in range(3) for c in range(3)]
        x0 = lo + np.array(e) * h
        cen = x0 + h / 2
        Cl, Ccl, Dl = np.zeros((27, 27)), np.zeros((27, 27)), np.zeros((27, 27))
        for (ia, va), (ib, vb), (ic, vc) in itertools.product(enumerate(V), repeat=3):
            phi = np.einsum("a,b,c->abc", va[0], vb[0], vc[0]).ravel()
            g = np.stack([np.einsum("a,b,c->abc", va[1], vb[0], vc[0]).ravel() / h[0],
                          np.einsum("a,b,c->abc", va[0], vb[1], vc[0]).ravel() / h[1],
                          np.einsum("a,b,c->abc", va[0], vb[0], vc[1]).ravel() / h[2]], 1)
            p = x0 + np.array([q[ia], q[ib], q[ic]]) * h
            wt = qw[ia] * qw[ib] * qw[ic] * np.prod(h)
            Cl += wt * np.outer(phi, g @ w_of(p))
            Ccl += wt * np.outer(phi, g @ w_of(cen))
            Dl += wt * div * np.outer(phi, phi)
        idx = np.ix_(nodes, nodes)
        C[idx] = C[idx].toarray() + Cl; Cc[idx] = Cc[idx].toarray() + Ccl; D[idx] = D[idx].toarray() + Dl
        # boundary faces of this element
        for ax in range(3):
            for side in (0, 1):
                if e[ax] != (0 if side == 0 else n[ax] - 1):
                    continue
                normal = np.zeros(3); normal[ax] = 1.0 if side else -1.0
                tang = [k for k in range(3) if k != ax]
                loc = [(a, b, c) for a in range(3) for b in range(3) for c in range(3) if (a, b, c)[ax] == 2 * side]
                fn = [gidx[(2 * e[0] + a, 2 * e[1] + b, 2 * e[2] + c)] for a, b, c in loc]
                Bl = np.zeros((9, 9))
                for (i1, v1), (i2, v2) in itertools.product(enumerate(V), repeat=2):
                    p = x0.copy(); p[ax] = x0[ax] + side * h[ax]
                    p[tang[0]] = x0[tang[0]] + q[i1] * h[tang[0]]; p[tang[1]] = x0[tang[1]] + q[i2] * h[tang[1]]
                    vals = []
                    for (a, b, c) in loc:
                        ab = [(a, b, c)[k] for k in tang]
                        vals.append(v1[0][ab[0]] * v2[0][ab[1]])
                    vals = np.array(vals)
                    wt = qw[i1] * qw[i2] * h[tang[0]] * h[tang[1]]
                    Bl += wt * (w_of(p) @ normal) * np.outer(vals, vals)
                fi = np.ix_(fn, fn)
                Bm[fi] = Bm[fi].toarray() + Bl
    return C.tocsr(), Cc.tocsr(), Bm.tocsr(), D.tocsr()


def run(args, prefix):
    r = subprocess.run([X] + args + ["--dump-operators", prefix, "--no-paraview"], cwd=W, env=ENV,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    if r.returncode != 0:
        raise RuntimeError(f"{prefix}: {r.stdout[-1500:]}")


def rel(A, B):
    d = (A - B)
    return (abs(d).max() if d.nnz else 0.0) / max(abs(B).max(), 1e-300)


def case(field, dom, n, state):
    unit = dom == (0, 1, 0, 1, 0, 1)
    tag = f"{field}_{'x'.join(map(str, n))}{'' if unit else '_nonunit'}_{state}"
    lo, hi = np.array(dom[0::2], float), np.array(dom[1::2], float)
    base = ["--dimension", "3", "--distributed-solve", "--control-full-domain", "--tracking-box3d"] + [repr(v) for v in dom] + \
           ["--domain3d"] + [repr(v) for v in dom] + ["--state-bc", state, "--nx", str(n[0]), "--ny", str(n[1]), "--nz", str(n[2]),
                                                    "--steps", "2", "--nu", "0.25", "--pc-schur-sweep"]
    run(base + ["--convection", "on", "--wind", field], f"{W}/{tag}_on")
    run(base + ["--convection", "off"], f"{W}/{tag}_off")
    Ld, L0 = coo(f"{W}/{tag}_on_K.coo"), coo(f"{W}/{tag}_off_K.coo")
    Cd = (Ld - L0).tocsr()
    xyz = np.loadtxt(f"{W}/{tag}_on_coords.txt", ndmin=2)
    lat = np.round((xyz - lo) / (hi - lo) * 2 * np.array(n)).astype(int)
    gidx = {tuple(l): i for i, l in enumerate(lat)}
    Cr, Cc, Br, Dr = reference(FIELDS[field], lo, hi, n, gidx)
    divw = np.trace(FIELDS[field])
    one = np.ones(len(xyz))
    if state == "natural":
        d = rel(Cd, Cr)
        rec(as_rows, tag, "C_dump = C_ref (quadrature-point wind, rel. max entrywise)", f"{d:.1e}", "<=1e-13", d <= 1e-13)
        dc = rel(Cd, Cc)
        rec(as_rows, tag, "NEG centroid-frozen wind C_cent: differs", f"{dc:.1e}", "> 1e-4", dc > 1e-4)
        S = (Cd + Cd.T).tocsr()
        d = rel(S, Br - Dr)
        rec(id_rows, tag, "C + C^T = B_w - D_w (before elimination)", f"{d:.1e}", "<=1e-13", d <= 1e-13)
        vol = np.prod(hi - lo)
        rec(id_rows, tag, "1^T D_w 1 = |Omega| div w", f"{one @ (Dr @ one):.12g}", f"{vol * divw:.12g}", abs(one @ (Dr @ one) - vol * divw) <= 1e-12 * max(1, vol))
        # 1^T B_w 1 = int_dOmega w.n = int div w (Gauss)
        rec(id_rows, tag, "1^T B_w 1 = int_dOmega w.n = |Omega| div w", f"{one @ (Br @ one):.12g}", f"{vol * divw:.12g}", abs(one @ (Br @ one) - vol * divw) <= 1e-12 * max(1, vol))
        if divw != 0:
            dn = rel(S, Br)
            rec(id_rows, tag, "NEG div-free identity C + C^T = B_w on the generic field: fails", f"{dn:.1e}", "> 1e-4", dn > 1e-4)
        else:
            rec(id_rows, tag, "D_w = 0 (div-free)", f"{abs(Dr).max():.1e}", 0, abs(Dr).max() == 0)
    else:
        M = coo(f"{W}/{tag}_on_M.coo")
        F = np.where(abs(M).sum(1).A1 > 0)[0]   # free DOFs: nonzero eliminated-mass rows
        CdF, CrF = Cd[F][:, F], Cr[F][:, F]
        d = rel(CdF, CrF)
        rec(as_rows, tag, "C_FF (eliminated) = C_ref[F,F]", f"{d:.1e}", "<=1e-13", d <= 1e-13)
        S = (CdF + CdF.T).tocsr()
        DF = Dr[F][:, F]
        e = (abs(S + DF).max() if (S + DF).nnz else 0.0) / abs(CdF).max()
        rec(id_rows, tag, "(C + C^T)_FF = -(D_w)_FF (after elimination; rel. to max|C|)", f"{e:.1e}", "<=1e-13", e <= 1e-13)
        if divw != 0:
            neg = abs(S).max() / abs(CdF).max()
            rec(id_rows, tag, "NEG skew C_FF (div-free identity) on the generic field: fails", f"{neg:.1e}", "> 1e-4", neg > 1e-4)
        rec(id_rows, tag, "eliminated (constrained) DOFs = boundary nodes", len(xyz) - len(F),
            int(np.any((lat == 0) | (lat == 2 * np.array(n)), 1).sum()), len(xyz) - len(F) == int(np.any((lat == 0) | (lat == 2 * np.array(n)), 1).sum()))


def main():
    for field in FIELDS:
        for state in ("natural", "dirichlet"):
            case(field, (0, 1, 0, 1, 0, 1), (2, 2, 2), state)
            case(field, (0, 2, -1, 1, 0, 0.5), (3, 2, 2), state)
            case(field, (0, 2, -1, 1, 0, 0.5), (2, 3, 4), state)
    for path, rows in ((sys.argv[1], id_rows), (sys.argv[2], as_rows)):
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("T14_ASM_RESULT", "PASS" if ok else "FAIL", f"(identity {sum(r['verdict'] == 'PASS' for r in id_rows)}/{len(id_rows)}, "
          f"assembly {sum(r['verdict'] == 'PASS' for r in as_rows)}/{len(as_rows)})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
