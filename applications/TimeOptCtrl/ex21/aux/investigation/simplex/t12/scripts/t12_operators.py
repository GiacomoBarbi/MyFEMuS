#!/usr/bin/env python3
"""T12: geometry / DOF and M, K, C operator validation on QUAD9, P2 TRI, HEX27, P2 TET (dump based,
serial) -> ../../t12_geometry.csv and ../../t12_operators.csv.

Each case dumps (--dump-operators with an explicit --mesh-type) the volume M and L = nu K (+ C)
BEFORE the homogeneous Dirichlet elimination (_M_full, _L_full), the eliminated operators and the
coordinates; heat and wind runs are compared to isolate C = L(w) - L(0).

GEOMETRY  elements (EX21_MESH) = n_x n_z (quad), 2 n_x n_z (tri), n_x n_y n_z (hex), 6 n_x n_y n_z (tet);
          P2/Q2 DOFs = the lattice prod(2 n_a + 1) (every P2 node is a lattice point and vice versa);
          FEMuS type-2 node count = lattice (+ n_el bubbles for TRI7; + faces + n_el for TET15:
          faces = (4 n_el + 2 (n_x n_y + n_x n_z + n_y n_z) * 2) / 2);  coordinates within the box,
          on the lattice, no duplicate; constrained DOFs = boundary lattice points
          (prod(2n+1) - prod(2n-1)), free = interior; |Omega| = 1^T M_full 1.
OPERATORS M_full symmetric, M_FF SPD; v^T M_full w = int f g for f, g in {1, x_a, x_a x_b} (exact);
          L_heat symmetric, L 1 = 0, v^T L w = nu int grad f . grad g (exact for quadratics),
          L_FF SPD;  C: orientation C I(x_a) = w_a M 1 (the transposed operator must FAIL),
          v^T C g = int v (w . grad g) exact; free-subspace skew (C + C^T)_FF = 0 for div w = 0 and
          zero boundary trace; NEG: (C - C^T)_FF and the transpose orientation are O(1).
usage: t12_operators.py"""
import csv, itertools, os, re, subprocess, sys
import numpy as np
import scipy.sparse as sps

X = os.environ.get("EX21_BIN", "/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21")
W = os.environ.get("T12_WORK", "/tmp/t12_ops"); os.makedirs(W + "/output", exist_ok=True)
D = os.path.dirname(os.path.abspath(__file__)) + "/.."
ENV = {"HOME": os.environ["HOME"], "PATH": "/home/giacomo/software/petsc/arch-linux-c-openblas-opt/bin:/usr/bin:/bin",
       "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
geo_rows, op_rows, ok = [], [], True
NU = 0.7


def coo(path):
    with open(path) as fh:
        fh.readline(); r, c = map(int, fh.readline().split()); fh.readline()
        d = np.loadtxt(fh, ndmin=2)
    return sps.csr_matrix((d[:, 2], (d[:, 0].astype(int), d[:, 1].astype(int))), shape=(r, c))


def rec(rows, case, check, value, ref, good):
    global ok
    ok &= bool(good)
    rows.append({"case": case, "check": check, "value": value, "reference": ref, "verdict": "PASS" if good else "FAIL"})
    print(f"T12 {case:<30s} {check:<58s} {str(value):<24s} ref {str(ref):<22s} {'PASS' if good else 'FAIL'}", flush=True)


def dump(mtype, n, dom, wind, tag):
    dim = 3 if mtype in ("hex", "tet") else 2
    # the 2D --domain must precede --control-full-domain (option order decides the region)
    pre = [] if dim == 3 else ["--domain"] + [repr(v) for v in dom]
    args = [X, "--mesh-type", mtype] + pre + ["--distributed-solve", "--control-full-domain", "--state-bc", "dirichlet",
            "--steps", "2", "--pc-schur-sweep", "--no-paraview", "--nu", repr(NU), "--dump-operators", f"{W}/{tag}"]
    if dim == 3:
        args += ["--dimension", "3", "--nx", str(n[0]), "--ny", str(n[1]), "--nz", str(n[2]),
                 "--domain3d"] + [repr(v) for v in dom] + ["--tracking-box3d"] + [repr(v) for v in dom]
    else:
        args += ["--nx", str(n[0]), "--nz", str(n[1]), "--tracking-box"] + [repr(v) for v in dom]
    args += ["--convection", "off"] if wind is None else ["--convection", "on", "--vx", repr(wind[0]), "--vy", repr(wind[1])] + \
            (["--vz", repr(wind[2])] if dim == 3 else [])
    r = subprocess.run(args, cwd=W, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    if r.returncode != 0:
        raise RuntimeError(f"{tag}: exit {r.returncode}: {r.stdout[-1500:]}")
    m = re.search(r"EX21_MESH type=(\S+) dim=(\d) elements=(\d+) nodes_biquadratic=(\d+) dofs_type1=(\d+) dofs_type2=(\d+)", r.stdout)
    return {k: int(v) for k, v in zip(("elements", "nodes", "d1", "d2"), m.groups()[2:])}


def case(mtype, n, dom, winds):
    dim = len(n)
    tag = f"{mtype}_{'x'.join(map(str, n))}_{'unit' if all(dom[2*a] == 0 and dom[2*a+1] == 1 for a in range(dim)) else 'nonunit'}"
    lo, hi = np.array(dom[0::2], float), np.array(dom[1::2], float)
    info = dump(mtype, n, dom, None, tag + "_heat")
    ne = int(np.prod(n)) * {"quad": 1, "tri": 2, "hex": 1, "tet": 6}[mtype]
    lattice = int(np.prod([2 * k + 1 for k in n])); inner = int(np.prod([2 * k - 1 for k in n]))
    rec(geo_rows, tag, "elements", info["elements"], ne, info["elements"] == ne)
    rec(geo_rows, tag, "P2/Q2 DOFs = lattice prod(2n+1)", info["d1"] if mtype in ("tri", "tet") else info["d2"], lattice,
        (info["d1"] if mtype in ("tri", "tet") else info["d2"]) == lattice)
    if mtype == "tri":
        rec(geo_rows, tag, "FEMuS type-2 nodes = lattice + n_el (TRI7 bubble)", info["d2"], lattice + ne, info["d2"] == lattice + ne)
    if mtype == "tet":
        bfaces = 2 * 2 * (n[0] * n[1] + n[0] * n[2] + n[1] * n[2])
        faces = (4 * ne + bfaces) // 2
        rec(geo_rows, tag, "FEMuS type-2 nodes = lattice + faces + n_el (TET15)", info["d2"], lattice + faces + ne,
            info["d2"] == lattice + faces + ne)
    p = f"{W}/{tag}_heat"
    xyz = np.loadtxt(p + "_coords.txt", ndmin=2)[:, :dim]
    Mf, Lf, Me, Ke = coo(p + "_M_full.coo"), coo(p + "_L_full.coo"), coo(p + "_M.coo"), coo(p + "_K.coo")
    N = Mf.shape[0]
    rec(geo_rows, tag, "dump size = DOFs", N, lattice, N == lattice)
    inside = bool(np.all(xyz >= lo - 1e-12) and np.all(xyz <= hi + 1e-12))
    rec(geo_rows, tag, "coordinates within the box", inside, True, inside)
    g = (xyz - lo) / (hi - lo) * (2 * np.array(n))
    on_lat = np.abs(g - np.round(g)).max()
    rec(geo_rows, tag, "coordinates on the lattice (max dev)", f"{on_lat:.1e}", "<= 1e-10", on_lat <= 1e-10)
    nuniq = len({tuple(r) for r in np.round(g).astype(int)})
    rec(geo_rows, tag, "no duplicated DOF (unique lattice points)", nuniq, N, nuniq == N)
    bnd = np.any(np.isclose(xyz, lo, atol=1e-12) | np.isclose(xyz, hi, atol=1e-12), axis=1)
    cons = np.isclose(Ke.diagonal(), 1.0) & np.isclose(abs(Ke).sum(1).A1, 1.0) & (abs(Me).sum(1).A1 == 0)
    rec(geo_rows, tag, "constrained DOFs = boundary lattice points", int(cons.sum()), lattice - inner,
        np.array_equal(cons, bnd) and int(cons.sum()) == lattice - inner)
    rec(geo_rows, tag, "free DOFs = interior lattice points", int((~cons).sum()), inner, int((~cons).sum()) == inner)
    one = np.ones(N)
    vol = float(np.prod(hi - lo))
    rec(geo_rows, tag, "|Omega| = 1^T M_full 1", f"{one @ (Mf @ one):.15g}", f"{vol:.15g}", abs(one @ (Mf @ one) - vol) <= 1e-13 * vol)
    # ---- operators
    F = ~cons
    def mono(e):   # exact int over the box of prod x_a^e_a
        return float(np.prod([(hi[a] ** (e[a] + 1) - lo[a] ** (e[a] + 1)) / (e[a] + 1) for a in range(dim)]))
    rec(op_rows, tag, "M_full symmetric max|M-M^T|/max|M|", f"{abs(Mf - Mf.T).max() / abs(Mf).max():.1e}", "<= 1e-15",
        abs(Mf - Mf.T).max() / abs(Mf).max() <= 1e-15)
    MF = Mf[F][:, F].toarray(); LF = Lf[F][:, F].toarray()
    eM = np.linalg.eigvalsh(MF)[0]
    rec(op_rows, tag, "M_FF SPD: lambda_min", f"{eM:.3e}", "> 0", eM > 0)
    exps = [tuple(int(i == a) for i in range(dim)) for a in range(dim)] + [tuple(int(i == a) + int(i == b) for i in range(dim)) for a in range(dim) for b in range(a, dim)]
    funcs = [tuple([0] * dim)] + exps
    worst = 0.0
    for e1, e2 in itertools.product(funcs, funcs):
        v = np.prod(xyz ** np.array(e1), axis=1); w = np.prod(xyz ** np.array(e2), axis=1)
        ex = mono(tuple(a + b for a, b in zip(e1, e2)))
        worst = max(worst, abs(v @ (Mf @ w) - ex) / max(abs(ex), 1.0))   # some moments vanish by symmetry
    rec(op_rows, tag, "v^T M w = int f g, f,g in {1,x_a,x_a x_b} (max err/max(1,|int|))", f"{worst:.1e}", "<= 1e-13", worst <= 1e-13)
    rec(op_rows, tag, "L_heat symmetric", f"{abs(Lf - Lf.T).max() / abs(Lf).max():.1e}", "<= 1e-14", abs(Lf - Lf.T).max() / abs(Lf).max() <= 1e-14)
    rec(op_rows, tag, "L_heat 1 = 0 (max|L 1|/max|L|)", f"{abs(Lf @ one).max() / abs(Lf).max():.1e}", "<= 1e-13",
        abs(Lf @ one).max() / abs(Lf).max() <= 1e-13)
    def grad(e, a):   # gradient component a of monomial e, as (coef, exps)
        if e[a] == 0: return 0.0, e
        return float(e[a]), tuple(e[i] - (i == a) for i in range(dim))
    worst = 0.0
    for e1, e2 in itertools.product(funcs, funcs):
        v = np.prod(xyz ** np.array(e1), axis=1); w = np.prod(xyz ** np.array(e2), axis=1)
        ex = 0.0
        for a in range(dim):
            c1, g1 = grad(e1, a); c2, g2 = grad(e2, a)
            if c1 and c2: ex += c1 * c2 * mono(tuple(x + y for x, y in zip(g1, g2)))
        ex *= NU
        got = v @ (Lf @ w)
        worst = max(worst, abs(got - ex) / max(abs(ex), 1.0))
    rec(op_rows, tag, "v^T L w = nu int grad f.grad g (max err/max(1,|int|))", f"{worst:.1e}", "<= 1e-12", worst <= 1e-12)
    eL = np.linalg.eigvalsh(0.5 * (LF + LF.T))[0]
    rec(op_rows, tag, "L_FF SPD (heat): lambda_min", f"{eL:.3e}", "> 0", eL > 0)
    for wi, wind in enumerate(winds):
        wt = f"{tag}_w{wi}"
        dump(mtype, n, dom, wind, wt)
        Lw = coo(f"{W}/{wt}_L_full.coo")
        C = (Lw - Lf).tocsr()
        Cmax = abs(C).max(); M1 = Mf @ one
        o = max(np.abs(C @ xyz[:, a] - wind[a] * M1).max() for a in range(dim)) / np.abs(M1).max()
        ot = max(np.abs(C.T @ xyz[:, a] - wind[a] * M1).max() for a in range(dim)) / np.abs(M1).max()
        rec(op_rows, wt, f"C orientation: C I(x_a) = w_a M 1 (w={wind})", f"{o:.1e}", "<= 1e-13", o <= 1e-13)
        rec(op_rows, wt, "NEG transposed orientation C^T I(x_a) = w_a M 1: fails", f"{ot:.1e}", "> 1e-3", ot > 1e-3)
        worst = 0.0
        for e1, e2 in itertools.product(funcs, funcs):
            v = np.prod(xyz ** np.array(e1), axis=1); w2 = np.prod(xyz ** np.array(e2), axis=1)
            ex = 0.0
            for a in range(dim):
                c2, g2 = grad(e2, a)
                if c2: ex += wind[a] * c2 * mono(tuple(x + y for x, y in zip(e1, g2)))
            worst = max(worst, abs(v @ (C @ w2) - ex) / max(abs(ex), 1.0))
        rec(op_rows, wt, "v^T C g = int v (w.grad g), quadratics (max err)", f"{worst:.1e}", "<= 1e-12", worst <= 1e-12)
        CF = C[F][:, F]
        sk = abs(CF + CF.T).max() / Cmax if (CF + CF.T).nnz else 0.0
        ns = abs(CF - CF.T).max() / Cmax
        # C = L(w) - L(0) is a difference of two assembled operators ~10x larger than C: round-off
        # ~1e-13 relative to max|C| (bar 1e-12, >= 10 orders below the NEG value)
        rec(op_rows, wt, "free-subspace skew max|(C+C^T)_FF|/max|C|", f"{sk:.1e}", "<= 1e-12", sk <= 1e-12)
        rec(op_rows, wt, "NEG (C-C^T)_FF / max|C| (not symmetric)", f"{ns:.1e}", "> 1e-2", ns > 1e-2)
        full_sk = abs(C + C.T).max() / Cmax
        rec(op_rows, wt, "full C + C^T != 0 (boundary terms, informational)", f"{full_sk:.1e}", "> 0", full_sk > 0)


W2 = [(1.0, 0.5), (-2.0, 1.0)]
W3 = [(1.0, 0.5, -0.75), (-2.0, 1.0, 0.5)]
for mt in ("quad", "tri"):
    for n in ((2, 2), (4, 4), (3, 5)):
        case(mt, n, (0, 1, 0, 1), W2)
    case(mt, (4, 2), (0, 2, -1, 0.5), W2)
for mt in ("hex", "tet"):
    case(mt, (2, 2, 2), (0, 1, 0, 1, 0, 1), W3)
    case(mt, (2, 3, 4), (0, 1, 0, 1, 0, 1), W3)
    case(mt, (3, 2, 2), (0, 2, -1, 1, 0, 0.5), W3)
for rows, f in ((geo_rows, "t12_geometry.csv"), (op_rows, "t12_operators.csv")):
    with open(f"{D}/../{f}", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("T12_OPERATORS_RESULT", "PASS" if ok else "FAIL", f"geometry {sum(r['verdict'] == 'PASS' for r in geo_rows)}/{len(geo_rows)}",
      f"operators {sum(r['verdict'] == 'PASS' for r in op_rows)}/{len(op_rows)}")
sys.exit(0 if ok else 1)
