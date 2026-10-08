#!/usr/bin/env python3
"""T11D: Dirichlet trace DOF layout and R = M_Gamma on all six faces (dump based, serial)
-> ../../t11d_trace.csv.

For `--dimension 3 --dirichlet-solve --dirichlet-face3d F --state-bc dirichlet
--dump-operators` (the dump also writes the controlled set C):
  * C = support diag(M_Gamma) (after elimination) = the nodes of int(Gamma_c), count
    (2 n_a - 1)(2 n_b - 1), all on the face plane, NO perimeter node (edge policy u = 0
    on dGamma_c);
  * the state-constrained set D (rows of K with unit diagonal, zero mass) = every
    boundary node except C (y = 0 on the five other faces AND on dGamma_c);
    C and D are disjoint; I = interior of the volume;
  * R = M_Gamma|_C symmetric, SPD (min eigenvalue > 0);
  * R = [M1(a) (x) M1(b)] restricted to the interior face grid (independently
    assembled 1D Q2 masses), rel. max entrywise;
  * v^T R v = int_Gamma b^2 dS for the bubble b = (s_a - lo)(hi - s_a)(s_b - lo)(hi - s_b)
    (Q2 per axis, vanishes on dGamma_c -> reproduced exactly);
  * M_Gamma rows off C are zero (the trace mass of D is eliminated).
usage: t11d_trace.py <out.csv>"""
import csv, itertools, os, subprocess, sys
import numpy as np
import scipy.sparse as sps

X = os.environ.get("EX21_BIN", "/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21")
W = os.environ.get("T11D_WORK", "/tmp/t11d_trace"); os.makedirs(W + "/output", exist_ok=True)
ENV = {"HOME": os.environ["HOME"], "PATH": "/home/giacomo/software/petsc/arch-linux-c-openblas-opt/bin:/usr/bin:/bin",
       "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
FACES = {"zmin": (2, False), "ymin": (1, False), "xmax": (0, True), "ymax": (1, True), "xmin": (0, False), "zmax": (2, True)}
rows, ok = [], True


def coo(path):
    with open(path) as fh:
        fh.readline(); r, c = map(int, fh.readline().split()); fh.readline()
        d = np.loadtxt(fh, ndmin=2)
    if d.size == 0:
        return sps.csr_matrix((r, c))
    return sps.csr_matrix((d[:, 2], (d[:, 0].astype(int), d[:, 1].astype(int))), shape=(r, c))


def mass1d(lo, hi, n):
    h = (hi - lo) / n
    m = h / 30.0 * np.array([[4, 2, -1], [2, 16, 2], [-1, 2, 4]], float)
    M = np.zeros((2 * n + 1, 2 * n + 1))
    for e in range(n):
        idx = [2 * e, 2 * e + 1, 2 * e + 2]
        M[np.ix_(idx, idx)] += m
    return M


def rec(case, check, value, ref, good):
    global ok
    ok &= bool(good)
    rows.append({"case": case, "check": check, "value": value, "reference": ref, "verdict": "PASS" if good else "FAIL"})
    print(f"T11D_TRACE {case:<24s} {check:<56s} {str(value):<24s} ref {str(ref):<20s} {'PASS' if good else 'FAIL'}")


def case(face, dom, n):
    unit = dom == (0, 1, 0, 1, 0, 1)
    tag = f"{face}_{'x'.join(map(str, n))}_{'unit' if unit else 'nonunit'}"
    prefix = f"{W}/{tag}"
    lo, hi = np.array(dom[0::2], float), np.array(dom[1::2], float)
    mid = lambda k, f: repr(float(lo[k] + f * (hi[k] - lo[k])))
    args = [X, "--dimension", "3", "--dirichlet-solve", "--dirichlet-face3d", face, "--state-bc", "dirichlet",
            "--nx", str(n[0]), "--ny", str(n[1]), "--nz", str(n[2]), "--steps", "2", "--no-paraview",
            "--convection", "on", "--vx", "1", "--vy", "0.5", "--vz", "-0.75",
            "--tracking-box3d", mid(0, .25), mid(0, .75), mid(1, .25), mid(1, .75), mid(2, .25), mid(2, .75),
            "--dump-operators", prefix]
    if not unit:
        args += ["--domain3d"] + [repr(v) for v in dom]
    r = subprocess.run(args, cwd=W, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if r.returncode != 0:
        raise RuntimeError(f"{tag}: exit {r.returncode}: {r.stdout[-1500:]} {r.stderr[-1500:]} {args}")
    MG, K, M = coo(prefix + "_M_Gamma.coo"), coo(prefix + "_K.coo"), coo(prefix + "_M.coo")
    Lf, Mf = coo(prefix + "_D_Lfull.coo"), coo(prefix + "_D_Mfull.coo")
    with open(prefix + "_D_C.txt") as fh:
        fh.readline(); fh.readline()
        Cl = np.array(sorted(int(v) for v in fh.read().split()), int)
    xyz = np.loadtxt(prefix + "_coords.txt", ndmin=2)
    ax, at_max = FACES[face]
    a, b = [k for k in range(3) if k != ax]
    eps = 1e-12
    on = lambda k, v: np.isclose(xyz[:, k], v, atol=eps)
    plane = on(ax, hi[ax] if at_max else lo[ax])
    inner = plane & ~on(a, lo[a]) & ~on(a, hi[a]) & ~on(b, lo[b]) & ~on(b, hi[b])
    boundary = np.zeros(len(xyz), bool)
    for k in range(3):
        boundary |= on(k, lo[k]) | on(k, hi[k])
    support = np.where(MG.diagonal() != 0)[0]
    expect = (2 * n[a] - 1) * (2 * n[b] - 1)
    rec(tag, "|C| = (2n_a-1)(2n_b-1)", len(Cl), expect, len(Cl) == expect)
    rec(tag, "C = support diag(M_Gamma)", "eq" if np.array_equal(Cl, support) else "differ", "eq", np.array_equal(Cl, support))
    rec(tag, "C on the face plane", int(plane[Cl].sum()), len(Cl), bool(plane[Cl].all()))
    rec(tag, "C excludes the perimeter dGamma_c (edge policy)", int((plane & ~inner)[Cl].sum()), 0, int((plane & ~inner)[Cl].sum()) == 0)
    rec(tag, "C = int(Gamma_c) exactly", "eq" if np.array_equal(Cl, np.where(inner)[0]) else "differ", "eq",
        np.array_equal(Cl, np.where(inner)[0]))
    elim = lambda KK, MM: np.where(np.isclose(KK.diagonal(), 1.0) & np.isclose(abs(KK).sum(1).A1, 1.0)
                                   & (abs(MM).sum(1).A1 == 0))[0]
    Dset = elim(Lf, Mf)                       # unhatted: only D eliminated
    want = np.where(boundary & ~inner)[0]
    rec(tag, "D (unhatted M,L) = boundary \\ int(Gamma_c) (incl. dGamma_c)", f"{len(Dset)}", f"{len(want)}", np.array_equal(Dset, want))
    rec(tag, "C and D disjoint", len(np.intersect1d(Cl, Dset)), 0, len(np.intersect1d(Cl, Dset)) == 0)
    H = elim(K, M)                            # hatted (state operators): D u C eliminated
    rec(tag, "hatted K, M eliminated set = D u C (y = P_I y_I + Q_C eta u)", f"{len(H)}", f"{len(want) + len(Cl)}",
        np.array_equal(H, np.union1d(want, Cl)))
    off = np.setdiff1d(np.arange(MG.shape[0]), Cl)
    rec(tag, "M_Gamma rows off C zero", int((abs(MG[off]).sum(1).A1 > 0).sum()), 0, int((abs(MG[off]).sum(1).A1 > 0).sum()) == 0)
    R = MG[Cl][:, Cl].toarray()
    rec(tag, "R symmetric max|R - R^T|", f"{abs(R - R.T).max():.1e}", 0, abs(R - R.T).max() == 0)
    ev = np.linalg.eigvalsh(R)
    rec(tag, "R SPD: min eigenvalue", f"{ev.min():.3e}", "> 0", ev.min() > 0)
    ga = np.round((xyz[Cl, a] - lo[a]) / (hi[a] - lo[a]) * 2 * n[a]).astype(int)
    gb = np.round((xyz[Cl, b] - lo[b]) / (hi[b] - lo[b]) * 2 * n[b]).astype(int)
    Kr = np.kron(mass1d(lo[a], hi[a], n[a]), mass1d(lo[b], hi[b], n[b]))
    lin = ga * (2 * n[b] + 1) + gb
    Kf = Kr[np.ix_(lin, lin)]
    d = np.abs(R - Kf).max() / np.abs(Kf).max()
    rec(tag, "R = [M1(a) (x) M1(b)]|_int (rel. max entrywise)", f"{d:.2e}", "<= 1e-14", d <= 1e-14)
    bub = lambda pa, pb: (pa - lo[a]) * (hi[a] - pa) * (pb - lo[b]) * (hi[b] - pb)
    v = bub(xyz[Cl, a], xyz[Cl, b])
    exact = ((hi[a] - lo[a]) ** 5 / 30.0) * ((hi[b] - lo[b]) ** 5 / 30.0)
    got = v @ R @ v
    rec(tag, "v^T R v = int bubble^2 dS", f"{got:.15g}", f"{exact:.15g}", abs(got - exact) <= 1e-13 * exact)


U = (0, 1, 0, 1, 0, 1)
for f in FACES:
    case(f, U, (3, 3, 3))
for f in ("zmin", "xmax", "ymax"):
    case(f, U, (2, 3, 4))
for f in ("zmin", "xmin", "ymax", "zmax"):
    case(f, (0, 2, -1, 1, 0, .5), (4, 4, 2))
with open(sys.argv[1], "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("T11D_TRACE_RESULT", "PASS" if ok else "FAIL", f"({sum(r['verdict'] == 'PASS' for r in rows)}/{len(rows)})")
sys.exit(0 if ok else 1)
