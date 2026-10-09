#!/usr/bin/env python3
"""T13: independent explicit check of the multi-face Dirichlet control-to-state map B_D and of
B_D^T (dump based, serial) -> ../../t13_gates_dirichlet_bd.csv (merged into t13_gates_dirichlet.csv).

As t11d_bd.py, for a UNION of faces.  Python rebuilds from the dumped unhatted M, L only and the
GEOMETRIC controlled set C (nodes on >= 1 controlled face, on no uncontrolled face):
  BE: (B_D u)_n = P_I [A Q_C u_n - (M/dt) Q_C u_{n-1}],          A  = M/dt + L
  CN: (B_D u)_n = P_I [A+ Q_C u_n + A- Q_C u_{n-1}],  A+ = M/dt + L/2, A- = -M/dt + L/2
checks: dumped C = geometric C; dumped B_D = formula; dumped B_D^T = formula^T = (dumped B_D)^T;
<B_D u, p> = <u, B_D^T p>; G1D on the union: C + C^T = sum_{f in F} (w.n_f) M_f on I u C
(M_f = the single-face M_Gamma dumped by a separate one-face run).
negative controls (must differ by >> roundoff):
  * WRONG C, single-face edge policy: internal edges of the union constrained (u = 0 there)
  * WRONG C, face omitted: C of the union minus the nodes of one controlled face
  * forward-L transpose (L instead of L^T; wrong under convection)
  * CN coupling p_{n-1} instead of p_{n+1}
  * G1D with one face dropped from sum_f (w.n_f) M_f
usage: t13_bd.py <out.csv>"""
import csv, itertools, os, subprocess, sys
import numpy as np
import scipy.sparse as sps

X = os.environ.get("EX21_BIN", "/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21")
W = os.environ.get("T13_WORK", "/tmp/t13_bd"); os.makedirs(W + "/output", exist_ok=True)
ENV = {"HOME": os.environ["HOME"], "PATH": "/home/giacomo/software/petsc/arch-linux-c-openblas-opt/bin:/usr/bin:/bin",
       "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
FACES = {"zmin": (2, False), "ymin": (1, False), "xmax": (0, True), "ymax": (1, True), "xmin": (0, False), "zmax": (2, True)}
ORDER = ["zmin", "ymin", "xmax", "ymax", "xmin", "zmax"]
rows, ok = [], True
TOL = 1e-13


def coo(path):
    with open(path) as fh:
        fh.readline(); r, c = map(int, fh.readline().split()); fh.readline()
        d = np.loadtxt(fh, ndmin=2)
    if d.size == 0:
        return sps.csr_matrix((r, c))
    return sps.csr_matrix((d[:, 2], (d[:, 0].astype(int), d[:, 1].astype(int))), shape=(r, c))


def rec(case, check, value, ref, good):
    global ok
    ok &= bool(good)
    rows.append({"case": case, "check": check, "value": value, "reference": ref, "verdict": "PASS" if good else "FAIL"})
    print(f"T13_BD {case:<34s} {check:<62s} {str(value):<12s} ref {str(ref):<12s} {'PASS' if good else 'FAIL'}")


def rel(a, b):
    a, b = sps.csr_matrix(a), sps.csr_matrix(b)
    s = max(abs(b).max(), 1e-300)
    d = a - b
    return (abs(d).max() if d.nnz else 0.0) / s


def run(faces, scheme, wind, n, nt, T, prefix):
    args = [X, "--dimension", "3", "--dirichlet-solve", "--dirichlet-face3d", ",".join(faces), "--state-bc", "dirichlet",
            "--nx", str(n[0]), "--ny", str(n[1]), "--nz", str(n[2]), "--steps", str(nt), "--T", repr(T),
            "--time-scheme", scheme, "--no-paraview", "--tracking-box3d", "0.25", "0.75", "0.25", "0.75", "0.25", "0.75",
            "--dump-operators", prefix]
    args += ["--convection", "off"] if wind is None else ["--convection", "on", "--vx", repr(wind[0]), "--vy", repr(wind[1]), "--vz", repr(wind[2])]
    r = subprocess.run(args, cwd=W, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if r.returncode != 0:
        raise RuntimeError(f"{prefix}: exit {r.returncode}: {r.stdout[-600:]} {r.stderr[-600:]}")


def geometry(xyz, faces):
    on = {f: np.isclose(xyz[:, FACES[f][0]], 1.0 if FACES[f][1] else 0.0, atol=1e-12) for f in FACES}
    union, unctl = np.zeros(len(xyz), bool), np.zeros(len(xyz), bool)
    for f in FACES:
        if f in faces:
            union |= on[f]
        else:
            unctl |= on[f]
    return on, union, unctl


def bd_formula(M, L, C, Dmask, steps, dt, sch):
    N = M.shape[0]
    QC = sps.csr_matrix((np.ones(len(C)), (C, np.arange(len(C)))), shape=(N, len(C)))
    pI = np.ones(N); pI[C] = 0.0; pI[Dmask] = 0.0
    PI = sps.diags(pI)
    Ap, Am = (M / dt + L, -M / dt) if sch == "BE" else (M / dt + 0.5 * L, -M / dt + 0.5 * L)
    blocks = [[None] * steps for _ in range(steps)]
    for k in range(steps):
        blocks[k][k] = PI @ Ap @ QC
        if k > 0:
            blocks[k][k - 1] = PI @ Am @ QC
    B = sps.bmat(blocks).tocsr()
    # embed into the full (Nt N) x (Nt N) column layout used by the dump
    cols = np.concatenate([k * N + C for k in range(steps)])
    E = sps.csr_matrix((np.ones(len(cols)), (np.arange(len(cols)), cols)), shape=(len(cols), steps * N))
    return (B @ E).tocsr(), B, cols, PI, QC, Ap, Am


def case(faces, scheme, wind, n, nt=3, T=0.3):
    tag = f"{'+'.join(faces)}_{scheme}_{'heat' if wind is None else 'wind'}_{'x'.join(map(str, n))}"
    prefix = f"{W}/{tag}"
    run(faces, scheme, wind, n, nt, T, prefix)
    M, L = coo(prefix + "_D_Mfull.coo"), coo(prefix + "_D_Lfull.coo")
    BD, BDT = coo(prefix + "_BD.coo"), coo(prefix + "_BDT.coo")
    with open(prefix + "_D_C.txt") as fh:
        fh.readline(); hdr = fh.readline().split()
        Cl = np.array([int(v) for v in fh.read().split()], int)
    steps, dt, sch, tr = int(hdr[0]), float(hdr[1]), hdr[2], int(hdr[3])
    N = M.shape[0]
    xyz = np.loadtxt(prefix + "_coords.txt", ndmin=2)
    on, union, unctl = geometry(xyz, faces)
    Cgeo = np.where(union & ~unctl)[0]
    rec(tag, "dumped C = geometric C (union minus uncontrolled faces)", len(Cl), len(Cgeo), np.array_equal(np.sort(Cl), Cgeo))
    rec(tag, "ApplyBDT uses L^T iff convection", tr, int(wind is not None), tr == int(wind is not None))
    Dmask = (union | unctl) & ~np.isin(np.arange(N), Cgeo)
    Bfull, Bf, cols, PI, QC, Ap, Am = bd_formula(M, L, Cgeo, Dmask, steps, dt, sch)
    rec(tag, "dumped B_D = formula B_D (rel.)", f"{rel(BD, Bfull):.1e}", f"<={TOL:.0e}", rel(BD, Bfull) <= TOL)
    rec(tag, "dumped B_D^T = (formula B_D)^T (rel.)", f"{rel(BDT, Bfull.T):.1e}", f"<={TOL:.0e}", rel(BDT, Bfull.T) <= TOL)
    rec(tag, "dumped B_D^T = (dumped B_D)^T (rel.)", f"{rel(BDT, BD.T):.1e}", f"<={TOL:.0e}", rel(BDT, BD.T) <= TOL)
    rng = np.random.default_rng(7)
    u, p = np.zeros(steps * N), rng.standard_normal(steps * N)
    u[cols] = rng.standard_normal(len(cols))
    lhs, rhs = p @ (BD @ u), u @ (BDT @ p)
    d = abs(lhs - rhs) / max(abs(lhs), abs(rhs))
    rec(tag, "<B_D u, p> = <u, B_D^T p> (rel.)", f"{d:.1e}", f"<={TOL:.0e}", d <= TOL)
    # ---- negatives: wrong C
    internal = np.zeros(N, bool)
    for f, g in itertools.combinations(faces, 2):
        internal |= on[f] & on[g]
    internal &= ~unctl
    if internal.any():
        Cw = np.setdiff1d(Cgeo, np.where(internal)[0])
        Bw = bd_formula(M, L, Cw, Dmask | internal, steps, dt, sch)[0]
        neg = rel(BD, Bw)
        rec(tag, f"NEG wrong C: internal edges constrained ({int(internal.sum())} nodes): detected", f"{neg:.1e}", "> 1e-6", neg > 1e-6)
    if len(faces) > 1:
        f0 = faces[-1]
        Cw = np.setdiff1d(Cgeo, np.where(on[f0])[0])
        Bw = bd_formula(M, L, Cw, Dmask | on[f0], steps, dt, sch)[0]
        neg = rel(BD, Bw)
        rec(tag, f"NEG wrong C: face {f0} omitted: detected", f"{neg:.1e}", "> 1e-6", neg > 1e-6)
    # ---- negatives: forward-L transpose, CN coupling
    fwd = [[None] * steps for _ in range(steps)]
    Fp, Fm = (M / dt + L, -M / dt) if sch == "BE" else (M / dt + 0.5 * L, -M / dt + 0.5 * L)
    for k in range(steps):
        fwd[k][k] = QC.T @ Fp @ PI
        if k + 1 < steps:
            fwd[k][k + 1] = QC.T @ Fm @ PI
    neg = rel(BDT[cols, :], sps.bmat(fwd))
    if wind is None:
        rec(tag, "NEG forward-L B_D^T: equals (L symmetric, heat)", f"{neg:.1e}", f"<={TOL:.0e}", neg <= TOL)
    else:
        rec(tag, "NEG forward-L B_D^T (L instead of L^T): detected", f"{neg:.1e}", "> 1e-6", neg > 1e-6)
    if sch == "CN":
        wr = [[None] * steps for _ in range(steps)]
        for k in range(steps):
            wr[k][k] = (PI @ Ap @ QC).T
            if k > 0:
                wr[k][k - 1] = (PI @ Am @ QC).T
        neg = rel(BDT[cols, :], sps.bmat(wr))
        rec(tag, "NEG CN coupling p_{n-1} instead of p_{n+1}: detected", f"{neg:.1e}", "> 1e-6", neg > 1e-6)


def g1d(faces, wind, n=(2, 2, 2)):
    """C + C^T = sum_{f in F} (w.n_f) M_f on I u C (div w = 0; C nodes see no uncontrolled face)."""
    tag = f"G1D_{'+'.join(faces)}_w({','.join(f'{v:g}' for v in wind)})"
    mats = []
    for w in (None, wind):
        prefix = f"{W}/g1d_{'+'.join(faces)}_{'heat' if w is None else 'wind'}"
        run(faces, "be", w, n, 2, 0.2, prefix)
        mats.append(coo(prefix + "_D_Lfull.coo"))
    L0, L1 = mats
    Mf = {}
    for f in faces:
        prefix = f"{W}/g1d_single_{f}"
        run([f], "be", None, n, 2, 0.2, prefix)
        Mf[f] = coo(prefix + "_M_Gamma_face.coo")
    wn = {f: (1.0 if FACES[f][1] else -1.0) * wind[FACES[f][0]] for f in faces}
    Cm = (L1 - L0).tocsr()
    free = np.where(~(np.isclose(L0.diagonal(), 1.0) & np.isclose(abs(L0).sum(1).A1, 1.0)))[0]
    S = (Cm + Cm.T)[free][:, free]
    ref = sum(wn[f] * Mf[f] for f in faces)[free][:, free]
    d = abs(S - ref).max() / max(abs(Cm).max(), 1e-300)
    rec(tag, "C + C^T = sum_f (w.n_f) M_f on I u C, rel. to max|C|", f"{d:.1e}", f"<={TOL:.0e}", d <= TOL)
    for f in faces:
        if wn[f] != 0.0:
            refw = sum(wn[g] * Mf[g] for g in faces if g != f)[free][:, free] if len(faces) > 1 else 0 * S
            neg = abs(S - refw).max() / max(abs(Cm).max(), 1e-300)
            rec(tag, f"NEG face {f} (w.n = {wn[f]:g}) dropped from the sum: detected", f"{neg:.1e}", "> 1e-6", neg > 1e-6)
            break


A = (1.0, 0.5, -0.75)
B = (-2.0, 1.0, 0.5)
for fs, w in ((["zmin", "zmax"], A), (["zmin", "xmin"], A), (["ymin", "xmax", "zmax"], B), (["zmin", "xmax", "xmin"], A), (ORDER, B)):
    g1d(sorted(fs, key=ORDER.index), w)
for fs in (["zmin", "zmax"], ["zmin", "xmin"], ["zmin", "ymin", "xmin"]):
    for scheme in ("be", "cn"):
        for wind in (None, A):
            case(sorted(fs, key=ORDER.index), scheme, wind, (2, 2, 2))
for fs in (["xmax", "ymax"], ["zmin", "xmax", "xmin"], ORDER):
    case(sorted(fs, key=ORDER.index), "cn", B, (2, 3, 3), nt=4, T=0.4)
with open(sys.argv[1], "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("T13_BD_RESULT", "PASS" if ok else "FAIL", f"({sum(r['verdict'] == 'PASS' for r in rows)}/{len(rows)})")
sys.exit(0 if ok else 1)
