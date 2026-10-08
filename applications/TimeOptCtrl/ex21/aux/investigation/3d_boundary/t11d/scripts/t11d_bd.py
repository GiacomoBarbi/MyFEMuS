#!/usr/bin/env python3
"""T11D: independent explicit check of the 3D Dirichlet control-to-state map B_D and of
its transpose B_D^T (dump based, serial) -> ../../t11d_bd_transpose.csv.

`--dump-operators` (3D Dirichlet) writes the unhatted M, L (C kept, D eliminated), the
controlled set C, dt, the scheme, and B_D / B_D^T built column by column by the C++
ApplyBD / ApplyBDT on unit space-time vectors (index n N + i, n = 0..Nt-1 <-> t_{n+1};
u_0 is data).  Python rebuilds from M, L, C only:
  BE: (B_D u)_n = P_I [A Q_C u_n - (M/dt) Q_C u_{n-1}],          A  = M/dt + L
  CN: (B_D u)_n = P_I [A+ Q_C u_n + A- Q_C u_{n-1}],  A+ = M/dt + L/2, A- = -M/dt + L/2
and checks
  * dumped B_D            = formula B_D                   (relative, max entrywise)
  * dumped B_D^T          = (formula B_D)^T on C columns   (exact transpose; uses L^T)
  * dumped B_D^T          = (dumped B_D)^T
  * <B_D u, p> = <u, B_D^T p> for random u (on C), p
negative controls (must be detected, i.e. differ by >> roundoff):
  * forward-L transpose  Q_C^T[(M/dt + L) P_I p_n - ...]   (wrong under convection)
  * wrong CN coupling    p_{n-1} instead of p_{n+1}
and the structural facts: C = int(Gamma_c) (count (2n_a-1)(2n_b-1), plane, perimeter
excluded), rows of B_D on C and D are zero (P_I), columns off C are zero.
usage: t11d_bd.py <out.csv>    (EX21_BIN overrides the executable)"""
import csv, os, subprocess, sys
import numpy as np
import scipy.sparse as sps

X = os.environ.get("EX21_BIN", "/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21")
W = os.environ.get("T11D_WORK", "/tmp/t11d_bd"); os.makedirs(W + "/output", exist_ok=True)
ENV = {"HOME": os.environ["HOME"], "PATH": "/home/giacomo/software/petsc/arch-linux-c-openblas-opt/bin:/usr/bin:/bin",
       "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
FACES = {"zmin": (2, False), "ymin": (1, False), "xmax": (0, True), "ymax": (1, True), "xmin": (0, False), "zmax": (2, True)}
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
    print(f"T11D_BD {case:<30s} {check:<58s} {str(value):<12s} ref {str(ref):<12s} {'PASS' if good else 'FAIL'}")


def rel(a, b):
    a, b = sps.csr_matrix(a), sps.csr_matrix(b)
    s = max(abs(b).max(), 1e-300)
    d = a - b
    return (abs(d).max() if d.nnz else 0.0) / s


def case(face, scheme, wind, n, nt=3, T=0.3):
    tag = f"{face}_{scheme}_{'heat' if wind is None else 'wind'}_{'x'.join(map(str, n))}"
    prefix = f"{W}/{tag}"
    args = [X, "--dimension", "3", "--dirichlet-solve", "--dirichlet-face3d", face, "--state-bc", "dirichlet",
            "--nx", str(n[0]), "--ny", str(n[1]), "--nz", str(n[2]), "--steps", str(nt), "--T", repr(T),
            "--time-scheme", scheme, "--no-paraview", "--tracking-box3d", "0.25", "0.75", "0.25", "0.75", "0.25", "0.75",
            "--dump-operators", prefix]
    args += ["--convection", "off"] if wind is None else ["--convection", "on", "--vx", repr(wind[0]), "--vy", repr(wind[1]), "--vz", repr(wind[2])]
    r = subprocess.run(args, cwd=W, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if r.returncode != 0:
        raise RuntimeError(f"{tag}: exit {r.returncode}: {r.stdout[-600:]} {r.stderr[-600:]}")
    M, L = coo(prefix + "_D_Mfull.coo"), coo(prefix + "_D_Lfull.coo")
    BD, BDT = coo(prefix + "_BD.coo"), coo(prefix + "_BDT.coo")
    with open(prefix + "_D_C.txt") as fh:
        fh.readline(); hdr = fh.readline().split()
        Cl = np.array([int(v) for v in fh.read().split()], int)
    steps, dt, sch, tr = int(hdr[0]), float(hdr[1]), hdr[2], int(hdr[3])
    N = M.shape[0]
    xyz = np.loadtxt(prefix + "_coords.txt", ndmin=2)
    ax, at_max = FACES[face]
    a, b = [k for k in range(3) if k != ax]
    plane = np.isclose(xyz[:, ax], 1.0 if at_max else 0.0, atol=1e-12)
    interior = plane & (xyz[:, a] > 1e-12) & (xyz[:, a] < 1 - 1e-12) & (xyz[:, b] > 1e-12) & (xyz[:, b] < 1 - 1e-12)
    rec(tag, "header: steps / dt / scheme", f"{steps}/{dt:.3g}/{sch}", f"{nt}/{T / nt:.3g}/{scheme.upper()}",
        steps == nt and abs(dt - T / nt) < 1e-15 and sch == scheme.upper())
    rec(tag, "|C| = (2n_a-1)(2n_b-1)", len(Cl), (2 * n[a] - 1) * (2 * n[b] - 1), len(Cl) == (2 * n[a] - 1) * (2 * n[b] - 1))
    rec(tag, "C = int(Gamma_c) node set (perimeter excluded)", "eq" if set(Cl) == set(np.where(interior)[0]) else "differ", "eq",
        set(Cl) == set(np.where(interior)[0]))
    rec(tag, "ApplyBDT uses L^T (transposes_L flag) iff convection", tr, int(wind is not None), tr == int(wind is not None))
    # ---- formula, independent of the C++ operator classes
    QC = sps.csr_matrix((np.ones(len(Cl)), (Cl, np.arange(len(Cl)))), shape=(N, len(Cl)))
    pI = np.ones(N); pI[Cl] = 0.0
    # D rows: L has diagonal 1 and M zero there; they are not free unknowns either
    Ddiag = np.isclose(L.diagonal(), 1.0) & (abs(L).sum(1).A1 == 1.0) & (abs(M).sum(1).A1 == 0.0)
    pI[Ddiag] = 0.0
    PI = sps.diags(pI)
    if sch == "BE":
        Ap, Am = M / dt + L, -M / dt
    else:
        Ap, Am = M / dt + 0.5 * L, -M / dt + 0.5 * L
    diag, low = PI @ Ap @ QC, PI @ Am @ QC
    blocks = [[None] * steps for _ in range(steps)]
    for k in range(steps):
        blocks[k][k] = diag
        if k > 0:
            blocks[k][k - 1] = low
    Bf = sps.bmat(blocks).tocsr()                          # (Nt N) x (Nt |C|)
    cols = np.concatenate([k * N + Cl for k in range(steps)])
    BDc = BD[:, cols]
    rec(tag, "dumped B_D = formula B_D (rel.)", f"{rel(BDc, Bf):.1e}", f"<={TOL:.0e}", rel(BDc, Bf) <= TOL)
    off = np.setdiff1d(np.arange(steps * N), cols)
    rec(tag, "dumped B_D columns off C are zero", int(abs(BD[:, off]).sum() > 0), 0, abs(BD[:, off]).sum() == 0)
    rows_off = np.concatenate([k * N + np.where(pI == 0)[0] for k in range(steps)])
    rec(tag, "dumped B_D rows on C u D are zero (P_I)", int(abs(BD[rows_off]).sum() > 0), 0, abs(BD[rows_off]).sum() == 0)
    rec(tag, "B_D nonzero (lifting couples C to I)", BD.nnz, "> 0", BD.nnz > 0)
    BDTc = BDT[cols, :]                                     # (Nt |C|) x (Nt N)
    rec(tag, "dumped B_D^T = (formula B_D)^T (rel.)", f"{rel(BDTc, Bf.T):.1e}", f"<={TOL:.0e}", rel(BDTc, Bf.T) <= TOL)
    rec(tag, "dumped B_D^T = (dumped B_D)^T (rel.)", f"{rel(BDT, BD.T):.1e}", f"<={TOL:.0e}", rel(BDT, BD.T) <= TOL)
    rng = np.random.default_rng(7)
    u, p = rng.standard_normal(Bf.shape[1]), rng.standard_normal(steps * N)
    lhs, rhs = p @ (BDc @ u), u @ (BDTc @ p)
    d = abs(lhs - rhs) / max(abs(lhs), abs(rhs))
    rec(tag, "<B_D u, p> = <u, B_D^T p> (rel.)", f"{d:.1e}", f"<={TOL:.0e}", d <= TOL)
    # ---- negative controls: the same transpose formula built wrongly
    if sch == "BE":
        Fp, Fm = M / dt + L, -M / dt
    else:
        Fp, Fm = M / dt + 0.5 * L, -M / dt + 0.5 * L
    fwd = [[None] * steps for _ in range(steps)]           # forward L (no transpose of L)
    for k in range(steps):
        fwd[k][k] = QC.T @ Fp @ PI
        if k + 1 < steps:
            fwd[k][k + 1] = QC.T @ Fm @ PI
    neg = rel(BDTc, sps.bmat(fwd))
    if wind is None:
        rec(tag, "NEG forward-L B_D^T: equals (L symmetric, heat)", f"{neg:.1e}", f"<={TOL:.0e}", neg <= TOL)
    else:
        rec(tag, "NEG forward-L B_D^T: detected", f"{neg:.1e}", "> 1e-6", neg > 1e-6)
    if sch == "CN":
        wr = [[None] * steps for _ in range(steps)]        # wrong coupling p_{n-1}
        for k in range(steps):
            wr[k][k] = (PI @ Ap @ QC).T
            if k > 0:
                wr[k][k - 1] = (PI @ Am @ QC).T
        neg = rel(BDTc, sps.bmat(wr))
        rec(tag, "NEG CN coupling p_{n-1} instead of p_{n+1}: detected", f"{neg:.1e}", "> 1e-6", neg > 1e-6)


def g1d(face, wind, n=(2, 2, 2)):
    """G1D: on the free set I u C the convection matrix C = L(w) - L(0) (unhatted, D
    eliminated) satisfies C + C^T = (w.n) M_Gamma (only Gamma_c survives the
    elimination; div w = 0).  Tangential winds give C + C^T = 0 on I u C."""
    ax, at_max = FACES[face]
    wn = (1.0 if at_max else -1.0) * wind[ax]
    tag = f"G1D_{face}_w({','.join(f'{v:g}' for v in wind)})"
    mats = []
    for w in (None, wind):
        prefix = f"{W}/g1d_{face}_{'heat' if w is None else 'wind'}"
        args = [X, "--dimension", "3", "--dirichlet-solve", "--dirichlet-face3d", face, "--state-bc", "dirichlet",
                "--nx", str(n[0]), "--ny", str(n[1]), "--nz", str(n[2]), "--steps", "2", "--no-paraview",
                "--tracking-box3d", "0.25", "0.75", "0.25", "0.75", "0.25", "0.75", "--dump-operators", prefix]
        args += ["--convection", "off"] if w is None else ["--convection", "on", "--vx", repr(w[0]), "--vy", repr(w[1]), "--vz", repr(w[2])]
        r = subprocess.run(args, cwd=W, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        if r.returncode != 0:
            raise RuntimeError(f"{tag}: exit {r.returncode}")
        mats.append((coo(prefix + "_D_Lfull.coo"), coo(prefix + "_M_Gamma.coo")))
    (L0, MG), (L1, _) = mats
    Cm = (L1 - L0).tocsr()
    free = np.where(~(np.isclose(L0.diagonal(), 1.0) & np.isclose(abs(L0).sum(1).A1, 1.0)))[0]
    S = (Cm + Cm.T)[free][:, free]
    ref = (wn * MG)[free][:, free]
    d = abs(S - ref).max() / max(abs(Cm).max(), 1e-300)
    rec(tag, f"C + C^T = (w.n) M_Gamma on I u C (w.n = {wn:g}), rel. to max|C|", f"{d:.1e}", f"<={TOL:.0e}", d <= TOL)
    if wn != 0.0:
        neg = abs(S).max() / max(abs(Cm).max(), 1e-300)
        rec(tag, "NEG plain skew C + C^T = 0 on I u C: violated (Gamma_c term)", f"{neg:.1e}", "> 1e-6", neg > 1e-6)


A = (1.0, 0.5, -0.75)
for face, wind in (("zmin", A), ("xmax", A), ("ymin", (-2.0, 1.0, 0.5)), ("zmin", (1.0, 0.5, 0.0))):
    g1d(face, wind)
for face in ("zmin", "xmax"):
    for scheme in ("be", "cn"):
        for wind in (None, A):
            case(face, scheme, wind, (2, 2, 2))
for face in ("ymax", "zmax"):
    case(face, "cn", (-2.0, 1.0, 0.5), (2, 3, 3), nt=4, T=0.4)
with open(sys.argv[1], "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("T11D_BD_RESULT", "PASS" if ok else "FAIL", f"({sum(r['verdict'] == 'PASS' for r in rows)}/{len(rows)})")
sys.exit(0 if ok else 1)
