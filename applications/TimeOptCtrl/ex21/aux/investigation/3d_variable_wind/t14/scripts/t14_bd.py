#!/usr/bin/env python3
"""T14: Dirichlet boundary control under VARIABLE wind, explicit dump checks (serial) -> ../../t14_bd.csv.

For --dirichlet-face3d F (single face and unions) with --wind harmonic-3d / affine-div-3d, BE / CN:
  * dumped B_D = the BE/CN formula rebuilt from the dumped unhatted M, L and the geometric C;
  * dumped B_D^T = (formula)^T = (dumped B_D)^T;  <B_D u, p> = <u, B_D^T p>;
  * G1D on the free set I u C:  C + C^T = (B_w - D_w)|_{I u C}, with B_w, D_w from the INDEPENDENT Python
    reference assembly (t14_assembly.reference; w at the quadrature points); C = L(wind) - L(off);
negatives (must differ by >> roundoff): L instead of L^T in B_D^T; wrong C (face omitted); the div-free
identity C + C^T = B_w on the generic field; the identity WITHOUT D_w.
usage: t14_bd.py <out.csv>"""
import csv, itertools, os, sys
import numpy as np
import scipy.sparse as sps
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t14_assembly as A

rows, ok = [], True
TOL = 1e-13
FACES = {"zmin": (2, 0), "ymin": (1, 0), "xmax": (0, 1), "ymax": (1, 1), "xmin": (0, 0), "zmax": (2, 1)}


def rec(case, check, value, ref, good):
    global ok
    ok &= bool(good)
    rows.append({"case": case, "check": check, "value": value, "reference": ref, "verdict": "PASS" if good else "FAIL"})
    print(f"T14_BD {case:<40s} {check:<64s} {str(value):<10s} ref {str(ref):<10s} {'PASS' if good else 'FAIL'}", flush=True)


def rel(a, b):
    d = sps.csr_matrix(a) - sps.csr_matrix(b)
    return (abs(d).max() if d.nnz else 0.0) / max(abs(sps.csr_matrix(b)).max(), 1e-300)


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
    cols = np.concatenate([k * N + C for k in range(steps)])
    E = sps.csr_matrix((np.ones(len(cols)), (np.arange(len(cols)), cols)), shape=(len(cols), steps * N))
    return (B @ E).tocsr(), cols, PI, QC


def case(field, faces, scheme, n=(2, 2, 2), nt=3):
    tag = f"{field}_{'+'.join(faces)}_{scheme}"
    pre = f"{A.W}/bd_{tag}"
    base = ["--dimension", "3", "--dirichlet-solve", "--dirichlet-face3d", ",".join(faces), "--state-bc", "dirichlet",
            "--nx", str(n[0]), "--ny", str(n[1]), "--nz", str(n[2]), "--steps", str(nt), "--T", "0.3", "--time-scheme", scheme,
            "--tracking-box3d", "0.25", "0.75", "0.25", "0.75", "0.25", "0.75", "--nu", "0.25"]
    A.run(base + ["--convection", "on", "--wind", field], pre)
    A.run(base + ["--convection", "off"], pre + "_off")
    M, L = A.coo(pre + "_D_Mfull.coo"), A.coo(pre + "_D_Lfull.coo")
    L0 = A.coo(pre + "_off_D_Lfull.coo")
    BD, BDT = A.coo(pre + "_BD.coo"), A.coo(pre + "_BDT.coo")
    with open(pre + "_D_C.txt") as fh:
        fh.readline(); hdr = fh.readline().split()
        Cl = np.array(sorted(int(v) for v in fh.read().split() if not v.startswith("#")), int)
    steps, dt, sch = int(hdr[0]), float(hdr[1]), hdr[2]
    xyz = np.loadtxt(pre + "_coords.txt", ndmin=2)
    lo, hi = np.zeros(3), np.ones(3)
    lat = np.round(xyz * 2 * np.array(n)).astype(int)
    on = {f: lat[:, a] == (2 * n[a] if m else 0) for f, (a, m) in FACES.items()}
    union, unctl = np.zeros(len(xyz), bool), np.zeros(len(xyz), bool)
    for f in FACES:
        (union if f in faces else unctl)[:] |= on[f]
    Cgeo = np.where(union & ~unctl)[0]
    rec(tag, "dumped C = geometric C", len(Cl), len(Cgeo), np.array_equal(Cl, Cgeo))
    Dmask = (union | unctl) & ~np.isin(np.arange(len(xyz)), Cgeo)
    Bf, cols, PI, QC = bd_formula(M, L, Cgeo, Dmask, steps, dt, sch)
    rec(tag, "dumped B_D = formula (rel.)", f"{rel(BD, Bf):.1e}", f"<={TOL:.0e}", rel(BD, Bf) <= TOL)
    rec(tag, "dumped B_D^T = (formula)^T = (dumped B_D)^T", f"{max(rel(BDT, Bf.T), rel(BDT, BD.T)):.1e}", f"<={TOL:.0e}",
        max(rel(BDT, Bf.T), rel(BDT, BD.T)) <= TOL)
    rng = np.random.default_rng(7)
    u, p = np.zeros(steps * len(xyz)), rng.standard_normal(steps * len(xyz)); u[cols] = rng.standard_normal(len(cols))
    lhs, rhs = p @ (BD @ u), u @ (BDT @ p)
    d = abs(lhs - rhs) / max(abs(lhs), abs(rhs))
    rec(tag, "<B_D u, p> = <u, B_D^T p>", f"{d:.1e}", f"<={TOL:.0e}", d <= TOL)
    Fp, Fm = (M / dt + L, -M / dt) if sch == "BE" else (M / dt + 0.5 * L, -M / dt + 0.5 * L)
    fwd = [[None] * steps for _ in range(steps)]
    for k in range(steps):
        fwd[k][k] = QC.T @ Fp @ PI
        if k + 1 < steps:
            fwd[k][k + 1] = QC.T @ Fm @ PI
    neg = rel(BDT[cols, :], sps.bmat(fwd))
    rec(tag, "NEG L instead of L^T in B_D^T: detected", f"{neg:.1e}", "> 1e-6", neg > 1e-6)
    if len(faces) > 1:
        Cw = np.setdiff1d(Cgeo, np.where(on[faces[-1]])[0])
        Bw = bd_formula(M, L, Cw, Dmask | on[faces[-1]], steps, dt, sch)[0]
        rec(tag, f"NEG wrong C (face {faces[-1]} omitted): detected", f"{rel(BD, Bw):.1e}", "> 1e-6", rel(BD, Bw) > 1e-6)
    # G1D with the independent reference
    gidx = {tuple(l): i for i, l in enumerate(lat)}
    _, _, Bref, Dref = A.reference(A.FIELDS[field], lo, hi, np.array(n), gidx)
    Cm = (L - L0).tocsr()
    free = np.where(~(np.isclose(L0.diagonal(), 1.0) & np.isclose(abs(L0).sum(1).A1, 1.0)))[0]
    S = (Cm + Cm.T)[free][:, free]
    scale = abs(Cm).max()
    d = abs(S - (Bref - Dref)[free][:, free]).max() / scale
    rec(tag, "G1D C + C^T = (B_w - D_w) on I u C (independent B_w, D_w)", f"{d:.1e}", f"<={TOL:.0e}", d <= TOL)
    if np.trace(A.FIELDS[field]) != 0:
        dn = abs(S - Bref[free][:, free]).max() / scale
        rec(tag, "NEG identity without D_w (div-free form) on the generic field", f"{dn:.1e}", "> 1e-6", dn > 1e-6)
    if np.isin(Cgeo, free).all() and len(Cgeo):
        dz = abs(S).max() / scale
        rec(tag, "NEG plain skew C + C^T = 0 on I u C", f"{dz:.1e}", "> 1e-6", dz > 1e-6)


for field in A.FIELDS:
    for faces in (["zmin"], ["zmin", "zmax"], ["zmin", "xmin"], ["zmin", "ymin", "xmin"]):
        for s in ("be", "cn"):
            case(field, faces, s)
with open(sys.argv[1], "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("T14_BD_RESULT", "PASS" if ok else "FAIL", f"({sum(r['verdict'] == 'PASS' for r in rows)}/{len(rows)})")
sys.exit(0 if ok else 1)
