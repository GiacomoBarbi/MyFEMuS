#!/usr/bin/env python3
"""T13: the H1 surface operators on a UNION of faces (dump based, serial) -> ../../t13_h1_surface.csv.

`--dimension 3 --dirichlet-solve --dirichlet-face3d F1,F2,... --control-h1-length 0.3
--control-h1-time-weight 1e-2 --dump-operators` (whole-union M/K before the elimination, the eliminated
ones, C, the sparse R_H1t).  Semantics: int_Gamma |grad_Gamma u|^2 = sum_f int_{Gamma_f} |grad_{Gamma_f} u|^2
with the continuous trace (shared edge DOFs are ONE DOF); no edge term.  Checks:
 union, before elimination:
   K = sum_f [K1(a_f)(x)M1(b_f) + M1(a_f)(x)K1(b_f)] (independent 1D Q2 matrices, actual lengths) <= 1e-14;
   K(union) = sum_f K(single face f) from separate one-face runs <= 1e-15;
   symmetry; PSD; null space dimension = number of connected components of the union (constants on
   each component: 2 for an opposite pair, 1 for any set with two axes); K 1 = 0;
   v^T K v = sum_f int_{Gamma_f} |grad_f v|^2 for v = 2x - 3y + z + 0.7 (linear) and v = x^2 + yz - 2z^2 + x
   (facewise biquadratic; reproduced exactly on every face);
 restricted to C (geometric union policy): eliminated rows/cols off C zero; K|_C = K_face|_C exactly;
   K|_C SPD (H1_0 of the union, u = 0 on its external boundary) -- except the closed surface (all six
   faces: no boundary, constants in the kernel), where R = M + ell^2 K is still SPD;
   R = M + ell^2 K SPD on C for ell in {0, 0.3, 3};
 R_H1t (sparse, factored by the core) = What (x) R + (gamma/(alpha dt^2)) T (x) M, BE and CN; symmetric; SPD;
 negatives: one face dropped from K; one-direction (1D-tangent analogue) K; R_H1t with R instead of M in the
   time term (double count), wrong sign, missing 1/dt, T_NN = 2.
usage: t13_h1_surface.py <out.csv>"""
import csv, itertools, os, subprocess, sys
import numpy as np
import scipy.sparse as sps

X = os.environ.get("EX21_BIN", "/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21")
W = os.environ.get("T13_WORK", "/tmp/t13_h1s"); os.makedirs(W + "/output", exist_ok=True)
ENV = {"HOME": os.environ["HOME"], "PATH": "/home/giacomo/software/petsc/arch-linux-c-openblas-opt/bin:/usr/bin:/bin",
       "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
FACES = {"zmin": (2, False), "ymin": (1, False), "xmax": (0, True), "ymax": (1, True), "xmin": (0, False), "zmax": (2, True)}
ORDER = ["zmin", "ymin", "xmax", "ymax", "xmin", "zmax"]
ELL, GAM, AU = 0.3, 1e-2, 1e-2
GL, GW = np.polynomial.legendre.leggauss(5)
rows, ok = [], True


def coo(path):
    with open(path) as fh:
        fh.readline(); r, c = map(int, fh.readline().split()); fh.readline()
        d = np.loadtxt(fh, ndmin=2)
    if d.size == 0:
        return sps.csr_matrix((r, c))
    return sps.csr_matrix((d[:, 2], (d[:, 0].astype(int), d[:, 1].astype(int))), shape=(r, c))


def q1(lo, hi, n, kind):
    h = (hi - lo) / n; A = np.zeros((2 * n + 1,) * 2)
    loc = (h / 30.0 * np.array([[4, 2, -1], [2, 16, 2], [-1, 2, 4]], float) if kind == "M"
           else 1.0 / (3.0 * h) * np.array([[7, -8, 1], [-8, 16, -8], [1, -8, 7]], float))
    for e in range(n):
        A[2 * e:2 * e + 3, 2 * e:2 * e + 3] += loc
    return A


def rec(case, check, value, ref, good):
    global ok
    ok &= bool(good)
    rows.append({"case": case, "check": check, "value": value, "reference": ref, "verdict": "PASS" if good else "FAIL"})
    print(f"T13_H1S {case:<36s} {check:<66s} {str(value):<22s} ref {str(ref):<16s} {'PASS' if good else 'FAIL'}", flush=True)


def run(faces, dom, n, scheme, nt, prefix, box):
    args = [X, "--dimension", "3", "--dirichlet-solve", "--dirichlet-face3d", ",".join(faces), "--state-bc", "dirichlet",
            "--nx", str(n[0]), "--ny", str(n[1]), "--nz", str(n[2]), "--steps", str(nt), "--T", "0.3",
            "--time-scheme", scheme, "--no-paraview", "--convection", "on", "--vx", "1", "--vy", "0.5", "--vz", "-0.75",
            "--alpha-u", repr(AU), "--control-h1-length", repr(ELL), "--control-h1-time-weight", repr(GAM),
            "--tracking-box3d"] + box + ["--dump-operators", prefix]
    if dom != (0, 1, 0, 1, 0, 1):
        args += ["--domain3d"] + [repr(v) for v in dom]
    r = subprocess.run(args, cwd=W, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    if r.returncode != 0:
        raise RuntimeError(f"{prefix}: exit {r.returncode}: {r.stdout[-1500:]}")


def components(faces):
    parent = {f: f for f in faces}
    def find(f):
        while parent[f] != f:
            f = parent[f]
        return f
    for f, g in itertools.combinations(faces, 2):
        if FACES[f][0] != FACES[g][0]:
            parent[find(f)] = find(g)
    return len({find(f) for f in faces})


def face_int(lo, hi, faces, fun):
    s = 0.0
    for f in faces:
        ax, at_max = FACES[f]
        a, b = [k for k in range(3) if k != ax]
        pa, pb = lo[a] + (GL + 1) / 2 * (hi[a] - lo[a]), lo[b] + (GL + 1) / 2 * (hi[b] - lo[b])
        wa, wb = GW / 2 * (hi[a] - lo[a]), GW / 2 * (hi[b] - lo[b])
        for i, j in itertools.product(range(5), range(5)):
            c = np.zeros(3); c[ax] = hi[ax] if at_max else lo[ax]; c[a] = pa[i]; c[b] = pb[j]
            s += wa[i] * wb[j] * fun(c, ax)
    return s


def case(faces, dom, n, scheme="be", nt=3):
    faces = sorted(faces, key=ORDER.index)
    unit = dom == (0, 1, 0, 1, 0, 1)
    tag = f"{'+'.join(faces) if len(faces) < 6 else 'all6'}_{'x'.join(map(str, n))}{'' if unit else '_nonunit'}_{scheme}"
    lo, hi = np.array(dom[0::2], float), np.array(dom[1::2], float)
    box = [repr(float(lo[k] + f * (hi[k] - lo[k]))) for k in range(3) for f in (0.2, 0.8)]
    prefix = f"{W}/{tag}"
    run(faces, dom, n, scheme, nt, prefix, box)
    Kf, Mf = coo(prefix + "_K_Gamma_face.coo"), coo(prefix + "_M_Gamma_face.coo")
    Ke, Me, RT = coo(prefix + "_K_Gamma.coo"), coo(prefix + "_M_Gamma.coo"), coo(prefix + "_RT.coo")
    with open(prefix + "_D_C.txt") as fh:
        fh.readline(); hdr = fh.readline().split()
        Cl = np.array(sorted(int(v) for v in fh.read().split("\n") if v and not v.startswith("#")), int)
    steps, dt = int(hdr[0]), float(hdr[1])
    xyz = np.loadtxt(prefix + "_coords.txt", ndmin=2)
    L = np.stack([np.round((xyz[:, k] - lo[k]) / (hi[k] - lo[k]) * 2 * n[k]).astype(int) for k in range(3)], 1)
    on = {f: L[:, FACES[f][0]] == (2 * n[FACES[f][0]] if FACES[f][1] else 0) for f in FACES}
    union = np.zeros(len(L), bool)
    for f in faces:
        union |= on[f]
    N = Kf.shape[0]
    # independent reference, facewise
    Kref, Kparts = sps.lil_matrix((N, N)), {}
    for f in faces:
        ax = FACES[f][0]; a, b = [k for k in range(3) if k != ax]
        idx = np.where(on[f])[0]; lin = L[idx, a] * (2 * n[b] + 1) + L[idx, b]
        Kf_loc = (np.kron(q1(lo[a], hi[a], n[a], "K"), q1(lo[b], hi[b], n[b], "M"))
                  + np.kron(q1(lo[a], hi[a], n[a], "M"), q1(lo[b], hi[b], n[b], "K")))[np.ix_(lin, lin)]
        P = sps.lil_matrix((N, N)); P[np.ix_(idx, idx)] = Kf_loc; Kparts[f] = P.tocsr()
        Kref[np.ix_(idx, idx)] = Kref[np.ix_(idx, idx)].toarray() + Kf_loc
    Kref = Kref.tocsr()
    kmax = abs(Kref).max()
    d = abs(Kf - Kref).max() / kmax
    rec(tag, "K_union = sum_f [K1(x)M1 + M1(x)K1] (rel. max entrywise)", f"{d:.2e}", "<= 1e-14", d <= 1e-14)
    S = None
    for f in faces:
        p1 = f"{prefix}__single_{f}"
        run([f], dom, n, scheme, nt, p1, box)
        Ks = coo(p1 + "_K_Gamma_face.coo"); S = Ks if S is None else S + Ks
    d = (abs(Kf - S).max() if (Kf - S).nnz else 0.0) / kmax
    rec(tag, "K_union = sum_f K(single face f), separate runs", f"{d:.2e}", "<= 1e-15", d <= 1e-15)
    sym = (abs(Kf - Kf.T).max() if (Kf - Kf.T).nnz else 0.0) / kmax
    rec(tag, "K_union symmetric", f"{sym:.1e}", "<= 1e-15", sym <= 1e-15)
    idx = np.where(union)[0]
    ev = np.linalg.eigvalsh(Kf[idx][:, idx].toarray())
    rec(tag, "K_union PSD: lambda_min/lambda_max", f"{ev[0] / ev[-1]:.1e}", ">= -1e-14", ev[0] / ev[-1] >= -1e-14)
    nz, nc = int((np.abs(ev) <= 1e-12 * ev[-1]).sum()), components(faces)
    rec(tag, "null space dim = connected components of the union", nz, nc, nz == nc)
    k1v = abs(Kf @ np.ones(N)).max() / kmax
    rec(tag, "K_union 1 = 0", f"{k1v:.1e}", "<= 1e-14", k1v <= 1e-14)
    for lab, fv, grad in (("2x - 3y + z + 0.7", lambda c: 2 * c[0] - 3 * c[1] + c[2] + 0.7, lambda c: np.array([2.0, -3.0, 1.0])),
                          ("x^2 + yz - 2z^2 + x", lambda c: c[0] ** 2 + c[1] * c[2] - 2 * c[2] ** 2 + c[0],
                           lambda c: np.array([2 * c[0] + 1, c[2], c[1] - 4 * c[2]]))):
        v = np.array([fv(c) for c in xyz])
        got = v @ (Kf @ v)
        ex = face_int(lo, hi, faces, lambda c, ax: float(sum(g * g for k, g in enumerate(grad(c)) if k != ax)))
        rec(tag, f"v^T K v = sum_f int |grad_f v|^2, v = {lab}", f"{got:.15g}", f"{ex:.15g}", abs(got - ex) <= 1e-12 * max(1, ex))
    f0 = faces[-1]
    dn = abs(Kf - (Kref - Kparts[f0])).max() / kmax
    rec(tag, f"NEG face {f0} dropped from K: detected", f"{dn:.1e}", "> 1e-6", dn > 1e-6)
    # one in-plane direction only (1D-tangent analogue) on every face
    K1d = sps.lil_matrix((N, N))
    for f in faces:
        ax = FACES[f][0]; a, b = [k for k in range(3) if k != ax]
        ii = np.where(on[f])[0]; lin = L[ii, a] * (2 * n[b] + 1) + L[ii, b]
        K1d[np.ix_(ii, ii)] = K1d[np.ix_(ii, ii)].toarray() + np.kron(q1(lo[a], hi[a], n[a], "K"), q1(lo[b], hi[b], n[b], "M"))[np.ix_(lin, lin)]
    dn = abs(Kf - K1d.tocsr()).max() / kmax
    rec(tag, "NEG one-direction (1D-tangent analogue) K: detected", f"{dn:.1e}", "> 1e-6", dn > 1e-6)
    # ---- restriction to C
    off = np.setdiff1d(np.arange(N), Cl)
    rec(tag, "eliminated K rows/cols off C zero", int(abs(Ke[off]).sum() + abs(Ke[:, off]).sum() > 0), 0,
        abs(Ke[off]).sum() == 0 and abs(Ke[:, off]).sum() == 0)
    KC, MC = Ke[Cl][:, Cl].toarray(), Me[Cl][:, Cl].toarray()
    rec(tag, "K|_C = K_face|_C (exact restriction)", f"{np.abs(KC - Kf[Cl][:, Cl].toarray()).max():.1e}", 0,
        np.abs(KC - Kf[Cl][:, Cl].toarray()).max() == 0)
    eK = np.linalg.eigvalsh(KC)
    closed = len(faces) == 6
    if closed:
        nzC = int((np.abs(eK) <= 1e-12 * eK[-1]).sum())
        rec(tag, "closed surface: K|_C null space = constants (dim 1)", nzC, 1, nzC == 1)
    else:
        rec(tag, "K|_C SPD (H1_0 of the union): lambda_min", f"{eK[0]:.3e}", "> 0", eK[0] > 1e-12 * eK[-1])
    for ell in (0.0, 0.3, 3.0):
        e = np.linalg.eigvalsh(MC + ell * ell * KC)[0]
        rec(tag, f"R = M + ell^2 K SPD on C, ell = {ell:g}", f"{e:.3e}", "> 0", e > 0)
    w = np.ones(steps)
    if scheme == "cn":
        w[-1] = 0.5
    T = 2 * np.eye(steps) - np.eye(steps, k=1) - np.eye(steps, k=-1); T[-1, -1] = 1
    RTref = np.kron(np.diag(w), MC + ELL * ELL * KC) + GAM / (AU * dt * dt) * np.kron(T, MC)
    RTd = RT.toarray()
    d = np.abs(RTd - RTref).max() / np.abs(RTref).max()
    rec(tag, "sparse R_H1t = What(x)R + gamma/(alpha dt^2) T(x)M (rel.)", f"{d:.1e}", "<= 1e-14", d <= 1e-14)
    rec(tag, "R_H1t symmetric, SPD (lambda_min)", f"{np.linalg.eigvalsh(RTd)[0]:.3e}", "> 0",
        np.abs(RTd - RTd.T).max() <= 1e-15 * np.abs(RTd).max() and np.linalg.eigvalsh(RTd)[0] > 0)
    for lab, wrong in (("time term with R instead of M (double count)",
                        np.kron(np.diag(w), MC + ELL * ELL * KC) + GAM / (AU * dt * dt) * np.kron(T, MC + ELL * ELL * KC)),
                       ("wrong sign of the H1-time term", np.kron(np.diag(w), MC + ELL * ELL * KC) - GAM / (AU * dt * dt) * np.kron(T, MC)),
                       ("missing 1/dt", np.kron(np.diag(w), MC + ELL * ELL * KC) + GAM / (AU * dt) * np.kron(T, MC)),
                       ("T_NN = 2 (no natural end condition)",
                        np.kron(np.diag(w), MC + ELL * ELL * KC) + GAM / (AU * dt * dt) * np.kron(T + np.diag(np.eye(steps)[-1]), MC))):
        dd = np.abs(RTd - wrong).max() / np.abs(RTref).max()
        rec(tag, f"NEG {lab}: detected", f"{dd:.1e}", "> 1e-6", dd > 1e-6)


U = (0, 1, 0, 1, 0, 1)
SETS = [["zmin", "zmax"], ["zmin", "xmin"], ["xmax", "ymax"], ["zmin", "ymin", "xmin"], ["zmin", "xmin", "xmax"],
        ["zmin", "ymin", "xmax", "ymax", "xmin"], ORDER]
for fs in SETS:
    case(fs, U, (3, 3, 3), "be")
for fs in (SETS[0], SETS[1], SETS[3], SETS[6]):
    case(fs, U, (4, 3, 5), "cn")   # anisotropic; (0.2,0.8)^3 box stays off every face
for fs in (SETS[1], SETS[3]):
    case(fs, (0, 2, -1, 1, 0, .5), (4, 4, 4), "cn", 4)   # n_z = 4: the relative (0.2,0.8) box keeps off the faces
with open(sys.argv[1], "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
print("T13_H1S_RESULT", "PASS" if ok else "FAIL", f"({sum(r['verdict'] == 'PASS' for r in rows)}/{len(rows)})")
sys.exit(0 if ok else 1)
