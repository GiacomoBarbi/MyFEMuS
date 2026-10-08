#!/usr/bin/env python3
"""T11H1: independent validation of the 3D surface stiffness K_Gamma, of R_Gamma and of the
sparse reduced R_H1t (dump based, serial) -> ../../t11h1_surface_stiffness.csv.

`--dimension 3 --dirichlet-solve --dirichlet-face3d F --state-bc dirichlet --control-h1-length 0.3
--control-h1-time-weight 1e-2 --dump-operators` writes the whole-face M_Gamma/K_Gamma BEFORE the
elimination (_face), the eliminated ones, C, and the sparse R_H1t.  Checks:
 whole face:  support = face-plane nodes ((2n_a+1)(2n_b+1)), off-face rows 0; symmetry; PSD with
   the constants as the ONLY null vector; K 1 = 0; v^T K v = int |grad_G v|^2 dS exactly for a
   linear and a quadratic tangential v; K = K1(a) (x) M1(b) + M1(a) (x) K1(b) (independently
   assembled 1D Q2 matrices, actual edge lengths: anisotropic meshes, non-unit faces) to 1e-14;
 orientation: K on the two opposite faces of the same axis, matched by in-plane coordinates;
 restricted to C = int(Gamma_c):  eliminated K rows off C zero, K|_C = K_face|_C exactly,
   K|_C SPD (H^1_0), M|_C SPD, R = M + ell^2 K SPD for ell in {0, 0.1, 0.3, 1, 3, 10};
 R_H1t (sparse, the matrix the core factors) = What (x) R + (gamma/(alpha dt^2)) T (x) M (T =
   tridiag(-1,2,-1), last diagonal 1) rebuilt here, BE and CN, symmetric, SPD;
 preallocation (PETSc -info :mat): 0 mallocs during MatSetValues for every assembled matrix;
   max row nnz of K_Gamma (face) and R_H1t vs the allocated bound max(24, 3 row_max).
usage: t11h1_surface_stiffness.py <out.csv>"""
import csv, itertools, os, re, subprocess, sys
import numpy as np
import scipy.sparse as sps

X = os.environ.get("EX21_BIN", "/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21")
W = os.environ.get("T11H1_WORK", "/tmp/t11h1_ks"); os.makedirs(W + "/output", exist_ok=True)
ENV = {"HOME": os.environ["HOME"], "PATH": "/home/giacomo/software/petsc/arch-linux-c-openblas-opt/bin:/usr/bin:/bin",
       "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "PETSC_OPTIONS": "-info :mat"}
FACES = {"zmin": (2, False), "ymin": (1, False), "xmax": (0, True), "ymax": (1, True), "xmin": (0, False), "zmax": (2, True)}
ELL, GAM, AU = 0.3, 1e-2, 1e-2
rows, ok = [], True
store = {}


def coo(path):
    with open(path) as fh:
        fh.readline(); r, c = map(int, fh.readline().split()); fh.readline()
        d = np.loadtxt(fh, ndmin=2)
    if d.size == 0:
        return sps.csr_matrix((r, c))
    return sps.csr_matrix((d[:, 2], (d[:, 0].astype(int), d[:, 1].astype(int))), shape=(r, c))


def m1(lo, hi, n):
    h = (hi - lo) / n; M = np.zeros((2 * n + 1,) * 2)
    for e in range(n):
        M[2 * e:2 * e + 3, 2 * e:2 * e + 3] += h / 30.0 * np.array([[4, 2, -1], [2, 16, 2], [-1, 2, 4]], float)
    return M


def k1(lo, hi, n):
    h = (hi - lo) / n; K = np.zeros((2 * n + 1,) * 2)
    for e in range(n):
        K[2 * e:2 * e + 3, 2 * e:2 * e + 3] += 1.0 / (3.0 * h) * np.array([[7, -8, 1], [-8, 16, -8], [1, -8, 7]], float)
    return K


def rec(case, check, value, ref, good):
    global ok
    ok &= bool(good)
    rows.append({"case": case, "check": check, "value": value, "reference": ref, "verdict": "PASS" if good else "FAIL"})
    print(f"T11H1_KS {case:<26s} {check:<62s} {str(value):<24s} ref {str(ref):<18s} {'PASS' if good else 'FAIL'}", flush=True)


def gauss_int(f, lo, hi, a, b):
    gl, gw = np.polynomial.legendre.leggauss(5)
    pa, pb = lo[a] + (gl + 1) / 2 * (hi[a] - lo[a]), lo[b] + (gl + 1) / 2 * (hi[b] - lo[b])
    wa, wb = gw / 2 * (hi[a] - lo[a]), gw / 2 * (hi[b] - lo[b])
    return sum(wa[i] * wb[j] * f(pa[i], pb[j]) for i in range(5) for j in range(5))


def case(face, dom, n, scheme="be", nt=3):
    unit = dom == (0, 1, 0, 1, 0, 1)
    tag = f"{face}_{'x'.join(map(str, n))}_{'unit' if unit else 'nonunit'}_{scheme}"
    prefix = f"{W}/{tag}"
    lo, hi = np.array(dom[0::2], float), np.array(dom[1::2], float)
    ax, at_max = FACES[face]
    a, b = [k for k in range(3) if k != ax]
    box = []
    for k in range(3):   # tracking box strictly inside, away from the controlled face
        L = hi[k] - lo[k]
        if k == ax:
            box += [repr(float(lo[k] + 0.0 * L)), repr(float(lo[k] + 0.6 * L))] if at_max else [repr(float(lo[k] + 0.4 * L)), repr(float(hi[k]))]
        else:
            box += [repr(float(lo[k] + 0.2 * L)), repr(float(lo[k] + 0.8 * L))]
    args = [X, "--dimension", "3", "--dirichlet-solve", "--dirichlet-face3d", face, "--state-bc", "dirichlet",
            "--nx", str(n[0]), "--ny", str(n[1]), "--nz", str(n[2]), "--steps", str(nt), "--T", "0.3",
            "--time-scheme", scheme, "--no-paraview", "--convection", "on", "--vx", "1", "--vy", "0.5", "--vz", "-0.75",
            "--alpha-u", repr(AU), "--control-h1-length", repr(ELL), "--control-h1-time-weight", repr(GAM),
            "--tracking-box3d"] + box + ["--dump-operators", prefix]
    if not unit:
        args += ["--domain3d"] + [repr(v) for v in dom]
    r = subprocess.run(args, cwd=W, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    if r.returncode != 0:
        raise RuntimeError(f"{tag}: exit {r.returncode}: {r.stdout[-1500:]}")
    mallocs = [int(v) for v in re.findall(r"Number of mallocs during MatSetValues\(\) is (\d+)", r.stdout)]
    Kf, Mf = coo(prefix + "_K_Gamma_face.coo"), coo(prefix + "_M_Gamma_face.coo")
    Ke, Me = coo(prefix + "_K_Gamma.coo"), coo(prefix + "_M_Gamma.coo")
    RT = coo(prefix + "_RT.coo")
    with open(prefix + "_D_C.txt") as fh:
        fh.readline(); hdr = fh.readline().split()
        rest = [l for l in fh.read().split("\n") if l and not l.startswith("#")]
    Cl = np.array(sorted(int(v) for v in rest), int)
    steps, dt = int(hdr[0]), float(hdr[1])
    xyz = np.loadtxt(prefix + "_coords.txt", ndmin=2)
    on = lambda k, v: np.isclose(xyz[:, k], v, atol=1e-12)
    plane = on(ax, hi[ax] if at_max else lo[ax])
    idx = np.where(plane)[0]
    N = Kf.shape[0]
    # ---- whole face
    sup = np.abs(Kf).sum(1).A1 > 0
    rec(tag, "K_face support = face-plane nodes", int(sup.sum()), int(plane.sum()), np.array_equal(sup, plane))
    rec(tag, "face node count (2n_a+1)(2n_b+1)", int(plane.sum()), (2 * n[a] + 1) * (2 * n[b] + 1),
        int(plane.sum()) == (2 * n[a] + 1) * (2 * n[b] + 1))
    kmax = abs(Kf).max()
    sym = abs(Kf - Kf.T).max() / kmax
    rec(tag, "K_face symmetry max|K-K^T|/max|K|", f"{sym:.1e}", "<= 1e-15", sym <= 1e-15)
    S = Kf[idx][:, idx].toarray()
    ev = np.linalg.eigvalsh(S)
    rec(tag, "K_face PSD: lambda_min/lambda_max", f"{ev[0] / ev[-1]:.1e}", ">= -1e-14", ev[0] / ev[-1] >= -1e-14)
    nzero = int((np.abs(ev) <= 1e-12 * ev[-1]).sum())
    rec(tag, "K_face null space dimension (constants only)", nzero, 1, nzero == 1)
    k1v = abs(Kf @ np.ones(N)).max() / kmax
    rec(tag, "K_face 1 = 0 (max|K 1|/max|K|)", f"{k1v:.1e}", "<= 1e-14", k1v <= 1e-14)
    sa, sb = xyz[:, a], xyz[:, b]
    v = 2.0 * sa - 3.0 * sb + 0.7
    got, exact = v @ (Kf @ v), (4.0 + 9.0) * (hi[a] - lo[a]) * (hi[b] - lo[b])
    rec(tag, "v^T K v = int |grad_G v|^2, v = 2 s_a - 3 s_b + 0.7", f"{got:.15g}", f"{exact:.15g}", abs(got - exact) <= 1e-12 * exact)
    v = sa * sa + sa * sb
    got = v @ (Kf @ v)
    exact = gauss_int(lambda p, q: (2 * p + q) ** 2 + p * p, lo, hi, a, b)
    rec(tag, "v^T K v = int |grad_G v|^2, v = s_a^2 + s_a s_b", f"{got:.15g}", f"{exact:.15g}", abs(got - exact) <= 1e-12 * exact)
    ga = np.round((xyz[idx, a] - lo[a]) / (hi[a] - lo[a]) * 2 * n[a]).astype(int)
    gb = np.round((xyz[idx, b] - lo[b]) / (hi[b] - lo[b]) * 2 * n[b]).astype(int)
    lin = ga * (2 * n[b] + 1) + gb
    Kt = (np.kron(k1(lo[a], hi[a], n[a]), m1(lo[b], hi[b], n[b])) + np.kron(m1(lo[a], hi[a], n[a]), k1(lo[b], hi[b], n[b])))[np.ix_(lin, lin)]
    d = np.abs(S - Kt).max() / np.abs(Kt).max()
    rec(tag, "K_face = K1(a)(x)M1(b) + M1(a)(x)K1(b) (rel. max entrywise)", f"{d:.2e}", "<= 1e-14", d <= 1e-14)
    # negative controls on the assembly: a 1D-tangent analogue (one in-plane direction only) and the
    # reference-derivative stiffness without the metric inverse (G = I: wrong scaling on h != 2)
    K1dir = np.kron(k1(lo[a], hi[a], n[a]), m1(lo[b], hi[b], n[b]))[np.ix_(lin, lin)]
    dn1 = np.abs(S - K1dir).max() / np.abs(Kt).max()
    rec(tag, "NEG one-direction (1D-tangent analogue) K: detected", f"{dn1:.1e}", "> 1e-6", dn1 > 1e-6)
    ha, hb = (hi[a] - lo[a]) / n[a], (hi[b] - lo[b]) / n[b]
    Knom = (np.kron(k1(lo[a], hi[a], n[a]) * (ha / 2) ** 2, m1(lo[b], hi[b], n[b]))
            + np.kron(m1(lo[a], hi[a], n[a]), k1(lo[b], hi[b], n[b]) * (hb / 2) ** 2))[np.ix_(lin, lin)]
    dn2 = np.abs(S - Knom).max() / np.abs(Kt).max()
    rec(tag, "NEG no metric inverse (reference derivatives) K: detected", f"{dn2:.1e}", "> 1e-6", dn2 > 1e-6)
    store[(face, n, dom, scheme)] = (S, ga, gb)
    # ---- restriction to C
    off = np.setdiff1d(np.arange(N), Cl)
    rec(tag, "eliminated K_Gamma rows/cols off C zero (perimeter inactive)",
        int((abs(Ke[off]).sum(1).A1 > 0).sum() + (abs(Ke[:, off]).sum(0).A1 > 0).sum()), 0,
        (abs(Ke[off]).sum() == 0) and (abs(Ke[:, off]).sum() == 0))
    KC, KfC = Ke[Cl][:, Cl].toarray(), Kf[Cl][:, Cl].toarray()
    rec(tag, "K_Gamma|_C = K_face|_C (exact restriction)", f"{abs(KC - KfC).max():.1e}", 0, abs(KC - KfC).max() == 0)
    MC = Me[Cl][:, Cl].toarray()
    eK, eM = np.linalg.eigvalsh(KC)[0], np.linalg.eigvalsh(MC)[0]
    rec(tag, "K_Gamma|_C SPD (H1_0): lambda_min", f"{eK:.3e}", "> 0", eK > 0)
    rec(tag, "M_Gamma|_C SPD: lambda_min", f"{eM:.3e}", "> 0", eM > 0)
    for ell in (0.0, 0.1, 0.3, 1.0, 3.0, 10.0):
        e = np.linalg.eigvalsh(MC + ell * ell * KC)[0]
        rec(tag, f"R_Gamma = M + ell^2 K SPD on C, ell = {ell:g}: lambda_min", f"{e:.3e}", "> 0", e > 0)
    # ---- R_H1t
    nC = len(Cl)
    w = np.ones(steps)
    if scheme == "cn": w[-1] = 0.5
    T = 2 * np.eye(steps) - np.eye(steps, k=1) - np.eye(steps, k=-1); T[-1, -1] = 1
    RTref = np.kron(np.diag(w), MC + ELL * ELL * KC) + GAM / (AU * dt * dt) * np.kron(T, MC)
    RTd = RT.toarray()
    d = np.abs(RTd - RTref).max() / np.abs(RTref).max()
    rec(tag, "sparse R_H1t = What(x)R + gamma/(alpha dt^2) T(x)M (rel.)", f"{d:.1e}", "<= 1e-14", d <= 1e-14)
    rec(tag, "R_H1t symmetric", f"{np.abs(RTd - RTd.T).max():.1e}", 0, np.abs(RTd - RTd.T).max() <= 1e-15 * np.abs(RTd).max())
    e = np.linalg.eigvalsh(RTd)[0]
    rec(tag, "R_H1t SPD: lambda_min", f"{e:.3e}", "> 0", e > 0)
    # double count guard: the time term carries M_Gamma, not R_Gamma
    RTwrong = np.kron(np.diag(w), MC + ELL * ELL * KC) + GAM / (AU * dt * dt) * np.kron(T, MC + ELL * ELL * KC)
    dw = np.abs(RTd - RTwrong).max() / np.abs(RTref).max()
    rec(tag, "NEG time term with R_Gamma instead of M_Gamma (double count): detected", f"{dw:.1e}", "> 1e-6", dw > 1e-6)
    for lab, wrong in (("wrong sign of the H1-time term", np.kron(np.diag(w), MC + ELL * ELL * KC) - GAM / (AU * dt * dt) * np.kron(T, MC)),
                       ("H1-time weight gamma/(alpha dt) (missing 1/dt)", np.kron(np.diag(w), MC + ELL * ELL * KC) + GAM / (AU * dt) * np.kron(T, MC)),
                       ("H1-time without the natural last diagonal (T_NN = 2)",
                        np.kron(np.diag(w), MC + ELL * ELL * KC) + GAM / (AU * dt * dt) * np.kron(T + np.diag(np.eye(steps)[-1]), MC))):
        dd = np.abs(RTd - wrong).max() / np.abs(RTref).max()
        rec(tag, f"NEG {lab}: detected", f"{dd:.1e}", "> 1e-6", dd > 1e-6)
    # ---- preallocation
    rmaxK = int(np.diff(Kf.indptr).max()); rmaxRT = int(np.diff(RT.tocsr().indptr).max())
    rec(tag, "PETSc mallocs during MatSetValues (all matrices)", sum(mallocs), 0, len(mallocs) > 0 and sum(mallocs) == 0)
    rec(tag, "max row nnz K_Gamma_face / R_H1t (used)", f"{rmaxK}/{rmaxRT}", f"<= alloc {max(24, 3 * rmaxK)}", rmaxRT <= max(24, 3 * rmaxK))
    return tag


U = (0, 1, 0, 1, 0, 1)
for f in FACES:
    case(f, U, (3, 3, 3))
for f in ("zmin", "xmax", "ymax"):
    case(f, U, (2, 3, 4), scheme="cn")
for f in ("zmin", "xmin", "ymax", "zmax"):
    case(f, (0, 2, -1, 1, 0, .5), (4, 4, 2), scheme="cn", nt=4)
# orientation invariance: opposite faces of the same axis on the unit cube, matched by in-plane grid
for lo_f, hi_f in (("xmin", "xmax"), ("ymin", "ymax"), ("zmin", "zmax")):
    S1, a1, b1 = store[(lo_f, (3, 3, 3), U, "be")]; S2, a2, b2 = store[(hi_f, (3, 3, 3), U, "be")]
    o1 = np.lexsort((b1, a1)); o2 = np.lexsort((b2, a2))
    d = np.abs(S1[np.ix_(o1, o1)] - S2[np.ix_(o2, o2)]).max() / np.abs(S1).max()
    rec(f"{lo_f}/{hi_f}_3x3x3", "orientation: K on opposite faces (coordinate-matched)", f"{d:.1e}", "<= 1e-14", d <= 1e-14)
with open(sys.argv[1], "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("T11H1_KS_RESULT", "PASS" if ok else "FAIL", f"({sum(r['verdict'] == 'PASS' for r in rows)}/{len(rows)})")
sys.exit(0 if ok else 1)
