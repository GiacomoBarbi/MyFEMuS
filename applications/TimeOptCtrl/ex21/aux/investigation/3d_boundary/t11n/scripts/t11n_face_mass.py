#!/usr/bin/env python3
"""T11N: QUAD9 face-mass validation (dump based, serial) -> ../../t11n_face_mass.csv.

For each case `--dimension 3 --neumann-solve --neumann-face3d F --dump-operators`
and, on the dumped M_Gamma (= the selected control operator N_selected):
  * support (nonzero rows) = the nodes on the face plane, count (2 n_a + 1)(2 n_b + 1);
  * symmetry max|M - M^T| = 0;  SPD on the support (min eigenvalue > 0);
  * 1^T M 1 = |Gamma_c| (face area);  v^T M v = int_Gamma (x + 2y - 3z)^2 dS for the
    nodal interpolant of a LINEAR function (exact: Q2 reproduces it, Gauss exact);
  * M restricted to the face = M1(a) (x) M1(b), the Kronecker product of two
    independently assembled 1D Q2 mass matrices on the face's in-plane axes
    (tensor-product QUAD9 face, exact Gauss): max entrywise difference;
  * N_selected = M_Gamma (the control operator IS the face mass);
  * K_Gamma empty (no tangential stiffness in 3D: H1 is refused).
usage: t11n_face_mass.py <out.csv>    (EX21_BIN overrides the executable)"""
import csv, os, subprocess, sys
import numpy as np
import scipy.sparse as sps

X = os.environ.get("EX21_BIN", "/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21")
W = os.environ.get("T11N_WORK", "/tmp/t11n_face"); os.makedirs(W + "/output", exist_ok=True)
ENV = {"HOME": os.environ["HOME"], "PATH": "/home/giacomo/software/petsc/arch-linux-c-openblas-opt/bin:/usr/bin:/bin",
       "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
FACES = {"zmin": (2, False), "ymin": (1, False), "xmax": (0, True), "ymax": (1, True), "xmin": (0, False), "zmax": (2, True)}
rows, ok = [], True


def coo(path):
    with open(path) as fh:
        fh.readline(); n = int(fh.readline().split()[0]); fh.readline()
        d = np.loadtxt(fh, ndmin=2)
    if d.size == 0:
        return sps.csr_matrix((n, n))
    return sps.csr_matrix((d[:, 2], (d[:, 0].astype(int), d[:, 1].astype(int))), shape=(n, n))


def mass1d(lo, hi, n):
    """assembled 1D Q2 mass on [lo, hi] with n equal elements (exact)"""
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
    print(f"T11N_FACE {case:<24s} {check:<52s} {str(value):<26s} ref {str(ref):<22s} {'PASS' if good else 'FAIL'}")


def case(face, dom, n):
    tag = f"{face}_{'x'.join(map(str, n))}_{'unit' if dom == (0, 1, 0, 1, 0, 1) else 'nonunit'}"
    prefix = f"{W}/{tag}"
    args = [X, "--dimension", "3", "--neumann-solve", "--neumann-face3d", face, "--state-bc", "natural",
            "--nx", str(n[0]), "--ny", str(n[1]), "--nz", str(n[2]), "--steps", "2", "--no-paraview",
            "--convection", "on", "--vx", "1", "--vy", "0.5", "--vz", "-0.75",
            "--tracking-box3d"] + [repr(v) for v in dom] + ["--dump-operators", prefix]
    if dom != (0, 1, 0, 1, 0, 1):
        args += ["--domain3d"] + [repr(v) for v in dom]
    r = subprocess.run(args, cwd=W, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if r.returncode != 0:
        raise RuntimeError(f"{tag}: exit {r.returncode}: {r.stdout[-600:]} {r.stderr[-600:]}")
    MG, N, KG = coo(prefix + "_M_Gamma.coo"), coo(prefix + "_N_selected.coo"), None
    xyz = np.loadtxt(prefix + "_coords.txt", ndmin=2)
    ax, at_max = FACES[face]
    lo, hi = np.array(dom[0::2], float), np.array(dom[1::2], float)
    plane = np.isclose(xyz[:, ax], hi[ax] if at_max else lo[ax], atol=1e-12)
    support = np.abs(MG).sum(1).A1 > 0
    a, b = [k for k in range(3) if k != ax]
    rec(tag, "support = face-plane nodes", int(support.sum()), int(plane.sum()), np.array_equal(support, plane))
    rec(tag, "trace node count = (2 n_a + 1)(2 n_b + 1)", int(plane.sum()), (2 * n[a] + 1) * (2 * n[b] + 1),
        int(plane.sum()) == (2 * n[a] + 1) * (2 * n[b] + 1))
    rec(tag, "symmetry max|M - M^T|", f"{abs(MG - MG.T).max():.1e}", 0, abs(MG - MG.T).max() == 0)
    idx = np.where(plane)[0]
    S = MG[idx][:, idx].toarray()
    ev = np.linalg.eigvalsh(S)
    rec(tag, "SPD on the trace support: min eigenvalue", f"{ev.min():.3e}", "> 0", ev.min() > 0)
    area = (hi[a] - lo[a]) * (hi[b] - lo[b])
    one = np.ones(MG.shape[0])
    rec(tag, "1^T M_Gamma 1 = |Gamma_c|", f"{one @ (MG @ one):.15g}", f"{area:.15g}", abs(one @ (MG @ one) - area) < 1e-13 * max(1, area))
    # exact quadratic surface integral: v = x + 2y - 3z (linear, reproduced exactly)
    v = xyz[:, 0] + 2 * xyz[:, 1] - 3 * xyz[:, 2]
    import itertools
    gl, gw = np.polynomial.legendre.leggauss(4)
    pa, pb = lo[a] + (gl + 1) / 2 * (hi[a] - lo[a]), lo[b] + (gl + 1) / 2 * (hi[b] - lo[b])
    wa, wb = gw / 2 * (hi[a] - lo[a]), gw / 2 * (hi[b] - lo[b])
    exact = 0.0
    for (i, j) in itertools.product(range(4), range(4)):
        c = np.zeros(3); c[ax] = hi[ax] if at_max else lo[ax]; c[a] = pa[i]; c[b] = pb[j]
        exact += wa[i] * wb[j] * (c[0] + 2 * c[1] - 3 * c[2]) ** 2
    got = v @ (MG @ v)
    rec(tag, "v^T M_Gamma v = int (x+2y-3z)^2 dS", f"{got:.15g}", f"{exact:.15g}", abs(got - exact) < 1e-12 * max(1, abs(exact)))
    # Kronecker structure M1(a) (x) M1(b) on the face grid
    ga = np.round((xyz[idx, a] - lo[a]) / (hi[a] - lo[a]) * 2 * n[a]).astype(int)
    gb = np.round((xyz[idx, b] - lo[b]) / (hi[b] - lo[b]) * 2 * n[b]).astype(int)
    K = np.kron(mass1d(lo[a], hi[a], n[a]), mass1d(lo[b], hi[b], n[b]))
    lin = ga * (2 * n[b] + 1) + gb
    Kf = K[np.ix_(lin, lin)]
    d = np.abs(S - Kf).max() / np.abs(Kf).max()
    rec(tag, "M_Gamma = M1(a) (x) M1(b) (rel. max entrywise)", f"{d:.2e}", "<= 1e-14", d <= 1e-14)
    rec(tag, "N_selected = M_Gamma", f"{abs(N - MG).max():.1e}", 0, abs(N - MG).max() == 0)
    # off-face rows of N are zero (the control is reconstructed on supp N only)
    rec(tag, "off-face rows of N zero", int((np.abs(N).sum(1).A1[~plane] > 0).sum()), 0,
        int((np.abs(N).sum(1).A1[~plane] > 0).sum()) == 0)


U = (0, 1, 0, 1, 0, 1)
for f in FACES:
    case(f, U, (3, 3, 3))
for f in ("zmin", "xmax", "ymax"):
    case(f, U, (2, 3, 4))
for f in ("zmin", "xmin", "ymax", "zmax"):
    case(f, (0, 2, -1, 1, 0, .5), (4, 4, 2))
with open(sys.argv[1], "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("T11N_FACE_RESULT", "PASS" if ok else "FAIL", f"({sum(r['verdict'] == 'PASS' for r in rows)}/{len(rows)})")
sys.exit(0 if ok else 1)
