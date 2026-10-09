#!/usr/bin/env python3
"""T13: multi-face trace topology and union face operators (dump based, serial).

For every face set F (one / opposite / adjacent / three mutually adjacent / U-shape / five / all six)
and mesh, run `--dimension 3 --{neumann,dirichlet}-solve --{neumann,dirichlet}-face3d F --dump-operators`
and check against INDEPENDENT references built here from the lattice:

  M_ref = sum_f  M1(a_f) (x) M1(b_f)             (1D Q2 masses, placed on face f's nodes)
  K_ref = sum_f [K1(a_f) (x) M1(b_f) + M1(a_f) (x) K1(b_f)]   (facewise surface stiffness)

so shared edge nodes receive the contributions of both faces and are counted ONCE as DOFs.

Neumann (-> t13_trace.csv / t13_face_mass.csv):
  support(M_Gamma) = nodes on the union, count predicted by lattice enumeration;
  M_Gamma = M_ref; N_selected = M_Gamma; 1^T M 1 = sum_f |Gamma_f|; v^T M v = sum_f int_f q^2 dS
  for q = x^2 + y z - 2 z^2 + x (biquadratic on every face -> reproduced exactly);
  M_Gamma(union) = sum_f M_Gamma(single face f) (separate runs; no double counting).
Dirichlet:
  C predicted geometrically = nodes on >= 1 controlled face and on NO uncontrolled face (internal
  edges/corners of the union active, any uncontrolled face wins with zero); closed-form count per class;
  C = support diag(M_Gamma); D = boundary \\ C; hatted eliminated set = D u C; C and D disjoint;
  M_Gamma_face (before elimination) = M_ref, K_Gamma_face = K_ref; R = M_ref[C, C] SPD;
  v^T R v = sum_f int_f phi^2, v^T K_Gamma v = sum_f int_f |grad_f phi|^2 for
  phi = prod_{uncontrolled f} dist_f (vanishes on every uncontrolled face, biquadratic per face).
usage: t13_trace.py <trace.csv> <face_mass.csv>"""
import csv, itertools, os, subprocess, sys
import numpy as np
import scipy.sparse as sps

X = os.environ.get("EX21_BIN", "/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21")
W = os.environ.get("T13_WORK", "/tmp/t13_trace"); os.makedirs(W + "/output", exist_ok=True)
ENV = {"HOME": os.environ["HOME"], "PATH": "/home/giacomo/software/petsc/arch-linux-c-openblas-opt/bin:/usr/bin:/bin",
       "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
FACES = {"zmin": (2, False), "ymin": (1, False), "xmax": (0, True), "ymax": (1, True), "xmin": (0, False), "zmax": (2, True)}
ORDER = ["zmin", "ymin", "xmax", "ymax", "xmin", "zmax"]   # FEMuS ids 1..6
trace_rows, mass_rows, ok = [], [], True
GL, GW = np.polynomial.legendre.leggauss(5)


def coo(path):
    with open(path) as fh:
        fh.readline(); r, c = map(int, fh.readline().split()); fh.readline()
        d = np.loadtxt(fh, ndmin=2)
    if d.size == 0:
        return sps.csr_matrix((r, c))
    return sps.csr_matrix((d[:, 2], (d[:, 0].astype(int), d[:, 1].astype(int))), shape=(r, c))


def q1d(lo, hi, n, which):
    h = (hi - lo) / n
    if which == "M":
        m = h / 30.0 * np.array([[4, 2, -1], [2, 16, 2], [-1, 2, 4]], float)
    else:
        m = 1.0 / (3.0 * h) * np.array([[7, -8, 1], [-8, 16, -8], [1, -8, 7]], float)
    A = np.zeros((2 * n + 1, 2 * n + 1))
    for e in range(n):
        idx = [2 * e, 2 * e + 1, 2 * e + 2]
        A[np.ix_(idx, idx)] += m
    return A


def rec(table, case, check, value, ref, good):
    global ok
    ok &= bool(good)
    table.append({"case": case, "check": check, "value": value, "reference": ref, "verdict": "PASS" if good else "FAIL"})
    print(f"T13_TRACE {case:<34s} {check:<62s} {str(value):<22s} ref {str(ref):<20s} {'PASS' if good else 'FAIL'}")


def run(kind, faces, dom, n, prefix, extra=()):
    lo, hi = np.array(dom[0::2], float), np.array(dom[1::2], float)
    mid = lambda k, f: repr(float(lo[k] + f * (hi[k] - lo[k])))
    args = [X, "--dimension", "3", f"--{kind}-solve", f"--{kind}-face3d", ",".join(faces),
            "--state-bc", "dirichlet" if kind == "dirichlet" else "natural",
            "--nx", str(n[0]), "--ny", str(n[1]), "--nz", str(n[2]), "--steps", "2", "--no-paraview",
            "--convection", "on", "--vx", "1", "--vy", "0.5", "--vz", "-0.75",
            "--tracking-box3d", mid(0, .25), mid(0, .75), mid(1, .25), mid(1, .75), mid(2, .25), mid(2, .75),
            "--dump-operators", prefix] + list(extra)
    if dom != (0, 1, 0, 1, 0, 1):
        args += ["--domain3d"] + [repr(v) for v in dom]
    r = subprocess.run(args, cwd=W, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if r.returncode != 0:
        raise RuntimeError(f"{prefix}: exit {r.returncode}: {r.stdout[-1500:]} {r.stderr[-1500:]} {args}")
    return r.stdout


def lattice(xyz, lo, hi, n):
    return np.stack([np.round((xyz[:, k] - lo[k]) / (hi[k] - lo[k]) * 2 * n[k]).astype(int) for k in range(3)], 1)


def face_nodes(L, n, face):
    ax, at_max = FACES[face]
    return L[:, ax] == (2 * n[ax] if at_max else 0)


def references(L, lo, hi, n, faces):
    N = len(L)
    M, K = sps.lil_matrix((N, N)), sps.lil_matrix((N, N))
    for f in faces:
        ax, _ = FACES[f]
        a, b = [k for k in range(3) if k != ax]
        idx = np.where(face_nodes(L, n, f))[0]
        lin = L[idx, a] * (2 * n[b] + 1) + L[idx, b]
        Ma, Mb = q1d(lo[a], hi[a], n[a], "M"), q1d(lo[b], hi[b], n[b], "M")
        Ka, Kb = q1d(lo[a], hi[a], n[a], "K"), q1d(lo[b], hi[b], n[b], "K")
        Mf = np.kron(Ma, Mb)[np.ix_(lin, lin)]
        Kf = (np.kron(Ka, Mb) + np.kron(Ma, Kb))[np.ix_(lin, lin)]
        M[np.ix_(idx, idx)] = M[np.ix_(idx, idx)].toarray() + Mf
        K[np.ix_(idx, idx)] = K[np.ix_(idx, idx)].toarray() + Kf
    return M.tocsr(), K.tocsr()


def face_integral(lo, hi, faces, fun):
    """sum_f int_{Gamma_f} fun(x, y, z) dS by tensor Gauss (5 points per axis)"""
    s = 0.0
    for f in faces:
        ax, at_max = FACES[f]
        a, b = [k for k in range(3) if k != ax]
        pa, pb = lo[a] + (GL + 1) / 2 * (hi[a] - lo[a]), lo[b] + (GL + 1) / 2 * (hi[b] - lo[b])
        wa, wb = GW / 2 * (hi[a] - lo[a]), GW / 2 * (hi[b] - lo[b])
        for i, j in itertools.product(range(len(GL)), range(len(GL))):
            c = np.zeros(3); c[ax] = hi[ax] if at_max else lo[ax]; c[a] = pa[i]; c[b] = pb[j]
            s += wa[i] * wb[j] * fun(c, f)
    return s


def rel(A, B):
    d = abs(A - B).max() if (A - B).nnz else 0.0
    return d / max(abs(B).max(), 1e-300)


def predicted_counts(n, faces, kind):
    """lattice enumeration (independent of the dump)"""
    cnt = 0
    for i, j, k in itertools.product(range(2 * n[0] + 1), range(2 * n[1] + 1), range(2 * n[2] + 1)):
        on = {"xmin": i == 0, "xmax": i == 2 * n[0], "ymin": j == 0, "ymax": j == 2 * n[1], "zmin": k == 0, "zmax": k == 2 * n[2]}
        if not any(on[f] for f in faces):
            continue
        if kind == "neumann" or not any(on[f] for f in FACES if f not in faces):
            cnt += 1
    return cnt


def closed_form(n, cls):
    """hand-derived Dirichlet |C| per class (unit-normal axes as in the representative face sets)"""
    nx, ny, nz = [2 * v for v in n]
    return {"one:zmin": (nx - 1) * (ny - 1),
            "opposite:zmin,zmax": 2 * (nx - 1) * (ny - 1),
            "adjacent:zmin,xmin": (ny - 1) * (nx + nz - 1),
            "three:zmin,ymin,xmin": (nx * ny + ny * nz + nx * nz) - (nx + ny + nz) + 1,
            "all6": (nx + 1) * (ny + 1) * (nz + 1) - (nx - 1) * (ny - 1) * (nz - 1)}.get(cls)


def case(kind, faces, dom, n, cls):
    unit = dom == (0, 1, 0, 1, 0, 1)
    tag = f"{kind[0].upper()}_{'+'.join(faces)}_{'x'.join(map(str, n))}{'' if unit else '_nonunit'}"
    prefix = f"{W}/{tag}"
    run(kind, faces, dom, n, prefix)
    lo, hi = np.array(dom[0::2], float), np.array(dom[1::2], float)
    xyz = np.loadtxt(prefix + "_coords.txt", ndmin=2)
    L = lattice(xyz, lo, hi, n)
    on = {f: face_nodes(L, n, f) for f in FACES}
    union = np.zeros(len(L), bool)
    for f in faces:
        union |= on[f]
    unctl = np.zeros(len(L), bool)
    for f in FACES:
        if f not in faces:
            unctl |= on[f]
    boundary = union | unctl
    Mref, Kref = references(L, lo, hi, n, faces)
    MG = coo(prefix + "_M_Gamma.coo")
    area = sum((hi[[k for k in range(3) if k != FACES[f][0]][0]] - lo[[k for k in range(3) if k != FACES[f][0]][0]]) *
               (hi[[k for k in range(3) if k != FACES[f][0]][1]] - lo[[k for k in range(3) if k != FACES[f][0]][1]]) for f in faces)
    pred = predicted_counts(n, faces, kind)
    cf = closed_form(n, cls) if kind == "dirichlet" and unit else None
    if kind == "neumann":
        support = np.abs(MG).sum(1).A1 > 0
        rec(trace_rows, tag, "trace DOFs = nodes on the union (lattice prediction)", int(support.sum()), pred, int(support.sum()) == pred)
        rec(trace_rows, tag, "support(M_Gamma) = union of face planes", "eq" if np.array_equal(support, union) else "differ", "eq",
            np.array_equal(support, union))
        ie = 0   # inclusion-exclusion over the faces: |union| = sum_{S != {}} (-1)^{|S|+1} |cap S|
        for r in range(1, len(faces) + 1):
            for S in itertools.combinations(faces, r):
                cap = np.ones(len(L), bool)
                for f in S:
                    cap &= on[f]
                ie += (-1) ** (r + 1) * int(cap.sum())
        rec(trace_rows, tag, "shared edges/corners counted once: |union| = incl.-excl.", int(support.sum()), ie, int(support.sum()) == ie)
        N = coo(prefix + "_N_selected.coo")
        rec(mass_rows, tag, "N_selected = M_Gamma", f"{abs(N - MG).max():.1e}", 0, abs(N - MG).max() == 0)
    else:
        with open(prefix + "_D_C.txt") as fh:
            fh.readline(); fh.readline()
            Cl = np.array(sorted(int(v) for v in fh.read().split() if not v.startswith("#")), int)
        want = np.where(union & ~unctl)[0]
        rec(trace_rows, tag, "|C| = lattice prediction (controlled, no uncontrolled face)", len(Cl), pred, len(Cl) == pred)
        if cf is not None:
            rec(trace_rows, tag, f"|C| = closed form ({cls})", len(Cl), cf, len(Cl) == cf)
        rec(trace_rows, tag, "C = geometric set exactly", "eq" if np.array_equal(Cl, want) else "differ", "eq", np.array_equal(Cl, want))
        support = np.where(MG.diagonal() != 0)[0]
        rec(trace_rows, tag, "C = support diag(M_Gamma)", "eq" if np.array_equal(Cl, support) else "differ", "eq", np.array_equal(Cl, support))
        internal = np.zeros(len(L), bool)
        for f, g in itertools.combinations(faces, 2):
            internal |= on[f] & on[g]
        internal &= ~unctl
        rec(trace_rows, tag, "internal edge/corner nodes of the union active (in C)", int(np.isin(np.where(internal)[0], Cl).sum()),
            int(internal.sum()), bool(np.isin(np.where(internal)[0], Cl).all()))
        rec(trace_rows, tag, "nodes on an uncontrolled face never in C", int(unctl[Cl].sum()), 0, int(unctl[Cl].sum()) == 0)
        Lf, Mf = coo(prefix + "_D_Lfull.coo"), coo(prefix + "_D_Mfull.coo")
        elim = lambda KK, MM: np.where(np.isclose(KK.diagonal(), 1.0) & np.isclose(abs(KK).sum(1).A1, 1.0)
                                       & (abs(MM).sum(1).A1 == 0))[0]
        Dset, Dwant = elim(Lf, Mf), np.where(boundary & ~np.isin(np.arange(len(L)), Cl))[0]
        rec(trace_rows, tag, "D (unhatted) = boundary \\ C", len(Dset), len(Dwant), np.array_equal(Dset, Dwant))
        rec(trace_rows, tag, "C and D disjoint", len(np.intersect1d(Cl, Dset)), 0, len(np.intersect1d(Cl, Dset)) == 0)
        H = elim(coo(prefix + "_K.coo"), coo(prefix + "_M.coo"))
        rec(trace_rows, tag, "hatted eliminated set = D u C", len(H), len(Dwant) + len(Cl), np.array_equal(H, np.union1d(Dwant, Cl)))
        MF, KF = coo(prefix + "_M_Gamma_face.coo"), coo(prefix + "_K_Gamma_face.coo")
        d = rel(MF, Mref)
        rec(mass_rows, tag, "M_Gamma (before elim.) = sum_f M1 (x) M1", f"{d:.2e}", "<= 1e-14", d <= 1e-14)
        d = rel(KF, Kref)
        rec(mass_rows, tag, "K_Gamma (before elim.) = sum_f [K1(x)M1 + M1(x)K1]", f"{d:.2e}", "<= 1e-13", d <= 1e-13)
        R = MG[Cl][:, Cl].toarray()
        d = np.abs(R - Mref[Cl][:, Cl].toarray()).max() / np.abs(R).max()
        rec(mass_rows, tag, "R = M_ref[C, C]", f"{d:.2e}", "<= 1e-14", d <= 1e-14)
        ev = np.linalg.eigvalsh(R)
        rec(mass_rows, tag, "R symmetric, SPD (min eig)", f"{ev.min():.3e}", "> 0", abs(R - R.T).max() == 0 and ev.min() > 0)
        dist = lambda c, f: (c[FACES[f][0]] - lo[FACES[f][0]]) if not FACES[f][1] else (hi[FACES[f][0]] - c[FACES[f][0]])
        phi = lambda c: np.prod([dist(c, g) for g in FACES if g not in faces]) if len(faces) < 6 else 1.0
        v = np.array([phi(c) for c in xyz])
        ex = face_integral(lo, hi, faces, lambda c, f: phi(c) ** 2)
        got = v[Cl] @ R @ v[Cl]
        rec(mass_rows, tag, "v^T R v = sum_f int phi^2 (phi = prod dist to uncontrolled)", f"{got:.15g}", f"{ex:.15g}",
            abs(got - ex) <= 1e-13 * max(ex, 1e-300))
        h = 1e-6
        def grad2(c, f):
            ax = FACES[f][0]
            s = 0.0
            for k in range(3):
                if k == ax:
                    continue
                e = np.zeros(3); e[k] = 1.0
                s += ((phi(c + h * e) - phi(c - h * e)) / (2 * h)) ** 2
            return s
        exk = face_integral(lo, hi, faces, grad2)
        KG = coo(prefix + "_K_Gamma.coo")
        gotk = v[Cl] @ (KG[Cl][:, Cl] @ v[Cl])
        rec(mass_rows, tag, "v^T K_Gamma v = sum_f int |grad_f phi|^2", f"{gotk:.15g}", f"{exk:.15g}",
            abs(gotk - exk) <= 1e-9 * max(exk, 1e-12) or (exk == 0 and abs(gotk) < 1e-12))
    d = rel(MG if kind == "neumann" else coo(prefix + "_M_Gamma_face.coo"), Mref)
    if kind == "neumann":
        rec(mass_rows, tag, "M_Gamma = sum_f M1(a) (x) M1(b) (rel. max entrywise)", f"{d:.2e}", "<= 1e-14", d <= 1e-14)
    Mu = MG if kind == "neumann" else coo(prefix + "_M_Gamma_face.coo")
    one = np.ones(Mu.shape[0])
    rec(mass_rows, tag, "1^T M_Gamma 1 = sum_f |Gamma_f|", f"{one @ (Mu @ one):.15g}", f"{area:.15g}",
        abs(one @ (Mu @ one) - area) <= 1e-13 * area)
    qf = lambda c: c[0] ** 2 + c[1] * c[2] - 2 * c[2] ** 2 + c[0]
    v = np.array([qf(c) for c in xyz])
    ex = face_integral(lo, hi, faces, lambda c, f: qf(c) ** 2)
    got = v @ (Mu @ v)
    rec(mass_rows, tag, "v^T M v = sum_f int (x^2+yz-2z^2+x)^2 dS", f"{got:.15g}", f"{ex:.15g}", abs(got - ex) <= 1e-12 * max(1, abs(ex)))
    if len(faces) > 1:   # union = sum of the single-face operators (separate runs)
        S = None
        for f in faces:
            p1 = f"{W}/{tag}__single_{f}"
            run(kind, [f], dom, n, p1)
            Mf = coo(p1 + ("_M_Gamma.coo" if kind == "neumann" else "_M_Gamma_face.coo"))
            S = Mf if S is None else S + Mf
        d = rel(Mu, S)
        rec(mass_rows, tag, "M_Gamma(union) = sum_f M_Gamma(single f) (separate runs)", f"{d:.2e}", "<= 1e-15", d <= 1e-15)
        Sdbl = S + coo(f"{W}/{tag}__single_{faces[0]}" + ("_M_Gamma.coo" if kind == "neumann" else "_M_Gamma_face.coo"))
        rec(mass_rows, tag, "NEGATIVE: double-counted face -> area mismatch", f"{one @ (Sdbl @ one):.6g}", f"!= {area:.6g}",
            abs(one @ (Sdbl @ one) - area) > 1e-3 * area)


SETS = [("one:zmin", ["zmin"]), ("one:xmax", ["xmax"]),
        ("opposite:zmin,zmax", ["zmin", "zmax"]), ("opposite:xmin,xmax", ["xmax", "xmin"]), ("opposite:ymin,ymax", ["ymin", "ymax"]),
        ("adjacent:zmin,xmin", ["zmin", "xmin"]), ("adjacent:xmax,ymax", ["xmax", "ymax"]), ("adjacent:ymin,zmax", ["ymin", "zmax"]),
        ("three:zmin,ymin,xmin", ["zmin", "ymin", "xmin"]), ("three:xmax,ymax,zmax", ["xmax", "ymax", "zmax"]),
        ("U:zmin,xmax,xmin", ["zmin", "xmax", "xmin"]), ("five:no-zmax", ["zmin", "ymin", "xmax", "ymax", "xmin"]),
        ("all6", list(ORDER))]
U = (0, 1, 0, 1, 0, 1)
for kind in ("neumann", "dirichlet"):
    for cls, fs in SETS:
        fs = sorted(fs, key=ORDER.index)
        case(kind, fs, U, (3, 3, 3), cls)
    for cls, fs in [SETS[3], SETS[5], SETS[8], SETS[12]]:
        case(kind, sorted(fs, key=ORDER.index), U, (2, 3, 4), cls)
    for cls, fs in [SETS[2], SETS[6], SETS[9]]:
        case(kind, sorted(fs, key=ORDER.index), (0, 2, -1, 1, 0, .5), (4, 4, 2), cls)
for path, rows in ((sys.argv[1], trace_rows), (sys.argv[2], mass_rows)):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("T13_TRACE_RESULT", "PASS" if ok else "FAIL",
      f"(trace {sum(r['verdict'] == 'PASS' for r in trace_rows)}/{len(trace_rows)}, mass {sum(r['verdict'] == 'PASS' for r in mass_rows)}/{len(mass_rows)})")
sys.exit(0 if ok else 1)
