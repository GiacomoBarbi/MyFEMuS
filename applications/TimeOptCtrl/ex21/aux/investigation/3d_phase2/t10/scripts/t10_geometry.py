#!/usr/bin/env python3
"""T10 3D geometry / region-mask gates (dump based, serial).

Every case runs `--dump-operators` (3D, distributed control) and checks, from the
assembled matrices and the per-element region file `<prefix>_regions3d.txt`:

  * element centroids = the analytic centroids of the structured box mesh;
  * the region indices of every element = an INDEPENDENT Python evaluation of
    the strict centroid rule with lowest-index precedence (tracking, control,
    source);
  * selected volumes 1^T M_region 1 = (#selected elements) * hx hy hz, and the
    geometric volume when the boxes are mesh-aligned;
  * union semantics: disjoint union = sum, overlapping union < sum and equal
    (entrywise) to the explicitly constructed union box;
  * the selected control operator N (N_selected): M for a full-domain box,
    M_ctrl_box for any other box -- including a box full in x and z but not in
    y (the ControlRegionIsWholeDomain 3D fix);
  * natural BC: no constrained row and (K + C) 1 = 0 (w.grad 1 = 0, K 1 = 0);
    Dirichlet: the constrained rows are exactly the boundary nodes;
  * non-unit rectangular domains: coordinates, total volume, boundary set.
usage: t10_geometry.py <out.csv>     (EX21_BIN overrides the executable)
"""
import csv
import itertools
import os
import subprocess
import sys

import numpy as np
import scipy.sparse as sps

X = os.environ.get("EX21_BIN", "/home/giacomo/software/FEMuS/femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21")
W = os.environ.get("T10_WORK", "/tmp/t10_geometry")
os.makedirs(W + "/output", exist_ok=True)
ENV = {"HOME": os.environ["HOME"], "PATH": "/home/giacomo/software/petsc/arch-linux-c-openblas-opt/bin:/usr/bin:/bin",
       "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
rows, ok = [], True


def record(case, check, value, reference, good, note=""):
    global ok
    ok &= bool(good)
    rows.append({"case": case, "check": check, "value": value, "reference": reference,
                 "verdict": "PASS" if good else "FAIL", "note": note})
    print(f"T10_GEOM {case:<26s} {check:<46s} {str(value):<24s} ref {str(reference):<24s} {'PASS' if good else 'FAIL'}")


def coo(path):
    with open(path) as fh:
        fh.readline()
        n = int(fh.readline().split()[0])
        fh.readline()
        d = np.loadtxt(fh, ndmin=2)
    if d.size == 0:
        return sps.csr_matrix((n, n))
    return sps.csr_matrix((d[:, 2], (d[:, 0].astype(int), d[:, 1].astype(int))), shape=(n, n))


def run(tag, args):
    prefix = f"{W}/{tag}"
    cmd = [X, "--dimension", "3", "--distributed-solve", "--steps", "2", "--no-paraview",
           "--dump-operators", prefix] + args
    r = subprocess.run(cmd, cwd=W, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if r.returncode != 0:
        raise RuntimeError(f"{tag}: exit {r.returncode}\n{r.stdout[-800:]}\n{r.stderr[-800:]}")
    M = {k: coo(f"{prefix}_{k}.coo") for k in ("M", "K", "M_track", "M_ctrl_box", "N_selected")}
    if os.path.exists(f"{prefix}_M_source.coo"):
        M["M_source"] = coo(f"{prefix}_M_source.coo")
    reg = np.loadtxt(f"{prefix}_regions3d.txt", ndmin=2)
    xyz = np.loadtxt(f"{prefix}_coords.txt", ndmin=2)
    return M, reg, xyz


def lowest_index(c, boxes):
    for k, b in enumerate(boxes):
        if b[0] < c[0] < b[1] and b[2] < c[1] < b[3] and b[4] < c[2] < b[5]:
            return k + 1
    return 0


def vol(A):
    return float(np.ones(A.shape[0]) @ (A @ np.ones(A.shape[0])))


def case(name, dom, n, track, ctrl, src=(), bc="natural", aligned_volumes=None, extra=()):
    """dom = (x0,x1,y0,y1,z0,z1); n = (nx,ny,nz); track/ctrl/src = lists of 6-tuples."""
    args = ["--nx", str(n[0]), "--ny", str(n[1]), "--nz", str(n[2]), "--state-bc", bc]
    if dom != (0, 1, 0, 1, 0, 1):
        args += ["--domain3d"] + [repr(v) for v in dom]
    args += ["--tracking-box3d"] + [repr(v) for v in track[0]]
    for b in track[1:]:
        args += ["--tracking-box3d-add"] + [repr(v) for v in b]
    args += ["--control-box3d"] + [repr(v) for v in ctrl[0]]
    for b in ctrl[1:]:
        args += ["--control-box3d-add"] + [repr(v) for v in b]
    if src:
        args += ["--source", "1"]
        for b in src:
            args += ["--source-box3d-add"] + [repr(v) for v in b]
    args += list(extra)
    M, reg, xyz = run(name, args)
    h = [(dom[2 * a + 1] - dom[2 * a]) / n[a] for a in range(3)]
    cell = h[0] * h[1] * h[2]
    # analytic centroids, in FEMuS element order matched by coordinates
    cent = reg[:, 1:4]
    ana = {(round(dom[0] + (i + .5) * h[0], 10), round(dom[2] + (j + .5) * h[1], 10), round(dom[4] + (k + .5) * h[2], 10))
           for i in range(n[0]) for j in range(n[1]) for k in range(n[2])}
    got = {tuple(np.round(c, 10)) for c in cent}
    record(name, "element centroids = analytic grid", len(got & ana), len(ana), got == ana and len(cent) == len(ana))
    for col, boxes, lab in ((4, track, "tracking"), (5, ctrl, "control"), (6, src, "source")):
        ind = np.array([lowest_index(c, boxes) for c in cent]) if boxes else np.zeros(len(cent))
        mism = int(np.sum(ind != reg[:, col]))
        record(name, f"{lab} index = independent centroid rule", mism, 0, mism == 0,
               f"selected={int(np.sum(ind > 0))} by_index={[int(np.sum(ind == q)) for q in range(1, len(boxes) + 1)]}")
    nsel = {lab: int(np.sum(reg[:, col] > 0)) for col, lab in ((4, "track"), (5, "ctrl"), (6, "src"))}
    if bc != "natural":   # Dirichlet-eliminated masses are not volumes
        return M, reg, xyz, cell
    record(name, "1^T M 1 = domain volume", f"{vol(M['M']):.15g}",
           f"{(dom[1]-dom[0])*(dom[3]-dom[2])*(dom[5]-dom[4]):.15g}",
           abs(vol(M["M"]) - (dom[1]-dom[0])*(dom[3]-dom[2])*(dom[5]-dom[4])) < 1e-12)
    for key, lab in (("M_track", "track"), ("M_ctrl_box", "ctrl"), ("M_source", "src")):
        if key not in M:
            continue
        v, ref = vol(M[key]), nsel[lab] * cell
        record(name, f"1^T {key} 1 = n_sel hx hy hz", f"{v:.15g}", f"{ref:.15g}", abs(v - ref) < 1e-13,
               f"n_sel={nsel[lab]}")
        if aligned_volumes and lab in aligned_volumes:
            record(name, f"1^T {key} 1 = geometric union volume", f"{v:.15g}", f"{aligned_volumes[lab]:.15g}",
                   abs(v - aligned_volumes[lab]) < 1e-13)
    return M, reg, xyz, cell


def main(out):
    U = (0, 1, 0, 1, 0, 1)
    full = [U]
    # G1 one aligned control box
    case("one_box", U, (4, 4, 4), full, [(0, .5, 0, .5, 0, .5)], aligned_volumes={"track": 1.0, "ctrl": .125})
    # G2 two disjoint boxes: union = sum
    Ma, *_ = case("box_A", U, (4, 4, 4), full, [(0, .5, 0, .5, 0, .5)])
    Mb, *_ = case("box_B", U, (4, 4, 4), full, [(.5, 1, .5, 1, .5, 1)])
    Mab, *_ = case("disjoint_AB", U, (4, 4, 4), full, [(0, .5, 0, .5, 0, .5), (.5, 1, .5, 1, .5, 1)],
                   aligned_volumes={"ctrl": .25})
    d = abs(Mab["M_ctrl_box"] - Ma["M_ctrl_box"] - Mb["M_ctrl_box"]).max()
    record("disjoint_AB", "M_ctrl(A u B) - M_ctrl(A) - M_ctrl(B) (max)", f"{d:.3e}", 0, d < 1e-15)
    # G3 two overlapping boxes: union != sum, = explicit union box
    Mo, reg_o, *_ = case("overlap_CD", U, (4, 4, 4), full, [(0, .75, 0, .5, 0, .5), (.25, 1, 0, .5, 0, .5)],
                         aligned_volumes={"ctrl": .25})
    Mc, *_ = case("box_C", U, (4, 4, 4), full, [(0, .75, 0, .5, 0, .5)])
    Md, *_ = case("box_D", U, (4, 4, 4), full, [(.25, 1, 0, .5, 0, .5)])
    Mu, *_ = case("union_box_CD", U, (4, 4, 4), full, [(0, 1, 0, .5, 0, .5)])
    s = vol(Mc["M_ctrl_box"]) + vol(Md["M_ctrl_box"])
    record("overlap_CD", "vol(C u D) < vol(C) + vol(D)", f"{vol(Mo['M_ctrl_box']):.15g}", f"{s:.15g}",
           vol(Mo["M_ctrl_box"]) < s - 1e-9, "overlap counted once")
    d = abs(Mo["M_ctrl_box"] - Mu["M_ctrl_box"]).max()
    record("overlap_CD", "M_ctrl(C u D) = M_ctrl(explicit union box)", f"{d:.3e}", 0, d == 0.0)
    prec = int(np.sum(reg_o[:, 5] == 1)), int(np.sum(reg_o[:, 5] == 2))
    record("overlap_CD", "lowest-index precedence: (#idx1, #idx2)", prec, (12, 4), prec == (12, 4),
           "C has 3x2x2=12 elements; D adds only x-centroid 0.875 (4)")
    # G4 full-domain-equivalent box: N = M (DISTRIBUTED_WHOLE); partial-y box: N = M_ctrl_box
    Mf, *_ = case("full_domain_box", U, (4, 4, 4), full, [U])
    d1 = abs(Mf["N_selected"] - Mf["M"]).max()
    d2 = abs(Mf["M_ctrl_box"] - Mf["M"]).max()
    record("full_domain_box", "N_selected = M ; M_ctrl_box = M (max)", f"{d1:.1e};{d2:.1e}", 0, d1 == 0 and d2 < 1e-15)
    Mp, *_ = case("partial_y_box", U, (4, 4, 4), full, [(0, 1, 0, .5, 0, 1)], aligned_volumes={"ctrl": .5})
    d1 = abs(Mp["N_selected"] - Mp["M_ctrl_box"]).max()
    d2 = abs(Mp["N_selected"] - Mp["M"]).max()
    record("partial_y_box", "N_selected = M_ctrl_box != M (3D whole-domain fix)", f"{d1:.1e};{d2:.3e}", "0;>0",
           d1 == 0 and d2 > 1e-6)
    # G5 non-aligned box, strict centroid rule (a centroid ON a face is outside)
    case("nonaligned_box", U, (4, 4, 4), full, [(.1, .6, .2, .9, .3, .7)])
    _, reg_f, *_ = case("centroid_on_face", U, (4, 4, 4), full, [(0, .375, 0, 1, 0, 1)])
    k = int(np.sum(reg_f[:, 5] > 0))
    record("centroid_on_face", "box x<0.375: centroids x=0.375 excluded", k, 16, k == 16,
           "only the x-centroid 0.125 layer (4x4) is selected")
    # G6 tracking boxes: one, two disjoint, overlapping (union = explicit union)
    Mt1, *_ = case("track_two_disjoint", U, (4, 4, 4), [(0, .25, 0, 1, 0, 1), (.75, 1, 0, 1, 0, 1)], [U],
                   aligned_volumes={"track": .5})
    Mt2, *_ = case("track_overlap", U, (4, 4, 4), [(0, .5, 0, 1, 0, .5), (0, .5, .25, 1, 0, 1)], [U])
    Mt3, *_ = case("track_union_explicit", U, (4, 4, 4), [(0, .5, 0, 1, 0, .5), (0, .5, .25, 1, .5, 1)], [U])
    # the overlap of the two boxes is (0,.5)x(.25,1)x(0,.5); union is not a single box, so compare the
    # overlapping description with a NON-overlapping partition of the same union
    d = abs(Mt2["M_track"] - Mt3["M_track"]).max()
    record("track_overlap", "M_track(overlapping) = M_track(disjoint partition of same union)", f"{d:.3e}", 0, d == 0.0)
    # G7 non-colocated control / tracking: 2 + 2 boxes, zero element overlap
    Mn, reg_n, *_ = case("noncolocated_2x2", U, (4, 4, 4),
                         [(0, .5, .5, 1, 0, .5), (.5, 1, 0, .5, .5, 1)],
                         [(0, .5, 0, .5, 0, .5), (.5, 1, .5, 1, .5, 1)], bc="natural",
                         aligned_volumes={"track": .25, "ctrl": .25})
    both = int(np.sum((reg_n[:, 4] > 0) & (reg_n[:, 5] > 0)))
    record("noncolocated_2x2", "elements in tracking AND control", both, 0, both == 0)
    had = abs(Mn["M_track"].multiply(Mn["M_ctrl_box"])).sum()
    shared = int(np.sum((abs(Mn["M_track"]).sum(1).A1 > 0) & (abs(Mn["M_ctrl_box"]).sum(1).A1 > 0)))
    record("noncolocated_2x2", "shared DOFs (box-face nodes only), info", shared, "-", True,
           f"sum|M_track o M_ctrl|={had:.3e} (interface entries)")
    # G8 sources: one, two, overlap counted once, whole domain = M
    Ms1, *_ = case("source_one", U, (4, 4, 4), full, [U], src=[(0, .25, 0, .25, 0, .25)], aligned_volumes={"src": 1 / 64})
    Ms2, *_ = case("source_two", U, (4, 4, 4), full, [U], src=[(0, .25, 0, .25, 0, .25), (.75, 1, .75, 1, .75, 1)],
                   aligned_volumes={"src": 2 / 64})
    Mso, *_ = case("source_overlap", U, (4, 4, 4), full, [U], src=[(0, .5, 0, .25, 0, .25), (.25, .75, 0, .25, 0, .25)],
                   aligned_volumes={"src": 3 / 64})
    Msu, *_ = case("source_overlap_union", U, (4, 4, 4), full, [U], src=[(0, .75, 0, .25, 0, .25)])
    d = abs(Mso["M_source"] - Msu["M_source"]).max()
    record("source_overlap", "M_source(overlap) = M_source(explicit union)", f"{d:.3e}", 0, d == 0.0)
    Msw, *_ = case("source_whole", U, (4, 4, 4), full, [U], src=[U])
    d = abs(Msw["M_source"] - Msw["M"]).max()
    record("source_whole", "M_source(full-domain box) = M", f"{d:.3e}", 0, d < 1e-16)
    # G9 natural vs Dirichlet state BC (conv on): constrained set and (K+C) 1 = 0
    wind = ["--convection", "on", "--vx", "1", "--vy", "0.5", "--vz", "-0.75"]
    Mnat, _, xyz, _ = case("natural_bc_wind", U, (3, 3, 3), full, [U], bc="natural", extra=wind)
    r1 = abs(Mnat["K"] @ np.ones(Mnat["K"].shape[0])).max()
    record("natural_bc_wind", "(K + C) 1 = 0 (no row constrained)", f"{r1:.3e}", 0, r1 < 1e-13)
    Mdir, _, xyz, _ = case("dirichlet_bc_wind", U, (3, 3, 3), full, [U], bc="dirichlet", extra=wind)
    diag = Mdir["K"].diagonal()
    unit = np.array([abs(Mdir["K"].getrow(i)).sum() == 1.0 and diag[i] == 1.0 for i in range(len(diag))])
    bnd = np.any((np.abs(xyz) < 1e-12) | (np.abs(xyz - 1) < 1e-12), axis=1)
    record("dirichlet_bc_wind", "constrained rows (K=e_i) = boundary nodes", int(unit.sum()), int(bnd.sum()),
           np.array_equal(unit, bnd))
    # G10 non-unit domains
    for nm, dom, n in (("nonunit_A", (0, 2, -1, 1, 0, .5), (4, 4, 2)), ("nonunit_B", (-1.5, .5, 2, 3, -.25, 1.75), (4, 2, 4))):
        Mx, reg_x, xyz, cell = case(nm, dom, n, [(dom[0], (dom[0] + dom[1]) / 2, dom[2], dom[3], dom[4], dom[5])],
                                    [(dom[0], dom[1], dom[2], (dom[2] + dom[3]) / 2, dom[4], dom[5])],
                                    src=[(dom[0], dom[1], dom[2], dom[3], dom[4], (dom[4] + dom[5]) / 2)], bc="natural",
                                    aligned_volumes={"track": .5 * (dom[1]-dom[0])*(dom[3]-dom[2])*(dom[5]-dom[4]),
                                                     "ctrl": .5 * (dom[1]-dom[0])*(dom[3]-dom[2])*(dom[5]-dom[4]),
                                                     "src": .5 * (dom[1]-dom[0])*(dom[3]-dom[2])*(dom[5]-dom[4])})
        lo, hi = xyz.min(0), xyz.max(0)
        good = np.allclose(lo, dom[0::2]) and np.allclose(hi, dom[1::2])
        record(nm, "coordinate extents = domain", f"{np.round(lo,6).tolist()}..{np.round(hi,6).tolist()}", dom, good)
        n_dofs = (2 * n[0] + 1) * (2 * n[1] + 1) * (2 * n[2] + 1)
        record(nm, "spatial DOFs = prod(2n+1)", xyz.shape[0], n_dofs, xyz.shape[0] == n_dofs)
        Md_, _, xyz, _ = case(nm + "_dirichlet", dom, n, [(dom[0], dom[1], dom[2], dom[3], dom[4], dom[5])],
                              [(dom[0], dom[1], dom[2], dom[3], dom[4], dom[5])], bc="dirichlet")
        diag = Md_["K"].diagonal()
        unit = np.array([abs(Md_["K"].getrow(i)).sum() == 1.0 and diag[i] == 1.0 for i in range(len(diag))])
        bnd = np.any(np.isclose(xyz, np.array(dom[0::2])) | np.isclose(xyz, np.array(dom[1::2])), axis=1)
        record(nm, "Dirichlet constrained rows = boundary nodes", int(unit.sum()), int(bnd.sum()), np.array_equal(unit, bnd))
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("T10_GEOM_RESULT", "PASS" if ok else "FAIL", f"({sum(r['verdict'] == 'PASS' for r in rows)}/{len(rows)})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
