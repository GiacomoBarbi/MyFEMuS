#!/usr/bin/env python3
"""T12 spatial ladders -> ../../t12_tri_space.csv (quad vs tri, conv-sin3) and ../../t12_tet_space.csv (hex vs tet,
conv-sin3-3d).  Errors y, p, u in L2(0,T;L2(Omega)) from EX21_EXACT_RESULT / EX21_EXACT3D_RESULT (FEM-integrated
against the exact solution with the element's own quadrature), orders vs h = 1/n AND vs the spatial DOF count
(order_dofs = -d log e / d log N^{1/dim}, identical to the h-order here because P2 and Q2 share the lattice);
elements, DOFs, nnz of the eliminated K (EX21 operator statistics are not printed -> from the dump battery),
outer iterations, spatial-solve iterations, wall, MaxRSS."""
import csv, glob, math, os, re
D = os.path.dirname(os.path.abspath(__file__)) + "/.."
def load(tag):
    rows = []
    for f in glob.glob(f"{D}/logs/exact/{tag}_*_n*_nt*.log"):
        s = open(f).read()
        m = re.search(r"EX21_EXACT3?D?_RESULT .*", s)
        if not m: continue
        L = m.group(0); g = lambda k: re.search(rf" {k}=(\S+)", L).group(1)
        mesh = re.search(r"EX21_MESH type=(\S+) dim=(\d) elements=(\d+) nodes_biquadratic=(\d+) dofs_type1=(\d+) dofs_type2=(\d+)", s)
        mt = mesh.group(1); dim = int(mesh.group(2))
        n = int(re.search(r"_n(\d+)_", f).group(1))
        r = {"ladder": tag, "mesh": mt, "dim": dim, "n": n, "nt": int(g("steps")), "scheme": g("scheme"),
             "elements": int(mesh.group(3)), "dofs": int(mesh.group(5)) if mt in ("tri", "tet") else int(mesh.group(6)),
             "h": 1.0 / n}
        for k in ("y_L2L2", "p_L2L2", "u_L2L2", "y_L2L2_rel", "p_L2L2_rel", "u_L2L2_rel"): r[k] = float(g(k))
        r["outer_its"] = g("outer_its"); r["true_residual"] = g("true_residual")
        ys = re.search(r"ysplit_its=(\d+) ysplit_max=(\d+)", s); r["ysplit_its_max"] = ys.group(2) if ys else ""
        mr = re.search(r"MAXRSS=(\d+) wall=(\S+)", s)
        r["maxrss_MB"], r["wall_s"] = int(mr.group(1)) // 1024, float(mr.group(2))
        rows.append(r)
    rows.sort(key=lambda r: r["n"])
    for i, r in enumerate(rows):
        for k in ("y", "p", "u"):
            if i == 0: r[f"ord_{k}_h"] = r[f"ord_{k}_dofs"] = ""; continue
            q = rows[i - 1]
            r[f"ord_{k}_h"] = f"{math.log(q[k + '_L2L2'] / r[k + '_L2L2']) / math.log(r['n'] / q['n']):.2f}"
            r[f"ord_{k}_dofs"] = f"{math.log(q[k + '_L2L2'] / r[k + '_L2L2']) / math.log((r['dofs'] / q['dofs']) ** (1.0 / r['dim'])):.2f}"
    return rows
def write(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
two = load("q2") + load("t2"); three = load("h3") + load("t3")
write(f"{D}/../t12_tri_space.csv", two); write(f"{D}/../t12_tet_space.csv", three)
for r in two + three:
    print(f"{r['mesh']:4s} n={r['n']:<3d} el={r['elements']:<7d} N={r['dofs']:<6d} y={r['y_L2L2']:.3e}({r['ord_y_h']:>5}) "
          f"p={r['p_L2L2']:.3e}({r['ord_p_h']:>5}) u={r['u_L2L2']:.3e}({r['ord_u_h']:>5}) out={r['outer_its']} ysplit={r['ysplit_its_max']} "
          f"MB={r['maxrss_MB']} wall={r['wall_s']}")
