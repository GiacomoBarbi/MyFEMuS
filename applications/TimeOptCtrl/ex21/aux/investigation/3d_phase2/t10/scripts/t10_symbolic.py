#!/usr/bin/env python3
"""T10: symbolic verification of the 3D exact natural-BC distributed OC (natural-conv-3d).

  R_k(s) = exp(-k s) (2 + cos(pi s)),  k_i = w_i / nu
  p = alpha_u B(t) R_kx(x) R_ky(y) R_kz(z),  B = sin(T - t)
  y = A(t) cos(pi x) cos(pi y) cos(pi z),    A = cos(2t)
  u = -(eta/alpha_u) p
  f   = y_t - nu Lap y + w.grad y - eta u
  y_d = y + (p_t + nu Lap p + w.grad p)/alpha_t
Checks (exact sympy simplification unless stated):
  N1 state PDE residual = 0            N2 adjoint PDE residual = 0
  N3 nu dn y = 0 on the six faces      N4 nu dn p + (w.n) p = 0 on the six faces
  N5 p(T) = 0                          N6 alpha_u u + eta p = 0
  N7 the C++ literal port (ExactOc3D.cpp NatR/NatC/Product, Source3D, Target3D)
     equals sympy at 375 points (max abs difference / max |field| <= 1e-13; the
     scale matters: k = w/nu = -8 makes R ~ e^8)
  N8 max over each face of |(w.n) p| (> 0 on every face: the wrong adjoint BC
     nu dn p = 0 is then violated there and detectable)
  N9 the WRONG adjoint BC residual nu dn p on each face is nonzero
Winds: two numeric winds (default (1, 0.5, -0.75) and (-2, 1, 0.5)), zero wind
(checks N1-N7), and a symbolic (vx, vy, vz) for N1-N6.
"""
import math
import sys

import sympy as sp

x, y, z, t = sp.symbols("x y z t", real=True)
VX, VY, VZ = sp.symbols("v_x v_y v_z", real=True)
nu, T, eta, alpha_u, alpha_t = sp.Rational(1, 4), 1, 1, sp.Rational(1, 100), 1
pi = sp.pi
ok = True
rows = []


def report(name, value, tol=0.0, expect_nonzero=False):
    global ok
    if isinstance(value, sp.Basic) and not value.is_number:
        v = sp.simplify(value)
        good = (v == 0) if not expect_nonzero else (v != 0)
        shown = str(v)[:40]
    else:
        v = abs(float(value))
        good = (v <= tol) if not expect_nonzero else (v > 1e-8)
        shown = f"{v:.3e}"
    ok &= good
    print(f"T10_SYM {name:<64s} {shown:<12s} {'PASS' if good else 'FAIL'}")
    rows.append((name, shown, "PASS" if good else "FAIL"))


def R(k, s):
    return sp.exp(-k * s) * (2 + sp.cos(pi * s))


def check(w, tag):
    vx, vy, vz = w
    A, B = sp.cos(2 * t), sp.sin(T - t)
    Y = A * sp.cos(pi * x) * sp.cos(pi * y) * sp.cos(pi * z)
    P = alpha_u * B * R(vx / nu, x) * R(vy / nu, y) * R(vz / nu, z)
    U = -eta * P / alpha_u
    lap = lambda q: sp.diff(q, x, 2) + sp.diff(q, y, 2) + sp.diff(q, z, 2)
    adv = lambda q: vx * sp.diff(q, x) + vy * sp.diff(q, y) + vz * sp.diff(q, z)
    f = sp.diff(Y, t) - nu * lap(Y) + adv(Y) - eta * U
    yd = Y + (sp.diff(P, t) + nu * lap(P) + adv(P)) / alpha_t
    report(f"N1 {tag} state residual", sp.simplify(sp.diff(Y, t) - nu * lap(Y) + adv(Y) - f - eta * U))
    report(f"N2 {tag} adjoint residual",
           sp.simplify(-sp.diff(P, t) - nu * lap(P) - adv(P) - alpha_t * (Y - yd)))
    faces = [(x, 0, -1, 0), (x, 1, 1, 0), (y, 0, -1, 1), (y, 1, 1, 1), (z, 0, -1, 2), (z, 1, 1, 2)]
    for var, val, sgn, ax in faces:
        dn = lambda q: sgn * sp.diff(q, var)
        wn = sgn * w[ax]
        report(f"N3 {tag} nu dn y on {var}={val}", sp.simplify((nu * dn(Y)).subs(var, val)))
        report(f"N4 {tag} nu dn p + (w.n) p on {var}={val}",
               sp.simplify((nu * dn(P) + wn * P).subs(var, val)))
    report(f"N5 {tag} p(T)", sp.simplify(P.subs(t, T)))
    report(f"N6 {tag} stationarity", sp.simplify(alpha_u * U + eta * P))
    if any(isinstance(c, sp.Symbol) for c in w):
        return
    fw = [float(c) for c in w]

    # ---- N7: literal port of ExactOc3D.cpp (natural-conv-3d branch)
    def natr(k, s):
        e = math.exp(-k * s)
        g, dg, ddg = 2 + math.cos(math.pi * s), -math.pi * math.sin(math.pi * s), -math.pi ** 2 * math.cos(math.pi * s)
        return e * g, e * (dg - k * g), e * (ddg - 2 * k * dg + k * k * g)

    def natc(s):
        return math.cos(math.pi * s), -math.pi * math.sin(math.pi * s), -math.pi ** 2 * math.cos(math.pi * s)

    def prod(X, Yy, Z):
        return (X[0] * Yy[0] * Z[0],
                fw[0] * X[1] * Yy[0] * Z[0] + fw[1] * X[0] * Yy[1] * Z[0] + fw[2] * X[0] * Yy[0] * Z[1],
                X[2] * Yy[0] * Z[0] + X[0] * Yy[2] * Z[0] + X[0] * Yy[0] * Z[2])

    def cpp(xv, yv, zv, tv):
        nuf, au = 0.25, 0.01
        Av, dAv, Bv, dBv = math.cos(2 * tv), -2 * math.sin(2 * tv), math.sin(1 - tv), -math.cos(1 - tv)
        Ys = prod(natc(xv), natc(yv), natc(zv))
        Ps = prod(natr(fw[0] / nuf, xv), natr(fw[1] / nuf, yv), natr(fw[2] / nuf, zv))
        u = -1.0 * (au * Bv * Ps[0]) / au
        src = dAv * Ys[0] - nuf * Av * Ys[2] + Av * Ys[1] - u
        q = au * (dBv * Ps[0] + nuf * Bv * Ps[2] + Bv * Ps[1])
        return src, Av * Ys[0] + q

    fl, ydl = sp.lambdify((x, y, z, t), f), sp.lambdify((x, y, z, t), yd)
    wf = wy = sf = sy = 0.0
    pts = [0.0, 0.17, 0.5, 0.83, 1.0]
    for xv in pts:
        for yv in pts:
            for zv in pts:
                for tv in (0.0, 0.4, 1.0):
                    a, b = cpp(xv, yv, zv, tv)
                    fv, yv_ = fl(xv, yv, zv, tv), ydl(xv, yv, zv, tv)
                    wf, sf = max(wf, abs(a - fv)), max(sf, abs(fv))
                    wy, sy = max(wy, abs(b - yv_)), max(sy, abs(yv_))
    # relative to the field scale: with k = w/nu up to 8 the profiles reach e^8
    report(f"N7 {tag} C++ source vs sympy (375 pts, max abs / max|f|)", wf / sf, 1e-13)
    report(f"N7 {tag} C++ target vs sympy (375 pts, max abs / max|y_d|)", wy / sy, 1e-13)
    if all(c == 0 for c in fw):
        return
    # ---- N8 / N9: |(w.n) p| and the wrong-BC residual nu dn p on every face
    Pl = sp.lambdify((x, y, z, t), P)
    dP = [sp.lambdify((x, y, z, t), sp.diff(P, v)) for v in (x, y, z)]
    grid = [i / 20 for i in range(21)]
    for var, val, sgn, ax in faces:
        wn = sgn * fw[ax]
        m_wnp = m_wrong = 0.0
        for a_ in grid:
            for b_ in grid:
                for tv in (0.0, 0.5):
                    pt = {0: (val, a_, b_), 1: (a_, val, b_), 2: (a_, b_, val)}[ax]
                    m_wnp = max(m_wnp, abs(wn * Pl(*pt, tv)))
                    m_wrong = max(m_wrong, abs(0.25 * sgn * dP[ax](*pt, tv)))
        report(f"N8 {tag} max|(w.n) p| on {var}={val} (must be > 0)", m_wnp, expect_nonzero=True)
        report(f"N9 {tag} max|nu dn p| (wrong BC residual) on {var}={val}", m_wrong, expect_nonzero=True)


winds = [(1.0, 0.5, -0.75), (-2.0, 1.0, 0.5)]
if len(sys.argv) >= 4:
    winds[0] = tuple(float(v) for v in sys.argv[1:4])
check((0, 0, 0), "w=0")
for w in winds:
    check(tuple(sp.nsimplify(v) for v in w), "w=(" + ",".join(f"{v:g}" for v in w) + ")")
check((VX, VY, VZ), "w=symbolic")
print("T10_SYM_RESULT", "PASS" if ok else "FAIL", f"({sum(r[2] == 'PASS' for r in rows)}/{len(rows)})")
sys.exit(0 if ok else 1)
