#!/usr/bin/env python3
"""T9: symbolic verification of the 3D exact distributed OC (conv-sin-3d and conv-sin3-3d;
the sin^3 case replaces sin(pi .) by sin^3(pi .) in S).

  S = sin(pi x) sin(pi y) sin(pi z),  y = A(t) S,  A = cos 2t,
  p = alpha_u B(t) S,  B = sin(T - t),  u = -(eta/alpha_u) p,
  f   = y_t - nu Lap y + w.grad y - eta u,
  y_d = y + (p_t + nu Lap p + w.grad p)/alpha_t.
Checks (exact sympy simplification; every residual must be 0):
  K1 state PDE      y_t - nu Lap y + w.grad y - f - eta u = 0
  K2 adjoint PDE    -p_t - nu Lap p - w.grad p - alpha_t (y - y_d) = 0
  K3 homogeneous Dirichlet: y = p = u = 0 on all six faces
  K4 p(T) = 0
  K5 stationarity   alpha_u u + eta p = 0
  K6 the C++ literal port of f and y_d (ExactOc3D.cpp) = sympy, at 125 points
for w = 0 and for the nonzero wind given on the command line
(default (1, 0.5, -0.75)); also for a SYMBOLIC wind (vx, vy, vz).
usage: t9_symbolic.py [vx vy vz]
"""
import math
import sys

import sympy as sp

x, y, z, t = sp.symbols("x y z t", real=True)
VX, VY, VZ = sp.symbols("v_x v_y v_z", real=True)
nu, T, eta, alpha_u, alpha_t = sp.Rational(1, 4), 1, 1, sp.Rational(1, 100), 1
pi = sp.pi
ok = True


def report(name, value, tol=0.0):
    global ok
    v = abs(float(value)) if not isinstance(value, sp.Basic) or value.is_number else None
    good = (v is not None and v <= tol) or (v is None and sp.simplify(value) == 0)
    ok &= good
    shown = f"{v:.3e}" if v is not None else str(sp.simplify(value))
    print(f"T9_SYM {name:<60s} {shown:<12s} {'PASS' if good else 'FAIL'}")


def check(w, tag, cubic=False):
    vx, vy, vz = w
    e = 3 if cubic else 1
    tag = tag + (" sin^3" if cubic else " sin")
    S = sp.sin(pi * x) ** e * sp.sin(pi * y) ** e * sp.sin(pi * z) ** e
    A, B = sp.cos(2 * t), sp.sin(T - t)
    Y, P = A * S, alpha_u * B * S
    U = -eta * P / alpha_u
    lap = lambda q: sp.diff(q, x, 2) + sp.diff(q, y, 2) + sp.diff(q, z, 2)
    adv = lambda q: vx * sp.diff(q, x) + vy * sp.diff(q, y) + vz * sp.diff(q, z)
    f = sp.diff(Y, t) - nu * lap(Y) + adv(Y) - eta * U
    yd = Y + (sp.diff(P, t) + nu * lap(P) + adv(P)) / alpha_t
    report(f"K1 {tag} state residual", sp.simplify(sp.diff(Y, t) - nu * lap(Y) + adv(Y) - f - eta * U))
    report(f"K2 {tag} adjoint residual", sp.simplify(-sp.diff(P, t) - nu * lap(P) - adv(P) - alpha_t * (Y - yd)))
    for var in (x, y, z):
        for val in (0, 1):
            report(f"K3 {tag} y,p,u on {var}={val}",
                   sum(sp.Abs(sp.simplify(q.subs(var, val))) for q in (Y, P, U)))
    report(f"K4 {tag} p(T)", sp.simplify(P.subs(t, T)))
    report(f"K5 {tag} stationarity alpha_u u + eta p", sp.simplify(alpha_u * U + eta * P))
    if any(isinstance(c, sp.Symbol) for c in w):
        return
    # K6: literal port of ExactOc3D.cpp (Source3D, Target3D)
    fw = [float(c) for c in w]
    on = any(c != 0 for c in fw)

    def fac(xi):   # literal port of Factor() in ExactOc3D.cpp
        sn, cs = math.sin(math.pi * xi), math.cos(math.pi * xi)
        if not cubic:
            return sn, math.pi * cs, -math.pi ** 2 * sn
        return sn ** 3, 3 * math.pi * sn * sn * cs, 3 * math.pi ** 2 * sn * (2 * cs * cs - sn * sn)

    def wgs(xv, yv, zv):
        if not on:
            return 0.0
        if cubic:
            X, Y, Z = fac(xv), fac(yv), fac(zv)
            return fw[0] * X[1] * Y[0] * Z[0] + fw[1] * X[0] * Y[1] * Z[0] + fw[2] * X[0] * Y[0] * Z[1]
        sx, sy, sz = math.sin(math.pi * xv), math.sin(math.pi * yv), math.sin(math.pi * zv)
        cx, cy, cz = math.cos(math.pi * xv), math.cos(math.pi * yv), math.cos(math.pi * zv)
        return math.pi * (fw[0] * cx * sy * sz + fw[1] * sx * cy * sz + fw[2] * sx * sy * cz)

    def cpp(xv, yv, zv, tv):
        Av, dAv, Bv, dBv = math.cos(2 * tv), -2 * math.sin(2 * tv), math.sin(1 - tv), -math.cos(1 - tv)
        if cubic:
            X, Y, Z = fac(xv), fac(yv), fac(zv)
            s = X[0] * Y[0] * Z[0]
            mlap = -(X[2] * Y[0] * Z[0] + X[0] * Y[2] * Z[0] + X[0] * Y[0] * Z[2])
            uv = -1.0 * (0.01 * Bv * s) / 0.01
            src = dAv * s + 0.25 * Av * mlap + Av * wgs(xv, yv, zv) - uv
            q = 0.01 * (dBv * s - 0.25 * Bv * mlap + Bv * wgs(xv, yv, zv))
            return src, Av * s + q
        s = math.sin(math.pi * xv) * math.sin(math.pi * yv) * math.sin(math.pi * zv)
        lp = 3 * math.pi ** 2
        uv = -1.0 * (0.01 * Bv * s) / 0.01
        src = (dAv + 0.25 * lp * Av) * s + Av * wgs(xv, yv, zv) - uv
        q = 0.01 * ((dBv - 0.25 * lp * Bv) * s + Bv * wgs(xv, yv, zv))
        return src, Av * s + q

    fl, ydl = sp.lambdify((x, y, z, t), f), sp.lambdify((x, y, z, t), yd)
    wf = wy = 0.0
    pts = [0.0, 0.17, 0.5, 0.83, 1.0]
    for xv in pts:
        for yv in pts:
            for zv in pts:
                for tv in (0.0, 0.4, 1.0):
                    a, b = cpp(xv, yv, zv, tv)
                    wf = max(wf, abs(a - fl(xv, yv, zv, tv)))
                    wy = max(wy, abs(b - ydl(xv, yv, zv, tv)))
    report(f"K6 {tag} C++ source vs sympy (375 pts, abs)", wf, 1e-12)
    report(f"K6 {tag} C++ target vs sympy (375 pts, abs)", wy, 1e-12)


winds = [float(v) for v in sys.argv[1:4]] if len(sys.argv) >= 4 else [1.0, 0.5, -0.75]
for cubic in (False, True):
    check((0, 0, 0), "w=0", cubic)
    check(tuple(sp.nsimplify(v) for v in winds), "w=(" + ",".join(f"{v:g}" for v in winds) + ")", cubic)
    check((VX, VY, VZ), "w=symbolic", cubic)
print("T9_SYM_RESULT", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
