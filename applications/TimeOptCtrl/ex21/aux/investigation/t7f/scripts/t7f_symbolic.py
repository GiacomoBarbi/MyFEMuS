#!/usr/bin/env python3
"""T7F: symbolic verification of the natural-conv exact optimal control.

  p = alpha_u B(t) R_kx(x) R_kz(z),  R_k(s) = exp(-k s)(2 + cos(pi s)),
  k = (vx, vy)/nu,  B = sin(T - t);  y = A(t) cos(pi x) cos(pi z),  A = cos 2t;
  u = -(eta/alpha_u) p;  f = y_t - nu Lap y + w.grad y - eta u;
  y_d = y + (p_t + nu Lap p + w.grad p)/alpha_t.
Checks, for each wind given on the command line (default (1, 0.5) and (-1, -0.5)):
  N1  adjoint BC  nu dn p + (w.n) p = 0 on the four faces (symbolic)
  N2  the convective part is genuinely exercised: max |(w.n) p| on each face > 0,
      i.e. the WRONG condition nu dn p = 0 is violated by exactly that amount
  N3  state BC  nu dn y = 0 on the four faces (symbolic);  p(T) = 0
  N4  the C++ literal port (NatR, NatPParts, Source, Target) vs sympy at 3x5x5 points
  N5  optimality: adjoint residual -p_t - nu Lap p - w.grad p = alpha_t (y - y_d) and
      state residual y_t - nu Lap y + w.grad y = f + eta u, both identically 0
usage: t7f_symbolic.py [vx vy [vx vy ...]]
"""
import math
import sys

import sympy as sp

x, z, t = sp.symbols("x z t", real=True)
nu, T, eta, alpha_u, alpha_t = sp.Rational(1, 4), 1, 1, sp.Rational(1, 100), 1
pi = sp.pi
ok_all = True


def report(name, value, tol, larger=False):
    global ok_all
    v = abs(float(value))
    ok = v > tol if larger else v <= tol
    ok_all &= ok
    print(f"T7F_SYM {name:<58s} {v:.3e}  {'PASS' if ok else 'FAIL'} ({'>' if larger else '<='} {tol:.0e})")


def check(vx, vy):
    kx, kz = vx / nu, vy / nu
    R = lambda k, s: sp.exp(-k * s) * (2 + sp.cos(pi * s))
    B, A = sp.sin(T - t), sp.cos(2 * t)
    p = alpha_u * B * R(kx, x) * R(kz, z)
    y = A * sp.cos(pi * x) * sp.cos(pi * z)
    u = -eta * p / alpha_u
    lap = lambda q: sp.diff(q, x, 2) + sp.diff(q, z, 2)
    f = sp.diff(y, t) - nu * lap(y) + vx * sp.diff(y, x) + vy * sp.diff(y, z) - eta * u
    yd = y + (sp.diff(p, t) + nu * lap(p) + vx * sp.diff(p, x) + vy * sp.diff(p, z)) / alpha_t
    tag = f"w=({vx},{vy})"
    # faces: (variable, value, normal component sign, w.n)
    faces = [(x, 0, -1, -vx), (x, 1, 1, vx), (z, 0, -1, -vy), (z, 1, 1, vy)]
    for var, val, sgn, wn in faces:
        dn = sgn * sp.diff(p, var)
        report(f"N1 {tag} nu dn p + (w.n)p on {var}={val}", sp.simplify((nu * dn + wn * p).subs(var, val)), 0)
        other = z if var == x else x
        mx = max(abs(float((wn * p).subs({var: val, other: s, t: 0.3}))) for s in (0, 0.25, 0.5, 0.75, 1))
        report(f"N2 {tag} max|(w.n)p| on {var}={val} (t=0.3, 5 pts)", mx, 1e-6, larger=True)
        report(f"N3 {tag} nu dn y on {var}={val}", sp.simplify((nu * sgn * sp.diff(y, var)).subs(var, val)), 0)
    report(f"N3 {tag} p(T)", sp.simplify(p.subs(t, T)), 0)
    r_adj = -sp.diff(p, t) - nu * lap(p) - vx * sp.diff(p, x) - vy * sp.diff(p, z) - alpha_t * (y - yd)
    r_st = sp.diff(y, t) - nu * lap(y) + vx * sp.diff(y, x) + vy * sp.diff(y, z) - f - eta * u
    report(f"N5 {tag} adjoint residual (symbolic)", sp.simplify(r_adj), 0)
    report(f"N5 {tag} state residual (symbolic)", sp.simplify(r_st), 0)

    # N4: literal port of the C++ (ExactOcBenchmark.cpp NatR / NatPParts / Source / Target)
    nuf, au, kxf, kzf = 0.25, 0.01, float(vx) / 0.25, float(vy) / 0.25

    def NatR(k, s):
        e = math.exp(-k * s)
        g, dg, ddg = 2 + math.cos(math.pi * s), -math.pi * math.sin(math.pi * s), -math.pi**2 * math.cos(math.pi * s)
        return e * g, e * (dg - k * g), e * (ddg - 2 * k * dg + k * k * g)

    def NatP(xv, zv, b):
        X, Z = NatR(kxf, xv), NatR(kzf, zv)
        s = au * b
        return s * X[0] * Z[0], s * X[1] * Z[0], s * X[0] * Z[1], s * (X[2] * Z[0] + X[0] * Z[2])

    def NatY(xv, zv, a):
        cx, cz, sx, sz = math.cos(math.pi * xv), math.cos(math.pi * zv), math.sin(math.pi * xv), math.sin(math.pi * zv)
        return a * cx * cz, -a * math.pi * sx * cz, -a * math.pi * cx * sz, -2 * math.pi**2 * a * cx * cz

    def cpp_source(xv, zv, tv):
        yv, ytv = NatY(xv, zv, math.cos(2 * tv)), NatY(xv, zv, -2 * math.sin(2 * tv))
        u_ = -NatP(xv, zv, math.sin(1 - tv))[0] / au
        return ytv[0] - nuf * yv[3] + vx * yv[1] + vy * yv[2] - u_

    def cpp_target(xv, zv, tv):
        pv, ptv = NatP(xv, zv, math.sin(1 - tv)), NatP(xv, zv, -math.cos(1 - tv))
        return NatY(xv, zv, math.cos(2 * tv))[0] + (ptv[0] + nuf * pv[3] + vx * pv[1] + vy * pv[2]) / 1.0

    fl, ydl, pl = (sp.lambdify((x, z, t), e) for e in (f, yd, p))
    w = [0.0, 0.0, 0.0]
    for xv in (0.0, 0.37, 1.0):
        for zv in (0.0, 0.21, 0.5, 0.83, 1.0):
            for tv in (0.0, 0.3, 0.6, 0.9, 1.0):
                w[0] = max(w[0], abs(cpp_source(xv, zv, tv) - fl(xv, zv, tv)) / max(1, abs(fl(xv, zv, tv))))
                w[1] = max(w[1], abs(cpp_target(xv, zv, tv) - ydl(xv, zv, tv)) / max(1, abs(ydl(xv, zv, tv))))
                w[2] = max(w[2], abs(NatP(xv, zv, math.sin(1 - tv))[0] - pl(xv, zv, tv)))
    report(f"N4 {tag} C++ source vs sympy (75 pts, rel)", w[0], 1e-12)
    report(f"N4 {tag} C++ target vs sympy (75 pts, rel)", w[1], 1e-12)
    report(f"N4 {tag} C++ p vs sympy (75 pts, abs)", w[2], 1e-14)


winds = [float(v) for v in sys.argv[1:]] or [1.0, 0.5, -1.0, -0.5]
for i in range(0, len(winds), 2):
    check(sp.nsimplify(winds[i]), sp.nsimplify(winds[i + 1]))
print("T7F_SYM_RESULT", "PASS" if ok_all else "FAIL")
sys.exit(0 if ok_all else 1)
