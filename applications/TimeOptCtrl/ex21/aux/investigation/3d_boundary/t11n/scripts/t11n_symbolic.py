#!/usr/bin/env python3
"""T11N: symbolic verification of the 3D exact Neumann-control OC (neumann-conv-3d).

  Gamma_c = {z = 0}.  X(s) = 1 - cos(2 pi s);  R_k(s) = exp(-k s)(2 + cos(pi s)), kz = vz/nu
  p = alpha_u B(t) X(x) X(y) R_kz(z),  B = sin(T - t)
  u = -(eta/alpha_u) p(x, y, 0, t)                           (on Gamma_c)
  y = A(t) cos(pi x) cos(pi y) cos(pi z) + kappa(t) X(x) X(y) H(z),
      H = z - z^2/2,  kappa = eta^2 B R_kz(0)/nu,  A = cos 2t
  f = y_t - nu Lap y + w.grad y;   y_d = y + (p_t + nu Lap p + w.grad p)/alpha_t
Checks (exact sympy simplification unless stated):
  M1 state PDE  y_t - nu Lap y + w.grad y - f = 0
  M2 adjoint PDE  -p_t - nu Lap p - w.grad p - alpha_t (y - y_d) = 0
  M3 controlled flux on Gamma_c:  nu dn y - eta u = 0  (n = -e_z)
  M4 natural flux nu dn y = 0 on the five other faces
  M5 adjoint Robin  nu dn p + (w.n) p = 0 on all six faces
  M6 p(T) = 0;   M7 stationarity alpha_u u + eta p|_Gamma_c = 0
  M8 C++ literal port (ExactOc3D.cpp Neu*) = sympy at 375 points (relative to the field scale)
  M9 max over Gamma_c (and z = 1) of |(w.n) p| > 0 and of the WRONG-BC residual
     |nu dn p| > 0 (the Robin term is exercised; on the four lateral faces
     p = dn p = 0 identically -- smoothness at the edges of Gamma_c forces it)
Winds: w = 0, A = (1, 0.5, -0.75), B = (-2, 1, 0.5), C = (1, 0.5, 0) (tangential on
Gamma_c), and a symbolic (vx, vy, vz) for M1-M7.
"""
import math
import sys

import sympy as sp

x, y, z, t = sp.symbols("x y z t", real=True)
VX, VY, VZ = sp.symbols("v_x v_y v_z", real=True)
nu, T, eta, alpha_u, alpha_t = sp.Rational(1, 4), 1, 1, sp.Rational(1, 100), 1
pi = sp.pi
ok, rows = True, []


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
    rows.append(good)
    print(f"T11N_SYM {name:<62s} {shown:<12s} {'PASS' if good else 'FAIL'}")


def check(w, tag):
    vx, vy, vz = w
    A, B = sp.cos(2 * t), sp.sin(T - t)
    Xf = lambda s: 1 - sp.cos(2 * pi * s)
    R = lambda k, s: sp.exp(-k * s) * (2 + sp.cos(pi * s))
    kz = vz / nu
    P = alpha_u * B * Xf(x) * Xf(y) * R(kz, z)
    U = -eta * P.subs(z, 0) / alpha_u
    kappa = eta ** 2 * B * R(kz, 0) / nu
    Y = A * sp.cos(pi * x) * sp.cos(pi * y) * sp.cos(pi * z) + kappa * Xf(x) * Xf(y) * (z - z ** 2 / 2)
    lap = lambda q: sp.diff(q, x, 2) + sp.diff(q, y, 2) + sp.diff(q, z, 2)
    adv = lambda q: vx * sp.diff(q, x) + vy * sp.diff(q, y) + vz * sp.diff(q, z)
    f = sp.diff(Y, t) - nu * lap(Y) + adv(Y)
    yd = Y + (sp.diff(P, t) + nu * lap(P) + adv(P)) / alpha_t
    report(f"M1 {tag} state residual", sp.simplify(sp.diff(Y, t) - nu * lap(Y) + adv(Y) - f))
    report(f"M2 {tag} adjoint residual", sp.simplify(-sp.diff(P, t) - nu * lap(P) - adv(P) - alpha_t * (Y - yd)))
    faces = [(z, 0, -1, 2), (y, 0, -1, 1), (x, 1, 1, 0), (y, 1, 1, 1), (x, 0, -1, 0), (z, 1, 1, 2)]  # FEMuS 1..6
    for var, val, sgn, ax in faces:
        dn = lambda q: sgn * sp.diff(q, var)
        if (var, val) == (z, 0):
            report(f"M3 {tag} nu dn y - eta u on Gamma_c (z=0)", sp.simplify((nu * dn(Y)).subs(z, 0) - eta * U))
        else:
            report(f"M4 {tag} nu dn y on {var}={val}", sp.simplify((nu * dn(Y)).subs(var, val)))
        report(f"M5 {tag} nu dn p + (w.n) p on {var}={val}",
               sp.simplify((nu * dn(P) + sgn * w[ax] * P).subs(var, val)))
    report(f"M6 {tag} p(T)", sp.simplify(P.subs(t, T)))
    report(f"M7 {tag} stationarity on Gamma_c", sp.simplify(alpha_u * U + eta * P.subs(z, 0)))
    if any(isinstance(c, sp.Symbol) for c in w):
        return
    fw = [float(c) for c in w]

    # ---- M8: literal port of ExactOc3D.cpp (Neu*, NatR, NatC, Product)
    def natr(k, s):
        e = math.exp(-k * s)
        g, dg, ddg = 2 + math.cos(math.pi * s), -math.pi * math.sin(math.pi * s), -math.pi ** 2 * math.cos(math.pi * s)
        return e * g, e * (dg - k * g), e * (ddg - 2 * k * dg + k * k * g)
    natc = lambda s: (math.cos(math.pi * s), -math.pi * math.sin(math.pi * s), -math.pi ** 2 * math.cos(math.pi * s))
    neux = lambda s: (1 - math.cos(2 * math.pi * s), 2 * math.pi * math.sin(2 * math.pi * s), 4 * math.pi ** 2 * math.cos(2 * math.pi * s))
    neuh = lambda s: (s - 0.5 * s * s, 1 - s, -1.0)

    def prod(X, Yy, Z):
        return (X[0] * Yy[0] * Z[0],
                fw[0] * X[1] * Yy[0] * Z[0] + fw[1] * X[0] * Yy[1] * Z[0] + fw[2] * X[0] * Yy[0] * Z[1],
                X[2] * Yy[0] * Z[0] + X[0] * Yy[2] * Z[0] + X[0] * Yy[0] * Z[2])

    def cpp(xv, yv, zv, tv):
        nuf, au, kzf = 0.25, 0.01, fw[2] / 0.25
        Av, dAv, Bv, dBv = math.cos(2 * tv), -2 * math.sin(2 * tv), math.sin(1 - tv), -math.cos(1 - tv)
        Ys, G = prod(natc(xv), natc(yv), natc(zv)), prod(neux(xv), neux(yv), neuh(zv))
        Ps = prod(neux(xv), neux(yv), natr(kzf, zv))
        k, kd = Bv * natr(kzf, 0)[0] / nuf, dBv * natr(kzf, 0)[0] / nuf
        src = dAv * Ys[0] + kd * G[0] - nuf * (Av * Ys[2] + k * G[2]) + Av * Ys[1] + k * G[1]
        yv_ = Av * Ys[0] + k * G[0]
        q = au * (dBv * Ps[0] + nuf * Bv * Ps[2] + Bv * Ps[1])
        return src, yv_ + q, -(au * Bv * prod(neux(xv), neux(yv), natr(kzf, 0.0))[0]) / au

    fl, ydl, ul = sp.lambdify((x, y, z, t), f), sp.lambdify((x, y, z, t), yd), sp.lambdify((x, y, t), U)
    m = {"f": [0, 0], "yd": [0, 0], "u": [0, 0]}
    pts = [0.0, 0.17, 0.5, 0.83, 1.0]
    for xv in pts:
        for yv in pts:
            for zv in pts:
                for tv in (0.0, 0.4, 1.0):
                    a, b, c = cpp(xv, yv, zv, tv)
                    for key, got, ref in (("f", a, fl(xv, yv, zv, tv)), ("yd", b, ydl(xv, yv, zv, tv)), ("u", c, ul(xv, yv, tv))):
                        m[key][0] = max(m[key][0], abs(got - ref)); m[key][1] = max(m[key][1], abs(ref))
    for key in m:
        report(f"M8 {tag} C++ {key} vs sympy (max abs / max |field|)", m[key][0] / max(m[key][1], 1e-300), 1e-13)
    if all(c == 0 for c in fw):
        return
    Pl = sp.lambdify((x, y, z, t), P)
    dPz = sp.lambdify((x, y, z, t), sp.diff(P, z))
    grid = [i / 20 for i in range(21)]
    for val, sgn in ((0, -1), (1, 1)):
        wn = sgn * fw[2]
        m_wnp = max(abs(wn * Pl(a_, b_, val, tv)) for a_ in grid for b_ in grid for tv in (0.0, 0.5))
        m_wrong = max(abs(0.25 * sgn * dPz(a_, b_, val, tv)) for a_ in grid for b_ in grid for tv in (0.0, 0.5))
        lab = "Gamma_c z=0" if val == 0 else "z=1"
        if fw[2] != 0:
            report(f"M9 {tag} max|(w.n) p| on {lab} (> 0)", m_wnp, expect_nonzero=True)
            report(f"M9 {tag} max|nu dn p| wrong-BC residual on {lab} (> 0)", m_wrong, expect_nonzero=True)
        else:
            print(f"T11N_SYM M9 {tag} w_z = 0: (w.n) p = 0 on {lab} (tangential wind; Robin = Neumann there) INFO")


check((0, 0, 0), "w=0")
for w in ((1.0, 0.5, -0.75), (-2.0, 1.0, 0.5), (1.0, 0.5, 0.0)):
    check(tuple(sp.nsimplify(v) for v in w), "w=(" + ",".join(f"{v:g}" for v in w) + ")")
check((VX, VY, VZ), "w=symbolic")
print("T11N_SYM_RESULT", "PASS" if ok else "FAIL", f"({sum(rows)}/{len(rows)})")
sys.exit(0 if ok else 1)
