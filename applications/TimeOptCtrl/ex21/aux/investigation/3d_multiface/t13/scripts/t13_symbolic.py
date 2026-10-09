#!/usr/bin/env python3
"""T13: symbolic verification of the multi-face exact cases (ExactOc3D.cpp, g_multi).

neumann-opp-3d (Gamma_c = zmin u zmax), neumann-adj-3d (zmin u xmin, w_x = w_z),
dirichlet-opp-3d (zmin u zmax, omega = (0,1)^2 x (1/4,3/4); L2 and H1 space/time).

Checks (exact sympy simplification unless a tolerance is shown):
 Neumann (N-cases):
  M1 state PDE residual = 0 (f = y_t - nu Lap y + w.grad y by construction; checked)
  M2 adjoint PDE residual = 0 with y_d = y + (p_t + nu Lap p + w.grad p)/alpha_t
  M3 nu dn y = eta u on EVERY controlled face;  M4 nu dn y = 0 on every uncontrolled face
  M5 adjoint Robin nu dn p + (w.n) p = 0 on all six faces;  M6 p(T) = 0
  M7 stationarity alpha_u u + eta p = 0 on every controlled face
  M8 edge compatibility: at every edge between two faces A, B:  d_{n_B} g_A = d_{n_A} g_B
     (g = prescribed flux: eta u or 0)  -- the condition for a smooth y; checked on all 12 edges
  M9 adj: the internal edge x = z = 0 carries a NONZERO control (u not forced to 0 there)
  N1 NEGATIVE adj with w_x != w_z: the internal-edge compatibility fails (the case refuses it)
  N2 NEGATIVE opp: y without the zmax corrector (kappa1 = 0) violates M3 on zmax
 Dirichlet (dirichlet-opp-3d):
  D1 Qs0, Qs1 solve the homogeneous backward adjoint (with X(x) X(y), any constant wind)
  D2 Qs0(0) = 0, Qs1(1) = 0  => p = 0 on all six faces
  D3 adjoint residual r = 0 for z <= 1/4 and z >= 3/4 (outside omega): the cutoff residual lives in omega
  D4 dz Qs1(1, t) = -exp(-a) dz Qs0(0, t)  =>  nu dn p|zmax = exp(-a) nu dn p|zmin
  D5 L2 stationarity alpha_u u = eta nu dn p on both faces with u1 = exp(-a) u0
  D6 H1 space: -Lap_Gamma S = 2 pi^2 S on both faces (S = sin sin) and S = 0 on dGamma_c
     (facewise eigenfunction; the union is two disjoint faces) => the temporal problem
     a U - gamma U_tt = F is the same on both faces up to the factor exp(-a) (linear)
  D7 y = eta u on both controlled faces, y = 0 on the four lateral faces
  D8 numeric: p(T - tau) -> 0 as tau -> 0 inside the cube (max over a grid), tol 1e-12
  N3 NEGATIVE: u1 = u0 (no exp(-a) factor) violates D5 on zmax under w_z != 0
Winds: symbolic (vx, vy, vz) where feasible; numeric A = (1, 0.5, -0.75), N = (0, 0, -0.75), adj (0.6, -0.4, 0.6).
usage: t13_symbolic.py <out.csv>"""
import csv
import itertools
import math
import sys

import sympy as sp

x, y, z, t, tau = sp.symbols("x y z t tau", real=True)
VX, VY, VZ = sp.symbols("v_x v_y v_z", real=True)
nu, T, eta, alpha_u, alpha_t = sp.Rational(1, 4), 1, 1, sp.Rational(1, 100), 1
pi = sp.pi
ok, rows = True, []
FACES = [(z, 0, -1, 2, "zmin"), (y, 0, -1, 1, "ymin"), (x, 1, 1, 0, "xmax"), (y, 1, 1, 1, "ymax"), (x, 0, -1, 0, "xmin"), (z, 1, 1, 2, "zmax")]
VARS = [x, y, z]


def report(name, value, tol=0.0, expect_nonzero=False):
    global ok
    if isinstance(value, sp.Basic) and not value.is_number:
        v = sp.simplify(value)
        good = (v == 0) if not expect_nonzero else (v != 0)
        shown = str(v)[:40]
    else:
        v = abs(complex(value)) if isinstance(value, (complex, sp.Basic)) else abs(float(value))
        v = float(v)
        good = (v <= tol) if not expect_nonzero else (v > 1e-8)
        shown = f"{v:.3e}"
    ok &= bool(good)
    rows.append({"check": name, "value": shown, "verdict": "PASS" if good else "FAIL"})
    print(f"T13_SYM {name:<78s} {shown:<14s} {'PASS' if good else 'FAIL'}")


lap = lambda q: sp.diff(q, x, 2) + sp.diff(q, y, 2) + sp.diff(q, z, 2)
A, B = sp.cos(2 * t), sp.sin(T - t)
Xf = lambda s: 1 - sp.cos(2 * pi * s)
R = lambda k, s: sp.exp(-k * s) * (2 + sp.cos(pi * s))
H0 = lambda s: s - s ** 2 / 2
H1 = lambda s: s ** 2 / 2
Fk = lambda k, s: sp.exp(-k * s) * (1 - s) ** 2 * (1 + 2 * s)


def neumann(case, w, tag):
    vx, vy, vz = w
    adv = lambda q: vx * sp.diff(q, x) + vy * sp.diff(q, y) + vz * sp.diff(q, z)
    if case == "opp":
        kz = vz / nu
        P = alpha_u * B * Xf(x) * Xf(y) * R(kz, z)
        k0, k1 = eta ** 2 * B * R(kz, 0) / nu, -eta ** 2 * B * R(kz, 1) / nu
        Y = A * sp.cos(pi * x) * sp.cos(pi * y) * sp.cos(pi * z) + k0 * Xf(x) * Xf(y) * H0(z) + k1 * Xf(x) * Xf(y) * H1(z)
        ctl = {"zmin", "zmax"}
    else:
        k = vx / nu
        P = alpha_u * B * Fk(k, x) * Xf(y) * Fk(k, z)
        fp0 = sp.diff(Fk(k, x), x).subs(x, 0)
        G = Fk(k, x) * H0(z) + H0(x) * Fk(k, z) - fp0 * H0(x) * H0(z)
        Y = A * sp.cos(pi * x) * sp.cos(pi * y) * sp.cos(pi * z) + eta ** 2 * B / nu * Xf(y) * G
        ctl = {"zmin", "xmin"}
    U = -eta * P / alpha_u                      # evaluated on each controlled face
    f = sp.diff(Y, t) - nu * lap(Y) + adv(Y)
    yd = Y + (sp.diff(P, t) + nu * lap(P) + adv(P)) / alpha_t
    report(f"M1 {tag} state residual", sp.simplify(sp.diff(Y, t) - nu * lap(Y) + adv(Y) - f))
    report(f"M2 {tag} adjoint residual", sp.simplify(-sp.diff(P, t) - nu * lap(P) - adv(P) - alpha_t * (Y - yd)))
    flux = {}
    for var, val, sgn, ax, name in FACES:
        dn = lambda q: sgn * sp.diff(q, var)
        if name in ctl:
            report(f"M3 {tag} nu dn y - eta u on {name}", sp.simplify((nu * dn(Y) - eta * U).subs(var, val)))
            flux[name] = eta * U
        else:
            report(f"M4 {tag} nu dn y on {name}", sp.simplify((nu * dn(Y)).subs(var, val)))
            flux[name] = sp.Integer(0)
        report(f"M5 {tag} nu dn p + (w.n) p on {name}", sp.simplify((nu * dn(P) + sgn * w[ax] * P).subs(var, val)))
        if name in ctl:
            report(f"M7 {tag} stationarity alpha_u u + eta p on {name}", sp.simplify((alpha_u * U + eta * P).subs(var, val)))
    report(f"M6 {tag} p(T)", sp.simplify(P.subs(t, T)))
    # M8: every edge (two faces with different axes)
    worst = 0.0
    for (va, la, sa, aa, na), (vb, lb, sb, ab, nb) in itertools.combinations(FACES, 2):
        if aa == ab:
            continue
        lhs = (sb * sp.diff(flux[na], vb)).subs({va: la, vb: lb})
        rhs = (sa * sp.diff(flux[nb], va)).subs({va: la, vb: lb})
        d = sp.simplify(lhs - rhs).subs({VX: 0.37, VY: -0.21, VZ: 0.53})   # symbolic wind: a numeric sample
        if d != 0:
            fn = sp.lambdify([x, y, z, t], d, "math")
            free = [v for v in VARS if v not in (va, vb)][0]
            vals = [abs(fn(*[{va: la, vb: lb}.get(v, s) for v in VARS], tt)) for s in (0.13, 0.5, 0.77) for tt in (0.1, 0.6)]
            worst = max(worst, max(vals))
    report(f"M8 {tag} edge compatibility d_nB g_A = d_nA g_B on all 12 edges (max)", worst, 1e-12)
    if case == "adj":
        ue = sp.lambdify([y, t], U.subs({x: 0, z: 0}).subs({VX: 0.37, VY: -0.21, VZ: 0.53}), "math")
        report(f"M9 {tag} control on the internal edge x=z=0 nonzero", max(abs(ue(s, 0.3)) for s in (0.3, 0.5)), expect_nonzero=True)
    if case == "opp":
        Yb = Y - k1 * Xf(x) * Xf(y) * H1(z)
        d = sp.lambdify([x, y, t], (nu * sp.diff(Yb, z) - eta * U).subs(z, 1).subs({VX: 0.37, VY: -0.21, VZ: 0.53}), "math")
        report(f"N2 {tag} NEG y without the zmax corrector: flux on zmax violated", abs(d(0.4, 0.3, 0.2)), expect_nonzero=True)


def neumann_adj_negative():
    vx, vz = sp.Rational(1), sp.Rational(-1, 2)
    P = Fk(vx / nu, x) * Xf(y) * R(vz / nu, z)   # Robin on x = 0 with k_x, on z = 0 with k_z
    d = (sp.diff(P, x) - sp.diff(P, z)).subs({x: 0, z: 0})
    report("N1 adj NEG w_x != w_z: p_x - p_z at x=z=0 (= -(w_x - w_z) p/nu) nonzero", float(d.subs(y, sp.Rational(1, 3))), expect_nonzero=True)


def dirichlet(w, tag):
    vx, vy, vz = w
    adv = lambda q: vx * sp.diff(q, x) + vy * sp.diff(q, y) + vz * sp.diff(q, z)
    cx, cy, a = vx / (2 * nu), vy / (2 * nu), vz / (2 * nu)
    X = lambda c, s: sp.exp(-c * s) * sp.sin(pi * s)
    lam = 2 * nu * pi ** 2 + nu * (cx ** 2 + cy ** 2)
    cc = vz ** 2 / (4 * nu) + lam
    Phi = lambda xi: sp.exp(-xi ** 2 / (4 * nu * tau)) / sp.sqrt(4 * pi * nu * tau)
    Qs0 = sp.exp(-a * z - cc * tau) * (Phi(z - sp.Rational(6, 5)) - Phi(z + sp.Rational(6, 5)))
    Qs1 = sp.exp(-a * z - cc * tau) * (Phi(z + sp.Rational(1, 5)) - Phi(z - sp.Rational(11, 5)))
    S = X(cx, x) * X(cy, y)
    for nm, Q in (("Qs0", Qs0), ("Qs1", Qs1)):
        p = S * Q.subs(tau, T - t)
        r = -sp.diff(p, t) - nu * lap(p) - adv(p)
        fn = sp.lambdify([x, y, z, t], sp.simplify(r / p), "math")
        m = max(abs(fn(xx, yy, zz, tt)) for xx, yy, zz, tt in itertools.product((0.3, 0.7), (0.4,), (0.1, 0.5, 0.9), (0.2, 0.7)))
        report(f"D1 {tag} {nm}: homogeneous adjoint residual / p (max on grid)", m, 1e-11)
    report(f"D2 {tag} Qs0(z=0) = 0", sp.simplify(Qs0.subs(z, 0)))
    report(f"D2 {tag} Qs1(z=1) = 0", sp.simplify(Qs1.subs(z, 1)))
    d0, d1 = sp.diff(Qs0, z).subs(z, 0), sp.diff(Qs1, z).subs(z, 1)
    report(f"D4 {tag} dz Qs1(1) + exp(-a) dz Qs0(0) = 0", sp.simplify(d1 + sp.exp(-a) * d0))
    # stationarity: u0 = (eta nu/alpha_u) dn p|zmin = -(eta nu/alpha_u) S dz Qs0(0); u1 = (eta nu/alpha_u) S dz Qs1(1)
    u0 = -eta * nu / alpha_u * S * d0
    u1 = eta * nu / alpha_u * S * d1
    report(f"D5 {tag} L2 stationarity zmax: u1 - exp(-a) u0 = 0", sp.simplify(u1 - sp.exp(-a) * u0))
    if vz != 0:
        report(f"N3 {tag} NEG u1 = u0 (no exp(-a)) violates zmax stationarity", float((u1 - u0).subs({x: 0.4, y: 0.6, tau: 0.5})),
               expect_nonzero=True)
    if vx == 0 and vy == 0:
        Sx = sp.sin(pi * x) * sp.sin(pi * y)
        report(f"D6 {tag} -Lap_Gamma S - 2 pi^2 S on zmin and zmax (same S)", sp.simplify(-(sp.diff(Sx, x, 2) + sp.diff(Sx, y, 2)) - 2 * pi ** 2 * Sx))
        report(f"D6 {tag} S = 0 on the perimeter of each face", sp.simplify(Sx.subs(x, 0)) + sp.simplify(Sx.subs(x, 1)) + sp.simplify(Sx.subs(y, 0)) + sp.simplify(Sx.subs(y, 1)))
    # D3: residual of the cut-off p outside omega
    sstep = sp.Symbol("s")
    for side, psi in (("z<=1/4 (psi=0)", 0), ("z>=3/4 (psi=1)", 1)):
        Q = (1 - psi) * Qs0 + psi * Qs1
        p = S * Q.subs(tau, T - t)
        r = -sp.diff(p, t) - nu * lap(p) - adv(p)
        fn = sp.lambdify([x, y, z, t], sp.simplify(r / p), "math")
        zs = (0.05, 0.2) if psi == 0 else (0.8, 0.95)
        m = max(abs(fn(xx, 0.4, zz, tt)) for xx, zz, tt in itertools.product((0.3, 0.7), zs, (0.2, 0.7)))
        report(f"D3 {tag} adjoint residual outside omega, {side} (rel., grid)", m, 1e-11)
    # D7: y = eta u on the controlled faces, 0 on the lateral faces (U: any temporal factor)
    Ut = sp.Function("U")(t)
    e1 = sp.exp(-a)
    Y = sp.cos(2 * t) * (sp.sin(pi * x) * sp.sin(pi * y) * sp.sin(pi * z)) ** 3 + eta * S * (Ut * (1 - z) ** 3 + e1 * Ut * z ** 3)
    report(f"D7 {tag} y - eta u0 on zmin", sp.simplify(Y.subs(z, 0) - eta * S * Ut))
    report(f"D7 {tag} y - eta u1 on zmax", sp.simplify(Y.subs(z, 1) - eta * e1 * S * Ut))
    for var, val in ((x, 0), (x, 1), (y, 0), (y, 1)):
        report(f"D7 {tag} y on {var}={val}", sp.simplify(Y.subs(var, val)))
    # D8: p(T) = 0 (tau -> 0+) inside the cube
    fq = sp.lambdify([z, tau], Qs0 * 0 + ((1 - 0) * Qs0), "math")
    fq1 = sp.lambdify([z, tau], Qs1, "math")
    m = max(abs(f(zz, 1e-4)) for f in (fq, fq1) for zz in [i / 20 for i in range(21)])
    report(f"D8 {tag} max |Qs0|, |Qs1| at tau = 1e-4 on [0,1]", m, 1e-12)


for w, tag in (((VX, VY, VZ), "opp wsym"), ((0, 0, 0), "opp heat"), ((1, sp.Rational(1, 2), sp.Rational(-3, 4)), "opp wA")):
    neumann("opp", w, tag)
for w, tag in (((VX, VY, VX), "adj w(a,b,a)sym"), ((0, 0, 0), "adj heat"), ((sp.Rational(3, 5), sp.Rational(-2, 5), sp.Rational(3, 5)), "adj w(.6,-.4,.6)")):
    neumann("adj", w, tag)
neumann_adj_negative()
for w, tag in (((0, 0, 0), "dopp heat"), ((0, 0, sp.Rational(-3, 4)), "dopp wN"), ((1, sp.Rational(1, 2), sp.Rational(-3, 4)), "dopp wA")):
    dirichlet(w, tag)
with open(sys.argv[1], "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
print("T13_SYM_RESULT", "PASS" if ok else "FAIL", f"({sum(r['verdict'] == 'PASS' for r in rows)}/{len(rows)})")
sys.exit(0 if ok else 1)
