#!/usr/bin/env python3
"""T7D: symbolic verification of the dirichlet-conv exact optimal control.

Checks (sympy, exact where possible, high-precision sampling otherwise):
  S1  Qs solves the homogeneous backward adjoint  -Q_t - nu Q_zz - vy Q_z + pi^2 nu Q = 0
  S2  Qs(0, t) = 0 (adjoint vanishes on Gamma_c) and Qs -> 0 as t -> T (p(T) = 0)
  S3  Zr satisfies the Robin adjoint BC  nu Zr' + vy Zr = 0  at z = 1
  S4  the C-infinity step: phi = 0 on z <= 1/2, 1 on z >= 9/10 => r = 0 on z < 1/2
  S5  full adjoint residual r = -p_t - nu Lap p - w.grad p equals the C++
      DirAdjointResidual formula (hand-derived derivatives) at sample points
  S6  state: y = eta u on Gamma_c, nu dn y = 0 on the other faces; C++ source
      f = y_t - nu Lap y + w.grad y matches sympy
  S7  x-faces: dp/dx = 0, dy/dx = 0 (w.n = vx = 0 there)
  S8  stationarity  alpha_u (u - ell^2 u_ss) = eta nu dn p  on Gamma_c (dn = -d/dz)
  S9  ||p(t_{n-1/2})||_L2 against the C++ slot p_exact (from a log, optional)
Usage: t7d_symbolic.py [exact-oc log with EX21_EXACT_SLOT lines] [ell]
"""
import re
import sys

import mpmath as mp
import sympy as sp

mp.mp.dps = 30
x, z, t = sp.symbols("x z t", real=True)
nu, vy, T, eta, alpha_u, ell = sp.Rational(1, 4), sp.Rational(1, 2), 1, 1, sp.Rational(1, 100), 0
if len(sys.argv) > 2:
    ell = sp.nsimplify(sys.argv[2])
Z1, S = sp.Rational(6, 5), sp.Rational(2, 5)
Za, Zb = sp.Rational(1, 2), sp.Rational(9, 10)
pi = sp.pi
tau = T - t

ok_all = True


def report(name, value, tol):
    global ok_all
    v = abs(float(value))
    ok = v <= tol
    ok_all &= ok
    print(f"T7D_SYM {name:<44s} {v:.3e}  {'PASS' if ok else 'FAIL'} (tol {tol:.0e})")


# ---- adjoint pieces ---------------------------------------------------------
def Phi(xi):
    return sp.exp(-xi**2 / (4 * nu * tau)) / sp.sqrt(4 * pi * nu * tau)


a = vy / (2 * nu)
cc = vy**2 / (4 * nu) + pi**2 * nu
Qs = sp.exp(-a * z - cc * tau) * (Phi(z - Z1) - Phi(z + Z1))
L_adj = lambda Q: -sp.diff(Q, t) - nu * sp.diff(Q, z, 2) - vy * sp.diff(Q, z) + pi**2 * nu * Q

# S1: exact symbolic simplification
res1 = sp.simplify(sp.expand(L_adj(Qs) * sp.exp(a * z + cc * tau)))
print("T7D_SYM S1 adjoint PDE (symbolic simplify)       ", res1)
ok_all &= res1 == 0

# S2
report("S2 Qs(0,t) (symbolic)", sp.simplify(Qs.subs(z, 0)), 0)
lim_vals = [abs(Qs.subs({z: zz, t: 1 - sp.Rational(1, 10**6)}).evalf(30)) for zz in (0, sp.Rational(1, 2), sp.Rational(9, 10), 1)]
report("S2 max|Qs(z, T-1e-6)| over z in {0,.5,.9,1}", max(lim_vals), 1e-300)

# S3 Robin profile
b = vy / nu
Zr = sp.exp(-b * z) * (2 + sp.cos(pi * z))
report("S3 nu Zr'(1) + vy Zr(1)", sp.simplify((nu * sp.diff(Zr, z) + vy * Zr).subs(z, 1)), 0)

# S4 step
s = (z - Za) / (Zb - Za)
psi = lambda q: sp.exp(-1 / q)
phi_mid = psi(s) / (psi(s) + psi(1 - s))
phi = sp.Piecewise((0, z <= Za), (1, z >= Zb), (phi_mid, True))
B = sp.sin(T - t)
Q = (1 - phi) * Qs + phi * B * Zr
p = S * sp.cos(pi * x) * Q
# r on z < 1/2 is exactly L_adj(Qs) = 0 (S1); on z > 9/10 it is L_adj(B Zr):
rR = sp.simplify(L_adj(B * Zr))
print("T7D_SYM S4 residual on z>=0.9 (tracked, nonzero) ", sp.simplify(rR / (sp.exp(-b * z))))

# S5: compare against the C++ hand derivatives (literal port)
import math


def cpp_DirQs(zv, tv):
    tau_ = 1.0 - tv
    if tau_ <= 1e-14:
        return (0, 0, 0, 0, 0)
    nu_, vy_ = 0.25, 0.5
    a_, c_ = vy_ / (2 * nu_), vy_ * vy_ / (4 * nu_) + math.pi**2 * nu_

    def ph(xi):
        p0 = math.exp(-xi * xi / (4 * nu_ * tau_)) / math.sqrt(4 * math.pi * nu_ * tau_)
        p1 = -xi / (2 * nu_ * tau_) * p0
        p2 = (xi * xi / (4 * nu_**2 * tau_**2) - 1 / (2 * nu_ * tau_)) * p0
        p3 = (-xi**3 / (8 * nu_**3 * tau_**3) + 3 * xi / (4 * nu_**2 * tau_**2)) * p0
        return p0, p1, p2, p3

    A_, B_ = ph(zv - 1.2), ph(zv + 1.2)
    V, Vz, Vzz, Vzzz = (A_[i] - B_[i] for i in range(4))
    E = math.exp(-a_ * zv - c_ * tau_)
    return (E * V, E * (Vz - a_ * V), E * (Vzz - 2 * a_ * Vz + a_ * a_ * V),
            E * (nu_ * Vzz - c_ * V), E * (-c_ * (Vz - a_ * V) + nu_ * Vzzz - a_ * nu_ * Vzz))


def cpp_step(zv):
    L = 0.4
    s_ = (zv - 0.5) / L
    if s_ <= 0:
        return 0, 0, 0
    if s_ >= 1:
        return 1, 0, 0

    def ps(q):
        g0 = math.exp(-1 / q)
        return g0, g0 / q**2, g0 * (1 / q**4 - 2 / q**3)

    g0, g1, g2 = ps(s_)
    h0, h1, h2 = ps(1 - s_)
    h1 = -h1
    D, Dp, N, Np = g0 + h0, g1 + h1, g1 * h0 - g0 * h1, g2 * h0 - g0 * h2
    return g0 / D, N / D**2 / L, (Np * D - 2 * N * Dp) / D**3 / L**2


def cpp_DirQ(zv, tv):
    q = cpp_DirQs(zv, tv)
    b_ = 2.0
    e = math.exp(-b_ * zv)
    g, dg, ddg = 2 + math.cos(math.pi * zv), -math.pi * math.sin(math.pi * zv), -math.pi**2 * math.cos(math.pi * zv)
    Zr_, Zr1, Zr2 = e * g, e * (dg - b_ * g), e * (ddg - 2 * b_ * dg + b_ * b_ * g)
    Bt, dBt = math.sin(1 - tv), -math.cos(1 - tv)
    f, f1, f2 = cpp_step(zv)
    Qr, Qr1, Qr2 = Bt * Zr_, Bt * Zr1, Bt * Zr2
    return ((1 - f) * q[0] + f * Qr, (1 - f) * q[1] + f * Qr1 + f1 * (Qr - q[0]),
            (1 - f) * q[2] + f * Qr2 + 2 * f1 * (Qr1 - q[1]) + f2 * (Qr - q[0]),
            -(1 - f) * q[3] + f * dBt * Zr_)


Qmid = (1 - phi_mid) * Qs + phi_mid * B * Zr
derivs = {"v": Qmid, "z": sp.diff(Qmid, z), "zz": sp.diff(Qmid, z, 2), "t": sp.diff(Qmid, t)}
derivs_s = {"v": Qs, "z": sp.diff(Qs, z), "zz": sp.diff(Qs, z, 2), "t": sp.diff(Qs, t)}
fn = {k: sp.lambdify((z, t), e, "mpmath") for k, e in derivs.items()}
fn_s = {k: sp.lambdify((z, t), e, "mpmath") for k, e in derivs_s.items()}
worst = {k: 0.0 for k in derivs}
pts = [(zv, tv) for zv in (0.0, 0.13, 0.37, 0.5, 0.55, 0.7, 0.83, 0.9, 0.95, 1.0)
       for tv in (0.0, 0.31, 0.5, 0.77, 0.95, 0.999)]
for zv, tv in pts:
    c = cpp_DirQ(zv, tv)
    use = fn if 0.5 < zv < 0.9 else fn_s
    for i, k in enumerate(("v", "z", "zz", "t")):
        if zv >= 0.9:
            ref = {"v": B * Zr, "z": sp.diff(B * Zr, z), "zz": sp.diff(B * Zr, z, 2), "t": sp.diff(B * Zr, t)}[k]
            ref = float(ref.subs({z: zv, t: tv}))
        else:
            ref = float(use[k](mp.mpf(zv), mp.mpf(tv)))
        scale = max(1.0, abs(ref))
        worst[k] = max(worst[k], abs(c[i] - ref) / scale)
for k in worst:
    report(f"S5 C++ DirQ.{k} vs sympy (60 pts, rel)", worst[k], 1e-11)

# ---- control and state ------------------------------------------------------
Qs_z0 = sp.diff(Qs, z).subs(z, 0)
U = -eta * nu * S * Qs_z0 / (alpha_u * (1 + pi**2 * ell**2))
u = U * sp.cos(pi * x)
y = sp.cos(2 * t) * sp.cos(pi * x) * sp.sin(pi * z / 2) + eta * u

# S8 stationarity: alpha_u (u - ell^2 u_xx) - eta nu dn p = 0 on z = 0, dn = -dz
p_s = S * sp.cos(pi * x) * Qs   # p near Gamma_c (phi = 0 there)
stat = alpha_u * (u - ell**2 * sp.diff(u, x, 2)) - eta * nu * (-sp.diff(p_s, z).subs(z, 0))
report("S8 stationarity on Gamma_c (symbolic)", sp.simplify(stat), 0)
# S6/S7 boundary conditions
report("S6 y - eta u on z=0 (symbolic)", sp.simplify((y - eta * u).subs(z, 0)), 0)
report("S6 dy/dz on z=1 (symbolic)", sp.simplify(sp.diff(y, z).subs(z, 1)), 0)
report("S7 dy/dx on x=0,1 (symbolic)", sp.simplify(sp.diff(y, x).subs(x, 0)) + sp.simplify(sp.diff(y, x).subs(x, 1)), 0)
report("S7 dp/dx on x=0,1 (symbolic)", sp.simplify(sp.diff(p_s, x).subs(x, 0)) + sp.simplify(sp.diff(p_s, x).subs(x, 1)), 0)

# S6: C++ source f = y_t - nu Lap y + vy y_z (with C++ DirU, DirUt)
f_sym = sp.diff(y, t) - nu * (sp.diff(y, x, 2) + sp.diff(y, z, 2)) + vy * sp.diff(y, z)
f_l = sp.lambdify((x, z, t), f_sym, "mpmath")
U_l = sp.lambdify(t, U, "mpmath")
Ut_l = sp.lambdify(t, sp.diff(U, t), "mpmath")
ellf = float(ell)


def cpp_source(xv, zv, tv):
    q = cpp_DirQs(0.0, tv)
    den = 0.01 * (1 + math.pi**2 * ellf**2)
    Uc, Utc = -0.25 * 0.4 * q[1] / den, 0.25 * 0.4 * q[4] / den
    cx, sz, cz = math.cos(math.pi * xv), math.sin(0.5 * math.pi * zv), math.cos(0.5 * math.pi * zv)
    A_, dA_ = math.cos(2 * tv), -2 * math.sin(2 * tv)
    yt = dA_ * cx * sz + Utc * cx
    lap = -1.25 * math.pi**2 * A_ * cx * sz - math.pi**2 * Uc * cx
    yz = 0.5 * math.pi * A_ * cx * cz
    return yt - 0.25 * lap + 0.5 * yz, Uc


wf = wu = 0.0
for xv in (0.0, 0.21, 0.66):
    for zv, tv in pts:
        cf, cu = cpp_source(xv, zv, tv)
        rf = float(f_l(mp.mpf(xv), mp.mpf(zv), mp.mpf(tv))) if tv < 0.9999 else cf
        ru = float(U_l(mp.mpf(tv))) if tv < 0.9999 else cu
        wf = max(wf, abs(cf - rf) / max(1, abs(rf)))
        wu = max(wu, abs(cu - ru) / max(1, abs(ru)))
report("S6 C++ source f vs sympy (rel)", wf, 1e-11)
report("S8 C++ DirU vs sympy (rel)", wu, 1e-11)
print(f"T7D_SYM info max|U| on [0,T] ~ {max(abs(float(U_l(mp.mpf(k / 200)))) for k in range(200)):.6f}")

# S9: compare ||p(t_m)||_L2 with the C++ slot p_exact
if len(sys.argv) > 1 and sys.argv[1]:
    rows = [l for l in open(sys.argv[1]) if l.startswith("EX21_EXACT_SLOT")]
    cn = "scheme=CN" in open(sys.argv[1]).read()
    steps = len(rows)
    worst = 0.0
    for l in rows[:: max(1, steps // 16)]:
        n = int(re.search(r"n=(\d+)", l).group(1))
        px = float(re.search(r"p_exact=(\S+)", l).group(1))
        tm = (n - 0.5) / steps if cn else n / steps
        qq = lambda zv: cpp_DirQ(float(zv), tm)[0] ** 2
        nrm = math.sqrt(0.4**2 * 0.5 * float(mp.quad(qq, [0, 0.5, 0.9, 1])))
        worst = max(worst, abs(nrm - px) / max(px, 1e-30))
    report(f"S9 ||p(t)|| vs C++ slot p_exact ({steps} steps)", worst, 1e-6)

print("T7D_SYM_RESULT", "PASS" if ok_all else "FAIL")
sys.exit(0 if ok_all else 1)
