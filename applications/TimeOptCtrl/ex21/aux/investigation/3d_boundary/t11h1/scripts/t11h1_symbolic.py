#!/usr/bin/env python3
"""T11H1: symbolic / high-precision verification of the exact H1 benchmark dirichlet-h1-3d.

p is the T11D adjoint (t11d_symbolic.py): p = X_cx(x) X_cy(y) (1 - phi(z)) Qs(z,tau), tau = T - t.
u = S(x,y) U(t), S = X_cx X_cy (= sin pi x sin pi y when ell > 0: no tangential wind);
y = cos 2t sin^3 sin^3 sin^3 + eta u (1 - z)^3;  f = y_t - nu Lap y + w.grad y;  y_d = y - r/alpha_t.
Stationarity (T7D sign; n = -e_z):  alpha_u (u - ell^2 Lap_G u) - gamma_u u_tt = eta nu dn p,
natural u_t(T) = 0 (gamma_u > 0), u(0) = u_0 DATA (y_0 is built from U(T)).
Checks:
 H1  -Lap_G S = 2 pi^2 S (c = 0) exactly; S = 0 on dGamma_c (u = 0 there)          [sympy]
 H2  dQs/dz(0,tau) = C0 tau^{-3/2} exp(-A0/tau - cc tau)                          [sympy]
 H3  E(b,tau) closed form (erfc) = int_0^tau s^{-3/2} e^{-A0/s - b s} ds            [mpmath quad, 40 digits]
 H4  U closed form = -(1/(gamma k)) int_0^tau sinh(k(tau-s)) F(s) ds  (direct quadrature, NOT the erfc
     formula) and U_tau likewise                                                   [mpmath]
 H5  ODE residual a U - gamma U_tautau - F = 0 with U_tautau by mpmath differentiation of the closed form
 H6  endpoints: U_tau(0) = 0 (natural u_t(T) = 0), U(tau -> 0) -> 0 to all orders
 H7  stationarity on Gamma_c at sample points: alpha_u (u - ell^2 Lap_G u) - gamma u_tt - eta nu dn p = 0
     (Lap_G by sympy of the literal S, dn p by sympy of p, U and U_tt by mpmath)
 H8  state: y = eta u on Gamma_c, y = 0 on the five other faces, y(0)|Gamma_c = eta u(0)  [sympy, U symbolic]
 H9  adjoint (unchanged T11D p): r = 0 for z <= 1/2; p = 0 on the six faces             [sympy]
 H10 the admissible family U + A cosh(k tau) (ODE + natural endpoint; A sets the datum u_0)
 H11 C++ literal port (ExactOc3D.cpp DirH1Time, DirY, DirSource3D, DirExactUGrad) vs the mpmath reference
Cases: heat, normal wind (0,0,-0.75), opposite normal wind (0,0,0.5) with ell in {0, 0.3} and gamma in {0, 1e-2};
tangential wind A = (1,0.5,-0.75) with ell = 0, gamma = 1e-2 (H1-time only: S = X_cx X_cy needs no eigen-property)."""
import math, sys
import mpmath as mp
import sympy as sp

mp.mp.dps = 40
x, y, z, t, tau = sp.symbols("x y z t tau", real=True)
nu, eta, au, at, T, Z1 = sp.Rational(1, 4), 1, sp.Rational(1, 100), 1, 1, sp.Rational(6, 5)
pi = sp.pi
ok, n_ok = True, 0


def report(name, value, tol=0.0):
    global ok, n_ok
    if isinstance(value, sp.Basic) and not value.is_number:
        v = sp.simplify(value); good = v == 0; shown = str(v)[:30]
    else:
        v = abs(float(value)); good = v <= tol; shown = f"{v:.3e}"
    ok &= bool(good); n_ok += bool(good)
    print(f"T11H1_SYM {name:<74s} {shown:<12s} {'PASS' if good else 'FAIL'}", flush=True)


def check(w, ell, gam, tag):
    vx, vy, vz = [sp.nsimplify(v) for v in w]
    cx, cy = vx / (2 * nu), vy / (2 * nu)
    Xf = lambda c, s: sp.exp(-c * s) * sp.sin(pi * s)
    S = Xf(cx, x) * Xf(cy, y)
    a_c = vz / (2 * nu)
    cc = vz ** 2 / (4 * nu) + nu * (2 * pi ** 2 + cx ** 2 + cy ** 2)
    G = lambda xi: sp.exp(-xi ** 2 / (4 * nu * tau)) / sp.sqrt(4 * pi * nu * tau)
    Qs = sp.exp(-a_c * z - cc * tau) * (G(z - Z1) - G(z + Z1))
    A0, C0 = Z1 ** 2 / (4 * nu), Z1 / (nu * sp.sqrt(4 * pi * nu))
    lam = 2 * pi ** 2
    if ell > 0:
        report(f"H1 {tag} -Lap_G S - 2 pi^2 S", -(sp.diff(S, x, 2) + sp.diff(S, y, 2)) - lam * S)
    for var, val in ((x, 0), (x, 1), (y, 0), (y, 1)):
        report(f"H1 {tag} S on dGamma_c {var}={val}", S.subs(var, val))
    report(f"H2 {tag} dQs/dz(0) - C0 tau^-3/2 e^(-A0/tau - cc tau)",
           sp.diff(Qs, z).subs(z, 0) - C0 * tau ** sp.Rational(-3, 2) * sp.exp(-A0 / tau - cc * tau))
    # ---- mpmath reference of the temporal factor
    z1m = mp.mpf(6) / 5   # exactly the sympy Z1 (not the binary double 1.2)
    A0m, C0m = z1m ** 2 / (4 * mp.mpf(0.25)), z1m / (mp.mpf(0.25) * mp.sqrt(4 * mp.pi * mp.mpf(0.25)))
    ccm = mp.mpf(str(sp.N(cc, 40)))
    am = mp.mpf("0.01") * (1 + mp.mpf(ell) ** 2 * 2 * mp.pi ** 2)
    F = lambda s: -mp.mpf(1) * mp.mpf(0.25) * C0m * s ** mp.mpf(-1.5) * mp.e ** (-A0m / s - ccm * s) if s > 0 else mp.mpf(0)
    gm = mp.mpf(gam)
    if gam > 0:
        k = mp.sqrt(am / gm)
        Eb = lambda b, tt: mp.sqrt(mp.pi) / (2 * mp.sqrt(A0m)) * (mp.e ** (2 * mp.sqrt(A0m * b)) * mp.erfc(mp.sqrt(A0m / tt) + mp.sqrt(b * tt))
                                                               + mp.e ** (-2 * mp.sqrt(A0m * b)) * mp.erfc(mp.sqrt(A0m / tt) - mp.sqrt(b * tt)))
        Ucf = lambda tt: (mp.mpf(0.25) * C0m / (2 * gm * k)) * (mp.e ** (k * tt) * Eb(ccm + k, tt) - mp.e ** (-k * tt) * Eb(ccm - k, tt))
        Utcf = lambda tt: (mp.mpf(0.25) * C0m / (2 * gm)) * (mp.e ** (k * tt) * Eb(ccm + k, tt) + mp.e ** (-k * tt) * Eb(ccm - k, tt))
        m3 = 0
        for b in (ccm + k, ccm - k):
            for tt in (mp.mpf("0.05"), mp.mpf("0.3"), mp.mpf(1)):
                q = mp.quad(lambda s: s ** mp.mpf(-1.5) * mp.e ** (-A0m / s - b * s), [0, tt / 4, tt])
                m3 = max(m3, abs(Eb(b, tt) - q) / abs(q))
        report(f"H3 {tag} E(b,tau) closed form vs quadrature (rel)", m3, 1e-25)
        m4 = m4t = 0
        for tt in (mp.mpf("0.02"), mp.mpf("0.1"), mp.mpf("0.4"), mp.mpf("0.75"), mp.mpf(1)):
            q = -1 / (gm * k) * mp.quad(lambda s: mp.sinh(k * (tt - s)) * F(s), [0, tt / 4, tt])
            qt = -1 / gm * mp.quad(lambda s: mp.cosh(k * (tt - s)) * F(s), [0, tt / 4, tt])
            m4 = max(m4, abs(Ucf(tt) - q) / abs(q)); m4t = max(m4t, abs(Utcf(tt) - qt) / abs(qt))
        report(f"H4 {tag} U closed form vs direct convolution quadrature (rel)", m4, 1e-20)   # mp.quad of the sinh-weighted kernel reaches ~1e-23 at 40 digits
        report(f"H4 {tag} U_tau closed form vs direct quadrature (rel)", m4t, 1e-20)
        m5 = 0
        for tt in (mp.mpf("0.05"), mp.mpf("0.3"), mp.mpf("0.9")):
            res = am * Ucf(tt) - gm * mp.diff(Ucf, tt, 2) - F(tt)
            m5 = max(m5, abs(res) / abs(F(tt)))
        report(f"H5 {tag} ODE a U - gamma U_tautau - F (rel to |F|)", m5, 1e-20)
        report(f"H6 {tag} natural endpoint U_tau(tau=1e-3) (u_t(T) = 0)", abs(Utcf(mp.mpf("0.001"))), 1e-150)
        report(f"H6 {tag} U(tau=1e-3) -> 0 (u(T) = 0)", abs(Ucf(mp.mpf("0.001"))), 1e-150)
        Um, Utt = Ucf, lambda tt: mp.diff(Ucf, tt, 2)
        Utm = lambda tt: -Utcf(tt)
    else:
        Um = lambda tt: F(tt) / am
        Utt = lambda tt: mp.diff(lambda s: F(s) / am, tt, 2)
        Utm = lambda tt: -mp.diff(lambda s: F(s) / am, tt)
    # ---- H7 stationarity on Gamma_c, mixed sympy/mpmath
    P = S * Qs   # near z = 0, phi = 0
    dnp = (-sp.diff(P, z)).subs(z, 0)
    LapS = sp.diff(S, x, 2) + sp.diff(S, y, 2)
    fS, fL, fD = (sp.lambdify((x, y), e, "mpmath") for e in (S, LapS, sp.diff(dnp, tau, 0) / 1))
    dn_l = sp.lambdify((x, y, tau), dnp, "mpmath")
    m7 = 0
    for (xv, yv) in ((0.3, 0.6), (0.5, 0.5), (0.81, 0.17)):
        for tt in (0.07, 0.4, 0.95):
            xv_, yv_, tt_ = mp.mpf(xv), mp.mpf(yv), mp.mpf(tt)
            res = mp.mpf("0.01") * (fS(xv_, yv_) * Um(tt_) - mp.mpf(ell) ** 2 * fL(xv_, yv_) * Um(tt_)) \
                - gm * fS(xv_, yv_) * Utt(tt_) - eta * nu * dn_l(xv_, yv_, tt_)
            m7 = max(m7, abs(res) / abs(mp.mpf(0.25) * dn_l(xv_, yv_, tt_)))
    report(f"H7 {tag} stationarity alpha_u(u - ell^2 Lap_G u) - gamma u_tt - eta nu dn p", m7, 1e-18)
    # ---- H8 state BCs with a symbolic temporal factor
    Uf = sp.Function("U")(t)
    Y = sp.cos(2 * t) * sp.sin(pi * x) ** 3 * sp.sin(pi * y) ** 3 * sp.sin(pi * z) ** 3 + eta * S * Uf * (1 - z) ** 3
    report(f"H8 {tag} y - eta u on Gamma_c", Y.subs(z, 0) - eta * S * Uf)
    for var, val in ((x, 0), (x, 1), (y, 0), (y, 1), (z, 1)):
        report(f"H8 {tag} y on {var}={val}", Y.subs(var, val))
    # ---- H9 adjoint (T11D p)
    adv = lambda q: vx * sp.diff(q, x) + vy * sp.diff(q, y) + vz * sp.diff(q, z)
    Pt = S * Qs.subs(tau, T - t)
    r = -sp.diff(Pt, t) - nu * (sp.diff(Pt, x, 2) + sp.diff(Pt, y, 2) + sp.diff(Pt, z, 2)) - adv(Pt)
    report(f"H9 {tag} adjoint r = 0 for z <= 1/2", sp.simplify(r / Pt))
    report(f"H9 {tag} p on z=0 (images)", Qs.subs(z, 0))
    # ---- H10 the admissible family: U + A cosh(k tau) also solves a U - gamma U_tautau = F with the natural
    # endpoint U_tau(0) = 0 (u_t(T) = 0); A only changes the DATUM u_0 = U(tau = T) S, which y_0 fixes.  The
    # benchmark takes A = 0 (u(T) = 0).  [A first draft asserted U_gamma -> F/a as gamma -> 0: FALSE for this
    # member, which grows like exp(k tau - 2 sqrt(A0 k)); the gamma = 0 benchmark is a different member.]
    if gam > 0:
        As, ks, gs, as_ = sp.symbols("A k g a", positive=True)
        hom = As * sp.cosh(ks * tau)
        report(f"H10 {tag} homogeneous mode: a A cosh - g (A cosh)'' with k^2 = a/g",
               sp.simplify((as_ * hom - gs * sp.diff(hom, tau, 2)).subs(ks, sp.sqrt(as_ / gs))))
        report(f"H10 {tag} homogeneous mode: d/dtau at tau = 0 (natural endpoint kept)", sp.diff(hom, tau).subs(tau, 0))
    # ---- H11 C++ literal port
    fw = [float(v) for v in w]
    nuf, auf, kPi = 0.25, 0.01, math.pi

    def cpp_qs3(tv):
        tau_ = 1.0 - tv
        if tau_ <= 1e-14: return (0.0,) * 5
        cx_, cy_ = fw[0] / (2 * nuf), fw[1] / (2 * nuf)
        lam_ = 2 * nuf * kPi ** 2 + nuf * (cx_ ** 2 + cy_ ** 2)
        a_, cc_ = fw[2] / (2 * nuf), fw[2] ** 2 / (4 * nuf) + lam_

        def ph(xi):
            p0 = math.exp(-xi * xi / (4 * nuf * tau_)) / math.sqrt(4 * kPi * nuf * tau_)
            return (p0, -xi / (2 * nuf * tau_) * p0, (xi * xi / (4 * nuf ** 2 * tau_ ** 2) - 1 / (2 * nuf * tau_)) * p0,
                    (-xi ** 3 / (8 * nuf ** 3 * tau_ ** 3) + 3 * xi / (4 * nuf ** 2 * tau_ ** 2)) * p0)
        A_, B_ = ph(-1.2), ph(1.2)
        V, Vz, Vzz, Vzzz = (A_[i] - B_[i] for i in range(4))
        E_ = math.exp(-cc_ * tau_)
        return (E_ * V, E_ * (Vz - a_ * V), 0, E_ * (nuf * Vzz - cc_ * V), E_ * (-cc_ * (Vz - a_ * V) + nuf * Vzzz - a_ * nuf * Vzz))

    def cpp_time(tv):
        tau_ = 1.0 - tv
        if tau_ <= 1e-14: return (0.0, 0.0)
        a = auf * (1.0 + ell * ell * 2.0 * kPi * kPi)
        if gam <= 0:
            q = cpp_qs3(tv); return (-nuf * q[1] / a, nuf * q[4] / a)
        k_ = math.sqrt(a / gam)
        cx_, cy_ = fw[0] / (2 * nuf), fw[1] / (2 * nuf)
        cc_ = fw[2] ** 2 / (4 * nuf) + 2 * nuf * kPi * kPi + nuf * (cx_ ** 2 + cy_ ** 2)
        A0_, C0_ = 1.2 * 1.2 / (4 * nuf), 1.2 / (nuf * math.sqrt(4 * kPi * nuf))
        def E(b):
            sa, rr, qq = math.sqrt(A0_ * b), math.sqrt(A0_ / tau_), math.sqrt(b * tau_)
            return math.sqrt(kPi) / (2 * math.sqrt(A0_)) * (math.exp(2 * sa) * math.erfc(rr + qq) + math.exp(-2 * sa) * math.erfc(rr - qq))
        ep, em = math.exp(k_ * tau_) * E(cc_ + k_), math.exp(-k_ * tau_) * E(cc_ - k_)
        s_ = nuf * C0_ / (2 * gam)
        return (s_ / k_ * (ep - em), -s_ * (ep + em))

    m11 = mt = 0
    for tv in (0.0, 0.25, 0.6, 0.93, 0.999):
        Uc, Utc = cpp_time(tv)
        tt = mp.mpf(1) - mp.mpf(tv)
        Ur, Utr = Um(tt), Utm(tt)
        scale = max(abs(Um(mp.mpf(s))) for s in ("0.2", "0.5", "1"))
        m11 = max(m11, abs(Uc - float(Ur)) / float(scale))
        mt = max(mt, abs(Utc - float(Utr)) / float(max(abs(Utm(mp.mpf(s))) for s in ("0.2", "0.5", "1"))))
    report(f"H11 {tag} C++ U(t) vs mpmath reference (max abs / scale)", m11, 1e-12)
    report(f"H11 {tag} C++ U_t(t) vs mpmath reference (max abs / scale)", mt, 1e-10)


cases = [((0, 0, 0), 0.3, 0.0, "heat ell=0.3"), ((0, 0, 0), 0.0, 1e-2, "heat gamma=1e-2"),
         ((0, 0, 0), 0.3, 1e-2, "heat combined"),
         ((0, 0, -0.75), 0.3, 0.0, "wN ell=0.3"), ((0, 0, -0.75), 0.0, 1e-2, "wN gamma=1e-2"),
         ((0, 0, -0.75), 0.3, 1e-2, "wN combined"), ((0, 0, 0.5), 0.3, 1e-2, "w(0,0,0.5) combined"),
         ((1, 0.5, -0.75), 0.0, 1e-2, "wA gamma=1e-2 (ell=0)")]
for w, ell, gam, tag in cases:
    check(w, ell, gam, tag)
print("T11H1_SYM_RESULT", "PASS" if ok else "FAIL", f"({n_ok} checks passed)")
sys.exit(0 if ok else 1)
