#!/usr/bin/env python3
"""T11D: symbolic verification of the 3D exact Dirichlet-control OC (dirichlet-conv-3d).

  Gamma_c = {z = 0};  y = 0 on the other five faces;  omega = (0,1)^2 x (1/2, 1)
  c_a = w_a/(2 nu);  X_c(s) = exp(-c s) sin(pi s);  tau = T - t
  Qs(z,tau) = exp(-a z - cc tau) [G(z - z1, tau) - G(z + z1, tau)],  z1 = 1.2,
     G = heat kernel of nu,  a = w_z/(2 nu),  cc = w_z^2/(4 nu) + nu(2 pi^2 + c_x^2 + c_y^2)
  phi = septic C^3 smoothstep 0 -> 1 on [1/2, 1];  Q = (1 - phi) Qs
  p = X_cx(x) X_cy(y) Q(z, tau)
  u = -(eta nu/alpha_u) X_cx X_cy dQs/dz(0, tau)                 (on Gamma_c)
  y = cos(2t) sin^3(pi x) sin^3(pi y) sin^3(pi z) + eta u(x, y, t) (1 - z)^3
  f = y_t - nu Lap y + w.grad y;  y_d = y - r/alpha_t,  r = -p_t - nu Lap p - w.grad p
Checks:
  D1 state PDE residual = 0
  D2 adjoint: r = -p_t - nu Lap p - w.grad p = 0 for z <= 1/2 (outside omega, where the
     KKT has no tracking), and = alpha_t (y - y_d) in omega (by construction)
  D3 y = eta u on Gamma_c;  D4 y = 0 on the five other faces
  D5 p = 0 on all six faces;  D6 p(T) = 0 (limit tau -> 0+, z in [0,1]: images at +-1.2)
  D7 stationarity alpha_u u = eta nu dn p on Gamma_c (n = -e_z; ex21 sign p = -lambda)
  D8 edge compatibility: u = 0 on dGamma_c;  y(t=0) on Gamma_c = eta u(0)
  D9 phi is C^3 at 1/2 and 1 (jumps of phi, phi', phi'', phi''' are 0; phi(1/2)=0, phi(1)=1)
  D10 C++ literal port (ExactOc3D.cpp Dir*) = sympy (f, y_d, y, p, u) at 5^3 x 3 points
Winds: w = 0, A = (1, 0.5, -0.75), B = (-2, 1, 0.5), N = (0, 0, -0.75) (normal; used for
the convergence ladders), and a symbolic (v_x, v_y, v_z) for D1-D8.
"""
import math, sys
import sympy as sp

x, y, z, t = sp.symbols("x y z t", real=True)
VX, VY, VZ = sp.symbols("v_x v_y v_z", real=True)
nu, T, eta, alpha_u, alpha_t = sp.Rational(1, 4), 1, 1, sp.Rational(1, 100), 1
pi, Z1 = sp.pi, sp.Rational(6, 5)
ok, rows = True, []


def report(name, value, tol=0.0):
    global ok
    if isinstance(value, sp.Basic) and not value.is_number:
        v = sp.simplify(value); good = v == 0; shown = str(v)[:40]
    else:
        v = abs(float(value)); good = v <= tol; shown = f"{v:.3e}"
    ok &= bool(good); rows.append(good)
    print(f"T11D_SYM {name:<66s} {shown:<12s} {'PASS' if good else 'FAIL'}")


def fields(w):
    vx, vy, vz = w
    cx, cy = vx / (2 * nu), vy / (2 * nu)
    tau = T - t
    G = lambda xi: sp.exp(-xi ** 2 / (4 * nu * tau)) / sp.sqrt(4 * pi * nu * tau)
    a = vz / (2 * nu)
    cc = vz ** 2 / (4 * nu) + nu * (2 * pi ** 2 + cx ** 2 + cy ** 2)
    Qs = sp.exp(-a * z - cc * tau) * (G(z - Z1) - G(z + Z1))
    Xf = lambda c, s: sp.exp(-c * s) * sp.sin(pi * s)
    Ps = Xf(cx, x) * Xf(cy, y) * Qs                      # p for z <= 1/2
    U = -eta * nu / alpha_u * Xf(cx, x) * Xf(cy, y) * sp.diff(Qs, z).subs(z, 0)
    Y = sp.cos(2 * t) * sp.sin(pi * x) ** 3 * sp.sin(pi * y) ** 3 * sp.sin(pi * z) ** 3 + eta * U * (1 - z) ** 3
    return Ps, Qs, U, Y, Xf, cx, cy


lap = lambda q: sp.diff(q, x, 2) + sp.diff(q, y, 2) + sp.diff(q, z, 2)
s_ = sp.symbols("s", real=True)
PHI = 35 * s_ ** 4 - 84 * s_ ** 5 + 70 * s_ ** 6 - 20 * s_ ** 7


def check(w, tag):
    vx, vy, vz = w
    adv = lambda q: vx * sp.diff(q, x) + vy * sp.diff(q, y) + vz * sp.diff(q, z)
    Ps, Qs, U, Y, Xf, cx, cy = fields(w)
    f = sp.diff(Y, t) - nu * lap(Y) + adv(Y)
    report(f"D1 {tag} state residual", sp.diff(Y, t) - nu * lap(Y) + adv(Y) - f)
    r = -sp.diff(Ps, t) - nu * lap(Ps) - adv(Ps)
    report(f"D2 {tag} adjoint r = 0 where phi = 0 (z <= 1/2)", sp.simplify(r / Ps))
    report(f"D3 {tag} y - eta u on Gamma_c (z=0)", Y.subs(z, 0) - eta * U)
    for var, val in ((y, 0), (x, 1), (y, 1), (x, 0), (z, 1)):
        report(f"D4 {tag} y on {var}={val}", Y.subs(var, val))
    for var, val in ((x, 0), (x, 1), (y, 0), (y, 1)):
        report(f"D5 {tag} p on {var}={val}", Ps.subs(var, val))
    report(f"D5 {tag} p on z=0 (image cancellation)", Qs.subs(z, 0))
    print(f"T11D_SYM D5 {tag} p on z=1: (1 - phi(1)) = 0 factor (D9) INFO")
    report(f"D7 {tag} alpha_u u - eta nu dn p on Gamma_c", alpha_u * U - eta * nu * (-sp.diff(Ps, z)).subs(z, 0))
    for var, val in ((x, 0), (x, 1), (y, 0), (y, 1)):
        report(f"D8 {tag} u on dGamma_c {var}={val}", U.subs(var, val))
    report(f"D8 {tag} y(t=0)|Gamma_c - eta u(0)", (Y.subs(z, 0) - eta * U).subs(t, 0))
    if any(isinstance(c, sp.Symbol) for c in w):
        return
    # D6: p(T) = 0 as a limit: max over z in [0,1] of |Qs| at tau -> 0
    Qn = sp.lambdify((z, t), Qs, "mpmath")
    import mpmath
    mpmath.mp.dps = 40
    m = max(abs(Qn(mpmath.mpf(zz) / 20, 1 - mpmath.mpf("1e-4"))) for zz in range(21))
    report(f"D6 {tag} max_z |Qs(z, tau=1e-4)| (-> 0: p(T) = 0)", float(m), 1e-150)
    fw = [float(c) for c in w]
    # ---- D10: literal float port of ExactOc3D.cpp
    nuf, au, kPi = 0.25, 0.01, math.pi

    def dirx(c, s):
        e, sn, cs = math.exp(-c * s), math.sin(kPi * s), math.cos(kPi * s)
        return (e * sn, e * (kPi * cs - c * sn), e * ((c * c - kPi * kPi) * sn - 2 * c * kPi * cs))

    def qs3(zz, tt):
        tau_ = 1.0 - tt
        if tau_ <= 1e-14:
            return (0.0,) * 5
        cx_, cy_ = fw[0] / (2 * nuf), fw[1] / (2 * nuf)
        lam = 2 * nuf * kPi ** 2 + nuf * (cx_ ** 2 + cy_ ** 2)
        a_, cc_ = fw[2] / (2 * nuf), fw[2] ** 2 / (4 * nuf) + lam

        def ph(xi):
            p0 = math.exp(-xi * xi / (4 * nuf * tau_)) / math.sqrt(4 * kPi * nuf * tau_)
            return (p0, -xi / (2 * nuf * tau_) * p0, (xi * xi / (4 * nuf ** 2 * tau_ ** 2) - 1 / (2 * nuf * tau_)) * p0,
                    (-xi ** 3 / (8 * nuf ** 3 * tau_ ** 3) + 3 * xi / (4 * nuf ** 2 * tau_ ** 2)) * p0)
        A_, B_ = ph(zz - 1.2), ph(zz + 1.2)
        V, Vz, Vzz, Vzzz = (A_[i] - B_[i] for i in range(4))
        E = math.exp(-a_ * zz - cc_ * tau_)
        return (E * V, E * (Vz - a_ * V), E * (Vzz - 2 * a_ * Vz + a_ * a_ * V), E * (nuf * Vzz - cc_ * V),
                E * (-cc_ * (Vz - a_ * V) + nuf * Vzzz - a_ * nuf * Vzz))

    def step(zz):
        L = 0.5; s = (zz - 0.5) / L
        if s <= 0: return 0.0, 0.0, 0.0
        if s >= 1: return 1.0, 0.0, 0.0
        return (s ** 4 * (35 - 84 * s + 70 * s * s - 20 * s ** 3), 140 * s ** 3 * (1 - s) ** 3 / L,
                420 * s * s * (1 - s) ** 2 * (1 - 2 * s) / L ** 2)

    def q3(zz, tt):
        q = qs3(zz, tt); f0, f1, f2 = step(zz)
        return ((1 - f0) * q[0], -f1 * q[0] + (1 - f0) * q[1], -f2 * q[0] - 2 * f1 * q[1] + (1 - f0) * q[2], -(1 - f0) * q[3])

    def sin3(s):
        sn, cs = math.sin(kPi * s), math.cos(kPi * s)
        return (sn ** 3, 3 * kPi * sn * sn * cs, 3 * kPi ** 2 * sn * (2 * cs * cs - sn * sn))

    def cpp(xv, yv, zv, tv):
        cx_, cy_ = fw[0] / (2 * nuf), fw[1] / (2 * nuf)
        X_, Y_ = dirx(cx_, xv), dirx(cy_, yv)
        S = [sin3(xv), sin3(yv), sin3(zv)]
        Sv = S[0][0] * S[1][0] * S[2][0]
        Sw = fw[0] * S[0][1] * S[1][0] * S[2][0] + fw[1] * S[0][0] * S[1][1] * S[2][0] + fw[2] * S[0][0] * S[1][0] * S[2][1]
        Sl = S[0][2] * S[1][0] * S[2][0] + S[0][0] * S[1][2] * S[2][0] + S[0][0] * S[1][0] * S[2][2]
        q0 = qs3(0.0, tv)
        k, kd = -nuf / au * q0[1], nuf / au * q0[4]
        g, g1, g2 = (1 - zv) ** 3, -3 * (1 - zv) ** 2, 6 * (1 - zv)
        Av, dAv = math.cos(2 * tv), -2 * math.sin(2 * tv)
        yv_ = Av * Sv + k * X_[0] * Y_[0] * g
        wg = Av * Sw + k * (fw[0] * X_[1] * Y_[0] * g + fw[1] * X_[0] * Y_[1] * g + fw[2] * X_[0] * Y_[0] * g1)
        lp = Av * Sl + k * ((X_[2] * Y_[0] + X_[0] * Y_[2]) * g + X_[0] * Y_[0] * g2)
        dt_ = dAv * Sv + kd * X_[0] * Y_[0] * g
        src = dt_ - nuf * lp + wg
        Q = q3(zv, tv)
        rr = (-X_[0] * Y_[0] * Q[3] - nuf * (X_[2] * Y_[0] * Q[0] + X_[0] * Y_[2] * Q[0] + X_[0] * Y_[0] * Q[2])
              - (fw[0] * X_[1] * Y_[0] * Q[0] + fw[1] * X_[0] * Y_[1] * Q[0] + fw[2] * X_[0] * Y_[0] * Q[1]))
        uu = -nuf / au * X_[0] * Y_[0] * q0[1]
        return {"f": src, "yd": yv_ - rr, "y": yv_, "p": X_[0] * Y_[0] * Q[0], "u": uu}

    phi_z = sp.Piecewise((0, z <= sp.Rational(1, 2)), (1, z >= 1), (PHI.subs(s_, 2 * z - 1), True))
    P = (1 - phi_z) * Ps
    rfull = -sp.diff(P, t) - nu * lap(P) - adv(P)
    yd = Y - rfull / alpha_t
    ref = {k: sp.lambdify((x, y, z, t), e) for k, e in (("f", f), ("yd", yd), ("y", Y), ("p", P), ("u", U))}
    m = {k: [0.0, 0.0] for k in ref}
    pts = [0.0, 0.17, 0.5, 0.83, 1.0]
    for xv in pts:
        for yv in pts:
            for zv in (0.0, 0.3, 0.5, 0.71, 0.93):
                for tv in (0.0, 0.4, 0.9):
                    got = cpp(xv, yv, zv, tv)
                    for k in ref:
                        rv = float(ref[k](xv, yv, zv, tv))
                        m[k][0] = max(m[k][0], abs(got[k] - rv)); m[k][1] = max(m[k][1], abs(rv))
    for k in m:
        report(f"D10 {tag} C++ {k} vs sympy (max abs / max |field|)", m[k][0] / max(m[k][1], 1e-300), 1e-12)


# D9: smoothstep
for k in range(4):
    d = sp.diff(PHI, s_, k)
    report(f"D9 phi^({k}) at s=0 (matches 0)", d.subs(s_, 0))
    report(f"D9 phi^({k}) at s=1 (matches {'1' if k == 0 else '0'})", d.subs(s_, 1) - (1 if k == 0 else 0))
check((0, 0, 0), "w=0")
for w in ((1.0, 0.5, -0.75), (-2.0, 1.0, 0.5), (0.0, 0.0, -0.75)):
    check(tuple(sp.nsimplify(v) for v in w), "w=(" + ",".join(f"{v:g}" for v in w) + ")")
check((VX, VY, VZ), "w=symbolic")
print("T11D_SYM_RESULT", "PASS" if ok else "FAIL", f"({sum(rows)}/{len(rows)})")
sys.exit(0 if ok else 1)
