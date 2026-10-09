#!/usr/bin/env python3
"""T14: symbolic verification of the variable-wind formulation and exact cases (ExactOc3D.cpp, g_var).

Fields: w = H x, H symmetric (quadratic potential Phi = x^T H x / 2); harmonic-3d (tr H = 0) and
affine-div-3d (tr H = 1); a fully SYMBOLIC symmetric H where feasible.  Checks (exact unless a tolerance):
 W1  -div(w p) = -w.grad p - (div w) p                       (symbolic p, symbolic H)
 W2  presets: w = grad Phi, div w = 0 / 1, w.n nonconstant on every face
 D*  conv-sin3-3d (distributed, homogeneous Dirichlet state): state / FULL adjoint residuals; y, p, grad y,
     grad p vanish on all six faces; p(T) = 0; stationarity alpha_u u + eta p = 0;
     NEG: target WITHOUT the (div w) p term leaves an adjoint residual for affine-div (none for harmonic)
 N*  natural-var-3d: p = alpha_u B exp(-Phi/nu) Q: Robin nu dn p + (w.n) p = 0 on all six faces for a SYMBOLIC
     symmetric H; nu dn y = 0; state / full adjoint residuals; p(T) = 0;
     NEG: Robin with the FACE-AVERAGED w.n (a facewise constant normal velocity) fails
 M*  neumann-var-3d / -opp-3d: nu dn y = eta u on each controlled face, 0 on the others; Robin on all faces;
     stationarity on Gamma_c; residuals; p(T) = 0
 S*  dirichlet-var-3d swirl: div w = 0; w = (0, 0, gamma) for z <= 1/2; tangential parts 0 on the lateral faces;
     the T11D adjoint has zero residual below omega; y = eta u on zmin, 0 on the other faces
 P*  literal ports of the C++ formulas (PotShape, VarG, VarY / VarTarget3D) vs sympy at sample points
usage: t14_symbolic.py <out.csv>"""
import csv, itertools, math, sys
import sympy as sp

x, y, z, t = sp.symbols("x y z t", real=True)
X3 = (x, y, z)
nu_q, T, eta, alpha_u, alpha_t = sp.Rational(1, 4), 1, 1, sp.Rational(1, 100), 1
pi = sp.pi
rows, ok = [], True
FACES = [(z, 0, -1, "zmin"), (y, 0, -1, "ymin"), (x, 1, 1, "xmax"), (y, 1, 1, "ymax"), (x, 0, -1, "xmin"), (z, 1, 1, "zmax")]
HARM = sp.Matrix([[0, 1, sp.Rational(-3, 4)], [1, 0, sp.Rational(1, 2)], [sp.Rational(-3, 4), sp.Rational(1, 2), 0]])
AFF = HARM + sp.diag(sp.Rational(1, 2), sp.Rational(1, 4), sp.Rational(1, 4))
PTS = [(0.13, 0.41, 0.77, 0.3), (0.62, 0.27, 0.35, 0.8), (0.9, 0.55, 0.1, 0.5)]


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
    ok &= bool(good)
    rows.append({"check": name, "value": shown, "verdict": "PASS" if good else "FAIL"})
    print(f"T14_SYM {name:<84s} {shown:<14s} {'PASS' if good else 'FAIL'}", flush=True)


def grad(q): return [sp.diff(q, v) for v in X3]
def lap(q): return sum(sp.diff(q, v, 2) for v in X3)
def num(expr, extra=None):
    f = sp.lambdify((x, y, z, t), expr, "math")
    return max(abs(f(*p)) for p in PTS)


def wind(H):
    w = list(H * sp.Matrix(X3))
    Phi = sp.Rational(1, 2) * (sp.Matrix(X3).T * H * sp.Matrix(X3))[0]
    return w, Phi


A_, B_ = sp.cos(2 * t), sp.sin(T - t)
# ---- W1 / W2
hs = sp.symbols("h11 h12 h13 h22 h23 h33", real=True)
HS = sp.Matrix([[hs[0], hs[1], hs[2]], [hs[1], hs[3], hs[4]], [hs[2], hs[4], hs[5]]])
wS, PhiS = wind(HS)
pf = sp.Function("p")(x, y, z)
divwp = sum(sp.diff(wS[k] * pf, X3[k]) for k in range(3))
report("W1 -div(w p) + w.grad p + (div w) p = 0 (symbolic H, symbolic p)",
       sp.simplify(-divwp + sum(wS[k] * sp.diff(pf, X3[k]) for k in range(3)) + sum(sp.diff(wS[k], X3[k]) for k in range(3)) * pf))
for lab, H, dv in (("harmonic-3d", HARM, 0), ("affine-div-3d", AFF, 1)):
    w, Phi = wind(H)
    report(f"W2 {lab}: w = grad Phi", sum(sp.simplify(w[k] - sp.diff(Phi, X3[k])) ** 2 for k in range(3)))
    report(f"W2 {lab}: div w - {dv}", sp.simplify(sum(sp.diff(w[k], X3[k]) for k in range(3)) - dv))
    for var, val, sgn, name in FACES:
        wn = (sgn * w["xyz".index(str(var))]).subs(var, val)
        report(f"W2 {lab}: w.n nonconstant on {name}", len(wn.free_symbols), expect_nonzero=True)
# ---- D: conv-sin3-3d
S3 = sp.sin(pi * x) ** 3 * sp.sin(pi * y) ** 3 * sp.sin(pi * z) ** 3
for lab, H in (("harmonic-3d", HARM), ("affine-div-3d", AFF)):
    w, Phi = wind(H); dv = sum(sp.diff(w[k], X3[k]) for k in range(3))
    nu = nu_q
    Y = A_ * S3; P = alpha_u * B_ * S3; U = -eta * P / alpha_u
    adv = lambda q: sum(w[k] * sp.diff(q, X3[k]) for k in range(3))
    f = sp.diff(Y, t) - nu * lap(Y) + adv(Y) - eta * U
    yd = Y + (sp.diff(P, t) + nu * lap(P) + adv(P) + dv * P) / alpha_t
    report(f"D1 {lab}: state residual", num(sp.diff(Y, t) - nu * lap(Y) + adv(Y) - f - eta * U), 1e-12)
    adj = -sp.diff(P, t) - nu * lap(P) - sum(sp.diff(w[k] * P, X3[k]) for k in range(3)) - alpha_t * (Y - yd)
    report(f"D2 {lab}: FULL adjoint residual -p_t - nu Lap p - div(w p) - alpha_t (y - y_d)", num(adj), 1e-12)
    for var, val, sgn, name in FACES:
        report(f"D3 {lab}: y, p, grad y, grad p on {name}",
               sum(abs(sp.simplify(e.subs(var, val))) for e in [Y, P] + grad(Y) + grad(P)) if False else
               sum(sp.simplify(e.subs(var, val)) ** 2 for e in [Y, P] + grad(Y) + grad(P)))
    report(f"D4 {lab}: p(T)", sp.simplify(P.subs(t, T)))
    report(f"D5 {lab}: stationarity alpha_u u + eta p", sp.simplify(alpha_u * U + eta * P))
    yd_wrong = Y + (sp.diff(P, t) + nu * lap(P) + adv(P)) / alpha_t
    adjw = -sp.diff(P, t) - nu * lap(P) - sum(sp.diff(w[k] * P, X3[k]) for k in range(3)) - alpha_t * (Y - yd_wrong)
    if lab == "affine-div-3d":
        report(f"D6 {lab}: NEG target without (div w) p: adjoint residual", num(adjw), expect_nonzero=True)
    else:
        report(f"D6 {lab}: target without (div w) p coincides (div w = 0)", num(adjw), 1e-12)
# ---- N: natural-var-3d with SYMBOLIC H (Robin), numeric presets for residuals
nuN = sp.Rational(1, 2)
Q = (2 + sp.cos(pi * x)) * (2 + sp.cos(pi * y)) * (2 + sp.cos(pi * z))
PS = alpha_u * B_ * sp.exp(-PhiS / nuN) * Q
for var, val, sgn, name in FACES:
    dn = lambda q: sgn * sp.diff(q, var)
    wn = sgn * wS["xyz".index(str(var))]
    report(f"N1 natural-var Robin nu dn p + (w.n) p on {name} (symbolic H)", sp.simplify((nuN * dn(PS) + wn * PS).subs(var, val)))
for lab, H in (("harmonic-3d", HARM), ("affine-div-3d", AFF)):
    w, Phi = wind(H); dv = sum(sp.diff(w[k], X3[k]) for k in range(3))
    adv = lambda q: sum(w[k] * sp.diff(q, X3[k]) for k in range(3))
    Y = A_ * sp.cos(pi * x) * sp.cos(pi * y) * sp.cos(pi * z)
    P = alpha_u * B_ * sp.exp(-Phi / nuN) * Q; U = -eta * P / alpha_u
    f = sp.diff(Y, t) - nuN * lap(Y) + adv(Y) - eta * U
    yd = Y + (sp.diff(P, t) + nuN * lap(P) + adv(P) + dv * P) / alpha_t
    adj = -sp.diff(P, t) - nuN * lap(P) - sum(sp.diff(w[k] * P, X3[k]) for k in range(3)) - alpha_t * (Y - yd)
    report(f"N2 {lab}: FULL adjoint residual", num(adj), 1e-12)
    report(f"N3 {lab}: p(T)", sp.simplify(P.subs(t, T)))
    for var, val, sgn, name in FACES:
        report(f"N4 {lab}: nu dn y on {name}", sp.simplify((sgn * sp.diff(Y, var)).subs(var, val)))
    # NEG: facewise-constant normal velocity (face average of w.n) in the Robin condition
    var, val, sgn, name = FACES[4]   # xmin
    wn = (sgn * w[0]).subs(x, 0)
    wn_avg = sp.integrate(sp.integrate(wn, (y, 0, 1)), (z, 0, 1))
    report(f"N5 {lab}: NEG Robin with the face-averaged w.n on xmin", num((nuN * sgn * sp.diff(P, x) + wn_avg * P).subs(x, 0)),
           expect_nonzero=True)
    # P1 port: PotShape formulas (gradient, Laplacian of exp(-Phi/nu) Q)
    E = sp.exp(-Phi / nuN); Pp = E * Q
    Qx, Qy, Qz = grad(Q); lapQ = lap(Q); ww = sum(c * c for c in w); wgQ = w[0] * Qx + w[1] * Qy + w[2] * Qz
    port_lap = E * (lapQ - 2 * wgQ / nuN - Q * dv / nuN + Q * ww / nuN ** 2)
    report(f"P1 {lab}: C++ PotShape Laplacian vs sympy", num(port_lap - lap(Pp)) / num(lap(Pp)), 1e-12)
    report(f"P1 {lab}: C++ PotShape gradient vs sympy",
           sum(num(E * (gq - Q * w[k] / nuN) - sp.diff(Pp, X3[k])) for k, gq in enumerate((Qx, Qy, Qz))), 1e-12)
# ---- M: neumann-var-3d (zmin) and -opp-3d (zmin u zmax)
Xf = lambda s: 1 - sp.cos(2 * pi * s)
for case, faces in (("neumann-var-3d", ["zmin"]), ("neumann-var-opp-3d", ["zmin", "zmax"])):
    for lab, H in (("harmonic-3d", HARM), ("affine-div-3d", AFF)):
        w, Phi = wind(H); dv = sum(sp.diff(w[k], X3[k]) for k in range(3))
        adv = lambda q: sum(w[k] * sp.diff(q, X3[k]) for k in range(3))
        P = alpha_u * B_ * sp.exp(-Phi / nuN) * Xf(x) * Xf(y) * (2 + sp.cos(pi * z))
        U = -eta * P / alpha_u
        Y = A_ * sp.cos(pi * x) * sp.cos(pi * y) * sp.cos(pi * z)
        kap = eta ** 2 * B_ / nuN
        G0 = (sp.exp(-Phi / nuN) * Xf(x) * Xf(y) * (2 + sp.cos(pi * z))).subs(z, 0)
        Y = Y + kap * G0 * (z - z ** 2 / 2)
        if "zmax" in faces:
            G1 = (sp.exp(-Phi / nuN) * Xf(x) * Xf(y) * (2 + sp.cos(pi * z))).subs(z, 1)
            Y = Y - kap * G1 * z ** 2 / 2
        f = sp.diff(Y, t) - nuN * lap(Y) + adv(Y)
        yd = Y + (sp.diff(P, t) + nuN * lap(P) + adv(P) + dv * P) / alpha_t
        tag = f"{case} {lab}"
        report(f"M1 {tag}: FULL adjoint residual", num(-sp.diff(P, t) - nuN * lap(P) - sum(sp.diff(w[k] * P, X3[k]) for k in range(3)) - alpha_t * (Y - yd)), 1e-12)
        for var, val, sgn, name in FACES:
            dn = lambda q: sgn * sp.diff(q, var)
            flux = (nuN * dn(Y) - (eta * U if name in faces else 0)).subs(var, val)
            report(f"M2 {tag}: nu dn y {'- eta u ' if name in faces else ''}on {name}", num(flux), 1e-12)
            wn = sgn * w["xyz".index(str(var))]
            report(f"M3 {tag}: Robin on {name}", num((nuN * dn(P) + wn * P).subs(var, val)), 1e-12)
        report(f"M4 {tag}: p(T)", sp.simplify(P.subs(t, T)))
        report(f"M5 {tag}: stationarity on Gamma_c", sp.simplify(alpha_u * U + eta * P))
        # P2 port: VarG Laplacian (x, y only) on z = 0
        R = Xf(x) * Xf(y) * 3; phi0 = Phi.subs(z, 0); E0 = sp.exp(-phi0 / nuN)
        Rx, Ry = sp.diff(R, x), sp.diff(R, y)
        port = E0 * (sp.diff(R, x, 2) + sp.diff(R, y, 2) - 2 * (sp.diff(phi0, x) * Rx + sp.diff(phi0, y) * Ry) / nuN
                     - R * (H[0, 0] + H[1, 1]) / nuN + R * (sp.diff(phi0, x) ** 2 + sp.diff(phi0, y) ** 2) / nuN ** 2)
        exact = sp.diff(E0 * R, x, 2) + sp.diff(E0 * R, y, 2)
        report(f"P2 {tag}: C++ VarG Laplacian vs sympy", num(port - exact) / num(exact), 1e-12)
# ---- S: dirichlet-var-3d swirl
gam = sp.Rational(-3, 4)
s_ = lambda v: sp.sin(pi * v) ** 2
ds_ = lambda v: pi * sp.sin(2 * pi * v)
gs = sp.Function("g")(z)
wsw = [sp.Rational(1, 2) * s_(x) * ds_(y) * gs, -sp.Rational(1, 2) * ds_(x) * s_(y) * gs, gam]
report("S1 swirl div w = 0 (any g(z))", sp.simplify(sum(sp.diff(wsw[k], X3[k]) for k in range(3))))
report("S1 swirl ds/dx = derivative of s (port: pi sin 2 pi x)", sp.simplify(sp.diff(s_(x), x) - ds_(x)))
for var, val, name in ((x, 0, "xmin"), (x, 1, "xmax"), (y, 0, "ymin"), (y, 1, "ymax")):
    report(f"S2 swirl tangential parts on {name}", sp.simplify(wsw[0].subs(var, val) ** 2 + wsw[1].subs(var, val) ** 2))
sstep = sp.Symbol("s")
septic = sstep ** 4 * (35 - 84 * sstep + 70 * sstep ** 2 - 20 * sstep ** 3)
report("S3 septic step g(1/2) = 0, g'(1/2) = 0, g''(1/2) = 0 (w = (0,0,gamma) for z <= 1/2)",
       sum(sp.diff(septic, sstep, k).subs(sstep, 0) ** 2 for k in range(3)))
# T11D adjoint below omega with the constant wind (0, 0, gamma): homogeneous (residual / p on a grid)
nuD = nu_q
a = gam / (2 * nuD); cc = gam ** 2 / (4 * nuD) + 2 * nuD * pi ** 2
tau = T - t
Phi_k = lambda xi: sp.exp(-xi ** 2 / (4 * nuD * tau)) / sp.sqrt(4 * pi * nuD * tau)
Qs = sp.exp(-a * z - cc * tau) * (Phi_k(z - sp.Rational(6, 5)) - Phi_k(z + sp.Rational(6, 5)))
Pd = sp.sin(pi * x) * sp.sin(pi * y) * Qs
res = -sp.diff(Pd, t) - nuD * lap(Pd) - gam * sp.diff(Pd, z)
fr = sp.lambdify((x, y, z, t), sp.simplify(res / Pd), "math")
report("S4 T11D adjoint homogeneous for the wind (0,0,gamma) (z <= 1/2, residual/p on a grid)",
       max(abs(fr(xx, yy, zz, tt)) for xx, yy, zz, tt in itertools.product((0.3, 0.7), (0.4,), (0.1, 0.3, 0.45), (0.2, 0.7))), 1e-11)
for var, val, name in ((x, 0, "xmin"), (x, 1, "xmax"), (y, 0, "ymin"), (y, 1, "ymax"), (z, 0, "zmin")):
    report(f"S5 dirichlet-var p = 0 on {name}", sp.simplify(Pd.subs(var, val)))
Ut = sp.Function("u")(x, y, t)
Yd = A_ * S3 + eta * Ut * (1 - z) ** 3
report("S6 dirichlet-var y - eta u on zmin", sp.simplify(Yd.subs(z, 0) - eta * Ut))
report("S6 dirichlet-var y on zmax", sp.simplify(Yd.subs(z, 1)))
with open(sys.argv[1], "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
print("T14_SYM_RESULT", "PASS" if ok else "FAIL", f"({sum(r['verdict'] == 'PASS' for r in rows)}/{len(rows)})")
sys.exit(0 if ok else 1)
