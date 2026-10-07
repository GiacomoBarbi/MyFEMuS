#!/usr/bin/env python3
"""T7N manufactured exact optimal control: convection + Neumann boundary control.

Omega = (0,1)^2 (x, z), Gamma_c = {z = 0} (FEMuS face 1), constant wind w = (wx, wz),
state (ex21 strong form):  y_t - nu Lap y + w.grad y = f,
   nu dn y = eta u on Gamma_c,  nu dn y = 0 on the other faces,  y(0) = y0;
objective  J = alpha_t/2 int ||y - y_d||^2_Omega + alpha_u/2 int ||u||^2_{Gamma_c};
DTO adjoint (ex21 sign, u = -(eta/alpha_u) p):
   -p_t - nu Lap p - w.grad p = alpha_t (y - y_d),
   nu dn p + (w.n) p = 0 on ALL of dOmega (Gamma_c included),  p(T) = 0;
stationarity on Gamma_c:  alpha_u u + eta p = 0.
Fields:
   X(x) = 1 - cos(2 pi x)            (X = X' = 0 at x = 0, 1)
   Z(z) = exp(-wz z/nu) (c0 + cos(pi z)),  c0 = 2   (nu Z' + wz Z = 0 at z = 0, 1)
   p = alpha_u B(t) X Z,   B = sin(T - t)
   u = -eta B X Z(0)  on Gamma_c
   y = A(t) cos(pi x) cos(pi z) + kappa B(t) X(x) H(z),  A = cos 2t,
       H = z - z^2/2 (H'(0) = 1, H'(1) = 0),  kappa = eta^2 Z(0)/nu
   f   = y_t - nu Lap y + w.grad y
   y_d = y - (-p_t - nu Lap p - w.grad p)/alpha_t
Checks every PDE / BC residual symbolically and at sample points; prints the
closed forms used by ExactOcBenchmark.cpp (case neumann-conv)."""
import sympy as sp
x, z, t = sp.symbols("x z t", real=True)
nu, wx, wz, eta, au, at, T = sp.symbols("nu w_x w_z eta alpha_u alpha_t T", positive=True)
c0 = 2
X = 1 - sp.cos(2 * sp.pi * x)
Z = sp.exp(-wz * z / nu) * (c0 + sp.cos(sp.pi * z))
B = sp.sin(T - t)
A = sp.cos(2 * t)
H = z - z**2 / 2
p = au * B * X * Z
Z0 = Z.subs(z, 0)
kappa = eta**2 * Z0 / nu
y = A * sp.cos(sp.pi * x) * sp.cos(sp.pi * z) + kappa * B * X * H
u = -(eta / au) * p.subs(z, 0)
lap = lambda q: sp.diff(q, x, 2) + sp.diff(q, z, 2)
f = sp.diff(y, t) - nu * lap(y) + wx * sp.diff(y, x) + wz * sp.diff(y, z)
yd = y - (-sp.diff(p, t) - nu * lap(p) - wx * sp.diff(p, x) - wz * sp.diff(p, z)) / at
checks = {
    # state PDE and adjoint PDE (identities by construction; recomputed)
    "state PDE  y_t - nu Lap y + w.grad y - f": sp.diff(y, t) - nu * lap(y) + wx * sp.diff(y, x) + wz * sp.diff(y, z) - f,
    "adjoint PDE -p_t - nu Lap p - w.grad p - alpha_t (y - y_d)":
        -sp.diff(p, t) - nu * lap(p) - wx * sp.diff(p, x) - wz * sp.diff(p, z) - at * (y - yd),
    # state BCs (outward normals): x=0 n=(-1,0); x=1 n=(1,0); z=0 n=(0,-1) [Gamma_c]; z=1 n=(0,1)
    "state  x=0: nu dn y": (-nu * sp.diff(y, x)).subs(x, 0),
    "state  x=1: nu dn y": (nu * sp.diff(y, x)).subs(x, 1),
    "state  z=1: nu dn y": (nu * sp.diff(y, z)).subs(z, 1),
    "state  z=0 (Gamma_c): nu dn y - eta u": (-nu * sp.diff(y, z)).subs(z, 0) - eta * u,
    # adjoint BCs nu dn p + (w.n) p
    "adjoint x=0: nu dn p + (w.n) p": (-nu * sp.diff(p, x) - wx * p).subs(x, 0),
    "adjoint x=1: nu dn p + (w.n) p": (nu * sp.diff(p, x) + wx * p).subs(x, 1),
    "adjoint z=0: nu dn p + (w.n) p": (-nu * sp.diff(p, z) - wz * p).subs(z, 0),
    "adjoint z=1: nu dn p + (w.n) p": (nu * sp.diff(p, z) + wz * p).subs(z, 1),
    "terminal p(T)": p.subs(t, T),
    "stationarity alpha_u u + eta p on Gamma_c": au * u + eta * p.subs(z, 0),
}
ok = True
for name, expr in checks.items():
    s = sp.simplify(expr)
    print(f"{name:62s}: {s}")
    ok = ok and s == 0
# the WRONG (symmetric-diffusion) adjoint BC nu dn p = 0 is violated by w.n p:
wrong = sp.simplify((-nu * sp.diff(p, z)).subs(z, 0))
print("wrong BC residual at z=0 (nu dn p, should be -(w.n)p = wz p):", sp.simplify(wrong - wz * p.subs(z, 0)) == 0)
# sample-point numerics (nu = 1/4, w = (1, 1/2), eta = 1, alpha_u = 1e-2, alpha_t = 1, T = 1)
vals = {nu: sp.Rational(1, 4), wx: 1, wz: sp.Rational(1, 2), eta: 1, au: sp.Rational(1, 100), at: 1, T: 1}
pts = [(0.3, 0.7, 0.2), (0.0, 0.4, 0.5), (1.0, 0.0, 0.9), (0.6, 1.0, 0.1), (0.25, 0.0, 0.0)]
worst = 0.0
for name, expr in checks.items():
    for (a, b, c) in pts:
        v = abs(float(expr.subs(vals).subs({x: a, z: b, t: c})))
        if name.startswith("state  ") or name.startswith("adjoint "):
            pass
        worst = max(worst, v) if (("x=0" in name and a == 0.0) or ("x=1" in name and a == 1.0) or
                                  ("z=0" in name and b == 0.0) or ("z=1" in name and b == 1.0) or
                                  "PDE" in name or "terminal" in name or "stationarity" in name) else worst
print(f"max |residual| at sample points: {worst:.3e}")
uval = float(u.subs(vals).subs({x: 0.5, t: 0.0}))
print(f"u(0.5, t=0) = {uval:.6f} (nontrivial); Z(0) = {float(Z0.subs(vals)):.6f}")
print("SYMBOLIC CHECKS", "PASS" if ok else "FAIL")
print("\nclosed forms (C++):")
for nm, e in (("f", f), ("y_d", yd)):
    print(nm, "=", sp.simplify(e))
