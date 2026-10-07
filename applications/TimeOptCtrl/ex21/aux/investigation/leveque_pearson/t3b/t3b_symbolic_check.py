#!/usr/bin/env python3
"""Symbolic check of the T3b manufactured case `lift-track` (ExactOcBenchmark.cpp),
ex21 convention, data written exactly as the C++ forms them:
  g  = 1/2 + exp((vx x + vy z - vx - vy)/nu)          (-nu Lap g + v.grad g = 0)
  y  = g + cos(2t) S1,  S1 = sin^3(pi x) sin^3(pi z)
  p  = alpha_u sin(T - t) S2,  S2 = sin(2 pi x) sin(pi z)   (s = sin, NOT cubed)
  u  = -eta p / alpha_u
  f  = y_t - nu Lap y + v.grad y - eta u
  yd = y + (p_t + nu Lap p + v.grad p)/alpha_t
Checks: the three optimality residuals, y = g and p = 0 on dOmega, f = 0 on dOmega
(source convention exact), and y_d - g != 0 on dOmega (the T3b term is exercised)."""
import sympy as sp
x, z, t = sp.symbols("x z t", real=True)
T, nu, vx, vy, au, eta, at = sp.symbols("T nu v_x v_y alpha_u eta alpha_t", positive=True)
lap = lambda q: sp.diff(q, x, 2) + sp.diff(q, z, 2)
adv = lambda q: vx * sp.diff(q, x) + vy * sp.diff(q, z)
g = sp.Rational(1, 2) + sp.exp((vx * x + vy * z - vx - vy) / nu)
S1 = sp.sin(sp.pi * x) ** 3 * sp.sin(sp.pi * z) ** 3
S2 = sp.sin(2 * sp.pi * x) * sp.sin(sp.pi * z)
y = g + sp.cos(2 * t) * S1
p = au * sp.sin(T - t) * S2
u = -eta * p / au
f = sp.diff(sp.cos(2 * t), t) * S1 - nu * sp.cos(2 * t) * lap(S1) + sp.cos(2 * t) * adv(S1) - eta * u \
    - nu * lap(g) + adv(g)
yd = y + (sp.diff(p, t) + nu * lap(p) + adv(p)) / at
ok = True
def rep(label, e, want_zero=True):
    global ok
    r = sp.simplify(sp.expand(e))
    print(f"  {label:58s} {r}")
    ok &= (r == 0) == want_zero
rep("state    y_t - nu Lap y + v.grad y - f - eta u", sp.diff(y, t) - nu * lap(y) + adv(y) - f - eta * u)
rep("adjoint -p_t - nu Lap p - v.grad p - alpha_t (y - y_d)", -sp.diff(p, t) - nu * lap(p) - adv(p) - at * (y - yd))
rep("gradient alpha_u u + eta p", au * u + eta * p)
rep("steady lift  -nu Lap g + v.grad g", -nu * lap(g) + adv(g))
rep("p(T)", p.subs(t, T))
for side, s in (("x=0", {x: 0}), ("x=1", {x: 1}), ("z=0", {z: 0}), ("z=1", {z: 1})):
    rep(f"y - g on {side}", (y - g).subs(s))
    rep(f"p on {side}", p.subs(s))
    rep(f"f on {side}", f.subs(s))
    print(f"  y_d - g on {side}: {sp.simplify((yd - g).subs(s))}")
ok &= sp.simplify((yd - g).subs(x, 0)) != 0
print("ALL CHECKS PASS (y_d - g != 0 on the boundary)" if ok else "FAILURE")
raise SystemExit(0 if ok else 1)
