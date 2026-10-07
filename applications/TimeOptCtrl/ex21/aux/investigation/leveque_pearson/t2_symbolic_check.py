#!/usr/bin/env python3
"""Symbolic verification of the exact optimal-control solutions used by T2 and T1
(ex21/include/ExactOcBenchmark.cpp), in the ex21 sign convention:

    state     y_t - nu Lap y + v.grad y - f - eta u          = 0
    adjoint  -p_t - nu Lap p - v.grad p - alpha_t (y - y_d)  = 0
    gradient  alpha_u u + eta p                              = 0
    p(T) = 0, y = p = 0 on dOmega.

The data f, y_d are written EXACTLY as the C++ code forms them (not re-derived
from the residuals), so a sign slip in the code would show here as a nonzero
residual.  Also checks the boundary traces of f and y_d (the reason for the two
shapes), and, for T1, the identity p_ex21 = -p_LP against Leveque-Pearson's own
(OTD) optimality system.
"""
import sympy as sp

x, z, t = sp.symbols("x z t", real=True)
T, nu, vx, vy, au, eta, at = sp.symbols("T nu v_x v_y alpha_u eta alpha_t", positive=True)
lap = lambda g: sp.diff(g, x, 2) + sp.diff(g, z, 2)
adv = lambda g, a, b: a * sp.diff(g, x) + b * sp.diff(g, z)

ok = True
def report(label, expr):
    global ok
    r = sp.simplify(sp.expand_trig(sp.expand(expr)))
    print(f"  {label:60s} {r}")
    ok &= (r == 0)

for shape in ("sin3", "sin"):
    s = (lambda a: sp.sin(a) ** 3) if shape == "sin3" else sp.sin
    S1 = s(sp.pi * x) * s(sp.pi * z)
    S2 = s(2 * sp.pi * x) * s(sp.pi * z)
    a, b = sp.cos(2 * t), sp.sin(T - t)
    y = a * S1
    p = au * b * S2
    u = -eta * p / au                                   # ExactU
    # --- data exactly as in ExactOcBenchmark.cpp (eta = 1, alpha_t = 1 there,
    #     kept symbolic here so the residuals test the general form)
    f = sp.diff(a, t) * S1 - nu * a * lap(S1) + a * adv(S1, vx, vy) - eta * u
    q = au * (sp.diff(b, t) * S2 + nu * b * lap(S2) + b * adv(S2, vx, vy))
    yd = y + q / at
    print(f"T2 shape={shape}")
    report("state  y_t - nu Lap y + v.grad y - f - eta u", sp.diff(y, t) - nu * lap(y) + adv(y, vx, vy) - f - eta * u)
    report("adjoint -p_t - nu Lap p - v.grad p - alpha_t (y - y_d)", -sp.diff(p, t) - nu * lap(p) - adv(p, vx, vy) - at * (y - yd))
    report("gradient alpha_u u + eta p", au * u + eta * p)
    report("p(T)", p.subs(t, T))
    for side, sub in (("x=0", {x: 0}), ("x=1", {x: 1}), ("z=0", {z: 0}), ("z=1", {z: 1})):
        report(f"y on {side}", y.subs(sub))
        report(f"p on {side}", p.subs(sub))
    # boundary trace of the data (expected 0 for sin3, nonzero for sin with v != 0)
    fx0 = sp.simplify(f.subs(x, 0))
    yd0 = sp.simplify(yd.subs(x, 0))
    print(f"  f   on x=0 : {fx0}")
    print(f"  y_d on x=0 : {yd0}")
    if shape == "sin3":
        ok &= fx0 == 0 and yd0 == 0
    # the numerical values used by the runs
    vals = {nu: sp.Rational(1, 4), vx: 1, vy: sp.Rational(1, 2), T: 1, au: sp.Rational(1, 100), eta: 1, at: 1}
    print("  convection/diffusion size at (x,z)=(0.3,0.4), t=0: |v.grad y| / |nu Lap y| =",
          sp.N(sp.Abs(adv(y, vx, vy) / (nu * lap(y))).subs(vals).subs({x: 0.3, z: 0.4, t: 0}), 4))

# --- T1: Leveque-Pearson section 5.1 (from ex20 EX20_LEVEQUE_PEARSON_2022_BENCHMARK.md section 4)
beta = sp.symbols("beta", positive=True)
s = sp.cos(sp.pi * x / 2) * sp.cos(sp.pi * z / 2)
yLP = 1 + (2 / (sp.pi**2 * beta) * sp.exp(T) - 2 / ((2 + sp.pi**2) * beta) * sp.exp(t)) * s
pLP = (sp.exp(T) - sp.exp(t)) * s
uLP = pLP / beta
yhat = 1 + ((2 / (sp.pi**2 * beta) + sp.pi**2 / 2) * sp.exp(T)
            + (1 - 2 / ((2 + sp.pi**2) * beta) - sp.pi**2 / 2) * sp.exp(t)) * s
print("T1 Leveque-Pearson 5.1, paper (OTD) convention")
report("y_t - Lap y - u", sp.diff(yLP, t) - lap(yLP) - uLP)
report("-p_t - Lap p - (yhat - y)", -sp.diff(pLP, t) - lap(pLP) - (yhat - yLP))
report("beta u - p", beta * uLP - pLP)
report("p(T)", pLP.subs(t, T))
# ex21 convention with the shift w = y - 1, p_ex21 = -p_LP, alpha_u = beta, eta = alpha_t = 1
w = yLP - 1
p21 = -pLP
u21 = -p21 / beta
what = yhat - 1
print("T1 in the ex21 convention (w = y - 1, p_ex21 = -p_LP, alpha_u = beta)")
report("state  w_t - Lap w - 0 - u", sp.diff(w, t) - lap(w) - u21)
report("adjoint -p_t - Lap p - (w - what)", -sp.diff(p21, t) - lap(p21) - (w - what))
report("gradient beta u + p", beta * u21 + p21)
report("u_ex21 - u_LP", u21 - uLP)
for side, sub in (("x=-1", {x: -1}), ("x=1", {x: 1}), ("z=-1", {z: -1}), ("z=1", {z: 1})):
    report(f"w on {side}", w.subs(sub))
    report(f"what on {side}", what.subs(sub))

print("ALL ZERO" if ok else "FAILURE")
raise SystemExit(0 if ok else 1)
