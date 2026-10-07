// Exact optimal-control solutions of ex21 (see ExactOcBenchmark.hpp).

#include "ExactOcBenchmark.hpp"

#include <mpi.h>
#include <petscksp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include "Discretization.hpp"
#include "PetscMatrix.hpp"
#include "Solver.hpp"
#include "SpaceTimeOperators.hpp"

using namespace femus;

namespace {

enum class Case { NONE, CONV_SIN3, CONV_SIN, LP_HEAT, LIFT_CONST, LIFT_EXP, LP52, LIFT_TRACK,
                  NEU_CONV, DIR_CONV, NAT_CONV };
Case g_case = Case::NONE;

/** The T2-type manufactured cases (exact y, p, u on (0,1)^2). */
bool Convective() {
  return g_case == Case::CONV_SIN3 || g_case == Case::CONV_SIN ||
         g_case == Case::LIFT_CONST || g_case == Case::LIFT_EXP ||
         g_case == Case::LIFT_TRACK || g_case == Case::NEU_CONV ||
         g_case == Case::DIR_CONV || g_case == Case::NAT_CONV;
}
/** T7D: true Dirichlet boundary control on Gamma_c = {z = 0}. */
bool IsDirichletCase() { return g_case == Case::DIR_CONV; }
/** T7N: Neumann boundary control on Gamma_c = {z = 0} (natural state BC). */
bool IsNeumannCase() { return g_case == Case::NEU_CONV; }
/** T7F: distributed control, natural state BC on all of dOmega. */
bool IsNaturalCase() { return g_case == Case::NAT_CONV; }
bool IsLiftedCase() {
  return g_case == Case::LIFT_CONST || g_case == Case::LIFT_EXP ||
         g_case == Case::LIFT_TRACK;
}

const char* CaseName() {
  switch (g_case) {
    case Case::CONV_SIN3: return "conv-sin3";
    case Case::CONV_SIN: return "conv-sin";
    case Case::LP_HEAT: return "lp-heat";
    case Case::LIFT_CONST: return "lift-const";
    case Case::LIFT_EXP: return "lift-exp";
    case Case::LP52: return "lp52";
    case Case::LIFT_TRACK: return "lift-track";
    case Case::NEU_CONV: return "neumann-conv";
    case Case::DIR_CONV: return "dirichlet-conv";
    case Case::NAT_CONV: return "natural-conv";
    default: return "none";
  }
}

// ---- T2: separable shapes s(kx x) s(kz z) on (0,1)^2 -----------------------
struct Shape {
  double s = 0.0, sx = 0.0, sz = 0.0, lap = 0.0;
};

void Factor(double a, double& s, double& ds, double& d2s, bool cubic) {
  const double sn = std::sin(a), cs = std::cos(a);
  if (!cubic) {
    s = sn; ds = cs; d2s = -sn;
    return;
  }
  s = sn * sn * sn;
  ds = 3.0 * sn * sn * cs;
  d2s = 6.0 * sn * cs * cs - 3.0 * sn * sn * sn;
}

Shape Separable(double kx, double kz, double x, double z, bool cubic) {
  double fx = 0, dfx = 0, d2fx = 0, fz = 0, dfz = 0, d2fz = 0;
  Factor(kx * x, fx, dfx, d2fx, cubic);
  Factor(kz * z, fz, dfz, d2fz, cubic);
  Shape r;
  r.s = fx * fz;
  r.sx = kx * dfx * fz;
  r.sz = kz * fx * dfz;
  r.lap = kx * kx * d2fx * fz + kz * kz * fx * d2fz;
  return r;
}
// S1 is sin^3 except for conv-sin.  S2 (the adjoint shape) is sin^3 except for
// conv-sin and lift-track: with s = sin, p = 0 on dOmega but its normal
// derivative is not, so y_d = y + (p_t + nu Lap p + w.grad p) differs from the
// state trace g on dOmega -- the T3b affine-tracking test.
Shape S1(double x, double z) {
  return Separable(kPi, kPi, x, z, g_case != Case::CONV_SIN);
}
Shape S2(double x, double z) {
  return Separable(2.0 * kPi, kPi, x, z,
                   g_case != Case::CONV_SIN && g_case != Case::LIFT_TRACK);
}

double A(double t) { return std::cos(2.0 * t); }
double dA(double t) { return -2.0 * std::sin(2.0 * t); }
double B(const ProblemConfig& c, double t) { return std::sin(c.final_time() - t); }
double dB(const ProblemConfig& c, double t) { return -std::cos(c.final_time() - t); }
/** w(x, z) . grad s (constant (vx, vy) unless a velocity field is set). */
double Advect(const ProblemConfig& c, double x, double z, const Shape& s) {
  double w[2];
  EvaluateVelocity(c, x, z, w);
  return w[0] * s.sx + w[1] * s.sz;
}

// ---- inhomogeneous Dirichlet lift g(x, z) of the lift-* cases --------------
//   lift-const: g = 1
//   lift-exp:   g = 1/2 + exp((vx x + vy z - vx - vy)/nu), a steady solution
//               of -nu Lap g + v . grad g = 0 for the CONSTANT (vx, vy)
//               (|grad| of the exponent = |v|/nu); max 3/2 at (1, 1).
struct Lifted {
  double g = 0.0, gx = 0.0, gz = 0.0, lap = 0.0;
};
Lifted Lift(const ProblemConfig& c, double x, double z) {
  Lifted r;
  if (g_case == Case::LIFT_CONST) {
    r.g = 1.0;
  } else if (g_case == Case::LIFT_EXP || g_case == Case::LIFT_TRACK) {
    const double cx = c.convection.vx / c.diffusivity, cz = c.convection.vy / c.diffusivity;
    const double e = std::exp(cx * x + cz * z - cx - cz);
    r.g = 0.5 + e;
    r.gx = cx * e;
    r.gz = cz * e;
    r.lap = (cx * cx + cz * cz) * e;
  }
  return r;
}
double DirichletData(const ProblemConfig& c, double x, double z) {
  return Lift(c, x, z).g;
}

// ---- Leveque-Pearson section 5.2 (LP52) --------------------------------------
//   Omega = (-1,1)^2, T = 2, f = 0, wind lp-recirculating, yhat = exp(-10(1-x1)),
//   y = 1 on {1} x [-1,1] (closed: the corners (1, +-1) take 1), 0 elsewhere on
//   the boundary; y(., 0) = 1 if x1 = 1, 0 otherwise.
bool OnRightEdge(double x) { return std::fabs(x - 1.0) < 1e-12; }
double Lp52Target(const ProblemConfig&, double x, double, double) {
  return std::exp(-10.0 * (1.0 - x));
}
double Lp52Boundary(const ProblemConfig&, double x, double) {
  return OnRightEdge(x) ? 1.0 : 0.0;
}
double Lp52Initial(const ProblemConfig&, double x, double) {
  return OnRightEdge(x) ? 1.0 : 0.0;
}
double Lp52Source(const ProblemConfig&, double, double, double) { return 0.0; }

// ---- T1: Leveque-Pearson section 5.1 (shifted state w = y - 1) -------------
double Lps(double x, double z) {
  return std::cos(kPi * x / 2.0) * std::cos(kPi * z / 2.0);
}

// ---- T7N neumann-conv (boundary_convection/scripts/t7n_symbolic.py) --------
//   Omega = (0,1)^2, Gamma_c = {z = 0}, constant wind w = (vx, vy), natural
//   (homogeneous Neumann) state BC elsewhere.  With
//     X(x) = 1 - cos(2 pi x)                      X = X' = 0 at x = 0, 1
//     Z(z) = exp(-b z) (2 + cos(pi z)), b = vy/nu  nu Z' + vy Z = 0 at z = 0, 1
//     H(z) = z - z^2/2                            H'(0) = 1, H'(1) = 0
//   the adjoint p = alpha_u B(t) X Z satisfies the DTO (exact-transpose)
//   boundary condition nu dn p + (w.n) p = 0 on all four faces and p(T) = 0;
//   u = -(eta/alpha_u) p on Gamma_c; the state
//     y = A(t) cos(pi x) cos(pi z) + kappa B(t) X H,  kappa = eta^2 Z(0)/nu,
//   has nu dn y = eta u on Gamma_c and nu dn y = 0 on the other faces.
struct Fn { double v = 0.0, d = 0.0, dd = 0.0; };
Fn NeuX(double x) {
  return {1.0 - std::cos(2.0 * kPi * x), 2.0 * kPi * std::sin(2.0 * kPi * x),
          4.0 * kPi * kPi * std::cos(2.0 * kPi * x)};
}
Fn NeuZ(const ProblemConfig& c, double z) {
  const double b = c.convection.vy / c.diffusivity, e = std::exp(-b * z);
  const double g = 2.0 + std::cos(kPi * z), dg = -kPi * std::sin(kPi * z),
               ddg = -kPi * kPi * std::cos(kPi * z);
  return {e * g, e * (dg - b * g), e * (ddg - 2.0 * b * dg + b * b * g)};
}
Fn NeuH(double z) { return {z - 0.5 * z * z, 1.0 - z, -1.0}; }
/** value, x/z derivatives and Laplacian of the separable parts */
struct Field { double v = 0.0, x = 0.0, z = 0.0, lap = 0.0; };
Field NeuYParts(const ProblemConfig& c, double x, double z, double a, double b) {
  // a * cos(pi x) cos(pi z) + b * kappa * X(x) H(z)
  const double kappa = c.eta * c.eta * NeuZ(c, 0.0).v / c.diffusivity;
  const Fn X = NeuX(x), Hz = NeuH(z);
  const double cx = std::cos(kPi * x), cz = std::cos(kPi * z);
  Field r;
  r.v = a * cx * cz + b * kappa * X.v * Hz.v;
  r.x = -a * kPi * std::sin(kPi * x) * cz + b * kappa * X.d * Hz.v;
  r.z = -a * kPi * cx * std::sin(kPi * z) + b * kappa * X.v * Hz.d;
  r.lap = -2.0 * kPi * kPi * a * cx * cz + b * kappa * (X.dd * Hz.v + X.v * Hz.dd);
  return r;
}
Field NeuPParts(const ProblemConfig& c, double x, double z, double b) {
  // alpha_u * b * X(x) Z(z)
  const Fn X = NeuX(x), Z = NeuZ(c, z);
  const double s = c.alpha_u * b;
  return {s * X.v * Z.v, s * X.d * Z.v, s * X.v * Z.d, s * (X.dd * Z.v + X.v * Z.dd)};
}

// ---- T7F natural-conv (t7f/scripts/t7f_symbolic.py) -----------------------
//   Omega = (0,1)^2, distributed control and tracking on the whole domain,
//   natural (homogeneous Neumann) state BC on ALL of dOmega, constant wind
//   w = (vx, vy) with both components free (w.n != 0 on every face).
//     R_k(s) = exp(-k s) (2 + cos(pi s)):  R_k' = -k R_k at s = 0 and s = 1,
//   so with kx = vx/nu, kz = vy/nu the adjoint
//     p = alpha_u B(t) R_kx(x) R_kz(z),   B = sin(T - t),
//   satisfies the DTO (exact-transpose) boundary condition
//     nu dn p + (w.n) p = 0   on all four faces, with p != 0 there,
//   and p(T) = 0;  u = -(eta/alpha_u) p;  the state
//     y = A(t) cos(pi x) cos(pi z)   has nu dn y = 0 on all four faces.
//   f = y_t - nu Lap y + w.grad y - eta u,  y_d = y + (p_t + nu Lap p + w.grad p)/alpha_t.
Fn NatR(double k, double s) {
  const double e = std::exp(-k * s);
  const double g = 2.0 + std::cos(kPi * s), dg = -kPi * std::sin(kPi * s),
               ddg = -kPi * kPi * std::cos(kPi * s);
  return {e * g, e * (dg - k * g), e * (ddg - 2.0 * k * dg + k * k * g)};
}
Field NatPParts(const ProblemConfig& c, double x, double z, double b) {
  // alpha_u * b * R_kx(x) R_kz(z)
  const Fn X = NatR(c.convection.vx / c.diffusivity, x),
           Z = NatR(c.convection.vy / c.diffusivity, z);
  const double s = c.alpha_u * b;
  return {s * X.v * Z.v, s * X.d * Z.v, s * X.v * Z.d, s * (X.dd * Z.v + X.v * Z.dd)};
}
Field NatYParts(double x, double z, double a) {
  // a * cos(pi x) cos(pi z)
  const double cx = std::cos(kPi * x), cz = std::cos(kPi * z);
  const double sx = std::sin(kPi * x), sz = std::sin(kPi * z);
  return {a * cx * cz, -a * kPi * sx * cz, -a * kPi * cx * sz, -2.0 * kPi * kPi * a * cx * cz};
}

// ---- T7D dirichlet-conv (boundary_convection/t7d/scripts/t7d_symbolic.py) --
//   Omega = (0,1)^2, Gamma_c = {z = 0} (Dirichlet control y = eta u), natural
//   state BC elsewhere, wind w = (0, vy) (vx must be 0: the x-faces carry the
//   tangential condition nu dp/dx = 0 satisfied by cos(pi x)), tracking box
//   z in (1/4, 1) (away from Gamma_c, as the Dirichlet KKT requires).
//   Adjoint p = S cos(pi x) Q(z,t),  Q = (1 - phi(z)) Qs + phi(z) B(t) Zr(z):
//     Qs = exp(-a z - c tau) [Phi(z - z1, tau) - Phi(z + z1, tau)], tau = T - t,
//          a = vy/(2 nu), c = vy^2/(4 nu) + pi^2 nu, Phi the 1D heat kernel
//          (diffusivity nu), z1 = 1.2 OUTSIDE the domain: Qs solves the
//          homogeneous backward adjoint -Q_t - nu Q_zz - vy Q_z + pi^2 nu Q = 0
//          exactly, Qs(0, t) = 0, Qs(., T) = 0, smooth on the closed domain;
//     phi a C-infinity step (0 for z <= 1/2, 1 for z >= 9/10);
//     Zr = exp(-(vy/nu) z)(2 + cos(pi z)): nu Zr' + vy Zr = 0 at z = 1;  B = sin(T-t).
//   => adjoint residual r = 0 for z < 1/2 (no tracking needed there),
//      p = 0 on Gamma_c, nu dn p + (w.n) p = 0 on the other faces, p(T) = 0.
//   Stationarity (L2 / H1-space, gamma_u = 0):
//     alpha_u (u - ell^2 u_ss) = eta nu dn p  =>  u = U(t) cos(pi x),
//     U = -eta nu S dQs/dz(0,t) / (alpha_u (1 + pi^2 ell^2)).
//   State y = A(t) cos(pi x) sin(pi z/2) + eta u  (y = eta u on Gamma_c,
//   nu dn y = 0 elsewhere).  f = y_t - nu Lap y + w.grad y,  y_d = y - r/alpha_t.
constexpr double kDirZ1 = 1.2, kDirS = 0.4, kDirZa = 0.5, kDirZb = 0.9;
struct Heat { double v = 0, z = 0, zz = 0, tau = 0, ztau = 0; };   // Qs and derivatives
Heat DirQs(const ProblemConfig& c, double z, double t) {
  Heat h;
  const double tau = c.final_time() - t;
  if (tau <= 1e-14) return h;
  const double nu = c.diffusivity, vy = c.convection.vy;
  const double a = vy / (2.0 * nu), cc = vy * vy / (4.0 * nu) + kPi * kPi * nu;
  auto phi = [&](double xi, double& p0, double& p1, double& p2, double& p3) {
    p0 = std::exp(-xi * xi / (4.0 * nu * tau)) / std::sqrt(4.0 * kPi * nu * tau);
    p1 = -xi / (2.0 * nu * tau) * p0;
    p2 = (xi * xi / (4.0 * nu * nu * tau * tau) - 1.0 / (2.0 * nu * tau)) * p0;
    p3 = (-xi * xi * xi / (8.0 * nu * nu * nu * tau * tau * tau) +
          3.0 * xi / (4.0 * nu * nu * tau * tau)) * p0;
  };
  double a0, a1, a2, a3, b0, b1, b2, b3;
  phi(z - kDirZ1, a0, a1, a2, a3);
  phi(z + kDirZ1, b0, b1, b2, b3);
  const double V = a0 - b0, Vz = a1 - b1, Vzz = a2 - b2, Vzzz = a3 - b3;
  const double E = std::exp(-a * z - cc * tau);
  h.v = E * V;
  h.z = E * (Vz - a * V);
  h.zz = E * (Vzz - 2.0 * a * Vz + a * a * V);
  h.tau = E * (nu * Vzz - cc * V);                                   // V_tau = nu V_zz
  h.ztau = E * (-cc * (Vz - a * V) + nu * Vzzz - a * nu * Vzz);
  return h;
}
/** C-infinity step 0 -> 1 on [kDirZa, kDirZb] and its first two z-derivatives. */
void DirStep(double z, double& f, double& f1, double& f2) {
  const double L = kDirZb - kDirZa, s = (z - kDirZa) / L;
  if (s <= 0.0) { f = f1 = f2 = 0.0; return; }
  if (s >= 1.0) { f = 1.0; f1 = f2 = 0.0; return; }
  auto psi = [](double q, double& g0, double& g1, double& g2) {
    g0 = std::exp(-1.0 / q); g1 = g0 / (q * q); g2 = g0 * (1.0 / (q * q * q * q) - 2.0 / (q * q * q));
  };
  double g0, g1, g2, h0, h1, h2;
  psi(s, g0, g1, g2);
  psi(1.0 - s, h0, h1, h2);
  h1 = -h1;   // d/ds psi(1 - s)
  const double D = g0 + h0, Dp = g1 + h1, N = g1 * h0 - g0 * h1, Np = g2 * h0 - g0 * h2;
  f = g0 / D;
  f1 = N / (D * D) / L;
  f2 = (Np * D - 2.0 * N * Dp) / (D * D * D) / (L * L);
}
struct ZField { double v = 0, z = 0, zz = 0, t = 0; };   // Q(z,t)
ZField DirQ(const ProblemConfig& c, double z, double t) {
  const Heat q = DirQs(c, z, t);
  const double b = c.convection.vy / c.diffusivity, e = std::exp(-b * z);
  const double g = 2.0 + std::cos(kPi * z), dg = -kPi * std::sin(kPi * z), ddg = -kPi * kPi * std::cos(kPi * z);
  const double Zr = e * g, Zr1 = e * (dg - b * g), Zr2 = e * (ddg - 2.0 * b * dg + b * b * g);
  const double Bt = std::sin(c.final_time() - t), dBt = -std::cos(c.final_time() - t);
  double f, f1, f2;
  DirStep(z, f, f1, f2);
  const double Qr = Bt * Zr, Qr1 = Bt * Zr1, Qr2 = Bt * Zr2;
  ZField r;
  r.v = (1.0 - f) * q.v + f * Qr;
  r.z = (1.0 - f) * q.z + f * Qr1 + f1 * (Qr - q.v);
  r.zz = (1.0 - f) * q.zz + f * Qr2 + 2.0 * f1 * (Qr1 - q.z) + f2 * (Qr - q.v);
  r.t = -(1.0 - f) * q.tau + f * dBt * Zr;
  return r;
}
double DirU(const ProblemConfig& c, double t) {   // U(t); u = U cos(pi x)
  return -c.eta * c.diffusivity * kDirS * DirQs(c, 0.0, t).z /
         (c.alpha_u * (1.0 + kPi * kPi * c.control_h1_length * c.control_h1_length));
}
double DirUt(const ProblemConfig& c, double t) {
  return c.eta * c.diffusivity * kDirS * DirQs(c, 0.0, t).ztau /
         (c.alpha_u * (1.0 + kPi * kPi * c.control_h1_length * c.control_h1_length));
}
double DirAdjointResidual(const ProblemConfig& c, double x, double z, double t) {
  // r = -p_t - nu Lap p - w.grad p,  p = S cos(pi x) Q
  const ZField q = DirQ(c, z, t);
  return kDirS * std::cos(kPi * x) *
         (-q.t + kPi * kPi * c.diffusivity * q.v - c.diffusivity * q.zz - c.convection.vy * q.z);
}

// ---- exact fields in the ex21 convention -----------------------------------
double ExactY(const ProblemConfig& c, double x, double z, double t) {
  if (IsDirichletCase())
    return A(t) * std::cos(kPi * x) * std::sin(0.5 * kPi * z) +
           c.eta * DirU(c, t) * std::cos(kPi * x);
  if (IsNeumannCase()) return NeuYParts(c, x, z, A(t), B(c, t)).v;
  if (IsNaturalCase()) return NatYParts(x, z, A(t)).v;
  if (IsLiftedCase()) return Lift(c, x, z).g + A(t) * S1(x, z).s;
  if (Convective()) return A(t) * S1(x, z).s;
  const double b = c.alpha_u, T = c.final_time(), p2 = kPi * kPi;
  return (2.0 / (p2 * b) * std::exp(T) - 2.0 / ((2.0 + p2) * b) * std::exp(t)) *
         Lps(x, z);   // w = y_LP - 1
}
double ExactP(const ProblemConfig& c, double x, double z, double t) {
  if (IsDirichletCase()) return kDirS * std::cos(kPi * x) * DirQ(c, z, t).v;
  if (IsNeumannCase()) return NeuPParts(c, x, z, B(c, t)).v;
  if (IsNaturalCase()) return NatPParts(c, x, z, B(c, t)).v;
  if (Convective()) return c.alpha_u * B(c, t) * S2(x, z).s;
  return -(std::exp(c.final_time()) - std::exp(t)) * Lps(x, z);   // -p_LP
}
double ExactU(const ProblemConfig& c, double x, double z, double t) {
  if (IsDirichletCase()) return DirU(c, t) * std::cos(kPi * x);
  return -c.eta * ExactP(c, x, z, t) / c.alpha_u;
}

// ---- data overrides --------------------------------------------------------
double Source(const ProblemConfig& c, double x, double z, double t) {
  if (!Convective()) return 0.0;
  if (IsDirichletCase()) {
    // f = y_t - nu Lap y + w.grad y, y = A cos(pi x) sin(pi z/2) + eta U cos(pi x)
    const double cx = std::cos(kPi * x), sz = std::sin(0.5 * kPi * z), cz = std::cos(0.5 * kPi * z);
    const double yt = dA(t) * cx * sz + c.eta * DirUt(c, t) * cx;
    const double lap = -1.25 * kPi * kPi * A(t) * cx * sz - kPi * kPi * c.eta * DirU(c, t) * cx;
    const double yz = 0.5 * kPi * A(t) * cx * cz;
    const double yx = -kPi * std::sin(kPi * x) * (A(t) * sz + c.eta * DirU(c, t));
    return yt - c.diffusivity * lap + c.convection.vx * yx + c.convection.vy * yz;
  }
  if (IsNaturalCase()) {
    // f = y_t - nu Lap y + w.grad y - eta u  (u = -(eta/alpha_u) p everywhere)
    const Field y = NatYParts(x, z, A(t)), yt = NatYParts(x, z, dA(t));
    return yt.v - c.diffusivity * y.lap + c.convection.vx * y.x + c.convection.vy * y.z -
           c.eta * ExactU(c, x, z, t);
  }
  if (IsNeumannCase()) {
    // f = y_t - nu Lap y + w.grad y  (the control acts on Gamma_c only)
    const Field y = NeuYParts(c, x, z, A(t), B(c, t));
    const Field yt = NeuYParts(c, x, z, dA(t), dB(c, t));
    return yt.v - c.diffusivity * y.lap + c.convection.vx * y.x + c.convection.vy * y.z;
  }
  const Shape s1 = S1(x, z);
  // f = y_t - nu Lap y + v.grad y - eta u
  const double f = dA(t) * s1.s - c.diffusivity * A(t) * s1.lap +
                   A(t) * Advect(c, x, z, s1) - c.eta * ExactU(c, x, z, t);
  if (!IsLiftedCase()) return f;
  // + (-nu Lap g + w . grad g): zero for lift-const, and for lift-exp with the
  // constant wind it was built for
  const Lifted g = Lift(c, x, z);
  double w[2];
  EvaluateVelocity(c, x, z, w);
  return f - c.diffusivity * g.lap + w[0] * g.gx + w[1] * g.gz;
}
double Target(const ProblemConfig& c, double x, double z, double t) {
  if (!Convective()) {
    const double b = c.alpha_u, T = c.final_time(), p2 = kPi * kPi;
    return ((2.0 / (p2 * b) + p2 / 2.0) * std::exp(T) +
            (1.0 - 2.0 / ((2.0 + p2) * b) - p2 / 2.0) * std::exp(t)) *
           Lps(x, z);   // yhat - 1
  }
  if (IsDirichletCase())   // y_d = y - r/alpha_t  (r = 0 for z < 1/2)
    return ExactY(c, x, z, t) - DirAdjointResidual(c, x, z, t) / c.alpha_track;
  if (IsNaturalCase()) {   // y_d = y + (p_t + nu Lap p + w.grad p) / alpha_track
    const Field p = NatPParts(c, x, z, B(c, t)), pt = NatPParts(c, x, z, dB(c, t));
    return ExactY(c, x, z, t) +
           (pt.v + c.diffusivity * p.lap + c.convection.vx * p.x + c.convection.vy * p.z) /
               c.alpha_track;
  }
  if (IsNeumannCase()) {
    const Field p = NeuPParts(c, x, z, B(c, t)), pt = NeuPParts(c, x, z, dB(c, t));
    return ExactY(c, x, z, t) +
           (pt.v + c.diffusivity * p.lap + c.convection.vx * p.x + c.convection.vy * p.z) /
               c.alpha_track;
  }
  const Shape s2 = S2(x, z);
  // y_d = y + (p_t + nu Lap p + v.grad p) / alpha_track
  const double q = c.alpha_u * (dB(c, t) * s2.s + c.diffusivity * B(c, t) * s2.lap +
                                B(c, t) * Advect(c, x, z, s2));
  return ExactY(c, x, z, t) + q / c.alpha_track;
}
double Initial(const ProblemConfig& c, double x, double z) {
  return ExactY(c, x, z, 0.0);
}

// ---- quadrature callbacks --------------------------------------------------
struct ExactContext {
  const ProblemConfig* config;
  double time;
  int field;        // 0 y, 1 p, 2 u, 3 y + 1 (lp-heat unshifted state)
  double sign;      // +1, or -1 for the wrong-sign diagnostic
};
double ExactAt(double x, double z, void* raw) {
  const ExactContext* e = static_cast<const ExactContext*>(raw);
  const ProblemConfig& c = *e->config;
  switch (e->field) {
    case 0: return e->sign * ExactY(c, x, z, e->time);
    case 1: return e->sign * ExactP(c, x, z, e->time);
    case 2: return e->sign * ExactU(c, x, z, e->time);
    case 4: return ExactY(c, x, z, e->time) - Target(c, x, z, e->time);
    default: return 1.0 + ExactY(c, x, z, e->time);
  }
}

/** ||x - b||_2 / ||b||_2. */
double RelDiff(Vec x, Vec b) {
  Vec d = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(x, &d));
  PetscCallAbort(PETSC_COMM_WORLD, VecWAXPY(d, -1.0, b, x));
  PetscReal nd = 0.0, nb = 0.0;
  PetscCallAbort(PETSC_COMM_WORLD, VecNorm(d, NORM_2, &nd));
  PetscCallAbort(PETSC_COMM_WORLD, VecNorm(b, NORM_2, &nb));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&d));
  return nb > 0.0 ? nd / nb : nd;
}

/** Space-time accumulator of one field: dt-weighted squared L2 errors and
 *  exact norms over all slots, over all but the last slot, and over the
 *  physical window t <= T/2; the last slot alone; the max over slots. */
struct Accumulator {
  double all_e = 0, all_x = 0, int_e = 0, int_x = 0, half_e = 0, half_x = 0;
  double last_e = 0, last_x = 0, max_e = 0;
  void Add(const SpatialFieldIntegrals& I, double dt, double t, double T,
           bool last) {
    all_e += dt * I.error_squared;
    all_x += dt * I.exact_squared;
    if (!last) { int_e += dt * I.error_squared; int_x += dt * I.exact_squared; }
    if (t <= 0.5 * T + 1e-12) { half_e += dt * I.error_squared; half_x += dt * I.exact_squared; }
    if (last) { last_e = I.error_squared; last_x = I.exact_squared; }
    max_e = std::max(max_e, std::sqrt(I.error_squared));
  }
  void Print(const char* name) const {
    auto rel = [](double e, double x) { return x > 0 ? std::sqrt(e / x) : std::sqrt(e); };
    std::printf(" %s_L2L2=%.10e %s_L2L2_rel=%.10e %s_int=%.10e %s_int_rel=%.10e"
                " %s_half=%.10e %s_half_rel=%.10e %s_last=%.10e %s_last_rel=%.10e"
                " %s_Linf_L2=%.10e",
                name, std::sqrt(all_e), name, rel(all_e, all_x), name,
                std::sqrt(int_e), name, rel(int_e, int_x), name, std::sqrt(half_e),
                name, rel(half_e, half_x), name, std::sqrt(last_e), name,
                rel(last_e, last_x), name, max_e);
  }
};

/** Writes the nonzeros of `a` as "i j value" lines (serial, tiny meshes). */
void DumpMatrix(Mat a, const std::string& path) {
  FILE* out = std::fopen(path.c_str(), "w");
  if (!out) throw std::runtime_error("cannot write " + path);
  PetscInt rows = 0, cols = 0;
  PetscCallAbort(PETSC_COMM_WORLD, MatGetSize(a, &rows, &cols));
  std::fprintf(out, "%d %d\n", (int)rows, (int)cols);
  for (PetscInt i = 0; i < rows; ++i) {
    PetscInt n = 0;
    const PetscInt* j = nullptr;
    const PetscScalar* v = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD, MatGetRow(a, i, &n, &j, &v));
    for (PetscInt k = 0; k < n; ++k)
      if (PetscRealPart(v[k]) != 0.0)
        std::fprintf(out, "%d %d %.17e\n", (int)i, (int)j[k], PetscRealPart(v[k]));
    PetscCallAbort(PETSC_COMM_WORLD, MatRestoreRow(a, i, &n, &j, &v));
  }
  std::fclose(out);
}

}  // namespace

void ApplyExactOptimalControl(ProblemConfig& config, const std::string& name,
                              double alpha_u, double final_time, double nu,
                              const double* velocity) {
  if (name == "conv-sin3") g_case = Case::CONV_SIN3;
  else if (name == "conv-sin") g_case = Case::CONV_SIN;
  else if (name == "lp-heat") g_case = Case::LP_HEAT;
  else if (name == "lift-const") g_case = Case::LIFT_CONST;
  else if (name == "lift-exp") g_case = Case::LIFT_EXP;
  else if (name == "lp52") g_case = Case::LP52;
  else if (name == "lift-track") g_case = Case::LIFT_TRACK;
  else if (name == "neumann-conv") g_case = Case::NEU_CONV;
  else if (name == "dirichlet-conv") g_case = Case::DIR_CONV;
  else if (name == "natural-conv") g_case = Case::NAT_CONV;
  else throw std::invalid_argument(
      "--exact-oc takes conv-sin3, conv-sin, lp-heat, lift-const, lift-exp,"
      " lift-track, lp52, neumann-conv, dirichlet-conv or natural-conv");
  config.dimension = 2u;
  config.state_bc = ProblemConfig::StateBoundaryCondition::HOMOGENEOUS_DIRICHLET;
  config.control.type = ProblemConfig::ControlType::DISTRIBUTED;
  config.additional_control_regions.clear();
  config.additional_tracking_regions.clear();
  config.source_regions.clear();
  config.alpha_track = 1.0;
  config.eta = 1.0;
  if (Convective()) {
    config.domain = {0.0, 1.0, 0.0, 1.0};
    // T6: --exact-oc-nu also sets nu here (the manufactured data use
    // c.diffusivity throughout); absent, the original 0.25.
    config.diffusivity = nu > 0.0 ? nu : 0.25;
    config.convection.enabled = true;
    config.convection.vx = 1.0;
    config.convection.vy = 0.5;
    config.alpha_u = alpha_u > 0.0 ? alpha_u : 1.0e-2;
    // a --wind field given on the command line is kept (variable-wind tests)
  } else if (g_case == Case::LP52) {
    config.domain = {-1.0, 1.0, -1.0, 1.0};
    config.diffusivity = nu > 0.0 ? nu : 1.0 / 20.0; // epsilon (LP: 1/20, 1/100, 1/500)
    config.convection.enabled = true;
    config.convection.vx = 0.0;
    config.convection.vy = 0.0;
    config.convection.field = WindLpRecirculating;   // LP's w, x2 = z
    config.convection.field_name = "lp-recirculating";
    config.alpha_u = alpha_u > 0.0 ? alpha_u : 1.0e-2;   // = beta exactly
  } else {
    config.domain = {-1.0, 1.0, -1.0, 1.0};
    config.diffusivity = 1.0;
    // v = 0; --convection on keeps the transposed (ex21) adjoint path active
    config.convection.vx = 0.0;
    config.convection.vy = 0.0;
    config.alpha_u = alpha_u > 0.0 ? alpha_u : 1.0e-3;   // = beta exactly
  }
  const double T = final_time > 0.0 ? final_time : (Convective() ? 1.0 : 2.0);
  config.time.dt = T / config.time.steps;
  config.control.region = {config.domain.xmin, config.domain.xmax,
                           config.domain.zmin, config.domain.zmax};
  config.tracking_box = config.control.region;
  config.data_overrides.source = Source;
  config.data_overrides.tracking_target = Target;
  config.data_overrides.initial_state = Initial;
  if (IsLiftedCase()) config.data_overrides.dirichlet_state = DirichletData;
  if (velocity) {   // T7N --exact-oc-velocity (constant wind; convective cases)
    if (!Convective() || config.convection.field)
      throw std::invalid_argument(
          "--exact-oc-velocity needs a constant-wind convective --exact-oc case");
    config.convection.vx = velocity[0];
    config.convection.vy = velocity[1];
  }
  if (IsDirichletCase()) {
    // Dirichlet boundary control on z = 0 (face 1), natural state BC elsewhere,
    // wind (0, vy) (default vy = 1/2), tracking box z in (1/4, 1) away from
    // Gamma_c.  ell (--control-h1-length) is kept; gamma_u (H1-time) is not
    // supported by this closed form (its stationarity is an ODE in time).
    if (config.convection.field)
      throw std::invalid_argument("dirichlet-conv uses a constant wind");
    if (!velocity) { config.convection.vx = 0.0; config.convection.vy = 0.5; }
    if (config.convection.vx != 0.0)
      throw std::invalid_argument("dirichlet-conv needs vx = 0 (cos(pi x) x-profile)");
    if (config.control_h1_time_weight != 0.0)
      throw std::invalid_argument("dirichlet-conv: the closed form has gamma_u = 0");
    config.state_bc = ProblemConfig::StateBoundaryCondition::NATURAL;
    config.control.type = ProblemConfig::ControlType::DIRICHLET;
    config.neumann_control_face = 1;
    config.neumann_control_boundary = "y0";
    config.tracking_box = {0.0, 1.0, 0.25, 1.0};
  }
  if (IsNaturalCase()) {
    // T7F: distributed control and tracking on the whole domain (set above),
    // natural state BC on every face; constant wind only (the R_k profiles).
    if (config.convection.field)
      throw std::invalid_argument("natural-conv uses a constant wind (--exact-oc-velocity)");
    config.state_bc = ProblemConfig::StateBoundaryCondition::NATURAL;
  }
  if (IsNeumannCase()) {
    // Neumann boundary control on the bottom face z = 0 (FEMuS box face 1),
    // natural state BC elsewhere; tracking over the whole domain.
    if (config.convection.field)
      throw std::invalid_argument("neumann-conv uses a constant wind (--exact-oc-velocity)");
    config.state_bc = ProblemConfig::StateBoundaryCondition::NATURAL;
    config.control.type = ProblemConfig::ControlType::NEUMANN;
    config.neumann_control_face = 1;
    config.neumann_control_boundary = "y0";
  }
  if (g_case == Case::LP52) {
    config.data_overrides.source = Lp52Source;
    config.data_overrides.tracking_target = Lp52Target;
    config.data_overrides.initial_state = Lp52Initial;
    config.data_overrides.dirichlet_state = Lp52Boundary;
  }
}

namespace {

/** LP section 5.2 has no exact solution: report the fields' ranges over the
 *  whole run, the t = 1 snapshot (y at the node t = 1; p and u at t = 1 as the
 *  mean of the two interval values around it under CN, the node value under
 *  BE), and write the snapshot as CSV (x, z, y, p_LP = -p_ex21, u). */
int ReportLp52(const ProblemConfig& config, const ApplicationOptions& options,
               const SpatialOperators& spatial,
               const DistributedOptimalControlSolve& solve, double march_rel,
               double stat_rel, double solve_wall, double total_wall) {
  const DistributedDataSideResult& f = solve.fields;
  const PetscInt local = spatial.layout.local_size;
  const unsigned steps = config.time.steps;
  const double dt = config.time.dt;
  const bool cn = IsCrankNicolson(config);
  const unsigned n1 = static_cast<unsigned>(std::llround(1.0 / dt));   // t_n1 = 1
  if (n1 < 1u || n1 >= steps || std::fabs(n1 * dt - 1.0) > 1e-12)
    throw std::runtime_error("lp52: t = 1 must be a time node");
  const PetscScalar *ys = nullptr, *ps = nullptr, *us = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(f.state, &ys));
  PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(f.adjoint, &ps));
  PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(f.control, &us));
  double ymin = 1e300, ymax = -1e300, yimin = 1e300, yimax = -1e300;
  bool finite = true;
  for (unsigned slot = 0; slot < steps; ++slot)
    for (PetscInt i = 0; i < local; ++i) {
      const double v = PetscRealPart(ys[static_cast<PetscInt>(slot) * local + i]);
      finite = finite && std::isfinite(v);
      ymin = std::min(ymin, v);
      ymax = std::max(ymax, v);
      const double x = spatial.layout.x[i], z = spatial.layout.z[i];
      if (std::fabs(std::fabs(x) - 1.0) > 1e-12 && std::fabs(std::fabs(z) - 1.0) > 1e-12) {
        yimin = std::min(yimin, v);
        yimax = std::max(yimax, v);
      }
    }
  if (!options.exact_oc_series.empty()) {
    // T6: where and when the extrema occur (slot s holds t = (s+1) dt).
    FILE* series = std::fopen(options.exact_oc_series.c_str(), "w");
    if (!series) throw std::runtime_error("cannot write " + options.exact_oc_series);
    std::fprintf(series, "slot,t,y_min,x_min,z_min,y_max,x_max,z_max\n");
    unsigned arg_slot = 0;
    double arg_value = 1e300;
    for (unsigned slot = 0; slot < steps; ++slot) {
      PetscInt imin = 0, imax = 0;
      for (PetscInt i = 0; i < local; ++i) {
        const PetscInt k = static_cast<PetscInt>(slot) * local;
        if (PetscRealPart(ys[k + i]) < PetscRealPart(ys[k + imin])) imin = i;
        if (PetscRealPart(ys[k + i]) > PetscRealPart(ys[k + imax])) imax = i;
      }
      const double vmin = PetscRealPart(ys[static_cast<PetscInt>(slot) * local + imin]);
      const double vmax = PetscRealPart(ys[static_cast<PetscInt>(slot) * local + imax]);
      if (vmin < arg_value) { arg_value = vmin; arg_slot = slot; }
      std::fprintf(series, "%u,%.10g,%.12e,%.10g,%.10g,%.12e,%.10g,%.10g\n", slot,
                   (slot + 1u) * dt, vmin, spatial.layout.x[imin], spatial.layout.z[imin],
                   vmax, spatial.layout.x[imax], spatial.layout.z[imax]);
    }
    std::fclose(series);
    FILE* field = std::fopen((options.exact_oc_series + ".argmin.csv").c_str(), "w");
    if (!field) throw std::runtime_error("cannot write the argmin field");
    std::fprintf(field, "# slot=%u t=%.10g\nx,z,y\n", arg_slot, (arg_slot + 1u) * dt);
    for (PetscInt i = 0; i < local; ++i)
      std::fprintf(field, "%.10g,%.10g,%.12e\n", spatial.layout.x[i], spatial.layout.z[i],
                   PetscRealPart(ys[static_cast<PetscInt>(arg_slot) * local + i]));
    std::fclose(field);
  }
  auto at1 = [&](const PetscScalar* a, PetscInt i, bool node) {
    const PetscInt k = static_cast<PetscInt>(n1 - 1u) * local + i;   // slot of t_n1
    if (node || !cn) return PetscRealPart(a[k]);
    return 0.5 * (PetscRealPart(a[k]) + PetscRealPart(a[k + local]));
  };
  double y1min = 1e300, y1max = -1e300, p1min = 1e300, p1max = -1e300,
         u1min = 1e300, u1max = -1e300;
  FILE* out = nullptr;
  if (!options.exact_oc_snapshot.empty()) {
    out = std::fopen(options.exact_oc_snapshot.c_str(), "w");
    if (!out) throw std::runtime_error("cannot write " + options.exact_oc_snapshot);
    std::fprintf(out, "x,z,y,p_LP,u\n");
  }
  for (PetscInt i = 0; i < local; ++i) {
    const double yv = at1(ys, i, true), pv = -at1(ps, i, false), uv = at1(us, i, false);
    y1min = std::min(y1min, yv); y1max = std::max(y1max, yv);
    p1min = std::min(p1min, pv); p1max = std::max(p1max, pv);
    u1min = std::min(u1min, uv); u1max = std::max(u1max, uv);
    if (out)
      std::fprintf(out, "%.10g,%.10g,%.12e,%.12e,%.12e\n", spatial.layout.x[i],
                   spatial.layout.z[i], yv, pv, uv);
  }
  if (out) std::fclose(out);
  PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(f.control, &us));
  PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(f.adjoint, &ps));
  PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(f.state, &ys));
  std::printf("EX21_LP52_RESULT scheme=%s nx=%u steps=%u dt=%.10e beta=%.6e eps=%.6g"
              " reason=%d outer_its=%d true_residual=%.3e state_rel=%.3e"
              " adjoint_rel=%.3e march_rel=%.3e alpha_u_u_plus_eta_p_rel=%.3e"
              " finite=%d y_min=%.6e y_max=%.6e y_int_min=%.6e y_int_max=%.6e"
              " y_t1_min=%.6e y_t1_max=%.6e pLP_t1_min=%.6e pLP_t1_max=%.6e"
              " u_t1_min=%.6e u_t1_max=%.6e J_track=%.12e J_reg=%.12e"
              " solve_wall=%.3f total_wall=%.3f\n",
              TimeSchemeName(config), config.mesh.nx, steps, dt, config.alpha_u,
              config.diffusivity, solve.reason, solve.outer_iterations,
              solve.true_residual, solve.kkt.state_rel, solve.kkt.adjoint_rel,
              march_rel, stat_rel, finite ? 1 : 0, ymin, ymax, yimin, yimax,
              y1min, y1max, p1min, p1max, u1min, u1max, f.J_track, f.J_reg,
              solve_wall, total_wall);
  return solve.accepted && finite ? 0 : 8;
}

}  // namespace

namespace {

/** T4: the isolated spatial-block experiment.  The three spatial blocks the
 *  SCHUR_SWEEP time sweeps solve, for theta = 1 (backward Euler) and
 *  theta = 1/2 (Crank-Nicolson diagonal):
 *      E-block  A(theta)            = M/dt + theta L          (y-split sweep)
 *      right    A(theta) + sqrt(b) N                          (matching factor)
 *      left     (A(theta) + sqrt(b) M_omega)^T                (matching factor)
 *  b = alpha_t eta^2 / alpha_u.  Each is solved with a KSP configured EXACTLY
 *  as Preconditioners.cpp CreateSweepSpatialSolver configures the nonsymmetric
 *  sweep solver (GMRES(50), right preconditioning, unpreconditioned norm,
 *  GAMG, rtol = --spatial-tol, 500 iterations, zero initial guess), for three
 *  seeded random right-hand sides and one smooth one (A applied to the nodal
 *  field sin(pi x/2) sin(pi z/2) + x, restricted to the free DOFs).  The KKT
 *  solver is not involved. */
int RunBlockExperiment(const ProblemConfig& config,
                       const ApplicationOptions& options,
                       const SpatialOperators& spatial) {
  const double dt = config.time.dt;
  const double b = config.alpha_track * config.eta * config.eta / config.alpha_u;
  const ProblemConfig::Domain& D = config.domain;
  const double h = (D.xmax - D.xmin) / config.mesh.nx;
  double wmax = std::hypot(config.convection.vx, config.convection.vy);
  if (config.convection.field) {
    wmax = 0.0;
    for (PetscInt i = 0; i < spatial.layout.local_size; ++i) {
      double w[2];
      EvaluateVelocity(config, spatial.layout.x[i], spatial.layout.z[i], w);
      wmax = std::max(wmax, std::hypot(w[0], w[1]));
    }
  }
  Vec smooth = nullptr, diag = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, MatCreateVecs(spatial.mass, &smooth, &diag));
  PetscCallAbort(PETSC_COMM_WORLD, MatGetDiagonal(spatial.mass, diag));
  {
    PetscScalar* a = nullptr;
    const PetscScalar* d = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD, VecGetArray(smooth, &a));
    PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(diag, &d));
    for (PetscInt i = 0; i < spatial.layout.local_size; ++i) {
      const double x = spatial.layout.x[i], z = spatial.layout.z[i];
      a[i] = PetscRealPart(d[i]) != 0.0
                 ? std::sin(kPi * x / 2.0) * std::sin(kPi * z / 2.0) + x : 0.0;
    }
    PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(diag, &d));
    PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArray(smooth, &a));
  }
  int failures = 0;
  for (double theta : {1.0, 0.5}) {
    Mat A = nullptr, R = nullptr, Lm = nullptr, Lt = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD, MatDuplicate(spatial.mass, MAT_COPY_VALUES, &A));
    PetscCallAbort(PETSC_COMM_WORLD, MatScale(A, 1.0 / dt));
    PetscCallAbort(PETSC_COMM_WORLD, MatAXPY(A, theta, spatial.stiffness, DIFFERENT_NONZERO_PATTERN));
    PetscCallAbort(PETSC_COMM_WORLD, MatDuplicate(A, MAT_COPY_VALUES, &R));
    PetscCallAbort(PETSC_COMM_WORLD, MatAXPY(R, std::sqrt(b), spatial.control, SUBSET_NONZERO_PATTERN));
    PetscCallAbort(PETSC_COMM_WORLD, MatDuplicate(A, MAT_COPY_VALUES, &Lm));
    PetscCallAbort(PETSC_COMM_WORLD, MatAXPY(Lm, std::sqrt(b), spatial.tracking, SUBSET_NONZERO_PATTERN));
    PetscCallAbort(PETSC_COMM_WORLD, MatTranspose(Lm, MAT_INITIAL_MATRIX, &Lt));
    struct Block { const char* name; Mat m; };
    for (const Block& blk : {Block{"E", A}, Block{"right", R}, Block{"left", Lt}}) {
      KSP ksp = nullptr;
      PetscCallAbort(PETSC_COMM_WORLD, KSPCreate(PETSC_COMM_WORLD, &ksp));
      PetscCallAbort(PETSC_COMM_WORLD, KSPSetOptionsPrefix(ksp, "block_test_"));
      PetscCallAbort(PETSC_COMM_WORLD, KSPSetOperators(ksp, blk.m, blk.m));
      PetscCallAbort(PETSC_COMM_WORLD, KSPSetType(ksp, KSPGMRES));
      PetscCallAbort(PETSC_COMM_WORLD, KSPGMRESSetRestart(ksp, 50));
      PetscCallAbort(PETSC_COMM_WORLD, KSPSetPCSide(ksp, PC_RIGHT));
      PetscCallAbort(PETSC_COMM_WORLD, KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED));
      PC pc = nullptr;
      PetscCallAbort(PETSC_COMM_WORLD, KSPGetPC(ksp, &pc));
      PetscCallAbort(PETSC_COMM_WORLD, PCSetType(pc, PCGAMG));
      PetscCallAbort(PETSC_COMM_WORLD, KSPSetTolerances(ksp, options.spatial_tolerance,
                                                        PETSC_DEFAULT, PETSC_DEFAULT, 500));
      PetscCallAbort(PETSC_COMM_WORLD, KSPSetInitialGuessNonzero(ksp, PETSC_FALSE));
      PetscCallAbort(PETSC_COMM_WORLD, KSPSetFromOptions(ksp));
      const double t_setup = MPI_Wtime();
      PetscCallAbort(PETSC_COMM_WORLD, KSPSetUp(ksp));
      const double setup = MPI_Wtime() - t_setup;
      Vec rhs = nullptr, x = nullptr;
      PetscCallAbort(PETSC_COMM_WORLD, MatCreateVecs(blk.m, &x, &rhs));
      std::string its, reasons;
      int max_its = 0, hits = 0;
      const double t_solve = MPI_Wtime();
      for (int k = 0; k < 4; ++k) {
        if (k < 3) {
          PetscRandom rnd = nullptr;
          PetscCallAbort(PETSC_COMM_WORLD, PetscRandomCreate(PETSC_COMM_WORLD, &rnd));
          PetscCallAbort(PETSC_COMM_WORLD, PetscRandomSetInterval(rnd, -1.0, 1.0));
          PetscCallAbort(PETSC_COMM_WORLD, PetscRandomSetSeed(rnd, 101u + 7u * k));
          PetscCallAbort(PETSC_COMM_WORLD, PetscRandomSeed(rnd));
          PetscCallAbort(PETSC_COMM_WORLD, VecSetRandom(rhs, rnd));
          PetscCallAbort(PETSC_COMM_WORLD, PetscRandomDestroy(&rnd));
          // the constrained rows are identity rows of the blocks: zero there
          PetscCallAbort(PETSC_COMM_WORLD, VecPointwiseMult(rhs, rhs, diag));
          PetscCallAbort(PETSC_COMM_WORLD, VecScale(rhs, 1.0 / dt));
        } else {
          PetscCallAbort(PETSC_COMM_WORLD, MatMult(blk.m, smooth, rhs));
        }
        PetscCallAbort(PETSC_COMM_WORLD, VecSet(x, 0.0));
        PetscCallAbort(PETSC_COMM_WORLD, KSPSolve(ksp, rhs, x));
        PetscInt n = 0;
        KSPConvergedReason reason;
        PetscCallAbort(PETSC_COMM_WORLD, KSPGetIterationNumber(ksp, &n));
        PetscCallAbort(PETSC_COMM_WORLD, KSPGetConvergedReason(ksp, &reason));
        its += (k ? "/" : "") + std::to_string(static_cast<int>(n));
        reasons += (k ? "/" : "") + std::to_string(static_cast<int>(reason));
        max_its = std::max(max_its, static_cast<int>(n));
        hits += n >= 500;
        failures += reason < 0;
      }
      const double solve = MPI_Wtime() - t_solve;
      std::printf("EX21_BLOCK theta=%.3g block=%s nx=%u dt=%.6e nu=%.6g h=%.6e max_w=%.6e"
                  " Pe_cell=%.6e adv_ratio=%.6e mass_over_diff=%.6e its=%s max_its=%d"
                  " cap_hits=%d reasons=%s setup_s=%.3f solve_s=%.3f\n",
                  theta, blk.name, config.mesh.nx, dt, config.diffusivity, h, wmax,
                  wmax * h / (2.0 * config.diffusivity), theta * wmax * dt / h,
                  h * h / (theta * dt * config.diffusivity), its.c_str(), max_its,
                  hits, reasons.c_str(), setup, solve);
      PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&x));
      PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&rhs));
      PetscCallAbort(PETSC_COMM_WORLD, KSPDestroy(&ksp));
    }
    for (Mat* m : {&A, &R, &Lm, &Lt}) PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(m));
  }
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&smooth));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&diag));
  return failures ? 8 : 0;
}

}  // namespace

int RunExactOptimalControl(const ProblemConfig& config,
                           const ApplicationOptions& options,
                           MultiLevelSolution& solution, Mesh* mesh,
                           unsigned solution_type) {
  int ranks = 0;
  MPI_Comm_size(PETSC_COMM_WORLD, &ranks);
  if (ranks != 1) throw std::runtime_error("--exact-oc runs serially");
  const char* name = CaseName();
  const double t0 = MPI_Wtime();
  SpatialOperators spatial(config, mesh, solution_type, solution);
  SpaceTimeOperators spacetime(config, spatial);
  const PetscInt local = spatial.layout.local_size;
  const unsigned steps = config.time.steps;
  const double dt = config.time.dt, T = config.final_time();
  const bool cn = IsCrankNicolson(config);
  const ProblemConfig::Domain& D = config.domain;
  const double h = (D.xmax - D.xmin) / config.mesh.nx;
  // max |w| over the nodes (the constant |v| without a field); Pe_cell is
  // based on the element size h, Pe_node on the Q2 node spacing h/2
  double speed = std::hypot(config.convection.vx, config.convection.vy);
  if (config.convection.field) {
    speed = 0.0;
    for (PetscInt i = 0; i < local; ++i) {
      double w[2];
      EvaluateVelocity(config, spatial.layout.x[i], spatial.layout.z[i], w);
      speed = std::max(speed, std::hypot(w[0], w[1]));
    }
  }
  std::printf("EX21_EXACT_CONFIG case=%s nx=%u nz=%u steps=%u T=%.6g dt=%.10e"
              " scheme=%s nu=%.6g convection=%s vx=%.6g vy=%.6g alpha_u=%.6e"
              " h=%.6e Pe_cell=%.6e dofs=%d pc=%d wind=%s max_w=%.6e"
              " Pe_node=%.6e dirichlet_data=%s tol=%s\n",
              name, config.mesh.nx, config.mesh.nz, steps, T, dt,
              TimeSchemeName(config), config.diffusivity,
              HasConvection(config) ? "on" : "off", config.convection.vx,
              config.convection.vy, config.alpha_u, h,
              speed * h / (2.0 * config.diffusivity), (int)local,
              static_cast<int>(options.pc_mode),
              config.convection.field ? config.convection.field_name : "constant",
              speed, speed * h / (4.0 * config.diffusivity),
              config.data_overrides.dirichlet_state ? "lifted" : "homogeneous",
              options.exact_oc_production_tol ? "production" : "tight");

  if (options.exact_oc_block_test)
    return RunBlockExperiment(config, options, spatial);

  if (!options.exact_oc_dump.empty()) {
    // tiny meshes only: the spatial M, L = K + C and the assembled space-time
    // E, E^T exactly as the solver uses them (input of t0_cn_matrix_check.py)
    DumpMatrix(spatial.mass, options.exact_oc_dump + "_M.txt");
    DumpMatrix(spatial.stiffness, options.exact_oc_dump + "_L.txt");
    DumpMatrix(spatial.tracking, options.exact_oc_dump + "_Mtrack.txt");
    DumpMatrix(spatial.control, options.exact_oc_dump + "_N.txt");
    DumpMatrix(spacetime.assembled.ay, options.exact_oc_dump + "_E.txt");
    DumpMatrix(spacetime.assembled.ap, options.exact_oc_dump + "_ET.txt");
    if (spatial.assembled.W_lps)   // T6: the LPS part of L (already in _L)
      DumpMatrix(static_cast<femus::PetscMatrix*>(spatial.assembled.W_lps.get())->mat(),
                 options.exact_oc_dump + "_W.txt");
    std::printf("EX21_EXACT_DUMP prefix=%s\n", options.exact_oc_dump.c_str());
  }

  // ---- tightened production solve: the solver error must sit well below the
  // discretisation error (policy local to this driver; defaults unchanged)
  KktSolverPolicy policy = MakeKktSolverPolicy(options.pc_mode, 1.0e-12);
  policy.outer_rtol = 1.0e-12;
  policy.inner_rtol = 1.0e-10;
  policy.outer_maxits = 400;
  // --exact-oc-production-tol: the unmodified production policy instead
  if (options.exact_oc_production_tol)
    policy = MakeKktSolverPolicy(options.pc_mode, options.spatial_tolerance);
  const double t_solve = MPI_Wtime();
  DistributedOptimalControlSolve solve =
      SolveOptimalControl(config, spatial, spacetime, policy);
  const double solve_wall = MPI_Wtime() - t_solve;
  const DistributedDataSideResult& f = solve.fields;

  // ---- independent forward march with the computed control (spatial LU)
  DistributedDataSideResult data = BuildDistributedRightHandSide(
      config, spatial.layout, spatial.mass, spatial.tracking,
      spacetime.distributed.ay, spatial.stiffness, spatial.source,
      &spatial.lift);
  Mat diagonal = nullptr, coupling = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatDuplicate(spatial.stiffness, MAT_COPY_VALUES, &diagonal));
  PetscCallAbort(PETSC_COMM_WORLD, MatScale(diagonal, cn ? 0.5 : 1.0));
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatAXPY(diagonal, 1.0 / dt, spatial.mass, SUBSET_NONZERO_PATTERN));
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatDuplicate(spatial.mass, MAT_COPY_VALUES, &coupling));
  PetscCallAbort(PETSC_COMM_WORLD, MatScale(coupling, 1.0 / dt));
  if (cn)
    PetscCallAbort(PETSC_COMM_WORLD, MatAXPY(coupling, -0.5, spatial.stiffness,
                                             DIFFERENT_NONZERO_PATTERN));
  KSP lu = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, KSPCreate(PETSC_COMM_WORLD, &lu));
  PetscCallAbort(PETSC_COMM_WORLD, KSPSetOptionsPrefix(lu, "exact_march_"));
  PetscCallAbort(PETSC_COMM_WORLD, KSPSetOperators(lu, diagonal, diagonal));
  PetscCallAbort(PETSC_COMM_WORLD, KSPSetType(lu, KSPPREONLY));
  PC pc = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, KSPGetPC(lu, &pc));
  PetscCallAbort(PETSC_COMM_WORLD, PCSetType(pc, PCLU));
  PetscCallAbort(PETSC_COMM_WORLD, KSPSetUp(lu));
  Vec r = nullptr, y = nullptr, prev = nullptr, tmp = nullptr, u = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, MatCreateVecs(spatial.mass, &r, &y));
  for (Vec* v : {&prev, &tmp, &u}) PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(r, v));
  Vec marched = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(f.state, &marched));
  if (IsDirichletCase()) {
    // T7D: march E_I Y = rhs - eta B_D U slot by slot with the SAME blocks the
    // AY shell applies (CN: the explicitly embedded A_plus_I / A_minus_I; BE:
    // M_I/dt + K_I and -M_I/dt on the hatted matrices), LU per slot, then the
    // controlled rows carry y_C = eta u.
    DirichletControlOperators& dc = *spatial.dirichlet;
    DistributedDataSideResult dd = BuildDistributedRightHandSide(
        config, spatial.layout, spatial.mass_full, spatial.tracking, spacetime.distributed.ay,
        dc.unhatted_stiffness(), spatial.source, &spatial.lift);
    PetscCallAbort(PETSC_COMM_WORLD, dc.ZeroControlledRows(dd.rhs_y));
    Vec bd = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(dd.rhs_y, &bd));
    PetscCallAbort(PETSC_COMM_WORLD, dc.ApplyBD(f.control, bd));
    PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(dd.rhs_y, -config.eta, bd));
    const DistributedSpaceTimeContext& ctx = spacetime.distributed.ay_context;
    Mat dblk = ctx.diagonal_block, nblk = ctx.neighbour_block, own_d = nullptr, own_n = nullptr;
    if (!dblk) {   // backward Euler: M_I/dt + K_I and -M_I/dt
      PetscCallAbort(PETSC_COMM_WORLD, MatDuplicate(spatial.stiffness, MAT_COPY_VALUES, &own_d));
      PetscCallAbort(PETSC_COMM_WORLD, MatAXPY(own_d, 1.0 / dt, spatial.mass, DIFFERENT_NONZERO_PATTERN));
      PetscCallAbort(PETSC_COMM_WORLD, MatDuplicate(spatial.mass, MAT_COPY_VALUES, &own_n));
      PetscCallAbort(PETSC_COMM_WORLD, MatScale(own_n, -1.0 / dt));
      dblk = own_d; nblk = own_n;
    }
    KSP dlu = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD, KSPCreate(PETSC_COMM_WORLD, &dlu));
    PetscCallAbort(PETSC_COMM_WORLD, KSPSetOptionsPrefix(dlu, "exact_march_d_"));
    PetscCallAbort(PETSC_COMM_WORLD, KSPSetOperators(dlu, dblk, dblk));
    PetscCallAbort(PETSC_COMM_WORLD, KSPSetType(dlu, KSPPREONLY));
    PC dpc = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD, KSPGetPC(dlu, &dpc));
    PetscCallAbort(PETSC_COMM_WORLD, PCSetType(dpc, PCLU));
    PetscCallAbort(PETSC_COMM_WORLD, KSPSetUp(dlu));
    for (unsigned slot = 0; slot < steps; ++slot) {
      CopyDataSideSliceToSpatial(dd.rhs_y, slot, local, r);
      if (slot > 0u) {
        PetscCallAbort(PETSC_COMM_WORLD, MatMult(nblk, prev, tmp));
        PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(r, -1.0, tmp));
      }
      PetscCallAbort(PETSC_COMM_WORLD, KSPSolve(dlu, r, y));
      PetscCallAbort(PETSC_COMM_WORLD, VecCopy(y, prev));
      CopySpatialToDataSideSlice(y, slot, local, 1.0, marched, false);
    }
    {
      PetscScalar* m = nullptr;
      const PetscScalar* uu = nullptr;
      PetscCallAbort(PETSC_COMM_WORLD, VecGetArray(marched, &m));
      PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(f.control, &uu));
      const std::vector<unsigned char>& mask = dc.control_mask();
      for (unsigned slot = 0; slot < steps; ++slot)
        for (PetscInt i = 0; i < local; ++i)
          if (mask[static_cast<std::size_t>(i)])
            m[static_cast<PetscInt>(slot) * local + i] = config.eta * uu[static_cast<PetscInt>(slot) * local + i];
      PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(f.control, &uu));
      PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArray(marched, &m));
    }
    PetscCallAbort(PETSC_COMM_WORLD, KSPDestroy(&dlu));
    PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&bd));
    PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&own_d));
    PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&own_n));
  }
  for (unsigned slot = 0; slot < steps && !IsDirichletCase(); ++slot) {
    CopyDataSideSliceToSpatial(data.rhs_y, slot, local, r);
    CopyDataSideSliceToSpatial(f.control, slot, local, u);
    PetscCallAbort(PETSC_COMM_WORLD, MatMult(spatial.control, u, tmp));
    PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(r, config.eta, tmp));
    if (slot > 0u) {
      PetscCallAbort(PETSC_COMM_WORLD, MatMult(coupling, prev, tmp));
      PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(r, 1.0, tmp));
    }
    PetscCallAbort(PETSC_COMM_WORLD, KSPSolve(lu, r, y));
    PetscCallAbort(PETSC_COMM_WORLD, VecCopy(y, prev));
    CopySpatialToDataSideSlice(y, slot, local, 1.0, marched, false);
    // the march is of the homogeneous unknown; the physical state adds G
    if (spatial.lift.g)
      CopySpatialToDataSideSlice(spatial.lift.g, slot, local, 1.0, marched, true);
  }
  const double march_rel = RelDiff(f.state, marched);
  // discrete stationarity alpha_u u + eta p = 0 on every slot (relative)
  Vec stat = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(f.control, &stat));
  PetscCallAbort(PETSC_COMM_WORLD, VecSet(stat, 0.0));
  PetscCallAbort(PETSC_COMM_WORLD, VecAXPBYPCZ(stat, config.alpha_u, config.eta,
                                               0.0, f.control, f.adjoint));
  PetscReal stat_norm = 0.0, au_norm = 0.0;
  PetscCallAbort(PETSC_COMM_WORLD, VecNorm(stat, NORM_2, &stat_norm));
  PetscCallAbort(PETSC_COMM_WORLD, VecNorm(f.control, NORM_2, &au_norm));
  if (IsDirichletCase()) {
    // T7D G8D: Dirichlet stationarity  alpha_u R_H1t u - eta B_D^T p = (gamma_u/dt) q_0
    // (ComputeKktResiduals), and the strong trace y_C = eta u.
    DirichletControlOperators& dc = *spatial.dirichlet;
    double trace_err = 0.0, n_c = 0.0, zmin = 1e300, zmax = -1e300;
    const PetscScalar *yy = nullptr, *uu = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(f.state, &yy));
    PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(f.control, &uu));
    const std::vector<unsigned char>& mask = dc.control_mask();
    for (PetscInt i = 0; i < local; ++i)
      if (mask[static_cast<std::size_t>(i)]) {
        n_c += 1.0; zmin = std::min(zmin, spatial.layout.z[i]); zmax = std::max(zmax, spatial.layout.z[i]);
        for (unsigned slot = 0; slot < steps; ++slot) {
          const PetscInt k = static_cast<PetscInt>(slot) * local + i;
          trace_err = std::max(trace_err, std::fabs(PetscRealPart(yy[k]) - config.eta * PetscRealPart(uu[k])));
        }
      }
    PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(f.control, &uu));
    PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(f.state, &yy));
    std::printf("EX21_DIRICHLET_STATIONARITY control_nodes=%.0f z_range=[%.6g,%.6g]"
                " kkt_stationarity_rel=%.3e kkt_stationarity_abs=%.3e max|y_C-eta u|=%.3e"
                " ell=%.6g gamma_u=%.6g transposes_stiffness=%d\n", n_c, zmin, zmax,
                solve.kkt.stationarity_rel, solve.kkt.stationarity_abs, trace_err,
                config.control_h1_length, config.control_h1_time_weight,
                dc.transposes_stiffness() ? 1 : 0);
    stat_norm = solve.kkt.stationarity_rel;
    au_norm = 1.0 / config.alpha_u;   // reported alpha_u_u_plus_eta_p_rel = stationarity_rel
  }
  if (IsNeumannCase()) {
    // T7N G7: Neumann stationarity on Gamma_c in the control metric,
    //   ||M_Gamma (alpha_u u + eta p)|| / ||alpha_u M_Gamma u||  (all slots),
    // since u = 0 but p != 0 off Gamma_c; and u must vanish off Gamma_c.
    double num = 0.0, den = 0.0, off = 0.0;
    const std::vector<unsigned char> active = ActiveControlDofs(spatial.control, spatial.layout);
    for (unsigned slot = 0; slot < steps; ++slot) {
      CopyDataSideSliceToSpatial(stat, slot, local, r);
      PetscCallAbort(PETSC_COMM_WORLD, MatMult(spatial.control, r, tmp));
      PetscReal a = 0.0;
      PetscCallAbort(PETSC_COMM_WORLD, VecNorm(tmp, NORM_2, &a));
      num += a * a;
      CopyDataSideSliceToSpatial(f.control, slot, local, r);
      PetscCallAbort(PETSC_COMM_WORLD, MatMult(spatial.control, r, tmp));
      PetscCallAbort(PETSC_COMM_WORLD, VecNorm(tmp, NORM_2, &a));
      den += config.alpha_u * config.alpha_u * a * a;
      const PetscScalar* uv = nullptr;
      PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(r, &uv));
      for (PetscInt i = 0; i < local; ++i)
        if (!active[static_cast<std::size_t>(i)]) off = std::max(off, std::fabs(PetscRealPart(uv[i])));
      PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(r, &uv));
    }
    long long n_active = 0;
    double zmax = -1e300, zmin = 1e300;
    for (PetscInt i = 0; i < local; ++i)
      if (active[static_cast<std::size_t>(i)]) {
        ++n_active;
        zmax = std::max(zmax, spatial.layout.z[i]);
        zmin = std::min(zmin, spatial.layout.z[i]);
      }
    std::printf("EX21_NEUMANN_STATIONARITY face=%u control_nodes=%lld z_range=[%.6g,%.6g]"
                " |M_G(alpha_u u + eta p)|/|alpha_u M_G u|=%.3e max|u|_off_Gamma_c=%.3e\n",
                config.neumann_control_face, n_active, zmin, zmax,
                den > 0 ? std::sqrt(num / den) : std::sqrt(num), off);
    // the reported stationarity (EX21_EXACT_RESULT) is the Gamma_c one
    stat_norm = std::sqrt(num);
    au_norm = std::sqrt(den) / config.alpha_u;
  }

  // ---- boundary values: y = g (0 without data), p = u = 0 on dOmega
  double bnd_y = 0.0, bnd_p = 0.0, bnd_u = 0.0;
  long long bnd_nodes = 0;
  {
    const PetscScalar *ys = nullptr, *ps = nullptr, *us = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(f.state, &ys));
    PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(f.adjoint, &ps));
    PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(f.control, &us));
    for (PetscInt i = 0; i < local; ++i) {
      const double x = spatial.layout.x[i], z = spatial.layout.z[i];
      if (!(std::fabs(x - D.xmin) < 1e-12 || std::fabs(x - D.xmax) < 1e-12 ||
            std::fabs(z - D.zmin) < 1e-12 || std::fabs(z - D.zmax) < 1e-12))
        continue;
      ++bnd_nodes;
      const double g = config.data_overrides.dirichlet_state
                           ? config.data_overrides.dirichlet_state(config, x, z) : 0.0;
      for (unsigned slot = 0; slot < steps; ++slot) {
        const PetscInt k = static_cast<PetscInt>(slot) * local + i;
        bnd_y = std::max(bnd_y, std::fabs(PetscRealPart(ys[k]) - g));
        bnd_p = std::max(bnd_p, std::fabs(PetscRealPart(ps[k])));
        bnd_u = std::max(bnd_u, std::fabs(PetscRealPart(us[k])));
      }
    }
    PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(f.control, &us));
    PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(f.adjoint, &ps));
    PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(f.state, &ys));
  }
  std::printf("EX21_EXACT_BOUNDARY nodes=%lld max|y-g|=%.3e max|p|=%.3e max|u|=%.3e\n",
              bnd_nodes, bnd_y, bnd_p, bnd_u);
  // ---- T3b objective checks.  (i) J_track recomputed DIRECTLY from the
  // returned physical state with the full tracking matrix (lifted cases);
  // (ii) the continuous J of the exact solution (manufactured cases), by
  // composite 3-point Gauss in time (64 subintervals) x element quadrature.
  {
    double j_direct = -1.0, j_exact_track = -1.0, j_exact_reg = -1.0;
    if (spatial.lift.g && spatial.lift.tracking_full) {
      Vec d = nullptr, md = nullptr;
      PetscCallAbort(PETSC_COMM_WORLD, MatCreateVecs(spatial.lift.tracking_full, &d, &md));
      double sum = 0.0;
      for (unsigned slot = 0; slot <= steps; ++slot) {
        if (slot == 0u && !cn) continue;
        const double t = slot * dt;
        if (slot > 0u) CopyDataSideSliceToSpatial(f.state, slot - 1u, local, d);
        PetscScalar* a = nullptr;
        PetscCallAbort(PETSC_COMM_WORLD, VecGetArray(d, &a));
        for (PetscInt i = 0; i < local; ++i) {
          const double x = spatial.layout.x[i], z = spatial.layout.z[i];
          const double yv = slot == 0u ? EvaluateInitialState(config, x, z) : PetscRealPart(a[i]);
          a[i] = yv - EvaluateTrackingTarget(config, x, z, t);
        }
        PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArray(d, &a));
        PetscCallAbort(PETSC_COMM_WORLD, MatMult(spatial.lift.tracking_full, d, md));
        PetscScalar q = 0.0;
        PetscCallAbort(PETSC_COMM_WORLD, VecDot(d, md, &q));
        const double w = !cn ? 1.0 : ((slot == 0u || slot == steps) ? 0.5 : 1.0);
        sum += dt * w * 0.5 * config.alpha_track * PetscRealPart(q);
      }
      j_direct = sum;
      PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&md));
      PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&d));
    }
    if (Convective()) {
      std::vector<double> zero(static_cast<std::size_t>(local), 0.0);
      const double gx[3] = {-std::sqrt(0.6), 0.0, std::sqrt(0.6)};
      const double gw[3] = {5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0};
      const unsigned intervals = 64;
      const double ht = T / intervals;
      j_exact_track = j_exact_reg = 0.0;
      for (unsigned k = 0; k < intervals; ++k)
        for (int q = 0; q < 3; ++q) {
          const double t = (k + 0.5) * ht + 0.5 * ht * gx[q];
          ExactContext cr{&config, t, 4, 1.0}, cu{&config, t, 2, 1.0};
          j_exact_track += 0.5 * ht * gw[q] * 0.5 * config.alpha_track *
              IntegrateSpatialField(mesh, solution_type, zero.data(), ExactAt, &cr).exact_squared;
          // the regularisation lives on the control set: Omega (distributed)
          // or Gamma_c (T7N Neumann control)
          if (IsDirichletCase()) {
            const double Uv = DirU(config, t), ell = config.control_h1_length;
            j_exact_reg += 0.5 * ht * gw[q] * 0.5 * config.alpha_u * Uv * Uv * 0.5 *
                           (1.0 + kPi * kPi * ell * ell);
            continue;
          }
          j_exact_reg += 0.5 * ht * gw[q] * 0.5 * config.alpha_u *
              (IsNeumannCase()
                   ? IntegrateBoundaryField(mesh, solution_type, config.neumann_control_face,
                                            zero.data(), ExactAt, &cu).exact_squared
                   : IntegrateSpatialField(mesh, solution_type, zero.data(), ExactAt, &cu)
                         .exact_squared);
        }
    }
    std::printf("EX21_EXACT_OBJECTIVE J_track=%.12e J_reg=%.12e J_total=%.12e"
                " J_track_direct=%.12e direct_rel=%.3e J_exact_track=%.12e"
                " J_exact_reg=%.12e J_exact_total=%.12e J_err_rel=%.6e\n",
                f.J_track, f.J_reg, f.J_track + f.J_reg, j_direct,
                j_direct >= 0 ? std::fabs(j_direct - f.J_track) / std::max(1e-300, std::fabs(j_direct)) : -1.0,
                j_exact_track, j_exact_reg, j_exact_track + j_exact_reg,
                j_exact_track >= 0
                    ? std::fabs(f.J_track + f.J_reg - j_exact_track - j_exact_reg) /
                          (j_exact_track + j_exact_reg)
                    : -1.0);
  }
  // ---- solver cost: per-sweep-context counters, as Benchmark2D reports them
  {
    auto counter = [&](const char* nm, const InnerIterationCounter& c) {
      std::printf(" %s_solves=%lld %s_its=%lld %s_max=%lld", nm, c.solves, nm,
                  c.iterations, nm, c.max_iterations);
    };
    std::printf("EX21_EXACT_SOLVER_STATS outer_its=%d reason=%d true_residual=%.3e"
                " state_rel=%.3e adjoint_rel=%.3e sweeps=%d",
                solve.outer_iterations, solve.reason, solve.true_residual,
                solve.kkt.state_rel, solve.kkt.adjoint_rel, solve.sweeps_used ? 1 : 0);
    counter("right", solve.right_inner);
    counter("left", solve.left_inner);
    counter("ay", solve.ay_inner);
    counter("ysplit", solve.y_split);
    counter("schur_ksp", solve.p_split);
    std::printf(" sweep_failures=%lld solve_wall=%.3f accepted=%d status=%s\n",
                solve.sweep_failures, solve_wall, solve.accepted ? 1 : 0,
                solve.status.c_str());
  }
  if (g_case == Case::LP52) {
    const int status = ReportLp52(config, options, spatial, solve, march_rel,
                                  stat_norm / std::max(1e-300, config.alpha_u * au_norm),
                                  solve_wall, MPI_Wtime() - t0);
    PetscCallAbort(PETSC_COMM_WORLD, KSPDestroy(&lu));
    for (Vec* v : {&r, &y, &prev, &tmp, &u, &marched, &stat})
      PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(v));
    PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&coupling));
    PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&diagonal));
    return status;
  }

  // ---- errors against the exact solution (element quadrature, true L2)
  // state y_n at t_n; adjoint and control at the interval midpoint t_{n-1/2}
  // under CN (the interval quantities of the DTO scheme) and at t_n under BE.
  Accumulator ey, ep, eu, ep_node, ep_sign;
  double y_T = 0, y_T_rel = 0, interp_T = 0, y_rel_unshifted_e = 0,
         y_rel_unshifted_x = 0;
  std::vector<double> slot_y(steps), slot_p(steps), slot_u(steps), slot_px(steps);
  std::vector<double> values;
  for (unsigned slot = 0; slot < steps; ++slot) {
    const double tn = (slot + 1u) * dt, tm = (slot + 0.5) * dt;
    const double tp = cn ? tm : tn;
    const bool last = slot + 1u == steps;
    auto integrate = [&](Vec field, int which, double time, double sign) {
      CopyDataSideSliceToSpatial(field, slot, local, r);
      const PetscScalar* a = nullptr;
      PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(r, &a));
      ExactContext ctx{&config, time, which, sign};
      const SpatialFieldIntegrals I =
          IntegrateSpatialField(mesh, solution_type, a, ExactAt, &ctx);
      PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(r, &a));
      return I;
    };
    const SpatialFieldIntegrals Iy = integrate(f.state, 0, tn, 1.0);
    const SpatialFieldIntegrals Ip = integrate(f.adjoint, 1, tp, 1.0);
    // T7N: the Neumann control is a trace on Gamma_c -> L2(Gamma_c) error with
    // the face quadrature that assembled M_Gamma (never a nodal max norm)
    // T7D: the Dirichlet control is NODAL in time (u_n at t_n, both schemes).
    const double tu = IsDirichletCase() ? tn : tp;
    const SpatialFieldIntegrals Iu = [&]() {
      if (!IsNeumannCase() && !IsDirichletCase()) return integrate(f.control, 2, tp, 1.0);
      CopyDataSideSliceToSpatial(f.control, slot, local, r);
      const PetscScalar* a = nullptr;
      PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(r, &a));
      ExactContext ctx{&config, tu, 2, 1.0};
      const SpatialFieldIntegrals I = IntegrateBoundaryField(
          mesh, solution_type, config.neumann_control_face, a, ExactAt, &ctx);
      PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(r, &a));
      return I;
    }();
    ey.Add(Iy, dt, tn, T, last);
    ep.Add(Ip, dt, tp, T, last);
    eu.Add(Iu, dt, IsDirichletCase() ? tn : tp, T, last);
    ep_node.Add(integrate(f.adjoint, 1, tn, 1.0), dt, tn, T, last);
    ep_sign.Add(integrate(f.adjoint, 1, tp, -1.0), dt, tp, T, last);
    slot_y[slot] = std::sqrt(Iy.error_squared);
    slot_p[slot] = std::sqrt(Ip.error_squared);
    slot_u[slot] = std::sqrt(Iu.error_squared);
    slot_px[slot] = std::sqrt(Ip.exact_squared);
    if (g_case == Case::LP_HEAT) {
      // relative to the UNSHIFTED state y = 1 + w (same error, larger norm)
      const SpatialFieldIntegrals I1 = integrate(f.state, 3, tn, 1.0);
      y_rel_unshifted_e += dt * Iy.error_squared;
      y_rel_unshifted_x += dt * I1.exact_squared;
    }
    if (last) {
      y_T = std::sqrt(Iy.error_squared);
      y_T_rel = y_T / std::sqrt(Iy.exact_squared);
      // interpolation error of the exact state at T (reference floor)
      PetscScalar* a = nullptr;
      PetscCallAbort(PETSC_COMM_WORLD, VecGetArray(r, &a));
      for (PetscInt i = 0; i < local; ++i)
        a[i] = ExactY(config, spatial.layout.x[i], spatial.layout.z[i], tn);
      ExactContext ctx{&config, tn, 0, 1.0};
      interp_T = std::sqrt(
          IntegrateSpatialField(mesh, solution_type, a, ExactAt, &ctx).error_squared);
      PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArray(r, &a));
    }
  }

  // ex20-comparable metric for lp-heat: relative nodal l2 over the interior
  // space-time nodes, state including the shift (ex20 EX20_LEVEQUE_PEARSON
  // section 12, y_l2rel / p_l2rel), adjoint at t_{n-1/2} (CN) or t_n (BE)
  double ex20_y_e = 0, ex20_y_x = 0, ex20_p_e = 0, ex20_p_x = 0;
  if (g_case == Case::LP_HEAT) {
    const double edge = 1.0 - 1e-12;
    const PetscScalar *ys = nullptr, *ps = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(f.state, &ys));
    PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(f.adjoint, &ps));
    for (unsigned slot = 0; slot < steps; ++slot) {
      const double tn = (slot + 1u) * dt, tp = cn ? (slot + 0.5) * dt : tn;
      for (PetscInt i = 0; i < local; ++i) {
        const double x = spatial.layout.x[i], z = spatial.layout.z[i];
        if (std::fabs(x) > edge || std::fabs(z) > edge) continue;
        const PetscInt k = static_cast<PetscInt>(slot) * local + i;
        const double wy = 1.0 + ExactY(config, x, z, tn);
        const double wp = ExactP(config, x, z, tp);
        ex20_y_e += std::pow(1.0 + PetscRealPart(ys[k]) - wy, 2);
        ex20_y_x += wy * wy;
        ex20_p_e += std::pow(PetscRealPart(ps[k]) - wp, 2);
        ex20_p_x += wp * wp;
      }
    }
    PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(f.adjoint, &ps));
    PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(f.state, &ys));
  }

  std::printf("EX21_EXACT_RESULT case=%s scheme=%s nx=%u steps=%u dt=%.10e"
              " alpha_u=%.6e reason=%d outer_its=%d true_residual=%.3e"
              " state_rel=%.3e adjoint_rel=%.3e stationarity_rel=%.3e"
              " alpha_u_u_plus_eta_p_rel=%.3e march_rel=%.3e J_track=%.12e"
              " J_reg=%.12e y_T=%.10e y_T_rel=%.10e interp_y_T=%.10e",
              name, TimeSchemeName(config), config.mesh.nx, steps, dt,
              config.alpha_u, solve.reason, solve.outer_iterations,
              solve.true_residual, solve.kkt.state_rel, solve.kkt.adjoint_rel,
              solve.kkt.stationarity_rel,
              au_norm > 0 ? stat_norm / (config.alpha_u * au_norm) : stat_norm,
              march_rel, f.J_track, f.J_reg, y_T, y_T_rel, interp_T);
  ey.Print("y");
  ep.Print("p");
  eu.Print("u");
  ep_node.Print("p_nodetime");
  ep_sign.Print("p_wrongsign");
  if (g_case == Case::LP_HEAT)
    std::printf(" y_L2L2_rel_unshifted=%.10e ex20_y_l2rel=%.10e ex20_p_l2rel=%.10e",
                std::sqrt(y_rel_unshifted_e / y_rel_unshifted_x),
                std::sqrt(ex20_y_e / ex20_y_x), std::sqrt(ex20_p_e / ex20_p_x));
  std::printf(" solve_wall=%.3f total_wall=%.3f\n", solve_wall, MPI_Wtime() - t0);
  for (unsigned slot = 0; slot < steps; ++slot)
    std::printf("EX21_EXACT_SLOT n=%u t_n=%.10e e_y=%.10e e_p=%.10e e_u=%.10e"
                " p_exact=%.10e\n",
                slot + 1u, (slot + 1u) * dt, slot_y[slot], slot_p[slot],
                slot_u[slot], slot_px[slot]);

  PetscCallAbort(PETSC_COMM_WORLD, KSPDestroy(&lu));
  for (Vec* v : {&r, &y, &prev, &tmp, &u, &marched, &stat})
    PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(v));
  PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&coupling));
  PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&diagonal));
  return solve.accepted ? 0 : 8;
}
