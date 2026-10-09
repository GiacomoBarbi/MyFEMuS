// T9 (3D phase 1): exact distributed optimal control on the unit cube
// (see ExactOc3D.hpp for the formulas).

#include "ExactOc3D.hpp"

#include <mpi.h>
#include <petscksp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>

#include "Discretization.hpp"
#include "Preconditioners.hpp"
#include "Solver.hpp"
#include "SpaceTimeOperators.hpp"

using namespace femus;

namespace {

// Shape: conv-sin-3d uses s(xi) = sin(pi xi); conv-sin3-3d uses sin^3(pi xi),
// whose data (f, y, grad y, Lap y) vanish on dOmega -- the 3D analogue of the
// 2D conv-sin3 (EX21_CONVECTION_VALIDATION.md: with sin and w != 0 the source
// has a nonzero boundary trace, which the inherited load M_elim I_h f drops,
// limiting y to h^{5/2}).
bool g_cubic = false;
// T10: natural-conv-3d (3D extension of the T7F natural-conv, t10_symbolic.py):
//   natural state BC on all six faces, constant wind w = (vx, vy, vz),
//   R_k(s) = exp(-k s) (2 + cos(pi s)),  R_k' = -k R_k at s = 0 and s = 1,
//   p = alpha_u B(t) R_kx(x) R_ky(y) R_kz(z),  k_i = w_i/nu  (0 with convection
//   off), so nu dn p + (w.n) p = 0 on every face with p != 0 there; p(T) = 0;
//   y = A(t) cos(pi x) cos(pi y) cos(pi z)  (nu dn y = 0 on every face);
//   u = -(eta/alpha_u) p;  f = y_t - nu Lap y + w.grad y - eta u;
//   y_d = y + (p_t + nu Lap p + w.grad p)/alpha_t.
bool g_natural = false;
// T11N: neumann-conv-3d (3D extension of the T7N neumann-conv, t11n_symbolic.py):
//   Gamma_c = {z = 0} (FEMuS face 1), Neumann control nu dn y = eta u there,
//   natural state BC on the other five faces, constant wind w = (vx, vy, vz);
//   X(s) = 1 - cos(2 pi s) (X = X' = 0 at s = 0, 1),  Z = R_kz, kz = vz/nu;
//   p = alpha_u B(t) X(x) X(y) Z(z): nu dn p + (w.n) p = 0 on all six faces
//     (trivially on the four lateral faces, where p = dn p = 0 -- smooth
//     compatibility at the edges of Gamma_c forces this; through w_z on z = 0, 1);
//   u = -(eta/alpha_u) p|_{z=0} = -eta B X(x) X(y) Z(0);
//   y = A cos(pi x) cos(pi y) cos(pi z) + kappa(t) X(x) X(y) H(z),
//     H = z - z^2/2 (H'(0) = 1, H'(1) = 0), kappa = eta^2 B Z(0)/nu, so that
//     -nu dz y = eta u on z = 0 and nu dn y = 0 on the other faces;
//   f = y_t - nu Lap y + w.grad y;  y_d = y + (p_t + nu Lap p + w.grad p)/alpha_t.
bool g_neumann = false;
// T11D-L2: dirichlet-conv-3d (t11d_symbolic.py).  Gamma_c = {z = 0} (FEMuS face 1),
//   Dirichlet control y = eta u there, homogeneous Dirichlet y = 0 on the other
//   five faces, tracking omega = (0,1)^2 x (1/2,1), constant wind w.
//   X_c(s) = exp(-c s) sin(pi s), c = w_s/(2 nu):  -nu X'' - w_s X' = lambda X,
//     lambda = nu pi^2 + w_s^2/(4 nu);  p = X(x) X(y) Q(z,t),
//   Q = (1 - phi(z)) Qs,  Qs = exp(-a z - c tau)[Phi(z - z1, tau) - Phi(z + z1, tau)],
//     tau = T - t, a = w_z/(2 nu), c = w_z^2/(4 nu) + lambda_x + lambda_y, z1 = 1.2
//     (outside), Phi the 1D heat kernel (diffusivity nu): Qs solves the
//     homogeneous backward adjoint exactly, Qs(0,t) = 0, Qs(.,T) = 0;
//     phi a C^3 septic step (0 for z <= 1/2, 1 at z = 1)  =>  p = 0 on the
//     whole boundary, p(T) = 0, adjoint residual r = 0 for z <= 1/2;
//   stationarity (L2):  alpha_u u = eta nu dn p  =>  u = -(eta nu/alpha_u) X X dQs/dz(0,t)
//     (u = 0 on the perimeter of Gamma_c);
//   y = A(t) sin^3(pi x) sin^3(pi y) sin^3(pi z) + eta u(x,y,t) (1 - z)^3;
//   (sin^3 and (1 - z)^3 make f vanish on the five homogeneous faces when the
//   wind is normal to Gamma_c, w = (0, 0, w_z): the inherited load M_elim I_h f
//   drops f there (T9 h^{5/2} effect); with tangential wind components the
//   lateral trace of f is 2 pi w_x k ... != 0 -- valid data, but y then converges
//   at h^{5/2}.  The convergence studies use w = (0, 0, w_z).)
//   f = y_t - nu Lap y + w.grad y;  y_d = y - r/alpha_t  (r = -p_t - nu Lap p - w.grad p).
bool g_dirichlet = false;
// T11H1 dirichlet-h1-3d: the same p, the control u = S(x,y) U(t) with S = X X and
// the temporal factor U solving the H1 stationarity (ExactOc3D.hpp); data reads ell
// and gamma_u from the config.  g_dirichlet stays true (shared Dirichlet driver).
bool g_dir_h1 = false;
// T14 variable-wind exact cases: 1 natural-var-3d, 2 neumann-var-3d (zmin), 3 neumann-var-opp-3d
// (zmin u zmax), 4 dirichlet-var-3d (zmin, owned swirl wind).  conv-sin3-3d also accepts a 3D field.
int g_var = 0;

struct F1 { double v, d, dd; };   // 1D factor and its first two derivatives
F1 Factor(double xi) {
  const double sn = std::sin(kPi * xi), cs = std::cos(kPi * xi);
  if (!g_cubic) return {sn, kPi * cs, -kPi * kPi * sn};
  return {sn * sn * sn, 3.0 * kPi * sn * sn * cs,
          3.0 * kPi * kPi * sn * (2.0 * cs * cs - sn * sn)};
}

double S(double x, double y, double z) {
  if (!g_cubic) return std::sin(kPi * x) * std::sin(kPi * y) * std::sin(kPi * z);
  return Factor(x).v * Factor(y).v * Factor(z).v;
}
/** w . grad S for the constant wind of the config (0 with convection off). */
double WindGradS(const ProblemConfig& c, double x, double y, double z) {
  if (!HasConvection(c)) return 0.0;
  if (g_cubic) {
    const F1 X = Factor(x), Y = Factor(y), Z = Factor(z);
    if (c.convection.field3d) {   // T14: w(x) . grad S
      double w[3];
      EvaluateVelocity3D(c, x, y, z, w);
      return w[0] * X.d * Y.v * Z.v + w[1] * X.v * Y.d * Z.v + w[2] * X.v * Y.v * Z.d;
    }
    return c.convection.vx * X.d * Y.v * Z.v + c.convection.vy * X.v * Y.d * Z.v +
           c.convection.vz * X.v * Y.v * Z.d;
  }
  const double sx = std::sin(kPi * x), sy = std::sin(kPi * y), sz = std::sin(kPi * z);
  const double cx = std::cos(kPi * x), cy = std::cos(kPi * y), cz = std::cos(kPi * z);
  return kPi * (c.convection.vx * cx * sy * sz + c.convection.vy * sx * cy * sz +
                c.convection.vz * sx * sy * cz);
}
/** -Lap S (= 3 pi^2 S for the sin shape). */
double MinusLapS(double x, double y, double z) {
  if (!g_cubic) return 3.0 * kPi * kPi * S(x, y, z);
  const F1 X = Factor(x), Y = Factor(y), Z = Factor(z);
  return -(X.dd * Y.v * Z.v + X.v * Y.dd * Z.v + X.v * Y.v * Z.dd);
}
/** T10: natural-conv-3d 1D factors.  NatR: R_k and its derivatives; NatC:
 *  cos(pi s) and its derivatives. */
F1 NatR(double k, double s) {
  const double e = std::exp(-k * s);
  const double g = 2.0 + std::cos(kPi * s), dg = -kPi * std::sin(kPi * s),
               ddg = -kPi * kPi * std::cos(kPi * s);
  return {e * g, e * (dg - k * g), e * (ddg - 2.0 * k * dg + k * k * g)};
}
F1 NatC(double s) {
  return {std::cos(kPi * s), -kPi * std::sin(kPi * s), -kPi * kPi * std::cos(kPi * s)};
}
/** value, w.grad and Laplacian of the product X(x) Y(y) Z(z) */
struct Prod { double v, wgrad, lap; };
Prod Product(const ProblemConfig& c, const F1& X, const F1& Y, const F1& Z) {
  const double w[3] = {HasConvection(c) ? c.convection.vx : 0.0,
                       HasConvection(c) ? c.convection.vy : 0.0,
                       HasConvection(c) ? c.convection.vz : 0.0};
  return {X.v * Y.v * Z.v,
          w[0] * X.d * Y.v * Z.v + w[1] * X.v * Y.d * Z.v + w[2] * X.v * Y.v * Z.d,
          X.dd * Y.v * Z.v + X.v * Y.dd * Z.v + X.v * Y.v * Z.dd};
}
Prod NatYShape(const ProblemConfig& c, double x, double y, double z) {
  return Product(c, NatC(x), NatC(y), NatC(z));
}
Prod NatPShape(const ProblemConfig& c, double x, double y, double z) {
  const double nu = c.diffusivity;
  const double kx = HasConvection(c) ? c.convection.vx / nu : 0.0,
               ky = HasConvection(c) ? c.convection.vy / nu : 0.0,
               kz = HasConvection(c) ? c.convection.vz / nu : 0.0;
  return Product(c, NatR(kx, x), NatR(ky, y), NatR(kz, z));
}

/** T11D: dirichlet-conv-3d factors. */
constexpr double kDir3Z1 = 1.2, kDir3Za = 0.5, kDir3Zb = 1.0;
F1 DirX(double c, double s) {   // exp(-c s) sin(pi s)
  const double e = std::exp(-c * s), sn = std::sin(kPi * s), cs = std::cos(kPi * s);
  return {e * sn, e * (kPi * cs - c * sn), e * ((c * c - kPi * kPi) * sn - 2.0 * c * kPi * cs)};
}
double DirC(const ProblemConfig& c, unsigned axis) {   // w_axis / (2 nu)
  if (!HasConvection(c)) return 0.0;
  const double w = axis == 0 ? c.convection.vx : (axis == 1 ? c.convection.vy : c.convection.vz);
  return w / (2.0 * c.diffusivity);
}
struct Heat3 { double v = 0, z = 0, zz = 0, tau = 0, ztau = 0; };   // Qs and derivatives
Heat3 DirQs3(const ProblemConfig& c, double z, double t) {
  Heat3 h;
  const double tau = c.final_time() - t;
  if (tau <= 1e-14) return h;
  const double nu = c.diffusivity;
  const double wz = HasConvection(c) ? c.convection.vz : 0.0;
  const double cx = DirC(c, 0), cy = DirC(c, 1);
  const double lam = 2.0 * nu * kPi * kPi + nu * (cx * cx + cy * cy);   // lambda_x + lambda_y
  const double a = wz / (2.0 * nu), cc = wz * wz / (4.0 * nu) + lam;
  auto phi = [&](double xi, double& p0, double& p1, double& p2, double& p3) {
    p0 = std::exp(-xi * xi / (4.0 * nu * tau)) / std::sqrt(4.0 * kPi * nu * tau);
    p1 = -xi / (2.0 * nu * tau) * p0;
    p2 = (xi * xi / (4.0 * nu * nu * tau * tau) - 1.0 / (2.0 * nu * tau)) * p0;
    p3 = (-xi * xi * xi / (8.0 * nu * nu * nu * tau * tau * tau) +
          3.0 * xi / (4.0 * nu * nu * tau * tau)) * p0;
  };
  double a0, a1, a2, a3, b0, b1, b2, b3;
  phi(z - kDir3Z1, a0, a1, a2, a3);
  phi(z + kDir3Z1, b0, b1, b2, b3);
  const double V = a0 - b0, Vz = a1 - b1, Vzz = a2 - b2, Vzzz = a3 - b3;
  const double E = std::exp(-a * z - cc * tau);
  h.v = E * V;
  h.z = E * (Vz - a * V);
  h.zz = E * (Vzz - 2.0 * a * Vz + a * a * V);
  h.tau = E * (nu * Vzz - cc * V);                                   // V_tau = nu V_zz
  h.ztau = E * (-cc * (Vz - a * V) + nu * Vzzz - a * nu * Vzz);
  return h;
}
/** C^3 septic smoothstep 0 -> 1 on [Za, Zb] = [1/2, 1]: 35s^4 - 84s^5 + 70s^6 - 20s^7.
 *  (Not the T7D C-infinity exp(-1/s) step: its second derivative is huge in thin
 *  zones that coarse 3D meshes cannot resolve.)  z = 1/2 is an element face for
 *  even n, so the finite smoothness there does not limit Q2. */
void DirStep3(double z, double& f, double& f1, double& f2) {
  const double L = kDir3Zb - kDir3Za, s = (z - kDir3Za) / L;
  if (s <= 0.0) { f = f1 = f2 = 0.0; return; }
  if (s >= 1.0) { f = 1.0; f1 = f2 = 0.0; return; }
  const double s2 = s * s, s3 = s2 * s, s4 = s3 * s;
  f = s4 * (35.0 - 84.0 * s + 70.0 * s2 - 20.0 * s3);
  f1 = 140.0 * s3 * (1.0 - s) * (1.0 - s) * (1.0 - s) / L;
  f2 = 420.0 * s2 * (1.0 - s) * (1.0 - s) * (1.0 - 2.0 * s) / (L * L);
}
struct ZQ { double v, z, zz, t; };   // Q(z,t) = (1 - phi) Qs and derivatives (t: d/dt = -d/dtau)
ZQ DirQ3(const ProblemConfig& c, double z, double t) {
  const Heat3 q = DirQs3(c, z, t);
  double f, f1, f2;
  DirStep3(z, f, f1, f2);
  return {(1.0 - f) * q.v, -f1 * q.v + (1.0 - f) * q.z,
          -f2 * q.v - 2.0 * f1 * q.z + (1.0 - f) * q.zz, -(1.0 - f) * q.tau};
}

/** T11H1: the temporal factor U(t) of u = S(x,y) U(t) (S = X_cx(x) X_cy(y)), and
 *  U_t, for dirichlet-h1-3d.  Stationarity on Gamma_c (n = -e_z, dn p = -S dQs/dz(0)):
 *      a U - gamma_u U_tt = F(tau) := -eta nu dQs/dz(0, tau),   a = alpha_u (1 + ell^2 lambda),
 *  lambda = 2 pi^2 (-Delta_Gamma S = lambda S; ell > 0 needs w_x = w_y = 0 so that X = sin).
 *  gamma_u = 0: U = F/a.  gamma_u > 0 (k = sqrt(a/gamma_u), tau = T - t): the natural
 *  condition u_t(T) = 0 is U_tau(0) = 0; u(0) = u_0 is DATA (y_0 is built from U(T)), so
 *  any U = U_p + A cosh(k tau) is admissible -- the benchmark takes A = 0:
 *      U(tau) = -(1/(gamma k)) int_0^tau sinh(k(tau - s)) F(s) ds,
 *  in closed form with dQs/dz(0,s) = C0 s^{-3/2} exp(-A0/s - cc s), C0 = z1/(nu sqrt(4 pi nu)),
 *  A0 = z1^2/(4 nu) and
 *      E(b,tau) = int_0^tau s^{-3/2} e^{-A0/s - b s} ds
 *               = sqrt(pi)/(2 sqrt(A0)) [e^{2 sqrt(A0 b)} erfc(sqrt(A0/tau) + sqrt(b tau))
 *                                        + e^{-2 sqrt(A0 b)} erfc(sqrt(A0/tau) - sqrt(b tau))]  (b > 0):
 *      U = (eta nu C0/(2 gamma k)) [e^{k tau} E(cc+k) - e^{-k tau} E(cc-k)],
 *      U_tau = (eta nu C0/(2 gamma)) [e^{k tau} E(cc+k) + e^{-k tau} E(cc-k)].
 *  cc > k is required (refused otherwise).  Verified by t11h1_symbolic.py. */
struct DirTime { double U, Ut; };
double DirCC(const ProblemConfig& c) {
  const double nu = c.diffusivity, wz = HasConvection(c) ? c.convection.vz : 0.0;
  const double cx = DirC(c, 0), cy = DirC(c, 1);
  return wz * wz / (4.0 * nu) + 2.0 * nu * kPi * kPi + nu * (cx * cx + cy * cy);
}
double DirH1A(const ProblemConfig& c) {   // alpha_u (1 + ell^2 lambda), lambda = 2 pi^2
  return c.alpha_u * (1.0 + c.control_h1_length * c.control_h1_length * 2.0 * kPi * kPi);
}
DirTime DirH1Time(const ProblemConfig& c, double t) {
  const double tau = c.final_time() - t;
  if (tau <= 1e-14) return {0.0, 0.0};
  const double nu = c.diffusivity, a = DirH1A(c), g = c.control_h1_time_weight;
  if (g <= 0.0) {
    const Heat3 q = DirQs3(c, 0.0, t);
    return {-c.eta * nu * q.z / a, c.eta * nu * q.ztau / a};
  }
  const double k = std::sqrt(a / g), cc = DirCC(c);
  const double A0 = kDir3Z1 * kDir3Z1 / (4.0 * nu);
  const double C0 = kDir3Z1 / (nu * std::sqrt(4.0 * kPi * nu));
  auto E = [&](double b) {
    const double sa = std::sqrt(A0 * b), r = std::sqrt(A0 / tau), q = std::sqrt(b * tau);
    return std::sqrt(kPi) / (2.0 * std::sqrt(A0)) *
           (std::exp(2.0 * sa) * std::erfc(r + q) + std::exp(-2.0 * sa) * std::erfc(r - q));
  };
  const double ep = std::exp(k * tau) * E(cc + k), em = std::exp(-k * tau) * E(cc - k);
  const double s = c.eta * nu * C0 / (2.0 * g);
  return {s / k * (ep - em), -s * (ep + em)};   // U, U_t = -U_tau
}

/** T11N: neumann-conv-3d factors. */
F1 NeuX(double s) {
  return {1.0 - std::cos(2.0 * kPi * s), 2.0 * kPi * std::sin(2.0 * kPi * s),
          4.0 * kPi * kPi * std::cos(2.0 * kPi * s)};
}
F1 NeuH(double z) { return {z - 0.5 * z * z, 1.0 - z, -1.0}; }
double NeuKz(const ProblemConfig& c) {
  return HasConvection(c) ? c.convection.vz / c.diffusivity : 0.0;
}
Prod NeuPShape(const ProblemConfig& c, double x, double y, double z) {
  return Product(c, NeuX(x), NeuX(y), NatR(NeuKz(c), z));
}
Prod NeuGShape(const ProblemConfig& c, double x, double y, double z) {   // X(x) X(y) H(z)
  return Product(c, NeuX(x), NeuX(y), NeuH(z));
}

double A(double t) { return std::cos(2.0 * t); }
double dA(double t) { return -2.0 * std::sin(2.0 * t); }
double B(const ProblemConfig& c, double t) { return std::sin(c.final_time() - t); }
double dB(const ProblemConfig& c, double t) { return -std::cos(c.final_time() - t); }

double ExactY(const ProblemConfig&, double x, double y, double z, double t) {
  return A(t) * S(x, y, z);
}
double ExactP(const ProblemConfig& c, double x, double y, double z, double t) {
  return c.alpha_u * B(c, t) * S(x, y, z);
}
double ExactU(const ProblemConfig& c, double x, double y, double z, double t) {
  return -c.eta * ExactP(c, x, y, z, t) / c.alpha_u;
}
double Source3D(const ProblemConfig& c, double x, double y, double z, double t) {
  if (g_cubic)   // f = A' S - nu A Lap S + A w.grad S - eta u
    return dA(t) * S(x, y, z) + c.diffusivity * A(t) * MinusLapS(x, y, z) +
           A(t) * WindGradS(c, x, y, z) - c.eta * ExactU(c, x, y, z, t);
  const double s = S(x, y, z), lap = 3.0 * kPi * kPi;
  return (dA(t) + c.diffusivity * lap * A(t)) * s + A(t) * WindGradS(c, x, y, z) -
         c.eta * ExactU(c, x, y, z, t);
}
double Target3D(const ProblemConfig& c, double x, double y, double z, double t) {
  if (g_cubic) {   // y_d = y + alpha_u (B' S + nu B Lap S + B w.grad S)/alpha_t
    double q = c.alpha_u * (dB(c, t) * S(x, y, z) - c.diffusivity * B(c, t) * MinusLapS(x, y, z) +
                            B(c, t) * WindGradS(c, x, y, z));
    if (c.convection.field3d)   // T14: the FULL adjoint, div(w p) = w.grad p + (div w) p
      q += c.alpha_u * B(c, t) * EvaluateDivergence3D(c, x, y, z) * S(x, y, z);
    return ExactY(c, x, y, z, t) + q / c.alpha_track;
  }
  const double s = S(x, y, z), lap = 3.0 * kPi * kPi;
  // y_d = y + (p_t + nu Lap p + w.grad p)/alpha_t
  const double q = c.alpha_u * ((dB(c, t) - c.diffusivity * lap * B(c, t)) * s +
                                B(c, t) * WindGradS(c, x, y, z));
  return ExactY(c, x, y, z, t) + q / c.alpha_track;
}
double Initial3D(const ProblemConfig& c, double x, double y, double z) {
  return ExactY(c, x, y, z, 0.0);
}

// T10 natural-conv-3d: separate functions (the T9 bodies above stay unchanged,
// so the phase-1 cases compile -- and round -- exactly as before).
double NatExactY(const ProblemConfig& c, double x, double y, double z, double t) {
  return A(t) * NatYShape(c, x, y, z).v;
}
double NatExactP(const ProblemConfig& c, double x, double y, double z, double t) {
  return c.alpha_u * B(c, t) * NatPShape(c, x, y, z).v;
}
double NatExactU(const ProblemConfig& c, double x, double y, double z, double t) {
  return -c.eta * NatExactP(c, x, y, z, t) / c.alpha_u;
}
double NatSource3D(const ProblemConfig& c, double x, double y, double z, double t) {
  // f = A' Y - nu A Lap Y + A w.grad Y - eta u
  const Prod Y = NatYShape(c, x, y, z);
  return dA(t) * Y.v - c.diffusivity * A(t) * Y.lap + A(t) * Y.wgrad -
         c.eta * NatExactU(c, x, y, z, t);
}
double NatTarget3D(const ProblemConfig& c, double x, double y, double z, double t) {
  // y_d = y + alpha_u (B' P + nu B Lap P + B w.grad P)/alpha_t
  const Prod P = NatPShape(c, x, y, z);
  const double q = c.alpha_u * (dB(c, t) * P.v + c.diffusivity * B(c, t) * P.lap +
                                B(c, t) * P.wgrad);
  return NatExactY(c, x, y, z, t) + q / c.alpha_track;
}
double NatInitial3D(const ProblemConfig& c, double x, double y, double z) {
  return NatExactY(c, x, y, z, 0.0);
}

// T11N neumann-conv-3d
double NeuKappa(const ProblemConfig& c, double t) {
  return c.eta * c.eta * B(c, t) * NatR(NeuKz(c), 0.0).v / c.diffusivity;
}
double NeuKappaDot(const ProblemConfig& c, double t) {
  return c.eta * c.eta * dB(c, t) * NatR(NeuKz(c), 0.0).v / c.diffusivity;
}
double NeuExactY(const ProblemConfig& c, double x, double y, double z, double t) {
  return A(t) * NatYShape(c, x, y, z).v + NeuKappa(c, t) * NeuGShape(c, x, y, z).v;
}
double NeuExactP(const ProblemConfig& c, double x, double y, double z, double t) {
  return c.alpha_u * B(c, t) * NeuPShape(c, x, y, z).v;
}
/** the exact control on Gamma_c = {z = 0}: -(eta/alpha_u) p(x, y, 0, t) */
double NeuExactU(const ProblemConfig& c, double x, double y, double, double t) {
  return -c.eta * NeuExactP(c, x, y, 0.0, t) / c.alpha_u;
}
double NeuSource3D(const ProblemConfig& c, double x, double y, double z, double t) {
  // f = y_t - nu Lap y + w.grad y  (the control enters through the boundary)
  const Prod Y = NatYShape(c, x, y, z), G = NeuGShape(c, x, y, z);
  const double k = NeuKappa(c, t);
  return dA(t) * Y.v + NeuKappaDot(c, t) * G.v - c.diffusivity * (A(t) * Y.lap + k * G.lap) +
         A(t) * Y.wgrad + k * G.wgrad;
}
double NeuTarget3D(const ProblemConfig& c, double x, double y, double z, double t) {
  const Prod P = NeuPShape(c, x, y, z);
  const double q = c.alpha_u * (dB(c, t) * P.v + c.diffusivity * B(c, t) * P.lap +
                                B(c, t) * P.wgrad);
  return NeuExactY(c, x, y, z, t) + q / c.alpha_track;
}
double NeuInitial3D(const ProblemConfig& c, double x, double y, double z) {
  return NeuExactY(c, x, y, z, 0.0);
}

struct Context3D {
  const ProblemConfig* config;
  double time;
  int field;   // 0 y, 1 p, 2 u
};
double ExactAt3D(double x, double y, double z, void* raw) {
  const Context3D* e = static_cast<const Context3D*>(raw);
  switch (e->field) {
    case 0: return ExactY(*e->config, x, y, z, e->time);
    case 1: return ExactP(*e->config, x, y, z, e->time);
    default: return ExactU(*e->config, x, y, z, e->time);
  }
}
// T11D dirichlet-conv-3d
double DirExactP(const ProblemConfig& c, double x, double y, double z, double t) {
  return DirX(DirC(c, 0), x).v * DirX(DirC(c, 1), y).v * DirQ3(c, z, t).v;
}
/** the exact control on Gamma_c: -(eta nu/alpha_u) X(x) X(y) dQs/dz(0, t) */
double DirExactUxy(const ProblemConfig& c, double x, double y, double t) {
  if (g_dir_h1)   // T11H1: u = S U(t)
    return DirX(DirC(c, 0), x).v * DirX(DirC(c, 1), y).v * DirH1Time(c, t).U;
  return -c.eta * c.diffusivity / c.alpha_u * DirX(DirC(c, 0), x).v * DirX(DirC(c, 1), y).v *
         DirQs3(c, 0.0, t).z;
}
/** T11H1: the exact surface gradient of u on Gamma_c = {z = 0}: U (X' Y, X Y', 0). */
void DirExactUGrad(double x, double y, double, void* raw, double* g) {
  const Context3D* e = static_cast<const Context3D*>(raw);
  const ProblemConfig& c = *e->config;
  const F1 X = DirX(DirC(c, 0), x), Y = DirX(DirC(c, 1), y);
  const double U = g_dir_h1 ? DirH1Time(c, e->time).U
                            : -c.eta * c.diffusivity / c.alpha_u * DirQs3(c, 0.0, e->time).z;
  g[0] = U * X.d * Y.v;
  g[1] = U * X.v * Y.d;
  g[2] = 0.0;
}
double DirExactU(const ProblemConfig& c, double x, double y, double, double t) {
  return DirExactUxy(c, x, y, t);
}
/** y = A S3(x)S3(y)S3(z) + eta u(x,y,t)(1 - z)^3 with value, w.grad, Laplacian, d/dt */
struct YParts { double v, wgrad, lap, dt; };
F1 Sin3F(double s) {   // sin^3(pi s)
  const double sn = std::sin(kPi * s), cs = std::cos(kPi * s);
  return {sn * sn * sn, 3.0 * kPi * sn * sn * cs, 3.0 * kPi * kPi * sn * (2.0 * cs * cs - sn * sn)};
}
/** T14 dirichlet-var-3d: the owned swirl wind w = (a s(x) s'(y) g(z), -a s'(x) s(y) g(z), vz),
 *  s = sin^2(pi .), g the T11D septic step on [1/2, 1], a = 1/2: div w = 0; w = (0, 0, vz) for
 *  z <= 1/2 (where the T11D adjoint is the exact homogeneous solution); tangential parts vanish on
 *  the lateral faces. */
void WindSwirl3D(const ProblemConfig& c, double x, double y, double z, double w[3]) {
  double g, g1, g2;
  DirStep3(z, g, g1, g2);
  const double sx = std::sin(kPi * x) * std::sin(kPi * x), sy = std::sin(kPi * y) * std::sin(kPi * y);
  const double dsx = kPi * std::sin(2.0 * kPi * x), dsy = kPi * std::sin(2.0 * kPi * y);
  w[0] = 0.5 * sx * dsy * g;
  w[1] = -0.5 * dsx * sy * g;
  w[2] = c.convection.vz;
}
struct YParts;
YParts DirYVar(const ProblemConfig& c, double x, double y, double z, double t);

YParts DirY(const ProblemConfig& c, double x, double y, double z, double t) {
  if (g_var == 4) return DirYVar(c, x, y, z, t);   // T14
  const Prod S = Product(c, Sin3F(x), Sin3F(y), Sin3F(z));
  const double wx = HasConvection(c) ? c.convection.vx : 0.0, wy = HasConvection(c) ? c.convection.vy : 0.0,
               wz = HasConvection(c) ? c.convection.vz : 0.0;
  const F1 X = DirX(DirC(c, 0), x), Y = DirX(DirC(c, 1), y);
  double k = -c.eta * c.eta * c.diffusivity / c.alpha_u * DirQs3(c, 0.0, t).z;   // eta u = k X Y
  double kd = c.eta * c.eta * c.diffusivity / c.alpha_u * DirQs3(c, 0.0, t).ztau;
  if (g_dir_h1) {   // T11H1: eta u = eta U(t) X Y
    const DirTime ht = DirH1Time(c, t);
    k = c.eta * ht.U;
    kd = c.eta * ht.Ut;
  }
  const double g = (1.0 - z) * (1.0 - z) * (1.0 - z), g1 = -3.0 * (1.0 - z) * (1.0 - z),
               g2 = 6.0 * (1.0 - z);
  YParts r;
  r.v = A(t) * S.v + k * X.v * Y.v * g;
  r.wgrad = A(t) * S.wgrad + k * (wx * X.d * Y.v * g + wy * X.v * Y.d * g + wz * X.v * Y.v * g1);
  r.lap = A(t) * S.lap + k * ((X.dd * Y.v + X.v * Y.dd) * g + X.v * Y.v * g2);
  r.dt = dA(t) * S.v + kd * X.v * Y.v * g;
  return r;
}
double DirExactY(const ProblemConfig& c, double x, double y, double z, double t) {
  return DirY(c, x, y, z, t).v;
}
double DirSource3D(const ProblemConfig& c, double x, double y, double z, double t) {
  const YParts Y = DirY(c, x, y, z, t);
  return Y.dt - c.diffusivity * Y.lap + Y.wgrad;
}
YParts DirYVar(const ProblemConfig& c, double x, double y, double z, double t) {
  const F1 sx = Sin3F(x), sy = Sin3F(y), sz = Sin3F(z);
  double w[3];
  EvaluateVelocity3D(c, x, y, z, w);
  const F1 X = DirX(DirC(c, 0), x), Y = DirX(DirC(c, 1), y);
  const double k = -c.eta * c.eta * c.diffusivity / c.alpha_u * DirQs3(c, 0.0, t).z;
  const double kd = c.eta * c.eta * c.diffusivity / c.alpha_u * DirQs3(c, 0.0, t).ztau;
  const double g = (1.0 - z) * (1.0 - z) * (1.0 - z), g1 = -3.0 * (1.0 - z) * (1.0 - z),
               g2 = 6.0 * (1.0 - z);
  YParts r;
  r.v = A(t) * sx.v * sy.v * sz.v + k * X.v * Y.v * g;
  r.wgrad = A(t) * (w[0] * sx.d * sy.v * sz.v + w[1] * sx.v * sy.d * sz.v + w[2] * sx.v * sy.v * sz.d) +
            k * (w[0] * X.d * Y.v * g + w[1] * X.v * Y.d * g + w[2] * X.v * Y.v * g1);
  r.lap = A(t) * (sx.dd * sy.v * sz.v + sx.v * sy.dd * sz.v + sx.v * sy.v * sz.dd) +
          k * ((X.dd * Y.v + X.v * Y.dd) * g + X.v * Y.v * g2);
  r.dt = dA(t) * sx.v * sy.v * sz.v + kd * X.v * Y.v * g;
  return r;
}

double DirTarget3D(const ProblemConfig& c, double x, double y, double z, double t) {
  if (g_var == 4) {   // T14: r = -p_t - nu Lap p - w(x).grad p (div w = 0); 0 for z <= 1/2
    const F1 X = DirX(DirC(c, 0), x), Y = DirX(DirC(c, 1), y);
    const ZQ Q = DirQ3(c, z, t);
    double w[3];
    EvaluateVelocity3D(c, x, y, z, w);
    const double r = -X.v * Y.v * Q.t -
                     c.diffusivity * (X.dd * Y.v * Q.v + X.v * Y.dd * Q.v + X.v * Y.v * Q.zz) -
                     (w[0] * X.d * Y.v * Q.v + w[1] * X.v * Y.d * Q.v + w[2] * X.v * Y.v * Q.z);
    return DirExactY(c, x, y, z, t) - r / c.alpha_track;
  }
  // r = -p_t - nu Lap p - w.grad p  (zero for z <= 1/2);  y_d = y - r/alpha_t
  const F1 X = DirX(DirC(c, 0), x), Y = DirX(DirC(c, 1), y);
  const ZQ Q = DirQ3(c, z, t);
  const double wx = HasConvection(c) ? c.convection.vx : 0.0, wy = HasConvection(c) ? c.convection.vy : 0.0,
               wz = HasConvection(c) ? c.convection.vz : 0.0;
  const double r = -X.v * Y.v * Q.t -
                   c.diffusivity * (X.dd * Y.v * Q.v + X.v * Y.dd * Q.v + X.v * Y.v * Q.zz) -
                   (wx * X.d * Y.v * Q.v + wy * X.v * Y.d * Q.v + wz * X.v * Y.v * Q.z);
  return DirExactY(c, x, y, z, t) - r / c.alpha_track;
}
double DirInitial3D(const ProblemConfig& c, double x, double y, double z) {
  return DirExactY(c, x, y, z, 0.0);
}
double DirExactAt3D(double x, double y, double z, void* raw) {   // T11D
  const Context3D* e = static_cast<const Context3D*>(raw);
  switch (e->field) {
    case 0: return DirExactY(*e->config, x, y, z, e->time);
    case 1: return DirExactP(*e->config, x, y, z, e->time);
    default: return DirExactU(*e->config, x, y, z, e->time);
  }
}
double NeuExactAt3D(double x, double y, double z, void* raw) {   // T11N
  const Context3D* e = static_cast<const Context3D*>(raw);
  switch (e->field) {
    case 0: return NeuExactY(*e->config, x, y, z, e->time);
    case 1: return NeuExactP(*e->config, x, y, z, e->time);
    default: return NeuExactU(*e->config, x, y, z, e->time);
  }
}
double NatExactAt3D(double x, double y, double z, void* raw) {   // T10
  const Context3D* e = static_cast<const Context3D*>(raw);
  switch (e->field) {
    case 0: return NatExactY(*e->config, x, y, z, e->time);
    case 1: return NatExactP(*e->config, x, y, z, e->time);
    default: return NatExactU(*e->config, x, y, z, e->time);
  }
}

// ===========================================================================
// T13: multi-face exact cases (t13_symbolic.py).  g_multi = 1 neumann-opp-3d,
// 2 neumann-adj-3d, 3 dirichlet-opp-3d (L2, H1-space, H1-time, combined).
//
// neumann-opp-3d   Gamma_c = zmin u zmax.  p = alpha_u B X(x) X(y) R_kz(z) (the T11N p:
//   nu dn p + (w.n) p = 0 on all six faces, any constant wind); u = -(eta/alpha_u) p on
//   both faces; y = A cos cos cos + kappa0 X X H0(z) + kappa1 X X H1(z),
//   H0 = z - z^2/2 (H0'(0) = 1, H0'(1) = 0), H1 = z^2/2 (H1'(0) = 0, H1'(1) = 1),
//   kappa0 = eta^2 B R(0)/nu, kappa1 = -eta^2 B R(1)/nu.
// neumann-adj-3d   Gamma_c = zmin u xmin, wind with w_x = w_z (refused otherwise: smooth
//   compatibility at the internal edge x = z = 0 needs p_x = p_z there, and the adjoint
//   Robin conditions give p_x - p_z = -(w_x - w_z) p/nu).
//   F_k(s) = exp(-k s)(1 - s)^2 (1 + 2 s): F'(0) = -k F(0) (Robin, controlled face),
//   F(1) = F'(1) = 0 (uncontrolled faces x = 1, z = 1: p = dn p = 0, edge compatibility);
//   p = alpha_u B F_k(x) X(y) F_k(z), k = w_x/nu;  u = -(eta/alpha_u) p on both faces;
//   y = A cos cos cos + kappa X(y) G(x,z), kappa = eta^2 B/nu,
//   G = F(x) H0(z) + H0(x) F(z) - F'(0) H0(x) H0(z):  G_z(x,0) = F(x), G_x(0,z) = F(z),
//   G_z(x,1) = G_x(1,z) = 0.
// dirichlet-opp-3d Gamma_c = zmin u zmax, omega = (0,1)^2 x (1/4,3/4) (n multiple of 4).
//   p = X(x) X(y) Q,  Q = (1 - psi) Qs0 + psi Qs1, psi the septic step on [1/4, 3/4];
//   Qs0 = the T11D image solution (sources +-1.2, Qs0(0) = 0), Qs1 = exp(-a z - c tau)
//   [Phi(z + 0.2) - Phi(z - 2.2)] (Qs1(1) = 0).  dz Qs1(1) = -exp(-a) dz Qs0(0), hence on
//   zmax the stationarity gives u1 = exp(-a) u0 (a = w_z/(2 nu)) for L2 AND for H1-space /
//   H1-time (both faces are disjoint, S = X X is a facewise surface eigenfunction, the
//   temporal problem is linear): u0 = S U(t) (DirH1Time), u1 = exp(-a) S U(t);
//   y = A S3 S3 S3 + eta S(x,y) [U(t)(1 - z)^3 + exp(-a) U(t) z^3].
int g_multi = 0;
constexpr double kOppZa = 0.25, kOppZb = 0.75, kOppSrc1 = -0.2, kOppImg1 = 2.2;
double NeuH1v(double z) { return 0.5 * z * z; }
// --- neumann-opp-3d
double NopKappa0(const ProblemConfig& c, double t) { return c.eta * c.eta * B(c, t) * NatR(NeuKz(c), 0.0).v / c.diffusivity; }
double NopKappa1(const ProblemConfig& c, double t) { return -c.eta * c.eta * B(c, t) * NatR(NeuKz(c), 1.0).v / c.diffusivity; }
double NopKappa0Dot(const ProblemConfig& c, double t) { return c.eta * c.eta * dB(c, t) * NatR(NeuKz(c), 0.0).v / c.diffusivity; }
double NopKappa1Dot(const ProblemConfig& c, double t) { return -c.eta * c.eta * dB(c, t) * NatR(NeuKz(c), 1.0).v / c.diffusivity; }
Prod NopG1Shape(const ProblemConfig& c, double x, double y, double z) {   // X(x) X(y) H1(z)
  return Product(c, NeuX(x), NeuX(y), F1{NeuH1v(z), z, 1.0});
}
// --- neumann-adj-3d
double NadK(const ProblemConfig& c) { return HasConvection(c) ? c.convection.vx / c.diffusivity : 0.0; }
F1 NadF(double k, double s) {
  const double e = std::exp(-k * s);
  const double q = (1.0 - s) * (1.0 - s) * (1.0 + 2.0 * s), q1 = -6.0 * s + 6.0 * s * s, q2 = -6.0 + 12.0 * s;
  return {e * q, e * (q1 - k * q), e * (q2 - 2.0 * k * q1 + k * k * q)};
}
Prod NadPShape(const ProblemConfig& c, double x, double y, double z) {
  return Product(c, NadF(NadK(c), x), NeuX(y), NadF(NadK(c), z));
}
Prod NadTShape(const ProblemConfig& c, double x, double y, double z) {   // X(y) G(x,z)
  const double k = NadK(c), fp0 = -k;
  const F1 Fx = NadF(k, x), Fz = NadF(k, z), Hx = NeuH(x), Hz = NeuH(z), Y = NeuX(y);
  const double G = Fx.v * Hz.v + Hx.v * Fz.v - fp0 * Hx.v * Hz.v;
  const double Gx = Fx.d * Hz.v + Hx.d * Fz.v - fp0 * Hx.d * Hz.v;
  const double Gz = Fx.v * Hz.d + Hx.v * Fz.d - fp0 * Hx.v * Hz.d;
  const double Gxx = Fx.dd * Hz.v + Hx.dd * Fz.v - fp0 * Hx.dd * Hz.v;
  const double Gzz = Fx.v * Hz.dd + Hx.v * Fz.dd - fp0 * Hx.v * Hz.dd;
  const double w[3] = {HasConvection(c) ? c.convection.vx : 0.0, HasConvection(c) ? c.convection.vy : 0.0,
                       HasConvection(c) ? c.convection.vz : 0.0};
  return {Y.v * G, w[0] * Y.v * Gx + w[1] * Y.d * G + w[2] * Y.v * Gz, Y.dd * G + Y.v * (Gxx + Gzz)};
}
double NadKappa(const ProblemConfig& c, double t) { return c.eta * c.eta * B(c, t) / c.diffusivity; }
double NadKappaDot(const ProblemConfig& c, double t) { return c.eta * c.eta * dB(c, t) / c.diffusivity; }
// --- shared Neumann multi-face y, p, u, f, y_d
Prod MnPShape(const ProblemConfig& c, double x, double y, double z) {
  return g_multi == 2 ? NadPShape(c, x, y, z) : NeuPShape(c, x, y, z);
}
struct MnY { double v, dt, lap, wgrad; };
MnY MnYParts(const ProblemConfig& c, double x, double y, double z, double t) {
  const Prod Y = NatYShape(c, x, y, z);
  MnY r{A(t) * Y.v, dA(t) * Y.v, A(t) * Y.lap, A(t) * Y.wgrad};
  if (g_multi == 2) {
    const Prod T = NadTShape(c, x, y, z);
    const double k = NadKappa(c, t);
    r.v += k * T.v; r.dt += NadKappaDot(c, t) * T.v; r.lap += k * T.lap; r.wgrad += k * T.wgrad;
  } else {
    const Prod G0 = NeuGShape(c, x, y, z), G1 = NopG1Shape(c, x, y, z);
    const double k0 = NopKappa0(c, t), k1 = NopKappa1(c, t);
    r.v += k0 * G0.v + k1 * G1.v;
    r.dt += NopKappa0Dot(c, t) * G0.v + NopKappa1Dot(c, t) * G1.v;
    r.lap += k0 * G0.lap + k1 * G1.lap;
    r.wgrad += k0 * G0.wgrad + k1 * G1.wgrad;
  }
  return r;
}
double MnExactY(const ProblemConfig& c, double x, double y, double z, double t) { return MnYParts(c, x, y, z, t).v; }
double MnExactP(const ProblemConfig& c, double x, double y, double z, double t) {
  return c.alpha_u * B(c, t) * MnPShape(c, x, y, z).v;
}
/** the exact control at a point of Gamma_c (either face): -(eta/alpha_u) p */
double MnExactU(const ProblemConfig& c, double x, double y, double z, double t) {
  return -c.eta * MnExactP(c, x, y, z, t) / c.alpha_u;
}
double MnSource3D(const ProblemConfig& c, double x, double y, double z, double t) {
  const MnY Y = MnYParts(c, x, y, z, t);
  return Y.dt - c.diffusivity * Y.lap + Y.wgrad;
}
double MnTarget3D(const ProblemConfig& c, double x, double y, double z, double t) {
  const Prod P = MnPShape(c, x, y, z);
  const double q = c.alpha_u * (dB(c, t) * P.v + c.diffusivity * B(c, t) * P.lap + B(c, t) * P.wgrad);
  return MnExactY(c, x, y, z, t) + q / c.alpha_track;
}
double MnInitial3D(const ProblemConfig& c, double x, double y, double z) { return MnExactY(c, x, y, z, 0.0); }
double MnExactAt3D(double x, double y, double z, void* raw) {
  const Context3D* e = static_cast<const Context3D*>(raw);
  switch (e->field) {
    case 0: return MnExactY(*e->config, x, y, z, e->time);
    case 1: return MnExactP(*e->config, x, y, z, e->time);
    default: return MnExactU(*e->config, x, y, z, e->time);
  }
}
// --- dirichlet-opp-3d
/** the T11D image solution with the sources at z = src and z = img: exp(-a z - cc tau) [Phi(z - src) - Phi(z - img)] */
Heat3 OppQs(const ProblemConfig& c, double z, double t, double src, double img) {
  Heat3 h;
  const double tau = c.final_time() - t;
  if (tau <= 1e-14) return h;
  const double nu = c.diffusivity;
  const double wz = HasConvection(c) ? c.convection.vz : 0.0;
  const double a = wz / (2.0 * nu), cc = DirCC(c);
  auto phi = [&](double xi, double& p0, double& p1, double& p2, double& p3) {
    p0 = std::exp(-xi * xi / (4.0 * nu * tau)) / std::sqrt(4.0 * kPi * nu * tau);
    p1 = -xi / (2.0 * nu * tau) * p0;
    p2 = (xi * xi / (4.0 * nu * nu * tau * tau) - 1.0 / (2.0 * nu * tau)) * p0;
    p3 = (-xi * xi * xi / (8.0 * nu * nu * nu * tau * tau * tau) + 3.0 * xi / (4.0 * nu * nu * tau * tau)) * p0;
  };
  double a0, a1, a2, a3, b0, b1, b2, b3;
  phi(z - src, a0, a1, a2, a3);
  phi(z - img, b0, b1, b2, b3);
  const double V = a0 - b0, Vz = a1 - b1, Vzz = a2 - b2, Vzzz = a3 - b3;
  const double E = std::exp(-a * z - cc * tau);
  h.v = E * V;
  h.z = E * (Vz - a * V);
  h.zz = E * (Vzz - 2.0 * a * Vz + a * a * V);
  h.tau = E * (nu * Vzz - cc * V);
  h.ztau = E * (-cc * (Vz - a * V) + nu * Vzzz - a * nu * Vzz);
  return h;
}
void OppStep(double z, double& f, double& f1, double& f2) {   // septic step 0 -> 1 on [1/4, 3/4]
  const double L = kOppZb - kOppZa, s = (z - kOppZa) / L;
  if (s <= 0.0) { f = f1 = f2 = 0.0; return; }
  if (s >= 1.0) { f = 1.0; f1 = f2 = 0.0; return; }
  const double s2 = s * s, s3 = s2 * s, s4 = s3 * s;
  f = s4 * (35.0 - 84.0 * s + 70.0 * s2 - 20.0 * s3);
  f1 = 140.0 * s3 * (1.0 - s) * (1.0 - s) * (1.0 - s) / L;
  f2 = 420.0 * s2 * (1.0 - s) * (1.0 - s) * (1.0 - 2.0 * s) / (L * L);
}
ZQ OppQ(const ProblemConfig& c, double z, double t) {
  const Heat3 q0 = OppQs(c, z, t, kDir3Z1, -kDir3Z1), q1 = OppQs(c, z, t, kOppSrc1, kOppImg1);
  double f, f1, f2;
  OppStep(z, f, f1, f2);
  return {(1.0 - f) * q0.v + f * q1.v,
          -f1 * q0.v + (1.0 - f) * q0.z + f1 * q1.v + f * q1.z,
          -f2 * q0.v - 2.0 * f1 * q0.z + (1.0 - f) * q0.zz + f2 * q1.v + 2.0 * f1 * q1.z + f * q1.zz,
          -((1.0 - f) * q0.tau + f * q1.tau)};
}
double OppE(const ProblemConfig& c) {   // u_zmax / u_zmin = exp(-a), a = w_z/(2 nu)
  return std::exp(-(HasConvection(c) ? c.convection.vz : 0.0) / (2.0 * c.diffusivity));
}
double DopExactP(const ProblemConfig& c, double x, double y, double z, double t) {
  return DirX(DirC(c, 0), x).v * DirX(DirC(c, 1), y).v * OppQ(c, z, t).v;
}
double DopExactU(const ProblemConfig& c, double x, double y, double z, double t) {
  const double S = DirX(DirC(c, 0), x).v * DirX(DirC(c, 1), y).v;
  return S * DirH1Time(c, t).U * (z > 0.5 ? OppE(c) : 1.0);
}
void DopExactUGrad(double x, double y, double z, void* raw, double* g) {
  const Context3D* e = static_cast<const Context3D*>(raw);
  const ProblemConfig& c = *e->config;
  const F1 X = DirX(DirC(c, 0), x), Y = DirX(DirC(c, 1), y);
  const double U = DirH1Time(c, e->time).U * (z > 0.5 ? OppE(c) : 1.0);
  g[0] = U * X.d * Y.v;
  g[1] = U * X.v * Y.d;
  g[2] = 0.0;
}
YParts DopY(const ProblemConfig& c, double x, double y, double z, double t) {
  const Prod S3 = Product(c, Sin3F(x), Sin3F(y), Sin3F(z));
  const double wx = HasConvection(c) ? c.convection.vx : 0.0, wy = HasConvection(c) ? c.convection.vy : 0.0,
               wz = HasConvection(c) ? c.convection.vz : 0.0;
  const F1 X = DirX(DirC(c, 0), x), Y = DirX(DirC(c, 1), y);
  const DirTime ht = DirH1Time(c, t);
  const double e1 = OppE(c);
  const double k = c.eta * ht.U, kd = c.eta * ht.Ut;
  // h(z) = (1 - z)^3 + e1 z^3
  const double g = (1.0 - z) * (1.0 - z) * (1.0 - z) + e1 * z * z * z,
               g1 = -3.0 * (1.0 - z) * (1.0 - z) + 3.0 * e1 * z * z, g2 = 6.0 * (1.0 - z) + 6.0 * e1 * z;
  YParts r;
  r.v = A(t) * S3.v + k * X.v * Y.v * g;
  r.wgrad = A(t) * S3.wgrad + k * (wx * X.d * Y.v * g + wy * X.v * Y.d * g + wz * X.v * Y.v * g1);
  r.lap = A(t) * S3.lap + k * ((X.dd * Y.v + X.v * Y.dd) * g + X.v * Y.v * g2);
  r.dt = dA(t) * S3.v + kd * X.v * Y.v * g;
  return r;
}
double DopExactY(const ProblemConfig& c, double x, double y, double z, double t) { return DopY(c, x, y, z, t).v; }
double DopSource3D(const ProblemConfig& c, double x, double y, double z, double t) {
  const YParts Y = DopY(c, x, y, z, t);
  return Y.dt - c.diffusivity * Y.lap + Y.wgrad;
}
double DopTarget3D(const ProblemConfig& c, double x, double y, double z, double t) {
  const F1 X = DirX(DirC(c, 0), x), Y = DirX(DirC(c, 1), y);
  const ZQ Q = OppQ(c, z, t);
  const double wx = HasConvection(c) ? c.convection.vx : 0.0, wy = HasConvection(c) ? c.convection.vy : 0.0,
               wz = HasConvection(c) ? c.convection.vz : 0.0;
  const double r = -X.v * Y.v * Q.t - c.diffusivity * (X.dd * Y.v * Q.v + X.v * Y.dd * Q.v + X.v * Y.v * Q.zz) -
                   (wx * X.d * Y.v * Q.v + wy * X.v * Y.d * Q.v + wz * X.v * Y.v * Q.z);
  return DopExactY(c, x, y, z, t) - r / c.alpha_track;
}
double DopInitial3D(const ProblemConfig& c, double x, double y, double z) { return DopExactY(c, x, y, z, 0.0); }
double DopExactAt3D(double x, double y, double z, void* raw) {
  const Context3D* e = static_cast<const Context3D*>(raw);
  switch (e->field) {
    case 0: return DopExactY(*e->config, x, y, z, e->time);
    case 1: return DopExactP(*e->config, x, y, z, e->time);
    default: return DopExactU(*e->config, x, y, z, e->time);
  }
}
/** T13: controlled-node geometry of the face set (Neumann: on the union; Dirichlet: on the
 *  union and on no uncontrolled face) */
bool OnControlSet3D(const ProblemConfig& c, double x, double y, double z, bool dirichlet) {
  const double p[3] = {x, y, z};
  const double lo[3] = {c.domain.xmin, c.domain.ymin, c.domain.zmin}, hi[3] = {c.domain.xmax, c.domain.ymax, c.domain.zmax};
  bool on_c = false, on_u = false;
  for (unsigned id = 1; id <= 6u; ++id) {
    BoxFace3D f;
    LookupBoxFace3D(id, f);
    if (std::fabs(p[f.axis] - (f.at_max ? hi[f.axis] : lo[f.axis])) > 1e-12) continue;
    (IsControlFace3D(c, id) ? on_c : on_u) = true;
  }
  return on_c && (!dirichlet || !on_u);
}
long long ExpectedControlNodes3D(const ProblemConfig& c, bool dirichlet) {
  const unsigned n[3] = {2u * c.mesh.nx, 2u * c.mesh.ny, 2u * c.mesh.nz};
  long long count = 0;
  for (unsigned i = 0; i <= n[0]; ++i)
    for (unsigned j = 0; j <= n[1]; ++j)
      for (unsigned k = 0; k <= n[2]; ++k) {
        bool on_c = false, on_u = false;
        const unsigned idx[3] = {i, j, k};
        for (unsigned id = 1; id <= 6u; ++id) {
          BoxFace3D f;
          LookupBoxFace3D(id, f);
          if (idx[f.axis] != (f.at_max ? n[f.axis] : 0u)) continue;
          (IsControlFace3D(c, id) ? on_c : on_u) = true;
        }
        if (on_c && (!dirichlet || !on_u)) ++count;
      }
  return count;
}

// ===========================================================================
// T14: potential-wind exact cases (t14_symbolic.py).  w = grad Phi (quadratic preset), nu default 1/2.
//   P = exp(-Phi/nu) Q  =>  nu dn P + (w.n) P = exp(-Phi/nu) nu dn Q, so dn Q = 0 gives the exact
//   variable Robin adjoint condition (any div w).
//   natural-var-3d:     Q = (2 + cos pi x)(2 + cos pi y)(2 + cos pi z); y = A cos cos cos;
//                       distributed control u = -(eta/alpha_u) p everywhere.
//   neumann-var-3d:     Q = X(x) X(y) (2 + cos pi z), X = 1 - cos 2 pi s; Gamma_c = zmin;
//   neumann-var-opp-3d: the same on zmin u zmax;  u = -(eta/alpha_u) p on Gamma_c;
//     y = A cos cos cos + kappa0 G0(x,y) H0(z) [+ kappa1 G1(x,y) H1(z)],
//     G_k = exp(-Phi(x,y,z_k)/nu) X(x) X(y) (2 + cos pi z_k), kappa0 = eta^2 B/nu, kappa1 = -eta^2 B/nu.
//   p = alpha_u B(t) P;  y_d = y + (p_t + nu Lap p + w.grad p + (div w) p)/alpha_t (the full adjoint).
struct PotP { double v, gx, gy, gz, lap; };
PotP PotShape(const ProblemConfig& c, double x, double y, double z) {
  const double nu = c.diffusivity;
  const F1 QX = g_var == 1 ? NatR(0.0, x) : NeuX(x), QY = g_var == 1 ? NatR(0.0, y) : NeuX(y),
           QZ = NatR(0.0, z);
  double w[3];
  EvaluateVelocity3D(c, x, y, z, w);
  const double div = EvaluateDivergence3D(c, x, y, z);
  const double E = std::exp(-PotentialPhi3D(c, x, y, z) / nu);
  const double Q = QX.v * QY.v * QZ.v, Qx = QX.d * QY.v * QZ.v, Qy = QX.v * QY.d * QZ.v,
               Qz = QX.v * QY.v * QZ.d;
  const double lapQ = QX.dd * QY.v * QZ.v + QX.v * QY.dd * QZ.v + QX.v * QY.v * QZ.dd;
  const double wgQ = w[0] * Qx + w[1] * Qy + w[2] * Qz, ww = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
  return {E * Q, E * (Qx - Q * w[0] / nu), E * (Qy - Q * w[1] / nu), E * (Qz - Q * w[2] / nu),
          E * (lapQ - 2.0 * wgQ / nu - Q * div / nu + Q * ww / (nu * nu))};
}
double VarExactP(const ProblemConfig& c, double x, double y, double z, double t) {
  return c.alpha_u * B(c, t) * PotShape(c, x, y, z).v;
}
double VarExactU(const ProblemConfig& c, double x, double y, double z, double t) {
  return -c.eta * VarExactP(c, x, y, z, t) / c.alpha_u;
}
/** G_k(x, y) on the face z = zk and its x, y derivatives / Laplacian (Phi quadratic: Hessian H). */
struct G2 { double v, gx, gy, lap; };
G2 VarG(const ProblemConfig& c, double x, double y, double zk) {
  const double nu = c.diffusivity;
  const F1 X = NeuX(x), Y = NeuX(y);
  const double zc = NatR(0.0, zk).v;
  double w[3];
  EvaluateVelocity3D(c, x, y, zk, w);
  const double E = std::exp(-PotentialPhi3D(c, x, y, zk) / nu);
  const double R = X.v * Y.v * zc, Rx = X.d * Y.v * zc, Ry = X.v * Y.d * zc,
               lapR = (X.dd * Y.v + X.v * Y.dd) * zc;
  const double* H = c.convection.hessian;
  return {E * R, E * (Rx - R * w[0] / nu), E * (Ry - R * w[1] / nu),
          E * (lapR - 2.0 * (w[0] * Rx + w[1] * Ry) / nu - R * (H[0] + H[4]) / nu +
               R * (w[0] * w[0] + w[1] * w[1]) / (nu * nu))};
}
YParts VarY(const ProblemConfig& c, double x, double y, double z, double t) {
  const F1 cx = NatC(x), cy = NatC(y), cz = NatC(z);
  double w[3];
  EvaluateVelocity3D(c, x, y, z, w);
  const double Yc = cx.v * cy.v * cz.v;
  YParts r;
  r.v = A(t) * Yc;
  r.wgrad = A(t) * (w[0] * cx.d * cy.v * cz.v + w[1] * cx.v * cy.d * cz.v + w[2] * cx.v * cy.v * cz.d);
  r.lap = A(t) * (cx.dd * cy.v * cz.v + cx.v * cy.dd * cz.v + cx.v * cy.v * cz.dd);
  r.dt = dA(t) * Yc;
  if (g_var == 2 || g_var == 3) {
    const double kap = c.eta * c.eta / c.diffusivity;
    for (int face = 0; face < (g_var == 3 ? 2 : 1); ++face) {
      const G2 G = VarG(c, x, y, face == 0 ? 0.0 : 1.0);
      const F1 H = face == 0 ? NeuH(z) : F1{0.5 * z * z, z, 1.0};
      const double k = (face == 0 ? 1.0 : -1.0) * kap * B(c, t),
                   kd = (face == 0 ? 1.0 : -1.0) * kap * dB(c, t);
      r.v += k * G.v * H.v;
      r.wgrad += k * (w[0] * G.gx * H.v + w[1] * G.gy * H.v + w[2] * G.v * H.d);
      r.lap += k * (G.lap * H.v + G.v * H.dd);
      r.dt += kd * G.v * H.v;
    }
  }
  return r;
}
double VarExactY(const ProblemConfig& c, double x, double y, double z, double t) {
  return VarY(c, x, y, z, t).v;
}
double VarSource3D(const ProblemConfig& c, double x, double y, double z, double t) {
  const YParts Y = VarY(c, x, y, z, t);
  double f = Y.dt - c.diffusivity * Y.lap + Y.wgrad;
  if (g_var == 1) f -= c.eta * VarExactU(c, x, y, z, t);   // distributed control
  return f;
}
double VarTarget3D(const ProblemConfig& c, double x, double y, double z, double t) {
  const PotP P = PotShape(c, x, y, z);
  double w[3];
  EvaluateVelocity3D(c, x, y, z, w);
  const double div = EvaluateDivergence3D(c, x, y, z);
  const double q = c.alpha_u * (dB(c, t) * P.v + c.diffusivity * B(c, t) * P.lap +
                                B(c, t) * (w[0] * P.gx + w[1] * P.gy + w[2] * P.gz) + B(c, t) * div * P.v);
  return VarExactY(c, x, y, z, t) + q / c.alpha_track;
}
double VarInitial3D(const ProblemConfig& c, double x, double y, double z) { return VarExactY(c, x, y, z, 0.0); }
double VarExactAt3D(double x, double y, double z, void* raw) {
  const Context3D* e = static_cast<const Context3D*>(raw);
  switch (e->field) {
    case 0: return VarExactY(*e->config, x, y, z, e->time);
    case 1: return VarExactP(*e->config, x, y, z, e->time);
    default: return VarExactU(*e->config, x, y, z, e->time);
  }
}

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

}  // namespace

void ApplyExactOptimalControl3D(ProblemConfig& config, const std::string& name,
                                double alpha_u, double final_time, double nu) {
  if (name != "conv-sin-3d" && name != "conv-sin3-3d" && name != "natural-conv-3d" &&
      name != "neumann-conv-3d" && name != "dirichlet-conv-3d" && name != "dirichlet-h1-3d" &&
      name != "neumann-opp-3d" && name != "neumann-adj-3d" && name != "dirichlet-opp-3d" &&   // T13
      name != "natural-var-3d" && name != "neumann-var-3d" && name != "neumann-var-opp-3d" &&
      name != "dirichlet-var-3d")   // T14
    throw std::invalid_argument(
        "the 3D exact OC cases are conv-sin-3d, conv-sin3-3d, natural-conv-3d, neumann-conv-3d,"
        " dirichlet-conv-3d, dirichlet-h1-3d, neumann-opp-3d, neumann-adj-3d, dirichlet-opp-3d,"
        " natural-var-3d, neumann-var-3d, neumann-var-opp-3d and dirichlet-var-3d");
  g_var = name == "natural-var-3d" ? 1 : name == "neumann-var-3d" ? 2 : name == "neumann-var-opp-3d" ? 3
        : name == "dirichlet-var-3d" ? 4 : 0;
  // T14: a 3D wind field is accepted only where the exact data read it; never silently ignored
  if (config.convection.field3d && name != "conv-sin3-3d" && !(g_var >= 1 && g_var <= 3))
    throw std::invalid_argument(
        name + (g_var == 4 ? " owns its (swirl) wind: give --convection on and --vz only, no --wind"
                           : " uses a constant wind; --wind is accepted by conv-sin3-3d, natural-var-3d,"
                             " neumann-var-3d and neumann-var-opp-3d"));
  if (g_var >= 1 && g_var <= 3 && !(config.convection.field3d && config.convection.potential))
    throw std::invalid_argument(
        name + " needs a potential wind: --convection on --wind harmonic-3d | affine-div-3d");
  if (g_var == 4) {
    if (!HasConvection(config) || config.convection.vx != 0.0 || config.convection.vy != 0.0)
      throw std::invalid_argument(
          "dirichlet-var-3d needs --convection on with --vx 0 --vy 0 (--vz = the normal part of the swirl)");
    config.convection.field3d = WindSwirl3D;
    config.convection.divergence3d = nullptr;   // div w = 0
    config.convection.field3d_reads_vz = true;
    config.convection.potential = false;
    config.convection.field_name = "swirl-3d";
  }
  g_multi = name == "neumann-opp-3d" ? 1 : name == "neumann-adj-3d" ? 2 : name == "dirichlet-opp-3d" ? 3 : 0;
  g_cubic = name == "conv-sin3-3d";
  g_natural = name == "natural-conv-3d" || g_var == 1;
  g_neumann = name == "neumann-conv-3d" || g_multi == 1 || g_multi == 2 || g_var == 2 || g_var == 3;
  g_dir_h1 = name == "dirichlet-h1-3d" || g_multi == 3;   // T11H1 (T13: dirichlet-opp-3d, L2 and H1)
  g_dirichlet = name == "dirichlet-conv-3d" || g_dir_h1 || g_var == 4;
  if ((name == "dirichlet-conv-3d" || g_var == 4) &&
      (config.control_h1_length != 0.0 || config.control_h1_time_weight != 0.0))
    throw std::invalid_argument(
        name + " is the L2 benchmark; use --exact-oc dirichlet-h1-3d for H1-space / H1-time");
  config.dimension = 3u;
  config.domain = ProblemConfig::Domain();   // unit cube
  config.state_bc = (g_natural || g_neumann)
                        ? ProblemConfig::StateBoundaryCondition::NATURAL
                        : ProblemConfig::StateBoundaryCondition::HOMOGENEOUS_DIRICHLET;
  config.control.type = g_neumann ? ProblemConfig::ControlType::NEUMANN
                       : g_dirichlet ? ProblemConfig::ControlType::DIRICHLET
                                     : ProblemConfig::ControlType::DISTRIBUTED;
  if (g_neumann) {   // T11N: Gamma_c = {z = 0}, FEMuS face 1
    config.neumann_control_face = 1u;
    config.neumann_control_boundary = "zmin";
  }
  config.control.region = {0.0, 1.0, 0.0, 1.0};
  config.tracking_box = {0.0, 1.0, 0.0, 1.0};
  if (g_dirichlet) {
    // T11D: Gamma_c = {z = 0} (FEMuS face 1); homogeneous Dirichlet state on
    // the other faces; tracking omega = (0,1)^2 x (1/2,1), away from Gamma_c
    // (the KKT tracks free DOFs only).  Box z = storage axis 2.
    config.state_bc = ProblemConfig::StateBoundaryCondition::HOMOGENEOUS_DIRICHLET;
    config.neumann_control_face = 1u;
    config.neumann_control_boundary = "zmin";
    config.tracking_box = {0.0, 1.0, 0.5, 1.0};
  }
  if (g_multi) {   // T13: the face set of the case
    config.control_face_mask = g_multi == 2 ? ((1u << 1) | (1u << 5)) : ((1u << 1) | (1u << 6));
    config.neumann_control_face = 1u;
    config.neumann_control_boundary = g_multi == 2 ? "zmin,xmin" : "zmin,zmax";
    if (g_multi == 3) {
      config.tracking_box = {0.0, 1.0, kOppZa, kOppZb};   // omega = (0,1)^2 x (1/4,3/4)
      // omega must be a union of whole elements: z = 1/4, 3/4 element faces (n_z = 2 puts them on
      // the element centroids, i.e. an EMPTY tracking region; other n_z not multiple of 4 cut elements)
      if (config.mesh.nz % 4u != 0u)
        throw std::invalid_argument(
            "dirichlet-opp-3d needs --nz a multiple of 4 (omega = (0,1)^2 x (1/4,3/4) must be resolved by"
            " whole elements)");
    }
    if (g_multi == 2 && HasConvection(config) && config.convection.vx != config.convection.vz)
      throw std::invalid_argument(
          "neumann-adj-3d needs w_x = w_z (--vx = --vz): smooth compatibility at the internal edge"
          " x = z = 0 requires p_x = p_z there, and the adjoint Robin conditions give"
          " p_x - p_z = -(w_x - w_z) p/nu");
  }
  if (g_var == 3) {   // T14: neumann-var-opp-3d on zmin u zmax
    config.control_face_mask = (1u << 1) | (1u << 6);
    config.neumann_control_face = 1u;
    config.neumann_control_boundary = "zmin,zmax";
  }
  config.additional_control_regions.clear();
  config.additional_tracking_regions.clear();
  config.source_regions.clear();
  config.alpha_track = 1.0;
  config.eta = 1.0;
  // T14: the potential-wind cases default to nu = 1/2 (exp(-Phi/nu) within a factor ~90 on the cube)
  config.diffusivity = nu > 0.0 ? nu : (g_var >= 1 && g_var <= 3 ? 0.5 : 0.25);
  config.alpha_u = alpha_u > 0.0 ? alpha_u : 1.0e-2;
  const double T = final_time > 0.0 ? final_time : 1.0;
  config.time.dt = T / config.time.steps;
  if (g_dir_h1) {
    // T11H1: S = X X must be a surface-Laplacian eigenfunction when ell > 0, i.e. X = sin
    // (no tangential wind); gamma_u > 0 needs cc > k for the closed form (DirH1Time).
    if (config.control_h1_length > 0.0 && HasConvection(config) &&
        (config.convection.vx != 0.0 || config.convection.vy != 0.0))
      throw std::invalid_argument(
          std::string(g_multi ? "dirichlet-opp-3d" : "dirichlet-h1-3d") +   // T13: name the case
          " with ell > 0 needs a wind normal to Gamma_c (--vx 0 --vy 0): the control"
          " shape must be an eigenfunction of the surface Laplacian");
    if (config.control_h1_time_weight > 0.0 &&
        !(DirCC(config) > std::sqrt(DirH1A(config) / config.control_h1_time_weight)))
      throw std::invalid_argument(
          std::string(g_multi ? "dirichlet-opp-3d" : "dirichlet-h1-3d") +
          ": gamma_u too small for the closed form (needs cc > sqrt(alpha_u (1 + 2 pi^2"
          " ell^2) / gamma_u)); increase --control-h1-time-weight");
  }
  if (g_var >= 1 && g_var <= 3) {   // T14
    config.data_overrides.source3d = VarSource3D;
    config.data_overrides.target3d = VarTarget3D;
    config.data_overrides.initial3d = VarInitial3D;
    return;
  }
  if (g_multi) {   // T13
    config.data_overrides.source3d = g_multi == 3 ? DopSource3D : MnSource3D;
    config.data_overrides.target3d = g_multi == 3 ? DopTarget3D : MnTarget3D;
    config.data_overrides.initial3d = g_multi == 3 ? DopInitial3D : MnInitial3D;
    return;
  }
  config.data_overrides.source3d =
      g_dirichlet ? DirSource3D : (g_neumann ? NeuSource3D : (g_natural ? NatSource3D : Source3D));
  config.data_overrides.target3d =
      g_dirichlet ? DirTarget3D : (g_neumann ? NeuTarget3D : (g_natural ? NatTarget3D : Target3D));
  config.data_overrides.initial3d =
      g_dirichlet ? DirInitial3D : (g_neumann ? NeuInitial3D : (g_natural ? NatInitial3D : Initial3D));
}

int RunExactOptimalControl3D(const ProblemConfig& config,
                             const ApplicationOptions& options,
                             MultiLevelSolution& solution, Mesh* mesh,
                             unsigned solution_type) {
  int ranks = 0;
  MPI_Comm_size(PETSC_COMM_WORLD, &ranks);
  if (ranks != 1)
    throw std::runtime_error("--exact-oc conv-sin-3d runs serially (use --solve-only for MPI)");
  const double t0 = MPI_Wtime();
  SpatialOperators spatial(config, mesh, solution_type, solution);
  SpaceTimeOperators spacetime(config, spatial);
  const PetscInt local = spatial.layout.local_size;
  const unsigned steps = config.time.steps;
  const double dt = config.time.dt, T = config.final_time();
  const bool cn = IsCrankNicolson(config);
  const double h = 1.0 / config.mesh.nx;
  const double speed = std::sqrt(config.convection.vx * config.convection.vx +
                                 config.convection.vy * config.convection.vy +
                                 config.convection.vz * config.convection.vz);
  std::printf("EX21_EXACT3D_CONFIG case=%s nx=%u ny=%u nz=%u steps=%u T=%.6g dt=%.10e"
              " scheme=%s nu=%.6g convection=%s vx=%.6g vy=%.6g vz=%.6g alpha_u=%.6e"
              " h=%.6e Pe_cell=%.6e dofs=%d spacetime_unknowns=%lld pc=%d\n",
              g_var == 1 ? "natural-var-3d" : g_var == 2 ? "neumann-var-3d" : g_var == 3 ? "neumann-var-opp-3d"
              : g_var == 4 ? "dirichlet-var-3d" : g_multi == 1 ? "neumann-opp-3d" : g_multi == 2 ? "neumann-adj-3d" : g_multi == 3 ? "dirichlet-opp-3d"
              : g_dir_h1 ? "dirichlet-h1-3d" : g_dirichlet ? "dirichlet-conv-3d" : g_neumann ? "neumann-conv-3d"
                        : (g_natural ? "natural-conv-3d" : (g_cubic ? "conv-sin3-3d" : "conv-sin-3d")),
              config.mesh.nx, config.mesh.ny, config.mesh.nz, steps, T, dt,
              TimeSchemeName(config), config.diffusivity, HasConvection(config) ? "on" : "off",
              config.convection.vx, config.convection.vy, config.convection.vz,
              config.alpha_u, h, speed * h / (2.0 * config.diffusivity), (int)local,
              3LL * steps * local, static_cast<int>(options.pc_mode));

  // tightened production solve: the policy of the 2D exact driver
  KktSolverPolicy policy = MakeKktSolverPolicy(options.pc_mode, 1.0e-12);
  policy.outer_rtol = 1.0e-12;
  policy.inner_rtol = 1.0e-10;
  policy.outer_maxits = 400;
  if (options.exact_oc_production_tol)
    policy = MakeKktSolverPolicy(options.pc_mode, options.spatial_tolerance);
  const double t_solve = MPI_Wtime();
  DistributedOptimalControlSolve solve = SolveOptimalControl(config, spatial, spacetime, policy);
  const double solve_wall = MPI_Wtime() - t_solve;
  const DistributedDataSideResult& f = solve.fields;

  // ---- independent forward march with the computed control (spatial LU)
  // T11D: Dirichlet control -- the right-hand side from the UNHATTED operators
  // (u_0 = y_0|_C / eta enters the first row), C rows zeroed, minus eta B_D U per
  // slot (the T7D gate construction); the marched state then carries y_C = eta u.
  DistributedDataSideResult data =
      g_dirichlet ? BuildDistributedRightHandSide(config, spatial.layout, spatial.mass_full,
                                                  spatial.tracking, spacetime.distributed.ay,
                                                  spatial.dirichlet->unhatted_stiffness(),
                                                  spatial.source, &spatial.lift)
                  : BuildDistributedRightHandSide(
      config, spatial.layout, spatial.mass, spatial.tracking, spacetime.distributed.ay,
      spatial.stiffness, spatial.source, &spatial.lift);
  Vec bdu = nullptr;
  if (g_dirichlet) {
    PetscCallAbort(PETSC_COMM_WORLD, spatial.dirichlet->ZeroControlledRows(data.rhs_y));
    PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(f.control, &bdu));
    PetscCallAbort(PETSC_COMM_WORLD, spatial.dirichlet->ApplyBD(f.control, bdu));
  }
  Mat diagonal = nullptr, coupling = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, MatDuplicate(spatial.stiffness, MAT_COPY_VALUES, &diagonal));
  PetscCallAbort(PETSC_COMM_WORLD, MatScale(diagonal, cn ? 0.5 : 1.0));
  PetscCallAbort(PETSC_COMM_WORLD, MatAXPY(diagonal, 1.0 / dt, spatial.mass, SUBSET_NONZERO_PATTERN));
  PetscCallAbort(PETSC_COMM_WORLD, MatDuplicate(spatial.mass, MAT_COPY_VALUES, &coupling));
  PetscCallAbort(PETSC_COMM_WORLD, MatScale(coupling, 1.0 / dt));
  if (cn)
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatAXPY(coupling, -0.5, spatial.stiffness, DIFFERENT_NONZERO_PATTERN));
  KSP lu = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, KSPCreate(PETSC_COMM_WORLD, &lu));
  PetscCallAbort(PETSC_COMM_WORLD, KSPSetOptionsPrefix(lu, "exact_march_"));
  PetscCallAbort(PETSC_COMM_WORLD, KSPSetOperators(lu, diagonal, diagonal));
  PetscCallAbort(PETSC_COMM_WORLD, KSPSetType(lu, KSPPREONLY));
  PC pc = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, KSPGetPC(lu, &pc));
  PetscCallAbort(PETSC_COMM_WORLD, PCSetType(pc, PCLU));
  PetscCallAbort(PETSC_COMM_WORLD, KSPSetUp(lu));
  Vec r = nullptr, y = nullptr, prev = nullptr, tmp = nullptr, u = nullptr, marched = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, MatCreateVecs(spatial.mass, &r, &y));
  for (Vec* v : {&prev, &tmp, &u}) PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(r, v));
  PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(f.state, &marched));
  for (unsigned slot = 0; slot < steps; ++slot) {
    CopyDataSideSliceToSpatial(data.rhs_y, slot, local, r);
    CopyDataSideSliceToSpatial(f.control, slot, local, u);
    if (g_dirichlet) {
      CopyDataSideSliceToSpatial(bdu, slot, local, tmp);
      PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(r, -config.eta, tmp));
    } else {
    PetscCallAbort(PETSC_COMM_WORLD, MatMult(spatial.control, u, tmp));
    PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(r, config.eta, tmp));
    }
    if (slot > 0u) {
      PetscCallAbort(PETSC_COMM_WORLD, MatMult(coupling, prev, tmp));
      PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(r, 1.0, tmp));
    }
    PetscCallAbort(PETSC_COMM_WORLD, KSPSolve(lu, r, y));
    if (g_dirichlet) {   // T11D: the strong trace y_C = eta u
      PetscScalar* yy = nullptr;
      const PetscScalar* uu = nullptr;
      PetscCallAbort(PETSC_COMM_WORLD, VecGetArray(y, &yy));
      PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(u, &uu));
      const std::vector<unsigned char>& m = spatial.dirichlet->control_mask();
      for (PetscInt i = 0; i < local; ++i)
        if (m[static_cast<std::size_t>(i)]) yy[i] = config.eta * uu[i];
      PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(u, &uu));
      PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArray(y, &yy));
    }
    PetscCallAbort(PETSC_COMM_WORLD, VecCopy(y, prev));
    CopySpatialToDataSideSlice(y, slot, local, 1.0, marched, false);
  }
  const double march_rel = RelDiff(f.state, marched);
  // stationarity alpha_u u + eta p = 0 (full-domain L2 control)
  Vec stat = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(f.control, &stat));
  PetscCallAbort(PETSC_COMM_WORLD, VecSet(stat, 0.0));
  PetscCallAbort(PETSC_COMM_WORLD,
                 VecAXPBYPCZ(stat, config.alpha_u, config.eta, 0.0, f.control, f.adjoint));
  PetscReal stat_norm = 0.0, au_norm = 0.0;
  PetscCallAbort(PETSC_COMM_WORLD, VecNorm(stat, NORM_2, &stat_norm));
  PetscCallAbort(PETSC_COMM_WORLD, VecNorm(f.control, NORM_2, &au_norm));
  double stat_rel = stat_norm / (config.alpha_u * (au_norm > 0 ? au_norm : 1.0));
  if (g_dirichlet) {
    // T11D: Dirichlet stationarity alpha_u R_T U = eta B_D^T P on C (R = M_Gamma,
    // trapezoidal What_u under CN), the strong trace y_C = eta u, the control
    // off C (perimeter included) and C itself: its size (2n_x-1)(2n_y-1) and
    // that it lies on z = 0 strictly inside the face.
    DirichletControlOperators& dc = *spatial.dirichlet;
    Vec ru = nullptr, bp = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(f.control, &ru));
    PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(f.control, &bp));
    PetscCallAbort(PETSC_COMM_WORLD, dc.ApplyR(f.control, ru));
    PetscCallAbort(PETSC_COMM_WORLD, VecScale(ru, config.alpha_u));
    PetscCallAbort(PETSC_COMM_WORLD, dc.ApplyBDT(f.adjoint, bp));
    PetscReal ru_norm = 0.0, diff_norm = 0.0;
    PetscCallAbort(PETSC_COMM_WORLD, VecNorm(ru, NORM_2, &ru_norm));
    PetscCallAbort(PETSC_COMM_WORLD, VecScale(bp, config.eta));   // eta B_D^T P
    PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(bp, -1.0, ru));
    if (dc.has_time_regularization()) {
      // T11H1: alpha_u R_H1t U - eta B_D^T P = (gamma_u/dt) q_0, q_0 = (1/dt)[M_G u_0; 0; ..]
      Vec trace = nullptr, q0 = nullptr;
      PetscCallAbort(PETSC_COMM_WORLD, MatCreateVecs(dc.regularization(), &trace, nullptr));
      PetscScalar* a = nullptr;
      PetscCallAbort(PETSC_COMM_WORLD, VecGetArray(trace, &a));
      for (PetscInt i = 0; i < local; ++i)
        a[i] = dc.control_mask()[static_cast<std::size_t>(i)]
                   ? EvaluateInitialState(config, spatial.layout.x[i], spatial.layout.z[i],
                                          spatial.layout.Axis2(i)) / config.eta
                   : 0.0;
      PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArray(trace, &a));
      PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(f.control, &q0));
      PetscCallAbort(PETSC_COMM_WORLD, dc.ApplyAffineData(trace, q0));
      PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(bp, ControlTimeSourceWeight(config), q0));
      PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&q0));
      PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&trace));
    }
    PetscCallAbort(PETSC_COMM_WORLD, VecNorm(bp, NORM_2, &diff_norm));
    stat_rel = diff_norm / (ru_norm > 0 ? ru_norm : 1.0);
    const std::vector<unsigned char>& m = dc.control_mask();
    const PetscScalar *yy = nullptr, *uu = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(f.state, &yy));
    PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(f.control, &uu));
    double trace_err = 0.0, off_max = 0.0;
    long long n_c = 0, off_plane = 0;
    for (PetscInt i = 0; i < local; ++i) {
      if (m[static_cast<std::size_t>(i)]) {
        ++n_c;
        const double xi = spatial.layout.x[i], yi = spatial.layout.z[i], zi = spatial.layout.Axis2(i);
        if (IsMultiFace3D(config) ? !OnControlSet3D(config, xi, yi, zi, true)   // T13: the geometric C of the union
                    : (std::fabs(zi) > 1e-12 || xi < 1e-12 || xi > 1 - 1e-12 || yi < 1e-12 || yi > 1 - 1e-12)) ++off_plane;
      }
      for (unsigned slot = 0; slot < steps; ++slot) {
        const PetscInt k = static_cast<PetscInt>(slot) * local + i;
        if (m[static_cast<std::size_t>(i)])
          trace_err = std::max(trace_err, std::fabs(PetscRealPart(yy[k]) - config.eta * PetscRealPart(uu[k])));
        else
          off_max = std::max(off_max, std::fabs(PetscRealPart(uu[k])));
      }
    }
    PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(f.control, &uu));
    PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(f.state, &yy));
    std::printf("EX21_DIRICHLET3D_STATIONARITY face=%s(id %u) |alpha_u R_T U - eta B_D^T P|/|alpha_u R_T U|=%.3e"
                " max|y_C - eta u|=%.3e max|u|_off_C=%.3e C_nodes=%lld expected=%d C_off_interior=%lld"
                " B_D^T_transposes_L=%d\n",
                config.neumann_control_boundary.c_str(), config.neumann_control_face, stat_rel,
                trace_err, off_max, n_c,
                IsMultiFace3D(config) ? (int)ExpectedControlNodes3D(config, true) : (2 * (int)config.mesh.nx - 1) * (2 * (int)config.mesh.ny - 1),
                off_plane, dc.transposes_stiffness() ? 1 : 0);
    PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&ru));
    PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&bp));
  }
  if (g_neumann) {
    // T11N: stationarity on Gamma_c in the control metric,
    //   |M_Gamma (alpha_u u + eta p)| / |alpha_u M_Gamma u|,
    // the control off the face (must be exactly 0) and the active set: its
    // size and that every active node lies on the plane z = 0.
    const std::vector<unsigned char> active = ActiveControlDofs(spatial.control, spatial.layout);
    Vec v = nullptr, w = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(r, &v));
    PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(r, &w));
    double num = 0.0, den = 0.0, off_max = 0.0;
    for (unsigned slot = 0; slot < steps; ++slot) {
      CopyDataSideSliceToSpatial(f.control, slot, local, u);
      CopyDataSideSliceToSpatial(f.adjoint, slot, local, v);
      PetscCallAbort(PETSC_COMM_WORLD, VecAXPBY(v, config.alpha_u, config.eta, u));
      PetscCallAbort(PETSC_COMM_WORLD, MatMult(spatial.control, v, w));
      PetscReal a = 0.0;
      PetscCallAbort(PETSC_COMM_WORLD, VecNorm(w, NORM_2, &a));
      num += a * a;
      PetscCallAbort(PETSC_COMM_WORLD, MatMult(spatial.control, u, w));
      PetscCallAbort(PETSC_COMM_WORLD, VecNorm(w, NORM_2, &a));
      den += config.alpha_u * config.alpha_u * a * a;
      const PetscScalar* uu = nullptr;
      PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(u, &uu));
      for (PetscInt i = 0; i < local; ++i)
        if (!active[i]) off_max = std::max(off_max, std::fabs(PetscRealPart(uu[i])));
      PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(u, &uu));
    }
    long long n_active = 0, off_plane = 0;
    for (PetscInt i = 0; i < local; ++i)
      if (active[i]) {
        ++n_active;
        if (IsMultiFace3D(config) ? !OnControlSet3D(config, spatial.layout.x[i], spatial.layout.z[i], spatial.layout.Axis2(i), false)
                    : std::fabs(spatial.layout.Axis2(i) - config.domain.zmin) > 1e-12) ++off_plane;
      }
    stat_rel = std::sqrt(num / (den > 0.0 ? den : 1.0));
    std::printf("EX21_NEUMANN3D_STATIONARITY face=%s(id %u) |M_G(alpha_u u + eta p)|/|alpha_u M_G u|=%.3e"
                " max|u|_off_face=%.3e active_nodes=%lld expected=%d active_off_plane=%lld\n",
                config.neumann_control_boundary.c_str(), config.neumann_control_face, stat_rel, off_max,
                n_active,
                IsMultiFace3D(config) ? (int)ExpectedControlNodes3D(config, false) : (2 * (int)config.mesh.nx + 1) * (2 * (int)config.mesh.ny + 1),
                off_plane);
    PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&v));
    PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&w));
  }

  // ---- FEM-integrated space-time L2 errors
  struct Acc { double e = 0, x = 0, last_e = 0, last_x = 0, max_e = 0; } ey, ep, eu, eg;
  double eg_h = 0.0;   // T11H1: int |grad_G u_h|^2
  for (unsigned slot = 0; slot < steps; ++slot) {
    const double tn = (slot + 1u) * dt, tm = (slot + 0.5) * dt;
    const double tp = cn ? tm : tn;
    auto integrate = [&](Vec field, int which, double time) {
      CopyDataSideSliceToSpatial(field, slot, local, r);
      const PetscScalar* a = nullptr;
      PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(r, &a));
      Context3D ctx{&config, time, which};
      // T11N: the Neumann control lives on Gamma_c -- its error is the surface L2
      // norm on the face (the face quadrature that assembled M_Gamma)
      const SpatialFieldIntegrals I =
          (g_var >= 1 && g_var <= 3)   // T14: natural-var-3d (volume u), neumann-var-* (u on Gamma_c = sum_f)
              ? ((which == 2 && g_var != 1)
                     ? IntegrateBoundaryField3DFaces(mesh, solution_type, ControlFaceMask3D(config), a, VarExactAt3D, &ctx)
                     : IntegrateSpatialField3D(mesh, solution_type, a, VarExactAt3D, &ctx))
          : g_multi ? (which == 2 ? IntegrateBoundaryField3DFaces(mesh, solution_type, ControlFaceMask3D(config), a,
                                                                g_multi == 3 ? DopExactAt3D : MnExactAt3D, &ctx)
                                : IntegrateSpatialField3D(mesh, solution_type, a,
                                                          g_multi == 3 ? DopExactAt3D : MnExactAt3D, &ctx))   // T13: sum_f
          : g_dirichlet ? (which == 2 ? IntegrateBoundaryField3D(mesh, solution_type,
                                                               config.neumann_control_face, a,
                                                               DirExactAt3D, &ctx)
                                    : IntegrateSpatialField3D(mesh, solution_type, a, DirExactAt3D, &ctx))
          : g_neumann ? (which == 2 ? IntegrateBoundaryField3D(mesh, solution_type,
                                                             config.neumann_control_face, a,
                                                             NeuExactAt3D, &ctx)
                                  : IntegrateSpatialField3D(mesh, solution_type, a, NeuExactAt3D, &ctx))
                    : IntegrateSpatialField3D(mesh, solution_type, a, g_natural ? NatExactAt3D : ExactAt3D, &ctx);
      PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(r, &a));
      return I;
    };
    const SpatialFieldIntegrals Iy = integrate(f.state, 0, tn);
    const SpatialFieldIntegrals Ip = integrate(f.adjoint, 1, tp);
    // T11D: the Dirichlet control is NODAL in time (u_n at t_n, both schemes; T7D)
    const SpatialFieldIntegrals Iu = integrate(f.control, 2, g_dirichlet ? tn : tp);
    if (g_dir_h1) {   // T11H1: surface H1 seminorm error of u at t_n (nodal)
      CopyDataSideSliceToSpatial(f.control, slot, local, r);
      const PetscScalar* a = nullptr;
      PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(r, &a));
      Context3D ctx{&config, tn, 2};
      const SpatialFieldIntegrals Ig = g_multi
          ? IntegrateBoundaryGradient3DFaces(mesh, solution_type, ControlFaceMask3D(config), a, DopExactUGrad, &ctx)
          : IntegrateBoundaryGradient3D(mesh, solution_type, config.neumann_control_face, a, DirExactUGrad, &ctx);
      PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(r, &a));
      eg.e += dt * Ig.error_squared;
      eg.x += dt * Ig.exact_squared;
      eg_h += dt * Ig.field_squared;
    }
    for (auto pr : {std::make_pair(&ey, &Iy), std::make_pair(&ep, &Ip), std::make_pair(&eu, &Iu)}) {
      pr.first->e += dt * pr.second->error_squared;
      pr.first->x += dt * pr.second->exact_squared;
      pr.first->max_e = std::max(pr.first->max_e, std::sqrt(pr.second->error_squared));
      if (slot + 1u == steps) { pr.first->last_e = pr.second->error_squared; pr.first->last_x = pr.second->exact_squared; }
    }
  }
  auto counter = [&](const char* nm, const InnerIterationCounter& c) {
    std::printf(" %s_solves=%lld %s_its=%lld %s_max=%lld", nm, c.solves, nm, c.iterations, nm,
                c.max_iterations);
  };
  std::printf("EX21_EXACT3D_SOLVER_STATS outer_its=%d reason=%d true_residual=%.3e"
              " state_rel=%.3e adjoint_rel=%.3e sweeps=%d",
              solve.outer_iterations, solve.reason, solve.true_residual, solve.kkt.state_rel,
              solve.kkt.adjoint_rel, solve.sweeps_used ? 1 : 0);
  counter("right", solve.right_inner);
  counter("left", solve.left_inner);
  counter("ay", solve.ay_inner);
  counter("ysplit", solve.y_split);
  counter("schur_ksp", solve.p_split);
  std::printf(" sweep_failures=%lld solve_wall=%.3f accepted=%d status=%s\n",
              solve.sweep_failures, solve_wall, solve.accepted ? 1 : 0, solve.status.c_str());
  auto rel = [](double e, double x) { return x > 0 ? std::sqrt(e / x) : std::sqrt(e); };
  std::printf("EX21_EXACT3D_RESULT scheme=%s nx=%u ny=%u nz=%u steps=%u dt=%.10e reason=%d"
              " outer_its=%d true_residual=%.3e stationarity_rel=%.3e march_rel=%.3e"
              " J_track=%.12e J_reg=%.12e"
              " y_L2L2=%.10e y_L2L2_rel=%.10e y_last=%.10e y_Linf_L2=%.10e"
              " p_L2L2=%.10e p_L2L2_rel=%.10e p_last=%.10e p_Linf_L2=%.10e"
              " u_L2L2=%.10e u_L2L2_rel=%.10e u_last=%.10e u_Linf_L2=%.10e"
              " solve_wall=%.3f total_wall=%.3f\n",
              TimeSchemeName(config), config.mesh.nx, config.mesh.ny, config.mesh.nz, steps, dt,
              solve.reason, solve.outer_iterations, solve.true_residual, stat_rel, march_rel,
              f.J_track, f.J_reg,
              std::sqrt(ey.e), rel(ey.e, ey.x), std::sqrt(ey.last_e), ey.max_e,
              std::sqrt(ep.e), rel(ep.e, ep.x), std::sqrt(ep.last_e), ep.max_e,
              std::sqrt(eu.e), rel(eu.e, eu.x), std::sqrt(eu.last_e), eu.max_e,
              solve_wall, MPI_Wtime() - t0);
  if (g_dir_h1)
    std::printf("EX21_DIRICHLET3D_H1 ell=%.6g gamma_u=%.6g u_H1s_L2L2=%.10e u_H1s_L2L2_rel=%.10e"
                " gradG_u_h_L2L2=%.10e J_reg_time=%.12e surface_gradient_normal_leak=%.3e\n",
                config.control_h1_length, config.control_h1_time_weight, std::sqrt(eg.e),
                rel(eg.e, eg.x), std::sqrt(eg_h), f.J_reg_time, SurfaceGradientNormalLeak());
  for (Vec* v : {&r, &y, &prev, &tmp, &u, &marched, &stat})
    PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(v));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&bdu));   // T11D (null otherwise)
  PetscCallAbort(PETSC_COMM_WORLD, KSPDestroy(&lu));
  PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&diagonal));
  PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&coupling));
  return solve.accepted ? 0 : 8;
}
