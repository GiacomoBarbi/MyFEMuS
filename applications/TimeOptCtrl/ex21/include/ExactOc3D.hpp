// T9 (3D phase 1): exact distributed optimal control on the unit cube.
//
//   S(x, y, z) = sin(pi x) sin(pi y) sin(pi z)          (0 on dOmega)
//   y = A(t) S,  A = cos(2t);   p = alpha_u B(t) S,  B = sin(T - t)  (p(T) = 0)
//   u = -(eta/alpha_u) p = -eta B S              (L2, full-domain control)
//   f   = y_t - nu Lap y + w.grad y - eta u
//       = (A' + 3 pi^2 nu A + eta^2 B) S + A (w.grad S)
//   y_d = y + (p_t + nu Lap p + w.grad p)/alpha_t
//       = A S + (alpha_u/alpha_t) [(B' - 3 pi^2 nu B) S + B (w.grad S)]
// with a constant wind w = (vx, vy, vz) on storage axes (x, y, z) (0 when
// --convection is off).  ex21 convention: state y_t - nu Lap y + w.grad y =
// f + eta u, adjoint -p_t - nu Lap p - w.grad p = alpha_t (y - y_d),
// stationarity alpha_u u + eta p = 0.  Verified by
// aux/investigation/3d_phase1/t9/scripts/t9_symbolic.py.

#pragma once

#include <string>

#include "MultiLevelSolution.hpp"

#include "Config.hpp"
#include "Options.hpp"

namespace femus {
class Mesh;
}

/** --exact-oc conv-sin-3d (shape sin) or conv-sin3-3d (shape sin^3, data
 *  vanishing on dOmega; the 3D analogue of the 2D conv-sin3): dimension 3, distributed full-domain control and
 *  tracking, homogeneous Dirichlet state, the data above.
 *  T10: natural-conv-3d -- natural state BC on all six faces, y = A cos(pi x)
 *  cos(pi y) cos(pi z), p = alpha_u B R_kx(x) R_ky(y) R_kz(z) with
 *  R_k = exp(-k s)(2 + cos(pi s)), k_i = w_i/nu (ExactOc3D.cpp).
 *  T11N: neumann-conv-3d -- Neumann boundary control on Gamma_c = {z = 0}
 *  (FEMuS face 1), natural state BC elsewhere; p = alpha_u B X(x) X(y) R_kz(z),
 *  X = 1 - cos(2 pi s), u = -(eta/alpha_u) p|_{z=0}; the u error is the surface
 *  L2 norm on Gamma_c (ExactOc3D.cpp).
 *  T11D: dirichlet-conv-3d -- Dirichlet boundary control on Gamma_c = {z = 0}
 *  (FEMuS face 1), y = 0 on the other five faces (and on dGamma_c, so u = 0
 *  there); tracking omega = (0,1)^2 x (1/2,1).  p = X(x) X(y) (1 - phi(z)) Qs(z,t),
 *  X = exp(-c s) sin(pi s), c = w_axis/(2 nu), Qs an image heat kernel in z
 *  (p(T) = 0 as a limit), phi a C^3 septic step on [1/2, 1];
 *  u = -(eta nu/alpha_u) X X dQs/dz(0,t) (= eta nu dn p / alpha_u);
 *  y = cos(2t) sin^3 sin^3 sin^3 + eta u (1 - z)^3.  u is nodal in time (BE and
 *  CN) and its error is the surface L2 norm on Gamma_c.  Verified by
 *  aux/investigation/3d_boundary/t11d/scripts/t11d_symbolic.py.  `alpha_u` <= 0:
 *  1e-2; `final_time` <= 0: 1; `nu` <= 0: 1/4.  The wind is NOT set here:
 *  it is --convection on/off with --vx --vy --vz. */
void ApplyExactOptimalControl3D(ProblemConfig& config, const std::string& name,
                                double alpha_u, double final_time, double nu);

/** Serial driver: tightened production solve (the policy of the 2D exact
 *  driver), FEM-integrated L2(Omega) errors of y (at t_n) and p, u (interval
 *  midpoints under CN, t_n under BE; the Dirichlet control at t_n always)
 *  against the exact solution, independent LU forward march, stationarity
 *  (Dirichlet: EX21_DIRICHLET3D_STATIONARITY).  Prints EX21_EXACT3D_* lines. */
int RunExactOptimalControl3D(const ProblemConfig& config,
                             const ApplicationOptions& options,
                             femus::MultiLevelSolution& solution,
                             femus::Mesh* mesh, unsigned solution_type);
