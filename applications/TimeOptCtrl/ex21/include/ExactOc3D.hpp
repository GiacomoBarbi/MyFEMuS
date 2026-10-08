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
 *  R_k = exp(-k s)(2 + cos(pi s)), k_i = w_i/nu (ExactOc3D.cpp).  `alpha_u` <= 0:
 *  1e-2; `final_time` <= 0: 1; `nu` <= 0: 1/4.  The wind is NOT set here:
 *  it is --convection on/off with --vx --vy --vz. */
void ApplyExactOptimalControl3D(ProblemConfig& config, const std::string& name,
                                double alpha_u, double final_time, double nu);

/** Serial driver: tightened production solve (the policy of the 2D exact
 *  driver), FEM-integrated L2(Omega) errors of y (at t_n) and p, u (interval
 *  midpoints under CN, t_n under BE) against the exact solution, independent
 *  LU forward march, stationarity.  Prints EX21_EXACT3D_* lines. */
int RunExactOptimalControl3D(const ProblemConfig& config,
                             const ApplicationOptions& options,
                             femus::MultiLevelSolution& solution,
                             femus::Mesh* mesh, unsigned solution_type);
