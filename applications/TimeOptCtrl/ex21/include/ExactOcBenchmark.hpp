// Exact optimal-control solutions of ex21 (Leveque-Pearson validation phase,
// tests T1 and T2 of aux/investigation/LEVEQUE_PEARSON_SOURCE_AUDIT.md).
//
// ex21 convention (derived in aux/investigation/leveque_pearson/
// T2_EXACT_CONVECTION_OC.md, verified symbolically by t2_symbolic_check.py):
//
//   state      y_t - nu Lap y + v.grad y = f + eta u,   y = 0 on dOmega,
//   adjoint   -p_t - nu Lap p - v.grad p = alpha_track (y - y_d),
//              p = 0 on dOmega,  p(T) = 0,
//   gradient   alpha_u u + eta p = 0,
//
// with eta = 1, alpha_track = 1, full-domain control and tracking.
//
//   conv-sin3 / conv-sin (T2, SOURCE-INSPIRED, not a literature problem):
//     Omega = (0,1)^2, T = 1, nu = 1/4, v = (1, 1/2), alpha_u = 1e-2 (default),
//     S1 = s(pi x) s(pi z), S2 = s(2 pi x) s(pi z), s = sin^3 or sin,
//     y = cos(2t) S1,  p = alpha_u sin(T-t) S2,  u = -sin(T-t) S2,
//     f = y_t - nu Lap y + v.grad y - u,  y_d = y + p_t + nu Lap p + v.grad p.
//     sin3: y, p, f, y_d and their gradients vanish on dOmega (compatible with
//     the inherited M_elim I_h f load); sin: f, y_d != 0 on dOmega (v != 0).
//
//   lp-heat (T1): Leveque & Pearson (NLAA 2022) section 5.1, transcribed from
//     ex20/aux/EX20_LEVEQUE_PEARSON_2022_BENCHMARK.md section 4, with the exact
//     constant shift w = y - 1 of its section 5 and p_ex21 = -p_LP:
//     Omega = (-1,1)^2, T = 2, nu = 1, v = 0, f = 0, alpha_u = beta (default 1e-3).
//
// The preset installs data only; nx, nz, steps, scheme and pc stay under
// command-line control.  Combined with --verify-convection the convection gates
// (transpose, FD gradient, Hessian symmetry, production solve) run on the
// preset configuration.

#pragma once

#include <string>

#include "MultiLevelSolution.hpp"

#include "Config.hpp"
#include "Options.hpp"

namespace femus {
class Mesh;
}

/** `name`: conv-sin3, conv-sin, lp-heat, lift-const, lift-exp, lift-track,
 *  lp52 or neumann-conv (T7N: Neumann boundary control on z = 0).
 *  `alpha_u` <= 0 keeps the case default; `final_time` <= 0 keeps the case
 *  default; `velocity` (null: case default) sets the constant wind of the
 *  convective manufactured cases. */
void ApplyExactOptimalControl(ProblemConfig& config, const std::string& name,
                              double alpha_u, double final_time,
                              double nu = -1.0,    // lp52: epsilon (<= 0: 1/20)
                              const double* velocity = nullptr);

/** Serial driver: tightened production solve, space-time errors of y, p, u
 *  against the exact solution, independent forward march, optional dump of
 *  the spatial and space-time operators (tiny meshes, for t0_cn_matrix_check). */
int RunExactOptimalControl(const ProblemConfig& config,
                           const ApplicationOptions& options,
                           femus::MultiLevelSolution& solution,
                           femus::Mesh* mesh, unsigned solution_type);
