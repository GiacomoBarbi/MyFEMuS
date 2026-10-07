// Mathematics 2025 (Aulisa, Chierici, Gilliam, 13, 3707), Example 3: the
// source's own finite-horizon optimal-control problem (82)-(84), reproduced as
// an ex21 benchmark.  See aux/investigation/MATH2025_EX3_SOURCE_AUDIT.md.
//
//   Omega = [0,1], z_t = 2 z_xx - 2 z_x + chi_{Omega_d} d + chi_{Omega_b} u,
//   z(0,t) = z(1,t) = 0, z(x,0) = x(1-x)  (or 0, see --ex3-initial),
//   Omega_b = [1/5,2/5], Omega_c = [1/2,7/10], Omega_d = [4/5,9/10],
//   r = x sin(t) F(t-1.5,1.5),  d = x^2 sin(2t) F(t-1.5,1.5),
//   J = int_0^T [ 1/2 ||z - r||^2_{L2(Omega_c)} + eps/2 ||u||^2_{L2(Omega_b)} ] dt,
//   eps = 1e-9  (alpha_track = 1, alpha_u = eps, eta = 1).
//
// --math2025-ex3 installs the data (a preset: mesh size, step count, T, time
// scheme and preconditioner stay under command-line control) and selects the
// driver below.  Combined with --verify-convection it runs the convection
// gates on this configuration instead.

#pragma once

#include <string>

#include "MultiLevelSolution.hpp"

#include "Config.hpp"
#include "Options.hpp"

namespace femus {
class Mesh;
}

/** F(s, eps): the order-4 smoothstep -- 0 for s < -eps, 1 for s > eps, and on
 *  [-eps, eps] the unique degree-9 polynomial with F(-eps) = 0, F(eps) = 1 and
 *  vanishing derivatives of orders 1..4 at both ends. */
double Math2025SmoothStep(double s, double eps);

/** Install the Example 3 plant and functional.  `initial` is "source"
 *  (z0 = x(1-x), eq. (80)) or "zero"; `final_time` is T. */
void ApplyMath2025Example3(ProblemConfig& config, const std::string& initial,
                           double final_time);

/** Example 4 (colocated, pp. 27-29): Omega = [0,1] (ell = 1), Omega_b =
 *  Omega_c = [0.45,0.55], Omega_d = [0.7,0.8], the signals (78) of Example 3,
 *  T = final_time (source: 15).  The source states NO initial condition and no
 *  optimal-control epsilon: `initial` is "zero" or "x1mx" (x(1-x), the Example 3
 *  value, a CANDIDATE only), and alpha_u is left to the command line. */
void ApplyMath2025Example4(ProblemConfig& config, const std::string& initial,
                           double final_time);

/** Driver of both presets (the example is recorded by the Apply* call). */
int RunMath2025Example3(const ProblemConfig& config,
                        const ApplicationOptions& options,
                        femus::MultiLevelSolution& solution, femus::Mesh* mesh,
                        unsigned solution_type);
