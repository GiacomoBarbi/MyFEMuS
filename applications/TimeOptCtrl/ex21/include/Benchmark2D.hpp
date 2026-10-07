// Two-dimensional convection-diffusion literature benchmarks of ex21.
//
//  * Mathematics 2025 (Aulisa, Chierici, Gilliam, 13, 3707), Example 5
//    (pp. 30-31): the source's own finite-horizon optimal control on a
//    rectangle Omega = [0,W] x [0,H] (W, H NOT stated -- see
//    aux/investigation/MATH2025_EX5_SOURCE_AUDIT.md):
//      z_t = 2 Lap z - (2,2).grad z + chi_d d + chi_b u,  z = 0 on dOmega,
//      Omega_b = [0,W/3] x [0,H], Omega_c = [W/3,2W/3] x [0,H],
//      Omega_d = [2W/3,W] x [0,H],
//      r = x y(1-y)(y-1/2) sin(t) F(t-1.5,1.5),
//      d = x (y-1/2)^2 sin(2t) F(t-1.5,1.5),
//      J = int_0^T [ 1/2 ||z-r||^2_{L2(Omega_c)} + eps/2 ||u||^2_{L2(Omega_b)} ],
//      eps = 1e-7 (alpha_track = 1, alpha_u = eps, eta = 1, not overridable).
//
//  * ErrorFeedback 2026, Examples 9.5 / 9.6 (pp. 35-39): Omega = (0,pi) x (0,1),
//      z_t = Lap z - (2,1).grad z + chi_d d + chi_B u,  z = 0 on dOmega,
//      r = x exp(-(y-1/2)^2/s^2) sin(2t),  d = x exp(-(y-1/2)^2/s^2) cos(3t),
//      s = 0.15; regions (asserted by the user, see the audit):
//      9.5  Omega_B = Omega_C = [3pi/8,pi/2] x [1/3,2/3]
//      9.6  Omega_B = [pi/8,pi/4] x [1/3,2/3], Omega_C = [3pi/8,pi/2] x [1/3,2/3]
//      both Omega_d = [3pi/4,7pi/8] x [1/3,2/3].
//    The source's controller is a cascade/feedback (REFC) at unpublished
//    alpha, beta: alpha_u is left to the command line and is NEVER identified
//    with the source's parameters.  The source states no IC: --ef-initial.
//
// The presets install data only; nx, nz, steps, T, scheme and pc stay under
// command-line control.  Combined with --verify-convection they run the
// convection gates on the preset configuration instead.

#pragma once

#include <string>

#include "MultiLevelSolution.hpp"

#include "Config.hpp"
#include "Options.hpp"

namespace femus {
class Mesh;
}

/** Example 5 on [0,W] x [0,H].  `initial`: "xy" (the plant IC phi = xy of the
 *  cascade runs) or "zero".  `variant` is a SENSITIVITY diagnostic of the
 *  disturbance ("source" = eq. (87); "d-zero"; "d-noswitch" = (87) without F). */
void ApplyMath2025Example5(ProblemConfig& config, double width, double height,
                           const std::string& initial, double final_time,
                           const std::string& variant = "source");

/** ErrorFeedback Example 9.5 (`example` = 95) or 9.6 (96).  `initial`:
 *  "zero", "sinsin" (sin x sin pi y) or "poly" (16 x(pi-x) y(1-y) / pi^2). */
void ApplyErrorFeedback9x(ProblemConfig& config, int example,
                          const std::string& initial, double final_time);

/** Shared driver: gates, production solve, independent march, error series. */
int RunBenchmark2D(const ProblemConfig& config,
                   const ApplicationOptions& options,
                   femus::MultiLevelSolution& solution, femus::Mesh* mesh,
                   unsigned solution_type);
