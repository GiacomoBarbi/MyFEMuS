// ex21 convection gates.  Opt-in run modes, serial only, never on the
// production path.
//
//   --verify-convection   G1 sign/orientation of the assembled convection
//                         matrix C = L - K, and its skew-symmetry on the free
//                         DOFs;  G2 exact-transpose tests of the space-time
//                         shells, of their assembled copies and of the reduced
//                         Hessian;  G3 reduced-gradient finite differences on
//                         the REPORTED objective;  G4 the production KKT solve
//                         and the reduced gradient at its control.
//   --convection-mms      forward manufactured solution of
//                         y_t - nu Lap y + v . grad y = f  with homogeneous
//                         Dirichlet data: L2 errors and the all-at-once residual
//                         of the marched solution.
//
// Both modes use the configuration exactly as parsed (dimension, mesh, scheme,
// nu, v, alpha_u, ...); see EX21_CONVECTION_VALIDATION.md.

#pragma once

#include <string>

#include "MultiLevelSolution.hpp"

#include "Config.hpp"
#include "Options.hpp"

namespace femus {
class Mesh;
}

int RunConvectionVerification(const ProblemConfig& config,
                              const ApplicationOptions& options,
                              femus::MultiLevelSolution& solution,
                              femus::Mesh* mesh, unsigned solution_type);

/** `shape` = "sin" or "sin3", see ConvectionVerification.cpp. */
int RunConvectionManufactured(const ProblemConfig& config,
                              const std::string& shape,
                              femus::MultiLevelSolution& solution,
                              femus::Mesh* mesh, unsigned solution_type);
