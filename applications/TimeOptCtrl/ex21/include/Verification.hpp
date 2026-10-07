// Verification entry points.  A leaf of the dependency graph: it consumes the
// production code and nothing consumes it.  Only main() includes this header,
// to dispatch four opt-in run modes:
//
//     --dump-operators            ->  RunSpatialOperatorDump           spatial operator export
//     --verify-sweeps             ->  RunSweepVerification             time-sweep algebra
//     --verify-stage1             ->  RunVerificationStage1            algebraic test battery
//     --verify-boundary-reduced   ->  RunBoundaryReducedVerification   reduced Schur solver sentinel

#pragma once

#include <string>

namespace femus {
class Mesh;
class MultiLevelSolution;
}  // namespace femus

struct ProblemConfig;

int RunSpatialOperatorDump(const ProblemConfig& config,
                           femus::MultiLevelSolution& solution,
                           femus::Mesh* mesh, unsigned solution_type,
                           const std::string& prefix);

int RunSweepVerification(const ProblemConfig& config,
                         femus::MultiLevelSolution& solution, femus::Mesh* mesh,
                         unsigned solution_type, double spatial_tolerance);

int RunVerificationStage1(const ProblemConfig& config,
                          femus::MultiLevelSolution& solution,
                          femus::Mesh* mesh, unsigned solution_type,
                          const std::string& output_prefix);

/** Regression sentinel of the boundary-reduced Schur solver (Neumann control
 *  required): a production solve at alpha_u = 1e-8 and at 1e-12 with fixed
 *  bounds on the outer iterations and the true KKT residual, and one
 *  caller-owned operator reused across three solves.  Prints one BR* line per
 *  check and a BR_VERIFY summary; returns 0 iff every check passed. */
int RunBoundaryReducedVerification(const ProblemConfig& config,
                                   femus::MultiLevelSolution& solution,
                                   femus::Mesh* mesh, unsigned solution_type,
                                   double spatial_tolerance);
