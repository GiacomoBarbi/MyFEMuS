/**
 * ex21: all-at-once optimal control of a linear convection-diffusion state
 * equation, derived from ex20 (the frozen heat-equation reference).
 *
 *   min  (alpha_track/2) ||y - y_d||^2_{L2(Q_omega)}
 *      + (alpha_u/2)     ||u||^2_{L2(Q_C)}
 *   s.t. y_t - nu Lap y + v . grad y = f + eta u  in Q = Omega x (0,T),
 *
 * v is a constant velocity, enabled with --convection on (default off, which
 * is the ex20 problem).  Discretize-then-optimize: the adjoint operator is the
 * exact algebraic transpose E^T of the discrete state operator E; no
 * continuous-adjoint operator is assembled.  1D (--dimension 1) or 2D meshes.
 *
 * ex20 header follows.
 *
 *   s.t. y_t - nu Lap y = f + eta u  in Q = Omega x (0,T),
 *        natural boundary conditions, y(.,0) = y_0
 *
 * Q2 finite elements in space; backward Euler (the default) or Crank-Nicolson
 * in time, selected at run time with --time-scheme be|cn.  All time levels are
 * solved simultaneously as one space-time KKT system; see Solver.hpp for the
 * block structure and Discretization.hpp for the spatial operators.
 *
 * Control is distributed over a box or applied as a Neumann boundary flux.
 * Crank-Nicolson currently covers DISTRIBUTED control only (piecewise-constant
 * interval controls and the trapezoidal tracking quadrature); the boundary
 * control types and the two reduced Schur solvers stay backward Euler and are
 * rejected with --time-scheme cn.
 */

#include "FemusInit.hpp"
#include "Files.hpp"
#include "MultiLevelMesh.hpp"
#include "MultiLevelSolution.hpp"

#include "Benchmarks.hpp"
#include "Config.hpp"
#include "ConvectionVerification.hpp"
#include "Benchmark2D.hpp"
#include "ExactOcBenchmark.hpp"
#include "ExactOc3D.hpp"
#include "Math2025Benchmark.hpp"
#include "Discretization.hpp"
#include "Options.hpp"
#include "Output.hpp"
#include "Preconditioners.hpp"
#include "Solver.hpp"
#include "Verification.hpp"

using namespace femus;

/** The opt-in validation and benchmark drivers.  Called only when a
 *  non-production run mode was selected; returns the process exit status. */
static int RunNonProductionMode(const ProblemConfig& config,
                                const ApplicationOptions& options,
                                MultiLevelSolution& solution, Mesh* level_mesh,
                                unsigned solution_type) {
  switch (options.run_mode) {
    case RunMode::DUMP_OPERATORS:
      return RunSpatialOperatorDump(config, solution, level_mesh,
                                    solution_type,
                                    options.operator_dump_prefix);

    case RunMode::VERIFY_SWEEPS:
      return RunSweepVerification(config, solution, level_mesh, solution_type,
                                  options.spatial_tolerance);

    case RunMode::VERIFY_STAGE1:
      Files::CheckDir(options.output_path, "");
      return RunVerificationStage1(config, solution, level_mesh,
                                   solution_type, options.verification_prefix);

    case RunMode::VERIFY_BOUNDARY_REDUCED:
      return RunBoundaryReducedVerification(config, solution, level_mesh,
                                            solution_type,
                                            options.spatial_tolerance);

    case RunMode::LITERATURE_BENCHMARK:
      return RunLiteratureBenchmark(options.benchmark, config, solution,
                                    level_mesh, solution_type,
                                    options.pc_mode, options.spatial_tolerance,
                                    options.json_path);

    case RunMode::MANUFACTURED_BENCHMARK:
      return RunManufacturedBenchmark(options.manufactured, config, solution,
                                      level_mesh, solution_type,
                                      options.pc_mode,
                                      options.spatial_tolerance,
                                      options.json_path);

    case RunMode::VERIFY_CONVECTION:
      return RunConvectionVerification(config, options, solution, level_mesh,
                                       solution_type);

    case RunMode::CONVECTION_MMS:
      return RunConvectionManufactured(config, options.mms_shape, solution,
                                       level_mesh, solution_type);

    case RunMode::MATH2025_EX3:
      return RunMath2025Example3(config, options, solution, level_mesh,
                                 solution_type);

    case RunMode::BENCHMARK_2D:
      return RunBenchmark2D(config, options, solution, level_mesh,
                            solution_type);

    case RunMode::EXACT_OC:
      if (config.dimension == 3u)   // T9: the 3D exact driver
        return RunExactOptimalControl3D(config, options, solution, level_mesh,
                                        solution_type);
      return RunExactOptimalControl(config, options, solution, level_mesh,
                                    solution_type);

    case RunMode::PRODUCTION:
      break;
  }
  // Unreachable: main() dispatches here only for non-production modes.
  return 1;
}

int main(int argc, char** argv) {
  FemusInit initialization(argc, argv, MPI_COMM_WORLD);

  // Problem data and resolved runtime options.
  ProblemConfig config;
  const ApplicationOptions options = ParseCommandLine(argc, argv, config);
  // Stage 8D temporal-parallel prototypes.  Both are 0 unless asked for, and
  // the call is then a no-op that leaves the production sweep untouched.
  SetTemporalPrototypeOptions(options.pc_time_blocks, options.pc_sweep_depth);
  // ex21: CG is invalid on nonsymmetric sweep blocks; false keeps ex20's.
  SetNonsymmetricSpatialSweeps(HasConvection(config));
  if (options.paraview_enabled) Files::CheckDir(options.output_path, "");

  // Mesh and the Q2 state space.
  MultiLevelMesh mesh;
  BuildMesh(config, mesh);

  MultiLevelSolution solution(&mesh);
  // T11D: 3D Dirichlet control -- the controlled face is not part of the
  // homogeneous Dirichlet set (its interior is the control trace C).
  InitializeStateSpace(solution, options.region_fields, config.state_bc,
                       (config.dimension == 3u &&
                        config.control.type == ProblemConfig::ControlType::DIRICHLET)
                           ? config.neumann_control_face : 0u,
      config.IsSimplex());

  Mesh* level_mesh = mesh.GetLevel(0);
  const unsigned solution_type =
      solution.GetSolutionType(solution.GetIndex("Ti"));

  if (options.run_mode != RunMode::PRODUCTION)
    return RunNonProductionMode(config, options, solution, level_mesh,
                                solution_type);

  // Spatial finite-element operators, indexed by the mesh's own Q2 DOFs.
  SpatialOperators spatial(config, level_mesh, solution_type, solution);

  // All-at-once space-time operators.
  SpaceTimeOperators spacetime(config, spatial);

  // Solve the reduced KKT system.
  const KktSolverPolicy policy =
      MakeKktSolverPolicy(options.pc_mode, options.spatial_tolerance);
  DistributedOptimalControlSolve solve =
      SolveOptimalControl(config, spatial, spacetime, policy);

  WriteSolution(solve, config, options, solution, level_mesh, spatial);

  return solve.accepted ? 0 : 8;
}
