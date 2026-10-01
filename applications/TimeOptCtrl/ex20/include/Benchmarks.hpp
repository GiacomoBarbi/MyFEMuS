// Validation drivers, in two families.  Both are leaves of the dependency
// graph: they consume the production APIs and nothing in the production path
// refers to a benchmark.  A family is expressed entirely as (a) a ProblemConfig
// built from the generic data-field forms of Config.hpp and (b) analytic fields
// used only for a-posteriori error measurement.
//
// The two share one measurement core in Benchmarks.cpp: the Bochner error norms
// and the per-slice replica are the same, and only the analytic field and the
// volume/boundary quadrature rule differ.

#pragma once

#include "MultiLevelSolution.hpp"

#include <string>

#include "Config.hpp"
#include "Discretization.hpp"

// ===========================================================================
// B1 / B3   PUBLISHED BENCHMARKS
//
//   FUNG_HON_EXAMPLE_43 -- M. Fung and R. Hon, Example 4.3: a
//   distributed-control parabolic tracking problem on the unit square with
//   homogeneous Neumann data and a closed-form solution whose adjoint vanishes
//   identically.
//
//   STOLL_BREITEN_2015 -- M. Stoll and T. Breiten, SIAM J. Sci. Comput. 37(1),
//   2015, B1-B29, section 6.1, distributed control.  A moving negative Gaussian
//   target on the unit square with homogeneous Dirichlet state data.  There is
//   no closed-form solution: it reports tracking and solver metrics, not errors
//   against an exact field.  A LEVEL-B comparison -- same continuous problem and
//   same all-at-once structure, but Q2 rather than the paper's Q1, and uniform
//   rather than trapezoidal objective time weights.
// ===========================================================================

enum class LiteratureBenchmark {
  NONE,
  FUNG_HON_EXAMPLE_43,
  STOLL_BREITEN_2015
};

/** Map a command-line spelling to a benchmark.  Returns false, leaving `out`
 *  untouched, if the name is not recognised. */
bool ParseLiteratureBenchmark(const std::string& name,
                              LiteratureBenchmark& out);

/** Overwrite the PROBLEM DATA of `config` with the benchmark's definition:
 *  domain, final time, diffusivity, control type and region, tracking region,
 *  the three data-field forms and their amplitudes, and the objective weights
 *  that the benchmark fixes.
 *
 *  Deliberately NOT overwritten, so that a convergence study can vary them
 *  from the command line: mesh.nx, mesh.nz, time.steps and alpha_u. */
void ApplyLiteratureBenchmark(LiteratureBenchmark benchmark,
                              ProblemConfig& config);

/** Run the benchmark: solve, then measure FEM-integrated space-time errors of
 *  y, p and u against the analytic solution and report them on stdout as one
 *  machine-readable EX20_BENCH record.  Returns a process exit status. */
int RunLiteratureBenchmark(LiteratureBenchmark benchmark,
                           const ProblemConfig& config,
                           femus::MultiLevelSolution& solution,
                           femus::Mesh* mesh, unsigned solution_type,
                           DistributedSolvePc pc_mode,
                           double spatial_tolerance,
                           const std::string& json_path);

// ===========================================================================
// B2-D / B2-N   MANUFACTURED SOLUTIONS
//
// Unlike the published benchmarks these have a NON-ZERO exact adjoint and
// control:
//
//   B2-D  DISTRIBUTED_COSINE        distributed control on all of Omega
//   B2-N  NEUMANN_BOUNDARY_COSINE   Neumann boundary control on Gamma_c
//
// Both are stated directly in this application's sign convention,
//
//     y_t - nu Lap y = f + eta u
//    -p_t - nu Lap p = alpha_track (y - y_d)
//     alpha_u u + eta p = 0
//
// so no textbook adjoint is introduced and no sign is flipped afterwards.
// ===========================================================================

enum class ManufacturedCase {
  NONE,
  DISTRIBUTED_COSINE,
  NEUMANN_BOUNDARY_COSINE,
  // Affine-in-time variants.  Backward Euler differences are exact on affine
  // data, and p* is built to vanish at t_{N+1} = T + dt, so the discrete
  // terminal row closes exactly too.  The residual is then purely spatial,
  // which is what makes a clean Q2 spatial ladder possible at modest N_t.
  DISTRIBUTED_AFFINE,
  NEUMANN_BOUNDARY_AFFINE
};

bool ParseManufacturedCase(const std::string& name, ManufacturedCase& out);

/** Install the manufactured problem: domain, T, the unit coefficients the case
 *  fixes, the control and tracking regions, and the three analytic data
 *  overrides.  Mesh resolution and step count stay under caller control. */
void ApplyManufacturedCase(ManufacturedCase manufactured, ProblemConfig& config);

/** Solve the manufactured case and report FEM-integrated errors of y, p and u
 *  against the exact fields -- volume norms for y and p, and for B2-N a TRUE
 *  boundary norm on Gamma_c for u -- plus the objective comparison, as one
 *  machine-readable EX20_MMS record. */
int RunManufacturedBenchmark(ManufacturedCase manufactured,
                             const ProblemConfig& config,
                             femus::MultiLevelSolution& solution,
                             femus::Mesh* mesh, unsigned solution_type,
                             DistributedSolvePc pc_mode,
                             double spatial_tolerance,
                             const std::string& json_path);
