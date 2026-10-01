// Runtime options of one run: which run mode the command line selected, where
// output goes, what is written, which preconditioner is used.
//
// ProblemConfig (Config.hpp) carries the mathematics; ApplicationOptions
// carries everything else.  Both are filled by ParseCommandLine
// (CommandLine.cpp), and main() dispatches on `run_mode`.  Every member is a
// RESOLVED value: the raw command-line flags never leave the parser.

#pragma once

#include "FemusDefault.hpp"

#include <string>

#include "Benchmarks.hpp"
#include "Config.hpp"

enum class RunMode {
  PRODUCTION,
  VERIFY_STAGE1,
  VERIFY_SWEEPS,
  VERIFY_BOUNDARY_REDUCED,
  DUMP_OPERATORS,
  LITERATURE_BENCHMARK,
  MANUFACTURED_BENCHMARK
};

struct ApplicationOptions {
  RunMode run_mode = RunMode::PRODUCTION;
  DistributedSolvePc pc_mode = DistributedSolvePc::ADDITIVE_GAMG;

  double spatial_tolerance = 1.0e-10;        // CG tolerance of a spatial solve
  bool paraview_enabled = true;
  // Opt-in ParaView cell fields marking the discrete tracking / control
  // regions.  Off by default.
  bool region_fields = false;
  std::string paraview_order = "linear";
  std::string output_path = DEFAULT_OUTPUTDIR;
  std::string json_path;
  // Spatial operator export (serial, opt-in).
  std::string operator_dump_prefix;
  // Published-benchmark replication.  NONE keeps the production path.
  LiteratureBenchmark benchmark = LiteratureBenchmark::NONE;
  // Manufactured-solution validation.  NONE keeps the production path.
  ManufacturedCase manufactured = ManufacturedCase::NONE;
  // Output prefix of the --verify-stage1 battery: <output_path>/ex20_verify.
  std::string verification_prefix;

  // --- Stage 8D temporal-parallel prototypes (opt-in, OFF by default) ------
  // Both alter ONLY the inexactness of the triangular time sweeps used inside
  // the Schur preconditioner; the outer operator, the assembled system and the
  // solution they converge to are untouched.  0 means "disabled", and the
  // sweep then executes the production code path bit for bit.  They are
  // mutually exclusive.
  //
  //   pc_time_blocks P : temporal block Jacobi.  The N_t slots are cut into P
  //     contiguous blocks; the coupling across a block boundary is dropped, so
  //     the P blocks are independent.  Serial depth N_t/P, work N_t solves.
  //   pc_sweep_depth m : windowed (truncated-Neumann) sweep.  Every slot is
  //     recomputed from its own window of m predecessors started from zero, so
  //     all slots are independent.  Serial depth m, work ~ m N_t solves.
  unsigned pc_time_blocks = 0;
  unsigned pc_sweep_depth = 0;
};

/** Parse the command line into the problem data (`config`) and the resolved
 *  runtime options.  Installs a requested literature benchmark or manufactured
 *  case, validates the problem data and resolves every precedence rule.
 *  Throws std::invalid_argument on an unknown, incomplete or invalid option. */
ApplicationOptions ParseCommandLine(int argc, char** argv, ProblemConfig& config);
