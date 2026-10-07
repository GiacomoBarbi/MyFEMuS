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
  MANUFACTURED_BENCHMARK,
  // ex21 convection gates (ConvectionVerification.hpp)
  VERIFY_CONVECTION,
  CONVECTION_MMS,
  // ex21 literature benchmark: Mathematics 2025, Example 3 (Math2025Benchmark.hpp)
  MATH2025_EX3,
  // ex21 2D benchmarks: Mathematics 2025 Ex. 5, ErrorFeedback 9.5/9.6 (Benchmark2D.hpp)
  BENCHMARK_2D,
  // ex21 exact optimal-control solutions: Leveque-Pearson validation T1/T2
  // (ExactOcBenchmark.hpp)
  EXACT_OC
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
  // ex21 --convection-mms <shape>: "sin" (S = sin, f != 0 on the boundary
  // once v != 0) or "sin3" (S = sin^3, f = 0 on the boundary for every v).
  std::string mms_shape = "sin";
  // ex21 --ex3-initial: "source" (z0 = x(1-x), eq. (80)) or "zero".
  std::string ex3_initial = "source";
  // ex21 --ex3-coefficients a v: SENSITIVITY ONLY -- replaces the inherited
  // a = v = 2 of the Example 3 preset (negative = keep the preset value).
  double ex3_a = -1.0, ex3_v = -1.0;
  // ex21 --ex3-alpha-u: SENSITIVITY ONLY -- replaces the source's eps = 1e-9 of
  // the Example 3 preset (used for the colocated/noncolocated alpha_u sweep).
  double ex3_alpha_u = -1.0;
  // ex21 --ex4-initial: "zero" or "x1mx" (x(1-x)); Example 4 states no IC, so
  // both are sensitivity cases, neither is "the source's".
  std::string ex4_initial = "zero";
  // ex21 --ex5-geometry W H: Example 5 states neither (default pi x 1, [B]);
  // --ex5-initial: "xy" (plant IC of the cascade runs) or "zero".
  double ex5_width = 3.14159265358979323846, ex5_height = 1.0;
  std::string ex5_initial = "zero";
  // ex21 --ex5-variant: SENSITIVITY ONLY (source | d-zero | d-noswitch).
  std::string ex5_variant = "source";
  // ex21 --ef-initial: ErrorFeedback 9.5/9.6 state no IC: "zero", "sinsin"
  // (sin x sin pi y) or "poly" (16 x(pi-x) y(1-y)/pi^2) -- sensitivity cases.
  std::string ef_initial = "zero";

  // ex21 --exact-oc <case> (conv-sin3 | conv-sin | lp-heat), --exact-oc-alpha-u
  // (<= 0: the case default), --exact-oc-dump <prefix> (tiny meshes: write M,
  // L, E, E^T for aux/investigation/leveque_pearson/t0_cn_matrix_check.py).
  std::string exact_oc;
  double exact_oc_alpha_u = -1.0;
  std::string exact_oc_dump;
  // --exact-oc-production-tol: unmodified production solver policy (default:
  // tightened); --exact-oc-snapshot <csv>: lp52 t = 1 field snapshot.
  bool exact_oc_production_tol = false;
  std::string exact_oc_snapshot;
  // T6 (lp52, serial): per-slot y min/max with their locations, CSV; plus
  // <path>.argmin.csv, the whole y field at the slot of the all-time minimum.
  std::string exact_oc_series;
  // T4: --exact-oc-nu (lp52 epsilon override, <= 0 keeps 1/20) and
  // --exact-oc-block-test (isolated spatial-block GMRES+GAMG experiment
  // instead of the optimal-control solve).
  double exact_oc_nu = -1.0;
  // T7N: --exact-oc-velocity vx vy, the constant wind of the convective
  // manufactured cases (absent: each case's default)
  bool exact_oc_velocity_set = false;
  double exact_oc_velocity[2] = {0.0, 0.0};
  bool exact_oc_block_test = false;

  unsigned pc_time_blocks = 0;
  unsigned pc_sweep_depth = 0;
};

/** Parse the command line into the problem data (`config`) and the resolved
 *  runtime options.  Installs a requested literature benchmark or manufactured
 *  case, validates the problem data and resolves every precedence rule.
 *  Throws std::invalid_argument on an unknown, incomplete or invalid option. */
ApplicationOptions ParseCommandLine(int argc, char** argv, ProblemConfig& config);
