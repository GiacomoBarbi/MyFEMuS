// Command line: argv -> the problem data (ProblemConfig) and the resolved
// runtime options (ApplicationOptions).
//
//   * run modes are not mutually exclusive.  When several are given the winner
//     is fixed by the resolution order in ResolveRunOptions:
//         --dump-operators > --verify-sweeps > --verify-stage1 > production
//   * --pc-schur-sweep beats --pc-schur-matching, in either order.
//   * --control-h1-time-weight <gamma> adds the temporal H1 seminorm of the
//     DIRICHLET control; the default 0 is the accepted code path.  A negative
//     value, and any positive value with distributed or Neumann control, are
//     rejected by ProblemConfig::Validate().
//   * --time-scheme be|cn selects the time discretization; absent, and with an
//     explicit "be", the run is the unchanged backward-Euler path.  "cn" is
//     accepted for every production solve (distributed, Neumann, Dirichlet).
//   * without any --pc-* flag the preconditioner is ADDITIVE_GAMG, except for
//     Dirichlet control, whose default is SCHUR_DIRICHLET_REDUCED (ADDITIVE
//     does not converge beyond the smallest Dirichlet problems).
//   * an unknown option and a known option missing its value produce the same
//     "unknown or incomplete option: ..." message.
//   * duplicate value options, and --control-box / --control-full-domain, are
//     last-occurrence-wins, so the else-if chain must stay ordered.
//   * exceptions are raised in statement order: the parser first, then
//     ResolveProblemData in the order written there.

#include "Options.hpp"

#include <stdexcept>
#include <string>

#include "Benchmarks.hpp"
#include "Config.hpp"

/** Parser scratch state: the raw flags whose meaning is only decided after the
 *  whole command line has been read.  Never leaves this translation unit. */
struct RawCommandLineOptions {
  bool distributed_control = false;
  bool neumann_control = false;
  bool dirichlet_control = false;
  bool pc_schur_matching = false;
  bool pc_schur_sweep = false;
  bool pc_schur_boundary_reduced = false;
  bool pc_schur_dirichlet_reduced = false;
  bool verify_stage1 = false;
  bool verify_sweeps = false;
  bool verify_boundary_reduced = false;
  bool final_time_was_set = false;
  double requested_final_time = 0.0;
};

/** Option order decides the winner for duplicates and for --control-box /
 *  --control-full-domain, so the else-if chain must stay ordered. */
static void ParseArguments(int argc, char** argv, ProblemConfig& config,
                           ApplicationOptions& options,
                           RawCommandLineOptions& raw) {
  for (int argument = 1; argument < argc; ++argument) {
    const std::string option(argv[argument]);
    if (option == "--distributed-solve") {
      raw.distributed_control = true;
    } else if (option == "--neumann-solve") {
      raw.neumann_control = true;
    } else if (option == "--dirichlet-solve") {
      raw.dirichlet_control = true;
    } else if (option == "--control-h1-length" && argument + 1 < argc) {
      config.control_h1_length = std::stod(argv[++argument]);
    } else if (option == "--control-h1-time-weight" && argument + 1 < argc) {
      // gamma_u of the temporal seminorm of the Dirichlet control; 0 (the
      // default) leaves the accepted code path untouched.
      config.control_h1_time_weight = std::stod(argv[++argument]);
    } else if (option == "--pc-schur-matching") {
      raw.pc_schur_matching = true;
    } else if (option == "--pc-schur-sweep") {
      // Exact triangular time sweeps inside the matching Schur preconditioner
      raw.pc_schur_sweep = true;
    } else if (option == "--pc-schur-boundary-reduced") {
      // Exact reduced Schur solve for Neumann boundary control
      raw.pc_schur_boundary_reduced = true;
    } else if (option == "--pc-schur-dirichlet-reduced") {
      // Exact reduced Schur solve for Dirichlet boundary control
      raw.pc_schur_dirichlet_reduced = true;
    } else if (option == "--pc-time-blocks" && argument + 1 < argc) {
      options.pc_time_blocks =
          static_cast<unsigned>(std::stoul(argv[++argument]));
    } else if (option == "--pc-sweep-depth" && argument + 1 < argc) {
      options.pc_sweep_depth =
          static_cast<unsigned>(std::stoul(argv[++argument]));
    } else if (option == "--spatial-tol" && argument + 1 < argc) {
      options.spatial_tolerance = std::stod(argv[++argument]);
    } else if (option == "--verify-stage1") {
      raw.verify_stage1 = true;
    } else if (option == "--verify-sweeps") {
      raw.verify_sweeps = true;
    } else if (option == "--verify-boundary-reduced") {
      raw.verify_boundary_reduced = true;
    } else if (option == "--dump-operators" && argument + 1 < argc) {
      options.operator_dump_prefix = argv[++argument];
    } else if (option == "--time-scheme" && argument + 1 < argc) {
      // Runtime time discretization.  Absent, the default stays backward
      // Euler; "be" must reproduce the default run exactly.
      const std::string scheme(argv[++argument]);
      if (scheme == "be")
        config.time.scheme = ProblemConfig::TimeScheme::BACKWARD_EULER;
      else if (scheme == "cn")
        config.time.scheme = ProblemConfig::TimeScheme::CRANK_NICOLSON;
      else
        throw std::invalid_argument(
            "--time-scheme must be be (backward Euler) or cn"
            " (Crank-Nicolson)");
    } else if (option == "--dt" && argument + 1 < argc) {
      config.time.dt = std::stod(argv[++argument]);
    } else if (option == "--steps" && argument + 1 < argc) {
      config.time.steps = static_cast<unsigned>(std::stoul(argv[++argument]));
    } else if (option == "--T" && argument + 1 < argc) {
      raw.requested_final_time = std::stod(argv[++argument]);
      raw.final_time_was_set = true;
    } else if (option == "--nx" && argument + 1 < argc) {
      config.mesh.nx = static_cast<unsigned>(std::stoul(argv[++argument]));
    } else if (option == "--nz" && argument + 1 < argc) {
      config.mesh.nz = static_cast<unsigned>(std::stoul(argv[++argument]));
    } else if (option == "--control-box" && argument + 4 < argc) {
      config.control.region.xmin = std::stod(argv[++argument]);
      config.control.region.xmax = std::stod(argv[++argument]);
      config.control.region.zmin = std::stod(argv[++argument]);
      config.control.region.zmax = std::stod(argv[++argument]);
    } else if (option == "--control-full-domain") {
      config.control.region.xmin = config.domain.xmin;
      config.control.region.xmax = config.domain.xmax;
      config.control.region.zmin = config.domain.zmin;
      config.control.region.zmax = config.domain.zmax;
    } else if (option == "--write-region-fields") {
      options.region_fields = true;
    } else if (option == "--tracking-box" && argument + 4 < argc) {
      config.tracking_box.xmin = std::stod(argv[++argument]);
      config.tracking_box.xmax = std::stod(argv[++argument]);
      config.tracking_box.zmin = std::stod(argv[++argument]);
      config.tracking_box.zmax = std::stod(argv[++argument]);
    } else if (option == "--tracking-box-add" && argument + 4 < argc) {
      ProblemConfig::Box box;
      box.xmin = std::stod(argv[++argument]);
      box.xmax = std::stod(argv[++argument]);
      box.zmin = std::stod(argv[++argument]);
      box.zmax = std::stod(argv[++argument]);
      config.additional_tracking_regions.push_back(box);
    } else if (option == "--control-box-add" && argument + 4 < argc) {
      ProblemConfig::Box box;
      box.xmin = std::stod(argv[++argument]);
      box.xmax = std::stod(argv[++argument]);
      box.zmin = std::stod(argv[++argument]);
      box.zmax = std::stod(argv[++argument]);
      config.additional_control_regions.push_back(box);
    } else if (option == "--manufactured-benchmark" && argument + 1 < argc) {
      const std::string manufactured_name(argv[++argument]);
      if (!ParseManufacturedCase(manufactured_name,
                                               options.manufactured))
        throw std::invalid_argument(
            "--manufactured-benchmark must be distributed, neumann, "
            "distributed-affine or neumann-affine");
    } else if (option == "--literature-benchmark" && argument + 1 < argc) {
      const std::string benchmark_name(argv[++argument]);
      if (!ParseLiteratureBenchmark(benchmark_name,
                                                  options.benchmark))
        throw std::invalid_argument(
            "--literature-benchmark must be fung-hon-43 or stoll-breiten-15");
    } else if (option == "--target-type" && argument + 1 < argc) {
      const std::string target_type(argv[++argument]);
      if (target_type == "constant")
        config.tracking.target_type =
            ProblemConfig::TrackingTargetType::CONSTANT;
      else if (target_type == "sin_x_minus_t")
        config.tracking.target_type = ProblemConfig::
            TrackingTargetType::SIN_X_MINUS_T;
      else if (target_type == "one_minus_y_sin_x_minus_t")
        config.tracking.target_type = ProblemConfig::
            TrackingTargetType::ONE_MINUS_Y_SIN_X_MINUS_T;
      else
        throw std::invalid_argument(
            "--target-type must be constant, sin_x_minus_t, or "
            "one_minus_y_sin_x_minus_t");
    } else if (option == "--target-param" && argument + 1 < argc) {
      config.tracking.parameter = std::stod(argv[++argument]);
    } else if (option == "--alpha-u" && argument + 1 < argc) {
      config.alpha_u = std::stod(argv[++argument]);
    } else if (option == "--alpha-track" && argument + 1 < argc) {
      config.alpha_track = std::stod(argv[++argument]);
    } else if ((option == "--nu" || option == "--diffusivity") &&
               argument + 1 < argc) {
      config.diffusivity = std::stod(argv[++argument]);
    } else if (option == "--eta" && argument + 1 < argc) {
      config.eta = std::stod(argv[++argument]);
    } else if (option == "--desired-state" && argument + 1 < argc) {
      config.desired_state = std::stod(argv[++argument]);
    } else if (option == "--initial-state" && argument + 1 < argc) {
      config.initial_state = std::stod(argv[++argument]);
    } else if (option == "--source" && argument + 1 < argc) {
      config.source = std::stod(argv[++argument]);
    } else if (option == "--neumann-flux" && argument + 1 < argc) {
      config.prescribed_neumann_flux = std::stod(argv[++argument]);
    } else if (option == "--dump-json" && argument + 1 < argc) {
      options.json_path = argv[++argument];
    } else if (option == "--paraview-out" && argument + 1 < argc) {
      options.output_path = argv[++argument];
    } else if (option == "--no-paraview") {
      options.paraview_enabled = false;
    } else if (option == "--paraview-order" && argument + 1 < argc) {
      const std::string order(argv[++argument]);
      if (order != "p1" && order != "q2")
        throw std::invalid_argument("--paraview-order must be p1 or q2");
      options.paraview_order = order == "p1" ? "linear" : "biquadratic";
    } else {
      throw std::invalid_argument("unknown or incomplete option: " + option);
    }
  }
}

/** Problem-data resolution and validation.  Statement order decides which
 *  exception is raised when several inputs are invalid. */
static void ResolveProblemData(const RawCommandLineOptions& raw,
                               const ApplicationOptions& options,
                               ProblemConfig& config) {
  if ((raw.distributed_control && raw.neumann_control) ||
      (raw.dirichlet_control && (raw.distributed_control || raw.neumann_control)))
    throw std::invalid_argument(
        "choose only one of --distributed-solve, --neumann-solve and"
        " --dirichlet-solve");
  if (raw.final_time_was_set) {
    if (raw.requested_final_time <= 0.0 || config.time.steps == 0)
      throw std::invalid_argument("--T and --steps must be positive");
    config.time.dt = raw.requested_final_time / config.time.steps;
  }
  if (raw.distributed_control)
    config.control.type = ProblemConfig::ControlType::DISTRIBUTED;
  else if (raw.dirichlet_control)
    config.control.type = ProblemConfig::ControlType::DIRICHLET;
  else
    config.control.type = ProblemConfig::ControlType::NEUMANN;
  // A literature benchmark defines its own problem data and overwrites
  // whatever the command line said about it: domain, final time, diffusivity,
  // control type and region, tracking region, source / initial state / target,
  // alpha_track and eta.  Mesh resolution, step count and alpha_u are left
  // under command-line control, because a convergence study varies exactly
  // those.  This runs before Validate() so the benchmark's data is what gets
  // validated.
  if (options.benchmark != LiteratureBenchmark::NONE)
    ApplyLiteratureBenchmark(options.benchmark, config);

  // A manufactured validation case likewise defines its own problem data, and
  // additionally installs the analytic data overrides.  It is applied after the
  // literature benchmark so that, if both were somehow requested, the run mode
  // resolved in ResolveRunOptions and the data installed here agree.
  if (options.manufactured != ManufacturedCase::NONE)
    ApplyManufacturedCase(options.manufactured, config);

  config.Validate();
  // Stage 1 of the Crank-Nicolson extension covers the PRODUCTION distributed
  // solve only.  The verification batteries, the literature benchmarks and the
  // manufactured cases all encode backward-Euler quadrature and reference
  // values, so CN is rejected for them instead of silently reusing them.
  if (config.time.scheme == ProblemConfig::TimeScheme::CRANK_NICOLSON) {
    if (options.benchmark != LiteratureBenchmark::NONE ||
        options.manufactured != ManufacturedCase::NONE)
      throw std::invalid_argument(
          "--time-scheme cn is not supported by the literature and"
          " manufactured benchmarks, which encode backward-Euler quadrature");
    if (raw.verify_stage1 || raw.verify_sweeps || raw.verify_boundary_reduced ||
        !options.operator_dump_prefix.empty())
      throw std::invalid_argument(
          "--time-scheme cn is not supported by the verification run modes,"
          " which encode backward-Euler operators");
    // Every preconditioner is available under Crank-Nicolson: the Neumann
    // boundary-reduced solver since stage 2 and the Dirichlet reduced solver
    // since stage 3.  Only the verification run modes and the benchmarks
    // above stay backward-Euler.
  }
  // The reduced Schur solve eliminates the control-inactive coordinates: it
  // needs a small control support, which only the Neumann boundary control
  // provides.  Checked on the resolved control type, after any benchmark.
  if (raw.pc_schur_boundary_reduced &&
      config.control.type != ProblemConfig::ControlType::NEUMANN)
    throw std::invalid_argument(
        "--pc-schur-boundary-reduced requires Neumann boundary control"
        " (--neumann-solve)");
  if (raw.pc_schur_dirichlet_reduced &&
      config.control.type != ProblemConfig::ControlType::DIRICHLET)
    throw std::invalid_argument(
        "--pc-schur-dirichlet-reduced requires Dirichlet boundary control"
        " (--dirichlet-solve)");
  // Stage 8D prototypes: two inequivalent inexact sweeps, never combined.
  if (options.pc_time_blocks > 0 && options.pc_sweep_depth > 0)
    throw std::invalid_argument(
        "--pc-time-blocks and --pc-sweep-depth are mutually exclusive"
        " temporal prototypes");
  if (options.pc_time_blocks > config.time.steps)
    throw std::invalid_argument(
        "--pc-time-blocks must not exceed the number of time steps");
}

/** Output paths, preconditioner and run-mode precedence.  Nothing here throws. */
static void ResolveRunOptions(const RawCommandLineOptions& raw,
                              const ProblemConfig& config,
                              ApplicationOptions& options) {
  if (options.json_path.empty()) {
    options.json_path = options.output_path;
    if (!options.json_path.empty() && options.json_path.back() != '/') options.json_path += '/';
    options.json_path += "ex20_fields.json";
  }

  options.verification_prefix = options.output_path;
  if (!options.verification_prefix.empty() &&
      options.verification_prefix.back() != '/')
    options.verification_prefix += '/';
  options.verification_prefix += "ex20_verify";

  // Precedence: dirichlet-reduced > boundary-reduced > sweep > matching >
  // the default, which is ADDITIVE_GAMG for distributed and Neumann control
  // and SCHUR_DIRICHLET_REDUCED for Dirichlet control.
  const DistributedSolvePc default_mode =
      config.control.type == ProblemConfig::ControlType::DIRICHLET
          ? DistributedSolvePc::SCHUR_DIRICHLET_REDUCED
          : DistributedSolvePc::ADDITIVE_GAMG;
  options.pc_mode =
      raw.pc_schur_dirichlet_reduced
          ? DistributedSolvePc::SCHUR_DIRICHLET_REDUCED
          : raw.pc_schur_boundary_reduced
          ? DistributedSolvePc::SCHUR_BOUNDARY_REDUCED
          : raw.pc_schur_sweep
                ? DistributedSolvePc::SCHUR_SWEEP
                : (raw.pc_schur_matching ? DistributedSolvePc::SCHUR_MATCHING
                                         : default_mode);

  // The order of these tests is the run-mode precedence.
  if (!options.operator_dump_prefix.empty())
    options.run_mode = RunMode::DUMP_OPERATORS;
  else if (raw.verify_sweeps)
    options.run_mode = RunMode::VERIFY_SWEEPS;
  else if (raw.verify_stage1)
    options.run_mode = RunMode::VERIFY_STAGE1;
  else if (raw.verify_boundary_reduced)
    options.run_mode = RunMode::VERIFY_BOUNDARY_REDUCED;
  else if (options.benchmark != LiteratureBenchmark::NONE)
    options.run_mode = RunMode::LITERATURE_BENCHMARK;
  else if (options.manufactured != ManufacturedCase::NONE)
    options.run_mode = RunMode::MANUFACTURED_BENCHMARK;
  else
    options.run_mode = RunMode::PRODUCTION;
}

ApplicationOptions ParseCommandLine(int argc, char** argv,
                                    ProblemConfig& config) {
  ApplicationOptions options;
  RawCommandLineOptions raw;
  ParseArguments(argc, argv, config, options, raw);
  ResolveProblemData(raw, options, config);
  ResolveRunOptions(raw, config, options);
  return options;
}
