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
#include "ExactOc3D.hpp"

#include <stdexcept>
#include <string>
#include <vector>

#include "Benchmarks.hpp"
#include "Config.hpp"
#include "Benchmark2D.hpp"
#include "ExactOcBenchmark.hpp"
#include "Math2025Benchmark.hpp"

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
  bool verify_convection = false;   // ex21
  bool convection_mms = false;      // ex21
  bool math2025_ex3 = false;        // ex21
  bool math2025_ex4 = false;        // ex21
  bool math2025_ex5 = false;        // ex21
  int errorfeedback = 0;            // ex21: 95 or 96
  bool solve_only = false;          // ex21: 2D preset data, production path
  bool domain_given = false;        // T9: --domain (2D syntax, refused in 3D)
  bool dimension3_requested = false;  // T9: --dimension 3 (presets may not override it)
  bool ny_given = false;            // T9: --ny is 3D-only
  // T10: the 4-number (x, storage-axis-1) region syntax, recorded so 3D can
  // refuse it rather than reinterpret it ...
  bool tracking_box_2d = false;     // --tracking-box
  bool control_box_2d = false;      // --control-box
  bool control_full_domain = false; // --control-full-domain (re-applied in 3D)
  bool region_add_2d = false;       // --tracking-box-add / --control-box-add
  bool source_add_2d = false;       // --source-box-add
  // ... and the 3D-only 6-number syntax (x0 x1 y0 y1 z0 z1), applied after
  // parsing (CommandLine: ApplyRegions3D) and refused outside 3D.
  bool domain3d_given = false;
  ProblemConfig::Domain domain3d;
  bool tracking_box3d_given = false, control_box3d_given = false;
  ProblemConfig::Box tracking_box3d, control_box3d;
  std::vector<ProblemConfig::Box> tracking_add3d, control_add3d, source_add3d;
  std::string neumann_face3d;       // T11N: --neumann-face3d xmin|xmax|ymin|ymax|zmin|zmax
  bool final_time_was_set = false;
  double requested_final_time = 0.0;
};

/** T10: six numbers x0 x1 y0 y1 z0 z1 (storage axes 0, 1, 2) into a 3D box. */
static ProblemConfig::Box ReadBox3D(char** argv, int& argument);

/** The 3D exact OC cases (ExactOc3D): T9 conv-sin-3d / conv-sin3-3d, T10
 *  natural-conv-3d, T11N neumann-conv-3d. */
static bool IsExact3DCase(const std::string& name) {
  return name == "conv-sin-3d" || name == "conv-sin3-3d" || name == "natural-conv-3d" ||
         name == "neumann-conv-3d";
}

/** T10: the 3D domain and regions, after every preset / exact case has run
 *  (so the final dimension is known).  The 4-number (x, axis-1) syntax is not
 *  reinterpreted in 3D: it is refused, except the full-domain box whose second
 *  pair equals BOTH the y and the z extents (the T9 command lines) -- that box
 *  is the full domain under either reading.  The 3D defaults are not implied:
 *  the tracking region and the control region must be given. */
static void ApplyRegions3D(ProblemConfig& config, const RawCommandLineOptions& raw,
                           bool exact3d) {
  const bool any3d = raw.domain3d_given || raw.tracking_box3d_given ||
                     raw.control_box3d_given || !raw.tracking_add3d.empty() ||
                     !raw.control_add3d.empty() || !raw.source_add3d.empty();
  if (config.dimension != 3u) {
    if (any3d || !raw.neumann_face3d.empty())
      throw std::invalid_argument(
          "--domain3d / --tracking-box3d* / --control-box3d* / --source-box3d-add /"
          " --neumann-face3d are 3D-only (--dimension 3)");
    return;
  }
  auto refuse = [](const std::string& what) {
    throw std::invalid_argument("3D: " + what);
  };
  // T11N: the Neumann control face, explicit (no silent default): ex21's
  // default control type is NEUMANN, so a 3D run must say --neumann-solve and
  // name the face; the exact cases set their own face.
  const bool neumann = config.control.type == ProblemConfig::ControlType::NEUMANN;
  if (!raw.neumann_face3d.empty() && (!neumann || exact3d))
    refuse(exact3d ? "the 3D exact OC cases own their control face (no --neumann-face3d)"
                   : "--neumann-face3d needs Neumann boundary control (--neumann-solve)");
  if (neumann && !exact3d) {
    if (!raw.neumann_control)
      refuse("give the control type explicitly (--distributed-solve or --neumann-solve);"
             " the 2D default (Neumann on the bottom edge) has no 3D meaning");
    BoxFace3D face;
    if (raw.neumann_face3d.empty() || !LookupBoxFace3D(raw.neumann_face3d, face))
      refuse("Neumann control needs one whole face: --neumann-face3d"
             " xmin|xmax|ymin|ymax|zmin|zmax");
    config.neumann_control_face = face.id;
    config.neumann_control_boundary = face.name;
  }
  if (exact3d) {
    if (any3d || raw.domain_given || raw.tracking_box_2d || raw.control_box_2d ||
        raw.region_add_2d || raw.source_add_2d)
      refuse("the 3D exact OC cases own their domain and regions (no region options)");
    return;
  }
  if (raw.domain_given)
    refuse("--domain is the 2D (x, z) syntax; use --domain3d x0 x1 y0 y1 z0 z1");
  if (raw.region_add_2d)
    refuse("--tracking-box-add / --control-box-add are 2D; use --tracking-box3d-add /"
           " --control-box3d-add x0 x1 y0 y1 z0 z1");
  if (raw.source_add_2d)
    refuse("--source-box-add is 2D; use --source-box3d-add x0 x1 y0 y1 z0 z1");
  if (raw.domain3d_given) config.domain = raw.domain3d;
  const ProblemConfig::Domain& d = config.domain;
  ProblemConfig::Box full;
  full.xmin = d.xmin; full.xmax = d.xmax;
  full.ymin = d.ymin; full.ymax = d.ymax;
  full.zmin = d.zmin; full.zmax = d.zmax;
  // a 4-number box is interpretation-free iff it is the full domain under both
  // readings of its second pair (y or z)
  auto unambiguous_full = [&](const ProblemConfig::Box& b) {
    return b.xmin == d.xmin && b.xmax == d.xmax && b.zmin == d.ymin && b.zmax == d.ymax &&
           b.zmin == d.zmin && b.zmax == d.zmax;
  };
  if (raw.tracking_box3d_given && raw.tracking_box_2d)
    refuse("give the tracking box once (--tracking-box3d or the full-domain --tracking-box)");
  if (raw.tracking_box3d_given)
    config.tracking_box = raw.tracking_box3d;
  else if (raw.tracking_box_2d) {
    if (!unambiguous_full(config.tracking_box))
      refuse("--tracking-box (4 numbers) is ambiguous in 3D; use --tracking-box3d"
             " x0 x1 y0 y1 z0 z1");
    config.tracking_box = full;
  } else
    refuse("an explicit tracking region is required (--tracking-box3d x0 x1 y0 y1 z0 z1);"
           " the 2D default box has no 3D meaning");
  if (neumann) {   // T11N: the control lives on the face, not in volume boxes
    if (raw.control_box3d_given || raw.control_full_domain || raw.control_box_2d ||
        !raw.control_add3d.empty())
      refuse("control regions (--control-box3d*, --control-full-domain) are for distributed"
             " control; Neumann control acts on the face --neumann-face3d");

    config.control.region = full;   // inert for Neumann (N = M_Gamma); keeps ValidateBox valid
  } else if (raw.control_box3d_given)
    config.control.region = raw.control_box3d;
  else if (raw.control_full_domain)
    config.control.region = full;
  else if (raw.control_box_2d) {
    if (!unambiguous_full(config.control.region))
      refuse("--control-box (4 numbers) is ambiguous in 3D; use --control-box3d"
             " x0 x1 y0 y1 z0 z1");
    config.control.region = full;
  } else if (!neumann)
    refuse("an explicit control region is required (--control-full-domain or"
           " --control-box3d x0 x1 y0 y1 z0 z1); the 2D default box has no 3D meaning");
  for (const ProblemConfig::Box& b : raw.tracking_add3d)
    config.additional_tracking_regions.push_back(b);
  for (const ProblemConfig::Box& b : raw.control_add3d)
    config.additional_control_regions.push_back(b);
  for (const ProblemConfig::Box& b : raw.source_add3d) config.source_regions.push_back(b);
}

static ProblemConfig::Box ReadBox3D(char** argv, int& argument) {
  ProblemConfig::Box box;
  box.xmin = std::stod(argv[++argument]);
  box.xmax = std::stod(argv[++argument]);
  box.ymin = std::stod(argv[++argument]);
  box.ymax = std::stod(argv[++argument]);
  box.zmin = std::stod(argv[++argument]);
  box.zmax = std::stod(argv[++argument]);
  return box;
}

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
    } else if (option == "--ny" && argument + 1 < argc) {
      // T9: 3D only -- elements along storage axis 1 (physical y)
      raw.ny_given = true;
      config.mesh.ny = static_cast<unsigned>(std::stoul(argv[++argument]));
    } else if (option == "--nz" && argument + 1 < argc) {
      config.mesh.nz = static_cast<unsigned>(std::stoul(argv[++argument]));
    } else if (option == "--control-box" && argument + 4 < argc) {
      raw.control_box_2d = true;
      raw.control_full_domain = false;
      raw.control_box3d_given = false;
      config.control.region.xmin = std::stod(argv[++argument]);
      config.control.region.xmax = std::stod(argv[++argument]);
      config.control.region.zmin = std::stod(argv[++argument]);
      config.control.region.zmax = std::stod(argv[++argument]);
    } else if (option == "--control-full-domain") {
      raw.control_full_domain = true;
      raw.control_box_2d = false;
      raw.control_box3d_given = false;
      config.control.region.xmin = config.domain.xmin;
      config.control.region.xmax = config.domain.xmax;
      config.control.region.zmin = config.domain.zmin;
      config.control.region.zmax = config.domain.zmax;
    } else if (option == "--write-region-fields") {
      options.region_fields = true;
    } else if (option == "--tracking-box" && argument + 4 < argc) {
      raw.tracking_box_2d = true;
      config.tracking_box.xmin = std::stod(argv[++argument]);
      config.tracking_box.xmax = std::stod(argv[++argument]);
      config.tracking_box.zmin = std::stod(argv[++argument]);
      config.tracking_box.zmax = std::stod(argv[++argument]);
    } else if (option == "--tracking-box-add" && argument + 4 < argc) {
      raw.region_add_2d = true;
      ProblemConfig::Box box;
      box.xmin = std::stod(argv[++argument]);
      box.xmax = std::stod(argv[++argument]);
      box.zmin = std::stod(argv[++argument]);
      box.zmax = std::stod(argv[++argument]);
      config.additional_tracking_regions.push_back(box);
    } else if (option == "--control-box-add" && argument + 4 < argc) {
      raw.region_add_2d = true;
      ProblemConfig::Box box;
      box.xmin = std::stod(argv[++argument]);
      box.xmax = std::stod(argv[++argument]);
      box.zmin = std::stod(argv[++argument]);
      box.zmax = std::stod(argv[++argument]);
      config.additional_control_regions.push_back(box);
    } else if (option == "--dirichlet-core-solver" && argument + 1 < argc) {
      config.dirichlet_core.explicitly_set = true;
      config.dirichlet_core.solver_explicit = true;
      // T8D: dense (reference) | iterative (matrix-free CG core; the default)
      const std::string value(argv[++argument]);
      if (value == "dense")
        config.dirichlet_core.solver = ProblemConfig::DirichletCoreConfig::Solver::DENSE;
      else if (value == "iterative")
        config.dirichlet_core.solver = ProblemConfig::DirichletCoreConfig::Solver::ITERATIVE;
      else
        throw std::invalid_argument("--dirichlet-core-solver must be dense or iterative");
    } else if (option == "--dirichlet-core-pc" && argument + 1 < argc) {
      config.dirichlet_core.explicitly_set = true;
      const std::string value(argv[++argument]);
      if (value == "none")
        config.dirichlet_core.pc = ProblemConfig::DirichletCoreConfig::Pc::NONE;
      else if (value == "rt")
        config.dirichlet_core.pc = ProblemConfig::DirichletCoreConfig::Pc::RT;
      else
        throw std::invalid_argument("--dirichlet-core-pc must be none or rt");
    } else if (option == "--dirichlet-core-rtol" && argument + 1 < argc) {
      config.dirichlet_core.explicitly_set = true;
      config.dirichlet_core.rtol = std::stod(argv[++argument]);
    } else if (option == "--dirichlet-core-maxit" && argument + 1 < argc) {
      config.dirichlet_core.explicitly_set = true;
      config.dirichlet_core.maxit = std::stoi(argv[++argument]);
    } else if (option == "--verify-dirichlet-core") {
      config.dirichlet_core.verify = true;
    } else if (option == "--source-box-add" && argument + 4 < argc) {
      raw.source_add_2d = true;
      // T7F: one more box of the source-region union (ProblemConfig::
      // source_regions; previously preset-only).  The source data are the
      // usual ones (--source amplitude), loaded as M_source I_h f.
      ProblemConfig::Box box;
      box.xmin = std::stod(argv[++argument]);
      box.xmax = std::stod(argv[++argument]);
      box.zmin = std::stod(argv[++argument]);
      box.zmax = std::stod(argv[++argument]);
      config.source_regions.push_back(box);
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
    // ---- ex21: dimension, domain, state BC, convection, convection gates ----
    } else if (option == "--dimension" && argument + 1 < argc) {
      const std::string value(argv[++argument]);
      if (value != "1" && value != "2" && value != "3")
        throw std::invalid_argument("--dimension must be 1, 2 or 3");
      config.dimension = value == "1" ? 1u : (value == "3" ? 3u : 2u);
      raw.dimension3_requested = value == "3";
    // ---- T10: 3D-only region syntax (x0 x1 y0 y1 z0 z1), see ApplyRegions3D
    } else if (option == "--domain3d" && argument + 6 < argc) {
      const ProblemConfig::Box b = ReadBox3D(argv, argument);
      raw.domain3d_given = true;
      raw.domain3d.xmin = b.xmin; raw.domain3d.xmax = b.xmax;
      raw.domain3d.ymin = b.ymin; raw.domain3d.ymax = b.ymax;
      raw.domain3d.zmin = b.zmin; raw.domain3d.zmax = b.zmax;
    } else if (option == "--tracking-box3d" && argument + 6 < argc) {
      raw.tracking_box3d = ReadBox3D(argv, argument);
      raw.tracking_box3d_given = true;
    } else if (option == "--tracking-box3d-add" && argument + 6 < argc) {
      raw.tracking_add3d.push_back(ReadBox3D(argv, argument));
    } else if (option == "--control-box3d" && argument + 6 < argc) {
      raw.control_box3d = ReadBox3D(argv, argument);
      raw.control_box3d_given = true;
      raw.control_full_domain = false;
      raw.control_box_2d = false;
    } else if (option == "--control-box3d-add" && argument + 6 < argc) {
      raw.control_add3d.push_back(ReadBox3D(argv, argument));
    } else if (option == "--neumann-face3d" && argument + 1 < argc) {
      raw.neumann_face3d = argv[++argument];   // T11N, resolved in ApplyRegions3D
    } else if (option == "--source-box3d-add" && argument + 6 < argc) {
      raw.source_add3d.push_back(ReadBox3D(argv, argument));
    } else if (option == "--domain" && argument + 4 < argc) {
      raw.domain_given = true;
      config.domain.xmin = std::stod(argv[++argument]);
      config.domain.xmax = std::stod(argv[++argument]);
      config.domain.zmin = std::stod(argv[++argument]);
      config.domain.zmax = std::stod(argv[++argument]);
    } else if (option == "--state-bc" && argument + 1 < argc) {
      const std::string value(argv[++argument]);
      if (value == "natural")
        config.state_bc = ProblemConfig::StateBoundaryCondition::NATURAL;
      else if (value == "dirichlet")
        config.state_bc =
            ProblemConfig::StateBoundaryCondition::HOMOGENEOUS_DIRICHLET;
      else
        throw std::invalid_argument("--state-bc must be natural or dirichlet");
    } else if (option == "--convection" && argument + 1 < argc) {
      const std::string value(argv[++argument]);
      if (value != "on" && value != "off")
        throw std::invalid_argument("--convection must be on or off");
      config.convection.enabled = value == "on";
    } else if (option == "--vx" && argument + 1 < argc) {
      config.convection.vx = std::stod(argv[++argument]);
    } else if (option == "--vy" && argument + 1 < argc) {
      config.convection.vy = std::stod(argv[++argument]);
    } else if (option == "--vz" && argument + 1 < argc) {
      // T9: third wind component (storage axis 2); Validate refuses a
      // nonzero value outside --dimension 3.
      config.convection.vz = std::stod(argv[++argument]);
    } else if (option == "--wind" && argument + 1 < argc) {
      // ex21: a named variable velocity field (Config.hpp); requires
      // --convection on.  Absent: the constant (--vx, --vy).
      const std::string wind(argv[++argument]);
      if (wind == "lp-recirculating")
        config.convection.field = WindLpRecirculating;
      else if (wind == "q2-divfree")
        config.convection.field = WindQ2Divfree;
      else
        throw std::invalid_argument("--wind takes lp-recirculating or q2-divfree");
      config.convection.field_name =
          wind == "lp-recirculating" ? "lp-recirculating" : "q2-divfree";
    } else if (option == "--convection-stabilization" && argument + 1 < argc) {
      // T6: none (default) | lps (Leveque-Pearson LPS, ex21 Q2 version).
      const std::string s(argv[++argument]);
      if (s == "none")
        config.convection.stabilization = ProblemConfig::ConvectionConfig::Stabilization::NONE;
      else if (s == "lps")
        config.convection.stabilization = ProblemConfig::ConvectionConfig::Stabilization::LPS;
      else
        throw std::invalid_argument("--convection-stabilization takes none or lps");
    } else if (option == "--lps-projection" && argument + 1 < argc) {
      // T6: p0 (default; LP's patch mean) | p1 (diagnostic variant).
      const std::string s(argv[++argument]);
      if (s == "p0")
        config.convection.lps_projection = ProblemConfig::ConvectionConfig::LpsProjection::P0;
      else if (s == "p1")
        config.convection.lps_projection = ProblemConfig::ConvectionConfig::LpsProjection::P1;
      else
        throw std::invalid_argument("--lps-projection takes p0 or p1");
    } else if (option == "--lift-no-affine-tracking") {
      // T3b DIAGNOSTIC ONLY (negative control): drop the affine tracking term
      // of inhomogeneous Dirichlet data (the pre-T3b behaviour).
      config.lift_affine_tracking = false;
    } else if (option == "--verify-convection") {
      raw.verify_convection = true;
    } else if (option == "--convection-mms" && argument + 1 < argc) {
      raw.convection_mms = true;
      options.mms_shape = argv[++argument];
      if (options.mms_shape != "sin" && options.mms_shape != "sin3")
        throw std::invalid_argument("--convection-mms takes sin or sin3");
    } else if (option == "--math2025-ex3") {
      raw.math2025_ex3 = true;
    } else if (option == "--math2025-ex4") {
      raw.math2025_ex4 = true;
    } else if (option == "--math2025-ex5") {
      raw.math2025_ex5 = true;
    } else if (option == "--ex5-geometry" && argument + 2 < argc) {
      options.ex5_width = std::stod(argv[++argument]);
      options.ex5_height = std::stod(argv[++argument]);
    } else if (option == "--ex5-initial" && argument + 1 < argc) {
      options.ex5_initial = argv[++argument];
    } else if (option == "--ex5-variant" && argument + 1 < argc) {
      options.ex5_variant = argv[++argument];
    } else if (option == "--solve-only") {
      raw.solve_only = true;
    } else if (option == "--errorfeedback-95") {
      raw.errorfeedback = 95;
    } else if (option == "--errorfeedback-96") {
      raw.errorfeedback = 96;
    } else if (option == "--ef-initial" && argument + 1 < argc) {
      options.ef_initial = argv[++argument];
    } else if (option == "--ex4-initial" && argument + 1 < argc) {
      options.ex4_initial = argv[++argument];
    } else if (option == "--ex3-initial" && argument + 1 < argc) {
      options.ex3_initial = argv[++argument];
    } else if (option == "--ex3-alpha-u" && argument + 1 < argc) {
      options.ex3_alpha_u = std::stod(argv[++argument]);
    } else if (option == "--ex3-coefficients" && argument + 2 < argc) {
      options.ex3_a = std::stod(argv[++argument]);
      options.ex3_v = std::stod(argv[++argument]);
    } else if (option == "--exact-oc" && argument + 1 < argc) {
      options.exact_oc = argv[++argument];
    } else if (option == "--exact-oc-alpha-u" && argument + 1 < argc) {
      options.exact_oc_alpha_u = std::stod(argv[++argument]);
    } else if (option == "--exact-oc-dump" && argument + 1 < argc) {
      options.exact_oc_dump = argv[++argument];
    } else if (option == "--exact-oc-nu" && argument + 1 < argc) {
      options.exact_oc_nu = std::stod(argv[++argument]);
    } else if (option == "--exact-oc-block-test") {
      options.exact_oc_block_test = true;
    } else if (option == "--exact-oc-production-tol") {
      options.exact_oc_production_tol = true;
    } else if (option == "--exact-oc-velocity" && argument + 2 < argc) {
      options.exact_oc_velocity_set = true;
      options.exact_oc_velocity[0] = std::stod(argv[++argument]);
      options.exact_oc_velocity[1] = std::stod(argv[++argument]);
    } else if (option == "--exact-oc-series" && argument + 1 < argc) {
      options.exact_oc_series = argv[++argument];
    } else if (option == "--exact-oc-snapshot" && argument + 1 < argc) {
      options.exact_oc_snapshot = argv[++argument];
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
  // T8D finalisation: --verify-dirichlet-core compares the DENSE core with the
  // matrix-free action, so it runs the dense reference unless the iterative
  // core was requested explicitly -- which is refused rather than ignored.
  if (config.dirichlet_core.verify) {
    if (config.dirichlet_core.solver_explicit &&
        config.dirichlet_core.solver == ProblemConfig::DirichletCoreConfig::Solver::ITERATIVE)
      throw std::invalid_argument(
          "--verify-dirichlet-core runs the dense reference core; do not combine it"
          " with --dirichlet-core-solver iterative");
    config.dirichlet_core.solver = ProblemConfig::DirichletCoreConfig::Solver::DENSE;
  }
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

  // ex21: the Mathematics 2025 Example 3 preset owns the plant and the
  // functional (alpha_u = eps = 1e-9 is the source's value and is not
  // overridable); nx, steps, T (default 20), scheme and pc stay free.
  if (raw.math2025_ex3 && raw.math2025_ex4)
    throw std::invalid_argument("--math2025-ex3 and --math2025-ex4 are exclusive");
  // Example 4: alpha_u stays under command-line control (the source has no
  // optimal-control epsilon for it); T defaults to the source's 15.
  if (raw.math2025_ex4)
    ApplyMath2025Example4(config, options.ex4_initial,
                          raw.final_time_was_set ? raw.requested_final_time
                                                 : 15.0);
  if (raw.math2025_ex3)
  {
    ApplyMath2025Example3(config, options.ex3_initial,
                          raw.final_time_was_set ? raw.requested_final_time
                                                 : 20.0);
    if (options.ex3_a > 0.0) config.diffusivity = options.ex3_a;
    if (options.ex3_v >= 0.0) config.convection.vx = options.ex3_v;
    if (options.ex3_alpha_u > 0.0) config.alpha_u = options.ex3_alpha_u;
  }

  // ex21 2D benchmarks (Benchmark2D.hpp).  Example 5 owns eps = 1e-7 and T
  // defaults to 10 (figure axes); ErrorFeedback 9.5/9.6 leave alpha_u to the
  // command line and T defaults to 20 (figure axes).
  if ((raw.math2025_ex3 || raw.math2025_ex4) + raw.math2025_ex5 + (raw.errorfeedback != 0) > 1)
    throw std::invalid_argument("the literature presets are mutually exclusive");
  if (raw.math2025_ex5)
    ApplyMath2025Example5(config, options.ex5_width, options.ex5_height,
                          options.ex5_initial,
                          raw.final_time_was_set ? raw.requested_final_time : 10.0,
                          options.ex5_variant);
  if (raw.errorfeedback != 0)
    ApplyErrorFeedback9x(config, raw.errorfeedback, options.ef_initial,
                         raw.final_time_was_set ? raw.requested_final_time : 20.0);

  // ex21 exact optimal-control solutions (ExactOcBenchmark.hpp): own the data,
  // alpha_u (--exact-oc-alpha-u) and T (default 1 / 2); nx, steps, scheme and
  // pc stay free.
  if (!options.exact_oc.empty()) {
    if (raw.math2025_ex3 || raw.math2025_ex4 || raw.math2025_ex5 ||
        raw.errorfeedback != 0)
      throw std::invalid_argument("--exact-oc excludes the literature presets");
    if (IsExact3DCase(options.exact_oc)) {
      // T9: the 3D exact case (ExactOc3D.hpp); the wind stays the CLI one
      if (options.exact_oc_velocity_set)
        throw std::invalid_argument(
            "the 3D exact cases take their wind from --convection on --vx --vy --vz");
      ApplyExactOptimalControl3D(config, options.exact_oc, options.exact_oc_alpha_u,
                                 raw.final_time_was_set ? raw.requested_final_time : -1.0,
                                 options.exact_oc_nu);
    } else
    ApplyExactOptimalControl(config, options.exact_oc, options.exact_oc_alpha_u,
                             raw.final_time_was_set ? raw.requested_final_time
                                                    : -1.0,
                             options.exact_oc_nu,
                             options.exact_oc_velocity_set ? options.exact_oc_velocity
                                                           : nullptr);
  }

  ApplyRegions3D(config, raw, IsExact3DCase(options.exact_oc));   // T10/T11N
  config.Validate();
  // ex21: a nonzero velocity without --convection on would be silently
  // ignored, and the ex20 batteries encode a symmetric diffusion operator.
  if (!config.convection.enabled &&
      (config.convection.vx != 0.0 || config.convection.vy != 0.0 ||
       config.convection.vz != 0.0))
    throw std::invalid_argument("--vx/--vy/--vz require --convection on");
  // T9/T10: 3D run modes and solvers (the feature whitelist is
  // ProblemConfig::Validate3D; the 3D region syntax is ApplyRegions3D).  Refuse, never reinterpret: the 1D/2D
  // presets and exact cases above overwrite config.dimension, so the test is on
  // what the command line REQUESTED, not on the (possibly reset) config.
  if (raw.dimension3_requested && config.dimension != 3u)
    throw std::invalid_argument(
        "3D: --dimension 3 requested, but the selected preset / exact case"
        " is 1D/2D (refused, not reinterpreted)");
  if (raw.ny_given && config.dimension != 3u)
    throw std::invalid_argument("--ny is 3D-only (in 2D the second axis is --nz)");
  if (config.dimension == 3u) {
    auto refuse = [](const std::string& what) {
      throw std::invalid_argument("3D: " + what);
    };
    if (raw.math2025_ex3 || raw.math2025_ex4 || raw.math2025_ex5 || raw.errorfeedback != 0)
      refuse("the Math2025 / ErrorFeedback presets are 1D/2D");
    if (options.benchmark != LiteratureBenchmark::NONE ||
        options.manufactured != ManufacturedCase::NONE)
      refuse("the literature / manufactured batteries are 2D");
    if (raw.convection_mms) refuse("--convection-mms is 2D");
    if (raw.verify_stage1 || raw.verify_sweeps || raw.verify_boundary_reduced)
      refuse("the ex20 verification batteries are 2D");
    if (!options.exact_oc.empty() && !IsExact3DCase(options.exact_oc))
      refuse("the only 3D exact OC cases are conv-sin-3d, conv-sin3-3d, natural-conv-3d and"
             " neumann-conv-3d");
    if (raw.pc_schur_matching || raw.pc_schur_boundary_reduced || raw.pc_schur_dirichlet_reduced)
      refuse("only SCHUR_SWEEP (--pc-schur-sweep) or ADDITIVE_GAMG (BE) are validated");
    if (options.pc_time_blocks > 0 || options.pc_sweep_depth > 0)
      refuse("the inexact temporal sweep prototypes are not ported");
    if (!raw.pc_schur_sweep && config.time.scheme == ProblemConfig::TimeScheme::CRANK_NICOLSON)
      refuse("Crank-Nicolson needs --pc-schur-sweep (ADDITIVE_GAMG is not robust under CN,"
             " T7F/T8D finalisation)");
  }
  if (config.convection.enabled &&
      (options.benchmark != LiteratureBenchmark::NONE ||
       options.manufactured != ManufacturedCase::NONE || raw.verify_stage1 ||
       raw.verify_sweeps || raw.verify_boundary_reduced))
    throw std::invalid_argument(
        "--convection on is not supported by the ex20 benchmark and"
        " verification batteries (they encode a symmetric diffusion operator);"
        " use --verify-convection / --convection-mms");
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
  // T7N: the boundary-reduced Schur solver factors SPD blocks (Cholesky,
  // MAT_SPD, A_plus/A_minus taken as their own transposes): refused under
  // convection, never silently replaced.
  if (raw.pc_schur_boundary_reduced && config.convection.enabled)
    throw std::invalid_argument(
        "--pc-schur-boundary-reduced is not available with --convection on (it"
        " assumes a symmetric spatial operator); use --pc-schur-sweep (ADDITIVE_GAMG"
        " and SCHUR_MATCHING are not robust for Neumann control, ex20 stage 2)");
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

  // The order of these tests is the run-mode precedence.  The two ex21 gates
  // come first: they never fall through to another mode.
  if (raw.verify_convection)
    options.run_mode = RunMode::VERIFY_CONVECTION;
  else if (raw.convection_mms)
    options.run_mode = RunMode::CONVECTION_MMS;
  else if (raw.math2025_ex3 || raw.math2025_ex4)
    options.run_mode = RunMode::MATH2025_EX3;   // shared driver of Ex. 3 and 4
  // --solve-only keeps a 2D preset's data but takes the PRODUCTION path (the
  // MPI-distributed solve + summary/JSON output) instead of the serial
  // benchmark driver and its gates.
  else if ((raw.math2025_ex5 || raw.errorfeedback != 0) && !raw.solve_only)
    options.run_mode = RunMode::BENCHMARK_2D;
  else if (!options.exact_oc.empty() && !raw.solve_only)
    options.run_mode = RunMode::EXACT_OC;
  else if (!options.operator_dump_prefix.empty())
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
