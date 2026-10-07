// Validation drivers: the published benchmarks and the manufactured solutions.
//
// Both families depend on the solver and nothing in the production path depends
// on them.  They share one measurement core, MeasureSpaceTime, because the
// Bochner error norms and the per-slice replica they need are the same; only
// the analytic field and the volume/boundary quadrature rule differ.

#include "Benchmarks.hpp"

#include "PetscMatrix.hpp"

#include <petscmat.h>
#include <petscvec.h>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "Output.hpp"
#include "Preconditioners.hpp"
#include "Solver.hpp"
#include "SpaceTimeOperators.hpp"

using namespace femus;

// ===========================================================================
// SHARED MEASUREMENT CORE
// ===========================================================================

/** Context handed to the quadrature callbacks: which analytic field, at which
 *  instant.  A plain struct, so no allocation happens inside the element loop.
 *  Each driver sets only the selector of its own family; the other stays NONE. */
struct ExactFieldContext {
  const ProblemConfig* config = nullptr;
  double time = 0.0;
  LiteratureBenchmark benchmark = LiteratureBenchmark::NONE;
  ManufacturedCase manufactured = ManufacturedCase::NONE;
};

/** Space-time error norms of one field, in the Bochner norms
 *
 *    L2(0,T;L2)   = sqrt( sum_n dt ||e(t_n)||^2_{L2(Omega)} )
 *    Linf(0,T;L2) = max_n ||e(t_n)||_{L2(Omega)}
 *
 *  with the same right-endpoint rule in time that weights the backward-Euler
 *  objective, and exact element quadrature in space. */
struct SpaceTimeNorms {
  double l2_l2 = 0.0;
  double linf_l2 = 0.0;
  double exact_l2_l2 = 0.0;
  double exact_linf_l2 = 0.0;
};

/** Walk the time slices of a space-time vector, replicate each one so that every
 *  rank can read the DOFs of the elements it owns, and integrate.  `boundary`
 *  selects the Gamma_c surface rule instead of the volume rule -- that is the
 *  only difference between the y/p norms and the Neumann u norm. */
static SpaceTimeNorms MeasureSpaceTime(const ProblemConfig& config,
                                ExactFieldContext& context,
                                SpatialScalarFunction exact, Vec space_time,
                                Mat mass, const OwnedSpatialLayout& layout,
                                Mesh* mesh, unsigned solution_type,
                                bool boundary) {
  SpaceTimeNorms norms;
  Vec spatial = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, MatCreateVecs(mass, &spatial, nullptr));
  context.config = &config;

  for (unsigned slot = 0; slot < config.time.steps; ++slot) {
    CopyDataSideSliceToSpatial(space_time, slot, layout.local_size, spatial);
    // Every rank needs the DOFs of every element it owns, including those it
    // does not own algebraically; the simplest correct way is a full replica.
    VecScatter scatter = nullptr;
    Vec replica = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD,
                   VecScatterCreateToAll(spatial, &scatter, &replica));
    PetscCallAbort(PETSC_COMM_WORLD,
                   VecScatterBegin(scatter, spatial, replica, INSERT_VALUES,
                                   SCATTER_FORWARD));
    PetscCallAbort(PETSC_COMM_WORLD,
                   VecScatterEnd(scatter, spatial, replica, INSERT_VALUES,
                                 SCATTER_FORWARD));
    const PetscScalar* values = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(replica, &values));

    context.time = (slot + 1u) * config.time.dt;
    const SpatialFieldIntegrals local =
        boundary ? IntegrateBoundaryField(mesh, solution_type,
                                          config.neumann_control_face, values,
                                          exact, &context)
                 : IntegrateSpatialField(mesh, solution_type, values, exact,
                                         &context);

    PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(replica, &values));
    PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&replica));
    PetscCallAbort(PETSC_COMM_WORLD, VecScatterDestroy(&scatter));

    double partial[2] = {local.error_squared, local.exact_squared};
    double total[2] = {0.0, 0.0};
    MPI_Allreduce(partial, total, 2, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
    const double err = std::sqrt(total[0] > 0.0 ? total[0] : 0.0);
    const double ref = std::sqrt(total[1] > 0.0 ? total[1] : 0.0);
    norms.l2_l2 += config.time.dt * total[0];
    norms.exact_l2_l2 += config.time.dt * total[1];
    if (err > norms.linf_l2) norms.linf_l2 = err;
    if (ref > norms.exact_linf_l2) norms.exact_linf_l2 = ref;
  }
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&spatial));
  norms.l2_l2 = std::sqrt(norms.l2_l2);
  norms.exact_l2_l2 = std::sqrt(norms.exact_l2_l2);
  return norms;
}

static double Relative(double error, double reference) {
  return reference > 0.0 ? error / reference : error;
}

/** Same spellings as PrintSolverStats. */
static const char* PcName(DistributedSolvePc mode) {
  if (mode == DistributedSolvePc::SCHUR_BOUNDARY_REDUCED)
    return "SCHUR_BOUNDARY_REDUCED";
  if (mode == DistributedSolvePc::SCHUR_SWEEP) return "SCHUR_SWEEP";
  if (mode == DistributedSolvePc::SCHUR_MATCHING) return "SCHUR_MATCHING";
  return "ADDITIVE";
}

// ---------------------------------------------------------------------------
// Benchmark catalogue
// ---------------------------------------------------------------------------

bool ParseLiteratureBenchmark(const std::string& name,
                              LiteratureBenchmark& out) {
  if (name == "fung-hon-43") {
    out = LiteratureBenchmark::FUNG_HON_EXAMPLE_43;
    return true;
  }
  if (name == "stoll-breiten-15") {
    out = LiteratureBenchmark::STOLL_BREITEN_2015;
    return true;
  }
  return false;
}

const char* LiteratureBenchmarkName(LiteratureBenchmark benchmark) {
  if (benchmark == LiteratureBenchmark::FUNG_HON_EXAMPLE_43)
    return "fung-hon-43";
  if (benchmark == LiteratureBenchmark::STOLL_BREITEN_2015)
    return "stoll-breiten-15";
  return "none";
}

/** Stoll and Breiten 2015, section 6.1, distributed control.
 *
 *  Stated by the paper:
 *      Omega = (0,1)^2,  T = 1,  y_t - laplacian y = u,  so nu = 1
 *      homogeneous (zero) Dirichlet state data on all of dOmega
 *      y_obs = -64 exp(-((x0 - t/2)^2 + (x1 - t/2)^2))
 *      beta in {1e-4, 1e-6, 1e-8}
 *
 *  NOT stated by the paper, and therefore ASSUMPTIONS of this benchmark:
 *      y_0 = 0               -- the paper carries y_0 only symbolically
 *      source f = 0          -- implied by the zero-Dirichlet statement
 *      tracking on all of Omega   -- Omega_1 is never specified
 *      distributed control on all of Omega -- Omega_2 is never specified,
 *                               though "distributed control (square matrix)"
 *                               implies it
 *  The emitted record prints these four so they are never mistaken for data.
 *
 *  In the paper's notation beta is the control regularization, which is
 *  alpha_u here, with alpha_track = 1 and eta = 1. */
static void ApplyStollBreiten2015(ProblemConfig& config) {
  config.domain.xmin = 0.0;
  config.domain.xmax = 1.0;
  config.domain.zmin = 0.0;
  config.domain.zmax = 1.0;

  // T = 1; the step count stays under command-line control.
  config.time.dt = 1.0 / static_cast<double>(config.time.steps);

  config.diffusivity = 1.0;
  config.alpha_track = 1.0;
  config.eta = 1.0;

  config.control.type = ProblemConfig::ControlType::DISTRIBUTED;
  config.control.region.xmin = config.domain.xmin;
  config.control.region.xmax = config.domain.xmax;
  config.control.region.zmin = config.domain.zmin;
  config.control.region.zmax = config.domain.zmax;

  config.tracking_box.xmin = config.domain.xmin;
  config.tracking_box.xmax = config.domain.xmax;
  config.tracking_box.zmin = config.domain.zmin;
  config.tracking_box.zmax = config.domain.zmax;

  config.tracking.target_type =
      ProblemConfig::TrackingTargetType::MOVING_GAUSSIAN;

  config.data.source_form = ProblemConfig::DataFieldForm::CONSTANT;
  config.source = 0.0;
  config.data.initial_form = ProblemConfig::DataFieldForm::CONSTANT;
  config.initial_state = 0.0;

  config.state_bc = ProblemConfig::StateBoundaryCondition::HOMOGENEOUS_DIRICHLET;
}

void ApplyLiteratureBenchmark(LiteratureBenchmark benchmark,
                              ProblemConfig& config) {
  if (benchmark == LiteratureBenchmark::STOLL_BREITEN_2015) {
    ApplyStollBreiten2015(config);
    return;
  }
  if (benchmark != LiteratureBenchmark::FUNG_HON_EXAMPLE_43) return;

  // Fung/Hon Example 4.3.
  //
  //   Omega = (0,1)^2,  T = 1,  a = 1e-3,  homogeneous Neumann on all of dOmega
  //   f(x,t) = (8 pi^2 a - 1) e^{-t} cos(2 pi x1) cos(2 pi x2)
  //   g(x,t) =                e^{-t} cos(2 pi x1) cos(2 pi x2)
  //   exact:  y = e^{-t} cos(2 pi x1) cos(2 pi x2),   p = 0
  //
  // Verified algebraically, not copied on trust:
  //   Delta y = -8 pi^2 y, hence y_t - a Delta y = (-1 + 8 pi^2 a) y = f.  ok
  //   dy/dx1 ~ sin(2 pi x1) vanishes at x1 = 0 and 1, same in x2, so the
  //   homogeneous Neumann condition is satisfied exactly.  ok
  //   p == 0 turns the adjoint equation -p_t - a Delta p + (y - g) = 0 into
  //   y = g, which the published g indeed equals.  ok
  //
  // The paper does not state y_0; it is forced to be y(.,0) = cos cos.
  config.domain.xmin = 0.0;
  config.domain.xmax = 1.0;
  config.domain.zmin = 0.0;
  config.domain.zmax = 1.0;

  // T = 1 with a uniform step; the number of steps stays under caller control.
  config.time.dt = 1.0 / static_cast<double>(config.time.steps);

  config.diffusivity = 1.0e-3;

  // Distributed control acting on the whole domain, so that N = M.
  config.control.type = ProblemConfig::ControlType::DISTRIBUTED;
  config.control.region.xmin = config.domain.xmin;
  config.control.region.xmax = config.domain.xmax;
  config.control.region.zmin = config.domain.zmin;
  config.control.region.zmax = config.domain.zmax;

  // Tracking over the whole domain, so that M_omega = M.
  config.tracking_box.xmin = config.domain.xmin;
  config.tracking_box.xmax = config.domain.xmax;
  config.tracking_box.zmin = config.domain.zmin;
  config.tracking_box.zmax = config.domain.zmax;

  config.tracking.target_type =
      ProblemConfig::TrackingTargetType::EXP_MINUS_T_COS2PI;
  config.desired_state = 1.0;

  config.data.source_form = ProblemConfig::DataFieldForm::EXP_MINUS_T_COS2PI;
  config.source = 8.0 * kPi * kPi * config.diffusivity - 1.0;

  config.data.initial_form = ProblemConfig::DataFieldForm::EXP_MINUS_T_COS2PI;
  config.initial_state = 1.0;

  // Fung/Hon's objective is (1/2)||y-g||^2 + (gamma/2)||u||^2, i.e. unit
  // tracking weight and alpha_u = gamma.  alpha_u is left to the caller.
  config.alpha_track = 1.0;
  config.eta = 1.0;
}

double BenchmarkExactState(LiteratureBenchmark benchmark,
                           const ProblemConfig&, double x, double z,
                           double time) {
  if (benchmark == LiteratureBenchmark::FUNG_HON_EXAMPLE_43)
    return UnitCosineMode(x, z, time);
  return 0.0;
}

double BenchmarkExactAdjoint(LiteratureBenchmark, const ProblemConfig&, double,
                             double, double) {
  // Fung/Hon Example 4.3 has p == 0, and p_ex20 = -p_paper leaves that alone.
  return 0.0;
}

double BenchmarkExactControl(LiteratureBenchmark benchmark,
                             const ProblemConfig& config, double x, double z,
                             double time) {
  // Stationarity: alpha_u u + eta p = 0.
  return -(config.eta / config.alpha_u) *
         BenchmarkExactAdjoint(benchmark, config, x, z, time);
}

// ===========================================================================
// B1 / B3   PUBLISHED BENCHMARKS
// ===========================================================================

static double ExactStateCallback(double x, double z, void* context) {
  const ExactFieldContext& c = *static_cast<ExactFieldContext*>(context);
  return BenchmarkExactState(c.benchmark, *c.config, x, z, c.time);
}

static double ExactAdjointCallback(double x, double z, void* context) {
  const ExactFieldContext& c = *static_cast<ExactFieldContext*>(context);
  return BenchmarkExactAdjoint(c.benchmark, *c.config, x, z, c.time);
}

static double ExactControlCallback(double x, double z, void* context) {
  const ExactFieldContext& c = *static_cast<ExactFieldContext*>(context);
  return BenchmarkExactControl(c.benchmark, *c.config, x, z, c.time);
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Stoll-Breiten metrics
//
// Every norm below is the space-time norm induced by a spatial operator A:
//
//     ||v||_A = sqrt( sum_k w_k  v_k^T A v_k )
//
// with w_k = dt for the native metric and w_k = dt * d_k, d = (1/2,1,...,1,1/2),
// for the paper-weight analogue.  The paper's own objective uses the
// trapezoidal weights; this application's KKT system uses the uniform ones and
// is NOT changed here -- the trapezoidal figure is reported alongside, as a
// metric only.
// ---------------------------------------------------------------------------

/** sum_k w_k v_k^T A v_k, reduced across the communicator.
 *  `trapezoid` selects the (1/2, 1, ..., 1, 1/2) endpoint weights. */
static double SpaceTimeQuadratic(Vec field, Mat op, const OwnedSpatialLayout& layout,
                          unsigned steps, double dt, bool trapezoid) {
  Vec spatial = nullptr, product = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, MatCreateVecs(op, &spatial, &product));
  double local = 0.0;
  for (unsigned slot = 0; slot < steps; ++slot) {
    CopyDataSideSliceToSpatial(field, slot, layout.local_size, spatial);
    PetscCallAbort(PETSC_COMM_WORLD, MatMult(op, spatial, product));
    PetscScalar dot = 0.0;
    PetscCallAbort(PETSC_COMM_WORLD, VecDot(spatial, product, &dot));
    double w = dt;
    if (trapezoid && (slot == 0u || slot + 1u == steps)) w *= 0.5;
    local += w * PetscRealPart(dot);
  }
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&product));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&spatial));
  return local;
}

/** The tracking target sampled on the owned space-time layout, at the same
 *  end-of-interval instants the state equation uses. */
static void FillTargetField(const ProblemConfig& config,
                     const OwnedSpatialLayout& layout, unsigned steps, Vec out) {
  PetscScalar* values = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, VecGetArray(out, &values));
  for (unsigned slot = 0; slot < steps; ++slot) {
    const double time = (slot + 1u) * config.time.dt;
    const PetscInt offset = static_cast<PetscInt>(slot) * layout.local_size;
    for (PetscInt i = 0; i < layout.local_size; ++i)
      values[offset + i] =
          EvaluateTrackingTarget(config, layout.x[i], layout.z[i], time);
  }
  PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArray(out, &values));
}

static int RunStollBreiten2015(const ProblemConfig& config,
                        MultiLevelSolution& solution, Mesh* mesh,
                        unsigned solution_type, DistributedSolvePc pc_mode,
                        double spatial_tolerance,
                        const std::string& json_path) {
  const double t_total0 = MPI_Wtime();
  SpatialOperators spatial(config, mesh, solution_type, solution);
  SpaceTimeOperators spacetime(config, spatial);
  Mat mass = spatial.mass;
  Mat tracking = spatial.tracking;
  Mat control = spatial.control;
  const OwnedSpatialLayout& layout = spatial.layout;

  DistributedOptimalControlSolve solve = SolveOptimalControl(
      config, spatial, spacetime,
      MakeKktSolverPolicy(pc_mode, spatial_tolerance));

  // --- metrics -------------------------------------------------------------
  Vec target = nullptr, difference = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(solve.fields.state, &target));
  PetscCallAbort(PETSC_COMM_WORLD,
                 VecDuplicate(solve.fields.state, &difference));
  FillTargetField(config, layout, config.time.steps, target);
  PetscCallAbort(PETSC_COMM_WORLD, VecCopy(solve.fields.state, difference));
  PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(difference, -1.0, target));

  const unsigned n = config.time.steps;
  const double dt = config.time.dt;
  double local[6];
  local[0] = SpaceTimeQuadratic(difference, tracking, layout, n, dt, false);
  local[1] = SpaceTimeQuadratic(target, tracking, layout, n, dt, false);
  local[2] = SpaceTimeQuadratic(difference, tracking, layout, n, dt, true);
  local[3] = SpaceTimeQuadratic(target, tracking, layout, n, dt, true);
  local[4] = SpaceTimeQuadratic(solve.fields.state, mass, layout, n, dt, false);
  local[5] = SpaceTimeQuadratic(solve.fields.adjoint, mass, layout, n, dt,
                                false);
  double global[6];
  MPI_Allreduce(local, global, 6, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
  double u_local =
      SpaceTimeQuadratic(solve.fields.control, control, layout, n, dt, false);
  double u_global = 0.0;
  MPI_Allreduce(&u_local, &u_global, 1, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);

  PetscReal diff_eucl = 0.0, target_eucl = 0.0;
  PetscCallAbort(PETSC_COMM_WORLD, VecNorm(difference, NORM_2, &diff_eucl));
  PetscCallAbort(PETSC_COMM_WORLD, VecNorm(target, NORM_2, &target_eucl));

  const double rel_tracking =
      global[1] > 0.0 ? std::sqrt(global[0] / global[1]) : 0.0;
  const double rel_tracking_trap =
      global[3] > 0.0 ? std::sqrt(global[2] / global[3]) : 0.0;
  const double rel_tracking_eucl =
      target_eucl > 0.0 ? diff_eucl / target_eucl : 0.0;

  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&difference));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&target));

  if (!json_path.empty()) {
    const RootDistributedFields fields = GatherDistributedSolveToRoot(
        solve, layout, mass, config.time.steps);
    WriteParallelSolveJsonRankZero(json_path, config, solve, fields);
  }

  PetscInt spatial_global = 0;
  PetscCallAbort(PETSC_COMM_WORLD, MatGetSize(mass, &spatial_global, nullptr));
  solve.timings.total = MPI_Wtime() - t_total0;

  if (mesh->processor_id() == 0) {
    // y0, source, and the two full-domain regions are ASSUMPTIONS: the paper
    // does not specify them.  They are printed so a reader never mistakes them
    // for published data.
    std::printf(
        "EX20_SB15 control=distributed nx=%u nz=%u dofs=%d nt=%u dt=%.17g "
        "T=%.17g nu=%.17g alpha_track=%.17g beta=%.17g eta=%.17g "
        "y0=%.17g source=%.17g bc=dirichlet0 "
        "assumed_y0=1 assumed_source=1 assumed_full_tracking=1 "
        "assumed_full_control=1 pc=%s "
        "rel_tracking=%.17g rel_tracking_trap=%.17g rel_tracking_eucl=%.17g "
        "y_norm=%.17g u_norm=%.17g p_norm=%.17g yd_norm=%.17g "
        "J_track=%.17g J_reg=%.17g J_total=%.17g "
        "state_res=%.17g adjoint_res=%.17g stationarity_res=%.17g "
        "reduced_res=%.17g active_dofs=%lld "
        "reason=%d outer=%d inner_y=%lld inner_p=%lld "
        "right_iters=%lld left_iters=%lld ay_iters=%lld "
        "sweeps=%d sweep_failures=%lld "
        "operators_time=%.17g pc_setup_time=%.17g solve_time=%.17g "
        "total_time=%.17g\n",
        config.mesh.nx, config.mesh.nz, static_cast<int>(spatial_global),
        config.time.steps, config.time.dt, config.final_time(),
        config.diffusivity, config.alpha_track, config.alpha_u, config.eta,
        config.initial_state, config.source, PcName(pc_mode),
        rel_tracking, rel_tracking_trap, rel_tracking_eucl,
        std::sqrt(global[4]), std::sqrt(u_global), std::sqrt(global[5]),
        std::sqrt(global[1]),
        solve.fields.J_track, solve.fields.J_reg,
        solve.fields.J_track + solve.fields.J_reg,
        solve.kkt.state_rel, solve.kkt.adjoint_rel,
        solve.kkt.stationarity_abs, solve.kkt.reduced_abs,
        solve.kkt.active_dofs,
        solve.reason, solve.outer_iterations, solve.inner_y_iterations,
        solve.inner_p_iterations, solve.right_inner.iterations,
        solve.left_inner.iterations, solve.ay_inner.iterations,
        solve.sweeps_used ? 1 : 0, solve.sweep_failures,
        solve.timings.operators, solve.timings.pc_setup,
        solve.timings.ksp_solve, solve.timings.total);
  }

  const int status = solve.accepted ? 0 : 8;
  return status;
}

int RunLiteratureBenchmark(LiteratureBenchmark benchmark,
                           const ProblemConfig& config,
                           MultiLevelSolution& solution, Mesh* mesh,
                           unsigned solution_type, DistributedSolvePc pc_mode,
                           double spatial_tolerance,
                           const std::string& json_path) {
  if (benchmark == LiteratureBenchmark::STOLL_BREITEN_2015)
    return RunStollBreiten2015(config, solution, mesh, solution_type, pc_mode,
                               spatial_tolerance, json_path);

  const double t_total0 = MPI_Wtime();
  SpatialOperators spatial(config, mesh, solution_type, solution);
  SpaceTimeOperators spacetime(config, spatial);
  Mat mass = spatial.mass;
  Mat tracking = spatial.tracking;
  Mat control = spatial.control;
  const OwnedSpatialLayout& layout = spatial.layout;

  DistributedOptimalControlSolve solve = SolveOptimalControl(
      config, spatial, spacetime,
      MakeKktSolverPolicy(pc_mode, spatial_tolerance));

  ExactFieldContext context;
  context.benchmark = benchmark;
  const SpaceTimeNorms state = MeasureSpaceTime(
      config, context, ExactStateCallback, solve.fields.state, mass, layout,
      mesh, solution_type, false);
  const SpaceTimeNorms adjoint = MeasureSpaceTime(
      config, context, ExactAdjointCallback, solve.fields.adjoint, mass,
      layout, mesh, solution_type, false);
  const SpaceTimeNorms control_norms = MeasureSpaceTime(
      config, context, ExactControlCallback, solve.fields.control, mass,
      layout, mesh, solution_type, false);

  if (!json_path.empty()) {
    const RootDistributedFields fields = GatherDistributedSolveToRoot(
        solve, layout, mass, config.time.steps);
    WriteParallelSolveJsonRankZero(json_path, config, solve, fields);
  }

  solve.timings.total = MPI_Wtime() - t_total0;

  if (mesh->processor_id() == 0) {
    const double h = (config.domain.xmax - config.domain.xmin) /
                     static_cast<double>(config.mesh.nx);
    // One machine-readable record per run; the convergence driver parses this
    // and nothing else.  Errors are FEM-integrated (element quadrature against
    // the analytic field), not nodal differences.
    std::printf(
        "EX20_BENCH benchmark=%s nx=%u nz=%u h=%.17g steps=%u dt=%.17g "
        "T=%.17g nu=%.17g alpha_u=%.17g alpha_track=%.17g pc=%s "
        "reason=%d outer=%d true_residual=%.17g "
        "y_l2l2=%.17g y_l2l2_rel=%.17g y_linf_l2=%.17g y_linf_l2_rel=%.17g "
        "p_l2l2=%.17g p_linf_l2=%.17g u_l2l2=%.17g u_linf_l2=%.17g "
        "y_exact_l2l2=%.17g J_total=%.17g J_track=%.17g J_reg=%.17g "
        "kkt_state_rel=%.17g kkt_adjoint_rel=%.17g kkt_stationarity_rel=%.17g "
        "t_total=%.17g\n",
        LiteratureBenchmarkName(benchmark), config.mesh.nx, config.mesh.nz, h,
        config.time.steps, config.time.dt, config.final_time(),
        config.diffusivity, config.alpha_u, config.alpha_track,
        PcName(pc_mode), solve.reason, solve.outer_iterations,
        solve.true_residual, state.l2_l2,
        Relative(state.l2_l2, state.exact_l2_l2), state.linf_l2,
        Relative(state.linf_l2, state.exact_linf_l2), adjoint.l2_l2,
        adjoint.linf_l2, control_norms.l2_l2, control_norms.linf_l2,
        state.exact_l2_l2, solve.fields.J_track + solve.fields.J_reg,
        solve.fields.J_track, solve.fields.J_reg, solve.kkt.state_rel,
        solve.kkt.adjoint_rel, solve.kkt.stationarity_rel,
        solve.timings.total);
    std::fflush(stdout);
  }

  PrintSolverStats(config, solve, pc_mode);

  const int success = solve.accepted ? 0 : 8;
  return success;
}

// ---------------------------------------------------------------------------
// The analytic fields.
//
// Written exactly as derived in the report.  Every formula below was verified
// symbolically -- all three continuous KKT residuals, the terminal condition
// and every boundary flux reduce to zero identically -- before it was typed
// here.
// ---------------------------------------------------------------------------

static const double kFourPiSq = 4.0 * kPi * kPi;    // -Lap cos(2 pi x)          in 1-D
static const double kEightPiSq = 8.0 * kPi * kPi;   // -Lap cos(2pi x)cos(2pi z) in 2-D
static const double kFivePiSq = 5.0 * kPi * kPi;    // -Lap cos(2pi x)cos(pi z)

/** phi = cos(2 pi x) cos(2 pi z),  the B2-D spatial mode.  -Lap phi = 8 pi^2 phi. */
static double ModePhi(double x, double z) {
  return std::cos(2.0 * kPi * x) * std::cos(2.0 * kPi * z);
}

/** psi = cos(2 pi x) cos(pi z),  the B2-N adjoint mode.  -Lap psi = 5 pi^2 psi.
 *  d psi/dz ~ sin(pi z) vanishes at BOTH z=0 and z=1, so psi is homogeneous
 *  Neumann on the whole boundary -- which the adjoint requires. */
static double ModePsi(double x, double z) {
  return std::cos(2.0 * kPi * x) * std::cos(kPi * z);
}

static double ModeCos2PiX(double x) { return std::cos(2.0 * kPi * x); }

/** q(z) = z^2 - 2z.  q'(0) = -2 gives the controlled flux on Gamma_c; q'(1) = 0
 *  makes the opposite face flux-free.  q'' = 2 is the source contribution. */
static double ShapeQ(double z) { return z * z - 2.0 * z; }

// --- B2-D -----------------------------------------------------------------

static double DistributedState(const ProblemConfig&, double x, double z, double t) {
  return std::exp(-t) * ModePhi(x, z);
}

static double DistributedAdjoint(const ProblemConfig& config, double x, double z,
                          double t) {
  return (config.final_time() - t) * std::exp(-t) * ModePhi(x, z);
}

/** f = y_t - nu Lap y - eta u = e^{-t} phi [ (8 pi^2 - 1) + (T - t) ]. */
static double DistributedSource(const ProblemConfig& config, double x, double z,
                         double t) {
  const double T = config.final_time();
  return std::exp(-t) * ModePhi(x, z) * ((kEightPiSq - 1.0) + (T - t));
}

/** y_d = y - (-p_t - nu Lap p)/alpha_track = -(1 + 8 pi^2)(T - t) e^{-t} phi. */
static double DistributedTarget(const ProblemConfig& config, double x, double z,
                         double t) {
  const double T = config.final_time();
  return -(1.0 + kEightPiSq) * (T - t) * std::exp(-t) * ModePhi(x, z);
}

static double DistributedInitial(const ProblemConfig& config, double x, double z) {
  return DistributedState(config, x, z, 0.0);
}

// --- B2-N -----------------------------------------------------------------

/** y = -(1/2)(T - t) e^{-t} cos(2 pi x) q(z).
 *  nu d_n y = eta u on z=0, and zero on the other three sides. */
static double NeumannState(const ProblemConfig& config, double x, double z, double t) {
  const double T = config.final_time();
  return -0.5 * (T - t) * std::exp(-t) * ModeCos2PiX(x) * ShapeQ(z);
}

static double NeumannAdjoint(const ProblemConfig& config, double x, double z,
                      double t) {
  return (config.final_time() - t) * std::exp(-t) * ModePsi(x, z);
}

/** f = y_t - nu Lap y.  The control enters through the BOUNDARY, so it does
 *  NOT appear here -- that asymmetry with B2-D is the point of the case. */
static double NeumannSource(const ProblemConfig& config, double x, double z,
                     double t) {
  const double T = config.final_time();
  return 0.5 * std::exp(-t) * ModeCos2PiX(x) *
         (ShapeQ(z) * (1.0 + (1.0 - kFourPiSq) * (T - t)) + 2.0 * (T - t));
}

/** y_d = y - (-p_t - nu Lap p)/alpha_track. */
static double NeumannTarget(const ProblemConfig& config, double x, double z,
                     double t) {
  const double T = config.final_time();
  return -std::exp(-t) * ModeCos2PiX(x) *
         (0.5 * (T - t) * ShapeQ(z) +
          std::cos(kPi * z) * (1.0 + (1.0 + kFivePiSq) * (T - t)));
}

static double NeumannInitial(const ProblemConfig& config, double x, double z) {
  return NeumannState(config, x, z, 0.0);
}

// --- affine-in-time variants (clean spatial ladder) -------------------------
//
// s(t) = T + dt - t is affine and vanishes at t_{N+1} = T + dt, so BOTH the
// backward-Euler differences and the implicit terminal condition p^{N+1} = 0
// are satisfied exactly.  Derivation: aux/manuscript/stage8/
// SPATIAL_MANUFACTURED_DERIVATION.md.

static double AffineS(const ProblemConfig& config, double t) {
  return config.final_time() + config.time.dt - t;
}

// D-affine:  y* = (1+t) phi,  p* = s phi,  u* = -p*
static double DistAffineState(const ProblemConfig&, double x, double z, double t) {
  return (1.0 + t) * ModePhi(x, z);
}
static double DistAffineAdjoint(const ProblemConfig& config, double x, double z,
                                double t) {
  return AffineS(config, t) * ModePhi(x, z);
}
/** f = y_t - nu Lap y - eta u = phi [ 1 + 8 pi^2 (1+t) + s ]. */
static double DistAffineSource(const ProblemConfig& config, double x, double z,
                               double t) {
  return ModePhi(x, z) *
         (1.0 + kEightPiSq * (1.0 + t) + AffineS(config, t));
}
/** y_d = y - (-p_t - nu Lap p)/alpha_track = phi [ t - 8 pi^2 s ]. */
static double DistAffineTarget(const ProblemConfig& config, double x, double z,
                               double t) {
  return ModePhi(x, z) * (t - kEightPiSq * AffineS(config, t));
}
static double DistAffineInitial(const ProblemConfig& config, double x, double z) {
  return DistAffineState(config, x, z, 0.0);
}

// N-affine:  p* = s psi,  y* = -(1/2) s C(x) q(z),  u* = -p*|_Gamma_c
static double NeuAffineState(const ProblemConfig& config, double x, double z,
                             double t) {
  return -0.5 * AffineS(config, t) * ModeCos2PiX(x) * ShapeQ(z);
}
static double NeuAffineAdjoint(const ProblemConfig& config, double x, double z,
                               double t) {
  return AffineS(config, t) * ModePsi(x, z);
}
/** f = y_t - nu Lap y = (1/2) C [ q + s (2 - 4 pi^2 q) ]; no volume control. */
static double NeuAffineSource(const ProblemConfig& config, double x, double z,
                              double t) {
  const double s = AffineS(config, t);
  return 0.5 * ModeCos2PiX(x) *
         (ShapeQ(z) + s * (2.0 - kFourPiSq * ShapeQ(z)));
}
/** y_d = y - (-p_t - nu Lap p)/alpha_track
 *      = -(1/2) s C q - psi [ 1 + 5 pi^2 s ]. */
static double NeuAffineTarget(const ProblemConfig& config, double x, double z,
                              double t) {
  const double s = AffineS(config, t);
  return -0.5 * s * ModeCos2PiX(x) * ShapeQ(z) -
         ModePsi(x, z) * (1.0 + kFivePiSq * s);
}
static double NeuAffineInitial(const ProblemConfig& config, double x, double z) {
  return NeuAffineState(config, x, z, 0.0);
}

static bool IsAffineCase(ManufacturedCase m) {
  return m == ManufacturedCase::DISTRIBUTED_AFFINE ||
         m == ManufacturedCase::NEUMANN_BOUNDARY_AFFINE;
}

// ---------------------------------------------------------------------------
// Time moments of the exact objective:  I_k = int_0^T (T-t)^k e^{-2t} dt.
// Substituting s = T - t gives I_k = e^{-2T} int_0^T s^k e^{2s} ds.
// ---------------------------------------------------------------------------

/** The three case-dependent constants of the objective:
 *    c            : y - y_d = e^{-t}(mode)[1 + c(T-t)], c set by -Lap(mode)
 *    track_space  : integral over Omega of the tracking mode squared
 *    reg_space    : integral of the control mode squared, over Omega for
 *                   distributed control and over Gamma_c for boundary control */
static void ObjectiveConstants(ManufacturedCase manufactured, double& c,
                        double& track_space, double& reg_space) {
  if (manufactured == ManufacturedCase::DISTRIBUTED_AFFINE) {
    // y - y_d = phi [1 + 8 pi^2 s];  u* = -s phi
    c = kEightPiSq;
    track_space = 0.25;
    reg_space = 0.25;
    return;
  }
  if (manufactured == ManufacturedCase::NEUMANN_BOUNDARY_AFFINE) {
    // y - y_d = psi [1 + 5 pi^2 s];  u* = -s C(x) on Gamma_c
    c = kFivePiSq;
    track_space = 0.25;
    reg_space = 0.5;
    return;
  }
  if (manufactured == ManufacturedCase::DISTRIBUTED_COSINE) {
    c = 1.0 + kEightPiSq;
    track_space = 0.25;   // int_Omega cos^2(2pi x) cos^2(2pi z) = 1/4
    reg_space = 0.25;     // u* = -(T-t) e^{-t} phi, the same mode
  } else {
    c = 1.0 + kFivePiSq;
    track_space = 0.25;   // int_Omega cos^2(2pi x) cos^2(pi z) = 1/4
    reg_space = 0.5;      // u* on Gamma_c: int_0^1 cos^2(2pi x) dx = 1/2
  }
}

static void ObjectiveMoments(double T, double& i0, double& i1, double& i2) {
  const double e2t = std::exp(2.0 * T);
  const double decay = std::exp(-2.0 * T);
  const double j0 = (e2t - 1.0) / 2.0;
  const double j1 = T * e2t / 2.0 - j0 / 2.0;
  const double j2 = T * T * e2t / 2.0 - j1;
  i0 = decay * j0;
  i1 = decay * j1;
  i2 = decay * j2;
}

// ---------------------------------------------------------------------------
// Catalogue
// ---------------------------------------------------------------------------

bool ParseManufacturedCase(const std::string& name, ManufacturedCase& out) {
  if (name == "distributed") {
    out = ManufacturedCase::DISTRIBUTED_COSINE;
    return true;
  }
  if (name == "neumann") {
    out = ManufacturedCase::NEUMANN_BOUNDARY_COSINE;
    return true;
  }
  if (name == "distributed-affine") {
    out = ManufacturedCase::DISTRIBUTED_AFFINE;
    return true;
  }
  if (name == "neumann-affine") {
    out = ManufacturedCase::NEUMANN_BOUNDARY_AFFINE;
    return true;
  }
  return false;
}

const char* ManufacturedCaseName(ManufacturedCase manufactured) {
  if (manufactured == ManufacturedCase::DISTRIBUTED_COSINE)
    return "distributed";
  if (manufactured == ManufacturedCase::DISTRIBUTED_AFFINE)
    return "distributed-affine";
  if (manufactured == ManufacturedCase::NEUMANN_BOUNDARY_AFFINE)
    return "neumann-affine";
  if (manufactured == ManufacturedCase::NEUMANN_BOUNDARY_COSINE)
    return "neumann";
  return "none";
}

void ApplyManufacturedCase(ManufacturedCase manufactured,
                           ProblemConfig& config) {
  if (manufactured == ManufacturedCase::NONE) return;

  config.domain.xmin = 0.0;
  config.domain.xmax = 1.0;
  config.domain.zmin = 0.0;
  config.domain.zmax = 1.0;
  config.time.dt = 1.0 / static_cast<double>(config.time.steps);  // T = 1

  // Both cases are posed with unit coefficients, so that u* = -p* exactly and
  // the derivations stay readable.
  config.diffusivity = 1.0;
  config.alpha_track = 1.0;
  config.alpha_u = 1.0;
  config.eta = 1.0;

  // Tracking over all of Omega in both cases: M_omega = M.
  config.tracking_box.xmin = config.domain.xmin;
  config.tracking_box.xmax = config.domain.xmax;
  config.tracking_box.zmin = config.domain.zmin;
  config.tracking_box.zmax = config.domain.zmax;

  if (manufactured == ManufacturedCase::DISTRIBUTED_COSINE) {
    config.control.type = ProblemConfig::ControlType::DISTRIBUTED;
    config.control.region.xmin = config.domain.xmin;
    config.control.region.xmax = config.domain.xmax;
    config.control.region.zmin = config.domain.zmin;
    config.control.region.zmax = config.domain.zmax;
    config.data_overrides.source = DistributedSource;
    config.data_overrides.initial_state = DistributedInitial;
    config.data_overrides.tracking_target = DistributedTarget;
  } else if (manufactured == ManufacturedCase::DISTRIBUTED_AFFINE) {
    config.control.type = ProblemConfig::ControlType::DISTRIBUTED;
    config.control.region.xmin = config.domain.xmin;
    config.control.region.xmax = config.domain.xmax;
    config.control.region.zmin = config.domain.zmin;
    config.control.region.zmax = config.domain.zmax;
    config.data_overrides.source = DistAffineSource;
    config.data_overrides.initial_state = DistAffineInitial;
    config.data_overrides.tracking_target = DistAffineTarget;
  } else if (manufactured == ManufacturedCase::NEUMANN_BOUNDARY_AFFINE) {
    config.control.type = ProblemConfig::ControlType::NEUMANN;
    config.data_overrides.source = NeuAffineSource;
    config.data_overrides.initial_state = NeuAffineInitial;
    config.data_overrides.tracking_target = NeuAffineTarget;
  } else {
    // Control on the EXISTING bottom face only; the geometry is not touched.
    config.control.type = ProblemConfig::ControlType::NEUMANN;
    config.data_overrides.source = NeumannSource;
    config.data_overrides.initial_state = NeumannInitial;
    config.data_overrides.tracking_target = NeumannTarget;
  }
}

double ManufacturedExactState(ManufacturedCase manufactured,
                              const ProblemConfig& config, double x, double z,
                              double time) {
  if (manufactured == ManufacturedCase::DISTRIBUTED_COSINE)
    return DistributedState(config, x, z, time);
  if (manufactured == ManufacturedCase::NEUMANN_BOUNDARY_COSINE)
    return NeumannState(config, x, z, time);
  if (manufactured == ManufacturedCase::DISTRIBUTED_AFFINE)
    return DistAffineState(config, x, z, time);
  if (manufactured == ManufacturedCase::NEUMANN_BOUNDARY_AFFINE)
    return NeuAffineState(config, x, z, time);
  return 0.0;
}

double ManufacturedExactAdjoint(ManufacturedCase manufactured,
                                const ProblemConfig& config, double x, double z,
                                double time) {
  if (manufactured == ManufacturedCase::DISTRIBUTED_COSINE)
    return DistributedAdjoint(config, x, z, time);
  if (manufactured == ManufacturedCase::NEUMANN_BOUNDARY_COSINE)
    return NeumannAdjoint(config, x, z, time);
  if (manufactured == ManufacturedCase::DISTRIBUTED_AFFINE)
    return DistAffineAdjoint(config, x, z, time);
  if (manufactured == ManufacturedCase::NEUMANN_BOUNDARY_AFFINE)
    return NeuAffineAdjoint(config, x, z, time);
  return 0.0;
}

double ManufacturedExactControl(ManufacturedCase manufactured,
                                const ProblemConfig& config, double x, double z,
                                double time) {
  // Stationarity alpha_u u + eta p = 0 holds in BOTH cases; for B2-N it holds
  // on Gamma_c, where this is evaluated.
  return -(config.eta / config.alpha_u) *
         ManufacturedExactAdjoint(manufactured, config, x, z, time);
}

double ManufacturedExactObjective(ManufacturedCase manufactured,
                                  const ProblemConfig& config, double& j_track,
                                  double& j_reg) {
  const double T = config.final_time();
  double i0 = 0.0, i1 = 0.0, i2 = 0.0;
  if (IsAffineCase(manufactured)) {
    // s = T + dt - t runs over [dt, T + dt] as t runs over [0, T]; the moments
    // are int s^k ds, with no exponential factor.
    const double a = config.time.dt, b = T + config.time.dt;
    i0 = b - a;
    i1 = (b * b - a * a) / 2.0;
    i2 = (b * b * b - a * a * a) / 3.0;
  } else {
    ObjectiveMoments(T, i0, i1, i2);
  }

  // y - y_d = e^{-t} (mode) [1 + c (T - t)] in both cases, with c set by the
  // Laplacian eigenvalue of the adjoint mode.  The spatial factors are the
  // exact integrals of the squared modes.
  double c = 0.0, track_space = 0.0, reg_space = 0.0;
  ObjectiveConstants(manufactured, c, track_space, reg_space);

  j_track = 0.5 * config.alpha_track * track_space *
            (i0 + 2.0 * c * i1 + c * c * i2);
  j_reg = 0.5 * config.alpha_u * reg_space * i2;
  return j_track + j_reg;
}

double ManufacturedDiscreteObjective(ManufacturedCase manufactured,
                                     const ProblemConfig& config,
                                     double& j_track, double& j_reg) {
  const double T = config.final_time();
  const double dt = config.time.dt;
  double c = 0.0, track_space = 0.0, reg_space = 0.0;
  ObjectiveConstants(manufactured, c, track_space, reg_space);

  // Same slots and same weight dt as ReconstructDistributedDataSide: the exact
  // fields are sampled at t_{n+1} = (n+1) dt, n = 0 .. steps-1.
  double track_sum = 0.0, reg_sum = 0.0;
  const bool affine = IsAffineCase(manufactured);
  for (unsigned slot = 0; slot < config.time.steps; ++slot) {
    const double t = (slot + 1u) * dt;
    const double decay = affine ? 1.0 : std::exp(-2.0 * t);
    const double s = affine ? (T + dt - t) : (T - t);
    const double bracket = 1.0 + c * s;
    track_sum += dt * decay * bracket * bracket;
    reg_sum += dt * decay * s * s;
  }
  j_track = 0.5 * config.alpha_track * track_space * track_sum;
  j_reg = 0.5 * config.alpha_u * reg_space * reg_sum;
  return j_track + j_reg;
}

// ===========================================================================
// B2-D / B2-N   MANUFACTURED SOLUTIONS
// ===========================================================================

static double StateCallback(double x, double z, void* raw) {
  const ExactFieldContext& c = *static_cast<ExactFieldContext*>(raw);
  return ManufacturedExactState(c.manufactured, *c.config, x, z, c.time);
}
static double AdjointCallback(double x, double z, void* raw) {
  const ExactFieldContext& c = *static_cast<ExactFieldContext*>(raw);
  return ManufacturedExactAdjoint(c.manufactured, *c.config, x, z, c.time);
}
static double ControlCallback(double x, double z, void* raw) {
  const ExactFieldContext& c = *static_cast<ExactFieldContext*>(raw);
  return ManufacturedExactControl(c.manufactured, *c.config, x, z, c.time);
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

int RunManufacturedBenchmark(ManufacturedCase manufactured,
                             const ProblemConfig& config,
                             MultiLevelSolution& solution, Mesh* mesh,
                             unsigned solution_type, DistributedSolvePc pc_mode,
                             double spatial_tolerance,
                             const std::string& json_path) {
  const double t_total0 = MPI_Wtime();
  SpatialOperators spatial(config, mesh, solution_type, solution);
  SpaceTimeOperators spacetime(config, spatial);
  Mat mass = spatial.mass;
  Mat control = spatial.control;
  const OwnedSpatialLayout& layout = spatial.layout;

  DistributedOptimalControlSolve solve = SolveOptimalControl(
      config, spatial, spacetime,
      MakeKktSolverPolicy(pc_mode, spatial_tolerance));

  const bool boundary_control =
      config.control.type == ProblemConfig::ControlType::NEUMANN;

  ExactFieldContext context;
  context.manufactured = manufactured;
  const SpaceTimeNorms y = MeasureSpaceTime(
      config, context, StateCallback, solve.fields.state, mass, layout,
      mesh, solution_type, false);
  const SpaceTimeNorms p = MeasureSpaceTime(
      config, context, AdjointCallback, solve.fields.adjoint, mass, layout,
      mesh, solution_type, false);
  // For Neumann control u lives on Gamma_c, so it is measured with the surface
  // rule.  Using the volume norm there would be meaningless.
  const SpaceTimeNorms u = MeasureSpaceTime(
      config, context, ControlCallback, solve.fields.control, mass, layout,
      mesh, solution_type, boundary_control);

  double j_track_exact = 0.0, j_reg_exact = 0.0;
  const double j_exact = ManufacturedExactObjective(manufactured, config,
                                                    j_track_exact,
                                                    j_reg_exact);
  double j_track_disc = 0.0, j_reg_disc = 0.0;
  const double j_disc = ManufacturedDiscreteObjective(manufactured, config,
                                                      j_track_disc,
                                                      j_reg_disc);
  const double j_h = solve.fields.J_track + solve.fields.J_reg;

  if (!json_path.empty()) {
    const RootDistributedFields fields = GatherDistributedSolveToRoot(
        solve, layout, mass, config.time.steps);
    WriteParallelSolveJsonRankZero(json_path, config, solve, fields);
  }
  solve.timings.total = MPI_Wtime() - t_total0;

  if (mesh->processor_id() == 0) {
    const double h = (config.domain.xmax - config.domain.xmin) /
                     static_cast<double>(config.mesh.nx);
    std::printf(
        "EX20_MMS case=%s control=%s nx=%u nz=%u h=%.17g steps=%u dt=%.17g "
        "T=%.17g nu=%.17g alpha_u=%.17g alpha_track=%.17g eta=%.17g pc=%s "
        "reason=%d outer=%d true_residual=%.17g "
        "y_l2l2=%.17g y_l2l2_rel=%.17g y_linf_l2=%.17g y_linf_l2_rel=%.17g "
        "p_l2l2=%.17g p_l2l2_rel=%.17g p_linf_l2=%.17g p_linf_l2_rel=%.17g "
        "u_l2l2=%.17g u_l2l2_rel=%.17g u_linf_l2=%.17g u_linf_l2_rel=%.17g "
        "u_norm_kind=%s "
        "y_exact_l2l2=%.17g p_exact_l2l2=%.17g u_exact_l2l2=%.17g "
        "J_h=%.17g J_track_h=%.17g J_reg_h=%.17g "
        "J_exact=%.17g J_track_exact=%.17g J_reg_exact=%.17g "
        "J_disc=%.17g J_track_disc=%.17g J_reg_disc=%.17g "
        "J_abs_error=%.17g J_rel_error=%.17g "
        "J_disc_abs_error=%.17g J_disc_rel_error=%.17g "
        "kkt_state_abs=%.17g kkt_state_rel=%.17g "
        "kkt_adjoint_abs=%.17g kkt_adjoint_rel=%.17g "
        "kkt_stationarity_abs=%.17g kkt_stationarity_rel=%.17g "
        "kkt_reduced_abs=%.17g control_offactive_max=%.17g "
        "active_dofs=%lld t_total=%.17g\n",
        ManufacturedCaseName(manufactured),
        boundary_control ? "neumann_boundary" : "distributed", config.mesh.nx,
        config.mesh.nz, h, config.time.steps, config.time.dt,
        config.final_time(), config.diffusivity, config.alpha_u,
        config.alpha_track, config.eta, PcName(pc_mode), solve.reason,
        solve.outer_iterations, solve.true_residual,
        y.l2_l2, Relative(y.l2_l2, y.exact_l2_l2), y.linf_l2,
        Relative(y.linf_l2, y.exact_linf_l2),
        p.l2_l2, Relative(p.l2_l2, p.exact_l2_l2), p.linf_l2,
        Relative(p.linf_l2, p.exact_linf_l2),
        u.l2_l2, Relative(u.l2_l2, u.exact_l2_l2), u.linf_l2,
        Relative(u.linf_l2, u.exact_linf_l2),
        boundary_control ? "boundary_L2_Gamma_c" : "volume_L2_Omega",
        y.exact_l2_l2, p.exact_l2_l2, u.exact_l2_l2,
        j_h, solve.fields.J_track, solve.fields.J_reg,
        j_exact, j_track_exact, j_reg_exact,
        j_disc, j_track_disc, j_reg_disc,
        std::fabs(j_h - j_exact),
        j_exact != 0.0 ? std::fabs(j_h - j_exact) / std::fabs(j_exact) : 0.0,
        std::fabs(j_h - j_disc),
        j_disc != 0.0 ? std::fabs(j_h - j_disc) / std::fabs(j_disc) : 0.0,
        solve.kkt.state_abs, solve.kkt.state_rel, solve.kkt.adjoint_abs,
        solve.kkt.adjoint_rel, solve.kkt.stationarity_abs,
        solve.kkt.stationarity_rel, solve.kkt.reduced_abs,
        solve.kkt.control_offactive_max, solve.kkt.active_dofs,
        solve.timings.total);
    std::fflush(stdout);
  }

  PrintSolverStats(config, solve, pc_mode);

  const int success = solve.accepted ? 0 : 8;
  return success;
}

