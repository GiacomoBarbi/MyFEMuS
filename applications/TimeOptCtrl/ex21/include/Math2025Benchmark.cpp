// Mathematics 2025 Example 3 preset and driver.

#include "Math2025Benchmark.hpp"

#include <mpi.h>
#include <petscksp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include "Discretization.hpp"
#include "Solver.hpp"
#include "SpaceTimeOperators.hpp"

using namespace femus;

double Math2025SmoothStep(double s, double eps) {
  if (s <= -eps) return 0.0;
  if (s >= eps) return 1.0;
  const double t = (s + eps) / (2.0 * eps);
  const double t5 = t * t * t * t * t;
  return t5 * (126.0 + t * (-420.0 + t * (540.0 + t * (-315.0 + t * 70.0))));
}

namespace {

double Switch(double t) { return Math2025SmoothStep(t - 1.5, 1.5); }

double Reference(const ProblemConfig&, double x, double, double t) {
  return x * std::sin(t) * Switch(t);
}

// The disturbance FIELD d(x,t); its support Omega_d comes from
// config.source_regions (the load is M_source I_h d), never from an indicator.
double Disturbance(const ProblemConfig&, double x, double, double t) {
  return x * x * std::sin(2.0 * t) * Switch(t);
}

double InitialSource(const ProblemConfig&, double x, double) {
  return x * (1.0 - x);
}

double InitialZero(const ProblemConfig&, double, double) { return 0.0; }

double SourceZero(const ProblemConfig&, double, double, double) { return 0.0; }

// Which preset was installed (3 or 4); read by the shared driver.
int g_example = 3;

double Dot(Vec a, Vec b) {
  PetscScalar v = 0.0;
  PetscCallAbort(PETSC_COMM_WORLD, VecDot(a, b, &v));
  return PetscRealPart(v);
}

/** 1^T A 1: the measure of the region a mass-type matrix integrates over. */
double Measure(Mat a) {
  Vec one = nullptr, r = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, MatCreateVecs(a, &one, &r));
  PetscCallAbort(PETSC_COMM_WORLD, VecSet(one, 1.0));
  PetscCallAbort(PETSC_COMM_WORLD, MatMult(a, one, r));
  const double m = Dot(one, r);
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&r));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&one));
  return m;
}

double InitialExact(double x, double, void* context) {
  return *static_cast<const bool*>(context) ? x * (1.0 - x) : 0.0;
}

}  // namespace

void ApplyMath2025Example3(ProblemConfig& config, const std::string& initial,
                           double final_time) {
  if (initial != "source" && initial != "zero")
    throw std::invalid_argument("--ex3-initial takes source or zero");
  config.dimension = 1u;
  config.domain = {0.0, 1.0, 0.0, 1.0};
  config.diffusivity = 2.0;                       // a = 2   [B] inherited, see audit
  config.convection.enabled = true;
  config.convection.vx = 2.0;                     // A = a d2/dx2 - v d/dx, v = 2 [B]
  config.convection.vy = 0.0;
  config.state_bc = ProblemConfig::StateBoundaryCondition::HOMOGENEOUS_DIRICHLET;
  config.control.type = ProblemConfig::ControlType::DISTRIBUTED;
  config.control.region = {0.2, 0.4, 0.0, 1.0};   // Omega_b = [1/5, 2/5]
  config.tracking_box = {0.5, 0.7, 0.0, 1.0};     // Omega_c = [1/2, 7/10]
  config.additional_control_regions.clear();
  config.additional_tracking_regions.clear();
  config.source_regions = {{0.8, 0.9, 0.0, 1.0}}; // Omega_d = [4/5, 9/10]
  config.alpha_track = 1.0;                       // 1/2 int (z - r)^2
  config.alpha_u = 1.0e-9;                        // eps = 1e-9, NOT tuned
  config.eta = 1.0;
  config.data_overrides.source = Disturbance;
  config.data_overrides.tracking_target = Reference;
  config.data_overrides.initial_state =
      initial == "source" ? InitialSource : InitialZero;
  config.time.dt = final_time / config.time.steps;
}

void ApplyMath2025Example4(ProblemConfig& config, const std::string& initial,
                           double final_time) {
  if (initial != "zero" && initial != "x1mx")
    throw std::invalid_argument("--ex4-initial takes zero or x1mx");
  // Same plant family as Example 3 ("the same system operator A ... in (65)",
  // a = v = 2 inherited [B]); only the geometry, T and the IC differ.
  const double alpha_u = config.alpha_u;          // no source eps: CLI value
  ApplyMath2025Example3(config, initial == "x1mx" ? "source" : "zero",
                        final_time);
  config.alpha_u = alpha_u;
  config.control.region = {0.45, 0.55, 0.0, 1.0};   // Omega_b = Omega_c
  config.tracking_box = {0.45, 0.55, 0.0, 1.0};
  config.source_regions = {{0.7, 0.8, 0.0, 1.0}};   // Omega_d = [0.7, 0.8]
  g_example = 4;
}

int RunMath2025Example3(const ProblemConfig& config,
                        const ApplicationOptions& options,
                        MultiLevelSolution& solution, Mesh* mesh,
                        unsigned solution_type) {
  const bool ex4 = (g_example == 4);
  const char* tag = ex4 ? "EX21_EX4" : "EX21_EX3";
  int ranks = 0;
  MPI_Comm_size(PETSC_COMM_WORLD, &ranks);
  if (ranks != 1) throw std::runtime_error("--math2025-ex3/ex4 run serially");
  const bool source_ic =
      config.data_overrides.initial_state == InitialSource;
  const double t0 = MPI_Wtime();
  SpatialOperators spatial(config, mesh, solution_type, solution);
  SpaceTimeOperators spacetime(config, spatial);
  const PetscInt local = spatial.layout.local_size;
  const unsigned steps = config.time.steps;
  const double dt = config.time.dt;
  std::printf("%s_CONFIG nx=%u steps=%u T=%.6g dt=%.6e scheme=%s nu=%.6g"
              " vx=%.6g alpha_u=%.3e alpha_track=%.6g initial=%s pc=%d\n",
              tag, config.mesh.nx, steps, config.final_time(), dt,
              TimeSchemeName(config), config.diffusivity, config.convection.vx,
              config.alpha_u, config.alpha_track, source_ic ? "x(1-x)" : "zero",
              static_cast<int>(options.pc_mode));

  // ---- gate 1: region measures (exact when the endpoints are element faces)
  if (ex4)
    std::printf("%s_GATE_REGIONS |Omega_b|=%.15f (%g) |Omega_c|=%.15f (%g)"
                " |Omega_d|=%.15f (%g) |Omega|_eliminated=%.15f\n", tag,
                Measure(spatial.control),
                config.control.region.xmax - config.control.region.xmin,
                Measure(spatial.tracking),
                config.tracking_box.xmax - config.tracking_box.xmin,
                Measure(spatial.source),
                config.source_regions[0].xmax - config.source_regions[0].xmin,
                Measure(spatial.mass));
  else
    std::printf("EX21_EX3_GATE_REGIONS |Omega_b|=%.15f (0.2) |Omega_c|=%.15f (0.2)"
                " |Omega_d|=%.15f (0.1) |Omega|_eliminated=%.15f\n",
                Measure(spatial.control), Measure(spatial.tracking),
                Measure(spatial.source), Measure(spatial.mass));

  // ---- gate 2: initial condition (x(1-x) is in Q2: interpolation is exact)
  Vec y0 = nullptr, work = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, MatCreateVecs(spatial.mass, &y0, &work));
  {
    PetscScalar* a = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD, VecGetArray(y0, &a));
    double trace = 0.0;
    for (PetscInt i = 0; i < local; ++i) {
      a[i] = EvaluateInitialState(config, spatial.layout.x[i], 0.0);
      if (spatial.layout.x[i] < 1e-12 || spatial.layout.x[i] > 1.0 - 1e-12)
        trace = std::max(trace, std::fabs(a[i]));
    }
    bool flag = source_ic;
    const SpatialFieldIntegrals I =
        IntegrateSpatialField(mesh, solution_type, a, InitialExact, &flag);
    PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArray(y0, &a));
    std::printf("%s_GATE_IC ||I_h z0 - z0||_L2=%.3e ||z0||_L2=%.12f"
                " (exact 1/sqrt(30)=%.12f for x(1-x)) max|z0| on boundary=%.3e\n",
                tag, std::sqrt(I.error_squared), std::sqrt(I.field_squared),
                1.0 / std::sqrt(30.0), trace);
  }

  // ---- gate 3: data -- switch-on, reference and disturbance at fixed times;
  // the disturbance load against int_{0.8}^{0.9} x^2 dx = 0.0723333...
  // Sample points inside Omega_c and Omega_d (0.6 / 0.85 for Ex. 3,
  // 0.5 / 0.75 for Ex. 4).
  for (double t : {0.0, 0.75, 1.5, 2.25, 3.0, 10.0})
    if (ex4)
      std::printf("%s_GATE_DATA t=%.2f F=%.15f r(0.5,t)=%.15f d(0.75,t)=%.15f\n",
                  tag, t, Switch(t), Reference(config, 0.5, 0, t),
                  Disturbance(config, 0.75, 0, t));
    else
      std::printf("EX21_EX3_GATE_DATA t=%.2f F=%.15f r(0.6,t)=%.15f d(0.85,t)=%.15f\n",
                  t, Switch(t), Reference(config, 0.6, 0, t),
                  Disturbance(config, 0.85, 0, t));
  {
    PetscScalar* a = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD, VecGetArray(work, &a));
    for (PetscInt i = 0; i < local; ++i)
      a[i] = spatial.layout.x[i] * spatial.layout.x[i];
    PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArray(work, &a));
    Vec one = nullptr, r = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(work, &one));
    PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(work, &r));
    PetscCallAbort(PETSC_COMM_WORLD, VecSet(one, 1.0));
    PetscCallAbort(PETSC_COMM_WORLD, MatMult(spatial.source, work, r));
    // exact int_{Omega_d} x^2 dx (x^2 is in Q2, so the load is exact)
    std::printf("%s_GATE_LOAD 1^T M_d I_h(x^2)=%.15f exact=%.15f\n", tag,
                Dot(one, r), ex4 ? (0.512 - 0.343) / 3.0 : (0.729 - 0.512) / 3.0);
    PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&one));
    PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&r));
  }

  // ---- the optimal-control solve (production path)
  const KktSolverPolicy policy =
      MakeKktSolverPolicy(options.pc_mode, options.spatial_tolerance);
  const double t_solve = MPI_Wtime();
  DistributedOptimalControlSolve solve =
      SolveOptimalControl(config, spatial, spacetime, policy);
  const double solve_wall = MPI_Wtime() - t_solve;
  const DistributedDataSideResult& f = solve.fields;

  // ---- gate 4: independent forward march with the optimal control
  //   BE: (M/dt + L) y_n = rhs_n + N u_n + (M/dt) y_{n-1}
  //   CN: (M/dt + L/2) y_n = rhs_n + N u_n + (M/dt - L/2) y_{n-1}
  // rhs_n (initial state, disturbance) is rebuilt by the data-side builder.
  const bool cn = IsCrankNicolson(config);
  DistributedDataSideResult data = BuildDistributedRightHandSide(
      config, spatial.layout, spatial.mass, spatial.tracking,
      spacetime.distributed.ay, spatial.stiffness, spatial.source);
  Mat diagonal = nullptr, coupling = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatDuplicate(spatial.stiffness, MAT_COPY_VALUES, &diagonal));
  PetscCallAbort(PETSC_COMM_WORLD, MatScale(diagonal, cn ? 0.5 : 1.0));
  PetscCallAbort(PETSC_COMM_WORLD, MatAXPY(diagonal, 1.0 / dt, spatial.mass,
                                           SUBSET_NONZERO_PATTERN));
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatDuplicate(spatial.mass, MAT_COPY_VALUES, &coupling));
  PetscCallAbort(PETSC_COMM_WORLD, MatScale(coupling, 1.0 / dt));
  if (cn)
    PetscCallAbort(PETSC_COMM_WORLD, MatAXPY(coupling, -0.5, spatial.stiffness,
                                             DIFFERENT_NONZERO_PATTERN));
  KSP lu = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, KSPCreate(PETSC_COMM_WORLD, &lu));
  PetscCallAbort(PETSC_COMM_WORLD, KSPSetOptionsPrefix(lu, "ex3_march_"));
  PetscCallAbort(PETSC_COMM_WORLD, KSPSetOperators(lu, diagonal, diagonal));
  PetscCallAbort(PETSC_COMM_WORLD, KSPSetType(lu, KSPPREONLY));
  PC pc = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, KSPGetPC(lu, &pc));
  PetscCallAbort(PETSC_COMM_WORLD, PCSetType(pc, PCLU));
  PetscCallAbort(PETSC_COMM_WORLD, KSPSetUp(lu));
  Vec r = nullptr, y = nullptr, prev = nullptr, tmp = nullptr, u = nullptr,
      target = nullptr, diff = nullptr, mdiff = nullptr;
  for (Vec* v : {&r, &y, &prev, &tmp, &u, &target, &diff, &mdiff})
    PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(y0, v));
  double march_diff = 0.0, march_norm = 0.0, boundary_max = 0.0;
  double control_l2_sq = 0.0;
  std::vector<double> times(1, 0.0), errors;
  // e(0) = || z0 - r(0) ||_{M_omega}
  {
    PetscScalar* a = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD, VecGetArray(target, &a));
    for (PetscInt i = 0; i < local; ++i)
      a[i] = EvaluateTrackingTarget(config, spatial.layout.x[i], 0.0, 0.0);
    PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArray(target, &a));
    PetscCallAbort(PETSC_COMM_WORLD, VecWAXPY(diff, -1.0, target, y0));
    PetscCallAbort(PETSC_COMM_WORLD, MatMult(spatial.tracking, diff, mdiff));
    errors.push_back(std::sqrt(std::max(0.0, Dot(diff, mdiff))));
  }
  for (unsigned slot = 0; slot < steps; ++slot) {
    const double t = (slot + 1u) * dt;
    CopyDataSideSliceToSpatial(data.rhs_y, slot, local, r);
    CopyDataSideSliceToSpatial(f.control, slot, local, u);
    PetscCallAbort(PETSC_COMM_WORLD, MatMult(spatial.control, u, tmp));
    PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(r, config.eta, tmp));
    control_l2_sq += dt * Dot(u, tmp);
    if (slot > 0u) {
      PetscCallAbort(PETSC_COMM_WORLD, MatMult(coupling, prev, tmp));
      PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(r, 1.0, tmp));
    }
    PetscCallAbort(PETSC_COMM_WORLD, KSPSolve(lu, r, y));
    PetscCallAbort(PETSC_COMM_WORLD, VecCopy(y, prev));
    // compare with the production state of this slot
    CopyDataSideSliceToSpatial(f.state, slot, local, tmp);
    PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(tmp, -1.0, y));
    PetscReal nd = 0.0, ny = 0.0;
    PetscCallAbort(PETSC_COMM_WORLD, VecNorm(tmp, NORM_INFINITY, &nd));
    PetscCallAbort(PETSC_COMM_WORLD, VecNorm(y, NORM_INFINITY, &ny));
    march_diff = std::max(march_diff, static_cast<double>(nd));
    march_norm = std::max(march_norm, static_cast<double>(ny));
    // tracking error ||y_n - r(t_n)||_{L2(Omega_c)} of the PRODUCTION state
    CopyDataSideSliceToSpatial(f.state, slot, local, y);
    PetscScalar* a = nullptr;
    const PetscScalar* ys = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD, VecGetArray(target, &a));
    PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(y, &ys));
    for (PetscInt i = 0; i < local; ++i) {
      a[i] = EvaluateTrackingTarget(config, spatial.layout.x[i], 0.0, t);
      if (spatial.layout.x[i] < 1e-12 || spatial.layout.x[i] > 1.0 - 1e-12)
        boundary_max = std::max(boundary_max, std::fabs(ys[i]));
    }
    PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(y, &ys));
    PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArray(target, &a));
    PetscCallAbort(PETSC_COMM_WORLD, VecWAXPY(diff, -1.0, target, y));
    PetscCallAbort(PETSC_COMM_WORLD, MatMult(spatial.tracking, diff, mdiff));
    times.push_back(t);
    errors.push_back(std::sqrt(std::max(0.0, Dot(diff, mdiff))));
  }
  std::printf("%s_GATE_STATE independent_march_vs_production"
              " max|dy|/max|y|=%.3e max|y| on the Dirichlet boundary=%.3e\n",
              tag, march_diff / march_norm, boundary_max);

  // ---- Example 4 only: the uncontrolled, undisturbed FREE response of the
  // initial state seen on Omega_c, ||chi_c S(t) z0||_{L2}.  The source's
  // cascade controller assumes a zero plant state, so the plant error of
  // Figs. 39-40 differs from the cascade error by exactly this field; it is
  // the observable that can identify the (unstated) plant IC.
  if (ex4) {
    ProblemConfig free_config = config;
    free_config.data_overrides.source = SourceZero;
    DistributedDataSideResult free_data = BuildDistributedRightHandSide(
        free_config, spatial.layout, spatial.mass, spatial.tracking,
        spacetime.distributed.ay, spatial.stiffness, spatial.source);
    for (unsigned slot = 0; slot < steps; ++slot) {
      CopyDataSideSliceToSpatial(free_data.rhs_y, slot, local, r);
      if (slot > 0u) {
        PetscCallAbort(PETSC_COMM_WORLD, MatMult(coupling, prev, tmp));
        PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(r, 1.0, tmp));
      }
      PetscCallAbort(PETSC_COMM_WORLD, KSPSolve(lu, r, y));
      PetscCallAbort(PETSC_COMM_WORLD, VecCopy(y, prev));
      PetscCallAbort(PETSC_COMM_WORLD, MatMult(spatial.tracking, y, mdiff));
      std::printf("%s_FREE t=%.10f free_c=%.12e\n", tag, (slot + 1u) * dt,
                  std::sqrt(std::max(0.0, Dot(y, mdiff))));
    }
  }

  // ---- metrics on the plotted source interval [0, min(T,20)] and tail
  auto window = [&](double lo, double hi, double& sup, double& rms) {
    sup = 0.0;
    double integral = 0.0, length = 0.0;
    for (std::size_t k = 0; k < times.size(); ++k) {
      if (times[k] < lo - 1e-12 || times[k] > hi + 1e-12) continue;
      sup = std::max(sup, errors[k]);
      if (k + 1 < times.size() && times[k + 1] <= hi + 1e-12) {
        const double h = times[k + 1] - times[k];
        integral += 0.5 * h * (errors[k] * errors[k] + errors[k + 1] * errors[k + 1]);
        length += h;
      }
    }
    rms = length > 0 ? std::sqrt(integral / length) : 0.0;
  };
  const double t_plot = std::min(ex4 ? 15.0 : 20.0, config.final_time());
  double sup_all, rms_all, sup_tail, rms_tail;
  window(0.0, t_plot, sup_all, rms_all);
  // Example 4: tail [5, 15] -- the signals are fully on from t = 3 (F) and the
  // source's figures are periodic from there.  Printed under its own keys.
  window(ex4 ? 5.0 : 10.0, t_plot, sup_tail, rms_tail);
  const double total_wall = MPI_Wtime() - t0;
  if (ex4) {
    double sup_late, rms_late;
    window(10.0, t_plot, sup_late, rms_late);
    std::printf("%s_TAIL sup_e_0_15=%.10e rms_e_0_15=%.10e sup_e_5_15=%.10e"
                " rms_e_5_15=%.10e sup_e_10_15=%.10e rms_e_10_15=%.10e\n", tag,
                sup_all, rms_all, sup_tail, rms_tail, sup_late, rms_late);
  }
  std::printf("%s_RESULT reason=%d outer_its=%d true_residual=%.3e"
              " inner_y_its=%lld inner_p_its=%lld sweeps=%d sweep_failures=%lld"
              " J_track=%.12e J_reg=%.12e J_total=%.12e u_L2=%.12e"
              " sup_e_0_20=%.10e rms_e_0_20=%.10e sup_e_10_20=%.10e rms_e_10_20=%.10e"
              " e_T=%.10e solve_wall=%.3f total_wall=%.3f\n",
              tag, solve.reason, solve.outer_iterations, solve.true_residual,
              solve.inner_y_iterations, solve.inner_p_iterations,
              solve.sweeps_used ? 1 : 0, solve.sweep_failures, f.J_track,
              f.J_reg, f.J_track + f.J_reg, std::sqrt(control_l2_sq), sup_all,
              rms_all, sup_tail, rms_tail, errors.back(), solve_wall, total_wall);
  for (std::size_t k = 0; k < times.size(); ++k)
    std::printf("%s_SERIES t=%.10f e_c=%.12e\n", tag, times[k], errors[k]);

  PetscCallAbort(PETSC_COMM_WORLD, KSPDestroy(&lu));
  for (Vec* v : {&r, &y, &prev, &tmp, &u, &target, &diff, &mdiff, &y0, &work})
    PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(v));
  PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&coupling));
  PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&diagonal));
  return solve.accepted ? 0 : 8;
}
