// Two-dimensional literature benchmark presets and driver (see Benchmark2D.hpp).

#include "Benchmark2D.hpp"

#include <mpi.h>
#include <petscksp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include "Discretization.hpp"
#include "PetscMatrix.hpp"
#include "Math2025Benchmark.hpp"
#include "Solver.hpp"
#include "SpaceTimeOperators.hpp"

using namespace femus;

namespace {

// ---- which preset is installed (read by the shared driver)
enum class Case { NONE, EX5, EF95, EF96 };
Case g_case = Case::NONE;
double g_width = M_PI, g_height = 1.0;   // Ex. 5 geometry
const char* g_initial_name = "zero";

constexpr double kSigma = 0.15;          // EF sigma_y

double Switch(double t) { return Math2025SmoothStep(t - 1.5, 1.5); }

// ---- Example 5 data
double Ex5Reference(const ProblemConfig&, double x, double y, double t) {
  return x * y * (1.0 - y) * (y - 0.5) * std::sin(t) * Switch(t);
}
// SENSITIVITY ONLY (--ex5-variant): "source" (eq. (87)), "d-zero" (no
// disturbance) or "d-noswitch" (eq. (87) without F) -- diagnostics of the
// transient, never the benchmark data.
std::string g_ex5_variant = "source";
double Ex5Disturbance(const ProblemConfig&, double x, double y, double t) {
  if (g_ex5_variant == "d-zero") return 0.0;
  const double f = g_ex5_variant == "d-noswitch" ? 1.0 : Switch(t);
  return x * (y - 0.5) * (y - 0.5) * std::sin(2.0 * t) * f;
}
double InitialXY(const ProblemConfig&, double x, double y) { return x * y; }

// ---- ErrorFeedback 9.5 / 9.6 data (no switch-on: d(.,0) != 0)
double Gauss(double y) { return std::exp(-(y - 0.5) * (y - 0.5) / (kSigma * kSigma)); }
double EfReference(const ProblemConfig&, double x, double y, double t) {
  return x * Gauss(y) * std::sin(2.0 * t);
}
double EfDisturbance(const ProblemConfig&, double x, double y, double t) {
  return x * Gauss(y) * std::cos(3.0 * t);
}
double InitialSinSin(const ProblemConfig&, double x, double y) {
  return std::sin(x) * std::sin(M_PI * y);
}
double InitialPoly(const ProblemConfig&, double x, double y) {
  return 16.0 * x * (M_PI - x) * y * (1.0 - y) / (M_PI * M_PI);
}

double InitialZero(const ProblemConfig&, double, double) { return 0.0; }

double Dot(Vec a, Vec b) {
  PetscScalar v = 0.0;
  PetscCallAbort(PETSC_COMM_WORLD, VecDot(a, b, &v));
  return PetscRealPart(v);
}

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

double Area(const ProblemConfig::Box& b) {
  return (b.xmax - b.xmin) * (b.zmax - b.zmin);
}

double InitialExact(double x, double z, void* context) {
  return EvaluateInitialState(*static_cast<const ProblemConfig*>(context), x, z);
}

void CommonPlant(ProblemConfig& config, double final_time) {
  config.dimension = 2u;
  config.convection.enabled = true;
  config.state_bc = ProblemConfig::StateBoundaryCondition::HOMOGENEOUS_DIRICHLET;
  config.control.type = ProblemConfig::ControlType::DISTRIBUTED;
  config.additional_control_regions.clear();
  config.additional_tracking_regions.clear();
  config.alpha_track = 1.0;
  config.eta = 1.0;
  config.time.dt = final_time / config.time.steps;
}

}  // namespace

void ApplyMath2025Example5(ProblemConfig& config, double width, double height,
                           const std::string& initial, double final_time,
                           const std::string& variant) {
  if (initial != "xy" && initial != "zero")
    throw std::invalid_argument("--ex5-initial takes xy or zero");
  if (variant != "source" && variant != "d-zero" && variant != "d-noswitch")
    throw std::invalid_argument("--ex5-variant takes source, d-zero or d-noswitch");
  g_ex5_variant = variant;
  if (!(width > 0.0 && height > 0.0))
    throw std::invalid_argument("--ex5-geometry needs W > 0 and H > 0");
  CommonPlant(config, final_time);
  config.domain = {0.0, width, 0.0, height};
  config.diffusivity = 2.0;                      // a = 2           [A]
  config.convection.vx = 2.0;                    // V = (u,v) = (2,2) [A]
  config.convection.vy = 2.0;                    //   A = a Lap - V.grad
  config.control.region = {0.0, width / 3.0, 0.0, height};
  config.tracking_box = {width / 3.0, 2.0 * width / 3.0, 0.0, height};
  config.source_regions = {{2.0 * width / 3.0, width, 0.0, height}};
  config.alpha_u = 1.0e-7;                       // eps = 1e-7 [A], NOT tuned
  config.data_overrides.source = Ex5Disturbance;
  config.data_overrides.tracking_target = Ex5Reference;
  config.data_overrides.initial_state = initial == "xy" ? InitialXY : InitialZero;
  g_case = Case::EX5;
  g_width = width;
  g_height = height;
  g_initial_name = initial == "xy" ? "xy" : "zero";
}

void ApplyErrorFeedback9x(ProblemConfig& config, int example,
                          const std::string& initial, double final_time) {
  if (example != 95 && example != 96)
    throw std::invalid_argument("ErrorFeedback example must be 9.5 or 9.6");
  if (initial != "zero" && initial != "sinsin" && initial != "poly")
    throw std::invalid_argument("--ef-initial takes zero, sinsin or poly");
  CommonPlant(config, final_time);
  config.domain = {0.0, M_PI, 0.0, 1.0};
  config.diffusivity = 1.0;                      // normalised      [A]
  config.convection.vx = 2.0;                    // V = (2,1)       [A]
  config.convection.vy = 1.0;
  const double y0 = 1.0 / 3.0, y1 = 2.0 / 3.0;
  config.tracking_box = {3.0 * M_PI / 8.0, M_PI / 2.0, y0, y1};
  config.control.region = example == 95 ? config.tracking_box
                                        : ProblemConfig::Box{M_PI / 8.0, M_PI / 4.0, y0, y1};
  config.source_regions = {{3.0 * M_PI / 4.0, 7.0 * M_PI / 8.0, y0, y1}};
  // alpha_u: command line (no source epsilon exists)
  config.data_overrides.source = EfDisturbance;
  config.data_overrides.tracking_target = EfReference;
  config.data_overrides.initial_state =
      initial == "sinsin" ? InitialSinSin : initial == "poly" ? InitialPoly : InitialZero;
  g_case = example == 95 ? Case::EF95 : Case::EF96;
  g_initial_name = initial == "sinsin" ? "sinsin" : initial == "poly" ? "poly" : "zero";
}

int RunBenchmark2D(const ProblemConfig& config, const ApplicationOptions& options,
                   MultiLevelSolution& solution, Mesh* mesh,
                   unsigned solution_type) {
  const char* tag = g_case == Case::EX5 ? "EX21_EX5" : g_case == Case::EF95 ? "EX21_EF95" : "EX21_EF96";
  int ranks = 0;
  MPI_Comm_size(PETSC_COMM_WORLD, &ranks);
  if (ranks != 1) throw std::runtime_error("the 2D benchmarks run serially");
  const double t0 = MPI_Wtime();
  SpatialOperators spatial(config, mesh, solution_type, solution);
  SpaceTimeOperators spacetime(config, spatial);
  const PetscInt local = spatial.layout.local_size;
  const unsigned steps = config.time.steps;
  const double dt = config.time.dt;
  const ProblemConfig::Domain& D = config.domain;
  auto on_boundary = [&](PetscInt i) {
    const double x = spatial.layout.x[i], z = spatial.layout.z[i];
    return std::fabs(x - D.xmin) < 1e-12 || std::fabs(x - D.xmax) < 1e-12 ||
           std::fabs(z - D.zmin) < 1e-12 || std::fabs(z - D.zmax) < 1e-12;
  };

  // cell Peclet number |V| h / (2 nu) with h the element size (Q2: two
  // node spacings per element); reported for the coarser direction.
  const double hx = (D.xmax - D.xmin) / config.mesh.nx;
  const double hz = (D.zmax - D.zmin) / config.mesh.nz;
  const double speed = std::hypot(config.convection.vx, config.convection.vy);
  std::printf("%s_CONFIG W=%.15g H=%.15g nx=%u nz=%u steps=%u T=%.6g dt=%.6e"
              " scheme=%s nu=%.6g vx=%.6g vy=%.6g alpha_u=%.3e alpha_track=%.6g"
              " initial=%s variant=%s pc=%d hx=%.6e hz=%.6e Pe_cell=%.6e Pe_x=%.6e Pe_z=%.6e\n",
              tag, D.xmax - D.xmin, D.zmax - D.zmin, config.mesh.nx,
              config.mesh.nz, steps, config.final_time(), dt,
              TimeSchemeName(config), config.diffusivity, config.convection.vx,
              config.convection.vy, config.alpha_u, config.alpha_track,
              g_initial_name, g_case == Case::EX5 ? g_ex5_variant.c_str() : "source",
              static_cast<int>(options.pc_mode), hx, hz,
              speed * std::max(hx, hz) / (2.0 * config.diffusivity),
              std::fabs(config.convection.vx) * hx / (2.0 * config.diffusivity),
              std::fabs(config.convection.vy) * hz / (2.0 * config.diffusivity));

  // ---- gate 1: region measures.  The production matrices have the
  // Dirichlet rows/columns eliminated (boxes touching dOmega lose boundary
  // rows), so the centroid selection is gated on an UN-eliminated assembly of
  // the same boxes; the eliminated measures are printed for information.
  ProblemConfig natural = config;
  natural.state_bc = ProblemConfig::StateBoundaryCondition::NATURAL;
  natural.convection.enabled = false;
  NativeSpatialOperators raw = BuildNativeSpatialOperators(
      natural, mesh, solution_type, nullptr, 0u);
  auto mat = [](const std::unique_ptr<SparseMatrix>& m) {
    return static_cast<PetscMatrix*>(m.get())->mat();
  };
  std::printf("%s_GATE_REGIONS |Omega_b|=%.15f (%.15f) |Omega_c|=%.15f (%.15f)"
              " |Omega_d|=%.15f (%.15f) |Omega|=%.15f (%.15f)"
              " eliminated: b=%.6f c=%.6f d=%.6f Omega=%.6f\n",
              tag, Measure(mat(raw.M_ctrl_box)), Area(config.control.region),
              Measure(mat(raw.M_track)), Area(config.tracking_box),
              Measure(mat(raw.M_source)), Area(config.source_regions[0]),
              Measure(mat(raw.M)), (D.xmax - D.xmin) * (D.zmax - D.zmin),
              Measure(spatial.control), Measure(spatial.tracking),
              Measure(spatial.source), Measure(spatial.mass));

  // ---- gate 2: initial condition (interpolation error; boundary trace that
  // the Dirichlet state space removes)
  Vec y0 = nullptr, work = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, MatCreateVecs(spatial.mass, &y0, &work));
  {
    PetscScalar* a = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD, VecGetArray(y0, &a));
    double trace = 0.0;
    for (PetscInt i = 0; i < local; ++i) {
      a[i] = EvaluateInitialState(config, spatial.layout.x[i], spatial.layout.z[i]);
      if (on_boundary(i)) trace = std::max(trace, std::fabs(a[i]));
    }
    ProblemConfig copy = config;
    const SpatialFieldIntegrals I =
        IntegrateSpatialField(mesh, solution_type, a, InitialExact, &copy);
    // the discrete state space has zero trace: e(0) uses the restricted y0
    for (PetscInt i = 0; i < local; ++i)
      if (on_boundary(i)) a[i] = 0.0;
    PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArray(y0, &a));
    std::printf("%s_GATE_IC ||I_h z0 - z0||_L2=%.3e ||z0||_L2=%.12f"
                " max|z0| on dOmega (removed)=%.6e\n", tag,
                std::sqrt(I.error_squared), std::sqrt(I.field_squared), trace);
  }

  // ---- gate 3: data at fixed times, at a point of Omega_c and of Omega_d
  {
    const ProblemConfig::Box& c = config.tracking_box;
    const ProblemConfig::Box& d = config.source_regions[0];
    const double xc = 0.5 * (c.xmin + c.xmax), yc = c.zmin + 0.3 * (c.zmax - c.zmin);
    const double xd = 0.5 * (d.xmin + d.xmax), yd = d.zmin + 0.3 * (d.zmax - d.zmin);
    for (double t : {0.0, 0.75, 1.5, 3.0, 10.0})
      std::printf("%s_GATE_DATA t=%.2f r(%.6f,%.6f)=%.15e d(%.6f,%.6f)=%.15e\n",
                  tag, t, xc, yc, EvaluateTrackingTarget(config, xc, yc, t), xd,
                  yd, config.data_overrides.source(config, xd, yd, t));
  }
  // ---- gate 4: load 1^T M_d I_h(x^2 y^2) against the exact integral (Q2-exact)
  {
    PetscScalar* a = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD, VecGetArray(work, &a));
    for (PetscInt i = 0; i < local; ++i) {
      const double x = spatial.layout.x[i], z = spatial.layout.z[i];
      a[i] = x * x * z * z;
    }
    PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArray(work, &a));
    Vec one = nullptr, r = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(work, &one));
    PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(work, &r));
    PetscCallAbort(PETSC_COMM_WORLD, VecSet(one, 1.0));
    PetscCallAbort(PETSC_COMM_WORLD, MatMult(mat(raw.M_source), work, r));
    const ProblemConfig::Box& d = config.source_regions[0];
    const double exact = (std::pow(d.xmax, 3) - std::pow(d.xmin, 3)) / 3.0 *
                         (std::pow(d.zmax, 3) - std::pow(d.zmin, 3)) / 3.0;
    const double full = Dot(one, r);
    PetscCallAbort(PETSC_COMM_WORLD, MatMult(spatial.source, work, r));
    // the production load drops the boundary rows (inherited M_elim I_h f)
    std::printf("%s_GATE_LOAD 1^T M_d I_h(x^2 y^2)=%.15e exact=%.15e"
                " production(eliminated)=%.15e\n", tag, full, exact, Dot(one, r));
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

  // ---- gate 5: independent forward march with the optimal control (LU)
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
  PetscCallAbort(PETSC_COMM_WORLD, KSPSetOptionsPrefix(lu, "b2d_march_"));
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
  auto fill_target = [&](double t) {
    PetscScalar* a = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD, VecGetArray(target, &a));
    for (PetscInt i = 0; i < local; ++i)
      a[i] = EvaluateTrackingTarget(config, spatial.layout.x[i], spatial.layout.z[i], t);
    PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArray(target, &a));
  };
  double march_diff = 0.0, march_norm = 0.0, boundary_max = 0.0;
  double control_l2_sq = 0.0;
  std::vector<double> times(1, 0.0), errors;
  fill_target(0.0);
  PetscCallAbort(PETSC_COMM_WORLD, VecWAXPY(diff, -1.0, target, y0));
  PetscCallAbort(PETSC_COMM_WORLD, MatMult(spatial.tracking, diff, mdiff));
  errors.push_back(std::sqrt(std::max(0.0, Dot(diff, mdiff))));
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
    CopyDataSideSliceToSpatial(f.state, slot, local, tmp);
    PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(tmp, -1.0, y));
    PetscReal nd = 0.0, ny = 0.0;
    PetscCallAbort(PETSC_COMM_WORLD, VecNorm(tmp, NORM_INFINITY, &nd));
    PetscCallAbort(PETSC_COMM_WORLD, VecNorm(y, NORM_INFINITY, &ny));
    march_diff = std::max(march_diff, static_cast<double>(nd));
    march_norm = std::max(march_norm, static_cast<double>(ny));
    // tracking error of the PRODUCTION state
    CopyDataSideSliceToSpatial(f.state, slot, local, y);
    {
      const PetscScalar* ys = nullptr;
      PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(y, &ys));
      for (PetscInt i = 0; i < local; ++i)
        if (on_boundary(i)) boundary_max = std::max(boundary_max, std::fabs(ys[i]));
      PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(y, &ys));
    }
    fill_target(t);
    PetscCallAbort(PETSC_COMM_WORLD, VecWAXPY(diff, -1.0, target, y));
    PetscCallAbort(PETSC_COMM_WORLD, MatMult(spatial.tracking, diff, mdiff));
    times.push_back(t);
    errors.push_back(std::sqrt(std::max(0.0, Dot(diff, mdiff))));
  }
  std::printf("%s_GATE_STATE independent_march_vs_production"
              " max|dy|/max|y|=%.3e max|y| on the Dirichlet boundary=%.3e\n",
              tag, march_diff / march_norm, boundary_max);

  // ---- Example 5 only: the uncontrolled, undisturbed FREE response of the
  // cascade runs' plant IC phi = xy seen on Omega_c, ||chi_c S(t) xy||_{L2}
  // (whatever this run's own IC).  For t < 0.5 the cascade error is ~0 (r and
  // d are switched off by F), so Fig. 45's plant error IS this field: it
  // depends on the geometry (W, H) and not on any controller.
  if (g_case == Case::EX5) {
    ProblemConfig free_config = config;
    free_config.data_overrides.source =
        [](const ProblemConfig&, double, double, double) { return 0.0; };
    free_config.data_overrides.initial_state = InitialXY;
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

  // ---- windows
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
  const double T = config.final_time();
  const double lows[3] = {g_case == Case::EX5 ? 0.0 : 2.0,
                          g_case == Case::EX5 ? 3.0 : 5.0,
                          g_case == Case::EX5 ? 0.0 : 10.0};
  const double highs[3] = {T, T, g_case == Case::EX5 ? 9.5 : T};
  std::printf("%s_TAIL", tag);
  for (int w = 0; w < 3; ++w) {
    double s, q;
    window(lows[w], highs[w], s, q);
    std::printf(" sup_%g_%g=%.10e rms_%g_%g=%.10e", lows[w], highs[w], s,
                lows[w], highs[w], q);
  }
  std::printf("\n");
  std::printf("%s_RESULT reason=%d outer_its=%d true_residual=%.3e"
              " inner_y_its=%lld inner_p_its=%lld sweeps=%d sweep_failures=%lld"
              " J_track=%.12e J_reg=%.12e J_total=%.12e u_L2=%.12e e_T=%.10e"
              " solve_wall=%.3f total_wall=%.3f\n",
              tag, solve.reason, solve.outer_iterations, solve.true_residual,
              solve.inner_y_iterations, solve.inner_p_iterations,
              solve.sweeps_used ? 1 : 0, solve.sweep_failures, f.J_track,
              f.J_reg, f.J_track + f.J_reg, std::sqrt(control_l2_sq),
              errors.back(), solve_wall, MPI_Wtime() - t0);
  // Solver cost: the existing per-sweep-context counters (spatial solves and
  // iterations of the right / left / ay sweeps of the matching Schur
  // preconditioner and of the y-split sweep) and the solve timings.
  auto counter = [&](const char* name, const InnerIterationCounter& c) {
    std::printf(" %s_solves=%lld %s_its=%lld %s_max=%lld", name, c.solves, name,
                c.iterations, name, c.max_iterations);
  };
  std::printf("%s_SOLVER_STATS outer_its=%d sweeps=%d", tag,
              solve.outer_iterations, solve.sweeps_used ? 1 : 0);
  counter("right", solve.right_inner);
  counter("left", solve.left_inner);
  counter("ay", solve.ay_inner);
  counter("ysplit", solve.y_split);
  // the Schur split's own GMRES (one entry per outer PC application)
  counter("schur_ksp", solve.p_split);
  std::printf(" sweep_failures=%lld operators_s=%.3f pc_setup_s=%.3f"
              " ksp_solve_s=%.3f\n", solve.sweep_failures,
              solve.timings.operators, solve.timings.pc_setup,
              solve.timings.ksp_solve);
  for (std::size_t k = 0; k < times.size(); ++k)
    std::printf("%s_SERIES t=%.10f e_c=%.12e\n", tag, times[k], errors[k]);

  PetscCallAbort(PETSC_COMM_WORLD, KSPDestroy(&lu));
  for (Vec* v : {&r, &y, &prev, &tmp, &u, &target, &diff, &mdiff, &y0, &work})
    PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(v));
  PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&coupling));
  PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&diagonal));
  return solve.accepted ? 0 : 8;
}
