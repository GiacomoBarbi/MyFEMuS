// T9 (3D phase 1): exact distributed optimal control on the unit cube
// (see ExactOc3D.hpp for the formulas).

#include "ExactOc3D.hpp"

#include <mpi.h>
#include <petscksp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>

#include "Discretization.hpp"
#include "Preconditioners.hpp"
#include "Solver.hpp"
#include "SpaceTimeOperators.hpp"

using namespace femus;

namespace {

// Shape: conv-sin-3d uses s(xi) = sin(pi xi); conv-sin3-3d uses sin^3(pi xi),
// whose data (f, y, grad y, Lap y) vanish on dOmega -- the 3D analogue of the
// 2D conv-sin3 (EX21_CONVECTION_VALIDATION.md: with sin and w != 0 the source
// has a nonzero boundary trace, which the inherited load M_elim I_h f drops,
// limiting y to h^{5/2}).
bool g_cubic = false;

struct F1 { double v, d, dd; };   // 1D factor and its first two derivatives
F1 Factor(double xi) {
  const double sn = std::sin(kPi * xi), cs = std::cos(kPi * xi);
  if (!g_cubic) return {sn, kPi * cs, -kPi * kPi * sn};
  return {sn * sn * sn, 3.0 * kPi * sn * sn * cs,
          3.0 * kPi * kPi * sn * (2.0 * cs * cs - sn * sn)};
}

double S(double x, double y, double z) {
  if (!g_cubic) return std::sin(kPi * x) * std::sin(kPi * y) * std::sin(kPi * z);
  return Factor(x).v * Factor(y).v * Factor(z).v;
}
/** w . grad S for the constant wind of the config (0 with convection off). */
double WindGradS(const ProblemConfig& c, double x, double y, double z) {
  if (!HasConvection(c)) return 0.0;
  if (g_cubic) {
    const F1 X = Factor(x), Y = Factor(y), Z = Factor(z);
    return c.convection.vx * X.d * Y.v * Z.v + c.convection.vy * X.v * Y.d * Z.v +
           c.convection.vz * X.v * Y.v * Z.d;
  }
  const double sx = std::sin(kPi * x), sy = std::sin(kPi * y), sz = std::sin(kPi * z);
  const double cx = std::cos(kPi * x), cy = std::cos(kPi * y), cz = std::cos(kPi * z);
  return kPi * (c.convection.vx * cx * sy * sz + c.convection.vy * sx * cy * sz +
                c.convection.vz * sx * sy * cz);
}
/** -Lap S (= 3 pi^2 S for the sin shape). */
double MinusLapS(double x, double y, double z) {
  if (!g_cubic) return 3.0 * kPi * kPi * S(x, y, z);
  const F1 X = Factor(x), Y = Factor(y), Z = Factor(z);
  return -(X.dd * Y.v * Z.v + X.v * Y.dd * Z.v + X.v * Y.v * Z.dd);
}
double A(double t) { return std::cos(2.0 * t); }
double dA(double t) { return -2.0 * std::sin(2.0 * t); }
double B(const ProblemConfig& c, double t) { return std::sin(c.final_time() - t); }
double dB(const ProblemConfig& c, double t) { return -std::cos(c.final_time() - t); }

double ExactY(const ProblemConfig&, double x, double y, double z, double t) {
  return A(t) * S(x, y, z);
}
double ExactP(const ProblemConfig& c, double x, double y, double z, double t) {
  return c.alpha_u * B(c, t) * S(x, y, z);
}
double ExactU(const ProblemConfig& c, double x, double y, double z, double t) {
  return -c.eta * ExactP(c, x, y, z, t) / c.alpha_u;
}
double Source3D(const ProblemConfig& c, double x, double y, double z, double t) {
  if (g_cubic)   // f = A' S - nu A Lap S + A w.grad S - eta u
    return dA(t) * S(x, y, z) + c.diffusivity * A(t) * MinusLapS(x, y, z) +
           A(t) * WindGradS(c, x, y, z) - c.eta * ExactU(c, x, y, z, t);
  const double s = S(x, y, z), lap = 3.0 * kPi * kPi;
  return (dA(t) + c.diffusivity * lap * A(t)) * s + A(t) * WindGradS(c, x, y, z) -
         c.eta * ExactU(c, x, y, z, t);
}
double Target3D(const ProblemConfig& c, double x, double y, double z, double t) {
  if (g_cubic) {   // y_d = y + alpha_u (B' S + nu B Lap S + B w.grad S)/alpha_t
    const double q = c.alpha_u * (dB(c, t) * S(x, y, z) - c.diffusivity * B(c, t) * MinusLapS(x, y, z) +
                                  B(c, t) * WindGradS(c, x, y, z));
    return ExactY(c, x, y, z, t) + q / c.alpha_track;
  }
  const double s = S(x, y, z), lap = 3.0 * kPi * kPi;
  // y_d = y + (p_t + nu Lap p + w.grad p)/alpha_t
  const double q = c.alpha_u * ((dB(c, t) - c.diffusivity * lap * B(c, t)) * s +
                                B(c, t) * WindGradS(c, x, y, z));
  return ExactY(c, x, y, z, t) + q / c.alpha_track;
}
double Initial3D(const ProblemConfig& c, double x, double y, double z) {
  return ExactY(c, x, y, z, 0.0);
}

struct Context3D {
  const ProblemConfig* config;
  double time;
  int field;   // 0 y, 1 p, 2 u
};
double ExactAt3D(double x, double y, double z, void* raw) {
  const Context3D* e = static_cast<const Context3D*>(raw);
  switch (e->field) {
    case 0: return ExactY(*e->config, x, y, z, e->time);
    case 1: return ExactP(*e->config, x, y, z, e->time);
    default: return ExactU(*e->config, x, y, z, e->time);
  }
}

double RelDiff(Vec x, Vec b) {
  Vec d = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(x, &d));
  PetscCallAbort(PETSC_COMM_WORLD, VecWAXPY(d, -1.0, b, x));
  PetscReal nd = 0.0, nb = 0.0;
  PetscCallAbort(PETSC_COMM_WORLD, VecNorm(d, NORM_2, &nd));
  PetscCallAbort(PETSC_COMM_WORLD, VecNorm(b, NORM_2, &nb));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&d));
  return nb > 0.0 ? nd / nb : nd;
}

}  // namespace

void ApplyExactOptimalControl3D(ProblemConfig& config, const std::string& name,
                                double alpha_u, double final_time, double nu) {
  if (name != "conv-sin-3d" && name != "conv-sin3-3d")
    throw std::invalid_argument("the 3D exact OC cases are conv-sin-3d and conv-sin3-3d");
  g_cubic = name == "conv-sin3-3d";
  config.dimension = 3u;
  config.domain = ProblemConfig::Domain();   // unit cube
  config.state_bc = ProblemConfig::StateBoundaryCondition::HOMOGENEOUS_DIRICHLET;
  config.control.type = ProblemConfig::ControlType::DISTRIBUTED;
  config.control.region = {0.0, 1.0, 0.0, 1.0};
  config.tracking_box = {0.0, 1.0, 0.0, 1.0};
  config.additional_control_regions.clear();
  config.additional_tracking_regions.clear();
  config.source_regions.clear();
  config.alpha_track = 1.0;
  config.eta = 1.0;
  config.diffusivity = nu > 0.0 ? nu : 0.25;
  config.alpha_u = alpha_u > 0.0 ? alpha_u : 1.0e-2;
  const double T = final_time > 0.0 ? final_time : 1.0;
  config.time.dt = T / config.time.steps;
  config.data_overrides.source3d = Source3D;
  config.data_overrides.target3d = Target3D;
  config.data_overrides.initial3d = Initial3D;
}

int RunExactOptimalControl3D(const ProblemConfig& config,
                             const ApplicationOptions& options,
                             MultiLevelSolution& solution, Mesh* mesh,
                             unsigned solution_type) {
  int ranks = 0;
  MPI_Comm_size(PETSC_COMM_WORLD, &ranks);
  if (ranks != 1)
    throw std::runtime_error("--exact-oc conv-sin-3d runs serially (use --solve-only for MPI)");
  const double t0 = MPI_Wtime();
  SpatialOperators spatial(config, mesh, solution_type, solution);
  SpaceTimeOperators spacetime(config, spatial);
  const PetscInt local = spatial.layout.local_size;
  const unsigned steps = config.time.steps;
  const double dt = config.time.dt, T = config.final_time();
  const bool cn = IsCrankNicolson(config);
  const double h = 1.0 / config.mesh.nx;
  const double speed = std::sqrt(config.convection.vx * config.convection.vx +
                                 config.convection.vy * config.convection.vy +
                                 config.convection.vz * config.convection.vz);
  std::printf("EX21_EXACT3D_CONFIG case=%s nx=%u ny=%u nz=%u steps=%u T=%.6g dt=%.10e"
              " scheme=%s nu=%.6g convection=%s vx=%.6g vy=%.6g vz=%.6g alpha_u=%.6e"
              " h=%.6e Pe_cell=%.6e dofs=%d spacetime_unknowns=%lld pc=%d\n",
              g_cubic ? "conv-sin3-3d" : "conv-sin-3d",
              config.mesh.nx, config.mesh.ny, config.mesh.nz, steps, T, dt,
              TimeSchemeName(config), config.diffusivity, HasConvection(config) ? "on" : "off",
              config.convection.vx, config.convection.vy, config.convection.vz,
              config.alpha_u, h, speed * h / (2.0 * config.diffusivity), (int)local,
              3LL * steps * local, static_cast<int>(options.pc_mode));

  // tightened production solve: the policy of the 2D exact driver
  KktSolverPolicy policy = MakeKktSolverPolicy(options.pc_mode, 1.0e-12);
  policy.outer_rtol = 1.0e-12;
  policy.inner_rtol = 1.0e-10;
  policy.outer_maxits = 400;
  if (options.exact_oc_production_tol)
    policy = MakeKktSolverPolicy(options.pc_mode, options.spatial_tolerance);
  const double t_solve = MPI_Wtime();
  DistributedOptimalControlSolve solve = SolveOptimalControl(config, spatial, spacetime, policy);
  const double solve_wall = MPI_Wtime() - t_solve;
  const DistributedDataSideResult& f = solve.fields;

  // ---- independent forward march with the computed control (spatial LU)
  DistributedDataSideResult data = BuildDistributedRightHandSide(
      config, spatial.layout, spatial.mass, spatial.tracking, spacetime.distributed.ay,
      spatial.stiffness, spatial.source, &spatial.lift);
  Mat diagonal = nullptr, coupling = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, MatDuplicate(spatial.stiffness, MAT_COPY_VALUES, &diagonal));
  PetscCallAbort(PETSC_COMM_WORLD, MatScale(diagonal, cn ? 0.5 : 1.0));
  PetscCallAbort(PETSC_COMM_WORLD, MatAXPY(diagonal, 1.0 / dt, spatial.mass, SUBSET_NONZERO_PATTERN));
  PetscCallAbort(PETSC_COMM_WORLD, MatDuplicate(spatial.mass, MAT_COPY_VALUES, &coupling));
  PetscCallAbort(PETSC_COMM_WORLD, MatScale(coupling, 1.0 / dt));
  if (cn)
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatAXPY(coupling, -0.5, spatial.stiffness, DIFFERENT_NONZERO_PATTERN));
  KSP lu = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, KSPCreate(PETSC_COMM_WORLD, &lu));
  PetscCallAbort(PETSC_COMM_WORLD, KSPSetOptionsPrefix(lu, "exact_march_"));
  PetscCallAbort(PETSC_COMM_WORLD, KSPSetOperators(lu, diagonal, diagonal));
  PetscCallAbort(PETSC_COMM_WORLD, KSPSetType(lu, KSPPREONLY));
  PC pc = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, KSPGetPC(lu, &pc));
  PetscCallAbort(PETSC_COMM_WORLD, PCSetType(pc, PCLU));
  PetscCallAbort(PETSC_COMM_WORLD, KSPSetUp(lu));
  Vec r = nullptr, y = nullptr, prev = nullptr, tmp = nullptr, u = nullptr, marched = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, MatCreateVecs(spatial.mass, &r, &y));
  for (Vec* v : {&prev, &tmp, &u}) PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(r, v));
  PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(f.state, &marched));
  for (unsigned slot = 0; slot < steps; ++slot) {
    CopyDataSideSliceToSpatial(data.rhs_y, slot, local, r);
    CopyDataSideSliceToSpatial(f.control, slot, local, u);
    PetscCallAbort(PETSC_COMM_WORLD, MatMult(spatial.control, u, tmp));
    PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(r, config.eta, tmp));
    if (slot > 0u) {
      PetscCallAbort(PETSC_COMM_WORLD, MatMult(coupling, prev, tmp));
      PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(r, 1.0, tmp));
    }
    PetscCallAbort(PETSC_COMM_WORLD, KSPSolve(lu, r, y));
    PetscCallAbort(PETSC_COMM_WORLD, VecCopy(y, prev));
    CopySpatialToDataSideSlice(y, slot, local, 1.0, marched, false);
  }
  const double march_rel = RelDiff(f.state, marched);
  // stationarity alpha_u u + eta p = 0 (full-domain L2 control)
  Vec stat = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(f.control, &stat));
  PetscCallAbort(PETSC_COMM_WORLD, VecSet(stat, 0.0));
  PetscCallAbort(PETSC_COMM_WORLD,
                 VecAXPBYPCZ(stat, config.alpha_u, config.eta, 0.0, f.control, f.adjoint));
  PetscReal stat_norm = 0.0, au_norm = 0.0;
  PetscCallAbort(PETSC_COMM_WORLD, VecNorm(stat, NORM_2, &stat_norm));
  PetscCallAbort(PETSC_COMM_WORLD, VecNorm(f.control, NORM_2, &au_norm));
  const double stat_rel = stat_norm / (config.alpha_u * (au_norm > 0 ? au_norm : 1.0));

  // ---- FEM-integrated space-time L2 errors
  struct Acc { double e = 0, x = 0, last_e = 0, last_x = 0, max_e = 0; } ey, ep, eu;
  for (unsigned slot = 0; slot < steps; ++slot) {
    const double tn = (slot + 1u) * dt, tm = (slot + 0.5) * dt;
    const double tp = cn ? tm : tn;
    auto integrate = [&](Vec field, int which, double time) {
      CopyDataSideSliceToSpatial(field, slot, local, r);
      const PetscScalar* a = nullptr;
      PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(r, &a));
      Context3D ctx{&config, time, which};
      const SpatialFieldIntegrals I =
          IntegrateSpatialField3D(mesh, solution_type, a, ExactAt3D, &ctx);
      PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(r, &a));
      return I;
    };
    const SpatialFieldIntegrals Iy = integrate(f.state, 0, tn);
    const SpatialFieldIntegrals Ip = integrate(f.adjoint, 1, tp);
    const SpatialFieldIntegrals Iu = integrate(f.control, 2, tp);
    for (auto pr : {std::make_pair(&ey, &Iy), std::make_pair(&ep, &Ip), std::make_pair(&eu, &Iu)}) {
      pr.first->e += dt * pr.second->error_squared;
      pr.first->x += dt * pr.second->exact_squared;
      pr.first->max_e = std::max(pr.first->max_e, std::sqrt(pr.second->error_squared));
      if (slot + 1u == steps) { pr.first->last_e = pr.second->error_squared; pr.first->last_x = pr.second->exact_squared; }
    }
  }
  auto counter = [&](const char* nm, const InnerIterationCounter& c) {
    std::printf(" %s_solves=%lld %s_its=%lld %s_max=%lld", nm, c.solves, nm, c.iterations, nm,
                c.max_iterations);
  };
  std::printf("EX21_EXACT3D_SOLVER_STATS outer_its=%d reason=%d true_residual=%.3e"
              " state_rel=%.3e adjoint_rel=%.3e sweeps=%d",
              solve.outer_iterations, solve.reason, solve.true_residual, solve.kkt.state_rel,
              solve.kkt.adjoint_rel, solve.sweeps_used ? 1 : 0);
  counter("right", solve.right_inner);
  counter("left", solve.left_inner);
  counter("ay", solve.ay_inner);
  counter("ysplit", solve.y_split);
  counter("schur_ksp", solve.p_split);
  std::printf(" sweep_failures=%lld solve_wall=%.3f accepted=%d status=%s\n",
              solve.sweep_failures, solve_wall, solve.accepted ? 1 : 0, solve.status.c_str());
  auto rel = [](double e, double x) { return x > 0 ? std::sqrt(e / x) : std::sqrt(e); };
  std::printf("EX21_EXACT3D_RESULT scheme=%s nx=%u ny=%u nz=%u steps=%u dt=%.10e reason=%d"
              " outer_its=%d true_residual=%.3e stationarity_rel=%.3e march_rel=%.3e"
              " J_track=%.12e J_reg=%.12e"
              " y_L2L2=%.10e y_L2L2_rel=%.10e y_last=%.10e y_Linf_L2=%.10e"
              " p_L2L2=%.10e p_L2L2_rel=%.10e p_last=%.10e p_Linf_L2=%.10e"
              " u_L2L2=%.10e u_L2L2_rel=%.10e u_last=%.10e u_Linf_L2=%.10e"
              " solve_wall=%.3f total_wall=%.3f\n",
              TimeSchemeName(config), config.mesh.nx, config.mesh.ny, config.mesh.nz, steps, dt,
              solve.reason, solve.outer_iterations, solve.true_residual, stat_rel, march_rel,
              f.J_track, f.J_reg,
              std::sqrt(ey.e), rel(ey.e, ey.x), std::sqrt(ey.last_e), ey.max_e,
              std::sqrt(ep.e), rel(ep.e, ep.x), std::sqrt(ep.last_e), ep.max_e,
              std::sqrt(eu.e), rel(eu.e, eu.x), std::sqrt(eu.last_e), eu.max_e,
              solve_wall, MPI_Wtime() - t0);
  for (Vec* v : {&r, &y, &prev, &tmp, &u, &marched, &stat})
    PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(v));
  PetscCallAbort(PETSC_COMM_WORLD, KSPDestroy(&lu));
  PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&diagonal));
  PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&coupling));
  return solve.accepted ? 0 : 8;
}
