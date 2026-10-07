// ex21 convection gates.

#include "ConvectionVerification.hpp"

#include <mpi.h>
#include <petscksp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "Discretization.hpp"
#include "PetscMatrix.hpp"
#include "Solver.hpp"
#include "SpaceTimeOperators.hpp"

using namespace femus;

namespace {

double Dot(Vec a, Vec b) {
  PetscScalar value = 0.0;
  PetscCallAbort(PETSC_COMM_WORLD, VecDot(a, b, &value));
  return PetscRealPart(value);
}

double Norm(Vec a, NormType type = NORM_2) {
  PetscReal value = 0.0;
  PetscCallAbort(PETSC_COMM_WORLD, VecNorm(a, type, &value));
  return value;
}

/** ||a - b|| / ||b|| (absolute when b == 0). */
double RelativeDifference(Vec a, Vec b, NormType type = NORM_2) {
  Vec d = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(a, &d));
  PetscCallAbort(PETSC_COMM_WORLD, VecWAXPY(d, -1.0, b, a));
  const double nd = Norm(d, type), nb = Norm(b, type);
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&d));
  return nb > 0.0 ? nd / nb : nd;
}

Vec NewLike(Vec v) {
  Vec out = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(v, &out));
  PetscCallAbort(PETSC_COMM_WORLD, VecSet(out, 0.0));
  return out;
}

/** Deterministic pseudo-random vector (fixed seed per call site). */
void FillRandom(Vec v, unsigned long seed) {
  PetscRandom r = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, PetscRandomCreate(PETSC_COMM_WORLD, &r));
  PetscCallAbort(PETSC_COMM_WORLD, PetscRandomSetInterval(r, -1.0, 1.0));
  PetscCallAbort(PETSC_COMM_WORLD, PetscRandomSetSeed(r, seed));
  PetscCallAbort(PETSC_COMM_WORLD, PetscRandomSeed(r));
  PetscCallAbort(PETSC_COMM_WORLD, VecSetRandom(v, r));
  PetscCallAbort(PETSC_COMM_WORLD, PetscRandomDestroy(&r));
}

/** out = (diag(scale) (x) S) in, slot by slot. */
void ApplyBlockDiagonal(Mat spatial, Vec in, Vec out, unsigned steps,
                        PetscInt local, const std::vector<double>& scale) {
  Vec s_in = nullptr, s_out = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, MatCreateVecs(spatial, &s_in, &s_out));
  PetscCallAbort(PETSC_COMM_WORLD, VecSet(out, 0.0));
  for (unsigned slot = 0; slot < steps; ++slot) {
    CopyDataSideSliceToSpatial(in, slot, local, s_in);
    PetscCallAbort(PETSC_COMM_WORLD, MatMult(spatial, s_in, s_out));
    CopySpatialToDataSideSlice(s_out, slot, local, scale[slot], out, false);
  }
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&s_out));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&s_in));
}

/** A direct (serial LU) solver for an assembled matrix. */
KSP MakeDirect(Mat a, const char* prefix) {
  KSP ksp = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCallAbort(PETSC_COMM_WORLD, KSPSetOptionsPrefix(ksp, prefix));
  PetscCallAbort(PETSC_COMM_WORLD, KSPSetOperators(ksp, a, a));
  PetscCallAbort(PETSC_COMM_WORLD, KSPSetType(ksp, KSPPREONLY));
  PC pc = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, KSPGetPC(ksp, &pc));
  PetscCallAbort(PETSC_COMM_WORLD, PCSetType(pc, PCLU));
  PetscCallAbort(PETSC_COMM_WORLD, KSPSetFromOptions(ksp));
  PetscCallAbort(PETSC_COMM_WORLD, KSPSetUp(ksp));
  return ksp;
}

/** GMRES on a MATRIX-FREE operator, preconditioned by the LU of an assembled
 *  matrix: converges in one iteration iff the shell equals that matrix. */
KSP MakeShellSolver(Mat shell, Mat assembled, const char* prefix) {
  KSP ksp = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCallAbort(PETSC_COMM_WORLD, KSPSetOptionsPrefix(ksp, prefix));
  PetscCallAbort(PETSC_COMM_WORLD, KSPSetOperators(ksp, shell, assembled));
  PetscCallAbort(PETSC_COMM_WORLD, KSPSetType(ksp, KSPGMRES));
  PetscCallAbort(PETSC_COMM_WORLD, KSPSetPCSide(ksp, PC_RIGHT));
  PetscCallAbort(PETSC_COMM_WORLD,
                 KSPSetTolerances(ksp, 1.0e-14, 0.0, PETSC_DEFAULT, 50));
  PC pc = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, KSPGetPC(ksp, &pc));
  PetscCallAbort(PETSC_COMM_WORLD, PCSetType(pc, PCLU));
  PetscCallAbort(PETSC_COMM_WORLD, KSPSetFromOptions(ksp));
  PetscCallAbort(PETSC_COMM_WORLD, KSPSetUp(ksp));
  return ksp;
}

int Solve(KSP ksp, Vec b, Vec x) {
  PetscCallAbort(PETSC_COMM_WORLD, VecSet(x, 0.0));
  PetscCallAbort(PETSC_COMM_WORLD, KSPSolve(ksp, b, x));
  PetscInt its = 0;
  PetscCallAbort(PETSC_COMM_WORLD, KSPGetIterationNumber(ksp, &its));
  return static_cast<int>(its);
}

void RequireSerial(const char* mode) {
  int ranks = 0;
  MPI_Comm_size(PETSC_COMM_WORLD, &ranks);
  if (ranks != 1)
    throw std::runtime_error(std::string(mode) + " runs serially (np = 1)");
}

/** The reduced problem J(U) of the REPORTED objective, up to the additive
 *  constant sum_n dt (alpha_t/2) w_n y_d^T M_omega y_d:
 *
 *    E Y = rhs_y + eta (I (x) N) U
 *    J   = sum_n dt [ (alpha_t/2) w_n Y_n^T M_omega Y_n + Y_n^T rhs_p,n
 *                     + (alpha_u/2) U_n^T N U_n ]
 *
 *  (rhs_p = -alpha_t (W (x) M_omega) Y_d, so the cross term is exact), and
 *
 *    grad J = dt alpha_u (I (x) N) U + eta (I (x) N) Lambda,
 *    E^T Lambda = dt [ alpha_t (W (x) M_omega) Y + rhs_p ].
 *
 *  The forward solve always uses the LU of the assembled E; the adjoint solve
 *  is whichever KSP the caller passes, so the SAME gradient formula is
 *  evaluated with the exact transpose, the production shell and a deliberately
 *  wrong adjoint. */
struct ReducedProblem {
  const ProblemConfig* config = nullptr;
  Mat control = nullptr, tracking = nullptr;
  Vec rhs_y = nullptr, rhs_p = nullptr;
  KSP forward = nullptr;
  PetscInt local = 0;
  std::vector<double> ones, weights;

  void State(Vec U, Vec Y) const {
    Vec b = NewLike(rhs_y);
    ApplyBlockDiagonal(control, U, b, config->time.steps, local, ones);
    PetscCallAbort(PETSC_COMM_WORLD, VecAYPX(b, config->eta, rhs_y));
    Solve(forward, b, Y);
    PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&b));
  }

  double Objective(Vec U) const {
    const double dt = config->time.dt;
    Vec Y = NewLike(rhs_y), MY = NewLike(rhs_y), NU = NewLike(rhs_y);
    State(U, Y);
    ApplyBlockDiagonal(tracking, Y, MY, config->time.steps, local, weights);
    ApplyBlockDiagonal(control, U, NU, config->time.steps, local, ones);
    const double j = dt * (0.5 * config->alpha_track * Dot(Y, MY) +
                           Dot(Y, rhs_p) + 0.5 * config->alpha_u * Dot(U, NU));
    for (Vec* v : {&Y, &MY, &NU}) PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(v));
    return j;
  }

  /** T3b: the PHYSICAL tracking functional of inhomogeneous Dirichlet data,
   *    dt sum_n w_n (alpha_t/2) (Y_n + G - D_n)^T M_full (Y_n + G - D_n)
   *    + dt (alpha_u/2) U^T N U
   *  (the CN initial node is a constant and is omitted).  Independent of
   *  rhs_p: it is the functional the adjoint gradient must differentiate. */
  Mat tracking_full = nullptr;
  Vec lift = nullptr, target = nullptr;
  double ObjectivePhysical(Vec U) const {
    const double dt = config->time.dt;
    Vec Y = NewLike(rhs_y), NU = NewLike(rhs_y);
    State(U, Y);
    Vec d = nullptr, md = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD, MatCreateVecs(tracking_full, &d, &md));
    double j = 0.0;
    for (unsigned slot = 0; slot < config->time.steps; ++slot) {
      CopyDataSideSliceToSpatial(Y, slot, local, d);
      PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(d, 1.0, lift));
      CopyDataSideSliceToSpatial(target, slot, local, md);
      PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(d, -1.0, md));
      PetscCallAbort(PETSC_COMM_WORLD, MatMult(tracking_full, d, md));
      j += dt * weights[slot] * 0.5 * config->alpha_track * Dot(d, md);
    }
    ApplyBlockDiagonal(control, U, NU, config->time.steps, local, ones);
    j += dt * 0.5 * config->alpha_u * Dot(U, NU);
    for (Vec* v : {&Y, &NU, &d, &md}) PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(v));
    return j;
  }

  /** Returns the adjoint-solve iteration count. */
  int Gradient(Vec U, KSP adjoint, Vec g) const {
    const double dt = config->time.dt;
    Vec Y = NewLike(rhs_y), G = NewLike(rhs_y), L = NewLike(rhs_y);
    State(U, Y);
    ApplyBlockDiagonal(tracking, Y, G, config->time.steps, local, weights);
    PetscCallAbort(PETSC_COMM_WORLD, VecScale(G, config->alpha_track));
    PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(G, 1.0, rhs_p));
    PetscCallAbort(PETSC_COMM_WORLD, VecScale(G, dt));
    const int its = Solve(adjoint, G, L);
    // g = dt alpha_u N U + eta N Lambda = N (dt alpha_u U + eta Lambda)
    PetscCallAbort(PETSC_COMM_WORLD, VecAXPBY(L, dt * config->alpha_u,
                                              config->eta, U));
    ApplyBlockDiagonal(control, L, g, config->time.steps, local, ones);
    for (Vec* v : {&Y, &G, &L}) PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(v));
    return its;
  }
};

/** T7D: the Dirichlet-control reduced problem (boundary_convection/T7D_*.md
 *  section 1), built only from the production operators:
 *
 *    E_I Y = rhs_y - eta B_D U         (rhs_y: free rows, u_0 data included)
 *    J(U)  = dt sum_n [ (alpha_t/2) w_n Y_n^T M_w Y_n + Y_n^T rhs_p,n ]
 *          + (alpha_u/2) dt sum_n What_u,n U_n^T R_Gamma U_n
 *          + (gamma_u/(2 dt)) sum_n (U_n - U_{n-1})^T M_Gamma (U_n - U_{n-1})   (U_0 = u_0)
 *    grad J = -eta B_D^T Lambda + alpha_u dt R_H1t U - gamma_u q_0,
 *    E_I^T Lambda = dt [ alpha_t (W (x) M_w) Y + rhs_p ],
 *
 *  with R_H1t = What_u (x) R_Gamma + rho (1/dt) T (x) M_Gamma, rho = gamma_u/(alpha_u dt)
 *  (DirichletControlOperators::ApplyR) and q_0 = (1/dt) [M_Gamma u_0; 0; ...].
 *  U lives on C (I entries ignored).  `legacy` evaluates the gradient with the
 *  pre-T7D B_D^T (negative control). */
struct DirichletReducedProblem {
  const ProblemConfig* config = nullptr;
  DirichletControlOperators* d = nullptr;
  Mat tracking = nullptr;
  Vec rhs_y = nullptr, rhs_p = nullptr, q0 = nullptr, trace = nullptr;
  KSP forward = nullptr;
  PetscInt local = 0;
  std::vector<double> weights;

  void MaskC(Vec U) const {
    PetscScalar* a = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD, VecGetArray(U, &a));
    const std::vector<unsigned char>& m = d->control_mask();
    for (unsigned n = 0; n < config->time.steps; ++n)
      for (PetscInt i = 0; i < local; ++i)
        if (!m[static_cast<std::size_t>(i)]) a[static_cast<PetscInt>(n) * local + i] = 0.0;
    PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArray(U, &a));
  }
  void State(Vec U, Vec Y) const {
    Vec b = NewLike(rhs_y);
    PetscCallAbort(PETSC_COMM_WORLD, d->ApplyBD(U, b));
    PetscCallAbort(PETSC_COMM_WORLD, VecAYPX(b, -config->eta, rhs_y));
    Solve(forward, b, Y);
    PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&b));
  }
  double Objective(Vec U) const {
    const double dt = config->time.dt;
    Vec Y = NewLike(rhs_y), MY = NewLike(rhs_y), RU = NewLike(rhs_y);
    State(U, Y);
    ApplyBlockDiagonal(tracking, Y, MY, config->time.steps, local, weights);
    double j = dt * (0.5 * config->alpha_track * Dot(Y, MY) + Dot(Y, rhs_p));
    // L2 / H1-space part: (alpha_u/2) dt sum What_u U^T R_Gamma U (R embedded:
    // identity on I, so U is masked to C first)
    Vec Uc = NewLike(rhs_y);
    PetscCallAbort(PETSC_COMM_WORLD, VecCopy(U, Uc));
    MaskC(Uc);
    Vec s_in = nullptr, s_out = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD, MatCreateVecs(d->regularization(), &s_in, &s_out));
    for (unsigned n = 0; n < config->time.steps; ++n) {
      CopyDataSideSliceToSpatial(Uc, n, local, s_in);
      PetscCallAbort(PETSC_COMM_WORLD, MatMult(d->regularization(), s_in, s_out));
      j += 0.5 * config->alpha_u * dt * d->SlotWeight(n) * Dot(s_in, s_out);
    }
    if (d->has_time_regularization()) {
      // (gamma/(2dt)) sum (U_n - U_{n-1})^T M_Gamma (U_n - U_{n-1}), U_0 = u_0
      Vec prev = nullptr, cur = nullptr;
      PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(s_in, &prev));
      PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(s_in, &cur));
      PetscCallAbort(PETSC_COMM_WORLD, VecCopy(trace, prev));
      for (unsigned n = 0; n < config->time.steps; ++n) {
        CopyDataSideSliceToSpatial(Uc, n, local, cur);
        PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(prev, -1.0, cur));   // prev - cur
        PetscCallAbort(PETSC_COMM_WORLD, d->ApplyFaceMass(prev, s_out));
        j += 0.5 * config->control_h1_time_weight / dt * Dot(prev, s_out);
        PetscCallAbort(PETSC_COMM_WORLD, VecCopy(cur, prev));
      }
      PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&prev));
      PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&cur));
    }
    for (Vec* v : {&Y, &MY, &RU, &Uc, &s_in, &s_out}) PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(v));
    return j;
  }
  int Gradient(Vec U, KSP adjoint, Vec g, bool legacy) const {
    const double dt = config->time.dt;
    Vec Y = NewLike(rhs_y), G = NewLike(rhs_y), L = NewLike(rhs_y), Uc = NewLike(rhs_y);
    State(U, Y);
    ApplyBlockDiagonal(tracking, Y, G, config->time.steps, local, weights);
    PetscCallAbort(PETSC_COMM_WORLD, VecScale(G, config->alpha_track));
    PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(G, 1.0, rhs_p));
    PetscCallAbort(PETSC_COMM_WORLD, VecScale(G, dt));
    d->ZeroControlledRows(G);   // the C rows of E_I^T are identity rows: Lambda_C = 0
    const int its = Solve(adjoint, G, L);
    if (legacy) PetscCallAbort(PETSC_COMM_WORLD, d->ApplyBDTLegacy(L, g));
    else PetscCallAbort(PETSC_COMM_WORLD, d->ApplyBDT(L, g));
    PetscCallAbort(PETSC_COMM_WORLD, VecScale(g, -config->eta));
    PetscCallAbort(PETSC_COMM_WORLD, VecCopy(U, Uc));
    MaskC(Uc);
    PetscCallAbort(PETSC_COMM_WORLD, d->ApplyR(Uc, G));
    PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(g, config->alpha_u * dt, G));
    if (d->has_time_regularization())
      PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(g, -config->control_h1_time_weight, q0));
    for (Vec* v : {&Y, &G, &L, &Uc}) PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(v));
    return its;
  }
};

}  // namespace

/** T7D gates for Dirichlet boundary control (serial; called from
 *  RunConvectionVerification after the generic G1/G2 gates).  Prints:
 *    G3D  <B_D u, p> - <u, B_D^T p>  (and the pre-T7D B_D^T, negative control)
 *    R    R_H1t symmetry (and R_Gamma positivity) on random C vectors
 *    G5D  reduced Hessian symmetry, G4D FD reduced gradient, for the exact LU
 *         adjoint, the production shell, the legacy B_D^T and the ex20-style
 *         untransposed E
 *    G6D  production KKT solve: true residual, stationarity (reduced gradient at
 *         the computed optimum), independent forward march, KKT stationarity */
static void RunDirichletConvectionGates(const ProblemConfig& config,
                                        const ApplicationOptions& options,
                                        SpatialOperators& spatial,
                                        SpaceTimeOperators& spacetime) {
  DirichletControlOperators& d = *spatial.dirichlet;
  Mat ay = spacetime.assembled.ay, ap = spacetime.assembled.ap;
  const unsigned steps = config.time.steps;
  const PetscInt local = spatial.layout.local_size;
  Vec U = nullptr, P = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, MatCreateVecs(ay, &U, &P));
  FillRandom(U, 101u);
  FillRandom(P, 202u);
  Vec BU = NewLike(U), BtP = NewLike(U), BtPl = NewLike(U);
  PetscCallAbort(PETSC_COMM_WORLD, d.ApplyBD(U, BU));
  PetscCallAbort(PETSC_COMM_WORLD, d.ApplyBDT(P, BtP));
  PetscCallAbort(PETSC_COMM_WORLD, d.ApplyBDTLegacy(P, BtPl));
  const double lhs = Dot(BU, P), rhs = Dot(U, BtP), rhsl = Dot(U, BtPl);
  const double sc = Norm(BU) * Norm(P);
  std::printf("EX21_VC_G3D_BD_TRANSPOSE <B_D u,p>=%.17g <u,B_D^T p>=%.17g rel=%.3e"
              " legacy_B_D^T: <u,B_D^T_old p>=%.17g rel=%.3e transposes_stiffness=%d\n",
              lhs, rhs, std::fabs(lhs - rhs) / sc, rhsl, std::fabs(lhs - rhsl) / sc,
              d.transposes_stiffness() ? 1 : 0);
  // R_H1t symmetry and positivity
  {
    Vec U1 = NewLike(U), U2 = NewLike(U), R1 = NewLike(U), R2 = NewLike(U);
    FillRandom(U1, 303u); FillRandom(U2, 404u);
    PetscCallAbort(PETSC_COMM_WORLD, d.ApplyR(U1, R1));
    PetscCallAbort(PETSC_COMM_WORLD, d.ApplyR(U2, R2));
    const double a = Dot(R1, U2), b = Dot(U1, R2);
    double pmin = 1e300;
    for (unsigned k = 0; k < 10u; ++k) {
      FillRandom(U1, 505u + k);
      PetscCallAbort(PETSC_COMM_WORLD, d.ApplyR(U1, R1));
      Vec Uc = NewLike(U1);
      PetscCallAbort(PETSC_COMM_WORLD, VecCopy(U1, Uc));
      PetscCallAbort(PETSC_COMM_WORLD, d.ZeroControlledRows(U1));      // U1 := I part
      PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(Uc, -1.0, U1));          // Uc := C part
      pmin = std::min(pmin, Dot(Uc, R1) / std::max(Dot(Uc, Uc), 1e-300));
      PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&Uc));
    }
    std::printf("EX21_VC_RD_REGULARIZATION ell=%.6g gamma_u=%.6g <R u1,u2>-<u1,R u2> rel=%.3e"
                " min Rayleigh(u_C)=%.3e\n", config.control_h1_length,
                config.control_h1_time_weight, std::fabs(a - b) / std::max(std::fabs(a), 1e-300), pmin);
    for (Vec* v : {&U1, &U2, &R1, &R2}) PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(v));
  }
  // reduced problem
  DistributedDataSideResult data = BuildDistributedRightHandSide(
      config, spatial.layout, spatial.mass_full, spatial.tracking, ay,
      d.unhatted_stiffness(), spatial.source, &spatial.lift);
  PetscCallAbort(PETSC_COMM_WORLD, d.ZeroControlledRows(data.rhs_y));
  DirichletReducedProblem rp;
  rp.config = &config; rp.d = &d; rp.tracking = spatial.tracking;
  rp.rhs_y = data.rhs_y; rp.rhs_p = data.rhs_p; rp.local = local;
  rp.weights.resize(steps);
  for (unsigned n = 0; n < steps; ++n) rp.weights[n] = TrackingSlotWeight(config, n);
  rp.forward = MakeDirect(ay, "vcd_forward_");
  Vec trace_spatial = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, MatCreateVecs(d.regularization(), &trace_spatial, nullptr));
  {
    PetscScalar* a = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD, VecGetArray(trace_spatial, &a));
    const std::vector<unsigned char>& m = d.control_mask();
    for (PetscInt i = 0; i < local; ++i)
      a[i] = m[static_cast<std::size_t>(i)]
                 ? EvaluateInitialState(config, spatial.layout.x[i], spatial.layout.z[i], spatial.layout.Axis2(i)) / config.eta
                 : 0.0;
    PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArray(trace_spatial, &a));
  }
  rp.trace = trace_spatial;
  rp.q0 = NewLike(U);
  PetscCallAbort(PETSC_COMM_WORLD, d.ApplyAffineData(trace_spatial, rp.q0));
  KSP adj_exact = MakeDirect(ap, "vcd_adjoint_");
  KSP adj_shell = MakeShellSolver(spacetime.distributed.ap, ap, "vcd_adjoint_shell_");
  Mat stiffness_t = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, MatTranspose(spatial.stiffness, MAT_INITIAL_MATRIX, &stiffness_t));
  AssembledSpaceTimeOperators wrong_ops;
  BuildAssembledSpaceTimeOperators(wrong_ops, spatial.mass, stiffness_t, steps,
                                   config.time.dt, config.time.scheme);
  KSP adj_wrong = MakeDirect(wrong_ops.ap, "vcd_adjoint_wrong_");
  struct Variant { const char* name; KSP ksp; bool legacy; };
  const Variant variants[] = {{"exact_LU(E^T)", adj_exact, false},
                              {"production_shell(E^T)", adj_shell, false},
                              {"legacy_B_D^T", adj_exact, true},
                              {"ex20_style_untransposed_E", adj_wrong, false}};
  {
    Vec U1 = NewLike(U), U2 = NewLike(U), g0 = NewLike(U), h1 = NewLike(U), h2 = NewLike(U), z = NewLike(U);
    FillRandom(U1, 7u); FillRandom(U2, 77u);
    rp.MaskC(U1); rp.MaskC(U2);
    for (const Variant& v : variants) {
      rp.Gradient(z, v.ksp, g0, v.legacy);
      rp.Gradient(U1, v.ksp, h1, v.legacy);
      rp.Gradient(U2, v.ksp, h2, v.legacy);
      PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(h1, -1.0, g0));
      PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(h2, -1.0, g0));
      const double a = Dot(h1, U2), b = Dot(U1, h2);
      std::printf("EX21_VC_G5D_REDUCED_HESSIAN adjoint=%s <H u1,u2>=%.17g <u1,H u2>=%.17g rel=%.3e\n",
                  v.name, a, b, std::fabs(a - b) / std::max(std::fabs(a), std::fabs(b)));
    }
    for (Vec* w : {&U1, &U2, &g0, &h1, &h2, &z}) PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(w));
  }
  {
    Vec U0 = NewLike(U), D = NewLike(U), g = NewLike(U), up = NewLike(U), um = NewLike(U);
    FillRandom(U0, 3u); FillRandom(D, 33u);
    rp.MaskC(U0); rp.MaskC(D);
    const double step_scale = Norm(U0) / Norm(D);
    for (const Variant& v : variants) {
      rp.Gradient(U0, v.ksp, g, v.legacy);
      const double directional = Dot(g, D);
      double best = 1e300;
      for (double eps : {1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6}) {
        const double h = eps * step_scale;
        PetscCallAbort(PETSC_COMM_WORLD, VecWAXPY(up, h, D, U0));
        PetscCallAbort(PETSC_COMM_WORLD, VecWAXPY(um, -h, D, U0));
        const double fd = (rp.Objective(up) - rp.Objective(um)) / (2.0 * h);
        best = std::min(best, std::fabs(fd - directional) / std::fabs(directional));
      }
      std::printf("EX21_VC_G4D_FD adjoint=%s best_rel=%.3e directional=%.17g\n", v.name, best, directional);
    }
    for (Vec* w : {&U0, &D, &g, &up, &um}) PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(w));
  }
  // production solve
  {
    const KktSolverPolicy policy = MakeKktSolverPolicy(options.pc_mode, options.spatial_tolerance);
    DistributedOptimalControlSolve solve = SolveOptimalControl(config, spatial, spacetime, policy);
    Vec g = NewLike(U), Yref = NewLike(U), RU = NewLike(U), Uc = NewLike(U);
    PetscCallAbort(PETSC_COMM_WORLD, VecCopy(solve.fields.control, Uc));
    rp.MaskC(Uc);
    rp.Gradient(Uc, adj_exact, g, false);
    PetscCallAbort(PETSC_COMM_WORLD, d.ApplyR(Uc, RU));
    PetscCallAbort(PETSC_COMM_WORLD, VecScale(RU, config.alpha_u * config.time.dt));
    rp.State(Uc, Yref);
    // the production state carries y_C = eta u on C
    {
      PetscScalar* y = nullptr;
      const PetscScalar* u = nullptr;
      PetscCallAbort(PETSC_COMM_WORLD, VecGetArray(Yref, &y));
      PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(Uc, &u));
      const std::vector<unsigned char>& m = d.control_mask();
      for (unsigned n = 0; n < steps; ++n)
        for (PetscInt i = 0; i < local; ++i)
          if (m[static_cast<std::size_t>(i)]) y[static_cast<PetscInt>(n) * local + i] = config.eta * u[static_cast<PetscInt>(n) * local + i];
      PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(Uc, &u));
      PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArray(Yref, &y));
    }
    std::printf("EX21_VC_G6D_PRODUCTION pc=%d reason=%d outer_its=%d true_residual=%.3e accepted=%d"
                " |grad J(u*)|/|dt alpha_u R u*|=%.3e |y_prod - y(u*)|/|y|=%.3e"
                " kkt_state_rel=%.3e kkt_adjoint_rel=%.3e kkt_stationarity_rel=%.3e"
                " J_track=%.10e J_reg=%.10e\n",
                static_cast<int>(options.pc_mode), solve.reason, solve.outer_iterations,
                solve.true_residual, solve.accepted ? 1 : 0, Norm(g) / Norm(RU),
                RelativeDifference(solve.fields.state, Yref), solve.kkt.state_rel,
                solve.kkt.adjoint_rel, solve.kkt.stationarity_rel, solve.fields.J_track,
                solve.fields.J_reg);
    for (Vec* w : {&g, &Yref, &RU, &Uc}) PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(w));
  }
  for (KSP* k : {&rp.forward, &adj_exact, &adj_shell, &adj_wrong}) PetscCallAbort(PETSC_COMM_WORLD, KSPDestroy(k));
  DestroyAssembledSpaceTimeOperators(wrong_ops);
  PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&stiffness_t));
  for (Vec* w : {&U, &P, &BU, &BtP, &BtPl, &rp.q0, &trace_spatial})
    PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(w));
}

// ===========================================================================
// --verify-convection
// ===========================================================================

int RunConvectionVerification(const ProblemConfig& config,
                              const ApplicationOptions& options,
                              MultiLevelSolution& solution, Mesh* mesh,
                              unsigned solution_type) {
  RequireSerial("--verify-convection");
  const bool dirichlet = config.state_bc ==
      ProblemConfig::StateBoundaryCondition::HOMOGENEOUS_DIRICHLET;
  std::printf("EX21_VC_CONFIG dim=%u nx=%u nz=%u steps=%u dt=%.6g scheme=%s"
              " nu=%.6g convection=%s vx=%.6g vy=%.6g state_bc=%s alpha_u=%.6g"
              " alpha_track=%.6g%s%s%s\n",
              config.dimension, config.mesh.nx, config.mesh.nz,
              config.time.steps, config.time.dt, TimeSchemeName(config),
              config.diffusivity, HasConvection(config) ? "on" : "off",
              config.convection.vx, config.convection.vy,
              dirichlet ? "dirichlet" : "natural", config.alpha_u,
              config.alpha_track,
              config.convection.field ? " wind=" : "",
              config.convection.field ? config.convection.field_name : "",
              config.data_overrides.dirichlet_state ? " dirichlet_data=lifted" : "");
  if (config.dimension == 3u)   // T9: the 3D extents and wind component
    std::printf("EX21_VC_CONFIG3D nx=%u ny=%u nz=%u vx=%.6g vy=%.6g vz=%.6g\n",
                config.mesh.nx, config.mesh.ny, config.mesh.nz, config.convection.vx,
                config.convection.vy, config.convection.vz);

  // ---------------------------------------------------------------- G1
  // C = L - K with the convection term switched on and off on the SAME mesh.
  // Orientation: for a LINEAR field y, (v . grad y) is the constant v . a, so
  // the strong-form row  (C y)_i = int (v . grad y) phi_i = (v . a)(M 1)_i
  // holds EXACTLY on every row (natural BC: no row is eliminated).  The
  // transposed orientation gives (C^T y)_i, which does not.
  {
    ProblemConfig with = config, without = config;
    with.state_bc = without.state_bc =
        ProblemConfig::StateBoundaryCondition::NATURAL;
    with.convection.enabled = true;
    without.convection.enabled = false;
    without.convection.vx = without.convection.vy = without.convection.vz = 0.0;
    // G1 tests the Galerkin convection term alone (T6: LPS is gated in G5).
    with.convection.stabilization = without.convection.stabilization =
        ProblemConfig::ConvectionConfig::Stabilization::NONE;
    // T7D: the orientation is a property of the UNCONSTRAINED L; Dirichlet
    // control would eliminate the C rows and columns, so G1 assembles it as a
    // whole-domain distributed problem instead (no other run mode changes).
    if (config.control.type == ProblemConfig::ControlType::DIRICHLET) {
      with.control.type = without.control.type = ProblemConfig::ControlType::DISTRIBUTED;
      with.control.region = without.control.region = {config.domain.xmin, config.domain.xmax,
                                                       config.domain.zmin, config.domain.zmax};
    }
    SpatialOperators L(with, mesh, solution_type, solution);
    SpatialOperators K(without, mesh, solution_type, solution);
    Mat C = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatDuplicate(L.stiffness, MAT_COPY_VALUES, &C));
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatAXPY(C, -1.0, K.stiffness, SAME_NONZERO_PATTERN));
    PetscReal c_norm = 0.0, k_norm = 0.0;
    PetscCallAbort(PETSC_COMM_WORLD, MatNorm(C, NORM_FROBENIUS, &c_norm));
    PetscCallAbort(PETSC_COMM_WORLD, MatNorm(K.stiffness, NORM_FROBENIUS, &k_norm));
    Vec one = nullptr, m_one = nullptr, field = nullptr, c_field = nullptr,
        expected = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD, MatCreateVecs(C, &one, &m_one));
    for (Vec* v : {&field, &c_field, &expected})
      PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(one, v));
    PetscCallAbort(PETSC_COMM_WORLD, VecSet(one, 1.0));
    PetscCallAbort(PETSC_COMM_WORLD, MatMult(L.mass, one, m_one));
    for (unsigned axis = 0; axis < config.dimension; ++axis) {
      // T9: storage axis 2 (3D only) carries vz
      const double v_axis = axis == 0u ? config.convection.vx
                            : axis == 1u ? config.convection.vy : config.convection.vz;
      PetscScalar* f = nullptr;
      PetscCallAbort(PETSC_COMM_WORLD, VecGetArray(field, &f));
      for (PetscInt i = 0; i < L.layout.local_size; ++i)
        f[i] = axis == 0u ? L.layout.x[i] : axis == 1u ? L.layout.z[i] : L.layout.axis2[i];
      PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArray(field, &f));
      if (config.convection.field) {
        // Variable field: (C lin)_i = int w_axis phi_i = (M I_h w_axis)_i,
        // exact when w_axis is in Q2 (both named fields are).
        PetscCallAbort(PETSC_COMM_WORLD, VecGetArray(c_field, &f));
        for (PetscInt i = 0; i < L.layout.local_size; ++i) {
          double w[2];
          EvaluateVelocity(config, L.layout.x[i], L.layout.z[i], w);
          f[i] = w[axis];
        }
        PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArray(c_field, &f));
        PetscCallAbort(PETSC_COMM_WORLD, MatMult(L.mass, c_field, expected));
      } else {
        PetscCallAbort(PETSC_COMM_WORLD, VecCopy(m_one, expected));
        PetscCallAbort(PETSC_COMM_WORLD, VecScale(expected, v_axis));
      }
      const double scale = std::max(Norm(expected, NORM_INFINITY),
                                    Norm(m_one, NORM_INFINITY));
      PetscCallAbort(PETSC_COMM_WORLD, MatMult(C, field, c_field));
      PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(c_field, -1.0, expected));
      const double strong = Norm(c_field, NORM_INFINITY) / scale;
      PetscCallAbort(PETSC_COMM_WORLD, MatMultTranspose(C, field, c_field));
      PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(c_field, -1.0, expected));
      const double transposed = Norm(c_field, NORM_INFINITY) / scale;
      std::printf("EX21_VC_G1_ORIENTATION axis=%s v=%.6g"
                  " |C*lin - v*M1|/|M1|=%.3e |C^T*lin - v*M1|/|M1|=%.3e\n",
                  axis == 0u ? "x" : (config.dimension == 3u ? (axis == 1u ? "y" : "z") : "z"),
                  v_axis, strong, transposed);
    }
    std::printf("EX21_VC_G1_NORMS ||C||_F=%.6e ||K||_F=%.6e\n", c_norm, k_norm);
    for (Vec* v : {&one, &m_one, &field, &c_field, &expected})
      PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(v));
    PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&C));
  }
  // Skew symmetry on the free DOFs: with div v = 0,
  //   C_ij + C_ji = int v . grad(phi_i phi_j) = int_{dOmega} (v . n) phi_i phi_j,
  // which vanishes whenever i or j is interior, i.e. on every entry that
  // survives the homogeneous Dirichlet elimination.
  if (dirichlet) {
    ProblemConfig with = config, without = config;
    with.convection.enabled = true;
    without.convection.enabled = false;
    without.convection.vx = without.convection.vy = 0.0;
    // G1 tests the Galerkin convection term alone (T6: LPS is gated in G5).
    with.convection.stabilization = without.convection.stabilization =
        ProblemConfig::ConvectionConfig::Stabilization::NONE;
    // T7D: the orientation is a property of the UNCONSTRAINED L; Dirichlet
    // control would eliminate the C rows and columns, so G1 assembles it as a
    // whole-domain distributed problem instead (no other run mode changes).
    if (config.control.type == ProblemConfig::ControlType::DIRICHLET) {
      with.control.type = without.control.type = ProblemConfig::ControlType::DISTRIBUTED;
      with.control.region = without.control.region = {config.domain.xmin, config.domain.xmax,
                                                       config.domain.zmin, config.domain.zmax};
    }
    SpatialOperators L(with, mesh, solution_type, solution);
    SpatialOperators K(without, mesh, solution_type, solution);
    Mat C = nullptr, Ct = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatDuplicate(L.stiffness, MAT_COPY_VALUES, &C));
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatAXPY(C, -1.0, K.stiffness, SAME_NONZERO_PATTERN));
    PetscCallAbort(PETSC_COMM_WORLD, MatTranspose(C, MAT_INITIAL_MATRIX, &Ct));
    PetscReal c_norm = 0.0, skew = 0.0, asym = 0.0;
    PetscCallAbort(PETSC_COMM_WORLD, MatNorm(C, NORM_FROBENIUS, &c_norm));
    PetscCallAbort(PETSC_COMM_WORLD, MatAXPY(Ct, 1.0, C, SAME_NONZERO_PATTERN));
    PetscCallAbort(PETSC_COMM_WORLD, MatNorm(Ct, NORM_FROBENIUS, &skew));
    // ||L - L^T||: nonzero exactly when convection is.
    Mat Lt = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatTranspose(L.stiffness, MAT_INITIAL_MATRIX, &Lt));
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatAXPY(Lt, -1.0, L.stiffness, SAME_NONZERO_PATTERN));
    PetscCallAbort(PETSC_COMM_WORLD, MatNorm(Lt, NORM_FROBENIUS, &asym));
    std::printf("EX21_VC_G1_SKEW_DIRICHLET ||C+C^T||_F/||C||_F=%.3e"
                " ||L-L^T||_F=%.6e ||C||_F=%.6e\n",
                c_norm > 0 ? skew / c_norm : skew, asym, c_norm);
    for (Mat* m : {&C, &Ct, &Lt}) PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(m));
  }

  // ---------------------------------------------------------------- G5
  // T6 LPS matrix gates (only with --convection-stabilization lps), on W as
  // assembled (no boundary elimination):
  //   symmetry  ||W - W^T||_F / ||W||_F
  //   PSD       min over 20 random x of x^T W x / (||W||_F ||x||^2)
  //   reversal  ||W_w - W_{-w}||_F / ||W_w||_F  (LP: W_{-w} = W_w), same active set
  //   constants ||W 1||_inf / ||W||_F  (w . grad 1 = 0: exactly 0)
  //   Pe switch with nu raised until max Pe_k <= 1: active = 0 and ||W||_F = 0
  if (config.convection.stabilization ==
      ProblemConfig::ConvectionConfig::Stabilization::LPS) {
    LpsStats sp, sm, s0;
    std::unique_ptr<SparseMatrix> Wp = BuildLpsMatrix(config, mesh, solution_type, 1.0, &sp);
    std::unique_ptr<SparseMatrix> Wm = BuildLpsMatrix(config, mesh, solution_type, -1.0, &sm);
    Mat wp = static_cast<PetscMatrix*>(Wp.get())->mat();
    Mat wm = static_cast<PetscMatrix*>(Wm.get())->mat();
    PetscReal norm = 0.0, asym = 0.0, rev = 0.0;
    PetscCallAbort(PETSC_COMM_WORLD, MatNorm(wp, NORM_FROBENIUS, &norm));
    Mat wt = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD, MatTranspose(wp, MAT_INITIAL_MATRIX, &wt));
    PetscCallAbort(PETSC_COMM_WORLD, MatAXPY(wt, -1.0, wp, DIFFERENT_NONZERO_PATTERN));
    PetscCallAbort(PETSC_COMM_WORLD, MatNorm(wt, NORM_FROBENIUS, &asym));
    PetscCallAbort(PETSC_COMM_WORLD, MatAXPY(wm, -1.0, wp, DIFFERENT_NONZERO_PATTERN));
    PetscCallAbort(PETSC_COMM_WORLD, MatNorm(wm, NORM_FROBENIUS, &rev));
    Vec x = nullptr, wx = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD, MatCreateVecs(wp, &x, &wx));
    double psd_min = 1.0e300;
    for (unsigned trial = 0; trial < 20u; ++trial) {
      FillRandom(x, 500u + trial);
      PetscCallAbort(PETSC_COMM_WORLD, MatMult(wp, x, wx));
      const double nx = Norm(x);
      psd_min = std::min(psd_min, Dot(x, wx) / (std::max(norm, 1e-300) * nx * nx));
    }
    PetscCallAbort(PETSC_COMM_WORLD, VecSet(x, 1.0));
    PetscCallAbort(PETSC_COMM_WORLD, MatMult(wp, x, wx));
    const double constants = Norm(wx, NORM_INFINITY) / std::max(norm, 1e-300);
    ProblemConfig low = config;
    low.diffusivity = config.diffusivity * std::max(1.0, sp.pe_max) * (1.0 + 1e-9);
    std::unique_ptr<SparseMatrix> W0 = BuildLpsMatrix(low, mesh, solution_type, 1.0, &s0);
    PetscReal norm0 = 0.0;
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatNorm(static_cast<PetscMatrix*>(W0.get())->mat(), NORM_FROBENIUS, &norm0));
    std::printf("EX21_VC_G5_LPS projection=%s patches=%lld active=%lld"
                " delta_min=%.6e delta_max=%.6e pe_max=%.6e ||W||_F=%.6e"
                " sym_rel=%.3e psd_min_rel=%.3e reversal_rel=%.3e"
                " reversal_active=%lld W1_rel=%.3e\n",
                config.convection.lps_projection ==
                        ProblemConfig::ConvectionConfig::LpsProjection::P1 ? "p1" : "p0",
                sp.patches, sp.active, sp.delta_min, sp.delta_max, sp.pe_max, norm,
                norm > 0 ? asym / norm : asym, psd_min, norm > 0 ? rev / norm : rev,
                sm.active, constants);
    std::printf("EX21_VC_G5_LPS_PE_SWITCH nu_low=%.6e pe_max_low=%.6e active=%lld"
                " ||W||_F=%.3e\n", low.diffusivity, s0.pe_max, s0.active, norm0);
    PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&wt));
    PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&x));
    PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&wx));
  }

  // ---------------------------------------------------------------- G2
  SpatialOperators spatial(config, mesh, solution_type, solution);
  SpaceTimeOperators spacetime(config, spatial);
  Mat AY = spacetime.distributed.ay, AP = spacetime.distributed.ap;
  Mat ay = spacetime.assembled.ay, ap = spacetime.assembled.ap;
  Vec X = nullptr, Y = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, MatCreateVecs(ay, &X, &Y));
  FillRandom(X, 21u);
  FillRandom(Y, 2021u);
  Vec EX = NewLike(X), ETY = NewLike(X), w1 = NewLike(X), w2 = NewLike(X);
  PetscCallAbort(PETSC_COMM_WORLD, MatMult(AY, X, EX));
  PetscCallAbort(PETSC_COMM_WORLD, MatMult(AP, Y, ETY));
  const double lhs = Dot(EX, Y), rhs = Dot(X, ETY);
  const double scale = Norm(EX) * Norm(Y);
  std::printf("EX21_VC_G2_SHELL_TRANSPOSE <E x,y>=%.17g <x,E^T y>=%.17g"
              " |diff|/(|Ex||y|)=%.3e\n", lhs, rhs, std::fabs(lhs - rhs) / scale);
  PetscCallAbort(PETSC_COMM_WORLD, MatMult(ay, X, w1));
  const double ay_vs_shell = RelativeDifference(w1, EX);
  PetscCallAbort(PETSC_COMM_WORLD, MatMult(ap, Y, w1));
  const double ap_vs_shell = RelativeDifference(w1, ETY);
  PetscCallAbort(PETSC_COMM_WORLD, MatMultTranspose(ay, Y, w2));
  const double ayT_vs_shell = RelativeDifference(w2, ETY);
  std::printf("EX21_VC_G2_ASSEMBLED |E_aij x - E x|/|E x|=%.3e"
              " |ET_aij y - ET y|/|ET y|=%.3e |E_aij^T y - ET y|/|ET y|=%.3e\n",
              ay_vs_shell, ap_vs_shell, ayT_vs_shell);
  // Negative control: the ex20 adjoint product (L NOT transposed).
  spacetime.distributed.ap_context.transpose_spatial = false;
  PetscCallAbort(PETSC_COMM_WORLD, MatMult(AP, Y, w1));
  spacetime.distributed.ap_context.transpose_spatial = HasConvection(config);
  const double wrong = Dot(X, w1);
  std::printf("EX21_VC_G2_NEGATIVE_CONTROL ex20-style adjoint (L untransposed):"
              " |<E x,y>-<x,E20 y>|/(|Ex||y|)=%.3e\n",
              std::fabs(lhs - wrong) / scale);

  // T7D: Dirichlet boundary control has its own reduced problem (B_D, R_H1t).
  if (config.control.type == ProblemConfig::ControlType::DIRICHLET) {
    RunDirichletConvectionGates(config, options, spatial, spacetime);
    for (Vec* w : {&X, &Y, &EX, &ETY, &w1, &w2})
      PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(w));
    return 0;
  }

  // Reduced problem pieces.
  DistributedDataSideResult data = BuildDistributedRightHandSide(
      config, spatial.layout, spatial.mass, spatial.tracking, ay,
      spatial.stiffness, spatial.source, &spatial.lift);
  ReducedProblem reduced;
  reduced.config = &config;
  reduced.control = spatial.control;
  reduced.tracking = spatial.tracking;
  reduced.rhs_y = data.rhs_y;
  reduced.rhs_p = data.rhs_p;
  reduced.local = spatial.layout.local_size;
  reduced.ones.assign(config.time.steps, 1.0);
  reduced.weights.resize(config.time.steps);
  for (unsigned n = 0; n < config.time.steps; ++n)
    reduced.weights[n] = TrackingSlotWeight(config, n);
  reduced.forward = MakeDirect(ay, "vc_forward_");
  KSP adjoint_exact = MakeDirect(ap, "vc_adjoint_");
  KSP adjoint_shell = MakeShellSolver(AP, ap, "vc_adjoint_shell_");
  // The ex20-style adjoint, assembled: (E[L^T])^T has diagonal M/dt + L
  // instead of M/dt + L^T.
  Mat stiffness_t = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatTranspose(spatial.stiffness, MAT_INITIAL_MATRIX, &stiffness_t));
  AssembledSpaceTimeOperators wrong_ops;
  BuildAssembledSpaceTimeOperators(wrong_ops, spatial.mass, stiffness_t,
                                   config.time.steps, config.time.dt,
                                   config.time.scheme);
  KSP adjoint_wrong = MakeDirect(wrong_ops.ap, "vc_adjoint_wrong_");

  // ---------------------------------------------------------------- G1N / G8b
  // T7N (natural state BC only, constant wind, no LPS: the gate lines appear
  // only then, so every Dirichlet-state gate log is unchanged).
  //   G1N  the boundary identity of the strong-form Galerkin convection,
  //          C + C^T = sum_faces (w.n_f) M_f ,   M_f the trace mass of face f,
  //        i.e. the exact transpose carries the boundary term that turns the
  //        natural state condition into nu dn p + (w.n) p = 0 for the adjoint.
  //   G8b  a boundary-WRONG adjoint: the spatial adjoint operator nu K - C
  //        (the correct interior operator with the symmetric-diffusion
  //        condition nu dn p = 0), differing from L^T exactly by C + C^T.
  //        It must fail the transpose, Hessian and FD gates whenever w.n != 0.
  const bool natural_gate =
      config.state_bc == ProblemConfig::StateBoundaryCondition::NATURAL &&
      HasConvection(config) && !config.convection.field &&
      config.convection.stabilization == ProblemConfig::ConvectionConfig::Stabilization::NONE &&
      config.dimension == 2u;
  AssembledSpaceTimeOperators wrong_bc_ops;
  KSP adjoint_wrong_bc = nullptr;
  Mat wrong_bc_spatial = nullptr;
  if (natural_gate) {
    ProblemConfig without = config;
    without.convection.enabled = false;
    without.convection.vx = without.convection.vy = 0.0;
    SpatialOperators K(without, mesh, solution_type, solution);
    Mat C = nullptr, Ct = nullptr, Bw = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD, MatDuplicate(spatial.stiffness, MAT_COPY_VALUES, &C));
    PetscCallAbort(PETSC_COMM_WORLD, MatAXPY(C, -1.0, K.stiffness, DIFFERENT_NONZERO_PATTERN));
    PetscCallAbort(PETSC_COMM_WORLD, MatTranspose(C, MAT_INITIAL_MATRIX, &Ct));
    PetscCallAbort(PETSC_COMM_WORLD, MatAXPY(Ct, 1.0, C, DIFFERENT_NONZERO_PATTERN));  // C + C^T
    PetscReal c_norm = 0.0, sym_norm = 0.0, defect = 0.0;
    PetscCallAbort(PETSC_COMM_WORLD, MatNorm(C, NORM_FROBENIUS, &c_norm));
    PetscCallAbort(PETSC_COMM_WORLD, MatNorm(Ct, NORM_FROBENIUS, &sym_norm));
    PetscCallAbort(PETSC_COMM_WORLD, MatDuplicate(Ct, MAT_COPY_VALUES, &Bw));
    PetscCallAbort(PETSC_COMM_WORLD, MatZeroEntries(Bw));
    std::string faces;
    for (unsigned face = 1; face <= 4u; ++face) {
      ProblemConfig fc = config;
      fc.control.type = ProblemConfig::ControlType::NEUMANN;
      fc.neumann_control_face = face;
      SpatialOperators F(fc, mesh, solution_type, solution);
      Mat Mf = static_cast<PetscMatrix*>(F.assembled.M_Gamma.get())->mat();
      // the face's side from its support (the rows with a nonzero diagonal)
      Vec d = nullptr;
      PetscCallAbort(PETSC_COMM_WORLD, MatCreateVecs(Mf, &d, nullptr));
      PetscCallAbort(PETSC_COMM_WORLD, MatGetDiagonal(Mf, d));
      const PetscScalar* dv = nullptr;
      PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(d, &dv));
      double xs = 0.0, zs = 0.0, n = 0.0;
      for (PetscInt i = 0; i < F.layout.local_size; ++i)
        if (PetscRealPart(dv[i]) != 0.0) { xs += F.layout.x[i]; zs += F.layout.z[i]; n += 1.0; }
      PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(d, &dv));
      PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&d));
      double sums[3] = {xs, zs, n}, g[3] = {0, 0, 0};
      MPI_Allreduce(sums, g, 3, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
      const double xc = g[0] / g[2], zc = g[1] / g[2];
      const double tol = 1e-9;
      double nx = 0.0, nz = 0.0;
      if (std::fabs(xc - config.domain.xmin) < tol) nx = -1.0;
      else if (std::fabs(xc - config.domain.xmax) < tol) nx = 1.0;
      else if (std::fabs(zc - config.domain.zmin) < tol) nz = -1.0;
      else if (std::fabs(zc - config.domain.zmax) < tol) nz = 1.0;
      const double wn = config.convection.vx * nx + config.convection.vy * nz;
      char buf[96];
      std::snprintf(buf, sizeof buf, "%s%u:n=(%+.0f,%+.0f),w.n=%+.4g", faces.empty() ? "" : ";",
                    face, nx, nz, wn);
      faces += buf;
      if (wn != 0.0)
        PetscCallAbort(PETSC_COMM_WORLD, MatAXPY(Bw, wn, Mf, DIFFERENT_NONZERO_PATTERN));
    }
    PetscReal bw_norm = 0.0;
    PetscCallAbort(PETSC_COMM_WORLD, MatNorm(Bw, NORM_FROBENIUS, &bw_norm));
    PetscCallAbort(PETSC_COMM_WORLD, MatAXPY(Bw, -1.0, Ct, DIFFERENT_NONZERO_PATTERN));
    PetscCallAbort(PETSC_COMM_WORLD, MatNorm(Bw, NORM_FROBENIUS, &defect));
    std::printf("EX21_VC_G1N_BOUNDARY_IDENTITY ||C+C^T - sum_f (w.n_f) M_f||_F/||C+C^T||_F=%.3e"
                " ||C+C^T||_F=%.6e ||sum (w.n) M_f||_F=%.6e ||C||_F=%.6e faces=%s\n",
                sym_norm > 0 ? defect / sym_norm : defect, sym_norm, bw_norm, c_norm,
                faces.c_str());
    // G8b operator: S with S^T = nu K - C, i.e. S = 2 nu K - L^T (K symmetric)
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatTranspose(spatial.stiffness, MAT_INITIAL_MATRIX, &wrong_bc_spatial));
    PetscCallAbort(PETSC_COMM_WORLD, MatScale(wrong_bc_spatial, -1.0));
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatAXPY(wrong_bc_spatial, 2.0, K.stiffness, DIFFERENT_NONZERO_PATTERN));
    BuildAssembledSpaceTimeOperators(wrong_bc_ops, spatial.mass, wrong_bc_spatial,
                                     config.time.steps, config.time.dt,
                                     config.time.scheme);
    adjoint_wrong_bc = MakeDirect(wrong_bc_ops.ap, "vc_adjoint_wrong_bc_");
    // its space-time transpose defect against the forward E
    Vec w3 = NewLike(X);
    PetscCallAbort(PETSC_COMM_WORLD, MatMult(wrong_bc_ops.ap, Y, w3));
    const double lhs3 = Dot(EX, Y), rhs3 = Dot(X, w3);
    std::printf("EX21_VC_G8B_WRONG_BC_TRANSPOSE adjoint=nu_K_minus_C (nu dn p = 0)"
                " |<E x,y>-<x,E_wrongbc^T y>|/(|Ex||y|)=%.3e\n",
                std::fabs(lhs3 - rhs3) / scale);
    PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&w3));
    for (Mat* m : {&C, &Ct, &Bw}) PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(m));
  }

  // Reduced Hessian symmetry: H U = grad J(U) - grad J(0) is linear in U.
  {
    Vec U1 = NewLike(X), U2 = NewLike(X), g0 = NewLike(X), h1 = NewLike(X),
        h2 = NewLike(X), zero = NewLike(X);
    FillRandom(U1, 7u);
    FillRandom(U2, 77u);
    struct Variant { const char* name; KSP ksp; };
    std::vector<Variant> variants{Variant{"exact_LU(E^T)", adjoint_exact},
                                  Variant{"production_shell(E^T)", adjoint_shell},
                                  Variant{"ex20_style_wrong", adjoint_wrong}};
    if (adjoint_wrong_bc)   // T7N G8b
      variants.push_back(Variant{"wrong_bc_nuK_minus_C", adjoint_wrong_bc});
    for (const Variant& v : variants) {
      reduced.Gradient(zero, v.ksp, g0);
      reduced.Gradient(U1, v.ksp, h1);
      reduced.Gradient(U2, v.ksp, h2);
      PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(h1, -1.0, g0));
      PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(h2, -1.0, g0));
      const double a = Dot(h1, U2), b = Dot(U1, h2);
      std::printf("EX21_VC_G2_REDUCED_HESSIAN adjoint=%s <H u1,u2>=%.17g"
                  " <u1,H u2>=%.17g rel=%.3e\n", v.name, a, b,
                  std::fabs(a - b) / std::max(std::fabs(a), std::fabs(b)));
    }
    for (Vec* w : {&U1, &U2, &g0, &h1, &h2, &zero})
      PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(w));
  }

  // ---------------------------------------------------------------- G3
  {
    Vec U0 = NewLike(X), D = NewLike(X), g = NewLike(X), up = NewLike(X),
        um = NewLike(X);
    FillRandom(U0, 3u);
    FillRandom(D, 33u);
    const double step_scale = Norm(U0) / Norm(D);
    struct Variant { const char* name; KSP ksp; };
    std::vector<Variant> variants{Variant{"exact_LU(E^T)", adjoint_exact},
                                  Variant{"production_shell(E^T)", adjoint_shell},
                                  Variant{"ex20_style_wrong", adjoint_wrong}};
    if (adjoint_wrong_bc)   // T7N G8b
      variants.push_back(Variant{"wrong_bc_nuK_minus_C", adjoint_wrong_bc});
    for (const Variant& v : variants) {
      const int its = reduced.Gradient(U0, v.ksp, g);
      const double directional = Dot(g, D);
      double best = 1.0e300;
      for (double eps : {1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7}) {
        const double h = eps * step_scale;
        PetscCallAbort(PETSC_COMM_WORLD, VecWAXPY(up, h, D, U0));
        PetscCallAbort(PETSC_COMM_WORLD, VecWAXPY(um, -h, D, U0));
        const double fd = (reduced.Objective(up) - reduced.Objective(um)) /
                          (2.0 * h);
        const double rel = std::fabs(fd - directional) / std::fabs(directional);
        best = std::min(best, rel);
        std::printf("EX21_VC_G3_FD adjoint=%s eps=%.0e fd=%.17g adj=%.17g"
                    " rel=%.3e\n", v.name, eps, fd, directional, rel);
      }
      std::printf("EX21_VC_G3_SUMMARY adjoint=%s adjoint_its=%d best_rel=%.3e\n",
                  v.name, its, best);
    }
    for (Vec* w : {&U0, &D, &g, &up, &um})
      PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(w));
  }

  // ---------------------------------------------------------------- G3P
  // T3b: with inhomogeneous Dirichlet data, the adjoint gradient against the
  // PHYSICAL full-mass objective.  Passes iff rhs_p carries the affine
  // tracking term M_F,Gamma (g - y_d)_Gamma (fails under the diagnostic
  // --lift-no-affine-tracking, the pre-T3b behaviour).
  if (spatial.lift.g && spatial.lift.tracking_full) {
    reduced.tracking_full = spatial.lift.tracking_full;
    reduced.lift = spatial.lift.g;
    reduced.target = NewLike(X);
    {
      PetscScalar* a = nullptr;
      PetscCallAbort(PETSC_COMM_WORLD, VecGetArray(reduced.target, &a));
      for (unsigned slot = 0; slot < config.time.steps; ++slot)
        for (PetscInt i = 0; i < reduced.local; ++i)
          a[static_cast<PetscInt>(slot) * reduced.local + i] = EvaluateTrackingTarget(
              config, spatial.layout.x[i], spatial.layout.z[i], (slot + 1u) * config.time.dt);
      PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArray(reduced.target, &a));
    }
    Vec U0 = NewLike(X), D = NewLike(X), g = NewLike(X), up = NewLike(X),
        um = NewLike(X);
    FillRandom(U0, 3u);
    FillRandom(D, 33u);
    const double step_scale = Norm(U0) / Norm(D);
    reduced.Gradient(U0, adjoint_exact, g);
    const double directional = Dot(g, D);
    double best = 1.0e300;
    for (double eps : {1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6}) {
      const double h = eps * step_scale;
      PetscCallAbort(PETSC_COMM_WORLD, VecWAXPY(up, h, D, U0));
      PetscCallAbort(PETSC_COMM_WORLD, VecWAXPY(um, -h, D, U0));
      const double fd = (reduced.ObjectivePhysical(up) -
                         reduced.ObjectivePhysical(um)) / (2.0 * h);
      const double rel = std::fabs(fd - directional) / std::fabs(directional);
      best = std::min(best, rel);
      std::printf("EX21_VC_G3P_FD eps=%.0e fd=%.17g adj=%.17g rel=%.3e\n", eps,
                  fd, directional, rel);
    }
    std::printf("EX21_VC_G3P_SUMMARY affine_tracking=%d best_rel=%.3e\n",
                spatial.lift.affine_tracking ? 1 : 0, best);
    for (Vec* w : {&U0, &D, &g, &up, &um, &reduced.target})
      PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(w));
  }

  // ---------------------------------------------------------------- G4
  {
    const KktSolverPolicy policy =
        MakeKktSolverPolicy(options.pc_mode, options.spatial_tolerance);
    DistributedOptimalControlSolve solve =
        SolveOptimalControl(config, spatial, spacetime, policy);
    Vec g = NewLike(X), Yref = NewLike(X), NU = NewLike(X);
    reduced.Gradient(solve.fields.control, adjoint_exact, g);
    ApplyBlockDiagonal(spatial.control, solve.fields.control, NU,
                       config.time.steps, reduced.local, reduced.ones);
    PetscCallAbort(PETSC_COMM_WORLD, VecScale(NU, config.time.dt * config.alpha_u));
    reduced.State(solve.fields.control, Yref);
    // inhomogeneous Dirichlet data: the production state is y_0 + G
    if (spatial.lift.g)
      for (unsigned slot = 0; slot < config.time.steps; ++slot)
        CopySpatialToDataSideSlice(spatial.lift.g, slot, reduced.local, 1.0,
                                   Yref, true);
    std::printf("EX21_VC_G4_PRODUCTION reason=%d outer_its=%d true_residual=%.3e"
                " |grad J(u*)|/|dt alpha_u N u*|=%.3e"
                " |y_prod - E^-1(rhs+eta N u*)|/|y|=%.3e J_track=%.10e J_reg=%.10e\n",
                solve.reason, solve.outer_iterations, solve.true_residual,
                Norm(g) / Norm(NU), RelativeDifference(solve.fields.state, Yref),
                solve.fields.J_track, solve.fields.J_reg);
    for (Vec* w : {&g, &Yref, &NU}) PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(w));
  }

  if (adjoint_wrong_bc) {
    PetscCallAbort(PETSC_COMM_WORLD, KSPDestroy(&adjoint_wrong_bc));
    DestroyAssembledSpaceTimeOperators(wrong_bc_ops);
    PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&wrong_bc_spatial));
  }
  for (KSP* k : {&reduced.forward, &adjoint_exact, &adjoint_shell, &adjoint_wrong})
    PetscCallAbort(PETSC_COMM_WORLD, KSPDestroy(k));
  DestroyAssembledSpaceTimeOperators(wrong_ops);
  PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&stiffness_t));
  for (Vec* w : {&X, &Y, &EX, &ETY, &w1, &w2})
    PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(w));
  return 0;
}

// ===========================================================================
// --convection-mms
//
//   y(x, z, t) = cos(2t) S(x, z),  S = s(xi) s(zeta)  (2D),  S = s(xi)  (1D)
//   xi = (x - xmin)/Lx, zeta = (z - zmin)/Lz,  f = y_t - nu Lap y + v . grad y,
//   shape "sin":  s = sin(pi .)    y = 0 on the boundary, but f != 0 there as
//                                  soon as v != 0 (the v . grad y term);
//   shape "sin3": s = sin^3(pi .)  y, grad y, Lap y and hence f all vanish
//                                  on the boundary, for every v.
// The two shapes separate the convection operator from the data-assembly
// convention that drops the boundary nodal values of f (see the report).
// ===========================================================================

namespace {

struct MmsField {
  double s = 0.0, sx = 0.0, sz = 0.0, lap = 0.0;
};

static bool g_mms_cubic = false;   // set by RunConvectionManufactured

/** One factor s(a) of the separable shape and its first two derivatives in a. */
void Factor(double a, double& s, double& ds, double& d2s) {
  const double sn = std::sin(a), cs = std::cos(a);
  if (!g_mms_cubic) {
    s = sn; ds = cs; d2s = -sn;
    return;
  }
  s = sn * sn * sn;
  ds = 3.0 * sn * sn * cs;
  d2s = 6.0 * sn * cs * cs - 3.0 * sn * sn * sn;
}

MmsField Shape(const ProblemConfig& c, double x, double z) {
  const double lx = c.domain.xmax - c.domain.xmin;
  const double kx = kPi / lx;
  double sx = 0, dsx = 0, d2sx = 0;
  Factor(kx * (x - c.domain.xmin), sx, dsx, d2sx);
  MmsField f;
  if (c.dimension == 1u) {
    f.s = sx;
    f.sx = kx * dsx;
    f.lap = kx * kx * d2sx;
    return f;
  }
  const double lz = c.domain.zmax - c.domain.zmin;
  const double kz = kPi / lz;
  double sz = 0, dsz = 0, d2sz = 0;
  Factor(kz * (z - c.domain.zmin), sz, dsz, d2sz);
  f.s = sx * sz;
  f.sx = kx * dsx * sz;
  f.sz = kz * sx * dsz;
  f.lap = kx * kx * d2sx * sz + kz * kz * sx * d2sz;
  return f;
}

double MmsExact(const ProblemConfig& c, double x, double z, double t) {
  return std::cos(2.0 * t) * Shape(c, x, z).s;
}

double MmsSource(const ProblemConfig& c, double x, double z, double t) {
  const MmsField f = Shape(c, x, z);
  const double g = std::cos(2.0 * t), dg = -2.0 * std::sin(2.0 * t);
  double w[2];
  EvaluateVelocity(c, x, z, w);
  const double advection = c.convection.enabled
      ? w[0] * f.sx + w[1] * f.sz : 0.0;
  return dg * f.s - c.diffusivity * g * f.lap + g * advection;
}

double MmsInitial(const ProblemConfig& c, double x, double z) {
  return MmsExact(c, x, z, 0.0);
}

struct ExactContext {
  const ProblemConfig* config;
  double time;
};

double ExactAt(double x, double z, void* raw) {
  const ExactContext* e = static_cast<const ExactContext*>(raw);
  return MmsExact(*e->config, x, z, e->time);
}

}  // namespace

int RunConvectionManufactured(const ProblemConfig& parsed,
                              const std::string& shape,
                              MultiLevelSolution& solution, Mesh* mesh,
                              unsigned solution_type) {
  RequireSerial("--convection-mms");
  g_mms_cubic = (shape == "sin3");
  if (parsed.state_bc !=
      ProblemConfig::StateBoundaryCondition::HOMOGENEOUS_DIRICHLET)
    throw std::invalid_argument(
        "--convection-mms needs --state-bc dirichlet (the exact solution"
        " vanishes on the boundary)");
  ProblemConfig config = parsed;
  config.data_overrides.source = MmsSource;
  config.data_overrides.initial_state = MmsInitial;

  SpatialOperators spatial(config, mesh, solution_type, solution);
  SpaceTimeOperators spacetime(config, spatial);
  DistributedDataSideResult data = BuildDistributedRightHandSide(
      config, spatial.layout, spatial.mass, spatial.tracking,
      spacetime.distributed.ay, spatial.stiffness, spatial.source);

  // March the block-bidiagonal E Y = rhs_y slot by slot with a spatial LU:
  //   BE:  (M/dt + L) y_n = rhs_n + (M/dt) y_{n-1}
  //   CN:  (M/dt + L/2) y_n = rhs_n + (M/dt - L/2) y_{n-1}
  // and then check the AAO residual ||E Y - rhs|| with the production shell.
  const bool cn = IsCrankNicolson(config);
  const double dt = config.time.dt;
  Mat diagonal = nullptr, coupling = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatDuplicate(spatial.stiffness, MAT_COPY_VALUES, &diagonal));
  PetscCallAbort(PETSC_COMM_WORLD, MatScale(diagonal, cn ? 0.5 : 1.0));
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatAXPY(diagonal, 1.0 / dt, spatial.mass, SUBSET_NONZERO_PATTERN));
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatDuplicate(spatial.mass, MAT_COPY_VALUES, &coupling));
  PetscCallAbort(PETSC_COMM_WORLD, MatScale(coupling, 1.0 / dt));
  if (cn)
    PetscCallAbort(PETSC_COMM_WORLD, MatAXPY(coupling, -0.5, spatial.stiffness,
                                             DIFFERENT_NONZERO_PATTERN));
  KSP spatial_lu = MakeDirect(diagonal, "mms_spatial_");
  Vec Y = NewLike(data.rhs_y);
  Vec r = nullptr, y = nullptr, prev = nullptr, tmp = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, MatCreateVecs(diagonal, &y, &r));
  PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(y, &prev));
  PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(y, &tmp));
  const PetscInt local = spatial.layout.local_size;
  double err_final = 0.0, rel_final = 0.0, err_max = 0.0;
  for (unsigned slot = 0; slot < config.time.steps; ++slot) {
    CopyDataSideSliceToSpatial(data.rhs_y, slot, local, r);
    if (slot > 0u) {
      PetscCallAbort(PETSC_COMM_WORLD, MatMult(coupling, prev, tmp));
      PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(r, 1.0, tmp));
    }
    Solve(spatial_lu, r, y);
    CopySpatialToDataSideSlice(y, slot, local, 1.0, Y, false);
    PetscCallAbort(PETSC_COMM_WORLD, VecCopy(y, prev));
    ExactContext ctx{&config, (slot + 1u) * dt};
    const PetscScalar* values = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(y, &values));
    const SpatialFieldIntegrals I = IntegrateSpatialField(
        mesh, solution_type, values, ExactAt, &ctx);
    PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(y, &values));
    const double err = std::sqrt(I.error_squared);
    err_max = std::max(err_max, err);
    if (slot + 1u == config.time.steps) {
      err_final = err;
      rel_final = err / std::sqrt(I.exact_squared);
    }
  }
  Vec EY = NewLike(Y);
  PetscCallAbort(PETSC_COMM_WORLD, MatMult(spacetime.distributed.ay, Y, EY));
  const double aao = RelativeDifference(EY, data.rhs_y);
  std::printf("EX21_MMS shape=%s dim=%u nx=%u nz=%u steps=%u T=%.6g dt=%.6e scheme=%s"
              " nu=%.6g convection=%s vx=%.6g vy=%.6g"
              " err_L2_T=%.10e rel_err_L2_T=%.10e err_L2_max=%.10e"
              " aao_residual=%.3e%s%s\n",
              shape.c_str(), config.dimension, config.mesh.nx, config.mesh.nz,
              config.time.steps, config.final_time(), dt, TimeSchemeName(config),
              config.diffusivity, HasConvection(config) ? "on" : "off",
              config.convection.vx, config.convection.vy, err_final, rel_final,
              err_max, aao, config.convection.field ? " wind=" : "",
              config.convection.field ? config.convection.field_name : "");
  PetscCallAbort(PETSC_COMM_WORLD, KSPDestroy(&spatial_lu));
  for (Vec* w : {&Y, &r, &y, &prev, &tmp, &EY})
    PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(w));
  PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&coupling));
  PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&diagonal));
  return 0;
}
