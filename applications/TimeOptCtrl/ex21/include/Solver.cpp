// Solver / KKT layer.
//
// One lifetime invariant that is easy to break and invisible in a diff:
//
//   * SolveOptimalControlOwnershipDistributed creates every preconditioner
//     context as a stack local and PETSc stores raw pointers to them.  The
//     contexts outlive the KSP/PC/Mat objects that reference them only because
//     they share this function's frame.

#include "Solver.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <memory>

#include <mpi.h>
#include <petscksp.h>

#include "BoundaryReducedSchur.hpp"
#include "Config.hpp"
#include "DirichletReducedSchur.hpp"
#include "Discretization.hpp"
#include "Preconditioners.hpp"
#include "SpaceTimeOperators.hpp"

/** Release the five vectors, in the order the data side always released
 *  them.  VecDestroy of a null handle is a no-op, so an empty (moved-from or
 *  default) result is released safely. */
static void DestroyDataSideVectors(DistributedDataSideResult& result) {
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&result.rhs_p));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&result.rhs_y));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&result.control));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&result.adjoint));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&result.state));
}

/** Move the handles and the objective values out of `other`, leaving it
 *  empty so that its destructor releases nothing. */
/** Space-time vector of the tracking target y_d at the N discrete times
 *  t^k = k dt, k = 1..N: the data both the right-hand side and the
 *  objective are built from. */
static void FillTrackingTarget(const ProblemConfig& config,
                        const OwnedSpatialLayout& layout, Vec target) {
  PetscScalar* values = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, VecGetArray(target, &values));
  for (unsigned slot = 0; slot < config.time.steps; ++slot) {
    const double time = (slot + 1u) * config.time.dt;
    const PetscInt offset = static_cast<PetscInt>(slot) * layout.local_size;
    for (PetscInt i = 0; i < layout.local_size; ++i)
      values[offset + i] =
          EvaluateTrackingTarget(config, layout.x[i], layout.z[i], time, layout.Axis2(i));
  }
  PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArray(target, &values));
}

/** T3b affine tracking term of inhomogeneous Dirichlet data (see
 *  T3B_AFFINE_DIRICHLET_TRACKING.md):
 *
 *      out_n += scale * w_n * [ M_track_full R_Gamma (g - y_d(t_n)) ]_F
 *
 *  R_Gamma keeps the constrained entries; the constrained rows of the result
 *  are zeroed.  `target` is the space-time nodal target (FillTrackingTarget).
 *  Identically zero when the tracking region does not touch Gamma or when
 *  y_d = g on Gamma. */
static void AddAffineTracking(const ProblemConfig& config,
                              const OwnedSpatialLayout& layout,
                              const DirichletLift& lift, Vec target,
                              double scale, Vec out) {
  Vec v = nullptr, mv = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, MatCreateVecs(lift.tracking_full, &v, &mv));
  const PetscScalar* g = nullptr;
  for (unsigned slot = 0; slot < config.time.steps; ++slot) {
    CopyDataSideSliceToSpatial(target, slot, layout.local_size, v);
    PetscScalar* a = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(lift.g, &g));
    PetscCallAbort(PETSC_COMM_WORLD, VecGetArray(v, &a));
    for (PetscInt i = 0; i < layout.local_size; ++i)
      a[i] = lift.constrained[static_cast<std::size_t>(i)] ? g[i] - a[i] : 0.0;
    PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArray(v, &a));
    PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(lift.g, &g));
    PetscCallAbort(PETSC_COMM_WORLD, MatMult(lift.tracking_full, v, mv));
    PetscCallAbort(PETSC_COMM_WORLD, VecGetArray(mv, &a));
    for (PetscInt i = 0; i < layout.local_size; ++i)
      if (lift.constrained[static_cast<std::size_t>(i)]) a[i] = 0.0;
    PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArray(mv, &a));
    CopySpatialToDataSideSlice(mv, slot, layout.local_size,
                               scale * TrackingSlotWeight(config, slot), out, true);
  }
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&mv));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&v));
}

static bool HasAffineTracking(const DirichletLift* lift) {
  return lift && lift->g && lift->tracking_full && lift->affine_tracking;
}

static void TakeDataSide(DistributedDataSideResult& into,
                  DistributedDataSideResult& other) {
  into.state = other.state;
  into.adjoint = other.adjoint;
  into.control = other.control;
  into.rhs_y = other.rhs_y;
  into.rhs_p = other.rhs_p;
  into.J_track = other.J_track;
  into.J_reg = other.J_reg;
  into.J_track_unweighted = other.J_track_unweighted;
  into.J_reg_unweighted = other.J_reg_unweighted;
  into.J_reg_time = other.J_reg_time;
  other.state = nullptr;
  other.adjoint = nullptr;
  other.control = nullptr;
  other.rhs_y = nullptr;
  other.rhs_p = nullptr;
}

DistributedDataSideResult::~DistributedDataSideResult() {
  DestroyDataSideVectors(*this);
}

DistributedDataSideResult::DistributedDataSideResult(
    DistributedDataSideResult&& other) {
  TakeDataSide(*this, other);
}

DistributedDataSideResult& DistributedDataSideResult::operator=(
    DistributedDataSideResult&& other) {
  if (this != &other) {
    DestroyDataSideVectors(*this);
    TakeDataSide(*this, other);
  }
  return *this;
}

DistributedDataSideResult BuildDistributedRightHandSide(
    const ProblemConfig& config, const OwnedSpatialLayout& layout,
    Mat mass, Mat tracking, Mat space_time_layout, Mat stiffness,
    Mat source_mass, const DirichletLift* lift) {
  // ex21: the source load uses the region mass when one is given; the initial
  // state term always uses `mass`.
  Mat source_operator = source_mass ? source_mass : mass;
  if (IsCrankNicolson(config) && stiffness == nullptr)
    throw std::invalid_argument(
        "the Crank-Nicolson right-hand side needs the stiffness matrix");
  DistributedDataSideResult result;
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatCreateVecs(space_time_layout, &result.rhs_y, nullptr));
  PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(result.rhs_y, &result.rhs_p));
  Vec target = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(result.rhs_y, &target));
  PetscCallAbort(PETSC_COMM_WORLD, VecSet(result.rhs_y, 0.0));
  PetscCallAbort(PETSC_COMM_WORLD, VecSet(result.rhs_p, 0.0));
  FillTrackingTarget(config, layout, target);

  Vec spatial_input = nullptr, spatial_result = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatCreateVecs(mass, &spatial_input, &spatial_result));
  PetscScalar* local = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, VecGetArray(spatial_input, &local));
  for (PetscInt i = 0; i < layout.local_size; ++i)
    local[i] = EvaluateInitialState(config, layout.x[i], layout.z[i], layout.Axis2(i));
  PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArray(spatial_input, &local));
  if (IsCrankNicolson(config) &&
      config.state_bc ==
          ProblemConfig::StateBoundaryCondition::HOMOGENEOUS_DIRICHLET) {
    // ex21 fix.  The eliminated (constrained) rows of CN read
    //   (1/2) y_n,G + (1/2) y_{n-1},G = rhs_G,
    // so a nonzero initial trace y_0,G is propagated as y_n,G = (-1)^n y_0,G
    // instead of being held at zero (backward Euler has a zero sub-diagonal on
    // G and is unaffected).  The homogeneous Dirichlet state has a zero trace:
    // y_0 is restricted to it.  The constrained rows are exactly the rows where
    // the eliminated mass matrix has a zero diagonal; their columns are already
    // zero in M and K, so no unconstrained row changes.
    Vec diagonal = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(spatial_input, &diagonal));
    PetscCallAbort(PETSC_COMM_WORLD, MatGetDiagonal(mass, diagonal));
    const PetscScalar* d = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(diagonal, &d));
    PetscCallAbort(PETSC_COMM_WORLD, VecGetArray(spatial_input, &local));
    for (PetscInt i = 0; i < layout.local_size; ++i)
      if (PetscRealPart(d[i]) == 0.0) local[i] = 0.0;
    PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArray(spatial_input, &local));
    PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(diagonal, &d));
    PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&diagonal));
  }
  if (IsCrankNicolson(config)) {
    // First step: rhs_1 = 1/2 (F_1 + F_0) - A_minus y_0
    //                   = 1/2 (F_1 + F_0) + (M/dt - K/2) y_0.
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatMult(mass, spatial_input, spatial_result));
    CopySpatialToDataSideSlice(spatial_result, 0u, layout.local_size,
                               1.0 / config.time.dt, result.rhs_y, true);
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatMult(stiffness, spatial_input, spatial_result));
    CopySpatialToDataSideSlice(spatial_result, 0u, layout.local_size, -0.5,
                               result.rhs_y, true);
    // The Crank-Nicolson source contribution of interval n is the TRAPEZOIDAL
    // average 1/2 (F_n + F_{n-1}) with F_k = M f(t_k), t_k = k dt: each nodal
    // sample F_k is formed once and added to the one or two intervals it
    // belongs to.  No right-endpoint (backward-Euler) sampling occurs here.
    for (unsigned node = 0; node <= config.time.steps; ++node) {
      const double time = node * config.time.dt;
      PetscCallAbort(PETSC_COMM_WORLD, VecGetArray(spatial_input, &local));
      for (PetscInt i = 0; i < layout.local_size; ++i)
        local[i] = EvaluateSource(config, layout.x[i], layout.z[i], time, layout.Axis2(i));
      PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArray(spatial_input, &local));
      PetscCallAbort(PETSC_COMM_WORLD,
                     MatMult(source_operator, spatial_input, spatial_result));
      // t_node is the right end of interval `node` (slot node-1) and the left
      // end of interval `node+1` (slot node).
      if (node > 0u)
        CopySpatialToDataSideSlice(spatial_result, node - 1u, layout.local_size,
                                   0.5, result.rhs_y, true);
      if (node < config.time.steps)
        CopySpatialToDataSideSlice(spatial_result, node, layout.local_size, 0.5,
                                   result.rhs_y, true);
    }
  } else {
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatMult(mass, spatial_input, spatial_result));
    CopySpatialToDataSideSlice(spatial_result, 0u, layout.local_size,
                               1.0 / config.time.dt, result.rhs_y, true);
    // The source is sampled per time slot, at the same end-of-interval instant
    // t_{n+1} = (slot + 1) dt at which the backward-Euler state equation and the
    // tracking target are sampled.
    for (unsigned slot = 0; slot < config.time.steps; ++slot) {
      const double time = (slot + 1u) * config.time.dt;
      PetscCallAbort(PETSC_COMM_WORLD, VecGetArray(spatial_input, &local));
      for (PetscInt i = 0; i < layout.local_size; ++i)
        local[i] = EvaluateSource(config, layout.x[i], layout.z[i], time, layout.Axis2(i));
      PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArray(spatial_input, &local));
      PetscCallAbort(PETSC_COMM_WORLD,
                     MatMult(source_operator, spatial_input, spatial_result));
      CopySpatialToDataSideSlice(spatial_result, slot, layout.local_size, 1.0,
                                 result.rhs_y, true);
    }
  }

  // -alpha_track w_n M_omega y_d(t_{n+1}); w_n = 1 under backward Euler and
  // (1,...,1,1/2) under the Crank-Nicolson trapezoidal quadrature.
  for (unsigned slot = 0; slot < config.time.steps; ++slot) {
    CopyDataSideSliceToSpatial(target, slot, layout.local_size, spatial_input);
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatMult(tracking, spatial_input, spatial_result));
    CopySpatialToDataSideSlice(spatial_result, slot, layout.local_size,
                               -config.alpha_track *
                                   TrackingSlotWeight(config, slot),
                               result.rhs_p, false);
  }
  // ex21 inhomogeneous Dirichlet data: the constant lifting load
  // -(L_full G)_free on every slot (BE: L y_n; CN: L (y_n + y_{n-1})/2 -- the
  // time-independent lift contributes L G to both).  Null: nothing is done.
  if (lift && lift->load)
    for (unsigned slot = 0; slot < config.time.steps; ++slot)
      CopySpatialToDataSideSlice(lift->load, slot, layout.local_size, 1.0,
                                 result.rhs_y, true);
  // T3b: the affine tracking term.  The adjoint row is
  //   E^T p - alpha_t W M_FF y = -alpha_t W M_FF y_d + alpha_t W M_F,Gamma (g - y_d)_Gamma,
  // since E^T p = alpha_t W [M_FF (y - y_d) + M_F,Gamma (g - y_d)_Gamma].
  if (HasAffineTracking(lift))
    AddAffineTracking(config, layout, *lift, target, config.alpha_track,
                      result.rhs_p);

  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&spatial_result));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&spatial_input));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&target));
  return result;
}

/** The spatial initial control trace u_0 = y_0|_Gamma_c / eta, zero outside C.
 *  It is DATA of the Dirichlet problem -- the control at t = 0 is the trace of
 *  the prescribed initial state -- and enters the temporal regularization as
 *  the affine term q_0 = (1/dt) [M_Gamma u_0; 0; ...; 0]. */
static void FillInitialControlTrace(const ProblemConfig& config,
                                    const OwnedSpatialLayout& layout,
                                    DirichletControlOperators& dirichlet,
                                    Vec spatial) {
  const std::vector<unsigned char>& mask = dirichlet.control_mask();
  PetscScalar* a = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, VecGetArray(spatial, &a));
  for (PetscInt i = 0; i < layout.local_size; ++i)
    a[i] = mask[static_cast<std::size_t>(i)]
               ? EvaluateInitialState(config, layout.x[i], layout.z[i], layout.Axis2(i)) /
                     config.eta
               : 0.0;
  PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArray(spatial, &a));
}

/** `dirichlet` (Dirichlet control only): the control is
 *  u = (eta/alpha_u) R^{-1} B_D^T p, the state's controlled boundary DOFs are
 *  set to eta u (they are identity rows of E_I, zero in the KKT unknown) and
 *  the regularization is the H1(Gamma_c) form R. */
static DistributedDataSideResult ReconstructDistributedDataSide(
    const ProblemConfig& config, const OwnedSpatialLayout& layout,
    Mat tracking, Mat control, Mat space_time_layout, Vec state,
    Vec adjoint, DirichletControlOperators* dirichlet,
    const DirichletLift* lift = nullptr) {
  DistributedDataSideResult result;
  // T3b: with inhomogeneous Dirichlet data the objective is evaluated on the
  // PHYSICAL state y0 + G with the full tracking matrix (free-free, cross and
  // boundary-boundary terms); `state` is the homogeneous unknown y0.
  const bool physical = lift && lift->g && lift->tracking_full;
  Mat tracking_value = physical ? lift->tracking_full : tracking;
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatCreateVecs(space_time_layout, &result.state, nullptr));
  PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(result.state, &result.adjoint));
  PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(result.state, &result.control));
  Vec target = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(result.state, &target));
  // The control active set comes from the assembled operator N itself, not
  // from a second geometric box test.
  const std::vector<unsigned char> active = ActiveControlDofs(control, layout);
  PetscCallAbort(PETSC_COMM_WORLD, VecCopy(state, result.state));
  PetscCallAbort(PETSC_COMM_WORLD, VecCopy(adjoint, result.adjoint));
  FillTrackingTarget(config, layout, target);

  if (dirichlet) {
    // u = (eta/alpha_u) R^{-1} B_D^T p on C; the boundary state is eta u.
    PetscCallAbort(PETSC_COMM_WORLD,
                   dirichlet->ReconstructControl(config.eta / config.alpha_u,
                                                 result.adjoint,
                                                 result.control));
    if (dirichlet->has_time_regularization()) {
      // Q_u U - eta B_D^T P = (gamma_u/dt) q_0, so the control carries the
      // affine part Q_u^{-1} ((gamma_u/dt) q_0) = rho R_H1t^{-1} q_0 on top of the
      // adjoint-driven part above.
      Vec trace = nullptr;
      PetscCallAbort(PETSC_COMM_WORLD,
                     MatCreateVecs(dirichlet->regularization(), &trace, nullptr));
      FillInitialControlTrace(config, layout, *dirichlet, trace);
      PetscCallAbort(PETSC_COMM_WORLD,
                     dirichlet->AddAffineControl(trace, result.control));
      PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&trace));
    }
    const PetscScalar* u = nullptr;
    PetscScalar* y = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(result.control, &u));
    PetscCallAbort(PETSC_COMM_WORLD, VecGetArray(result.state, &y));
    const std::vector<unsigned char>& mask = dirichlet->control_mask();
    for (unsigned slot = 0; slot < config.time.steps; ++slot) {
      const PetscInt offset = static_cast<PetscInt>(slot) * layout.local_size;
      for (PetscInt i = 0; i < layout.local_size; ++i)
        if (mask[static_cast<std::size_t>(i)])
          y[offset + i] = config.eta * u[offset + i];
    }
    PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArray(result.state, &y));
    PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(result.control, &u));
  } else {
    // u = -(eta/alpha_u) p on the active set, 0 elsewhere.
    PetscScalar* p = nullptr;
    PetscScalar* u = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD, VecGetArray(result.adjoint, &p));
    PetscCallAbort(PETSC_COMM_WORLD, VecGetArray(result.control, &u));
    for (unsigned slot = 0; slot < config.time.steps; ++slot) {
      const PetscInt offset = static_cast<PetscInt>(slot) * layout.local_size;
      for (PetscInt i = 0; i < layout.local_size; ++i) {
        u[offset + i] = active[i]
                            ? -(config.eta / config.alpha_u) * p[offset + i]
                            : 0.0;
      }
    }
    PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArray(result.control, &u));
    PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArray(result.adjoint, &p));
  }

  Vec spatial_input = nullptr, spatial_result = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatCreateVecs(tracking, &spatial_input, &spatial_result));
  Vec difference = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(spatial_input, &difference));
  for (unsigned slot = 0; slot < config.time.steps; ++slot) {
    CopyDataSideSliceToSpatial(result.state, slot, layout.local_size,
                               difference);
    CopyDataSideSliceToSpatial(target, slot, layout.local_size, spatial_input);
    PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(difference, -1.0, spatial_input));
    if (physical)
      PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(difference, 1.0, lift->g));
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatMult(tracking_value, difference, spatial_result));
    PetscScalar quadratic = 0.0;
    PetscCallAbort(PETSC_COMM_WORLD,
                   VecDot(difference, spatial_result, &quadratic));
    result.J_track_unweighted += TrackingSlotWeight(config, slot) * 0.5 *
                                 config.alpha_track * PetscRealPart(quadratic);

    CopyDataSideSliceToSpatial(result.control, slot, layout.local_size,
                               spatial_input);
    // The regularization form: N for distributed/Neumann control, the
    // H1(Gamma_c) matrix R for Dirichlet control.  Dirichlet controls are
    // NODAL, so under Crank-Nicolson they carry the trapezoidal weights
    // What_u = diag(1,...,1,1/2); the distributed and Neumann controls are
    // interval quantities whose quadrature is exact with uniform weights.
    const double control_weight =
        dirichlet ? TrackingSlotWeight(config, slot) : 1.0;
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatMult(dirichlet ? dirichlet->regularization() : control,
                           spatial_input, spatial_result));
    PetscCallAbort(PETSC_COMM_WORLD,
                   VecDot(spatial_input, spatial_result, &quadratic));
    result.J_reg_unweighted +=
        control_weight * 0.5 * config.alpha_u * PetscRealPart(quadratic);
  }
  if (dirichlet && dirichlet->has_time_regularization()) {
    // TEMPORAL seminorm, exact for the P1-in-time control:
    //   J_H1t = (gamma_u/(2 dt)) sum_{n=1}^{N} (u_n-u_{n-1})^T M_Gamma (u_n-u_{n-1}),
    // whose n = 1 difference uses the DATA u_0 = y_0|_Gamma_c/eta and therefore
    // carries the additive constant (gamma_u/(2 dt)) u_0^T M_Gamma u_0.  The
    // metric is M_Gamma, NOT the spatial regularization R_Gamma.  The common
    // factor dt is applied to J_reg below, so the sum is divided by dt twice
    // here: once for the quadrature weight and once for the difference quotient.
    Vec previous = nullptr, difference_u = nullptr, mass_u = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(spatial_input, &previous));
    PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(spatial_input, &difference_u));
    PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(spatial_input, &mass_u));
    FillInitialControlTrace(config, layout, *dirichlet, previous);
    double sum = 0.0;
    for (unsigned slot = 0; slot < config.time.steps; ++slot) {
      CopyDataSideSliceToSpatial(result.control, slot, layout.local_size,
                                 difference_u);
      PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(difference_u, -1.0, previous));
      PetscCallAbort(PETSC_COMM_WORLD,
                     dirichlet->ApplyFaceMass(difference_u, mass_u));
      PetscScalar quadratic = 0.0;
      PetscCallAbort(PETSC_COMM_WORLD,
                     VecDot(difference_u, mass_u, &quadratic));
      sum += PetscRealPart(quadratic);
      CopyDataSideSliceToSpatial(result.control, slot, layout.local_size,
                                 previous);
    }
    const double energy =
        0.5 * config.control_h1_time_weight * sum / config.time.dt;
    // `energy` already carries the dt of the time quadrature, so it IS the
    // reported J_reg_time; J_reg_unweighted is multiplied by dt below.
    result.J_reg_time = energy;
    result.J_reg_unweighted += energy / config.time.dt;
    for (Vec* v : {&mass_u, &difference_u, &previous})
      PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(v));
  }
  if (dirichlet && IsCrankNicolson(config)) {
    // The trapezoidal control quadrature also weighs the INITIAL node, whose
    // control u_0 = y_0|_C / eta is DATA: an additive constant of the
    // objective VALUE with no derivative, so it enters here and nowhere in
    // the KKT system.
    const std::vector<unsigned char>& mask = dirichlet->control_mask();
    PetscScalar* initial = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD, VecGetArray(spatial_input, &initial));
    for (PetscInt i = 0; i < layout.local_size; ++i)
      initial[i] = mask[static_cast<std::size_t>(i)]
                       ? EvaluateInitialState(config, layout.x[i], layout.z[i], layout.Axis2(i)) /
                             config.eta
                       : 0.0;
    PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArray(spatial_input, &initial));
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatMult(dirichlet->regularization(), spatial_input,
                           spatial_result));
    PetscScalar quadratic = 0.0;
    PetscCallAbort(PETSC_COMM_WORLD,
                   VecDot(spatial_input, spatial_result, &quadratic));
    result.J_reg_unweighted +=
        0.5 * 0.5 * config.alpha_u * PetscRealPart(quadratic);
  }
  if (IsCrankNicolson(config)) {
    // The trapezoidal rule also weighs the INITIAL node, which is data and not
    // an unknown: it is an additive constant of the objective VALUE and has no
    // derivative, so it enters here and nowhere in the KKT system.
    PetscScalar* initial = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD, VecGetArray(difference, &initial));
    for (PetscInt i = 0; i < layout.local_size; ++i)
      initial[i] = EvaluateInitialState(config, layout.x[i], layout.z[i], layout.Axis2(i)) -
                   EvaluateTrackingTarget(config, layout.x[i], layout.z[i], 0.0, layout.Axis2(i));
    PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArray(difference, &initial));
    // physical: y_0 (whose trace is g) against y_d(0), full tracking matrix
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatMult(tracking_value, difference, spatial_result));
    PetscScalar quadratic = 0.0;
    PetscCallAbort(PETSC_COMM_WORLD,
                   VecDot(difference, spatial_result, &quadratic));
    result.J_track_unweighted +=
        0.5 * 0.5 * config.alpha_track * PetscRealPart(quadratic);
  }
  // Time quadrature of
  //   (alpha_t/2)\int_0^T ||y-y_d||^2_omega dt + (alpha_u/2)\int_0^T ||u||^2_C dt
  // Backward Euler: the right-rectangle rule over the N nodes t^k = k*dt, so
  // every term carries the same weight dt.  Crank-Nicolson: the trapezoidal
  // rule for the tracking term (half weights at t=0 and t=T, applied above)
  // and the EXACT integral of the piecewise-constant control, which is again
  // dt per interval.  Either way the common factor is dt.
  result.J_track = config.time.dt * result.J_track_unweighted;
  result.J_reg = config.time.dt * result.J_reg_unweighted;

  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&difference));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&spatial_result));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&spatial_input));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&target));
  return result;
}

/** The composite [y; p] the outer KSP works on is the standard vector
 *  MatCreateVecs hands out for the MatNest (ex20 never selects VECNEST), so
 *  each rank's block is exactly its local y followed by its local p. */
static void CopyTwoLocalSubvectorsToComposite(Vec first, Vec second,
                                       Vec composite) {
  PetscInt first_local = 0, second_local = 0, composite_local = 0;
  PetscCallAbort(PETSC_COMM_WORLD, VecGetLocalSize(first, &first_local));
  PetscCallAbort(PETSC_COMM_WORLD, VecGetLocalSize(second, &second_local));
  PetscCallAbort(PETSC_COMM_WORLD,
                 VecGetLocalSize(composite, &composite_local));
  if (composite_local != first_local + second_local)
    throw std::runtime_error("invalid local MatNest composite layout");
  const PetscScalar* first_values = nullptr;
  const PetscScalar* second_values = nullptr;
  PetscScalar* composite_values = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(first, &first_values));
  PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(second, &second_values));
  PetscCallAbort(PETSC_COMM_WORLD, VecGetArray(composite, &composite_values));
  for (PetscInt i = 0; i < first_local; ++i)
    composite_values[i] = first_values[i];
  for (PetscInt i = 0; i < second_local; ++i)
    composite_values[first_local + i] = second_values[i];
  PetscCallAbort(PETSC_COMM_WORLD,
                 VecRestoreArray(composite, &composite_values));
  PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(second, &second_values));
  PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(first, &first_values));
}

static void CopyCompositeToTwoLocalSubvectors(Vec composite, Vec first,
                                       Vec second) {
  PetscInt first_local = 0, second_local = 0, composite_local = 0;
  PetscCallAbort(PETSC_COMM_WORLD, VecGetLocalSize(first, &first_local));
  PetscCallAbort(PETSC_COMM_WORLD, VecGetLocalSize(second, &second_local));
  PetscCallAbort(PETSC_COMM_WORLD,
                 VecGetLocalSize(composite, &composite_local));
  if (composite_local != first_local + second_local)
    throw std::runtime_error("invalid local MatNest composite split");
  const PetscScalar* composite_values = nullptr;
  PetscScalar* first_values = nullptr;
  PetscScalar* second_values = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD,
                 VecGetArrayRead(composite, &composite_values));
  PetscCallAbort(PETSC_COMM_WORLD, VecGetArray(first, &first_values));
  PetscCallAbort(PETSC_COMM_WORLD, VecGetArray(second, &second_values));
  for (PetscInt i = 0; i < first_local; ++i)
    first_values[i] = composite_values[i];
  for (PetscInt i = 0; i < second_local; ++i)
    second_values[i] = composite_values[first_local + i];
  PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArray(second, &second_values));
  PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArray(first, &first_values));
  PetscCallAbort(PETSC_COMM_WORLD,
                 VecRestoreArrayRead(composite, &composite_values));
}

// ===========================================================================
// KKT residual diagnostics.
// ===========================================================================

/** Apply the space-time block-diagonal operator (W (x) spatial), with
 *  W = diag(scale, ..., scale, final_scale); `final_scale` < 0 means "equal to
 *  `scale`", the uniform backward-Euler case. */
static void ApplySpatialBlockDiagonal(Mat spatial, const OwnedSpatialLayout& layout,
                               unsigned steps, double scale, Vec input,
                               Vec output, double final_weight = 1.0) {
  Vec spatial_in = nullptr, spatial_out = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatCreateVecs(spatial, &spatial_in, &spatial_out));
  for (unsigned slot = 0; slot < steps; ++slot) {
    const double slot_scale =
        (slot + 1u == steps) ? scale * final_weight : scale;
    CopyDataSideSliceToSpatial(input, slot, layout.local_size, spatial_in);
    PetscCallAbort(PETSC_COMM_WORLD, MatMult(spatial, spatial_in, spatial_out));
    CopySpatialToDataSideSlice(spatial_out, slot, layout.local_size, slot_scale,
                               output, false);
  }
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&spatial_out));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&spatial_in));
}

/** Residuals of the three KKT blocks for the converged triple (y, u, p):
 *
 *      r_y = E y - eta N u - g                        STATE
 *      r_p = E^T p - alpha_track M_omega (y - y_d)    ADJOINT
 *      r_u = N (alpha_u u + eta p)                    STATIONARITY
 *
 *  r_y and r_p are computed through the reconstructed u and the rebuilt target
 *  y_d rather than through the off-diagonal block and the right-hand side, so
 *  they detect a broken reconstruction.  Where the active set is correct they
 *  satisfy ||true_residual||^2 = ||r_y||^2 + ||r_p||^2.
 *
 *  r_u is a consistency check only: u is DEFINED as -(eta/alpha_u) p on the
 *  active set, so r_u vanishes identically unless the active set and N
 *  disagree.  It says nothing about whether (y, p) is optimal.  An independent
 *  optimality test would need a separately computed adjoint. */
static KktResidualReport ComputeKktResiduals(
    const ProblemConfig& config, const OwnedSpatialLayout& layout,
    Mat tracking, Mat control, DistributedSpaceTimeOperators& operators,
    Vec rhs_y, Vec rhs_p, Vec y, Vec p, Vec u,
    DirichletControlOperators* dirichlet, const DirichletLift* lift = nullptr) {
  KktResidualReport report;
  Vec t1 = nullptr, t2 = nullptr, r = nullptr, target = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(y, &t1));
  PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(y, &t2));
  PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(y, &r));
  PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(y, &target));
  FillTrackingTarget(config, layout, target);

  double n1 = 0.0, n2 = 0.0, n3 = 0.0;

  // STATE:  r_y = E y - eta * (I(x)N) u - g   (E = E_CN under Crank-Nicolson;
  //         the distributed CN control is piecewise constant per interval, so
  //         the control operator stays block diagonal, eta (I(x)N))
  //         (Dirichlet control:  r_y = E_I y + eta B_D u - g)
  PetscCallAbort(PETSC_COMM_WORLD, MatMult(operators.ay, y, t1));
  if (dirichlet) {
    PetscCallAbort(PETSC_COMM_WORLD, dirichlet->ApplyBD(u, t2));
    PetscCallAbort(PETSC_COMM_WORLD, VecScale(t2, -config.eta));
  } else {
    ApplySpatialBlockDiagonal(control, layout, config.time.steps, config.eta, u,
                              t2);
  }
  PetscCallAbort(PETSC_COMM_WORLD, VecNorm(t1, NORM_2, &n1));
  PetscCallAbort(PETSC_COMM_WORLD, VecNorm(t2, NORM_2, &n2));
  PetscCallAbort(PETSC_COMM_WORLD, VecNorm(rhs_y, NORM_2, &n3));
  PetscCallAbort(PETSC_COMM_WORLD, VecCopy(t1, r));
  PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(r, -1.0, t2));
  PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(r, -1.0, rhs_y));
  PetscCallAbort(PETSC_COMM_WORLD, VecNorm(r, NORM_2, &report.state_abs));
  report.state_rel =
      report.state_abs / std::max(1.0, std::max(n1, std::max(n2, n3)));

  // ADJOINT:  r_p = E^T p - alpha_t * (W(x)M_omega) (y - y_d),
  //           W = I (backward Euler) or diag(1,...,1,1/2) (Crank-Nicolson)
  PetscCallAbort(PETSC_COMM_WORLD, MatMult(operators.ap, p, t1));
  PetscCallAbort(PETSC_COMM_WORLD, VecCopy(y, r));
  PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(r, -1.0, target));
  ApplySpatialBlockDiagonal(tracking, layout, config.time.steps,
                            config.alpha_track, r, t2,
                            TrackingSlotWeight(config, config.time.steps - 1u));
  // T3b: + alpha_t w_n M_F,Gamma (g - y_d)_Gamma
  if (HasAffineTracking(lift))
    AddAffineTracking(config, layout, *lift, target, config.alpha_track, t2);
  PetscCallAbort(PETSC_COMM_WORLD, VecNorm(t1, NORM_2, &n1));
  PetscCallAbort(PETSC_COMM_WORLD, VecNorm(t2, NORM_2, &n2));
  PetscCallAbort(PETSC_COMM_WORLD, VecCopy(t1, r));
  PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(r, -1.0, t2));
  PetscCallAbort(PETSC_COMM_WORLD, VecNorm(r, NORM_2, &report.adjoint_abs));
  report.adjoint_rel =
      report.adjoint_abs / std::max(1.0, std::max(n1, n2));

  // STATIONARITY:  r_u = (I(x)N) (alpha_u u + eta p)
  //                (Dirichlet control:  r_u = alpha_u R u - eta B_D^T p,
  //                 a genuine optimality test: u is not a function of p alone)
  if (dirichlet) {
    // r_u = Q_u u - eta B_D^T p - (gamma_u/dt) q_0, Q_u = alpha_u R_H1t.  The
    // affine term is present only when gamma_u > 0, where it is an equal
    // partner of the other two: dropping it would make an H1-in-time optimum
    // look non-stationary.
    PetscCallAbort(PETSC_COMM_WORLD, dirichlet->ApplyR(u, t2));
    PetscCallAbort(PETSC_COMM_WORLD, VecScale(t2, config.alpha_u));
    PetscCallAbort(PETSC_COMM_WORLD, VecNorm(t2, NORM_2, &n1));
    PetscCallAbort(PETSC_COMM_WORLD, VecCopy(t2, t1));
    PetscCallAbort(PETSC_COMM_WORLD, dirichlet->ApplyBDT(p, t2));
    PetscCallAbort(PETSC_COMM_WORLD, VecScale(t2, config.eta));
    PetscCallAbort(PETSC_COMM_WORLD, VecNorm(t2, NORM_2, &n2));
    PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(t1, -1.0, t2));
    if (dirichlet->has_time_regularization()) {
      Vec trace = nullptr;
      PetscCallAbort(PETSC_COMM_WORLD,
                     MatCreateVecs(dirichlet->regularization(), &trace, nullptr));
      FillInitialControlTrace(config, layout, *dirichlet, trace);
      PetscCallAbort(PETSC_COMM_WORLD, dirichlet->ApplyAffineData(trace, t2));
      const double source = ControlTimeSourceWeight(config);
      PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(t1, -source, t2));
      PetscReal affine = 0.0;
      PetscCallAbort(PETSC_COMM_WORLD, VecNorm(t2, NORM_2, &affine));
      n2 = std::max(n2, source * affine);
      PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&trace));
    }
  } else {
    PetscCallAbort(PETSC_COMM_WORLD, VecCopy(u, r));
    PetscCallAbort(PETSC_COMM_WORLD, VecScale(r, config.alpha_u));
    PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(r, config.eta, p));
    ApplySpatialBlockDiagonal(control, layout, config.time.steps, 1.0, r, t1);
    ApplySpatialBlockDiagonal(control, layout, config.time.steps,
                              config.alpha_u, u, t2);
    PetscCallAbort(PETSC_COMM_WORLD, VecNorm(t2, NORM_2, &n1));
    ApplySpatialBlockDiagonal(control, layout, config.time.steps, config.eta, p,
                              t2);
    PetscCallAbort(PETSC_COMM_WORLD, VecNorm(t2, NORM_2, &n2));
  }
  PetscCallAbort(PETSC_COMM_WORLD,
                 VecNorm(t1, NORM_2, &report.stationarity_abs));
  report.stationarity_rel =
      report.stationarity_abs / std::max(1.0, std::max(n1, n2));

  // Reduced 2x2 block residuals, for comparison with true_residual.
  PetscCallAbort(PETSC_COMM_WORLD, MatMult(operators.ay, y, t1));
  PetscCallAbort(PETSC_COMM_WORLD, MatMult(operators.ayp, p, t2));
  PetscCallAbort(PETSC_COMM_WORLD, VecCopy(t1, r));
  PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(r, 1.0, t2));
  PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(r, -1.0, rhs_y));
  PetscCallAbort(PETSC_COMM_WORLD, VecNorm(r, NORM_2, &report.block_y_abs));
  PetscCallAbort(PETSC_COMM_WORLD, MatMult(operators.apy, y, t1));
  PetscCallAbort(PETSC_COMM_WORLD, MatMult(operators.ap, p, t2));
  PetscCallAbort(PETSC_COMM_WORLD, VecCopy(t1, r));
  PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(r, 1.0, t2));
  PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(r, -1.0, rhs_p));
  PetscCallAbort(PETSC_COMM_WORLD, VecNorm(r, NORM_2, &report.block_p_abs));
  report.reduced_abs = std::sqrt(report.block_y_abs * report.block_y_abs +
                                 report.block_p_abs * report.block_p_abs);

  // The reconstructed control must be identically zero outside the active set
  // of N.  `geometric_mask_dofs` reports what the purely geometric box
  // predicate would select, so the two selections can be compared.
  const std::vector<unsigned char> active = ActiveControlDofs(control, layout);
  const PetscScalar* control_values = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(u, &control_values));
  double local_max = 0.0;
  long long local_offactive = 0, local_active = 0, local_mask = 0;
  for (PetscInt i = 0; i < layout.local_size; ++i) {
    const bool masked =
        OwnedDofIsControlled(config, layout.x[i], layout.z[i], layout.Axis2(i));
    if (active[i]) ++local_active;
    if (masked) ++local_mask;
    if (active[i]) continue;
    bool any = false;
    for (unsigned slot = 0; slot < config.time.steps; ++slot) {
      const double value = std::fabs(PetscRealPart(
          control_values[static_cast<PetscInt>(slot) * layout.local_size + i]));
      if (value > local_max) local_max = value;
      if (value != 0.0) any = true;
    }
    if (any) ++local_offactive;
  }
  PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(u, &control_values));
  MPI_Allreduce(&local_max, &report.control_offactive_max, 1, MPI_DOUBLE,
                MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&local_offactive, &report.control_offactive_dofs, 1,
                MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&local_active, &report.active_dofs, 1, MPI_LONG_LONG,
                MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&local_mask, &report.geometric_mask_dofs, 1, MPI_LONG_LONG,
                MPI_SUM, MPI_COMM_WORLD);

  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&target));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&r));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&t2));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&t1));
  return report;
}

static void PrintKktResiduals(const KktResidualReport& report) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0) return;
  std::cout << std::setprecision(17)
            << "EX20_KKT state_abs=" << report.state_abs
            << " state_rel=" << report.state_rel
            << " adjoint_abs=" << report.adjoint_abs
            << " adjoint_rel=" << report.adjoint_rel
            << " stationarity_abs=" << report.stationarity_abs
            << " stationarity_rel=" << report.stationarity_rel
            << " block_y_abs=" << report.block_y_abs
            << " block_p_abs=" << report.block_p_abs
            << " reduced_abs=" << report.reduced_abs
            << " active_dofs=" << report.active_dofs
            << " geometric_mask_dofs=" << report.geometric_mask_dofs
            << " offactive_dofs=" << report.control_offactive_dofs
            << " offactive_max_u=" << report.control_offactive_max
            << std::endl;
}

/** Build the outer KSP and its PCFIELDSPLIT preconditioner from `policy`.
 *
 *  Every PETSc call, value and ordering here is the solver's configuration
 *  sequence; nothing else configures it.
 *
 *  LIFETIME: `matching_context`, `y_split_sweep`, `y_counter` and `p_counter`
 *  live in the CALLER's frame and PETSc stores raw pointers to them.  They must
 *  outlive the returned KSP. */
static KSP ConfigureKktSolver(const KktSolverPolicy& policy,
                       const ProblemConfig& config,
                       DistributedSpaceTimeOperators& operators,
                       AssembledSpaceTimeOperators& assembled,
                       Mat mass,
                       DistributedMatchingSchurPCContext& matching_context,
                       TimeSweepContext& y_split_sweep,
                       InnerIterationCounter& y_counter,
                       InnerIterationCounter& p_counter,
                       BoundaryReducedSchurOperators* reduced,
                       DirichletReducedSchurOperators* dirichlet_reduced) {
  KSP outer = nullptr;
  PC outer_pc = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, KSPCreate(PETSC_COMM_WORLD, &outer));
  PetscCallAbort(PETSC_COMM_WORLD,
                 KSPSetOperators(outer, operators.nest, operators.nest));
  PetscCallAbort(PETSC_COMM_WORLD, KSPSetType(outer, KSPFGMRES));
  PetscCallAbort(PETSC_COMM_WORLD,
                 KSPGMRESSetRestart(outer, static_cast<PetscInt>(policy.outer_restart)));
  PetscCallAbort(PETSC_COMM_WORLD,
                 KSPSetTolerances(outer, policy.outer_rtol, PETSC_DEFAULT,
                                  PETSC_DEFAULT,
                                  static_cast<PetscInt>(policy.outer_maxits)));
  PetscCallAbort(PETSC_COMM_WORLD,
                 KSPSetInitialGuessNonzero(outer, PETSC_FALSE));
  PetscCallAbort(PETSC_COMM_WORLD, KSPGetPC(outer, &outer_pc));
  PetscCallAbort(PETSC_COMM_WORLD, PCSetType(outer_pc, PCFIELDSPLIT));
  if (IsMatchingSchur(policy.pc_mode)) {
    PetscCallAbort(PETSC_COMM_WORLD,
                   PCFieldSplitSetType(outer_pc, PC_COMPOSITE_SCHUR));
    PetscCallAbort(PETSC_COMM_WORLD,
                   PCFieldSplitSetSchurFactType(
                       outer_pc, PC_FIELDSPLIT_SCHUR_FACT_FULL));
    PetscCallAbort(PETSC_COMM_WORLD,
                   PCFieldSplitSetSchurPre(
                       outer_pc, PC_FIELDSPLIT_SCHUR_PRE_USER,
                       matching_context.schur_preconditioner));
  } else if (IsBoundaryReduced(policy.pc_mode) ||
             IsDirichletReduced(policy.pc_mode)) {
    // The Schur split is solved exactly by a PCSHELL, which needs no
    // preconditioning matrix.
    PetscCallAbort(PETSC_COMM_WORLD,
                   PCFieldSplitSetType(outer_pc, PC_COMPOSITE_SCHUR));
    PetscCallAbort(PETSC_COMM_WORLD,
                   PCFieldSplitSetSchurFactType(
                       outer_pc, PC_FIELDSPLIT_SCHUR_FACT_FULL));
    PetscCallAbort(PETSC_COMM_WORLD,
                   PCFieldSplitSetSchurPre(outer_pc, PC_FIELDSPLIT_SCHUR_PRE_SELF,
                                           nullptr));
  } else {
    PetscCallAbort(PETSC_COMM_WORLD,
                   PCFieldSplitSetType(outer_pc, PC_COMPOSITE_ADDITIVE));
  }
  IS nest_rows[2] = {nullptr, nullptr};
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatNestGetISs(operators.nest, nest_rows, nullptr));
  PetscCallAbort(PETSC_COMM_WORLD,
                 PCFieldSplitSetIS(outer_pc, "y", nest_rows[0]));
  PetscCallAbort(PETSC_COMM_WORLD,
                 PCFieldSplitSetIS(
                     outer_pc,
                     IsMatchingSchur(policy.pc_mode) ||
                             IsBoundaryReduced(policy.pc_mode) ||
                             IsDirichletReduced(policy.pc_mode)
                         ? "schur" : "p",
                     nest_rows[1]));
  PetscCallAbort(PETSC_COMM_WORLD, KSPSetFromOptions(outer));
  PetscCallAbort(PETSC_COMM_WORLD, KSPSetUp(outer));

  PetscInt number_of_splits = 0;
  KSP* split_solvers = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD,
                 PCFieldSplitGetSubKSP(outer_pc, &number_of_splits,
                                       &split_solvers));
  if (number_of_splits != 2) {
    if (split_solvers)
      PetscCallAbort(PETSC_COMM_WORLD, PetscFree(split_solvers));
    throw std::runtime_error("parallel fieldsplit did not create y,p solvers");
  }
  if (IsMatchingSchur(policy.pc_mode)) {
    if (policy.pc_mode == DistributedSolvePc::SCHUR_SWEEP) {
      // The A00 solves that drive the Schur-complement action become
      // EXACT triangular sweeps, so the outer Schur operator is applied
      // essentially exactly instead of to a 1e-6 Krylov tolerance.
      const bool crank_nicolson = IsCrankNicolson(config);
      InitializeTimeSweepContext(
          y_split_sweep, TimeSweepDirection::FORWARD,
          matching_context.sweep_matrices.A,
          crank_nicolson ? matching_context.sweep_matrices.A_coupling : mass,
          config.time.steps, config.time.dt, policy.spatial_rtol,
          "fieldsplit_y_sweep_", "y", crank_nicolson ? 1.0 : 0.0);
      PetscCallAbort(PETSC_COMM_WORLD,
                     KSPSetOperators(split_solvers[0], operators.ay, operators.ay));
      PetscCallAbort(PETSC_COMM_WORLD,
                     KSPSetType(split_solvers[0], KSPPREONLY));
      PC y_pc = nullptr;
      PetscCallAbort(PETSC_COMM_WORLD, KSPGetPC(split_solvers[0], &y_pc));
      PetscCallAbort(PETSC_COMM_WORLD, PCSetType(y_pc, PCSHELL));
      PetscCallAbort(PETSC_COMM_WORLD, PCShellSetContext(y_pc, &y_split_sweep));
      PetscCallAbort(PETSC_COMM_WORLD, PCShellSetApply(y_pc, ApplyTimeSweep));
      PetscCallAbort(PETSC_COMM_WORLD, KSPSetUp(split_solvers[0]));
    } else {
      ConfigureFieldSplitInnerGmresGamg(split_solvers[0], operators.ay,
                                        assembled.ay, policy.inner_rtol);
    }
    PetscCallAbort(PETSC_COMM_WORLD, KSPSetType(split_solvers[1], KSPGMRES));
    PetscCallAbort(PETSC_COMM_WORLD, KSPGMRESSetRestart(split_solvers[1], 30));
    PetscCallAbort(PETSC_COMM_WORLD,
                   KSPSetTolerances(split_solvers[1], policy.inner_rtol,
                                    PETSC_DEFAULT, PETSC_DEFAULT, 100));
    PetscCallAbort(PETSC_COMM_WORLD,
                   KSPSetInitialGuessNonzero(split_solvers[1], PETSC_FALSE));
    PC schur_pc = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD, KSPGetPC(split_solvers[1], &schur_pc));
    PetscCallAbort(PETSC_COMM_WORLD, PCSetType(schur_pc, PCSHELL));
    PetscCallAbort(PETSC_COMM_WORLD,
                   PCShellSetContext(schur_pc, &matching_context));
    PetscCallAbort(PETSC_COMM_WORLD,
                   PCShellSetApply(
                       schur_pc, ApplyDistributedMatchingSchurPreconditioner));
    PetscCallAbort(PETSC_COMM_WORLD, KSPSetFromOptions(split_solvers[1]));
    PetscCallAbort(PETSC_COMM_WORLD, KSPSetUp(split_solvers[1]));
  } else if (IsBoundaryReduced(policy.pc_mode) ||
             IsDirichletReduced(policy.pc_mode)) {
    // Split 0: the exact forward time sweep on the reduced operator's A
    // (A_I, the hatted operator, for Dirichlet control).
    // The forward sweep on E (E_CN under Crank-Nicolson) couples through
    // (1/dt) M with backward Euler and through the reduced operator's own
    // -A_minus block, with unit scale, under Crank-Nicolson.  Both reduced
    // solvers publish that pair; using the mass matrix instead would make the
    // A00 solve inexact, and with it PETSc's Schur complement, so the exact
    // PCSHELL of split 1 would no longer be its inverse.
    Mat y_coupling = reduced ? reduced->coupling()
                             : dirichlet_reduced->coupling();
    const double y_coupling_scale =
        IsCrankNicolson(config)
            ? (reduced ? reduced->coupling_scale()
                       : dirichlet_reduced->coupling_scale())
            : 0.0;
    InitializeTimeSweepContext(
        y_split_sweep, TimeSweepDirection::FORWARD,
        reduced ? reduced->diagonal() : dirichlet_reduced->diagonal(),
        y_coupling, config.time.steps, config.time.dt, policy.spatial_rtol,
        "fieldsplit_y_sweep_", "y", y_coupling_scale);
    PetscCallAbort(PETSC_COMM_WORLD,
                   KSPSetOperators(split_solvers[0], operators.ay, operators.ay));
    PetscCallAbort(PETSC_COMM_WORLD,
                   KSPSetType(split_solvers[0], KSPPREONLY));
    PC y_pc = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD, KSPGetPC(split_solvers[0], &y_pc));
    PetscCallAbort(PETSC_COMM_WORLD, PCSetType(y_pc, PCSHELL));
    PetscCallAbort(PETSC_COMM_WORLD, PCShellSetContext(y_pc, &y_split_sweep));
    PetscCallAbort(PETSC_COMM_WORLD, PCShellSetApply(y_pc, ApplyTimeSweep));
    PetscCallAbort(PETSC_COMM_WORLD, KSPSetUp(split_solvers[0]));
    // Split 1: the exact reduced Schur solve; the shell context is the owner,
    // which outlives this KSP.
    PetscCallAbort(PETSC_COMM_WORLD, KSPSetType(split_solvers[1], KSPPREONLY));
    PC schur_pc = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD, KSPGetPC(split_solvers[1], &schur_pc));
    PetscCallAbort(PETSC_COMM_WORLD, PCSetType(schur_pc, PCSHELL));
    if (reduced) {
      PetscCallAbort(PETSC_COMM_WORLD, PCShellSetContext(schur_pc, reduced));
      PetscCallAbort(PETSC_COMM_WORLD,
                     PCShellSetApply(schur_pc, ApplyBoundaryReducedSchur));
      PetscCallAbort(PETSC_COMM_WORLD,
                     PCShellSetName(schur_pc, "boundary-reduced Schur"));
    } else {
      PetscCallAbort(PETSC_COMM_WORLD,
                     PCShellSetContext(schur_pc, dirichlet_reduced));
      PetscCallAbort(PETSC_COMM_WORLD,
                     PCShellSetApply(schur_pc, ApplyDirichletReducedSchur));
      PetscCallAbort(PETSC_COMM_WORLD,
                     PCShellSetName(schur_pc, "Dirichlet reduced Schur"));
    }
    PetscCallAbort(PETSC_COMM_WORLD, KSPSetFromOptions(split_solvers[1]));
    PetscCallAbort(PETSC_COMM_WORLD, KSPSetUp(split_solvers[1]));
  } else {
    ConfigureFieldSplitInnerGmresGamg(split_solvers[0], operators.ay,
                                      assembled.ay, policy.inner_rtol);
    ConfigureFieldSplitInnerGmresGamg(split_solvers[1], operators.ap,
                                      assembled.ap, policy.inner_rtol);
  }
  PetscCallAbort(PETSC_COMM_WORLD,
                 KSPMonitorSet(split_solvers[0], CountInnerIterations,
                               &y_counter, nullptr));
  PetscCallAbort(PETSC_COMM_WORLD,
                 KSPMonitorSet(split_solvers[1], CountInnerIterations,
                               &p_counter, nullptr));
  PetscCallAbort(PETSC_COMM_WORLD, PetscFree(split_solvers));
  return outer;
}

DistributedOptimalControlSolve SolveOptimalControlOwnershipDistributed(
    const ProblemConfig& config, const OwnedSpatialLayout& layout,
    Mat mass, Mat tracking, Mat control, Mat stiffness,
    DistributedSpaceTimeOperators& operators,
    AssembledSpaceTimeOperators& assembled,
    const KktSolverPolicy& policy,
    BoundaryReducedSchurOperators* reduced,
    DirichletControlOperators* dirichlet, Mat mass_full, Mat source_mass,
    const DirichletLift* lift) {
  if ((dirichlet != nullptr) !=
      (config.control.type == ProblemConfig::ControlType::DIRICHLET))
    throw std::invalid_argument(
        "Dirichlet control operators must be given exactly for Dirichlet"
        " control");
  if (dirichlet && policy.pc_mode != DistributedSolvePc::ADDITIVE_GAMG &&
      !IsDirichletReduced(policy.pc_mode))
    throw std::invalid_argument(
        "Dirichlet control is solved with ADDITIVE_GAMG or"
        " SCHUR_DIRICHLET_REDUCED only: the matching / sweep / boundary-reduced"
        " Schur preconditioners need a control operator that is block diagonal"
        " in time");
  if (!dirichlet && IsDirichletReduced(policy.pc_mode))
    throw std::invalid_argument(
        "SCHUR_DIRICHLET_REDUCED requires Dirichlet control");
  if (IsCrankNicolson(config) && stiffness == nullptr)
    throw std::invalid_argument(
        "Crank-Nicolson needs the stiffness matrix (right-hand side and"
        " sweeps)");
  if (reduced && !IsBoundaryReduced(policy.pc_mode))
    throw std::invalid_argument(
        "a boundary-reduced Schur operator was given to a solve of another mode");
  // A caller-owned reduced operator is accepted only for the problem it was
  // built from (see BoundaryReducedSchur.hpp, REUSE); checked before any
  // object of this solve exists.
  if (reduced) {
    std::string reason;
    if (!reduced->IsCompatible(config, mass, stiffness, control, tracking,
                               &reason))
      throw std::invalid_argument(
          "BoundaryReducedSchurOperators is incompatible with this solve: " +
          reason);
  }
  DistributedOptimalControlSolve result;
  // rhs_y / rhs_p are the two Nest sub-vectors of the right-hand side.
  // Dirichlet control: the data terms (initial state, source) are formed
  // with the unhatted mass matrix -- the initial boundary trace enters the
  // first step through (M/dt) y_0 -- and then restricted to the free rows.
  // Dirichlet control: the data terms are formed with the UNHATTED M and K.
  // Under Crank-Nicolson the first-step term is (M/dt - K/2) y_0 on the full
  // DOF set; because eta u_0 = y_0|_C, that single term carries both
  // -Aminus_II y_I,0 and -eta Gminus_IC u_0, so u_0 is data and is never
  // zeroed -- exactly as backward Euler's (M/dt) y_0 does.
  DistributedDataSideResult rhs_data = BuildDistributedRightHandSide(
      config, layout, dirichlet ? mass_full : mass, tracking, operators.ay,
      dirichlet ? dirichlet->unhatted_stiffness() : stiffness, source_mass,
      lift);
  if (dirichlet)
    PetscCallAbort(PETSC_COMM_WORLD,
                   dirichlet->ZeroControlledRows(rhs_data.rhs_y));
  if (dirichlet && dirichlet->has_time_regularization()) {
    // Eliminating U from the affine control stationarity
    //     Q_u U - eta B_D^T P = (gamma_u/dt) q_0
    // leaves the state equation  E_I Y + eta^2 B_D Q_u^{-1} B_D^T P
    //                              = g - eta (gamma_u/dt) B_D Q_u^{-1} q_0,
    // so the state right-hand side gains the term below.  It vanishes
    // identically at gamma_u = 0, where this branch is not entered at all.
    Vec trace = nullptr, correction = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatCreateVecs(dirichlet->regularization(), &trace, nullptr));
    PetscCallAbort(PETSC_COMM_WORLD,
                   VecDuplicate(rhs_data.rhs_y, &correction));
    FillInitialControlTrace(config, layout, *dirichlet, trace);
    PetscCallAbort(PETSC_COMM_WORLD,
                   dirichlet->ApplyAffineStateCorrection(trace, correction));
    PetscCallAbort(PETSC_COMM_WORLD,
                   VecAXPY(rhs_data.rhs_y, -config.eta, correction));
    PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&correction));
    PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&trace));
  }

  Vec solution_y = nullptr, solution_p = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatCreateVecs(operators.ay, &solution_y, nullptr));
  PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(solution_y, &solution_p));
  PetscCallAbort(PETSC_COMM_WORLD, VecSet(solution_y, 0.0));
  PetscCallAbort(PETSC_COMM_WORLD, VecSet(solution_p, 0.0));
  Vec rhs = nullptr, solution = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatCreateVecs(operators.nest, &solution, &rhs));
  CopyTwoLocalSubvectorsToComposite(rhs_data.rhs_y, rhs_data.rhs_p, rhs);
  PetscCallAbort(PETSC_COMM_WORLD, VecSet(solution, 0.0));

  DistributedMatchingAuxiliaryMatrices matching_matrices;
  DistributedMatchingSchurPCContext matching_context;
  TimeSweepContext y_split_sweep;          // SCHUR_SWEEP and SCHUR_BOUNDARY_REDUCED
  // SCHUR_BOUNDARY_REDUCED: the reduced operator is the caller's or a
  // solve-local one; either way it is prepared for this beta here and is
  // referenced by the split-1 PCSHELL until the outer KSP is destroyed.
  std::unique_ptr<BoundaryReducedSchurOperators> local_reduced;
  const double t_pc0 = MPI_Wtime();
  if (IsBoundaryReduced(policy.pc_mode)) {
    if (config.control.type != ProblemConfig::ControlType::NEUMANN)
      throw std::invalid_argument(
          "SCHUR_BOUNDARY_REDUCED requires Neumann control");
    if (!reduced) {
      local_reduced.reset(new BoundaryReducedSchurOperators(
          config, mass, stiffness, control, tracking));
      reduced = local_reduced.get();
    }
    PetscCallAbort(PETSC_COMM_WORLD,
                   reduced->Prepare(config.alpha_track * config.eta * config.eta /
                                    config.alpha_u));
  }
  // SCHUR_DIRICHLET_REDUCED: solve-local exact Schur operator (responses,
  // Gram blocks) prepared for this beta; referenced by the split-1 PCSHELL
  // until the outer KSP is destroyed.
  std::unique_ptr<DirichletReducedSchurOperators> dirichlet_reduced;
  if (IsDirichletReduced(policy.pc_mode)) {
    dirichlet_reduced.reset(new DirichletReducedSchurOperators(
        config, *dirichlet, mass, stiffness, tracking));
    PetscCallAbort(PETSC_COMM_WORLD,
                   dirichlet_reduced->Prepare(config.alpha_track * config.eta *
                                              config.eta / config.alpha_u));
    // T8D: dense K vs matrix-free action and SPD checks (dense mode only)
    if (config.dirichlet_core.verify && !dirichlet_reduced->iterative_core())
      PetscCallAbort(PETSC_COMM_WORLD, dirichlet_reduced->VerifyCore());
  }
  if (IsMatchingSchur(policy.pc_mode)) {
    matching_matrices = BuildDistributedMatchingAuxiliaryMatrices(
        config, tracking, control, assembled);
    const bool sweeps = (policy.pc_mode == DistributedSolvePc::SCHUR_SWEEP);
    InitializeDistributedMatchingSchurPCContext(
        matching_context, matching_matrices, operators.ay, assembled.ay,
        policy.inner_rtol, policy.legacy_schur_order,
        sweeps ? &config : nullptr, mass, stiffness, control, tracking,
        policy.spatial_rtol);
  }

  InnerIterationCounter y_counter, p_counter;
  KSP outer = ConfigureKktSolver(policy, config, operators, assembled, mass,
                                 matching_context, y_split_sweep,
                                 y_counter, p_counter, reduced,
                                 dirichlet_reduced.get());

  result.timings.pc_setup = MPI_Wtime() - t_pc0;
  const double t_solve0 = MPI_Wtime();
  PetscCallAbort(PETSC_COMM_WORLD, KSPSolve(outer, rhs, solution));
  result.timings.ksp_solve = MPI_Wtime() - t_solve0;
  KSPConvergedReason reason;
  PetscInt outer_iterations = 0;
  PetscCallAbort(PETSC_COMM_WORLD, KSPGetConvergedReason(outer, &reason));
  PetscCallAbort(PETSC_COMM_WORLD,
                 KSPGetIterationNumber(outer, &outer_iterations));
  PetscCallAbort(PETSC_COMM_WORLD,
                 KSPGetResidualNorm(outer, &result.reported_residual));
  result.reason = static_cast<int>(reason);
  result.outer_iterations = static_cast<int>(outer_iterations);
  result.inner_y_iterations = y_counter.iterations;
  result.inner_p_iterations = p_counter.iterations;
  result.inner_y_solves = y_counter.solves;
  result.inner_p_solves = p_counter.solves;
  // Full per-solver statistics.
  result.y_split = y_counter;
  result.p_split = p_counter;
  if (IsMatchingSchur(policy.pc_mode)) {
    if (policy.pc_mode == DistributedSolvePc::SCHUR_SWEEP) {
      result.sweeps_used = true;
      // For the sweep path the counters live in the sweep contexts and count
      // SPATIAL CG solves/iterations, not space-time Krylov iterations.
      result.right_inner = matching_context.right_sweep.counter;
      result.left_inner = matching_context.left_sweep.counter;
      result.ay_inner = matching_context.ay_sweep.counter;
      result.y_split = y_split_sweep.counter;
      result.sweep_failures = matching_context.right_sweep.failures +
                              matching_context.left_sweep.failures +
                              matching_context.ay_sweep.failures +
                              y_split_sweep.failures;
    } else {
      result.right_inner = matching_context.right_counter;
      result.left_inner = matching_context.left_counter;
      result.ay_inner = matching_context.ay_counter;
    }
  } else if (IsBoundaryReduced(policy.pc_mode)) {
    result.sweeps_used = true;
    result.y_split = y_split_sweep.counter;
    result.sweep_failures = y_split_sweep.failures;
    result.reduced.active = true;
    result.reduced.support = reduced->control_support();
    result.reduced.dimension = reduced->reduced_dimension();
    result.reduced.forward_backend = reduced->forward_backend();
    result.reduced.z_backend = reduced->z_backend();
    result.reduced.prepared_beta = reduced->prepared_beta();
    result.reduced.reused_invariant_operator = !local_reduced;
  } else if (IsDirichletReduced(policy.pc_mode)) {
    result.sweeps_used = true;
    result.y_split = y_split_sweep.counter;
    result.sweep_failures = y_split_sweep.failures;
    result.reduced.active = true;
    result.reduced.dirichlet = true;
    result.reduced.support = dirichlet_reduced->control_count();
    result.reduced.dimension = dirichlet_reduced->reduced_dimension();
    result.reduced.forward_backend = dirichlet_reduced->factor_backend();
    result.reduced.z_backend = "woodbury:" + dirichlet_reduced->factor_backend();
    result.reduced.prepared_beta = dirichlet_reduced->prepared_beta();
    result.reduced.reused_invariant_operator = false;
    result.reduced.setup_seconds = dirichlet_reduced->setup_seconds();
    result.reduced.sweep_solves = dirichlet_reduced->sweep_solves();
    if (dirichlet_reduced->iterative_core()) {
      // T8D: inner core statistics (printed only in the new mode)
      const DirichletReducedSchurOperators::CoreStats& st = dirichlet_reduced->core_stats();
      PetscCallAbort(PETSC_COMM_WORLD, PetscPrintf(PETSC_COMM_WORLD,
          "EX21_T8D_CORE_STATS mode=iterative pc=%s rtol=%.1e m=%d core_solves=%lld"
          " core_its_total=%lld core_its_max=%lld core_its_mean=%.2f core_failures=%lld"
          " shell_applications=%lld core_seconds=%.3f shell_seconds=%.3f setup_seconds=%.3f"
          " rt_backend=%s\n",
          config.dirichlet_core.pc == ProblemConfig::DirichletCoreConfig::Pc::RT ? "rt" : "none",
          config.dirichlet_core.rtol, (int)dirichlet_reduced->reduced_dimension(), st.solves,
          st.iterations, st.max_iterations,
          st.solves ? double(st.iterations) / st.solves : 0.0, st.failures, st.applications,
          st.seconds, st.apply_seconds, dirichlet_reduced->setup_seconds(),
          dirichlet->reduced_r_backend().c_str()));
    }
  }
  CopyCompositeToTwoLocalSubvectors(solution, solution_y, solution_p);

  Vec residual = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(rhs, &residual));
  PetscCallAbort(PETSC_COMM_WORLD, MatMult(operators.nest, solution, residual));
  PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(residual, -1.0, rhs));
  PetscCallAbort(PETSC_COMM_WORLD,
                 VecNorm(residual, NORM_2, &result.true_residual));
  {
    // T6 failure guard: see DistributedOptimalControlSolve::accepted.
    PetscReal rtol = 0.0, abstol = 0.0, dtol = 0.0;
    PetscInt maxits = 0;
    PetscCallAbort(PETSC_COMM_WORLD,
                   KSPGetTolerances(outer, &rtol, &abstol, &dtol, &maxits));
    PetscCallAbort(PETSC_COMM_WORLD, VecNorm(rhs, NORM_2, &result.rhs_norm));
    result.acceptance_tolerance =
        kSolveAcceptanceFactor * std::max(static_cast<double>(rtol) * result.rhs_norm,
                                          static_cast<double>(abstol));
    result.status =
        result.reason <= 0 ? "ksp_not_converged"
        : result.sweep_failures > 0 ? "sweep_failures"
        : !(result.true_residual <= result.acceptance_tolerance) ? "true_residual"
        : "ok";
    result.accepted = (result.status == "ok");
    if (!result.accepted) {
      int rank = 0;
      MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
      if (rank == 0) {
        std::printf("EX21_SOLVE_REJECTED status=%s reason=%d outer_its=%d"
                    " sweep_failures=%lld true_residual=%.3e rhs_norm=%.3e"
                    " acceptance_tolerance=%.3e (rtol=%.1e factor=%.0f)\n",
                    result.status.c_str(), result.reason, result.outer_iterations,
                    result.sweep_failures, result.true_residual, result.rhs_norm,
                    result.acceptance_tolerance, static_cast<double>(rtol),
                    kSolveAcceptanceFactor);
        std::fprintf(stderr, "ERROR: ex21 KKT solve REJECTED (%s): reason=%d"
                     " sweep_failures=%lld true_residual=%.3e"
                     " acceptance_tolerance=%.3e\n", result.status.c_str(),
                     result.reason, result.sweep_failures, result.true_residual,
                     result.acceptance_tolerance);
        std::fflush(stdout);
      }
    }
  }

  // Control reconstruction, objective values and KKT residuals from the
  // solved sub-vectors.
  result.fields = ReconstructDistributedDataSide(
      config, layout, tracking, control, operators.ay, solution_y, solution_p,
      dirichlet, lift);

  result.kkt = ComputeKktResiduals(
      config, layout, tracking, control, operators, rhs_data.rhs_y,
      rhs_data.rhs_p, solution_y, solution_p, result.fields.control,
      dirichlet, lift);
  PrintKktResiduals(result.kkt);

  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&residual));
  PetscCallAbort(PETSC_COMM_WORLD, KSPDestroy(&outer));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&solution));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&rhs));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&solution_p));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&solution_y));
  // rhs_data releases its own vectors when the frame exits.
  if (IsMatchingSchur(policy.pc_mode)) {
    if (policy.pc_mode == DistributedSolvePc::SCHUR_SWEEP)
      DestroyTimeSweepContext(y_split_sweep);
    DestroyDistributedMatchingSchurPCContext(matching_context);
    DestroyDistributedMatchingAuxiliaryMatrices(matching_matrices);
  } else if (IsBoundaryReduced(policy.pc_mode) ||
             IsDirichletReduced(policy.pc_mode)) {
    DestroyTimeSweepContext(y_split_sweep);
  }
  // local_reduced / dirichlet_reduced, if any, are released here: after the
  // outer KSP that held their address.
  return result;
}


// ---------------------------------------------------------------------------
// The production solve on the two operator owners.
// ---------------------------------------------------------------------------

DistributedOptimalControlSolve SolveOptimalControl(
    const ProblemConfig& config, const SpatialOperators& spatial,
    SpaceTimeOperators& spacetime, const KktSolverPolicy& policy,
    BoundaryReducedSchurOperators* reduced) {
  DistributedOptimalControlSolve solve = SolveOptimalControlOwnershipDistributed(
      config, spatial.layout, spatial.mass, spatial.tracking, spatial.control,
      spatial.stiffness, spacetime.distributed, spacetime.assembled, policy,
      reduced, spatial.dirichlet.get(), spatial.mass_full, spatial.source,
      &spatial.lift);
  solve.timings.operators = spacetime.elapsed;
  // ex21 inhomogeneous Dirichlet data: the solver's unknown is y - G; the
  // returned state is the physical one.  The adjoint and the control are
  // those of the homogeneous problem (zero on the constrained DOFs).
  if (spatial.lift.g)
    for (unsigned slot = 0; slot < config.time.steps; ++slot)
      CopySpatialToDataSideSlice(spatial.lift.g, slot,
                                 spatial.layout.local_size, 1.0,
                                 solve.fields.state, true);
  return solve;
}
