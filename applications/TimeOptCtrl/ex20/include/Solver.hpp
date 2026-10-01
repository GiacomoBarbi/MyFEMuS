// Solver layer: right-hand side, control reconstruction, KKT residuals and the
// outer all-at-once solve.
//
// Unknown ordering: [y, p], each of length steps * spatial_dofs, laid out
// rank-major (every rank owns all time levels of its own spatial DOFs).
//
//   [  E                     (eta^2/alpha_u) I (x) N ] [y]   [ rhs_y ]
//   [ -alpha_track I (x) M_omega          E^T        ] [p] = [ rhs_p ]
//
//   E   = blockdiag(M/dt + K) - blocksubdiag(M/dt)
//   E^T = blockdiag(M/dt + K) - blocksuperdiag(M/dt)
//
// with backward Euler.  With Crank-Nicolson (distributed control only) the
// state operator is E_CN = blockdiag(M/dt + K/2) + blocksubdiag(-M/dt + K/2)
// and the (2,1) block acquires the trapezoidal slot weights, i.e. the system is
//
//   [  E_CN                        (eta^2/alpha_u) I (x) N ] [y]   [ rhs_y ]
//   [ -alpha_track W (x) M_omega              E_CN^T       ] [p] = [ rhs_p ]
//   W = diag(1, ..., 1, 1/2)       (W = I for backward Euler).
//
// p is the dt-SCALED Lagrange multiplier P_Lagrange/dt under both schemes --
// that is what removes dt from the two off-diagonal blocks and makes the
// control reconstruction u = -(eta/alpha_u) p scheme independent.  With the
// unscaled multiplier the same Crank-Nicolson system reads
//   [ E_CN  (eta^2/(alpha_u dt)) I(x)N ; -alpha_t (W_y (x) M_omega)  E_CN^T ]
// with W_y = dt W; see EX20_CRANK_NICOLSON_STAGE1_DISTRIBUTED.md.
//
//   N        control mass matrix: the box mass matrix for distributed control,
//            the trace mass matrix M_Gamma for Neumann control
//   M_omega  tracking mass matrix
//
// The solver is FGMRES on this 2x2 system with a y/p field split.  The Schur
// modes eliminate the y block; the additive mode preconditions y and p
// independently.  The control is not a solver unknown: it is reconstructed
// algebraically from p as u = -(eta/alpha_u) p on the active set.
//
// The solve returns a DistributedOptimalControlSolve, which owns its vectors.

#pragma once

#include <petscksp.h>
#include <petscmat.h>

#include <string>

#include "Config.hpp"
#include "Discretization.hpp"
#include "Preconditioners.hpp"
#include "SpaceTimeOperators.hpp"

class BoundaryReducedSchurOperators;

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/** The data-side vectors and the objective values.
 *
 *  BuildDistributedRightHandSide fills rhs_y / rhs_p; ReconstructDistributed-
 *  DataSide fills state / adjoint / control and J_*.  Unused handles stay
 *  null.  Owns the vectors: the destructor releases them, exactly once.  No
 *  PETSc object stores this struct's address, so it may be moved (the handles
 *  travel and the source is emptied); copying is deleted because a copy would
 *  destroy the same vectors twice. */
struct DistributedDataSideResult {
  DistributedDataSideResult() = default;
  ~DistributedDataSideResult();
  DistributedDataSideResult(DistributedDataSideResult&& other);
  DistributedDataSideResult& operator=(DistributedDataSideResult&& other);
  DistributedDataSideResult(const DistributedDataSideResult&) = delete;
  DistributedDataSideResult& operator=(const DistributedDataSideResult&) = delete;

  Vec state = nullptr;
  Vec adjoint = nullptr;
  Vec control = nullptr;
  Vec rhs_y = nullptr;
  Vec rhs_p = nullptr;
  // J_track / J_reg carry the backward-Euler quadrature weight dt; the
  // *_unweighted members are the same sums without it.
  double J_track = 0.0;
  double J_reg = 0.0;
  double J_track_unweighted = 0.0;
  double J_reg_unweighted = 0.0;
  // The temporal part of J_reg, reported separately:
  //   J_reg_time = (gamma_u/(2 dt)) sum_{n=1}^{N} (u_n-u_{n-1})^T M_Gamma (u_n-u_{n-1}),
  // including the constant (gamma_u/(2 dt)) u_0^T M_Gamma u_0 that the DATA
  // u_0 = y_0|_Gamma_c/eta contributes through the n = 1 difference.  It is
  // already contained in J_reg; 0 unless gamma_u > 0.
  double J_reg_time = 0.0;
};

// Optimality-system residuals; see ComputeKktResiduals for the definitions.
struct KktResidualReport {
  double state_abs = 0.0;
  double state_rel = 0.0;
  double adjoint_abs = 0.0;
  double adjoint_rel = 0.0;
  double stationarity_abs = 0.0;
  double stationarity_rel = 0.0;
  double block_y_abs = 0.0;
  double block_p_abs = 0.0;
  double reduced_abs = 0.0;
  double control_offactive_max = 0.0;
  long long control_offactive_dofs = 0;
  long long active_dofs = 0;
  long long geometric_mask_dofs = 0;
};

/** Metadata of the reduced Schur solves (pc modes SCHUR_BOUNDARY_REDUCED and
 *  SCHUR_DIRICHLET_REDUCED); `active` is false for every other mode and the
 *  remaining members are then meaningless. */
struct BoundaryReducedSchurStats {
  bool active = false;
  bool dirichlet = false;    // SCHUR_DIRICHLET_REDUCED (else SCHUR_BOUNDARY_REDUCED)
  PetscInt support = 0;      // |R|, the control-active spatial DOFs
  PetscInt dimension = 0;    // steps * |R|, the dense reduced dimension
  std::string forward_backend;
  std::string z_backend;
  double prepared_beta = 0.0;   // alpha_track eta^2 / alpha_u used by Prepare
  // true: the caller supplied an existing, compatible operator (its S0/S1
  // and spatial factors were reused); false: built for this solve.
  bool reused_invariant_operator = false;
  // SCHUR_DIRICHLET_REDUCED only: wall clock of the operator construction
  // (responses, Gram blocks) and the direct spatial solves of its sweeps.
  double setup_seconds = 0.0;
  long long sweep_solves = 0;
};

/** The solved KKT system: state, adjoint and reconstructed control (owned
 *  through `fields`), the convergence reason, the iteration counters and the
 *  KKT residuals.  Movable, not copyable, like `fields`. */
struct DistributedOptimalControlSolve {
  int reason = 0;
  int outer_iterations = 0;
  double reported_residual = 0.0;
  double true_residual = 0.0;
  long long inner_y_iterations = 0;
  long long inner_p_iterations = 0;
  long long inner_y_solves = 0;
  long long inner_p_solves = 0;
  DistributedDataSideResult fields;
  KktResidualReport kkt;
  // Cost decomposition and per-internal-solver statistics.
  SolveTimings timings;
  InnerIterationCounter y_split, p_split, right_inner, left_inner, ay_inner;
  bool sweeps_used = false;
  long long sweep_failures = 0;
  BoundaryReducedSchurStats reduced;
};

// ---------------------------------------------------------------------------
// Solver policy
//
// What the application chooses; ConfigureKktSolver in Solver.cpp turns it into
// PETSc objects.  The defaults below ARE the production settings.
//
//   unknowns      [y, p]
//   outer         FGMRES on the 2x2 MatNest
//   block PC      PCFIELDSPLIT
//   split type    ADDITIVE_GAMG -> additive, y and p preconditioned separately
//                 SCHUR_*       -> Schur, eliminating the y block
//   Schur factor  FULL
//   Schur approx  matching, applied as a PCSHELL
//   E solve       GMRES/GAMG, or an exact time sweep in SCHUR_SWEEP
//   spatial solve CG/GAMG inside each time sweep
// ---------------------------------------------------------------------------

struct KktSolverPolicy {
  // Field-split composition and inner-solver choice.
  DistributedSolvePc pc_mode = DistributedSolvePc::ADDITIVE_GAMG;

  // Outer FGMRES on the whole KKT system.
  double outer_rtol = 1.0e-8;
  unsigned outer_maxits = 200;
  unsigned outer_restart = 30;

  // Krylov tolerance of each field-split inner solve.
  double inner_rtol = 1.0e-6;

  // CG tolerance of each spatial solve inside a time sweep.
  double spatial_rtol = 1.0e-10;

  // Swaps the two matching-Schur factors; verification driver only.
  bool legacy_schur_order = false;
};

/** The production policy.  The three arguments are the only values any driver
 *  varies; everything else takes the default above. */
inline KktSolverPolicy MakeKktSolverPolicy(
    DistributedSolvePc pc_mode, double spatial_rtol = 1.0e-10,
    bool legacy_schur_order = false) {
  KktSolverPolicy policy;
  policy.pc_mode = pc_mode;
  policy.spatial_rtol = spatial_rtol;
  policy.legacy_schur_order = legacy_schur_order;
  return policy;
}

// ---------------------------------------------------------------------------
// Data side, control reconstruction and the outer solve
// ---------------------------------------------------------------------------

/** The right-hand side of the reduced KKT system.
 *
 *  Backward Euler:
 *      rhs_y = M y_0 / dt (first slot) + M f(t_k)
 *      rhs_p = -alpha_track M_omega y_d(t_k).
 *  Crank-Nicolson (distributed control only):
 *      rhs_y = (M/dt - K/2) y_0 (first slot) + (M/2)(f(t_{k-1}) + f(t_k))
 *      rhs_p = -alpha_track w_k M_omega y_d(t_k),  w = (1, ..., 1, 1/2),
 *  so `stiffness` must be supplied whenever `config` selects Crank-Nicolson. */
DistributedDataSideResult BuildDistributedRightHandSide(
    const ProblemConfig& config, const OwnedSpatialLayout& layout,
    Mat mass, Mat tracking, Mat space_time_layout, Mat stiffness = nullptr);

/** `stiffness` builds A = M/dt + K for the time sweeps; it may be null when the
 *  policy selects a mode that does not sweep.  `reduced` (SCHUR_BOUNDARY_REDUCED
 *  only, null otherwise) is a caller-owned reduced operator built for the same
 *  spatial operators, `steps` and `dt`; it is prepared for the configured beta
 *  and reused, not destroyed.  Null selects a solve-local one.
 *  `dirichlet` and `mass_full` (Dirichlet control only, null otherwise): the
 *  boundary control operators and the unhatted mass matrix that forms the
 *  data terms of the state right-hand side (initial state, source) on the free
 *  rows; both borrowed.  Dirichlet control is solved with ADDITIVE_GAMG or
 *  SCHUR_DIRICHLET_REDUCED (a solve-local exact Schur operator). */
DistributedOptimalControlSolve SolveOptimalControlOwnershipDistributed(
    const ProblemConfig& config, const OwnedSpatialLayout& layout,
    Mat mass, Mat tracking, Mat control, Mat stiffness,
    DistributedSpaceTimeOperators& operators,
    AssembledSpaceTimeOperators& assembled,
    const KktSolverPolicy& policy = KktSolverPolicy(),
    BoundaryReducedSchurOperators* reduced = nullptr,
    DirichletControlOperators* dirichlet = nullptr, Mat mass_full = nullptr);

// ---------------------------------------------------------------------------
// The production pipeline, in the order every driver runs it:
//
//   SpatialOperators        Q2 finite-element operators on the spatial mesh
//   SpaceTimeOperators      the all-at-once blocks built from them
//   SolveOptimalControl     the solved KKT system, its fields and diagnostics
//
// Each type owns what it builds and releases it in the order the solver
// requires, so a driver declares them as locals and nothing else.
// ---------------------------------------------------------------------------

/** Solve the reduced KKT system on the operators above.  The returned solve
 *  owns its vectors; `timings.operators` is the operator-phase wall clock. */
DistributedOptimalControlSolve SolveOptimalControl(
    const ProblemConfig& config, const SpatialOperators& spatial,
    SpaceTimeOperators& spacetime, const KktSolverPolicy& policy,
    BoundaryReducedSchurOperators* reduced = nullptr);
