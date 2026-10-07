// Preconditioner layer: the inner solvers of the [y, p] field split.
//
// Additive mode preconditions each diagonal block with GMRES/GAMG.
//
// Matching-Schur mode replaces the exact Schur complement
//     S = E^T + alpha_track (eta^2/alpha_u) (I (x) M_omega) E^{-1} (I (x) N)
// by the matching approximation
//     Shat = left . E^{-1} . right,   right = E   + sqrt(beta) I (x) N
//                                     left  = E^T + sqrt(beta) I (x) M_omega
//     beta = alpha_track eta^2 / alpha_u,
// applied as Shat^{-1} = right^{-1} E left^{-1}.
//
// Each of E, right and left is block bidiagonal in time, hence exactly
// invertible by substitution; the sweep mode uses that instead of a Krylov
// solve on the whole space-time operator.

#pragma once

#include <petscksp.h>
#include <petscmat.h>

#include "Config.hpp"
#include "SpaceTimeOperators.hpp"

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

struct DistributedMatchingAuxiliaryMatrices {
  Mat right = nullptr;
  Mat left = nullptr;
};

struct InnerIterationCounter {
  long long iterations = 0;
  long long solves = 0;
  // Per-solve maximum: separates "many cheap solves" from "few expensive ones".
  long long max_iterations = 0;
  long long current = 0;
  double mean() const {
    return solves > 0 ? static_cast<double>(iterations) / solves : 0.0;
  }
};

// Wall-clock cost decomposition.
struct SolveTimings {
  double operators = 0.0;
  double pc_setup = 0.0;
  double ksp_solve = 0.0;
  double total = 0.0;
};

struct SpatialSweepMatrices {
  // Backward Euler: A = M/dt + K, and the block coupling is (1/dt) M.
  // Crank-Nicolson: A = A_plus = M/dt + K/2, and the block coupling is the
  // owned matrix A_coupling = -A_minus = M/dt - K/2 with unit scale.
  Mat A = nullptr;
  Mat A_right = nullptr;  // A + sqrt(beta) N
  Mat A_left = nullptr;   // A + sqrt(beta) M_omega
  // Crank-Nicolson only (null under backward Euler):
  Mat A_coupling = nullptr;    // M/dt - K/2
  // The trapezoidal half weight of the LAST tracking slot makes the matching
  // `left` factor slot dependent: A + sqrt(beta) w_n M_omega with w_N = 1/2.
  Mat A_left_final = nullptr;  // A + sqrt(beta)/2 M_omega
  // ex21, convection only (null otherwise).  The `left` factor belongs to
  // E^T, so with a nonsymmetric L its blocks are TRANSPOSED: A_left and
  // A_left_final are then built from A^T, and its Crank-Nicolson coupling is
  // A_coupling^T = M/dt - L^T/2 (backward Euler couples through the symmetric
  // M and needs nothing new).
  Mat A_coupling_transpose = nullptr;
};

enum class TimeSweepDirection { FORWARD, BACKWARD };

/** Stage 8D opt-in temporal-parallel prototypes, applied to every triangular
 *  time sweep in the process.  (0, 0) -- the default, never set on the
 *  production path -- leaves the exact sweep untouched.  At most one may be
 *  non-zero; the command line enforces that. */
void SetTemporalPrototypeOptions(unsigned time_blocks, unsigned sweep_depth);

/** ex21: whether the spatial blocks of the time sweeps may be nonsymmetric
 *  (convection).  false -- the default -- keeps the ex20 CG+GAMG spatial
 *  solves; true switches them to right-preconditioned GMRES+GAMG, because CG
 *  is not valid for a nonsymmetric matrix.  Set once by main(). */
void SetNonsymmetricSpatialSweeps(bool nonsymmetric);

struct TimeSweepContext {
  TimeSweepDirection direction = TimeSweepDirection::FORWARD;
  Mat diagonal = nullptr;      // A, A_R, A_L or A_ZZ (SPD, time independent)
  Mat coupling = nullptr;      // M, M_ZZ or the Crank-Nicolson M/dt - K/2
  // The block coupling is `inverse_dt` * coupling: 1/dt with the backward-Euler
  // mass coupling, 1 with the Crank-Nicolson -A_minus coupling.
  double inverse_dt = 1.0;
  unsigned steps = 0;
  PetscInt spatial_local = 0;
  // The spatial solver on `diagonal`, built ONCE and owned by the context:
  // CG + GAMG (InitializeTimeSweepContext) or an adopted direct solver
  // (InitializeTimeSweepContextWithSpatialSolver).
  KSP spatial = nullptr;
  // Optional SECOND diagonal block, used for the LAST time slot only: the
  // Crank-Nicolson matching `left` factor carries the trapezoidal half weight
  // there.  Null (backward Euler and every other sweep) means one block for
  // every slot, which is the original code path.
  Mat diagonal_last = nullptr;
  KSP spatial_last = nullptr;
  Vec rhs = nullptr, sol = nullptr, tmp = nullptr, prev = nullptr;
  // instrumentation
  InnerIterationCounter counter;
  long long failures = 0;
  const char* name = "sweep";
};

struct DistributedMatchingSchurPCContext {
  DistributedMatchingAuxiliaryMatrices* matrices = nullptr;
  Mat ay_matrix_free = nullptr;
  Mat ay_assembled = nullptr;
  KSP right_solver = nullptr;
  KSP left_solver = nullptr;
  KSP ay_solver = nullptr;
  Vec work_right = nullptr;
  Vec work_ay = nullptr;
  Mat schur_preconditioner = nullptr;
  // Production applies Shat^{-1} = right^{-1} E left^{-1}.  Setting this true
  // swaps the two factors; only the verification driver does so.
  bool legacy_factor_order = false;
  // Instrumentation for the three internal solvers, and the optional exact
  // triangular-sweep replacements for them.
  InnerIterationCounter right_counter, left_counter, ay_counter;
  bool use_sweeps = false;
  SpatialSweepMatrices sweep_matrices;
  TimeSweepContext right_sweep, left_sweep, ay_sweep;
};

// ---------------------------------------------------------------------------
// Additive fieldsplit inner solver
// ---------------------------------------------------------------------------

void ConfigureFieldSplitInnerGmresGamg(KSP solver, Mat matrix_free_operator,
                                       Mat assembled_preconditioner,
                                       double tolerance = 1.0e-6);

// ---------------------------------------------------------------------------
// Inner-iteration instrumentation (KSPMonitor callback)
// ---------------------------------------------------------------------------

PetscErrorCode CountInnerIterations(KSP, PetscInt iteration, PetscReal,
                                    void* raw_context);

// ---------------------------------------------------------------------------
// Matching-Schur auxiliary operators, spatial sweep matrices, time sweeps and
// the matching-Schur preconditioner context
// ---------------------------------------------------------------------------

DistributedMatchingAuxiliaryMatrices
BuildDistributedMatchingAuxiliaryMatrices(
    const ProblemConfig& config, Mat tracking, Mat control,
    const AssembledSpaceTimeOperators& assembled);

void DestroyDistributedMatchingAuxiliaryMatrices(
    DistributedMatchingAuxiliaryMatrices& matrices);

SpatialSweepMatrices BuildSpatialSweepMatrices(const ProblemConfig& config,
                                               Mat mass, Mat stiffness,
                                               Mat control, Mat tracking);

void DestroySpatialSweepMatrices(SpatialSweepMatrices& m);

/** The exact triangular sweep on `context`; ApplyTimeSweep is its PCSHELL
 *  entry point. */
PetscErrorCode RunTimeSweep(TimeSweepContext& context, Vec input, Vec output);

PetscErrorCode ApplyTimeSweep(PC preconditioner, Vec input, Vec output);

/** `coupling_scale` 0 selects the backward-Euler 1/dt; pass 1.0 with the
 *  Crank-Nicolson coupling M/dt - K/2.  `diagonal_last`, when non-null, is the
 *  diagonal block of the LAST time slot (its own spatial solver is built and
 *  owned by the context). */
void InitializeTimeSweepContext(TimeSweepContext& context,
                                TimeSweepDirection direction, Mat diagonal,
                                Mat coupling, unsigned steps, double dt,
                                double spatial_tolerance, const char* prefix,
                                const char* name,
                                double coupling_scale = 0.0,
                                Mat diagonal_last = nullptr);

/** Same context, but with a caller-created spatial solver for `diagonal`
 *  (already configured; set up here).  OWNERSHIP: the context adopts `spatial`
 *  and DestroyTimeSweepContext releases it; the caller must not destroy it.
 *  `coupling_scale` follows InitializeTimeSweepContext: 0 selects 1/dt. */
void InitializeTimeSweepContextWithSpatialSolver(
    TimeSweepContext& context, TimeSweepDirection direction, Mat diagonal,
    Mat coupling, unsigned steps, double dt, const char* name, KSP spatial,
    double coupling_scale = 0.0);

void DestroyTimeSweepContext(TimeSweepContext& context);

PetscErrorCode ApplyDistributedMatchingSchurPreconditioner(
    PC preconditioner, Vec input, Vec output);

void InitializeDistributedMatchingSchurPCContext(
    DistributedMatchingSchurPCContext& context,
    DistributedMatchingAuxiliaryMatrices& matrices, Mat ay_matrix_free,
    Mat ay_assembled, double inner_tolerance,
    bool legacy_factor_order = false,
    // Non-null sweep_config replaces the three whole-space-time GMRES/GAMG
    // solves by exact triangular time sweeps.
    const ProblemConfig* sweep_config = nullptr, Mat mass = nullptr,
    Mat stiffness = nullptr, Mat control = nullptr, Mat tracking = nullptr,
    double spatial_tolerance = 1.0e-10);

void DestroyDistributedMatchingSchurPCContext(
    DistributedMatchingSchurPCContext& context);
