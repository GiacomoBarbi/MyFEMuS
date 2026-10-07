// Preconditioner layer.

#include "Preconditioners.hpp"

#include <petscksp.h>

#include <cmath>
#include <vector>

#include "Config.hpp"
#include "SpaceTimeOperators.hpp"

void ConfigureFieldSplitInnerGmresGamg(KSP solver, Mat matrix_free_operator,
                                       Mat assembled_preconditioner,
                                       double tolerance) {
  PetscCallAbort(PETSC_COMM_WORLD,
                 KSPSetOperators(solver, matrix_free_operator,
                                 assembled_preconditioner));
  PetscCallAbort(PETSC_COMM_WORLD, KSPSetType(solver, KSPGMRES));
  PetscCallAbort(PETSC_COMM_WORLD, KSPGMRESSetRestart(solver, 30));
  PC preconditioner = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, KSPGetPC(solver, &preconditioner));
  PetscCallAbort(PETSC_COMM_WORLD, PCSetType(preconditioner, PCGAMG));
  PetscCallAbort(PETSC_COMM_WORLD,
                 KSPSetTolerances(solver, tolerance, PETSC_DEFAULT,
                                  PETSC_DEFAULT, 100));
  PetscCallAbort(PETSC_COMM_WORLD,
                 KSPSetInitialGuessNonzero(solver, PETSC_FALSE));
  PetscCallAbort(PETSC_COMM_WORLD, KSPSetFromOptions(solver));
  PetscCallAbort(PETSC_COMM_WORLD, KSPSetUp(solver));
}

/** space_time += (W (x) spatial), W = diag(scale, ..., scale, final_scale).
 *  `final_scale` equals `scale` for every backward-Euler use; the
 *  Crank-Nicolson matching `left` factor gives the last slot the trapezoidal
 *  half weight. */
static void AddDistributedSpatialBlockDiagonal(Mat space_time, Mat spatial,
                                        unsigned steps, double scale,
                                        double final_scale) {
  PetscInt spatial_begin = 0, spatial_end = 0;
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatGetOwnershipRange(spatial, &spatial_begin, &spatial_end));
  const PetscInt spatial_local = spatial_end - spatial_begin;
  const PetscInt* spatial_ranges = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatGetOwnershipRanges(spatial, &spatial_ranges));
  int rank = 0, ranks = 0;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  MPI_Comm_size(PETSC_COMM_WORLD, &ranks);
  const PetscInt space_time_begin =
      static_cast<PetscInt>(steps) * spatial_ranges[rank];
  for (unsigned slot = 0; slot < steps; ++slot) {
    for (PetscInt spatial_row = spatial_begin; spatial_row < spatial_end;
         ++spatial_row) {
      PetscInt count = 0;
      const PetscInt* spatial_columns = nullptr;
      const PetscScalar* spatial_values = nullptr;
      PetscCallAbort(PETSC_COMM_WORLD,
                     MatGetRow(spatial, spatial_row, &count, &spatial_columns,
                               &spatial_values));
      std::vector<PetscInt> columns(count);
      std::vector<PetscScalar> values(count);
      const double slot_scale =
          (slot + 1u == steps) ? final_scale : scale;
      for (PetscInt entry = 0; entry < count; ++entry) {
        columns[entry] = OwnershipPermutedSpaceTimeIndex(
            spatial_columns[entry], slot, steps, spatial_ranges, ranks);
        values[entry] = slot_scale * spatial_values[entry];
      }
      const PetscInt row = space_time_begin +
          static_cast<PetscInt>(slot) * spatial_local +
          spatial_row - spatial_begin;
      if (count)
        PetscCallAbort(PETSC_COMM_WORLD,
                       MatSetValues(space_time, 1, &row, count, columns.data(),
                                    values.data(), ADD_VALUES));
      PetscCallAbort(PETSC_COMM_WORLD,
                     MatRestoreRow(spatial, spatial_row, &count,
                                   &spatial_columns, &spatial_values));
    }
  }
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatAssemblyBegin(space_time, MAT_FINAL_ASSEMBLY));
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatAssemblyEnd(space_time, MAT_FINAL_ASSEMBLY));
}

DistributedMatchingAuxiliaryMatrices
BuildDistributedMatchingAuxiliaryMatrices(
    const ProblemConfig& config, Mat tracking, Mat control,
    const AssembledSpaceTimeOperators& assembled) {
  DistributedMatchingAuxiliaryMatrices matrices;
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatDuplicate(assembled.ay, MAT_COPY_VALUES, &matrices.right));
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatDuplicate(assembled.ap, MAT_COPY_VALUES, &matrices.left));
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatSetOption(matrices.right, MAT_NEW_NONZERO_ALLOCATION_ERR,
                              PETSC_TRUE));
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatSetOption(matrices.left, MAT_NEW_NONZERO_ALLOCATION_ERR,
                              PETSC_TRUE));
  // The reduced 2x2 in ex20's dt-scaled-adjoint convention is
  //     [ E                 (eta^2/alpha_u) (I (x) N)          ]
  //     [ -alpha_t (W (x) M_omega)        E^T                  ]
  // with W = I (backward Euler) or diag(1,...,1,1/2) (Crank-Nicolson).  Its
  // Schur complement is  S = E^T + beta (W (x) M_omega) E^{-1} (I (x) N)  with
  // the SAME beta = alpha_t eta^2/alpha_u under both schemes; the slot weight
  // w_n multiplies the M_omega factor only, so the matching split that
  // reproduces  beta w_n  is  right = E + sqrt(beta) (I (x) N)  (slot
  // independent) and  left = E^T + sqrt(beta) (W (x) M_omega).
  const double beta = config.alpha_track * config.eta * config.eta /
                      config.alpha_u;
  const double root_beta = std::sqrt(beta);
  AddDistributedSpatialBlockDiagonal(matrices.right, control,
                                     config.time.steps, root_beta, root_beta);
  AddDistributedSpatialBlockDiagonal(
      matrices.left, tracking, config.time.steps, root_beta,
      root_beta * TrackingSlotWeight(config, config.time.steps - 1u));
  return matrices;
}

void DestroyDistributedMatchingAuxiliaryMatrices(
    DistributedMatchingAuxiliaryMatrices& matrices) {
  PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&matrices.left));
  PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&matrices.right));
}


PetscErrorCode CountInnerIterations(KSP, PetscInt iteration, PetscReal,
                                    void* raw_context) {
  InnerIterationCounter* context =
      static_cast<InnerIterationCounter*>(raw_context);
  if (iteration == 0) {
    ++context->solves;
    context->current = 0;
  } else {
    ++context->iterations;
    ++context->current;
    if (context->current > context->max_iterations)
      context->max_iterations = context->current;
  }
  return PETSC_SUCCESS;
}

// ===========================================================================
// EXACT TRIANGULAR TIME SWEEPS
//
// The space-time operators are block bidiagonal:
//
//   E      = blockdiag(A) - blocksubdiag(B)          A = M/dt + K,  B = M/dt
//   E^T    = blockdiag(A) - blocksuperdiag(B)
//   right  = E   + sqrt(beta) (I_N (x) N)        diagonal blocks A_R = A + sqrt(beta) N
//   left   = E^T + sqrt(beta) (I_N (x) M_omega)  diagonal blocks A_L = A + sqrt(beta) M_omega
//
// so each of the four systems is EXACTLY solvable by substitution in time:
//
//   forward  (E, right):   A_* z_1 = r_1 ;  A_* z_k = r_k + B z_{k-1}, k = 2..N
//   backward (E^T, left):  A_* z_N = r_N ;  A_* z_k = r_k + B z_{k+1}, k = N-1..1
//
// Crank-Nicolson keeps this kernel unchanged and only changes the two blocks:
// A = A_plus = M/dt + K/2 and B = -A_minus = M/dt - K/2 with unit scale
// (instead of A = M/dt + K, B = M, scale 1/dt).  Its `left` factor carries the
// trapezoidal half weight on the last slot, which is the one place where the
// diagonal block is slot dependent (TimeSweepContext::diagonal_last).
//
// A is SPD (M SPD, K SPSD, dt > 0); N and M_omega are symmetric PSD; hence
// A_R and A_L are SPD as well, so every spatial solve is SPD and CG+GAMG
// applies.  The spatial matrix is time-independent, so ONE KSP/GAMG hierarchy
// is built and reused across all N time levels.
//
// Because each rank owns all N time levels of its own spatial DOFs, the sweep
// is sequential in time but needs no time-direction communication; all
// parallelism stays inside the spatial solves.
// ===========================================================================

/** A = (1/dt) M + K, and the two matching variants A_R, A_L.  Ownership of
 *  the returned matrices passes to the caller. */
SpatialSweepMatrices BuildSpatialSweepMatrices(const ProblemConfig& config,
                                               Mat mass, Mat stiffness,
                                               Mat control, Mat tracking) {
  SpatialSweepMatrices out;
  const bool crank_nicolson = IsCrankNicolson(config);
  PetscCallAbort(PETSC_COMM_WORLD, MatDuplicate(mass, MAT_COPY_VALUES, &out.A));
  PetscCallAbort(PETSC_COMM_WORLD, MatScale(out.A, 1.0 / config.time.dt));
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatAXPY(out.A, crank_nicolson ? 0.5 : 1.0, stiffness,
                         SAME_NONZERO_PATTERN));
  if (crank_nicolson) {
    // -A_minus = M/dt - K/2, the block coupling of every Crank-Nicolson sweep.
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatDuplicate(mass, MAT_COPY_VALUES, &out.A_coupling));
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatScale(out.A_coupling, 1.0 / config.time.dt));
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatAXPY(out.A_coupling, -0.5, stiffness,
                           SAME_NONZERO_PATTERN));
  }
  const double beta = config.alpha_track * config.eta * config.eta /
                      config.alpha_u;
  const double root_beta = std::sqrt(beta);
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatDuplicate(out.A, MAT_COPY_VALUES, &out.A_right));
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatAXPY(out.A_right, root_beta, control, SUBSET_NONZERO_PATTERN));
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatDuplicate(out.A, MAT_COPY_VALUES, &out.A_left));
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatAXPY(out.A_left, root_beta, tracking, SUBSET_NONZERO_PATTERN));
  if (crank_nicolson) {
    // The last tracking slot carries the trapezoidal half weight.
    const double final_weight =
        TrackingSlotWeight(config, config.time.steps - 1u);
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatDuplicate(out.A, MAT_COPY_VALUES, &out.A_left_final));
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatAXPY(out.A_left_final, root_beta * final_weight, tracking,
                           SUBSET_NONZERO_PATTERN));
  }
  if (HasConvection(config)) {
    // ex21: `left` is the E^T factor.  Rebuild its diagonal blocks from A^T
    // (tracking is symmetric, so adding it after the transpose is the same as
    // transposing A + sqrt(beta) M_omega) and transpose the CN coupling.  The
    // SPD/symmetric flags below are NOT set: L is not symmetric.
    const double final_weight =
        TrackingSlotWeight(config, config.time.steps - 1u);
    PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&out.A_left));
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatTranspose(out.A, MAT_INITIAL_MATRIX, &out.A_left));
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatAXPY(out.A_left, root_beta, tracking,
                           SUBSET_NONZERO_PATTERN));
    if (out.A_left_final) {
      PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&out.A_left_final));
      PetscCallAbort(PETSC_COMM_WORLD,
                     MatTranspose(out.A, MAT_INITIAL_MATRIX, &out.A_left_final));
      PetscCallAbort(PETSC_COMM_WORLD,
                     MatAXPY(out.A_left_final, root_beta * final_weight,
                             tracking, SUBSET_NONZERO_PATTERN));
    }
    if (out.A_coupling)
      PetscCallAbort(PETSC_COMM_WORLD,
                     MatTranspose(out.A_coupling, MAT_INITIAL_MATRIX,
                                  &out.A_coupling_transpose));
    return out;
  }
  for (Mat m : {out.A, out.A_right, out.A_left}) {
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatSetOption(m, MAT_SYMMETRIC, PETSC_TRUE));
    PetscCallAbort(PETSC_COMM_WORLD, MatSetOption(m, MAT_SPD, PETSC_TRUE));
  }
  if (out.A_left_final) {
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatSetOption(out.A_left_final, MAT_SYMMETRIC, PETSC_TRUE));
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatSetOption(out.A_left_final, MAT_SPD, PETSC_TRUE));
  }
  if (out.A_coupling)
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatSetOption(out.A_coupling, MAT_SYMMETRIC, PETSC_TRUE));
  return out;
}

void DestroySpatialSweepMatrices(SpatialSweepMatrices& m) {
  PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&m.A_coupling_transpose));
  PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&m.A_left_final));
  PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&m.A_coupling));
  PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&m.A_left));
  PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&m.A_right));
  PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&m.A));
}

// ---------------------------------------------------------------------------
// Named PETSc log events.  Purely diagnostic: PetscLogEventBegin/End are
// no-ops unless PETSc logging is switched on (-log_view), they touch no
// algebra, no tolerance and add no synchronisation.  Note that
// EX20_SPATIAL_SOLVE is NESTED inside the two sweep events, so the sweep
// times INCLUDE the spatial solves and the two must not be added together.
// ---------------------------------------------------------------------------
static PetscLogEvent kEventSweepForward = 0;
static PetscLogEvent kEventSweepBackward = 0;
static PetscLogEvent kEventSpatialSolve = 0;

static void RegisterEx20SweepEvents() {
  static bool registered = false;
  if (registered) return;
  registered = true;
  PetscLogEventRegister("EX20_E_FORWARD", PC_CLASSID, &kEventSweepForward);
  PetscLogEventRegister("EX20_ET_BACKWARD", PC_CLASSID, &kEventSweepBackward);
  PetscLogEventRegister("EX20_SPATIAL_SOLVE", KSP_CLASSID, &kEventSpatialSolve);
}

// --- Stage 8D temporal-parallel prototypes -------------------------------
//
// Process-wide knobs, both 0 (disabled) unless the command line asks for them.
// They are read ONLY by RunTimeSweep, and only to make the triangular sweep
// deliberately inexact; with both at 0 the production loop below is executed
// unchanged, instruction for instruction.  A single pair of statics keeps
// every sweep context (matching Schur, boundary-reduced, Dirichlet-reduced)
// consistent without touching any existing constructor signature.
static unsigned g_pc_time_blocks = 0;
static unsigned g_pc_sweep_depth = 0;

void SetTemporalPrototypeOptions(unsigned time_blocks, unsigned sweep_depth) {
  g_pc_time_blocks = time_blocks;
  g_pc_sweep_depth = sweep_depth;
}

// ex21: false (the ex20 behaviour) unless main() reports a convection run.
static bool g_nonsymmetric_spatial_sweeps = false;

void SetNonsymmetricSpatialSweeps(bool nonsymmetric) {
  g_nonsymmetric_spatial_sweeps = nonsymmetric;
}

/** Length of the contiguous temporal blocks when block Jacobi is on, 0 when it
 *  is off.  The last block absorbs the remainder, so P blocks of length
 *  ceil(N_t/P) cover N_t slots with at most P-1 boundaries dropped. */
static unsigned TemporalBlockLength(unsigned steps) {
  if (g_pc_time_blocks <= 1u || steps == 0u) return 0u;
  const unsigned blocks = g_pc_time_blocks < steps ? g_pc_time_blocks : steps;
  return (steps + blocks - 1u) / blocks;
}

/** Windowed (truncated-Neumann) sweep: slot i is recomputed on its own window
 *  of at most `depth` predecessors, started from zero initial data.  The
 *  windows overlap and are mutually independent, so the serial depth is
 *  `depth` for every slot while the work grows to ~ depth * N_t solves.
 *
 *  This is NOT the same operator as block Jacobi with blocks of length
 *  `depth`: there, a slot sees only as far back as its own block start, so its
 *  effective lookback varies between 1 and `depth` across the block.  Here
 *  every slot sees exactly `depth` (except the first few, which see all there
 *  is, and are therefore exact).  depth >= N_t reproduces the exact sweep. */
static PetscErrorCode RunWindowedTimeSweep(TimeSweepContext& context, Vec input,
                                           Vec output, unsigned depth) {
  TimeSweepContext* c = &context;
  const bool forward = (c->direction == TimeSweepDirection::FORWARD);
  const PetscLogEvent sweep_event =
      forward ? kEventSweepForward : kEventSweepBackward;
  PetscCall(PetscLogEventBegin(sweep_event, 0, 0, 0, 0));
  for (unsigned i = 0; i < c->steps; ++i) {
    const unsigned first = (i + 1u > depth) ? (i + 1u - depth) : 0u;
    for (unsigned j = first; j <= i; ++j) {
      const unsigned slot = forward ? j : (c->steps - 1u - j);
      CopyDataSideSliceToSpatial(input, slot, c->spatial_local, c->rhs);
      if (j > first) {
        PetscCall(MatMult(c->coupling, c->prev, c->tmp));
        PetscCall(VecAXPY(c->rhs, c->inverse_dt, c->tmp));
      }
      KSP spatial_solver =
          (c->spatial_last && slot + 1u == c->steps) ? c->spatial_last
                                                     : c->spatial;
      PetscCall(VecSet(c->sol, 0.0));
      PetscCall(PetscLogEventBegin(kEventSpatialSolve, 0, 0, 0, 0));
      PetscCall(KSPSolve(spatial_solver, c->rhs, c->sol));
      PetscCall(PetscLogEventEnd(kEventSpatialSolve, 0, 0, 0, 0));
      PetscInt its = 0;
      KSPConvergedReason reason;
      PetscCall(KSPGetIterationNumber(spatial_solver, &its));
      PetscCall(KSPGetConvergedReason(spatial_solver, &reason));
      ++c->counter.solves;
      c->counter.iterations += its;
      if (its > c->counter.max_iterations) c->counter.max_iterations = its;
      if (reason < 0) ++c->failures;
      PetscCall(VecCopy(c->sol, c->prev));
    }
    // Only the window's last solve is a slot of the result.
    const unsigned slot = forward ? i : (c->steps - 1u - i);
    CopySpatialToDataSideSlice(c->sol, slot, c->spatial_local, 1.0, output,
                               false);
  }
  PetscCall(PetscLogEventEnd(sweep_event, 0, 0, 0, 0));
  return PETSC_SUCCESS;
}

PetscErrorCode RunTimeSweep(TimeSweepContext& context, Vec input, Vec output) {
  TimeSweepContext* c = &context;
  const bool forward = (c->direction == TimeSweepDirection::FORWARD);
  RegisterEx20SweepEvents();
  const PetscLogEvent sweep_event =
      forward ? kEventSweepForward : kEventSweepBackward;
  if (g_pc_sweep_depth > 0u)
    return RunWindowedTimeSweep(context, input, output, g_pc_sweep_depth);
  // 0 unless temporal block Jacobi was requested: then the coupling into the
  // first slot of every block is dropped, which is what makes the blocks
  // independent.  The arithmetic of every other slot is untouched.
  const unsigned block_length = TemporalBlockLength(c->steps);
  PetscCall(PetscLogEventBegin(sweep_event, 0, 0, 0, 0));
  for (unsigned i = 0; i < c->steps; ++i) {
    const unsigned slot = forward ? i : (c->steps - 1u - i);
    const bool block_start = (block_length > 0u) && (i % block_length == 0u);
    CopyDataSideSliceToSpatial(input, slot, c->spatial_local, c->rhs);
    if (i > 0 && !block_start) {
      PetscCall(MatMult(c->coupling, c->prev, c->tmp));
      PetscCall(VecAXPY(c->rhs, c->inverse_dt, c->tmp));
    }
    // One diagonal block per slot, except that a context carrying
    // `spatial_last` uses it on the final slot (Crank-Nicolson matching
    // `left`, whose last tracking weight is 1/2).
    KSP spatial_solver =
        (c->spatial_last && slot + 1u == c->steps) ? c->spatial_last
                                                   : c->spatial;
    PetscCall(VecSet(c->sol, 0.0));
    PetscCall(PetscLogEventBegin(kEventSpatialSolve, 0, 0, 0, 0));
    PetscCall(KSPSolve(spatial_solver, c->rhs, c->sol));
    PetscCall(PetscLogEventEnd(kEventSpatialSolve, 0, 0, 0, 0));
    PetscInt its = 0;
    KSPConvergedReason reason;
    PetscCall(KSPGetIterationNumber(spatial_solver, &its));
    PetscCall(KSPGetConvergedReason(spatial_solver, &reason));
    ++c->counter.solves;
    c->counter.iterations += its;
    if (its > c->counter.max_iterations) c->counter.max_iterations = its;
    if (reason < 0) ++c->failures;
    CopySpatialToDataSideSlice(c->sol, slot, c->spatial_local, 1.0, output,
                               false);
    PetscCall(VecCopy(c->sol, c->prev));
  }
  PetscCall(PetscLogEventEnd(sweep_event, 0, 0, 0, 0));
  return PETSC_SUCCESS;
}

PetscErrorCode ApplyTimeSweep(PC preconditioner, Vec input, Vec output) {
  TimeSweepContext* c = nullptr;
  PetscCall(PCShellGetContext(preconditioner, reinterpret_cast<void**>(&c)));
  if (!c) return PETSC_ERR_ARG_NULL;
  PetscCall(RunTimeSweep(*c, input, output));
  return PETSC_SUCCESS;
}

/** The context fields and the four spatial work vectors; the spatial solver
 *  is installed afterwards by AdoptSpatialSolver. */
static void InitializeTimeSweepWorkspace(TimeSweepContext& context,
                                         TimeSweepDirection direction,
                                         Mat diagonal, Mat coupling,
                                         unsigned steps, double dt,
                                         const char* name,
                                         double coupling_scale = 0.0) {
  context.direction = direction;
  context.diagonal = diagonal;
  context.coupling = coupling;
  // 0 keeps the backward-Euler (1/dt) M coupling; a positive value is the
  // explicit scale of an already-formed coupling block (Crank-Nicolson: 1).
  context.inverse_dt = coupling_scale != 0.0 ? coupling_scale : 1.0 / dt;
  context.steps = steps;
  context.name = name;
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatGetLocalSize(diagonal, &context.spatial_local, nullptr));
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatCreateVecs(diagonal, &context.rhs, nullptr));
  PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(context.rhs, &context.sol));
  PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(context.rhs, &context.tmp));
  PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(context.rhs, &context.prev));
}

/** The context takes ownership of `spatial` (released by
 *  DestroyTimeSweepContext) and sets it up once, here, not inside the time
 *  loop. */
static void AdoptSpatialSolver(TimeSweepContext& context, KSP spatial) {
  context.spatial = spatial;
  PetscCallAbort(PETSC_COMM_WORLD, KSPSetUp(context.spatial));
}

void InitializeTimeSweepContextWithSpatialSolver(
    TimeSweepContext& context, TimeSweepDirection direction, Mat diagonal,
    Mat coupling, unsigned steps, double dt, const char* name, KSP spatial,
    double coupling_scale) {
  InitializeTimeSweepWorkspace(context, direction, diagonal, coupling, steps,
                               dt, name, coupling_scale);
  AdoptSpatialSolver(context, spatial);
}

/** One CG+GAMG spatial solver, reused for every time level. */
/** The CG+GAMG spatial solver on one diagonal block, with the sweep's
 *  tolerance and options prefix.  Set up by the caller through
 *  AdoptSpatialSolver, so the GAMG hierarchy is built once. */
static KSP CreateSweepSpatialSolver(Mat diagonal, double spatial_tolerance,
                                    const char* prefix) {
  KSP spatial = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, KSPCreate(PETSC_COMM_WORLD, &spatial));
  PetscCallAbort(PETSC_COMM_WORLD, KSPSetOptionsPrefix(spatial, prefix));
  PetscCallAbort(PETSC_COMM_WORLD,
                 KSPSetOperators(spatial, diagonal, diagonal));
  if (g_nonsymmetric_spatial_sweeps) {
    // ex21 convection: the block is nonsymmetric, so CG is invalid.  GMRES with
    // RIGHT preconditioning keeps the same (unpreconditioned) residual norm and
    // therefore the same meaning of the tolerance.
    PetscCallAbort(PETSC_COMM_WORLD, KSPSetType(spatial, KSPGMRES));
    PetscCallAbort(PETSC_COMM_WORLD, KSPGMRESSetRestart(spatial, 50));
    PetscCallAbort(PETSC_COMM_WORLD, KSPSetPCSide(spatial, PC_RIGHT));
  } else {
    PetscCallAbort(PETSC_COMM_WORLD, KSPSetType(spatial, KSPCG));
  }
  PetscCallAbort(PETSC_COMM_WORLD,
                 KSPSetNormType(spatial, KSP_NORM_UNPRECONDITIONED));
  PC pc = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, KSPGetPC(spatial, &pc));
  PetscCallAbort(PETSC_COMM_WORLD, PCSetType(pc, PCGAMG));
  PetscCallAbort(PETSC_COMM_WORLD,
                 KSPSetTolerances(spatial, spatial_tolerance, PETSC_DEFAULT,
                                  PETSC_DEFAULT, 500));
  PetscCallAbort(PETSC_COMM_WORLD,
                 KSPSetInitialGuessNonzero(spatial, PETSC_FALSE));
  PetscCallAbort(PETSC_COMM_WORLD, KSPSetFromOptions(spatial));
  return spatial;
}

void InitializeTimeSweepContext(TimeSweepContext& context,
                                TimeSweepDirection direction, Mat diagonal,
                                Mat coupling, unsigned steps, double dt,
                                double spatial_tolerance, const char* prefix,
                                const char* name, double coupling_scale,
                                Mat diagonal_last) {
  InitializeTimeSweepWorkspace(context, direction, diagonal, coupling, steps,
                               dt, name, coupling_scale);
  KSP spatial = CreateSweepSpatialSolver(diagonal, spatial_tolerance, prefix);
  context.spatial = spatial;
  // Build the GAMG hierarchy ONCE, here, not inside the time loop.
  AdoptSpatialSolver(context, spatial);
  if (diagonal_last) {
    context.diagonal_last = diagonal_last;
    context.spatial_last =
        CreateSweepSpatialSolver(diagonal_last, spatial_tolerance, prefix);
    PetscCallAbort(PETSC_COMM_WORLD, KSPSetUp(context.spatial_last));
  }
}

void DestroyTimeSweepContext(TimeSweepContext& context) {
  PetscCallAbort(PETSC_COMM_WORLD, KSPDestroy(&context.spatial_last));
  PetscCallAbort(PETSC_COMM_WORLD, KSPDestroy(&context.spatial));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&context.prev));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&context.tmp));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&context.sol));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&context.rhs));
}

/** Wrap a sweep as a KSP so that it is a drop-in replacement for the
 *  GMRES/GAMG whole-space-time solvers: KSPPREONLY + PCSHELL means one
 *  KSPSolve == exactly one triangular sweep. */
static void ConfigureTimeSweepSolver(TimeSweepContext& context, Mat space_time_operator,
                              const char* prefix, KSP* solver) {
  PetscCallAbort(PETSC_COMM_WORLD, KSPCreate(PETSC_COMM_WORLD, solver));
  PetscCallAbort(PETSC_COMM_WORLD, KSPSetOptionsPrefix(*solver, prefix));
  PetscCallAbort(PETSC_COMM_WORLD,
                 KSPSetOperators(*solver, space_time_operator, space_time_operator));
  PetscCallAbort(PETSC_COMM_WORLD, KSPSetType(*solver, KSPPREONLY));
  PC pc = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, KSPGetPC(*solver, &pc));
  PetscCallAbort(PETSC_COMM_WORLD, PCSetType(pc, PCSHELL));
  PetscCallAbort(PETSC_COMM_WORLD, PCShellSetContext(pc, &context));
  PetscCallAbort(PETSC_COMM_WORLD, PCShellSetApply(pc, ApplyTimeSweep));
  PetscCallAbort(PETSC_COMM_WORLD, PCShellSetName(pc, context.name));
  PetscCallAbort(PETSC_COMM_WORLD, KSPSetUp(*solver));
}

// ===========================================================================
// End triangular time sweeps.
// ===========================================================================

static void ConfigureDistributedMatchingAuxiliarySolve(
    Mat operator_matrix, Mat preconditioner_matrix, const char* prefix,
    double tolerance, KSP* solver) {
  PetscCallAbort(PETSC_COMM_WORLD, KSPCreate(PETSC_COMM_WORLD, solver));
  PetscCallAbort(PETSC_COMM_WORLD, KSPSetOptionsPrefix(*solver, prefix));
  PetscCallAbort(PETSC_COMM_WORLD,
                 KSPSetOperators(*solver, operator_matrix,
                                 preconditioner_matrix));
  PetscCallAbort(PETSC_COMM_WORLD, KSPSetType(*solver, KSPGMRES));
  PetscCallAbort(PETSC_COMM_WORLD, KSPGMRESSetRestart(*solver, 30));
  PetscCallAbort(PETSC_COMM_WORLD,
                 KSPSetTolerances(*solver, tolerance, PETSC_DEFAULT,
                                  PETSC_DEFAULT, 100));
  PetscCallAbort(PETSC_COMM_WORLD,
                 KSPSetInitialGuessNonzero(*solver, PETSC_FALSE));
  PC preconditioner = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, KSPGetPC(*solver, &preconditioner));
  PetscCallAbort(PETSC_COMM_WORLD, PCSetType(preconditioner, PCGAMG));
  PetscCallAbort(PETSC_COMM_WORLD, KSPSetFromOptions(*solver));
  PetscCallAbort(PETSC_COMM_WORLD, KSPSetUp(*solver));
}

static PetscErrorCode MatShellMultDistributedMatchingSchur(
    Mat matrix, Vec input, Vec output) {
  DistributedMatchingSchurPCContext* context = nullptr;
  PetscCall(MatShellGetContext(matrix, reinterpret_cast<void**>(&context)));
  if (!context || !context->matrices) return PETSC_ERR_ARG_NULL;
  PetscCall(MatMult(context->matrices->right, input, context->work_right));
  PetscCall(VecSet(context->work_ay, 0.0));
  PetscCall(KSPSolve(context->ay_solver, context->work_right,
                     context->work_ay));
  PetscCall(MatMult(context->matrices->left, context->work_ay, output));
  return PETSC_SUCCESS;
}

PetscErrorCode ApplyDistributedMatchingSchurPreconditioner(
    PC preconditioner, Vec input, Vec output) {
  DistributedMatchingSchurPCContext* context = nullptr;
  PetscCall(
      PCShellGetContext(preconditioner, reinterpret_cast<void**>(&context)));
  if (!context || !context->matrices) return PETSC_ERR_ARG_NULL;
  // The context's own MatMult computes
  //     Shat = left . E^{-1} . right,   left  = E^T + sqrt(beta) M_omega
  //                                     right = E   + sqrt(beta) N
  // whose inverse is Shat^{-1} = right^{-1} . E . left^{-1}: solve with `left`
  // first, multiply by E, then solve with `right`.
  KSP first_solver = context->legacy_factor_order ? context->right_solver
                                                  : context->left_solver;
  KSP second_solver = context->legacy_factor_order ? context->left_solver
                                                   : context->right_solver;
  PetscCall(VecSet(context->work_right, 0.0));
  PetscCall(KSPSolve(first_solver, input, context->work_right));
  PetscCall(MatMult(context->ay_matrix_free, context->work_right,
                    context->work_ay));
  PetscCall(VecSet(output, 0.0));
  PetscCall(KSPSolve(second_solver, context->work_ay, output));
  return PETSC_SUCCESS;
}

void InitializeDistributedMatchingSchurPCContext(
    DistributedMatchingSchurPCContext& context,
    DistributedMatchingAuxiliaryMatrices& matrices, Mat ay_matrix_free,
    Mat ay_assembled, double inner_tolerance,
    bool legacy_factor_order,
    // Non-null sweep_config replaces the three whole-space-time GMRES/GAMG
    // solves by exact triangular time sweeps.
    const ProblemConfig* sweep_config, Mat mass,
    Mat stiffness, Mat control, Mat tracking,
    double spatial_tolerance) {
  context.legacy_factor_order = legacy_factor_order;
  context.matrices = &matrices;
  context.ay_matrix_free = ay_matrix_free;
  context.ay_assembled = ay_assembled;
  context.use_sweeps = (sweep_config != nullptr);

  if (context.use_sweeps) {
    const ProblemConfig& cfg = *sweep_config;
    context.sweep_matrices =
        BuildSpatialSweepMatrices(cfg, mass, stiffness, control, tracking);
    // Backward Euler sweeps couple through (1/dt) M; Crank-Nicolson sweeps
    // couple through the owned -A_minus = M/dt - K/2 with unit scale, and the
    // `left` factor uses a second diagonal block on the final slot.
    const bool crank_nicolson = IsCrankNicolson(cfg);
    Mat sweep_coupling =
        crank_nicolson ? context.sweep_matrices.A_coupling : mass;
    const double sweep_coupling_scale = crank_nicolson ? 1.0 : 0.0;
    InitializeTimeSweepContext(
        context.right_sweep, TimeSweepDirection::FORWARD,
        context.sweep_matrices.A_right, sweep_coupling, cfg.time.steps,
        cfg.time.dt, spatial_tolerance, "sweep_right_", "right",
        sweep_coupling_scale);
    // ex21: the backward (E^T) sweep couples through the TRANSPOSED
    // Crank-Nicolson block when L is nonsymmetric.
    Mat left_coupling = context.sweep_matrices.A_coupling_transpose
                            ? context.sweep_matrices.A_coupling_transpose
                            : sweep_coupling;
    InitializeTimeSweepContext(
        context.left_sweep, TimeSweepDirection::BACKWARD,
        context.sweep_matrices.A_left, left_coupling, cfg.time.steps,
        cfg.time.dt, spatial_tolerance, "sweep_left_", "left",
        sweep_coupling_scale, context.sweep_matrices.A_left_final);
    InitializeTimeSweepContext(
        context.ay_sweep, TimeSweepDirection::FORWARD, context.sweep_matrices.A,
        sweep_coupling, cfg.time.steps, cfg.time.dt, spatial_tolerance,
        "sweep_ay_", "ay", sweep_coupling_scale);
    ConfigureTimeSweepSolver(context.right_sweep, matrices.right,
                             "fieldsplit_schur_right_", &context.right_solver);
    ConfigureTimeSweepSolver(context.left_sweep, matrices.left,
                             "fieldsplit_schur_left_", &context.left_solver);
    ConfigureTimeSweepSolver(context.ay_sweep, ay_matrix_free,
                             "fieldsplit_schur_ay_", &context.ay_solver);
  } else {
    ConfigureDistributedMatchingAuxiliarySolve(
        matrices.right, matrices.right, "fieldsplit_schur_right_",
        inner_tolerance, &context.right_solver);
    ConfigureDistributedMatchingAuxiliarySolve(
        matrices.left, matrices.left, "fieldsplit_schur_left_",
        inner_tolerance, &context.left_solver);
    ConfigureDistributedMatchingAuxiliarySolve(
        ay_matrix_free, ay_assembled, "fieldsplit_schur_ay_",
        inner_tolerance, &context.ay_solver);
    PetscCallAbort(PETSC_COMM_WORLD,
                   KSPMonitorSet(context.right_solver, CountInnerIterations,
                                 &context.right_counter, nullptr));
    PetscCallAbort(PETSC_COMM_WORLD,
                   KSPMonitorSet(context.left_solver, CountInnerIterations,
                                 &context.left_counter, nullptr));
    PetscCallAbort(PETSC_COMM_WORLD,
                   KSPMonitorSet(context.ay_solver, CountInnerIterations,
                                 &context.ay_counter, nullptr));
  }
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatCreateVecs(ay_matrix_free, &context.work_right, nullptr));
  PetscCallAbort(PETSC_COMM_WORLD,
                 VecDuplicate(context.work_right, &context.work_ay));
  PetscInt local_rows = 0, global_rows = 0;
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatGetLocalSize(ay_matrix_free, &local_rows, nullptr));
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatGetSize(ay_matrix_free, &global_rows, nullptr));
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatCreateShell(PETSC_COMM_WORLD, local_rows, local_rows,
                                global_rows, global_rows, &context,
                                &context.schur_preconditioner));
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatShellSetOperation(
                     context.schur_preconditioner, MATOP_MULT,
                     reinterpret_cast<void (*)(void)>(
                         MatShellMultDistributedMatchingSchur)));
}

void DestroyDistributedMatchingSchurPCContext(
    DistributedMatchingSchurPCContext& context) {
  PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&context.schur_preconditioner));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&context.work_ay));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&context.work_right));
  PetscCallAbort(PETSC_COMM_WORLD, KSPDestroy(&context.ay_solver));
  PetscCallAbort(PETSC_COMM_WORLD, KSPDestroy(&context.left_solver));
  PetscCallAbort(PETSC_COMM_WORLD, KSPDestroy(&context.right_solver));
  if (context.use_sweeps) {
    DestroyTimeSweepContext(context.ay_sweep);
    DestroyTimeSweepContext(context.left_sweep);
    DestroyTimeSweepContext(context.right_sweep);
    DestroySpatialSweepMatrices(context.sweep_matrices);
  }
}


