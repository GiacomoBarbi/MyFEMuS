// Space-time operator layer: the four blocks of the all-at-once KKT matrix.
//
//   E   = blockdiag(M/dt + K) - blocksubdiag(M/dt)     forward  (state)
//   E^T = blockdiag(M/dt + K) - blocksuperdiag(M/dt)   backward (adjoint)
//
// with backward Euler.  With Crank-Nicolson (distributed control only),
//
//   A_plus  =  M/dt + K/2 ,   A_minus = -M/dt + K/2
//   E_CN    = blockdiag(A_plus) + blocksubdiag(A_minus)
//   E_CN^T  = blockdiag(A_plus) + blocksuperdiag(A_minus)
//
// (M and K symmetric, so A_plus and A_minus are their own transposes).  The
// (2,1) block additionally carries the trapezoidal slot weights W_y/dt =
// diag(1,...,1,1/2).
//
// E and E^T are applied matrix-free (MatShell); assembled AIJ copies exist only
// as GAMG preconditioning operands.  The two off-diagonal blocks are block
// diagonal in time: (eta^2/alpha_u) I (x) N and -alpha_track I (x) M_omega --
// except for Dirichlet control, whose (1,2) block (eta^2/alpha_u) B_D R^{-1}
// B_D^T couples neighbouring time levels and is applied through
// DirichletControlOperators (E is then E_I: identity rows on the controlled
// boundary DOFs, from the eliminated M and K).
//
// Each rank owns ALL time levels of its own spatial DOFs, so a space-time
// vector is rank-major (see OwnershipPermutedSpaceTimeIndex) and no operator
// application communicates in the time direction.
//
// Pure PETSc: this header names no FEMuS type; the owner constructor in the
// .cpp only reads the spatial owner's borrowed Mat handles.

#pragma once

#include <petscmat.h>

#include "Config.hpp"
#include "DirichletControl.hpp"

struct SpatialOperators;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

enum class DistributedSpaceTimeKind { AY, AP, AYP, APY };

struct DistributedSpaceTimeContext {
  DistributedSpaceTimeKind kind = DistributedSpaceTimeKind::AY;
  Mat mass = nullptr;
  Mat stiffness = nullptr;
  Mat diagonal_mass = nullptr;
  unsigned steps = 0;
  PetscInt spatial_local = 0;
  PetscInt spatial_global = 0;
  double dt = 1.0;
  double scale = 1.0;
  // AY / AP: the time discretization.  BACKWARD_EULER keeps the original
  // blocks (diagonal M/dt + K, neighbour -M/dt); CRANK_NICOLSON uses
  // A_plus = M/dt + K/2 on the diagonal and A_minus = -M/dt + K/2 on the
  // neighbour, so the off-diagonal block carries a stiffness product too.
  ProblemConfig::TimeScheme scheme =
      ProblemConfig::TimeScheme::BACKWARD_EULER;
  // Crank-Nicolson with DIRICHLET control only: explicit, correctly embedded
  // A_plus and A_minus.  The hatted matrices carry the strong elimination as
  // K_I = 1 and M_I = 0 on the C diagonal, so M/dt + K/2 would put 1/2 there
  // instead of the identity the elimination needs (and 1/2 instead of 0 on the
  // neighbour).  When these are non-null the shell multiplies by them directly
  // instead of recombining M and K, which is also what makes the C rows exact.
  Mat diagonal_block = nullptr;
  Mat neighbour_block = nullptr;
  // AYP / APY: the scale of the LAST time slot.  Equal to `scale` for backward
  // Euler; the Crank-Nicolson (2,1) block carries the trapezoidal half weight
  // there (W_y = dt diag(1,...,1,1/2)).
  double final_scale = 1.0;
  Vec spatial_input = nullptr;
  Vec spatial_other = nullptr;
  Vec spatial_mass_result = nullptr;
  Vec spatial_stiffness_result = nullptr;
  Vec spatial_other_mass_result = nullptr;
  Vec spatial_other_stiffness_result = nullptr;   // Crank-Nicolson only
};

/** The four matrix-free blocks and the 2x2 MatNest over them.
 *
 *  A plain aggregate of handles and contexts, NOT an owner: it is a member of
 *  SpaceTimeOperators (below), which builds it in place and destroys it.
 *  The four shells store raw pointers to the four contexts below, so the
 *  object must never be copied or moved once built. */
struct DistributedSpaceTimeOperators {
  DistributedSpaceTimeContext ay_context;
  DistributedSpaceTimeContext ap_context;
  DistributedSpaceTimeContext ayp_context;
  DistributedSpaceTimeContext apy_context;
  DirichletControlShellContext dirichlet_context;   // Dirichlet control: the (1,2) block
  Mat ay = nullptr;
  Mat ap = nullptr;
  Mat ayp = nullptr;
  Mat apy = nullptr;
  Mat nest = nullptr;
};

/** The assembled AIJ copies of E and E^T; a member of SpaceTimeOperators. */
struct AssembledSpaceTimeOperators {
  Mat ay = nullptr;
  Mat ap = nullptr;
};

// ---------------------------------------------------------------------------
// Matrix-free and assembled space-time operators.
//
// These four routines are the construction and destruction steps of the
// owner SpaceTimeOperators (below) and are called only from its
// constructor and destructor.  Both builders fill the caller's object IN
// PLACE, so the shell contexts are created at their final address.
// ---------------------------------------------------------------------------

/** `dirichlet` (Dirichlet control only, null otherwise) supplies the (1,2)
 *  block; it is borrowed and must outlive the operators. */
void BuildDistributedSpaceTimeOperators(
    DistributedSpaceTimeOperators& operators, const ProblemConfig& config,
    Mat mass, Mat stiffness, Mat tracking, Mat control,
    DirichletControlOperators* dirichlet = nullptr,
    Mat cn_diagonal = nullptr, Mat cn_neighbour = nullptr);

void DestroyDistributedSpaceTimeOperators(
    DistributedSpaceTimeOperators& operators);

/** `scheme` selects the block pattern: backward Euler keeps the mass-only
 *  sub-diagonal, Crank-Nicolson assembles -M/dt + K/2 there (the union
 *  stiffness pattern, preallocated accordingly). */
void BuildAssembledSpaceTimeOperators(
    AssembledSpaceTimeOperators& operators, Mat mass, Mat stiffness,
    unsigned steps, double dt,
    ProblemConfig::TimeScheme scheme =
        ProblemConfig::TimeScheme::BACKWARD_EULER,
    Mat cn_diagonal = nullptr, Mat cn_neighbour = nullptr);

void DestroyAssembledSpaceTimeOperators(
    AssembledSpaceTimeOperators& operators);

// ---------------------------------------------------------------------------
// Ownership-permuted global row index.  Public because the matching-Schur
// auxiliary assembly (AddDistributedSpatialBlockDiagonal) also needs it.
// ---------------------------------------------------------------------------

PetscInt OwnershipPermutedSpaceTimeIndex(
    PetscInt spatial_dof, unsigned slot, unsigned steps,
    const PetscInt* spatial_ranges, int ranks);

// ---------------------------------------------------------------------------
// Time-slice vector transfers, used by the solver, the preconditioner sweeps,
// the KKT diagnostics and the output gather.
// ---------------------------------------------------------------------------

void CopyDataSideSliceToSpatial(Vec space_time, unsigned slot,
                                PetscInt spatial_local, Vec spatial);

void CopySpatialToDataSideSlice(Vec spatial, unsigned slot,
                                PetscInt spatial_local, double scale,
                                Vec space_time, bool add);

// ---------------------------------------------------------------------------
// The space-time operator owner
// ---------------------------------------------------------------------------

/** The all-at-once blocks: E and E^T matrix-free, plus the assembled AIJ copies
 *  GAMG preconditions with.  The single owner of the space-time operator set.
 *
 *  `distributed` holds the four MatShell contexts the shells point at.  The
 *  constructor builds it IN PLACE (the contexts are created at their final
 *  address) and copying and moving are deleted, so those addresses stay valid
 *  until the destructor destroys the shells first and the contexts after.
 *  The borrowed spatial Mat handles must outlive this object: declare it after
 *  the SpatialOperators it is built from. */
struct SpaceTimeOperators {
  SpaceTimeOperators(const ProblemConfig& config,
                     const SpatialOperators& spatial);
  ~SpaceTimeOperators();

  SpaceTimeOperators(const SpaceTimeOperators&) = delete;
  SpaceTimeOperators& operator=(const SpaceTimeOperators&) = delete;
  SpaceTimeOperators(SpaceTimeOperators&&) = delete;
  SpaceTimeOperators& operator=(SpaceTimeOperators&&) = delete;

  DistributedSpaceTimeOperators distributed;
  AssembledSpaceTimeOperators assembled;
  // Crank-Nicolson with Dirichlet control: the owned, correctly embedded
  // A_plus and A_minus the blocks above borrow (null otherwise).
  Mat cn_diagonal = nullptr;
  Mat cn_neighbour = nullptr;
  double elapsed = 0.0;
};
