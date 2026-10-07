// Boundary-reduced Schur operators; see BoundaryReducedSchur.hpp.

#include "BoundaryReducedSchur.hpp"

#include <petscblaslapack.h>
#include <petscksp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "Config.hpp"
#include "SpaceTimeOperators.hpp"

// A control DOF is one whose diagonal entry of N is not negligible relative to
// the largest one; every row of N outside the support must then be zero to
// the same relative level, which the constructor verifies.
static const double kSupportTolerance = 1.0e-12;

/** A direct solver for one SPD spatial matrix: KSPPREONLY with the first
 *  available factorization of the preference list (Cholesky before LU;
 *  CHOLMOD/PETSc on one rank, MUMPS on several).  Production defaults are set
 *  before KSPSetFromOptions, so the `prefix` options can override them.  The
 *  returned KSP is not yet set up; the caller owns it. */
static KSP CreateDirectSpatialSolver(Mat matrix, const char* prefix,
                                     std::string* backend) {
  int ranks = 1;
  MPI_Comm_size(PETSC_COMM_WORLD, &ranks);
  struct Candidate {
    const char* package;
    MatFactorType factor;
  };
  const Candidate serial[] = {{MATSOLVERCHOLMOD, MAT_FACTOR_CHOLESKY},
                              {MATSOLVERPETSC, MAT_FACTOR_LU}};
  const Candidate parallel[] = {{MATSOLVERMUMPS, MAT_FACTOR_CHOLESKY},
                                {MATSOLVERMUMPS, MAT_FACTOR_LU}};
  const Candidate* candidates = ranks > 1 ? parallel : serial;
  const Candidate* chosen = nullptr;
  for (int i = 0; i < 2 && !chosen; ++i) {
    PetscBool available = PETSC_FALSE;
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatGetFactorAvailable(matrix, candidates[i].package,
                                         candidates[i].factor, &available));
    if (available) chosen = &candidates[i];
  }
  if (!chosen)
    SETERRABORT(PETSC_COMM_WORLD, PETSC_ERR_SUP,
                "no direct spatial solver available: the boundary-reduced Schur"
                " operators need CHOLMOD or PETSc LU on one rank and MUMPS on"
                " several ranks");
  KSP solver = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, KSPCreate(PETSC_COMM_WORLD, &solver));
  PetscCallAbort(PETSC_COMM_WORLD, KSPSetOptionsPrefix(solver, prefix));
  PetscCallAbort(PETSC_COMM_WORLD, KSPSetOperators(solver, matrix, matrix));
  PetscCallAbort(PETSC_COMM_WORLD, KSPSetType(solver, KSPPREONLY));
  PetscCallAbort(PETSC_COMM_WORLD, KSPSetNormType(solver, KSP_NORM_NONE));
  PC pc = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, KSPGetPC(solver, &pc));
  PetscCallAbort(PETSC_COMM_WORLD,
                 PCSetType(pc, chosen->factor == MAT_FACTOR_CHOLESKY ? PCCHOLESKY
                                                                     : PCLU));
  PetscCallAbort(PETSC_COMM_WORLD, PCFactorSetMatSolverType(pc, chosen->package));
  PetscCallAbort(PETSC_COMM_WORLD, KSPSetFromOptions(solver));
  *backend = std::string(chosen->package) + ':' +
             (chosen->factor == MAT_FACTOR_CHOLESKY ? "cholesky" : "lu");
  return solver;
}

BoundaryReducedSchurOperators::BoundaryReducedSchurOperators(
    const ProblemConfig& config, Mat mass, Mat stiffness, Mat control,
    Mat tracking)
    : mass_(mass), stiffness_(stiffness), control_(control),
      tracking_(tracking), control_type_(config.control.type),
      steps_(config.time.steps), dt_(config.time.dt),
      scheme_(config.time.scheme) {
  int ranks = 1;
  MPI_Comm_size(PETSC_COMM_WORLD, &ranks);
  PetscCallAbort(PETSC_COMM_WORLD,
                 PetscObjectGetId((PetscObject)mass_, &mass_id_));
  PetscCallAbort(PETSC_COMM_WORLD,
                 PetscObjectGetId((PetscObject)stiffness_, &stiffness_id_));
  PetscCallAbort(PETSC_COMM_WORLD,
                 PetscObjectGetId((PetscObject)control_, &control_id_));
  PetscCallAbort(PETSC_COMM_WORLD,
                 PetscObjectGetId((PetscObject)tracking_, &tracking_id_));
  PetscCallAbort(PETSC_COMM_WORLD, MatGetSize(control_, &n_, nullptr));
  PetscCallAbort(PETSC_COMM_WORLD, MatGetLocalSize(control_, &local_, nullptr));
  const PetscInt* ranges = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, MatGetOwnershipRanges(control_, &ranges));
  spatial_ranges_.assign(ranges, ranges + ranks + 1);
  int rank = 0;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  begin_ = spatial_ranges_[rank];

  BuildPartition();
  CheckDenseStorage();
  BuildOperators();
  BuildSweeps();
  BuildInvariant();
  BuildTransfer();
  PetscCallAbort(PETSC_COMM_WORLD,
                 VecCreateMPI(PETSC_COMM_WORLD,
                              static_cast<PetscInt>(steps_) * local_,
                              PETSC_DECIDE, &work_));
  for (Vec* v : {&z_, &t_, &q_, &c_, &c_tmp_})
    PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(work_, v));
  PetscCallAbort(PETSC_COMM_WORLD, MatCreateVecs(mass_, &spatial_a_, nullptr));
  PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(spatial_a_, &spatial_b_));
  PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(spatial_a_, &spatial_c_));
}

BoundaryReducedSchurOperators::~BoundaryReducedSchurOperators() {
  for (Vec* v : {&spatial_c_, &spatial_b_, &spatial_a_, &c_tmp_, &c_, &q_, &t_, &z_})
    PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(v));
  // The dense solver factored S_work in place, so it goes before the matrix.
  PetscCallAbort(PETSC_COMM_WORLD, KSPDestroy(&dense_));
  PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&S_work_));
  for (Vec* v : {&seq_sol_, &seq_rhs_, &R_root2_, &R_root_})
    PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(v));
  PetscCallAbort(PETSC_COMM_WORLD, VecScatterDestroy(&to_root_));
  PetscCallAbort(PETSC_COMM_WORLD, ISDestroy(&is_R_root_));
  PetscCallAbort(PETSC_COMM_WORLD, ISDestroy(&is_R_canonical_));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&work_));
  PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&S1_));
  PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&S0_));
  DestroyTimeSweepContext(backward_Z_);
  DestroyTimeSweepContext(forward_);
  PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&M_ZZ_));
  PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&A_ZZ_));
  PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&coupling_));
  PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&A_minus_));
  PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&A_));
  PetscCallAbort(PETSC_COMM_WORLD, ISDestroy(&is_R_owned_));
}

bool BoundaryReducedSchurOperators::IsCompatible(
    const ProblemConfig& config, Mat mass, Mat stiffness, Mat control,
    Mat tracking, std::string* reason) const {
  // A borrowed matrix matches when it is the same PETSc object: same handle
  // and same id (the id guards against a new object at a recycled address).
  auto same = [](Mat stored, PetscObjectId stored_id, Mat given) {
    if (given != stored) return false;
    PetscObjectId id = 0;
    PetscCallAbort(PETSC_COMM_WORLD, PetscObjectGetId((PetscObject)given, &id));
    return id == stored_id;
  };
  const char* mismatch =
      !same(mass_, mass_id_, mass)                ? "mass operator changed"
      : !same(stiffness_, stiffness_id_, stiffness) ? "stiffness operator changed"
      : !same(control_, control_id_, control)     ? "control operator changed"
      : !same(tracking_, tracking_id_, tracking)  ? "tracking operator changed"
      : config.time.steps != steps_               ? "steps changed"
      : config.time.dt != dt_                     ? "dt changed"
      : config.control.type != control_type_      ? "control type changed"
      : config.time.scheme != scheme_            ? "time scheme changed"
                                                  : nullptr;
  if (mismatch && reason) *reason = mismatch;
  return mismatch == nullptr;
}

/** R from the diagonal of N on the owned rows, all-gathered and sorted so
 *  that every rank holds the same list; the rows outside R are verified to be
 *  zero (N must vanish on Z for the Z block of the Schur complement to be
 *  E^T_ZZ).  The owned space-time indices of R follow the production
 *  rank-major layout. */
void BoundaryReducedSchurOperators::BuildPartition() {
  int ranks = 1;
  MPI_Comm_size(PETSC_COMM_WORLD, &ranks);
  Vec diagonal = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, MatCreateVecs(control_, &diagonal, nullptr));
  PetscCallAbort(PETSC_COMM_WORLD, MatGetDiagonal(control_, diagonal));
  PetscReal scale = 0.0;
  PetscCallAbort(PETSC_COMM_WORLD, VecNorm(diagonal, NORM_INFINITY, &scale));
  const PetscScalar* d = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(diagonal, &d));
  double off_support_local = 0.0;
  for (PetscInt i = 0; i < local_; ++i) {
    const PetscInt row = begin_ + i;
    if (PetscRealPart(d[i]) > kSupportTolerance * scale) {
      R_owned_.push_back(row);
      continue;
    }
    PetscInt count = 0;
    const PetscScalar* values = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatGetRow(control_, row, &count, nullptr, &values));
    for (PetscInt k = 0; k < count; ++k)
      off_support_local =
          std::max(off_support_local, std::abs(PetscRealPart(values[k])));
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatRestoreRow(control_, row, &count, nullptr, &values));
  }
  PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(diagonal, &d));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&diagonal));

  double off_support = 0.0;
  MPI_Allreduce(&off_support_local, &off_support, 1, MPI_DOUBLE, MPI_MAX,
                PETSC_COMM_WORLD);
  int owned_count = static_cast<int>(R_owned_.size());
  std::vector<int> counts(ranks), displacements(ranks + 1, 0);
  MPI_Allgather(&owned_count, 1, MPI_INT, counts.data(), 1, MPI_INT,
                PETSC_COMM_WORLD);
  for (int r = 0; r < ranks; ++r) displacements[r + 1] = displacements[r] + counts[r];
  R_global_.assign(displacements[ranks], 0);
  MPI_Allgatherv(R_owned_.data(), owned_count, MPIU_INT, R_global_.data(),
                 counts.data(), displacements.data(), MPIU_INT, PETSC_COMM_WORLD);
  // The owned lists are increasing and the ownership ranges are contiguous and
  // ordered by rank, so the concatenation is already sorted: rank r's R DOFs
  // occupy R_global_[displacements[r] .. displacements[r+1]).
  support_counts_ = counts;
  support_offsets_ = displacements;

  // The three conditions are evaluated from replicated data, so every rank
  // takes the same branch.
  if (R_global_.empty())
    SETERRABORT(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
                "boundary-reduced Schur: the control operator has no support");
  if (static_cast<PetscInt>(R_global_.size()) == n_)
    SETERRABORT(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
                "boundary-reduced Schur: the control operator acts on every"
                " spatial DOF (no interior block to eliminate)");
  if (off_support > kSupportTolerance * scale)
    SETERRABORT(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
                "boundary-reduced Schur: the control operator has entries"
                " outside its diagonal support");

  std::vector<PetscInt> owned;
  owned.reserve(R_owned_.size() * steps_);
  for (PetscInt g : R_owned_)
    for (unsigned slot = 0; slot < steps_; ++slot)
      owned.push_back(OwnershipPermutedSpaceTimeIndex(
          g, slot, steps_, spatial_ranges_.data(), ranks));
  PetscCallAbort(PETSC_COMM_WORLD,
                 ISCreateGeneral(PETSC_COMM_WORLD,
                                 static_cast<PetscInt>(owned.size()),
                                 owned.data(), PETSC_COPY_VALUES, &is_R_owned_));
}

/** A = M/dt + K, and the embedded restrictions: zeroing the R rows and
 *  columns of A_ZZ (diagonal 1) and of M_ZZ (diagonal 0) embeds the restricted
 *  Z recurrence in the full spatial layout; with an R-masked right-hand side
 *  the sweep solution has zero R components and its Z components equal the
 *  restricted solve. */
void BoundaryReducedSchurOperators::BuildOperators() {
  // Backward Euler: A = M/dt + K and the coupling block is (1/dt) M, so M_ZZ is
  // built from the mass matrix and `coupling_` stays null.
  // Crank-Nicolson: A = A_plus = M/dt + K/2, A_minus = -M/dt + K/2, and the
  // coupling block is the owned -A_minus = M/dt - K/2 with unit scale.
  const bool crank_nicolson =
      scheme_ == ProblemConfig::TimeScheme::CRANK_NICOLSON;
  PetscCallAbort(PETSC_COMM_WORLD, MatDuplicate(mass_, MAT_COPY_VALUES, &A_));
  PetscCallAbort(PETSC_COMM_WORLD, MatScale(A_, 1.0 / dt_));
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatAXPY(A_, crank_nicolson ? 0.5 : 1.0, stiffness_,
                         SAME_NONZERO_PATTERN));
  if (crank_nicolson) {
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatDuplicate(mass_, MAT_COPY_VALUES, &A_minus_));
    PetscCallAbort(PETSC_COMM_WORLD, MatScale(A_minus_, -1.0 / dt_));
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatAXPY(A_minus_, 0.5, stiffness_, SAME_NONZERO_PATTERN));
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatDuplicate(A_minus_, MAT_COPY_VALUES, &coupling_));
    PetscCallAbort(PETSC_COMM_WORLD, MatScale(coupling_, -1.0));
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatSetOption(A_minus_, MAT_SYMMETRIC, PETSC_TRUE));
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatSetOption(coupling_, MAT_SYMMETRIC, PETSC_TRUE));
  }
  PetscCallAbort(PETSC_COMM_WORLD, MatDuplicate(A_, MAT_COPY_VALUES, &A_ZZ_));
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatDuplicate(crank_nicolson ? coupling_ : mass_,
                              MAT_COPY_VALUES, &M_ZZ_));
  const PetscInt count = static_cast<PetscInt>(R_owned_.size());
  for (Mat m : {A_ZZ_, M_ZZ_})
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatSetOption(m, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE));
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatZeroRowsColumns(A_ZZ_, count, R_owned_.data(), 1.0,
                                    nullptr, nullptr));
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatZeroRowsColumns(M_ZZ_, count, R_owned_.data(), 0.0,
                                    nullptr, nullptr));
  // M is SPD and K symmetric positive semidefinite, so A and A_ZZ are SPD
  // under both schemes (A_plus = M/dt + K/2 as well).  M_ZZ is symmetric; it
  // is positive semidefinite under backward Euler and, being M/dt - K/2,
  // symmetric indefinite under Crank-Nicolson -- it is never factored, only
  // multiplied, so only its symmetry is claimed below.
  for (Mat m : {A_, A_ZZ_}) {
    PetscCallAbort(PETSC_COMM_WORLD, MatSetOption(m, MAT_SYMMETRIC, PETSC_TRUE));
    PetscCallAbort(PETSC_COMM_WORLD, MatSetOption(m, MAT_SPD, PETSC_TRUE));
  }
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatSetOption(M_ZZ_, MAT_SYMMETRIC, PETSC_TRUE));
}

void BoundaryReducedSchurOperators::BuildSweeps() {
  // The generic sweep kernel computes diagonal^{-1}(rhs + scale * coupling *
  // x_neighbour).  Backward Euler passes (mass, 1/dt); Crank-Nicolson passes
  // the owned -A_minus with unit scale, which is exactly
  // A_plus x_k = rhs_k - A_minus x_{k-/+1}.
  InitializeTimeSweepContextWithSpatialSolver(
      forward_, TimeSweepDirection::FORWARD, A_, coupling(), steps_, dt_,
      "forward",
      CreateDirectSpatialSolver(A_, "schur_reduced_forward_", &forward_backend_),
      coupling_scale());
  InitializeTimeSweepContextWithSpatialSolver(
      backward_Z_, TimeSweepDirection::BACKWARD, A_ZZ_, M_ZZ_, steps_, dt_, "z",
      CreateDirectSpatialSolver(A_ZZ_, "schur_reduced_z_", &z_backend_),
      coupling_scale());
}

PetscErrorCode BoundaryReducedSchurOperators::SolveZ(Vec rhs, Vec solution) {
  PetscCall(VecCopy(rhs, work_));
  PetscCall(VecISSet(work_, is_R_owned_, 0.0));
  PetscCall(RunTimeSweep(backward_Z_, work_, solution));
  PetscCall(VecISSet(solution, is_R_owned_, 0.0));
  return PETSC_SUCCESS;
}

PetscErrorCode BoundaryReducedSchurOperators::SolveForward(Vec rhs,
                                                           Vec solution) {
  PetscCall(RunTimeSweep(forward_, rhs, solution));
  return PETSC_SUCCESS;
}

// ---------------------------------------------------------------------------
// Dense reduced operators
// ---------------------------------------------------------------------------

/** n x columns dense block with the spatial row layout. */
static Mat CreateSpatialBlock(PetscInt local, PetscInt global, PetscInt columns) {
  Mat block = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatCreateDense(PETSC_COMM_WORLD, local, PETSC_DECIDE, global,
                                columns, nullptr, &block));
  PetscCallAbort(PETSC_COMM_WORLD, MatAssemblyBegin(block, MAT_FINAL_ASSEMBLY));
  PetscCallAbort(PETSC_COMM_WORLD, MatAssemblyEnd(block, MAT_FINAL_ASSEMBLY));
  return block;
}

/** C = A B for a sparse A and a dense B; the product object is created on the
 *  first call and reused afterwards (same operand objects). */
static void SparseDenseProduct(Mat A, Mat B, Mat* C) {
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatMatMult(A, B, *C ? MAT_REUSE_MATRIX : MAT_INITIAL_MATRIX,
                            PETSC_DEFAULT, C));
}

/** Zero the given local rows of a dense block. */
static void ZeroLocalRows(Mat block, const std::vector<PetscInt>& local_rows) {
  PetscScalar* a = nullptr;
  PetscInt lda = 0, columns = 0;
  PetscCallAbort(PETSC_COMM_WORLD, MatDenseGetArray(block, &a));
  PetscCallAbort(PETSC_COMM_WORLD, MatDenseGetLDA(block, &lda));
  PetscCallAbort(PETSC_COMM_WORLD, MatGetSize(block, nullptr, &columns));
  for (PetscInt column = 0; column < columns; ++column)
    for (PetscInt row : local_rows) a[column * lda + row] = 0.0;
  PetscCallAbort(PETSC_COMM_WORLD, MatDenseRestoreArray(block, &a));
}

/** Copy the given local rows of a dense block into `out` (column-major,
 *  rows.size() x columns). */
static void CopyLocalRows(Mat block, const std::vector<PetscInt>& local_rows,
                          PetscInt columns, std::vector<double>& out) {
  const PetscScalar* a = nullptr;
  PetscInt lda = 0;
  const PetscInt rows = static_cast<PetscInt>(local_rows.size());
  out.resize(static_cast<std::size_t>(rows) * columns);
  PetscCallAbort(PETSC_COMM_WORLD, MatDenseGetArrayRead(block, &a));
  PetscCallAbort(PETSC_COMM_WORLD, MatDenseGetLDA(block, &lda));
  for (PetscInt column = 0; column < columns; ++column)
    for (PetscInt r = 0; r < rows; ++r)
      out[column * rows + r] = PetscRealPart(a[column * lda + local_rows[r]]);
  PetscCallAbort(PETSC_COMM_WORLD, MatDenseRestoreArrayRead(block, &a));
}

/** Gather the owned R rows of a dense block (given column-major in `local`,
 *  |R_owned| x columns) to rank 0 in the canonical spatial R order:
 *  `root` (rank 0) is column-major |R| x columns. */
static void GatherSupportRows(const std::vector<double>& local, PetscInt columns,
                              const std::vector<int>& counts,
                              const std::vector<int>& offsets,
                              std::vector<double>& root) {
  int rank = 0, ranks = 1;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  MPI_Comm_size(PETSC_COMM_WORLD, &ranks);
  const int n_R = offsets[ranks];
  std::vector<int> recv_counts(ranks), recv_offsets(ranks + 1, 0);
  for (int r = 0; r < ranks; ++r) {
    recv_counts[r] = counts[r] * static_cast<int>(columns);
    recv_offsets[r + 1] = recv_offsets[r] + recv_counts[r];
  }
  std::vector<double> packed(rank == 0 ? static_cast<std::size_t>(n_R) * columns : 0);
  MPI_Gatherv(local.data(), static_cast<int>(local.size()), MPI_DOUBLE,
              packed.data(), recv_counts.data(), recv_offsets.data(), MPI_DOUBLE,
              0, PETSC_COMM_WORLD);
  if (rank != 0) return;
  // packed: per rank, column-major counts[r] x columns  ->  column-major n_R x columns
  root.assign(static_cast<std::size_t>(n_R) * columns, 0.0);
  for (int r = 0; r < ranks; ++r)
    for (PetscInt column = 0; column < columns; ++column)
      for (int i = 0; i < counts[r]; ++i)
        root[column * n_R + offsets[r] + i] =
            packed[recv_offsets[r] + column * counts[r] + i];
}

/** S0 and S1 from the lag responses of the time-independent operators
 *  (A = M/dt + K, B = M/dt):
 *      X_0 = A^{-1} N Q_R,                X_l = A^{-1} B X_{l-1}
 *      Lam_0 = A_ZZ^{-1} A_ZR,  Lam_1 = A_ZZ^{-1}(-B_ZR + B_ZZ Lam_0),
 *      Lam_l = A_ZZ^{-1} B_ZZ Lam_{l-1}
 *  are the lag-l blocks of E^{-1}(I (x) N) Q_R and of (E_ZZ)^{-1} E_ZR, so
 *  every column of a later time slot is a time shift of the slot-0 column.
 *  With T0_m = (A_R Lam_m)^T - (M_R Lam_{m-1})^T / dt and
 *  Pi_{m,l} = Lam_m^T M_omega X_l the n_R x n_R blocks (slots i, j) are
 *      S0_{ij} = d_{ij} A_RR - d_{i,j-1} M_RR/dt - T0_{j-i}           (j >= i)
 *      S1_{ij} = (M_omega X_{i-j})_R [i >= j] - sum_{k >= max(i,j)} Pi_{k-i,k-j}
 *  Pi is streamed: each Lam_m is formed once, its row block
 *  Pi_{m,*} = Lam_m^T [M_omega X_0 ... M_omega X_{Nt-1}] is reduced to rank 0
 *  and accumulated into S1, and Lam_m is discarded.  The products use the
 *  rows where M_omega has support and the owned R rows of A and M only. */
void BoundaryReducedSchurOperators::BuildInvariant() {
  int rank = 0, ranks = 1;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  MPI_Comm_size(PETSC_COMM_WORLD, &ranks);
  const PetscInt n_R = control_support();
  const PetscInt nt_R = reduced_dimension();
  const double inverse_dt = 1.0 / dt_;
  // The three scheme-dependent block roles of the construction.  Backward
  // Euler keeps its original matrices and factors verbatim:
  //   forward coupling   (1/dt) M                    / -A_minus  (scale 1)
  //   super-diagonal     -(1/dt) M  (of E^T)         / A_minus
  //   Z coupling         (1/dt) M_ZZ                 / (-A_minus)_ZZ (scale 1)
  // `second_` is the super-diagonal block of E^T and `second_coeff` its scale,
  // so that lag 1 of E^T_RR, the lag-1 term of the Lambda recursion and the
  // T0 correction all read "second_coeff * second_".
  const bool crank_nicolson =
      scheme_ == ProblemConfig::TimeScheme::CRANK_NICOLSON;
  Mat forward_coupling = crank_nicolson ? coupling_ : mass_;
  const double forward_scale = crank_nicolson ? 1.0 : inverse_dt;
  Mat second = crank_nicolson ? A_minus_ : mass_;
  const double second_coeff = crank_nicolson ? 1.0 : -inverse_dt;
  const double z_coupling_scale = crank_nicolson ? 1.0 : inverse_dt;
  // Factor handles for the block solves; they are owned by the two PCs and
  // released with the sweep solvers, never here.
  Mat factor_A = nullptr, factor_ZZ = nullptr;
  {
    PC pc = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD, KSPGetPC(forward_.spatial, &pc));
    PetscCallAbort(PETSC_COMM_WORLD, PCFactorGetMatrix(pc, &factor_A));
    PetscCallAbort(PETSC_COMM_WORLD, KSPGetPC(backward_Z_.spatial, &pc));
    PetscCallAbort(PETSC_COMM_WORLD, PCFactorGetMatrix(pc, &factor_ZZ));
  }

  // Local rows of the support of M_omega (compaction of the products only;
  // the R rows of every product are kept separately, so an overlap of the
  // tracking support with R is handled exactly).
  std::vector<PetscInt> box_rows, R_local_rows;
  {
    Vec diagonal = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD, MatCreateVecs(tracking_, &diagonal, nullptr));
    PetscCallAbort(PETSC_COMM_WORLD, MatGetDiagonal(tracking_, diagonal));
    PetscReal scale = 0.0;
    PetscCallAbort(PETSC_COMM_WORLD, VecNorm(diagonal, NORM_INFINITY, &scale));
    const PetscScalar* d = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(diagonal, &d));
    for (PetscInt i = 0; i < local_; ++i)
      if (PetscRealPart(d[i]) > kSupportTolerance * scale) box_rows.push_back(i);
    PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(diagonal, &d));
    PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&diagonal));
  }
  for (PetscInt g : R_owned_) R_local_rows.push_back(g - begin_);
  const PetscInt n_box = static_cast<PetscInt>(box_rows.size());
  const PetscInt n_R_local = static_cast<PetscInt>(R_local_rows.size());

  // Owned R rows of A and M (all columns) for (E^T W)_R.
  Mat A_R = nullptr, M_R = nullptr;
  {
    IS rows = nullptr, columns = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD,
                   ISCreateGeneral(PETSC_COMM_WORLD, n_R_local, R_owned_.data(),
                                   PETSC_COPY_VALUES, &rows));
    PetscCallAbort(PETSC_COMM_WORLD,
                   ISCreateStride(PETSC_COMM_WORLD, local_, begin_, 1, &columns));
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatCreateSubMatrix(A_, rows, columns, MAT_INITIAL_MATRIX, &A_R));
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatCreateSubMatrix(second, rows, columns, MAT_INITIAL_MATRIX, &M_R));
    PetscCallAbort(PETSC_COMM_WORLD, ISDestroy(&columns));
    PetscCallAbort(PETSC_COMM_WORLD, ISDestroy(&rows));
  }
  std::vector<PetscInt> all_local_rows(n_R_local);   // rows of the A_R/M_R products
  for (PetscInt r = 0; r < n_R_local; ++r) all_local_rows[r] = r;

  // Unit columns of the spatial R DOFs; A_RR, M_RR (rank 0); A_ZR, B_ZR dt.
  Mat Q = CreateSpatialBlock(local_, n_, n_R);
  {
    PetscScalar* a = nullptr;
    PetscInt lda = 0;
    PetscCallAbort(PETSC_COMM_WORLD, MatDenseGetArray(Q, &a));
    PetscCallAbort(PETSC_COMM_WORLD, MatDenseGetLDA(Q, &lda));
    for (PetscInt column = 0; column < n_R; ++column) {
      const PetscInt g = R_global_[column];
      if (g >= begin_ && g < begin_ + local_) a[column * lda + (g - begin_)] = 1.0;
    }
    PetscCallAbort(PETSC_COMM_WORLD, MatDenseRestoreArray(Q, &a));
  }
  std::vector<double> local_rows_buffer, A_RR, M_RR;
  Mat AQ = nullptr, MQ = nullptr;
  SparseDenseProduct(A_, Q, &AQ);
  SparseDenseProduct(second, Q, &MQ);
  CopyLocalRows(AQ, R_local_rows, n_R, local_rows_buffer);
  GatherSupportRows(local_rows_buffer, n_R, support_counts_, support_offsets_, A_RR);
  CopyLocalRows(MQ, R_local_rows, n_R, local_rows_buffer);
  GatherSupportRows(local_rows_buffer, n_R, support_counts_, support_offsets_, M_RR);
  ZeroLocalRows(AQ, R_local_rows);   // A_ZR
  ZeroLocalRows(MQ, R_local_rows);   // B_ZR dt

  // ---- X lag sweep: keep M_omega X_l on the tracking rows and on the R rows.
  std::vector<std::vector<double>> MwX_box(steps_), MwX_R(steps_);
  Mat rhs = CreateSpatialBlock(local_, n_, n_R);
  Mat X = CreateSpatialBlock(local_, n_, n_R);
  Mat NQ = nullptr, MX = nullptr, MwX = nullptr;
  for (unsigned l = 0; l < steps_; ++l) {
    if (l == 0) {
      SparseDenseProduct(control_, Q, &NQ);
      PetscCallAbort(PETSC_COMM_WORLD, MatCopy(NQ, rhs, SAME_NONZERO_PATTERN));
    } else {
      // X_l = A^{-1} (forward coupling) X_{l-1}:  (1/dt) M under backward
      // Euler, -A_minus = M/dt - K/2 under Crank-Nicolson.
      SparseDenseProduct(forward_coupling, X, &MX);
      PetscCallAbort(PETSC_COMM_WORLD, MatCopy(MX, rhs, SAME_NONZERO_PATTERN));
      PetscCallAbort(PETSC_COMM_WORLD, MatScale(rhs, forward_scale));
    }
    PetscCallAbort(PETSC_COMM_WORLD, MatMatSolve(factor_A, rhs, X));
    SparseDenseProduct(tracking_, X, &MwX);
    CopyLocalRows(MwX, box_rows, n_R, MwX_box[l]);
    CopyLocalRows(MwX, R_local_rows, n_R, local_rows_buffer);
    GatherSupportRows(local_rows_buffer, n_R, support_counts_, support_offsets_, MwX_R[l]);
  }
  for (Mat* m : {&NQ, &MX, &MwX, &X}) PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(m));

  // ---- rank-0 accumulators (column-major nt_R x nt_R) and the Pi row block.
  std::vector<double> S0_values, S1_values;
  if (rank == 0) {
    S0_values.assign(static_cast<std::size_t>(nt_R) * nt_R, 0.0);
    S1_values.assign(static_cast<std::size_t>(nt_R) * nt_R, 0.0);
  }
  auto entry = [nt_R](std::vector<double>& values, PetscInt row, PetscInt column) -> double& {
    return values[static_cast<std::size_t>(column) * nt_R + row];
  };
  std::vector<double> Pi_local(static_cast<std::size_t>(n_R) * nt_R, 0.0);
  std::vector<double> Pi_root(rank == 0 ? Pi_local.size() : 0);
  std::vector<double> Lam_box, T0;

  // ---- Lambda lag sweep, streamed: one lag at a time.
  Mat Lam = CreateSpatialBlock(local_, n_, n_R);
  Mat Lam_prev = CreateSpatialBlock(local_, n_, n_R);
  Mat MZL = nullptr, ARL = nullptr, MRL = nullptr;
  for (unsigned m = 0; m < steps_; ++m) {
    PetscCallAbort(PETSC_COMM_WORLD, MatZeroEntries(rhs));
    if (m == 0) PetscCallAbort(PETSC_COMM_WORLD, MatAXPY(rhs, 1.0, AQ, SAME_NONZERO_PATTERN));
    if (m == 1) PetscCallAbort(PETSC_COMM_WORLD, MatAXPY(rhs, second_coeff, MQ, SAME_NONZERO_PATTERN));
    if (m >= 1) {
      SparseDenseProduct(M_ZZ_, Lam_prev, &MZL);
      PetscCallAbort(PETSC_COMM_WORLD, MatAXPY(rhs, z_coupling_scale, MZL, SAME_NONZERO_PATTERN));
    }
    PetscCallAbort(PETSC_COMM_WORLD, MatMatSolve(factor_ZZ, rhs, Lam));
    ZeroLocalRows(Lam, R_local_rows);   // the R components of the Z solve are zero
    // T0_m = (A_R Lam_m - M_R Lam_{m-1} / dt)^T, gathered to rank 0
    SparseDenseProduct(A_R, Lam, &ARL);
    if (m >= 1) {
      SparseDenseProduct(M_R, Lam_prev, &MRL);
      PetscCallAbort(PETSC_COMM_WORLD, MatAXPY(ARL, second_coeff, MRL, SAME_NONZERO_PATTERN));
    }
    CopyLocalRows(ARL, all_local_rows, n_R, local_rows_buffer);
    GatherSupportRows(local_rows_buffer, n_R, support_counts_, support_offsets_, T0);
    // Pi_{m,l} = Lam_m^T M_omega X_l over the tracking rows, all l at once
    CopyLocalRows(Lam, box_rows, n_R, Lam_box);
    if (n_box > 0) {
      PetscBLASInt bn = 0, bk = 0;
      PetscCallAbort(PETSC_COMM_WORLD, PetscBLASIntCast(n_R, &bn));
      PetscCallAbort(PETSC_COMM_WORLD, PetscBLASIntCast(n_box, &bk));
      const PetscScalar one = 1.0, zero = 0.0;
      for (unsigned l = 0; l < steps_; ++l)
        BLASgemm_("T", "N", &bn, &bn, &bk, &one, Lam_box.data(), &bk,
                  MwX_box[l].data(), &bk, &zero,
                  Pi_local.data() + static_cast<std::size_t>(l) * n_R * n_R, &bn);
    } else {
      std::fill(Pi_local.begin(), Pi_local.end(), 0.0);
    }
    MPI_Reduce(Pi_local.data(), Pi_root.data(), static_cast<int>(Pi_local.size()),
               MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);
    if (rank == 0) {
      // S0: the lag-m block above the diagonal;  S1: -Pi_{m,l} into every
      // (i, j) = (k - m, k - l), k = max(m, l) .. steps-1.
      for (unsigned i = 0; i + m < steps_; ++i) {
        const unsigned j = i + m;
        for (PetscInt a = 0; a < n_R; ++a)
          for (PetscInt b = 0; b < n_R; ++b) {
            double value = -T0[static_cast<std::size_t>(a) * n_R + b];   // (A_R Lam)^T: transpose
            if (m == 0) value += A_RR[static_cast<std::size_t>(b) * n_R + a];
            if (m == 1) value += second_coeff * M_RR[static_cast<std::size_t>(b) * n_R + a];
            entry(S0_values, static_cast<PetscInt>(i) * n_R + a,
                  static_cast<PetscInt>(j) * n_R + b) = value;
          }
      }
      // (Psi C_ZR)_{i,j} = sum_{k >= max(i,j)} what_k Lam_{k-i}^T (M_w X_{k-j})_Z
      // with i = k - m, j = k - l: the trapezoidal weight belongs to the
      // SUMMATION index k, not to the row or the column, so it cannot be
      // folded into Pi and must be applied here, inside the k loop.
      for (unsigned l = 0; l < steps_; ++l) {
        const double* block = Pi_root.data() + static_cast<std::size_t>(l) * n_R * n_R;
        for (unsigned k = std::max(m, l); k < steps_; ++k) {
          const PetscInt row0 = static_cast<PetscInt>(k - m) * n_R;
          const PetscInt column0 = static_cast<PetscInt>(k - l) * n_R;
          const double weight = SlotWeight(k);
          for (PetscInt b = 0; b < n_R; ++b)
            for (PetscInt a = 0; a < n_R; ++a)
              entry(S1_values, row0 + a, column0 + b) -= weight * block[static_cast<std::size_t>(b) * n_R + a];
        }
      }
    }
    PetscCallAbort(PETSC_COMM_WORLD, MatCopy(Lam, Lam_prev, SAME_NONZERO_PATTERN));
  }
  for (Mat* m : {&MZL, &ARL, &MRL, &Lam_prev, &Lam, &rhs, &AQ, &MQ, &Q, &M_R, &A_R})
    PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(m));

  if (rank != 0) return;
  // C_RR: the R rows of what_i M_omega X_{i-j} on and below the block diagonal.
  // C is block Toeplitz under backward Euler; under Crank-Nicolson the
  // trapezoidal weight of the ROW slot breaks that, which is why the weight is
  // read from i and not from the lag i - j.
  for (unsigned i = 0; i < steps_; ++i)
    for (unsigned j = 0; j <= i; ++j) {
      const std::vector<double>& block = MwX_R[i - j];
      const double weight = SlotWeight(i);
      for (PetscInt b = 0; b < n_R; ++b)
        for (PetscInt a = 0; a < n_R; ++a)
          entry(S1_values, static_cast<PetscInt>(i) * n_R + a,
                static_cast<PetscInt>(j) * n_R + b) += weight * block[static_cast<std::size_t>(b) * n_R + a];
    }
  for (int which = 0; which < 2; ++which) {
    Mat* target = which == 0 ? &S0_ : &S1_;
    const std::vector<double>& values = which == 0 ? S0_values : S1_values;
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatCreateSeqDense(PETSC_COMM_SELF, nt_R, nt_R, nullptr, target));
    PetscScalar* a = nullptr;
    PetscInt lda = 0;
    PetscCallAbort(PETSC_COMM_WORLD, MatDenseGetArray(*target, &a));
    PetscCallAbort(PETSC_COMM_WORLD, MatDenseGetLDA(*target, &lda));
    for (PetscInt column = 0; column < nt_R; ++column)
      for (PetscInt row = 0; row < nt_R; ++row)
        a[column * lda + row] = values[static_cast<std::size_t>(column) * nt_R + row];
    PetscCallAbort(PETSC_COMM_WORLD, MatDenseRestoreArray(*target, &a));
    PetscCallAbort(PETSC_COMM_WORLD, MatAssemblyBegin(*target, MAT_FINAL_ASSEMBLY));
    PetscCallAbort(PETSC_COMM_WORLD, MatAssemblyEnd(*target, MAT_FINAL_ASSEMBLY));
  }
}

/** The three dense reduced operators (S0, S1, S_work) live on rank 0; their
 *  storage is bounded by -schur_reduced_max_dense_mb (default 1024 MiB).
 *  Evaluated from replicated data, so every rank takes the same branch. */
void BoundaryReducedSchurOperators::CheckDenseStorage() {
  PetscInt limit_mb = 1024;
  PetscCallAbort(PETSC_COMM_WORLD,
                 PetscOptionsGetInt(nullptr, nullptr, "-schur_reduced_max_dense_mb",
                                    &limit_mb, nullptr));
  const std::size_t nt_R = static_cast<std::size_t>(reduced_dimension());
  const std::size_t max_dimension = static_cast<std::size_t>(PETSC_MAX_INT);
  // nt_R * nt_R must be representable as PetscInt (dense sequential matrix)
  // and 3 * nt_R * nt_R * sizeof(PetscScalar) must not overflow size_t.
  const bool representable = nt_R <= max_dimension / (nt_R > 0 ? nt_R : 1);
  const std::size_t entries = representable ? nt_R * nt_R : 0;
  const bool countable = representable && entries <= (static_cast<std::size_t>(-1) / (3 * sizeof(PetscScalar)));
  const double required_mb = 3.0 * static_cast<double>(nt_R) * static_cast<double>(nt_R) *
                             static_cast<double>(sizeof(PetscScalar)) / (1024.0 * 1024.0);
  if (!representable || !countable ||
      required_mb > static_cast<double>(limit_mb))
    SETERRABORT(PETSC_COMM_WORLD, PETSC_ERR_MEM,
                "boundary-reduced Schur: the reduced dimension nt_R = %d needs"
                " %.1f MB of dense storage on rank 0 (S0, S1, S_work) but"
                " -schur_reduced_max_dense_mb is %d",
                static_cast<int>(reduced_dimension()), required_mb,
                static_cast<int>(limit_mb));
}

/** One persistent scatter between the full layout and the canonical rank-0
 *  reduced vector (row j <-> slot j / |R|, R DOF R_global[j mod |R|]); the
 *  rank-0 sequential mirrors feed the dense solver. */
void BoundaryReducedSchurOperators::BuildTransfer() {
  int rank = 0, ranks = 1;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  MPI_Comm_size(PETSC_COMM_WORLD, &ranks);
  const PetscInt nt_R = reduced_dimension();
  std::vector<PetscInt> canonical, root;
  if (rank == 0) {
    canonical.reserve(nt_R);
    root.reserve(nt_R);
    for (unsigned slot = 0; slot < steps_; ++slot)
      for (PetscInt g : R_global_) {
        canonical.push_back(OwnershipPermutedSpaceTimeIndex(
            g, slot, steps_, spatial_ranges_.data(), ranks));
        root.push_back(static_cast<PetscInt>(root.size()));
      }
  }
  PetscCallAbort(PETSC_COMM_WORLD,
                 ISCreateGeneral(PETSC_COMM_WORLD, static_cast<PetscInt>(canonical.size()),
                                 canonical.data(), PETSC_COPY_VALUES, &is_R_canonical_));
  PetscCallAbort(PETSC_COMM_WORLD,
                 ISCreateGeneral(PETSC_COMM_WORLD, static_cast<PetscInt>(root.size()),
                                 root.data(), PETSC_COPY_VALUES, &is_R_root_));
  PetscCallAbort(PETSC_COMM_WORLD,
                 VecCreateMPI(PETSC_COMM_WORLD, rank == 0 ? nt_R : 0, nt_R, &R_root_));
  PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(R_root_, &R_root2_));
  Vec full = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD,
                 VecCreateMPI(PETSC_COMM_WORLD, static_cast<PetscInt>(steps_) * local_,
                              PETSC_DECIDE, &full));
  PetscCallAbort(PETSC_COMM_WORLD,
                 VecScatterCreate(full, is_R_canonical_, R_root_, is_R_root_, &to_root_));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&full));
  if (rank == 0) {
    PetscCallAbort(PETSC_COMM_WORLD, VecCreateSeq(PETSC_COMM_SELF, nt_R, &seq_rhs_));
    PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(seq_rhs_, &seq_sol_));
  }
}

PetscErrorCode BoundaryReducedSchurOperators::Prepare(double beta) {
  if (prepared_ && beta == prepared_beta_) return PETSC_SUCCESS;
  int rank = 0;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  if (rank == 0) {
    // S_work holds its own in-place factor, so its solver goes first.
    PetscCall(KSPDestroy(&dense_));
    PetscCall(MatDestroy(&S_work_));
    PetscCall(MatDuplicate(S0_, MAT_COPY_VALUES, &S_work_));
    PetscCall(MatAXPY(S_work_, beta, S1_, SAME_NONZERO_PATTERN));
    PetscCall(KSPCreate(PETSC_COMM_SELF, &dense_));
    PetscCall(KSPSetOptionsPrefix(dense_, "schur_reduced_dense_"));
    PetscCall(KSPSetOperators(dense_, S_work_, S_work_));
    PetscCall(KSPSetType(dense_, KSPPREONLY));
    PC pc = nullptr;
    PetscCall(KSPGetPC(dense_, &pc));
    PetscCall(PCSetType(pc, PCLU));
    PetscCall(PCFactorSetUseInPlace(pc, PETSC_TRUE));
    PetscCall(KSPSetFromOptions(dense_));
    PetscCall(KSPSetUp(dense_));
  }
  prepared_ = true;
  prepared_beta_ = beta;
  return PETSC_SUCCESS;
}

PetscErrorCode BoundaryReducedSchurOperators::ApplyTransposeE(Vec v, Vec out) {
  // (E^T v)_k = A v_k - (1/dt) M v_{k+1}  /  A_plus v_k + A_minus v_{k+1}.
  const bool crank_nicolson =
      scheme_ == ProblemConfig::TimeScheme::CRANK_NICOLSON;
  Mat neighbour = crank_nicolson ? A_minus_ : mass_;
  const double neighbour_scale = crank_nicolson ? 1.0 : -1.0 / dt_;
  for (unsigned k = 0; k < steps_; ++k) {
    CopyDataSideSliceToSpatial(v, k, local_, spatial_a_);
    PetscCall(MatMult(A_, spatial_a_, spatial_b_));
    if (k + 1 < steps_) {
      CopyDataSideSliceToSpatial(v, k + 1, local_, spatial_a_);
      PetscCall(MatMult(neighbour, spatial_a_, spatial_c_));
      PetscCall(VecAXPY(spatial_b_, neighbour_scale, spatial_c_));
    }
    CopySpatialToDataSideSlice(spatial_b_, k, local_, 1.0, out, false);
  }
  return PETSC_SUCCESS;
}

PetscErrorCode BoundaryReducedSchurOperators::ApplyC(Vec v, Vec out) {
  for (unsigned k = 0; k < steps_; ++k) {
    CopyDataSideSliceToSpatial(v, k, local_, spatial_a_);
    PetscCall(MatMult(control_, spatial_a_, spatial_b_));
    CopySpatialToDataSideSlice(spatial_b_, k, local_, 1.0, c_tmp_, false);
  }
  PetscCall(SolveForward(c_tmp_, out));
  // (C v)_k = what_k M_omega (E^{-1} N v)_k: the Crank-Nicolson trapezoidal
  // weight halves the FINAL slot; SlotWeight is 1 everywhere otherwise.
  for (unsigned k = 0; k < steps_; ++k) {
    CopyDataSideSliceToSpatial(out, k, local_, spatial_a_);
    PetscCall(MatMult(tracking_, spatial_a_, spatial_b_));
    CopySpatialToDataSideSlice(spatial_b_, k, local_, SlotWeight(k), c_tmp_,
                               false);
  }
  PetscCall(VecCopy(c_tmp_, out));
  return PETSC_SUCCESS;
}

PetscErrorCode BoundaryReducedSchurOperators::ApplySchur(double beta, Vec v, Vec Sv) {
  PetscCall(ApplyTransposeE(v, Sv));
  PetscCall(ApplyC(v, c_));
  PetscCall(VecAXPY(Sv, beta, c_));
  return PETSC_SUCCESS;
}

PetscErrorCode BoundaryReducedSchurOperators::Solve(Vec b, Vec x) {
  int rank = 0;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  PetscCheck(prepared_, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE,
             "boundary-reduced Schur: Solve before Prepare");
  // y_Z = S_ZZ^{-1} b_Z ;  r = b - E^T Q_Z y_Z  (its R rows are r_R)
  PetscCall(SolveZ(b, z_));
  PetscCall(ApplyTransposeE(z_, t_));
  PetscCall(VecAYPX(t_, -1.0, b));
  // x_R = S_work^{-1} r_R on rank 0
  PetscCall(VecScatterBegin(to_root_, t_, R_root_, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(to_root_, t_, R_root_, INSERT_VALUES, SCATTER_FORWARD));
  if (rank == 0) {
    const PetscScalar* source = nullptr;
    PetscScalar* target = nullptr;
    PetscCall(VecGetArrayRead(R_root_, &source));
    PetscCall(VecGetArray(seq_rhs_, &target));
    for (PetscInt i = 0; i < reduced_dimension(); ++i) target[i] = source[i];
    PetscCall(VecRestoreArray(seq_rhs_, &target));
    PetscCall(VecRestoreArrayRead(R_root_, &source));
    PetscCall(KSPSolve(dense_, seq_rhs_, seq_sol_));
    PetscCall(VecGetArrayRead(seq_sol_, &source));
    PetscCall(VecGetArray(R_root2_, &target));
    for (PetscInt i = 0; i < reduced_dimension(); ++i) target[i] = source[i];
    PetscCall(VecRestoreArray(R_root2_, &target));
    PetscCall(VecRestoreArrayRead(seq_sol_, &source));
  }
  // q = Q_R x_R ;  x = q + S_ZZ^{-1} (b - S q)_Z
  PetscCall(VecSet(q_, 0.0));
  PetscCall(VecScatterBegin(to_root_, R_root2_, q_, INSERT_VALUES, SCATTER_REVERSE));
  PetscCall(VecScatterEnd(to_root_, R_root2_, q_, INSERT_VALUES, SCATTER_REVERSE));
  PetscCall(ApplySchur(prepared_beta_, q_, t_));
  PetscCall(VecAYPX(t_, -1.0, b));
  PetscCall(SolveZ(t_, x));
  PetscCall(VecAXPY(x, 1.0, q_));
  return PETSC_SUCCESS;
}

PetscErrorCode ApplyBoundaryReducedSchur(PC pc, Vec b, Vec x) {
  BoundaryReducedSchurOperators* reduced = nullptr;
  PetscCall(PCShellGetContext(pc, reinterpret_cast<void**>(&reduced)));
  if (!reduced) return PETSC_ERR_ARG_NULL;
  PetscCall(reduced->Solve(b, x));
  return PETSC_SUCCESS;
}
