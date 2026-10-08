// Dirichlet boundary control operators; see DirichletControl.hpp.

#include "DirichletControl.hpp"

#include <petscblaslapack.h>
#include <petscksp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "Config.hpp"

// A controlled DOF is one whose diagonal entry of the trace mass matrix is not
// negligible relative to the largest one (the same support rule as the
// boundary-reduced partition).
static const double kSupportTolerance = 1.0e-12;

KSP CreateDirichletDirectSolver(Mat matrix, const char* prefix,
                                std::string* backend) {
  return CreateDirichletDirectSolver(matrix, prefix, backend, false);
}

/** T7D: `lu_only` skips the Cholesky candidates (a nonsymmetric matrix, e.g.
 *  the convection block A_I, must never be handed to a Cholesky factor). */
KSP CreateDirichletDirectSolver(Mat matrix, const char* prefix,
                                std::string* backend, bool lu_only) {
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
  for (int i = lu_only ? 1 : 0; i < 2 && !chosen; ++i) {
    PetscBool available = PETSC_FALSE;
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatGetFactorAvailable(matrix, candidates[i].package,
                                         candidates[i].factor, &available));
    if (available) chosen = &candidates[i];
  }
  if (!chosen)
    SETERRABORT(PETSC_COMM_WORLD, PETSC_ERR_SUP,
                "Dirichlet control: no direct solver available (CHOLMOD/PETSc"
                " LU on one rank, MUMPS on several)");
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
  PetscCallAbort(PETSC_COMM_WORLD, KSPSetUp(solver));
  *backend = std::string(chosen->package) + ':' +
             (chosen->factor == MAT_FACTOR_CHOLESKY ? "cholesky" : "lu");
  return solver;
}

DirichletControlOperators::DirichletControlOperators(
    const ProblemConfig& config, Mat mass, Mat stiffness, Mat trace_mass,
    Mat trace_stiffness)
    : mass_(mass), stiffness_(stiffness), scheme_(config.time.scheme),
      steps_(config.time.steps), dt_(config.time.dt),
      ell_(config.control_h1_length), alpha_u_(config.alpha_u),
      gamma_(config.control_h1_time_weight),
      rho_(ControlTimeRatio(config)) {
  transpose_stiffness_ = HasConvection(config);
  sparse_core_ = config.dirichlet_core.solver ==
                 ProblemConfig::DirichletCoreConfig::Solver::ITERATIVE;
  PetscCallAbort(PETSC_COMM_WORLD, MatGetSize(mass_, &n_, nullptr));
  PetscCallAbort(PETSC_COMM_WORLD, MatGetLocalSize(mass_, &local_, nullptr));
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatGetOwnershipRange(mass_, &begin_, nullptr));
  BuildPartition(trace_mass);
  BuildRegularization(trace_mass, trace_stiffness);
  PetscCallAbort(PETSC_COMM_WORLD, MatCreateVecs(mass_, &spatial_a_, nullptr));
  for (Vec* v : {&spatial_b_, &spatial_c_, &spatial_d_})
    PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(spatial_a_, v));
  PetscCallAbort(PETSC_COMM_WORLD,
                 VecCreateMPI(PETSC_COMM_WORLD,
                              static_cast<PetscInt>(steps_) * local_,
                              PETSC_DECIDE, &work_));
  if (gamma_ > 0.0) {
    PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(work_, &work2_));
    BuildTemporalRegularization(trace_mass);
  }
}

DirichletControlOperators::~DirichletControlOperators() {
  PetscCallAbort(PETSC_COMM_WORLD, KSPDestroy(&RT_solver_));
  PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&RT_));
  PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&M_face_));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&work2_));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&work_));
  for (Vec* v : {&spatial_d_, &spatial_c_, &spatial_b_, &spatial_a_})
    PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(v));
  PetscCallAbort(PETSC_COMM_WORLD, KSPDestroy(&R_solver_));
  PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&R_));
}

/** C from the diagonal of the trace mass matrix on the owned rows; the count
 *  is all-reduced so that every rank sees the same |C|. */
void DirichletControlOperators::BuildPartition(Mat trace_mass) {
  Vec diagonal = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, MatCreateVecs(trace_mass, &diagonal, nullptr));
  PetscCallAbort(PETSC_COMM_WORLD, MatGetDiagonal(trace_mass, diagonal));
  PetscReal scale = 0.0;
  PetscCallAbort(PETSC_COMM_WORLD, VecNorm(diagonal, NORM_INFINITY, &scale));
  const PetscScalar* d = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(diagonal, &d));
  mask_.assign(static_cast<std::size_t>(local_), 0);
  for (PetscInt i = 0; i < local_; ++i)
    if (PetscRealPart(d[i]) > kSupportTolerance * scale) {
      mask_[i] = 1;
      C_owned_.push_back(begin_ + i);
    }
  PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(diagonal, &d));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&diagonal));
  PetscInt owned = static_cast<PetscInt>(C_owned_.size());
  MPI_Allreduce(&owned, &control_count_, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD);
  if (control_count_ == 0)
    SETERRABORT(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
                "Dirichlet control: the control face has no DOFs");
  if (control_count_ == n_)
    SETERRABORT(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
                "Dirichlet control: every spatial DOF is controlled");
}

/** Embed face matrices into the full DOF layout: the C rows carry
 *  sum_k factor_k * source_k, the I rows the identity.  Assembled row by row
 *  into a fresh AIJ matrix -- the face matrices store no diagonal entry
 *  outside the face, so a row/column elimination cannot be used to embed
 *  them.  SPD on the full layout whenever the face sum is SPD on C, and a
 *  solve with a right-hand side supported on C returns a solution supported
 *  on C.  The preallocation counts every source, whether or not its factor
 *  vanishes. */
static Mat EmbedFaceMatrix(const std::vector<std::pair<Mat, double>>& sources,
                           const std::vector<unsigned char>& mask,
                           PetscInt local, PetscInt begin, PetscInt n) {
  std::vector<PetscInt> nnz(static_cast<std::size_t>(local), 1);
  for (PetscInt i = 0; i < local; ++i) {
    if (!mask[static_cast<std::size_t>(i)]) continue;
    // (MatRestoreRow may clear the count: keep it before restoring.)
    PetscInt total = 1;
    for (const std::pair<Mat, double>& source : sources) {
      PetscInt count = 0;
      PetscCallAbort(PETSC_COMM_WORLD,
                     MatGetRow(source.first, begin + i, &count, nullptr, nullptr));
      total += count;
      PetscCallAbort(PETSC_COMM_WORLD,
                     MatRestoreRow(source.first, begin + i, &count, nullptr, nullptr));
    }
    nnz[static_cast<std::size_t>(i)] = total;
  }
  Mat embedded = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, MatCreate(PETSC_COMM_WORLD, &embedded));
  PetscCallAbort(PETSC_COMM_WORLD, MatSetSizes(embedded, local, local, n, n));
  PetscCallAbort(PETSC_COMM_WORLD, MatSetType(embedded, MATAIJ));
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatSeqAIJSetPreallocation(embedded, 0, nnz.data()));
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatMPIAIJSetPreallocation(embedded, 0, nnz.data(), 0, nnz.data()));
  PetscCallAbort(PETSC_COMM_WORLD, MatSetUp(embedded));
  for (PetscInt i = 0; i < local; ++i) {
    const PetscInt row = begin + i;
    if (!mask[static_cast<std::size_t>(i)]) {
      const PetscScalar one = 1.0;
      PetscCallAbort(PETSC_COMM_WORLD,
                     MatSetValues(embedded, 1, &row, 1, &row, &one, ADD_VALUES));
      continue;
    }
    for (const std::pair<Mat, double>& source : sources) {
      const double factor = source.second;
      if (factor == 0.0) continue;
      PetscInt count = 0;
      const PetscInt* columns = nullptr;
      const PetscScalar* values = nullptr;
      PetscCallAbort(PETSC_COMM_WORLD,
                     MatGetRow(source.first, row, &count, &columns, &values));
      for (PetscInt k = 0; k < count; ++k) {
        const PetscScalar value = factor * values[k];
        PetscCallAbort(PETSC_COMM_WORLD,
                       MatSetValues(embedded, 1, &row, 1, &columns[k], &value,
                                    ADD_VALUES));
      }
      PetscCallAbort(PETSC_COMM_WORLD,
                     MatRestoreRow(source.first, row, &count, &columns, &values));
    }
  }
  PetscCallAbort(PETSC_COMM_WORLD, MatAssemblyBegin(embedded, MAT_FINAL_ASSEMBLY));
  PetscCallAbort(PETSC_COMM_WORLD, MatAssemblyEnd(embedded, MAT_FINAL_ASSEMBLY));
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatSetOption(embedded, MAT_SYMMETRIC, PETSC_TRUE));
  PetscCallAbort(PETSC_COMM_WORLD, MatSetOption(embedded, MAT_SPD, PETSC_TRUE));
  return embedded;
}

/** R = M_Gamma + ell^2 K_Gamma on the C rows, the identity on the I rows. */
void DirichletControlOperators::BuildRegularization(Mat trace_mass,
                                                    Mat trace_stiffness) {
  R_ = EmbedFaceMatrix({{trace_mass, 1.0}, {trace_stiffness, ell_ * ell_}},
                       mask_, local_, begin_, n_);
  R_solver_ = CreateDirichletDirectSolver(R_, "dirichlet_control_r_", &backend_);
}

/** The replicated sorted C list and the position of each owned C DOF in it. */
void DirichletControlOperators::BuildGlobalControlList() {
  int ranks = 1;
  MPI_Comm_size(PETSC_COMM_WORLD, &ranks);
  std::vector<int> counts(ranks), displacements(ranks);
  int mine = static_cast<int>(C_owned_.size());
  MPI_Allgather(&mine, 1, MPI_INT, counts.data(), 1, MPI_INT, PETSC_COMM_WORLD);
  int total = 0;
  for (int r = 0; r < ranks; ++r) { displacements[r] = total; total += counts[r]; }
  C_global_.assign(static_cast<std::size_t>(total), 0);
  MPI_Allgatherv(C_owned_.data(), mine, MPIU_INT, C_global_.data(), counts.data(),
                 displacements.data(), MPIU_INT, PETSC_COMM_WORLD);
  std::sort(C_global_.begin(), C_global_.end());
  for (PetscInt g : C_owned_)
    C_owned_position_.push_back(static_cast<PetscInt>(
        std::lower_bound(C_global_.begin(), C_global_.end(), g) - C_global_.begin()));
}

/** The C x C block of an embedded face matrix, replicated dense column-major. */
void DirichletControlOperators::BuildFaceBlock(Mat embedded,
                                               std::vector<double>& dense) const {
  const std::size_t block = static_cast<std::size_t>(control_count_) * control_count_;
  std::vector<double> partial(block, 0.0);
  dense.assign(block, 0.0);
  for (std::size_t i = 0; i < C_owned_.size(); ++i) {
    PetscInt ncols = 0;
    const PetscInt* cols = nullptr;
    const PetscScalar* vals = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatGetRow(embedded, C_owned_[i], &ncols, &cols, &vals));
    const PetscInt row = C_owned_position_[i];
    for (PetscInt j = 0; j < ncols; ++j) {
      auto it = std::lower_bound(C_global_.begin(), C_global_.end(), cols[j]);
      if (it == C_global_.end() || *it != cols[j]) continue;
      const PetscInt col = static_cast<PetscInt>(it - C_global_.begin());
      partial[static_cast<std::size_t>(col) * control_count_ + row] +=
          PetscRealPart(vals[j]);
    }
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatRestoreRow(embedded, C_owned_[i], &ncols, &cols, &vals));
  }
  MPI_Allreduce(partial.data(), dense.data(), static_cast<int>(block), MPI_DOUBLE,
                MPI_SUM, PETSC_COMM_WORLD);
}

/** gamma_u > 0 only: the embedded M_Gamma, the dense C-blocks and the dense
 *  replicated Cholesky factor of
 *      R_H1t = What_u (x) R_Gamma + (rho/dt) T (x) M_Gamma,
 *  T = tridiag(-1, 2, -1) with 1 on the LAST diagonal entry.  R_H1t is SPD:
 *  R_Gamma is SPD on C, What_u > 0 and T = D^T D is SPD, so the Cholesky
 *  cannot fail for any rho >= 0 -- a failure means the face blocks are wrong
 *  and is reported as such. */
void DirichletControlOperators::BuildTemporalRegularization(Mat trace_mass) {
  M_face_ = EmbedFaceMatrix({{trace_mass, 1.0}}, mask_, local_, begin_, n_);
  BuildGlobalControlList();
  BuildFaceBlock(R_, R_dense_);
  BuildFaceBlock(M_face_, M_dense_);
  // T8D iterative core: R_H1t^{-1} uses the sparse reduced factor instead of
  // the dense (N_t|C|)^2 Cholesky built below (the dense path is unchanged).
  if (sparse_core_) {
    ReducedRegularization();
    return;
  }
  const PetscInt nC = control_count_;
  const PetscInt dim = static_cast<PetscInt>(steps_) * nC;
  Q_dense_.assign(static_cast<std::size_t>(dim) * dim, 0.0);
  const double coupling = rho_ / dt_;
  for (unsigned m = 0; m < steps_; ++m) {
    const double control_weight = SlotWeight(m);
    const double diagonal = coupling * TemporalDiagonal(m);
    for (PetscInt cp = 0; cp < nC; ++cp) {
      for (PetscInt c = 0; c < nC; ++c) {
        const std::size_t entry = static_cast<std::size_t>(cp) * nC + c;
        const std::size_t row = static_cast<std::size_t>(m) * nC + c;
        const std::size_t col = static_cast<std::size_t>(m) * nC + cp;
        Q_dense_[col * dim + row] =
            control_weight * R_dense_[entry] + diagonal * M_dense_[entry];
      }
    }
    if (m + 1 >= steps_) continue;
    for (PetscInt cp = 0; cp < nC; ++cp) {
      for (PetscInt c = 0; c < nC; ++c) {
        const double value = -coupling * M_dense_[static_cast<std::size_t>(cp) * nC + c];
        const std::size_t row = static_cast<std::size_t>(m) * nC + c;
        const std::size_t col = static_cast<std::size_t>(m + 1) * nC + cp;
        Q_dense_[col * dim + row] = value;
        Q_dense_[row * dim + col] = value;   // M_Gamma is symmetric
      }
    }
  }
  PetscBLASInt bdim = 0, info = 0;
  PetscCallAbort(PETSC_COMM_WORLD, PetscBLASIntCast(dim, &bdim));
  LAPACKpotrf_("L", &bdim, Q_dense_.data(), &bdim, &info);
  if (info != 0)
    SETERRABORT(PETSC_COMM_WORLD, PETSC_ERR_LIB,
                "Dirichlet control: dense Cholesky of R_H1t failed");
  reduced_a_.assign(static_cast<std::size_t>(dim), 0.0);
  reduced_b_.assign(static_cast<std::size_t>(dim), 0.0);
}

/** Local slice helpers on the rank-major space-time layout. */
static PetscErrorCode SliceToSpatial(Vec space_time, unsigned slot,
                                     PetscInt local, Vec spatial) {
  const PetscScalar* s = nullptr;
  PetscScalar* d = nullptr;
  PetscCall(VecGetArrayRead(space_time, &s));
  PetscCall(VecGetArray(spatial, &d));
  const PetscInt offset = static_cast<PetscInt>(slot) * local;
  for (PetscInt i = 0; i < local; ++i) d[i] = s[offset + i];
  PetscCall(VecRestoreArray(spatial, &d));
  PetscCall(VecRestoreArrayRead(space_time, &s));
  return PETSC_SUCCESS;
}

static PetscErrorCode SpatialToSlice(Vec spatial, unsigned slot, PetscInt local,
                                     Vec space_time) {
  const PetscScalar* s = nullptr;
  PetscScalar* d = nullptr;
  PetscCall(VecGetArrayRead(spatial, &s));
  PetscCall(VecGetArray(space_time, &d));
  const PetscInt offset = static_cast<PetscInt>(slot) * local;
  for (PetscInt i = 0; i < local; ++i) d[offset + i] = s[i];
  PetscCall(VecRestoreArray(space_time, &d));
  PetscCall(VecRestoreArrayRead(spatial, &s));
  return PETSC_SUCCESS;
}

/** Keep (`keep_control` = true) or drop the C entries of a spatial vector. */
static PetscErrorCode MaskSpatial(Vec v, const std::vector<unsigned char>& mask,
                                  bool keep_control) {
  PetscScalar* a = nullptr;
  PetscCall(VecGetArray(v, &a));
  for (std::size_t i = 0; i < mask.size(); ++i)
    if ((mask[i] != 0) != keep_control) a[i] = 0.0;
  PetscCall(VecRestoreArray(v, &a));
  return PETSC_SUCCESS;
}

/** out = A v_cur - (M/dt) v_prev   (backward Euler)
 *      = A_plus v_cur + A_minus v_prev   (Crank-Nicolson),
 *  on the unhatted matrices: the single place where the time scheme enters B_D
 *  (current = Q_C u_n, previous = Q_C u_{n-1}) and B_D^T (current = p_n,
 *  previous = p_{n+1}).  T7D: `transpose` (B_D^T) applies the TRANSPOSED
 *  blocks A^T = M/dt + L^T, A_plus^T, A_minus^T -- the mass matrix is
 *  symmetric, so only the stiffness product changes, and only when the
 *  stiffness is nonsymmetric (transpose_stiffness_, convection on). */
PetscErrorCode DirichletControlOperators::ApplyStep(Vec spatial_current,
                                                    Vec spatial_previous,
                                                    bool have_previous,
                                                    Vec spatial_out,
                                                    bool transpose) {
  auto stiffness_product = [&](Vec in, Vec out) -> PetscErrorCode {
    if (transpose && transpose_stiffness_) return MatMultTranspose(stiffness_, in, out);
    return MatMult(stiffness_, in, out);
  };
  if (scheme_ == ProblemConfig::TimeScheme::CRANK_NICOLSON) {
    PetscCall(stiffness_product(spatial_current, spatial_out));
    PetscCall(VecScale(spatial_out, 0.5));
    PetscCall(MatMult(mass_, spatial_current, spatial_d_));
    PetscCall(VecAXPY(spatial_out, 1.0 / dt_, spatial_d_));
    if (have_previous) {
      PetscCall(stiffness_product(spatial_previous, spatial_d_));
      PetscCall(VecAXPY(spatial_out, 0.5, spatial_d_));
      PetscCall(MatMult(mass_, spatial_previous, spatial_d_));
      PetscCall(VecAXPY(spatial_out, -1.0 / dt_, spatial_d_));
    }
    return PETSC_SUCCESS;
  }
  PetscCall(stiffness_product(spatial_current, spatial_out));
  PetscCall(MatMult(mass_, spatial_current, spatial_d_));
  PetscCall(VecAXPY(spatial_out, 1.0 / dt_, spatial_d_));
  if (have_previous) {
    PetscCall(MatMult(mass_, spatial_previous, spatial_d_));
    PetscCall(VecAXPY(spatial_out, -1.0 / dt_, spatial_d_));
  }
  return PETSC_SUCCESS;
}

PetscErrorCode DirichletControlOperators::ApplyBD(Vec u, Vec out) {
  for (unsigned n = 0; n < steps_; ++n) {
    PetscCall(SliceToSpatial(u, n, local_, spatial_a_));
    PetscCall(MaskSpatial(spatial_a_, mask_, true));   // Q_C u_n
    if (n > 0) {
      PetscCall(SliceToSpatial(u, n - 1, local_, spatial_b_));
      PetscCall(MaskSpatial(spatial_b_, mask_, true)); // Q_C u_{n-1}
    }
    PetscCall(ApplyStep(spatial_a_, spatial_b_, n > 0, spatial_c_));
    PetscCall(MaskSpatial(spatial_c_, mask_, false));  // P_I
    PetscCall(SpatialToSlice(spatial_c_, n, local_, out));
  }
  return PETSC_SUCCESS;
}

PetscErrorCode DirichletControlOperators::ApplyBDT(Vec p, Vec out) {
  return ApplyBDTImpl(p, out, true);
}

PetscErrorCode DirichletControlOperators::ApplyBDTLegacy(Vec p, Vec out) {
  return ApplyBDTImpl(p, out, false);
}

/** (B_D^T P)_n = Q_C^T [ A^T P_I p_n - (M/dt) P_I p_{n+1} ]          (BE)
 *             = Q_C^T [ A_plus^T P_I p_n + A_minus^T P_I p_{n+1} ]    (CN):
 *  the exact transpose of ApplyBD (T7D derivation,
 *  boundary_convection/T7D_DIRICHLET_CONVECTION.md section 1). */
PetscErrorCode DirichletControlOperators::ApplyBDTImpl(Vec p, Vec out, bool transpose) {
  for (unsigned n = 0; n < steps_; ++n) {
    PetscCall(SliceToSpatial(p, n, local_, spatial_a_));
    PetscCall(MaskSpatial(spatial_a_, mask_, false));  // P_I p_n
    const bool have_next = (n + 1 < steps_);
    if (have_next) {
      PetscCall(SliceToSpatial(p, n + 1, local_, spatial_b_));
      PetscCall(MaskSpatial(spatial_b_, mask_, false));
    }
    PetscCall(ApplyStep(spatial_a_, spatial_b_, have_next, spatial_c_, transpose));
    PetscCall(MaskSpatial(spatial_c_, mask_, true));   // Q_C^T
    PetscCall(SpatialToSlice(spatial_c_, n, local_, out));
  }
  return PETSC_SUCCESS;
}

/** z = R_H1t^{-1} b.
 *
 *  gamma_u = 0 (the accepted fast path, untouched): R_H1t = R_T =
 *  What_u (x) R_Gamma is block diagonal, so ONE factorization of R_Gamma is
 *  applied slot by slot and scaled by 1/What_u (2 R_Gamma^{-1} on the final
 *  Crank-Nicolson slot).  Under backward Euler every weight is 1 and the
 *  scaling is a no-op.
 *
 *  gamma_u > 0: R_H1t couples consecutive slots, so the per-slot solve is
 *  wrong and the dense replicated Cholesky factor of the whole N_t|C| system
 *  is used instead -- one gather, one triangular solve pair, one scatter. */
PetscErrorCode DirichletControlOperators::SolveR(Vec b, Vec z) {
  if (gamma_ > 0.0 && sparse_core_) {
    // T8D: the same R_H1t, solved by its sparse direct factor on the
    // distributed reduced layout (no gather of N_t|C| values).
    Vec r = nullptr, x = nullptr;
    PetscCall(CreateReducedVector(&r));
    PetscCall(VecDuplicate(r, &x));
    PetscCall(GatherReduced(b, r));
    PetscCall(SolveReducedR(r, x));
    PetscCall(ScatterReduced(x, z));
    PetscCall(VecDestroy(&x));
    PetscCall(VecDestroy(&r));
    return PETSC_SUCCESS;
  }
  if (gamma_ > 0.0) {
    const PetscInt dim = static_cast<PetscInt>(steps_) * control_count_;
    std::fill(reduced_a_.begin(), reduced_a_.end(), 0.0);
    {
      const PetscScalar* a = nullptr;
      PetscCall(VecGetArrayRead(b, &a));
      for (unsigned n = 0; n < steps_; ++n)
        for (std::size_t i = 0; i < C_owned_.size(); ++i)
          reduced_a_[static_cast<std::size_t>(n) * control_count_ +
                     C_owned_position_[i]] =
              PetscRealPart(a[static_cast<PetscInt>(n) * local_ +
                              (C_owned_[i] - begin_)]);
      PetscCall(VecRestoreArrayRead(b, &a));
    }
    MPI_Allreduce(reduced_a_.data(), reduced_b_.data(), static_cast<int>(dim),
                  MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
    PetscBLASInt bdim = 0, one = 1, info = 0;
    PetscCall(PetscBLASIntCast(dim, &bdim));
    LAPACKpotrs_("L", &bdim, &one, Q_dense_.data(), &bdim, reduced_b_.data(),
                 &bdim, &info);
    if (info != 0)
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_LIB,
              "Dirichlet control: dense triangular solve failed (info %d)",
              (int)info);
    PetscCall(VecSet(z, 0.0));
    PetscScalar* d = nullptr;
    PetscCall(VecGetArray(z, &d));
    for (unsigned n = 0; n < steps_; ++n)
      for (std::size_t i = 0; i < C_owned_.size(); ++i)
        d[static_cast<PetscInt>(n) * local_ + (C_owned_[i] - begin_)] =
            reduced_b_[static_cast<std::size_t>(n) * control_count_ +
                       C_owned_position_[i]];
    PetscCall(VecRestoreArray(z, &d));
    return PETSC_SUCCESS;
  }
  for (unsigned n = 0; n < steps_; ++n) {
    PetscCall(SliceToSpatial(b, n, local_, spatial_a_));
    PetscCall(MaskSpatial(spatial_a_, mask_, true));
    PetscCall(KSPSolve(R_solver_, spatial_a_, spatial_b_));
    PetscCall(MaskSpatial(spatial_b_, mask_, true));
    const double weight = SlotWeight(n);
    if (weight != 1.0) PetscCall(VecScale(spatial_b_, 1.0 / weight));
    PetscCall(SpatialToSlice(spatial_b_, n, local_, z));
  }
  return PETSC_SUCCESS;
}

/** out = R_H1t u = (What_u (x) R_Gamma) u + (rho/dt) (T (x) M_Gamma) u.
 *  The temporal term is applied matrix-free, row by row of T:
 *      (T u)_n = T_nn u_n - u_{n-1} - u_{n+1},
 *  with the missing neighbours dropped and T_nn = 2 except on the LAST slot,
 *  where it is 1.  With gamma_u = 0 the second half is skipped entirely and
 *  the accepted block-diagonal product is what runs. */
PetscErrorCode DirichletControlOperators::ApplyR(Vec u, Vec out) {
  for (unsigned n = 0; n < steps_; ++n) {
    PetscCall(SliceToSpatial(u, n, local_, spatial_a_));
    PetscCall(MaskSpatial(spatial_a_, mask_, true));
    PetscCall(MatMult(R_, spatial_a_, spatial_b_));
    PetscCall(MaskSpatial(spatial_b_, mask_, true));
    const double weight = SlotWeight(n);
    if (weight != 1.0) PetscCall(VecScale(spatial_b_, weight));
    if (gamma_ > 0.0) {
      PetscCall(VecCopy(spatial_a_, spatial_c_));
      PetscCall(VecScale(spatial_c_, TemporalDiagonal(n)));
      for (int neighbour = -1; neighbour <= 1; neighbour += 2) {
        const int slot = static_cast<int>(n) + neighbour;
        if (slot < 0 || slot >= static_cast<int>(steps_)) continue;
        PetscCall(SliceToSpatial(u, static_cast<unsigned>(slot), local_,
                                 spatial_d_));
        PetscCall(MaskSpatial(spatial_d_, mask_, true));
        PetscCall(VecAXPY(spatial_c_, -1.0, spatial_d_));
      }
      PetscCall(MatMult(M_face_, spatial_c_, spatial_d_));
      PetscCall(MaskSpatial(spatial_d_, mask_, true));
      PetscCall(VecAXPY(spatial_b_, rho_ / dt_, spatial_d_));
    }
    PetscCall(SpatialToSlice(spatial_b_, n, local_, out));
  }
  return PETSC_SUCCESS;
}

PetscErrorCode DirichletControlOperators::ApplyFaceMass(Vec v, Vec out) {
  if (gamma_ <= 0.0)
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ORDER,
            "Dirichlet control: M_Gamma is built only when gamma_u > 0");
  PetscCall(VecCopy(v, spatial_a_));
  PetscCall(MaskSpatial(spatial_a_, mask_, true));
  PetscCall(MatMult(M_face_, spatial_a_, out));
  PetscCall(MaskSpatial(out, mask_, true));
  return PETSC_SUCCESS;
}

/** q_0 = (1/dt) [M_Gamma u_0; 0; ...; 0]: the linear term the P1-in-time
 *  energy leaves behind because u_0 = y_0|_Gamma_c/eta is DATA and not an
 *  unknown.  Omitting it would implement the Hessian of the temporal
 *  seminorm without its gradient, i.e. a different functional. */
PetscErrorCode DirichletControlOperators::ApplyAffineData(Vec initial_trace,
                                                          Vec q0) {
  PetscCall(VecSet(q0, 0.0));
  if (gamma_ <= 0.0) return PETSC_SUCCESS;
  PetscCall(ApplyFaceMass(initial_trace, spatial_b_));
  PetscCall(VecScale(spatial_b_, 1.0 / dt_));
  PetscCall(SpatialToSlice(spatial_b_, 0, local_, q0));
  return PETSC_SUCCESS;
}

/** out = rho R_H1t^{-1} q_0 = Q_u^{-1} (gamma_u q_0). */
PetscErrorCode DirichletControlOperators::AffineControl(Vec initial_trace,
                                                        Vec out) {
  PetscCall(VecSet(out, 0.0));
  if (gamma_ <= 0.0) return PETSC_SUCCESS;
  PetscCall(ApplyAffineData(initial_trace, work2_));
  PetscCall(SolveR(work2_, out));
  PetscCall(VecScale(out, rho_));
  return PETSC_SUCCESS;
}

PetscErrorCode DirichletControlOperators::AddAffineControl(Vec initial_trace,
                                                           Vec u) {
  if (gamma_ <= 0.0) return PETSC_SUCCESS;
  PetscCall(AffineControl(initial_trace, work_));
  PetscCall(VecAXPY(u, 1.0, work_));
  return PETSC_SUCCESS;
}

PetscErrorCode DirichletControlOperators::ApplyAffineStateCorrection(
    Vec initial_trace, Vec out) {
  PetscCall(VecSet(out, 0.0));
  if (gamma_ <= 0.0) return PETSC_SUCCESS;
  PetscCall(AffineControl(initial_trace, work_));
  PetscCall(ApplyBD(work_, out));
  return PETSC_SUCCESS;
}

PetscErrorCode DirichletControlOperators::ApplyND(double scale, Vec x, Vec out) {
  PetscCall(ApplyBDT(x, work_));
  PetscCall(SolveR(work_, out));
  PetscCall(VecCopy(out, work_));
  PetscCall(ApplyBD(work_, out));
  PetscCall(VecScale(out, scale));
  return PETSC_SUCCESS;
}

PetscErrorCode DirichletControlOperators::ReconstructControl(double factor,
                                                             Vec p, Vec u) {
  PetscCall(ApplyBDT(p, work_));
  PetscCall(SolveR(work_, u));
  PetscCall(VecScale(u, factor));
  return PETSC_SUCCESS;
}

PetscErrorCode DirichletControlOperators::ZeroControlledRows(Vec v) {
  PetscScalar* a = nullptr;
  PetscCall(VecGetArray(v, &a));
  for (unsigned n = 0; n < steps_; ++n) {
    const PetscInt offset = static_cast<PetscInt>(n) * local_;
    for (PetscInt i = 0; i < local_; ++i)
      if (mask_[static_cast<std::size_t>(i)]) a[offset + i] = 0.0;
  }
  PetscCall(VecRestoreArray(v, &a));
  return PETSC_SUCCESS;
}

PetscErrorCode ApplyDirichletControlShell(Mat shell, Vec x, Vec y) {
  DirichletControlShellContext* context = nullptr;
  PetscCall(MatShellGetContext(shell, &context));
  if (!context || !context->operators) return PETSC_ERR_ARG_NULL;
  PetscCall(context->operators->ApplyND(context->scale, x, y));
  return PETSC_SUCCESS;
}

// ===========================================================================
// T8D: the distributed reduced control space and the sparse exact R_H1t.
// ===========================================================================

PetscErrorCode DirichletControlOperators::CreateReducedVector(Vec* v) const {
  PetscCall(VecCreateMPI(PETSC_COMM_WORLD, reduced_local(), PETSC_DECIDE, v));
  return PETSC_SUCCESS;
}

PetscErrorCode DirichletControlOperators::GatherReduced(Vec space_time, Vec reduced) const {
  const PetscScalar* a = nullptr;
  PetscScalar* r = nullptr;
  const PetscInt owned = static_cast<PetscInt>(C_owned_.size());
  PetscCall(VecGetArrayRead(space_time, &a));
  PetscCall(VecGetArray(reduced, &r));
  for (unsigned n = 0; n < steps_; ++n)
    for (PetscInt i = 0; i < owned; ++i)
      r[static_cast<PetscInt>(n) * owned + i] =
          a[static_cast<PetscInt>(n) * local_ + (C_owned_[i] - begin_)];
  PetscCall(VecRestoreArray(reduced, &r));
  PetscCall(VecRestoreArrayRead(space_time, &a));
  return PETSC_SUCCESS;
}

PetscErrorCode DirichletControlOperators::ScatterReduced(Vec reduced, Vec space_time) const {
  const PetscScalar* r = nullptr;
  PetscScalar* a = nullptr;
  const PetscInt owned = static_cast<PetscInt>(C_owned_.size());
  PetscCall(VecSet(space_time, 0.0));
  PetscCall(VecGetArrayRead(reduced, &r));
  PetscCall(VecGetArray(space_time, &a));
  for (unsigned n = 0; n < steps_; ++n)
    for (PetscInt i = 0; i < owned; ++i)
      a[static_cast<PetscInt>(n) * local_ + (C_owned_[i] - begin_)] =
          r[static_cast<PetscInt>(n) * owned + i];
  PetscCall(VecRestoreArray(space_time, &a));
  PetscCall(VecRestoreArrayRead(reduced, &r));
  return PETSC_SUCCESS;
}

Mat DirichletControlOperators::ReducedRegularization() {
  if (!RT_) BuildReducedRegularization();
  return RT_;
}

/** R_H1t = What_u (x) R_Gamma + (rho/dt) T (x) M_Gamma assembled SPARSE on the
 *  reduced layout: global index of (slot n, C DOF owned by rank r at owned
 *  position i) = offset_r + n |C_r| + i.  The only replicated data are the
 *  |C|-long position maps.  The C x C couplings are read from the embedded
 *  R_ (and M_face_ when gamma_u > 0), rows owned locally, so no dense block is
 *  needed; T = tridiag(-1, 2, -1) with 1 on the last slot, as in the dense
 *  BuildTemporalRegularization. */
void DirichletControlOperators::BuildReducedRegularization() {
  if (C_global_.empty() && control_count_ > 0) BuildGlobalControlList();
  int ranks = 1, rank = 0;
  MPI_Comm_size(PETSC_COMM_WORLD, &ranks);
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  const PetscInt owned = static_cast<PetscInt>(C_owned_.size());
  std::vector<PetscInt> counts(ranks), first(ranks + 1, 0);
  MPI_Allgather(&owned, 1, MPIU_INT, counts.data(), 1, MPIU_INT, PETSC_COMM_WORLD);
  std::vector<PetscInt> offset(ranks + 1, 0);
  for (int r = 0; r < ranks; ++r) {
    first[r + 1] = first[r] + counts[r];                    // C positions of rank r
    offset[r + 1] = offset[r] + static_cast<PetscInt>(steps_) * counts[r];
  }
  // position p in C_global_ -> (owner rank, owned index); ranks own contiguous
  // ascending position ranges because ownership ranges are ascending.
  auto global_index = [&](unsigned n, PetscInt p) -> PetscInt {
    int r = static_cast<int>(std::upper_bound(first.begin(), first.end(), p) - first.begin()) - 1;
    return offset[r] + static_cast<PetscInt>(n) * counts[r] + (p - first[r]);
  };
  auto position = [&](PetscInt g) -> PetscInt {
    auto it = std::lower_bound(C_global_.begin(), C_global_.end(), g);
    return (it == C_global_.end() || *it != g) ? -1
                                               : static_cast<PetscInt>(it - C_global_.begin());
  };
  const PetscInt local = reduced_local();
  PetscCallAbort(PETSC_COMM_WORLD, MatCreate(PETSC_COMM_WORLD, &RT_));
  PetscCallAbort(PETSC_COMM_WORLD, MatSetSizes(RT_, local, local, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCallAbort(PETSC_COMM_WORLD, MatSetType(RT_, MATAIJ));
  const double coupling = gamma_ > 0.0 ? rho_ / dt_ : 0.0;
  // T11D: preallocation from the actual trace stencil.  A 1D (EDGE3) trace row
  // couples 5 DOFs, a QUAD9 face row 25; the temporal coupling triples it.
  // Never below the historical 24 (2D: identical allocation; values never
  // depend on it).
  PetscInt row_max = 0;
  for (PetscInt i = 0; i < owned; ++i)
    for (int which = 0; which < (coupling > 0.0 ? 2 : 1); ++which) {
      PetscInt ncols = 0;
      PetscCallAbort(PETSC_COMM_WORLD,
                     MatGetRow(which == 0 ? R_ : M_face_, C_owned_[i], &ncols, nullptr, nullptr));
      row_max = std::max(row_max, ncols);
      PetscCallAbort(PETSC_COMM_WORLD,
                     MatRestoreRow(which == 0 ? R_ : M_face_, C_owned_[i], &ncols, nullptr, nullptr));
    }
  PetscInt row_max_global = 0;
  MPI_Allreduce(&row_max, &row_max_global, 1, MPIU_INT, MPI_MAX, PETSC_COMM_WORLD);
  const PetscInt prealloc = std::max<PetscInt>(24, (coupling > 0.0 ? 3 : 1) * row_max_global);
  PetscCallAbort(PETSC_COMM_WORLD, MatSeqAIJSetPreallocation(RT_, prealloc, nullptr));
  PetscCallAbort(PETSC_COMM_WORLD, MatMPIAIJSetPreallocation(RT_, prealloc, nullptr, prealloc, nullptr));
  for (PetscInt i = 0; i < owned; ++i) {
    const PetscInt prow = C_owned_position_.empty()
        ? position(C_owned_[i]) : C_owned_position_[i];
    // gather the C columns of R_Gamma and M_Gamma for this row
    std::vector<std::pair<PetscInt, double>> rcols, mcols;
    for (int which = 0; which < (coupling > 0.0 ? 2 : 1); ++which) {
      Mat source = which == 0 ? R_ : M_face_;
      PetscInt ncols = 0;
      const PetscInt* cols = nullptr;
      const PetscScalar* vals = nullptr;
      PetscCallAbort(PETSC_COMM_WORLD, MatGetRow(source, C_owned_[i], &ncols, &cols, &vals));
      for (PetscInt j = 0; j < ncols; ++j) {
        const PetscInt p = position(cols[j]);
        if (p < 0 || PetscRealPart(vals[j]) == 0.0) continue;
        (which == 0 ? rcols : mcols).push_back({p, PetscRealPart(vals[j])});
      }
      PetscCallAbort(PETSC_COMM_WORLD, MatRestoreRow(source, C_owned_[i], &ncols, &cols, &vals));
    }
    for (unsigned n = 0; n < steps_; ++n) {
      const PetscInt row = global_index(n, prow);
      const double w = SlotWeight(n);
      for (const auto& e : rcols)
        PetscCallAbort(PETSC_COMM_WORLD,
                       MatSetValue(RT_, row, global_index(n, e.first), w * e.second, ADD_VALUES));
      if (coupling > 0.0)
        for (const auto& e : mcols) {
          PetscCallAbort(PETSC_COMM_WORLD,
                         MatSetValue(RT_, row, global_index(n, e.first),
                                     coupling * TemporalDiagonal(n) * e.second, ADD_VALUES));
          if (n > 0)
            PetscCallAbort(PETSC_COMM_WORLD,
                           MatSetValue(RT_, row, global_index(n - 1, e.first),
                                       -coupling * e.second, ADD_VALUES));
          if (n + 1 < steps_)
            PetscCallAbort(PETSC_COMM_WORLD,
                           MatSetValue(RT_, row, global_index(n + 1, e.first),
                                       -coupling * e.second, ADD_VALUES));
        }
    }
  }
  PetscCallAbort(PETSC_COMM_WORLD, MatAssemblyBegin(RT_, MAT_FINAL_ASSEMBLY));
  PetscCallAbort(PETSC_COMM_WORLD, MatAssemblyEnd(RT_, MAT_FINAL_ASSEMBLY));
  PetscCallAbort(PETSC_COMM_WORLD, MatSetOption(RT_, MAT_SYMMETRIC, PETSC_TRUE));
  PetscCallAbort(PETSC_COMM_WORLD, MatSetOption(RT_, MAT_SPD, PETSC_TRUE));
  RT_solver_ = CreateDirichletDirectSolver(RT_, "dirichlet_core_rt_", &RT_backend_);
}

PetscErrorCode DirichletControlOperators::SolveReducedR(Vec r, Vec z) {
  if (!RT_) BuildReducedRegularization();
  PetscCall(KSPSolve(RT_solver_, r, z));
  return PETSC_SUCCESS;
}
