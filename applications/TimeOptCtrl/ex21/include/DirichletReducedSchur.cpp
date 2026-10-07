// Exact Schur solve for Dirichlet control; see DirichletReducedSchur.hpp.

#include "DirichletReducedSchur.hpp"

#include <petscblaslapack.h>
#include <petscksp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "Config.hpp"
#include "DirichletControl.hpp"
#include "Preconditioners.hpp"

DirichletReducedSchurOperators::DirichletReducedSchurOperators(
    const ProblemConfig& config, DirichletControlOperators& dirichlet,
    Mat mass, Mat stiffness, Mat tracking)
    : dirichlet_(dirichlet), scheme_(config.time.scheme), tracking_(tracking),
      mass_(mass), steps_(config.time.steps), dt_(config.time.dt),
      convection_(HasConvection(config)) {
  const double t0 = MPI_Wtime();
  core_config_ = config.dirichlet_core;
  iterative_ = core_config_.solver == ProblemConfig::DirichletCoreConfig::Solver::ITERATIVE;
  verify_ = core_config_.verify;
  PetscCallAbort(PETSC_COMM_WORLD, MatGetSize(mass, &n_, nullptr));
  PetscCallAbort(PETSC_COMM_WORLD, MatGetLocalSize(mass, &local_, nullptr));
  PetscCallAbort(PETSC_COMM_WORLD, MatGetOwnershipRange(mass, &begin_, nullptr));
  if (dirichlet.spatial_local() != local_ || dirichlet.steps() != steps_)
    throw std::invalid_argument(
        "DirichletReducedSchurOperators: layout mismatch with the control operators");

  // The replicated sorted C list and the positions of the owned C DOFs in it.
  {
    int ranks = 1;
    MPI_Comm_size(PETSC_COMM_WORLD, &ranks);
    const std::vector<PetscInt>& owned = dirichlet.owned_control_dofs();
    std::vector<int> counts(ranks), displacements(ranks);
    int mine = static_cast<int>(owned.size());
    MPI_Allgather(&mine, 1, MPI_INT, counts.data(), 1, MPI_INT, PETSC_COMM_WORLD);
    int total = 0;
    for (int r = 0; r < ranks; ++r) { displacements[r] = total; total += counts[r]; }
    C_global_.assign(static_cast<std::size_t>(total), 0);
    MPI_Allgatherv(owned.data(), mine, MPIU_INT, C_global_.data(), counts.data(),
                   displacements.data(), MPIU_INT, PETSC_COMM_WORLD);
    std::sort(C_global_.begin(), C_global_.end());
    n_C_ = total;
    for (PetscInt g : owned)
      C_owned_position_.push_back(static_cast<PetscInt>(
          std::lower_bound(C_global_.begin(), C_global_.end(), g) - C_global_.begin()));
  }

  BuildOperators(mass, stiffness);
  PetscCallAbort(PETSC_COMM_WORLD, MatCreateVecs(mass, &spatial_a_, nullptr));
  PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(spatial_a_, &spatial_b_));
  PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(spatial_a_, &spatial_c_));
  PetscCallAbort(PETSC_COMM_WORLD,
                 VecCreateMPI(PETSC_COMM_WORLD, static_cast<PetscInt>(steps_) * local_,
                              PETSC_DECIDE, &w_));
  for (Vec* v : {&t_, &s_, &q_})
    PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(w_, v));
  if (!iterative_) {
    // the dense reference: responses (N_t local |C|), Gram blocks
    // (N_t(N_t+1)/2 |C|^2, replicated) and, in Prepare, the dense K ((N_t|C|)^2)
    reduced_a_.assign(static_cast<std::size_t>(reduced_dimension()), 0.0);
    reduced_b_.assign(reduced_a_.size(), 0.0);
    BuildResponses();
    BuildGram();
    BuildRegularizationBlock();
  }
  if (iterative_ || verify_) BuildCoreSolver();
  setup_seconds_ = MPI_Wtime() - t0;
}

DirichletReducedSchurOperators::~DirichletReducedSchurOperators() {
  // The two sweep contexts share the spatial solver: each adopted one
  // reference, so both destructions are needed and neither frees it twice.
  DestroyTimeSweepContext(backward_);
  DestroyTimeSweepContext(forward_);
  for (Vec* v : {&q_, &s_, &t_, &w_, &spatial_c_, &spatial_b_, &spatial_a_})
    PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(v));
  PetscCallAbort(PETSC_COMM_WORLD, KSPDestroy(&core_ksp_));
  PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&core_shell_));
  for (Vec* v : {&core_r_, &core_z_, &core_tmp_, &k_a_, &k_b_})
    PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(v));
  PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&coupling_transpose_));
  PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&A_transpose_));
  PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&coupling_));
  PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&A_minus_));
  PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&A_));
}

/** A_I = M_I/dt + K_I, its direct factor, the forward and backward sweeps. */
void DirichletReducedSchurOperators::BuildOperators(Mat mass, Mat stiffness) {
  // Backward Euler: A_I = M_I/dt + K_I, sweep coupling (mass_, 1/dt).
  // Crank-Nicolson: A_I = Aplus_I = M_I/dt + K_I/2, Aminus_I = -M_I/dt + K_I/2
  // and the sweep couples through the owned -Aminus_I with unit scale.
  const bool crank_nicolson =
      scheme_ == ProblemConfig::TimeScheme::CRANK_NICOLSON;
  PetscCallAbort(PETSC_COMM_WORLD, MatDuplicate(stiffness, MAT_COPY_VALUES, &A_));
  if (crank_nicolson)
    PetscCallAbort(PETSC_COMM_WORLD, MatScale(A_, 0.5));
  PetscCallAbort(PETSC_COMM_WORLD, MatAXPY(A_, 1.0 / dt_, mass, DIFFERENT_NONZERO_PATTERN));
  if (crank_nicolson) {
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatDuplicate(stiffness, MAT_COPY_VALUES, &A_minus_));
    PetscCallAbort(PETSC_COMM_WORLD, MatScale(A_minus_, 0.5));
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatAXPY(A_minus_, -1.0 / dt_, mass, DIFFERENT_NONZERO_PATTERN));
    // The hatted matrices carry the ELIMINATION on C: K_I has 1 on the C
    // diagonal and M_I has 0, so that backward Euler's A = M_I/dt + K_I has
    // identity C rows and its coupling M_I has zero C rows -- that is what
    // makes the embedded space-time operator E_I leave the C components
    // untouched.  Halving K_I destroys both properties (1/2 instead of 1 and
    // instead of 0), so the two Crank-Nicolson blocks must be re-embedded
    // explicitly: A_plus with identity C rows, A_minus with zero C rows.
    const std::vector<PetscInt>& owned = dirichlet_.owned_control_dofs();
    const PetscInt count = static_cast<PetscInt>(owned.size());
    for (Mat m : {A_, A_minus_})
      PetscCallAbort(PETSC_COMM_WORLD,
                     MatSetOption(m, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE));
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatZeroRowsColumns(A_, count, owned.empty() ? nullptr : owned.data(),
                                      1.0, nullptr, nullptr));
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatZeroRowsColumns(A_minus_, count,
                                      owned.empty() ? nullptr : owned.data(),
                                      0.0, nullptr, nullptr));
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatDuplicate(A_minus_, MAT_COPY_VALUES, &coupling_));
    PetscCallAbort(PETSC_COMM_WORLD, MatScale(coupling_, -1.0));
    if (!convection_)
      for (Mat m : {A_minus_, coupling_})
        PetscCallAbort(PETSC_COMM_WORLD, MatSetOption(m, MAT_SYMMETRIC, PETSC_TRUE));
  }
  Mat sweep_coupling = crank_nicolson ? coupling_ : mass_;
  const double sweep_scale = crank_nicolson ? 1.0 : 0.0;   // 0 selects 1/dt
  if (convection_) {
    // T7D: A_I = M_I/dt + L_I (Aplus_I) is NOT symmetric.  The forward sweep
    // (E_I) factors A_I by LU; the backward sweep (E_I^T) runs on A_I^T with the
    // transposed coupling (BE: the hatted mass, symmetric; CN: -Aminus_I^T) and
    // its own LU factor.  No symmetry flag is set.
    // one reference each: spatial_ is adopted by the forward sweep only,
    // spatial_transpose_ by the backward sweep only
    spatial_ = CreateDirichletDirectSolver(A_, "dirichlet_reduced_a_", &backend_, true);
    PetscCallAbort(PETSC_COMM_WORLD, MatTranspose(A_, MAT_INITIAL_MATRIX, &A_transpose_));
    std::string backend_t;
    spatial_transpose_ = CreateDirichletDirectSolver(A_transpose_, "dirichlet_reduced_at_",
                                                     &backend_t, true);
    Mat backward_coupling = sweep_coupling;
    if (crank_nicolson) {
      PetscCallAbort(PETSC_COMM_WORLD,
                     MatTranspose(coupling_, MAT_INITIAL_MATRIX, &coupling_transpose_));
      backward_coupling = coupling_transpose_;
    }
    InitializeTimeSweepContextWithSpatialSolver(forward_, TimeSweepDirection::FORWARD,
                                                A_, sweep_coupling, steps_, dt_,
                                                "dirichlet-forward", spatial_,
                                                sweep_scale);
    InitializeTimeSweepContextWithSpatialSolver(backward_, TimeSweepDirection::BACKWARD,
                                                A_transpose_, backward_coupling, steps_, dt_,
                                                "dirichlet-backward", spatial_transpose_,
                                                sweep_scale);
    return;
  }
  PetscCallAbort(PETSC_COMM_WORLD, MatSetOption(A_, MAT_SYMMETRIC, PETSC_TRUE));
  PetscCallAbort(PETSC_COMM_WORLD, MatSetOption(A_, MAT_SPD, PETSC_TRUE));
  spatial_ = CreateDirichletDirectSolver(A_, "dirichlet_reduced_a_", &backend_);
  PetscCallAbort(PETSC_COMM_WORLD,
                 PetscObjectReference(reinterpret_cast<PetscObject>(spatial_)));
  InitializeTimeSweepContextWithSpatialSolver(forward_, TimeSweepDirection::FORWARD,
                                              A_, sweep_coupling, steps_, dt_,
                                              "dirichlet-forward", spatial_,
                                              sweep_scale);
  InitializeTimeSweepContextWithSpatialSolver(backward_, TimeSweepDirection::BACKWARD,
                                              A_, sweep_coupling, steps_, dt_,
                                              "dirichlet-backward", spatial_,
                                              sweep_scale);
}

/** Backward Euler: y^c_1 = A_I^{-1} g_c, y^c_2 = A_I^{-1}((M_I/dt) y^c_1 - h_c),
 *  y^c_k = A_I^{-1} (M_I/dt) y^c_{k-1}, with g_c = P_I A Q_C e_c and
 *  h_c = P_I (M/dt) Q_C e_c on the unhatted matrices.
 *  Crank-Nicolson: y^c_1 = Aplus^{-1} g_c,
 *  y^c_2 = Aplus^{-1}(h_c - Aminus y^c_1), y^c_k = Aplus^{-1}(-Aminus y^c_{k-1}),
 *  with g_c = P_I A_plus Q_C e_c and h_c = P_I A_minus Q_C e_c -- the SAME
 *  formulas with (A, -M/dt) -> (A_plus, A_minus); note the lag-1 term enters
 *  with a PLUS because B_D_CN's sub-diagonal block is Gminus = (A_minus)_IC
 *  itself, where the backward-Euler one is -(M/dt)_IC.  The free-state
 *  response to the unit control of face DOF c at slot 1. */
void DirichletReducedSchurOperators::BuildResponses() {
  const bool crank_nicolson =
      scheme_ == ProblemConfig::TimeScheme::CRANK_NICOLSON;
  responses_.assign(steps_, std::vector<double>(
      static_cast<std::size_t>(local_) * n_C_, 0.0));
  const std::vector<unsigned char>& mask = dirichlet_.control_mask();
  Mat M_full = dirichlet_.unhatted_mass(), K_full = dirichlet_.unhatted_stiffness();
  Vec e = nullptr, g = nullptr, h = nullptr, y = nullptr, prev = nullptr;
  for (Vec* v : {&e, &g, &h, &y, &prev})
    PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(spatial_a_, v));
  auto zero_C = [&](Vec v) {
    PetscScalar* a = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD, VecGetArray(v, &a));
    for (PetscInt i = 0; i < local_; ++i) if (mask[i]) a[i] = 0.0;
    PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArray(v, &a));
  };
  auto store = [&](Vec v, unsigned k, PetscInt c) {
    const PetscScalar* a = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(v, &a));
    double* column = responses_[k].data() + static_cast<std::size_t>(c) * local_;
    for (PetscInt i = 0; i < local_; ++i) column[i] = PetscRealPart(a[i]);
    PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(v, &a));
  };
  for (PetscInt c = 0; c < n_C_; ++c) {
    PetscCallAbort(PETSC_COMM_WORLD, VecSet(e, 0.0));
    PetscCallAbort(PETSC_COMM_WORLD, VecSetValue(e, C_global_[c], 1.0, INSERT_VALUES));
    PetscCallAbort(PETSC_COMM_WORLD, VecAssemblyBegin(e));
    PetscCallAbort(PETSC_COMM_WORLD, VecAssemblyEnd(e));
    if (crank_nicolson) {
      // g = P_I A_plus Q_C e_c = P_I (M/dt + K/2) e_c
      // h = P_I A_minus Q_C e_c = P_I (-M/dt + K/2) e_c
      PetscCallAbort(PETSC_COMM_WORLD, MatMult(M_full, e, spatial_c_));
      PetscCallAbort(PETSC_COMM_WORLD, MatMult(K_full, e, g));
      PetscCallAbort(PETSC_COMM_WORLD, VecScale(g, 0.5));
      PetscCallAbort(PETSC_COMM_WORLD, VecCopy(g, h));
      PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(g, 1.0 / dt_, spatial_c_));
      PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(h, -1.0 / dt_, spatial_c_));
    } else {
      PetscCallAbort(PETSC_COMM_WORLD, MatMult(M_full, e, h));
      PetscCallAbort(PETSC_COMM_WORLD, VecScale(h, 1.0 / dt_));
      PetscCallAbort(PETSC_COMM_WORLD, MatMult(K_full, e, g));
      PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(g, 1.0, h));
    }
    zero_C(g);
    zero_C(h);
    for (unsigned k = 0; k < steps_; ++k) {
      if (k == 0) {
        PetscCallAbort(PETSC_COMM_WORLD, VecCopy(g, spatial_b_));
      } else if (crank_nicolson) {
        PetscCallAbort(PETSC_COMM_WORLD, MatMult(coupling_, prev, spatial_b_));
        if (k == 1) PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(spatial_b_, 1.0, h));
      } else {
        PetscCallAbort(PETSC_COMM_WORLD, MatMult(mass_, prev, spatial_b_));
        PetscCallAbort(PETSC_COMM_WORLD, VecScale(spatial_b_, 1.0 / dt_));
        if (k == 1) PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(spatial_b_, -1.0, h));
      }
      PetscCallAbort(PETSC_COMM_WORLD, KSPSolve(spatial_, spatial_b_, y));
      store(y, k, c);
      PetscCallAbort(PETSC_COMM_WORLD, VecCopy(y, prev));
    }
  }
  for (Vec* v : {&prev, &y, &h, &g, &e})
    PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(v));
}

/** Gamma_ab = Y_a^T M_w Y_b for a >= b (replicated |C| x |C| blocks). */
void DirichletReducedSchurOperators::BuildGram() {
  const std::size_t block = static_cast<std::size_t>(n_C_) * n_C_;
  gram_.assign(static_cast<std::size_t>(steps_) * (steps_ + 1) / 2,
               std::vector<double>(block, 0.0));
  std::vector<double> Z(static_cast<std::size_t>(local_) * n_C_), partial(block);
  PetscBLASInt bn_C = 0, blocal = 0;
  PetscCallAbort(PETSC_COMM_WORLD, PetscBLASIntCast(n_C_, &bn_C));
  PetscCallAbort(PETSC_COMM_WORLD, PetscBLASIntCast(local_, &blocal));
  const double one = 1.0, zero = 0.0;
  for (unsigned b = 0; b < steps_; ++b) {
    // Z = M_w Y_b, column by column.
    for (PetscInt c = 0; c < n_C_; ++c) {
      PetscScalar* a = nullptr;
      PetscCallAbort(PETSC_COMM_WORLD, VecGetArray(spatial_a_, &a));
      const double* column = responses_[b].data() + static_cast<std::size_t>(c) * local_;
      for (PetscInt i = 0; i < local_; ++i) a[i] = column[i];
      PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArray(spatial_a_, &a));
      PetscCallAbort(PETSC_COMM_WORLD, MatMult(tracking_, spatial_a_, spatial_b_));
      const PetscScalar* z = nullptr;
      PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(spatial_b_, &z));
      double* out = Z.data() + static_cast<std::size_t>(c) * local_;
      for (PetscInt i = 0; i < local_; ++i) out[i] = PetscRealPart(z[i]);
      PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(spatial_b_, &z));
    }
    for (unsigned a = b; a < steps_; ++a) {
      std::vector<double>& target = gram_[static_cast<std::size_t>(a) * (a + 1) / 2 + b];
      if (local_ > 0) {
        BLASgemm_("T", "N", &bn_C, &bn_C, &blocal, &one, responses_[a].data(), &blocal,
                  Z.data(), &blocal, &zero, partial.data(), &bn_C);
      } else {
        std::fill(partial.begin(), partial.end(), 0.0);
      }
      MPI_Allreduce(partial.data(), target.data(), static_cast<int>(block), MPI_DOUBLE,
                    MPI_SUM, PETSC_COMM_WORLD);
    }
  }
}

/** R_Gamma restricted to C, replicated dense (from the embedded R). */
void DirichletReducedSchurOperators::BuildRegularizationBlock() {
  const std::size_t block = static_cast<std::size_t>(n_C_) * n_C_;
  std::vector<double> partial(block, 0.0);
  R_dense_.assign(block, 0.0);
  Mat R = dirichlet_.regularization();
  const std::vector<PetscInt>& owned = dirichlet_.owned_control_dofs();
  for (std::size_t i = 0; i < owned.size(); ++i) {
    PetscInt ncols = 0;
    const PetscInt* cols = nullptr;
    const PetscScalar* vals = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD, MatGetRow(R, owned[i], &ncols, &cols, &vals));
    const PetscInt row = C_owned_position_[i];
    for (PetscInt j = 0; j < ncols; ++j) {
      auto it = std::lower_bound(C_global_.begin(), C_global_.end(), cols[j]);
      if (it == C_global_.end() || *it != cols[j]) continue;
      const PetscInt col = static_cast<PetscInt>(it - C_global_.begin());
      partial[static_cast<std::size_t>(col) * n_C_ + row] += PetscRealPart(vals[j]);
    }
    PetscCallAbort(PETSC_COMM_WORLD, MatRestoreRow(R, owned[i], &ncols, &cols, &vals));
  }
  MPI_Allreduce(partial.data(), R_dense_.data(), static_cast<int>(block), MPI_DOUBLE,
                MPI_SUM, PETSC_COMM_WORLD);
}

PetscErrorCode DirichletReducedSchurOperators::Prepare(double beta) {
  if (beta == prepared_beta_) return PETSC_SUCCESS;
  if (iterative_) {   // T8D: nothing to assemble or factor; the shell reads beta
    prepared_beta_ = beta;
    return PETSC_SUCCESS;
  }
  const PetscInt dim = reduced_dimension();
  AssembleDenseK(beta, K_);
  if (verify_) K_unfactored_ = K_;
  PetscBLASInt bdim = 0, info = 0;
  PetscCall(PetscBLASIntCast(dim, &bdim));
  LAPACKpotrf_("L", &bdim, K_.data(), &bdim, &info);
  if (info != 0)
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_LIB,
            "Dirichlet reduced Schur: dense Cholesky failed (info %d)", (int)info);
  prepared_beta_ = beta;
  return PETSC_SUCCESS;
}

/** K = R_H1t + beta H, dense (N_t|C|)^2 column-major, both triangles. */
void DirichletReducedSchurOperators::AssembleDenseK(double beta,
                                                    std::vector<double>& K) const {
  const PetscInt dim = reduced_dimension();
  K.assign(static_cast<std::size_t>(dim) * dim, 0.0);
  const std::size_t block = static_cast<std::size_t>(n_C_) * n_C_;
  std::vector<double> H(block);
  for (unsigned m = 0; m < steps_; ++m) {
    for (unsigned mp = m; mp < steps_; ++mp) {
      // H_{m,m'} = sum_{j=0}^{N_t-m'-1} what_y(j+m') Gamma_{j+d, j}, d = m'-m.
      // The TRACKING weight belongs to the summation index j+m' -- the state
      // slot the Gram block is evaluated at -- not to m or m', so it cannot be
      // folded into Gamma and is applied here.  It is 1 for every backward-
      // Euler slot, so that path is unchanged.
      const unsigned d = mp - m;
      std::fill(H.begin(), H.end(), 0.0);
      for (unsigned j = 0; j + mp < steps_; ++j) {
        const std::vector<double>& G =
            gram_[static_cast<std::size_t>(j + d) * (j + d + 1) / 2 + j];
        const double weight = SlotWeight(j + mp);
        for (std::size_t k = 0; k < block; ++k) H[k] += weight * G[k];
      }
      // The CONTROL weight What_u enters only the block-diagonal term, which
      // is R_T = What_u (x) R_Gamma.  Conceptually distinct from What_y above,
      // numerically the same 1/2 on the last slot.
      const double control_weight = dirichlet_.SlotWeight(m);
      // TEMPORAL regularization (gamma_u > 0 only).  Q_u = alpha_u R_H1t with
      // R_H1t = What_u (x) R_Gamma + (rho/dt) T (x) M_Gamma, so the Woodbury
      // matrix is K = R_H1t + beta H: the SAME dense object, with the tridiagonal
      // T (x) M_Gamma added on the d = 0 and d = 1 blocks.  rho = 0 leaves K
      // exactly as stages 1-3 built it.
      const double rho = dirichlet_.time_ratio();
      const std::vector<double>& M_face = dirichlet_.face_mass_block();
      const double temporal_diagonal =
          rho > 0.0 ? (rho / dt_) * dirichlet_.TemporalDiagonal(m) : 0.0;
      const double temporal_coupling = rho > 0.0 ? -(rho / dt_) : 0.0;
      for (PetscInt cp = 0; cp < n_C_; ++cp) {
        for (PetscInt c = 0; c < n_C_; ++c) {
          double value = beta * H[static_cast<std::size_t>(cp) * n_C_ + c];
          if (d == 0)
            value += control_weight * R_dense_[static_cast<std::size_t>(cp) * n_C_ + c];
          if (rho > 0.0 && d <= 1)
            value += (d == 0 ? temporal_diagonal : temporal_coupling) *
                     M_face[static_cast<std::size_t>(cp) * n_C_ + c];
          const std::size_t row = static_cast<std::size_t>(m) * n_C_ + c;
          const std::size_t col = static_cast<std::size_t>(mp) * n_C_ + cp;
          K[col * dim + row] = value;
          K[row * dim + col] = value;   // (m',m) block = transpose
        }
      }
    }
  }
}

/** out_n = what_y(n) M_w v_n on every slot: the operator (What_y (x) M_w) of
 *  the Woodbury correction.  what_y == 1 under backward Euler. */
PetscErrorCode DirichletReducedSchurOperators::ApplyTrackingSlices(Vec v, Vec out) {
  const PetscScalar* s = nullptr;
  PetscScalar* d = nullptr;
  for (unsigned n = 0; n < steps_; ++n) {
    const PetscInt offset = static_cast<PetscInt>(n) * local_;
    PetscScalar* a = nullptr;
    PetscCall(VecGetArrayRead(v, &s));
    PetscCall(VecGetArray(spatial_a_, &a));
    for (PetscInt i = 0; i < local_; ++i) a[i] = s[offset + i];
    PetscCall(VecRestoreArray(spatial_a_, &a));
    PetscCall(VecRestoreArrayRead(v, &s));
    PetscCall(MatMult(tracking_, spatial_a_, spatial_b_));
    const double weight = SlotWeight(n);
    if (weight != 1.0) PetscCall(VecScale(spatial_b_, weight));
    const PetscScalar* b = nullptr;
    PetscCall(VecGetArrayRead(spatial_b_, &b));
    PetscCall(VecGetArray(out, &d));
    for (PetscInt i = 0; i < local_; ++i) d[offset + i] = b[i];
    PetscCall(VecRestoreArray(out, &d));
    PetscCall(VecRestoreArrayRead(spatial_b_, &b));
  }
  return PETSC_SUCCESS;
}

PetscErrorCode DirichletReducedSchurOperators::Solve(Vec b, Vec x) {
  if (prepared_beta_ < 0.0)
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ORDER,
            "Dirichlet reduced Schur: Prepare(beta) must precede Solve");
  // w = E_I^{-T} b;  t = B_D^T w.
  PetscCall(RunTimeSweep(backward_, b, w_));
  PetscCall(dirichlet_.ApplyBDT(w_, t_));
  if (iterative_) {
    // T8D: z = K^{-1} r by CG on the distributed reduced layout.
    const double t0 = MPI_Wtime();
    PetscCall(dirichlet_.GatherReduced(t_, core_r_));
    PetscCall(KSPSolve(core_ksp_, core_r_, core_z_));
    PetscInt its = 0;
    KSPConvergedReason reason;
    PetscCall(KSPGetIterationNumber(core_ksp_, &its));
    PetscCall(KSPGetConvergedReason(core_ksp_, &reason));
    stats_.solves += 1;
    stats_.iterations += its;
    stats_.max_iterations = std::max<long long>(stats_.max_iterations, its);
    if (reason < 0) stats_.failures += 1;
    stats_.seconds += MPI_Wtime() - t0;
    PetscCall(dirichlet_.ScatterReduced(core_z_, t_));
    PetscCall(dirichlet_.ApplyBD(t_, s_));
    PetscCall(RunTimeSweep(forward_, s_, q_));
    PetscCall(ApplyTrackingSlices(q_, s_));
    PetscCall(RunTimeSweep(backward_, s_, q_));
    PetscCall(VecWAXPY(x, -prepared_beta_, q_, w_));
    return PETSC_SUCCESS;
  }
  // Gather the C entries into the replicated reduced vector and solve K z = r.
  const PetscInt dim = reduced_dimension();
  std::fill(reduced_a_.begin(), reduced_a_.end(), 0.0);
  const std::vector<PetscInt>& owned = dirichlet_.owned_control_dofs();
  {
    const PetscScalar* t = nullptr;
    PetscCall(VecGetArrayRead(t_, &t));
    for (unsigned n = 0; n < steps_; ++n)
      for (std::size_t i = 0; i < owned.size(); ++i)
        reduced_a_[static_cast<std::size_t>(n) * n_C_ + C_owned_position_[i]] =
            PetscRealPart(t[static_cast<PetscInt>(n) * local_ + (owned[i] - begin_)]);
    PetscCall(VecRestoreArrayRead(t_, &t));
  }
  MPI_Allreduce(reduced_a_.data(), reduced_b_.data(), static_cast<int>(dim), MPI_DOUBLE,
                MPI_SUM, PETSC_COMM_WORLD);
  {
    PetscBLASInt bdim = 0, one = 1, info = 0;
    PetscCall(PetscBLASIntCast(dim, &bdim));
    LAPACKpotrs_("L", &bdim, &one, K_.data(), &bdim, reduced_b_.data(), &bdim, &info);
    if (info != 0)
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_LIB,
              "Dirichlet reduced Schur: dense triangular solve failed (info %d)", (int)info);
  }
  // Scatter z back to the owned C entries; then E_I^{-T} M_w E_I^{-1} B_D z.
  PetscCall(VecSet(t_, 0.0));
  {
    PetscScalar* t = nullptr;
    PetscCall(VecGetArray(t_, &t));
    for (unsigned n = 0; n < steps_; ++n)
      for (std::size_t i = 0; i < owned.size(); ++i)
        t[static_cast<PetscInt>(n) * local_ + (owned[i] - begin_)] =
            reduced_b_[static_cast<std::size_t>(n) * n_C_ + C_owned_position_[i]];
    PetscCall(VecRestoreArray(t_, &t));
  }
  PetscCall(dirichlet_.ApplyBD(t_, s_));
  PetscCall(RunTimeSweep(forward_, s_, q_));
  PetscCall(ApplyTrackingSlices(q_, s_));
  PetscCall(RunTimeSweep(backward_, s_, q_));
  PetscCall(VecWAXPY(x, -prepared_beta_, q_, w_));
  return PETSC_SUCCESS;
}

PetscErrorCode DirichletReducedSchurOperators::ApplySchur(double beta, Vec v, Vec Sv) {
  // (E_I^T v)_k = A_I v_k - (M_I/dt) v_{k+1}   (backward Euler)
  //             = Aplus_I v_k + Aminus_I v_{k+1}   (Crank-Nicolson)
  const bool crank_nicolson =
      scheme_ == ProblemConfig::TimeScheme::CRANK_NICOLSON;
  Mat neighbour = crank_nicolson ? A_minus_ : mass_;
  const double neighbour_scale = crank_nicolson ? 1.0 : -1.0 / dt_;
  auto slice = [&](Vec source, unsigned k, Vec spatial) -> PetscErrorCode {
    const PetscScalar* s = nullptr;
    PetscScalar* a = nullptr;
    PetscCall(VecGetArrayRead(source, &s));
    PetscCall(VecGetArray(spatial, &a));
    for (PetscInt i = 0; i < local_; ++i) a[i] = s[static_cast<PetscInt>(k) * local_ + i];
    PetscCall(VecRestoreArray(spatial, &a));
    PetscCall(VecRestoreArrayRead(source, &s));
    return PETSC_SUCCESS;
  };
  for (unsigned k = 0; k < steps_; ++k) {
    PetscCall(slice(v, k, spatial_a_));
    PetscCall(MatMult(A_, spatial_a_, spatial_b_));
    if (k + 1 < steps_) {
      PetscCall(slice(v, k + 1, spatial_a_));
      PetscCall(MatMult(neighbour, spatial_a_, spatial_c_));
      PetscCall(VecAXPY(spatial_b_, neighbour_scale, spatial_c_));
    }
    const PetscScalar* s = nullptr;
    PetscScalar* d = nullptr;
    PetscCall(VecGetArrayRead(spatial_b_, &s));
    PetscCall(VecGetArray(Sv, &d));
    for (PetscInt i = 0; i < local_; ++i) d[static_cast<PetscInt>(k) * local_ + i] = s[i];
    PetscCall(VecRestoreArray(Sv, &d));
    PetscCall(VecRestoreArrayRead(spatial_b_, &s));
  }
  // + beta M_w E_I^{-1} N_D v
  PetscCall(dirichlet_.ApplyND(1.0, v, t_));
  PetscCall(RunTimeSweep(forward_, t_, s_));
  PetscCall(ApplyTrackingSlices(s_, q_));
  PetscCall(VecAXPY(Sv, beta, q_));
  return PETSC_SUCCESS;
}

PetscErrorCode ApplyDirichletReducedSchur(PC pc, Vec b, Vec x) {
  DirichletReducedSchurOperators* operators = nullptr;
  PetscCall(PCShellGetContext(pc, reinterpret_cast<void**>(&operators)));
  if (!operators) return PETSC_ERR_ARG_NULL;
  PetscCall(operators->Solve(b, x));
  return PETSC_SUCCESS;
}

// ===========================================================================
// T8D: matrix-free Woodbury core.
// ===========================================================================

static PetscErrorCode CoreShellMult(Mat shell, Vec v, Vec out) {
  DirichletReducedSchurOperators* self = nullptr;
  PetscCall(MatShellGetContext(shell, &self));
  PetscCall(self->ApplyCore(v, out));
  return PETSC_SUCCESS;
}

static PetscErrorCode CorePcApply(PC pc, Vec r, Vec z) {
  DirichletControlOperators* dirichlet = nullptr;
  PetscCall(PCShellGetContext(pc, &dirichlet));
  PetscCall(dirichlet->SolveReducedR(r, z));
  return PETSC_SUCCESS;
}

/** K v = R_H1t v + beta B_D^T E_I^{-T} (What_y (x) M_w) E_I^{-1} B_D v. */
PetscErrorCode DirichletReducedSchurOperators::ApplyCore(Vec v, Vec out) {
  const double t0 = MPI_Wtime();
  PetscCall(dirichlet_.ScatterReduced(v, k_a_));      // Q_C v on the space-time layout
  PetscCall(dirichlet_.ApplyBD(k_a_, k_b_));          // B_D v
  PetscCall(RunTimeSweep(forward_, k_b_, k_a_));      // E_I^{-1}
  PetscCall(ApplyTrackingSlices(k_a_, k_b_));         // What_y (x) M_w
  PetscCall(RunTimeSweep(backward_, k_b_, k_a_));     // E_I^{-T}
  PetscCall(dirichlet_.ApplyBDT(k_a_, k_b_));         // B_D^T
  PetscCall(dirichlet_.GatherReduced(k_b_, out));
  PetscCall(VecScale(out, prepared_beta_));
  PetscCall(MatMult(dirichlet_.ReducedRegularization(), v, core_tmp_));
  PetscCall(VecAXPY(out, 1.0, core_tmp_));            // + R_H1t v
  stats_.applications += 1;
  stats_.apply_seconds += MPI_Wtime() - t0;
  return PETSC_SUCCESS;
}

void DirichletReducedSchurOperators::BuildCoreSolver() {
  for (Vec* v : {&core_r_, &core_z_, &core_tmp_})
    PetscCallAbort(PETSC_COMM_WORLD, dirichlet_.CreateReducedVector(v));
  PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(w_, &k_a_));
  PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(w_, &k_b_));
  const PetscInt local = dirichlet_.reduced_local();
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatCreateShell(PETSC_COMM_WORLD, local, local, PETSC_DETERMINE,
                                PETSC_DETERMINE, this, &core_shell_));
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatShellSetOperation(core_shell_, MATOP_MULT,
                                      reinterpret_cast<void (*)(void)>(CoreShellMult)));
  PetscCallAbort(PETSC_COMM_WORLD, MatSetOption(core_shell_, MAT_SYMMETRIC, PETSC_TRUE));
  PetscCallAbort(PETSC_COMM_WORLD, KSPCreate(PETSC_COMM_WORLD, &core_ksp_));
  PetscCallAbort(PETSC_COMM_WORLD, KSPSetOptionsPrefix(core_ksp_, "dirichlet_core_"));
  PetscCallAbort(PETSC_COMM_WORLD, KSPSetOperators(core_ksp_, core_shell_, core_shell_));
  PetscCallAbort(PETSC_COMM_WORLD, KSPSetType(core_ksp_, KSPCG));
  PetscCallAbort(PETSC_COMM_WORLD,
                 KSPSetTolerances(core_ksp_, core_config_.rtol, 0.0, PETSC_DEFAULT,
                                  core_config_.maxit));
  PetscCallAbort(PETSC_COMM_WORLD, KSPSetNormType(core_ksp_, KSP_NORM_UNPRECONDITIONED));
  PC pc = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, KSPGetPC(core_ksp_, &pc));
  if (core_config_.pc == ProblemConfig::DirichletCoreConfig::Pc::RT) {
    dirichlet_.ReducedRegularization();   // build + factor once
    PetscCallAbort(PETSC_COMM_WORLD, PCSetType(pc, PCSHELL));
    PetscCallAbort(PETSC_COMM_WORLD, PCShellSetContext(pc, &dirichlet_));
    PetscCallAbort(PETSC_COMM_WORLD, PCShellSetApply(pc, CorePcApply));
    PetscCallAbort(PETSC_COMM_WORLD, PCShellSetName(pc, "R_H1t (sparse direct)"));
  } else {
    dirichlet_.ReducedRegularization();   // the shell still needs R_H1t v
    PetscCallAbort(PETSC_COMM_WORLD, PCSetType(pc, PCNONE));
  }
  PetscCallAbort(PETSC_COMM_WORLD, KSPSetFromOptions(core_ksp_));
}

/** Verification (dense mode + --verify-dirichlet-core, after Prepare). */
PetscErrorCode DirichletReducedSchurOperators::VerifyCore() {
  if (iterative_ || K_unfactored_.empty())
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ORDER,
            "VerifyCore needs the dense mode with --verify-dirichlet-core after Prepare");
  int rank = 0;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  const PetscInt dim = reduced_dimension();
  const std::vector<double>& K = K_unfactored_;
  // ---- SPD of the dense core -------------------------------------------
  double asym = 0.0, norm = 0.0;
  for (PetscInt j = 0; j < dim; ++j)
    for (PetscInt i = 0; i < dim; ++i) {
      const double a = K[static_cast<std::size_t>(j) * dim + i],
                   b = K[static_cast<std::size_t>(i) * dim + j];
      asym += (a - b) * (a - b);
      norm += a * a;
    }
  // eigenvalues of K and of the pencil (K, R_H1t) (R_H1t = K at beta = 0)
  std::vector<double> R0;
  AssembleDenseK(0.0, R0);
  double lmin = 0, lmax = 0, gmin = 0, gmax = 0, rmin = 0, rmax = 0;
  {
    PetscBLASInt n = 0, info = 0, lwork = -1, itype = 1;
    PetscCall(PetscBLASIntCast(dim, &n));
    std::vector<double> A(K), W(dim), work(1);
    LAPACKsyev_("N", "L", &n, A.data(), &n, W.data(), work.data(), &lwork, &info);
    lwork = static_cast<PetscBLASInt>(work[0]);
    work.resize(static_cast<std::size_t>(lwork));
    LAPACKsyev_("N", "L", &n, A.data(), &n, W.data(), work.data(), &lwork, &info);
    lmin = W.front(); lmax = W.back();
    A = R0;
    LAPACKsyev_("N", "L", &n, A.data(), &n, W.data(), work.data(), &lwork, &info);
    rmin = W.front(); rmax = W.back();
    A = K;
    std::vector<double> B(R0);
    lwork = -1;
    LAPACKsygv_(&itype, "N", "L", &n, A.data(), &n, B.data(), &n, W.data(), work.data(), &lwork, &info);
    lwork = static_cast<PetscBLASInt>(work[0]);
    work.resize(static_cast<std::size_t>(lwork));
    LAPACKsygv_(&itype, "N", "L", &n, A.data(), &n, B.data(), &n, W.data(), work.data(), &lwork, &info);
    gmin = W.front(); gmax = W.back();
  }
  // random Rayleigh quotients (deterministic LCG)
  unsigned long long state = 12345ull;
  auto rnd = [&]() { state = state * 6364136223846793005ull + 1442695040888963407ull;
                     return static_cast<double>(state >> 11) / 9007199254740992.0 - 0.5; };
  double min_rayleigh = 1e300;
  for (int k = 0; k < 10; ++k) {
    std::vector<double> x(dim), Kx(dim, 0.0);
    for (auto& v : x) v = rnd();
    double xx = 0, xKx = 0;
    for (PetscInt j = 0; j < dim; ++j)
      for (PetscInt i = 0; i < dim; ++i) Kx[i] += K[static_cast<std::size_t>(j) * dim + i] * x[j];
    for (PetscInt i = 0; i < dim; ++i) { xx += x[i] * x[i]; xKx += x[i] * Kx[i]; }
    min_rayleigh = std::min(min_rayleigh, xKx / xx);
  }
  if (rank == 0)
    PetscCall(PetscPrintf(PETSC_COMM_SELF,
        "EX21_T8D_CORE_SPD m=%d nC=%d steps=%u beta=%.6e ||K-K^T||_F/||K||_F=%.3e lambda_min=%.6e"
        " lambda_max=%.6e cond_K=%.6e min_rayleigh_random=%.6e R_lambda_min=%.6e R_lambda_max=%.6e"
        " cond_R=%.6e RinvK_lambda_min=%.6e RinvK_lambda_max=%.6e cond_RinvK=%.6e\n",
        (int)dim, (int)n_C_, steps_, prepared_beta_, std::sqrt(asym / norm), lmin, lmax,
        lmax / lmin, min_rayleigh, rmin, rmax, rmax / rmin, gmin, gmax, gmax / gmin));
  // ---- dense-vs-shell action ---------------------------------------------
  const std::vector<PetscInt>& owned = dirichlet_.owned_control_dofs();
  const PetscInt nown = static_cast<PetscInt>(owned.size());
  auto dense_index = [&](unsigned n, PetscInt i) {
    return static_cast<std::size_t>(n) * n_C_ + C_owned_position_[i];
  };
  struct Test { const char* name; };
  const char* names[] = {"random", "constant", "time_low_freq", "time_high_freq",
                         "dofindex_smooth", "dofindex_oscillatory", "random_2", "slot1_only", "last_slot_only"};
  for (int t = 0; t < 9; ++t) {
    std::vector<double> x(dim);
    for (unsigned n = 0; n < steps_; ++n)
      for (PetscInt p = 0; p < n_C_; ++p) {
        const double tn = (n + 1.0) / steps_, s = n_C_ > 1 ? double(p) / (n_C_ - 1) : 0.0;
        double v = 0;
        switch (t) {
          case 0: case 6: v = rnd(); break;
          case 1: v = 1.0; break;
          case 2: v = std::sin(PETSC_PI * tn); break;
          case 3: v = (n % 2 ? -1.0 : 1.0); break;
          case 4: v = std::cos(PETSC_PI * s); break;
          case 5: v = std::cos(PETSC_PI * s * (n_C_ - 1)); break;
          case 7: v = n == 0 ? 1.0 + s : 0.0; break;
          case 8: v = n + 1 == steps_ ? 1.0 - s : 0.0; break;
        }
        x[static_cast<std::size_t>(n) * n_C_ + p] = v;
      }
    std::vector<double> Kx(dim, 0.0);
    for (PetscInt j = 0; j < dim; ++j)
      for (PetscInt i = 0; i < dim; ++i) Kx[i] += K[static_cast<std::size_t>(j) * dim + i] * x[j];
    PetscScalar* a = nullptr;
    PetscCall(VecGetArray(core_r_, &a));
    for (unsigned n = 0; n < steps_; ++n)
      for (PetscInt i = 0; i < nown; ++i) a[n * nown + i] = x[dense_index(n, i)];
    PetscCall(VecRestoreArray(core_r_, &a));
    PetscCall(ApplyCore(core_r_, core_z_));
    const PetscScalar* z = nullptr;
    PetscCall(VecGetArrayRead(core_z_, &z));
    double local[2] = {0, 0}, global[2] = {0, 0};
    for (unsigned n = 0; n < steps_; ++n)
      for (PetscInt i = 0; i < nown; ++i) {
        const double d = PetscRealPart(z[n * nown + i]) - Kx[dense_index(n, i)];
        local[0] += d * d;
        local[1] += Kx[dense_index(n, i)] * Kx[dense_index(n, i)];
      }
    PetscCall(VecRestoreArrayRead(core_z_, &z));
    MPI_Allreduce(local, global, 2, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
    if (rank == 0)
      PetscCall(PetscPrintf(PETSC_COMM_SELF,
          "EX21_T8D_CORE_ACTION vector=%s ||K_dense v - K_shell v||/||K_dense v||=%.3e\n",
          names[t], std::sqrt(global[0] / global[1])));
  }
  // ---- shell symmetry <Kx, y> = <x, Ky> --------------------------------
  {
    Vec x1 = nullptr, x2 = nullptr, k1 = nullptr, k2 = nullptr;
    for (Vec* v : {&x1, &x2, &k1, &k2}) PetscCall(dirichlet_.CreateReducedVector(v));
    PetscRandom rctx;
    PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rctx));
    PetscCall(PetscRandomSetSeed(rctx, 7));
    PetscCall(PetscRandomSeed(rctx));
    PetscCall(VecSetRandom(x1, rctx));
    PetscCall(VecSetRandom(x2, rctx));
    PetscCall(ApplyCore(x1, k1));
    PetscCall(ApplyCore(x2, k2));
    PetscScalar a12 = 0, a21 = 0;
    PetscReal n1 = 0, n2 = 0;
    PetscCall(VecDot(k1, x2, &a12));
    PetscCall(VecDot(x1, k2, &a21));
    PetscCall(VecNorm(k1, NORM_2, &n1));
    PetscCall(VecNorm(x2, NORM_2, &n2));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "EX21_T8D_CORE_SHELL_SYMMETRY |<Kx,y>-<x,Ky>|/(|Kx||y|)=%.3e\n",
        std::fabs(PetscRealPart(a12 - a21)) / (n1 * n2)));
    // the sparse R_H1t vs the dense R block (beta = 0 assembly)
    PetscCall(VecSetRandom(x1, rctx));
    PetscCall(MatMult(dirichlet_.ReducedRegularization(), x1, k1));
    std::vector<double> xd(dim, 0.0), xs(dim, 0.0);
    const PetscScalar* a = nullptr;
    PetscCall(VecGetArrayRead(x1, &a));
    for (unsigned n = 0; n < steps_; ++n)
      for (PetscInt i = 0; i < nown; ++i) xs[dense_index(n, i)] = PetscRealPart(a[n * nown + i]);
    PetscCall(VecRestoreArrayRead(x1, &a));
    MPI_Allreduce(xs.data(), xd.data(), static_cast<int>(dim), MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
    std::vector<double> Rx(dim, 0.0);
    for (PetscInt j = 0; j < dim; ++j)
      for (PetscInt i = 0; i < dim; ++i) Rx[i] += R0[static_cast<std::size_t>(j) * dim + i] * xd[j];
    PetscCall(VecGetArrayRead(k1, &a));
    double local[2] = {0, 0}, global[2] = {0, 0};
    for (unsigned n = 0; n < steps_; ++n)
      for (PetscInt i = 0; i < nown; ++i) {
        const double d = PetscRealPart(a[n * nown + i]) - Rx[dense_index(n, i)];
        local[0] += d * d; local[1] += Rx[dense_index(n, i)] * Rx[dense_index(n, i)];
      }
    PetscCall(VecRestoreArrayRead(k1, &a));
    MPI_Allreduce(local, global, 2, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "EX21_T8D_CORE_RT_SPARSE_VS_DENSE rel=%.3e backend=%s\n",
        std::sqrt(global[0] / global[1]), dirichlet_.reduced_r_backend().c_str()));
    PetscCall(PetscRandomDestroy(&rctx));
    for (Vec* v : {&x1, &x2, &k1, &k2}) PetscCall(VecDestroy(v));
  }
  return PETSC_SUCCESS;
}
