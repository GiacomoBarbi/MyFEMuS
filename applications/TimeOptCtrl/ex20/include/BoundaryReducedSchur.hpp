// Boundary-reduced Schur operators: the spatial partition into the control
// support R = support(N) and its complement Z, the embedded restricted
// operators of the Z block, the direct spatial solvers of the exact time
// sweeps built on them, and the dense reduced operators S0, S1.
//
// With the KKT unknowns split into R (control-active DOFs at every time level)
// and Z, the control operator N vanishes on Z, so the Z block of the Schur
// complement S = E^T + beta C, C = (I (x) M_omega) E^{-1} (I (x) N), is exactly
// E^T_ZZ and is inverted by a backward time sweep on the restricted spatial
// operators.  The restriction is embedded in the full spatial layout: A_ZZ is
// A with the R rows and columns replaced by the identity and M_ZZ is M with the
// R rows and columns zeroed, so that a sweep on (A_ZZ, M_ZZ) with an R-masked
// right-hand side has zero R components and Z components equal to the
// restricted recurrence.
//
// Eliminating Z gives the reduced operator, exactly affine in beta:
//     S_red(beta) = S_RR - S_RZ S_ZZ^{-1} S_ZR = S0 + beta S1
//     S0 = E^T_RR - E^T_RZ (E^T_ZZ)^{-1} E^T_ZR
//     S1 = C_RR   - E^T_RZ (E^T_ZZ)^{-1} C_ZR
//
// CRANK-NICOLSON.  The SAME reduction holds, and is affine in beta for the same
// reason: N = M_Gamma is supported on R, so C_CN = (What_y (x) M_omega)
// E_CN^{-1} (I (x) N) again has zero COLUMNS on Z, which is what makes S_ZZ and
// S_RZ beta-free.  Only the blocks change:
//     E_CN^T   = blockdiag(A_plus) + blocksuperdiag(A_minus)
//     A_plus   = M/dt + K/2 ,  A_minus = -M/dt + K/2
//     C_CN(i,j) = what_i M_omega X_{i-j} N ,  what = (1, ..., 1, 1/2)
// so C_CN is NOT block Toeplitz: the trapezoidal weight sits on the ROW index.
// The lag families become
//     X_0   = A_plus^{-1} N Q ,        X_l   = A_plus^{-1} (-A_minus) X_{l-1}
//     Lam_0 = A_plus_ZZ^{-1} A_plus_ZR
//     Lam_1 = A_plus_ZZ^{-1} (A_minus_ZR - A_minus_ZZ Lam_0)
//     Lam_m = A_plus_ZZ^{-1} (-A_minus_ZZ Lam_{m-1})
// and the weight enters S1 twice: what_i on the C_RR row index, and what_k
// INSIDE the sum (Psi C_ZR)_{i,j} = sum_{k >= max(i,j)} what_k Lam_{k-i}^T
// (M_omega X_{k-j})_Z, where k is the summation index, not i or j.  Setting
// (A_plus, A_minus) -> (A, -M/dt) and what == 1 recovers the backward-Euler
// formulas above exactly; the derivation and both limits are checked on random
// dense operators by aux/cn_stage2_neumann/derivation_check.py.
//
// S0 and S1 depend on the mesh, the operators, `steps` and `dt` only; they are
// dense on rank 0 in the canonical reduced order (time slot major, then the
// sorted global spatial R DOF) and immutable once built.  Prepare(beta) forms
// and factors S_work = S0 + beta S1 on rank 0; Solve then applies the exact
// block elimination
//     y_Z = S_ZZ^{-1} b_Z,  r_R = b_R - S_RZ y_Z,  x_R = S_red^{-1} r_R,
//     x_Z = S_ZZ^{-1}(b_Z - S_ZR x_R),  x = Q_R x_R + Q_Z x_Z
// with the R components moved between the full space-time layout and rank 0
// by one persistent scatter in the canonical order.
//
// LIFETIME: `mass`, `stiffness`, `control` and `tracking` are borrowed for the
// whole life of the object and must outlive it.  Everything else is owned.
// The object is built for one mesh, one spatial operator set, one `steps` and
// one `dt`; it is destroyed before any of those change.  It is neither copyable
// nor movable because PETSc callbacks receive its address.
//
// REUSE: one object may serve any number of solves whose alpha-independent
// operator is the one it was built from.  S0 and S1 depend on the four
// borrowed matrices (which carry the mesh, the diffusivity, the state
// boundary conditions, the control boundary and the tracking region), on
// `steps` and on `dt`; alpha_u, alpha_track and eta enter only through the
// beta of Prepare, and the target, source and initial state only through the
// right-hand side.  IsCompatible states that contract: it compares the
// matrices as PETSc objects (not by value: a rebuilt matrix with equal
// entries is a different object whose lifetime this object knows nothing
// about) and the scalar invariants.  It cannot tell a destroyed matrix from a
// live one; keeping the borrowed matrices alive remains the caller's duty.

#pragma once

#include <petscksp.h>
#include <petscmat.h>

#include <string>
#include <vector>

#include "Config.hpp"
#include "Preconditioners.hpp"

class BoundaryReducedSchurOperators {
 public:
  /** Collective on PETSC_COMM_WORLD.  Builds the partition from the support of
   *  `control`, the diagonal block (A = M/dt + K for backward Euler,
   *  A_plus = M/dt + K/2 for Crank-Nicolson), its Z restriction, the temporal
   *  coupling block and its Z restriction, the two direct spatial solvers and
   *  the dense reduced operators S0, S1.  The time scheme comes from
   *  `config.time.scheme`. */
  BoundaryReducedSchurOperators(const ProblemConfig& config, Mat mass,
                                Mat stiffness, Mat control, Mat tracking);
  ~BoundaryReducedSchurOperators();

  BoundaryReducedSchurOperators(const BoundaryReducedSchurOperators&) = delete;
  BoundaryReducedSchurOperators& operator=(
      const BoundaryReducedSchurOperators&) = delete;
  BoundaryReducedSchurOperators(BoundaryReducedSchurOperators&&) = delete;
  BoundaryReducedSchurOperators& operator=(BoundaryReducedSchurOperators&&) =
      delete;

  /** Whether this object can serve a solve of `config` on the given
   *  matrices: the same four PETSc objects it was built from, the same
   *  `steps`, `dt` and control type.  Constant-time metadata comparison, no
   *  PETSc work; `reason`, if given, receives the first mismatch. */
  bool IsCompatible(const ProblemConfig& config, Mat mass, Mat stiffness,
                    Mat control, Mat tracking,
                    std::string* reason = nullptr) const;

  /** solution = (E^T_ZZ)^{-1} rhs_Z on the full space-time layout: the R
   *  components of `rhs` are ignored and those of `solution` are zero. */
  PetscErrorCode SolveZ(Vec rhs, Vec solution);

  /** solution = E^{-1} rhs, the exact forward sweep with the direct solver. */
  PetscErrorCode SolveForward(Vec rhs, Vec solution);

  /** S_work = S0 + beta S1, factored on rank 0 (collective; every rank must
   *  call it).  Returns without work when `beta` equals the prepared value. */
  PetscErrorCode Prepare(double beta);
  double prepared_beta() const { return prepared_beta_; }

  /** x = S_red(beta)^{-1} b for the prepared beta, exact block elimination on
   *  the full space-time layout (collective).  No PETSc object is created. */
  PetscErrorCode Solve(Vec b, Vec x);

  /** Sv = E^T v + beta (I (x) M_omega) E^{-1} (I (x) N) v (collective). */
  PetscErrorCode ApplySchur(double beta, Vec v, Vec Sv);

  PetscInt spatial_size() const { return n_; }
  PetscInt control_support() const { return static_cast<PetscInt>(R_global_.size()); }
  PetscInt reduced_dimension() const { return static_cast<PetscInt>(steps_) * control_support(); }
  /** Sorted global spatial DOFs of R, identical on every rank. */
  const std::vector<PetscInt>& support() const { return R_global_; }
  /** The owned subset of support(), increasing. */
  const std::vector<PetscInt>& owned_support() const { return R_owned_; }
  /** Owned space-time indices of R (any order). */
  IS owned_support_indices() const { return is_R_owned_; }
  Mat diagonal() const { return A_; }
  Mat restricted_diagonal() const { return A_ZZ_; }
  Mat restricted_coupling() const { return M_ZZ_; }
  /** The temporal coupling block of the FULL-space forward sweep and its
   *  scale: (mass, 1/dt) under backward Euler -- the block is (1/dt) M -- and
   *  (-A_minus, 1) under Crank-Nicolson.  A caller that builds its own sweep on
   *  `diagonal()` must use this pair, not the mass matrix. */
  Mat coupling() const { return coupling_ ? coupling_ : mass_; }
  double coupling_scale() const { return coupling_ ? 1.0 : 1.0 / dt_; }
  ProblemConfig::TimeScheme scheme() const { return scheme_; }
  const std::string& forward_backend() const { return forward_backend_; }
  const std::string& z_backend() const { return z_backend_; }
  /** The dense reduced operators, rank 0 only (null elsewhere); the handles
   *  stay owned by this object. */
  Mat S0() const { return S0_; }
  Mat S1() const { return S1_; }
  /** S0 + beta S1 for the prepared beta, rank 0 only; factored in place by the
   *  dense solver, so it is not the plain combination after Prepare. */
  Mat S_work() const { return S_work_; }

 private:
  void BuildPartition();
  void CheckDenseStorage();
  void BuildOperators();
  void BuildSweeps();
  void BuildInvariant();
  void BuildTransfer();
  // (E^T v)_k = A v_k - (1/dt) M v_{k+1}   (backward Euler)
  //           = A_plus v_k + A_minus v_{k+1}  (Crank-Nicolson)
  PetscErrorCode ApplyTransposeE(Vec v, Vec out);
  // (C v)_k = what_k M_omega (E^{-1} N v)_k;  what == 1 under backward Euler.
  PetscErrorCode ApplyC(Vec v, Vec out);
  /** The trapezoidal tracking weight of a slot: 1 everywhere under backward
   *  Euler, and 1 except 1/2 on the final slot under Crank-Nicolson. */
  double SlotWeight(unsigned slot) const {
    return (scheme_ == ProblemConfig::TimeScheme::CRANK_NICOLSON &&
            slot + 1u == steps_) ? 0.5 : 1.0;
  }

  Mat mass_ = nullptr;        // borrowed
  Mat stiffness_ = nullptr;   // borrowed
  Mat control_ = nullptr;     // borrowed
  Mat tracking_ = nullptr;    // borrowed
  // PETSc object ids of the borrowed matrices: unique for the life of the
  // process, so a later object at a recycled address does not pass for them.
  PetscObjectId mass_id_ = 0, stiffness_id_ = 0, control_id_ = 0,
                tracking_id_ = 0;
  ProblemConfig::ControlType control_type_ = ProblemConfig::ControlType::NEUMANN;

  unsigned steps_ = 0;
  double dt_ = 0.0;
  PetscInt n_ = 0, local_ = 0, begin_ = 0;
  std::vector<PetscInt> spatial_ranges_;
  std::vector<PetscInt> R_global_;
  std::vector<PetscInt> R_owned_;
  std::vector<int> support_counts_;    // |R_owned| of every rank: rank r's R DOFs
  std::vector<int> support_offsets_;   //   are R_global_[offsets[r] .. offsets[r+1])
  IS is_R_owned_ = nullptr;

  ProblemConfig::TimeScheme scheme_ =
      ProblemConfig::TimeScheme::BACKWARD_EULER;
  // Backward Euler: A = M/dt + K, the coupling block is (1/dt) M (borrowed
  // `mass_`, so `coupling_` and `A_minus_` stay null).
  // Crank-Nicolson: A = A_plus = M/dt + K/2, A_minus = -M/dt + K/2 and the
  // coupling block is the owned -A_minus = M/dt - K/2 with unit scale.
  Mat A_ = nullptr;      // M/dt + K   (backward Euler) / A_plus (Crank-Nicolson)
  Mat A_minus_ = nullptr;   // Crank-Nicolson only
  Mat coupling_ = nullptr;  // Crank-Nicolson only: -A_minus
  Mat A_ZZ_ = nullptr;   // A, R rows/columns -> identity
  Mat M_ZZ_ = nullptr;   // the coupling block, R rows/columns -> zero
  TimeSweepContext forward_;     // E^{-1}: FORWARD on (A, M)
  TimeSweepContext backward_Z_;  // (E^T_ZZ)^{-1}: BACKWARD on (A_ZZ, M_ZZ)
  std::string forward_backend_;
  std::string z_backend_;
  Vec work_ = nullptr;   // full space-time layout (SolveZ)
  Mat S0_ = nullptr;     // rank 0: reduced_dimension() square, dense
  Mat S1_ = nullptr;     // rank 0: reduced_dimension() square, dense

  // R transfer: full layout <-> canonical rank-0 vector (local size
  // reduced_dimension() on rank 0, 0 elsewhere) and its sequential mirrors.
  IS is_R_canonical_ = nullptr;
  IS is_R_root_ = nullptr;
  VecScatter to_root_ = nullptr;
  Vec R_root_ = nullptr, R_root2_ = nullptr;
  Vec seq_rhs_ = nullptr, seq_sol_ = nullptr;   // rank 0
  // beta-dependent dense state, rank 0
  Mat S_work_ = nullptr;
  KSP dense_ = nullptr;
  bool prepared_ = false;
  double prepared_beta_ = 0.0;
  // apply workspace: full space-time layout and spatial slices
  Vec z_ = nullptr, t_ = nullptr, q_ = nullptr, c_ = nullptr, c_tmp_ = nullptr;
  Vec spatial_a_ = nullptr, spatial_b_ = nullptr, spatial_c_ = nullptr;
};

/** PCSHELL apply callback of the Schur split: the shell context is the
 *  address of a prepared BoundaryReducedSchurOperators, which must outlive
 *  the PC. */
PetscErrorCode ApplyBoundaryReducedSchur(PC pc, Vec b, Vec x);
