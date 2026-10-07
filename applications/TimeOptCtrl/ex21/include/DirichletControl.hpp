// Dirichlet boundary control operators: the controlled boundary DOF set C of
// the control face, the H1(Gamma_c) regularization R_Gamma = M_Gamma + ell^2
// K_Gamma with its direct factor, and the control-to-state operator B_D of
// the strongly eliminated backward-Euler scheme.
//
// With the spatial DOFs split into the free set I and the controlled
// Dirichlet set C (the support of the trace mass matrix M_Gamma), imposing
// y_C,n = u_n in the backward-Euler step  A y_n - (M/dt) y_{n-1} = M f_n,
// A = M/dt + K, leaves the I rows
//     A_II y_I,n - M_II/dt y_I,n-1 + A_IC u_n - M_IC/dt u_{n-1} = (M f_n)_I.
//
// CRANK-NICOLSON.  The control stays NODAL (it is the trace of the state, so
// it lives at the state nodes) and the same elimination applies to
//     A_plus y_n + A_minus y_{n-1} = 1/2 (F_n + F_{n-1}),
//     A_plus = M/dt + K/2 ,  A_minus = -M/dt + K/2
// leaving
//     Aplus_II y_I,n + Aminus_II y_I,n-1 + Gplus_IC u_n + Gminus_IC u_n-1 = ...
// with Gplus_IC = (A_plus)_IC and Gminus_IC = (A_minus)_IC, i.e. the SAME
// ApplyStep with (A, -M/dt) replaced by (A_plus, A_minus).  u_0 = y_0|_C / eta
// stays DATA and reaches the first row through the full-space (M/dt - K/2) y_0
// term of the right-hand side, exactly as backward Euler's (M/dt) y_0 does.
// The control regularization acquires the trapezoidal nodal weights
// What_u = diag(1, ..., 1, 1/2): R_T = What_u (x) R_Gamma, so SolveR applies
// 2 R_Gamma^{-1} on the final slot and ApplyR applies R_Gamma/2 there.
// On the full DOF layout, with the C rows of the state operator turned into
// identity rows (E_I: the accepted AY shell on M, K with the C rows and
// columns eliminated), the boundary contribution is the lower bidiagonal
// space-time operator
//     (B_D u)_n = P_I [ A Q_C u_n - (M/dt) Q_C u_{n-1} ]          (u_0 = 0)
// where Q_C extends a boundary vector by zero and P_I zeroes the C rows, and
// A, M are the UNHATTED matrices.  Its transpose,
//     (B_D^T p)_n = Q_C^T [ A p_n - (M/dt) p_{n+1} ],
// is the residual of the adjoint equation on the C rows: the variational
// (discrete) normal derivative of the adjoint on Gamma_c.
//
// Regularization: r(u,v) = (u,v)_Gamma + ell^2 (d_s u, d_s v)_Gamma on the
// face, i.e. R_Gamma = M_Gamma + ell^2 K_Gamma, symmetric positive definite on
// C; stored embedded (identity outside C) and factored once.  The reduced
// KKT block (1,2) is (eta^2/alpha_u) N_D with N_D = B_D R^{-1} B_D^T, applied
// matrix-free and never assembled.
//
// TEMPORAL H1 SEMINORM.  With gamma_u > 0 the objective gains
//     (gamma_u/2) \int_0^T ||d_t u||^2_{L2(Gamma_c)} dt
// whose metric is the face MASS matrix M_Gamma, NOT R_Gamma.  The control is
// P1 in time on the nodes t^0..t^N with u_0 = y_0|_Gamma_c/eta as DATA, so the
// integral is exact:
//     J_H1t = (gamma_u/(2 dt)) sum_{n=1}^{N} (u_n-u_{n-1})^T M_Gamma (u_n-u_{n-1}).
// Its Hessian is L_t = (1/dt) T (x) M_Gamma with T = tridiag(-1, 2, -1) whose
// LAST diagonal entry is 1 (slot N has no successor), i.e. T = D^T D for the
// first-difference D, hence SPD; its linear part is the affine data term
//     q_0 = (1/dt) [M_Gamma u_0; 0; ...; 0],   grad J_H1t = gamma_u (L_t U - q_0),
// and it contributes the constant gamma_u/(2 dt) u_0^T M_Gamma u_0 to the
// objective VALUE.  ex20 carries the dt-scaled multiplier p = P_Lagrange/dt,
// so every KKT row is the true derivative DIVIDED BY dt: the spatial term
// (alpha_u/2) dt sum_n w_n u_n^T R_Gamma u_n contributes alpha_u R_T with no
// dt, while the temporal energy, which has no dt of its own, contributes
// gamma_u/dt.  The control operator of the KKT system is therefore
//     Q_u = alpha_u (What_u (x) R_Gamma) + (gamma_u/dt) L_t = alpha_u R_H1t,
//     R_H1t = What_u (x) R_Gamma + rho L_t,   rho = gamma_u/(alpha_u dt),
// and the control stationarity reads  Q_u U - eta B_D^T P = (gamma_u/dt) q_0.
// EVERY accepted call site keeps its scaling: SolveR and ApplyR apply R_H1t^{-1}
// and R_H1t, which for gamma_u = 0 are literally the per-slot R_Gamma solve and
// product of stages 1-3.  Because R_H1t couples consecutive slots it cannot be
// applied slot by slot; for gamma_u > 0 it is assembled as a dense replicated
// (N_t |C|) x (N_t |C|) SPD matrix and Cholesky-factored once, exactly as the
// reduced Schur operator's K already is.
// The derivation, the SPD property, the 3x3 <-> 2x2 equivalence, the Woodbury
// form with the temporally coupled Q_u and the gamma_u -> 0 limit are checked
// on random dense operators by aux/dirichlet_h1_time/derivation_check.py.
//
// LIFETIME: the four matrices are borrowed and must outlive the object; all
// PETSc objects created here are owned.  Not copyable, not movable (the shell
// callback keeps its address).

#pragma once

#include <petscksp.h>
#include <petscmat.h>

#include <string>
#include <vector>

#include "Config.hpp"

/** Direct solver for an SPD matrix: Cholesky (CHOLMOD on one rank, MUMPS on
 *  several) with PETSc LU / MUMPS LU as the fallback.  The caller owns the
 *  KSP; `backend` receives "package:factor". */
KSP CreateDirichletDirectSolver(Mat matrix, const char* prefix,
                                std::string* backend, bool lu_only);
/** Cholesky first (SPD matrices); `lu_only` overload above: nonsymmetric. */
KSP CreateDirichletDirectSolver(Mat matrix, const char* prefix,
                                std::string* backend);

class DirichletControlOperators {
 public:
  /** Collective.  `mass`/`stiffness` are the unhatted M and K, `trace_mass`
   *  and `trace_stiffness` the face matrices M_Gamma and K_Gamma; C is the
   *  support of `trace_mass`. */
  DirichletControlOperators(const ProblemConfig& config, Mat mass,
                            Mat stiffness, Mat trace_mass,
                            Mat trace_stiffness);
  ~DirichletControlOperators();
  DirichletControlOperators(const DirichletControlOperators&) = delete;
  DirichletControlOperators& operator=(const DirichletControlOperators&) = delete;

  /** out = B_D u on the full space-time layout (out_C = 0). */
  PetscErrorCode ApplyBD(Vec u, Vec out);
  /** out = B_D^T p (supported on C). */
  PetscErrorCode ApplyBDT(Vec p, Vec out);
  /** T7D DIAGNOSTIC ONLY (negative control): the pre-T7D B_D^T, which applies
   *  the FORWARD step (A, not A^T) -- exact without convection, wrong with it.
   *  Never used by the solver. */
  PetscErrorCode ApplyBDTLegacy(Vec p, Vec out);
  /** True when B_D^T transposes the (nonsymmetric) stiffness: convection on. */
  bool transposes_stiffness() const { return transpose_stiffness_; }
  /** z = R_H1t^{-1} b on C, zero outside.  With gamma_u = 0 this is the
   *  accepted per-slot solve z_n = (1/What_u(n)) R_Gamma^{-1} (b_n)_C; with
   *  gamma_u > 0 it is one dense replicated Cholesky solve over all slots. */
  PetscErrorCode SolveR(Vec b, Vec z);
  /** out = R_H1t u on C, zero outside: (What_u (x) R_Gamma) u + rho L_t u. */
  PetscErrorCode ApplyR(Vec u, Vec out);
  /** out = scale * B_D R^{-1} B_D^T x: the KKT block (1,2) without its
   *  coefficient.  No PETSc object is created. */
  PetscErrorCode ApplyND(double scale, Vec x, Vec out);
  /** u = factor * R^{-1} B_D^T p: the control from the adjoint
   *  (factor = eta / alpha_u). */
  PetscErrorCode ReconstructControl(double factor, Vec p, Vec u);
  /** Zero the C entries of a full space-time vector (P_I). */
  PetscErrorCode ZeroControlledRows(Vec v);

  /** q_0 = (1/dt) [M_Gamma u_0; 0; ...; 0] on the space-time layout, from the
   *  SPATIAL initial trace u_0 = y_0|_Gamma_c / eta (only its C entries are
   *  read).  Zero when gamma_u = 0. */
  PetscErrorCode ApplyAffineData(Vec initial_trace, Vec q0);
  /** u += rho R_H1t^{-1} q_0 = Q_u^{-1} ((gamma_u/dt) q_0): the affine part of
   *  the control stationarity.  No-op when gamma_u = 0. */
  PetscErrorCode AddAffineControl(Vec initial_trace, Vec u);
  /** out = B_D (rho R_H1t^{-1} q_0): the affine state-equation correction.
   *  The state right-hand side gains -eta times this.  Zero when gamma_u = 0. */
  PetscErrorCode ApplyAffineStateCorrection(Vec initial_trace, Vec out);
  /** out = M_Gamma v on C, zero outside (spatial).  Requires gamma_u > 0. */
  PetscErrorCode ApplyFaceMass(Vec v, Vec out);

  /** gamma_u and rho = gamma_u/alpha_u. */
  double time_weight() const { return gamma_; }
  double time_ratio() const { return rho_; }
  bool has_time_regularization() const { return gamma_ > 0.0; }
  /** The diagonal entry T_mm of the temporal stiffness: 2 on slots
   *  1..N-1 and 1 on the LAST slot, which has no successor. */
  double TemporalDiagonal(unsigned slot) const {
    return slot + 1u == steps_ ? 1.0 : 2.0;
  }
  /** M_Gamma restricted to C, replicated dense |C| x |C| column-major, and
   *  the sorted global C list that indexes it.  Non-empty only when
   *  gamma_u > 0 (the reduced Schur operator needs them to add rho L_t). */
  const std::vector<double>& face_mass_block() const { return M_dense_; }
  const std::vector<PetscInt>& global_control_dofs() const { return C_global_; }

  /** The trapezoidal nodal control weight of a slot: 1 everywhere under
   *  backward Euler, and 1 except 1/2 on the final slot under Crank-Nicolson.
   *  This is What_u; SolveR uses its reciprocal. */
  double SlotWeight(unsigned slot) const {
    return (scheme_ == ProblemConfig::TimeScheme::CRANK_NICOLSON &&
            slot + 1u == steps_) ? 0.5 : 1.0;
  }
  ProblemConfig::TimeScheme scheme() const { return scheme_; }

  unsigned steps() const { return steps_; }
  PetscInt spatial_local() const { return local_; }
  /** Global count of controlled DOFs. */
  PetscInt control_count() const { return control_count_; }
  /** Owned controlled spatial DOFs (global row numbers, increasing). */
  const std::vector<PetscInt>& owned_control_dofs() const { return C_owned_; }
  /** Owned-row mask: 1 on C, 0 on I. */
  const std::vector<unsigned char>& control_mask() const { return mask_; }
  /** The embedded regularization matrix (identity outside C); owned here. */
  Mat regularization() const { return R_; }
  const std::string& factor_backend() const { return backend_; }
  double h1_length() const { return ell_; }
  /** The borrowed unhatted M and K. */
  Mat unhatted_mass() const { return mass_; }
  Mat unhatted_stiffness() const { return stiffness_; }

  // ---- T8D: the DISTRIBUTED reduced control space ------------------------
  // A reduced vector holds the C entries of a space-time vector, local size
  // N_t |C_owned|, ordered [slot][owned C DOF] -- gather and scatter are
  // purely local copies.  Nothing of size N_t |C| is replicated.
  /** True with --dirichlet-core-solver iterative: no dense (N_t|C|)^2 object;
   *  R_H1t^{-1} uses the sparse direct factor of reduced_regularization(). */
  bool sparse_core() const { return sparse_core_; }
  PetscInt reduced_local() const {
    return static_cast<PetscInt>(steps_) * static_cast<PetscInt>(C_owned_.size());
  }
  PetscErrorCode CreateReducedVector(Vec* v) const;
  /** reduced = the C entries of a space-time vector (local copy). */
  PetscErrorCode GatherReduced(Vec space_time, Vec reduced) const;
  /** space_time = 0 off C, the reduced entries on C (local copy). */
  PetscErrorCode ScatterReduced(Vec reduced, Vec space_time) const;
  /** The exact sparse R_H1t = What_u (x) R_Gamma + (rho/dt) T (x) M_Gamma on the
   *  reduced layout (MPIAIJ, SPD), built and factored on first use. */
  Mat ReducedRegularization();
  /** z = R_H1t^{-1} r on the reduced layout (sparse direct factor). */
  PetscErrorCode SolveReducedR(Vec r, Vec z);
  const std::string& reduced_r_backend() const { return RT_backend_; }

 private:
  void BuildPartition(Mat trace_mass);
  void BuildRegularization(Mat trace_mass, Mat trace_stiffness);
  /** gamma_u > 0 only: the embedded M_Gamma, the replicated C list, the dense
   *  R_Gamma and M_Gamma blocks and the Cholesky factor of R_H1t. */
  void BuildTemporalRegularization(Mat trace_mass);
  void BuildGlobalControlList();
  void BuildFaceBlock(Mat embedded, std::vector<double>& dense) const;
  /** out = rho R_H1t^{-1} q_0 on the space-time layout. */
  PetscErrorCode AffineControl(Vec initial_trace, Vec out);
  PetscErrorCode ApplyStep(Vec spatial_current, Vec spatial_previous,
                           bool have_previous, Vec spatial_out,
                           bool transpose = false);
  PetscErrorCode ApplyBDTImpl(Vec p, Vec out, bool transpose);
  // T7D: B_D^T needs A^T = M/dt + L^T (CN: A_plus^T, A_minus^T).  M is
  // symmetric; L is not under convection.  False (no convection) keeps the
  // ex20 products bit for bit.
  bool transpose_stiffness_ = false;

  Mat mass_ = nullptr;        // borrowed, unhatted
  Mat stiffness_ = nullptr;   // borrowed, unhatted
  ProblemConfig::TimeScheme scheme_ =
      ProblemConfig::TimeScheme::BACKWARD_EULER;
  unsigned steps_ = 0;
  double dt_ = 0.0;
  double ell_ = 0.0;
  PetscInt n_ = 0, local_ = 0, begin_ = 0;
  PetscInt control_count_ = 0;
  std::vector<PetscInt> C_owned_;
  std::vector<unsigned char> mask_;
  Mat R_ = nullptr;           // owned: M_Gamma + ell^2 K_Gamma, identity on I
  KSP R_solver_ = nullptr;    // owned: direct factor of R_
  std::string backend_;
  Vec spatial_a_ = nullptr, spatial_b_ = nullptr, spatial_c_ = nullptr,
      spatial_d_ = nullptr;   // owned spatial workspace
  Vec work_ = nullptr;        // owned full-layout workspace (ApplyND)
  // Temporal regularization (gamma_u > 0 only).  Everything below is empty or
  // null when gamma_u = 0, and no code path touches it then.
  double alpha_u_ = 0.0;
  double gamma_ = 0.0;        // gamma_u
  double rho_ = 0.0;          // gamma_u / alpha_u
  Mat M_face_ = nullptr;      // owned: M_Gamma embedded, identity on I
  std::vector<PetscInt> C_global_;           // sorted, replicated
  std::vector<PetscInt> C_owned_position_;   // position in C_global_ of each owned C DOF
  std::vector<double> R_dense_;              // R_Gamma on C, |C| x |C| column-major
  std::vector<double> M_dense_;              // M_Gamma on C, |C| x |C| column-major
  std::vector<double> Q_dense_;              // Cholesky factor of R_H1t, (N_t|C|)^2
  std::vector<double> reduced_a_, reduced_b_;
  Vec work2_ = nullptr;       // owned full-layout workspace (affine terms)
  // T8D (sparse_core_ or on first ReducedRegularization() call)
  void BuildReducedRegularization();
  bool sparse_core_ = false;
  Mat RT_ = nullptr;          // owned: sparse R_H1t on the reduced layout
  KSP RT_solver_ = nullptr;   // owned: its direct (Cholesky) factor
  std::string RT_backend_;
};

/** Context and MULT callback of the matrix-free KKT block
 *  scale * N_D = scale * B_D R^{-1} B_D^T (MatShell over the space-time layout). */
struct DirichletControlShellContext {
  DirichletControlOperators* operators = nullptr;
  double scale = 0.0;
};
PetscErrorCode ApplyDirichletControlShell(Mat shell, Vec x, Vec y);
