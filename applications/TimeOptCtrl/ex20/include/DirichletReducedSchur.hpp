// Exact Schur solve for Dirichlet boundary control by a low-rank (Woodbury)
// reduction to the control space.
//
// The reduced KKT system of Dirichlet control is
//     [ E_I            (eta^2/alpha_u) N_D ] [y]   [g_y]
//     [ -alpha_t M_w   E_I^T               ] [p] = [g_p],   N_D = B_D R^{-1} B_D^T,
// so its Schur complement (y eliminated) is
//     S_D = E_I^T + beta M_w E_I^{-1} B_D R^{-1} B_D^T,   beta = alpha_t eta^2/alpha_u.
// N_D is block tridiagonal in time, which rules out the matching/sweep Schur
// approximations (they need a spatial N) and the Neumann boundary-reduced
// operator (it needs I (x) N).  But the coupling has rank <= N_t |C|, and the
// Woodbury identity applied to S_D = E_I^T (I + beta E_I^{-T} M_w E_I^{-1} B_D
// R^{-1} B_D^T) gives the EXACT inverse
//     S_D^{-1} b = w - beta E_I^{-T} M_w E_I^{-1} B_D (R + beta H)^{-1} B_D^T w,
//     w = E_I^{-T} b,   H = B_D^T E_I^{-T} M_w E_I^{-1} B_D   (N_t|C| x N_t|C|).
// H depends on the mesh, the operators, `steps` and `dt` only; beta enters
// through the dense SPD matrix K = I (x) R_Gamma + beta H, factored by Prepare.
//
// H is assembled from the state responses of the unit controls: with
// B_D = I (x) G - Sub (x) Hs (G = P_I A Q_C, Hs = P_I (M/dt) Q_C on the
// unhatted matrices) and E_I lower block-Toeplitz, the response to the unit
// control of slot m and face DOF c is the response to slot 1 shifted by m-1
// slots, so only the |C| response sequences y^c_1..N_t are computed (N_t |C|
// direct spatial solves).  With the Gram blocks Gamma_ab = Y_a^T M_w Y_b,
//     H_{m,m'} = sum_{j=1}^{N_t-m'+1} Gamma_{j+d, j},   d = m' - m >= 0,
// and H_{m',m} = H_{m,m'}^T.  Everything dense is replicated on every rank
// (sizes: |C| = 2 nx + 1 face DOFs per slot; N_t |C| ~ 10^3-10^4).
//
// CRANK-NICOLSON.  Every step above survives unchanged in FORM; three things
// change, and they are independent of one another:
//   * the blocks:  E_CN = blockdiag(Aplus_I) + blocksubdiag(Aminus_I),
//     B_D_CN = blockdiag(Gplus_IC) + blocksubdiag(Gminus_IC), so the response
//     recursion becomes  y^c_1 = Aplus^-1 g_c,
//     y^c_2 = Aplus^-1 (h_c - Aminus y^c_1),  y^c_k = Aplus^-1 (-Aminus y^c_{k-1})
//     with g_c = P_I A_plus Q_C e_c and h_c = P_I A_minus Q_C e_c;
//   * the TRACKING weights What_y = diag(1,...,1,1/2) sit inside H, on the
//     SUMMATION index of the shifted Gram sum:
//         H_{m,m'} = sum_j what_{j+m'} Gamma_{j+d, j},
//     so they cannot be folded into Gamma and are applied in Prepare;
//   * the CONTROL weights What_u = diag(1,...,1,1/2) sit in R_T = What_u (x)
//     R_Gamma, i.e. only in the block-diagonal (d = 0) term of K.
// The two half weights are numerically equal and conceptually distinct: one
// belongs to the state quadrature, the other to the control quadrature.
// The Woodbury identity itself is scheme independent -- H = Z^T (What_y (x)
// M_w) Z with Z = E^-1 B_D is symmetric PSD for any symmetric weights, so
// K = R_T + beta H stays SPD.  Setting (Aplus, Aminus, what) -> (A, -M/dt, 1)
// recovers the backward-Euler formulas above; the derivation and both limits
// are checked on random dense operators by
// aux/cn_stage3_dirichlet/derivation_check.py.
//
// One application of S_D^{-1} costs three exact time sweeps (direct spatial
// factor of A_I = M_I/dt + K_I), one B_D and one B_D^T, one dense triangular
// solve pair.  Used as the PCSHELL of the Schur split (pc mode
// SCHUR_DIRICHLET_REDUCED); FGMRES then converges in O(1) iterations.
//
// LIFETIME: the DirichletControlOperators and the matrices are borrowed and
// must outlive the object; everything else is owned.  Not copyable or
// movable (the PCSHELL callback keeps its address).

#pragma once

#include <petscksp.h>
#include <petscmat.h>

#include <string>
#include <vector>

#include "Config.hpp"
#include "DirichletControl.hpp"
#include "Preconditioners.hpp"

class DirichletReducedSchurOperators {
 public:
  /** Collective.  `mass`/`stiffness` are the hatted M_I, K_I (identity /
   *  zero on C), `tracking` the (hatted) tracking mass matrix; the unhatted
   *  matrices come from `dirichlet`. */
  DirichletReducedSchurOperators(const ProblemConfig& config,
                                 DirichletControlOperators& dirichlet,
                                 Mat mass, Mat stiffness, Mat tracking);
  ~DirichletReducedSchurOperators();
  DirichletReducedSchurOperators(const DirichletReducedSchurOperators&) = delete;
  DirichletReducedSchurOperators& operator=(
      const DirichletReducedSchurOperators&) = delete;

  /** K = I (x) R_Gamma + beta H, Cholesky-factored (replicated; collective).
   *  Returns without work when `beta` equals the prepared value. */
  PetscErrorCode Prepare(double beta);
  double prepared_beta() const { return prepared_beta_; }

  /** x = S_D^{-1} b for the prepared beta (collective). */
  PetscErrorCode Solve(Vec b, Vec x);
  /** Sv = E_I^T v + beta M_w E_I^{-1} B_D R^{-1} B_D^T v (verification). */
  PetscErrorCode ApplySchur(double beta, Vec v, Vec Sv);

  /** A_I = M_I/dt + K_I, the diagonal block of E_I (owned here);
   *  Aplus_I = M_I/dt + K_I/2 under Crank-Nicolson. */
  Mat diagonal() const { return A_; }
  /** The temporal coupling block of a sweep on `diagonal()` and its scale:
   *  (hatted mass, 1/dt) under backward Euler -- the block is (1/dt) M_I --
   *  and (-Aminus_I, 1) under Crank-Nicolson.  A caller that builds its own
   *  sweep on `diagonal()` must use this pair, not the mass matrix. */
  Mat coupling() const { return coupling_ ? coupling_ : mass_; }
  double coupling_scale() const { return coupling_ ? 1.0 : 1.0 / dt_; }
  PetscInt control_count() const { return n_C_; }
  PetscInt reduced_dimension() const { return static_cast<PetscInt>(steps_) * n_C_; }
  const std::string& factor_backend() const { return backend_; }
  /** Direct spatial solves of the sweeps (forward and backward together). */
  long long sweep_solves() const { return forward_.counter.solves + backward_.counter.solves; }
  double setup_seconds() const { return setup_seconds_; }
  /** Read-only ingredients of the dense reduced system (verification):
   *  the Gram blocks Gamma_ab (index a(a+1)/2 + b, a >= b, |C| x |C|
   *  column-major, replicated), R_Gamma on C (|C| x |C|, replicated) and the
   *  sorted global C list. */
  const std::vector<std::vector<double>>& gram_blocks() const { return gram_; }
  const std::vector<double>& regularization_block() const { return R_dense_; }
  const std::vector<PetscInt>& control_dofs() const { return C_global_; }
  unsigned steps() const { return steps_; }
  /** The trapezoidal TRACKING weight of a slot (What_y); 1 under backward
   *  Euler.  The control weight What_u lives in DirichletControlOperators. */
  double SlotWeight(unsigned slot) const {
    return (scheme_ == ProblemConfig::TimeScheme::CRANK_NICOLSON &&
            slot + 1u == steps_) ? 0.5 : 1.0;
  }

 private:
  void BuildOperators(Mat mass, Mat stiffness);
  void BuildResponses();
  void BuildGram();
  void BuildRegularizationBlock();
  PetscErrorCode ApplyTrackingSlices(Vec v, Vec out);   // (I (x) M_w) v

  DirichletControlOperators& dirichlet_;
  ProblemConfig::TimeScheme scheme_ =
      ProblemConfig::TimeScheme::BACKWARD_EULER;
  Mat tracking_ = nullptr;   // borrowed
  // Backward Euler: A = M_I/dt + K_I and the sweep couples through (1/dt) M_I
  // (the borrowed `mass_`, so `coupling_` stays null).
  // Crank-Nicolson: A = Aplus_I = M_I/dt + K_I/2, Aminus_I = -M_I/dt + K_I/2
  // and the sweep couples through the owned -Aminus_I with unit scale.
  Mat A_ = nullptr;          // owned: M_I/dt + K_I  /  Aplus_I
  Mat A_minus_ = nullptr;    // owned, Crank-Nicolson only: Aminus_I
  Mat coupling_ = nullptr;   // owned, Crank-Nicolson only: -Aminus_I
  Mat mass_ = nullptr;       // borrowed hatted M_I (sweep coupling)
  KSP spatial_ = nullptr;    // direct factor of A_, adopted by the sweeps
  std::string backend_;
  TimeSweepContext forward_, backward_;
  unsigned steps_ = 0;
  double dt_ = 0.0;
  PetscInt n_ = 0, local_ = 0, begin_ = 0, n_C_ = 0;
  std::vector<PetscInt> C_global_;                // sorted, replicated
  std::vector<PetscInt> C_owned_position_;        // position in C_global_ of each owned C DOF
  std::vector<std::vector<double>> responses_;    // [k]: local x |C|, column-major, y^c_{k+1}
  std::vector<std::vector<double>> gram_;         // [a*(a+1)/2 + b], a >= b: |C| x |C|, replicated
  std::vector<double> R_dense_;                   // R_Gamma on C, |C| x |C|, replicated
  std::vector<double> K_;                         // I (x) R_Gamma + beta H, Cholesky factor
  double prepared_beta_ = -1.0;
  double setup_seconds_ = 0.0;
  Vec w_ = nullptr, t_ = nullptr, s_ = nullptr, q_ = nullptr;   // owned space-time workspace
  Vec spatial_a_ = nullptr, spatial_b_ = nullptr, spatial_c_ = nullptr;   // owned spatial workspace
  std::vector<double> reduced_a_, reduced_b_;                 // replicated reduced vectors
};

/** PCSHELL apply: context = DirichletReducedSchurOperators*. */
PetscErrorCode ApplyDirichletReducedSchur(PC pc, Vec b, Vec x);
