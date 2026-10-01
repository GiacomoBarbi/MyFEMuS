// Problem configuration and the pure model predicates derived from it.
//
// Base layer of the application: depends on nothing but the C++ standard
// library.  No FEMuS, PETSc or MPI type appears here.

#pragma once

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

/** pi to full double precision.  Declared here because <cmath> does not
 *  guarantee M_PI in a standard-conforming mode. */
constexpr double kPi = 3.14159265358979323846;

struct ProblemConfig {
  // NEUMANN and DIRICHLET act on the boundary face `neumann_control_face`
  // (the flux, respectively the trace, of the state); DISTRIBUTED acts on a
  // volume region.
  enum class ControlType { NEUMANN, DISTRIBUTED, DIRICHLET };
  enum class TrackingTargetType {
    CONSTANT,
    SIN_X_MINUS_T,
    ONE_MINUS_Y_SIN_X_MINUS_T,
    EXP_MINUS_T_COS2PI,
    // -64 exp(-((x - t/2)^2 + (z - t/2)^2)): a negative Gaussian whose centre
    // travels along the diagonal from the origin to (1/2, 1/2) over [0, 1].
    MOVING_GAUSSIAN
  };

  /** Functional form of a scalar data field (source term, initial state).
   *
   *  Every form is a fixed analytic shape multiplied by the corresponding
   *  scalar amplitude already stored in ProblemConfig (`source`,
   *  `initial_state`); CONSTANT returns the amplitude itself.
   *
   *  Forms are named after the mathematics, not after any benchmark that
   *  happens to use them. */
  enum class DataFieldForm {
    CONSTANT,
    // amplitude * exp(-t) * cos(2*pi*x) * cos(2*pi*z)
    EXP_MINUS_T_COS2PI
  };

  /** Boundary condition imposed on the state.
   *
   *  NATURAL is the default: it leaves every boundary unconstrained
   *  (homogeneous Neumann).
   *
   *  HOMOGENEOUS_DIRICHLET constrains the state to zero on the whole external
   *  boundary.  It is opt-in, and Validate() rejects it for boundary control,
   *  where the control lives on the boundary the constraint would remove. */
  enum class StateBoundaryCondition { NATURAL, HOMOGENEOUS_DIRICHLET };

  struct Domain {
    double xmin = 0.0;
    double xmax = 1.0;
    double zmin = 0.0;
    double zmax = 1.0;
  } domain;

  struct MeshConfig {
    unsigned nx = 4;
    unsigned nz = 4;
  } mesh;

  /** Time discretization of the state equation.
   *
   *  BACKWARD_EULER is the production default.  CRANK_NICOLSON is available
   *  for every control type, with the trapezoidal tracking quadrature:
   *  piecewise-constant interval controls for DISTRIBUTED and NEUMANN, nodal
   *  controls for DIRICHLET (whose control IS the state trace).  See
   *  EX20_CRANK_NICOLSON_STAGE{1_DISTRIBUTED,2_NEUMANN,3_DIRICHLET}.md. */
  enum class TimeScheme { BACKWARD_EULER, CRANK_NICOLSON };

  struct TimeConfig {
    unsigned steps = 3;
    double dt = 0.05;
    TimeScheme scheme = TimeScheme::BACKWARD_EULER;
  } time;

  /** A rectangular sub-region of the domain: the one geometric type behind
   *  the primary tracking box, the distributed-control box and the additional
   *  regions of either kind.  Which role a box plays is said by the member
   *  that holds it, not by its type. */
  struct Box {
    double xmin = 0.0;
    double xmax = 1.0;
    double zmin = 0.0;
    double zmax = 1.0;
  };

  Box tracking_box{0.25, 0.75, 0.25, 0.75};

  struct ControlConfig {
    ControlType type = ControlType::NEUMANN;
    Box region{0.25, 0.75, 0.25, 0.75};
  } control;

  struct TrackingConfig {
    TrackingTargetType target_type = TrackingTargetType::CONSTANT;
    double parameter = 1.0;
  } tracking;

  /** Tracking / distributed-control regions BEYOND the primary box.
   *
   *  The discrete regions are the UNIONS
   *
   *      omega = tracking_box   U  additional_tracking_regions
   *      C     = control.region U  additional_control_regions
   *
   *  and the corresponding matrices are still
   *
   *      (M_omega)_ab = int_omega phi_a phi_b ,   N_ab = int_C phi_a phi_b .
   *
   *  Both lists default to EMPTY, in which case every region predicate reduces
   *  to the single-box test.
   *
   *  These apply to VOLUME regions only.  Neumann boundary control is
   *  untouched: its region is Gamma_c, selected by neumann_control_face. */
  std::vector<Box> additional_tracking_regions;
  std::vector<Box> additional_control_regions;

  // AMPLITUDES of the three scalar data fields; the *_form selectors below say
  // which analytic shape each amplitude multiplies.
  double desired_state = 0.5;
  double initial_state = 0.0;
  double source = 0.0;

  struct DataConfig {
    DataFieldForm source_form = DataFieldForm::CONSTANT;
    DataFieldForm initial_form = DataFieldForm::CONSTANT;
  } data;

  /** Optional analytic overrides of the three data fields.
   *
   *  A non-null pointer REPLACES the DataFieldForm dispatch for that field.
   *  Production never sets one; this is what lets the production layers stay
   *  ignorant of every benchmark -- the config carries a function, not a
   *  catalogue of test cases.
   *
   *  Plain function pointers -- no std::function, no allocation, no ownership,
   *  and callable from inside the DOF loops of the data side. */
  struct DataOverrides {
    double (*source)(const ProblemConfig&, double x, double z,
                     double time) = nullptr;
    double (*initial_state)(const ProblemConfig&, double x, double z) = nullptr;
    double (*tracking_target)(const ProblemConfig&, double x, double z,
                              double time) = nullptr;
  } data_overrides;

  double prescribed_neumann_flux = 0.0;

  // Physical diffusivity, used directly by ApplyStiffness.
  double diffusivity = 1.0;

  // Full bottom boundary y/z=0 in the FEMuS box face numbering: the control
  // face Gamma_c of both boundary control types.
  unsigned neumann_control_face = 1;
  std::string neumann_control_boundary = "y0";

  // Dirichlet control regularization  r(u,v) = (u,v)_Gamma + ell^2 (d_s u, d_s v)_Gamma:
  // the H1(Gamma_c) length scale ell; 0 is the L2(Gamma_c) norm.
  double control_h1_length = 0.0;

  // Dirichlet control TEMPORAL regularization
  //   (gamma_u/2) \int_0^T ||d_t u||^2_{L2(Gamma_c)} dt :
  // the weight gamma_u.  The metric is the face MASS matrix M_Gamma, NOT the
  // spatial regularization R_Gamma = M_Gamma + ell^2 K_Gamma -- the two are
  // independent and both may be active.  0 disables the term and reproduces
  // the accepted (stage 1-3) code path numerically.
  double control_h1_time_weight = 0.0;

  StateBoundaryCondition state_bc = StateBoundaryCondition::NATURAL;

  double alpha_u = 0.01;
  double eta = 1.0;

  // Direct coefficient of 0.5*alpha_track*||y-y_d||^2_Mtrack.
  // It is a physical/discrete-functional input and is independent of the
  // mesh: no hidden or mesh-dependent normalization is applied.
  double alpha_track = 0.1364210526315784;

  double final_time() const { return time.steps * time.dt; }

  /** The two box checks every region undergoes, with the region's own
   *  messages: a non-empty extent, then containment in the domain. */
  static void ValidateBox(const Box& box, const Domain& domain,
                          const char* invalid_message,
                          const char* outside_message) {
    if (!(box.xmin < box.xmax && box.zmin < box.zmax))
      throw std::invalid_argument(invalid_message);
    if (box.xmin < domain.xmin || box.xmax > domain.xmax ||
        box.zmin < domain.zmin || box.zmax > domain.zmax)
      throw std::invalid_argument(outside_message);
  }

  void Validate() const {
    // The Neumann control lives on the boundary, so constraining the state
    // there would delete the control itself.
    if (state_bc == StateBoundaryCondition::HOMOGENEOUS_DIRICHLET &&
        control.type == ControlType::NEUMANN)
      throw std::invalid_argument(
          "homogeneous Dirichlet state BC is incompatible with Neumann control");
    if (state_bc == StateBoundaryCondition::HOMOGENEOUS_DIRICHLET &&
        control.type == ControlType::DIRICHLET)
      throw std::invalid_argument(
          "homogeneous Dirichlet state BC is incompatible with Dirichlet control");
    if (control_h1_length < 0.0 || !std::isfinite(control_h1_length))
      throw std::invalid_argument("invalid control H1 length");
    if (control_h1_time_weight < 0.0 || !std::isfinite(control_h1_time_weight))
      throw std::invalid_argument("invalid control H1-in-time weight");
    // The temporal seminorm of the control is derived and implemented for the
    // NODAL Dirichlet control only.  The distributed and Neumann controls are
    // piecewise-constant INTERVAL quantities whose discrete time derivative is
    // a different object; they are rejected rather than silently reinterpreted.
    if (control_h1_time_weight > 0.0 && control.type != ControlType::DIRICHLET)
      throw std::invalid_argument(
          "--control-h1-time-weight requires Dirichlet boundary control"
          " (--dirichlet-solve)");
    if (!(domain.xmin < domain.xmax && domain.zmin < domain.zmax))
      throw std::invalid_argument("invalid domain bounds");
    if (mesh.nx == 0 || mesh.nz == 0)
      throw std::invalid_argument("mesh dimensions must be positive");
    if (time.steps == 0 || time.dt <= 0.0)
      throw std::invalid_argument("invalid time discretization");
    ValidateBox(tracking_box, domain, "invalid tracking box",
                "tracking box lies outside the domain");
    ValidateBox(control.region, domain, "invalid distributed-control box",
                "distributed-control box lies outside the domain");
    for (const Box& box : additional_tracking_regions)
      ValidateBox(box, domain, "invalid additional tracking region",
                  "additional tracking region lies outside the domain");
    for (const Box& box : additional_control_regions)
      ValidateBox(box, domain, "invalid additional control region",
                  "additional control region lies outside the domain");
    if (diffusivity <= 0.0 || alpha_u <= 0.0 || alpha_track < 0.0 ||
        eta == 0.0 || !std::isfinite(tracking.parameter))
      throw std::invalid_argument("invalid physical/control coefficient");
    // Crank-Nicolson covers every control type: DISTRIBUTED (stage 1) and
    // NEUMANN (stage 2) with piecewise-constant interval controls, DIRICHLET
    // (stage 3) with nodal controls and u_0 = y_0|_Gamma_c / eta as data.
  }
};

enum class DistributedSolvePc {
  ADDITIVE_GAMG,
  SCHUR_MATCHING,
  // Matching Schur whose three internal space-time solves are exact
  // triangular time sweeps with spatial CG+GAMG.
  SCHUR_SWEEP,
  // Exact Schur solve by elimination of the control-inactive coordinates
  // (boundary control): the reduced dense operator on the control support,
  // affine in beta, is factored once per beta.
  SCHUR_BOUNDARY_REDUCED,
  // Exact Schur solve for Dirichlet boundary control: Woodbury reduction of
  // the low-rank coupling to the control space (dense N_t |C| operator,
  // factored once per beta).
  SCHUR_DIRICHLET_REDUCED
};

inline bool IsMatchingSchur(DistributedSolvePc mode) {
  return mode == DistributedSolvePc::SCHUR_MATCHING ||
         mode == DistributedSolvePc::SCHUR_SWEEP;
}

inline bool IsBoundaryReduced(DistributedSolvePc mode) {
  return mode == DistributedSolvePc::SCHUR_BOUNDARY_REDUCED;
}

inline bool IsDirichletReduced(DistributedSolvePc mode) {
  return mode == DistributedSolvePc::SCHUR_DIRICHLET_REDUCED;
}

inline bool ControlRegionIsWholeDomain(const ProblemConfig& config) {
  const double tolerance = 64.0 * std::numeric_limits<double>::epsilon();
  return std::fabs(config.control.region.xmin - config.domain.xmin) <=
             tolerance &&
         std::fabs(config.control.region.xmax - config.domain.xmax) <=
             tolerance &&
         std::fabs(config.control.region.zmin - config.domain.zmin) <=
             tolerance &&
         std::fabs(config.control.region.zmax - config.domain.zmax) <=
             tolerance;
}

/** True when the run uses the Crank-Nicolson state discretisation.  Every
 *  production call site branches on this and leaves the backward-Euler
 *  formulas -- and their floating-point operation ordering -- untouched. */
inline bool IsCrankNicolson(const ProblemConfig& config) {
  return config.time.scheme == ProblemConfig::TimeScheme::CRANK_NICOLSON;
}

inline const char* TimeSchemeName(const ProblemConfig& config) {
  return IsCrankNicolson(config) ? "CN" : "BE";
}

/** Temporal weight of tracking slot `slot` (the unknown y_{slot+1}), WITHOUT
 *  the dt factor.
 *
 *  Backward Euler: the right-rectangle rule gives every node weight 1.
 *  Crank-Nicolson: the trapezoidal rule gives the final node weight 1/2 (the
 *  y_0 half-weight is an additive constant of the objective, not an unknown).
 *
 *  This is W_y/dt of the derivation; ex20's adjoint is the dt-scaled multiplier
 *  P = P_Lagrange/dt, which is exactly what removes the remaining dt from the
 *  (2,1) block -- see the report, section "reduced 2x2". */
inline double TrackingSlotWeight(const ProblemConfig& config, unsigned slot) {
  if (!IsCrankNicolson(config)) return 1.0;
  return (slot + 1u == config.time.steps) ? 0.5 : 1.0;
}

/** The single separable space-time mode shared by the non-constant data
 *  forms: exp(-t) cos(2*pi*x) cos(2*pi*z), of unit amplitude.
 *
 *  cos(2*pi*x) cos(2*pi*z) is an eigenfunction of the Laplacian on the unit
 *  square with homogeneous Neumann data, which is why analytic parabolic test
 *  problems posed on this geometry keep reusing it. */
inline double UnitCosineMode(double x, double z, double time) {
  const double two_pi = 2.0 * kPi;
  return std::exp(-time) * std::cos(two_pi * x) * std::cos(two_pi * z);
}

/** Distributed source term f(x, z, t) of the state equation. */
inline double EvaluateSource(const ProblemConfig& config, double x, double z,
                             double time) {
  if (config.data_overrides.source)
    return config.data_overrides.source(config, x, z, time);
  if (config.data.source_form ==
      ProblemConfig::DataFieldForm::EXP_MINUS_T_COS2PI)
    return config.source * UnitCosineMode(x, z, time);
  return config.source;
}

/** Initial state y_0(x, z) of the state equation. */
inline double EvaluateInitialState(const ProblemConfig& config, double x,
                                   double z) {
  if (config.data_overrides.initial_state)
    return config.data_overrides.initial_state(config, x, z);
  if (config.data.initial_form ==
      ProblemConfig::DataFieldForm::EXP_MINUS_T_COS2PI)
    return config.initial_state * UnitCosineMode(x, z, 0.0);
  return config.initial_state;
}

inline double EvaluateTrackingTarget(const ProblemConfig& config, double x,
                                     double z, double time) {
  if (config.data_overrides.tracking_target)
    return config.data_overrides.tracking_target(config, x, z, time);
  if (config.tracking.target_type ==
      ProblemConfig::TrackingTargetType::SIN_X_MINUS_T)
    return std::sin(config.tracking.parameter * (x - time));
  if (config.tracking.target_type ==
      ProblemConfig::TrackingTargetType::ONE_MINUS_Y_SIN_X_MINUS_T)
    return (1.0 - z) *
           std::sin(config.tracking.parameter * (x - time));
  if (config.tracking.target_type ==
      ProblemConfig::TrackingTargetType::EXP_MINUS_T_COS2PI)
    return config.desired_state * UnitCosineMode(x, z, time);
  if (config.tracking.target_type ==
      ProblemConfig::TrackingTargetType::MOVING_GAUSSIAN) {
    const double dx = x - 0.5 * time;
    const double dz = z - 0.5 * time;
    return -64.0 * std::exp(-(dx * dx + dz * dz));
  }
  return config.desired_state;
}

/** The temporal regularization is active only when its weight is positive;
 *  every call site uses this predicate so the gamma_u = 0 path is the
 *  accepted one, byte for byte. */
inline bool HasControlTimeRegularization(const ProblemConfig& config) {
  return config.control_h1_time_weight > 0.0;
}

/** The temporal weight AS IT ENTERS THE KKT SYSTEM, gamma_u/dt.
 *
 *  ex20 carries the dt-scaled multiplier p = P_Lagrange/dt, i.e. every KKT row
 *  is the true derivative divided by dt.  The spatial regularization
 *  (alpha_u/2) dt sum_n w_n u_n^T R_Gamma u_n then contributes alpha_u R_T U
 *  with no dt, but the temporal energy gamma_u/(2 dt) sum (u_n-u_{n-1})^T
 *  M_Gamma (u_n-u_{n-1}) has no dt of its own, so dividing by dt leaves
 *  gamma_u/dt in front of L_t and of q_0.  Getting this factor wrong makes the
 *  solver minimise a functional that differs from the reported one by a power
 *  of dt, which is why the discrete gradient is checked by finite differences
 *  against the reported objective. */
inline double ControlTimeSourceWeight(const ProblemConfig& config) {
  return config.control_h1_time_weight / config.time.dt;
}

/** rho = gamma_u / (alpha_u dt): the temporal stiffness in units of the spatial
 *  regularization.  The control operator of the KKT system is
 *      Q_u = alpha_u (What_u (x) R_Gamma + rho L_t) = alpha_u R_H1t,
 *  so every accepted call site that divides by alpha_u keeps its scaling and
 *  only R_Gamma -> R_H1t changes. */
inline double ControlTimeRatio(const ProblemConfig& config) {
  return ControlTimeSourceWeight(config) / config.alpha_u;
}

inline const char* TrackingTargetName(const ProblemConfig& config) {
  if (config.tracking.target_type ==
      ProblemConfig::TrackingTargetType::SIN_X_MINUS_T)
    return "sin_x_minus_t";
  if (config.tracking.target_type ==
      ProblemConfig::TrackingTargetType::ONE_MINUS_Y_SIN_X_MINUS_T)
    return "one_minus_y_sin_x_minus_t";
  if (config.tracking.target_type ==
      ProblemConfig::TrackingTargetType::EXP_MINUS_T_COS2PI)
    return "exp_minus_t_cos2pi";
  return "constant";
}

/** Purely geometric control mask, for diagnostics only.
 *
 *  This predicate does NOT build the control: it disagrees with the support of
 *  the assembled operator N whenever the mesh is not aligned with the control
 *  box, in both directions.  The production active set is ActiveControlDofs();
 *  this one is kept so the verification driver can report the difference. */
inline bool OwnedDofIsControlled(const ProblemConfig& config, double x, double z) {
  const double tolerance = 1.0e-12;
  if (config.control.type == ProblemConfig::ControlType::NEUMANN ||
      config.control.type == ProblemConfig::ControlType::DIRICHLET)
    return std::fabs(z - config.domain.zmin) <= tolerance;
  return x >= config.control.region.xmin - tolerance &&
         x <= config.control.region.xmax + tolerance &&
         z >= config.control.region.zmin - tolerance &&
         z <= config.control.region.zmax + tolerance;
}
