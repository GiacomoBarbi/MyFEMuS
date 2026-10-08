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

/** T11N: the six faces of the 3D HEX27 box in the FEMuS face numbering, as
 *  measured from the JacobianSur outward normals in T10 (G1N face report):
 *    1 zmin (0,0,-1)   2 ymin (0,-1,0)   3 xmax (+1,0,0)
 *    4 ymax (0,+1,0)   5 xmin (-1,0,0)   6 zmax (0,0,+1)
 *  (storage axes 0, 1, 2 = x, y, z).  The assembly re-verifies the normal and
 *  the area of the selected face at run time (Discretization.cpp). */
struct BoxFace3D {
  unsigned id;      // FEMuS face id
  unsigned axis;    // storage axis normal to the face
  bool at_max;      // outward normal +e_axis (true) or -e_axis (false)
  const char* name;
};
inline bool LookupBoxFace3D(unsigned id, BoxFace3D& face) {
  static const BoxFace3D faces[6] = {{1u, 2u, false, "zmin"}, {2u, 1u, false, "ymin"},
                                     {3u, 0u, true, "xmax"},  {4u, 1u, true, "ymax"},
                                     {5u, 0u, false, "xmin"}, {6u, 2u, true, "zmax"}};
  for (const BoxFace3D& f : faces)
    if (f.id == id) { face = f; return true; }
  return false;
}
inline bool LookupBoxFace3D(const std::string& name, BoxFace3D& face) {
  for (unsigned id = 1; id <= 6u; ++id)
    if (LookupBoxFace3D(id, face) && name == face.name) return true;
  return false;
}

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

  /** Storage axes (T9): axis 0 = x always.  In 2D the second storage axis is
   *  historically called "z" (zmin/zmax, nz, vy).  In 3D the storage axes are
   *  (x, y, z): axis 1 = y (ymin/ymax, ny, vy) and axis 2 = z (zmin/zmax, nz,
   *  vz).  ymin/ymax and ny are read only in 3D. */
  struct Domain {
    double xmin = 0.0;
    double xmax = 1.0;
    double zmin = 0.0;
    double zmax = 1.0;
    double ymin = 0.0;   // 3D only (storage axis 1)
    double ymax = 1.0;   // 3D only
  } domain;

  struct MeshConfig {
    unsigned nx = 4;
    unsigned nz = 4;
    unsigned ny = 4;     // 3D only (storage axis 1)
  } mesh;

  /** Spatial dimension of the mesh: 2 (QUAD9 box, the ex20 default), 1, or
   *  (T9, phase 1) 3 = HEX27 box with the restricted feature set of Validate.
   *  Original note: 2 (QUAD9 box, the ex20 default) or 1
   *  (EDGE3 interval [xmin, xmax]).  In 1D every z coordinate is 0, the z
   *  extents of the domain and of every box are ignored, and only DISTRIBUTED
   *  control is available (there is no control face). */
  unsigned dimension = 2;

  /** Linear convection  y_t - nu Lap y + v . grad y = f + eta u  (ex21).
   *
   *  Strong (non-conservative) form with a constant velocity v = (vx, vy);
   *  vy acts along the second mesh coordinate, which this code calls z.  The
   *  Galerkin term is  C_ij = int_Omega (v . grad phi_j) phi_i  and it is added
   *  to the stiffness, so the spatial operator becomes L = K + C (row i = test
   *  function, column j = trial function), which is NOT symmetric.
   *
   *  `enabled == false` (the default) is the ex20 code path, operation for
   *  operation.  `enabled == true` with v = 0 assembles a zero C but already
   *  routes every adjoint product through the transpose, which is what the
   *  zero-convection baseline tests. */
  struct ConvectionConfig {
    bool enabled = false;
    double vx = 0.0;
    double vy = 0.0;
    double vz = 0.0;   // T9: 3D only (storage axis 2); must stay 0 in 1D/2D
    /** Optional VARIABLE, time-independent velocity w(x, z), evaluated at
     *  every quadrature point of the convection assembly.  Null (the default)
     *  is the constant pair (vx, vy) above, operation for operation.  The
     *  Galerkin term stays  C_ij = int (w . grad phi_j) phi_i  and the adjoint
     *  stays the exact transpose of the assembled operator. */
    void (*field)(const ProblemConfig&, double x, double z, double w[2]) = nullptr;
    const char* field_name = nullptr;   // for logs only
    /** T6: optional convection stabilisation, OFF by default.  LPS is the
     *  ex21 Q2 version of Leveque-Pearson's (NLAA 2022, eqs. (7)-(8)) local
     *  projection stabilisation: L = nu K + N + W, with
     *    W_ij = sum_k delta_k int_{P_k} kappa_k(w.grad phi_i) kappa_k(w.grad phi_j),
     *  kappa_k = id - pi_k, P_k = ONE Q2 element (the same square as an LP
     *  2x2 Q1 patch on LP's node grid), pi_k the L2 projection onto P0
     *  (LP's patch mean, eq. (8)) or, as a labelled diagnostic variant, onto
     *  P1 on the patch.  delta_k = h_k/(2|w_k|)(1 - 1/Pe_k) if
     *  Pe_k = |w_k| h_k/(2 nu) > 1 and 0 otherwise; w_k = w at the patch
     *  centroid, h_k = chord of the patch through its centroid along w_k.
     *  W is symmetric PSD and even in w, so the adjoint (exact transpose of
     *  the assembled L) carries the same W.  See BuildLpsMatrix. */
    enum class Stabilization { NONE, LPS };
    Stabilization stabilization = Stabilization::NONE;
    enum class LpsProjection { P0, P1 };
    LpsProjection lps_projection = LpsProjection::P0;
  } convection;

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
    // T10: 3D only, storage axis 1 (as Domain): in 3D the box is
    // [xmin,xmax] x [ymin,ymax] x [zmin,zmax] over storage axes (0, 1, 2).
    // Never read in 1D/2D, so the 4-value aggregate initialisers keep working.
    double ymin = 0.0;
    double ymax = 1.0;
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

  /** ex21: support of the distributed source term.  EMPTY (the default, ex20)
   *  means the source acts on the whole domain, rhs = M I_h f.  Otherwise the
   *  load is  int_{union of boxes} I_h f phi_i = M_source I_h f, with M_source
   *  the mass matrix over the boxes (element-centroid selection, like the
   *  control box) -- the discretisation of a disturbance B_d d = chi_{Omega_d} d
   *  without interpolating a discontinuous indicator. */
  std::vector<Box> source_regions;

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
    /** ex21: time-independent INHOMOGENEOUS Dirichlet state data g(x, z).
     *  Only with state_bc == HOMOGENEOUS_DIRICHLET (the constrained set is the
     *  same) and distributed control.  Implemented by LIFTING: the solver's
     *  unknown is y_0 = y - G (G = g at the constrained DOFs, 0 elsewhere), its
     *  free rows receive the constant load -(L_full G)_free, and the returned
     *  state is y_0 + G.  The adjoint, the control and the observation spaces
     *  are those of the homogeneous problem (see Discretization.cpp).  Null
     *  (the default) is the homogeneous problem, operation for operation. */
    double (*dirichlet_state)(const ProblemConfig&, double x, double z) = nullptr;
    /** T9: 3D data (x, y, z) = storage axes (0, 1, 2); read only when
     *  dimension == 3.  Null in 3D falls back to the built-in forms that do
     *  not depend on the second/third coordinate (Validate refuses the rest). */
    double (*source3d)(const ProblemConfig&, double x, double y, double z,
                       double time) = nullptr;
    double (*initial3d)(const ProblemConfig&, double x, double y, double z) = nullptr;
    double (*target3d)(const ProblemConfig&, double x, double y, double z,
                       double time) = nullptr;
  } data_overrides;

  /** T3b DIAGNOSTIC ONLY (negative control): false drops the affine tracking
   *  term M_F,Gamma (g - y_d)_Gamma of inhomogeneous Dirichlet data from the
   *  adjoint right-hand side, reproducing the pre-T3b behaviour.  Production
   *  never sets it; it has no effect without dirichlet_state. */
  bool lift_affine_tracking = true;

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

  /** T8D: how the Dirichlet-reduced Woodbury core K = R_H1t + beta H is solved.
   *  T8D finalisation: ITERATIVE + RT + rtol 1e-8 is the PRODUCTION DEFAULT;
   *  DENSE stays available as the reference / gold solver.
   *  DENSE (the validated reference): replicated dense Cholesky of K
   *  (and, for gamma_u > 0, of R_H1t), O((N_t |C|)^2) memory.
   *  ITERATIVE: K is a MatShell (B_D, forward sweep, tracking, backward sweep,
   *  B_D^T, + R_H1t) solved by CG on a DISTRIBUTED reduced vector, optionally
   *  preconditioned by R_H1t (sparse direct factor); R_H1t^{-1} itself also
   *  uses the sparse factor.  No O((N_t |C|)^2) object exists in this mode. */
  struct DirichletCoreConfig {
    enum class Solver { DENSE, ITERATIVE } solver = Solver::ITERATIVE;
    enum class Pc { NONE, RT } pc = Pc::RT;
    double rtol = 1e-8;
    // true when ANY core option was given on the command line: such options
    // are refused without Dirichlet control (the default itself is inert there)
    bool explicitly_set = false;
    // true when --dirichlet-core-solver itself was given (verify needs dense)
    bool solver_explicit = false;
    int maxit = 2000;
    // verification: compare the dense K with the matrix-free action, SPD checks
    bool verify = false;
  } dirichlet_core;

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
                          const char* outside_message,
                          unsigned dimension = 2u) {
    if (!(box.xmin < box.xmax && box.zmin < box.zmax))
      throw std::invalid_argument(invalid_message);
    if (box.xmin < domain.xmin || box.xmax > domain.xmax ||
        box.zmin < domain.zmin || box.zmax > domain.zmax)
      throw std::invalid_argument(outside_message);
    // T10: the third extent (storage axis 1) in 3D
    if (dimension == 3u) {
      if (!(box.ymin < box.ymax)) throw std::invalid_argument(invalid_message);
      if (box.ymin < domain.ymin || box.ymax > domain.ymax)
        throw std::invalid_argument(outside_message);
    }
  }
  // In 1D the z extents are inert: every z coordinate is 0 and the box tests
  // ignore z, so the default z ranges of the domain and boxes are left alone.

  void Validate() const {
    if (dimension != 1u && dimension != 2u && dimension != 3u)
      throw std::invalid_argument("dimension must be 1, 2 or 3");
    if (dimension != 3u && convection.vz != 0.0)
      throw std::invalid_argument("--vz is a 3D wind component (--dimension 3)");
    if (dimension == 3u) Validate3D();
    if (dimension == 1u && control.type != ControlType::DISTRIBUTED)
      throw std::invalid_argument(
          "1D meshes support DISTRIBUTED control only (no control face)");
    if (dimension == 1u && convection.vy != 0.0)
      throw std::invalid_argument("1D meshes have no vy component");
    if (!std::isfinite(convection.vx) || !std::isfinite(convection.vy))
      throw std::invalid_argument("invalid convection velocity");
    if (convection.field && !convection.enabled)
      throw std::invalid_argument("a velocity field requires --convection on");
    if (convection.stabilization != ConvectionConfig::Stabilization::NONE &&
        !convection.enabled)
      throw std::invalid_argument(
          "--convection-stabilization lps requires --convection on");
    if (data_overrides.dirichlet_state &&
        (state_bc != StateBoundaryCondition::HOMOGENEOUS_DIRICHLET ||
         control.type != ControlType::DISTRIBUTED))
      throw std::invalid_argument(
          "inhomogeneous Dirichlet state data need the Dirichlet state BC and"
          " distributed control");
    // ex21 stage 1 refused every boundary control under convection: the
    // boundary-control paths contain reduced operators and factorizations
    // derived for a SYMMETRIC stiffness (EX21_SOLVER_NONSYMMETRY_AUDIT.md).
    // T7N: NEUMANN control is accepted -- its control enters only through the
    // symmetric trace mass M_Gamma, and the generic KKT path (exact transpose
    // L^T in E^T, transposed sweep blocks) carries the convective adjoint
    // boundary condition nu dn p + (w.n) p = 0 by construction
    // (boundary_convection/T7N_NEUMANN_CONVECTION.md).  Its SPD-specific
    // boundary-reduced Schur solver stays refused (CommandLine.cpp).
    // T7D: DIRICHLET control is accepted too -- B_D^T now applies the
    // transposed blocks A^T / A_plus^T / A_minus^T (DirichletControl.cpp), and
    // the Dirichlet-reduced Schur solver factors A_I by LU with a transposed
    // backward sweep under convection (boundary_convection/T7D_*.md).
    // The Neumann control lives on the boundary, so constraining the state
    // there would delete the control itself.
    if (state_bc == StateBoundaryCondition::HOMOGENEOUS_DIRICHLET &&
        control.type == ControlType::NEUMANN)
      throw std::invalid_argument(
          "homogeneous Dirichlet state BC is incompatible with Neumann control");
    // T11D: in 3D the controlled face is excluded from the constrained set
    // (y = eta u on Gamma_c, y = 0 on the other five faces); 2D unchanged.
    if (state_bc == StateBoundaryCondition::HOMOGENEOUS_DIRICHLET &&
        control.type == ControlType::DIRICHLET && dimension != 3u)
      throw std::invalid_argument(
          "homogeneous Dirichlet state BC is incompatible with Dirichlet control");
    // T8D: the core options act on Dirichlet control only; refuse rather than ignore.
    if ((dirichlet_core.explicitly_set || dirichlet_core.verify) &&
        control.type != ControlType::DIRICHLET)
      throw std::invalid_argument(
          "--dirichlet-core-* options / --verify-dirichlet-core require"
          " Dirichlet control (--dirichlet-solve)");
    if (!(dirichlet_core.rtol > 0.0) || dirichlet_core.maxit <= 0)
      throw std::invalid_argument("--dirichlet-core-rtol / -maxit must be positive");
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
                "tracking box lies outside the domain", dimension);
    ValidateBox(control.region, domain, "invalid distributed-control box",
                "distributed-control box lies outside the domain", dimension);
    for (const Box& box : additional_tracking_regions)
      ValidateBox(box, domain, "invalid additional tracking region",
                  "additional tracking region lies outside the domain", dimension);
    for (const Box& box : additional_control_regions)
      ValidateBox(box, domain, "invalid additional control region",
                  "additional control region lies outside the domain", dimension);
    for (const Box& box : source_regions)
      ValidateBox(box, domain, "invalid source region",
                  "source region lies outside the domain", dimension);
    if (diffusivity <= 0.0 || alpha_u <= 0.0 || alpha_track < 0.0 ||
        eta == 0.0 || !std::isfinite(tracking.parameter))
      throw std::invalid_argument("invalid physical/control coefficient");
    // Crank-Nicolson covers every control type: DISTRIBUTED (stage 1) and
    // NEUMANN (stage 2) with piecewise-constant interval controls, DIRICHLET
    // (stage 3) with nodal controls and u_0 = y_0|_Gamma_c / eta as data.
  }

  /** 3D whitelist.  T9 (phase 1): distributed control, homogeneous Dirichlet
   *  state, constant wind (or none), BE/CN, L2 regularisation.  T10 (phase 2)
   *  adds: rectangular boxes (xmin,xmax)x(ymin,ymax)x(zmin,zmax), control and
   *  tracking boxes with multi-zone unions, source regions, and the natural
   *  homogeneous state BC.  Everything else is REFUSED, never reinterpreted.
   *  (Solver, run-mode and CLI restrictions: CommandLine.cpp.) */
  void Validate3D() const {
    auto refuse = [](const char* what) {
      throw std::invalid_argument(std::string("3D: ") + what);
    };
    // T11N / T11D-L2: Neumann and Dirichlet boundary control on ONE whole face.
    if (control.type == ControlType::DIRICHLET) {
      BoxFace3D face;
      if (!LookupBoxFace3D(neumann_control_face, face))
        refuse("Dirichlet control needs one whole box face (--dirichlet-face3d xmin|xmax|ymin|ymax|zmin|zmax)");
      if (state_bc != StateBoundaryCondition::HOMOGENEOUS_DIRICHLET)
        refuse("Dirichlet control needs --state-bc dirichlet (y = eta u on the face,"
               " y = 0 on the other five faces; the 2D natural-elsewhere variant is not ported)");
      if (!additional_control_regions.empty())
        refuse("control boxes are distributed-control regions; Dirichlet control acts on one face");
    }
    if (control.type == ControlType::NEUMANN) {
      BoxFace3D face;
      if (!LookupBoxFace3D(neumann_control_face, face))
        refuse("Neumann control needs one whole box face (--neumann-face3d xmin|xmax|ymin|ymax|zmin|zmax)");
      if (!additional_control_regions.empty())
        refuse("control boxes are distributed-control regions; Neumann control acts on one face");
    }
    if (!(domain.xmin < domain.xmax && domain.ymin < domain.ymax &&
          domain.zmin < domain.zmax) ||
        !std::isfinite(domain.xmin) || !std::isfinite(domain.xmax) ||
        !std::isfinite(domain.ymin) || !std::isfinite(domain.ymax) ||
        !std::isfinite(domain.zmin) || !std::isfinite(domain.zmax))
      refuse("invalid --domain3d (need xmin < xmax, ymin < ymax, zmin < zmax)");
    if (mesh.nx == 0 || mesh.ny == 0 || mesh.nz == 0)
      refuse("--nx, --ny, --nz must be positive");
    if (control_h1_length != 0.0 || control_h1_time_weight != 0.0)
      refuse("only L2 control regularisation (no H1-space / H1-time)");
    if (data_overrides.dirichlet_state) refuse("inhomogeneous Dirichlet lifting is not ported");
    if (convection.field) refuse("only a constant wind (named/variable winds are not ported)");
    if (convection.stabilization != ConvectionConfig::Stabilization::NONE)
      refuse("no convection stabilisation (LPS is not ported)");
    if (!std::isfinite(convection.vz)) refuse("invalid --vz");
    if (!convection.enabled && (convection.vx != 0.0 || convection.vy != 0.0))
      refuse("--vx/--vy/--vz need --convection on");
    // data forms that read the 2D second coordinate
    if (data.source_form != DataFieldForm::CONSTANT || data.initial_form != DataFieldForm::CONSTANT)
      refuse("only constant built-in source/initial data (the cos(2 pi x) cos(2 pi z) forms are 2D)");
    if (tracking.target_type != TrackingTargetType::CONSTANT &&
        tracking.target_type != TrackingTargetType::SIN_X_MINUS_T)
      refuse("only the constant and sin_x_minus_t built-in targets (the others are 2D-shaped)");
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
             tolerance &&
         // T10: in 3D the box must also span the y extent (storage axis 1);
         // otherwise the full mass M would silently replace the box operator.
         (config.dimension != 3u ||
          (std::fabs(config.control.region.ymin - config.domain.ymin) <= tolerance &&
           std::fabs(config.control.region.ymax - config.domain.ymax) <= tolerance));
}

/** True when the run uses the Crank-Nicolson state discretisation.  Every
 *  production call site branches on this and leaves the backward-Euler
 *  formulas -- and their floating-point operation ordering -- untouched. */
/** True when the spatial operator L = K + C may be nonsymmetric, i.e. when
 *  convection is enabled.  Every call site that would otherwise rely on the
 *  symmetry of K (adjoint products, sweep blocks, spatial Krylov method)
 *  branches on this; false keeps the ex20 code path unchanged. */
inline bool HasConvection(const ProblemConfig& config) {
  return config.convection.enabled;
}

/** The convection velocity at (x, z): the constant (vx, vy) unless a field
 *  is installed. */
inline void EvaluateVelocity(const ProblemConfig& config, double x, double z,
                             double w[2]) {
  w[0] = config.convection.vx;
  w[1] = config.convection.vy;
  if (config.convection.field) config.convection.field(config, x, z, w);
}

/** Named divergence-free velocity fields (--wind).
 *
 *  lp-recirculating: w = (2 z (1 - x^2), -2 x (1 - z^2)), the recirculating
 *    wind of Leveque & Pearson (NLAA 2022) section 5.2 (their x2 is z here);
 *    w . n = 0 on the boundary of (-1,1)^2.
 *  q2-divfree: w = (1/2 + x^2, 1/4 - 2 x z), div w = 0, both components
 *    biquadratic (exactly representable in Q2), a validation field. */
inline void WindLpRecirculating(const ProblemConfig&, double x, double z,
                                double w[2]) {
  w[0] = 2.0 * z * (1.0 - x * x);
  w[1] = -2.0 * x * (1.0 - z * z);
}
inline void WindQ2Divfree(const ProblemConfig&, double x, double z,
                          double w[2]) {
  w[0] = 0.5 + x * x;
  w[1] = 0.25 - 2.0 * x * z;
}

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
                             double time, double axis2 = 0.0) {
  // T9: in 3D (x, z, axis2) are the storage axes (0, 1, 2) = physical (x, y, z)
  if (config.dimension == 3u && config.data_overrides.source3d)
    return config.data_overrides.source3d(config, x, z, axis2, time);
  if (config.data_overrides.source)
    return config.data_overrides.source(config, x, z, time);
  if (config.data.source_form ==
      ProblemConfig::DataFieldForm::EXP_MINUS_T_COS2PI)
    return config.source * UnitCosineMode(x, z, time);
  return config.source;
}

/** Initial state y_0(x, z) of the state equation. */
inline double EvaluateInitialState(const ProblemConfig& config, double x,
                                   double z, double axis2 = 0.0) {
  if (config.dimension == 3u && config.data_overrides.initial3d)
    return config.data_overrides.initial3d(config, x, z, axis2);
  if (config.data_overrides.initial_state)
    return config.data_overrides.initial_state(config, x, z);
  if (config.data.initial_form ==
      ProblemConfig::DataFieldForm::EXP_MINUS_T_COS2PI)
    return config.initial_state * UnitCosineMode(x, z, 0.0);
  return config.initial_state;
}

inline double EvaluateTrackingTarget(const ProblemConfig& config, double x,
                                     double z, double time, double axis2 = 0.0) {
  if (config.dimension == 3u && config.data_overrides.target3d)
    return config.data_overrides.target3d(config, x, z, axis2, time);
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
inline bool OwnedDofIsControlled(const ProblemConfig& config, double x, double z,
                                 double axis2 = 0.0) {
  const double tolerance = 1.0e-12;
  // T11N: 3D Neumann control -- the nodes on the selected face plane.
  // T11D: 3D Dirichlet control -- the same plane MINUS its perimeter (the
  // perimeter is on the homogeneous faces: u = 0 there).
  if (config.dimension == 3u && (config.control.type == ProblemConfig::ControlType::NEUMANN ||
                                 config.control.type == ProblemConfig::ControlType::DIRICHLET)) {
    BoxFace3D face;
    if (!LookupBoxFace3D(config.neumann_control_face, face)) return false;
    const double c[3] = {x, z, axis2};
    const double lo[3] = {config.domain.xmin, config.domain.ymin, config.domain.zmin};
    const double hi[3] = {config.domain.xmax, config.domain.ymax, config.domain.zmax};
    if (std::fabs(c[face.axis] - (face.at_max ? hi[face.axis] : lo[face.axis])) > tolerance)
      return false;
    if (config.control.type == ProblemConfig::ControlType::NEUMANN) return true;
    for (unsigned a = 0; a < 3u; ++a)
      if (a != face.axis &&
          (c[a] <= lo[a] + tolerance || c[a] >= hi[a] - tolerance)) return false;
    return true;
  }
  if (config.control.type == ProblemConfig::ControlType::NEUMANN ||
      config.control.type == ProblemConfig::ControlType::DIRICHLET)
    return std::fabs(z - config.domain.zmin) <= tolerance;
  // T10: in 3D (x, z, axis2) are storage axes (0, 1, 2) = box (x, y, z) extents
  if (config.dimension == 3u)
    return x >= config.control.region.xmin - tolerance &&
           x <= config.control.region.xmax + tolerance &&
           z >= config.control.region.ymin - tolerance &&
           z <= config.control.region.ymax + tolerance &&
           axis2 >= config.control.region.zmin - tolerance &&
           axis2 <= config.control.region.zmax + tolerance;
  return x >= config.control.region.xmin - tolerance &&
         x <= config.control.region.xmax + tolerance &&
         z >= config.control.region.zmin - tolerance &&
         z <= config.control.region.zmax + tolerance;
}
