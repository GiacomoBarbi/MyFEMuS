// FEMuS-facing discretization layer: mesh, the state and output finite-element
// spaces, spatial operator assembly, the owned spatial DOF layout and the
// algebraic active-control set.
//
// The five spatial operators assembled here are Q2 finite-element matrices over
// the same mesh and DOF map:
//
//   M             mass
//   K             nu-weighted stiffness
//   M_track       mass over the tracking region
//   M_ctrl_box    mass over the distributed-control region
//   M_Gamma       trace mass over the control boundary Gamma_c
//   K_Gamma       tangential stiffness over Gamma_c (Dirichlet control
//                 regularization; the face-derivative part of H1(Gamma_c))
//
// Whole-domain distributed control uses M itself as the control operator: the
// mass matrix over the whole domain IS M (same contributions, same DOF map,
// same constrained-DOF treatment), so no separate copy is assembled.  The
// operator dump still writes M under the historical name "M_ctrl_whole".
//
// This is the only module that knows about femus::Mesh, MultiLevelSolution,
// SparseMatrix and element quadrature at the same time.

#pragma once

#include "MultiLevelMesh.hpp"
#include "MultiLevelSolution.hpp"
#include "Solution.hpp"
#include "SparseMatrix.hpp"

#include <petscmat.h>

#include <memory>
#include <string>
#include <vector>

#include "Config.hpp"
#include "DirichletControl.hpp"

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** ex21 inhomogeneous Dirichlet state data (lifting; see Config.hpp
 *  DataOverrides::dirichlet_state and BuildNativeSpatialOperators).
 *
 *    g              G: g on the constrained set Gamma, 0 on the free set F
 *    load           -(L_full G) with the Gamma rows zeroed (state free rows)
 *    tracking_full  M_track BEFORE the constrained-DOF elimination: the
 *                   F-Gamma coupling of the tracking functional (T3b) and the
 *                   physical objective value
 *    constrained    local mask of Gamma
 *    affine_tracking  false only in the negative-control diagnostic
 *                   (ProblemConfig::lift_affine_tracking).
 *  Owned by SpatialOperators; every member is null/empty without data. */
struct DirichletLift {
  Vec g = nullptr;
  Vec load = nullptr;
  Mat tracking_full = nullptr;
  std::vector<unsigned char> constrained;
  bool affine_tracking = true;
};

/** T6 LPS assembly statistics (global over ranks).  `patches` counts every
 *  element (patch), `active` those with Pe_k > 1 (delta_k > 0). */
struct LpsStats {
  long long patches = 0;
  long long active = 0;
  double delta_min = 0.0;    // over active patches (0 if none)
  double delta_max = 0.0;
  double pe_max = 0.0;       // over all patches
};

/** The assembled spatial operators, one FEMuS SparseMatrix each. */
struct NativeSpatialOperators {
  std::unique_ptr<femus::SparseMatrix> M;           // mass
  std::unique_ptr<femus::SparseMatrix> K;           // nu-weighted stiffness
  std::unique_ptr<femus::SparseMatrix> M_track;     // mass over the tracking region
  std::unique_ptr<femus::SparseMatrix> M_ctrl_box;  // mass over the control region
  std::unique_ptr<femus::SparseMatrix> M_Gamma;     // trace mass over Gamma_c
  std::unique_ptr<femus::SparseMatrix> K_Gamma;     // tangential stiffness over Gamma_c
  std::unique_ptr<femus::SparseMatrix> M_source;    // ex21: mass over the source regions
  // T6, only with --convection-stabilization lps (null otherwise): the LPS
  // matrix W, already ADDED to K; kept for diagnostics and dumps (constrained
  // rows/columns eliminated with diagonal 0, like M).
  std::unique_ptr<femus::SparseMatrix> W_lps;
  LpsStats lps;
};

/** T6: assemble the LPS matrix W (ProblemConfig::ConvectionConfig, Config.hpp)
 *  for the wind  wind_sign * w  (wind_sign = -1: the reversal gate).  No
 *  boundary elimination.  W is symmetric positive semidefinite by
 *  construction: per patch,
 *    W_k = delta_k sum_g omega_g (f_g - Pi f_g)(f_g - Pi f_g)^T,
 *  f_g,i = w(x_g) . grad phi_i(x_g), Pi the L2(P_k) projection onto the
 *  patch space computed with the SAME quadrature.  Statistics into `stats`. */
std::unique_ptr<femus::SparseMatrix> BuildLpsMatrix(
    const ProblemConfig& config, femus::Mesh* mesh, unsigned solution_type,
    double wind_sign, LpsStats* stats);

struct OwnedSpatialLayout {
  PetscInt begin = 0;
  PetscInt end = 0;
  PetscInt local_size = 0;
  std::vector<double> x;
  std::vector<double> z;
};

// ---------------------------------------------------------------------------
// Mesh and finite-element spaces
// ---------------------------------------------------------------------------

void BuildMesh(const ProblemConfig& config, femus::MultiLevelMesh& mesh);

/** Declare the state space and the output carrier fields.
 *
 *  `with_region_fields` additionally declares "tracking_region" and
 *  "control_region" as piecewise-constant element fields, which FEMuS emits as
 *  VTK <CellData> -- the right representation for a centroid-selected region.
 *  Off by default. */
void InitializeStateSpace(
    femus::MultiLevelSolution& solution, bool with_region_fields = false,
    ProblemConfig::StateBoundaryCondition state_bc =
        ProblemConfig::StateBoundaryCondition::NATURAL);

/** 1-based index of the region of the tracking / distributed-control UNION
 *  containing this element, or 0 if it is in none.
 *
 *  Region 1 is the primary box; 2..n are the additional regions in the order
 *  given.  An element inside several overlapping boxes reports the LOWEST
 *  index and is reported once, so a caller that assembles on a nonzero return
 *  assembles it exactly once whatever the overlap. */
unsigned TrackingRegionIndex(const ProblemConfig& config, femus::Mesh* mesh,
                             unsigned element);
unsigned ControlRegionIndex(const ProblemConfig& config, femus::Mesh* mesh,
                            unsigned element);

// ---------------------------------------------------------------------------
// Spatial finite-element operators, DOF layout and active control set
//
// Rows are the mesh's own solution DOFs (Mesh::GetSolutionDof) and the row
// distribution is Mesh::_dofOffset[solution_type]; for the one-variable Q2
// state space these are exactly the system DOFs a FEMuS LinearEquation would
// assign, so no system is needed to index the operators.
// ---------------------------------------------------------------------------

/** `solution` is the level Solution carrying FEMuS's generated constrained set
 *  for the state variable `solution_index`; it is read only when the
 *  configuration asks for homogeneous Dirichlet state constraints. */
NativeSpatialOperators BuildNativeSpatialOperators(
    const ProblemConfig& config, femus::Mesh* mesh, unsigned solution_type,
    const femus::Solution* solution, unsigned solution_index,
    DirichletLift* lift = nullptr);

/** The control operator N that the configuration selects:
 *
 *      NEUMANN                  ->  M_Gamma       (trace mass on Gamma_c)
 *      DIRICHLET                ->  M_Gamma       (its support is the controlled
 *                                                  set C; the coupling itself is
 *                                                  DirichletControlOperators)
 *      control box == domain    ->  M             (mass over the whole domain)
 *      otherwise                ->  M_ctrl_box    */
Mat SelectedControlOperator(const ProblemConfig& config,
                            const NativeSpatialOperators& spatial);

OwnedSpatialLayout BuildOwnedSpatialLayout(femus::Mesh* mesh,
                                           unsigned solution_type,
                                           Mat spatial_matrix);

std::vector<unsigned char> ActiveControlDofs(Mat control,
                                             const OwnedSpatialLayout& layout);

/** The assembled spatial operators, and the DOF layout the space-time vectors
 *  inherit from them.
 *
 *  Owns the matrices: the Mat handles below are valid exactly as long as this
 *  object is. */
struct SpatialOperators {
  SpatialOperators(const ProblemConfig& config, femus::Mesh* mesh,
                   unsigned solution_type,
                   const femus::MultiLevelSolution& solution);

  ~SpatialOperators();
  SpatialOperators(const SpatialOperators&) = delete;
  SpatialOperators& operator=(const SpatialOperators&) = delete;

  NativeSpatialOperators assembled;
  Mat mass = nullptr;        // M   (Dirichlet control: C rows/columns eliminated, diagonal 0)
  Mat stiffness = nullptr;   // K   (Dirichlet control: C rows/columns eliminated, diagonal 1)
  Mat tracking = nullptr;    // M_omega
  Mat control = nullptr;     // N: the box mass matrix, or M_Gamma
  // ex21: M_source when config.source_regions is non-empty, else null (the
  // source then uses `mass`, the ex20 load).
  Mat source = nullptr;
  OwnedSpatialLayout layout;
  // Dirichlet control only (null otherwise): M and K before the elimination
  // of the controlled boundary DOFs (owned copies; the coupling B_D needs
  // their C columns), and the owner of the boundary DOF set C, the H1(Gamma_c)
  // regularization and B_D itself.  `dirichlet` borrows the two copies.
  Mat mass_full = nullptr;
  Mat stiffness_full = nullptr;
  // ex21 inhomogeneous Dirichlet state data (null otherwise): the lift G
  // (g on the constrained DOFs, 0 elsewhere) and the constant free-row load
  // -(L_full G)_free; see BuildNativeSpatialOperators.
  DirichletLift lift;
  std::unique_ptr<DirichletControlOperators> dirichlet;

  // Wall-clock start of the operator phase, reused for the total-time field of
  // the summary record.
  double wall_start = 0.0;
};

// ---------------------------------------------------------------------------
// FEM-integrated error norms
// ---------------------------------------------------------------------------

/** An analytic scalar field sampled at a physical point, with an opaque
 *  caller-supplied context.  A plain function pointer, not std::function, so
 *  that the element quadrature loop below allocates nothing. */
typedef double (*SpatialScalarFunction)(double x, double z, void* context);

/** Rank-local contributions to the element-quadrature integrals below.  They
 *  are partial sums over the elements owned by the calling rank; the caller
 *  reduces them across the communicator and then takes square roots. */
struct SpatialFieldIntegrals {
  double error_squared = 0.0;   // integral of (field_h - exact)^2
  double exact_squared = 0.0;   // integral of exact^2
  double field_squared = 0.0;   // integral of field_h^2
};

/** Integrate a discrete field against an analytic one using the same element
 *  quadrature that assembles the mass matrix.
 *
 *  A true finite-element error: the discrete field is evaluated at each Gauss
 *  point through its shape functions, not compared nodally -- a nodal or
 *  sampled norm can superconverge and misreport the order of accuracy.
 *
 *  `global_values` is indexed by GLOBAL solution DOF and must be readable for
 *  every DOF of every element this rank owns (VecScatterCreateToAll). */
SpatialFieldIntegrals IntegrateSpatialField(
    femus::Mesh* mesh, unsigned solution_type, const double* global_values,
    SpatialScalarFunction exact, void* context);

/** The same integrals over one boundary face set instead of the volume.
 *
 *  `face_id` selects the boundary the way the operator assembly does, so
 *  config.neumann_control_face integrates over exactly the faces carrying
 *  M_Gamma.  The quadrature is the same JacobianSur rule that assembled
 *  M_Gamma, so the error is consistent with the operator the solver used. */
SpatialFieldIntegrals IntegrateBoundaryField(
    femus::Mesh* mesh, unsigned solution_type, unsigned face_id,
    const double* global_values, SpatialScalarFunction exact, void* context);
