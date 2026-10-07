// FEMuS discretization layer.

#include "Discretization.hpp"

#include "PetscMatrix.hpp"

#include <mpi.h>
#include <petscmat.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>

#include "Config.hpp"

using namespace femus;

// FEMuS's boundary-condition callback is a plain function pointer with no
// context argument, so the requested mode has to reach it through file scope.
// This is the convention the neighbouring FEMuS applications use.  It is
// written once by InitializeStateSpace and read only by the callback.
static ProblemConfig::StateBoundaryCondition g_state_bc =
    ProblemConfig::StateBoundaryCondition::NATURAL;

/** NATURAL leaves every boundary unconstrained.  HOMOGENEOUS_DIRICHLET
 *  constrains the state to zero on the whole external boundary; the carrier
 *  fields Y, P and u_q stay unconstrained because they are output only. */
static bool StateBoundaryCondition(const std::vector<double>&,
                            const char name[],
                            double& value,
                            const int,
                            const double) {
  value = 0.0;
  if (g_state_bc == ProblemConfig::StateBoundaryCondition::NATURAL)
    return false;
  return std::string(name) == "Ti";
}

void BuildMesh(const ProblemConfig& config, MultiLevelMesh& mesh) {
  config.Validate();
  if (config.dimension == 1u)
    // Q2 line elements on [xmin, xmax]: faces 1 (x = xmin) and 2 (x = xmax).
    mesh.GenerateCoarseBoxMesh(config.mesh.nx, 0, 0, config.domain.xmin,
                               config.domain.xmax, 0.0, 0.0, 0.0, 0.0, EDGE3,
                               "seventh");
  else if (config.dimension == 3u)
    // T9: Q2 HEX27 box, storage axes (x, y, z) with (nx, ny, nz) elements.
    mesh.GenerateCoarseBoxMesh(config.mesh.nx, config.mesh.ny, config.mesh.nz,
                               config.domain.xmin, config.domain.xmax,
                               config.domain.ymin, config.domain.ymax,
                               config.domain.zmin, config.domain.zmax,
                               HEX27, "seventh");
  else
    mesh.GenerateCoarseBoxMesh(config.mesh.nx, config.mesh.nz, 0,
                               config.domain.xmin, config.domain.xmax,
                               config.domain.zmin, config.domain.zmax,
                               0.0, 0.0, QUAD9, "seventh");
  mesh.EraseCoarseLevels(0);
}

void InitializeStateSpace(MultiLevelSolution& solution,
                          bool with_region_fields,
                          ProblemConfig::StateBoundaryCondition state_bc) {
  g_state_bc = state_bc;
  if (with_region_fields) {
    // Piecewise constant, one DOF per element: FEMuS emits solutions of this
    // type as VTK <CellData>, which a centroid-selected region needs -- nodal
    // interpolation would smear the region boundary.
    //
    // DECLARATION ORDER MATTERS.  VTKWriter's CellData loop names the
    // array from the solution's own index but reads the data from _Sol[i],
    // where i is the position in the requested variable list.  Declaring these
    // two first, and listing them first in the variable list passed to Write(),
    // makes the two indices agree for exactly the element fields.
    // WriteParallelVtk keeps the same order; the two must stay in step.
    solution.AddSolution("tracking_region", DISCONTINUOUS_POLYNOMIAL, ZERO);
    solution.AddSolution("control_region", DISCONTINUOUS_POLYNOMIAL, ZERO);
  }
  solution.AddSolution("Ti", LAGRANGE, SECOND, 2);
  solution.AddSolution("Y", LAGRANGE, SECOND, 2);
  solution.AddSolution("P", LAGRANGE, SECOND, 2);
  solution.AddSolution("u_q", LAGRANGE, SECOND, 2);
  solution.Initialize("All");
  solution.AttachSetBoundaryConditionFunction(StateBoundaryCondition);
  solution.GenerateBdc("Ti", "Time_dependent");
}

namespace {

struct ElementData {
  short unsigned geometry = 0;
  std::vector<unsigned> dofs;
  std::vector<std::vector<double> > coordinates;
};

}  // namespace

static ElementData GatherElementData(Mesh* mesh, unsigned solution_type,
                              unsigned element) {
  ElementData data;
  const unsigned dimension = mesh->GetDimension();
  data.geometry = mesh->GetElementType(element);
  const unsigned element_dofs =
      mesh->GetElementDofNumber(element, solution_type);
  data.dofs.resize(element_dofs);
  data.coordinates.assign(dimension,
                          std::vector<double>(element_dofs, 0.0));

  for (unsigned a = 0; a < element_dofs; ++a) {
    data.dofs[a] = mesh->GetSolutionDof(a, element, solution_type);
    const unsigned coordinate_dof = mesh->GetSolutionDof(a, element, 2);
    for (unsigned axis = 0; axis < dimension; ++axis)
      data.coordinates[axis][a] =
          (*mesh->_topology->_Sol[axis])(coordinate_dof);
  }
  return data;
}

/** The element centroid, as the arithmetic mean of its coordinate DOFs. */
static void ElementCentroid(Mesh* mesh, unsigned element, double& x_centroid,
                     double& z_centroid) {
  const unsigned coordinate_dofs = mesh->GetElementDofNumber(element, 2);
  x_centroid = 0.0;
  z_centroid = 0.0;
  for (unsigned a = 0; a < coordinate_dofs; ++a) {
    const unsigned coordinate_dof = mesh->GetSolutionDof(a, element, 2);
    x_centroid += (*mesh->_topology->_Sol[0])(coordinate_dof);
    z_centroid += (*mesh->_topology->_Sol[1])(coordinate_dof);
  }
  x_centroid /= coordinate_dofs;
  z_centroid /= coordinate_dofs;
}

/** Strict centroid-in-box test: a centroid exactly on a face is outside.
 *  `dimension == 1` tests x only (every z is 0 on a line mesh). */
static bool CentroidIsInBox(double x_centroid, double z_centroid, double xmin,
                     double xmax, double zmin, double zmax,
                     unsigned dimension = 2u) {
  if (dimension == 1u) return x_centroid > xmin && x_centroid < xmax;
  return x_centroid > xmin && x_centroid < xmax &&
         z_centroid > zmin && z_centroid < zmax;
}

/** Index of the region of the UNION that contains this element, 1-based, or 0
 *  if the element is in none of them.
 *
 *  Region 1 is always the primary box; regions 2..n are the additional ones in
 *  the order given.  An element lying in several overlapping boxes reports the
 *  LOWEST index -- and, crucially, is reported once, so a caller that assembles
 *  on a nonzero return assembles it exactly once regardless of overlap.
 *
 *  The centroid is computed once per element and reused for every box. */
unsigned TrackingRegionIndex(const ProblemConfig& config, Mesh* mesh,
                             unsigned element) {
  // T9: 3D phase 1 tracks the whole domain (Validate refuses boxes).
  if (config.dimension == 3u) return 1u;
  double x_centroid = 0.0;
  double z_centroid = 0.0;
  ElementCentroid(mesh, element, x_centroid, z_centroid);
  if (CentroidIsInBox(x_centroid, z_centroid, config.tracking_box.xmin,
                      config.tracking_box.xmax, config.tracking_box.zmin,
                      config.tracking_box.zmax, config.dimension))
    return 1u;
  for (std::size_t k = 0; k < config.additional_tracking_regions.size(); ++k) {
    const ProblemConfig::Box& box = config.additional_tracking_regions[k];
    if (CentroidIsInBox(x_centroid, z_centroid, box.xmin, box.xmax, box.zmin,
                        box.zmax, config.dimension))
      return static_cast<unsigned>(k + 2u);
  }
  return 0u;
}

unsigned ControlRegionIndex(const ProblemConfig& config, Mesh* mesh,
                            unsigned element) {
  // T9: 3D phase 1 controls the whole domain (Validate refuses boxes).
  if (config.dimension == 3u) return 1u;
  double x_centroid = 0.0;
  double z_centroid = 0.0;
  ElementCentroid(mesh, element, x_centroid, z_centroid);
  if (CentroidIsInBox(x_centroid, z_centroid, config.control.region.xmin,
                      config.control.region.xmax, config.control.region.zmin,
                      config.control.region.zmax, config.dimension))
    return 1u;
  for (std::size_t k = 0; k < config.additional_control_regions.size(); ++k) {
    const ProblemConfig::Box& box = config.additional_control_regions[k];
    if (CentroidIsInBox(x_centroid, z_centroid, box.xmin, box.xmax, box.zmin,
                        box.zmax, config.dimension))
      return static_cast<unsigned>(k + 2u);
  }
  return 0u;
}


// ---------------------------------------------------------------------------
// Homogeneous Dirichlet state constraints
//
// The spatial operators live outside the FEMuS _KK solve path, so GenerateBdc
// alone constrains nothing: the constraints have to be applied to the operators
// this application assembles.  The treatment differs per operator, because
// applying one rule everywhere would be wrong:
//
//   K        rows and columns zeroed, DIAGONAL 1
//   M        rows and columns zeroed, DIAGONAL 0
//   all mass-like operators (M_track, M_ctrl_*, M_Gamma)
//            rows and columns zeroed, DIAGONAL 0
//
// Consequences, with G the constrained set:
//
//   E   = blockdiag(M/dt + K) - blocksubdiag(M/dt)  has  E_GG = 0/dt + 1 = 1
//        and a zero subdiagonal on G, so row G is the identity equation
//        y_G^k = 0, decoupled in time and perfectly conditioned.
//   E^T is still the exact transpose: both eliminations are symmetric, so M
//        and K stay symmetric and the AP shell transposes the same matrices.
//   rhs_y and rhs_p are assembled as MatMult(M, .) and MatMult(M_track, .),
//        so their constrained rows vanish automatically -- no change to the
//        solver or the space-time layer is needed to make y_G = p_G = 0.
//   M_track with a ZERO diagonal keeps constrained DOFs out of the tracking
//        functional and out of the adjoint right-hand side.
//   N with a ZERO diagonal keeps them out of the control: ActiveControlDofs
//        selects on N_ii > tol * max_j N_jj, so N_ii = 0 excludes them by the
//        operator's own support rather than by a tolerance hack.
//
// A unit diagonal on the mass-like operators would do the opposite: it would
// put constrained DOFs into the active control set and give them tracking
// weight.  That is why the diagonal is 0 for every one of them.
// ---------------------------------------------------------------------------

/** The state DOFs FEMuS marked as constrained, as PETSc matrix rows.
 *
 *  Read from the generated BC data rather than from geometry.  FEMuS stores
 *  2 = free (natural), 1 = AMR artificial Dirichlet, 0 = Dirichlet, so the
 *  constrained test is the library's own `< 1.5`.  The rows of the assembled
 *  operators are the solution DOFs themselves, so the owned solution DOF
 *  range is the owned row range. */
static std::vector<PetscInt> ConstrainedStateRows(Mesh* mesh,
                                           const Solution* solution,
                                           unsigned solution_index,
                                           unsigned solution_type) {
  std::vector<PetscInt> rows;
  const unsigned rank = mesh->processor_id();
  const unsigned begin = mesh->_dofOffset[solution_type][rank];
  const unsigned end = mesh->_dofOffset[solution_type][rank + 1u];
  const PetscInt row_start = static_cast<PetscInt>(begin);
  const NumericVector& bdc = *solution->_Bdc[solution_index];
  for (unsigned solution_dof = begin; solution_dof < end; ++solution_dof)
    if (bdc(solution_dof) < 1.5)
      rows.push_back(row_start + static_cast<PetscInt>(solution_dof - begin));
  return rows;
}

/** Symmetric row/column elimination with the given diagonal value. */
static void EliminateConstrainedDofs(SparseMatrix* matrix,
                              const std::vector<PetscInt>& rows,
                              PetscScalar diagonal) {
  Mat m = static_cast<PetscMatrix*>(matrix)->mat();
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatSetOption(m, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE));
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatZeroRowsColumns(m, static_cast<PetscInt>(rows.size()),
                                    rows.empty() ? nullptr : rows.data(),
                                    diagonal, nullptr, nullptr));
}

std::unique_ptr<SparseMatrix> BuildLpsMatrix(
    const ProblemConfig& config, Mesh* mesh, unsigned solution_type,
    double wind_sign, LpsStats* stats) {
  typedef ProblemConfig::ConvectionConfig CC;
  const unsigned rank = mesh->processor_id();
  const int global_rows = mesh->GetTotalNumberOfDofs(solution_type);
  const int local_rows = mesh->_dofOffset[solution_type][rank + 1u] -
                         mesh->_dofOffset[solution_type][rank];
  std::unique_ptr<SparseMatrix> W = SparseMatrix::build();
  W->init(global_rows, global_rows, local_rows, local_rows, 64, 64);
  const unsigned dimension = mesh->GetDimension();
  const unsigned nq =
      config.convection.lps_projection == CC::LpsProjection::P1 ? 1u + dimension : 1u;
  const double nu = config.diffusivity;
  long long patches = 0, active = 0;
  double delta_min = std::numeric_limits<double>::infinity(), delta_max = 0.0,
         pe_max = 0.0;
  std::vector<double> phi, gradient, f, q(nq), A, C, G, local;
  double weight = 0.0;
  for (unsigned element = mesh->_elementOffset[rank];
       element < mesh->_elementOffset[rank + 1u]; ++element) {
    const ElementData data = GatherElementData(mesh, solution_type, element);
    const unsigned n = data.dofs.size();
    ++patches;
    // Patch centroid, extents and the wind at the centroid (LP: w_k).
    double xc = 0.0, zc = 0.0;
    ElementCentroid(mesh, element, xc, zc);
    double extent[2] = {0.0, 0.0};
    for (unsigned axis = 0; axis < dimension && axis < 2u; ++axis) {
      const std::vector<double>& c = data.coordinates[axis];
      extent[axis] = *std::max_element(c.begin(), c.end()) -
                     *std::min_element(c.begin(), c.end());
    }
    double wk[2] = {config.convection.vx, config.convection.vy};
    if (config.convection.field) config.convection.field(config, xc, zc, wk);
    wk[0] *= wind_sign;
    wk[1] *= wind_sign;
    if (dimension == 1u) wk[1] = 0.0;
    const double speed = std::hypot(wk[0], wk[1]);
    // h_k: length of the chord of the (rectangular) patch through its
    // centroid in the direction of w_k (LP: "a measure of the patch length in
    // the direction of the wind"); for a square of side H this is
    // H / max(|cos|, |sin|), as in t3/lp52_patch_peclet.py.
    double hk = std::numeric_limits<double>::infinity();
    for (unsigned axis = 0; axis < dimension && axis < 2u; ++axis)
      if (wk[axis] != 0.0)
        hk = std::min(hk, extent[axis] * speed / std::fabs(wk[axis]));
    const double pe = speed > 0.0 ? speed * hk / (2.0 * nu) : 0.0;
    pe_max = std::max(pe_max, pe);
    if (!(pe > 1.0)) continue;                     // delta_k = 0
    const double delta = hk / (2.0 * speed) * (1.0 - 1.0 / pe);
    ++active;
    delta_min = std::min(delta_min, delta);
    delta_max = std::max(delta_max, delta);
    A.assign(n * n, 0.0);
    C.assign(nq * n, 0.0);
    G.assign(nq * nq, 0.0);
    f.assign(n, 0.0);
    const elem_type* fe = mesh->_finiteElement[data.geometry][solution_type];
    for (unsigned gauss = 0; gauss < fe->GetGaussPointNumber(); ++gauss) {
      fe->Jacobian(data.coordinates, gauss, weight, phi, gradient);
      double x = 0.0, z = 0.0;
      for (unsigned a = 0; a < n; ++a) {
        x += phi[a] * data.coordinates[0][a];
        if (dimension > 1u) z += phi[a] * data.coordinates[1][a];
      }
      // The integrand of (7) uses the wind itself (variable), the parameter
      // delta_k its centroid value.
      double w[2] = {config.convection.vx, config.convection.vy};
      if (config.convection.field) config.convection.field(config, x, z, w);
      w[0] *= wind_sign;
      w[1] *= wind_sign;
      for (unsigned a = 0; a < n; ++a) {
        f[a] = 0.0;
        for (unsigned axis = 0; axis < dimension && axis < 2u; ++axis)
          f[a] += w[axis] * gradient[a * dimension + axis];
      }
      q[0] = 1.0;
      if (nq > 1u) q[1] = (x - xc) / extent[0];
      if (nq > 2u) q[2] = (z - zc) / extent[1];
      for (unsigned a = 0; a < n; ++a)
        for (unsigned b = 0; b < n; ++b) A[a * n + b] += weight * f[a] * f[b];
      for (unsigned m = 0; m < nq; ++m) {
        for (unsigned a = 0; a < n; ++a) C[m * n + a] += weight * q[m] * f[a];
        for (unsigned l = 0; l < nq; ++l) G[m * nq + l] += weight * q[m] * q[l];
      }
    }
    // D = G^{-1} C by Gauss-Jordan on the (at most 3x3, SPD) patch Gram matrix.
    std::vector<double> D = C;
    for (unsigned p = 0; p < nq; ++p) {
      const double pivot = G[p * nq + p];
      for (unsigned l = 0; l < nq; ++l) G[p * nq + l] /= pivot;
      for (unsigned a = 0; a < n; ++a) D[p * n + a] /= pivot;
      for (unsigned r = 0; r < nq; ++r) {
        if (r == p) continue;
        const double factor = G[r * nq + p];
        for (unsigned l = 0; l < nq; ++l) G[r * nq + l] -= factor * G[p * nq + l];
        for (unsigned a = 0; a < n; ++a) D[r * n + a] -= factor * D[p * n + a];
      }
    }
    // W_k = delta (A - C^T G^{-1} C), symmetrised entrywise (exact in exact
    // arithmetic; removes the round-off asymmetry of the C^T D product).
    local.assign(n * n, 0.0);
    for (unsigned a = 0; a < n; ++a)
      for (unsigned b = 0; b < n; ++b) {
        double ab = A[a * n + b], ba = A[b * n + a];
        for (unsigned m = 0; m < nq; ++m) {
          ab -= C[m * n + a] * D[m * n + b];
          ba -= C[m * n + b] * D[m * n + a];
        }
        local[a * n + b] = delta * 0.5 * (ab + ba);
      }
    std::vector<int> system_dofs(n);
    for (unsigned a = 0; a < n; ++a)
      system_dofs[a] = static_cast<int>(mesh->GetSolutionDof(a, element, solution_type));
    W->add_matrix_blocked(local, system_dofs, system_dofs);
  }
  W->close();
  if (stats) {
    long long counts[2] = {patches, active}, gcounts[2] = {0, 0};
    MPI_Allreduce(counts, gcounts, 2, MPI_LONG_LONG, MPI_SUM, PETSC_COMM_WORLD);
    double mins[1] = {delta_min}, gmins[1] = {0.0};
    double maxs[2] = {delta_max, pe_max}, gmaxs[2] = {0.0, 0.0};
    MPI_Allreduce(mins, gmins, 1, MPI_DOUBLE, MPI_MIN, PETSC_COMM_WORLD);
    MPI_Allreduce(maxs, gmaxs, 2, MPI_DOUBLE, MPI_MAX, PETSC_COMM_WORLD);
    stats->patches = gcounts[0];
    stats->active = gcounts[1];
    stats->delta_min = gcounts[1] > 0 ? gmins[0] : 0.0;
    stats->delta_max = gmaxs[0];
    stats->pe_max = gmaxs[1];
  }
  return W;
}

NativeSpatialOperators BuildNativeSpatialOperators(
    const ProblemConfig& config, Mesh* mesh, unsigned solution_type,
    const Solution* solution, unsigned solution_index, DirichletLift* lift) {
  const unsigned rank = mesh->processor_id();
  const int global_rows = mesh->GetTotalNumberOfDofs(solution_type);
  const int local_rows = mesh->_dofOffset[solution_type][rank + 1u] -
                         mesh->_dofOffset[solution_type][rank];
  NativeSpatialOperators operators;
  for (std::unique_ptr<SparseMatrix>* matrix :
       {&operators.M, &operators.K, &operators.M_track, &operators.M_ctrl_box,
        &operators.M_Gamma, &operators.K_Gamma, &operators.M_source}) {
    *matrix = SparseMatrix::build();
    // T9: a Q2 HEX27 vertex row couples 5^3 = 125 DOFs; 1D/2D keep 64.
    const int row_nonzeros = config.dimension == 3u ? 125 : 64;
    (*matrix)->init(global_rows, global_rows, local_rows, local_rows,
                    row_nonzeros, row_nonzeros);
  }

  const unsigned dimension = mesh->GetDimension();
  // Convection velocity per mesh axis (axis 1 is the coordinate called z).
  const bool convection = HasConvection(config);
  // T9: the third component (storage axis 2) is read only in 3D; the 2D
  // loop below never touches it.
  const double velocity[3] = {config.convection.vx, config.convection.vy,
                              config.convection.vz};
  std::vector<double> phi;
  std::vector<double> gradient;
  double weight = 0.0;
  for (unsigned element = mesh->_elementOffset[rank];
       element < mesh->_elementOffset[rank + 1u]; ++element) {
    const ElementData data = GatherElementData(mesh, solution_type, element);
    const unsigned element_dofs = data.dofs.size();
    std::vector<int> system_dofs(element_dofs);
    for (unsigned a = 0; a < element_dofs; ++a)
      system_dofs[a] = static_cast<int>(
          mesh->GetSolutionDof(a, element, solution_type));
    std::vector<double> local_mass(element_dofs * element_dofs, 0.0);
    std::vector<double> local_stiffness(element_dofs * element_dofs, 0.0);
    for (unsigned gauss = 0;
         gauss < mesh->_finiteElement[data.geometry][solution_type]
                     ->GetGaussPointNumber(); ++gauss) {
      mesh->_finiteElement[data.geometry][solution_type]->Jacobian(
          data.coordinates, gauss, weight, phi, gradient);
      for (unsigned a = 0; a < element_dofs; ++a)
        for (unsigned b = 0; b < element_dofs; ++b) {
          double gradient_dot = 0.0;
          for (unsigned axis = 0; axis < dimension; ++axis)
            gradient_dot += gradient[a * dimension + axis] *
                            gradient[b * dimension + axis];
          local_mass[a * element_dofs + b] += phi[a] * phi[b] * weight;
          local_stiffness[a * element_dofs + b] +=
              config.diffusivity * gradient_dot * weight;
        }
      // Convection, strong form:  L_ab += int (v . grad phi_b) phi_a, row a
      // (test), column b (trial) -- add_matrix_blocked is row-oriented.  The
      // orientation is checked numerically by --verify-convection.
      // A variable field w(x) is evaluated at the physical quadrature point
      // (isoparametric map); the constant pair is used unchanged otherwise.
      double point_velocity[3] = {velocity[0], velocity[1], velocity[2]};
      if (convection && config.convection.field) {
        double x = 0.0, z = 0.0;
        for (unsigned a = 0; a < element_dofs; ++a) {
          x += phi[a] * data.coordinates[0][a];
          if (data.coordinates.size() > 1u) z += phi[a] * data.coordinates[1][a];
        }
        config.convection.field(config, x, z, point_velocity);
      }
      if (convection)
        for (unsigned a = 0; a < element_dofs; ++a)
          for (unsigned b = 0; b < element_dofs; ++b) {
            double advective = 0.0;
            for (unsigned axis = 0; axis < dimension && axis < 3u; ++axis)
              advective += point_velocity[axis] * gradient[b * dimension + axis];
            local_stiffness[a * element_dofs + b] += advective * phi[a] * weight;
          }
    }
    operators.M->add_matrix_blocked(local_mass, system_dofs, system_dofs);
    operators.K->add_matrix_blocked(local_stiffness, system_dofs, system_dofs);
    // Union of the primary tracking box and any additional tracking regions.
    // A nonzero index means "in the union"; the element is inserted ONCE,
    // however many boxes happen to contain it.
    if (TrackingRegionIndex(config, mesh, element) != 0u)
      operators.M_track->add_matrix_blocked(local_mass, system_dofs,
                                            system_dofs);
    // Same for the distributed-control region.
    if (ControlRegionIndex(config, mesh, element) != 0u)
      operators.M_ctrl_box->add_matrix_blocked(local_mass, system_dofs,
                                               system_dofs);
    // ex21: the source-region union (empty list: nothing is assembled).
    if (!config.source_regions.empty()) {
      double xc = 0.0, zc = 0.0;
      ElementCentroid(mesh, element, xc, zc);
      for (const ProblemConfig::Box& box : config.source_regions)
        if (CentroidIsInBox(xc, zc, box.xmin, box.xmax, box.zmin, box.zmax,
                            config.dimension)) {
          operators.M_source->add_matrix_blocked(local_mass, system_dofs,
                                                 system_dofs);
          break;
        }
    }

    // A 1D mesh has point faces and no control face (Validate allows only
    // DISTRIBUTED control there), so M_Gamma and K_Gamma stay empty.
    // T9: skipped in 3D too -- no boundary control in phase 1, and the
  // tangential stiffness below assumes one-dimensional faces.
  for (unsigned face = 0;
         config.dimension == 2u && face < mesh->GetElementFaceNumber(element);
         ++face) {
      const unsigned face_id = static_cast<unsigned>(
          -(mesh->el->GetFaceElementIndex(element, face) + 1));
      if (face_id != config.neumann_control_face) continue;
      const unsigned face_dofs =
          mesh->GetElementFaceDofNumber(element, face, solution_type);
      const unsigned face_geometry = mesh->GetElementFaceType(element, face);
      std::vector<int> face_system_dofs(face_dofs);
      std::vector<std::vector<double> > face_coordinates(
          dimension, std::vector<double>(face_dofs, 0.0));
      for (unsigned a = 0; a < face_dofs; ++a) {
        const unsigned local_node =
            mesh->GetLocalFaceVertexIndex(element, face, a);
        face_system_dofs[a] = static_cast<int>(
            mesh->GetSolutionDof(local_node, element, solution_type));
        const unsigned coordinate_dof =
            mesh->GetSolutionDof(local_node, element, 2);
        for (unsigned axis = 0; axis < dimension; ++axis)
          face_coordinates[axis][a] =
              (*mesh->_topology->_Sol[axis])(coordinate_dof);
      }
      std::vector<double> local_boundary(face_dofs * face_dofs, 0.0);
      std::vector<double> local_tangential(face_dofs * face_dofs, 0.0);
      std::vector<double> normal;
      const elem_type* face_element =
          mesh->_finiteElement[face_geometry][solution_type];
      for (unsigned gauss = 0; gauss < face_element->GetGaussPointNumber();
           ++gauss) {
        face_element->JacobianSur(face_coordinates, gauss, weight, phi,
                                  gradient, normal);
        // Tangential derivative along the (one-dimensional) face: with the
        // reference derivatives dphi/dxi and the arc-length Jacobian
        // J = |dx/dxi|, d_s phi = (dphi/dxi) / J and ds = J dxi, so the
        // face stiffness is sum_g (dphi_a/dxi)(dphi_b/dxi) w_g / J.
        // (JacobianSur provides no surface gradient.)
        const double* dphi_dxi = face_element->GetDPhiDXi(gauss);
        double tangent[2] = {0.0, 0.0};
        for (unsigned a = 0; a < face_dofs; ++a)
          for (unsigned axis = 0; axis < dimension && axis < 2; ++axis)
            tangent[axis] += dphi_dxi[a] * face_coordinates[axis][a];
        const double arc_jacobian =
            std::sqrt(tangent[0] * tangent[0] + tangent[1] * tangent[1]);
        const double gauss_weight =
            face_element->GetGaussWeight(gauss) / arc_jacobian;
        for (unsigned a = 0; a < face_dofs; ++a)
          for (unsigned b = 0; b < face_dofs; ++b) {
            local_boundary[a * face_dofs + b] +=
                phi[a] * phi[b] * weight;
            local_tangential[a * face_dofs + b] +=
                dphi_dxi[a] * dphi_dxi[b] * gauss_weight;
          }
      }
      operators.M_Gamma->add_matrix_blocked(local_boundary, face_system_dofs,
                                            face_system_dofs);
      operators.K_Gamma->add_matrix_blocked(local_tangential, face_system_dofs,
                                            face_system_dofs);
    }
  }
  operators.M->close();
  operators.K->close();
  operators.M_track->close();
  operators.M_ctrl_box->close();
  operators.M_Gamma->close();
  operators.K_Gamma->close();
  operators.M_source->close();

  // T6: optional LPS, assembled separately, diagnosed, then added to L
  // BEFORE the Dirichlet lifting and elimination (so the lifting load and the
  // eliminated operator both carry it).  Off: nothing here runs.
  if (convection &&
      config.convection.stabilization ==
          ProblemConfig::ConvectionConfig::Stabilization::LPS) {
    operators.W_lps = BuildLpsMatrix(config, mesh, solution_type, 1.0, &operators.lps);
    Mat w = static_cast<PetscMatrix*>(operators.W_lps.get())->mat();
    Mat k = static_cast<PetscMatrix*>(operators.K.get())->mat();
    PetscReal w_norm = 0.0, asym = 0.0, l_sym = 0.0, l_skew = 0.0;
    Mat wt = nullptr, lt = nullptr, sym = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD, MatNorm(w, NORM_FROBENIUS, &w_norm));
    PetscCallAbort(PETSC_COMM_WORLD, MatTranspose(w, MAT_INITIAL_MATRIX, &wt));
    PetscCallAbort(PETSC_COMM_WORLD, MatAXPY(wt, -1.0, w, DIFFERENT_NONZERO_PATTERN));
    PetscCallAbort(PETSC_COMM_WORLD, MatNorm(wt, NORM_FROBENIUS, &asym));
    // Unstabilised L = nu K + N: its symmetric part (diffusion + the boundary
    // part of N) and skew part (convection) as magnitude references.
    PetscCallAbort(PETSC_COMM_WORLD, MatTranspose(k, MAT_INITIAL_MATRIX, &lt));
    PetscCallAbort(PETSC_COMM_WORLD, MatDuplicate(k, MAT_COPY_VALUES, &sym));
    PetscCallAbort(PETSC_COMM_WORLD, MatAXPY(sym, 1.0, lt, DIFFERENT_NONZERO_PATTERN));
    PetscCallAbort(PETSC_COMM_WORLD, MatNorm(sym, NORM_FROBENIUS, &l_sym));
    PetscCallAbort(PETSC_COMM_WORLD, MatAXPY(lt, -1.0, k, DIFFERENT_NONZERO_PATTERN));
    PetscCallAbort(PETSC_COMM_WORLD, MatNorm(lt, NORM_FROBENIUS, &l_skew));
    l_sym *= 0.5;
    l_skew *= 0.5;
    for (Mat* m : {&wt, &lt, &sym}) PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(m));
    if (operators.lps.active > 0)
      PetscCallAbort(PETSC_COMM_WORLD, MatAXPY(k, 1.0, w, SUBSET_NONZERO_PATTERN));
    if (rank == 0)
      std::printf("EX21_LPS projection=%s patches=%lld active=%lld"
                  " delta_min=%.6e delta_max=%.6e pe_max=%.6e ||W||_F=%.6e"
                  " ||W-W^T||_F/||W||_F=%.3e ||sym(L0)||_F=%.6e ||skew(L0)||_F=%.6e"
                  " W/sym=%.3e W/skew=%.3e\n",
                  config.convection.lps_projection ==
                          ProblemConfig::ConvectionConfig::LpsProjection::P1
                      ? "p1" : "p0",
                  operators.lps.patches, operators.lps.active,
                  operators.lps.delta_min, operators.lps.delta_max,
                  operators.lps.pe_max, w_norm, w_norm > 0 ? asym / w_norm : asym,
                  l_sym, l_skew, l_sym > 0 ? w_norm / l_sym : 0.0,
                  l_skew > 0 ? w_norm / l_skew : 0.0);
  }

  if (config.state_bc ==
      ProblemConfig::StateBoundaryCondition::HOMOGENEOUS_DIRICHLET) {
    if (solution == nullptr)
      throw std::runtime_error(
          "homogeneous Dirichlet state BC needs the Solution to read the "
          "constrained set from");
    const std::vector<PetscInt> rows = ConstrainedStateRows(
        mesh, solution, solution_index, solution_type);
    // ex21 inhomogeneous Dirichlet data by lifting, BEFORE K is eliminated:
    //   G = g on the constrained rows, 0 elsewhere;  load = -(L_full G) with
    //   the constrained rows zeroed.  The free rows of  L_full (y_0 + G) are
    //   L_FF y_0,F + L_FG g, so  -L_FG g  is the constant load the homogeneous
    //   unknown y_0 sees (the time-derivative term M_FG dg/dt vanishes for
    //   time-independent g; the source and tracking conventions are those of
    //   the homogeneous problem, see Config.hpp).  Built only when asked for.
    if (config.data_overrides.dirichlet_state && lift) {
      Mat k = static_cast<PetscMatrix*>(operators.K.get())->mat();
      const OwnedSpatialLayout layout =
          BuildOwnedSpatialLayout(mesh, solution_type, k);
      PetscCallAbort(PETSC_COMM_WORLD, MatCreateVecs(k, &lift->g, &lift->load));
      PetscCallAbort(PETSC_COMM_WORLD, VecSet(lift->g, 0.0));
      lift->constrained.assign(static_cast<std::size_t>(layout.local_size), 0);
      for (PetscInt row : rows) {
        const PetscInt i = row - layout.begin;
        lift->constrained[static_cast<std::size_t>(i)] = 1;
        PetscCallAbort(PETSC_COMM_WORLD,
                       VecSetValue(lift->g, row,
                                   config.data_overrides.dirichlet_state(
                                       config, layout.x[i], layout.z[i]),
                                   INSERT_VALUES));
      }
      PetscCallAbort(PETSC_COMM_WORLD, VecAssemblyBegin(lift->g));
      PetscCallAbort(PETSC_COMM_WORLD, VecAssemblyEnd(lift->g));
      PetscCallAbort(PETSC_COMM_WORLD, MatMult(k, lift->g, lift->load));
      PetscCallAbort(PETSC_COMM_WORLD, VecScale(lift->load, -1.0));
      for (PetscInt row : rows)
        PetscCallAbort(PETSC_COMM_WORLD,
                       VecSetValue(lift->load, row, 0.0, INSERT_VALUES));
      PetscCallAbort(PETSC_COMM_WORLD, VecAssemblyBegin(lift->load));
      PetscCallAbort(PETSC_COMM_WORLD, VecAssemblyEnd(lift->load));
      // T3b: the un-eliminated tracking matrix, for the F-Gamma coupling
      // M_F,Gamma (g - y_d)_Gamma of the tracking derivative and for the
      // physical objective value (constrained values are data, not unknowns;
      // the KKT Hessian stays the eliminated M_FF).
      PetscCallAbort(PETSC_COMM_WORLD,
                     MatDuplicate(static_cast<PetscMatrix*>(operators.M_track.get())->mat(),
                                  MAT_COPY_VALUES, &lift->tracking_full));
      lift->affine_tracking = config.lift_affine_tracking;
    }
    // Diagonal 1 on K alone: it is what makes E_GG = 1.  Every other operator
    // takes diagonal 0 so that constrained DOFs carry neither tracking weight
    // nor control support.  See the note above BuildNativeSpatialOperators.
    // Exactly one elimination per physical matrix: M serves both as the mass
    // matrix and as the whole-domain control operator and is eliminated once.
    EliminateConstrainedDofs(operators.M.get(), rows, 0.0);
    EliminateConstrainedDofs(operators.K.get(), rows, 1.0);
    EliminateConstrainedDofs(operators.M_track.get(), rows, 0.0);
    EliminateConstrainedDofs(operators.M_ctrl_box.get(), rows, 0.0);
    EliminateConstrainedDofs(operators.M_Gamma.get(), rows, 0.0);
    EliminateConstrainedDofs(operators.K_Gamma.get(), rows, 0.0);
    EliminateConstrainedDofs(operators.M_source.get(), rows, 0.0);
    if (operators.W_lps) EliminateConstrainedDofs(operators.W_lps.get(), rows, 0.0);
  }

  return operators;
}


Mat SelectedControlOperator(const ProblemConfig& config,
                            const NativeSpatialOperators& spatial) {
  if (config.control.type == ProblemConfig::ControlType::NEUMANN ||
      config.control.type == ProblemConfig::ControlType::DIRICHLET)
    return static_cast<PetscMatrix*>(spatial.M_Gamma.get())->mat();
  if (ControlRegionIsWholeDomain(config))
    return static_cast<PetscMatrix*>(spatial.M.get())->mat();
  return static_cast<PetscMatrix*>(spatial.M_ctrl_box.get())->mat();
}

OwnedSpatialLayout BuildOwnedSpatialLayout(Mesh* mesh,
                                           unsigned solution_type,
                                           Mat spatial_matrix) {
  OwnedSpatialLayout layout;
  // FEMuS's Mesh owns the row distribution; the assembled operator adopted
  // it, so PETSc is the cross-check rather than the source.
  const unsigned rank = mesh->processor_id();
  layout.begin = static_cast<PetscInt>(mesh->_dofOffset[solution_type][rank]);
  layout.end =
      static_cast<PetscInt>(mesh->_dofOffset[solution_type][rank + 1u]);
  layout.local_size = layout.end - layout.begin;
  PetscInt petsc_begin = 0, petsc_end = 0;
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatGetOwnershipRange(spatial_matrix, &petsc_begin, &petsc_end));
  if (layout.begin != petsc_begin || layout.end != petsc_end)
    throw std::runtime_error("FEMuS/PETSc spatial ownership mismatch");
  layout.x.assign(layout.local_size, 0.0);
  layout.z.assign(layout.local_size, 0.0);
  const bool three_d = mesh->GetDimension() == 3u;   // T9
  layout.three_d = three_d;
  if (three_d) layout.axis2.assign(layout.local_size, 0.0);
  std::vector<unsigned char> seen(layout.local_size, 0u);
  for (unsigned element = mesh->_elementOffset[rank];
       element < mesh->_elementOffset[rank + 1u]; ++element) {
    const unsigned element_dofs =
        mesh->GetElementDofNumber(element, solution_type);
    for (unsigned a = 0; a < element_dofs; ++a) {
      const PetscInt system_dof = static_cast<PetscInt>(
          mesh->GetSolutionDof(a, element, solution_type));
      if (system_dof < layout.begin || system_dof >= layout.end) continue;
      const PetscInt local = system_dof - layout.begin;
      const unsigned coordinate_dof = mesh->GetSolutionDof(a, element, 2);
      layout.x[local] = (*mesh->_topology->_Sol[0])(coordinate_dof);
      layout.z[local] = (*mesh->_topology->_Sol[1])(coordinate_dof);
      if (three_d) layout.axis2[local] = (*mesh->_topology->_Sol[2])(coordinate_dof);
      seen[local] = 1u;
    }
  }
  for (PetscInt i = 0; i < layout.local_size; ++i)
    if (!seen[i])
      throw std::runtime_error("owned Q2 DOF has no locally visible coordinate");
  return layout;
}

/** The discrete active control set, read off the assembled control operator N.
 *
 *  N is a mass matrix over the controlled region C (the control box for
 *  distributed control, Gamma_c for Neumann control), so its diagonal is
 *
 *      N_ii = \int_C phi_i^2 >= 0,
 *
 *  strictly positive exactly when basis function i has support in C.  N_ii > 0
 *  is therefore an exact characterisation of the DOFs the control acts on.
 *
 *  The threshold is relative to max_j N_jj -- a global reduction, so identical
 *  on every rank -- never absolute: N_jj scales like h^2 for a volume control
 *  and like h for a boundary control, so a fixed cutoff would be mesh
 *  dependent.  Q2 mass-matrix diagonals vary by about one order of magnitude
 *  between node types, so 1e-12 * max_j N_jj separates the two populations with
 *  a wide margin.  If N == 0 the scale is 0 and the active set is empty. */
std::vector<unsigned char> ActiveControlDofs(Mat control,
                                             const OwnedSpatialLayout& layout) {
  Vec diagonal = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, MatCreateVecs(control, &diagonal, nullptr));
  PetscCallAbort(PETSC_COMM_WORLD, MatGetDiagonal(control, diagonal));
  double scale = 0.0;
  PetscCallAbort(PETSC_COMM_WORLD, VecNorm(diagonal, NORM_INFINITY, &scale));
  const double tolerance = 1.0e-12 * scale;
  const PetscScalar* values = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(diagonal, &values));
  std::vector<unsigned char> active(layout.local_size, 0u);
  for (PetscInt i = 0; i < layout.local_size; ++i)
    active[i] = (PetscRealPart(values[i]) > tolerance) ? 1u : 0u;
  PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(diagonal, &values));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&diagonal));
  return active;
}


// ---------------------------------------------------------------------------
// FEM-integrated error norms
//
// A generic numerical service: it knows nothing about any particular
// benchmark, and the analytic field arrives as a function pointer.
// ---------------------------------------------------------------------------

SpatialFieldIntegrals IntegrateSpatialField(
    Mesh* mesh, unsigned solution_type, const double* global_values,
    SpatialScalarFunction exact, void* context) {
  SpatialFieldIntegrals totals;
  const unsigned rank = mesh->processor_id();
  std::vector<double> phi;
  std::vector<double> gradient;
  double weight = 0.0;
  for (unsigned element = mesh->_elementOffset[rank];
       element < mesh->_elementOffset[rank + 1u]; ++element) {
    const ElementData data = GatherElementData(mesh, solution_type, element);
    const unsigned element_dofs = data.dofs.size();
    std::vector<double> nodal(element_dofs, 0.0);
    for (unsigned a = 0; a < element_dofs; ++a)
      nodal[a] = global_values[mesh->GetSolutionDof(a, element,
                                                    solution_type)];
    for (unsigned gauss = 0;
         gauss < mesh->_finiteElement[data.geometry][solution_type]
                     ->GetGaussPointNumber(); ++gauss) {
      mesh->_finiteElement[data.geometry][solution_type]->Jacobian(
          data.coordinates, gauss, weight, phi, gradient);
      // Isoparametric map: the physical Gauss point uses the same shape
      // functions as the field, so both are evaluated consistently.
      double x = 0.0, z = 0.0, discrete = 0.0;
      // A 1D mesh has one coordinate axis: z stays 0 there.
      const bool has_z = data.coordinates.size() > 1u;
      for (unsigned a = 0; a < element_dofs; ++a) {
        x += phi[a] * data.coordinates[0][a];
        if (has_z) z += phi[a] * data.coordinates[1][a];
        discrete += phi[a] * nodal[a];
      }
      const double analytic = exact(x, z, context);
      const double difference = discrete - analytic;
      totals.error_squared += difference * difference * weight;
      totals.exact_squared += analytic * analytic * weight;
      totals.field_squared += discrete * discrete * weight;
    }
  }
  return totals;
}

SpatialFieldIntegrals IntegrateSpatialField3D(
    Mesh* mesh, unsigned solution_type, const double* global_values,
    SpatialScalarFunction3D exact, void* context) {
  SpatialFieldIntegrals totals;
  const unsigned rank = mesh->processor_id();
  std::vector<double> phi;
  std::vector<double> gradient;
  double weight = 0.0;
  for (unsigned element = mesh->_elementOffset[rank];
       element < mesh->_elementOffset[rank + 1u]; ++element) {
    const ElementData data = GatherElementData(mesh, solution_type, element);
    const unsigned element_dofs = data.dofs.size();
    if (data.coordinates.size() != 3u)
      throw std::runtime_error("IntegrateSpatialField3D needs a 3D mesh");
    std::vector<double> nodal(element_dofs, 0.0);
    for (unsigned a = 0; a < element_dofs; ++a)
      nodal[a] = global_values[mesh->GetSolutionDof(a, element, solution_type)];
    for (unsigned gauss = 0;
         gauss < mesh->_finiteElement[data.geometry][solution_type]->GetGaussPointNumber();
         ++gauss) {
      mesh->_finiteElement[data.geometry][solution_type]->Jacobian(
          data.coordinates, gauss, weight, phi, gradient);
      double x = 0.0, y = 0.0, z = 0.0, discrete = 0.0;
      for (unsigned a = 0; a < element_dofs; ++a) {
        x += phi[a] * data.coordinates[0][a];
        y += phi[a] * data.coordinates[1][a];
        z += phi[a] * data.coordinates[2][a];
        discrete += phi[a] * nodal[a];
      }
      const double analytic = exact(x, y, z, context);
      const double difference = discrete - analytic;
      totals.error_squared += difference * difference * weight;
      totals.exact_squared += analytic * analytic * weight;
      totals.field_squared += discrete * discrete * weight;
    }
  }
  return totals;
}

/** Boundary counterpart of IntegrateSpatialField.  The face selection, the
 *  face DOF gathering and the JacobianSur quadrature are the same ones
 *  BuildNativeSpatialOperators uses to assemble M_Gamma; nothing about the
 *  control assembly is changed or duplicated with different conventions. */
SpatialFieldIntegrals IntegrateBoundaryField(
    Mesh* mesh, unsigned solution_type, unsigned face_id,
    const double* global_values, SpatialScalarFunction exact, void* context) {
  SpatialFieldIntegrals totals;
  const unsigned rank = mesh->processor_id();
  const unsigned dimension = mesh->GetDimension();
  std::vector<double> phi;
  std::vector<double> gradient;
  std::vector<double> normal;
  double weight = 0.0;
  for (unsigned element = mesh->_elementOffset[rank];
       element < mesh->_elementOffset[rank + 1u]; ++element) {
    for (unsigned face = 0; face < mesh->GetElementFaceNumber(element);
         ++face) {
      const unsigned this_face = static_cast<unsigned>(
          -(mesh->el->GetFaceElementIndex(element, face) + 1));
      if (this_face != face_id) continue;
      const unsigned face_dofs =
          mesh->GetElementFaceDofNumber(element, face, solution_type);
      const unsigned face_geometry = mesh->GetElementFaceType(element, face);
      std::vector<std::vector<double> > face_coordinates(
          dimension, std::vector<double>(face_dofs, 0.0));
      std::vector<double> nodal(face_dofs, 0.0);
      for (unsigned a = 0; a < face_dofs; ++a) {
        const unsigned local_node =
            mesh->GetLocalFaceVertexIndex(element, face, a);
        nodal[a] = global_values[mesh->GetSolutionDof(local_node, element,
                                                      solution_type)];
        const unsigned coordinate_dof =
            mesh->GetSolutionDof(local_node, element, 2);
        for (unsigned axis = 0; axis < dimension; ++axis)
          face_coordinates[axis][a] =
              (*mesh->_topology->_Sol[axis])(coordinate_dof);
      }
      for (unsigned gauss = 0;
           gauss < mesh->_finiteElement[face_geometry][solution_type]
                       ->GetGaussPointNumber(); ++gauss) {
        mesh->_finiteElement[face_geometry][solution_type]->JacobianSur(
            face_coordinates, gauss, weight, phi, gradient, normal);
        double x = 0.0, z = 0.0, discrete = 0.0;
        for (unsigned a = 0; a < face_dofs; ++a) {
          x += phi[a] * face_coordinates[0][a];
          z += phi[a] * face_coordinates[1][a];
          discrete += phi[a] * nodal[a];
        }
        const double analytic = exact(x, z, context);
        const double difference = discrete - analytic;
        totals.error_squared += difference * difference * weight;
        totals.exact_squared += analytic * analytic * weight;
        totals.field_squared += discrete * discrete * weight;
      }
    }
  }
  return totals;
}

// ---------------------------------------------------------------------------
// The spatial operator owner
// ---------------------------------------------------------------------------

/** Dirichlet control: the controlled boundary DOFs C (the support of M_Gamma)
 *  are eliminated from M, K, M_track and M_ctrl_box exactly like FEMuS's
 *  constrained state DOFs (K diagonal 1, mass-like diagonal 0), after the
 *  unhatted M and K have been copied (mass_full / stiffness_full) for the coupling
 *  operator.  M_Gamma and K_Gamma keep their C entries: they are the control
 *  space.  The C rows of every product then vanish and the state operator E
 *  has identity rows on C (E_I). */
static void EliminateDirichletControlDofs(NativeSpatialOperators& operators,
                                          Mat* mass_full, Mat* stiffness_full) {
  Mat trace = static_cast<PetscMatrix*>(operators.M_Gamma.get())->mat();
  Vec diagonal = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, MatCreateVecs(trace, &diagonal, nullptr));
  PetscCallAbort(PETSC_COMM_WORLD, MatGetDiagonal(trace, diagonal));
  PetscReal scale = 0.0;
  PetscCallAbort(PETSC_COMM_WORLD, VecNorm(diagonal, NORM_INFINITY, &scale));
  PetscInt begin = 0, end = 0;
  PetscCallAbort(PETSC_COMM_WORLD, VecGetOwnershipRange(diagonal, &begin, &end));
  const PetscScalar* d = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(diagonal, &d));
  std::vector<PetscInt> rows;
  for (PetscInt i = 0; i < end - begin; ++i)
    if (PetscRealPart(d[i]) > 1.0e-12 * scale) rows.push_back(begin + i);
  PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(diagonal, &d));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&diagonal));

  PetscCallAbort(PETSC_COMM_WORLD,
                 MatDuplicate(static_cast<PetscMatrix*>(operators.M.get())->mat(),
                              MAT_COPY_VALUES, mass_full));
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatDuplicate(static_cast<PetscMatrix*>(operators.K.get())->mat(),
                              MAT_COPY_VALUES, stiffness_full));
  EliminateConstrainedDofs(operators.M.get(), rows, 0.0);
  EliminateConstrainedDofs(operators.K.get(), rows, 1.0);
  EliminateConstrainedDofs(operators.M_track.get(), rows, 0.0);
  EliminateConstrainedDofs(operators.M_ctrl_box.get(), rows, 0.0);
}

SpatialOperators::SpatialOperators(const ProblemConfig& config,
                                   femus::Mesh* mesh, unsigned solution_type,
                                   const femus::MultiLevelSolution& solution)
    : wall_start(MPI_Wtime()) {
  // The level-0 Solution carries FEMuS's generated constrained set of the
  // state variable "Ti".
  assembled = BuildNativeSpatialOperators(
      config, mesh, solution_type, solution.GetSolutionLevel(0),
      solution.GetIndex("Ti"), &lift);
  if (config.control.type == ProblemConfig::ControlType::DIRICHLET) {
    // The tracking region must not touch the controlled boundary: the
    // implemented KKT tracks the state on the free DOFs only (see the
    // Dirichlet design, general case not implemented).
    Mat track = static_cast<femus::PetscMatrix*>(assembled.M_track.get())->mat();
    Mat trace = static_cast<femus::PetscMatrix*>(assembled.M_Gamma.get())->mat();
    Vec dt = nullptr, dg = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD, MatCreateVecs(track, &dt, nullptr));
    PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(dt, &dg));
    PetscCallAbort(PETSC_COMM_WORLD, MatGetDiagonal(track, dt));
    PetscCallAbort(PETSC_COMM_WORLD, MatGetDiagonal(trace, dg));
    PetscCallAbort(PETSC_COMM_WORLD, VecPointwiseMult(dt, dt, dg));
    PetscReal overlap = 0.0;
    PetscCallAbort(PETSC_COMM_WORLD, VecNorm(dt, NORM_INFINITY, &overlap));
    PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&dg));
    PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&dt));
    if (overlap > 0.0)
      throw std::invalid_argument(
          "Dirichlet control: the tracking region touches the control face"
          " (tracking of controlled boundary DOFs is not implemented)");
    EliminateDirichletControlDofs(assembled, &mass_full, &stiffness_full);
  }
  mass = static_cast<femus::PetscMatrix*>(assembled.M.get())->mat();
  stiffness = static_cast<femus::PetscMatrix*>(assembled.K.get())->mat();
  tracking = static_cast<femus::PetscMatrix*>(assembled.M_track.get())->mat();
  control = SelectedControlOperator(config, assembled);
  if (!config.source_regions.empty())
    source = static_cast<femus::PetscMatrix*>(assembled.M_source.get())->mat();
  layout = BuildOwnedSpatialLayout(mesh, solution_type, mass);
  if (config.control.type == ProblemConfig::ControlType::DIRICHLET)
    dirichlet.reset(new DirichletControlOperators(
        config, mass_full, stiffness_full,
        static_cast<femus::PetscMatrix*>(assembled.M_Gamma.get())->mat(),
        static_cast<femus::PetscMatrix*>(assembled.K_Gamma.get())->mat()));
}

SpatialOperators::~SpatialOperators() {
  // The Dirichlet owner borrows the two full copies: it goes first.
  dirichlet.reset();
  PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&lift.tracking_full));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&lift.load));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&lift.g));
  PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&stiffness_full));
  PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&mass_full));
}
