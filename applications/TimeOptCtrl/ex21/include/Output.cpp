// Output layer.
//
// Field names, order, spacing and numeric precision of the JSON and of the two
// stdout records are a parsed contract; see the header.

#include "Output.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>
#include <petscmat.h>

#include "Options.hpp"
#include "Config.hpp"
#include "Discretization.hpp"
#include "Preconditioners.hpp"
#include "Solver.hpp"
#include "SpaceTimeOperators.hpp"

using namespace femus;

/** A "prefix + value" fragment at full double precision, for the optional
 *  report fields that must not appear at all when they are inactive. */
static std::string PrecisionField(const char* prefix, double value) {
  std::ostringstream out;
  out << std::setprecision(17) << prefix << value;
  return out.str();
}

static std::vector<double> GatherSpaceTimeToRoot(
    Vec space_time, Mat spatial_matrix, PetscInt spatial_local,
    unsigned steps) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  PetscInt spatial_global = 0;
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatGetSize(spatial_matrix, &spatial_global, nullptr));
  Vec spatial = nullptr, root = nullptr;
  VecScatter scatter = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatCreateVecs(spatial_matrix, &spatial, nullptr));
  PetscCallAbort(PETSC_COMM_WORLD,
                 VecScatterCreateToZero(spatial, &scatter, &root));
  std::vector<double> gathered;
  if (rank == 0) gathered.assign(steps * spatial_global, 0.0);
  for (unsigned slot = 0; slot < steps; ++slot) {
    CopyDataSideSliceToSpatial(space_time, slot, spatial_local, spatial);
    PetscCallAbort(PETSC_COMM_WORLD,
                   VecScatterBegin(scatter, spatial, root, INSERT_VALUES,
                                   SCATTER_FORWARD));
    PetscCallAbort(PETSC_COMM_WORLD,
                   VecScatterEnd(scatter, spatial, root, INSERT_VALUES,
                                 SCATTER_FORWARD));
    if (rank == 0) {
      const PetscScalar* values = nullptr;
      PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(root, &values));
      for (PetscInt dof = 0; dof < spatial_global; ++dof)
        gathered[slot * spatial_global + dof] = values[dof];
      PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(root, &values));
    }
  }
  PetscCallAbort(PETSC_COMM_WORLD, VecScatterDestroy(&scatter));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&root));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&spatial));
  return gathered;
}

static std::vector<double> GatherSpatialOwnedValuesToRoot(
    const std::vector<double>& owned, Mat spatial_matrix) {
  Vec spatial = nullptr, root = nullptr;
  VecScatter scatter = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatCreateVecs(spatial_matrix, &spatial, nullptr));
  PetscScalar* local = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, VecGetArray(spatial, &local));
  for (unsigned i = 0; i < owned.size(); ++i) local[i] = owned[i];
  PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArray(spatial, &local));
  PetscCallAbort(PETSC_COMM_WORLD,
                 VecScatterCreateToZero(spatial, &scatter, &root));
  PetscCallAbort(PETSC_COMM_WORLD,
                 VecScatterBegin(scatter, spatial, root, INSERT_VALUES,
                                 SCATTER_FORWARD));
  PetscCallAbort(PETSC_COMM_WORLD,
                 VecScatterEnd(scatter, spatial, root, INSERT_VALUES,
                               SCATTER_FORWARD));
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::vector<double> gathered;
  if (rank == 0) {
    PetscInt size = 0;
    PetscCallAbort(PETSC_COMM_WORLD, VecGetSize(root, &size));
    gathered.resize(size);
    const PetscScalar* values = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(root, &values));
    for (PetscInt i = 0; i < size; ++i) gathered[i] = values[i];
    PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(root, &values));
  }
  PetscCallAbort(PETSC_COMM_WORLD, VecScatterDestroy(&scatter));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&root));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&spatial));
  return gathered;
}

RootDistributedFields GatherDistributedSolveToRoot(
    const DistributedOptimalControlSolve& solve,
    const OwnedSpatialLayout& layout, Mat spatial_matrix, unsigned steps) {
  RootDistributedFields root;
  const std::vector<double> x =
      GatherSpatialOwnedValuesToRoot(layout.x, spatial_matrix);
  const std::vector<double> z =
      GatherSpatialOwnedValuesToRoot(layout.z, spatial_matrix);
  // T9: the third storage coordinate, 3D only (layout.three_d is the same
  // on every rank, so the collective gather is entered consistently).
  std::vector<double> w;
  if (layout.three_d) w = GatherSpatialOwnedValuesToRoot(layout.axis2, spatial_matrix);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    root.coordinates.assign(x.size(), std::vector<double>(layout.three_d ? 3 : 2, 0.0));
    for (unsigned i = 0; i < x.size(); ++i) {
      root.coordinates[i][0] = x[i];
      root.coordinates[i][1] = z[i];
      if (layout.three_d) root.coordinates[i][2] = w[i];
    }
  }
  root.state = GatherSpaceTimeToRoot(
      solve.fields.state, spatial_matrix, layout.local_size, steps);
  root.adjoint = GatherSpaceTimeToRoot(
      solve.fields.adjoint, spatial_matrix, layout.local_size, steps);
  root.control = GatherSpaceTimeToRoot(
      solve.fields.control, spatial_matrix, layout.local_size, steps);
  return root;
}

void WriteParallelSolveJsonRankZero(
    const std::string& path, const ProblemConfig& config,
    const DistributedOptimalControlSolve& solve,
    const RootDistributedFields& fields) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0) return;
  std::ofstream stream(path.c_str());
  if (!stream)
    throw std::runtime_error("cannot open parallel solve JSON: " + path);
  const unsigned spatial_dofs = fields.coordinates.size();
  stream << std::setprecision(17)
         << "{\n  \"config\":{\"control\":\""
         << (config.control.type == ProblemConfig::ControlType::NEUMANN
                 ? "NEUMANN"
             : config.control.type == ProblemConfig::ControlType::DIRICHLET
                 ? "DIRICHLET" : "DISTRIBUTED")
         << "\",\"mesh\":[" << config.mesh.nx << ','
         << (config.dimension == 3u ? std::to_string(config.mesh.ny) + "," : std::string())   // T9
         << config.mesh.nz
         << "],\"N\":" << config.time.steps << ",\"dt\":"
         << config.time.dt << ",\"time_scheme\":\""
         << TimeSchemeName(config) << "\",\"target_type\":\""
         << TrackingTargetName(config) << '"';
  // ex21: written only when active, so the default output is ex20's.
  if (config.dimension == 1u) stream << ",\"dimension\":1";
  if (HasConvection(config))
    stream << ",\"convection\":{\"nu\":" << config.diffusivity
           << ",\"vx\":" << config.convection.vx
           << ",\"vy\":" << config.convection.vy << '}';
  stream << "},\n"
         << "  \"solve\":{\"reason\":" << solve.reason
         << ",\"iterations\":" << solve.outer_iterations
         << ",\"reported_residual\":" << solve.reported_residual
         << ",\"true_residual\":" << solve.true_residual
         << ",\"inner_y_iterations\":" << solve.inner_y_iterations
         << ",\"inner_p_iterations\":" << solve.inner_p_iterations
         << ",\"inner_y_solves\":" << solve.inner_y_solves
         << ",\"inner_p_solves\":" << solve.inner_p_solves
         << "},\n"
         << "  \"kkt\":{\"state_kkt_residual_abs\":" << solve.kkt.state_abs
         << ",\"state_kkt_residual_rel\":" << solve.kkt.state_rel
         << ",\"adjoint_kkt_residual_abs\":" << solve.kkt.adjoint_abs
         << ",\"adjoint_kkt_residual_rel\":" << solve.kkt.adjoint_rel
         << ",\"stationarity_consistency_residual_abs\":"
         << solve.kkt.stationarity_abs
         << ",\"stationarity_consistency_residual_rel\":"
         << solve.kkt.stationarity_rel
         << ",\"reduced_block_y_abs\":" << solve.kkt.block_y_abs
         << ",\"reduced_block_p_abs\":" << solve.kkt.block_p_abs
         << ",\"active_control_dofs\":" << solve.kkt.active_dofs
         << ",\"legacy_geometric_mask_dofs\":" << solve.kkt.geometric_mask_dofs
         << ",\"control_offactive_dofs\":" << solve.kkt.control_offactive_dofs
         << ",\"control_offactive_max\":" << solve.kkt.control_offactive_max
         << ",\"note\":\"r_y and r_p test consistency of the converged triple"
            " with the discrete state and adjoint equations; r_u is only a"
            " stationarity CONSISTENCY check because u is reconstructed from"
            " p, and is NOT an independent reduced-gradient optimality test\""
         << "},\n"
         // J_track/J_reg/J_total carry the dt quadrature weight; the
         // *_unweighted fields are the same sums without it.
         << "  \"functional\":{\"J_track\":" << solve.fields.J_track
         << ",\"J_reg\":" << solve.fields.J_reg
         << ",\"J_total\":"
         << solve.fields.J_track + solve.fields.J_reg
         << ",\"J_track_unweighted\":" << solve.fields.J_track_unweighted
         << ",\"J_reg_unweighted\":" << solve.fields.J_reg_unweighted
         << ",\"dt_weighted\":true"
         // The temporal part of J_reg and its weight are emitted ONLY when the
         // temporal regularization is active, so every gamma_u = 0 run -- that
         // is, every accepted stage 1-3 run -- produces a byte-identical file.
         << (HasControlTimeRegularization(config)
                 ? PrecisionField(",\"J_reg_time\":", solve.fields.J_reg_time) +
                       PrecisionField(",\"control_h1_time_weight\":",
                                      config.control_h1_time_weight)
                 : std::string())
         << "},\n";
  // Present only for the reduced Schur solves (Neumann boundary-reduced,
  // Dirichlet reduced).
  if (solve.reduced.active)
    stream << "  \"boundary_reduced\":{\"pc\":\""
           << (solve.reduced.dirichlet ? "SCHUR_DIRICHLET_REDUCED"
                                       : "SCHUR_BOUNDARY_REDUCED")
           << "\",\"dimension\":" << solve.reduced.dimension
           << ",\"support\":" << solve.reduced.support
           << ",\"forward_backend\":\"" << solve.reduced.forward_backend
           << "\",\"z_backend\":\"" << solve.reduced.z_backend
           << "\",\"prepared_beta\":" << solve.reduced.prepared_beta
           << "},\n";
  stream << "  \"spatial_coordinates\":[";
  for (unsigned i = 0; i < spatial_dofs; ++i) {
    if (i) stream << ',';
    stream << '[' << fields.coordinates[i][0] << ','
           << fields.coordinates[i][1];
    if (fields.coordinates[i].size() == 3u) stream << ',' << fields.coordinates[i][2];   // T9: 3D
    stream << ']';
  }
  const char* names[] = {"state", "adjoint", "control_full"};
  const std::vector<double>* values[] = {
      &fields.state, &fields.adjoint, &fields.control};
  stream << ']';
  for (unsigned field = 0; field < 3u; ++field) {
    stream << ",\n  \"" << names[field] << "\":[";
    for (unsigned slot = 0; slot < config.time.steps; ++slot) {
      if (slot) stream << ',';
      stream << "{\"slot\":" << slot << ",\"time\":"
             << (slot + 1u) * config.time.dt << ",\"values\":[";
      for (unsigned dof = 0; dof < spatial_dofs; ++dof) {
        if (dof) stream << ',';
        stream << (*values[field])[slot * spatial_dofs + dof];
      }
      stream << "]}";
    }
    stream << ']';
  }
  stream << "\n}\n";
}

void WriteParallelVtk(
    const std::string& output_path, const std::string& order,
    MultiLevelSolution& solution, Mesh* mesh,
    const OwnedSpatialLayout& layout,
    const DistributedDataSideResult& fields, unsigned steps,
    const ProblemConfig& config, bool region_fields) {
  solution.SetWriter(VTK);
  Solution* level_solution = solution.GetSolutionLevel(0);
  const unsigned state_index = solution.GetIndex("Y");
  const unsigned adjoint_index = solution.GetIndex("P");
  const unsigned control_index = solution.GetIndex("u_q");
  std::vector<std::string> output_fields = {"Y", "P", "u_q"};
  // Region markers are prepended, not appended: see the ordering note in
  // InitializeStateSpace.  They must occupy positions 0 and 1 of this list.
  // The region markers are time independent, so they are filled once, before
  // the slot loop, and simply re-emitted with every time step.
  if (region_fields) {
    Solution* region_solution = solution.GetSolutionLevel(0);
    const unsigned tracking_index = solution.GetIndex("tracking_region");
    const unsigned control_index_marker = solution.GetIndex("control_region");
    const unsigned tracking_type = solution.GetSolutionType(tracking_index);
    const unsigned control_type = solution.GetSolutionType(control_index_marker);
    region_solution->_Sol[tracking_index]->zero();
    region_solution->_Sol[control_index_marker]->zero();
    const unsigned marker_rank = mesh->processor_id();
    for (unsigned element = mesh->_elementOffset[marker_rank];
         element < mesh->_elementOffset[marker_rank + 1u]; ++element) {
      // One DOF per element for a ZERO-order discontinuous field.
      region_solution->_Sol[tracking_index]->set(
          mesh->GetSolutionDof(0, element, tracking_type),
          static_cast<double>(TrackingRegionIndex(config, mesh, element)));
      region_solution->_Sol[control_index_marker]->set(
          mesh->GetSolutionDof(0, element, control_type),
          static_cast<double>(ControlRegionIndex(config, mesh, element)));
    }
    region_solution->_Sol[tracking_index]->close();
    region_solution->_Sol[control_index_marker]->close();
    output_fields.insert(output_fields.begin(), "control_region");
    output_fields.insert(output_fields.begin(), "tracking_region");
  }
  const PetscScalar* state = nullptr;
  const PetscScalar* adjoint = nullptr;
  const PetscScalar* control = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(fields.state, &state));
  PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(fields.adjoint, &adjoint));
  PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(fields.control, &control));

  for (unsigned slot = 0; slot < steps; ++slot) {
    level_solution->_Sol[state_index]->zero();
    level_solution->_Sol[adjoint_index]->zero();
    level_solution->_Sol[control_index]->zero();
    const PetscInt offset =
        static_cast<PetscInt>(slot) * layout.local_size;
    // The space-time rows are the solution DOFs themselves and this rank's
    // block of a slice is exactly its owned DOF range, so each owned DOF is
    // written once; close() then delivers the owned values to the ghosts.
    for (PetscInt i = 0; i < layout.local_size; ++i) {
      const unsigned application_dof =
          static_cast<unsigned>(layout.begin + i);
      level_solution->_Sol[state_index]->set(
          application_dof, PetscRealPart(state[offset + i]));
      level_solution->_Sol[adjoint_index]->set(
          application_dof, PetscRealPart(adjoint[offset + i]));
      level_solution->_Sol[control_index]->set(
          application_dof, PetscRealPart(control[offset + i]));
    }
    level_solution->_Sol[state_index]->close();
    level_solution->_Sol[adjoint_index]->close();
    level_solution->_Sol[control_index]->close();
    solution.GetWriter()->Write(output_path, order.c_str(), output_fields,
                                slot + 1u);
  }

  PetscCallAbort(PETSC_COMM_WORLD,
                 VecRestoreArrayRead(fields.control, &control));
  PetscCallAbort(PETSC_COMM_WORLD,
                 VecRestoreArrayRead(fields.adjoint, &adjoint));
  PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(fields.state, &state));
}


void PrintParallelSolveSummary(const ProblemConfig& config,
                               const DistributedOptimalControlSolve& solve,
                               DistributedSolvePc pc_mode,
                               const std::string& json_path) {
  int rank = 0, ranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  long long inner_y_min = 0, inner_y_max = 0;
  long long inner_p_min = 0, inner_p_max = 0;
  MPI_Reduce(&solve.inner_y_iterations, &inner_y_min, 1, MPI_LONG_LONG,
             MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&solve.inner_y_iterations, &inner_y_max, 1, MPI_LONG_LONG,
             MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&solve.inner_p_iterations, &inner_p_min, 1, MPI_LONG_LONG,
             MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&solve.inner_p_iterations, &inner_p_max, 1, MPI_LONG_LONG,
             MPI_MAX, 0, MPI_COMM_WORLD);
  // Dirichlet control: the H1(Gamma_c) length scale of the regularization.
  std::string dirichlet_suffix;
  if (config.control.type == ProblemConfig::ControlType::DIRICHLET) {
    std::ostringstream suffix;
    suffix << std::setprecision(17) << " h1_length=" << config.control_h1_length;
    if (HasControlTimeRegularization(config))
      suffix << " h1_time_weight=" << config.control_h1_time_weight;
    dirichlet_suffix = suffix.str();
  }
  if (rank == 0)
    std::cout << std::setprecision(17)
              << "EX20_PAR_SOLVE control="
              << (config.control.type == ProblemConfig::ControlType::NEUMANN
                      ? "NEUMANN"
                  : config.control.type == ProblemConfig::ControlType::DIRICHLET
                      ? "DIRICHLET"
                  : (ControlRegionIsWholeDomain(config)
                         ? "DISTRIBUTED_WHOLE" : "DISTRIBUTED_BOX"))
              << dirichlet_suffix
              << " target=" << TrackingTargetName(config)
              << " mesh=" << config.mesh.nx << 'x'
              << (config.dimension == 3u ? std::to_string(config.mesh.ny) + "x" : std::string())
              << config.mesh.nz
              << " N=" << config.time.steps
              << " time_scheme=" << TimeSchemeName(config)
              << (config.dimension == 1u ? " dimension=1" : "")
              << (HasConvection(config) ? " convection=on" : "")
              << " nrank=" << ranks
              << (pc_mode == DistributedSolvePc::SCHUR_BOUNDARY_REDUCED
                      ? " pc=SCHUR_BOUNDARY_REDUCED inner=EXACT_BLOCK_ELIMINATION"
                        " ownership=distributed"
                  : pc_mode == DistributedSolvePc::SCHUR_DIRICHLET_REDUCED
                      ? " pc=SCHUR_DIRICHLET_REDUCED inner=EXACT_WOODBURY"
                        " ownership=distributed"
                  : pc_mode == DistributedSolvePc::SCHUR_SWEEP
                      ? " pc=SCHUR_SWEEP inner=TIME_SWEEP+CG/GAMG"
                        " ownership=distributed"
                      : (pc_mode == DistributedSolvePc::SCHUR_MATCHING
                             ? " pc=SCHUR_MATCHING inner_pc=GAMG"
                               " ownership=distributed"
                             : " pc=ADDITIVE inner_pc=GAMG"
                               " Amat=MatShell Pmat=AIJ"))
              << " reason=" << solve.reason
              << " outer_iterations=" << solve.outer_iterations
              << " inner_y_iterations_minmax=[" << inner_y_min << ','
              << inner_y_max << "] inner_p_iterations_minmax=["
              << inner_p_min << ',' << inner_p_max << ']'
              << " inner_y_solves=" << solve.inner_y_solves
              << " inner_p_solves=" << solve.inner_p_solves
              << " true_residual=" << solve.true_residual
              << " state_kkt_residual_abs=" << solve.kkt.state_abs
              << " state_kkt_residual_rel=" << solve.kkt.state_rel
              << " adjoint_kkt_residual_abs=" << solve.kkt.adjoint_abs
              << " adjoint_kkt_residual_rel=" << solve.kkt.adjoint_rel
              << " stationarity_consistency_residual_abs="
              << solve.kkt.stationarity_abs
              << " stationarity_consistency_residual_rel="
              << solve.kkt.stationarity_rel
              // dt-weighted.
              << " J_track=" << solve.fields.J_track
              << " J_reg=" << solve.fields.J_reg
              << (HasControlTimeRegularization(config)
                      ? PrecisionField(" J_reg_time=", solve.fields.J_reg_time)
                      : std::string())
              << " json=" << json_path << std::endl;
}

void PrintSolverStats(const ProblemConfig& config,
                      const DistributedOptimalControlSolve& solve,
                      DistributedSolvePc pc_mode) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    auto stat = [](const InnerIterationCounter& c) {
      std::ostringstream o;
      o << c.solves << '/' << c.iterations << '/' << std::fixed
        << std::setprecision(2) << c.mean() << '/' << c.max_iterations;
      return o.str();
    };
    std::cout << std::setprecision(6)
              << "EX20_SOLVERSTATS pc="
              << (pc_mode == DistributedSolvePc::SCHUR_BOUNDARY_REDUCED
                      ? "SCHUR_BOUNDARY_REDUCED"
                  : pc_mode == DistributedSolvePc::SCHUR_DIRICHLET_REDUCED
                      ? "SCHUR_DIRICHLET_REDUCED"
                  : pc_mode == DistributedSolvePc::SCHUR_SWEEP ? "SCHUR_SWEEP"
                  : (pc_mode == DistributedSolvePc::SCHUR_MATCHING
                         ? "SCHUR_MATCHING" : "ADDITIVE"))
              << " nx=" << config.mesh.nx << " N=" << config.time.steps
              << " alpha_u=" << config.alpha_u
              << " outer=" << solve.outer_iterations
              << " reason=" << solve.reason
              << " sweeps=" << (solve.sweeps_used ? 1 : 0)
              << " sweep_failures=" << solve.sweep_failures
              << "  [solves/iters/mean/max]"
              << " y=" << stat(solve.y_split)
              << " p=" << stat(solve.p_split)
              << " right=" << stat(solve.right_inner)
              << " left=" << stat(solve.left_inner)
              << " ay=" << stat(solve.ay_inner)
              << " t_operators=" << solve.timings.operators
              << " t_pcsetup=" << solve.timings.pc_setup
              << " t_kspsolve=" << solve.timings.ksp_solve
              << " t_total=" << solve.timings.total
              << std::endl;
    if (solve.reduced.active) {
      std::cout << std::setprecision(17)
                << "EX20_REDUCED_SCHUR pc="
                << (pc_mode == DistributedSolvePc::SCHUR_DIRICHLET_REDUCED
                        ? "SCHUR_DIRICHLET_REDUCED" : "SCHUR_BOUNDARY_REDUCED")
                << " dimension=" << solve.reduced.dimension
                << " support=" << solve.reduced.support
                << " forward_backend=" << solve.reduced.forward_backend
                << " z_backend=" << solve.reduced.z_backend
                << " prepared_beta=" << solve.reduced.prepared_beta;
      if (pc_mode == DistributedSolvePc::SCHUR_DIRICHLET_REDUCED)
        std::cout << std::setprecision(6)
                  << " setup_seconds=" << solve.reduced.setup_seconds
                  << " sweep_solves=" << solve.reduced.sweep_solves;
      std::cout << std::endl;
    }
  }
}


void WriteSolution(DistributedOptimalControlSolve& solve,
                   const ProblemConfig& config,
                   const ApplicationOptions& options,
                   MultiLevelSolution& solution, Mesh* mesh,
                   const SpatialOperators& spatial) {
  const RootDistributedFields fields = GatherDistributedSolveToRoot(
      solve, spatial.layout, spatial.mass, config.time.steps);
  WriteParallelSolveJsonRankZero(options.json_path, config, solve, fields);
  if (options.paraview_enabled)
    WriteParallelVtk(options.output_path, options.paraview_order, solution,
                     mesh, spatial.layout, solve.fields,
                     config.time.steps, config, options.region_fields);
  PrintParallelSolveSummary(config, solve, options.pc_mode, options.json_path);
  solve.timings.total = MPI_Wtime() - spatial.wall_start;
  PrintSolverStats(config, solve, options.pc_mode);
}
