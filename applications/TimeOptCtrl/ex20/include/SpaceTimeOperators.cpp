// Space-time operator layer.

#include "SpaceTimeOperators.hpp"

#include <mpi.h>
#include <petscmat.h>

#include <algorithm>
#include <unordered_map>
#include <vector>

#include "Discretization.hpp"

// ---------------------------------------------------------------------------
// Time-slice transfers between a space-time array and a spatial vector.
//
// These two are the single implementation of the slice copy: the MatShell
// callback uses them directly on the arrays it holds open for the whole
// product (returning PETSc error codes, as a callback must), and the public
// Vec-level pair at the end of this file wraps them for everybody else.
// ---------------------------------------------------------------------------

static PetscErrorCode CopyLocalSliceToSpatial(const PetscScalar* source,
                                       PetscInt offset, PetscInt count,
                                       Vec destination) {
  PetscScalar* values = nullptr;
  PetscCall(VecGetArray(destination, &values));
  for (PetscInt i = 0; i < count; ++i)
    values[i] = source[offset + i];
  PetscCall(VecRestoreArray(destination, &values));
  return PETSC_SUCCESS;
}

static PetscErrorCode AddSpatialToLocalSlice(Vec source, PetscScalar factor,
                                      PetscInt offset, PetscInt count,
                                      PetscScalar* destination) {
  const PetscScalar* values = nullptr;
  PetscCall(VecGetArrayRead(source, &values));
  for (PetscInt i = 0; i < count; ++i)
    destination[offset + i] += factor * values[i];
  PetscCall(VecRestoreArrayRead(source, &values));
  return PETSC_SUCCESS;
}

static PetscErrorCode MatShellMultDistributedSpaceTime(Mat matrix, Vec input,
                                                Vec output) {
  DistributedSpaceTimeContext* context = nullptr;
  PetscCall(MatShellGetContext(matrix, reinterpret_cast<void**>(&context)));
  if (!context) return PETSC_ERR_ARG_NULL;
  const PetscScalar* input_values = nullptr;
  PetscScalar* output_values = nullptr;
  PetscCall(VecGetArrayRead(input, &input_values));
  PetscCall(VecGetArray(output, &output_values));
  const PetscInt local_total = context->steps * context->spatial_local;
  for (PetscInt i = 0; i < local_total; ++i) output_values[i] = 0.0;

  for (unsigned slot = 0; slot < context->steps; ++slot) {
    const PetscInt offset = static_cast<PetscInt>(slot) *
                            context->spatial_local;
    PetscCall(CopyLocalSliceToSpatial(input_values, offset,
                                      context->spatial_local,
                                      context->spatial_input));
    if (context->kind == DistributedSpaceTimeKind::AY ||
        context->kind == DistributedSpaceTimeKind::AP) {
      // Crank-Nicolson halves the stiffness on the diagonal block and adds
      // the SAME halved stiffness to the neighbour block; backward Euler is
      // the untouched original path (weight 1 on the diagonal stiffness, a
      // pure mass neighbour), so its operation ordering is unchanged.
      const bool crank_nicolson =
          context->scheme == ProblemConfig::TimeScheme::CRANK_NICOLSON;
      if (context->diagonal_block) {
        // Crank-Nicolson with Dirichlet control: explicit A_plus / A_minus,
        // whose C rows carry the strong elimination exactly (identity on the
        // diagonal block, zero on the neighbour).
        PetscCall(MatMult(context->diagonal_block, context->spatial_input,
                          context->spatial_mass_result));
        PetscCall(AddSpatialToLocalSlice(context->spatial_mass_result, 1.0,
                                         offset, context->spatial_local,
                                         output_values));
        const bool has_neighbour =
            (context->kind == DistributedSpaceTimeKind::AY && slot > 0u) ||
            (context->kind == DistributedSpaceTimeKind::AP &&
             slot + 1u < context->steps);
        if (has_neighbour) {
          const unsigned other_slot =
              context->kind == DistributedSpaceTimeKind::AY ? slot - 1u
                                                            : slot + 1u;
          PetscCall(CopyLocalSliceToSpatial(
              input_values, static_cast<PetscInt>(other_slot) * context->spatial_local,
              context->spatial_local, context->spatial_other));
          PetscCall(MatMult(context->neighbour_block, context->spatial_other,
                            context->spatial_other_mass_result));
          PetscCall(AddSpatialToLocalSlice(context->spatial_other_mass_result,
                                           1.0, offset, context->spatial_local,
                                           output_values));
        }
        continue;
      }
      PetscCall(MatMult(context->mass, context->spatial_input,
                        context->spatial_mass_result));
      PetscCall(MatMult(context->stiffness, context->spatial_input,
                        context->spatial_stiffness_result));
      PetscCall(AddSpatialToLocalSlice(context->spatial_mass_result,
                                       1.0 / context->dt, offset,
                                       context->spatial_local, output_values));
      PetscCall(AddSpatialToLocalSlice(context->spatial_stiffness_result,
                                       crank_nicolson ? 0.5 : 1.0,
                                       offset, context->spatial_local,
                                       output_values));
      const bool has_other =
          (context->kind == DistributedSpaceTimeKind::AY && slot > 0u) ||
          (context->kind == DistributedSpaceTimeKind::AP &&
           slot + 1u < context->steps);
      if (has_other) {
        const unsigned other_slot =
            context->kind == DistributedSpaceTimeKind::AY ? slot - 1u
                                                           : slot + 1u;
        const PetscInt other_offset =
            static_cast<PetscInt>(other_slot) * context->spatial_local;
        PetscCall(CopyLocalSliceToSpatial(input_values, other_offset,
                                          context->spatial_local,
                                          context->spatial_other));
        PetscCall(MatMult(context->mass, context->spatial_other,
                          context->spatial_other_mass_result));
        PetscCall(AddSpatialToLocalSlice(context->spatial_other_mass_result,
                                         -1.0 / context->dt, offset,
                                         context->spatial_local, output_values));
        if (crank_nicolson) {
          PetscCall(MatMult(context->stiffness, context->spatial_other,
                            context->spatial_other_stiffness_result));
          PetscCall(AddSpatialToLocalSlice(
              context->spatial_other_stiffness_result, 0.5, offset,
              context->spatial_local, output_values));
        }
      }
    } else {
      // Block diagonal in time; the last slot may carry its own scale (the
      // trapezoidal half weight of the Crank-Nicolson (2,1) block).
      const double slot_scale = (slot + 1u == context->steps)
                                    ? context->final_scale : context->scale;
      PetscCall(MatMult(context->diagonal_mass, context->spatial_input,
                        context->spatial_mass_result));
      PetscCall(AddSpatialToLocalSlice(context->spatial_mass_result,
                                       slot_scale, offset,
                                       context->spatial_local, output_values));
    }
  }
  PetscCall(VecRestoreArray(output, &output_values));
  PetscCall(VecRestoreArrayRead(input, &input_values));
  return PETSC_SUCCESS;
}

static void InitializeDistributedSpaceTimeContext(
    DistributedSpaceTimeContext& context, DistributedSpaceTimeKind kind,
    Mat mass, Mat stiffness, Mat diagonal_mass, unsigned steps,
    double dt, double scale, ProblemConfig::TimeScheme scheme,
    double final_scale, Mat diagonal_block = nullptr,
    Mat neighbour_block = nullptr) {
  context.kind = kind;
  context.mass = mass;
  context.stiffness = stiffness;
  context.diagonal_mass = diagonal_mass;
  context.steps = steps;
  context.dt = dt;
  context.scale = scale;
  context.scheme = scheme;
  context.final_scale = final_scale;
  context.diagonal_block = diagonal_block;
  context.neighbour_block = neighbour_block;
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatGetLocalSize(mass, &context.spatial_local, nullptr));
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatGetSize(mass, &context.spatial_global, nullptr));
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatCreateVecs(mass, &context.spatial_input, nullptr));
  PetscCallAbort(PETSC_COMM_WORLD,
                 VecDuplicate(context.spatial_input, &context.spatial_other));
  PetscCallAbort(PETSC_COMM_WORLD,
                 VecDuplicate(context.spatial_input,
                              &context.spatial_mass_result));
  PetscCallAbort(PETSC_COMM_WORLD,
                 VecDuplicate(context.spatial_input,
                              &context.spatial_stiffness_result));
  PetscCallAbort(PETSC_COMM_WORLD,
                 VecDuplicate(context.spatial_input,
                              &context.spatial_other_mass_result));
  PetscCallAbort(PETSC_COMM_WORLD,
                 VecDuplicate(context.spatial_input,
                              &context.spatial_other_stiffness_result));
}

static Mat CreateDistributedSpaceTimeShell(DistributedSpaceTimeContext& context) {
  Mat matrix = nullptr;
  const PetscInt local = context.steps * context.spatial_local;
  const PetscInt global = context.steps * context.spatial_global;
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatCreateShell(PETSC_COMM_WORLD, local, local, global, global,
                                &context, &matrix));
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatShellSetOperation(
                     matrix, MATOP_MULT,
                     reinterpret_cast<void (*)(void)>(
                         MatShellMultDistributedSpaceTime)));
  return matrix;
}

static void DestroyDistributedSpaceTimeContext(
    DistributedSpaceTimeContext& context) {
  PetscCallAbort(PETSC_COMM_WORLD,
                 VecDestroy(&context.spatial_other_stiffness_result));
  PetscCallAbort(PETSC_COMM_WORLD,
                 VecDestroy(&context.spatial_other_mass_result));
  PetscCallAbort(PETSC_COMM_WORLD,
                 VecDestroy(&context.spatial_stiffness_result));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&context.spatial_mass_result));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&context.spatial_other));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&context.spatial_input));
}

/** The Dirichlet (1,2) block: a shell over the same layout whose MULT is
 *  scale * B_D R^{-1} B_D^T on the borrowed Dirichlet operators. */
static Mat CreateDirichletControlShell(DirichletControlShellContext& context,
                                       PetscInt local, PetscInt global) {
  Mat matrix = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatCreateShell(PETSC_COMM_WORLD, local, local, global, global,
                                &context, &matrix));
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatShellSetOperation(
                     matrix, MATOP_MULT,
                     reinterpret_cast<void (*)(void)>(
                         ApplyDirichletControlShell)));
  return matrix;
}

void BuildDistributedSpaceTimeOperators(
    DistributedSpaceTimeOperators& operators, const ProblemConfig& config,
    Mat mass, Mat stiffness, Mat tracking, Mat control,
    DirichletControlOperators* dirichlet, Mat cn_diagonal, Mat cn_neighbour) {
  const ProblemConfig::TimeScheme scheme = config.time.scheme;
  // (1,2): eta^2/alpha_u (I (x) N), block diagonal with a uniform scale under
  // BOTH schemes -- the Crank-Nicolson control is piecewise constant on each
  // interval (W_u = dt I), and ex20's dt-scaled adjoint absorbs the dt.
  const double control_scale = config.eta * config.eta / config.alpha_u;
  // (2,1): -alpha_track (W_y/dt (x) M_omega); W_y/dt = diag(1,...,1,1/2) for
  // Crank-Nicolson, the identity for backward Euler.
  const double tracking_scale = -config.alpha_track;
  const double tracking_final_scale =
      tracking_scale * TrackingSlotWeight(config, config.time.steps - 1u);
  InitializeDistributedSpaceTimeContext(
      operators.ay_context, DistributedSpaceTimeKind::AY, mass, stiffness,
      nullptr, config.time.steps, config.time.dt, 1.0, scheme, 1.0,
      cn_diagonal, cn_neighbour);
  InitializeDistributedSpaceTimeContext(
      operators.ap_context, DistributedSpaceTimeKind::AP, mass, stiffness,
      nullptr, config.time.steps, config.time.dt, 1.0, scheme, 1.0,
      cn_diagonal, cn_neighbour);
  InitializeDistributedSpaceTimeContext(
      operators.ayp_context, DistributedSpaceTimeKind::AYP, mass, nullptr,
      control, config.time.steps, config.time.dt, control_scale, scheme,
      control_scale);
  InitializeDistributedSpaceTimeContext(
      operators.apy_context, DistributedSpaceTimeKind::APY, mass, nullptr,
      tracking, config.time.steps, config.time.dt, tracking_scale, scheme,
      tracking_final_scale);
  operators.ay = CreateDistributedSpaceTimeShell(operators.ay_context);
  operators.ap = CreateDistributedSpaceTimeShell(operators.ap_context);
  if (dirichlet) {
    operators.dirichlet_context.operators = dirichlet;
    operators.dirichlet_context.scale =
        config.eta * config.eta / config.alpha_u;
    operators.ayp = CreateDirichletControlShell(
        operators.dirichlet_context,
        operators.ayp_context.steps * operators.ayp_context.spatial_local,
        operators.ayp_context.steps * operators.ayp_context.spatial_global);
  } else {
    operators.ayp = CreateDistributedSpaceTimeShell(operators.ayp_context);
  }
  operators.apy = CreateDistributedSpaceTimeShell(operators.apy_context);
  Mat blocks[4] = {operators.ay, operators.ayp,
                   operators.apy, operators.ap};
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatCreateNest(PETSC_COMM_WORLD, 2, nullptr, 2, nullptr,
                               blocks, &operators.nest));
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatAssemblyBegin(operators.nest, MAT_FINAL_ASSEMBLY));
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatAssemblyEnd(operators.nest, MAT_FINAL_ASSEMBLY));
}

void DestroyDistributedSpaceTimeOperators(
    DistributedSpaceTimeOperators& operators) {
  PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&operators.nest));
  PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&operators.apy));
  PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&operators.ayp));
  PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&operators.ap));
  PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&operators.ay));
  DestroyDistributedSpaceTimeContext(operators.apy_context);
  DestroyDistributedSpaceTimeContext(operators.ayp_context);
  DestroyDistributedSpaceTimeContext(operators.ap_context);
  DestroyDistributedSpaceTimeContext(operators.ay_context);
}

PetscInt OwnershipPermutedSpaceTimeIndex(
    PetscInt spatial_dof, unsigned slot, unsigned steps,
    const PetscInt* spatial_ranges, int ranks) {
  const PetscInt* upper =
      std::upper_bound(spatial_ranges, spatial_ranges + ranks + 1,
                       spatial_dof);
  int owner = static_cast<int>(upper - spatial_ranges) - 1;
  if (owner < 0 || owner >= ranks)
    throw std::runtime_error("spatial column outside PETSc ownership");
  const PetscInt owner_local =
      spatial_ranges[owner + 1] - spatial_ranges[owner];
  return static_cast<PetscInt>(steps) * spatial_ranges[owner] +
         static_cast<PetscInt>(slot) * owner_local +
         (spatial_dof - spatial_ranges[owner]);
}

/** One spatial row of `mass_weight * M + stiffness_weight * K`.
 *
 *  Backward Euler calls it with (1/dt, 1) for the diagonal block only;
 *  Crank-Nicolson calls it with (1/dt, 1/2) for the diagonal block A_plus and
 *  with (-1/dt, 1/2) for the sub-diagonal block A_minus. */
static std::unordered_map<PetscInt, PetscScalar> CombinedSpatialRow(
    Mat mass, Mat stiffness, PetscInt row, double mass_weight,
    double stiffness_weight = 1.0) {
  std::unordered_map<PetscInt, PetscScalar> values;
  PetscInt count = 0;
  const PetscInt* columns = nullptr;
  const PetscScalar* entries = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatGetRow(mass, row, &count, &columns, &entries));
  for (PetscInt i = 0; i < count; ++i)
    values[columns[i]] += mass_weight * entries[i];
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatRestoreRow(mass, row, &count, &columns, &entries));
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatGetRow(stiffness, row, &count, &columns, &entries));
  for (PetscInt i = 0; i < count; ++i)
    values[columns[i]] += stiffness_weight * entries[i];
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatRestoreRow(stiffness, row, &count, &columns, &entries));
  return values;
}

void BuildAssembledSpaceTimeOperators(AssembledSpaceTimeOperators& operators,
                                      Mat mass, Mat stiffness, unsigned steps,
                                      double dt,
                                      ProblemConfig::TimeScheme scheme,
                                      Mat cn_diagonal, Mat cn_neighbour) {
  PetscInt spatial_begin = 0, spatial_end = 0, spatial_global = 0;
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatGetOwnershipRange(mass, &spatial_begin, &spatial_end));
  PetscCallAbort(PETSC_COMM_WORLD, MatGetSize(mass, &spatial_global, nullptr));
  const PetscInt spatial_local = spatial_end - spatial_begin;
  const PetscInt local_size = static_cast<PetscInt>(steps) * spatial_local;
  const PetscInt global_size = static_cast<PetscInt>(steps) * spatial_global;
  const PetscInt* spatial_ranges = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatGetOwnershipRanges(mass, &spatial_ranges));
  int rank = 0, ranks = 0;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  MPI_Comm_size(PETSC_COMM_WORLD, &ranks);
  const PetscInt expected_begin =
      static_cast<PetscInt>(steps) * spatial_ranges[rank];
  const PetscInt expected_end =
      static_cast<PetscInt>(steps) * spatial_ranges[rank + 1];
  std::vector<PetscInt> diagonal_nnz(local_size, 0);
  std::vector<PetscInt> offdiagonal_nnz(local_size, 0);
  const double inverse_dt = 1.0 / dt;
  // Backward Euler: diagonal M/dt + K, sub-diagonal -M/dt (the MASS pattern).
  // Crank-Nicolson: diagonal A_plus = M/dt + K/2, sub-diagonal
  // A_minus = -M/dt + K/2 (the UNION M+K pattern), so both blocks are built by
  // the same CombinedSpatialRow and preallocated from it.
  const bool crank_nicolson =
      scheme == ProblemConfig::TimeScheme::CRANK_NICOLSON;
  const double diagonal_stiffness_weight = crank_nicolson ? 0.5 : 1.0;
  // Crank-Nicolson with Dirichlet control supplies the two blocks explicitly;
  // CombinedSpatialRow then reads them with weights (1, 0) instead of
  // recombining M and K, so the eliminated C rows are assembled exactly.
  Mat diagonal_source_mass = cn_diagonal ? cn_diagonal : mass;
  Mat diagonal_source_stiffness = cn_diagonal ? cn_diagonal : stiffness;
  const double diagonal_mass_weight = cn_diagonal ? 1.0 : inverse_dt;
  const double diagonal_k_weight = cn_diagonal ? 0.0 : diagonal_stiffness_weight;
  Mat sub_source_mass = cn_neighbour ? cn_neighbour : mass;
  Mat sub_source_stiffness = cn_neighbour ? cn_neighbour : stiffness;
  const double sub_mass_weight = cn_neighbour ? 1.0 : -inverse_dt;
  const double sub_k_weight = cn_neighbour ? 0.0 : 0.5;
  for (unsigned slot = 0; slot < steps; ++slot) {
    for (PetscInt spatial_row = spatial_begin; spatial_row < spatial_end;
         ++spatial_row) {
      const PetscInt local_row = static_cast<PetscInt>(slot) * spatial_local +
                                 spatial_row - spatial_begin;
      const std::unordered_map<PetscInt, PetscScalar> diagonal =
          CombinedSpatialRow(diagonal_source_mass, diagonal_source_stiffness,
                             spatial_row, diagonal_mass_weight,
                             diagonal_k_weight);
      for (const auto& entry : diagonal) {
        const PetscInt column = OwnershipPermutedSpaceTimeIndex(
            entry.first, slot, steps, spatial_ranges, ranks);
        if (column >= expected_begin && column < expected_end)
          ++diagonal_nnz[local_row];
        else
          ++offdiagonal_nnz[local_row];
      }
      if (slot == 0u) continue;
      if (crank_nicolson) {
        const std::unordered_map<PetscInt, PetscScalar> subdiagonal =
            CombinedSpatialRow(sub_source_mass, sub_source_stiffness,
                               spatial_row, sub_mass_weight, sub_k_weight);
        for (const auto& entry : subdiagonal) {
          const PetscInt column = OwnershipPermutedSpaceTimeIndex(
              entry.first, slot - 1u, steps, spatial_ranges, ranks);
          if (column >= expected_begin && column < expected_end)
            ++diagonal_nnz[local_row];
          else
            ++offdiagonal_nnz[local_row];
        }
        continue;
      }
      PetscInt count = 0;
      const PetscInt* columns = nullptr;
      const PetscScalar* entries = nullptr;
      PetscCallAbort(PETSC_COMM_WORLD,
                     MatGetRow(mass, spatial_row, &count, &columns, &entries));
      for (PetscInt i = 0; i < count; ++i) {
        const PetscInt column = OwnershipPermutedSpaceTimeIndex(
            columns[i], slot - 1u, steps, spatial_ranges, ranks);
        if (column >= expected_begin && column < expected_end)
          ++diagonal_nnz[local_row];
        else
          ++offdiagonal_nnz[local_row];
      }
      PetscCallAbort(PETSC_COMM_WORLD,
                     MatRestoreRow(mass, spatial_row, &count, &columns, &entries));
    }
  }

  PetscCallAbort(PETSC_COMM_WORLD,
                 MatCreateAIJ(PETSC_COMM_WORLD, local_size, local_size,
                              global_size, global_size, 0,
                              diagonal_nnz.data(), 0, offdiagonal_nnz.data(),
                              &operators.ay));
  PetscInt ay_begin = 0, ay_end = 0;
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatGetOwnershipRange(operators.ay, &ay_begin, &ay_end));
  if (ay_begin != expected_begin || ay_end != expected_end)
    throw std::runtime_error("assembled space-time ownership mismatch");
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatSetOption(operators.ay, MAT_NEW_NONZERO_ALLOCATION_ERR,
                              PETSC_TRUE));
  for (unsigned slot = 0; slot < steps; ++slot) {
    for (PetscInt spatial_row = spatial_begin; spatial_row < spatial_end;
         ++spatial_row) {
      const PetscInt row = expected_begin +
          static_cast<PetscInt>(slot) * spatial_local +
          spatial_row - spatial_begin;
      const std::unordered_map<PetscInt, PetscScalar> diagonal =
          CombinedSpatialRow(diagonal_source_mass, diagonal_source_stiffness,
                             spatial_row, diagonal_mass_weight,
                             diagonal_k_weight);
      std::vector<PetscInt> columns;
      std::vector<PetscScalar> values;
      columns.reserve(diagonal.size());
      values.reserve(diagonal.size());
      for (const auto& entry : diagonal) {
        columns.push_back(OwnershipPermutedSpaceTimeIndex(
            entry.first, slot, steps, spatial_ranges, ranks));
        values.push_back(entry.second);
      }
      PetscCallAbort(PETSC_COMM_WORLD,
                     MatSetValues(operators.ay, 1, &row, columns.size(),
                                  columns.data(), values.data(), INSERT_VALUES));
      if (slot == 0u) continue;
      if (crank_nicolson) {
        const std::unordered_map<PetscInt, PetscScalar> subdiagonal =
            CombinedSpatialRow(sub_source_mass, sub_source_stiffness,
                               spatial_row, sub_mass_weight, sub_k_weight);
        columns.clear();
        values.clear();
        columns.reserve(subdiagonal.size());
        values.reserve(subdiagonal.size());
        for (const auto& entry : subdiagonal) {
          columns.push_back(OwnershipPermutedSpaceTimeIndex(
              entry.first, slot - 1u, steps, spatial_ranges, ranks));
          values.push_back(entry.second);
        }
        PetscCallAbort(PETSC_COMM_WORLD,
                       MatSetValues(operators.ay, 1, &row, columns.size(),
                                    columns.data(), values.data(),
                                    INSERT_VALUES));
        continue;
      }
      PetscInt count = 0;
      const PetscInt* mass_columns = nullptr;
      const PetscScalar* mass_values = nullptr;
      PetscCallAbort(PETSC_COMM_WORLD,
                     MatGetRow(mass, spatial_row, &count, &mass_columns,
                               &mass_values));
      columns.resize(count);
      values.resize(count);
      for (PetscInt i = 0; i < count; ++i) {
        columns[i] = OwnershipPermutedSpaceTimeIndex(
            mass_columns[i], slot - 1u, steps, spatial_ranges, ranks);
        values[i] = -inverse_dt * mass_values[i];
      }
      PetscCallAbort(PETSC_COMM_WORLD,
                     MatSetValues(operators.ay, 1, &row, count, columns.data(),
                                  values.data(), INSERT_VALUES));
      PetscCallAbort(PETSC_COMM_WORLD,
                     MatRestoreRow(mass, spatial_row, &count, &mass_columns,
                                   &mass_values));
    }
  }
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatAssemblyBegin(operators.ay, MAT_FINAL_ASSEMBLY));
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatAssemblyEnd(operators.ay, MAT_FINAL_ASSEMBLY));
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatTranspose(operators.ay, MAT_INITIAL_MATRIX, &operators.ap));
}

void DestroyAssembledSpaceTimeOperators(
    AssembledSpaceTimeOperators& operators) {
  PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&operators.ap));
  PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&operators.ay));
}


void CopyDataSideSliceToSpatial(Vec space_time, unsigned slot,
                                PetscInt spatial_local, Vec spatial) {
  const PetscScalar* source = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(space_time, &source));
  const PetscInt offset = static_cast<PetscInt>(slot) * spatial_local;
  PetscCallAbort(PETSC_COMM_WORLD,
                 CopyLocalSliceToSpatial(source, offset, spatial_local, spatial));
  PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(space_time, &source));
}

void CopySpatialToDataSideSlice(Vec spatial, unsigned slot,
                                PetscInt spatial_local, double scale,
                                Vec space_time, bool add) {
  PetscScalar* destination = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, VecGetArray(space_time, &destination));
  const PetscInt offset = static_cast<PetscInt>(slot) * spatial_local;
  if (add) {
    PetscCallAbort(PETSC_COMM_WORLD,
                   AddSpatialToLocalSlice(spatial, scale, offset, spatial_local,
                                          destination));
  } else {
    // Insert has no counterpart in the shell callback, which only accumulates.
    const PetscScalar* source = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(spatial, &source));
    for (PetscInt i = 0; i < spatial_local; ++i)
      destination[offset + i] = scale * source[i];
    PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(spatial, &source));
  }
  PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArray(space_time, &destination));
}

// ---------------------------------------------------------------------------
// The space-time operator owner
// ---------------------------------------------------------------------------

// `distributed` is built IN PLACE: its four MatShell operators store raw
// pointers to the contexts inside it, so the contexts are initialised at the
// member's own address and the object is neither copied nor moved afterwards.
SpaceTimeOperators::SpaceTimeOperators(const ProblemConfig& config,
                                       const SpatialOperators& spatial) {
  // Crank-Nicolson with Dirichlet control: build A_plus and A_minus once,
  // re-embedding the eliminated boundary rows.  The hatted operators carry the
  // strong elimination as K_I = 1 and M_I = 0 on the C diagonal; halving K_I
  // would put 1/2 where the diagonal block needs the identity and where the
  // neighbour block needs zero, so both are re-embedded explicitly here and
  // the shell and the assembled twin then use them directly.
  if (IsCrankNicolson(config) && spatial.dirichlet) {
    const std::vector<PetscInt>& owned =
        spatial.dirichlet->owned_control_dofs();
    const PetscInt count = static_cast<PetscInt>(owned.size());
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatDuplicate(spatial.stiffness, MAT_COPY_VALUES, &cn_diagonal));
    PetscCallAbort(PETSC_COMM_WORLD, MatScale(cn_diagonal, 0.5));
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatAXPY(cn_diagonal, 1.0 / config.time.dt, spatial.mass,
                           DIFFERENT_NONZERO_PATTERN));
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatDuplicate(spatial.stiffness, MAT_COPY_VALUES, &cn_neighbour));
    PetscCallAbort(PETSC_COMM_WORLD, MatScale(cn_neighbour, 0.5));
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatAXPY(cn_neighbour, -1.0 / config.time.dt, spatial.mass,
                           DIFFERENT_NONZERO_PATTERN));
    for (Mat m : {cn_diagonal, cn_neighbour})
      PetscCallAbort(PETSC_COMM_WORLD,
                     MatSetOption(m, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE));
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatZeroRowsColumns(cn_diagonal, count,
                                      owned.empty() ? nullptr : owned.data(),
                                      1.0, nullptr, nullptr));
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatZeroRowsColumns(cn_neighbour, count,
                                      owned.empty() ? nullptr : owned.data(),
                                      0.0, nullptr, nullptr));
  }
  BuildDistributedSpaceTimeOperators(distributed, config, spatial.mass,
                                     spatial.stiffness, spatial.tracking,
                                     spatial.control, spatial.dirichlet.get(),
                                     cn_diagonal, cn_neighbour);
  BuildAssembledSpaceTimeOperators(assembled, spatial.mass, spatial.stiffness,
                                   config.time.steps, config.time.dt,
                                   config.time.scheme, cn_diagonal,
                                   cn_neighbour);
  elapsed = MPI_Wtime() - spatial.wall_start;
}

SpaceTimeOperators::~SpaceTimeOperators() {
  DestroyAssembledSpaceTimeOperators(assembled);
  DestroyDistributedSpaceTimeOperators(distributed);
  PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&cn_neighbour));
  PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&cn_diagonal));
}
