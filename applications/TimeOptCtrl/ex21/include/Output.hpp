// Output layer: gather the distributed solution to rank 0, write the JSON field
// dump and the ParaView/VTK series, emit the two stdout summary records, and
// run the whole output phase of a production solve (WriteSolution).
//
// The EX20_PAR_SOLVE and EX20_SOLVERSTATS records and the JSON writer are a
// PARSED CONTRACT: external harnesses read them field by field and byte-compare
// the JSON at setprecision(17).  Do not reformat, reorder or rename a field.

#pragma once

#include "MultiLevelSolution.hpp"

#include <petscmat.h>

#include <string>
#include <vector>

#include "Config.hpp"
#include "Discretization.hpp"
#include "Solver.hpp"

// ---------------------------------------------------------------------------
// Rank-0 gathered fields
// ---------------------------------------------------------------------------

struct RootDistributedFields {
  std::vector<std::vector<double> > coordinates;
  std::vector<double> state;
  std::vector<double> adjoint;
  std::vector<double> control;
};

RootDistributedFields GatherDistributedSolveToRoot(
    const DistributedOptimalControlSolve& solve,
    const OwnedSpatialLayout& layout, Mat spatial_matrix, unsigned steps);

// ---------------------------------------------------------------------------
// File output
// ---------------------------------------------------------------------------

void WriteParallelSolveJsonRankZero(
    const std::string& path, const ProblemConfig& config,
    const DistributedOptimalControlSolve& solve,
    const RootDistributedFields& fields);

// ---------------------------------------------------------------------------
// Machine-readable stdout records.  Field names, order, spacing and precision
// are a parsed contract.
// ---------------------------------------------------------------------------

void PrintSolverStats(const ProblemConfig& config,
                      const DistributedOptimalControlSolve& solve,
                      DistributedSolvePc pc_mode);

// ---------------------------------------------------------------------------
// The whole output phase of a production solve
// ---------------------------------------------------------------------------

struct ApplicationOptions;

/** Gather to rank 0 and emit the JSON field dump, the ParaView series and the
 *  two machine-readable stdout records. */
void WriteSolution(DistributedOptimalControlSolve& solve,
                   const ProblemConfig& config,
                   const ApplicationOptions& options,
                   femus::MultiLevelSolution& solution, femus::Mesh* mesh,
                   const SpatialOperators& spatial);
