// Verification battery.  Nothing here is reachable from the production solve
// path; the only exposed symbols are the three drivers declared in the header.

#include "Verification.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "MultiLevelSolution.hpp"
#include "PetscMatrix.hpp"

#include <petscksp.h>

#include "BoundaryReducedSchur.hpp"
#include "Config.hpp"
#include "Discretization.hpp"
#include "Preconditioners.hpp"
#include "SpaceTimeOperators.hpp"
#include "Solver.hpp"

using namespace femus;

/** Fill v with a deterministic pseudo-random vector that is independent of the
 *  MPI rank count: the same global sequence is generated on every rank and each
 *  rank keeps the entries it owns. */
void FillDeterministicRandomVec(Vec v, unsigned seed) {
  PetscInt begin = 0, end = 0, global = 0;
  PetscCallAbort(PETSC_COMM_WORLD, VecGetOwnershipRange(v, &begin, &end));
  PetscCallAbort(PETSC_COMM_WORLD, VecGetSize(v, &global));
  std::mt19937 generator(seed);
  std::uniform_real_distribution<double> distribution(-1.0, 1.0);
  PetscScalar* values = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, VecGetArray(v, &values));
  for (PetscInt g = 0; g < global; ++g) {
    const double sample = distribution(generator);
    if (g >= begin && g < end) values[g - begin] = sample;
  }
  PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArray(v, &values));
}

/** Marker of the structural row support of a spatial matrix, over the locally
 *  owned rows only.  1 == the row carries at least one stored entry. */
std::vector<unsigned char> SpatialRowSupport(
    Mat spatial, const OwnedSpatialLayout& layout) {
  std::vector<unsigned char> support(layout.local_size, 0u);
  for (PetscInt row = layout.begin; row < layout.end; ++row) {
    PetscInt count = 0;
    const PetscInt* columns = nullptr;
    const PetscScalar* entries = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatGetRow(spatial, row, &count, &columns, &entries));
    bool nonzero = false;
    for (PetscInt i = 0; i < count; ++i)
      if (entries[i] != 0.0) nonzero = true;
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatRestoreRow(spatial, row, &count, &columns, &entries));
    support[row - layout.begin] = nonzero ? 1u : 0u;
  }
  return support;
}

/** Space-time vector holding the interpolated tracking target y_d at the N
 *  discrete times t^k = k*dt, k = 1..N: the same field
 *  BuildDistributedRightHandSide samples internally without returning it. */
// ===========================================================================
// Algebraic test battery, opt-in via --verify-stage1.
// ===========================================================================

struct VerificationLog {
  std::ofstream file;
  bool root = false;
  void operator()(const std::ostringstream& text) {
    if (!root) return;
    std::cout << text.str() << std::endl;
    if (file.is_open()) file << text.str() << '\n';
  }
};

void CreateDirectFactorSolver(Mat matrix, const char* prefix, KSP* solver) {
  PetscCallAbort(PETSC_COMM_WORLD, KSPCreate(PETSC_COMM_WORLD, solver));
  PetscCallAbort(PETSC_COMM_WORLD, KSPSetOptionsPrefix(*solver, prefix));
  PetscCallAbort(PETSC_COMM_WORLD, KSPSetOperators(*solver, matrix, matrix));
  PetscCallAbort(PETSC_COMM_WORLD, KSPSetType(*solver, KSPPREONLY));
  PC preconditioner = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, KSPGetPC(*solver, &preconditioner));
  PetscCallAbort(PETSC_COMM_WORLD, PCSetType(preconditioner, PCLU));
  // The symmetric arrangement has a singular (1,1) block alpha_t*M_omega whose
  // rows outside the tracking region are structurally empty, so PETSc's own
  // SeqAIJ LU (which requires a diagonal entry) cannot factor it.  MUMPS also
  // fails on it (KSP_DIVERGED_PC_FAILED); UMFPACK factors it to 3e-16.
  // Overridable at run time through the KSP prefixes vay_/vleft_/vright_/
  // vfull_/vsym_ using the PETSC_OPTIONS environment variable.
  PetscCallAbort(PETSC_COMM_WORLD,
                 PCFactorSetMatSolverType(preconditioner, MATSOLVERUMFPACK));
  PetscCallAbort(PETSC_COMM_WORLD, KSPSetFromOptions(*solver));
  PetscCallAbort(PETSC_COMM_WORLD, KSPSetUp(*solver));
}

/** ||a - b||_2 and the relative version normalised by max(||a||,||b||,1). */
void CompareVectors(Vec a, Vec b, Vec work, double* absolute,
                    double* relative) {
  double na = 0.0, nb = 0.0;
  PetscCallAbort(PETSC_COMM_WORLD, VecNorm(a, NORM_2, &na));
  PetscCallAbort(PETSC_COMM_WORLD, VecNorm(b, NORM_2, &nb));
  PetscCallAbort(PETSC_COMM_WORLD, VecCopy(a, work));
  PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(work, -1.0, b));
  PetscCallAbort(PETSC_COMM_WORLD, VecNorm(work, NORM_2, absolute));
  *relative = *absolute / std::max(1.0, std::max(na, nb));
}

void DumpDenseOperator(const std::string& path, Mat layout_matrix,
                       const std::function<void(Vec, Vec)>& apply) {
  PetscInt size = 0;
  PetscCallAbort(PETSC_COMM_WORLD, MatGetSize(layout_matrix, &size, nullptr));
  Vec basis = nullptr, image = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatCreateVecs(layout_matrix, &basis, &image));
  std::vector<double> dense(static_cast<size_t>(size) * size, 0.0);
  for (PetscInt j = 0; j < size; ++j) {
    PetscCallAbort(PETSC_COMM_WORLD, VecSet(basis, 0.0));
    PetscCallAbort(PETSC_COMM_WORLD, VecSetValue(basis, j, 1.0, INSERT_VALUES));
    PetscCallAbort(PETSC_COMM_WORLD, VecAssemblyBegin(basis));
    PetscCallAbort(PETSC_COMM_WORLD, VecAssemblyEnd(basis));
    apply(basis, image);
    const PetscScalar* values = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(image, &values));
    for (PetscInt i = 0; i < size; ++i)
      dense[static_cast<size_t>(i) * size + j] = PetscRealPart(values[i]);
    PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(image, &values));
  }
  std::ofstream stream(path.c_str());
  stream << std::setprecision(17);
  for (PetscInt i = 0; i < size; ++i) {
    for (PetscInt j = 0; j < size; ++j)
      stream << (j ? " " : "") << dense[static_cast<size_t>(i) * size + j];
    stream << '\n';
  }
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&image));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&basis));
}

void CopySpaceTimeBlockIntoMonolithic(Mat source, Mat destination,
                                      PetscInt row_offset,
                                      PetscInt col_offset, double scale,
                                      PetscInt block) {
  for (PetscInt r = 0; r < block; ++r) {
    PetscInt count = 0;
    const PetscInt* columns = nullptr;
    const PetscScalar* entries = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatGetRow(source, r, &count, &columns, &entries));
    std::vector<PetscInt> target_columns(count);
    std::vector<PetscScalar> values(count);
    for (PetscInt i = 0; i < count; ++i) {
      target_columns[i] = columns[i] + col_offset;
      values[i] = scale * entries[i];
    }
    const PetscInt row = r + row_offset;
    if (count)
      PetscCallAbort(PETSC_COMM_WORLD,
                     MatSetValues(destination, 1, &row, count,
                                  target_columns.data(), values.data(),
                                  ADD_VALUES));
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatRestoreRow(source, r, &count, &columns, &entries));
  }
}

void CopySpatialBlockDiagonalIntoMonolithic(Mat spatial, Mat destination,
                                            PetscInt row_offset,
                                            PetscInt col_offset, double scale,
                                            unsigned steps,
                                            PetscInt spatial_n) {
  for (unsigned slot = 0; slot < steps; ++slot) {
    const PetscInt shift = static_cast<PetscInt>(slot) * spatial_n;
    for (PetscInt s = 0; s < spatial_n; ++s) {
      PetscInt count = 0;
      const PetscInt* columns = nullptr;
      const PetscScalar* entries = nullptr;
      PetscCallAbort(PETSC_COMM_WORLD,
                     MatGetRow(spatial, s, &count, &columns, &entries));
      std::vector<PetscInt> target_columns(count);
      std::vector<PetscScalar> values(count);
      for (PetscInt i = 0; i < count; ++i) {
        target_columns[i] = columns[i] + shift + col_offset;
        values[i] = scale * entries[i];
      }
      const PetscInt row = s + shift + row_offset;
      if (count)
        PetscCallAbort(PETSC_COMM_WORLD,
                       MatSetValues(destination, 1, &row, count,
                                    target_columns.data(), values.data(),
                                    ADD_VALUES));
      PetscCallAbort(PETSC_COMM_WORLD,
                     MatRestoreRow(spatial, s, &count, &columns, &entries));
    }
  }
}

/** TEST 7: the complete 2x2 system as one explicit serial AIJ matrix.
 *  symmetric == false gives the production arrangement
 *      [ E , delta*N ; -alpha_t*M_omega , E^T ]
 *  symmetric == true gives the equivalent symmetric indefinite arrangement
 *      [ alpha_t*M_omega , E^T ; E , -delta*N ]   (unknowns (y, phi = -p)). */
Mat AssembleMonolithicSerial(const ProblemConfig& config,
                             const AssembledSpaceTimeOperators& assembled,
                             Mat control, Mat tracking, unsigned steps,
                             PetscInt spatial_n, bool symmetric) {
  const PetscInt block = static_cast<PetscInt>(steps) * spatial_n;
  const double delta = config.eta * config.eta / config.alpha_u;
  Mat full = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, MatCreate(PETSC_COMM_WORLD, &full));
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatSetSizes(full, 2 * block, 2 * block, 2 * block, 2 * block));
  PetscCallAbort(PETSC_COMM_WORLD, MatSetType(full, MATAIJ));
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatSeqAIJSetPreallocation(full, 200, nullptr));
  if (!symmetric) {
    CopySpaceTimeBlockIntoMonolithic(assembled.ay, full, 0, 0, 1.0, block);
    CopySpatialBlockDiagonalIntoMonolithic(control, full, 0, block, delta,
                                           steps, spatial_n);
    CopySpatialBlockDiagonalIntoMonolithic(tracking, full, block, 0,
                                           -config.alpha_track, steps,
                                           spatial_n);
    CopySpaceTimeBlockIntoMonolithic(assembled.ap, full, block, block, 1.0,
                                     block);
  } else {
    CopySpatialBlockDiagonalIntoMonolithic(tracking, full, 0, 0,
                                           config.alpha_track, steps,
                                           spatial_n);
    CopySpaceTimeBlockIntoMonolithic(assembled.ap, full, 0, block, 1.0, block);
    CopySpaceTimeBlockIntoMonolithic(assembled.ay, full, block, 0, 1.0, block);
    CopySpatialBlockDiagonalIntoMonolithic(control, full, block, block, -delta,
                                           steps, spatial_n);
  }
  PetscCallAbort(PETSC_COMM_WORLD, MatAssemblyBegin(full, MAT_FINAL_ASSEMBLY));
  PetscCallAbort(PETSC_COMM_WORLD, MatAssemblyEnd(full, MAT_FINAL_ASSEMBLY));
  return full;
}

void ConcatenateSerial(Vec first, Vec second, Vec composite) {
  PetscInt block = 0;
  PetscCallAbort(PETSC_COMM_WORLD, VecGetLocalSize(first, &block));
  const PetscScalar* a = nullptr;
  const PetscScalar* b = nullptr;
  PetscScalar* c = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(first, &a));
  PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(second, &b));
  PetscCallAbort(PETSC_COMM_WORLD, VecGetArray(composite, &c));
  for (PetscInt i = 0; i < block; ++i) {
    c[i] = a[i];
    c[block + i] = b[i];
  }
  PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArray(composite, &c));
  PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(second, &b));
  PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(first, &a));
}

void SplitSerial(Vec composite, Vec first, Vec second) {
  PetscInt block = 0;
  PetscCallAbort(PETSC_COMM_WORLD, VecGetLocalSize(first, &block));
  const PetscScalar* c = nullptr;
  PetscScalar* a = nullptr;
  PetscScalar* b = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(composite, &c));
  PetscCallAbort(PETSC_COMM_WORLD, VecGetArray(first, &a));
  PetscCallAbort(PETSC_COMM_WORLD, VecGetArray(second, &b));
  for (PetscInt i = 0; i < block; ++i) {
    a[i] = c[i];
    b[i] = c[block + i];
  }
  PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArray(second, &b));
  PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArray(first, &a));
  PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(composite, &c));
}

// ===========================================================================
// Spatial operator export, opt-in via --dump-operators.
// Writes the spatial operators and the DOF coordinates in the PETSc row
// ordering used by the JSON field dump, so an external driver can compare them
// entry by entry against an independent assembly.  Serial only.
// ===========================================================================
void DumpSpatialOperatorsSerial(const std::string& prefix,
                                const NativeSpatialOperators& spatial,
                                Mat control_selected,
                                const OwnedSpatialLayout& layout) {
  int ranks = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  if (ranks != 1)
    throw std::runtime_error("--dump-operators requires a single MPI rank");

  // The dump keeps its historical six logical names.  "M_ctrl_whole" -- the
  // mass matrix over the whole domain -- is the physical M and is written
  // from it under that name, so downstream readers see the same file set.
  const struct {
    const char* label;
    femus::SparseMatrix* matrix;
  } operators[] = {{"M", spatial.M.get()},
                   {"K", spatial.K.get()},
                   {"M_track", spatial.M_track.get()},
                   {"M_ctrl_whole", spatial.M.get()},
                   {"M_ctrl_box", spatial.M_ctrl_box.get()},
                   {"M_Gamma", spatial.M_Gamma.get()}};
  for (const auto& op : operators) {
    Mat matrix = static_cast<PetscMatrix*>(op.matrix)->mat();
    const std::string path = prefix + "_" + op.label + ".coo";
    std::ofstream stream(path.c_str());
    if (!stream)
      throw std::runtime_error("cannot open operator dump: " + path);
    PetscInt rows = 0, columns = 0;
    PetscCallAbort(PETSC_COMM_WORLD, MatGetSize(matrix, &rows, &columns));
    stream << std::setprecision(17) << "# rows cols\n"
           << rows << ' ' << columns << '\n' << "# row col value\n";
    for (PetscInt r = 0; r < rows; ++r) {
      PetscInt count = 0;
      const PetscInt* cols = nullptr;
      const PetscScalar* vals = nullptr;
      PetscCallAbort(PETSC_COMM_WORLD,
                     MatGetRow(matrix, r, &count, &cols, &vals));
      for (PetscInt i = 0; i < count; ++i)
        stream << r << ' ' << cols[i] << ' ' << PetscRealPart(vals[i]) << '\n';
      PetscCallAbort(PETSC_COMM_WORLD,
                     MatRestoreRow(matrix, r, &count, &cols, &vals));
    }
  }
  // The operator actually selected as N for this run.
  {
    const std::string path = prefix + "_N_selected.coo";
    std::ofstream stream(path.c_str());
    PetscInt rows = 0;
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatGetSize(control_selected, &rows, nullptr));
    stream << std::setprecision(17) << "# rows cols\n"
           << rows << ' ' << rows << '\n' << "# row col value\n";
    for (PetscInt r = 0; r < rows; ++r) {
      PetscInt count = 0;
      const PetscInt* cols = nullptr;
      const PetscScalar* vals = nullptr;
      PetscCallAbort(PETSC_COMM_WORLD,
                     MatGetRow(control_selected, r, &count, &cols, &vals));
      for (PetscInt i = 0; i < count; ++i)
        stream << r << ' ' << cols[i] << ' ' << PetscRealPart(vals[i]) << '\n';
      PetscCallAbort(PETSC_COMM_WORLD,
                     MatRestoreRow(control_selected, r, &count, &cols, &vals));
    }
  }
  {
    const std::string path = prefix + "_coords.txt";
    std::ofstream stream(path.c_str());
    stream << std::setprecision(17) << "# x z\n";
    for (PetscInt i = 0; i < layout.local_size; ++i) {
      stream << layout.x[i] << ' ' << layout.z[i];
      if (layout.three_d) stream << ' ' << layout.axis2[i];   // T9: 3D
      stream << '\n';
    }
  }
  std::cout << "EX20_OPERATOR_DUMP prefix=" << prefix
            << " n_spatial=" << layout.local_size << std::endl;
}

int RunSpatialOperatorDump(const ProblemConfig& config,
                           MultiLevelSolution& solution, Mesh* mesh,
                           unsigned solution_type,
                           const std::string& prefix) {
  SpatialOperators spatial(config, mesh, solution_type, solution);
  DumpSpatialOperatorsSerial(prefix, spatial.assembled, spatial.control,
                             spatial.layout);
  // T7F: the source-region mass, only when source regions exist (so every
  // pre-existing dump keeps its file set).
  if (!config.source_regions.empty()) {
    Mat matrix = static_cast<PetscMatrix*>(spatial.assembled.M_source.get())->mat();
    PetscInt rows = 0;
    PetscCallAbort(PETSC_COMM_WORLD, MatGetSize(matrix, &rows, nullptr));
    std::ofstream stream((prefix + "_M_source.coo").c_str());
    stream << std::setprecision(17) << "# rows cols\n"
           << rows << ' ' << rows << '\n' << "# row col value\n";
    for (PetscInt r = 0; r < rows; ++r) {
      PetscInt count = 0;
      const PetscInt* cols = nullptr;
      const PetscScalar* vals = nullptr;
      PetscCallAbort(PETSC_COMM_WORLD, MatGetRow(matrix, r, &count, &cols, &vals));
      for (PetscInt i = 0; i < count; ++i)
        stream << r << ' ' << cols[i] << ' ' << PetscRealPart(vals[i]) << '\n';
      PetscCallAbort(PETSC_COMM_WORLD, MatRestoreRow(matrix, r, &count, &cols, &vals));
    }
  }
  // T10: 3D only -- per element the centroid and the tracking / control /
  // source region indices (lowest matching box, 0 = none), so the centroid
  // rule and the precedence can be checked independently of the assembly.
  if (config.dimension == 3u) {
    std::ofstream stream((prefix + "_regions3d.txt").c_str());
    stream << std::setprecision(17)
           << "# element xc yc zc tracking_index control_index source_index\n";
    const unsigned rank = mesh->processor_id();
    for (unsigned element = mesh->_elementOffset[rank];
         element < mesh->_elementOffset[rank + 1u]; ++element) {
      double c[3];
      ElementCentroid3DOf(mesh, element, c);
      stream << element << ' ' << c[0] << ' ' << c[1] << ' ' << c[2] << ' '
             << TrackingRegionIndex(config, mesh, element) << ' '
             << ControlRegionIndex(config, mesh, element) << ' '
             << SourceRegionIndex3D(config, mesh, element) << '\n';
    }
  }
  // T11D: 3D Dirichlet control (serial) -- the unhatted M and L (C kept, D
  // eliminated), the controlled set C, and B_D / B_D^T EXPLICITLY, column by
  // column from DirichletControlOperators::ApplyBD / ApplyBDT on unit
  // space-time vectors, so that the BE/CN formulas and the transpose can be
  // checked independently (t11d_bd_transpose.py).  Space-time index = n N + i.
  if (config.dimension == 3u && config.control.type == ProblemConfig::ControlType::DIRICHLET &&
      spatial.dirichlet) {
    DirichletControlOperators& d = *spatial.dirichlet;
    auto write_mat = [&](Mat m, const std::string& path) {
      PetscInt rows = 0, cols = 0;
      PetscCallAbort(PETSC_COMM_WORLD, MatGetSize(m, &rows, &cols));
      std::ofstream out(path.c_str());
      out << std::setprecision(17) << "# rows cols\n" << rows << ' ' << cols << "\n# row col value\n";
      for (PetscInt r = 0; r < rows; ++r) {
        PetscInt count = 0;
        const PetscInt* c = nullptr;
        const PetscScalar* v = nullptr;
        PetscCallAbort(PETSC_COMM_WORLD, MatGetRow(m, r, &count, &c, &v));
        for (PetscInt k = 0; k < count; ++k)
          if (PetscRealPart(v[k]) != 0.0) out << r << ' ' << c[k] << ' ' << PetscRealPart(v[k]) << '\n';
        PetscCallAbort(PETSC_COMM_WORLD, MatRestoreRow(m, r, &count, &c, &v));
      }
    };
    write_mat(d.unhatted_mass(), prefix + "_D_Mfull.coo");
    write_mat(d.unhatted_stiffness(), prefix + "_D_Lfull.coo");
    {
      std::ofstream out((prefix + "_D_C.txt").c_str());
      out << "# controlled DOFs C (global), steps dt scheme transposes_L\n"
          << config.time.steps << ' ' << std::setprecision(17) << config.time.dt << ' '
          << (IsCrankNicolson(config) ? "CN" : "BE") << ' ' << (d.transposes_stiffness() ? 1 : 0) << '\n';
      for (PetscInt g : d.owned_control_dofs()) out << g << '\n';
    }
    const PetscInt N = spatial.layout.local_size, T = static_cast<PetscInt>(config.time.steps);
    Vec e = nullptr, out_v = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD, VecCreateSeq(PETSC_COMM_SELF, T * N, &e));
    PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(e, &out_v));
    for (int which = 0; which < 2; ++which) {   // 0: B_D (columns over C), 1: B_D^T (columns over all)
      std::ofstream out((prefix + (which == 0 ? "_BD.coo" : "_BDT.coo")).c_str());
      out << std::setprecision(17) << "# rows cols\n" << T * N << ' ' << T * N << "\n# row col value\n";
      for (PetscInt n = 0; n < T; ++n)
        for (PetscInt i = 0; i < N; ++i) {
          if (which == 0 && !d.control_mask()[static_cast<std::size_t>(i)]) continue;
          PetscCallAbort(PETSC_COMM_WORLD, VecSet(e, 0.0));
          PetscCallAbort(PETSC_COMM_WORLD, VecSetValue(e, n * N + i, 1.0, INSERT_VALUES));
          PetscCallAbort(PETSC_COMM_WORLD, VecAssemblyBegin(e));
          PetscCallAbort(PETSC_COMM_WORLD, VecAssemblyEnd(e));
          PetscCallAbort(PETSC_COMM_WORLD, which == 0 ? d.ApplyBD(e, out_v) : d.ApplyBDT(e, out_v));
          const PetscScalar* a = nullptr;
          PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(out_v, &a));
          for (PetscInt r = 0; r < T * N; ++r)
            if (PetscRealPart(a[r]) != 0.0) out << r << ' ' << n * N + i << ' ' << PetscRealPart(a[r]) << '\n';
          PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(out_v, &a));
        }
    }
    PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&e));
    PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&out_v));
  }
  return 0;
}

// ===========================================================================
// Algebraic verification of the triangular time sweeps.  Each sweep is checked
// against (a) the residual of the assembled space-time operator it claims to
// invert and (b) a direct factorisation of that same operator.  Serial: the
// direct reference needs a serial LU.
// ===========================================================================
int RunSweepVerification(const ProblemConfig& config,
                         MultiLevelSolution& solution, Mesh* mesh,
                         unsigned solution_type, double spatial_tolerance) {
  int rank = 0, ranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);

  SpatialOperators spatial(config, mesh, solution_type, solution);
  SpaceTimeOperators spacetime(config, spatial);
  Mat mass = spatial.mass;
  Mat stiffness = spatial.stiffness;
  Mat tracking = spatial.tracking;
  Mat control = spatial.control;
  const OwnedSpatialLayout& layout = spatial.layout;
  DistributedSpaceTimeOperators& operators = spacetime.distributed;
  AssembledSpaceTimeOperators& assembled = spacetime.assembled;
  DistributedMatchingAuxiliaryMatrices matching =
      BuildDistributedMatchingAuxiliaryMatrices(config, tracking, control,
                                                assembled);
  SpatialSweepMatrices sweep_matrices =
      BuildSpatialSweepMatrices(config, mass, stiffness, control, tracking);

  struct Target {
    const char* name;
    Mat assembled_operator;
    Mat spatial_diagonal;
    TimeSweepDirection direction;
  };
  const Target targets[] = {
      {"E", assembled.ay, sweep_matrices.A, TimeSweepDirection::FORWARD},
      {"E^T", assembled.ap, sweep_matrices.A, TimeSweepDirection::BACKWARD},
      {"right", matching.right, sweep_matrices.A_right,
       TimeSweepDirection::FORWARD},
      {"left", matching.left, sweep_matrices.A_left,
       TimeSweepDirection::BACKWARD},
  };

  Vec v = nullptr, z = nullptr, zd = nullptr, w = nullptr, work = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, MatCreateVecs(assembled.ay, &v, nullptr));
  PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(v, &z));
  PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(v, &zd));
  PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(v, &w));
  PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(v, &work));

  if (rank == 0)
    std::cout << std::setprecision(6)
              << "S4_SWEEP_VERIFY config nx=" << config.mesh.nx
              << " N=" << config.time.steps << " alpha_u=" << config.alpha_u
              << " spatial_tol=" << spatial_tolerance
              << " control="
              << (config.control.type == ProblemConfig::ControlType::NEUMANN
                      ? "NEUMANN" : "DISTRIBUTED")
              << std::endl;

  for (const Target& t : targets) {
    TimeSweepContext ctx;
    InitializeTimeSweepContext(ctx, t.direction, t.spatial_diagonal, mass,
                               config.time.steps, config.time.dt,
                               spatial_tolerance, "verify_sweep_", t.name);
    PC pc = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD, PCCreate(PETSC_COMM_WORLD, &pc));
    PetscCallAbort(PETSC_COMM_WORLD, PCSetType(pc, PCSHELL));
    PetscCallAbort(PETSC_COMM_WORLD,
                   PCSetOperators(pc, t.assembled_operator, t.assembled_operator));
    PetscCallAbort(PETSC_COMM_WORLD, PCShellSetContext(pc, &ctx));
    PetscCallAbort(PETSC_COMM_WORLD, PCShellSetApply(pc, ApplyTimeSweep));
    PetscCallAbort(PETSC_COMM_WORLD, PCSetUp(pc));

    // The direct reference is a serial LU of the FULL space-time matrix; it is
    // only affordable for small systems.  Above the guard the residual test
    // still runs and vs_direct_rel is reported as -1 (not computed).
    PetscInt space_time_rows = 0;
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatGetSize(t.assembled_operator, &space_time_rows, nullptr));
    KSP direct = nullptr;
    if (ranks == 1 && space_time_rows <= 40000)
      CreateDirectFactorSolver(t.assembled_operator, "s4direct_", &direct);

    double max_residual = 0.0, max_direct = 0.0;
    for (unsigned trial = 0; trial < 4u; ++trial) {
      FillDeterministicRandomVec(v, 606000u + 101u * trial);
      PetscCallAbort(PETSC_COMM_WORLD, PCApply(pc, v, z));
      // residual of the assembled operator
      PetscCallAbort(PETSC_COMM_WORLD, MatMult(t.assembled_operator, z, w));
      PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(w, -1.0, v));
      double rn = 0.0, vn = 0.0;
      PetscCallAbort(PETSC_COMM_WORLD, VecNorm(w, NORM_2, &rn));
      PetscCallAbort(PETSC_COMM_WORLD, VecNorm(v, NORM_2, &vn));
      max_residual = std::max(max_residual, rn / std::max(vn, 1e-300));
      if (direct) {
        PetscCallAbort(PETSC_COMM_WORLD, KSPSolve(direct, v, zd));
        double dn = 0.0, zn = 0.0;
        PetscCallAbort(PETSC_COMM_WORLD, VecCopy(z, work));
        PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(work, -1.0, zd));
        PetscCallAbort(PETSC_COMM_WORLD, VecNorm(work, NORM_2, &dn));
        PetscCallAbort(PETSC_COMM_WORLD, VecNorm(zd, NORM_2, &zn));
        max_direct = std::max(max_direct, dn / std::max(zn, 1e-300));
      }
    }
    if (rank == 0)
      std::cout << std::setprecision(6) << "S4_SWEEP " << t.name
                << " residual_rel=" << max_residual
                << " vs_direct_rel=" << (direct ? max_direct : -1.0)
                << " spatial_solves=" << ctx.counter.solves
                << " spatial_iters=" << ctx.counter.iterations
                << " mean=" << ctx.counter.mean()
                << " max=" << ctx.counter.max_iterations
                << " failures=" << ctx.failures << std::endl;
    if (direct) PetscCallAbort(PETSC_COMM_WORLD, KSPDestroy(&direct));
    PetscCallAbort(PETSC_COMM_WORLD, PCDestroy(&pc));
    DestroyTimeSweepContext(ctx);
  }

  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&work));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&w));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&zd));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&z));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&v));
  DestroySpatialSweepMatrices(sweep_matrices);
  DestroyDistributedMatchingAuxiliaryMatrices(matching);

  return 0;
}

int RunVerificationStage1(const ProblemConfig& config,
                          MultiLevelSolution& solution, Mesh* mesh,
                          unsigned solution_type,
                          const std::string& output_prefix) {
  int rank = 0, ranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  VerificationLog log;
  log.root = (rank == 0);
  if (log.root) log.file.open((output_prefix + "_stage1.log").c_str());

  SpatialOperators spatial(config, mesh, solution_type, solution);
  SpaceTimeOperators spacetime(config, spatial);
  Mat mass = spatial.mass;
  Mat stiffness = spatial.stiffness;
  Mat tracking = spatial.tracking;
  Mat control = spatial.control;
  const OwnedSpatialLayout& layout = spatial.layout;
  DistributedSpaceTimeOperators& operators = spacetime.distributed;
  AssembledSpaceTimeOperators& assembled = spacetime.assembled;

  PetscInt spatial_global = 0, space_time_global = 0;
  PetscCallAbort(PETSC_COMM_WORLD, MatGetSize(mass, &spatial_global, nullptr));
  PetscCallAbort(PETSC_COMM_WORLD,
                 MatGetSize(operators.ay, &space_time_global, nullptr));

  {
    std::ostringstream s;
    s << std::setprecision(17) << "V0_CONFIG control="
      << (config.control.type == ProblemConfig::ControlType::NEUMANN
              ? "NEUMANN"
              : (ControlRegionIsWholeDomain(config) ? "DISTRIBUTED_WHOLE"
                                                    : "DISTRIBUTED_BOX"))
      << " nx=" << config.mesh.nx << " nz=" << config.mesh.nz
      << " steps=" << config.time.steps << " dt=" << config.time.dt
      << " alpha_u=" << config.alpha_u << " alpha_track=" << config.alpha_track
      << " eta=" << config.eta << " nu=" << config.diffusivity
      << " n_spatial=" << spatial_global << " n_spacetime=" << space_time_global
      << " ranks=" << ranks;
    log(s);
  }

  Vec v = nullptr, w = nullptr, a = nullptr, b = nullptr, work = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, MatCreateVecs(operators.ay, &v, nullptr));
  PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(v, &w));
  PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(v, &a));
  PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(v, &b));
  PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(v, &work));

  // ---------------------------------------------------------------------
  // TEST 1: matrix-free shell versus assembled AIJ, for E and E^T.
  // Identity checked: E_shell v == E_AIJ v for all v (same operator, two
  // independent implementations).  Expected tolerance: pure floating-point
  // summation-order difference, i.e. relative error <= a few 1e-16.
  // ---------------------------------------------------------------------
  double e_abs = 0.0, e_rel = 0.0, et_abs = 0.0, et_rel = 0.0;
  double transpose_abs = 0.0, transpose_rel = 0.0;
  for (unsigned trial = 0; trial < 8u; ++trial) {
    FillDeterministicRandomVec(v, 20260830u + 7919u * trial);
    double abs_error = 0.0, rel_error = 0.0;
    PetscCallAbort(PETSC_COMM_WORLD, MatMult(operators.ay, v, a));
    PetscCallAbort(PETSC_COMM_WORLD, MatMult(assembled.ay, v, b));
    CompareVectors(a, b, work, &abs_error, &rel_error);
    e_abs = std::max(e_abs, abs_error);
    e_rel = std::max(e_rel, rel_error);
    PetscCallAbort(PETSC_COMM_WORLD, MatMult(operators.ap, v, a));
    PetscCallAbort(PETSC_COMM_WORLD, MatMult(assembled.ap, v, b));
    CompareVectors(a, b, work, &abs_error, &rel_error);
    et_abs = std::max(et_abs, abs_error);
    et_rel = std::max(et_rel, rel_error);
    // E_AIJ^T == ET_AIJ, checked through the PETSc transpose product.
    PetscCallAbort(PETSC_COMM_WORLD, MatMultTranspose(assembled.ay, v, a));
    PetscCallAbort(PETSC_COMM_WORLD, MatMult(assembled.ap, v, b));
    CompareVectors(a, b, work, &abs_error, &rel_error);
    transpose_abs = std::max(transpose_abs, abs_error);
    transpose_rel = std::max(transpose_rel, rel_error);
  }
  {
    std::ostringstream s;
    s << std::setprecision(17)
      << "V1_RANDOM E_max_abs=" << e_abs << " E_max_rel=" << e_rel
      << " ET_max_abs=" << et_abs << " ET_max_rel=" << et_rel
      << " AIJtranspose_max_abs=" << transpose_abs
      << " AIJtranspose_max_rel=" << transpose_rel << " trials=8";
    log(s);
  }

  // Exhaustive canonical-basis comparison for small systems: this compares
  // every column of the two operators, not just their action on 8 vectors.
  if (space_time_global <= 1024) {
    double col_e_abs = 0.0, col_e_rel = 0.0, col_et_abs = 0.0,
           col_et_rel = 0.0;
    for (PetscInt j = 0; j < space_time_global; ++j) {
      PetscCallAbort(PETSC_COMM_WORLD, VecSet(v, 0.0));
      PetscCallAbort(PETSC_COMM_WORLD, VecSetValue(v, j, 1.0, INSERT_VALUES));
      PetscCallAbort(PETSC_COMM_WORLD, VecAssemblyBegin(v));
      PetscCallAbort(PETSC_COMM_WORLD, VecAssemblyEnd(v));
      double abs_error = 0.0, rel_error = 0.0;
      PetscCallAbort(PETSC_COMM_WORLD, MatMult(operators.ay, v, a));
      PetscCallAbort(PETSC_COMM_WORLD, MatMult(assembled.ay, v, b));
      CompareVectors(a, b, work, &abs_error, &rel_error);
      col_e_abs = std::max(col_e_abs, abs_error);
      col_e_rel = std::max(col_e_rel, rel_error);
      PetscCallAbort(PETSC_COMM_WORLD, MatMult(operators.ap, v, a));
      PetscCallAbort(PETSC_COMM_WORLD, MatMult(assembled.ap, v, b));
      CompareVectors(a, b, work, &abs_error, &rel_error);
      col_et_abs = std::max(col_et_abs, abs_error);
      col_et_rel = std::max(col_et_rel, rel_error);
    }
    std::ostringstream s;
    s << std::setprecision(17) << "V1_COLUMNS columns=" << space_time_global
      << " E_max_abs=" << col_e_abs << " E_max_rel=" << col_e_rel
      << " ET_max_abs=" << col_et_abs << " ET_max_rel=" << col_et_rel;
    log(s);
  }

  // ---------------------------------------------------------------------
  // TEST 2: block structure identities, all through bilinear forms because
  // the shells implement MATOP_MULT only.
  //   A22 == A11^T   <=>   <A11 x, y> == <x, A22 y>
  //   A12 == A12^T   <=>   <A12 x, y> == <A12 y, x>
  //   A21 == A21^T   <=>   <A21 x, y> == <A21 y, x>
  // Expected tolerance: cancellation of O(1) inner products, i.e. relative
  // error of order the machine epsilon times the condition of the sum,
  // realistically <= 1e-14.
  // ---------------------------------------------------------------------
  double sym_a22 = 0.0, sym_a12 = 0.0, sym_a21 = 0.0;
  for (unsigned trial = 0; trial < 8u; ++trial) {
    FillDeterministicRandomVec(v, 555000u + 31u * trial);
    FillDeterministicRandomVec(w, 777000u + 37u * trial);
    PetscScalar left_value = 0.0, right_value = 0.0;
    PetscCallAbort(PETSC_COMM_WORLD, MatMult(operators.ay, v, a));
    PetscCallAbort(PETSC_COMM_WORLD, VecDot(a, w, &left_value));
    PetscCallAbort(PETSC_COMM_WORLD, MatMult(operators.ap, w, b));
    PetscCallAbort(PETSC_COMM_WORLD, VecDot(v, b, &right_value));
    sym_a22 = std::max(sym_a22,
                       std::fabs(PetscRealPart(left_value - right_value)) /
                           std::max(1.0, std::fabs(PetscRealPart(left_value))));
    PetscCallAbort(PETSC_COMM_WORLD, MatMult(operators.ayp, v, a));
    PetscCallAbort(PETSC_COMM_WORLD, VecDot(a, w, &left_value));
    PetscCallAbort(PETSC_COMM_WORLD, MatMult(operators.ayp, w, b));
    PetscCallAbort(PETSC_COMM_WORLD, VecDot(b, v, &right_value));
    sym_a12 = std::max(sym_a12,
                       std::fabs(PetscRealPart(left_value - right_value)) /
                           std::max(1.0, std::fabs(PetscRealPart(left_value))));
    PetscCallAbort(PETSC_COMM_WORLD, MatMult(operators.apy, v, a));
    PetscCallAbort(PETSC_COMM_WORLD, VecDot(a, w, &left_value));
    PetscCallAbort(PETSC_COMM_WORLD, MatMult(operators.apy, w, b));
    PetscCallAbort(PETSC_COMM_WORLD, VecDot(b, v, &right_value));
    sym_a21 = std::max(sym_a21,
                       std::fabs(PetscRealPart(left_value - right_value)) /
                           std::max(1.0, std::fabs(PetscRealPart(left_value))));
  }
  {
    std::ostringstream s;
    s << std::setprecision(17) << "V2_SYMMETRY A22_vs_A11T_rel=" << sym_a22
      << " A12_sym_rel=" << sym_a12 << " A21_sym_rel=" << sym_a21;
    log(s);
  }

  // Equivalence of the production nonsymmetric arrangement with the symmetric
  // indefinite arrangement.  With phi = -p and delta = eta^2/alpha_u:
  //   symmetric row 1:  s1 = alpha_t M_omega y + E^T phi
  //   symmetric row 2:  s2 = E y - delta N phi
  //   production row 1: n1 = E y + delta N p       must equal  s2
  //   production row 2: n2 = -alpha_t M_omega y + E^T p  must equal  -s1
  // alpha_t*M_omega is obtained as -A21 and delta*N as A12, so no new
  // operator is introduced by this test.
  double equiv_row1 = 0.0, equiv_row2 = 0.0;
  Vec phi = nullptr, s1 = nullptr, s2 = nullptr, n1 = nullptr, n2 = nullptr;
  PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(v, &phi));
  PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(v, &s1));
  PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(v, &s2));
  PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(v, &n1));
  PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(v, &n2));
  for (unsigned trial = 0; trial < 4u; ++trial) {
    FillDeterministicRandomVec(v, 909000u + 11u * trial);   // y
    FillDeterministicRandomVec(w, 313000u + 13u * trial);   // p
    PetscCallAbort(PETSC_COMM_WORLD, VecCopy(w, phi));
    PetscCallAbort(PETSC_COMM_WORLD, VecScale(phi, -1.0));
    // s1 = alpha_t M_omega y + E^T phi  =  -(A21 y) + A22 phi
    PetscCallAbort(PETSC_COMM_WORLD, MatMult(operators.apy, v, a));
    PetscCallAbort(PETSC_COMM_WORLD, MatMult(operators.ap, phi, b));
    PetscCallAbort(PETSC_COMM_WORLD, VecCopy(b, s1));
    PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(s1, -1.0, a));
    // s2 = E y - delta N phi  =  A11 y - A12 phi
    PetscCallAbort(PETSC_COMM_WORLD, MatMult(operators.ay, v, a));
    PetscCallAbort(PETSC_COMM_WORLD, MatMult(operators.ayp, phi, b));
    PetscCallAbort(PETSC_COMM_WORLD, VecCopy(a, s2));
    PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(s2, -1.0, b));
    // n1 = A11 y + A12 p
    PetscCallAbort(PETSC_COMM_WORLD, MatMult(operators.ayp, w, b));
    PetscCallAbort(PETSC_COMM_WORLD, VecCopy(a, n1));
    PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(n1, 1.0, b));
    // n2 = A21 y + A22 p
    PetscCallAbort(PETSC_COMM_WORLD, MatMult(operators.apy, v, a));
    PetscCallAbort(PETSC_COMM_WORLD, MatMult(operators.ap, w, b));
    PetscCallAbort(PETSC_COMM_WORLD, VecCopy(a, n2));
    PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(n2, 1.0, b));
    double abs_error = 0.0, rel_error = 0.0;
    CompareVectors(n1, s2, work, &abs_error, &rel_error);
    equiv_row1 = std::max(equiv_row1, rel_error);
    PetscCallAbort(PETSC_COMM_WORLD, VecCopy(s1, a));
    PetscCallAbort(PETSC_COMM_WORLD, VecScale(a, -1.0));
    CompareVectors(n2, a, work, &abs_error, &rel_error);
    equiv_row2 = std::max(equiv_row2, rel_error);
  }
  {
    std::ostringstream s;
    s << std::setprecision(17)
      << "V2_SYMFORM row1_rel=" << equiv_row1 << " row2_rel=" << equiv_row2;
    log(s);
  }
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&n2));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&n1));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&s2));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&s1));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&phi));

  // ---------------------------------------------------------------------
  // TEST 6a: structural row support of N versus the nodal reconstruction mask.
  // ---------------------------------------------------------------------
  {
    // Cross-check the diagonal-based active set against the structural row
    // support of N and against the purely geometric mask.  active == support
    // is the correctness requirement; active vs mask sizes the disagreement.
    const std::vector<unsigned char> support =
        SpatialRowSupport(control, layout);
    const std::vector<unsigned char> active = ActiveControlDofs(control, layout);
    long long in_active = 0, in_support = 0, in_mask = 0;
    long long active_ne_support = 0, active_not_mask = 0, mask_not_active = 0;
    double sample_x = -1.0, sample_z = -1.0;
    for (PetscInt i = 0; i < layout.local_size; ++i) {
      const bool masked =
          OwnedDofIsControlled(config, layout.x[i], layout.z[i], layout.Axis2(i));
      if (active[i]) ++in_active;
      if (support[i]) ++in_support;
      if (masked) ++in_mask;
      if (static_cast<bool>(active[i]) != static_cast<bool>(support[i]))
        ++active_ne_support;
      if (active[i] && !masked) ++active_not_mask;
      if (!active[i] && masked) {
        ++mask_not_active;
        if (sample_x < 0.0) {
          sample_x = layout.x[i];
          sample_z = layout.z[i];
        }
      }
    }
    long long g[6] = {0, 0, 0, 0, 0, 0};
    long long l[6] = {in_active, in_support, in_mask,
                      active_ne_support, active_not_mask, mask_not_active};
    MPI_Allreduce(l, g, 6, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    std::ostringstream s;
    s << std::setprecision(17) << "V6_ACTIVE n_spatial=" << spatial_global
      << " active_dofs=" << g[0] << " support_rows=" << g[1]
      << " geometric_mask_dofs=" << g[2]
      << " active_ne_support=" << g[3]
      << " active_not_mask=" << g[4] << " mask_not_active=" << g[5]
      << " sample_mask_not_active=(" << sample_x << ',' << sample_z << ')';
    log(s);
  }

  int status = 0;

  // ---------------------------------------------------------------------
  // TESTS 4 and 7 need exact factorisations; restrict them to one MPI rank.
  // ---------------------------------------------------------------------
  if (ranks == 1) {
    DistributedMatchingAuxiliaryMatrices matching =
        BuildDistributedMatchingAuxiliaryMatrices(config, tracking, control,
                                                  assembled);
    KSP lu_right = nullptr, lu_left = nullptr, lu_ay = nullptr;
    CreateDirectFactorSolver(matching.right, "vright_", &lu_right);
    CreateDirectFactorSolver(matching.left, "vleft_", &lu_left);
    CreateDirectFactorSolver(assembled.ay, "vay_", &lu_ay);

    Vec t1 = nullptr, t2 = nullptr, out = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(v, &t1));
    PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(v, &t2));
    PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(v, &out));

    // Shat = left * E^{-1} * right   (the operator the shell MatMult claims).
    std::function<void(Vec, Vec)> exact_shat = [&](Vec in, Vec result) {
      PetscCallAbort(PETSC_COMM_WORLD, MatMult(matching.right, in, t1));
      PetscCallAbort(PETSC_COMM_WORLD, KSPSolve(lu_ay, t1, t2));
      PetscCallAbort(PETSC_COMM_WORLD, MatMult(matching.left, t2, result));
    };
    // The mirrored candidate right * E^{-1} * left.
    std::function<void(Vec, Vec)> exact_shat_mirror = [&](Vec in, Vec result) {
      PetscCallAbort(PETSC_COMM_WORLD, MatMult(matching.left, in, t1));
      PetscCallAbort(PETSC_COMM_WORLD, KSPSolve(lu_ay, t1, t2));
      PetscCallAbort(PETSC_COMM_WORLD, MatMult(matching.right, t2, result));
    };
    // Swapped factor order  left^{-1} E right^{-1}  (legacy_factor_order == true)
    std::function<void(Vec, Vec)> exact_pc_current = [&](Vec in, Vec result) {
      PetscCallAbort(PETSC_COMM_WORLD, KSPSolve(lu_right, in, t1));
      PetscCallAbort(PETSC_COMM_WORLD, MatMult(assembled.ay, t1, t2));
      PetscCallAbort(PETSC_COMM_WORLD, KSPSolve(lu_left, t2, result));
    };
    // Production factor order  right^{-1} E left^{-1}  ==  Shat^{-1}
    std::function<void(Vec, Vec)> exact_pc_intended = [&](Vec in, Vec result) {
      PetscCallAbort(PETSC_COMM_WORLD, KSPSolve(lu_left, in, t1));
      PetscCallAbort(PETSC_COMM_WORLD, MatMult(assembled.ay, t1, t2));
      PetscCallAbort(PETSC_COMM_WORLD, KSPSolve(lu_right, t2, result));
    };

    // TEST 4.3: is Shat(PC(v)) == v?  Only the true inverse gives ~0.
    double current_inverse = 0.0, intended_inverse = 0.0;
    Vec pc_result = nullptr, shat_result = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(v, &pc_result));
    PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(v, &shat_result));
    for (unsigned trial = 0; trial < 4u; ++trial) {
      FillDeterministicRandomVec(v, 424000u + 17u * trial);
      double abs_error = 0.0, rel_error = 0.0;
      exact_pc_current(v, pc_result);
      exact_shat(pc_result, shat_result);
      CompareVectors(shat_result, v, work, &abs_error, &rel_error);
      current_inverse = std::max(current_inverse, rel_error);
      exact_pc_intended(v, pc_result);
      exact_shat(pc_result, shat_result);
      CompareVectors(shat_result, v, work, &abs_error, &rel_error);
      intended_inverse = std::max(intended_inverse, rel_error);
    }
    {
      std::ostringstream s;
      s << std::setprecision(17)
        << "V4_INVERSE_EXACT Shat_times_PClegacy_minus_I_rel="
        << current_inverse
        << " Shat_times_PCcorrected_minus_I_rel=" << intended_inverse;
      log(s);
    }

    // TEST 4.1/4.2: identify what the production shell and PCShell compute,
    // by comparing them against the two exact candidates.  The production
    // objects use GMRES/GAMG at rtol 1e-6, so the matching candidate should
    // agree to ~1e-6 relative and the other to O(1).
    // Constructed with NO factor-order argument, i.e. exactly as production
    // builds it.  The acceptance test below therefore measures the shipped
    // default, not a flag setting.
    DistributedMatchingSchurPCContext production;
    InitializeDistributedMatchingSchurPCContext(production, matching,
                                                operators.ay, assembled.ay,
                                                1.0e-13);
    Vec production_result = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(v, &production_result));
    double shell_vs_shat = 0.0, shell_vs_mirror = 0.0;
    double pc_default_vs_current = 0.0, pc_default_vs_intended = 0.0;
    double pc_flag_vs_current = 0.0, pc_flag_vs_intended = 0.0;
    PC shell_pc = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD, PCCreate(PETSC_COMM_WORLD, &shell_pc));
    PetscCallAbort(PETSC_COMM_WORLD, PCSetType(shell_pc, PCSHELL));
    PetscCallAbort(PETSC_COMM_WORLD,
                   PCSetOperators(shell_pc, assembled.ay, assembled.ay));
    PetscCallAbort(PETSC_COMM_WORLD, PCShellSetContext(shell_pc, &production));
    PetscCallAbort(PETSC_COMM_WORLD,
                   PCShellSetApply(shell_pc,
                                   ApplyDistributedMatchingSchurPreconditioner));
    PetscCallAbort(PETSC_COMM_WORLD, PCSetUp(shell_pc));
    for (unsigned trial = 0; trial < 3u; ++trial) {
      FillDeterministicRandomVec(v, 616000u + 19u * trial);
      double abs_error = 0.0, rel_error = 0.0;
      PetscCallAbort(PETSC_COMM_WORLD,
                     MatMult(production.schur_preconditioner, v,
                             production_result));
      exact_shat(v, shat_result);
      CompareVectors(production_result, shat_result, work, &abs_error,
                     &rel_error);
      shell_vs_shat = std::max(shell_vs_shat, rel_error);
      exact_shat_mirror(v, shat_result);
      CompareVectors(production_result, shat_result, work, &abs_error,
                     &rel_error);
      shell_vs_mirror = std::max(shell_vs_mirror, rel_error);

      production.legacy_factor_order = false;   // production default
      PetscCallAbort(PETSC_COMM_WORLD, PCApply(shell_pc, v, production_result));
      exact_pc_current(v, pc_result);
      CompareVectors(production_result, pc_result, work, &abs_error,
                     &rel_error);
      pc_default_vs_current = std::max(pc_default_vs_current, rel_error);
      exact_pc_intended(v, pc_result);
      CompareVectors(production_result, pc_result, work, &abs_error,
                     &rel_error);
      pc_default_vs_intended = std::max(pc_default_vs_intended, rel_error);

      production.legacy_factor_order = true;    // the swapped factor order
      PetscCallAbort(PETSC_COMM_WORLD, PCApply(shell_pc, v, production_result));
      exact_pc_current(v, pc_result);
      CompareVectors(production_result, pc_result, work, &abs_error,
                     &rel_error);
      pc_flag_vs_current = std::max(pc_flag_vs_current, rel_error);
      exact_pc_intended(v, pc_result);
      CompareVectors(production_result, pc_result, work, &abs_error,
                     &rel_error);
      pc_flag_vs_intended = std::max(pc_flag_vs_intended, rel_error);
      production.legacy_factor_order = false;
    }
    {
      // The shipped default (pcdefault_*) must match the production ordering
      // right^{-1} E left^{-1} = Shat^{-1}; the swapped ordering must be
      // reachable only through the flag.
      std::ostringstream s;
      s << std::setprecision(17)
        << "V4_IDENTIFY shell_vs_leftEinvright_rel=" << shell_vs_shat
        << " shell_vs_rightEinvleft_rel=" << shell_vs_mirror
        << " pcdefault_vs_legacy_rel=" << pc_default_vs_current
        << " pcdefault_vs_corrected_rel=" << pc_default_vs_intended
        << " pclegacyflag_vs_legacy_rel=" << pc_flag_vs_current
        << " pclegacyflag_vs_corrected_rel=" << pc_flag_vs_intended;
      log(s);
    }
    {
      // || Shat * PC_production - I ||, using the production PCApply with its
      // shipped default and exact factor solves for Shat.  Must be at roundoff.
      double production_inverse = 0.0;
      Vec pc_out = nullptr, shat_out = nullptr;
      PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(v, &pc_out));
      PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(v, &shat_out));
      for (unsigned trial = 0; trial < 4u; ++trial) {
        FillDeterministicRandomVec(v, 828000u + 23u * trial);
        double abs_error = 0.0, rel_error = 0.0;
        PetscCallAbort(PETSC_COMM_WORLD, PCApply(shell_pc, v, pc_out));
        exact_shat(pc_out, shat_out);
        CompareVectors(shat_out, v, work, &abs_error, &rel_error);
        production_inverse = std::max(production_inverse, rel_error);
      }
      std::ostringstream s;
      s << std::setprecision(17)
        << "V4_PRODUCTION_INVERSE Shat_times_PCproduction_minus_I_rel="
        << production_inverse
        << " inner_rtol=1e-13 note=limited_by_inner_Krylov_tolerance";
      log(s);
      PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&shat_out));
      PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&pc_out));
    }
    PetscCallAbort(PETSC_COMM_WORLD, PCDestroy(&shell_pc));
    PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&production_result));
    DestroyDistributedMatchingSchurPCContext(production);

    // TEST 4.4: dense dumps of S, PC_current*S and PC_intended*S for the
    // eigenvalue study (small systems only).
    if (space_time_global <= 256) {
      std::function<void(Vec, Vec)> exact_schur = [&](Vec in, Vec result) {
        PetscCallAbort(PETSC_COMM_WORLD, MatMult(operators.ayp, in, t1));
        PetscCallAbort(PETSC_COMM_WORLD, KSPSolve(lu_ay, t1, t2));
        PetscCallAbort(PETSC_COMM_WORLD, MatMult(operators.apy, t2, t1));
        PetscCallAbort(PETSC_COMM_WORLD, MatMult(operators.ap, in, result));
        PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(result, -1.0, t1));
      };
      Vec inner = nullptr;
      PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(v, &inner));
      std::function<void(Vec, Vec)> pc_current_times_s = [&](Vec in,
                                                             Vec result) {
        exact_schur(in, inner);
        exact_pc_current(inner, result);
      };
      std::function<void(Vec, Vec)> pc_intended_times_s = [&](Vec in,
                                                              Vec result) {
        exact_schur(in, inner);
        exact_pc_intended(inner, result);
      };
      DumpDenseOperator(output_prefix + "_schur_S.txt", operators.ay,
                        exact_schur);
      DumpDenseOperator(output_prefix + "_schur_PCcurrent_S.txt", operators.ay,
                        pc_current_times_s);
      DumpDenseOperator(output_prefix + "_schur_PCintended_S.txt",
                        operators.ay, pc_intended_times_s);
      PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&inner));
      std::ostringstream s;
      s << "V4_DUMP size=" << space_time_global << " prefix=" << output_prefix;
      log(s);
    }

    PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&shat_result));
    PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&pc_result));

    // -------------------------------------------------------------------
    // TEST 7: explicit monolithic assembly + direct LU reference solve.
    // -------------------------------------------------------------------
    DistributedDataSideResult data = BuildDistributedRightHandSide(
        config, layout, mass, tracking, operators.ay);
    Mat full = AssembleMonolithicSerial(config, assembled, control, tracking,
                                        config.time.steps, spatial_global,
                                        false);
    Vec full_rhs = nullptr, full_solution = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatCreateVecs(full, &full_solution, &full_rhs));
    ConcatenateSerial(data.rhs_y, data.rhs_p, full_rhs);
    KSP direct = nullptr;
    CreateDirectFactorSolver(full, "vfull_", &direct);
    PetscCallAbort(PETSC_COMM_WORLD, KSPSolve(direct, full_rhs, full_solution));
    Vec y_direct = nullptr, p_direct = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(v, &y_direct));
    PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(v, &p_direct));
    SplitSerial(full_solution, y_direct, p_direct);
    // Residual of the direct solution in the monolithic matrix.
    Vec full_residual = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(full_rhs, &full_residual));
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatMult(full, full_solution, full_residual));
    PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(full_residual, -1.0, full_rhs));
    double direct_residual = 0.0, rhs_norm = 0.0;
    PetscCallAbort(PETSC_COMM_WORLD,
                   VecNorm(full_residual, NORM_2, &direct_residual));
    PetscCallAbort(PETSC_COMM_WORLD, VecNorm(full_rhs, NORM_2, &rhs_norm));

    // Iterative solve with the production stack, both PC modes.
    for (unsigned mode = 0; mode < 2u; ++mode) {
      DistributedOptimalControlSolve iterative =
          SolveOptimalControlOwnershipDistributed(
              config, layout, mass, tracking, control, nullptr, operators,
              assembled,
              MakeKktSolverPolicy(mode == 0u
                                      ? DistributedSolvePc::ADDITIVE_GAMG
                                      : DistributedSolvePc::SCHUR_MATCHING));
      double y_abs = 0.0, y_rel = 0.0, p_abs = 0.0, p_rel = 0.0;
      CompareVectors(iterative.fields.state, y_direct, work, &y_abs, &y_rel);
      CompareVectors(iterative.fields.adjoint, p_direct, work, &p_abs, &p_rel);
      // Control reconstructed from each adjoint, with the production mask.
      Vec u_direct = nullptr;
      PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(v, &u_direct));
      {
        const PetscScalar* p_values = nullptr;
        PetscScalar* u_values = nullptr;
        PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(p_direct, &p_values));
        PetscCallAbort(PETSC_COMM_WORLD, VecGetArray(u_direct, &u_values));
        // Same active set as production.
        const std::vector<unsigned char> direct_active =
            ActiveControlDofs(control, layout);
        for (unsigned slot = 0; slot < config.time.steps; ++slot) {
          const PetscInt offset =
              static_cast<PetscInt>(slot) * layout.local_size;
          for (PetscInt i = 0; i < layout.local_size; ++i)
            u_values[offset + i] =
                direct_active[i]
                    ? -(config.eta / config.alpha_u) * p_values[offset + i]
                    : 0.0;
        }
        PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArray(u_direct, &u_values));
        PetscCallAbort(PETSC_COMM_WORLD,
                       VecRestoreArrayRead(p_direct, &p_values));
      }
      double u_abs = 0.0, u_rel = 0.0;
      CompareVectors(iterative.fields.control, u_direct, work, &u_abs, &u_rel);
      std::ostringstream s;
      s << std::setprecision(17) << "V7_DIRECT pc="
        << (mode == 0u ? "ADDITIVE" : "SCHUR_MATCHING")
        << " direct_residual=" << direct_residual
        << " rhs_norm=" << rhs_norm << " y_abs=" << y_abs << " y_rel=" << y_rel
        << " p_abs=" << p_abs << " p_rel=" << p_rel << " u_abs=" << u_abs
        << " u_rel=" << u_rel << " iterative_true_residual="
        << iterative.true_residual << " outer_iterations="
        << iterative.outer_iterations << " reason=" << iterative.reason;
      log(s);
      PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&u_direct));
    }

    // Matching-Schur factor-order comparison, reachable only from here.
    // Production always uses right^{-1} E left^{-1}.
    for (unsigned legacy = 0; legacy < 2u; ++legacy) {
      DistributedOptimalControlSolve run =
          SolveOptimalControlOwnershipDistributed(
              config, layout, mass, tracking, control, nullptr, operators,
              assembled,
              MakeKktSolverPolicy(DistributedSolvePc::SCHUR_MATCHING, 1.0e-10,
                                  legacy == 1u));
      std::ostringstream s;
      s << std::setprecision(17) << "V4_OUTER order="
        << (legacy == 1u ? "LEGACY" : "CORRECTED_PRODUCTION")
        << " reason=" << run.reason
        << " outer_iterations=" << run.outer_iterations
        << " inner_y_iterations=" << run.inner_y_iterations
        << " inner_p_iterations=" << run.inner_p_iterations
        << " true_residual=" << run.true_residual;
      log(s);
    }

    // Symmetric arrangement, solved directly; phi must equal -p.
    Mat symmetric = AssembleMonolithicSerial(config, assembled, control,
                                             tracking, config.time.steps,
                                             spatial_global, true);
    Vec sym_rhs = nullptr, sym_solution = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatCreateVecs(symmetric, &sym_solution, &sym_rhs));
    Vec minus_rhs_p = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(data.rhs_p, &minus_rhs_p));
    PetscCallAbort(PETSC_COMM_WORLD, VecCopy(data.rhs_p, minus_rhs_p));
    PetscCallAbort(PETSC_COMM_WORLD, VecScale(minus_rhs_p, -1.0));
    ConcatenateSerial(minus_rhs_p, data.rhs_y, sym_rhs);
    // Algebraic equivalence by substitution: the symmetric system must be
    // satisfied exactly by (y, phi) = (y_direct, -p_direct).  This is
    // independent of whether a direct solver can factor the symmetric matrix.
    Vec substitution = nullptr, sym_residual = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(sym_rhs, &substitution));
    PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(sym_rhs, &sym_residual));
    {
      Vec minus_p = nullptr;
      PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(p_direct, &minus_p));
      PetscCallAbort(PETSC_COMM_WORLD, VecCopy(p_direct, minus_p));
      PetscCallAbort(PETSC_COMM_WORLD, VecScale(minus_p, -1.0));
      ConcatenateSerial(y_direct, minus_p, substitution);
      PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&minus_p));
    }
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatMult(symmetric, substitution, sym_residual));
    PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(sym_residual, -1.0, sym_rhs));
    double substitution_residual = 0.0, sym_rhs_norm = 0.0;
    PetscCallAbort(PETSC_COMM_WORLD,
                   VecNorm(sym_residual, NORM_2, &substitution_residual));
    PetscCallAbort(PETSC_COMM_WORLD, VecNorm(sym_rhs, NORM_2, &sym_rhs_norm));

    KSP sym_direct = nullptr;
    CreateDirectFactorSolver(symmetric, "vsym_", &sym_direct);
    PetscCallAbort(PETSC_COMM_WORLD,
                   KSPSolve(sym_direct, sym_rhs, sym_solution));
    KSPConvergedReason sym_reason;
    PetscCallAbort(PETSC_COMM_WORLD,
                   KSPGetConvergedReason(sym_direct, &sym_reason));
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatMult(symmetric, sym_solution, sym_residual));
    PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(sym_residual, -1.0, sym_rhs));
    double sym_direct_residual = 0.0;
    PetscCallAbort(PETSC_COMM_WORLD,
                   VecNorm(sym_residual, NORM_2, &sym_direct_residual));
    {
      std::ostringstream s;
      s << std::setprecision(17)
        << "V7_SYMSUBST substitution_residual=" << substitution_residual
        << " sym_rhs_norm=" << sym_rhs_norm
        << " direct_reason=" << static_cast<int>(sym_reason)
        << " direct_residual=" << sym_direct_residual;
      log(s);
    }
    PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&sym_residual));
    PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&substitution));
    if (2 * space_time_global <= 512) {
      std::function<void(Vec, Vec)> apply_symmetric = [&](Vec in, Vec result) {
        PetscCallAbort(PETSC_COMM_WORLD, MatMult(symmetric, in, result));
      };
      std::function<void(Vec, Vec)> apply_full = [&](Vec in, Vec result) {
        PetscCallAbort(PETSC_COMM_WORLD, MatMult(full, in, result));
      };
      DumpDenseOperator(output_prefix + "_monolithic_symmetric.txt", symmetric,
                        apply_symmetric);
      DumpDenseOperator(output_prefix + "_monolithic_production.txt", full,
                        apply_full);
      std::ostringstream s;
      s << "V7_DUMP size=" << 2 * space_time_global
        << " prefix=" << output_prefix;
      log(s);
    }
    Vec y_sym = nullptr, phi_sym = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(v, &y_sym));
    PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(v, &phi_sym));
    SplitSerial(sym_solution, y_sym, phi_sym);
    PetscCallAbort(PETSC_COMM_WORLD, VecScale(phi_sym, -1.0));
    double sym_y_abs = 0.0, sym_y_rel = 0.0, sym_p_abs = 0.0, sym_p_rel = 0.0;
    CompareVectors(y_sym, y_direct, work, &sym_y_abs, &sym_y_rel);
    CompareVectors(phi_sym, p_direct, work, &sym_p_abs, &sym_p_rel);
    // Structural symmetry of the symmetric arrangement.
    Mat sym_transpose = nullptr;
    double asymmetry = 0.0;
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatTranspose(symmetric, MAT_INITIAL_MATRIX, &sym_transpose));
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatAXPY(sym_transpose, -1.0, symmetric, DIFFERENT_NONZERO_PATTERN));
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatNorm(sym_transpose, NORM_FROBENIUS, &asymmetry));
    double sym_norm = 0.0;
    PetscCallAbort(PETSC_COMM_WORLD,
                   MatNorm(symmetric, NORM_FROBENIUS, &sym_norm));
    {
      std::ostringstream s;
      s << std::setprecision(17) << "V7_SYMMETRIC y_abs=" << sym_y_abs
        << " y_rel=" << sym_y_rel << " minus_phi_vs_p_abs=" << sym_p_abs
        << " minus_phi_vs_p_rel=" << sym_p_rel
        << " frobenius_asymmetry=" << asymmetry
        << " frobenius_norm=" << sym_norm;
      log(s);
    }
    PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&sym_transpose));
    PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&phi_sym));
    PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&y_sym));
    PetscCallAbort(PETSC_COMM_WORLD, KSPDestroy(&sym_direct));
    PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&minus_rhs_p));
    PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&sym_solution));
    PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&sym_rhs));
    PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&symmetric));

    PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&full_residual));
    PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&p_direct));
    PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&y_direct));
    PetscCallAbort(PETSC_COMM_WORLD, KSPDestroy(&direct));
    PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&full_solution));
    PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&full_rhs));
    PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&full));

    PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&out));
    PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&t2));
    PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&t1));
    PetscCallAbort(PETSC_COMM_WORLD, KSPDestroy(&lu_ay));
    PetscCallAbort(PETSC_COMM_WORLD, KSPDestroy(&lu_left));
    PetscCallAbort(PETSC_COMM_WORLD, KSPDestroy(&lu_right));
    DestroyDistributedMatchingAuxiliaryMatrices(matching);
  } else {
    std::ostringstream s;
    s << "V4_V7_SKIPPED reason=direct_factorisation_requires_one_rank ranks="
      << ranks;
    log(s);
  }

  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&work));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&b));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&a));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&w));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&v));

  return status;
}

// ===========================================================================
// End algebraic test battery.
// ===========================================================================

// ===========================================================================
// Boundary-reduced Schur solver sentinel
// ===========================================================================

/** One production solve of `config` at `alpha_u` with the boundary-reduced
 *  Schur preconditioner, through the public solver entry (a caller-owned
 *  `reduced` when given, a solve-local one otherwise).  Prints the BR line
 *  and returns whether the frozen bounds hold: convergence, at most
 *  `max_outer` outer iterations, true KKT residual at most `residual_bound`. */
static bool RunReducedSolveCheck(const char* label, const ProblemConfig& config,
                                 double alpha_u, const SpatialOperators& spatial,
                                 const KktSolverPolicy& policy,
                                 BoundaryReducedSchurOperators* reduced,
                                 int max_outer, double residual_bound,
                                 DistributedOptimalControlSolve* solve) {
  ProblemConfig c = config;
  c.alpha_u = alpha_u;
  SpaceTimeOperators spacetime(c, spatial);
  *solve = SolveOptimalControl(c, spatial, spacetime, policy, reduced);
  const bool ok = solve->accepted && solve->outer_iterations <= max_outer &&
                  solve->true_residual <= residual_bound;
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0)
    std::cout << std::setprecision(6) << label << (ok ? " PASS" : " FAIL")
              << " alpha_u=" << alpha_u << " reason=" << solve->reason
              << " outer=" << solve->outer_iterations << "/" << max_outer
              << " true_residual=" << solve->true_residual << "/"
              << residual_bound << " state_rel=" << solve->kkt.state_rel
              << " adjoint_rel=" << solve->kkt.adjoint_rel
              << std::setprecision(17) << " J_track=" << solve->fields.J_track
              << " J_reg=" << solve->fields.J_reg << std::endl;
  return ok;
}

int RunBoundaryReducedVerification(const ProblemConfig& config,
                                   MultiLevelSolution& solution, Mesh* mesh,
                                   unsigned solution_type,
                                   double spatial_tolerance) {
  int rank = 0, ranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  if (config.control.type != ProblemConfig::ControlType::NEUMANN) {
    if (rank == 0)
      std::cout << "BR_VERIFY FAIL reason=requires_neumann_control"
                << std::endl;
    return 1;
  }
  // Frozen bounds: the accepted solver converges in 2 outer iterations with
  // a true KKT residual far below these at every alpha down to 1e-12.
  const int kMaxOuter = 2;
  const double kResidualBasic = 1.0e-10, kResidualLowAlpha = 1.0e-9;
  const double kReuseTolerance = 1.0e-12;   // reused vs fresh solve, relative

  SpatialOperators spatial(config, mesh, solution_type, solution);
  const KktSolverPolicy policy = MakeKktSolverPolicy(
      DistributedSolvePc::SCHUR_BOUNDARY_REDUCED, spatial_tolerance);
  if (rank == 0)
    std::cout << "BR_VERIFY config nx=" << config.mesh.nx << " nz="
              << config.mesh.nz << " N=" << config.time.steps
              << " ranks=" << ranks << " pc=SCHUR_BOUNDARY_REDUCED"
              << std::endl;
  int passed = 0;

  // BR1: the production integration at alpha_u = 1e-8; BR2: the low-alpha
  // stress at 1e-12 (beta = 1e12).  Both with a solve-local operator.
  DistributedOptimalControlSolve basic, low;
  passed += RunReducedSolveCheck("BR1", config, 1.0e-8, spatial, policy,
                                 nullptr, kMaxOuter, kResidualBasic, &basic);
  passed += RunReducedSolveCheck("BR2", config, 1.0e-12, spatial, policy,
                                 nullptr, kMaxOuter, kResidualLowAlpha, &low);

  // BR3: one caller-owned operator across 1e-8, 1e-12, 1e-8.  The invariant
  // operators must survive untouched (S0/S1 equal to their snapshot, the same
  // spatial operator objects) and the reused solves must reproduce the
  // fresh-operator solves.
  {
    BoundaryReducedSchurOperators reduced(config, spatial.mass,
                                          spatial.stiffness, spatial.control,
                                          spatial.tracking);
    Mat S0_snapshot = nullptr, S1_snapshot = nullptr;
    if (rank == 0) {
      PetscCallAbort(PETSC_COMM_WORLD,
                     MatDuplicate(reduced.S0(), MAT_COPY_VALUES, &S0_snapshot));
      PetscCallAbort(PETSC_COMM_WORLD,
                     MatDuplicate(reduced.S1(), MAT_COPY_VALUES, &S1_snapshot));
    }
    const Mat diagonal = reduced.diagonal();
    const Mat restricted = reduced.restricted_diagonal();
    std::string reason;
    bool ok = reduced.IsCompatible(config, spatial.mass, spatial.stiffness,
                                   spatial.control, spatial.tracking, &reason);
    DistributedOptimalControlSolve first, second, third;
    ok = RunReducedSolveCheck("BR3a", config, 1.0e-8, spatial, policy, &reduced,
                              kMaxOuter, kResidualBasic, &first) && ok;
    ok = RunReducedSolveCheck("BR3b", config, 1.0e-12, spatial, policy,
                              &reduced, kMaxOuter, kResidualLowAlpha, &second)
         && ok;
    ok = RunReducedSolveCheck("BR3c", config, 1.0e-8, spatial, policy, &reduced,
                              kMaxOuter, kResidualBasic, &third) && ok;
    ok = ok && first.reduced.reused_invariant_operator &&
         second.reduced.reused_invariant_operator &&
         third.reduced.reused_invariant_operator &&
         reduced.prepared_beta() == first.reduced.prepared_beta &&
         reduced.diagonal() == diagonal &&
         reduced.restricted_diagonal() == restricted;
    // reused solves against the fresh ones of BR1/BR2, and third against first
    Vec work = nullptr;
    PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(basic.fields.state, &work));
    double worst = 0.0;
    const Vec pairs[][2] = {
        {first.fields.state, basic.fields.state},
        {first.fields.adjoint, basic.fields.adjoint},
        {first.fields.control, basic.fields.control},
        {second.fields.state, low.fields.state},
        {second.fields.adjoint, low.fields.adjoint},
        {second.fields.control, low.fields.control},
        {third.fields.state, first.fields.state},
        {third.fields.adjoint, first.fields.adjoint},
        {third.fields.control, first.fields.control}};
    for (const Vec* pair : pairs) {
      double absolute = 0.0, relative = 0.0;
      CompareVectors(pair[0], pair[1], work, &absolute, &relative);
      worst = std::max(worst, relative);
    }
    PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&work));
    ok = ok && worst <= kReuseTolerance;
    PetscBool S0_unchanged = PETSC_TRUE, S1_unchanged = PETSC_TRUE;
    if (rank == 0) {
      PetscCallAbort(PETSC_COMM_WORLD,
                     MatEqual(reduced.S0(), S0_snapshot, &S0_unchanged));
      PetscCallAbort(PETSC_COMM_WORLD,
                     MatEqual(reduced.S1(), S1_snapshot, &S1_unchanged));
      PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&S1_snapshot));
      PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&S0_snapshot));
    }
    int invariant = (S0_unchanged && S1_unchanged) ? 1 : 0;
    MPI_Bcast(&invariant, 1, MPI_INT, 0, MPI_COMM_WORLD);
    ok = ok && invariant;
    if (rank == 0)
      std::cout << std::setprecision(6) << "BR3" << (ok ? " PASS" : " FAIL")
                << " compatible=" << (reason.empty() ? 1 : 0)
                << " reused=" << first.reduced.reused_invariant_operator
                << second.reduced.reused_invariant_operator
                << third.reduced.reused_invariant_operator
                << " S0_S1_unchanged=" << invariant
                << " prepared_beta=" << reduced.prepared_beta()
                << " reused_vs_fresh_rel=" << worst << "/" << kReuseTolerance
                << std::endl;
    passed += ok;
  }

  if (rank == 0)
    std::cout << "BR_VERIFY " << (passed == 3 ? "PASS" : "FAIL") << " "
              << passed << "/3" << std::endl;
  return passed == 3 ? 0 : 1;
}
