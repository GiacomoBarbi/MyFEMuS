
#include "FemusInit.hpp"
#include "MultiLevelProblem.hpp"
#include "VTKWriter.hpp"
#include "TransientSystem.hpp"
#include "NonLinearImplicitSystem.hpp"
#include "Marker.hpp"
#include "Line.hpp"

#include "Fluid.hpp"
#include "Solid.hpp"
#include "Parameter.hpp"

#include "NumericVector.hpp"
#include "adept.h"

#include "../include/mpmFem.hpp"

using namespace femus;

double SetVariableTimeStep(const double time)
{
  double dt =  0.008;
  return dt;
}

bool SetBoundaryCondition(const std::vector < double >& x, const char name[], double& value, const int facename, const double time)
{
  bool test = 1; //dirichlet
  value = 0.;

  if (!strcmp(name, "DX")) {
    if (2 == facename || 4 == facename) {
      test = 0;
      value = 0;
    }
  }
  else if (!strcmp(name, "DY")) {
    if (3 == facename) {
      test = 0;
      value = 0;
    }
  }
  else if (!strcmp(name, "M")) {
    if (1 == facename) {
      test = 0;
      value = 0;
    }
  }

  return test;

}

int main(int argc, char** args)
{

  // init Petsc-MPI communicator
  FemusInit mpinit(argc, args, MPI_COMM_WORLD);

  MultiLevelMesh mlMsh;
  double scalingFactor = 1.;
  unsigned numberOfUniformLevels = 4; //for refinement in 3D
  //unsigned numberOfUniformLevels = 1;
  unsigned numberOfSelectiveLevels = 0;

  double Lref = 1.;
  double Uref = 1.;
  double rhos = 10000;
  double nu = 0.4;
  double E = 1.74 * 1.e6;

  Parameter par(Lref, Uref);

  // Generate Solid Object
  Solid solid;
  solid = Solid(par, E, nu, rhos, "Neo-Hookean");

  mlMsh.ReadCoarseMesh("../input/square.neu", "fifth", scalingFactor);
  mlMsh.RefineMesh(numberOfUniformLevels + numberOfSelectiveLevels, numberOfUniformLevels , NULL);

  unsigned dim = mlMsh.GetDimension();

  MultiLevelSolution mlSol(&mlMsh);
  // add variables to mlSol
  mlSol.AddSolution("DX", LAGRANGE, SECOND, 2);
  if (dim > 1) mlSol.AddSolution("DY", LAGRANGE, SECOND, 2);
  if (dim > 2) mlSol.AddSolution("DZ", LAGRANGE, SECOND, 2);

  mlSol.AddSolution("M", LAGRANGE, FIRST, 2);
  mlSol.AddSolution("Mat", DISCONTINOUS_POLYNOMIAL, ZERO, 0, false);

  mlSol.Initialize("All");

  mlSol.AttachSetBoundaryConditionFunction(SetBoundaryCondition);

  // ******* Set boundary conditions *******
  mlSol.GenerateBdc("DX", "Steady");
  if (dim > 1) mlSol.GenerateBdc("DY", "Steady");
  if (dim > 2) mlSol.GenerateBdc("DZ", "Steady");
  mlSol.GenerateBdc("M", "Steady");

  MultiLevelProblem ml_prob(&mlSol);

  ml_prob.parameters.set<Solid> ("SolidMPM") = solid;
  ml_prob.parameters.set<Solid> ("SolidFEM") = solid;

  // ******* Add MPM system to the MultiLevel problem *******
  TransientNonlinearImplicitSystem& system = ml_prob.add_system < TransientNonlinearImplicitSystem > ("MPM_FEM");
  system.AddSolutionToSystemPDE("DX");
  if (dim > 1)system.AddSolutionToSystemPDE("DY");
  if (dim > 2) system.AddSolutionToSystemPDE("DZ");

  // ******* System MPM Assembly *******
  system.SetAssembleFunction(AssembleMPMSys);
  //system.SetAssembleFunction(AssembleFEM);
  // ******* set MG-Solver *******
  system.SetMgType(V_CYCLE);


  system.SetAbsoluteLinearConvergenceTolerance(1.0e-10);
  system.SetMaxNumberOfLinearIterations(1);
  system.SetNonLinearConvergenceTolerance(1.e-9);
  system.SetMaxNumberOfNonLinearIterations(20);

  system.SetNumberPreSmoothingStep(1);
  system.SetNumberPostSmoothingStep(1);

  // ******* Set Preconditioner *******
  system.SetMgSmoother(GMRES_SMOOTHER);

  system.init();

  // ******* Set Smoother *******
  system.SetSolverFineGrids(GMRES);

  system.SetPreconditionerFineGrids(ILU_PRECOND);

  system.SetTolerances(1.e-10, 1.e-15, 1.e+50, 40, 40);

  unsigned rows = 60;
  unsigned columns = 120;
  unsigned size = rows * columns;

  std::vector < std::vector < double > > x; // marker
  std::vector < MarkerType > markerType;

  x.resize(size);
  markerType.resize(size);

  std::vector < std::vector < std::vector < double > > > line(1);
  std::vector < std::vector < std::vector < double > > > line0(1);

  for (unsigned j = 0; j < size; j++) {
    x[j].assign(dim, 0.);
    markerType[j] = VOLUME;
  }

  //BEGIN initialization
  for (unsigned i = 0; i < rows; i++) {
    for (unsigned j = 0; j < columns; j++) {

      x[i * columns + j][0] = -0.5 + ((0.625 - 0.000001) / (columns - 1)) * j;
      x[i * columns + j][1] = -0.0625 + ((0.25 - 0.000001) / (rows - 1)) * i;
      if (dim == 3) {
        x[j][2] = 0.;
      }
    }
  }
  //END

  unsigned solType = 2;
  linea = new Line(x, markerType, mlSol.GetLevel(numberOfUniformLevels - 1), solType);
  
  double beamArea = ( -0.5 + (0.625 - 0.000001) ) * (-0.0625 + (0.25 - 0.000001) );
  linea->SetParticlesMass(beamArea, rhos);

  linea->GetLine(line0[0]);
  PrintLine(DEFAULT_OUTPUTDIR, line0, false, 0);

  
  linea->GetParticlesToGridMaterial();
  
  // ******* Print solution *******
  mlSol.SetWriter(VTK);

  std::vector<std::string> mov_vars;
  mov_vars.push_back("DX");
  mov_vars.push_back("DY");
  mov_vars.push_back("DZ");
  mlSol.GetWriter()->SetMovingMesh(mov_vars);

  std::vector<std::string> print_vars;
  print_vars.push_back("All");

  mlSol.GetWriter()->SetDebugOutput(true);
  mlSol.GetWriter()->Write(DEFAULT_OUTPUTDIR, "biquadratic", print_vars, 0);


  system.AttachGetTimeIntervalFunction(SetVariableTimeStep);
  unsigned n_timesteps = 400;
  for (unsigned time_step = 1; time_step <= n_timesteps; time_step++) {

    system.CopySolutionToOldSolution();
    
    system.MGsolve();

    mlSol.GetWriter()->Write(DEFAULT_OUTPUTDIR, "biquadratic", print_vars, time_step);

    GridToParticlesProjection(ml_prob, *linea);

    linea->GetLine(line[0]);
    PrintLine(DEFAULT_OUTPUTDIR, line, false, time_step);



  }



  delete linea;
  return 0;

} //end main

