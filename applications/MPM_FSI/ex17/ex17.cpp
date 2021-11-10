#include "FemusInit.hpp"
#include "MultiLevelProblem.hpp"
#include "VTKWriter.hpp"
#include "MultiLevelSolution.hpp"
#include "MeshRefinement.hpp"

#include "TransientSystem.hpp"
#include "NonLinearImplicitSystem.hpp"

#include "Fluid.hpp"
#include "Solid.hpp"
#include "Parameter.hpp"

#include "PetscMatrix.hpp"

#include "./include/parameter.hpp"

using namespace femus;

parameter *par;

#include "marker.hpp"
#include "projection.hpp"
Projection *projection;

#include "background.hpp"


double TimeStepBeam(const double time);
bool BoundaryConditionBeam(const std::vector < double >& x, const char name[], double& value, const int facename, const double t);

double TimeStepTurek3(const double time);
bool BoundaryConditionTurek3(const std::vector < double >&x, const char name[], double &value, const int facename, const double t);

double TimeStepTurek1(const double time);
bool BoundaryConditionTurek1(const std::vector < double >&x, const char name[], double &value, const int facename, const double t);

double TimeStepTurek0(const double time);
bool BoundaryConditionTurek0(const std::vector < double >&x, const char name[], double &value, const int facename, const double t);


parameter beam = parameter(false, 1., {0., 0., 0.},
                           45., 0.05, 0.05, 0.05,
                           true, 7850., 1000., 0.3, 2.e05, 1.0e-03,
                           "../input/beam.neu", 1.e5, 1, 2,
                           "../input/fsi_bnc_2D.neu", 10000., 5,
                           BoundaryConditionBeam, TimeStepBeam);


// parameter turek1 = parameter(false, 0.8, {0., 0., 0.},
//                              45., 0.05, 0.05, 0.05,
//                              false, 1000., 1000., 0.4, 1400000., 1.,
//                              "../input/turekBeam2D.neu", 1., 3, 2,
//                              "../input/turek2D.neu", 1., 1,
//                              BoundaryConditionTurek1, TimeStepTurek1);

parameter turek1 = parameter(false, 0.8, {0., 0., 0.},
                             45., 0.05, 0.05, 0.05,
                             false, 1000., 1000., 0.4, 1400000., 1.,
                             "../input/turekBeam2DFine.neu", 1., 4, 3,
                             "../input/turek2DNew.neu", 1., -1,  // 5-2 >= 3
                             BoundaryConditionTurek1, TimeStepTurek1);


parameter turek3 = parameter(false, 0.8, {0., 0., 0.},
                             45., 0.05, 0.05, 0.05,
                             false, 1000., 1000., 0.4, 4*1400000., 1.,
                             "../input/turekBeam2DFine.neu", 1., 5, 1,
                             "../input/turek2DNew.neu", 1., -2,
                             BoundaryConditionTurek3, TimeStepTurek3);



parameter turek0 = parameter(false, 0.8, {0., 0., 0.},
                             45., 0.05, 0.05, 0.05,
                             true, 1000., 1000., 0.4, 4 * 14000., 1.,
                             "../input/turekBeam2DMarker.neu", 1., 3, 3,
                             "../input/turekBeam2DEnvelopeNew.neu", 1., -2,
                             BoundaryConditionTurek0, TimeStepTurek0);




//     parameter(bool weakP, double theta, std::vector < double > gravity,
//               double GAMMA, double gammacF, double gammacS, double gammap,
//               bool NeoHookean, double rhos, double rhof, double nu, double E, double muf,
//               std::string mMesh, double mScale, unsigned mUniform, unsigned mAdaptive,
//               std::string bMesh, double bScale, unsigned deltaUniform,
//               BoundaryFunc bdcFunction, TimeFunc timeFunction)



#include <iostream>
#include <vector>
#include <algorithm>


int main(int argc, char** args) {

//   std::vector<unsigned> a = {3, 2, 1};
// 
//   std::vector<unsigned*> vec(a.size());
//   for(unsigned i = 0; i < a.size(); i++) {
//     vec[i] = &a[i];
//   }
// 
//   std::sort(vec.begin(), vec.end(), [](const unsigned * a, const unsigned * b) {
//     return *a < *b;
//   });
// 
//   std::cout << "vec = " << static_cast<unsigned>(vec[0] - &a[0]) << ", " << vec[1] - &a[0] << ", " << vec[2] - &a[0] << '\n';
//   std::cout << "abc = " << &a[0] << ", " << &a[1] << ", " << &a[2] << '\n';
// 
//   std::cout << "vec = " << *vec[0] << ", " << *vec[1] << ", " << *vec[2] << '\n';
//   std::cout << "abc = " << a[0] << ", " << a[1] << ", " << a[2] << '\n';
// 
// 
//   return 1;




  par = &turek3;
  //par = &beam;

  // init Petsc-MPI communicator
  FemusInit mpinit(argc, args, MPI_COMM_WORLD);

  MultiLevelMesh mlMshM;
  mlMshM.ReadCoarseMesh(par->_mMesh.c_str(), "fifth", par->_mScale);
  mlMshM.RefineMesh(par->_mUniform, par->_mUniform, NULL); //uniform refinement, this goes with the background mesh refinement. For COMSOL we use 8 = 3 turekBeam2D

  for(unsigned i = 0; i < par->_mAdaptive; i++) {
    FlagElements(mlMshM, 0);
    mlMshM.AddAMRMeshLevel();
  }
  mlMshM.EraseCoarseLevels(par->_mUniform + par->_mAdaptive - 1u);

  MultiLevelSolution mlSolM(&mlMshM);
  InitializeMarkerVariables(mlSolM);

  UpdateMeshQuantities(&mlSolM);

  unsigned numberOfRefinementB = par->_mUniform;
  MultiLevelMesh mlMshB;

  mlMshB.ReadCoarseMesh(par->_bMesh.c_str(), "fifth", par->_bScale);
  mlMshB.RefineMesh(par->_bUniform, par->_bUniform, NULL);
  mlMshB.EraseCoarseLevels(par->_bUniform - 1u);

  MultiLevelSolution mlSolB(&mlMshB);
  InitializeBackgroundVariables(mlSolB);


  unsigned dim = mlMshB.GetDimension();
  // ******* Set boundary conditions *******
  mlSolB.GenerateBdc("DX", "Steady");
  if(dim > 1) mlSolB.GenerateBdc("DY", "Steady");
  if(dim > 2) mlSolB.GenerateBdc("DZ", "Steady");
  mlSolB.GenerateBdc("VX", "Time_dependent");
  if(dim > 1) mlSolB.GenerateBdc("VY", "Steady");
  if(dim > 2) mlSolB.GenerateBdc("VZ", "Steady");
  mlSolB.GenerateBdc("P", "Steady");

  double Lref = 1.;
  double Uref = 1.;

  Parameter physics(Lref, Uref);
  // Generate Solid Object
  Solid solid(physics, par->_E, par->_nu, par->_rhos, "Neo-Hookean");
  Fluid fluid(physics, par->_muf, par->_rhof, "Newtonian");
  MultiLevelProblem ml_prob(&mlSolB);
  ml_prob.parameters.set<Solid> ("SolidMPM") = solid;
  ml_prob.parameters.set<Fluid> ("FluidFEM") = fluid;

  // ******* Add MPM system to the MultiLevel problem *******
  TransientNonlinearImplicitSystem& system = ml_prob.add_system < TransientNonlinearImplicitSystem > ("MPM_FSI");
  system.AddSolutionToSystemPDE("DX");
  if(dim > 1) system.AddSolutionToSystemPDE("DY");
  if(dim > 2) system.AddSolutionToSystemPDE("DZ");
  system.AddSolutionToSystemPDE("VX");
  if(dim > 1) system.AddSolutionToSystemPDE("VY");
  if(dim > 2) system.AddSolutionToSystemPDE("VZ");
  system.AddSolutionToSystemPDE("P");

  system.SetSparsityPatternMinimumSize(250);

  // ******* System MPM-FSI Assembly *******
  system.SetAssembleFunction(AssembleMPMSys);
  // ******* set MG-Solver *******
  system.SetMgType(V_CYCLE);
  system.SetAbsoluteLinearConvergenceTolerance(1.0e-10);
  system.SetMaxNumberOfLinearIterations(1);
  system.SetNonLinearConvergenceTolerance(1.e-9);
  system.SetMaxNumberOfNonLinearIterations(5);
  system.SetNumberPreSmoothingStep(1);
  system.SetNumberPostSmoothingStep(1);

  // ******* Set Preconditioner *******
  system.SetLinearEquationSolverType(FEMuS_DEFAULT);
  system.init();

  // ******* Set Smoother *******
  system.SetSolverFineGrids(GMRES);
  system.SetPreconditionerFineGrids(ILU_PRECOND);
  system.SetTolerances(1.e-10, 1.e-15, 1.e+50, 40, 40);

  system.AttachGetTimeIntervalFunction(par->_timeFunction);

  //******* Print solution *******
  mlSolM.SetWriter(VTK);
  mlSolB.SetWriter(VTK);

  std::vector<std::string> mov_vars;
  mov_vars.push_back("DX");
  mov_vars.push_back("DY");
  if(dim == 3) mov_vars.push_back("DZ");
  mlSolM.GetWriter()->SetMovingMesh(mov_vars);
  //mlSolB.GetWriter()->SetMovingMesh(mov_vars);
  mlSolM.GetWriter()->SetDebugOutput(false);
  mlSolB.GetWriter()->SetDebugOutput(true);
  std::vector<std::string> print_vars;
  print_vars.push_back("All");

  mlSolM.GetWriter()->Write(DEFAULT_OUTPUTDIR, "biquadratic", print_vars, 0);
  mlSolB.GetWriter()->Write(DEFAULT_OUTPUTDIR, "linear", print_vars, 0);

  projection = new Projection(&mlSolM, &mlSolB);



  for(unsigned t = 1; t <= 20000; t++) {

    system.CopySolutionToOldSolution();

    projection->SetNewmarkParameters(par->_beta, par->_gamma, 1.);
    clock_t time = clock();
    projection->FromMarkerToBackground();
    std::cout << "forward projection time " << t << " = " << static_cast<double>((clock() - time)) / CLOCKS_PER_SEC << std::endl;
    time = clock();
    system.MGsolve();
    std::cout << "solve time " << t << " = " << static_cast<double>((clock() - time)) / CLOCKS_PER_SEC << std::endl;
    
    mlSolB.GetWriter()->Write(DEFAULT_OUTPUTDIR, "linear", print_vars, t);
    
    time = clock();
    projection->FromBackgroundToMarker();
    std::cout << "backward projection time " << t << " = " << static_cast<double>((clock() - time)) / CLOCKS_PER_SEC << std::endl;
    mlSolM.GetWriter()->Write(DEFAULT_OUTPUTDIR, "biquadratic", print_vars, t);
   
  }


  delete projection;
  return 0;

} //end main


bool BoundaryConditionBeam(const std::vector < double >& x, const char name[], double& value, const int facename, const double t) {
  bool test = 1; //dirichlet
  value = 0.;

  double H = 1.e-4; //channel length
  double U = 0.05;
  double t2 = t * t;

  if(!strcmp(name, "DX")) {
    if(3 == facename) {
      test = 0;
      value = 0;
    }
  }
  else if(!strcmp(name, "DY")) {
    if(2 == facename || 4 == facename) {
      test = 0;
      value = 0;
    }
  }
  else if(!strcmp(name, "VX")) {
    if(2 == facename) {
      test = 0;
      value = 0;
    }
    else if(4 == facename) {
      value = (U * t2 / sqrt(pow((0.04 - t2), 2.) + pow((0.1 * t), 2.))) * 4. * (H - x[1]) * x[1] / (H * H);
    }
  }
  else if(!strcmp(name, "VY")) {
    if(2 == facename) {
      test = 0;
      value = 0;
    }
  }
  else if(!strcmp(name, "P")) {
    if(par->_weakP || 2 != facename) {
      test = 0;
    }
    value = 0;
  }

  return test;

}

double TimeStepBeam(const double time) {
  double dt =  0.005;
  return dt;
}


bool BoundaryConditionTurek1(const std::vector < double >&x, const char name[], double &value, const int facename, const double t) {
  bool test = 1;      //dirichlet
  value = 0.;

  const double Ubar = 0.2;    // FSI1
  const double L = 0.41;
  const double H = 2.5;

  if(!strcmp(name, "DY")) {
    if(1 == facename || 2 == facename) {
      test = 0;
      value = 0.;
    }
  }

  else if(!strcmp(name, "DX")) {
    if(3 == facename) {    //fluid wall
      test = 0;
      value = 0.;
    }
  }

  else if(!strcmp(name, "VY")) {
    if(2 == facename) {     //outflow
      test = 0;
      value = 0.;
    }
  }

  else if(!strcmp(name, "VX")) {
    if(1 == facename) {     //inflow
      test = 1;
      if(t < 2.0) {
        value = 1.5 * Ubar * 4.0 / 0.1681 * (x[1] + 0.21) * (-x[1] + 0.2) * 0.5 * (1. - cos(0.5 * M_PI * t));
      }
      else {
        value = 1.5 * Ubar * 4.0 / 0.1681 * (x[1] + 0.21) * (-x[1] + 0.2);
      }
    }
    else if(2 == facename) {    //outflow
      test = 0;
      value = 0.;
    }
  }

  else if(!strcmp(name, "P")) {
    if(par->_weakP || 2 != facename) {
      test = 0;
    }
    value = 0;
  }

  return test;

}


bool BoundaryConditionTurek3(const std::vector < double >&x, const char name[], double &value, const int facename, const double t) {
  bool test = 1;      //dirichlet
  value = 0.;

  const double Ubar = 2.;    // FSI1
  const double L = 0.41;
  const double H = 2.5;

  if(!strcmp(name, "DY")) {
    if(1 == facename || 2 == facename) {
      test = 0;
      value = 0.;
    }
  }

  else if(!strcmp(name, "DX")) {
    if(3 == facename) {    //fluid wall
      test = 0;
      value = 0.;
    }
  }

  else if(!strcmp(name, "VY")) {
    if(2 == facename) {     //outflow
      test = 0;
      value = 0.;
    }
  }

  else if(!strcmp(name, "VX")) {
    if(1 == facename) {     //inflow
      test = 1;
      if(t < 2.0) {
        value = 1.5 * Ubar * 4.0 / 0.1681 * (x[1] + 0.21) * (-x[1] + 0.2) * 0.5 * (1. - cos(0.5 * M_PI * t));
      }
      else {
        value = 1.5 * Ubar * 4.0 / 0.1681 * (x[1] + 0.21) * (-x[1] + 0.2);
      }
    }
    else if(2 == facename) {    //outflow
      test = 0;
      value = 0.;
    }
  }

  else if(!strcmp(name, "P")) {
    if(par->_weakP || 2 != facename) {
      test = 0;
    }
    value = 0;
  }

  return test;

}



double TimeStepTurek1(const double time) {
  double dt;
  double dt0 = 0.1;
  double dt1 = 1.; //FSI3
  double dt2 = 1.; //FSI3

  double T0 = 2.;
  double T1 = 20.;
  if(time < T0)
    dt = (0.5 * (1. + cos(M_PI * time / T0))) * dt0    + (0.5 * (1. - cos(M_PI * time / T0))) * dt1;
  else if(time < T1)
    dt = (0.5 * (1. + cos(M_PI * (time - T0) / (T1 - T0)))) * dt1 + (0.5 * (1. - cos(M_PI * (time - T0) / (T1 - T0)))) * dt2;
  else
    dt = dt2;

  return dt;
}



double TimeStepTurek3(const double time) {
  double dt;
  double dt0 = 0.05;
  double dt1 = 0.005; //FSI3
  double dt2 = 0.001; //FSI3

  double T0 = 2.;
  double T1 = 15.;
  if(time < T0)
    dt = (0.5 * (1. + cos(M_PI * time / T0))) * dt0    + (0.5 * (1. - cos(M_PI * time / T0))) * dt1;
  else if(time < T1)
    dt = (0.5 * (1. + cos(M_PI * (time - T0) / (T1 - T0)))) * dt1 + (0.5 * (1. - cos(M_PI * (time - T0) / (T1 - T0)))) * dt2;
  else
    dt = dt2;

  return dt;
}


double TimeStepTurek0(const double time) {
  return 1.;
}

bool BoundaryConditionTurek0(const std::vector < double >&x, const char name[], double &value, const int facename, const double t) {
  bool test = 1;      //dirichlet
  value = 0.;

  //const double Ubar = 0.2;    // FSI1
  const double Ubar = 2;    // FSI3
  const double L = 0.41;
  const double H = 2.5;

  if(!strcmp(name, "DY")) {
    if(2 == facename || 3 == facename) {
      test = 0;
      value = 0.;
    }
  }

  else if(!strcmp(name, "DX")) {
    if(2 == facename || 3 == facename) {    //fluid wall
      test = 0;
      value = 0.;
    }
  }

//   else if(!strcmp(name, "VX")) {
//     if(3 == facename) {    //fluid wall
//       test = 0;
//       value = 0.;
//     }
//   }
//
//   else if(!strcmp(name, "VY")) {
//     if(3 == facename) {    //fluid wall
//       test = 0;
//       value = 0.;
//     }
//   }
//
  else if(!strcmp(name, "P")) {
    if(par->_weakP && 3 != facename) {
      test = 0;
      value = 0;
    }
    else {
      value = 10;
    }

  }

  return test;

}
