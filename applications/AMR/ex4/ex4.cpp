


#include "FemusInit.hpp"
#include "MultiLevelProblem.hpp"
#include "VTKWriter.hpp"
#include "adept.h"



using namespace femus;

bool SetBoundaryCondition(const std::vector < double >& x, const char SolName[], double& value, const int facename, const double time) {
  bool dirichlet = true; //dirichlet
  value = 1.;
  return dirichlet;
}


int main(int argc, char** args) {

  // init Petsc-MPI communicator
  FemusInit mpinit(argc, args, MPI_COMM_WORLD);

  
  //const elem_type * _finiteElement1 = new const elem_type_2D("tri","quadratic","seventh");
  //const elem_type * _finiteElement2 = new const elem_type_2D("tri","biquadratic","seventh");
  
  //const elem_type * _finiteElement3 = new const elem_type_3D("wedge","quadratic","seventh");
  //const elem_type * _finiteElement4 = new const elem_type_3D("wedge","biquadratic","third");
  
  const elem_type * _finiteElement5 = new const elem_type_3D("tet","quadratic","seventh");
  //const elem_type * _finiteElement6 = new const elem_type_3D("tet","biquadratic","seventh");
  
  
  
  
//   MultiLevelMesh mlMsh;
//   // read coarse level mesh and generate finers level meshes
//   double scalingFactor = 1.;
//   mlMsh.ReadCoarseMesh("./input/triangle.neu", "seventh", scalingFactor);
//   
//   mlMsh.PrintInfo();
// 
//   MultiLevelSolution mlSol(&mlMsh);
// 
//   // add variables to mlSol
// 
//   mlSol.AddSolution("V", LAGRANGE, SECOND);
//   
//   mlSol.Initialize("All");
//   
//   mlSol.AttachSetBoundaryConditionFunction(SetBoundaryCondition);
// 
//   mlSol.GenerateBdc("All");
//   
//   
//   
//   
//   
//   std::vector < std::string > variablesToBePrinted;
//   variablesToBePrinted.push_back("All");
// 
//   VTKWriter vtkIO(&mlSol);
//   vtkIO.Write(DEFAULT_OUTPUTDIR, "biquadratic", variablesToBePrinted);
  
  
  return 0;
}

