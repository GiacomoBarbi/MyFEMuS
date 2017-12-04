/** \file Ex11.cpp
 *  \brief This example shows how to set and solve the weak form
 *   of the Boussinesq appoximation of the Navier-Stokes Equation
 *
 *  \f{eqnarray*}
 *  && \mathbf{V} \cdot \nabla T - \nabla \cdot\alpha \nabla T = 0 \\
 *  && \mathbf{V} \cdot \nabla \mathbf{V} - \nabla \cdot \nu (\nabla \mathbf{V} +(\nabla \mathbf{V})^T)
 *  +\nabla P = \beta T \mathbf{j} \\
 *  && \nabla \cdot \mathbf{V} = 0
 *  \f}
 *  in a unit box domain (in 2D and 3D) with given temperature 0 and 1 on
 *  the left and right walls, respectively, and insulated walls elsewhere.
 *  \author Eugenio Aulisa
 */

#include "FemusInit.hpp"
#include "MultiLevelProblem.hpp"
#include "NumericVector.hpp"
#include "VTKWriter.hpp"
#include "GMVWriter.hpp"
#include "NonLinearImplicitSystem.hpp"
#include "adept.h"
#include "FieldSplitTree.hpp"
#include <stdlib.h>
#include "Marker.hpp"
#include "MyVector.hpp"

double Prandtl = 1.0;
double Rayleigh = 10000.;

using namespace femus;

double InitalValueT(const std::vector < double >& x) {
  return (x[0] + 0.5);
}

bool SetBoundaryCondition(const std::vector < double >& x, const char SolName[], double& value, const int facename, const double time) {
  bool dirichlet = true; //dirichlet
  value = 0.;

  if(!strcmp(SolName, "T")) {
    if(facename == 2) {
      value = 1.;
    }
    else if(facename == 3) {
      dirichlet = false; //Neumann
    }
  }
  else if(!strcmp(SolName, "P")) {
    dirichlet = false;
  }

  return dirichlet;
}

bool SetRefinementFlag(const std::vector < double >& x, const int& elemgroupnumber, const int& level)
{
  bool refine = false;
  unsigned level0 = 0;
//   double a = static_cast<double>(rand())/RAND_MAX;
//   if ( a < 0.25) refine	= true;

//   if( fabs(x[0] - 0.5) < 0.5/ pow(2,level) && fabs(x[1] - 0.5) < 0.5/ pow(2,level) ){
//     refine = true;
//   }

//  std::cout << level << std::endl;
  double pi = acos(-1.);
  double radius = 1.0 / 2.0 * (1.0 - pow(0.5,level - level0));
  double radius2 = radius * radius;

  if ( (x[0]*x[0] + x[1] * x[1] ) > radius2 + 1.0e-6) {
//    std::cout << level << std::endl;
    refine	= true;
  }
  return refine;
}


void PrintConvergenceInfo(char *stdOutfile, char* outfile, const unsigned &numofrefinements);
void PrintNonlinearTime(char *stdOutfile, char* outfile, const unsigned &numofrefinements);

//void AssembleBoussinesqAppoximation_AD(MultiLevelProblem& ml_prob);    //, unsigned level, const unsigned &levelMax, const bool &assembleMatrix );
void AssembleBoussinesqAppoximation(MultiLevelProblem& ml_prob);

int main(int argc, char** args) {

  unsigned precType = 0;

  if(argc >= 2) {
    Prandtl = strtod(args[1], NULL);
    std::cout << Prandtl << std::endl;
  }

  if(argc >= 3) {
    Rayleigh = strtod(args[2], NULL);
    std::cout << Rayleigh << std::endl;
  }

  // init Petsc-MPI communicator
  FemusInit mpinit(argc, args, MPI_COMM_WORLD);

  // define multilevel mesh
  MultiLevelMesh mlMsh;
  // read coarse level mesh and generate finers level meshes
  double scalingFactor = 1.;
 // mlMsh.ReadCoarseMesh("./input/cube_hex.neu","seventh",scalingFactor);
  mlMsh.ReadCoarseMesh("./input/hex_cube.neu", "seventh", scalingFactor);
  /* "seventh" is the order of accuracy that is used in the gauss integration scheme
     probably in the furure it is not going to be an argument of this function   */
  unsigned dim = mlMsh.GetDimension();

  unsigned numberOfUniformLevels = 2;
  unsigned numberOfSelectiveLevels = 3;
  //mlMsh.RefineMesh(numberOfUniformLevels , numberOfUniformLevels + numberOfSelectiveLevels, NULL);
  mlMsh.RefineMesh(numberOfUniformLevels + numberOfSelectiveLevels, numberOfUniformLevels, SetRefinementFlag);
  // erase all the coarse mesh levels
  // mlMsh.EraseCoarseLevels(1);

  // print mesh info
  mlMsh.PrintInfo();

  MultiLevelSolution mlSol(&mlMsh);

  // add variables to mlSol
  mlSol.AddSolution("T", LAGRANGE, FIRST);
  mlSol.AddSolution("U", LAGRANGE, SECOND);
  mlSol.AddSolution("V", LAGRANGE, SECOND);

  if(dim == 3) mlSol.AddSolution("W", LAGRANGE, SECOND);

  mlSol.AddSolution("P",  DISCONTINOUS_POLYNOMIAL, FIRST);
  mlSol.AssociatePropertyToSolution("P", "Pressure");
  mlSol.Initialize("All");
  mlSol.Initialize("T", InitalValueT);

  // attach the boundary condition function and generate boundary data
  mlSol.AttachSetBoundaryConditionFunction(SetBoundaryCondition);
  mlSol.FixSolutionAtOnePoint("P");
  mlSol.GenerateBdc("All");

  // define the multilevel problem attach the mlSol object to it
  MultiLevelProblem mlProb(&mlSol);

  // add system Poisson in mlProb as a Linear Implicit System
  NonLinearImplicitSystem& system = mlProb.add_system < NonLinearImplicitSystem > ("NS");
  // add solution "u" to system
  system.AddSolutionToSystemPDE("U");
  system.AddSolutionToSystemPDE("V");
  if(dim == 3) system.AddSolutionToSystemPDE("W");
  system.AddSolutionToSystemPDE("P");
  system.AddSolutionToSystemPDE("T");
  system.SetMgSmoother(GMRES_SMOOTHER);

  // attach the assembling function to system
  system.SetAssembleFunction(AssembleBoussinesqAppoximation);

  system.SetMaxNumberOfNonLinearIterations(20);
  system.SetNonLinearConvergenceTolerance(1.e-8);
  //system.SetMaxNumberOfResidualUpdatesForNonlinearIteration(10);
  //system.SetResidualUpdateConvergenceTolerance(1.e-15);

  //system.SetMaxNumberOfLinearIterations(10);
  //system.SetAbsoluteLinearConvergenceTolerance(1.e-15);

  system.SetMaxNumberOfLinearIterations(1);
  system.SetAbsoluteLinearConvergenceTolerance(1.e-15);

  system.SetMgType(V_CYCLE);
  system.SetNumberPreSmoothingStep(1);
  system.SetNumberPostSmoothingStep(1);
  // initilaize and solve the system
  system.init();

  system.SetSolverFineGrids(RICHARDSON);
  system.SetRichardsonScaleFactor(.6);
  system.SetPreconditionerFineGrids(ILU_PRECOND);

  system.SetTolerances(1.e-8, 1.e-15, 1.e+50, 100, 100);//GMRES tolerances

  system.ClearVariablesToBeSolved();
  system.AddVariableToBeSolved("All");
  system.SetNumberOfSchurVariables(1);
  system.SetElementBlockNumber("All");

  system.MGsolve();
/*
  std::vector< double > x(3);
  x[0] = 0.33375; //the marker is in element 117 (proc 1)
  x[1] = 0.0627;
  x[2] = 0.;
  Marker marker(x, VOLUME, mlMsh.GetLevel(numberOfUniformLevels - 1), 2, true);
  unsigned elem = marker.GetMarkerElement();

  std::vector<double> xi;
  marker.GetMarkerLocalCoordinates(xi);

  std::cout << "marker\n";
  std::cout << elem << " " << xi[0] << " " << xi[1] << " " << GetTemperatureValue(mlProb, elem, xi) << std::endl;
*/
  // print solutions
  std::vector < std::string > variablesToBePrinted;
  variablesToBePrinted.push_back("All");

  VTKWriter vtkIO(&mlSol);
  vtkIO.Write(DEFAULT_OUTPUTDIR, "biquadratic", variablesToBePrinted);

  mlMsh.PrintInfo();

// char infile[256]="FSVTtrueResidual.txt";
// char stdOutfile[256]="output.txt";
  
  char *stdOutfile =  new char[100];
  char *outfile =  new char[100];
  sprintf(stdOutfile, "trueResidualPr=%sRa=%s.txt", args[1], args[2]);
  sprintf(outfile, "convergencePr=%sRa=%s.txt", args[1], args[2]);
  std::cout << stdOutfile << std::endl;

  PrintConvergenceInfo(stdOutfile, outfile, numberOfUniformLevels);

/*  char *stdOutfile1 = new char[100];
  char *outfile1 = new char[100];
  sprintf(stdOutfile1, "%sprintout_infoPr=%sRa=%s_time.txt", args[1], args[2], args[3]);
  sprintf(outfile1, "%scomputational_timePr=%sRa=%s_time.txt", args[1], args[2], args[3]);

  PrintNonlinearTime(stdOutfile1, outfile1, numberOfUniformLevels);*/  
  return 0;
}


void AssembleBoussinesqAppoximation(MultiLevelProblem& ml_prob) {
  //  ml_prob is the global object from/to where get/set all the data
  //  level is the level of the PDE system to be assembled
  //  levelMax is the Maximum level of the MultiLevelProblem
  //  assembleMatrix is a flag that tells if only the residual or also the matrix should be assembled

  //  extract pointers to the several objects that we are going to use
  NonLinearImplicitSystem* mlPdeSys   = &ml_prob.get_system<NonLinearImplicitSystem> ("NS");   // pointer to the linear implicit system named "Poisson"
  const unsigned level = mlPdeSys->GetLevelToAssemble();

  Mesh*           msh         = ml_prob._ml_msh->GetLevel(level);    // pointer to the mesh (level) object
  elem*           el          = msh->el;  // pointer to the elem object in msh (level)

  MultiLevelSolution*   mlSol         = ml_prob._ml_sol;  // pointer to the multilevel solution object
  Solution*   sol         = ml_prob._ml_sol->GetSolutionLevel(level);    // pointer to the solution (level) object


  LinearEquationSolver* pdeSys        = mlPdeSys->_LinSolver[level];  // pointer to the equation (level) object

  bool assembleMatrix = mlPdeSys->GetAssembleMatrix();
  // call the adept stack object

  SparseMatrix*   KK          = pdeSys->_KK;  // pointer to the global stifness matrix object in pdeSys (level)
  NumericVector*  RES         = pdeSys->_RES; // pointer to the global residual vector object in pdeSys (level)

  const unsigned  dim = msh->GetDimension(); // get the domain dimension of the problem
  unsigned dim2 = (3 * (dim - 1) + !(dim - 1));        // dim2 is the number of second order partial derivatives (1,3,6 depending on the dimension)
  unsigned    iproc = msh->processor_id(); // get the process_id (for parallel computation)

  // reserve memory for the local standar vectors
  const unsigned maxSize = static_cast< unsigned >(ceil(pow(3, dim)));          // conservative: based on line3, quad9, hex27

  //solution variable
  unsigned solTIndex;
  solTIndex = mlSol->GetIndex("T");    // get the position of "T" in the ml_sol object
  unsigned solTType = mlSol->GetSolutionType(solTIndex);    // get the finite element type for "T"

  vector < unsigned > solVIndex(dim);
  solVIndex[0] = mlSol->GetIndex("U");    // get the position of "U" in the ml_sol object
  solVIndex[1] = mlSol->GetIndex("V");    // get the position of "V" in the ml_sol object

  if(dim == 3) solVIndex[2] = mlSol->GetIndex("W");       // get the position of "V" in the ml_sol object

  unsigned solVType = mlSol->GetSolutionType(solVIndex[0]);    // get the finite element type for "u"

  unsigned solPIndex;
  solPIndex = mlSol->GetIndex("P");    // get the position of "P" in the ml_sol object
  unsigned solPType = mlSol->GetSolutionType(solPIndex);    // get the finite element type for "u"

  unsigned solTPdeIndex;
  solTPdeIndex = mlPdeSys->GetSolPdeIndex("T");    // get the position of "T" in the pdeSys object

  // std::cout << solTIndex <<" "<<solTPdeIndex<<std::endl;


  vector < unsigned > solVPdeIndex(dim);
  solVPdeIndex[0] = mlPdeSys->GetSolPdeIndex("U");    // get the position of "U" in the pdeSys object
  solVPdeIndex[1] = mlPdeSys->GetSolPdeIndex("V");    // get the position of "V" in the pdeSys object

  if(dim == 3) solVPdeIndex[2] = mlPdeSys->GetSolPdeIndex("W");

  unsigned solPPdeIndex;
  solPPdeIndex = mlPdeSys->GetSolPdeIndex("P");    // get the position of "P" in the pdeSys object

  vector < double >  solT; // local solution
  vector < vector < double > >  solV(dim);    // local solution
  vector < double >  solP; // local solution

  vector < vector < double > > coordX(dim);    // local coordinates
  unsigned coordXType = 2; // get the finite element type for "x", it is always 2 (LAGRANGE QUADRATIC)

  solT.reserve(maxSize);

  for(unsigned  k = 0; k < dim; k++) {
    solV[k].reserve(maxSize);
    coordX[k].reserve(maxSize);
  }

  solP.reserve(maxSize);

  vector <double> phiV;  // local test function
  vector <double> phiV_x; // local test function first order partial derivatives
  vector <double> phiV_xx; // local test function second order partial derivatives

  phiV.reserve(maxSize);
  phiV_x.reserve(maxSize * dim);
  phiV_xx.reserve(maxSize * dim2);

  vector <double> phiT;  // local test function
  vector <double> phiT_x; // local test function first order partial derivatives
  vector <double> phiT_xx; // local test function second order partial derivatives

  phiT.reserve(maxSize);
  phiT_x.reserve(maxSize * dim);
  phiT_xx.reserve(maxSize * dim2);

  double* phiP;
  double weight; // gauss point weight

  vector< int > sysDof; // local to global pdeSys dofs
  sysDof.reserve((dim + 2) *maxSize);

  vector< double > Res; // local redidual vector
  Res.reserve((dim + 2) *maxSize);

  vector < double > Jac;
  Jac.reserve((dim + 2) *maxSize * (dim + 2) *maxSize);

  if(assembleMatrix)
    KK->zero(); // Set to zero all the entries of the Global Matrix

  //BEGIN element loop
  for(int iel = msh->_elementOffset[iproc]; iel < msh->_elementOffset[iproc + 1]; iel++) {

    //BEGIN local dof number extraction
    unsigned nDofsT = msh->GetElementDofNumber(iel, solTType);  //temperature
    unsigned nDofsV = msh->GetElementDofNumber(iel, solVType);  //velocity
    unsigned nDofsP = msh->GetElementDofNumber(iel, solPType);  //pressure
    unsigned nDofsX = msh->GetElementDofNumber(iel, coordXType); // coordinates
    unsigned nDofsTVP = nDofsT + dim * nDofsV + nDofsP; // all solutions
    //END local dof number extraction

    //BEGIN memory allocation
    Res.resize(nDofsTVP);
    std::fill(Res.begin(), Res.end(), 0);
    Jac.resize(nDofsTVP * nDofsTVP);
    std::fill(Jac.begin(), Jac.end(), 0);
    sysDof.resize(nDofsTVP);

    solT.resize(nDofsV);

    for(unsigned  k = 0; k < dim; k++) {
      solV[k].resize(nDofsV);
      coordX[k].resize(nDofsX);
    }

    solP.resize(nDofsP);
    //END memory allocation

    //BEGIN global to local extraction
    for(unsigned i = 0; i < nDofsT; i++) { //temperature
      unsigned solTDof = msh->GetSolutionDof(i, iel, solTType);  //local to global solution dof
      solT[i] = (*sol->_Sol[solTIndex])(solTDof);  //global to local solution value
      sysDof[i] = pdeSys->GetSystemDof(solTIndex, solTPdeIndex, i, iel);  //local to global system dof
    }

    for(unsigned i = 0; i < nDofsV; i++) { //velocity
      unsigned solVDof = msh->GetSolutionDof(i, iel, solVType);  //local to global solution dof

      for(unsigned  k = 0; k < dim; k++) {
        solV[k][i] = (*sol->_Sol[solVIndex[k]])(solVDof);  //global to local solution value
        sysDof[i + nDofsT + k * nDofsV] = pdeSys->GetSystemDof(solVIndex[k], solVPdeIndex[k], i, iel);  //local to global system dof
      }
    }

    for(unsigned i = 0; i < nDofsP; i++) { //pressure
      unsigned solPDof = msh->GetSolutionDof(i, iel, solPType);  //local to global solution dof
      solP[i] = (*sol->_Sol[solPIndex])(solPDof);  //global to local solution value
      sysDof[i + nDofsT + dim * nDofsV] = pdeSys->GetSystemDof(solPIndex, solPPdeIndex, i, iel);  //local to global system dof
    }

    for(unsigned i = 0; i < nDofsX; i++) { //coordinates
      unsigned coordXDof  = msh->GetSolutionDof(i, iel, coordXType);  //local to global coordinate dof

      for(unsigned k = 0; k < dim; k++) {
        coordX[k][i] = (*msh->_topology->_Sol[k])(coordXDof);  //global to local coordinate value
      }
    }

    //END global to local extraction

    //BEGIN Gauss point loop
    short unsigned ielGeom = msh->GetElementType(iel);

    for(unsigned ig = 0; ig < msh->_finiteElement[ielGeom][solVType]->GetGaussPointNumber(); ig++) {
      // *** get gauss point weight, test function and test function partial derivatives ***
      msh->_finiteElement[ielGeom][solTType]->Jacobian(coordX, ig, weight, phiT, phiT_x, phiT_xx);
      msh->_finiteElement[ielGeom][solVType]->Jacobian(coordX, ig, weight, phiV, phiV_x, phiV_xx);
      phiP = msh->_finiteElement[ielGeom][solPType]->GetPhi(ig);

      // evaluate the solution, the solution derivatives and the coordinates in the gauss point
      double solT_gss = 0;
      vector < double > gradSolT_gss(dim, 0.);

      for(unsigned i = 0; i < nDofsT; i++) {
        solT_gss += phiT[i] * solT[i];

        for(unsigned j = 0; j < dim; j++) {
          gradSolT_gss[j] += phiT_x[i * dim + j] * solT[i];
        }
      }

      vector < double > solV_gss(dim, 0);
      vector < vector < double > > gradSolV_gss(dim);

      for(unsigned  k = 0; k < dim; k++) {
        gradSolV_gss[k].resize(dim);
        std::fill(gradSolV_gss[k].begin(), gradSolV_gss[k].end(), 0);
      }

      for(unsigned i = 0; i < nDofsV; i++) {
        for(unsigned  k = 0; k < dim; k++) {
          solV_gss[k] += phiV[i] * solV[k][i];
        }

        for(unsigned j = 0; j < dim; j++) {
          for(unsigned  k = 0; k < dim; k++) {
            gradSolV_gss[k][j] += phiV_x[i * dim + j] * solV[k][i];
          }
        }
      }

      double solP_gss = 0;

      for(unsigned i = 0; i < nDofsP; i++) {
        solP_gss += phiP[i] * solP[i];
      }


      double alpha = 1.;
      double beta = 1.;//40000.;


      double Pr = Prandtl;
      double Ra = Rayleigh;

      //BEGIN phiT_i loop: Energy balance
      for(unsigned i = 0; i < nDofsT; i++) {
        unsigned irow = i;

        for(unsigned k = 0; k < dim; k++) {
          Res[irow] +=  -alpha / sqrt(Ra * Pr) * phiT_x[i * dim + k] * gradSolT_gss[k] * weight;
          Res[irow] +=  -phiT[i] * solV_gss[k] * gradSolT_gss[k] * weight;

          if(assembleMatrix) {
            unsigned irowMat = irow * nDofsTVP;

            for(unsigned j = 0; j < nDofsT; j++) {
              Jac[ irowMat + j ] +=  alpha / sqrt(Ra * Pr) * phiT_x[i * dim + k] * phiT_x[j * dim + k] * weight;
              Jac[ irowMat + j ] +=  phiT[i] * solV_gss[k] * phiT_x[j * dim + k] * weight;
            }

            for(unsigned j = 0; j < nDofsV; j++) {
              unsigned jcol = nDofsT + k * nDofsV + j;
              Jac[ irowMat + jcol ] += phiT[i] * phiV[j] * gradSolT_gss[k] * weight;
            }
          }

        }
      }

      //END phiT_i loop

      //BEGIN phiV_i loop: Momentum balance
      for(unsigned i = 0; i < nDofsV; i++) {

        for(unsigned k = 0; k < dim; k++) {
          unsigned irow = nDofsT + k * nDofsV + i;

          for(unsigned l = 0; l < dim; l++) {
            Res[irow] +=  -sqrt(Pr / Ra) * phiV_x[i * dim + l] * (gradSolV_gss[k][l] + gradSolV_gss[l][k]) * weight;
            Res[irow] +=  -phiV[i] * solV_gss[l] * gradSolV_gss[k][l] * weight;
          }

          Res[irow] += solP_gss * phiV_x[i * dim + k] * weight;

          if(k == 1) {
            Res[irow] += beta * solT_gss * phiV[i] * weight;
          }

          if(assembleMatrix) {
            unsigned irowMat = nDofsTVP * irow;

            for(unsigned l = 0; l < dim; l++) {
              for(unsigned j = 0; j < nDofsV; j++) {
                unsigned jcol1 = (nDofsT + k * nDofsV + j);
                unsigned jcol2 = (nDofsT + l * nDofsV + j);
                Jac[ irowMat + jcol1] += sqrt(Pr / Ra) * phiV_x[i * dim + l] * phiV_x[j * dim + l] * weight;
                Jac[ irowMat + jcol2] += sqrt(Pr / Ra) * phiV_x[i * dim + l] * phiV_x[j * dim + k] * weight;
                Jac[ irowMat + jcol1] += phiV[i] * solV_gss[l] * phiV_x[j * dim + l] * weight;
                Jac[ irowMat + jcol2] += phiV[i] * phiV[j] * gradSolV_gss[k][l] * weight;
              }
            }

            for(unsigned j = 0; j < nDofsP; j++) {
              unsigned jcol = (nDofsT + dim * nDofsV) + j;
              Jac[ irowMat + jcol] += - phiV_x[i * dim + k] * phiP[j] * weight;
            }

            if(k == 1) {
              for(unsigned j = 0; j < nDofsT; j++) {
                Jac[ irowMat + j ] += -beta * phiV[i] * phiT[j] * weight;
              }
            }
          }

        }
      }

      //END phiV_i loop

      //BEGIN phiP_i loop: mass balance
      for(unsigned i = 0; i < nDofsP; i++) {
        unsigned irow = nDofsT + dim * nDofsV + i;

        for(int k = 0; k < dim; k++) {
          Res[irow] += -(gradSolV_gss[k][k]) * phiP[i]  * weight;

          if(assembleMatrix) {
            unsigned irowMat = nDofsTVP * irow;

            for(unsigned j = 0; j < nDofsV; j++) {
              unsigned jcol = (nDofsT + k * nDofsV + j);
              Jac[ irowMat + jcol ] += phiP[i] * phiV_x[j * dim + k] * weight;
            }
          }

        }
      }

      //END phiP_i loop

    }

    //END Gauss point loop

    //BEGIN local to global Matrix/Vector assembly
    RES->add_vector_blocked(Res, sysDof);

    if(assembleMatrix) {
      KK->add_matrix_blocked(Jac, sysDof, sysDof);
    }

    //END local to global Matrix/Vector assembly

  }

  //END element loop

  RES->close();

  if(assembleMatrix) {
    KK->close();
  }

  // ***************** END ASSEMBLY *******************
}

void PrintConvergenceInfo(char *stdOutfile, char* outfile, const unsigned &numofrefinements) {

  std::cout << "END_COMPUTATION\n" << std::flush;

  std::ifstream inf;
  inf.open(stdOutfile);
  if(!inf) {
    std::cout << "Redirected standard output file not found\n";
    std::cout << "add option -std_output std_out_filename > std_out_filename\n";
    return;
  }

  unsigned counter1 = 0;
  std :: vector <unsigned> Level(numofrefinements, 0);
  std :: vector <unsigned> Num_Nonlinear(numofrefinements, 0);
  std :: vector <unsigned> Num_GMRES(numofrefinements, 0);
  std :: vector <unsigned> Ave_GMRES(numofrefinements, 0);

  std::ofstream outf;
  outf.open(outfile, std::ofstream::app);
  outf << std::endl << std::endl;
  outf << "Number_of_refinements=" << numofrefinements << std::endl;
  outf << "Nonlinear_Iteration,resid_norm0,resid_normN,N,convergence";

  std::string str1;
  inf >> str1;
  while(str1.compare("END_COMPUTATION") != 0) {

    if(str1.compare("Start") == 0) {
      inf >> str1;
      if(str1.compare("Level") == 0) {
        inf >> str1;
        if(str1.compare("Max") == 0) {
          int value;
          inf >> value;
          Level[counter1] = value;
        }
      }
    }

    if(str1.compare("Nonlinear") == 0) {
      inf >> str1;
      if(str1.compare("iteration") == 0) {
        inf >> str1;
        outf << std::endl << str1;
        Num_Nonlinear[counter1] += 1;
      }
    }
    else if(str1.compare("KSP") == 0) {
      inf >> str1;
      if(str1.compare("preconditioned") == 0) {
        inf >> str1;
        if(str1.compare("resid") == 0) {
          inf >> str1;
          if(str1.compare("norm") == 0) {
            double norm0 = 1.;
            double normN = 1.;
            unsigned counter = 0;
            inf >> norm0;
            outf << "," << norm0;
            for(unsigned i = 0; i < 11; i++) {
              inf >> str1;
            }
            while(str1.compare("norm") == 0) {
              inf >> normN;
              counter++;
              for(unsigned i = 0; i < 11; i++) {
                inf >> str1;
              }
              Num_GMRES[counter1] += 1;
            }
            outf << "," << normN;
            if(counter != 0) {
              outf << "," << counter << "," << pow(normN / norm0, 1. / counter);
            }
            else {
              outf << "Invalid solver, set -outer_ksp_solver \"gmres\"";
            }
          }
        }
      }
    }

    if(str1.compare("End") == 0) {
      inf >> str1;
      if(str1.compare("Level") == 0) {
        inf >> str1;
        if(str1.compare("Max") == 0) {
          counter1 ++ ;
        }
      }
    }
    inf >> str1;
  }


  outf << "\n" << "Level, Number of nonlinear,Number of GMRES, Average number of GMRES per nonlinear" << std::endl;
  for(unsigned i = 0; i < numofrefinements; i++) {
    outf << Level[i] << "," << Num_Nonlinear[i] << ","
         << Num_GMRES[i] << "," << double(Num_GMRES[i]) / double(Num_Nonlinear[i]) << std::endl;
  }

  outf.close();
  inf.close();
}

void PrintNonlinearTime(char *stdOutfile, char* outfile, const unsigned &numofrefinements) {

  std::cout << "END_COMPUTATION\n" << std::flush;

  std::ifstream inf;
  inf.open(stdOutfile);
  if(!inf) {
    std::cout << "Redirected standard output file not found\n";
    std::cout << "add option -std_output std_out_filename > std_out_filename\n";
    return;
  }

  std::ofstream outf;
  outf.open(outfile, std::ofstream::app);
  outf << std::endl;
  outf << "\nLevel_Max, Assembly Time, Solver Time, Nonlinear Time";

  unsigned counter = 0;
  std::vector <double> Assembly_Time(numofrefinements, 0.);
  std::vector <double> Nonlinear_Time(numofrefinements, 0.);

  std::string str1;
  inf >> str1;

  while(str1.compare("END_COMPUTATION") != 0) {
    if(str1.compare("Start") == 0) {
      inf >> str1;
      if(str1.compare("Level") == 0) {
        inf >> str1;
        if(str1.compare("Max") == 0) {
          int value;
          inf >> value;
          outf << "\n" << value << ", ";
        }
      }
    }
    else if(str1.compare("ASSEMBLY") == 0) {
      inf >> str1;
      if(str1.compare("TIME:") == 0) {
        double value1;
        inf >> value1;
        Assembly_Time[counter] += value1;
      }
    }
    else if(str1.compare("Nonlinear-Cycle") == 0) {
      inf >> str1;
      if(str1.compare("TIME:") == 0) {
        double value2;
        inf >> value2;
        Nonlinear_Time[counter] += value2;
      }
    }

    if(str1.compare("End") == 0) {
      inf >> str1;
      if(str1.compare("Level") == 0) {
        inf >> str1;
        if(str1.compare("Max") == 0) {
          outf << "Assembly Time ="
               << Assembly_Time[counter] << ", Solver Time ="
               << Nonlinear_Time[counter] - Assembly_Time[counter]
               << ", Nonliear Time =" << Nonlinear_Time[counter];
          counter ++;
        }
      }
    }
    inf >> str1;
  }

  outf << std::endl;

  for(unsigned i = 0; i < numofrefinements; i++) {
    double sum1 = Assembly_Time[i];
    double sum2 = Nonlinear_Time[i];
    for(unsigned j = 0; j < i; j++) {
      sum1 = sum1 + Assembly_Time[j];
      sum2 = sum2 + Nonlinear_Time[j];
    }
    outf << "\n" << "level" << " " << i + 1 << ", "
         << "Total Nonliear Time:" << sum2
         << " = Assembly Time (" << sum1 << ") + " << "Solver Time ("
         << sum2 - sum1 << ")";
  }

  outf << std::endl;
  outf.close();
  inf.close();
}

