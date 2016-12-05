/** \file Ex7.cpp
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
#include "XDMFWriter.hpp"
#include "NonLinearImplicitSystem.hpp"
#include "adept.h"
#include "Marker.hpp"


using namespace femus;

// bool SetBoundaryCondition(const std::vector < double >& x, const char SolName[], double& value, const int faceIndex, const double time) {
//   bool dirichlet = true; //dirichlet
//   value = 0.;
//
//   if (!strcmp(SolName, "T")) {
//     if (faceIndex == 2) {
//       value = 1.;
//     } else if (faceIndex == 3) {
//       dirichlet = false; //Neumann
//     }
//   } else if (!strcmp(SolName, "P")) {
//     dirichlet = false;
//   }
//
//   return dirichlet;
// }

unsigned dim;

void PrintLine(const std::string output_path, const std::vector< std::vector<double> > &xn, const bool &streamline = true, const unsigned &step = 0);

bool SetBoundaryCondition( const std::vector < double >& x, const char name[],
                           double& value, const int FaceName, const double time ) {
  bool test = 1; //Dirichlet
  value = 0.;
  //   cout << "Time bdc : " <<  time << endl;
  if( !strcmp( name, "U" ) ) {
    if( 1 == FaceName ) { //inflow
       double um = 1;
       double r2 = x[1]*x[1]+x[2]*x[2];
       value = (1. - r2) * um;
    }
    else if( 2 == FaceName ) { //outflow
      test = 0;
      value = 0.;
    }
    else if( 3 == FaceName ) { // no-slip fluid wall
      test = 1;
      value = 0.;
    }
  }
  else if( !strcmp( name, "V" ) ) {
    if( 1 == FaceName ) {       //inflow
      test = 1;
      value = 0.;
    }
    else if( 2 == FaceName ) {  //outflow
      test = 0;
      value = 0.;
    }
    else if( 3 == FaceName ) {  // no-slip fluid wall
      test = 1;
      value = 0.;
    }
  }
  else if( !strcmp( name, "W" ) ) {
    if( 1 == FaceName ) {      //inflow
      test = 1;
      value = 0.;
    }
    else if( 2 == FaceName ) { //outflow
      test = 0;
      value = 0.;
    }
    else if( 3 == FaceName ) { // no-slip fluid wall
      test = 1;
      value = 0.;
    }
  }
  else if( !strcmp( name, "P" ) ) {
    if( 1 == FaceName ) {
      test = 0;
      value = 0.;
    }
    else if( 2 == FaceName ) {
      test = 0;
      value = 0.;
    }
    else if( 3 == FaceName ) {
      test = 0;
      value = 0.;
    }
  }
  return test;
}

// //------------------------------------------------------------------------------------------------------------
unsigned numberOfUniformLevels;

bool SetRefinementFlag( const std::vector < double >& x, const int& elemgroupnumber, const int& level ) {

  bool refine = 0;

  //------------------ 3D --------------------------//
  //if (elemgroupnumber == 6 && level < 2) refine = 1;

  if( elemgroupnumber == 7 && level < numberOfUniformLevels ) refine = 1;

  if( elemgroupnumber == 8 && level < numberOfUniformLevels + 1 ) refine = 1;

  //------------------ 2D --------------------------//
  //if (elemgroupnumber == 7 && level < 2) refine = 1;

  //if (elemgroupnumber == 6 && level < 3) refine = 1;

  //if (elemgroupnumber == 5 && level < 4) refine = 1;

//   if (elemgroupnumber==6 && level<1) refine=1;
//   if (elemgroupnumber==7 && level<2) refine=1;
//   if (elemgroupnumber==8 && level<3) refine=1;

  return refine;

}

// //------------------------------------------------------------------------------------------------------------


void AssembleIncompressibleNavierStokes( MultiLevelProblem& mlProb );  //, unsigned level, const unsigned &levelMax, const bool &assembleMatrix );


int main( int argc, char** args ) {

  // init Petsc-MPI communicator
  FemusInit mpinit( argc, args, MPI_COMM_WORLD );

  // define multilevel mesh
  MultiLevelMesh mlMsh;
  // read coarse level mesh and generate finers level meshes
  double scalingFactor = 1.;
  //mlMsh.ReadCoarseMesh("./input/cylinder2Dnew.neu", "seventh", scalingFactor);
  mlMsh.ReadCoarseMesh( "./input/pipe.neu", "seventh", scalingFactor );
  /* "seventh" is the order of accuracy that is used in the gauss integration scheme
     probably in the furure it is not going to be an argument of this function   */
  dim = mlMsh.GetDimension();

  numberOfUniformLevels = 3;
  unsigned numberOfSelectiveLevels = 0;
  mlMsh.RefineMesh( numberOfUniformLevels + numberOfSelectiveLevels, numberOfUniformLevels , SetRefinementFlag );

  MultiLevelSolution mlSol( &mlMsh );

  // add variables to mlSol
  mlSol.AddSolution( "U", LAGRANGE, SECOND );
  mlSol.AddSolution( "V", LAGRANGE, SECOND );
  if( dim == 3 ) mlSol.AddSolution( "W", LAGRANGE, SECOND );

  //mlSol.AddSolution("P", LAGRANGE, FIRST);
  mlSol.AddSolution( "P",  DISCONTINOUS_POLYNOMIAL, FIRST );

  mlSol.AssociatePropertyToSolution( "P", "Pressure", false );
  mlSol.Initialize( "All" );

  // attach the boundary condition function and generate boundary data
  mlSol.AttachSetBoundaryConditionFunction( SetBoundaryCondition );

  mlSol.GenerateBdc( "All" );

  // define the multilevel problem attach the mlSol object to it
  MultiLevelProblem mlProb( &mlSol );

  // add system Poisson in mlProb as a Linear Implicit System
  NonLinearImplicitSystem& system = mlProb.add_system < NonLinearImplicitSystem > ( "NS" );

  system.AddSolutionToSystemPDE( "U" );
  system.AddSolutionToSystemPDE( "V" );
  if( dim == 3 ) system.AddSolutionToSystemPDE( "W" );

  system.AddSolutionToSystemPDE( "P" );

  //system.SetMgSmoother(GMRES_SMOOTHER);
  system.SetMgSmoother( ASM_SMOOTHER ); // Additive Swartz Method
  // attach the assembling function to system
  system.SetAssembleFunction( AssembleIncompressibleNavierStokes );

  system.SetMaxNumberOfNonLinearIterations(10);
  system.SetNonLinearConvergenceTolerance(1.e-8);
  system.SetMaxNumberOfResidualUpdatesForNonlinearIteration(10);
  system.SetResidualUpdateConvergenceTolerance(1.e-15);

  system.SetMgType( F_CYCLE );

  system.SetNumberPreSmoothingStep( 1 );
  system.SetNumberPostSmoothingStep( 1 );
  // initialize and solve the system
  system.init();

  system.SetSolverFineGrids(GMRES);
  system.SetTolerances( 1.e-20, 1.e-20, 1.e+50, 50, 10 );
  
//   system.SetSolverFineGrids(RICHARDSON);
//   system.SetRichardsonScaleFactor(.5);
//   system.SetTolerances( 1.e-20, 1.e-20, 1.e+50, 50, 10 );
  
  system.SetPreconditionerFineGrids( ILU_PRECOND );

  system.ClearVariablesToBeSolved();
  system.AddVariableToBeSolved( "All" );
  system.SetNumberOfSchurVariables( 1 );
  system.SetElementBlockNumber( 2 );
  //system.UseSamePreconditioner();
  system.MLsolve();

  // print solutions
  std::vector < std::string > variablesToBePrinted;
  variablesToBePrinted.push_back( "All" );

  VTKWriter vtkIO( &mlSol );
  vtkIO.SetDebugOutput( true );
  vtkIO.Write( DEFAULT_OUTPUTDIR, "biquadratic", variablesToBePrinted );

  // print mesh info
  mlMsh.PrintInfo();

  std::vector<double>x(dim,0.);
  std::cout << " --------------------------------------------------------------------------------------------- " << std::endl;
  Marker a1Quad(x, VOLUME, mlMsh.GetLevel(0), 2, true);
  //Marker a( x, VOLUME, mlMsh.GetLevel(numberOfUniformLevels + numberOfSelectiveLevels -1) );
  std::cout << " The coordinates of the marker are " << x[0] << " ," << x[1] << " ," << x[2] << std::endl;
  std::cout << " The marker type is " <<  a1Quad.GetMarkerType() << std::endl;

  double T = 30;
  unsigned n  = 100;


  std::vector < std::vector < double > > xn(n + 1);
  for(unsigned k = 0; k < n; k++) {
    a1Quad.GetMarkerCoordinates(xn[k]);
    a1Quad.Advection(mlSol.GetLevel(0), 2, T / n);
  }
  a1Quad.GetMarkerCoordinates(xn[n]);
  for(unsigned i = 0;  i < xn.size(); i++) {
    for(unsigned d = 0; d < xn[i].size(); d++) {
      std::cout << xn[i][d] << " ";
    }
    std::cout << std::endl;
  }
  PrintLine(DEFAULT_OUTPUTDIR, xn);
  
  
  

  return 0;
}


void AssembleIncompressibleNavierStokes( MultiLevelProblem& mlProb ) {
  //  mlProb is the global object from/to where get/set all the data
  //  level is the level of the PDE system to be assembled
  //  levelMax is the Maximum level of the MultiLevelProblem



  //  extract pointers to the several objects that we are going to use
  NonLinearImplicitSystem* mlPdeSys   = &mlProb.get_system<NonLinearImplicitSystem> ( "NS" ); // pointer to the linear implicit system named "Poisson"
  const unsigned level = mlPdeSys->GetLevelToAssemble();

  Mesh*           msh         = mlProb._ml_msh->GetLevel( level );  // pointer to the mesh (level) object
  elem*           el          = msh->el;  // pointer to the elem object in msh (level)

  MultiLevelSolution*   mlSol   = mlProb._ml_sol;  // pointer to the multilevel solution object
  Solution*   sol         = mlProb._ml_sol->GetSolutionLevel( level );  // pointer to the solution (level) object

  LinearEquationSolver* pdeSys        = mlPdeSys->_LinSolver[level];  // pointer to the equation (level) object


  bool assembleMatrix = mlPdeSys->GetAssembleMatrix();
  // call the adept stack object
  adept::Stack& s = FemusInit::_adeptStack;
  if( assembleMatrix ) s.continue_recording();
  else s.pause_recording();


  SparseMatrix*   KK          = pdeSys->_KK;  // pointer to the global stifness matrix object in pdeSys (level)
  NumericVector*  RES         = pdeSys->_RES; // pointer to the global residual vector object in pdeSys (level)

  const unsigned  dim = msh->GetDimension(); // get the domain dimension of the problem
  unsigned dim2 = ( 3 * ( dim - 1 ) + !( dim - 1 ) );  // dim2 is the number of second order partial derivatives (1,3,6 depending on the dimension)
  unsigned    iproc = msh->processor_id(); // get the process_id (for parallel computation)

  // reserve memory for the local standar vectors
  const unsigned maxSize = static_cast< unsigned >( ceil( pow( 3, dim ) ) );    // conservative: based on line3, quad9, hex27

  vector < unsigned > solVIndex( dim );
  solVIndex[0] = mlSol->GetIndex( "U" );  // get the position of "U" in the ml_sol object
  solVIndex[1] = mlSol->GetIndex( "V" );  // get the position of "V" in the ml_sol object
  if( dim == 3 ) solVIndex[2] = mlSol->GetIndex( "W" );   // get the position of "V" in the ml_sol object

  unsigned solVType = mlSol->GetSolutionType( solVIndex[0] );  // get the finite element type for "u"

  unsigned solPIndex;
  solPIndex = mlSol->GetIndex( "P" );  // get the position of "P" in the ml_sol object
  unsigned solPType = mlSol->GetSolutionType( solPIndex );  // get the finite element type for "u"

  vector < unsigned > solVPdeIndex( dim );
  solVPdeIndex[0] = mlPdeSys->GetSolPdeIndex( "U" );  // get the position of "U" in the pdeSys object
  solVPdeIndex[1] = mlPdeSys->GetSolPdeIndex( "V" );  // get the position of "V" in the pdeSys object
  if( dim == 3 ) solVPdeIndex[2] = mlPdeSys->GetSolPdeIndex( "W" );

  unsigned solPPdeIndex;
  solPPdeIndex = mlPdeSys->GetSolPdeIndex( "P" );  // get the position of "P" in the pdeSys object

  vector < vector < adept::adouble > >  solV( dim );  // local solution
  vector < adept::adouble >  solP; // local solution

  vector< vector < adept::adouble > > aResV( dim );  // local redidual vector
  vector< adept::adouble > aResP; // local redidual vector

  vector < vector < double > > coordX( dim );  // local coordinates
  unsigned coordXType = 2; // get the finite element type for "x", it is always 2 (LAGRANGE QUADRATIC)

  for( unsigned  k = 0; k < dim; k++ ) {
    solV[k].reserve( maxSize );
    aResV[k].reserve( maxSize );
    coordX[k].reserve( maxSize );
  }

  solP.reserve( maxSize );
  aResP.reserve( maxSize );


  vector <double> phiV;  // local test function
  vector <double> gradPhiV; // local test function first order partial derivatives
  vector <double> nablaPhiV; // local test function second order partial derivatives

  phiV.reserve( maxSize );
  gradPhiV.reserve( maxSize * dim );
  nablaPhiV.reserve( maxSize * dim2 );

  double* phiP;
  double weight; // gauss point weight

  vector< int > sysDof; // local to global pdeSys dofs
  sysDof.reserve( ( dim + 1 ) *maxSize );

  vector< double > Res; // local redidual vector
  Res.reserve( ( dim + 1 ) *maxSize );

  vector < double > Jac;
  Jac.reserve( ( dim + 1 ) *maxSize * ( dim + 1 ) *maxSize );

  if( assembleMatrix ) KK->zero();

  //BEGIN element loop for each process
  for( int iel = msh->_elementOffset[iproc]; iel < msh->_elementOffset[iproc + 1]; iel++ ) {

    short unsigned ielType = msh->GetElementType( iel );

    unsigned nDofsV = msh->GetElementDofNumber( iel, solVType );  // number of solution element dofs
    unsigned nDofsP = msh->GetElementDofNumber( iel, solPType );  // number of solution element dofs
    unsigned nDofsX = msh->GetElementDofNumber( iel, coordXType );  // number of coordinate element dofs

    unsigned nDofsTVP = dim * nDofsV + nDofsP;
    // resize local arrays
    sysDof.resize( nDofsTVP );

    for( unsigned  k = 0; k < dim; k++ ) {
      solV[k].resize( nDofsV );
      coordX[k].resize( nDofsX );
    }
    solP.resize( nDofsP );

    for( unsigned  k = 0; k < dim; k++ ) {
      aResV[k].assign( nDofsV, 0. ); //resize
    }
    aResP.assign( nDofsP, 0. ); //resize

    // local storage of global mapping and solution
    for( unsigned i = 0; i < nDofsV; i++ ) {
      unsigned solVDof = msh->GetSolutionDof( i, iel, solVType );  // global to global mapping between solution node and solution dof
      for( unsigned  k = 0; k < dim; k++ ) {
        solV[k][i] = ( *sol->_Sol[solVIndex[k]] )( solVDof );  // global extraction and local storage for the solution
        sysDof[i + k * nDofsV] = pdeSys->GetSystemDof( solVIndex[k], solVPdeIndex[k], i, iel );  // global to global mapping between solution node and pdeSys dof
      }
    }

    for( unsigned i = 0; i < nDofsP; i++ ) {
      unsigned solPDof = msh->GetSolutionDof( i, iel, solPType );  // global to global mapping between solution node and solution dof
      solP[i] = ( *sol->_Sol[solPIndex] )( solPDof );  // global extraction and local storage for the solution
      sysDof[i + dim * nDofsV] = pdeSys->GetSystemDof( solPIndex, solPPdeIndex, i, iel );  // global to global mapping between solution node and pdeSys dof
    }

    // local storage of coordinates
    for( unsigned i = 0; i < nDofsX; i++ ) {
      unsigned coordXDof  = msh->GetSolutionDof( i, iel, coordXType );  // global to global mapping between coordinates node and coordinate dof
      for( unsigned k = 0; k < dim; k++ ) {
        coordX[k][i] = ( *msh->_topology->_Sol[k] )( coordXDof );  // global extraction and local storage for the element coordinates
      }
    }

    //BEGIN a new recording of all the operations involving adept::adouble variables
    if( assembleMatrix ) s.new_recording();

    //BEGIN Gauss point loop
    for( unsigned ig = 0; ig < msh->_finiteElement[ielType][solVType]->GetGaussPointNumber(); ig++ ) {
      msh->_finiteElement[ielType][solVType]->Jacobian( coordX, ig, weight, phiV, gradPhiV, nablaPhiV );
      phiP = msh->_finiteElement[ielType][solPType]->GetPhi( ig );

      // evaluate the solution, the solution derivatives and the coordinates in the Gauss point

      vector < adept::adouble > solVig( dim, 0 );
      vector < vector < adept::adouble > > gradSolVig( dim );

      for( unsigned  k = 0; k < dim; k++ ) {
        gradSolVig[k].assign( dim, 0 );
      }

      for( unsigned i = 0; i < nDofsV; i++ ) {
        for( unsigned  k = 0; k < dim; k++ ) {
          solVig[k] += phiV[i] * solV[k][i];
        }

        for( unsigned j = 0; j < dim; j++ ) {
          for( unsigned  k = 0; k < dim; k++ ) {
            gradSolVig[k][j] += gradPhiV[i * dim + j] * solV[k][i];
          }
        }
      }

      adept::adouble solPig = 0;

      for( unsigned i = 0; i < nDofsP; i++ ) {
        solPig += phiP[i] * solP[i];
      }

      //BEGIN phiV loop (momentum)
      double nu = 0.1;
      for( unsigned i = 0; i < nDofsV; i++ ) {
        vector < adept::adouble > NSV( dim, 0. );

        for( unsigned j = 0; j < dim; j++ ) {
          for( unsigned  k = 0; k < dim; k++ ) {
            NSV[k]   +=  nu * gradPhiV[i * dim + j] * ( gradSolVig[k][j] + gradSolVig[j][k] );
            NSV[k]   +=  phiV[i] * ( solVig[j] * gradSolVig[k][j] );
          }
        }

        for( unsigned  k = 0; k < dim; k++ ) {
          NSV[k] += -solPig * gradPhiV[i * dim + k];
        }

        for( unsigned  k = 0; k < dim; k++ ) {
          aResV[k][i] += - NSV[k] * weight;
        }
      }
      //END phiV loop

      //BEGIN phiP loop (continuity)
      for( unsigned i = 0; i < nDofsP; i++ ) {
        for( int k = 0; k < dim; k++ ) {
          aResP[i] += - ( gradSolVig[k][k] ) * phiP[i]  * weight;
        }
      }
      //END phiP loop

    }
    //END Gauss point loop

    //BEGIN Extract and store the residual
    Res.resize( nDofsTVP );  //resize
    for( int i = 0; i < nDofsV; i++ ) {
      for( unsigned  k = 0; k < dim; k++ ) {
        Res[ i + k * nDofsV ] = -aResV[k][i].value();
      }
    }
    for( int i = 0; i < nDofsP; i++ ) {
      Res[ i + dim * nDofsV ] = -aResP[i].value();
    }
    RES->add_vector_blocked( Res, sysDof );
    //END Extract and store the residual

    //BEGIN Extract and store the jacobian
    if( assembleMatrix ) {
      Jac.resize( nDofsTVP * nDofsTVP );
      // define the dependent variables
      for( unsigned  k = 0; k < dim; k++ ) {
        s.dependent( &aResV[k][0], nDofsV );
      }
      s.dependent( &aResP[0], nDofsP );

      // define the independent variables
      for( unsigned  k = 0; k < dim; k++ ) {
        s.independent( &solV[k][0], nDofsV );
      }
      s.independent( &solP[0], nDofsP );

      // get the and store jacobian matrix (row-major)
      s.jacobian( &Jac[0] , true );
      KK->add_matrix_blocked( Jac, sysDof, sysDof );

      s.clear_independents();
      s.clear_dependents();
    }
    //END extract and store the jacobian
  }
  //END element loop for each process

  RES->close();
  if( assembleMatrix ) KK->close();

  // ***************** END ASSEMBLY *******************
}

#include "Files.hpp"
#include <b64/b64.h>

void PrintLine(const std::string output_path, const std::vector< std::vector<double> > &xn, const bool &streamline, const unsigned &step) {

  // *********** open vtu files *************
  std::ofstream fout;

  std::string dirnamePVTK = "./";
  Files files;
  files.CheckDir(output_path, dirnamePVTK);

  std::string filename_prefix = (streamline) ? "streamline" : "line";

  std::ostringstream filename;
  filename << output_path << "./" << filename_prefix << "." << step << ".vtu";

  fout.open(filename.str().c_str());
  if(!fout.is_open()) {
    std::cout << std::endl << " The output file " << filename.str() << " cannot be opened.\n";
    abort();
  }

  unsigned nvt = xn.size();
  unsigned nel = nvt - 1;

  const unsigned dim_array_coord [] = { nvt * 3 * sizeof(float) };
  const unsigned dim_array_conn[]   = { nel * 2 * sizeof(int) };
  const unsigned dim_array_off []   = { nel * sizeof(int) };
  const unsigned dim_array_type []  = { nel * sizeof(short unsigned) };

  unsigned buffer_size = (dim_array_coord[0] > dim_array_conn[0]) ? dim_array_coord[0] : dim_array_conn[0];
  void* buffer_void = new char [buffer_size];
  char* buffer_char = static_cast <char*>(buffer_void);

  size_t cch;
  cch = b64::b64_encode(&buffer_char[0], buffer_size , NULL, 0);
  vector <char> enc;
  enc.resize(cch);
  char* pt_char;

  // *********** write vtu header ************
  fout << "<?xml version=\"1.0\"?>" << std::endl;
  fout << "<VTKFile type = \"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">" << std::endl;
  fout << "  <UnstructuredGrid>" << std::endl;


  fout  << "    <Piece NumberOfPoints= \"" << nvt
        << "\" NumberOfCells= \"" << nel
        << "\" >" << std::endl;

  //-----------------------------------------------------------------------------------------------
  // print coordinates *********************************************Solu*******************************************
  fout  << "      <Points>" << std::endl;
  fout  << "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"binary\">" << std::endl;

  // point pointer to common mamory area buffer of void type;
  float* var_coord = static_cast< float* >(buffer_void);

  for(unsigned i = 0; i < nvt; i++) {
    for(unsigned d = 0; d < 3; d++) {
      var_coord[i * 3 + d] = (xn[i].size() > d) ? xn[i][d] : 0.;
    }
  }

  cch = b64::b64_encode(&dim_array_coord[0], sizeof(dim_array_coord), NULL, 0);
  b64::b64_encode(&dim_array_coord[0], sizeof(dim_array_coord), &enc[0], cch);
  pt_char = &enc[0];
  for(unsigned i = 0; i < cch; i++, pt_char++) fout << *pt_char;

  //print coordinates array
  cch = b64::b64_encode(&var_coord[0], dim_array_coord[0] , NULL, 0);
  b64::b64_encode(&var_coord[0], dim_array_coord[0], &enc[0], cch);
  pt_char = &enc[0];
  for(unsigned i = 0; i < cch; i++, pt_char++) fout << *pt_char;
  fout << std::endl;

  fout  << "        </DataArray>" << std::endl;
  fout  << "      </Points>" << std::endl;
  //-----------------------------------------------------------------------------------------------

  // Printing of element connectivity - offset - format type  *
  fout  << "      <Cells>" << std::endl;
  //-----------------------------------------------------------------------------------------------
  //print connectivity
  fout  << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"binary\">" << std::endl;

  // point pointer to common mamory area buffer of void type;
  int* var_conn = static_cast <int*>(buffer_void);
  unsigned icount = 0;
  for(unsigned iel = 0; iel < nel; iel++) {
    for(unsigned j = 0; j < 2; j++) {
      var_conn[icount] = iel + j;;
      icount++;
    }
  }

  //print connectivity dimension
  cch = b64::b64_encode(&dim_array_conn[0], sizeof(dim_array_conn), NULL, 0);
  b64::b64_encode(&dim_array_conn[0], sizeof(dim_array_conn), &enc[0], cch);
  pt_char = &enc[0];
  for(unsigned i = 0; i < cch; i++, pt_char++) fout << *pt_char;

  //print connectivity array
  cch = b64::b64_encode(&var_conn[0], dim_array_conn[0] , NULL, 0);
  b64::b64_encode(&var_conn[0], dim_array_conn[0], &enc[0], cch);
  pt_char = &enc[0];
  for(unsigned i = 0; i < cch; i++, pt_char++) fout << *pt_char;
  fout << std::endl;
  fout << "        </DataArray>" << std::endl;
  //------------------------------------------------------------------------------------------------
  fout  << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"binary\">" << std::endl;
  // point pointer to common memory area buffer of void type;
  int* var_off = static_cast <int*>(buffer_void);
  // print offset array
  for(int iel = 0; iel < nel; iel++) {
    var_off[iel] = (iel + 1) * 2;
  }

  //print offset dimension
  cch = b64::b64_encode(&dim_array_off[0], sizeof(dim_array_off), NULL, 0);
  b64::b64_encode(&dim_array_off[0], sizeof(dim_array_off), &enc[0], cch);
  pt_char = &enc[0];
  for(unsigned i = 0; i < cch; i++, pt_char++) fout << *pt_char;

  //print offset array
  cch = b64::b64_encode(&var_off[0], dim_array_off[0] , NULL, 0);
  b64::b64_encode(&var_off[0], dim_array_off[0], &enc[0], cch);
  pt_char = &enc[0];
  for(unsigned i = 0; i < cch; i++, pt_char++) fout << *pt_char;

  fout  << std::endl;

  fout  << "        </DataArray>" << std::endl;



  //--------------------------------------------------------------------------------------------------

  //Element format type : 23:Serendipity(8-nodes)  28:Quad9-Biquadratic
  fout  << "        <DataArray type=\"UInt16\" Name=\"types\" format=\"binary\">" << std::endl;

  // point pointer to common mamory area buffer of void type;
  unsigned short* var_type = static_cast <unsigned short*>(buffer_void);

  for(unsigned iel = 0; iel < nel; iel++) {
    var_type[iel] = 3;
  }

  //print element format dimension
  cch = b64::b64_encode(&dim_array_type[0], sizeof(dim_array_type), NULL, 0);
  b64::b64_encode(&dim_array_type[0], sizeof(dim_array_type), &enc[0], cch);
  pt_char = &enc[0];
  for(unsigned i = 0; i < cch; i++, pt_char++) fout << *pt_char;

  //print element format array
  cch = b64::b64_encode(&var_type[0], dim_array_type[0] , NULL, 0);
  b64::b64_encode(&var_type[0], dim_array_type[0], &enc[0], cch);
  pt_char = &enc[0];
  for(unsigned i = 0; i < cch; i++, pt_char++) fout << *pt_char;

  fout  << std::endl;
  fout  << "        </DataArray>" << std::endl;



  //----------------------------------------------------------------------------------------------------
//
  fout  << "      </Cells>" << std::endl;



  //-----------------------------------------------------------------------------------------------
  // Printing of element connectivity - offset - format type  *
//   fout  << "      <Cells>" << std::endl;
//   fout  << "      </Cells>" << std::endl;

  fout << "    </Piece>" << std::endl;
  fout << "  </UnstructuredGrid>" << std::endl;
  fout << "</VTKFile>" << std::endl;
  fout.close();

  delete [] var_coord;
  //--------------------------------------------------------------------------------------------------------
  return;
}


