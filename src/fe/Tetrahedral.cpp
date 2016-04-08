/*=========================================================================

  Program: FEMUS
  Module: Tetrahedral
  Authors: Eugenio Aulisa
 
  Copyright (c) FEMTTU
  All rights reserved. 

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

  =========================================================================*/

//----------------------------------------------------------------------------
// includes :
//----------------------------------------------------------------------------

#include "Basis.hpp"


namespace femus {

// ************** TETRAHEDRA ***************

  const double tet_lag::X[67][3] = { 
    {0, 0, 0},        {1, 0, 0},       {0, 1, 0},   {0, 0, 1},         		//0->4
    {0.5, 0, 0},      {0.5, 0.5, 0},   {0, 0.5, 0},
    {0.,  0, 0.5},    {0.5, 0., 0.5},  {0, 0.5, 0.5},                  		//5->9
    {0.25,0,0},       {0.25,0.25,0.},  {0,0.25,0},
    {0,0,0.25},       {0.25,0,0.25},   {0,0.25,0.25},                  		//10->15
    {0.75,0.25,0},    {0.5,0.25,0},    {0.75,0,0},
    {0.75,0,0.25},    {0.5,0.25,0.25}, {0.5,0,0.25},                   		//16->21
    {0,0.75,0},       {0.25,0.5,0},    {0.25,0.75,0},
    {0,0.75,0.25},    {0,0.5,0.25},    {0.25,0.5,0.25},                		//22->27
    {0.25,0,0.5},     {0.25,0.25,0.5}, {0,0.25,0.5},
    {0,0,0.75},       {0.25,0,0.75},   {0,0.25,0.75},                  		//28->33
    {0.25,0.25,0.25},                                                  		//34  
    {1./6.,1./6.,0.}, {1./6.,0,1./6.}, {1./6.,1./6.,1./6.}, {0.,1./6.,1./6.},           //faces of 1st external triangle   
    {2./3.,1./6.,0.}, {2./3.,1./6.,1./6.}, {1./2.,1./6.,1./6.}, {2./3.,0.,1./6.},       //faces of 2nd external triangle
    {1./6.,2./3,0.}, {0.,2./3.,1./6.}, {1./6.,1./2.,1./6.}, {1./6.,2./3.,1./6.},        //faces of 3rd external triangle
    {1./6.,1./6.,1./2.}, {1./6.,0.,2./3.}, {1./6.,1./6.,2./3.}, {0.,1./6.,2./3.},       //faces of 4th external triangle
    {1./3.,0.,1./3.}, {1./3.,1./3.,1./3.}, {1./3.,1./3.,0.}, {0.,1./3.,1./3.},          //external faces of internal tetrahedra
    {1./3.,1./6.,1./6.}, {1./3.,1./6.,1./3.}, {1./6.,1./3.,1./3.}, {1./6.,1./3.,1./6.}, //internal faces of internal tetrahedra
    {1./8.,1./8.,1./8.}, //baricenter of 1st external triangle
    {5./8.,1./8.,1./8.}, //baricenter of 2nd external triangle
    {1./8.,5./8.,1./8.}, //baricenter of 3rd external triangle
    {1./8.,1./8.,5./8.}, //baricenter of 4th external triangle
    {3./8.,1./2.,1./4.}, //baricenter of 1st internal triangle
    {1./4.,1./4.,3./8.}, //baricenter of 2nd internal triangle
    {1./4.,1./4.,1./2.}, //baricenter of 3rd internal triangle
    {1./2.,3./8.,1./2.}  //baricenter of 4th internal triangle
  };

//   const int tet_lag::IND[10][3] = { 
//     {0, 0, 0},{2, 0, 0},{0, 2, 0},{0, 0, 2},
//     {1, 0, 0},{1, 1, 0},{0, 1, 0},
//     {0, 0, 1},{1, 0, 1},{0, 1, 1}
//   };
  
  const int tet_lag::IND[15][3] = { 
    {0, 0, 0},{2, 0, 0},{0, 2, 0},{0, 0, 2}, // 0-3 vertices
    {1, 0, 0},{1, 1, 0},{0, 1, 0}, 
    {0, 0, 1},{1, 0, 1},{0, 1, 1}, // 4-9 midpoints 
    {7, 7, 0},{7, 0 ,7},{7, 7, 7},{0, 7, 7}, // 10-13 faces
    {8, 8, 8} // 14 interior
  };

  
//            3
//           /|\
//          / | \
//         /  |  \
//        9   |12 8
//       /    |    \
//      /   14|  11 \
//     / 13   7      \
//    2-------|5------1
//     \      |      /
//      \     |     /
//       \    |10  /
//        6   |   4
//         \  |  /
//          \ | /
//           \|/
//            0
  
  
  
  const int tet_lag::KVERT_IND[67][2] = { 
    {0,0},{1,0},{2,0},{3,3},
    {0,1},{1,1},{2,1},
    {0,3},{1,3},{2,3},
    {0,4},{0,5},{0,6},{0,7},{0,8},{0,9},
    {1,4},{1,5},{1,6},{1,7},{1,8},{1,9},
    {2,4},{2,5},{2,6},{2,7},{2,8},{2,9},
    {3,4},{3,5},{3,6},{3,7},{3,8},{3,9},
    {4,5},
    {0,10},{0,11},{0,12},{0,13}, //faces of external tetrahedra
    {1,10},{1,11},{1,12},{1,13},
    {2,10},{2,11},{2,12},{2,13},
    {3,10},{3,11},{3,12},{3,13},
    {4,13},{5,11},{6,13},{7,12}, //external faces of internal tetrahedra
    {4,10},{4,12},{5,10},{6,12}, //internal faces of internal tetrahedra
    {0,14},{1,14},{2,14},{3,14}, // baricenters
    {4,14},{5,14},{6,14},{7,14}
  };
  
  const double tet_const::X[32][3]= {
    {0.125,0.125,0.125},{0.625,0.125,0.125},{0.125,0.625,0.125},{0.125,0.125,0.625},
    {0.375,0.125,0.250},{0.250,0.250,0.375},{0.250,0.250,0.125},{0.125,0.375,0.250},
    {0.125,0.125,0.125},{0.625,0.125,0.125},{0.125,0.625,0.125},{0.125,0.125,0.625},
    {0.375,0.125,0.250},{0.250,0.250,0.375},{0.250,0.250,0.125},{0.125,0.375,0.250},
    {0.125,0.125,0.125},{0.625,0.125,0.125},{0.125,0.625,0.125},{0.125,0.125,0.625},
    {0.375,0.125,0.250},{0.250,0.250,0.375},{0.250,0.250,0.125},{0.125,0.375,0.250},
    {0.125,0.125,0.125},{0.625,0.125,0.125},{0.125,0.625,0.125},{0.125,0.125,0.625},
    {0.375,0.125,0.250},{0.250,0.250,0.375},{0.250,0.250,0.125},{0.125,0.375,0.250}
  };

  const int tet_const::IND[4][3]= { {0,0,0}, {1,0,0}, {0,1,0}, {0,0,1} };

  const int tet_const::KVERT_IND[32][2]= {
    {0,0},{1,0},{2,0},{3,0},{4,0},{5,0},{6,0},{7,0},
    {0,1},{1,1},{2,1},{3,1},{4,1},{5,1},{6,1},{7,1},
    {0,2},{1,2},{2,2},{3,2},{4,2},{5,2},{6,2},{7,2},
    {0,3},{1,3},{2,3},{3,3},{4,3},{5,3},{6,3},{7,3}
  };
  
  // ************************************************************  
  
  double tetpwl::eval_phi(const int *I,const double* x) const {
    return (1.-I[0])*(1.-I[1])*(1.-I[2]) + 
	    x[0]*eval_dphidx(I,x) + 
	    x[1]*eval_dphidy(I,x) + 
	    x[2]*eval_dphidz(I,x);
  }

  double tetpwl::eval_dphidx(const int *I,const double* x) const {
    return I[0];
  }

  double tetpwl::eval_dphidy(const int *I,const double* x) const {
    return I[1];
  }

  double tetpwl::eval_dphidz(const int *I,const double* x) const {
    return I[2];
  }

  // ************************************************************  
   
    
    
    
  //************************************************************

  double tetLinear::eval_phi(const int *I,const double* X) const {
    const double x=X[0];   const double y=X[1];   const double z=X[2];
    const int i=I[0];      const int j=I[1];      const int k=I[2];
    return (!i*!j*!k)*(1.-x-y-z)+ !(i-2)*x + !(j-2)*y + !(k-2)*z;
  }

  double tetLinear::eval_dphidx(const int *I,const double* X) const {
    const double x=X[0];   const double y=X[1];   const double z=X[2];
    const int i=I[0];      const int j=I[1];      const int k=I[2];
    return -(!i*!j*!k) + !(i-2);
  }

  double tetLinear::eval_dphidy(const int *I,const double* X) const {
    const double x=X[0];   const double y=X[1];   const double z=X[2];
    const int i=I[0];      const int j=I[1];      const int k=I[2];
    return -(!i*!j*!k) + !(j-2);
  }

  double tetLinear::eval_dphidz(const int *I,const double* X) const {
    const double x=X[0];   const double y=X[1];   const double z=X[2];
    const int i=I[0];      const int j=I[1];      const int k=I[2];
    return -(!i*!j*!k) + !(k-2);
  }

  //************************************************************

  double  tetQuadratic::eval_phi(const int *I,const double* X) const {
    
    const double x=X[0];   const double y=X[1];   const double z=X[2];
    const int i=I[0];      const int j=I[1];      const int k=I[2];
    double t=1.-(x+y+z);
 
    return
      !i     *( !j     *( !k*t*(2.*t-1.) + !(k-1)*4.*z*t + !(k-2)*(-z+2.*z*z) )+
		!(j-1) *( !k*4.*y*t      + !(k-1)*4.*y*z		      )+
		!(j-2) *( !k*(-y+2.*y*y) 				      ) )+
      !(i-1) *( !j     *( !k*4.*x*t      + !(k-1)*4.*x*z		      )+
		!(j-1) *( !k*4.*x*y					      ) )+
      !(i-2) *( !j     *( !k*(-x+2.*x*x)				      ) );
  }

  double  tetQuadratic::eval_dphidx(const int *I,const double* X) const {
    
    const double x=X[0];   const double y=X[1];   const double z=X[2];
    const int i=I[0];      const int j=I[1];      const int k=I[2];
    double t=1.-(x+y+z);
  
    return
      !i    *( !j     *( !k*(-4.*t+1.) + !(k-1)*(-4.)*z )+
	       !(j-1) *( !k*(-4.)*y 		      	) )+
      !(i-1)*( !j     *( !k*4.*(t-x)   + !(k-1)*4.*z    )+
	       !(j-1) *( !k*4.*y			) )+
      !(i-2)*( !j     *( !k*(-1.+4.*x)		      	) );
  }

  double  tetQuadratic::eval_dphidy(const int *I,const double* X) const {
    
    const double x=X[0];   const double y=X[1];   const double z=X[2];
    const int i=I[0];      const int j=I[1];      const int k=I[2];
    double t=1.-(x+y+z);
  
    return
      !i     *( !j     *( !k*(-4.*t+1.) + !(k-1)*(-4.)*z )+
	        !(j-1) *( !k*4.*(t-y)   + !(k-1)*4.*z    )+
	        !(j-2) *( !k*(-1.+4.*y)                  ) )+
      !(i-1) *( !j     *( !k*(-4.)*x	                 )+
	        !(j-1) *( !k*4.*x		         ) );
  }

  double  tetQuadratic::eval_dphidz(const int *I,const double* X) const {
    
    const double x=X[0];   const double y=X[1];   const double z=X[2];
    const int i=I[0];      const int j=I[1];      const int k=I[2];
    double t=1.-(x+y+z);
  
    return
      !i     *( !j     *( !k*(-4.*t+1.) + !(k-1)*4.*(t-z) + !(k-2)*(-1+4.*z) )+
		!(j-1) *( !k*(-4.)*y    + !(k-1)*4.*y			     ) )+
      !(i-1) *( !j     *( !k*(-4.)*x    + !(k-1)*4.*x			     ) );
    }

  double  tetQuadratic::eval_d2phidx2(const int *I,const double* X) const {
  
    const int i=I[0];  const int j=I[1];  const int k=I[2];
    return
      !i    *( !j *( !k*(4.)   ) )+
      !(i-1)*( !j *( !k*(-8.)  ) )+
      !(i-2)*( !j *( !k*(4.)   ) );   
  }

  double  tetQuadratic::eval_d2phidy2(const int *I,const double* X) const {
  
    const int i=I[0];  const int j=I[1];  const int k=I[2];
    return
      !i  *( !j     *( !k*(4.)  )+
	     !(j-1) *( !k*(-8.) )+
	     !(j-2) *( !k*(4.)  ) );
  }

  double  tetQuadratic::eval_d2phidz2(const int *I,const double* X) const {
  
    const int i=I[0];  const int j=I[1];  const int k=I[2];
    return
      !i *( !j *( !k*(4.) + !(k-1)*(-8.) + !(k-2)*(4.) ));
  }

  double  tetQuadratic::eval_d2phidxdy(const int *I,const double* X) const {
  
    const int i=I[0]; const int j=I[1];  const int k=I[2];
    return
      !i    *( !j     *( !k*(4.)  ) + !(j-1) *( !k*(-4.) ) )+
      !(i-1)*( !j     *( !k*(-4.) ) + !(j-1) *( !k*(4.)  ) );
  }

  double  tetQuadratic::eval_d2phidydz(const int *I,const double* X) const {
 
    const int i=I[0];  const int j=I[1];  const int k=I[2];
    return
      !i   *( !j     *( !k*(4.)   + !(k-1)*(-4.) )+
	      !(j-1) *( !k*(-4.)  + !(k-1)*(4.)	) );
  }

  double  tetQuadratic::eval_d2phidzdx(const int *I,const double* X) const {
 
    const int i=I[0];  const int j=I[1];  const int k=I[2];
    return
      !i    *( !j *( !k*(4.)  + !(k-1)*(-4.) ) )+
      !(i-1)*( !j *( !k*(-4.) + !(k-1)*4.    ) );
  
  } 

//************************************************************

  double  tetBiquadratic::eval_phi(const int *I,const double* X) const {
    
    const double x=X[0];   const double y=X[1];   const double z=X[2];
    const int i=I[0];      const int j=I[1];      const int k=I[2];
    double t=1.-(x+y+z);
 
    return
      !i     *( !j     *( !k*(t*(2.*t-1.)+3.*x*y*t+3.*x*z*t+3.*y*z*t-4.*x*y*z*t) +               //f0
                          !(k-1)*(4.*z*t-12.*z*y*t-12.*x*z*t+32.*x*y*z*t) +                      //f7
                          !(k-2)*((-z+2.*z*z)+3.*x*y*z+3.*x*z*t+3.*y*z*t-4.*x*y*z*t) ) +         //f3
		!(j-1) *( !k*(4.*y*t-12.*x*y*t-12.*y*z*t+32.*x*y*z*t) +                          //f6
		          !(k-1)*(4.*y*z-12.*y*z*t-12.*x*y*z+32.*x*y*z*t) ) +                    //f9
		!(j-2) *( !k*((-y+2.*y*y)+3.*x*y*t+3.*x*z*y+3.*y*z*t-4.*x*y*z*t) ) +             //f1
		!(j-7) *( !(k-7)*(27.*z*y*t-27.*4.*x*y*z*t) ) ) +                                //f13           
      !(i-1) *( !j     *( !k*(4.*x*t-12.*x*y*t-12.*x*z*t+32.*x*y*z*t) +                          //f4
                          !(k-1)*(4.*x*z-12.*x*z*t-12.*x*z*y+32.*x*y*z*t) ) +                    //f8
		!(j-1) *( !k*(4.*x*y-12.*x*y*t-12.*x*z*y+32.*x*y*z*t) ) ) +                      //f5
      !(i-2) *( !j     *( !k*((-x+2.*x*x)+3.*x*y*t+3.*x*z*t+3.*x*y*z-4.*x*y*z*t) ) ) +           //f2
      !(i-7) *( !j     *( !(k-7)*(27.*x*z*t-27.*4.*x*y*z*t) ) +                                  //f11
                !(j-7) *( !(k)*(27.*x*y*t-27.*4.*x*y*z*t)  +                                     //f10
                          !(k-7)*(27.*x*y*z-27.*4.*x*y*z*t) ) ) +                                //f12                              
      !(i-8) *( !(j-8) *( !(k-8)*(256.*x*y*z*t) ) );                                             //f14
  }

  double  tetBiquadratic::eval_dphidx(const int *I,const double* X) const {
    
    const double x=X[0];   const double y=X[1];   const double z=X[2];
    const int i=I[0];      const int j=I[1];      const int k=I[2];
    double t=1.-(x+y+z);
  
    return
      !i     *( !j     *( !k*(-4.*t+1. + 3.*y*(t-x) + 3.*z*(t-x) + 3.*(-y*z) - 4.*y*z*(t-x)) +         //d(f0)/dx          
                          !(k-1)*(-4.*z + 12.*y*z - 12.*z*(t-x) + 32.*y*z*(t-x)) +                 //d(f7)/dx
                          !(k-2)*(3.*y*z + 3.*z*(t-x) + 3.*(-y*z) - 4.*y*z*(t-x)) ) +                  //d(f3)/dx
		!(j-1) *( !k*(-4.*y - 12.*y*(t-x) + 12.*(y*z) + 32.*y*z*(t-x)) +                       //d(f6)/dx
		          !(k-1)*32.*y*z*(t-x) ) +                                                     //d(f9)/dx
		!(j-2) *( !k*(3.*y*(t-x) - 4.*y*z*(t-x) ) ) +                                          //d(f1)/dx
		!(j-7) *( !(k-7)*(-27.*y*z - 27.*4.*y*z*(t-x)) ) ) +                                   //d(f13)/dx           
      !(i-1) *( !j     *( !k*(4.*(t-x) - 12.*y*(t-x) - 12.*z*(t-x) + 32.*y*z*(t-x)) +                  //d(f4)/dx
                          !(k-1)*(4.*z - 12.*z*(t-x) - 12.*z*y + 32.*y*z*(t-x)) ) +                    //d(f8)/dx
		!(j-1) *( !k*(4.*y - 12.*y*(t-x) - 12.*z*y + 32.*y*z*(t-x)) ) ) +                      //d(f5)/dx
      !(i-2) *( !j     *( !k*(-1.+4.*x + 3.*y*(t-x) + 3.*z*(t-x) + 3.*y*z - 4.*y*z*(t-x)) ) ) +        //d(f2)/dx
      !(i-7) *( !j     *( !(k-7)*(27.*z*(t-x) - 27.*4.*y*z*(t-x) ) ) +                                 //d(f11)/dx
                !(j-7) *( !(k)*(27.*y*(t-x) - 27.*4.*y*z*(t-x))  +                                     //d(f10)/dx
                          !(k-7)*(27.*y*z - 27.*4.*y*z*(t-x)) ) ) +                                    //d(f12)/dx                              
      !(i-8) *( !(j-8) *( !(k-8)*(256.*y*z*(t-x)) ) );                                                 //d(f14)/dx
  }
  
  double  tetBiquadratic::eval_dphidy(const int *I,const double* X) const {
    
    const double x=X[0];   const double y=X[1];   const double z=X[2];
    const int i=I[0];      const int j=I[1];      const int k=I[2];
    double t=1.-(x+y+z);
  
    return
      !i     *( !j     *( !k*(-4.*t+1 + 3.*x*(t-y) - 3.*x*z + 3.*z*(t-y) - 4.*x*z*(t-y)) +          //d(f0)/dy
                          !(k-1)*(-4.*z - 12.*z*(t-y) + 12.*x*z + 32.*x*z*(t-y)) +                  //d(f7)/dy
                          !(k-2)*(3.*z*(t-y) - 4.*x*z*(t-y)) ) +                                    //d(f3)/dy
		!(j-1) *( !k*(4.*(t-y) - 12.*x*(t-y) - 12.*z*(t-y) + 32.*x*z*(t-y)) +               //d(f6)/dy
		          !(k-1)*(4.*z - 12.*z*(t-y) - 12.*x*z + 32.*x*z*(t-y)) ) +                 //d(f9)/dy
		!(j-2) *( !k*(-1.+4.*y + 3.*x*(t-y) + 3.*x*z + 3.*z*(t-y) - 4.*x*z*(t-y)) ) +       //d(f1)/dy
		!(j-7) *( !(k-7)*(27.*z*(t-y) - 27.*4.*x*z*(t-y)) ) ) +                             //d(f13)/dy           
      !(i-1) *( !j     *( !k*(-4.*x - 12.*x*(t-y) + 12.*x*z + 32.*x*z*(t-y)) +                      //d(f4)/dy
                          !(k-1)*(32.*x*z*(t-y)) ) +                                                //d(f8)/dy
		!(j-1) *( !k*(4.*x - 12.*x*(t-y) - 12.*x*z + 32.*x*z*(t-y)) ) ) +                   //d(f5)/dy
      !(i-2) *( !j     *( !k*(3.*x*(t-y) - 4.*x*z*(t-y)) ) ) +                                      //d(f2)/dy
      !(i-7) *( !j     *( !(k-7)*(-27.*x*z - 27.*4.*x*z*(t-y)) ) +                                  //d(f11)/dy
                !(j-7) *( !(k)*(27.*x*(t-y) - 27.*4.*x*z*(t-y))  +                                  //d(f10)/dy
                          !(k-7)*(27.*x*z - 27.*4.*x*z*(t-y)) ) ) +                                 //d(f12)/dy                              
      !(i-8) *( !(j-8) *( !(k-8)*(256.*x*z*(t-y)) ) );                                              //d(f14)/dy       
  }
    
  double  tetBiquadratic::eval_dphidz(const int *I,const double* X) const { 
    
    const double x=X[0];   const double y=X[1];   const double z=X[2];
    const int i=I[0];      const int j=I[1];      const int k=I[2];
    double t=1.-(x+y+z);
  
    return
      !i     *( !j     *( !k*(-4.*t+1. - 3.*x*y + 3.*x*(t-z) + 3.*y*(t-z) - 4.*x*y*(t-z) ) +              //d(f0)/dz
                          !(k-1)*(4.*(t-z) - 12.*y*(t-z) - 12.*x*(t-z) + 32.*x*y*(t-z)) +                    //d(f7)/dz
                          !(k-2)*((-1.+4.*z + 3.*x*y + 3.*x*(t-z) + 3.*y*(t-z) - 4.*x*y*(t-z)) ) ) +       //d(f3)/dz
		!(j-1) *( !k*(-4.*y + 12.*x*y - 12.*y*(t-z) + 32.*x*y*(t-z)) +                           //d(f6)/dz
		          !(k-1)*(4.*y - 12.*y*(t-z) - 12.*x*y + 32.*x*y*(t-z)) ) +                      //d(f9)/dz
		!(j-2) *( !k*(3.*y*(t-z) - 4.*x*y*(t-z)) ) +                                             //d(f1)/dz
		!(j-7) *( !(k-7)*(27.*y*(t-z) - 27.*4.*x*y*(t-z)) ) ) +                                  //d(f13)/dz           
      !(i-1) *( !j     *( !k*(-4.*x + 12.*x*y - 12.*x*(t-z) + 32.*x*y*(t-z)) +                           //d(f4)/dz
                          !(k-1)*(4.*x - 12.*x*(t-z) - 12.*x*y + 32.*x*y*(t-z)) ) +                      //d(f8)/dz
		!(j-1) *( !k*(32.*x*y*(t-z)) ) ) +                                                       //d(f5)/dz
      !(i-2) *( !j     *( !k*(3.*x*(t-z) - 4.*x*y*(t-z))	) ) +                                    //d(f2)/dz
      !(i-7) *( !j     *( !(k-7)*(27.*x*(t-z) - 27.*4.*x*y*(t-z)) ) +                                    //d(f11)/dz
                !(j-7) *( !(k)*(-27.*x*y - 27.*4.*x*y*(t-z))  +                                          //d(f10)/dz
                          !(k-7)*(27.*x*y - 27.*4.*x*y*(t-z)) ) ) +                                      //d(f12)/dz                      
      !(i-8) *( !(j-8) *( !(k-8)*(256.*x*y*(t-z)) ) ) ;                                                 //d(f14)/dz
    }  
  
  double  tetBiquadratic::eval_d2phidx2(const int *I,const double* X) const {
    
    const double y=X[1];   const double z=X[2];
    const int i=I[0];  const int j=I[1];  const int k=I[2];
    return
      !i     *( !j     *( !k*(4 - 6.*y - 6.*z + 8.*y*z) +                                  //d^2(f0)/dx^2
                          !(k-1)*(24.*z - 64.*y*z) +                                       //d^2(f7)/dx^2
                          !(k-2)*(-6.*z + 8.*y*z) ) +                                      //d^2(f3)/dx^2
		!(j-1) *( !k*(24.*y - 64.*y*z) +                                           //d^2(f6)/dx^2
		          !(k-1)*(-64.*y*z) ) +                                            //d^2(f9)/dx^2
		!(j-2) *( !k*(-6.*y + 8.*y*z) ) +                                          //d^2(f1)/dx^2
		!(j-7) *( !(k-7)*(27.*8.*y*z) ) ) +                                        //d^2(f13)/dx^2           
      !(i-1) *( !j     *( !k*(-8. + 24.*y + 24.*z - 64.*y*z) +                              //d^2(f4)/dx^2
                          !(k-1)*(24.*z - 64.*y*z) ) +                                     //d^2(f8)/dx^2
		!(j-1) *( !k*(24.*y - 64.*y*z) ) ) +                                       //d^2(f5)/dx^2
      !(i-2) *( !j     *( !k*(4.- 6.*y - 6.*z + 8.*y*z) ) ) +                              //d^2(f2)/dx^2
      !(i-7) *( !j     *( !(k-7)*(-2.*27.*z + 27.*8.*y*z) ) +                              //d^2(f11)/dx^2
                !(j-7) *( !(k)*(-2.*27.*y + 27.*8.*y*z)  +                                 //d^2(f10)/dx^2
                          !(k-7)*(27.*8.*y*z) ) ) +                                        //d^2(f12)/dx^2                  
      !(i-8) *( !(j-8) *( !(k-8)*(-2.*256.*y*z) ) );                                       //d^2(f14)/dx^2
  }
  
  double  tetBiquadratic::eval_d2phidy2(const int *I,const double* X) const {
    
    const double x=X[0];   const double z=X[2];
    const int i=I[0];  const int j=I[1];  const int k=I[2];
    return
      !i     *( !j     *( !k*(4. - 6.*x - 6.*z + 8.*x*z) +                            //d^2(f0)/dy^2
                          !(k-1)*(24.*z - 64.*x*z) +                                  //d^2(f7)/dy^2
                          !(k-2)*(-6.*z + 8.*x*z) ) +                                 //d^2(f3)/dy^2
		!(j-1) *( !k*(-8. + 24.*x + 24.*z - 64.*x*z) +                        //d^2(f6)/dy^2
		          !(k-1)*(24.*z - 64.*x*z) ) +                                //d^2(f9)/dy^2
		!(j-2) *( !k*(4. - 6.*x - 6.*z + 8.*x*z) ) +                          //d^2(f1)/dy^2
		!(j-7) *( !(k-7)*(-2.*27.*z + 27.*8.*x*z) ) ) +                       //d^2(f13)/dy^2
      !(i-1) *( !j     *( !k*(24.*x - 64.*x*z) +                                      //d^2(f4)/dy^2
                          !(k-1)*(-64.*x*z) ) +                                       //d^2(f8)/dy^2
		!(j-1) *( !k*(24.*x -64.*x*z) ) ) +                                   //d^2(f5)/dy^2
      !(i-2) *( !j     *( !k*(-6.*x + 8.*x*z) ) ) +                                   //d^2(f2)/dy^2
      !(i-7) *( !j     *( !(k-7)*(27.*8.*x*z) ) +                                     //d^2(f11)/dy^2
                !(j-7) *( !(k)*(-2.*27.*x + 27.*8.*x*z)  +                            //d^2(f10)/dy^2
                          !(k-7)*(27.*8.*x*z) ) ) +                                   //d^2(f12)/dy^2                  
      !(i-8) *( !(j-8) *( !(k-8)*(-2.*256.*x*z) ) );                                  //d^2(f14)/dy^2
  }

  double  tetBiquadratic::eval_d2phidz2(const int *I,const double* X) const {
    
    const double x=X[0];   const double y=X[1];
    const int i=I[0];  const int j=I[1];  const int k=I[2];
    return
      !i     *( !j     *( !k*(4. - 6.*x - 6.*y + 8.*x*y) +                            //d^2(f0)/dz^2
                          !(k-1)*(-8. + 24.*y + 24.*x - 64.*x*y) +                    //d^2(f7)/dz^2
                          !(k-2)*(4. - 6.*x - 6.*y + 8.*x*y) ) +                      //d^2(f3)/dz^2
		!(j-1) *( !k*(24.*y - 64.*x*y) +                                      //d^2(f6)/dz^2
		          !(k-1)*(24.*y - 64.*x*y) ) +                                //d^2(f9)/dz^2
		!(j-2) *( !k*(-6.*y + 8.*x*y) ) +                                     //d^2(f1)/dz^2
		!(j-7) *( !(k-7)*(-54.*y + 27.*8.*x*y) ) ) +                          //d^2(f13)/dz^2           
      !(i-1) *( !j     *( !k*(24.*x - 64.*x*y) +                                      //d^2(f4)/dz^2
                          !(k-1)*(24.*x - 64.*x*y) ) +                                //d^2(f8)/dz^2
		!(j-1) *( !k*(-64.*x*y) ) ) +                                         //d^2(f5)/dz^2
      !(i-2) *( !j     *( !k*(-6.*x + 8.*x*y) ) ) +                                   //d^2(f2)/dz^2
      !(i-7) *( !j     *( !(k-7)*(-2.*27.*x + 27.*8.*x*y) ) +                         //d^2(f11)/dz^2
                !(j-7) *( !(k)*(27.*8.*x*y)  +                                        //fd^2(f10)/dz^2
                          !(k-7)*(27.*8.*x*y) ) ) +                                   //fd^2(f12)/dz^2                  
      !(i-8) *( !(j-8) *( !(k-8)*(-2.*256.*x*y) ) );                                  //fd^2(f14)/dz^2
  }

  double  tetBiquadratic::eval_d2phidxdy(const int *I,const double* X) const {
  
    const double x=X[0];   const double y=X[1];   const double z=X[2];
    const int i=I[0];      const int j=I[1];      const int k=I[2];
    double t=1.-(x+y+z);
    return
      !i     *( !j     *( !k*(4. + 3.*(t-x-y) - 6.*z - 4.*z*(t-x-y)) +                           //d^2(f0)/dxdy
                          !(k-1)*( 24.*z + 32.*z*(t-x-y)) +                                      //d^2(f7)/dxdy
                          !(k-2)*(-3.*z - 4.*z*(t-x-y)) ) +                                      //d^2(f3)/dxdy
		!(j-1) *( !k*(-4. - 12.*(t-x-y) + 12.*z + 32.*z*(t-x-y)) +                       //d^2(f6)/dxdy
		          !(k-1)*(32.*z*(t-x-y)) ) +                                             //d^2(f9)/dxdy
		!(j-2) *( !k*(3.*(t-x-y) - 4.*z*(t-x-y)) ) +                                     //d^2(f1)/dxdy
		!(j-7) *( !(k-7)*(-27.*z - 27.*4.*z*(t-x-y)) ) ) +                               //d^2(f13)/dxdy
      !(i-1) *( !j     *( !k*(-4. - 12.*(t-x-y) + 12.*z + 32.*z*(t-x-y)) +                       //d^2(f4)/dxdy
                          !(k-1)*(32.*z*(t-x-y)) ) +                                             //d^2(f8)/dxdy
		!(j-1) *( !k*(4. - 12.*(t-x-y) - 12.*z + 32.*z*(t-x-y)) ) ) +                    //d^2(f5)/dxdy
      !(i-2) *( !j     *( !k*((3.*(t-x-y) - 4.*z*(t-x-y)) ) ) ) +                                //d^2(f2)/dxdy
      !(i-7) *( !j     *( !(k-7)*(-27.*z - 27.*4.*z*(t-x-y)) ) +                                 //d^2(f11)/dxdy
                !(j-7) *( !(k)*(27.*(t-x-y) - 27.*4.*z*(t-x-y))  +                               //fd^2(f10)/dxdy
                          !(k-7)*(27.*z - 27.*4.*z*(t-x-y)) ) ) +                                //d^2(f12)/dxdy                  
      !(i-8) *( !(j-8) *( !(k-8)*(256.*z*(t-x-y)) ) );                                           //d^2(f14)/dxdy
  }
  
  double  tetBiquadratic::eval_d2phidydz(const int *I,const double* X) const {
 
    const double x=X[0];   const double y=X[1];   const double z=X[2];
    const int i=I[0];      const int j=I[1];      const int k=I[2];
    double t=1.-(x+y+z);
    return
      !i     *( !j     *( !k*(4. - 6.*x + 3.*(t-y-z) - 4.*x*(t-y-z)) +                           //d^2(f0)/dydz
                          !(k-1)*(-4.- 12*(t-y-z)+ 12.*x +32.*x*(t-y-z)) +                                  //d^2(f7)/dydz
                          !(k-2)*(3.*(t-y-z) - 4.*x*(t-y-z)) ) +                                 //d^2(f3)/dydz
		!(j-1) *( !k*(-4. + 12.*x - 12.*(t-y-z) + 32.*x*(t-y-z)) +                       //d^2(f6)/dydz
		          !(k-1)*(4. - 12.*(t-y-z) - 12.*x + 32.*x*(t-y-z)) ) +                  //d^2(f9)/dydz
		!(j-2) *( !k*(-3.*x + 3.*x + 3.*(t-y-z) -4.*x*(t-y-z)) ) +                       //d^2(f1)/dydz
		!(j-7) *( !(k-7)*(27.*(t-y-z) - 27.*4.*x*(t-y-z)) ) ) +                          //fd^2(f13)/dydz
      !(i-1) *( !j     *( !k*(24.*x + 32.*x*(t-y-z)) +                                           //d^2(f4)/dydz
                          !(k-1)*(32.*x*(t-y-z)) ) +                                             //d^2(f8)/dydz
		!(j-1) *( !k*(32.*x*(t-y-z)) ) ) +                                               //d^2(f5)/dydz
      !(i-2) *( !j     *( !k*(-3.*x - 4.*x*(t-y-z)) ) ) +                                        //d^2(f2)/dydz
      !(i-7) *( !j     *( !(k-7)*(-27.*x - 27.*4.*x*(t-y-z)) ) +                                 //d^2(f11)/dydz
                !(j-7) *( !(k)*(-27.*x - 27.*4.*x*(t-y-z))  +                                    //d^2(f10)/dydz
                          !(k-7)*(27.*x - 27.*4.*x*(t-y-z)) ) ) +                                //d^2(f12)/dydz                  
      !(i-8) *( !(j-8)  *( !(k-8)*(256.*x*(t-y-z)) ) );                                          //d^2(14)/dydz
  }
  
  double  tetBiquadratic::eval_d2phidzdx(const int *I,const double* X) const { 
 
    const double x=X[0];   const double y=X[1];   const double z=X[2];
    const int i=I[0];      const int j=I[1];      const int k=I[2];
    double t=1.-(x+y+z);
    return
      !i     *( !j     *( !k*(4. - 6.*y + 3.*(t-x-z) - 4.*y*(t-x-z)) +                           //d^2(f0)/dzdx
                          !(k-1)*(-4. + 12.*y - 12.*(t-x-z) + 32.*y*(t-x-z)) +                   //d^2(f7)/dzdx
                          !(k-2)*(3.*(t-x-z) - 4.*y*(t-x-z)) ) +                                 //d^2(f3)/dzdx
		!(j-1) *( !k*(24.*y+ 32.*y*(t-x-z)) +                                            //d^2(f6)/dzdx
		          !(k-1)*(32.*y*(t-x-z)) ) +                                             //d^2(f9)/dzdx
		!(j-2) *( !k*(-3.*y - 4.*y*(t-x-z)) ) +                                          //d^2(f1)/dzdx
		!(j-7) *( !(k-7)*(-27.*y - 27.*4.*y*(t-x-z)) ) ) +                               //d^2(f13)/dzdx
      !(i-1) *( !j     *( !k*(-4. + 12.*y - 12.*(t-x-z) + 32.*y*(t-x-z)) +                       //d^2(f4)/dzdx
                          !(k-1)*(4. - 12.*(t-x-z) -  12.*y + 32.*y*(t-x-z)) ) +                 //d^2(f8)/dzdx
		!(j-1) *( !k*(32.*y*(t-x-z)) ) ) +                                               //d^2(f5)/dzdx
      !(i-2) *( !j     *( !k*(3.*(t-x-z) - 4.*y*(t-x-z)) ) ) +                                   //d^2(f2)/dzdx
      !(i-7) *( !j     *( !(k-7)*(27.*(t-x-z) - 27.*4.*y*(t-x-z)) ) +                            //d^2(f11)/dzdx
                !(j-7) *( !(k)*(-27.*y - 27.*4.*y*(t-x-z))  +                                    //d^2(f10)/dzdx
                          !(k-7)*(27.*y - 27.*4.*y*(t-x-z)) ) ) +                                //d^2(f12)/dzdx                  
      !(i-8) *( !(j-8) *( !(k-8)*(256.*y*(t-x-z)) ) );                                           //d^2(f14)/dzdx
  
  }

} //end namespace femus


