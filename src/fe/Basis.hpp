/*=========================================================================

  Program: FEMUS
  Module: basis
  Authors: Eugenio Aulisa, Giorgio Bornia
 
  Copyright (c) FEMTTU
  All rights reserved. 

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

  =========================================================================*/

/**
 * This class contains the fe basis function and their derivatives
 */


#ifndef __femus_fe_Basis_hpp__
#define __femus_fe_Basis_hpp__

#include <iostream>
#include <stdlib.h>  

namespace femus {
    
  class basis {
  public:
    
    const int _nc,_nf,_nlag0,_nlag1,_nlag2;
    
    basis(const int &nc,const int &nf,const int &nlag0, const int &nlag1, const int &nlag2):
      _nc(nc),
      _nf(nf),
      _nlag0(nlag0),
      _nlag1(nlag1),
      _nlag2(nlag2){ };
           
    virtual void PrintType() const = 0 ;
    virtual double eval_phi(const int *I,const double* x) const {
      std::cout<<"Error this phi is not available for this element \n"; abort();
    };
    virtual double eval_dphidx(const int *I,const double* x) const {
      std::cout<<"Error this dphidx is not available for this element dimension\n"; abort();
    };
    virtual double eval_dphidy(const int *I,const double* x) const {
      std::cout<<"Error this dphidy is not available for this element dimension\n"; abort();
    };
    virtual double eval_dphidz(const int *I,const double* x) const {
      std::cout<<"Error this dphidz is not available for this element dimension\n"; abort();
    };
    virtual double eval_d2phidx2(const int *I,const double* x) const {
      std::cout<<"Error this d2phix2 is not available for this element dimension\n"; abort();
    };
    virtual double eval_d2phidy2(const int *I,const double* x) const {
      std::cout<<"Error this d2phidy2 is not available for this element dimension\n"; abort();
    };
    virtual double eval_d2phidz2(const int *I,const double* x) const {
      std::cout<<"Error this d2phidz2 is not available for this element dimension\n"; abort();
    };
    virtual double eval_d2phidxdy(const int *I,const double* x) const {
      std::cout<<"Error this d2phidxdy is not available for this element dimension\n"; abort();
    };
    virtual double eval_d2phidydz(const int *I,const double* x) const {
      std::cout<<"Error this d2phidydz is not available for this element dimension\n"; abort();
    };
    virtual double eval_d2phidzdx(const int *I,const double* x) const {
      std::cout<<"Error this d2phidydz is not available for this element dimension\n"; abort();
    };
    
    virtual const double* getX(const int &i) const = 0;
    virtual const int* getIND(const int &i) const = 0;
    virtual const int* getKVERT_IND(const int &i) const = 0;
   
    
    
  protected:  
    //1D basis
    // linear lagrangian
    inline double lagLinear(const double& x, const int& i) const {
      return (!i)*0.5*(1.-x)+!(i-2)*0.5*(1.+x);
    }
    inline double dlagLinear(const double& x, const int& i) const {
      return (!i)*(-0.5)+!(i-2)*0.5;
    }
  
    //quadratic lagrangian  
    inline double th2(const double& x, const int& i) const {
      return !i*(0.5)*(1.-x) + !(i-1)*(1.-x)*(1.+x) + !(i-2)*(0.5)*(1.+x);
    }

    inline double dth2(const double& x, const int& i) const {
      return (!i)*(-0.5) + !(i-1)*(-2.*x) + !(i-2)*(0.5);
    }

    inline double d2th2(const double& x, const int& i) const {
      return !(i-1)*(-2.);
    }
   
    //bi-quadratic lagrangian  
    inline double lagBiquadratic(const double& x, const int& i) const {
      return !i*0.5*x*(x-1.) + !(i-1)*(1.-x)*(1.+x) + !(i-2)*0.5*x*(1.+x);
    }

    inline double dlagBiquadratic(const double& x, const int& i) const {
      return !i*(x-0.5) + !(i-1)*(-2.*x) + !(i-2)*(x+0.5);
    }

    inline double d2lagBiquadratic(const double& x, const int& i) const{
      return !i + !(i-1)*(-2.) + !(i-2);
    }
  
    //2D basis 
    // linear triangle 
    inline double triangleLinear(const double& x, const double& y, const int& i,const int& j) const {
      return (!i*!j)*(1.-x-y)+ !(i-2)*x + !(j-2)*y;
    }

    inline double dtriangleLineardx(const double& x, const double& y, const int& i,const int& j) const {
      return -(!i*!j) + !(i-2);
    }

    inline double dtriangleLineardy(const double& x, const double& y, const int& i,const int& j) const {
      return -(!i*!j) + !(j-2);
    }
  
    // quadratic triangle 
    inline double triangleQuadratic(const double& x, const double& y, const int& i,const int& j) const {
      return 
        !i     * (!j* (1.-x-y)*(1.-2.*x-2.*y) + !(j-1)* 4.*y*(1.-x-y) + !(j-2)*(-y+2.*y*y)) +
	!(i-1) * (!j* 4.*x*(1.-x-y) + !(j-1) * 4.*x*y) +
	!(i-2) * (!j*(-x+2.*x*x));  
    }

    inline double dtriangleQuadraticdx(const double& x, const double& y, const int& i,const int& j) const {
      return 
        !i     * (!j* (-3.+4.*x+4.*y) + !(j-1)*y*(-4.) ) +
	!(i-1) * (!j* 4.*(1.-2.*x-y)  + !(j-1)*y*(4.)) +
	!(i-2) * (!j*(-1 + 4.*x));
    }

    inline double dtriangleQuadraticdy(const double& x, const double& y, const int& i,const int& j) const {
      return 
        !j     * (!i* (-3.+4.*y+4.*x) + !(i-1)*x*(-4.) ) +
	!(j-1) * (!i* 4.*(1.-2.*y-x)  + !(i-1)*x*(4.)) +
	!(j-2) * (!i*(-1 + 4.*y));
    }
    
    inline double d2triangleQuadraticdx2(const double& x, const double& y, const int& i,const int& j) const {
      return !j*( (!i)*4. +!(i-1)*(-8.) + !(i-2)*4. );
    }
    
    inline double d2triangleQuadraticdy2(const double& x, const double& y, const int& i,const int& j) const {
      return !i*( (!j)*4. +!(j-1)*(-8.) + !(j-2)*4. );
    }
    
    inline double d2triangleQuadraticdxdy(const double& x, const double& y, const int& i,const int& j) const {
      return ( (!i)*(!j) + !(i-1)*!(j-1) )*4. + ( !(i-1)*(!j) + (!i)*!(j-1) )*(-4.);
    }
    
    //biquadratic triangle
    
    inline double triangleBiquadratic(const double& x, const double& y, const int& i,const int& j) const {
      return 
        !i     * ( !j* ((1.-x-y)*(1.-2.*x-2.*y) + 3.*x*y*(1-x-y)) +
                   !(j-1)* 4.*(y*(1.-x-y) - 3.*x*y*(1-x-y))  +
                   !(j-2)* (-y+2.*y*y + 3.*x*y*(1-x-y))  ) +        
	!(i-1) * ( !j* 4.*(x*(1.-x-y) - 3.*x*y*(1-x-y)) + 
	           !(j-1)* 4.*(x*y - 3.*x*y*(1-x-y)) ) +
	!(i-2) * ( !j* (-x+2.*x*x + 3.*x*y*(1-x-y)) ) +
	!(i-7) * ( !(j-7)* 27.*x*y*(1-x-y) );  
    }
    
    inline double dtriangleBiquadraticdx(const double& x, const double& y, const int& i,const int& j) const {
      return 
        !i     * ( !j* (-3.+4.*x+4.*y + 3.*(y-2.*x*y-y*y)) +
                   !(j-1)* 4.*(-y - 3.*(y-2.*x*y-y*y))  +
                   !(j-2)* 3.*(y-2.*x*y-y*y) ) +
	!(i-1) * ( !j* 4.*(1.-2.*x-y - 3.*(y-2.*x*y-y*y))  +
	           !(j-1)* 4.*(y - 3.*(y-2.*x*y-y*y)) ) +
	!(i-2) * ( !j* (-1+4.*x + 3.*(y-2.*x*y-y*y)) ) +
	!(i-7) * ( !(j-7)* 27.*(y-2.*x*y-y*y) );
    }
    
    inline double dtriangleBiquadraticdy(const double& x, const double& y, const int& i,const int& j) const {
      return 
        !j     * ( !i* (-3.+4.*y+4.*x + 3.*(x-x*x-2.*x*y)) +
                   !(i-1)* 4.*(-x - 3.*(x-x*x-2.*x*y))  +
                   !(i-2)* 3.*(x-x*x-2.*x*y) ) +
	!(j-1) * ( !i* 4.*(1.-2.*y-x - 3.*(x-x*x-2.*x*y))  +
	           !(i-1)* 4.*(x - 3.*(x-x*x-2.*x*y)) ) +
	!(j-2) * ( !i* (-1+4.*y + 3.*(x-x*x-2.*x*y)) ) +
	!(j-7) * ( !(i-7)* 27.*(x-x*x-2.*x*y) );
    }
    
    inline double d2triangleBiquadraticdx2(const double& x, const double& y, const int& i,const int& j) const {
      return 
        !i     * ( !j* (4. - 6.*y) +
                   !(j-1)* 4.*(6.*y)  +
                   !(j-2)* (-6.*y) ) +
	!(i-1) * ( !j* 4.*(-2. + 6.*y)  +
	           !(j-1)* 4.*(6.*y) ) +
	!(i-2) * ( !j* (4. - 6.*y) ) +
	!(i-7) * ( !(j-7)* (-54.*y) );
    }
    
    inline double d2triangleBiquadraticdy2(const double& x, const double& y, const int& i,const int& j) const {
      return 
        !j     * ( !i* (4. - 6.*x) +
                   !(i-1)* 4.*(6.*x) +
                   !(i-2)* (-6.*x) ) +
        !(j-1) * ( !i* 4.*(-2. + 6.*x)  +
	           !(i-1)* 4.*(6.*x) ) +
        !(j-2) * ( !i* (4. - 6.*x) ) +
        !(j-7) * ( !(i-7)* (-54.*x) );
      }
      
      inline double d2triangleBiquadraticdxdy(const double& x, const double& y, const int& i,const int& j) const {
      return
        !j     * ( !i* (4. + 3.*(1-2.*x-2.*y)) +
                   !(i-1)* 4.*(-1. - 3.*(1-2.*x-2.*y))  +
                   !(i-2)* 3.*(1-2.*x-2.*y) ) +
	!(j-1) * ( !i* 4.*(-1. - 3.*(1-2.*x-2.*y))  +
	           !(i-1)* 4.*(1. - 3.*(1-2.*x-2.*y)) ) +
	!(j-2) * ( !i* (3.*(1-2.*x-2.*y)) ) +
	!(j-7) * ( !(i-7)* 27.*(1-2.*x-2.*y) );
    }
   
  };

  //************************************************************
    
  class hex_lag : public basis {  
  public:
    hex_lag(const int& nc, const int& nf):
      basis(nc, nf, 8, 20, 27){ };
     
    const double* getX(const int &i) const{return X[i];};
    const int* getIND(const int &i) const{return IND[i];};
    const int* getKVERT_IND(const int &i) const {return KVERT_IND[i];};
    
  protected: 
    static const double X[125][3];
    static const int IND[27][3];
    static const int KVERT_IND[125][2];
  };
  
  class hexLinear: public hex_lag {
  public:
    hexLinear(): hex_lag(8, 27) {}; 
    void PrintType() const { std::cout<<" hexLinear ";};
    
    double eval_phi(const int *I,const double* x) const;
    double eval_dphidx(const int *I,const double* x) const;
    double eval_dphidy(const int *I,const double* x) const;
    double eval_dphidz(const int *I,const double* x) const;
  
    double eval_d2phidx2(const int *I,const double* x) const{ return 0.; };
    double eval_d2phidy2(const int *I,const double* x) const{ return 0.; };
    double eval_d2phidz2(const int *I,const double* x) const{ return 0.; };
    double eval_d2phidxdy(const int *I,const double* x) const;
    double eval_d2phidydz(const int *I,const double* x) const;
    double eval_d2phidzdx(const int *I,const double* x) const;
  };

  //************************************************************

  class hexQuadratic: public hex_lag { 
  public:
    hexQuadratic(): hex_lag(20, 81) {};    
    void PrintType() const { std::cout<<" hexQuadratic ";};
    
    double eval_phi(const int *I,const double* x) const;
    double eval_dphidx(const int *I,const double* x) const;
    double eval_dphidy(const int *I,const double* x) const;
    double eval_dphidz(const int *I,const double* x) const;
  
    double eval_d2phidx2(const int *I,const double* x) const;
    double eval_d2phidy2(const int *I,const double* x) const;
    double eval_d2phidz2(const int *I,const double* x) const;
    double eval_d2phidxdy(const int *I,const double* x) const;
    double eval_d2phidydz(const int *I,const double* x) const;
    double eval_d2phidzdx(const int *I,const double* x) const;
  
  };

  //************************************************************

  class hexBiquadratic: public hex_lag {
  public:
     hexBiquadratic(): hex_lag(27, 125) {};    
    void PrintType() const { std::cout<<" hexBiquadratic ";};
    
    double eval_phi(const int *I,const double* x) const;
    double eval_dphidx(const int *I,const double* x) const;
    double eval_dphidy(const int *I,const double* x) const;
    double eval_dphidz(const int *I,const double* x) const;
  
    double eval_d2phidx2(const int *I,const double* x) const;
    double eval_d2phidy2(const int *I,const double* x) const;
    double eval_d2phidz2(const int *I,const double* x) const;
    double eval_d2phidxdy(const int *I,const double* x) const;
    double eval_d2phidydz(const int *I,const double* x) const;
    double eval_d2phidzdx(const int *I,const double* x) const;
  
  };

  //************************************************************

  class hex_const : public basis {
  public: 
    hex_const(const int& nc, const int& nf):
      basis( nc, nf, 8, 20, 27){ };
    
    const double* getX(const int &i) const{return X[i];};
    const int* getIND(const int &i) const{return IND[i];};
    const int* getKVERT_IND(const int &i) const {return KVERT_IND[i];};
    
  protected: 
    static const double X[32][3];
    static const int IND[4][3];
    static const int KVERT_IND[32][2];
  };
  
  
  class hex0: public hex_const {
  public:
    hex0(): hex_const( 1, 8) {};    
    void PrintType() const { std::cout<<" hex0 ";};
    
    double eval_phi(const int *I,const double* x) const{ return 1.; };
    double eval_dphidx(const int *I,const double* x) const{ return 0.; };
    double eval_dphidy(const int *I,const double* x) const{ return 0.; };
    double eval_dphidz(const int *I,const double* x) const{ return 0.; };
    
    double eval_d2phidx2(const int *I,const double* x) const{ return 0.; };
    double eval_d2phidy2(const int *I,const double* x) const{ return 0.; };
    double eval_d2phidz2(const int *I,const double* x) const{ return 0.; };
    double eval_d2phidxdy(const int *I,const double* x) const{ return 0.; };
    double eval_d2phidydz(const int *I,const double* x) const{ return 0.; };
    double eval_d2phidzdx(const int *I,const double* x) const{ return 0.; };
  };

  //******************************************************************************

  class hexpwl: public hex_const {
  public:
    hexpwl(): hex_const( 4, 32) {};     
    void PrintType() const { std::cout<<" hexpwl ";};
    
    double eval_phi(const int *I,const double* x) const;
    double eval_dphidx(const int *I,const double* x) const;
    double eval_dphidy(const int *I,const double* x) const;
    double eval_dphidz(const int *I,const double* x) const;
  
    double eval_d2phidx2(const int *I,const double* x) const{ return 0.; };
    double eval_d2phidy2(const int *I,const double* x) const{ return 0.; };
    double eval_d2phidz2(const int *I,const double* x) const{ return 0.; };
    double eval_d2phidxdy(const int *I,const double* x) const{ return 0.; };
    double eval_d2phidydz(const int *I,const double* x) const{ return 0.; };
    double eval_d2phidzdx(const int *I,const double* x) const{ return 0.; };
  
  };

  //************************************************************
  
  class wedge_lag : public basis{
  public:
    wedge_lag(const int& nc, const int& nf):
      basis(nc, nf, 6, 15, 21){ };
    
    const double* getX(const int &i) const{return X[i];};
    const int* getIND(const int &i) const{return IND[i];};
    const int* getKVERT_IND(const int &i) const {return KVERT_IND[i];};
    
  protected: 
    static const double X[95][3];
    static const int IND[21][3];
    static const int KVERT_IND[95][2];
  };

  class wedgeLinear: public wedge_lag {
  public:
    wedgeLinear(): wedge_lag( 6, 18) {};    
    void PrintType() const { std::cout<<" wedgeLinear ";};
    
    double eval_phi(const int *I,const double* x) const;
    double eval_dphidx(const int *I,const double* x) const;
    double eval_dphidy(const int *I,const double* x) const;
    double eval_dphidz(const int *I,const double* x) const;
  
    double eval_d2phidx2(const int *I,const double* x) const{ return 0.; };
    double eval_d2phidy2(const int *I,const double* x) const{ return 0.; };
    double eval_d2phidz2(const int *I,const double* x) const{ return 0.; };
    double eval_d2phidxdy(const int *I,const double* x) const{ return 0.; };
    double eval_d2phidydz(const int *I,const double* x) const{ return 0.; };
    double eval_d2phidzdx(const int *I,const double* x) const{ return 0.; };
  
  };

  //************************************************************

  class wedgeQuadratic: public wedge_lag {
  public:
    wedgeQuadratic(): wedge_lag(15, 57) {};    
    void PrintType() const { std::cout<<" wedgeQuadratic ";};
    
    double eval_phi(const int *I,const double* x) const;
    double eval_dphidx(const int *I,const double* x) const;
    double eval_dphidy(const int *I,const double* x) const;
    double eval_dphidz(const int *I,const double* x) const;
  
    double eval_d2phidx2(const int *I,const double* x) const;
    double eval_d2phidy2(const int *I,const double* x) const;
    double eval_d2phidz2(const int *I,const double* x) const;
    double eval_d2phidxdy(const int *I,const double* x) const;
    double eval_d2phidydz(const int *I,const double* x) const;
    double eval_d2phidzdx(const int *I,const double* x) const;
  
  };

  //************************************************************
  
  class wedgeBiquadratic: public wedge_lag {
  public:
    wedgeBiquadratic(): wedge_lag(21, 95) {};    
    void PrintType() const { std::cout<<" wedgeBiquadratic ";};
    
    double eval_phi(const int *I,const double* x) const;
    double eval_dphidx(const int *I,const double* x) const;
    double eval_dphidy(const int *I,const double* x) const;
    double eval_dphidz(const int *I,const double* x) const;
  
    double eval_d2phidx2(const int *I,const double* x) const;
    double eval_d2phidy2(const int *I,const double* x) const;
    double eval_d2phidz2(const int *I,const double* x) const;
    double eval_d2phidxdy(const int *I,const double* x) const;
    double eval_d2phidydz(const int *I,const double* x) const;
    double eval_d2phidzdx(const int *I,const double* x) const;
  
  };

  class wedge_const : public basis {
  public: 
    wedge_const(const int& nc, const int& nf):
      basis( nc, nf, 6, 15, 18){ };
    
    const double* getX(const int &i) const{return X[i];};
    const int* getIND(const int &i) const{return IND[i];};
    const int* getKVERT_IND(const int &i) const {return KVERT_IND[i];};
    
  protected: 
    static const double X[32][3];
    static const int IND[4][3];
    static const int KVERT_IND[32][2];
  };
  
  class wedge0: public wedge_const {
  public:
    wedge0(): wedge_const(1, 8){ };    
    void PrintType() const { std::cout<<" wedge0 ";};
    
    double eval_phi(const int *I,const double* x) const{ return 1.; };
    double eval_dphidx(const int *I,const double* x) const{ return 0.; };
    double eval_dphidy(const int *I,const double* x) const{ return 0.; };
    double eval_dphidz(const int *I,const double* x) const{ return 0.; };
    
    double eval_d2phidx2(const int *I,const double* x) const{ return 0.; };
    double eval_d2phidy2(const int *I,const double* x) const{ return 0.; };
    double eval_d2phidz2(const int *I,const double* x) const{ return 0.; };
    double eval_d2phidxdy(const int *I,const double* x) const{ return 0.; };
    double eval_d2phidydz(const int *I,const double* x) const{ return 0.; };
    double eval_d2phidzdx(const int *I,const double* x) const{ return 0.; };
    
  };
  
  class wedgepwl: public wedge_const {
  public:
    wedgepwl(): wedge_const(4, 32){ };    
    void PrintType() const { std::cout<<" wedgepwl ";};
    
    double eval_phi(const int *I,const double* x) const;
    double eval_dphidx(const int *I,const double* x) const;
    double eval_dphidy(const int *I,const double* x) const;
    double eval_dphidz(const int *I,const double* x) const;
    
    double eval_d2phidx2(const int *I,const double* x) const{ return 0.; };
    double eval_d2phidy2(const int *I,const double* x) const{ return 0.; };
    double eval_d2phidz2(const int *I,const double* x) const{ return 0.; };
    double eval_d2phidxdy(const int *I,const double* x) const{ return 0.; };
    double eval_d2phidydz(const int *I,const double* x) const{ return 0.; };
    double eval_d2phidzdx(const int *I,const double* x) const{ return 0.; };
    
  };
  
  
  //************************************************************

  class tet_lag : public basis{
  public:
    tet_lag(const int& nc, const int& nf):
      basis(nc, nf, 4, 10, 15){ };
    const double* getX(const int &i) const{return X[i];};
    const int* getIND(const int &i) const{return IND[i];};
    const int* getKVERT_IND(const int &i) const {return KVERT_IND[i];};
  
  protected: 
    static const double X[67][3];
    static const int IND[15][3];
    static const int KVERT_IND[67][2];
  };
  
  
  
  class tetLinear: public tet_lag {
  public:
    tetLinear(): tet_lag(4, 10) {};    
    void PrintType() const { std::cout<<" tetLinear ";};
    
    double eval_phi(const int *I,const double* x) const;
    double eval_dphidx(const int *I,const double* x) const;
    double eval_dphidy(const int *I,const double* x) const;
    double eval_dphidz(const int *I,const double* x) const;
  
    double eval_d2phidx2(const int *I,const double* x) const{ return 0.; };
    double eval_d2phidy2(const int *I,const double* x) const{ return 0.; };
    double eval_d2phidz2(const int *I,const double* x) const{ return 0.; };
    double eval_d2phidxdy(const int *I,const double* x) const{ return 0.; };
    double eval_d2phidydz(const int *I,const double* x) const{ return 0.; };
    double eval_d2phidzdx(const int *I,const double* x) const{ return 0.; };
  };
  
  //************************************************************

  class tetQuadratic: public tet_lag {
  public:
    tetQuadratic(): tet_lag(10, 35) {};
    void PrintType() const { std::cout<<" tetQuadratic ";};
    
    double eval_phi(const int *I,const double* x) const;
    double eval_dphidx(const int *I,const double* x) const;
    double eval_dphidy(const int *I,const double* x) const;
    double eval_dphidz(const int *I,const double* x) const;
  
    double eval_d2phidx2(const int *I,const double* x) const;
    double eval_d2phidy2(const int *I,const double* x) const;
    double eval_d2phidz2(const int *I,const double* x) const;
    double eval_d2phidxdy(const int *I,const double* x) const;
    double eval_d2phidydz(const int *I,const double* x) const;
    double eval_d2phidzdx(const int *I,const double* x) const;
  };

  //************************************************************

  class tetBiquadratic: public tet_lag {
  public:
    tetBiquadratic(): tet_lag(15, 35) {};
    void PrintType() const { std::cout<<" tetBiquadratic ";};
    
    double eval_phi(const int *I,const double* x) const;
    double eval_dphidx(const int *I,const double* x) const;
    double eval_dphidy(const int *I,const double* x) const;
    double eval_dphidz(const int *I,const double* x) const;
  
    double eval_d2phidx2(const int *I,const double* x) const;
    double eval_d2phidy2(const int *I,const double* x) const;
    double eval_d2phidz2(const int *I,const double* x) const;
    double eval_d2phidxdy(const int *I,const double* x) const;
    double eval_d2phidydz(const int *I,const double* x) const;
    double eval_d2phidzdx(const int *I,const double* x) const;
  };

  //************************************************************

  class tet_const : public basis {
  public: 
    tet_const(const int& nc, const int& nf):
      basis( nc, nf, 4, 10, 10){ };
    
    const double* getX(const int &i) const{return X[i];};
    const int* getIND(const int &i) const{return IND[i];};
    const int* getKVERT_IND(const int &i) const {return KVERT_IND[i];};
    
  protected: 
    static const double X[32][3];
    static const int IND[4][3];
    static const int KVERT_IND[32][2];
  };
  
  class tet0: public tet_const {
  public:
    tet0(): tet_const(1, 8){ };
    void PrintType() const { std::cout<<" tet0 ";};  
    
    double eval_phi(const int *I,const double* x) const{ return 1.; };
    double eval_dphidx(const int *I,const double* x) const{ return 0.; };
    double eval_dphidy(const int *I,const double* x) const{ return 0.; };
    double eval_dphidz(const int *I,const double* x) const{ return 0.; };
    
    double eval_d2phidx2(const int *I,const double* x) const{ return 0.; };
    double eval_d2phidy2(const int *I,const double* x) const{ return 0.; };
    double eval_d2phidz2(const int *I,const double* x) const{ return 0.; };
    double eval_d2phidxdy(const int *I,const double* x) const{ return 0.; };
    double eval_d2phidydz(const int *I,const double* x) const{ return 0.; };
    double eval_d2phidzdx(const int *I,const double* x) const{ return 0.; };
    
  };
  
  class tetpwl: public tet_const {
  public:
    tetpwl(): tet_const(4, 32){ };
    void PrintType() const { std::cout<<" tetpwl ";};  
    
    double eval_phi(const int *I,const double* x) const;
    double eval_dphidx(const int *I,const double* x) const;
    double eval_dphidy(const int *I,const double* x) const;
    double eval_dphidz(const int *I,const double* x) const;
    
    double eval_d2phidx2(const int *I,const double* x) const{ return 0.; };
    double eval_d2phidy2(const int *I,const double* x) const{ return 0.; };
    double eval_d2phidz2(const int *I,const double* x) const{ return 0.; };
    double eval_d2phidxdy(const int *I,const double* x) const{ return 0.; };
    double eval_d2phidydz(const int *I,const double* x) const{ return 0.; };
    double eval_d2phidzdx(const int *I,const double* x) const{ return 0.; };
    
  };

  //************************************************************

  class quad_lag : public basis{
  public:
    quad_lag(const int& nc, const int& nf):
      basis(nc, nf, 4, 8, 9){ };
    const double* getX(const int &i) const{return X[i];};
    const int* getIND(const int &i) const{return IND[i];};
    const int* getKVERT_IND(const int &i) const {return KVERT_IND[i];};
  protected: 
    static const double X[25][2];
    static const int IND[9][2];
    static const int KVERT_IND[25][2];
    
  };
  
  
  class quadLinear: public quad_lag {
  public:
    quadLinear(): quad_lag(4, 9) {};
    void PrintType() const { std::cout<<" quadLinear ";};
    
    double eval_phi(const int *I,const double* x) const;
    double eval_dphidx(const int *I,const double* x) const;
    double eval_dphidy(const int *I,const double* x) const;
  
    double eval_d2phidx2(const int *I,const double* x) const{ return 0.; }
    double eval_d2phidy2(const int *I,const double* x) const{ return 0.; }
    double eval_d2phidxdy(const int *I,const double* x) const;
  };

  //************************************************************

  class quadQuadratic: public quad_lag {
  public:
    quadQuadratic(): quad_lag(8, 21) {};
    void PrintType() const { std::cout<<" quadQuadratic ";};
    
    double eval_phi(const int *I,const double* x) const;
    double eval_dphidx(const int *I,const double* x) const;
    double eval_dphidy(const int *I,const double* x) const;
  
    double eval_d2phidx2(const int *I,const double* x) const;
    double eval_d2phidy2(const int *I,const double* x) const;
    double eval_d2phidxdy(const int *I,const double* x) const;
  };

  //************************************************************

  class quadBiquadratic: public quad_lag {
  public:
    quadBiquadratic(): quad_lag(9, 25) {};
    void PrintType() const { std::cout<<" quadBiquadratic ";};
    
    double eval_phi(const int *I,const double* x) const;
    double eval_dphidx(const int *I,const double* x) const;
    double eval_dphidy(const int *I,const double* x) const;
  
    double eval_d2phidx2(const int *I,const double* x) const;
    double eval_d2phidy2(const int *I,const double* x) const;
    double eval_d2phidxdy(const int *I,const double* x) const;
  };

  //******************************************************************************

  class quad_const : public basis{
  public:
    quad_const(const int& nc, const int& nf):
      basis(nc, nf, 4, 8, 9){ };
    const double* getX(const int &i) const{return X[i];};
    const int* getIND(const int &i) const{return IND[i];};
    const int* getKVERT_IND(const int &i) const {return KVERT_IND[i];};
  
  protected: 
    static const double X[12][2];
    static const int IND[3][2];
    static const int KVERT_IND[12][2];
  };
  
  
  class quad0: public quad_const {
  public:
    quad0(): quad_const(1, 4) {};
    void PrintType() const { std::cout<<" quad0 ";};
    
    double eval_phi(const int *I,const double* x) const{ return 1.; };
    double eval_dphidx(const int *I,const double* x) const{ return 0.; };
    double eval_dphidy(const int *I,const double* x) const{ return 0.; };
    
    double eval_d2phidx2(const int *I,const double* x) const{ return 0.; };
    double eval_d2phidy2(const int *I,const double* x) const{ return 0.; };
    double eval_d2phidxdy(const int *I,const double* x) const{ return 0.; };
    
  };

  //******************************************************************************

  class quadpwl: public quad_const {
  public:
    quadpwl(): quad_const(3, 12) {};
    void PrintType() const { std::cout<<" quadpwl ";};
    
    double eval_phi(const int *I,const double* x) const;
    double eval_dphidx(const int *I,const double* x) const;
    double eval_dphidy(const int *I,const double* x) const;
  
    double eval_d2phidx2(const int *I,const double* x) const{ return 0.; };
    double eval_d2phidy2(const int *I,const double* x) const{ return 0.; };
    double eval_d2phidxdy(const int *I,const double* x) const{ return 0.; };
  };

  //************************************************************

  class tri_lag : public basis{
  public: 
    tri_lag(const int& nc, const int& nf):
      basis(nc, nf, 3, 6, 7){ };
    const double* getX(const int &i) const{return X[i];};
    const int* getIND(const int &i) const{return IND[i];};
    const int* getKVERT_IND(const int &i) const {return KVERT_IND[i];};
    
  protected:
//     static const double X[15][2];
//     static const int IND[6][2];
//     static const int KVERT_IND[15][2];
    static const double X[19][2];
    static const int IND[7][2];
    static const int KVERT_IND[19][2];
  };
  
  
  class triLinear: public tri_lag {
  public:
    triLinear(): tri_lag(3, 6) {}; 
    void PrintType() const { std::cout<<" triLinear ";};
    
    double eval_phi(const int *I,const double* x) const;
    double eval_dphidx(const int *I,const double* x) const;
    double eval_dphidy(const int *I,const double* x) const;
  
    double eval_d2phidx2(const int *I,const double* x) const{ return 0.; };
    double eval_d2phidy2(const int *I,const double* x) const{ return 0.; };
    double eval_d2phidxdy(const int *I,const double* x) const{ return 0.; };
  };

  //************************************************************

  class triQuadratic: public tri_lag {
  public:
    triQuadratic(): tri_lag(6, 15) {}; 
    void PrintType() const { std::cout<<" triQuadratic ";};
    
    double eval_phi(const int *I,const double* x) const;
    double eval_dphidx(const int *I,const double* x) const;
    double eval_dphidy(const int *I,const double* x) const;
  
    double eval_d2phidx2(const int *I,const double* x) const;
    double eval_d2phidy2(const int *I,const double* x) const;
    double eval_d2phidxdy(const int *I,const double* x) const;
  
  };

  //************************************************************
  
  class triBiquadratic: public tri_lag {
  public:
    triBiquadratic(): tri_lag(7, 19) {}; 
    void PrintType() const { std::cout<<" triBiquadratic ";};
    
    double eval_phi(const int *I,const double* x) const;
    double eval_dphidx(const int *I,const double* x) const;
    double eval_dphidy(const int *I,const double* x) const;
  
    double eval_d2phidx2(const int *I,const double* x) const;
    double eval_d2phidy2(const int *I,const double* x) const;
    double eval_d2phidxdy(const int *I,const double* x) const;
  
  };

  //************************************************************
  
  class tri_const : public basis{
  public:
    tri_const(const int& nc, const int& nf):
      basis(nc, nf, 3, 6, 7){ };
    const double* getX(const int &i) const{return X[i];};
    const int* getIND(const int &i) const{return IND[i];};
    const int* getKVERT_IND(const int &i) const {return KVERT_IND[i];};
  
  protected: 
    static const double X[12][2];
    static const int IND[3][2];
    static const int KVERT_IND[12][2];
  };
  
  class tri0: public tri_const {
  public:
    tri0(): tri_const(1, 4){ };    
    void PrintType() const { std::cout<<" tri0 ";};
        
    double eval_phi(const int *I,const double* x) const{ return 1.; };
    double eval_dphidx(const int *I,const double* x) const{ return 0.; };
    double eval_dphidy(const int *I,const double* x) const{ return 0.; };
    
    double eval_d2phidx2(const int *I,const double* x) const{ return 0.; };
    double eval_d2phidy2(const int *I,const double* x) const{ return 0.; };
    double eval_d2phidxdy(const int *I,const double* x) const{ return 0.; };
  };
  
  class tripwl: public tri_const {
  public:
    tripwl(): tri_const(3, 12){ };    
    void PrintType() const { std::cout<<" tripwl ";};
        
    double eval_phi(const int *I,const double* x) const;
    double eval_dphidx(const int *I,const double* x) const;
    double eval_dphidy(const int *I,const double* x) const;
    
    double eval_d2phidx2(const int *I,const double* x) const{ return 0.; };
    double eval_d2phidy2(const int *I,const double* x) const{ return 0.; };
    double eval_d2phidxdy(const int *I,const double* x) const{ return 0.; };
  };
  

  //************************************************************

  class line_lag : public basis{ 
  public:
    line_lag(const int& nc, const int& nf):
      basis(nc, nf, 2, 3, 3){ };
    const double* getX(const int &i) const{return X[i];};
    const int* getIND(const int &i) const{return IND[i];};
    const int* getKVERT_IND(const int &i) const {return KVERT_IND[i];};
    
  protected:
    static const double X[5][1];
    static const int IND[3][1];
    static const int KVERT_IND[5][2];
  };
  
  class lineLinear: public line_lag {
  public:
    lineLinear(): line_lag(2, 3) {}; 
    void PrintType() const { std::cout<<" lineLinear ";};
    
    double eval_phi(const int *I,const double* x) const;
    double eval_dphidx(const int *I,const double* x) const;
    double eval_d2phidx2(const int *I,const double* x) const{ return 0.; };
  
  };

  //************************************************************
  class lineBiquadratic: public line_lag {
  public:
    lineBiquadratic(): line_lag(3, 5) {}; 
    void PrintType() const { std::cout<<" lineBiquadratic ";};
    
    double eval_phi(const int *I,const double* x) const;
    double eval_dphidx(const int *I,const double* x) const;
    double eval_d2phidx2(const int *I,const double* x) const;
  };

  //************************************************************

  class line_const : public basis{
  public:
    line_const(const int& nc, const int& nf):
      basis(nc, nf, 2, 3, 3){ };
    const double* getX(const int &i) const{return X[i];};
    const int* getIND(const int &i) const{return IND[i];};
    const int* getKVERT_IND(const int &i) const {return KVERT_IND[i];};
  
  protected: 
    static const double X[4][1];
    static const int IND[2][1];
    static const int KVERT_IND[4][2];
  };
  
  
  class line0: public line_const {
  public:
    line0(): line_const(1, 2){ };  
    void PrintType() const { std::cout<<" line0 ";};
    
    double eval_phi(const int *I,const double* x) const{ return 1.; };
    double eval_dphidx(const int *I,const double* x) const{ return 0.; };
    double eval_d2phidx2(const int *I,const double* x) const{ return 0.; };
        
  };
  
  
  class linepwl: public line_const {
  public:
    linepwl(): line_const(2, 4){ };  
    void PrintType() const { std::cout<<" linepwl ";};
    
    double eval_phi(const int *I,const double* x) const;
    double eval_dphidx(const int *I,const double* x) const;
    double eval_d2phidx2(const int *I,const double* x) const{ return 0.; };
        
  };
  
  

}//end namespace femus


#endif
