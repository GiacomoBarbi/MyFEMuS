
/*=========================================================================

 Program: FEMuS
 Module: MyMatrix
 Authors: Eugenio Aulisa

 Copyright (c) FEMuS
 All rights reserved.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include <string>

#ifndef __femus_mesh_MyMatrix_hpp__
#define __femus_mesh_MyMatrix_hpp__


#include <iostream>
#include <vector>
#include <stdlib.h>

#include <mpi.h>
#include <boost/mpi/datatype.hpp>

#include "MyVector.hpp"

namespace femus {

  template <class Type> class MyMatrix {

    public:
      // ******************
      MyMatrix();

      // ******************
      MyMatrix(const unsigned &rsize, const unsigned &csize, const Type value = 0);

      // ******************
      MyMatrix(const std::vector < unsigned > &offset, const unsigned &csize, const Type value = 0);

      // ******************
      MyMatrix(const MyVector < unsigned > &rowSize, const Type value = 0);

      // ******************
      ~MyMatrix();

      //*******************
      void init();

      // ******************
      void resize(const unsigned &rsize, const unsigned &csize, const Type value = 0);

      // ******************
      void resize(const std::vector < unsigned > &offset, const unsigned &csize, const Type value = 0);

      // ******************
      void shrinkToFit(Type remove);
      
      void shrinkToFit(MyVector<unsigned> &r);
      
      // ******************
      void clear();

      // ******************
      unsigned size();
      
      unsigned size(const unsigned &i);

      // ******************
      unsigned begin();

      unsigned begin(const unsigned &i);

      // ******************
      unsigned end();

      unsigned end(const unsigned &i);

      // ******************
      void scatter(const std::vector < unsigned > &offset);

      void scatter();

      // ******************
      void buildOffset();

      // ******************
      std::vector<unsigned> getOffset();
      
      MyVector < unsigned > getRowSize(){
	return _rowSize;
      }

      // ******************
      void broadcast(const unsigned &lproc);

      // ******************
      void clearBroadcast();

      // ****************
      const std::string &status();

      // ******************

      Type* operator[](const unsigned &i);

      Type& operator()(const unsigned &i, const unsigned &j);

      // *****************
      friend std::ostream& operator<<(std::ostream& os, MyMatrix<Type>& mat) {

        os << mat.status() << std::endl;

        if(mat._matIsAllocated) {
          if(mat._serial) {
            for(unsigned i = mat.begin(); i < mat.end(); i++) {
	      os << i <<"   ";
              for(unsigned j = mat.begin(i); j < mat.end(i); j++) {
                os << mat[i][j] << " ";
              }
              os << std::endl;
            }
          }
          else {
            for(int j = 0; j < mat._nprocs; j++) {
              mat.broadcast(j);
              for(unsigned i = mat.begin(); i < mat.end(); i++) {
		os << i <<"   ";
                for(unsigned j = mat.begin(i); j < mat.end(i); j++) {
                  os << mat[i][j] << " ";
                }
                os << std::endl;
              }
              mat.clearBroadcast();
              os << std::endl;
            }
          }
        }
        return os;
      }
   
    private:

      std::string _status;
      bool _serial;
      bool _matIsAllocated;

      unsigned _iproc;
      unsigned _nprocs;
      MPI_Datatype _MY_MPI_DATATYPE;

      unsigned _begin;
      unsigned _end;
      unsigned _size;

      std::vector< Type > _mat;
      std::vector< Type > _mat2;

      //only to be used when matrix status is parallel
      std::vector < unsigned > _offset;
      MyVector < unsigned > _rowOffset;
      MyVector < unsigned > _rowSize;
      MyVector < unsigned > _matSize;
      unsigned _lproc;
  };





} //end namespace femus



#endif
