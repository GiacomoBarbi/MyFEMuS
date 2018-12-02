#include "GeomElemHex8.hpp"


namespace femus {




// =======================
GeomElemHex8::GeomElemHex8() : GeomElemBase() {  }
	    
	  
// =======================
          GeomElemHex8::~GeomElemHex8() {    }

// =======================
//this function is the same for all FE, but we had to do it identical for each one because of the fixed allocation
      float GeomElemHex8::get_embedding_matrix(const uint a ,const uint b,const uint c )  {  return _embedding_matrix[a][b][c];  }
      

// =======================
//the embedding matrix picks every child of an element
// and, for every child node, gives the value of the FATHER SHAPE FNCS at the CHILD NODE
//STATIC data member
	   const float GeomElemHex8::_embedding_matrix[8][8][8] =
      { // --------------------------------------
  // The 8 children of the Hex-type elements can be thought of as being
  // associated with the 8 vertices of the Hex.  Some of the children are
  // numbered the same as their corresponding vertex, while some are
  // not.  The children which are numbered differently have been marked
  // with ** in the comments below.
  {// embedding matrix for child 0 (child 0 is associated with vertex 0)
    //  0     1     2     3     4     5     6     7         //father----
    { 1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0}, // 0  //child ----
    { 0.5,  0.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0}, // 1
    { .25,  .25,  .25,  .25,  0.0,  0.0,  0.0,  0.0}, // 2
    { 0.5,  0.0,  0.0,  0.5,  0.0,  0.0,  0.0,  0.0}, // 3
    { 0.5,  0.0,  0.0,  0.0,  0.5,  0.0,  0.0,  0.0}, // 4
    { .25,  .25,  0.0,  0.0,  .25,  .25,  0.0,  0.0}, // 5
    {.125, .125, .125, .125, .125, .125, .125, .125}, // 6
    { .25,  0.0,  0.0,  .25,  .25,  0.0,  0.0,  .25}  // 7
  },
  { // embedding matrix for child 1 (child 1 is associated with vertex 1)
    //  0     1     2     3     4     5     6     7
    { 0.5,  0.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0}, // 0
    { 0.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0}, // 1
    { 0.0,  0.5,  0.5,  0.0,  0.0,  0.0,  0.0,  0.0}, // 2
    { .25,  .25,  .25,  .25,  0.0,  0.0,  0.0,  0.0}, // 3
    { .25,  .25,  0.0,  0.0,  .25,  .25,  0.0,  0.0}, // 4
    { 0.0,  0.5,  0.0,  0.0,  0.0,  0.5,  0.0,  0.0}, // 5
    { 0.0,  .25,  .25,  0.0,  0.0,  .25,  .25,  0.0}, // 6
    {.125, .125, .125, .125, .125, .125, .125, .125}  // 7
  },
  { // embedding matrix for child 2 (child 2 is associated with vertex 3**)
    //  0      1    2     3     4     5     6     7
    { 0.5,  0.0,  0.0,  0.5,  0.0,  0.0,  0.0,  0.0}, // 0
    { .25,  .25,  .25,  .25,  0.0,  0.0,  0.0,  0.0}, // 1
    { 0.0,  0.0,  0.5,  0.5,  0.0,  0.0,  0.0,  0.0}, // 2
    { 0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  0.0}, // 3
    { .25,  0.0,  0.0,  .25,  .25,  0.0,  0.0,  .25}, // 4
    {.125, .125, .125, .125, .125, .125, .125, .125}, // 5
    { 0.0,  0.0,  .25,  .25,  0.0,  0.0,  .25,  .25}, // 6
    { 0.0,  0.0,  0.0,  0.5,  0.0,  0.0,  0.0,  0.5}  // 7
  },
  {// embedding matrix for child 3 (child 3 is associated with vertex 2**)
    //  0      1    2     3     4     5     6     7
    { .25,  .25,  .25,  .25,  0.0,  0.0,  0.0,  0.0}, // 0
    { 0.0,  0.5,  0.5,  0.0,  0.0,  0.0,  0.0,  0.0}, // 1
    { 0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0}, // 2
    { 0.0,  0.0,  0.5,  0.5,  0.0,  0.0,  0.0,  0.0}, // 3
    {.125, .125, .125, .125, .125, .125, .125, .125}, // 4
    { 0.0,  .25,  .25,  0.0,  0.0,  .25,  .25,  0.0}, // 5
    { 0.0,  0.0,  0.5,  0.0,  0.0,  0.0,  0.5,  0.0}, // 6
    { 0.0,  0.0,  .25,  .25,  0.0,  0.0,  .25,  .25}  // 7
  },

  // embedding matrix for child 4 (child 4 is associated with vertex 4)
  {
    //  0      1    2     3     4     5     6     7
    { 0.5,  0.0,  0.0,  0.0,  0.5,  0.0,  0.0,  0.0}, // 0
    { .25,  .25,  0.0,  0.0,  .25,  .25,  0.0,  0.0}, // 1
    {.125, .125, .125, .125, .125, .125, .125, .125}, // 2
    { .25,  0.0,  0.0,  .25,  .25,  0.0,  0.0,  .25}, // 3
    { 0.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0}, // 4
    { 0.0,  0.0,  0.0,  0.0,  0.5,  0.5,  0.0,  0.0}, // 5
    { 0.0,  0.0,  0.0,  0.0,  .25,  .25,  .25,  .25}, // 6
    { 0.0,  0.0,  0.0,  0.0,  0.5,  0.0,  0.0,  0.5}  // 7
  },

  // embedding matrix for child 5 (child 5 is associated with vertex 5)
  {
    //  0      1    2     3     4     5     6     7
    { .25,  .25,  0.0,  0.0,  .25,  .25,  0.0,  0.0}, // 0
    { 0.0,  0.5,  0.0,  0.0,  0.0,  0.5,  0.0,  0.0}, // 1
    { 0.0,  .25,  .25,  0.0,  0.0,  .25,  .25,  0.0}, // 2
    {.125, .125, .125, .125, .125, .125, .125, .125}, // 3
    { 0.0,  0.0,  0.0,  0.0,  0.5,  0.5,  0.0,  0.0}, // 4
    { 0.0,  0.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0}, // 5
    { 0.0,  0.0,  0.0,  0.0,  0.0,  0.5,  0.5,  0.0}, // 6
    { 0.0,  0.0,  0.0,  0.0,  .25,  .25,  .25,  .25}  // 7
  },

  // embedding matrix for child 6 (child 6 is associated with vertex 7**)
  {
    //  0      1    2     3     4     5     6     7
    { .25,  0.0,  0.0,  .25,  .25,  0.0,  0.0,  .25}, // 0
    {.125, .125, .125, .125, .125, .125, .125, .125}, // 1
    { 0.0,  0.0,  .25,  .25,  0.0,  0.0,  .25,  .25}, // 2
    { 0.0,  0.0,  0.0,  0.5,  0.0,  0.0,  0.0,  0.5}, // 3
    { 0.0,  0.0,  0.0,  0.0,  0.5,  0.0,  0.0,  0.5}, // 4
    { 0.0,  0.0,  0.0,  0.0,  .25,  .25,  .25,  .25}, // 5
    { 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.5,  0.5}, // 6
    { 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  1.0}  // 7
  },

  // embedding matrix for child 7 (child 7 is associated with vertex 6**)
  {
    //  0      1    2     3     4     5     6     7
    {.125, .125, .125, .125, .125, .125, .125, .125}, // 0
    { 0.0,  .25,  .25,  0.0,  0.0,  .25,  .25,  0.0}, // 1
    { 0.0,  0.0,  0.5,  0.0,  0.0,  0.0,  0.5,  0.0}, // 2
    { 0.0,  0.0,  .25,  .25,  0.0,  0.0,  .25,  .25}, // 3
    { 0.0,  0.0,  0.0,  0.0,  .25,  .25,  .25,  .25}, // 4
    { 0.0,  0.0,  0.0,  0.0,  0.0,  0.5,  0.5,  0.0}, // 5
    { 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  1.0,  0.0}, // 6
    { 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.5,  0.5}  // 7
  }
};



  const double GeomElemHex8::_Prol[27*8/*NNDS*NNDSL*/] = { 
   1.,0.,0.,0.,0.,0.,0.,0.,
   0.,1.,0.,0.,0.,0.,0.,0.,
   0.,0.,1.,0.,0.,0.,0.,0.,
   0.,0.,0.,1.,0.,0.,0.,0.,
   0.,0.,0.,0.,1.,0.,0.,0.,
   0.,0.,0.,0.,0.,1.,0.,0.,
   0.,0.,0.,0.,0.,0.,1.,0.,
   0.,0.,0.,0.,0.,0.,0.,1.,
   0.5,0.5,0.,0.,0.,0.,0.,0.,
   0.,0.5,0.5,0.,0.,0.,0.,0.,
   0.,0.,0.5,0.5,0.,0.,0.,0.,
   0.5,0.,0.,0.5,0.,0.,0.,0.,  //end first four below
   0.5,0.,0.,0.,0.5,0.,0.,0.,
   0.,0.5,0.,0.,0.,0.5,0.,0.,
   0.,0.,/*0.*/0.5,/*0.5*/0.,0.,0.,0.5,0.,  //3rd and 7th, not 4th and 7th!
   0.,0.,0.,0.5,0.,0.,0.,0.5,
   0.,0.,0.,0.,0.5,0.5,0.,0.,
   0.,0.,0.,0.,0.,0.5,0.5,0.,
   0.,0.,0.,0.,0.,0.,0.5,0.5,
   0.,0.,0.,0.,0.5,0.,0.,0.5,
   0.25,0.25,0.25,0.25,0.,0.,0.,0.,
   0.25,0.25,0.,0.,0.25,0.25,0.,0., 
   0.,0.25,0.25,0.,0.,0.25,0.25,0.,
   0.,0.,0.25,0.25,0.,0.,0.25,0.25,
   0.25,0.,0.,0.25,0.25,0.,0.,0.25,
   0.,0.,0.,0.,0.25,0.25,0.25,0.25,
   0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125
   };


} //end namespace femus


	   
