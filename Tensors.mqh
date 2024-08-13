//+------------------------------------------------------------------+
//|                                                      Tensors.mqh |
//|                                    Copyright 2022, Fxalgebra.com |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, Fxalgebra.com"
#property link      "https://www.mql5.com/en/users/omegajoctan"

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

class CMatrix
  {
   public:
         matrix Matrix;
  };
  
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

class C3DTensor
  {
CMatrix* matrices[]; 

public:
                     C3DTensor(void); //For one dimension tensor
                    ~C3DTensor(void);
                    
                    bool   Init(uint size);
                    bool   Append(matrix<double> &__matrix__);
                    
                    CMatrix *GetObj(int index);
                    //virtual matrix operator[](const int index) { return Get(index); }
                    CMatrix* operator[](const int index) { return GetObj(index); }
                    void   Print_();
                    
                    void   Delete();
                    uint   Size(); //returns tensor's size
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
C3DTensor::C3DTensor(void)
 {   
 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
C3DTensor::~C3DTensor(void)
 {
   for (uint i=0; i<matrices.Size(); i++)
     if (CheckPointer(matrices[i]) != POINTER_INVALID)
       delete matrices[i];

   ArrayFree(matrices);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//|  This function initilalizes the 3D tensor by creating empty      |
//|  matrices to the tensor memory                                   |
//|                                                                  |
//+------------------------------------------------------------------+
bool C3DTensor::Init(uint size)
 {
   if (size==0)
     return false;
     
   ArrayResize(this.matrices, size);
   return true;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
uint C3DTensor::Size()
 {
   return this.matrices.Size();
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool C3DTensor::Append(matrix<double> &__matrix__)
 {
   if (ArrayResize(matrices, matrices.Size()+1)<0)
    return false;
    
   uint SIZE = matrices.Size();
   matrices[SIZE-1] = new CMatrix();
   matrices[SIZE-1].Matrix = __matrix__; //Add the new matrix to the newly created tensor index
   
   return true;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void C3DTensor::Print_(void)
 {
   for (uint i=0; i<matrices.Size(); i++)
     Print("TENSOR INDEX [",i,"] matrix-size=(",this.matrices[i].Matrix.Rows(),"x",this.matrices[i].Matrix.Cols(),")\n",this.matrices[i].Matrix); 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CMatrix *C3DTensor::GetObj(int index)
 {
   if (index<-1 || index > int(matrices.Size()))
    {
      printf("%s failed, index out of range. Line %d",__FUNCTION__, __LINE__);
      return this.matrices[index==-1?matrices.Size()-1: index];
    }
   
   return this.matrices[index==-1?matrices.Size()-1: index]; //if the selected position is -1 we obtain the last matrix in our tensor
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void C3DTensor::Delete(void)
 {
   for (ulong i=0; i<matrices.Size(); i++)
    {
      this.matrices[i].Matrix.Resize(0,0);
      ZeroMemory(this.matrices[i].Matrix);
    }
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+


//+------------------------------------------------------------------+
//|                                                                  |
//|   Tensorflows for Vector type of data                            |
//|                                                                  |
//+------------------------------------------------------------------+

class CVectors
  {
   public:
          vector Vector;
  };

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class C2DTensor
  {
CVectors             *vectors[];

public:
                     C2DTensor(void);
                    ~C2DTensor(void);
                     
                     bool   Init(uint size);
                     bool Append(vector &v);
                     
                     void Print_(void);
                     CVectors* operator[](const int index) { return GetObj(index); }
                     CVectors *GetObj(int index);
                     
                     void Delete();
                     uint   Size(); //returns tensor's size
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
C2DTensor::C2DTensor(void)
 {
   
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
C2DTensor::~C2DTensor(void)
 {
   for (uint i=0; i<vectors.Size(); i++)
     if (CheckPointer(vectors[i]) != POINTER_INVALID)
       delete vectors[i];

   ArrayFree(vectors);
 } 
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool C2DTensor::Init(uint size)
 {
   if (size==0)
     return false;
     
   ArrayResize(this.vectors, size);
   return true;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CVectors *C2DTensor::GetObj(int index)
 {
   if (index<-1 || index > int(vectors.Size()))
    {
      printf("%s failed, index out of range. Line %d",__FUNCTION__, __LINE__);
      return this.vectors[index==-1?vectors.Size()-1: index];
    }
   
   return this.vectors[index==-1?vectors.Size()-1: index]; //if the selected position is -1 we obtain the last matrix in our tensor
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void C2DTensor::Print_(void)
 {
   for (ulong i=0; i<vectors.Size(); i++)
     Print("TENSOR INDEX [",i,"] vector-size =(",this.vectors[i].Vector.Size(),")\n",this.vectors[i].Vector); 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
uint C2DTensor::Size(void)
 {
   return this.vectors.Size();
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void C2DTensor::Delete(void)
 {
   for (ulong i=0; i<vectors.Size(); i++)
    {
      this.vectors[i].Vector.Resize(0,0);
      ZeroMemory(this.vectors[i].Vector);
    }
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool C2DTensor::Append(vector &v)
 {
   if (ArrayResize(this.vectors, vectors.Size()+1)<0)
    return false;
    
   uint SIZE = vectors.Size();
   vectors[SIZE-1] = new CVectors();
   vectors[SIZE-1].Vector = v; //Add the new matrix to the newly created tensor index
   
   return true;   
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+