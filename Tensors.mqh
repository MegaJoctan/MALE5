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
                     C3DTensor(uint DIM); //For one dimension tensor
                    ~C3DTensor(void);
                    
                    uint   SIZE;
                    bool   Add(matrix<double> &mat_ , ulong POS);
                    bool   Append(matrix<double> &mat_);
                    matrix<double> Get(int POS);
                    void   Print_();
                    
                    void   Fill(double value);
                    void   MemoryClear();
                    string shape(); //returns the shape of the tensor
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
C3DTensor::C3DTensor(uint DIM)
 {
   SIZE = DIM;
   
   ArrayResize(matrices, SIZE);
   
   for (uint i=0; i<SIZE; i++)
       matrices[i] = new CMatrix;
     
   
   for (uint i=0; i<SIZE; i++)
     if (CheckPointer(matrices[i]) == POINTER_INVALID)
       {
         printf("Can't create a tensor, Invalid pointer Err %d ",GetLastError());
         return;
       }
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
C3DTensor::~C3DTensor(void)
 {
   for (uint i=0; i<SIZE; i++)
     if (CheckPointer(matrices[i]) != POINTER_INVALID)
       delete matrices[i];

   ArrayFree(matrices);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

bool  C3DTensor::Add(matrix<double> &mat_ , ulong POS)
 {
   if (POS > SIZE) 
     {
       Print(__FUNCTION__," Index Error POS =",POS," greater than TENSOR_DIM ",SIZE);
       
       return (false);
     }
     
    this.matrices[POS].Matrix = mat_;
   
   return (true);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

bool C3DTensor::Append(matrix<double> &mat_)
 {
   if (ArrayResize(matrices, SIZE+1)<0)
    return false;
    
   SIZE = matrices.Size();
   matrices[SIZE-1] = new CMatrix();
   matrices[SIZE-1].Matrix = mat_; //Add the new matrix to the newly created tensor index
   
   return true;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void C3DTensor::Print_(void)
 {
   for (ulong i=0; i<SIZE; i++)
     Print("TENSOR INDEX [",i,"] matrix-size=(",this.matrices[i].Matrix.Rows(),"x",this.matrices[i].Matrix.Cols(),")\n",this.matrices[i].Matrix); 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

matrix<double> C3DTensor::Get(int POS)
 {
   matrix<double> mat={};
   if (POS<-1 || POS > int(SIZE))
    {
      printf("%s failed, index out of range. Line %d",__FUNCTION__, __LINE__);
      return mat;
    }
   
   matrix temp = this.matrices[POS==-1?SIZE-1: POS].Matrix; //if the selected position is -1 we obtain the last matrix in our tensor
   mat.Assign(temp);
     
   return (mat); 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

void C3DTensor::Fill(double value)
 {
   for (ulong i=0; i<SIZE; i++)
     this.matrices[i].Matrix.Fill(value);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void C3DTensor::MemoryClear(void)
 {
   for (ulong i=0; i<SIZE; i++)
    {
      this.matrices[i].Matrix.Resize(1,0);
      ZeroMemory(this.matrices[i].Matrix);
    }
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
string C3DTensor::shape(void)
 {
   printf("Warning: %s assumes all matrices in the tensor have the same size",__FUNCTION__);
   return StringFormat("(%d, %d, %d)",this.SIZE,this.matrices[0].Matrix.Rows(),this.matrices[0].Matrix.Cols());
 }

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

private:
   uint SIZE;

public:
                     C2DTensor(uint DIM);
                    ~C2DTensor(void);
                    
                     bool Add(vector &v, ulong POS);
                     bool Append(vector &v);
                     void Print_(void);
                     vector Get(int POS);
                     void Fill(double value);
                     void MemoryClear();
                     string shape(void);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
C2DTensor::C2DTensor(uint DIM)
 {
   SIZE = DIM;
   
   ArrayResize(vectors, SIZE);
   
   for (uint i=0; i<SIZE; i++)
       vectors[i] = new CVectors;
     
   
   for (uint i=0; i<SIZE; i++)
     if (CheckPointer(vectors[i]) == POINTER_INVALID)
       {
         printf("Can't create a tensor, Invalid pointer Err %d ",GetLastError());
         return;
       }
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
C2DTensor::~C2DTensor(void)
 {
   for (uint i=0; i<SIZE; i++)
     if (CheckPointer(vectors[i]) != POINTER_INVALID)
       delete vectors[i];

   ArrayFree(vectors);
 } 
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool  C2DTensor::Add(vector &v, ulong POS)
 {
   if (POS > SIZE) 
     {
       Print(__FUNCTION__," Index Error POS =",POS," greater than TENSOR_DIM ",SIZE);
       
       return (false);
     }
     
    this.vectors[POS].Vector = v;
   
   return (true);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void C2DTensor::Print_(void)
 {
 
   for (ulong i=0; i<SIZE; i++)
     Print("TENSOR INDEX [",i,"] vector-size =(",this.vectors[i].Vector.Size(),")\n",this.vectors[i].Vector); 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector C2DTensor::Get(int POS)
 {
    if (POS<-1 || POS > int(SIZE))
       {
         printf("%s failed, index out of range. Line %d",__FUNCTION__, __LINE__);
         vector v = {};
         return v;
       }
     
   return (this.vectors[POS==-1?SIZE-1: POS].Vector); 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void C2DTensor::Fill(double value)
 {
   for (ulong i=0; i<SIZE; i++)
     this.vectors[i].Vector.Fill(value);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void C2DTensor::MemoryClear(void)
 {
   for (ulong i=0; i<SIZE; i++)
    {
      this.vectors[i].Vector.Resize(1,0);
      ZeroMemory(this.vectors[i].Vector);
    }
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
string C2DTensor::shape(void)
 {
   printf("Warning: %s assumes all vectors in the tensor have the same size",__FUNCTION__);
   return StringFormat("(%d, %d)",this.SIZE,this.vectors[0].Vector.Size());
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool C2DTensor::Append(vector &v)
 {
   if (ArrayResize(this.vectors, SIZE+1)<0)
    return false;
    
   SIZE = vectors.Size();
   vectors[SIZE-1] = new CVectors();
   vectors[SIZE-1].Vector = v; //Add the new matrix to the newly created tensor index
   
   return true;   
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
