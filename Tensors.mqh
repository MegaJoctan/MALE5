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

class CTensors
  {
CMatrix* matrices[]; 

public:
                     CTensors(uint DIM); //For one dimension tensor
                    ~CTensors(void);
                    
                    uint   SIZE;
                    bool   Add(matrix<double> &mat_ , ulong POS);
                    bool   Append(matrix<double> &mat_);
                    matrix<double> Get(ulong POS);
                    void   Print_();
                    
                    void   Fill(double value);
                    void   MemoryClear();
                    string shape(); //returns the shape of the tensor
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CTensors::CTensors(uint DIM)
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
CTensors::~CTensors(void)
 {
   for (uint i=0; i<SIZE; i++)
     if (CheckPointer(matrices[i]) != POINTER_INVALID)
       delete matrices[i];

   ArrayFree(matrices);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

bool  CTensors::Add(matrix<double> &mat_ , ulong POS)
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

bool CTensors::Append(matrix<double> &mat_)
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
void CTensors::Print_(void)
 {
   for (ulong i=0; i<SIZE; i++)
     Print("TENSOR INDEX [",i,"] matrix-size=(",this.matrices[i].Matrix.Rows(),"x",this.matrices[i].Matrix.Cols(),")\n",this.matrices[i].Matrix); 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

matrix<double> CTensors::Get(ulong POS)
 {
   matrix<double> mat={};
   if (POS > SIZE) 
     {
       Print(__FUNCTION__," Index Error POS =",POS," greater than TENSOR_DIM ",SIZE);
       return (mat);
     }
   
   matrix temp = this.matrices[POS].Matrix;
   mat.Assign(temp);
     
   return (mat); 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

void CTensors::Fill(double value)
 {
   for (ulong i=0; i<SIZE; i++)
     this.matrices[i].Matrix.Fill(value);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CTensors::MemoryClear(void)
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
string CTensors::shape(void)
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

//---

class CTensorsVectors
  {
CVectors             *vectors[];

private:
   uint SIZE;

public:
                     CTensorsVectors(uint DIM);
                    ~CTensorsVectors(void);
                    
                     bool Add(vector &v, ulong POS);
                     void Print_(void);
                     vector Get(ulong POS);
                     void Fill(double value);
                     void MemoryClear();
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CTensorsVectors::CTensorsVectors(uint DIM)
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
CTensorsVectors::~CTensorsVectors(void)
 {
   for (uint i=0; i<SIZE; i++)
     if (CheckPointer(vectors[i]) != POINTER_INVALID)
       delete vectors[i];

   ArrayFree(vectors);
 } 
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool  CTensorsVectors::Add(vector &v, ulong POS)
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
void CTensorsVectors::Print_(void)
 {
   for (ulong i=0; i<SIZE; i++)
     Print("TENSOR INDEX [",i,"] vector-size =(",this.vectors[i].Vector.Size(),")\n",this.vectors[i].Vector); 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CTensorsVectors::Get(ulong POS)
 {
   if (POS > SIZE) 
     {
       Print(__FUNCTION__," Index Error POS =",POS," greater than TENSOR_DIM ",SIZE);
       
       vector v = {};
       return (v);
     }
     
   return (this.vectors[POS].Vector); 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CTensorsVectors::Fill(double value)
 {
   for (ulong i=0; i<SIZE; i++)
     this.vectors[i].Vector.Fill(value);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CTensorsVectors::MemoryClear(void)
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
