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
#define TENSOR_COLS 2

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

private:

   matrix Matrix;
   ulong  m_rows, m_cols;

public:
                     CTensors(uint DIM); //For one dimension tensor
                    ~CTensors(void);
                    
                    uint   TENSOR_DIMENSION;
                    
                    bool   TensorAdd(matrix &mat_ , ulong POS);
                    bool   TensorAppend(vector &v, ulong POS);
                    
                    void   TensorPrint();
                    matrix Tensor(ulong POS);
                    void   TensorFill(double value);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CTensors::CTensors(uint DIM)
 {
   TENSOR_DIMENSION = DIM;
   
   ArrayResize(matrices, TENSOR_DIMENSION);
   
   
   for (uint i=0; i<TENSOR_DIMENSION; i++)
       matrices[i] = new CMatrix;
     
   
   for (uint i=0; i<TENSOR_DIMENSION; i++)
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
   for (uint i=0; i<TENSOR_DIMENSION; i++)
     if (CheckPointer(matrices[i]) != POINTER_INVALID)
       delete matrices[i];
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool  CTensors::TensorAdd(matrix &mat_ , ulong POS)
 {
   if (POS > TENSOR_DIMENSION) 
     {
       Print(__FUNCTION__," Index Error POS =",POS," greater than TENSOR_DIM ",TENSOR_DIMENSION);
       
       return (false);
     }
     
    this.matrices[POS].Matrix = mat_;
   
   return (true);
 }
//+------------------------------------------------------------------+
//|      Appending rows to the Tensor Matrix at POS index            |
//+------------------------------------------------------------------+
bool CTensors::TensorAppend(vector &v, ulong POS)
 {
   if (POS > TENSOR_DIMENSION) 
     {
       Print(__FUNCTION__," Index Error POS =",POS," greater than TENSOR_DIM ",TENSOR_DIMENSION);
       
       return (false);
     }
     
//---

   matrix mat = this.matrices[POS].Matrix;
   
   mat.Resize(mat.Rows()+1, mat.Cols());
   mat.Row(v, mat.Rows()-1);
   
   TensorAdd(mat, POS);
   
  return (true);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CTensors::TensorPrint(void)
 {
   for (ulong i=0; i<TENSOR_DIMENSION; i++)
     Print("TENSOR INDEX <<",i,">>\n",this.matrices[i].Matrix); 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
matrix CTensors::Tensor(ulong POS)
 {
   if (POS > TENSOR_DIMENSION) 
     {
       Print(__FUNCTION__," Index Error POS =",POS," greater than TENSOR_DIM ",TENSOR_DIMENSION);
       
       matrix mat = {};
       return (mat);
     }
     
   return (this.matrices[POS].Matrix); 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CTensors::TensorFill(double value)
 {
   for (ulong i=0; i<TENSOR_DIMENSION; i++)
     this.matrices[i].Matrix.Fill(value);
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

//---

class CTensorsVectors
  {
CVectors             *vectors[];

private:
   uint TENSOR_DIMENSION;

public:
                     CTensorsVectors(uint DIM);
                    ~CTensorsVectors(void);
                    
                     bool TensorAdd(vector &v, ulong POS);
                     void TensorPrint(void);
                     vector Tensor(ulong POS);
                     void TensorFill(double value);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CTensorsVectors::CTensorsVectors(uint DIM)
 {
   TENSOR_DIMENSION = DIM;
   
   ArrayResize(vectors, TENSOR_DIMENSION);
   
   
   for (uint i=0; i<TENSOR_DIMENSION; i++)
       vectors[i] = new CVectors;
     
   
   for (uint i=0; i<TENSOR_DIMENSION; i++)
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
   for (uint i=0; i<TENSOR_DIMENSION; i++)
     if (CheckPointer(vectors[i]) != POINTER_INVALID)
       delete vectors[i];
 } 
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool  CTensorsVectors::TensorAdd(vector &v, ulong POS)
 {
   if (POS > TENSOR_DIMENSION) 
     {
       Print(__FUNCTION__," Index Error POS =",POS," greater than TENSOR_DIM ",TENSOR_DIMENSION);
       
       return (false);
     }
     
    this.vectors[POS].Vector = v;
   
   return (true);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CTensorsVectors::TensorPrint(void)
 {
   for (ulong i=0; i<TENSOR_DIMENSION; i++)
     Print("TENSOR INDEX <<",i,">>\n",this.vectors[i].Vector); 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CTensorsVectors::Tensor(ulong POS)
 {
   if (POS > TENSOR_DIMENSION) 
     {
       Print(__FUNCTION__," Index Error POS =",POS," greater than TENSOR_DIM ",TENSOR_DIMENSION);
       
       vector v = {};
       return (v);
     }
     
   return (this.vectors[POS].Vector); 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CTensorsVectors::TensorFill(double value)
 {
   for (ulong i=0; i<TENSOR_DIMENSION; i++)
     this.vectors[i].Vector.Fill(value);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
