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
         vector Vector;
  };
  
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

class CTensors
  {
CMatrix* matrices[]; 

private:

   matrix Matrix;
   vector Vector;
   ulong  m_rows, m_cols;
   uint   M_TENSOR_DIM;

public:
                     CTensors(uint DIM); //For one dimension tensor
                    ~CTensors(void);
                    
                    bool   TensorAdd(matrix &mat_ , ulong POS);
                    void   TensorPrint();
                    matrix Tensor(ulong POS);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CTensors::CTensors(uint DIM)
 {
   M_TENSOR_DIM = DIM;
   
   ArrayResize(matrices, M_TENSOR_DIM);
   
   
   for (uint i=0; i<M_TENSOR_DIM; i++)
       matrices[i] = new CMatrix;
     
   
   for (uint i=0; i<M_TENSOR_DIM; i++)
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
   for (uint i=0; i<M_TENSOR_DIM; i++)
     if (CheckPointer(matrices[i]) != POINTER_INVALID)
       delete matrices[i];
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool  CTensors::TensorAdd(matrix &mat_ , ulong POS)
 {
   if (POS > M_TENSOR_DIM) 
     {
       Print(__FUNCTION__," Index Error POS =",POS," greater than TENSOR_DIM ",M_TENSOR_DIM);
       
       return (false);
     }
     
    matrices[POS].Matrix = mat_;
   
   return (true);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CTensors::TensorPrint(void)
 {
   for (ulong i=0; i<M_TENSOR_DIM; i++)
     Print("TENSOR INDEX <<",i,">>\n",this.matrices[i].Matrix); 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
matrix CTensors::Tensor(ulong POS)
 {
   if (POS > M_TENSOR_DIM) 
     {
       Print(__FUNCTION__," Index Error POS =",POS," greater than TENSOR_DIM ",M_TENSOR_DIM);
       
       matrix mat = {};
       return (mat);
     }
     
   return (this.matrices[POS].Matrix); 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
