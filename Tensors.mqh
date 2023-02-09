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
                     //CTensors(ulong& DIM[][TENSOR_COLS])
                    ~CTensors(void);
                    bool TensorAdd(matrix &mat_ , int POS);
                    void TensorPrint();
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CTensors::CTensors(uint DIM)
 {
   M_TENSOR_DIM = DIM;
   
   Print("Tensor dim ",M_TENSOR_DIM);
   
   ArrayResize(matrices, M_TENSOR_DIM);
   
   Print("matrices size ",ArraySize(matrices));
   
   
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
bool  CTensors::TensorAdd(matrix &mat_ , int POS)
 {
   if ((ulong)POS > M_TENSOR_DIM) 
     {
       Print(__FUNCTION__," Index Error POS greater than TENSOR_DIM ");
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
     Print("TENSOR INDEX ",i,"\n",this.matrices[i].Matrix); 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
