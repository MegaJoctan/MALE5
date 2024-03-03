//+------------------------------------------------------------------+
//|                                             cross_validation.mqh |
//|                                    Copyright 2022, Fxalgebra.com |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, Fxalgebra.com"
#property link      "https://www.mql5.com/en/users/omegajoctan"
//+------------------------------------------------------------------+
//| defines                                                          |
//+------------------------------------------------------------------+
#include <MALE5\Tensors.mqh>
#include <MALE5\MatrixExtend.mqh>

class CCrossValidation
  {
   CTensors *tensors[]; //Keep track of all the tensors in memory
   
public:
                     CCrossValidation();
                    ~CCrossValidation(void);
                    
                    CTensors *KFoldCV(matrix &data, uint n_spilts=5);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CCrossValidation::CCrossValidation()
 {
 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CCrossValidation::~CCrossValidation(void)
 {
   for (uint i=0; i<tensors.Size(); i++)
     if (CheckPointer(tensors[i]) != POINTER_INVALID)
       delete (tensors[i]);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CTensors *CCrossValidation::KFoldCV(matrix &data, uint n_spilts=5)
 {   
   ArrayResize(tensors, tensors.Size()+1);
   tensors[tensors.Size()-1] = new CTensors(n_spilts);
   
   int size = (int)MathFloor(data.Rows() / (double)n_spilts);
   
   matrix split_data = {};
   
   for (uint k=0, start = 0; k<n_spilts; k++)
    {      
      split_data = MatrixExtend::Get(data, start, (start+size)-1);
      
      tensors[tensors.Size()-1].Add(split_data, k);
      
      start += size;
    }
   return tensors[tensors.Size()-1];
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

 