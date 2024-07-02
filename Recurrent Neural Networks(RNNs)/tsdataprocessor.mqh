//+------------------------------------------------------------------+
//|                                                         RNNs.mqh |
//|                                     Copyright 2023, Omega Joctan |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, Omega Joctan"
#property link      "https://www.mql5.com/en/users/omegajoctan"

#include <MALE5\MatrixExtend.mqh>
#include <MALE5\Tensors.mqh>
//+------------------------------------------------------------------+
//| defines                                                          |
//+------------------------------------------------------------------+
class CTSDataProcessor 
  {
  
public:
                     CTSDataProcessor (void);
                    ~CTSDataProcessor (void);
                    
                    static C3DTensor *extract_ts(const matrix &mat, uint time_step);
                    static vector    extract_ts(const vector &vec, uint time_step);
                    
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CTSDataProcessor ::CTSDataProcessor (void)
 {
 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CTSDataProcessor ::~CTSDataProcessor (void)
 {
 
 } 
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
C3DTensor *CTSDataProcessor::extract_ts(const matrix &mat, uint time_step)
 {
   C3DTensor *ts_tensor;
   ts_tensor = new C3DTensor(0); //initialize a tensor with zero dimension
   
   matrix chunk = {};
   
    for (uint i=0; i<mat.Rows()-time_step; i++)
      {
        chunk = MatrixExtend::Slice(mat, i, i+time_step);
        ts_tensor.Append(chunk);
      }
  return ts_tensor;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CTSDataProcessor::extract_ts(const vector &vec, uint time_step)
 {
    C2DTensor *ts_tensor;
    ts_tensor = new C2DTensor(0);
    
    vector chunk(vec.Size()-time_step);
    for (uint i=0; i<vec.Size()-time_step; i++)
      chunk[i] = vec[i + time_step];
      
   return chunk;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

