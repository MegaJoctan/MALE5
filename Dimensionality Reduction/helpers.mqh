//+------------------------------------------------------------------+
//|                                                      helpers.mqh |
//|                                     Copyright 2023, Omega Joctan |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, Omega Joctan"
#property link      "https://www.mql5.com/en/users/omegajoctan"
//+------------------------------------------------------------------+
//| defines                                                          |
//+------------------------------------------------------------------+
class CDimensionReductionHelpers
  {
public:
                     CDimensionReductionHelpers(void);
                    ~CDimensionReductionHelpers(void);
                    
                    static matrix Slice(const matrix &mat, uint r_from_0_to);
                    static vector Slice(const vector &v, uint from_0_to);
                    static matrix subtract(const matrix&mat, const vector &v);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
matrix CDimensionReductionHelpers::Slice(const matrix &mat, uint r_from_0_to)
 {
   matrix ret(r_from_0_to, mat.Cols());
   
   for (uint i=0; i<mat.Rows(); i++)
     ret.Row(mat.Row(i), i);
  
   return ret;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CDimensionReductionHelpers::Slice(const vector &v, uint from_0_to)
 {
   vector ret(from_0_to);
   
   for (uint i=0; i<ret.Size(); i++)
     ret[i] = v[i];
  
   return ret;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
matrix CDimensionReductionHelpers::subtract(const matrix&mat, const vector &v)
 {
   matrix ret = mat;
   
   for (ulong i=0; i<mat.Rows(); i++)
     ret.Row(mat.Row(i)-v, i);
   
   return ret; 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

