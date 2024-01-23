//+------------------------------------------------------------------+
//|                                                      helpers.mqh |
//|                                     Copyright 2023, Omega Joctan |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, Omega Joctan"
#property link      "https://www.mql5.com/en/users/omegajoctan"
//+------------------------------------------------------------------+
//| Base class for dimension reduction, containing most useful       |
//| that are necessary for the algorithms in this folder             |
//+------------------------------------------------------------------+
class Base
  {
public:
                     Base(void);
                    ~Base(void);
                    
                    static matrix Slice(const matrix &mat, uint from_0_to, int axis=0);
                    static vector Slice(const vector &v, uint from_0_to);
                    static matrix subtract(const matrix&mat, const vector &v);
                    static void   ReplaceNaN(matrix &mat);
                    static matrix Sort(matrix &mat, vector &args);
                    static vector Sort(vector &v, vector &args);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
matrix Base::Slice(const matrix &mat, uint from_0_to, int axis=0)
 {
  matrix ret = {};
  
  switch(axis)
    {
     case  0: 
      ret.Resize(from_0_to, mat.Cols());
       
      for (uint i=0; i<mat.Rows(); i++)
        ret.Row(mat.Row(i), i);   
        
       break;
     case 1:
      ret.Resize(mat.Rows(), from_0_to);
      
      for (uint i=0; i<mat.Cols(); i++)
        ret.Col(mat.Col(i), i);   
        
       break;
     default:
       Print("%s Invalid axis %d axis can be either 0 or 1",__FUNCTION__,axis);
       break;
   }
   
   return ret;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector Base::Slice(const vector &v, uint from_0_to)
 {
   vector ret(from_0_to);
   
   for (uint i=0; i<ret.Size(); i++)
     ret[i] = v[i];
  
   return ret;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
matrix Base::subtract(const matrix&mat, const vector &v)
 {
   matrix ret = mat;
   
   for (ulong i=0; i<mat.Rows(); i++)
     ret.Row(mat.Row(i)-v, i);
   
   return ret; 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void Base::ReplaceNaN(matrix &mat)
 {
   for (ulong i = 0; i < mat.Rows(); i++) 
     for (ulong j = 0; j < mat.Cols(); j++) 
       if (!MathIsValidNumber(mat[i][j]))
          mat[i][j] = 0.0;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
matrix Base::Sort(matrix &mat, vector &args)
 {
   matrix m = mat;
   
   if (args.Size() != mat.Cols())
     {
       printf("%s Args size != mat.Cols ");
       return m;
     }
   
   for (ulong i=0; i<mat.Cols(); i++)
       m.Col(mat.Col((ulong)args[i]), i);
       
   return m;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector Base::Sort(vector &v, vector &args)
 {
   vector vec = v;
   
   if (args.Size() != v.Size())
     {
       printf("%s Args size != v.size ");
       return vec;
     }
   
   for (ulong i=0; i<v.Size(); i++)
     vec[i] = v[(int)args[i]];
       
   return vec;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
