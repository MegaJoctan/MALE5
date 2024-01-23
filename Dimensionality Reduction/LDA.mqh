//+------------------------------------------------------------------+
//|                                                          LDA.mqh |
//|                                     Copyright 2023, Omega Joctan |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, Omega Joctan"
#property link      "https://www.mql5.com/en/users/omegajoctan"
//+------------------------------------------------------------------+
//| defines                                                          |
//+------------------------------------------------------------------+

#include "base.mqh";

class CLDA
  {
protected:
   uint m_num_components;
   matrix projection_matrix;
   
public:
                     CLDA(uint num_components=NULL);
                    ~CLDA(void);
                    
                     matrix fit_transform(matrix &x, vector &y);
                     matrix transform(matrix &x);
                     vector transform(vector &x);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CLDA::CLDA(uint num_components=NULL)
:m_num_components(num_components)
 {
 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CLDA::~CLDA(void)
 {
 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
matrix  CLDA::fit_transform(matrix &x,vector &y)
 {
   vector classes = MatrixExtend::Unique(y);
   ulong num_classes = classes.Size();
   ulong num_features = x.Cols();
   
   vector class_means(classes.Size());
   
   for (ulong i=0; i<num_classes; i++)
    {
     matrix temp_x = {};
      for (ulong j=0, count=0; j<x.Rows(); j++)
         {
           if (y[j] == classes[i])
            {
               count++;
               temp_x.Resize(count, x.Cols());
               temp_x.Row(x.Row(j), count-1);
            }
         }
      class_means[i] = temp_x.Mean(0);
    }
    
  matrix ret={};
  return ret;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
