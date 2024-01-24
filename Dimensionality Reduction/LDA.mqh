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
   ulong num_features;
   
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
   num_features = x.Cols();
   
   matrix class_means(x.Cols(), classes.Size());
   
   for (ulong i=0; i<num_classes; i++)
    {
     matrix class_samples = {};
      for (ulong j=0, count=0; j<x.Rows(); j++)
         {
           if (y[j] == classes[i])
            {
               count++;
               class_samples.Resize(count, x.Cols());
               class_samples.Row(x.Row(j), count-1);
            }
         }
      
      class_means.Col(class_samples.Mean(0), i);
    }
  
  matrix SW, SB;
  SW.Init(num_features, num_features);
  SB.Init(num_features, num_features);
  
  for (ulong i=0; i<num_classes; i++)
   {
     matrix class_samples = {};
      for (ulong j=0, count=0; j<x.Rows(); j++)
         {
           if (y[j] == classes[i])
            {
               count++;
               class_samples.Resize(count, x.Cols());
               class_samples.Row(x.Row(j), count-1);
            }
         }
         
     matrix diff = Base::subtract(class_samples, class_means.Col(i)); 
     SW += diff.Transpose().MatMul(diff);
     
     vector mean_diff = class_means.Col(i) - x.Mean(0);
     SB += class_samples.Rows() * mean_diff.Outer(mean_diff);
   }
  
  matrix eigen_vectors;
  vector eigen_values;
  
  matrix SBSW = SW.Inv().MatMul(SB);
  if (!SBSW.Eig(eigen_vectors, eigen_values))
    {
      Print("%s Failed to calculate eigen values and vectors Err=%d",__FUNCTION__,GetLastError());
      DebugBreak();
      
      matrix empty = {};
      return empty;
    }
   
   if (eigen_vectors.Rows()==0 || eigen_values.Size()==0)
    {
      printf("%s Zero eigen values or eigen vectors, check your data",__FUNCTION__);
      DebugBreak();
      
      matrix empty = {};
      return empty;
    }
    
//--- Sort eigenvectors by decreasing eigenvalues
   
   vector args = MatrixExtend::ArgSort(eigen_values);
   MatrixExtend::Reverse(args);
   
   eigen_values = Base::Sort(eigen_values, args);
   eigen_vectors = Base::Sort(eigen_vectors, args);
   
//---
   
  if (this.m_num_components == NULL)
    this.projection_matrix = eigen_vectors;
  else
    this.projection_matrix = Base::Slice(eigen_vectors, this.m_num_components);
    
   return transform(x);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
matrix CLDA::transform(matrix &x)
 {
   if (this.projection_matrix.Rows() == 0)
    {
      printf("%s fit_transform method must be called befor transform",__FUNCTION__);
      matrix empty = {};
      return empty; 
    }
    
  return x.MatMul(this.projection_matrix);  
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CLDA::transform(vector &x)
 {
   matrix m = MatrixExtend::VectorToMatrix(x, this.num_features); 
   
   if (m.Rows()==0)
    {
      vector empty={};
      return empty; //return nothing since there is a failure in converting vector to matrix
    }
   
   m = transform(m);
   return MatrixExtend::MatrixToVector(m);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

