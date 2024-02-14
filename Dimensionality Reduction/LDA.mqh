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
#include <MALE5\MqPlotLib\plots.mqh>

enum lda_criterion //selecting best components criteria selection
  {
    CRITERION_VARIANCE,
    CRITERION_KAISER,
    CRITERION_SCREE_PLOT
  };

class CLDA
  {  
CPlots   plt;

protected:
   uint m_components;
   lda_criterion m_criterion;
   
   matrix projection_matrix;
   ulong num_features;
   double m_regparam;
   vector mean;
   
   uint calculate_variance(vector &eigen_values, double threshold=0.95);
   uint calculate_kaiser(vector &eigen_values);
   
   uint CLDA::extract_components(vector &eigen_values, double threshold=0.95);
   
public:
                     CLDA(uint k=NULL, lda_criterion CRITERION_=CRITERION_SCREE_PLOT, double reg_param =1e-6);
                    ~CLDA(void);
                    
                     matrix fit_transform(const matrix &x, const vector &y);
                     matrix transform(const matrix &x);
                     vector transform(const vector &x);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CLDA::CLDA(uint k=NULL, lda_criterion CRITERION_=CRITERION_SCREE_PLOT, double reg_param=1e-6)
:m_components(k),
 m_criterion(CRITERION_),
 m_regparam(reg_param)
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
matrix  CLDA::fit_transform(const matrix &x, const vector &y)
 {
   vector classes = MatrixExtend::Unique(y);
   ulong num_classes = classes.Size();
   num_features = x.Cols();
   
   this.mean = x.Mean(0);
   
   matrix x_centered = Base::subtract(x, this.mean);
   
   matrix class_means(classes.Size(), x.Cols());
   class_means.Fill(0.0);
   
   for (ulong i=0; i<num_classes; i++)
    {
     matrix class_samples = {};
      for (ulong j=0, count=0; j<x.Rows(); j++)
         {
           if (y[j] == classes[i])
            {  
               count++;
               class_samples.Resize(count, num_features);
               class_samples.Row(x.Row(j), count-1);
            }
         }
         
      class_means.Row(class_samples.Mean(0), i);
    }
    
    
  matrix SW, SB; //within and between scatter matrices 
  SW.Init(num_features, num_features);
  SB.Init(num_features, num_features);
  
  for (ulong i=0; i<num_classes; i++)
   {
     matrix class_samples = {};
      for (ulong j=0, count=0; j<x.Rows(); j++)
         {
           if (y[j] == classes[i]) //Collect a matrix for samples belonging to a particular class
            {
               count++;
               class_samples.Resize(count, num_features);
               class_samples.Row(x.Row(j), count-1);
            }
         }

         
     matrix diff = Base::subtract(class_samples, class_means.Row(i)); //Each row subtracted to the mean
     if (diff.Rows()==0 && diff.Cols()==0) //if the subtracted matrix is zero stop the program for possible bugs or errors
      {
        DebugBreak();
        return x_centered;
      }
     
     SW += diff.Transpose().MatMul(diff); //Find within scatter matrix 
     
     vector mean_diff = class_means.Row(i) - x_centered.Mean(0);
     SB += class_samples.Rows() * mean_diff.Outer(mean_diff); //compute between scatter matrix 
   }
  
//--- Regularization to avoid errors while calculating Eigen values and vectors
   
   SW += this.m_regparam * MatrixExtend::eye((uint)num_features);
   SB += this.m_regparam * MatrixExtend::eye((uint)num_features);

//---

  matrix eigen_vectors(x.Cols(), x.Cols());  eigen_vectors.Fill(0.0);
  vector eigen_values(x.Cols()); eigen_values.Fill(0.0);
  
  matrix SBSW = SW.Inv().MatMul(SB);
  
  Base::ReplaceNaN(SBSW);
  
  if (!SBSW.Eig(eigen_vectors, eigen_values))
    {
      Print("%s Failed to calculate eigen values and vectors Err=%d",__FUNCTION__,GetLastError());
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
   
  if (this.m_components == NULL)
   {
     this.m_components = extract_components(eigen_values);
     if (this.m_components==0)
      {
        printf("%s Failed to auto detect the best components\n You need to select the value of k yourself by looking at the scree plot",__FUNCTION__);
        this.m_components = (uint)x.Cols();
      }
   }
  else //plot the scree plot 
    extract_components(eigen_values);
    
  this.projection_matrix = Base::Slice(eigen_vectors, this.m_components);
    
  return x_centered.MatMul(projection_matrix.Transpose());
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
matrix CLDA::transform(const matrix &x)
 {
   if (this.projection_matrix.Rows() == 0)
    {
      printf("%s fit_transform method must be called befor transform",__FUNCTION__);
      matrix empty = {};
      return empty; 
    }
  matrix x_centered = Base::subtract(x, this.mean);
  
  return x_centered.MatMul(this.projection_matrix.Transpose());  
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CLDA::transform(const vector &x)
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
uint CLDA::calculate_variance(vector &eigen_values, double threshold=0.95)
 {
  uint k=0; 
  
   vector eigen_pow = MathPow(eigen_values, 2);
   vector cum_sum = eigen_pow.CumSum();
   double sum = eigen_pow.Sum();
   
   vector cumulative_variance =  cum_sum / sum;
   
   if (MQLInfoInteger(MQL_DEBUG))
     Print("Cummulative variance: ",cumulative_variance);
   
   vector v(cumulative_variance.Size());  v.Fill(0.0);
   for (ulong i=0; i<v.Size(); i++)
     v[i] = (cumulative_variance[i] >= threshold);
      
   k = (uint)v.ArgMax() + 1;
   
   return k;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
uint CLDA::calculate_kaiser(vector &eigen_values)
 {
  vector v(eigen_values.Size()); v.Fill(0.0);
   for (ulong i=0; i<eigen_values.Size(); i++)
     v[i] = (eigen_values[i] >= 1);
   
   return uint(v.Sum());
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
uint CLDA::extract_components(vector &eigen_values, double threshold=0.95)
 {
  uint k = 0;
  
   switch(m_criterion)
     {
      case  CRITERION_VARIANCE: 
         k = calculate_variance(eigen_values, threshold);
         
        break;
        
      case  CRITERION_KAISER:
         k = calculate_kaiser(eigen_values);
        
        break;
        
      case  CRITERION_SCREE_PLOT:
       {  
         vector v_cols(eigen_values.Size());
         
         for (ulong i=0; i<v_cols.Size(); i++)
             v_cols[i] = (int)i+1;
             
          vector vars = eigen_values;
          
          plt.ScatterCurvePlots("Scree plot",v_cols,vars,"EigenValue","LDA","EigenValue");

//---
      string warn = "\n<<<< WARNING >>>>\nThe Scree plot doesn't return the determined number of k m_components\nThe cummulative variance Or kaiser will return the number of k m_components instead\nThe k returned might be different from what you see on the scree plot";
             warn += "\nTo apply the same number of k m_components to the LDA from the scree plot\nCall the LDA model again with that value applied from the plot\n";
      
         Print(warn);
        
        //--- Kaiser
           
           k = calculate_kaiser(eigen_values);
            
            if (k==0) //kaiser wasn't suitable in this particular task
              k = calculate_variance(eigen_values, threshold);
        }          
           
        break;
     } 
     
   return (k);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

