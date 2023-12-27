//+------------------------------------------------------------------+
//|                                                          pca.mqh |
//|                                    Copyright 2022, Fxalgebra.com |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, Fxalgebra.com"
#property link      "https://www.mql5.com/en/users/omegajoctan"
//+------------------------------------------------------------------+
//|         Principle Component Analysis Library                     |
//+------------------------------------------------------------------+
#include <MALE5\MqPlotLib\plots.mqh>
#include <MALE5\matrix_utils.mqh>
#include "helpers.mqh"

enum criterion
  {
    CRITERION_VARIANCE,
    CRITERION_KAISER,
    CRITERION_SCREE_PLOT
  };
//+------------------------------------------------------------------+
//|            Principal Component Analysis Class                    |
//+------------------------------------------------------------------+
class CPCA
  {
CPlots   plt;
CMatrixutils matrix_utils;

protected:
   int               m_components;
   matrix            components_matrix;
   vector            mean;   
   int n_features;
                     
public:
                     CPCA(int k_components=2);
                    ~CPCA(void);
                    
                     matrix fit_transform(matrix &X);
                     matrix transform(matrix &X);
                     matrix extract_components(matrix &X, criterion CRITERION_);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CPCA::CPCA(int k_components=2)
 :m_components(k_components)
 {
 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CPCA::~CPCA(void)
 {
 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
matrix CPCA::fit_transform(matrix &X)
 { 
   n_features = (int)X.Cols();
   
   if (m_components>n_features)
     {
       printf("%s Number of dimensions K[%d] is supposed to be <= number of features %d",__FUNCTION__,m_components,n_features);
       this.m_components = (int)n_features;
     }

//---

   this.mean = X.Mean(0);
   
   matrix standardized_data = CDimensionReductionHelpers::subtract(X, this.mean);
   
   matrix cov_matrix = X.Cov(false);
   
   matrix eigen_vectors;
   vector eigen_values;
   
   if (!cov_matrix.Eig(eigen_vectors, eigen_values))
     printf("Failed to caculate Eigen matrix and vectors Err=%d",GetLastError());
   
//--- Sort eigenvectors by decreasing eigenvalues

   eigen_values = matrix_utils.Sort(eigen_values); //Sort ascending
   matrix_utils.Reverse(eigen_values); //Reverse the order
   
   /*
   Print("eigen values: ",eigen_values);
   Print("eigen vectors:\n",eigen_vectors);
   */
   
   if (m_components==0)
     components_matrix = eigen_vectors;
   else
      this.components_matrix = CDimensionReductionHelpers::Slice(eigen_vectors, m_components, 1);
   
   //Print("components_matrix\n",components_matrix);
   
//---
      
   return standardized_data.MatMul(components_matrix); //return the pca scores
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
matrix CPCA::transform(matrix &X)
 {
   if (X.Cols()!=this.n_features)
     {
       printf("%s Inconsistent input X matrix size, It is supposed to be of size %d same as the matrix used under fit_transform",__FUNCTION__,n_features);
       this.m_components = n_features;
     }
     
   matrix standardized_data = CDimensionReductionHelpers::subtract(X, this.mean);
  
   return standardized_data.MatMul(this.components_matrix); //return the pca scores
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
matrix CPCA::extract_components(matrix &X, criterion CRITERION_)
 {
   ulong rows = X.Rows(),
         cols = X.Cols();
         
   vector pca_scores_coefficients(cols);
   matrix pca_scores = this.fit_transform(X);
   
   for (ulong i=0; i<cols; i++)
       pca_scores_coefficients[i] = pca_scores.Col(i).Var(); //variance of the pca scores
     
  vector vars = pca_scores_coefficients;   
  vector vars_percents = (vars/(double)vars.Sum())*100.0;
  
//--- for Kaiser

  double vars_mean = pca_scores_coefficients.Mean();

//--- for scree
   

  matrix PCAS = {};
  
  double sum=0;
  ulong  max;
  vector<double> v_cols = {};
   
   switch(CRITERION_)
     {
  
      case  CRITERION_VARIANCE: 
      
       #ifdef DEBUG_MODE
        Print("vars percentages ",vars_percents);       
       #endif 
       
         for (int i=0, count=0; i<(int)cols; i++)
           { 
             count++;
             
              max = vars_percents.ArgMax();
              sum += vars_percents[max];
              
              vars_percents[max] = 0; 
              
              v_cols.Resize(count);
              v_cols[count-1] = (int)max;
           }
         
         PCAS.Resize(rows, v_cols.Size());
         
         for (ulong i=0; i<v_cols.Size(); i++)
            PCAS.Col(pca_scores.Col((ulong)v_cols[i]), i);
         
        break;
      case  CRITERION_KAISER:
      
      #ifdef DEBUG_MODE
         Print("var ",vars," scores mean ",vars_mean);
      #endif 
      
       vars = pca_scores_coefficients;
        for (ulong i=0, count=0; i<cols; i++)
           if (vars[i] > vars_mean)
             {
               count++;
       
               PCAS.Resize(rows, count);
               
               PCAS.Col(pca_scores.Col(i), count-1);
             }           
           
        break;
      case  CRITERION_SCREE_PLOT:
         
         v_cols.Resize(cols);
         
         for (ulong i=0; i<v_cols.Size(); i++)
             v_cols[i] = (int)i+1;
             
         
          vars = pca_scores_coefficients;
          
          if (MQLInfoInteger(MQL_DEBUG))
            Print("pca_scores_coefficients: ",vars," | ",v_cols);
          
          matrix_utils.Sort(vars); //Make sure they are in ascending first order
          matrix_utils.Reverse(vars);  //Set them to descending order
          
          plt.ScatterCurvePlots("Scree plot",v_cols,vars,"variance","PCA","Variance");

//---

       vars = pca_scores_coefficients;
        for (ulong i=0, count=0; i<cols; i++)
           if (vars[i] > vars_mean)
             {
               count++;
       
               PCAS.Resize(rows, count);
               
               PCAS.Col(pca_scores.Col(i), count-1);
             }    
             
        break;
     } 
   return (PCAS);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+