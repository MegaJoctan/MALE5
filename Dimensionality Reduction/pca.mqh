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
#include <MALE5\MatrixExtend.mqh>
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

protected:
   uint              m_components;
   criterion         m_criterion;
   
   matrix            components_matrix;
   vector            mean;   
   
   uint              n_features;
                     
                     
                     uint extract_components(vector &eigen_values, double threshold=0.95);
                     
public:
                     CPCA(int k=0, criterion CRITERION_=CRITERION_SCREE_PLOT);
                    ~CPCA(void);
                    
                     matrix fit_transform(matrix &X);
                     matrix transform(matrix &X);
                     vector transform(vector &X);
                     bool save(string dir);
                     bool load(string dir);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CPCA::CPCA(int k=0, criterion CRITERION_=CRITERION_SCREE_PLOT)
 :m_components(k),
  m_criterion(CRITERION_)
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
   n_features = (uint)X.Cols();
   
   if (m_components>n_features)
     {
       printf("%s Number of dimensions K[%d] is supposed to be <= number of features %d",__FUNCTION__,m_components,n_features);
       this.m_components = (int)n_features;
       Print(__LINE__);
     }

//---
   
   this.mean = X.Mean(0);
   
   matrix X_centered = CDimensionReductionHelpers::subtract(X, this.mean);   
   CDimensionReductionHelpers::ReplaceNaN(X_centered);
   
   matrix cov_matrix = cova(X_centered, false);
   
   matrix eigen_vectors;
   vector eigen_values;
    
   CDimensionReductionHelpers::ReplaceNaN(cov_matrix);
   
   if (!cov_matrix.Eig(eigen_vectors, eigen_values))
     printf("Failed to caculate Eigen matrix and vectors Err=%d",GetLastError());
   
//--- Sort eigenvectors by decreasing eigenvalues
   
   vector args = MatrixExtend::ArgSort(eigen_values); MatrixExtend::Reverse(args);
   
   eigen_values = CDimensionReductionHelpers::Sort(eigen_values, args);
   eigen_vectors = CDimensionReductionHelpers::Sort(eigen_vectors, args);
//---

   if (MQLInfoInteger(MQL_DEBUG))
      Print("Eigen values: ",eigen_values);
      
   if (m_components==0)
     m_components = this.extract_components(eigen_values);
   else
     this.extract_components(eigen_values);
   
   if (MQLInfoInteger(MQL_DEBUG)) 
     printf("%s Selected components %d",__FUNCTION__,m_components);
   
   this.components_matrix = CDimensionReductionHelpers::Slice(eigen_vectors, m_components, 1); //Get the components matrix
   //MatrixExtend::NormalizeDouble_(this.components_matrix, 5);
   //this.components_matrix = scaler.fit_transform(this.components_matrix.Transpose()); //Normalize components matrix
   
   this.components_matrix = this.components_matrix.Transpose();
   
   if (MQLInfoInteger(MQL_DEBUG))
     Print("components_matrix\n",components_matrix);
   
//---
      
   return X_centered.MatMul(components_matrix.Transpose()); //return the pca scores
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
     
   matrix X_centered = CDimensionReductionHelpers::subtract(X, this.mean);

   return X_centered.MatMul(this.components_matrix.Transpose()); //return the pca scores
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CPCA::transform(vector &X)
 {
   matrix INPUT_MAT = MatrixExtend::VectorToMatrix(X, X.Size());
   matrix OUTPUT_MAT = transform(INPUT_MAT);
      
   return MatrixExtend::MatrixToVector(OUTPUT_MAT);
 }
//+------------------------------------------------------------------+
//|   Select the number of components based on some criterion        |
//+------------------------------------------------------------------+
uint CPCA::extract_components(vector &eigen_values, double threshold=0.95)
 {
  uint k = 0;
  
   vector eigen_pow = MathPow(eigen_values, 2);
   vector cum_sum = eigen_pow.CumSum();
   double sum = eigen_pow.Sum();
   
   switch(m_criterion)
     {
      case  CRITERION_VARIANCE: 
         {              
            
            vector cumulative_variance =  cum_sum / sum;
            
            if (MQLInfoInteger(MQL_DEBUG))
              Print("Cummulative variance: ",cumulative_variance);
            
            vector v(cumulative_variance.Size());  v.Fill(0.0);
            for (ulong i=0; i<v.Size(); i++)
              v[i] = (cumulative_variance[i] >= threshold);
               
            k = (uint)v.ArgMax() + 1;
         }  
         
        break;
        
      case  CRITERION_KAISER:
         {
           vector v(eigen_values.Size()); v.Fill(0.0);
            for (ulong i=0; i<eigen_values.Size(); i++)
              v[i] = (eigen_values[i] >= 1);
            
            k = uint(v.Sum());
         } 
        
        break;
        
      case  CRITERION_SCREE_PLOT:
       {  
         vector v_cols(eigen_values.Size());
         
         for (ulong i=0; i<v_cols.Size(); i++)
             v_cols[i] = (int)i+1;
             
          vector vars = eigen_values;
          
          //matrix_utils.Sort(vars); //Make sure they are in ascending first order
          //matrix_utils.Reverse(vars);  //Set them to descending order
          
          plt.ScatterCurvePlots("Scree plot",v_cols,vars,"EigenValue","PCA","EigenValue");

//---
      string warn = "\n<<<< WARNING >>>>\nThe Scree plot doesn't return the determined number of k components\nThe cummulative variance will return the number of k components instead\nThe k returned might be different from what you see on the scree plot";
             warn += "\nTo apply the same number of k components to the PCA from the scree plot\nCall the PCA model again with that value applied from the plot\n";
      
         Print(warn);
        
        //--- Kaiser
        
           vector v(eigen_values.Size()); v.Fill(0.0);
            for (ulong i=0; i<eigen_values.Size(); i++)
              v[i] = (eigen_values[i] >= 1);
            
            k = uint(v.Sum());
        }          
           
        break;
     } 
     
   return (k);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CPCA::save(string dir)
 { 
   
   matrix m = MatrixExtend::VectorToMatrix(this.mean, this.mean.Size());
   
   if (!MatrixExtend::WriteCsv(dir+"\\PCA-Mean.csv",m,NULL,false,8))
     {
       Print("Failed to Save PCA-Mean information to ",dir);
       return false;
     }

//---

   if (!MatrixExtend::WriteCsv(dir+"\\PCA-ComponentsMatrix.csv",this.components_matrix,NULL,false,8))
     {
       Print("Failed to Save PCA-ComponentsMatrix information to ",dir);
       return false;
     }
     
   return true;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CPCA::load(string dir)
 {
   string header;
   matrix m = MatrixExtend::ReadCsv(dir+"\\PCA-Mean.csv",header);
   
   if (m.Rows()==0)
     return false;
     
   this.mean = MatrixExtend::MatrixToVector(m);
   
//---
   
   this.components_matrix = MatrixExtend::ReadCsv(dir+"\\PCA-ComponentsMatrix.csv",header);
   
   if (components_matrix.Rows()==0)
     return false;
     
   return true;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

matrix cova(matrix &data, bool row_var=true)
  {
    if (row_var)
        data = data.Transpose();  // Transpose if each row represents a data point

    // Step 1: Center the data
    matrix centered_data = CDimensionReductionHelpers::subtract(data, data.Mean(0));

    // Step 2: Calculate the covariance matrix
    matrix covariance_matrix = centered_data.Transpose().MatMul(centered_data) / (data.Rows() - 1);

    return covariance_matrix;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+