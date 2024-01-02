//+------------------------------------------------------------------+
//|                                                 CTruncatedSVD.mqh |
//|                                     Copyright 2023, Omega Joctan |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, Omega Joctan"
#property link      "https://www.mql5.com/en/users/omegajoctan"
//+------------------------------------------------------------------+
//| defines                                                          |
//+------------------------------------------------------------------+
#include "helpers.mqh"
#include <MALE5\MqPlotLib\plots.mqh>

class CTruncatedSVD
  {
CPlots   plt;

uint m_components;
ulong n_features;
matrix components_;
vector mean;
vector explained_variance_;

public:
                     CTruncatedSVD(uint k=0);
                    ~CTruncatedSVD(void);
                    
                    matrix fit_transform(matrix& X);
                    matrix transform(matrix &X);
                    vector transform(vector &X);
                    ulong _select_n_components(vector &singular_values);
  };
//+------------------------------------------------------------------+
//|  Once the k value is left to default value of zero, the function |
//| _select_n_components will be used to find the best number of     |
//| components to use                                                |
//+------------------------------------------------------------------+
CTruncatedSVD::CTruncatedSVD(uint k=0)
:m_components(k)
 {
 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CTruncatedSVD::~CTruncatedSVD(void)
 {
 
 } 
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
matrix CTruncatedSVD::fit_transform(matrix &X)
 {
  n_features = X.Cols();  
    
   if (m_components>n_features)
     {
       printf("%s Number of dimensions K[%d] is supposed to be <= number of features %d",__FUNCTION__,m_components,n_features);
       this.m_components = (uint)n_features;
     }
   
    this.mean = X.Mean(0);
    
    // Center the data (subtract mean)
    matrix X_centered = CDimensionReductionHelpers::subtract(X, this.mean);
    
    // Compute the covariance matrix
   
    CDimensionReductionHelpers::ReplaceNaN(X_centered);
    matrix cov_matrix = X_centered.Cov(false);
    
    CDimensionReductionHelpers::ReplaceNaN(cov_matrix);
    
   // Perform SVD on the covariance matrix
    matrix U={}, Vt={};
    vector Sigma={};
    
    if (!cov_matrix.SVD(U,Vt,Sigma))
       Print(__FUNCTION__," Line ",__LINE__," Failed to calculate SVD Err=",GetLastError());    
        
     if (m_components == 0)
       {
         m_components = (uint)this._select_n_components(Sigma);
         Print(__FUNCTION__," Best value of K = ",m_components);
       }
                 
    this.components_ = CDimensionReductionHelpers::Slice(Vt, this.m_components).Transpose();
    CDimensionReductionHelpers::ReplaceNaN(this.components_);
        
    if (MQLInfoInteger(MQL_DEBUG))
      Print("components_T[",components_.Rows(),"X",components_.Cols(),"]\n",this.components_);
    
    this.explained_variance_ = MathPow(CDimensionReductionHelpers::Slice(Sigma, this.m_components), 2) / (X.Rows() - 1);
    
    return X_centered.MatMul(components_);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
matrix CTruncatedSVD::transform(matrix &X)
 {
   matrix X_centered = CDimensionReductionHelpers::subtract(X, this.mean);
   
   if (X.Cols()!=this.n_features)
     {
       printf("%s Inconsistent input X matrix size, It is supposed to be of size %d same as the matrix used under fit_transform",__FUNCTION__,n_features);
       this.m_components = (uint)n_features;
     }
    
    return X_centered.MatMul(components_);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CTruncatedSVD::transform(vector &X)
 {
   matrix INPUT_MAT = CMatrixutils::VectorToMatrix(X, X.Size());
   matrix OUTPUT_MAT = transform(INPUT_MAT);
      
   return CMatrixutils::MatrixToVector(OUTPUT_MAT);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
ulong CTruncatedSVD::_select_n_components(vector &singular_values)
 {
    double total_variance = MathPow(singular_values.Sum(), 2);
    
    vector explained_variance_ratio = MathPow(singular_values, 2).CumSum() / total_variance;
    
    if (MQLInfoInteger(MQL_DEBUG))
      Print(__FUNCTION__," Explained variance ratio ",explained_variance_ratio);
    
    vector k(explained_variance_ratio.Size());
    
    for (uint i=0; i<k.Size(); i++)
      k[i] = i+1;
    
    plt.ScatterCurvePlots("Explained variance plot",k,explained_variance_ratio,"variance","components","Variance");
    
   return explained_variance_ratio.ArgMax() + 1;  //Choose k for maximum explained variance
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
