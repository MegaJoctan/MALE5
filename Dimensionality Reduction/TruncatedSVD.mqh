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

class CTruncatedSVD
  {
uint m_components;
ulong n_features;
matrix components_;
vector explained_variance_;

public:
                     CTruncatedSVD(uint k=2);
                    ~CTruncatedSVD(void);
                    
                    matrix fit_transform(matrix& X);
                    matrix transform(matrix &X);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CTruncatedSVD::CTruncatedSVD(uint k=2)
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
   
    // Center the data (subtract mean)
    matrix X_centered = CDimensionReductionHelpers::subtract(X, X.Mean(0));
    
    //Print("X_centered\n",X_centered);
    
   // Compute the covariance matrix
    //matrix cov_matrix = X_centered.Cov(false);
    
   // Perform SVD on the covariance matrix
    matrix U, Vt;
    vector Sigma;
    
    if (!X_centered.SVD(U,Vt,Sigma))
       Print(__FUNCTION__," Line ",__LINE__," Failed to calculate SVD Err=",GetLastError());    
    
    //Print("u\n",U,"\nsigma\n",Sigma,"\nVt\n",Vt);
    
    this.components_ = CDimensionReductionHelpers::Slice(Vt, this.m_components).Transpose();
    
    Print("components\n",CDimensionReductionHelpers::Slice(Vt, this.m_components),"\ncomponents_T\n",this.components_);
    
    this.explained_variance_ = MathPow(CDimensionReductionHelpers::Slice(Sigma, this.m_components), 2) / (X.Rows() - 1);
    
    return X_centered.MatMul(components_);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
matrix CTruncatedSVD::transform(matrix &X)
 {
   if (X.Cols()!=this.n_features)
     {
       printf("%s Inconsistent input X matrix size, It is supposed to be of size %d same as the matrix used under fit_transform",__FUNCTION__,n_features);
       this.m_components = (uint)n_features;
     }
     
  return X.MatMul(components_);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
