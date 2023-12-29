//+------------------------------------------------------------------+
//|                                                          NMF.mqh |
//|                                     Copyright 2023, Omega Joctan |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, Omega Joctan"
#property link      "https://www.mql5.com/en/users/omegajoctan"
//+------------------------------------------------------------------+
//| defines                                                          |
//+------------------------------------------------------------------+
#include <MALE5\matrix_utils.mqh>

class CNMF
  {
protected:
   uint m_components;
   uint m_max_iter;
   int m_randseed;
   ulong n_features;
   matrix W_;
   matrix H_;
   double m_tol; //loss tolerance
   
public:
                     CNMF(uint max_iter=100, double tol=1e-4, int random_state=-1);
                    ~CNMF(void);
                    
                    matrix fit_transform(matrix &X, uint k=2);
                    matrix transform(matrix &X);
                    uint select_best_components(matrix &X);
                    
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CNMF::CNMF(uint max_iter=100, double tol=1e-4,int random_state=-1)
 :m_max_iter(max_iter),
 m_randseed(random_state),
 m_tol(tol)
 {
   
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CNMF::~CNMF(void)
 {
 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
matrix CNMF::transform(matrix &X)
 {
  n_features = X.Cols();
  if (m_components>n_features)
     {
       printf("%s Number of dimensions K[%d] is supposed to be <= number of features %d",__FUNCTION__,m_components,n_features);
       this.m_components = (uint)n_features;
     }
     
  if (this.W_.Rows()==0 || this.H_.Rows()==0)
    {
      Print(__FUNCTION__," Model not fitted. Call fit method first.");
      matrix mat={};
      return mat;
    }
  
  return X.MatMul(this.H_.Transpose());
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
matrix CNMF::fit_transform(matrix &X, uint k=2)
 {
  ulong m = X.Rows(), n = X.Cols();
  double best_frobenius_norm = DBL_MIN;
  
   m_components = m_components == 0 ? (uint)n : k;      
   
   this.W_ = CMatrixutils::Random(0,1, m, this.m_components, this.m_randseed);
   this.H_ = CMatrixutils::Random(0,1,this.m_components, n, this.m_randseed);
   
//--- Update factors
      
   vector loss(this.m_max_iter);
    for (uint i=0; i<this.m_max_iter; i++)
      {
        // Update W
         this.W_ *= MathAbs((X.MatMul(this.H_.Transpose())) / (this.W_.MatMul(this.H_.MatMul(this.H_.Transpose()))+ 1e-10));
         
        // Update H
         this.H_ *= MathAbs((this.W_.Transpose().MatMul(X)) / (this.W_.Transpose().MatMul(this.W_.MatMul(this.H_))+ 1e-10));
         
         loss[i] = MathPow((X - W_.MatMul(H_)).Flat(1), 2);
                    
         // Calculate Frobenius norm of the difference
         
          double frobenius_norm = (X - W_.MatMul(H_)).Norm(MATRIX_NORM_FROBENIUS);

         //if (MQLInfoInteger(MQL_DEBUG))
         //  printf("%s [%d/%d] Loss = %.5f frobenius norm %.5f",__FUNCTION__,i+1,m_max_iter,loss[i],frobenius_norm);
         
          // heck convergence
          if (frobenius_norm < this.m_tol)
              break;
      }
  
  return this.W_.MatMul(this.H_); 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
uint CNMF::select_best_components(matrix &X)
{
    uint best_components = 1;
    this.m_components = (uint)X.Cols();
    
    vector explained_ratio(X.Cols());    
    for (uint k = 1; k <= X.Cols(); k++)
    {
       // Calculate explained variance or other criterion 
       matrix X_reduced = fit_transform(X, k);
   
       // Calculate explained variance as the ratio of squared Frobenius norms
       double explained_variance = 1.0 - (X-X_reduced).Norm(MATRIX_NORM_FROBENIUS) / (X.Norm(MATRIX_NORM_FROBENIUS));
        
        if (MQLInfoInteger(MQL_DEBUG))
            printf("k %d Explained Var %.5f",k,explained_variance);
       
       explained_ratio[k-1] = explained_variance;       
    }
    
    return uint(explained_ratio.ArgMax()+1);
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
