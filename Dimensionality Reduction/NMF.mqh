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
   
   void update_factors(matrix &X);
   void initialize_factors(matrix &X);
   double calculate_explained_variance(matrix &X);
public:
                     CNMF(uint max_iter=100, int random_state=-1);
                    ~CNMF(void);
                    
                    matrix fit_transform(matrix &X, uint k=2);
                    matrix transform(matrix &X);
                    uint select_best_components(matrix &X);
                    
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CNMF::CNMF(uint max_iter=100,int random_state=-1)
 :m_max_iter(max_iter),
 m_randseed(random_state)
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
void CNMF::initialize_factors(matrix &X)
 { 
   ulong m = X.Rows(), n = X.Cols();
   
   if (m_components == 0)
     this.m_components = (uint)n;
   
   this.W_ = CMatrixutils::Random(0,1, m, this.m_components, this.m_randseed);
   this.H_ = CMatrixutils::Random(0,1,this.m_components, n, this.m_randseed);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNMF::update_factors(matrix &X)
 {
    vector loss(X.Rows());
    for (uint i=0; i<this.m_max_iter; i++)
      {
        // Update W
         this.W_ *= MathAbs((X.MatMul(this.H_.Transpose())) / (this.W_.MatMul(this.H_.MatMul(this.H_.Transpose()))+ 1e-10));
         
        // Update H
         this.H_ *= MathAbs((this.W_.Transpose().MatMul(X)) / (this.W_.Transpose().MatMul(this.W_.MatMul(this.H_))+ 1e-10));
         
         loss[i] = MathPow((X - W_.MatMul(H_)).Flat(1), 2);
         
         //if (MQLInfoInteger(MQL_DEBUG))
         //  printf("%s [%d/%d] Loss = %.5f",__FUNCTION__,i,m_max_iter,loss[i]);
      }
 }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
matrix CNMF::fit_transform(matrix &X, uint k=2)
 {
  n_features = X.Cols();
  m_components = k;
  
  initialize_factors(X);
  update_factors(X); 
  
  return transform(X); 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CNMF::calculate_explained_variance(matrix &X)
{   
    matrix X_approx = fit_transform(X, (uint)X.Cols());
    matrix diff = (X - X_approx);

    double frobenius_norm_original = X.Norm(MATRIX_NORM_FROBENIUS);
    double frobenius_norm_diff = diff.Norm(MATRIX_NORM_FROBENIUS);

    // Calculate explained variance as the ratio of squared Frobenius norms
    double explained_variance = 1.0 - ((frobenius_norm_diff) / ( frobenius_norm_original));

    return explained_variance;
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
uint CNMF::select_best_components(matrix &X)
{
    uint best_components = 1;
    double best_explained_variance = -DBL_MAX;
    this.m_components = (uint)X.Cols();
        
    for (uint k = 1; k <= X.Cols(); k++)
    {
        // Calculate explained variance or other criterion
        double explained_variance = calculate_explained_variance(X);
        
        if (MQLInfoInteger(MQL_DEBUG))
            printf("k %d Explained Var %.5f",k,explained_variance);
              
        // Update best_components if the current k provides a better result
        if (explained_variance > best_explained_variance)
        {
            best_components = k;
            best_explained_variance = explained_variance;
        }
    }

    return best_components;
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
