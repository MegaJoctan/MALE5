//+------------------------------------------------------------------+
//|                                                 MatrixExtend::mqh |
//|                                  Copyright 2022, Omega Joctan  . |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, MetaQuotes Ltd."
#property link      "https://www.mql5.com/en/users/omegajoctan"
//+------------------------------------------------------------------+
//| defines                                                          |
//+------------------------------------------------------------------+
#include <MALE5\preprocessing.mqh>
#include <MALE5\MatrixExtend.mqh>
#include <MALE5\metrics.mqh>
 
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CLogisticRegression
  {
private:
                    
                    vector classes_in_data;
                    
                     bool istrained;          
                     bool checkIsTrained(string func)
                       {
                         if (!istrained)
                           {
                             Print(func," Tree not trained, Call fit function first to train the model");
                             return false;   
                           }
                         return (true);
                       }
                    
                    matrix weights;
                    double bias;
                    
                  //---
                    
                    uint m_epochs;
                    double m_alpha;
                    double m_tol;
                    
public:
                     CLogisticRegression(uint epochs=10, double alpha=0.01, double tol=1e-8);
                    ~CLogisticRegression(void);
                    
       
                    void fit(matrix &x, vector &y);        
                         
                    int    predict(vector &x);
                    vector predict(matrix &x);
                    double predict_proba(vector &x);
                    vector predict_proba(matrix &x);
                    
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CLogisticRegression::CLogisticRegression(uint epochs=10, double alpha=0.01, double tol=1e-8)
 :istrained(false),
  m_epochs(epochs),
  m_alpha(alpha),
  m_tol(tol)
 {
 
 }
//+------------------------------------------------------------------+
//| This is where the logistic model gets trained                    |
//+------------------------------------------------------------------+
void CLogisticRegression::fit(matrix &x, vector &y)
 {
   ulong m = x.Rows(), n = x.Cols();
   
   this.weights = MatrixExtend::Random(-1,1,n,1,42);
   
   matrix dw; //derivative wrt weights & 
   double db; //bias respectively
   vector preds;
   
   istrained = true;
   
   if (MQLInfoInteger(MQL_DEBUG))
      printf("x[%dx%d] w[%dx%d]",x.Rows(),x.Cols(),weights.Rows(),weights.Cols());
   
   double prev_cost = -DBL_MAX, cost =0;
   for (ulong i=0; i<m_epochs; i++)
     { 
       preds = this.predict_proba(x);    
       
       //-- Computing gradient(s)
       
       matrix error = MatrixExtend::VectorToMatrix(preds - y);
       
       dw = (1/(double)m) * x.Transpose().MatMul(error);
       db = (1/(double)m) * (preds - y).Sum();
       
       cost = Metrics::mse(y, preds);
       
       printf("[%d/%d] mse %.5f",i+1,m_epochs, cost);
       
       this.weights -= this.m_alpha * dw;
       this.bias -= this.bias * db;
       
       if (MathAbs(prev_cost - cost) < this.m_tol)
        {
          Print("Converged!!!");
          break;
        }
       
       prev_cost = cost;
     }
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CLogisticRegression::~CLogisticRegression(void)
 {   
   
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int CLogisticRegression::predict(vector &x)
 {
   if (!checkIsTrained(__FUNCTION__))
    return 0;
   
   matrix x_mat = MatrixExtend::VectorToMatrix(x, x.Size());
   
   matrix preds = (x_mat.MatMul(this.weights) + this.bias);
   
   preds.Activation(preds, AF_HARD_SIGMOID);
   
   if (preds.Rows()>1)
    {
      printf("%s The outcome from a sigmoid must be a scalar value",__FUNCTION__);
      return 0;
    }
   return (int)(preds[0][0]>=0.5);   
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CLogisticRegression::predict(matrix &x)
 {
   vector v(x.Rows());
   for (ulong i=0; i<x.Rows(); i++)
      v[i] = this.predict(x.Row(i));
      
   return v;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CLogisticRegression::predict_proba(vector &x)
 {
   if (!checkIsTrained(__FUNCTION__))
    return 0;
   
   matrix x_mat = MatrixExtend::VectorToMatrix(x, x.Size());
   
   matrix preds = (x_mat.MatMul(this.weights) + this.bias);
   
   preds.Activation(preds, AF_HARD_SIGMOID);
   
   if (preds.Rows()>1)
    {
      printf("%s The outcome from a sigmoid must be a scalar value",__FUNCTION__);
      return 0;
    }
   return preds[0][0];   
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CLogisticRegression::predict_proba(matrix &x)
 {
   vector v(x.Rows());
   for (ulong i=0; i<x.Rows(); i++)
      v[i] = this.predict_proba(x.Row(i));
      
   return v;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
