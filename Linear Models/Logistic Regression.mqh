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
                    
                    vector classes;
                    
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
                    
                    double Logit(vector &x);
                    vector Logit(matrix &x); //Logitstic loss function for training 

                    void ParameterEstimationGrad(matrix &x, vector &y, uint epochs=1000, double alpha=0.01, double tol=1e-8);
                    
                    matrix Betas;
                                        
                    vector Odds(vector &proba);
                    vector lnOdss(vector &odds);

                    
public:
                     CLogisticRegression(void);
                    ~CLogisticRegression(void);
                    
       
                    void fit(matrix &x, vector &y, double alpha=0.01, uint epochs=1000, double tol=1e-8);        
                         
                    int    predict(vector &v);
                    vector predict(matrix &mat);
                    vector predict_proba(vector &v);
                    
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CLogisticRegression::CLogisticRegression(void)
 :istrained(false)
 {
 
 }
//+------------------------------------------------------------------+
//| This is where the logistic model gets trained                    |
//+------------------------------------------------------------------+
void CLogisticRegression::fit(matrix &x, vector &y, double alpha=0.01, uint epochs=1000, double tol=1e-8)
 {
   ulong rows = x.Rows();
   ulong cols = x.Cols();

   Betas.Resize(cols+1, 1);
   Betas.Fill(0);
   
//---
   
   classes = MatrixExtend::Unique(y);
    
    
   ParameterEstimationGrad(x,y, epochs, alpha, tol);
   
//---
      
   #ifdef DEBUG_MODE
      Print("Betas\n",Betas);
   #endif 
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
vector CLogisticRegression::Odds(vector &proba)
 {
   return (proba/(1-proba)); 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CLogisticRegression::lnOdss(vector &odds)
 {
   return (log(odds)); 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CLogisticRegression::Logit(vector &x)
 { 
//---

   double sum = 0;
   
   for (ulong i=0; i<Betas.Rows(); i++)
      if (i == 0)
         sum += Betas[i][0];
      else
         sum += Betas[i][0] * x[i-1]; 
   
   return (1.0/(1.0 + exp(-sum))); 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CLogisticRegression::Logit(matrix &x)
 {
   vector v_out(x.Rows());
   
   vector betas_v = MatrixExtend::MatrixToVector(Betas), v;
   double sum =0;
   
   for (ulong i=0; i<x.Rows(); i++)
     {
        v = x.Row(i);
        
        sum = 0;
        for (ulong j=0; j<betas_v.Size(); j++)
           sum += betas_v[j] * v[j];
                
        v_out[i] = 1.0/(1.0 + exp(-sum));
     }
     
   return (v_out);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CLogisticRegression::ParameterEstimationGrad(matrix &x, vector &y, uint epochs=1000, double alpha=0.01, double tol=1e-8)
 {     
   matrix XDesignMatrix = MatrixExtend::DesignMatrix(x);
   matrix XT = XDesignMatrix.Transpose();
   
   vector P = {}; matrix PA = {}; 
   
   #ifdef DEBUG_MODE
      Print("classes ",classes);
   #endif 
   
   double prev_cost=0, curr_cost=0;
   
    for (ulong i=0; i<epochs; i++)
     {
       prev_cost = P.Loss(y,LOSS_BCE);;
       
       P = Logit(XDesignMatrix);
       
       PA = MatrixExtend::VectorToMatrix(P - y); 
       
       Betas -= (alpha/(double)x.Rows()) * (XT.MatMul(PA));
       
       curr_cost = P.Loss(y,LOSS_BCE);

       if (MathAbs(prev_cost - curr_cost) < tol)
        {
           printf("Finished convergence prev cost %.5f curr cost %.5f | tol =%.8f | given tolerance %.7f ",prev_cost,curr_cost,MathAbs(curr_cost-prev_cost),tol);
           break;
        }

       //printf("Epoch [%d/%d] Loss %.8f Accuracy %.3f tol %.8f", i+1, epochs, curr_cost, metrics.confusion_matrix(y, P, false),MathAbs(curr_cost-prev_cost));
     }
   istrained = true;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int CLogisticRegression::predict(vector &v)
 { 
   double p1 = Logit(v);
   vector v_out = {p1, 1-p1};
     
  return ((int)classes[v_out.ArgMax()]);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

vector CLogisticRegression::predict_proba(vector &v)
 {
   double p1 = Logit(v);
   vector v_out = {p1, 1-p1};
   
   return(v_out);
 }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CLogisticRegression::predict(matrix &mat)
 {
  ulong size = mat.Rows();
  
  vector v(size);
  
   for (ulong i=0; i<size; i++)
      v[i] = predict(mat.Row(i));
   
   return (v);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+