//+------------------------------------------------------------------+
//|                                                 matrix_utils.mqh |
//|                                  Copyright 2022, Omega Joctan  . |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, MetaQuotes Ltd."
#property link      "https://www.mql5.com/en/users/omegajoctan"
//+------------------------------------------------------------------+
//| defines                                                          |
//+------------------------------------------------------------------+
#include <MALE5\preprocessing.mqh>
#include <MALE5\matrix_utils.mqh>
#include <MALE5\metrics.mqh>

#define CLEAR_MEM(mat) mat.Resize(1,0);  ZeroMemory(mat);
 
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CLogisticRegression
  {
CPreprocessing      *normalize_x;
CMatrixutils         matrix_utils;
CMetrics             metrics;

private:
                    norm_technique M_NORM;
                    
                    bool isTrain;
                    matrix XMatrix; 
                    vector YVector;
                    
                    ulong m_rows, m_cols;
                    
                    double Logit(vector &x);
                    vector Logit(matrix &x); //Logitstic loss function for training 

                    void ParameterEstimationGrad(uint epochs=1000, double alpha=0.01, double tol=1e-8);
                    
public:
                     CLogisticRegression(matrix &x_matrix, vector &y_vector, norm_technique NORM_METHOD, double alpha=0.01, uint epochs=1000, double tol=1e-8);
                    ~CLogisticRegression(void);
                    
                    matrix Betas;
                    vector classes;
                    
                    vector Odds(vector &proba);
                    vector lnOdss(vector &odds);
                    
                    int    LogitPred(vector &v);
                    vector LogitPredProba(vector &v);
                    vector LogitPred(matrix &mat);
                    matrix LogitPredProba(matrix &mat);
                    
  };
//+------------------------------------------------------------------+
//| This is where the logistic model gets |
//+------------------------------------------------------------------+
CLogisticRegression::CLogisticRegression(matrix &x_matrix, vector &y_vector, norm_technique NORM_METHOD, double alpha=0.01, uint epochs=1000, double tol=1e-8)
 {
   m_rows = x_matrix.Rows();
   m_cols = x_matrix.Cols();

   Betas.Resize(m_cols+1, 1);
   Betas.Fill(0);
   
//---
   
   XMatrix = x_matrix;
   YVector = y_vector; 
   
   classes = matrix_utils.Classes(y_vector);
   
   M_NORM = NORM_METHOD;
    
   normalize_x = new CPreprocessing(XMatrix, M_NORM);

//---

   isTrain = true; //we are on isTrain 
    
   ParameterEstimationGrad(epochs, alpha, tol);
   
//---
   
   
   #ifdef DEBUG_MODE
      Print("Betas\n",Betas);
   #endif 
   
   isTrain  = false;
   
   CLEAR_MEM(XMatrix);
   CLEAR_MEM(YVector);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CLogisticRegression::~CLogisticRegression(void)
 {
   CLEAR_MEM(Betas);
   
   if (CheckPointer(normalize_x) != POINTER_INVALID)  
      delete (normalize_x);
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
  vector temp_x = x;
  
   if (!isTrain && M_NORM != NORM_NONE) 
         normalize_x.Normalization(temp_x);
 
//---

   double sum = 0;
   
   for (ulong i=0; i<Betas.Rows(); i++)
      if (i == 0)
         sum += Betas[i][0];
      else
         sum += Betas[i][0] * temp_x[i-1]; 
   
   return (1.0/(1.0 + exp(-sum))); 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CLogisticRegression::Logit(matrix &x)
 {
   vector v_out(x.Rows());
   
   vector betas_v = matrix_utils.MatrixToVector(Betas), v;
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
void CLogisticRegression::ParameterEstimationGrad(uint epochs=1000, double alpha=0.01, double tol=1e-8)
 {     
   matrix XDesignMatrix = matrix_utils.DesignMatrix(XMatrix);
   matrix XT = XDesignMatrix.Transpose();
   
   vector P = {}; matrix PA = {}; 
   
   #ifdef DEBUG_MODE
      Print("classes ",classes);
   #endif 
   
   double prev_cost=0, curr_cost=0;
   
    for (ulong i=0; i<epochs; i++)
     {
       prev_cost = P.Loss(YVector,LOSS_BCE);;
       
       P = Logit(XDesignMatrix);
       
       PA = matrix_utils.VectorToMatrix(P - YVector); 
       
       Betas -= (alpha/(double)m_rows) * (XT.MatMul(PA));
       
       curr_cost = P.Loss(YVector,LOSS_BCE);

       if (MathAbs(prev_cost - curr_cost) < tol)
        {
           printf("Finished convergence prev cost %.5f curr cost %.5f | tol =%.8f | given tolerance %.7f ",prev_cost,curr_cost,MathAbs(curr_cost-prev_cost),tol);
           break;
        }

       printf("Epoch [%d/%d] Loss %.8f Accuracy %.3f tol %.8f", i+1, epochs, curr_cost, metrics.confusion_matrix(YVector, round(P), classes,false),MathAbs(curr_cost-prev_cost));
     }
     
//--- Clear the training memory

   CLEAR_MEM(XDesignMatrix);
   CLEAR_MEM(XT);
   CLEAR_MEM(P);
   CLEAR_MEM(PA);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int CLogisticRegression::LogitPred(vector &v)
 { 
   double p1 = Logit(v);
   vector v_out = {p1, 1-p1};
     
  return ((int)classes[v_out.ArgMax()]);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

vector CLogisticRegression::LogitPredProba(vector &v)
 {
   double p1 = Logit(v);
   vector v_out = {p1, 1-p1};
   
   return(v_out);
 }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CLogisticRegression::LogitPred(matrix &mat)
 {
  ulong size = mat.Rows();
  
  vector v(size);
  
   for (ulong i=0; i<size; i++)
      v[i] = LogitPred(mat.Row(i));
   
   return (v);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
matrix CLogisticRegression::LogitPredProba(matrix &mat)
 {
   ulong rows = mat.Rows(), cols = mat.Cols();
   
   matrix mat_out(rows, 2);
   vector v;
   
   for (ulong i=0; i<rows; i++)
    { 
      v = LogitPredProba(mat.Row(i)); 
      mat_out.Row(v, i);
    } 
    
   return (mat_out); 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
