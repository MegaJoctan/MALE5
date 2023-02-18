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
#define LOG_REG //Linear regression model is called for Logistic regression purposes

#include <MALE5\Linear Regression\Linear Regression.mqh>

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

class CLogisticRegression: public CLinearRegression
  {     
     public:
         CLogisticRegression(matrix &x_matrix, vector &y_vector);
         CLogisticRegression(matrix<double> &x_matrix,vector &y_vector, double Lr, uint iters = 1000);
         
        ~CLogisticRegression(void);
                         
                         
         vector LogregModelPred(matrix &x_matrix);
         double LogregModelPred(vector &x_vector);

  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CLogisticRegression::CLogisticRegression(matrix &x_matrix, vector &y_vector):CLinearRegression(x_matrix,y_vector,NORM_STANDARDIZATION)
 { 
 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CLogisticRegression::CLogisticRegression(matrix<double> &x_matrix,vector &y_vector, double Lr, uint iters = 1000):CLinearRegression(x_matrix,y_vector,NORM_STANDARDIZATION,Lr,iters)
 {
   
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

double CLogisticRegression::LogregModelPred(vector &x_vector)
 {   
   return (round(1.0/(1.0 + exp(-LRModelPred(x_vector)))));
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CLogisticRegression::LogregModelPred(matrix &x_matrix)
 {  
    vector v(x_matrix.Rows());
    
      for (ulong i=0; i<v.Size(); i++)
         v[i] = LogregModelPred(x_matrix.Row(i));
    
    return (v);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
