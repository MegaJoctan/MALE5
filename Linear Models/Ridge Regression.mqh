//+------------------------------------------------------------------+
//|                                             Ridge Regression.mqh |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, Fxalgebra.com"
#property link      "https://www.mql5.com/en/users/omegajoctan"

//+------------------------------------------------------------------+
 
#include <MALE5\preprocessing.mqh>
#include <MALE5\matrix_utils.mqh>
#include <MALE5\Linear Regression\Linear Regression.mqh>

//+------------------------------------------------------------------+

class CRidgeregression
  {
   private:
   
   CPreprocessing pre_processing; 
   CMatrixutils matrix_utils;
   CLinearRegression *Linear_reg;
  
   protected: 
                        matrix XMatrix; //matrix of independent variables
                        matrix YMatrix;
                        vector yVector; // Vector of target variables
                        matrix Id_matrix; //Identity matrix
                        
                        matrix Betas;
                        ulong  n; //No of samples
                        ulong  k; //No of regressors 
                        
   public:
                        CRidgeregression(matrix &_matrix);
                       ~CRidgeregression(void);
                         
                       double RSS;
                       double Lr_accuracy;
                       
                       vector L2Norm(double lambda); //Ridge regression
                        
  };
  
//+------------------------------------------------------------------+

CRidgeregression::CRidgeregression(matrix &_matrix)
 {
    n = _matrix.Rows();
    k = _matrix.Cols();
    
    pre_processing.Standardization(_matrix); 
    
    matrix_utils.XandYSplitMatrices(_matrix,XMatrix,yVector);
    
    YMatrix = matrix_utils.VectorToMatrix(yVector);
    
//---

    Id_matrix.Resize(k,k);
    
    Id_matrix.Identity();

 }
 
//+------------------------------------------------------------------+

CRidgeregression::~CRidgeregression(void)
 {
   ZeroMemory(XMatrix);
   ZeroMemory(yVector);
   ZeroMemory(yVector);
   ZeroMemory(Id_matrix); 
 }
 
//+------------------------------------------------------------------+

vector CRidgeregression::L2Norm(double lambda)
 {    
   matrix design = matrix_utils.DesignMatrix(XMatrix);
   
   matrix XT = design.Transpose();
   
   matrix XTX = XT.MatMul(design);
   
   matrix lamdaxI = lambda * Id_matrix;
   
   matrix sum_matrix = XTX + lamdaxI;
   
   matrix Inverse_sum = sum_matrix.Inv();
   
   matrix XTy = XT.MatMul(YMatrix);
   
   Betas = Inverse_sum.MatMul(XTy);
 
   #ifdef DEBUG_MODE
      //Print("Betas\n",Betas);
   #endif 
   
  return(matrix_utils.MatrixToVector(Betas));
 }
 
//+------------------------------------------------------------------+
