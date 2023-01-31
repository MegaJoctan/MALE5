//+------------------------------------------------------------------+
//|                                             cross_validation.mqh |
//|                                    Copyright 2022, Fxalgebra.com |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, Fxalgebra.com"
#property link      "https://www.mql5.com/en/users/omegajoctan"
//+------------------------------------------------------------------+
//| defines                                                          |
//+------------------------------------------------------------------+

#include <MALE5\matrix_utils.mqh>
#include <MALE5\KNN nearest neighbors\KNN_nearest_neighbors.mqh>
#include <MALE5\Ridge & Lasso Regression\Ridge Regression.mqh>
#include <MALE5\metrics.mqh>

enum models //Models that need cross validation
  {
     KNN_NEAREST_NEIGHBORS,
     RIDGE_REGRESSION
  } selected_model;

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

class CCrossValidation
  {
   private:
      CMatrixutils      matrix_utils;
      CRidgeregression  *ridge_regression;
      CLinearRegression *Linear_reg;
      CKNNNearestNeighbors *nearest_neighbors; 
      
      matrix            Matrix;
      ulong             n;
      
   public:
                        CCrossValidation(matrix& matrix_, models MODEL);
                       ~CCrossValidation(void);
                       
                       double LeaveOneOut(double init, double step, double finale);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CCrossValidation::CCrossValidation(matrix& matrix_, models MODEL)
 { 
   selected_model = MODEL;
   Matrix = matrix_;
   n = Matrix.Rows();
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CCrossValidation::~CCrossValidation(void)
 {
 
 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

double CCrossValidation::LeaveOneOut(double init, double step, double finale)
 {
    matrix XMatrix;
    vector yVector;
    
    matrix_utils.XandYSplitMatrices(Matrix,XMatrix,yVector);
 
    matrix train = Matrix; vector test = {};
    
    int size = int(finale/step);
    vector validation_output(ulong(size));
    vector lambda_vector(ulong(size));
    
    vector forecast(n); 
    vector actual = yVector;
    
    double lambda = init;
    
     for (int i=0; i<size; i++)
       {
         lambda += step;
         
          for (ulong j=0; j<n; j++)
            {               
               train.Copy(Matrix);
               ZeroMemory(test);
               
               test = XMatrix.Row(j);
               
               matrix_utils.RemoveRow(train,j);
               
               vector coeff = {};
               double acc =0;
               
                switch(selected_model)
                  {
                   case  RIDGE_REGRESSION:

                        ridge_regression = new CRidgeregression(train);
                        coeff = ridge_regression.L2Norm(lambda); //ridge regression
                        
                        Linear_reg = new CLinearRegression(train,coeff);   

                        forecast[j] =  Linear_reg.LRModelPred(test);  
                        
                        //---
                        
                        delete (Linear_reg); 
                        delete (ridge_regression);
                        
                     break;
                     
                    case KNN_NEAREST_NEIGHBORS:
                    
                         nearest_neighbors = new CKNNNearestNeighbors(train);
                         
                         forecast[j] = nearest_neighbors.KNNAlgorithm(test);
                         
                     break;
                  }
            }
          
          //Print("---->\nforecast\n",forecast);
          //Print("actual\n",yVector);
          
          validation_output[i] = forecast.Loss(actual,LOSS_MSE)/double(n); 
          
          lambda_vector[i] = lambda;
          
          #ifdef DEBUG_MODE
             printf("%.5f LOOCV mse %.5f",lambda_vector[i],validation_output[i]);
          #endif           
       }

//---

      #ifdef  DEBUG_MODE
         matrix store_matrix(size,2);
         
         store_matrix.Col(validation_output,0);
         store_matrix.Col(lambda_vector,1); 
         
         string name = EnumToString(selected_model)+"\\LOOCV.csv";
         
         string header[2] = {"Validation output","lambda"};
         matrix_utils.WriteCsv(name,store_matrix,header,10);
      #endif 
      
    return(lambda_vector[validation_output.ArgMin()]);
 }

//+------------------------------------------------------------------+
