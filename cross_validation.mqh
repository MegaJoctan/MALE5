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
                       
                       vector LeaveOneOut(double init,double finale,double step);
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

vector CCrossValidation::LeaveOneOut(double init,double finale,double step)
 {
    matrix XMatrix;
    vector yVector;
    
    matrix_utils.XandYSplitMatrices(Matrix,XMatrix,yVector);
 
    matrix train = Matrix; vector test = {};
    
    int size = int(finale/step);
    vector validation_output(ulong(size));
    vector lambda_vector(ulong(size));
    
    vector forecast(n-1); 
    vector actual(n-1);
    
    double lambda = init;
    
     for (int i=0; i<(int)finale/step; i++)
       {
         lambda += step;
         
          for (ulong j=0; j<n-1; j++)
            {               
               train.Copy(Matrix);
               ZeroMemory(test);
               
               test = XMatrix.Row(j);
               actual[j] = yVector[j];
               
               matrix_utils.MatrixRemoveRow(train,j);
               
               //Print("Row ",j," test ",test,"\nTrain\n",train);
               vector coeff = {};
               
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
                    
                         nearest_neighbors = new CKNNNearestNeighbors(Matrix);
                         
                         forecast[j] = nearest_neighbors.KNNAlgorithm(test);
                         
                     break;
                  }
            }
          
          //Print("forecast\n",forecast);
          //Print("actual\n",yVector);
           
          validation_output[i] = forecast.Loss(actual,LOSS_MSE); 
          lambda_vector[i] = lambda;
          
          #ifdef DEBUG_MODE
          
            Print("mse ",validation_output[i]);
            
          #endif           
       }

//---

      #ifdef  DEBUG_MODE
         matrix store_matrix(size,2);
         
         store_matrix.Col(validation_output,0);
         store_matrix.Col(lambda_vector,1); 
         
         string name = EnumToString(selected_model)+"\\LOOCV.csv";
         
         matrix_utils.WriteCsv(name,store_matrix);
      #endif 
      
    return(validation_output);
 }

//+------------------------------------------------------------------+
