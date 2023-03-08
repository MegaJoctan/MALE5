//+------------------------------------------------------------------+
//|                                                     Reg test.mq5 |
//|                                    Copyright 2022, Fxalgebra.com |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, Fxalgebra.com"
#property link      "https://www.mql5.com/en/users/omegajoctan"
#property version   "1.00"
#property description "This is a Test EA file for testing the Linear Regression.mqh file for the MALE5 repository located at Linear Regression/"
//#define  DEBUG_MODE //uncomment this to debug my libraries

#include <MALE5\Linear Regression\Linear Regression.mqh>
#include <MALE5\matrix_utils.mqh>
#include <MALE5\metrics.mqh>

CMetrics metrics;
CLinearRegression *Lr;
CMatrixutils matrix_utils;

input string symbol_x = "Apple_Inc_(AAPL.O)"; 
input string symbol_y = "Microsoft_Corp_(MSFT.O)";

input ENUM_COPY_RATES copy_rates = COPY_RATES_CLOSE;
input int n_samples = 100; 

   
matrix XMATRIX(n_samples, 1); vector YVECTOR;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   
   vector x_vector; //This vector is going to help us store the values temporarily
   
   x_vector.CopyRates(symbol_x,PERIOD_CURRENT,copy_rates,0,n_samples); //start at bar located at  0 index or the current bar to n bars in the past
   
   //Print("x vector ",x_vector);
   XMATRIX.Col(x_vector, 0); //put this x vector to the first column of the matrix, 0 means 1 column 1 second column ....
   
   YVECTOR.CopyRates(symbol_y, PERIOD_CURRENT,copy_rates,0,n_samples); //start at bar located at  0 index or the current bar to n bars in the past
   
   //Print("y vector ",YVECTOR);
   //Print("x matrix\n",XMATRIX);
   
   Lr = new CLinearRegression(XMATRIX, YVECTOR, NORM_NONE);
   
   vector y_pred = Lr.LRModelPred(XMATRIX);
   
   Print("Trained Accuracy ",metrics.r_squared(YVECTOR, y_pred));
   
   //Print("Actuals ",YVECTOR);
   //Print("Pred ",y_pred);
   
   matrix save(n_samples,2); //creating a matrix that we are going to save to csv
   
   save.Col(YVECTOR, 0); //Store the actual values in the first column of this matrix
   save.Col(y_pred, 1);  //store the predicted values in the second column of this matrix
   
   string header[2] = {"Actual","Pred"};
   matrix_utils.WriteCsv("LINEAR REGRESSION\\LR performance.csv",save,header); //This file is at the directory MQL5/Files/LINEAR REGRESSION
   
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
      delete (Lr);
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
   
   vector x;
   
   x.CopyRates(symbol_x,PERIOD_CURRENT,copy_rates, 0,1); //Copy the current value of the symbol x into the x vector
   
   double pred = Lr.LRModelPred(x);
   
   Print(" x = ",x," LR Y Predicted ",pred);
  }
//+------------------------------------------------------------------+
