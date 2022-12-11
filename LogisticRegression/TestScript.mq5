//+------------------------------------------------------------------+
//|                                                   TestScript.mq5 |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
#include "LogisticRegressionLib.mqh";
CLogisticRegression *log_reg;
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---
    string file_name =  "Apple Dataset.csv";
    string delimiter = ",";
    
//---
    log_reg = new CLogisticRegression();
    log_reg.Init(file_name,delimiter,2,"3,5,6",0.7);
    double accuracy = 0;
    log_reg.LogisticRegressionMain(accuracy);
    
    printf("Tested model accuracy =%.4f",accuracy);
    
    
    delete log_reg;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

