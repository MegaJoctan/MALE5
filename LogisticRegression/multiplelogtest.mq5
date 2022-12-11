//+------------------------------------------------------------------+
//|                                              multiplelogtest.mq5 |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
#include "MultipledynamicLogisticRegression.mqh";
#include  "C:\Users\Omega Joctan\AppData\Roaming\MetaQuotes\Terminal\892B47EBC091D6EF95E3961284A76097\MQL5\Experts\DataScience\LinearRegression\MultipleMatLinearReg.mqh";
#include  "LogisticRegressionLib.mqh";

CMultipleMatLinearReg lr;
CMultipleLogisticRegression mlog;
CLogisticRegression logreg;
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---
      //mlog.FixMissingValues("5");
      //mlog.LabelEncoder("4","female,male");
      
      //mlog.MLRInit("titanic.csv",",",2,"6",false);
      string filename = "Apple Dataset.csv";
      string delimiter = ",";
      
      double A[], B[], C[], D[], E[], y[];
      logreg.GetDatatoArray(2,y,filename,delimiter);
      logreg.GetDatatoArray(3,A,filename,delimiter);
      logreg.GetDatatoArray(4,B,filename,delimiter);
      logreg.GetDatatoArray(5,C,filename,delimiter);
      logreg.GetDatatoArray(6,D,filename,delimiter);
      logreg.GetDatatoArray(7,E,filename,delimiter);
      
      ArrayPrint(A);
      
      Print(" corr coeff A vs y ",lr.corrcoef(A,y));
      Print(" corr coeff B vs y ",lr.corrcoef(B,y));
      Print(" corr coeff C vs y ",lr.corrcoef(C,y));
      Print(" corr coeff D vs y ",lr.corrcoef(D,y));
      Print(" corr coeff E vs y ",lr.corrcoef(E,y));
      
      double accuracy =0;
      mlog.MultipleLogisticRegression(accuracy);   
  }
//+------------------------------------------------------------------+
