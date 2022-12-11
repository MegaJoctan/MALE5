//+------------------------------------------------------------------+
//|                                          CrashClassifyScript.mq5 |
//|                                    Copyright 2022, Omega Joctan. |
//|                           https://www.mql5.com/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, Omega Joctan."
#property link      "https://www.mql5.com/users/omegajoctan"
#property version   "1.00"
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
#include "LogisticRegressionLib.mqh";
CLogisticRegression logreg; //logistic regression class
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---
     double stock_prices[];
     string dates[];
     
     string file_name = "NFLX.csv";
     
     StoreDataToArray(2,stock_prices,file_name);
     StoreDataToArray(1,dates,file_name);
     
     Print("stock prices size = ",ArraySize(stock_prices));
//---


     int crash[];
     if (ArraySize(stock_prices)>1)
      DetectCrash(stock_prices,crash);
     
     for (int i=1; i<ArraySize(stock_prices); i++)
      {
        Print(" DATE ",dates[i]," TREND ",crash[i-1]);
      }
       
     StoreArrayToCSV(crash,"Trend",dates,"DATE","Netflix Trends.csv",",");
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void DetectCrash(double &prices[], int& out_binary[])
 {
     double prev_high = prices[0];
     
     ArrayResize(out_binary,ArraySize(prices)-1); //we reduce the size by one since we ignore the current we predict the previous one
     for (int i=1; i<ArraySize(prices); i++)
        {
           int prev = i-1;
            if (prices[i] >= prev_high)
                prev_high = prices[i]; //grab the highest price 
                
            double percent_crash = ((prev_high - prices[i]) / prev_high) * 100.0; //convert crash to percentage
            //printf("crash percentage %.2f high price %.4f curr price %.4f ", percent_crash,prev_high,prices[i]);  
          
            //based on the definition of a crash; markets has to fall more than 10% percent
            if (percent_crash > 10)
                out_binary[prev] = 0; //downtrend (crash)
            else
                out_binary[prev] = 1; //uptrend (no crash )
        }
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void StoreArrayToCSV(int& Array[], string label, string &stArray[], string label2, string filename, string delimiter)
 {
    FileDelete(filename);   
  
    int handle = FileOpen(filename,FILE_CSV|FILE_READ|FILE_WRITE,delimiter);
    
    if (handle == INVALID_HANDLE)
      Print("Invalid csv handle Err=",GetLastError());

//---

     if (handle>0)
       {  
         FileWrite(handle,label,label2); 
            for (int i=0; i<ArraySize(Array); i++)
              {  
                string str1 = DoubleToString(Array[i],Digits());
                FileWrite(handle,str1,stArray[i+1]); 
              }
       }
       
     FileClose(handle); 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void StoreDataToArray(int from_column_number,double &toArr[],string file_name,string delimiter=",")
  {
    int counter=0;
    int column = 0, rows=0;
    
    int handle = FileOpen(file_name,FILE_CSV|FILE_READ|FILE_WRITE|FILE_ANSI,delimiter,CP_UTF8);
    
    if (handle == INVALID_HANDLE)
      Print("Invalid csv handle Err =",GetLastError());
      
    while (!FileIsEnding(handle))
      {
        string data = FileReadString(handle);
        
        column++;
//---      
        if (column==from_column_number)
           {
          
               if (rows>=1) //Avoid the first column which contains the column's header
                 {    
                     counter++;
                     ArrayResize(toArr,counter); 
                     toArr[counter-1]=(double)data;
                 }   
                  
           }
//---
        if (FileIsLineEnding(handle))
          {                     
            rows++;
            column=0;
          }
      }
    FileClose(handle);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void StoreDataToArray(int from_column_number,string &toArr[],string file_name,string delimiter=",")
  {
    int counter=0;
    int column = 0, rows=0;
    
    int handle = FileOpen(file_name,FILE_CSV|FILE_READ|FILE_WRITE|FILE_ANSI,delimiter,CP_UTF8);
    
    if (handle == INVALID_HANDLE)
      Print("Invalid csv handle Err=",GetLastError());
      
    while (!FileIsEnding(handle))
      {
        string data = FileReadString(handle);
        
        column++;
//---      
        if (column==from_column_number)
           {
               if (rows>=1) //Avoid the first column which contains the column's header
                 {   
                     counter++;
                     ArrayResize(toArr,counter); 
                     toArr[counter-1]=data;
                 }   
                  
           }
//---
        if (FileIsLineEnding(handle))
          {                     
            rows++;
            column=0;
          }
     }
    FileClose(handle);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

