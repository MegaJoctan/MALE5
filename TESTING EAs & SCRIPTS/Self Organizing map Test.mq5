//+------------------------------------------------------------------+
//|                                          Self Organizing map.mq5 |
//|                                     Copyright 2023, Omega Joctan |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, Omega Joctan"
#property link      "https://www.mql5.com/en/users/omegajoctan"
#property version   "1.00"
#property strict
#property description "This is a Test EA file for testing Naive Bayes.mqh file for the MALE5 repository located at /"

//#define DEBUG_MODE

#include <MALE5\Neural Networks\kohonen maps.mqh>
#include <MALE5\matrix_utils.mqh>

CMatrixutils matrix_utils;
CKohonenMaps *maps;


input int bars = 100;   //Train Bars
input bool save = true; //save clusters
input uint clst = 2;    //clusters
input double Lr = 0.01; //alpha
input uint  total_epochs = 100; //epochs
input norm_technique NORMALIZATION = NORM_MIN_MAX_SCALER;

int handles[5];
int period[5] = {10,20,30,50,100};
matrix Matrix(bars,5);

bool trained = false;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
/*
   matrix Matrix = {
                     {1.2, 2.3},
                     {0.7, 1.8},
                     {3.6, 4.8},
                     {2.8, 3.9},
                     {5.2, 6.7},
                     {4.8, 5.6}
                   };
*/
 
   vector v;
   
   for (int i=0; i<5; i++)
      {
         handles[i] = iMA(Symbol(),PERIOD_CURRENT,period[i],0,MODE_LWMA,PRICE_CLOSE);
         matrix_utils.CopyBufferVector(handles[i],0,0,bars, v);
         
         Matrix.Col(v, i);
      }

//---

   if (!MQLInfoInteger(MQL_TESTER)) 
      maps = new CKohonenMaps(Matrix,save,clst,Lr,total_epochs,NORMALIZATION); //Training
     
/*
   matrix new_data = {
         {0.5,1.5},
         {5.5, 6.0}
      };
*/ 
   
   
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
   delete(maps);
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
   TrainOnStrategyTester();
    
   vector new_data(5);
   vector v;
   
   for (int i=0; i<5; i++)
      {
         matrix_utils.CopyBufferVector(handles[i],0,0,1, v);
         
         new_data[i] = v[0];
      }
     
     matrix_utils.NormalizeVector(new_data,Digits());
     Comment("Indicator Readings ",new_data," predicted cluster ",maps.KOMPredCluster(new_data)); 
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void TrainOnStrategyTester()
 {
   if (MQLInfoInteger(MQL_TESTER))
    if (!trained)
     {
      vector v;
      
      Print("Tester train ");
      
      for (int i=0; i<5; i++)
         {
            matrix_utils.CopyBufferVector(handles[i],0,0,bars, v);
            
            Matrix.Col(v, i);
         }
   
      
      maps = new CKohonenMaps(Matrix,save,clst,Lr,total_epochs,NORMALIZATION); //Training
     }   
   trained = true;  
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

