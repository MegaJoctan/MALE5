//+------------------------------------------------------------------+
//|                                             Naive Bayes Test.mq5 |
//|                                    Copyright 2022, Fxalgebra.com |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, Fxalgebra.com"
#property link      "https://www.mql5.com/en/users/omegajoctan"
#property version   "1.00"
#property description "This is a Test EA file for testing Naive Bayes.mqh file for the MALE5 repository located at Naive Bayes/"

//#define  DEBUG_MODE
#define  MAGIC_NUMBER 144020230348

#include <MALE5\Naive Bayes\Naive Bayes.mqh>
#include <MALE5\matrix_utils.mqh>
#include <MALE5\metrics.mqh>
#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh> 

CTrade m_trade;
CPositionInfo m_position;

CMatrixutils matrix_utils;
CMetrics metrics;

//CNaiveBayes  *naive_bayes;
CGaussianNaiveBayes *gaussian_naive;

input uint rand_state = 42; //Random State
input int  TrainBars = 1000;
input ENUM_TIMEFRAMES TF = PERIOD_CURRENT;

input group "BEARS"
input int   bears_period = 13;

input group "BULLS"
input int   bulls_period = 13;

input group "RSI"
input int   rsi_period = 13;
input ENUM_APPLIED_PRICE rsi_price = PRICE_CLOSE; 

input group "VOLUMEs" 
input string volume_tick="Has been applied";

input group "MFI"
input int mfi_period = 14;

input group "TRADING INFO"
input int  slippage =100;

matrix Matrix(TrainBars, 6);
int handles[5];

double buffer[];
static bool train_state = false;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {   
//--- Preparing Data

   handles[0] = iBearsPower(Symbol(),TF, bears_period);
   handles[1] = iBullsPower(Symbol(),TF, bulls_period);
   handles[2] = iRSI(Symbol(),TF,rsi_period, rsi_price);
   handles[3] = iVolumes(Symbol(),TF,VOLUME_TICK);
   handles[4] = iMFI(Symbol(),TF,mfi_period,VOLUME_TICK );

//---

   m_trade.SetExpertMagicNumber(MAGIC_NUMBER);
   m_trade.SetTypeFillingBySymbol(Symbol());
   m_trade.SetMarginMode();
   m_trade.SetDeviationInPoints(slippage);

   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void TrainTest()
 {
//--- 
    
    
   vector col_v;
   
   for (ulong i=0; i<5; i++) //Independent vars
     { 
       CopyBuffer(handles[i],0,0,TrainBars, buffer);
       col_v = matrix_utils.ArrayToVector(buffer);
       
       Matrix.Col(col_v, i);
     }
     
//-- Target var

   vector open, close;
   col_v.Resize(TrainBars);

   close.CopyRates(Symbol(),TF, COPY_RATES_CLOSE,0,TrainBars);
   open.CopyRates(Symbol(),TF, COPY_RATES_OPEN,0,TrainBars);

   for (int i=0; i<TrainBars; i++)
      {
         if (close[i] > open[i]) //price went up
            col_v[i] = 1;
         else 
            col_v[i] = 0;
      }
   
   Matrix.Col(col_v, 5);
   
//---

   //matrix_utils.PrintShort(Matrix);

//--- Visualize the data 
  /*
   matrix vars_matrix = Matrix;
   
   string header[5] = {"Bears","Bulls","Rsi","Volumes","MFI"};
    
   matrix_utils.RemoveCol(vars_matrix, 5); //remove target variable
   
   if (!MQLInfoInteger(MQL_TESTER) && !MQLInfoInteger(MQL_OPTIMIZATION))
      matrix_utils.WriteCsv("NAIVE BAYES\\vars.csv",vars_matrix, header, 8);
     
   matrix_utils.PrintShort(Matrix);
   
   ArrayPrint(header);
   Print(vars_matrix.CorrCoef(false));
  */
    
//---
     
     Print("\n---> Training the Model\n");
     
     matrix x_train, x_test;
     vector y_train, y_test;
     
     matrix_utils.TrainTestSplitMatrices(Matrix,x_train,y_train,x_test,y_test,0.7,rand_state);
     
//--- Train
     
     gaussian_naive = new CGaussianNaiveBayes(x_train,y_train);  
     
     vector train_pred = gaussian_naive.GaussianNaiveBayes(x_train);
    
     vector c= gaussian_naive.classes;
    
     metrics.confusion_matrix(y_train,train_pred,c);
    
    
//--- Test
   
     Print("\n---> Testing the model\n");
     
     vector test_pred = gaussian_naive.GaussianNaiveBayes(x_test); //giving the model test data
     
     metrics.confusion_matrix(y_test,test_pred,c);
//---
 }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
   
   //delete (naive_bayes);
   delete (gaussian_naive);
   
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   if (!train_state)
      TrainTest();
  
    train_state = true;   
     
//---

   vector v_inputs(5); //5 independent variables
   double buff[1];  //current indicator value
    
   for (ulong i=0; i<5; i++) //Independent vars
     { 
       CopyBuffer(handles[i],0,0,1, buff);
       
       v_inputs[i] = buff[0];
     }

//---

   MqlTick ticks;
   SymbolInfoTick(Symbol(), ticks);

   int signal = -1;
   double min_volume = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_MIN);
   
   if (isNewBar())
     { 
       signal = gaussian_naive.GaussianNaiveBayes(v_inputs);
       
       Comment("SIGNAL ",signal);
       
       CloseAll();
        
        if (signal == 1)
          { 
            if (!PosExist())
              m_trade.Buy(min_volume, Symbol(), ticks.ask, 0 , 0,"Naive Buy");
          }
        else if (signal == 0)
          {
            if (!PosExist())
              m_trade.Sell(min_volume, Symbol(), ticks.bid, 0 , 0,"Naive Sell");
          }
     } 
   
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

static int BARS;

bool isNewBar()
   {
      if(BARS!=Bars(_Symbol,_Period))
        {
            BARS=Bars(_Symbol,_Period);
            return(true);
        }
      return(false);
   }
   
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

bool PosExist()
 {
   for (int i=PositionsTotal()-1; i>=0; i--)
      if (m_position.SelectByIndex(i))
         if (m_position.Magic() == MAGIC_NUMBER && m_position.Symbol()==Symbol())
            return(true);
            
     return (false);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CloseAll()
 {
   for (int i=PositionsTotal()-1; i>=0; i--)
      if (m_position.SelectByIndex(i))
         if (m_position.Magic() == MAGIC_NUMBER && m_position.Symbol()==Symbol())
            if (!m_trade.PositionClose(m_position.Ticket(),slippage))
               Print("Failed to close a position Err=",GetLastError());
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
