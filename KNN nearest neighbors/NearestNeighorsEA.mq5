//+------------------------------------------------------------------+
//|                                            NearestNeighorsEA.mq5 |
//|                                    Copyright 2022, Omega Joctan. |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, Omega Joctan."
#property link      "https://www.mql5.com/en/users/omegajoctan"
#property version   "1.00"
//+------------------------------------------------------------------+
#include "KNN_nearest_neighbors.mqh";
CKNNNearestNeighbors *nearest_neigbors;

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>

CTrade m_trade;
CPositionInfo m_postion;

//---

input uint k_val = 2;
input uint bars = 100;
input ENUM_TIMEFRAMES timeframe = PERIOD_D1;
input group "ATR";
input int   period = 14;
input group "VOLUME";
input ENUM_APPLIED_VOLUME applied_vol = VOLUME_TICK;
input group "TRADING";
input int   MAGIC = 3112022;
input int   Slippage = 100;
uint _k, _bars;

int atr_handle, volume_handle;
vector atr_buffer, volume_buffer;
matrix Matrix(bars, 3); //2 independent 1 target
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
    _k = k_val; _bars = bars; 
    
    m_trade.SetExpertMagicNumber(MAGIC);
    m_trade.SetTypeFillingBySymbol(Symbol());
    m_trade.SetMarginMode();
    m_trade.SetDeviationInPoints(Slippage);
    
//--- Preparing the dataset 
   
    atr_handle = iATR(Symbol(),timeframe,period);
    volume_handle = iVolumes(Symbol(),timeframe,applied_vol);
    
   
   //nearest_neigbors.TrainTest();
     
   if (MQLInfoInteger(MQL_TESTER) || MQLInfoInteger(MQL_OPTIMIZATION)) 
         isdebug = false;          
   else
     {
         gather_data();      
         nearest_neigbors = new CKNNNearestNeighbors(Matrix,_k); 
     }
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
    
    delete(nearest_neigbors);
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+

int signal = -1;

void OnTick()
  {
    
    vector x_vars(2);
    double atr_val[], volume_val[];
    
    CopyBuffer(atr_handle,0,0,1,atr_val);
    CopyBuffer(volume_handle,0,0,1,volume_val);
    
    x_vars[0] = atr_val[0]; 
    x_vars[1] = volume_val[0];
    
//---
    
    double volume = SymbolInfoDouble(Symbol(),SYMBOL_VOLUME_MIN);
    
    MqlTick ticks;
    SymbolInfoTick(Symbol(),ticks);
    
    double ask = ticks.ask, bid = ticks.bid;
    
    
      if (isNewBar() == true) //we are on the new candle
         { 
           if (MQLInfoInteger(MQL_TESTER) || MQLInfoInteger(MQL_OPTIMIZATION)) 
             {
               gather_data();      
               nearest_neigbors = new CKNNNearestNeighbors(Matrix,_k); 
               signal = nearest_neigbors.KNNAlgorithm(x_vars);
               
               delete(nearest_neigbors);
             }          

//---

            if (signal == 1)
              {
                 if (!CheckPosionType(POSITION_TYPE_BUY))
                  {
                    m_trade.Buy(volume,Symbol(),ask,0,0);
                    if (ClosePosType(POSITION_TYPE_SELL))
                      printf("Failed to close %s Err = %d",EnumToString(POSITION_TYPE_SELL),GetLastError());
                  }
              }
            else if (signal == 0)
              {
                if (!CheckPosionType(POSITION_TYPE_SELL))
                  {
                    m_trade.Sell(volume,Symbol(),bid,0,0);
                    if (ClosePosType(POSITION_TYPE_BUY))
                      printf("Failed to close %s Err = %d",EnumToString(POSITION_TYPE_BUY),GetLastError());
                  }
              }
         }
     
  }

//+------------------------------------------------------------------+

static int BARS;

bool isNewBar()
   {
      if(BARS!=Bars(Symbol(),timeframe))
        {
            BARS=Bars(Symbol(),timeframe);
            return(true);
        }
      return(false);
   }

//+------------------------------------------------------------------+

void CopyBuffer(int indicator_handle,int buffer_num,int start_pos,int count,vector &buffer_vec)
 {
   double buffer_Arr[];
   ArraySetAsSeries(buffer_Arr,true);
    
    CopyBuffer(indicator_handle,buffer_num,start_pos,count,buffer_Arr);
    
    ulong size = (ulong)ArraySize(buffer_Arr);
    buffer_vec.Resize(size);
    
    for (ulong i=0; i<size; i++)
      buffer_vec[i] = buffer_Arr[i];      
 }

//+------------------------------------------------------------------+

bool ClosePosType(ENUM_POSITION_TYPE type)
 {
    for (int i=PositionsTotal()-1; i>=0; i++)
      if (m_postion.SelectByIndex(i))
         if (m_postion.Magic() == MAGIC && m_postion.Symbol()==Symbol() && m_postion.PositionType() == type)
            {
               m_trade.PositionClose(m_postion.Ticket());
               return(true);
            }
     return(false);
 }

//+------------------------------------------------------------------+

bool CheckPosionType(ENUM_POSITION_TYPE type)
 {
    for (int i=PositionsTotal()-1; i>=0; i++)
      if (m_postion.SelectByIndex(i))
         if (m_postion.Magic() == MAGIC && m_postion.Symbol()==Symbol() && m_postion.PositionType() == type)
               return(true);


     return(false);
 }
 
//+------------------------------------------------------------------+
void gather_data()
 {
       CopyBuffer(atr_handle,0,1,_bars,atr_buffer);
       CopyBuffer(volume_handle,0,1,_bars,volume_buffer); 
       
       Matrix.Col(atr_buffer,0); //Independent var 1
       Matrix.Col(volume_buffer,1); //Independent var 2
       
   //--- Target variables
   
       vector Target_vector(_bars);
       
       MqlRates rates[];
       ArraySetAsSeries(rates,true);
       CopyRates(Symbol(),PERIOD_D1,1,_bars,rates);
       
       for (ulong i=0; i<Target_vector.Size(); i++) //putting the labels
        {
          if (rates[i].close > rates[i].open)
             Target_vector[i] = 1; //bullish
          else
             Target_vector[i] = 0;
        }
   
       Matrix.Col(Target_vector,2);
           
   //---
       
       if (isdebug) 
         Print("ATR,Volumes,Class Matrix\n",Matrix);
       
        nearest_neigbors = new CKNNNearestNeighbors(Matrix,_k); 
        //nearest_neigbors.TrainTest();             
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
