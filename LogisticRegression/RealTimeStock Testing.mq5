//+------------------------------------------------------------------+
//|                                        RealTimeStock Testing.mq5 |
//|                                    Copyright 2022, Omega Joctan. |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, Omega Joctan."
#property link      "https://www.mql5.com/en/users/omegajoctan"
#property version   "1.00"
#property tester_file "Predicted Apple Dataset.csv"
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
#include <Trade\Trade.mqh> 
#include <Trade\PositionInfo.mqh>
//--- 
CTrade                m_trade; 
CPositionInfo         m_position;
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
input   double      Lots = 0.01;

//---
string   file_name = "Predicted Apple Dataset.csv"; //csv file name
string   delimiter = ",";

int      Trend[];
string   dates[];
datetime date_datetime[];
//---
#define  PRINT_VAR(v) ""+#v+""
#define  MAGIC_NUMBER   30052022
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  { 
//---
      GetColumnDatatoArray(1,Trend);
      GetColumnDatatoArray(2,dates);
      
      if (ArraySize(Trend) !=  ArraySize(dates))
        {
           printf("Unbalanced size between %s Array and %s Array",PRINT_VAR(Trend),PRINT_VAR(dates));
           return(INIT_FAILED);
        }
      
      ConvertTimeToStandard();
      
      if (MQLInfoInteger(MQL_TESTER))
        Print("Testing DataSet Begins at ",date_datetime[0]);
//---
      
      m_trade.SetExpertMagicNumber(MAGIC_NUMBER);
      m_trade.SetTypeFillingBySymbol(Symbol());
      m_trade.SetMarginMode();
 
      return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
    if (reason == REASON_PARAMETERS)
       {
         GetColumnDatatoArray(1,Trend);
         GetColumnDatatoArray(2,dates);
         ConvertTimeToStandard();
       }
    else if (reason == REASON_RECOMPILE)
       {
         GetColumnDatatoArray(1,Trend);
         GetColumnDatatoArray(2,dates);
         ConvertTimeToStandard();
       }
   else 
      Comment("");
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//--- 

    datetime today[1];
    int trend_signal = -1; //1 is buy signal 0 is sell signal
    
    CopyTime(Symbol(),PERIOD_D1,0,1,today);
    
    if (isNewBar())
     for (int i=0; i<ArraySize(date_datetime); i++)
      {
          if (today[0] == date_datetime[i]) //train in that specific day only
              {
                 
                  if ((int)Trend[i] == 1)
                    trend_signal = 1;
                  else 
                     trend_signal = 0; 
                     
                  // close all the existing positions since we are coming up with new data signals   
                  ClosePosByType(POSITION_TYPE_BUY);
                  ClosePosByType(POSITION_TYPE_SELL);
                  break;
              }
          
          if (MQLInfoInteger(MQL_TESTER) && today[0] > date_datetime[ArrayMaximum(date_datetime)])
             {
                 Print("we've run out of the testing data, Tester will be cancelled");
                 ExpertRemove();
             }
     } 
     
//--- Time to trade

      MqlTick tick;
      SymbolInfoTick(Symbol(),tick);
      double ask = tick.ask , bid = tick.bid;

//---

      if (trend_signal == 1 && PositionCounter(POSITION_TYPE_BUY)<1)
        {
           m_trade.Buy(Lots,Symbol(),ask,0,0," Buy trade ");
           ClosePosByType(POSITION_TYPE_SELL); //if the model predicts a bullish market close all sell trades if available
        }
        
      if (trend_signal == 0 && PositionCounter(POSITION_TYPE_SELL)<1)
        {
            m_trade.Sell(Lots,Symbol(),bid,0,0,"Sell trade");
            ClosePosByType(POSITION_TYPE_BUY); //vice versa if the model predicts bear market
        }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void ConvertTimeToStandard()
 {
// A one time attempt to convert the date to yy.mm.dd
    
    ArrayResize(date_datetime,ArraySize(dates));
    for (int i=0; i<ArraySize(dates); i++)
       {
         StringReplace(dates[i],"/","."); //replace comma with period in each and every date
         //Print(dates[i]);
         string mm_dd_yy[];
         
         ushort sep = StringGetCharacter(".",0);
         StringSplit(dates[i],sep,mm_dd_yy); //separate month, day and year 
         
         //Print("mm dd yy date format");
         //ArrayPrint(mm_dd_yy);
         
         string year = mm_dd_yy[2];
         string  day = mm_dd_yy[1];
         string month = mm_dd_yy[0];
                
         dates[i] = year+"."+month+"."+day; //store to a yy.mm.dd format
         
         date_datetime[i] = StringToTime(dates[i]); //lastly convert the string datetime to an actual date and time
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
void ClosePosByType(ENUM_POSITION_TYPE type)
 {
    for (int i=PositionsTotal()-1; i>=0; i--)
      if (m_position.SelectByIndex(i))
         if (m_position.Magic() == MAGIC_NUMBER && m_position.Symbol()==Symbol() && m_position.PositionType()==type)
             {
               m_trade.PositionClose(m_position.Ticket());
             }
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int PositionCounter(ENUM_POSITION_TYPE type)
 {
    int counter=0;
    for (int i=PositionsTotal()-1; i>=0; i--)
      if (m_position.SelectByIndex(i))
         if (m_position.Magic() == MAGIC_NUMBER && m_position.Symbol()==Symbol() && m_position.PositionType()==type)
             {
               counter++;
             }
     return(counter);
 }  
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void GetColumnDatatoArray(int from_column_number, int &toArr[])
 {
//---
    int counter=0;    
    int column = 0, rows=0;
    
    int m_handle  = FileOpen(file_name,FILE_READ|FILE_CSV|FILE_ANSI,delimiter,CP_UTF8); 

    if (m_handle == INVALID_HANDLE)
         Print(__FUNCTION__," Invalid csv handle err=",GetLastError());
    
//---

    while (!FileIsEnding(m_handle))
      {
        string data = FileReadString(m_handle);
        
        column++;
//---      
        if (column==from_column_number)
           {
               if (rows>=1) //Avoid the first column which contains the column's header
                 {   
                     counter++;
                     ArrayResize(toArr,counter); 
                     toArr[counter-1]=(int)data;
                 }   
                  
           }
//---
        if (FileIsLineEnding(m_handle))
          {                     
            rows++;
            column=0;
          }
      }
    FileClose(m_handle);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void GetColumnDatatoArray(int from_column_number, string &toArr[])
 {
//---
    int counter=0;    
    int column = 0, rows=0;
    
    int m_handle  = FileOpen(file_name,FILE_READ|FILE_CSV|FILE_ANSI,delimiter); 

    if (m_handle == INVALID_HANDLE)
         Print(__FUNCTION__," Invalid csv handle err=",GetLastError());
      
//---

    while (!FileIsEnding(m_handle))
      {
        string data = FileReadString(m_handle);
        
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
        if (FileIsLineEnding(m_handle))
          {                     
            rows++;
            column=0;
          }
      }
    FileClose(m_handle);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+


