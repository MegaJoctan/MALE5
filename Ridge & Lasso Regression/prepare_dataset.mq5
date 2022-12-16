//+------------------------------------------------------------------+
//|                                              prepare_dataset.mq5 |
//|                                    Copyright 2022, Fxalgebra.com |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, Fxalgebra.com"
#property link      "https://www.mql5.com/en/users/omegajoctan"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+

#include <matrix_utils.mqh>
CMatrixutils matrix_utils;

matrix data_matrix(1000,6);

int stochastic_handle;
int rsi_handle;
int volume_handle;
int bulls_handle;
int bears_handle;

double stocastic_buffer[];
double rsi_buffer[];
double volume_buffer[];
double bears_buffer[];
double bulls_buffer[];

vector stoc_vector;
vector rsi_vector;
vector volume_vector;
vector bulls_vector;
vector bears_vector;

void OnStart()
  {
//---

      rsi_handle = iRSI(Symbol(),PERIOD_CURRENT,13,PRICE_CLOSE);
      stochastic_handle = iStochastic(Symbol(),PERIOD_CURRENT,5,3,3,MODE_SMA,STO_LOWHIGH);
      volume_handle = iVolumes(Symbol(),PERIOD_CURRENT,VOLUME_TICK);
      bears_handle = iBearsPower(Symbol(),PERIOD_CURRENT,13);
      bulls_handle = iBullsPower(Symbol(),PERIOD_CURRENT,13);
      
      vector price;
      price.CopyRates(Symbol(),PERIOD_CURRENT,COPY_RATES_CLOSE,0,1000);
      
      CopyBuffer(rsi_handle,0,0,1000,rsi_buffer);
      CopyBuffer(stochastic_handle,0,0,1000,stocastic_buffer);
      CopyBuffer(volume_handle,0,0,1000,volume_buffer);
      CopyBuffer(bulls_handle,0,0,1000,bulls_buffer);
      CopyBuffer(bears_handle,0,0,1000,bears_buffer);
      
      stoc_vector = matrix_utils.ArrayToVector(stocastic_buffer);
      rsi_vector = matrix_utils.ArrayToVector(rsi_buffer);
      volume_vector = matrix_utils.ArrayToVector(volume_buffer);
      bears_vector = matrix_utils.ArrayToVector(bulls_buffer);
      bulls_vector = matrix_utils.ArrayToVector(bears_buffer);
      
      data_matrix.Col(stoc_vector,0);
      data_matrix.Col(rsi_vector,1);
      data_matrix.Col(volume_vector,2);
      data_matrix.Col(bears_vector,3);
      data_matrix.Col(bulls_vector,4);
      data_matrix.Col(price,5);
      
      matrix_utils.WriteCsv("Oscillators.csv",data_matrix);
  }
//+------------------------------------------------------------------+
