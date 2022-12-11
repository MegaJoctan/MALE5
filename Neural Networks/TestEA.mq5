//+------------------------------------------------------------------+
//|                                                       TestEA.mq5 |
//|                                    Copyright 2022, Omega Joctan. |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, Omega Joctan."
#property link      "https://www.mql5.com/en/users/omegajoctan"
#property version   "1.00"
#property strict

//+------------------------------------------------------------------+
#define DEBUG_MODE

#include "neural_nn_lib.mqh";
CNeuralNets *neural_nets;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
    matrix Matrix = 
      {//weight(kg) | height(cm) | class
         {51, 167,   1}, //underweight
         {62, 182,   0}, //Normal
         {69, 176,   0}, //Normal
         {64, 173,   0}, //Normal
         {65, 172,   0}, //Normal
         {56, 174,   1}, //Underweight
         {58, 169,   0}, //Normal
         {57, 173,   0}, //Normal
         {55, 170,   0}  //Normal
      };
   
   neural_nets = new CNeuralNets(Matrix,AF_SIGMOID,5,BCE);
   neural_nets.BackPropagation(ADAM,0.1,1000);
   

   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
    delete(neural_nets);
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
   
  }
//+------------------------------------------------------------------+
