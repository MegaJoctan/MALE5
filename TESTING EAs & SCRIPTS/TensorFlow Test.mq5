//+------------------------------------------------------------------+
//|                                              TensorFlow Test.mq5 |
//|                                    Copyright 2022, Fxalgebra.com |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, Fxalgebra.com"
#property link      "https://www.mql5.com/en/users/omegajoctan"
#property version   "1.00"
#property description "This is a Test EA file for testing the Tensors.mqh file for the MALE5 repository located at /MALE5"

#include <MALE5\Tensors.mqh>

CTensors *weight_tensor;
CTensors *input_tensors;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   
   weight_tensor = new CTensors(2);
   input_tensors = new CTensors(2);
   
   //weight_tensor.TensorAdd()
   
   matrix input_1 = {{1,2}};
   matrix w_1 = {
                  {0.1,0.2,0.3},
                  {0.4,0.5,0.6}
                  };
                  
   input_tensors.TensorAdd(input_1,0);
   weight_tensor.TensorAdd(w_1,0);
   
   matrix input_2 = {{3,4}};
   matrix w_2 = {
                  {0.8,0.8},
                  {0.9, 0.9}
                };
   
   input_tensors.TensorAdd(input_2,1);
   weight_tensor.TensorAdd(w_2,1);
   
   Print("Input 1\n",input_tensors.Tensor(0),"\nweight 1\n",weight_tensor.Tensor(0));
   
//---

   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---

   delete (weight_tensor);
   delete (input_tensors);   
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
   
  }
//+------------------------------------------------------------------+
