//+------------------------------------------------------------------+
//|                                                   TestScript.mq5 |
//|                                    Copyright 2022, Omega Joctan. |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, Omega Joctan."
#property link      "https://www.mql5.com/en/users/omegajoctan"
#property version   "1.00"
//+------------------------------------------------------------------+

#include "KNN_nearest_neighbors.mqh";
CKNNNearestNeighbors *nearest_neighbors;

//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
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
    
    vector v = {57, 170};
    
    //Print("Dataset\n",Matrix);
    
    nearest_neighbors = new CKNNNearestNeighbors(Matrix,1);
    //nearest_neighbors.KNNAlgorithm(v);
    //vector CV = nearest_neighbors.CrossValidation_LOOCV();
    
    //Print("cv ",CV);
    nearest_neighbors.TrainTest();
    
   
    delete(nearest_neighbors);   
  }
//+------------------------------------------------------------------+
