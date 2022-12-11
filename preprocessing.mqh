//+------------------------------------------------------------------+
//|                                                preprocessing.mqh |
//|                                    Copyright 2022, Omega Joctan. |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, Omega Joctan."
#property link      "https://www.mql5.com/en/users/omegajoctan"
//+------------------------------------------------------------------+
//| defines                                                          |
//+------------------------------------------------------------------+
struct min_max
  {
    double min[];
    double max[];
  } min_max_scaler;

//+------------------------------------------------------------------+

struct mean_norm
 {
   double mean[];
   double min[];
   double max[];
 } mean_norm_scaler;

//+------------------------------------------------------------------+

struct standardization
 {
   double mean[];
   double std[];
 } standardization_scaler;

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

class CPreprocessing
  {      
      public:
                           CPreprocessing(void);
                          ~CPreprocessing(void);
                           
                           void MinMaxScaler(vector &v,ulong index=0);
                           void MinMaxScaler(matrix &mat);
                           void MeanNormalization(vector &v,ulong index=0);
                           void MeanNormalization(matrix &mat);
                           void Standardization(vector &v, ulong index=0);
                           void Standardization(matrix &mat);
                      //---
                           void ReverseMinMaxScaler(vector &v,ulong index=0);
                           void ReverseMinMaxScaler(matrix &mat);
                           void ReverseMeanNormalization(vector &v, ulong index=0);
                           void ReverseMeanNormalization(matrix &mat);
                           void ReverseStandardization(vector &v, ulong index=0);
                           void ReverseStandardization(matrix &mat);
  };

//+------------------------------------------------------------------+

CPreprocessing::CPreprocessing(void)
 {
   ArrayResize(min_max_scaler.max,1);
   ArrayResize(min_max_scaler.min,1);
   
   ArrayResize(mean_norm_scaler.max,1);
   ArrayResize(mean_norm_scaler.mean,1);
   ArrayResize(mean_norm_scaler.min,1);
   
   ArrayResize(standardization_scaler.mean,1);
   ArrayResize(standardization_scaler.std,1);
 }

//+------------------------------------------------------------------+

CPreprocessing::~CPreprocessing(void)
 {
 
 }

//+------------------------------------------------------------------+

void CPreprocessing::MinMaxScaler(vector &v,ulong index=0)
 {
   //Normalizing vector using Min-max scaler
   
   min_max_scaler.min[index] = v.Min();
   min_max_scaler.max[index] = v.Max();
   
   for (int i=0; i<(int)v.Size(); i++)
     v[i] = (v[i] - min_max_scaler.min[index]) / (min_max_scaler.max[index] - min_max_scaler.min[index]);  
   
 } 

//+------------------------------------------------------------------+

void CPreprocessing::MeanNormalization(vector &v,ulong index=0)
 {
   mean_norm_scaler.mean[index] = v.Mean();
   mean_norm_scaler.max[index] = v.Max();
   mean_norm_scaler.min[index] = v.Min();
          
   for (ulong i=0; i<v.Size(); i++)
      v[i] = (v[i] - mean_norm_scaler.mean[index]) / (mean_norm_scaler.max[index] - mean_norm_scaler.min[index]);
    
 }
 
//+------------------------------------------------------------------+

void CPreprocessing::Standardization(vector &v, ulong index=0)
 {
    standardization_scaler.mean[index] = v.Mean();
    standardization_scaler.std[index] = v.Std();
      
      for (ulong i=0; i<v.Size(); i++)
        v[i] = (v[i] - standardization_scaler.mean[index]) / standardization_scaler.std[index];
 }

//+------------------------------------------------------------------+

void CPreprocessing::MinMaxScaler(matrix &mat)
 {
   vector v = {}; 
   
   ArrayResize(min_max_scaler.min,(int)mat.Cols());
   ArrayResize(min_max_scaler.max,(int)mat.Cols());
   
    for (ulong i=0; i<mat.Cols(); i++)
       { 
          v = mat.Col(i); 
          MinMaxScaler(v,i);
          mat.Col(v,i);  
       }
 }

//+------------------------------------------------------------------+

void CPreprocessing::MeanNormalization(matrix &mat)
 {
   vector v = {}; 
   
   ArrayResize(mean_norm_scaler.max,(int)mat.Cols());
   ArrayResize(mean_norm_scaler.min,(int)mat.Cols());
   ArrayResize(mean_norm_scaler.mean,(int)mat.Cols());
   
    for (ulong i=0; i<mat.Cols(); i++)
       { 
          v = mat.Col(i); 
          MeanNormalization(v,i);
          mat.Col(v,i);  
       }
 }

//+------------------------------------------------------------------+

void CPreprocessing::Standardization(matrix &mat)
 {
   vector v = {}; 
   
   ArrayResize(standardization_scaler.mean,(int)mat.Cols());
   ArrayResize(standardization_scaler.std,(int)mat.Cols());
   
    for (ulong i=0; i<mat.Cols(); i++)
       { 
          v = mat.Col(i); 
          Standardization(v,i);
          mat.Col(v,i);  
       }
 }
 
//+------------------------------------------------------------------+
//|                                                                  |
//| REVERSING NORMALIZATION | RETURNING THE VALUES TO THEIR ORIGINAL |
//|   STATE BEFORE THEY WERE NORMALIZED                              |
//|                                                                  |
//+------------------------------------------------------------------+


void CPreprocessing::ReverseMinMaxScaler(vector &v,ulong index=0)
 {  
    for (ulong i=0; i<v.Size(); i++) 
       v[i] = (v[i]* (min_max_scaler.max[index] - min_max_scaler.min[index])) + min_max_scaler.min[index];  
 }

//+------------------------------------------------------------------+

void CPreprocessing::ReverseMeanNormalization(vector &v, ulong index=0)
 {
    for (ulong i=0; i<v.Size(); i++)
      v[i] = (v[i] * (mean_norm_scaler.max[index] - mean_norm_scaler.min[index]) ) + mean_norm_scaler.mean[index];
 }

//+------------------------------------------------------------------+

void CPreprocessing::ReverseStandardization(vector &v, ulong index=0)
 {
    for (ulong i=0; i<v.Size(); i++)
      v[i] = (v[i] * standardization_scaler.std[index]) + standardization_scaler.mean[index];
 }

//+------------------------------------------------------------------+

void CPreprocessing::ReverseMinMaxScaler(matrix &mat)
 {
    for (ulong i=0; i<mat.Cols(); i++)
       {
         vector v = mat.Col(i);
         ReverseMinMaxScaler(v,i);
         mat.Col(v,i);
       } 
 }

//+------------------------------------------------------------------+

void CPreprocessing::ReverseMeanNormalization(matrix &mat)
 {
    for (ulong i=0; i<mat.Cols(); i++)
       {
         vector v = mat.Col(i);
         ReverseMeanNormalization(v,i);
         mat.Col(v,i);
       }  
 }

//+------------------------------------------------------------------+

void CPreprocessing::ReverseStandardization(matrix &mat)
 {
    for (ulong i=0; i<mat.Cols(); i++)
       {
         vector v = mat.Col(i);
         ReverseStandardization(v,i);
         mat.Col(v,i);
       }  
 }

//+------------------------------------------------------------------+
