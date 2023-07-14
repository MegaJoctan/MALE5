//+------------------------------------------------------------------+
//|                                                preprocessing.mqh |
//|                                    Copyright 2022, Fxalgebra.com |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, Fxalgebra.com"
#property link      "https://www.mql5.com/en/users/omegajoctan"

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+


enum norm_technique
 {
   NORM_MIN_MAX_SCALER, //MIN MAX SCALER
   NORM_MEAN_NORM,      //MEAN NORMALIZATION
   NORM_STANDARDIZATION, //STANDARDIZATION
   NORM_NONE            //DO NOT NORMALIZE
 }; 

class CPreprocessing
  {
//---

struct standardization_struct
 {
   vector<double> mean;
   vector<double> std;
 };
    
struct min_max_struct
  {
    vector<double> min;
    vector<double> max;
  };

struct mean_norm_struct
 {
   vector<double> mean;
   vector<double> min;
   vector<double> max;
 };
 
private:
      ulong  m_rows, m_cols;
      norm_technique norm_method;

      void Standardization(vector &v);
      void Standardization(matrix &matrix_);
      
      void ReverseStandardization(vector &v);
      void ReverseStandardization(matrix &matrix_);
//---
      void MinMaxScaler(vector &v);
      void MinMaxScaler(matrix &matrix_);
      
      void ReverseMinMaxScaler(vector &v);
      void ReverseMinMaxScaler(matrix &matrix_);
//---

      void MeanNormalization(vector &v);
      void MeanNormalization(matrix &matrix_);
      
      void ReverseMeanNormalization(vector &v);
      void ReverseMeanNormalization(matrix &matrix_);      
//---
         
      
   public:
                        
                        CPreprocessing(matrix &matrix_, norm_technique NORM_MODE); 
                        
                       //---
                        
                        CPreprocessing(vector &mean_norm_max, vector &mean_norm_mean, vector &mean_norm_min);
                        CPreprocessing(vector &min_max_max, vector &min_max_min); 
                        CPreprocessing(vector &stdn_mean, vector &stdn_std, norm_technique NORM_MODE);
                        
                       ~CPreprocessing(void);
                       
                       standardization_struct standardization_scaler;
                       min_max_struct min_max_scaler;
                       mean_norm_struct mean_norm_scaler;
                       
                       void Normalization(vector &v);
                       void Normalization(matrix &matrix_);
                       
                       void ReverseNormalization(vector &v);
                       void ReverseNormalization(matrix &matrix_);
  };
//+------------------------------------------------------------------+
//| For normalizing and reverse normalizing the given x-matrix       |
//| This constructor obtains crucial information such as mean, min   |
//| max and Std deviation from the dataset, this information is used |
//| during reverse normalization for turning the data back to its    |
//| original state
//+------------------------------------------------------------------+
CPreprocessing::CPreprocessing(matrix &matrix_, norm_technique NORM_MODE)
 {    
   m_cols = matrix_.Cols();
   m_rows = matrix_.Rows();
   
   norm_method = NORM_MODE;
   
   vector v = {}; 
   
   switch(norm_method)
     {
      case NORM_STANDARDIZATION:
         standardization_scaler.mean.Resize(m_cols);
         standardization_scaler.std.Resize(m_cols);
         
          for (ulong i=0; i<m_cols; i++) { 
                v = matrix_.Col(i); 
                standardization_scaler.mean[i] = v.Mean();
                standardization_scaler.std[i] = v.Std();
             }
        
        break;
        
      case NORM_MEAN_NORM:
      
         mean_norm_scaler.mean.Resize(m_cols);
         mean_norm_scaler.min.Resize(m_cols);
         mean_norm_scaler.max.Resize(m_cols);
         
          for (ulong i=0; i<m_cols; i++) { 
                v = matrix_.Col(i); 
                
                mean_norm_scaler.min[i] = v.Min();
                mean_norm_scaler.max[i] = v.Max();
                mean_norm_scaler.mean[i] = v.Mean();
             }
             
        break;
        
      case NORM_MIN_MAX_SCALER:
         min_max_scaler.max.Resize(m_cols);
         min_max_scaler.min.Resize(m_cols);
         
          for (ulong i=0; i<m_cols; i++) { 
                v = matrix_.Col(i); 
                
                min_max_scaler.min[i] = v.Min();
                min_max_scaler.max[i] = v.Max();
             }
             
         break;
      
      case NORM_NONE:
            return;
         break;   
       
    }
     
   Normalization(matrix_);
 }
//+------------------------------------------------------------------+
//|   In case the Normalization techniques and normalization         |
//| information are known from pre-trained model or class instance   |
//  the following classes may be appropriate to use instead          |            
//+------------------------------------------------------------------+
CPreprocessing::CPreprocessing(vector &stdn_mean, vector &stdn_std, norm_technique NORM_MODE)
 {
   this.norm_method = NORM_STANDARDIZATION;
   this.m_cols = stdn_mean.Size();
   
   standardization_scaler.mean = stdn_mean;
   standardization_scaler.std = stdn_std;
 }
CPreprocessing::CPreprocessing(vector &min_max_max, vector &min_max_min)
 {
   this.norm_method =  NORM_MIN_MAX_SCALER;
   this.m_cols = min_max_max.Size();
  
   min_max_scaler.max = min_max_max;
   min_max_scaler.min = min_max_min;
 }
 
CPreprocessing::CPreprocessing(vector &mean_norm_max, vector &mean_norm_mean, vector &mean_norm_min)
 {
   this.norm_method = NORM_MEAN_NORM;
   this.m_cols = mean_norm_max.Size();
   
   mean_norm_scaler.max = mean_norm_max;
   mean_norm_scaler.mean = mean_norm_mean;
   mean_norm_scaler.min = mean_norm_min;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CPreprocessing::~CPreprocessing(void)
 {
   ZeroMemory(standardization_scaler.mean);
   ZeroMemory(standardization_scaler.std);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

void CPreprocessing::Standardization(vector &v)
 {
   for (ulong i=0; i<m_cols; i++)
      v[i] = (v[i] - standardization_scaler.mean[i]) / standardization_scaler.std[i];  
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CPreprocessing::Standardization(matrix &matrix_)
 {
  vector v;
  for (ulong i=0; i<m_rows; i++)
    {
       v = matrix_.Row(i);
       
       Standardization(v);
       matrix_.Row(v, i);  
    }
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CPreprocessing::ReverseStandardization(vector &v)
 {
    for (ulong i=0; i<m_cols; i++) 
        v[i] = (v[i] * standardization_scaler.std[i]) + standardization_scaler.mean[i];
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CPreprocessing::ReverseStandardization(matrix &matrix_)
 {
  for (ulong i=0; i<m_rows; i++)
    { 
      vector v = matrix_.Row(i);
      
      ReverseStandardization(v);
      matrix_.Row(v,i);
    }  
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

void CPreprocessing::Normalization(vector &v)
 {
   if (v.Size() != m_cols)
     {
       Print(__FUNCTION__," Can't Standardize the data | Vector v needs to have the same size as the Columns ",m_cols," of the Matrix given");
       return;
     }
   
   switch(norm_method)
     {
      case  NORM_STANDARDIZATION:
        Standardization(v);
        break;
        
      case NORM_MIN_MAX_SCALER:
         MinMaxScaler(v);
         break;
         
      case NORM_MEAN_NORM: 
         MeanNormalization(v);
         break;
     }
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CPreprocessing::Normalization(matrix &matrix_)
 {
   vector v;
   switch(norm_method)
     {
      case  NORM_STANDARDIZATION:
        Standardization(matrix_);
        break;
        
      case NORM_MIN_MAX_SCALER:
        MinMaxScaler(matrix_);
        break;
         
      case  NORM_MEAN_NORM:
        MeanNormalization(matrix_);
        break;
     }
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CPreprocessing::ReverseNormalization(vector &v)
 {
   if (v.Size() != m_cols)
     {
       Print(__FUNCTION__," Can't Reverse Standardize the data | Vector v sized ",v.Size()," needs to have the same size as the Columns ",m_cols," of the Matrix given");
       return;
     }
     
   switch(norm_method)
     {
      case  NORM_STANDARDIZATION:
        ReverseStandardization(v);
        break;
        
      case NORM_MIN_MAX_SCALER:
         ReverseMinMaxScaler(v);
         break;
      
      case NORM_MEAN_NORM:  
         ReverseMeanNormalization(v);
         break;   
     }
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

void CPreprocessing::ReverseNormalization(matrix &matrix_)
 {
  
  switch(norm_method)
    {
     case  NORM_STANDARDIZATION:
       ReverseStandardization(matrix_);
       break;
       
     case NORM_MIN_MAX_SCALER:
        ReverseMinMaxScaler(matrix_);
        break;
        
     case NORM_MEAN_NORM: 
        ReverseMeanNormalization(matrix_);
        break;
    }
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

void CPreprocessing::MinMaxScaler(vector &v)
 {
   if (v.Size() != m_cols)
     {
       Print(__FUNCTION__," Can't Normalize the data | Vector v sized ",v.Size()," needs to have the same size as the Columns ",m_cols," of the Matrix given");
       return;
     }
     
   for (ulong i=0; i<m_cols; i++)
     v[i] = (v[i] - min_max_scaler.min[i]) / (min_max_scaler.max[i] - min_max_scaler.min[i]);  
 } 

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+ 

void CPreprocessing::MinMaxScaler(matrix &matrix_)
 {
   vector v = {}; 
   
    for (ulong i=0; i<m_rows; i++)
       { 
          v = matrix_.Row(i); 
          MinMaxScaler(v);
          
          matrix_.Row(v,i);  
       }
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

void CPreprocessing::ReverseMinMaxScaler(matrix &matrix_)
 {
    for (ulong i=0; i<matrix_.Rows(); i++)
       {
         vector v = matrix_.Row(i);
         ReverseMinMaxScaler(v);
         
         matrix_.Row(v, i);
       } 
 }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

void CPreprocessing::ReverseMinMaxScaler(vector &v)
 {  
   if (v.Size() != m_cols)
     {
       Print(__FUNCTION__," Can't Reverse Normalize the data | Vector v sized ",v.Size()," needs to have the same size as the Columns ",m_cols," of the Matrix given");
       return;
     }
     
    for (ulong i=0; i<m_cols; i++) 
       v[i] = (v[i]* (min_max_scaler.max[i] - min_max_scaler.min[i])) + min_max_scaler.min[i];  
 }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

void CPreprocessing::MeanNormalization(vector &v)
 {
   if (v.Size() != m_cols)
     {
       Print(__FUNCTION__," Can't Normalize the data | Vector v sized ",v.Size()," needs to have the same size as the Columns ",m_cols," of the Matrix given");
       return;
     }
     
   for (ulong i=0; i<m_cols; i++)
      v[i] = (v[i] - mean_norm_scaler.mean[i]) / (mean_norm_scaler.max[i] - mean_norm_scaler.min[i]);
    
 }
 
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+


void CPreprocessing::MeanNormalization(matrix &matrix_)
 {
   vector v = {};  
   
    for (ulong i=0; i<matrix_.Rows(); i++)
       { 
          v = matrix_.Row(i); 
          MeanNormalization(v);
          
          matrix_.Row(v,i);  
       }
 }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

void CPreprocessing::ReverseMeanNormalization(vector &v)
 {
   if (v.Size() != m_cols)
     {
       Print(__FUNCTION__," Can't Reverse Normalize the data | Vector v sized ",v.Size()," needs to have the same size as the Columns ",m_cols," of the Matrix given");
       return;
     }
     
    for (ulong i=0; i<m_cols; i++)
      v[i] = (v[i] * (mean_norm_scaler.max[i] - mean_norm_scaler.min[i]) ) + mean_norm_scaler.mean[i];
 }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

void CPreprocessing::ReverseMeanNormalization(matrix &matrix_)
 {
    for (ulong i=0; i<m_rows; i++)
       {
         vector v = matrix_.Row(i);
         MeanNormalization(v);
         
         matrix_.Row(v,i);
       }  
 }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+