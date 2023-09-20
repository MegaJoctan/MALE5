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

      bool Standardization(vector &v);
      bool Standardization(matrix &matrix_);
      
      bool ReverseStandardization(vector &v);
      bool ReverseStandardization(matrix &matrix_);
//---
      bool MinMaxScaler(vector &v);
      bool MinMaxScaler(matrix &matrix_);
      
      bool ReverseMinMaxScaler(vector &v);
      bool ReverseMinMaxScaler(matrix &matrix_);
//---

      bool MeanNormalization(vector &v);
      bool MeanNormalization(matrix &matrix_);
      
      bool ReverseMeanNormalization(vector &v);
      bool ReverseMeanNormalization(matrix &matrix_);      
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
                       
                       bool Normalization(vector &v);
                       bool Normalization(matrix &matrix_);
                       
                       bool ReverseNormalization(vector &v);
                       bool ReverseNormalization(matrix &matrix_);
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

bool CPreprocessing::Standardization(vector &v)
 {
   if (v.Size() != m_cols)
     {
       Print(__FUNCTION__," Can't Normalize the data | Vector v sized ",v.Size()," needs to have the same size as the Columns ",m_cols," of the Matrix given");
       return false;
     }
     
   for (ulong i=0; i<m_cols; i++)
      v[i] = (v[i] - standardization_scaler.mean[i]) / standardization_scaler.std[i];  
    
   return true;  
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CPreprocessing::Standardization(matrix &matrix_)
 {
  vector v;
  bool norm = true;
  
  for (ulong i=0; i<matrix_.Rows(); i++)
    {
       v = matrix_.Row(i);
       
       if (!Standardization(v))
         {
            norm = false;
            break;
         }
       matrix_.Row(v, i);  
    }
    
   return norm;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CPreprocessing::ReverseStandardization(vector &v)
 {
   if (v.Size() != m_cols)
     {
       Print(__FUNCTION__," Can't Normalize the data | Vector v sized ",v.Size()," needs to have the same size as the Columns ",m_cols," of the Matrix given");
       return false;
     }
     
    for (ulong i=0; i<m_cols; i++) 
        v[i] = (v[i] * standardization_scaler.std[i]) + standardization_scaler.mean[i];
    
    return true;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CPreprocessing::ReverseStandardization(matrix &matrix_)
 {
 bool norm =true;
 
  for (ulong i=0; i<matrix_.Rows(); i++)
    { 
      vector v = matrix_.Row(i);
      
      if (!ReverseStandardization(v))
        {
          norm =  false;
          break;
        }
      matrix_.Row(v,i);
    }  
    
  return norm;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

bool CPreprocessing::Normalization(vector &v)
 {
   if (v.Size() != m_cols)
     {
       Print(__FUNCTION__," Can't Standardize the data | Vector v needs to have the same size as the Columns ",m_cols," of the Matrix given");
       return false;
     }
   
   bool norm = true;
   
   switch(norm_method)
     {
      case  NORM_STANDARDIZATION:
        if (!Standardization(v))
         norm = false;
        break;
        
      case NORM_MIN_MAX_SCALER:
         if (!MinMaxScaler(v))
         norm = false;
         break;
         
      case NORM_MEAN_NORM: 
         if (MeanNormalization(v))
          norm = false;
         break;
     }
   return norm;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CPreprocessing::Normalization(matrix &matrix_)
 {
   vector v;
   
   bool norm = true;
   switch(norm_method)
     {
      case  NORM_STANDARDIZATION:
        if (!Standardization(matrix_))
        norm = false;
        break;
        
      case NORM_MIN_MAX_SCALER:
        if (!MinMaxScaler(matrix_))
        norm =false;
        break;
         
      case  NORM_MEAN_NORM:
        if (!MeanNormalization(matrix_))
        norm =false;
        break;
     }
     
  return norm;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CPreprocessing::ReverseNormalization(vector &v)
 {
   if (v.Size() != m_cols)
     {
       Print(__FUNCTION__," Can't Reverse Standardize the data | Vector v sized ",v.Size()," needs to have the same size as the Columns ",m_cols," of the Matrix given");
       return false;
     }
   
   bool norm = true;  
   switch(norm_method)
     {
      case  NORM_STANDARDIZATION:
        if (!ReverseStandardization(v))
        norm =  false;
        break;
        
      case NORM_MIN_MAX_SCALER:
         if (!ReverseMinMaxScaler(v))
         norm = false;
         break;
      
      case NORM_MEAN_NORM:  
         if (!ReverseMeanNormalization(v))
         norm =  false;
         break;   
     }
   return norm;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

bool CPreprocessing::ReverseNormalization(matrix &matrix_)
 {
  bool norm = true;
  
  switch(norm_method)
    {
     case  NORM_STANDARDIZATION:
       if (!ReverseStandardization(matrix_))
       norm = false;
       break;
       
     case NORM_MIN_MAX_SCALER:
        ReverseMinMaxScaler(matrix_);
        norm = false;
        break;
        
     case NORM_MEAN_NORM: 
        ReverseMeanNormalization(matrix_);
        norm = false;
        break;
    }
  return norm;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

bool CPreprocessing::MinMaxScaler(vector &v)
 {
   if (v.Size() != m_cols)
     {
       Print(__FUNCTION__," Can't Normalize the data | Vector v sized ",v.Size()," needs to have the same size as the Columns ",m_cols," of the Matrix given");
       return false;
     }
     
   for (ulong i=0; i<m_cols; i++)
     v[i] = (v[i] - min_max_scaler.min[i]) / (min_max_scaler.max[i] - min_max_scaler.min[i]);  
     
    return true;
 } 

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+ 

bool CPreprocessing::MinMaxScaler(matrix &matrix_)
 {
   vector v = {}; 
   bool norm = true;
   
    for (ulong i=0; i<matrix_.Rows(); i++)
       { 
          v = matrix_.Row(i); 
          
          if (!MinMaxScaler(v))
           {
             norm = false;
             break;
           }
          
          matrix_.Row(v,i);  
       }
   return norm;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

bool CPreprocessing::ReverseMinMaxScaler(matrix &matrix_)
 {
 bool norm =true;
 
    for (ulong i=0; i<matrix_.Rows(); i++)
       {
         vector v = matrix_.Row(i);
         if (!ReverseMinMaxScaler(v))
           {
             norm = false;
             break;    
           }
         
         matrix_.Row(v, i);
       } 
   return norm;
 }
 
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

bool CPreprocessing::ReverseMinMaxScaler(vector &v)
 {  
   if (v.Size() != m_cols)
     {
       Print(__FUNCTION__," Can't Reverse Normalize the data | Vector v sized ",v.Size()," needs to have the same size as the Columns ",m_cols," of the Matrix given");
       return false;
     }
     
    for (ulong i=0; i<m_cols; i++) 
       v[i] = (v[i]* (min_max_scaler.max[i] - min_max_scaler.min[i])) + min_max_scaler.min[i];  
      
   
   return true;
 }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

bool CPreprocessing::MeanNormalization(vector &v)
 {
   if (v.Size() != m_cols)
     {
       Print(__FUNCTION__," Can't Normalize the data | Vector v sized ",v.Size()," needs to have the same size as the Columns ",m_cols," of the Matrix given");
       return false;
     }
     
   for (ulong i=0; i<m_cols; i++)
      v[i] = (v[i] - mean_norm_scaler.mean[i]) / (mean_norm_scaler.max[i] - mean_norm_scaler.min[i]);
    
   return true;
 }
 
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

bool CPreprocessing::MeanNormalization(matrix &matrix_)
 {
   vector v = {};  
   bool norm = true;
    for (ulong i=0; i<matrix_.Rows(); i++)
       { 
          v = matrix_.Row(i); 
          if (!MeanNormalization(v))
            {
               norm = false;
               break;
            }
          
          matrix_.Row(v,i);  
       }
   return norm;
 }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

bool CPreprocessing::ReverseMeanNormalization(vector &v)
 {
   if (v.Size() != m_cols)
     {
       Print(__FUNCTION__," Can't Reverse Normalize the data | Vector v sized ",v.Size()," needs to have the same size as the Columns ",m_cols," of the Matrix given");
       return false;
     }
     
    for (ulong i=0; i<m_cols; i++)
      v[i] = (v[i] * (mean_norm_scaler.max[i] - mean_norm_scaler.min[i]) ) + mean_norm_scaler.mean[i];
      
   return true;
 }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

bool CPreprocessing::ReverseMeanNormalization(matrix &matrix_)
 {
  bool norm =true;
  
    for (ulong i=0; i<matrix_.Rows(); i++)
       {
         vector v = matrix_.Row(i);
         if (!MeanNormalization(v))
           {
             norm = false;
             break;
           }
         matrix_.Row(v,i);
       }  
   return norm;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+