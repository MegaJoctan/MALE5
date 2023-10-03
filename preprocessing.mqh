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
 }; 


template <typename T_vector, typename T_matrix>
class CPreprocessing
  {
//---

struct standardization_struct
 {
   T_vector mean;
   T_vector std;
 };
    
struct min_max_struct
  {
    T_vector min;
    T_vector max;
  };

struct mean_norm_struct
 {
   T_vector mean;
   T_vector min;
   T_vector max;
 };
 
private:
      ulong  m_rows, m_cols;
      norm_technique norm_method;

      bool Standardization(T_vector &v);
      bool Standardization(T_matrix &matrix_);
      
      bool ReverseStandardization(T_vector &v);
      bool ReverseStandardization(T_matrix &matrix_);
//---
      bool MinMaxScaler(T_vector &v);
      bool MinMaxScaler(T_matrix &matrix_);
      
      bool ReverseMinMaxScaler(T_vector &v);
      bool ReverseMinMaxScaler(T_matrix &matrix_);
//---

      bool MeanNormalization(T_vector &v);
      bool MeanNormalization(T_matrix &matrix_);
      
      bool ReverseMeanNormalization(T_vector &v);
      bool ReverseMeanNormalization(T_matrix &matrix_);      
//---
         
      
   public:
                        
                        CPreprocessing(T_matrix &matrix_, norm_technique NORM_MODE); 
                        
                       //---
                        
                        CPreprocessing(T_vector &mean_norm_max, T_vector &mean_norm_mean, T_vector &mean_norm_min);
                        CPreprocessing(T_vector &min_max_max, T_vector &min_max_min); 
                        CPreprocessing(T_vector &stdn_mean, T_vector &stdn_std, norm_technique NORM_MODE);
                        
                       ~CPreprocessing(void);
                       
                       standardization_struct standardization_scaler;
                       min_max_struct min_max_scaler;
                       mean_norm_struct mean_norm_scaler;
                       
                       bool Normalization(T_vector &v);
                       bool Normalization(T_matrix &matrix_);
                       
                       bool ReverseNormalization(T_vector &v);
                       bool ReverseNormalization(T_matrix &matrix_);
  };
//+------------------------------------------------------------------+
//| For normalizing and reverse normalizing the given x-matrix       |
//| This constructor obtains crucial information such as mean, min   |
//| max and Std deviation from the dataset, this information is used |
//| during reverse normalization for turning the data back to its    |
//| original state                                                   |
//+------------------------------------------------------------------+
template <typename T_vector, typename T_matrix>
CPreprocessing::CPreprocessing(T_matrix &matrix_, norm_technique NORM_MODE)
 {    
   m_cols = matrix_.Cols();
   m_rows = matrix_.Rows();
   
   norm_method = NORM_MODE;
   
   T_vector v = {}; 
   
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
    }
     
   Normalization(matrix_);
 }
//+------------------------------------------------------------------+
//|   In case the Normalization techniques and normalization         |
//| information are known from pre-trained model or class instance   |
//  the following classes may be appropriate to use instead          |            
//+------------------------------------------------------------------+
template <typename T_vector, typename T_matrix>
CPreprocessing::CPreprocessing(T_vector &stdn_mean, T_vector &stdn_std, norm_technique NORM_MODE)
 {
   this.norm_method = NORM_STANDARDIZATION;
   this.m_cols = stdn_mean.Size();
   
   standardization_scaler.mean = stdn_mean;
   standardization_scaler.std = stdn_std;
 }
template <typename T_vector, typename T_matrix>
CPreprocessing::CPreprocessing(T_vector &min_max_max, T_vector &min_max_min)
 {
   this.norm_method =  NORM_MIN_MAX_SCALER;
   this.m_cols = min_max_max.Size();
  
   min_max_scaler.max = min_max_max;
   min_max_scaler.min = min_max_min;
 }
template <typename T_vector, typename T_matrix>
CPreprocessing::CPreprocessing(T_vector &mean_norm_max, T_vector &mean_norm_mean, T_vector &mean_norm_min)
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
template <typename T_vector, typename T_matrix>
CPreprocessing::~CPreprocessing(void)
 {
   ZeroMemory(standardization_scaler.mean);
   ZeroMemory(standardization_scaler.std);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
template <typename T_vector, typename T_matrix>
bool CPreprocessing::Standardization(T_vector &v)
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
template <typename T_vector, typename T_matrix>
bool CPreprocessing::Standardization(T_matrix &matrix_)
 {
  T_vector v;
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
template <typename T_vector, typename T_matrix>
bool CPreprocessing::ReverseStandardization(T_vector &v)
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
template <typename T_vector, typename T_matrix>
bool CPreprocessing::ReverseStandardization(T_matrix &matrix_)
 {
 bool norm =true;
 
  for (ulong i=0; i<matrix_.Rows(); i++)
    { 
      T_vector v = matrix_.Row(i);
      
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
template <typename T_vector, typename T_matrix>
bool CPreprocessing::Normalization(T_vector &v)
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
template <typename T_vector, typename T_matrix>
bool CPreprocessing::Normalization(T_matrix &matrix_)
 {
   T_vector v;
   
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
template <typename T_vector, typename T_matrix>
bool CPreprocessing::ReverseNormalization(T_vector &v)
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
template <typename T_vector, typename T_matrix>
bool CPreprocessing::ReverseNormalization(T_matrix &matrix_)
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
template <typename T_vector, typename T_matrix>
bool CPreprocessing::MinMaxScaler(T_vector &v)
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
template <typename T_vector, typename T_matrix>
bool CPreprocessing::MinMaxScaler(T_matrix &matrix_)
 {
   T_vector v = {}; 
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
template <typename T_vector, typename T_matrix>
bool CPreprocessing::ReverseMinMaxScaler(T_matrix &matrix_)
 {
 bool norm =true;
 
    for (ulong i=0; i<matrix_.Rows(); i++)
       {
         T_vector v = matrix_.Row(i);
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
template <typename T_vector, typename T_matrix>
bool CPreprocessing::ReverseMinMaxScaler(T_vector &v)
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
template <typename T_vector, typename T_matrix>
bool CPreprocessing::MeanNormalization(T_vector &v)
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
template <typename T_vector, typename T_matrix>
bool CPreprocessing::MeanNormalization(T_matrix &matrix_)
 {
   T_vector v = {};  
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
template <typename T_vector, typename T_matrix>
bool CPreprocessing::ReverseMeanNormalization(T_vector &v)
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
template <typename T_vector, typename T_matrix>
bool CPreprocessing::ReverseMeanNormalization(T_matrix &matrix_)
 {
  bool norm =true;
  
    for (ulong i=0; i<matrix_.Rows(); i++)
       {
         T_vector v = matrix_.Row(i);
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