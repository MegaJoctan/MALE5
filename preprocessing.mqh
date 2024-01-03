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
class StandardizationScaler
  {
protected:
   vector mean, std;
   
public:
                     StandardizationScaler(void);
                    ~StandardizationScaler(void);
                    
                    matrix fit_transform(const matrix &X);
                    matrix transform(const matrix &X);
                    vector transform(const vector &X);
                    bool save(string csv_name);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
StandardizationScaler::StandardizationScaler(void)
 {
 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
StandardizationScaler::~StandardizationScaler(void)
 {
 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
matrix StandardizationScaler::fit_transform(const matrix &X)
 { 
    for (ulong i=0; i<X.Cols(); i++)
      { 
         this.mean[i] = X.Col(i).Mean();
         this.std[i] = X.Col(i).Std();
      }

//---
   return this.transform(X);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
matrix StandardizationScaler::transform(const matrix &X)
 {
   matrix X_norm = X;
   
   for (ulong i=0; i<X.Cols(); i++)
     X_norm.Col(this.transform(X.Col(i)), i);
   
   return X_norm;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector StandardizationScaler::transform(const vector &X)
 {
   vector v(X.Size());
   if (this.mean.Size()==0 || this.std.Size()==0)
     {
       printf("%s Call the fit_transform function fist before attempting to transform the new data",__FUNCTION__);
       return v;
     }
   
   if (X.Size() != this.mean.Size())
     {
         printf("%s X of size [%d] doesn't match the same number of features in a given X matrix on the fit_transform function call",__FUNCTION__,this.mean.Size());
         return v;
     }
   
   for (ulong i=0; i<v.Size(); i++)
      v[i] = (X[i] - this.mean[i]) / (this.std[i] + 1e-10);  
   
   return v;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+


/*

#define  NaN DBL_MAX*2

class CPreprocessing
  {
//---
template <typename T>
struct standardization_struct
 {
   vector<T> mean;
   vector<T> std;
   
   void standardization_struct::standardization_struct(void)
     {
       mean.Fill(NaN);
       std.Fill(NaN);
     }
 };
template <typename T>
struct min_max_struct
  {
    vector<T> min;
    vector<T> max;
    
    void min_max_struct::min_max_struct(void)
      {
         min.Fill(NaN);
         max.Fill(NaN);
      }
  };
  
template <typename T>
struct mean_norm_struct
 {
   vector<T> mean;
   vector<T> min;
   vector<T> max;
   
   void mean_norm_struct::mean_norm_struct(void)
     {
         mean.Fill(NaN);
         min.Fill(NaN);
         max.Fill(NaN);
     }
 };
 
private:
      ulong  m_cols;
      norm_technique norm_method;
      
      template <typename T>
      bool Standardization(vector<T> &v);
      template <typename T>
      bool Standardization(matrix<T> &matrix_);
      template <typename T>
      bool ReverseStandardization(vector<T> &v);
      template <typename T>
      bool ReverseStandardization(matrix<T> &matrix_);
//---
      template <typename T>
      bool MinMaxScaler(vector<T> &v);
      template <typename T>
      bool MinMaxScaler(matrix<T> &matrix_);
      template <typename T>
      bool ReverseMinMaxScaler(vector<T> &v);
      template <typename T>
      bool ReverseMinMaxScaler(matrix<T> &matrix_);
//---
      template <typename T>
      bool MeanNormalization(vector<T> &v);
      template <typename T>
      bool MeanNormalization(matrix<T> &matrix_);
      template <typename T>
      bool ReverseMeanNormalization(vector<T> &v);
      template <typename T>
      bool ReverseMeanNormalization(matrix<T> &matrix_);      
//---     
      
   public:
                        template <typename T>
                        CPreprocessing(matrix<T> &matrix_, norm_technique NORM_MODE); 
                        
                       //---
                       
                        template <typename T>
                        CPreprocessing(vector<T> &mean_norm_max, vector<T> &mean_norm_mean, vector<T> &mean_norm_min); //for mean normalization
                        template <typename T>
                        CPreprocessing(vector<T> &min_max_max, vector<T> &min_max_min);  //for min max scaler
                        template <typename T>
                        CPreprocessing(vector<T> &stdn_mean, vector<T> &stdn_std, norm_technique NORM_MODE); //for standardization
                        
                       ~CPreprocessing(void);
                       
                       standardization_struct<double> standardization_scaler;
                       min_max_struct<double> min_max_scaler;
                       mean_norm_struct<double> mean_norm_scaler;
                       
                       template <typename T>
                       bool Normalization(vector<T> &v);
                       template <typename T>
                       bool Normalization(matrix<T> &matrix_);
                       
                       template <typename T>
                       bool ReverseNormalization(vector<T> &v);
                       template <typename T>
                       bool ReverseNormalization(matrix<T> &matrix_);
  };
//+------------------------------------------------------------------+
//| For normalizing and reverse normalizing the given x-matrix<T>       |
//| This constructor obtains crucial information such as mean, min   |
//| max and Std deviation from the dataset, this information is used |
//| during reverse normalization for turning the data back to its    |
//| original state                                                   |
//+------------------------------------------------------------------+
template <typename T>
CPreprocessing::CPreprocessing(matrix<T> &matrix_, norm_technique NORM_MODE)
 {    
   m_cols = matrix_.Cols();
   
   norm_method = NORM_MODE;
   
   vector<T> v = {}; 
   
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
             
         break;       
    }
   
      
   Normalization(matrix_);
 }
//+------------------------------------------------------------------+
//|   In case the Normalization techniques and normalization         |
//| information are known from pre-trained model or class instance   |
//  the following classes may be appropriate to use instead          |            
//+------------------------------------------------------------------+
template <typename T>
CPreprocessing::CPreprocessing(vector<T> &stdn_mean, vector<T> &stdn_std, norm_technique NORM_MODE)
 {
   this.norm_method = NORM_STANDARDIZATION;
   this.m_cols = stdn_mean.Size();
   
   standardization_scaler.mean = stdn_mean;
   standardization_scaler.std = stdn_std;
 }
template <typename T>
CPreprocessing::CPreprocessing(vector<T> &min_max_max, vector<T> &min_max_min)
 {
   this.norm_method =  NORM_MIN_MAX_SCALER;
   this.m_cols = min_max_max.Size();
      
   min_max_scaler.max = min_max_max;
   min_max_scaler.min = min_max_min;
 }

template <typename T>
CPreprocessing::CPreprocessing(vector<T> &mean_norm_max, vector<T> &mean_norm_mean, vector<T> &mean_norm_min)
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
 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
template <typename T>
bool CPreprocessing::Standardization(vector<T> &v)
 {
   if (v.Size() != m_cols)
     {
       Print(__FUNCTION__," Can't Normalize the data | Vector v sized ",v.Size()," needs to have the same size as the Columns ",m_cols," of the Matrix given");
       return false;
     }
     
   for (ulong i=0; i<m_cols; i++)
      v[i] = (v[i] - standardization_scaler.mean[i]) / (standardization_scaler.std[i] + 1e-10);  
    
   return true;  
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
template <typename T>
bool CPreprocessing::Standardization(matrix<T> &matrix_)
 {
  vector<T> v;
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
template <typename T>
bool CPreprocessing::ReverseStandardization(vector<T> &v)
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
template <typename T>
bool CPreprocessing::ReverseStandardization(matrix<T> &matrix_)
 {
 bool norm =true;
 
  for (ulong i=0; i<matrix_.Rows(); i++)
    { 
      vector<T> v = matrix_.Row(i);
      
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
template <typename T>
bool CPreprocessing::Normalization(vector<T> &v)
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
         
      case NORM_NONE:
             
         break;     
     }
   return norm;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
template <typename T>
bool CPreprocessing::Normalization(matrix<T> &matrix_)
 {
   vector<T> v;
   
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
        
      case NORM_NONE:
             
         break;     
     }
     
  return norm;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
template <typename T>
bool CPreprocessing::ReverseNormalization(vector<T> &v)
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
         
      case NORM_NONE:
             
         break;     
     }
   return norm;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
template <typename T>
bool CPreprocessing::ReverseNormalization(matrix<T> &matrix_)
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
        
      case NORM_NONE:
             
         break;     
    }
  return norm;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
template <typename T>
bool CPreprocessing::MinMaxScaler(vector<T> &v)
 {
   if (v.Size() != m_cols)
     {
       Print(__FUNCTION__," Can't Normalize the data | Vector v sized ",v.Size()," needs to have the same size as the Columns ",m_cols," of the Matrix given");
       return false;
     }
     
   for (ulong i=0; i<m_cols; i++)
     v[i] = (v[i] - min_max_scaler.min[i]) / ((min_max_scaler.max[i] - min_max_scaler.min[i]) + 1e-10);  
     
    return true;
 } 

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+ 
template <typename T>
bool CPreprocessing::MinMaxScaler(matrix<T> &matrix_)
 {
   vector<T> v = {}; 
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
template <typename T>
bool CPreprocessing::ReverseMinMaxScaler(matrix<T> &matrix_)
 {
 bool norm =true;
 
    for (ulong i=0; i<matrix_.Rows(); i++)
       {
         vector<T> v = matrix_.Row(i);
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
template <typename T>
bool CPreprocessing::ReverseMinMaxScaler(vector<T> &v)
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
template <typename T>
bool CPreprocessing::MeanNormalization(vector<T> &v)
 {
   if (v.Size() != m_cols)
     {
       Print(__FUNCTION__," Can't Normalize the data | Vector v sized ",v.Size()," needs to have the same size as the Columns ",m_cols," of the Matrix given");
       return false;
     }
     
   for (ulong i=0; i<m_cols; i++)
      v[i] = (v[i] - mean_norm_scaler.mean[i]) / ((mean_norm_scaler.max[i] - mean_norm_scaler.min[i]) + 1e-10);
    
   return true;
 }
 
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
template <typename T>
bool CPreprocessing::MeanNormalization(matrix<T> &matrix_)
 {
   vector<T> v = {};  
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
template <typename T>
bool CPreprocessing::ReverseMeanNormalization(vector<T> &v)
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
template <typename T>
bool CPreprocessing::ReverseMeanNormalization(matrix<T> &matrix_)
 {
  bool norm =true;
  
    for (ulong i=0; i<matrix_.Rows(); i++)
       {
         vector<T> v = matrix_.Row(i);
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
//|                                                                  |
//|            LabelEncoder class                                    |
//|                                                                  |
//|                                                                  |
//|                                                                  |
//+------------------------------------------------------------------+
*/

struct CLabelEncoder
  {
      private:
        int dummy;
        
         void Unique(const string &Array[], string &classes_arr[]) //From matrix<T> utils
          {
            string temp_arr[];
         
            ArrayResize(classes_arr,1);
            ArrayCopy(temp_arr,Array);
            
            classes_arr[0] = Array[0];
            
            for(int i=0, count =1; i<ArraySize(Array); i++)  //counting the different neighbors
              {
               for(int j=0; j<ArraySize(Array); j++)
                 {
                  if(Array[i] == temp_arr[j] && temp_arr[j] != "-nan")
                    {
                     bool count_ready = false;
         
                     for(int n=0; n<ArraySize(classes_arr); n++)
                        if(Array[i] == classes_arr[n])
                             count_ready = true;
         
                     if(!count_ready)
                       {
                        count++;
                        ArrayResize(classes_arr,count);
         
                        classes_arr[count-1] = Array[i]; 
         
                        temp_arr[j] = "-nan"; //modify so that it can no more be counted
                       }
                     else
                        break;
                     //Print("t vectors vector<T> ",v);
                    }
                  else
                     continue;
                 }
              }
          }
         //--- Sort the array based on the bubble algorithm
         
         bool BubbleSortStrings(string &arr[])
           {
            int arraySize = ArraySize(arr);
            
            if (arraySize == 0)
              {
               Print(__FUNCTION__," Failed to Sort | ArraySize = 0");
               return false;
              }
            
            for(int i = 0; i < arraySize - 1; i++)
              {
               for(int j = 0; j < arraySize - i - 1; j++)
                 {
                  if(StringCompare(arr[j], arr[j + 1], false) > 0)
                    {
                     // Swap arr[j] and arr[j + 1]
                     string temp = arr[j];
                     arr[j] = arr[j + 1];
                     arr[j + 1] = temp;
                    }
                 }
              }
             return true;
           }
       
      public:         
         vector encode(string &Arr[])
           {
            string unique_values[];
            Unique(Arr, unique_values);
            
            vector ret(ArraySize(Arr));
                                    
            if (!BubbleSortStrings(unique_values))
                return ret;
             
             for (int i=0; i<ArraySize(unique_values); i++)
                for (int j=0; j<ArraySize(Arr); j++)
                   if (unique_values[i] == Arr[j])
                     ret[j] = i+1;
                 
             return ret;
           }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
