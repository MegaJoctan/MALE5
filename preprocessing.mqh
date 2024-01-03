//+------------------------------------------------------------------+
//|                                                preprocessing.mqh |
//|                                    Copyright 2022, Fxalgebra.com |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, Fxalgebra.com"
#property link      "https://www.mql5.com/en/users/omegajoctan"

//+------------------------------------------------------------------+
//|   LABEL ENCODE CLASS FOR PREPROCESSING INFORMATION               |
//+------------------------------------------------------------------+

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

#include "MatrixExtend.mqh";
 

//+------------------------------------------------------------------+
//|                                                                  |
//|                                                                  |
//|               Standardization Scaler                             |
//|                                                                  |
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
                    bool save(string save_dir, string column_names, bool common_dir=false);
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
  this.mean.Resize(X.Cols());
  this.std.Resize(X.Cols());
  
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
   
   for (ulong i=0; i<X.Rows(); i++)
     X_norm.Row(this.transform(X.Row(i)), i);
   
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
bool StandardizationScaler::save(string save_dir, string column_names, bool common_dir=false)
 {
//---save mean

   matrix m = MatrixExtend::VectorToMatrix(this.mean, this.mean.Size());
   
   return MatrixExtend::WriteCsv(save_dir+"StandardizationScaler-Mean.csv", m, column_names, common_dir,8);
   
//--- save std

   m = MatrixExtend::VectorToMatrix(this.std, this.mean.Size());
   
   return MatrixExtend::WriteCsv(save_dir+"StandardizationScaler-Std.csv", m, column_names, common_dir,8);
 }
 
 
//+------------------------------------------------------------------+
//|                                                                  |
//|                                                                  |
//|                  Min-Max Scaler                                  |
//|                                                                  |
//|                                                                  |
//+------------------------------------------------------------------+

class MinMaxScaler
  {
protected:
   vector min, max;
   
public:
                     MinMaxScaler(void);
                    ~MinMaxScaler(void);
                    
                    matrix fit_transform(const matrix &X);
                    matrix transform(const matrix &X);
                    vector transform(const vector &X);
                    bool save(string save_dir, string column_names, bool common_dir=false);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
MinMaxScaler::MinMaxScaler(void)
 {
 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
MinMaxScaler::~MinMaxScaler(void)
 {
 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
matrix MinMaxScaler::fit_transform(const matrix &X)
 {
 
  this.min.Resize(X.Cols());
  this.max.Resize(X.Cols());
  
    for (ulong i=0; i<X.Cols(); i++)
      { 
         this.min[i] = X.Col(i).Min();
         this.max[i] = X.Col(i).Max();
      }
   
   if (MQLInfoInteger(MQL_DEBUG))
     Print("Min: ",this.min," Max: ",this.max);
   
//---
   return this.transform(X);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector MinMaxScaler::transform(const vector &X)
 {
   vector v(X.Size());
   if (this.min.Size()==0 || this.max.Size()==0)
     {
       printf("%s Call the fit_transform function fist before attempting to transform the new data",__FUNCTION__);
       return v;
     }
   
   if (X.Size() != this.min.Size())
     {
         printf("%s X of size [%d] doesn't match the same number of features in a given X matrix on the fit_transform function call",__FUNCTION__,this.min.Size());
         return v;
     }
     
   for (ulong i=0; i<X.Size(); i++)
      v[i] = (X[i] - this.min[i]) / ((this.max[i] - this.min[i]) + 1e-10);  
   
   return v;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
matrix MinMaxScaler::transform(const matrix &X)
 {
   matrix X_norm = X;
   
   for (ulong i=0; i<X.Rows(); i++)
     X_norm.Row(this.transform(X.Row(i)), i);
   
   return X_norm;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool MinMaxScaler::save(string save_dir,string column_names,bool common_dir=false)
 {
//---save min

   matrix m = MatrixExtend::VectorToMatrix(this.min, this.min.Size());
   
   return MatrixExtend::WriteCsv(save_dir+"MinMaxScaler-Min.csv", m, column_names, common_dir,8);
   
//--- save max

   m = MatrixExtend::VectorToMatrix(this.max, this.max.Size());
   
   return MatrixExtend::WriteCsv(save_dir+"MinMaxScaler-Max.csv", m, column_names, common_dir,8);
 }
 
//+------------------------------------------------------------------+
//|                                                                  |
//|                                                                  |
//|               Mean Normalization Scaler                          |
//|                                                                  |
//|                                                                  |
//+------------------------------------------------------------------+

class RobustScaler
  {
protected:
   vector median, quantile_range;
public:
                     RobustScaler(void);
                    ~RobustScaler(void);
                    
                    matrix fit_transform(const matrix &X);
                    matrix transform(const matrix &X);
                    vector transform(const vector &X);
                    bool save(string save_dir, string column_names, bool common_dir=false);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
RobustScaler::RobustScaler(void)
 {
 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
RobustScaler::~RobustScaler(void)
 {
 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
matrix RobustScaler::fit_transform(const matrix &X)
 {
  this.median.Resize(X.Cols());
  this.quantile_range.Resize(X.Cols());
  
    for (ulong i=0; i<X.Cols(); i++)
     {
       this.median[i] = X.Col(i).Median();
       this.quantile_range[i] = X.Col(i).Quantile(1);
     }
     
   if (MQLInfoInteger(MQL_DEBUG))
     Print("Median: ",this.median," Quantile: ",this.quantile_range);
   
   
//---
   return this.transform(X);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
matrix RobustScaler::transform(const matrix &X)
 {
   matrix X_norm = X;
   
   for (ulong i=0; i<X.Rows(); i++)
     X_norm.Row(this.transform(X.Row(i)), i);
   
   return X_norm;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector RobustScaler::transform(const vector &X)
 {
   vector v(X.Size());
   if (this.median.Size()==0)
     {
       printf("%s Call the fit_transform function fist before attempting to transform the new data",__FUNCTION__);
       return v;
     }
   
   if (X.Size() != this.median.Size())
     {
         printf("%s X of size [%d] doesn't match the same number of features in a given X matrix on the fit_transform function call",__FUNCTION__,this.median.Size());
         return v;
     }
     
    for (ulong i=0; i<X.Size(); i++)
      v[i] = (X[i] - this.median[i]) / (quantile_range[i] + 1e-10); 
    
    return v;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool RobustScaler::save(string save_dir,string column_names,bool common_dir=false)
 {
//--- save median

   matrix m = MatrixExtend::VectorToMatrix(this.median, this.median.Size());
   
   return MatrixExtend::WriteCsv(save_dir+"RobustScaler-Median.csv", m, column_names, common_dir,8);

//--- save quantile

   m = MatrixExtend::VectorToMatrix(this.quantile_range, this.quantile_range.Size());
   
   return MatrixExtend::WriteCsv(save_dir+"RobustScaler-Median.csv", m, column_names, common_dir,8);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
