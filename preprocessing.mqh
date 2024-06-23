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
   bool loaded_scaler;
   
public:
                     StandardizationScaler(void);
                     StandardizationScaler(const double &mean[], const double &std[]); //For Loading the pre-fitted scaler 
                    ~StandardizationScaler(void);
                    
                    virtual matrix fit_transform(const matrix &X);
                    virtual matrix transform(const matrix &X);
                    virtual vector transform(const vector &X);
                    
                    virtual bool   save(string save_dir);
                    
                    
                    virtual matrix inverse_transform(const matrix &X_scaled);
                    virtual vector inverse_transform(const vector &X_scaled);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
StandardizationScaler::StandardizationScaler(void)
 {
   loaded_scaler = false;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
StandardizationScaler::StandardizationScaler(const double &mean_[],const double &std_[])
 {
   this.mean = MatrixExtend::ArrayToVector(mean_);
   this.std = MatrixExtend::ArrayToVector(std_);
   
   loaded_scaler = true;
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
  
  if (loaded_scaler)
    {
      printf("% This is a loaded scaler | no need to fit to the new data, call another instance of a class",__FUNCTION__);
      return X;
    }
  
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
vector StandardizationScaler::inverse_transform(const vector &X_scaled)
 {
    vector X(X_scaled.Size());

    if (this.mean.Size() == 0 || this.std.Size() == 0) {
        printf("%s Call the fit_transform function first to fit the scaler or\n Load the pre-fitted scaler before attempting to transform the new data", __FUNCTION__);
        return X;
    }

    if (X_scaled.Size() != this.mean.Size()) {
        printf("%s Dimension mismatch between trained data sized=(%d) and the new data sized=(%d)", __FUNCTION__, this.mean.Size(), X_scaled.Size());
        return X;
    }

    for (ulong i = 0; i < X.Size(); i++) {
        X[i] = X_scaled[i] * (this.std[i] + 1e-10) + this.mean[i];
    }

    return X;
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

matrix StandardizationScaler::inverse_transform(const matrix &X_scaled)
 {
   matrix X = X_scaled;
   
   for (ulong i=0; i<X.Rows(); i++)
     X.Row(this.inverse_transform(X_scaled.Row(i)), i);
   
   return X;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector StandardizationScaler::transform(const vector &X)
 {
   vector v(X.Size());
   if (this.mean.Size()==0 || this.std.Size()==0)
     {
       printf("%s Call the fit_transform function first to fit the scaler or\n Load the pre-fitted scaler before attempting to transform the new data",__FUNCTION__);
       return v;
     }
   
   if (X.Size() != this.mean.Size())
     {
         printf("%s Dimension mismatch between trained data sized=(%d) and the new data sized=(%d)",__FUNCTION__,this.mean.Size(),X.Size());
         return v;
     }
   
   for (ulong i=0; i<v.Size(); i++)
      v[i] = (X[i] - this.mean[i]) / (this.std[i] + 1e-10);  
   
   return v;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool StandardizationScaler::save(string save_dir)
 {
//---save mean

   if (!MatrixExtend::write_bin(this.mean, save_dir+"\\mean.bin"))
     {
       printf("%s Failed Save the mean values of the Scaler",__FUNCTION__);
       return false;
     }
   
//--- save std

   if (!MatrixExtend::write_bin(this.std, save_dir+"\\std.bin"))
     {
       printf("%s Failed Save the Standard deviation values of the Scaler",__FUNCTION__);
       return false;
     }
     
   return true;
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
   bool loaded_scaler;
   
public:
                     MinMaxScaler(void);
                     MinMaxScaler(const double &min_[], const double &max_[]); //For Loading the pre-fitted scaler 
                     
                    ~MinMaxScaler(void);
                    
                    virtual matrix fit_transform(const matrix &X);
                    virtual matrix transform(const matrix &X);
                    virtual vector transform(const vector &X);
                    
                    virtual bool   save(string dir);
                                        
                    virtual matrix inverse_transform(const matrix &X_scaled);
                    virtual vector inverse_transform(const vector &X_scaled);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
MinMaxScaler::MinMaxScaler(void)
 {
   loaded_scaler =false;
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
MinMaxScaler::MinMaxScaler(const double &min_[],const double &max_[])
 {
   this.min = MatrixExtend::ArrayToVector(min_);
   this.max = MatrixExtend::ArrayToVector(max_);
   
   loaded_scaler = true;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
matrix MinMaxScaler::fit_transform(const matrix &X)
 {
  if (loaded_scaler)
    {
      printf("% This is a loaded scaler | no need to fit to the new data, call another instance of a class",__FUNCTION__);
      return X;
    }

//---

  this.min.Resize(X.Cols());
  this.max.Resize(X.Cols());
  
    for (ulong i=0; i<X.Cols(); i++)
      { 
         this.min[i] = X.Col(i).Min();
         this.max[i] = X.Col(i).Max();
      }
   
   if (MQLInfoInteger(MQL_DEBUG))
     Print("Min: ",this.min,"\nMax: ",this.max);
   
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
       printf("%s Call the fit_transform function fist to fit the scaler or\n the load function to load the pre-fitted scalerbefore attempting to transform the new data",__FUNCTION__);
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
matrix MinMaxScaler::inverse_transform(const matrix &X_scaled)
 {
   matrix X = X_scaled;
   
   for (ulong i=0; i<X.Rows(); i++)
     X.Row(this.inverse_transform(X_scaled.Row(i)), i);
   
   return X;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector MinMaxScaler::inverse_transform(const vector &X_scaled)
 {
   vector v(X_scaled.Size());
   if (this.min.Size()==0 || this.max.Size()==0)
     {
       printf("%s Call the fit_transform function fist to fit the scaler or\n the load function to load the pre-fitted scalerbefore attempting to transform the new data",__FUNCTION__);
       return v;
     }
   
   if (X_scaled.Size() != this.min.Size())
     {
         printf("%s X of size [%d] doesn't match the same number of features in a given X matrix on the fit_transform function call",__FUNCTION__,this.min.Size());
         return v;
     }

//--- Perform inverse transformation

    for (ulong i = 0; i < X_scaled.Size(); ++i) 
        v[i] = X_scaled[i] * (max[i] - min[i]) + min[i];

    return v;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool MinMaxScaler::save(string save_dir)
 {
//---save min

   if (!MatrixExtend::write_bin(this.min, save_dir+"\\min.bin"))
     {
       printf("%s Failed to save the Min values for the scaler",__FUNCTION__);
       return false;
     }
   
//--- save max
   
   if (!MatrixExtend::write_bin(this.max, save_dir+"\\max.bin"))
     {
       printf("%s Failed to save the Max values for the scaler",__FUNCTION__);
       return false;
     }
   
   return true;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+


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
   vector median, quantile;
   bool loaded_scaler;
   
public:
                     RobustScaler(void);
                     RobustScaler(const double &median_[], const double &quantile_[]);
                    ~RobustScaler(void);
                    
                    virtual matrix fit_transform(const matrix &X);
                    virtual matrix transform(const matrix &X);
                    virtual vector transform(const vector &X);
                    
                    virtual bool   save(string save_dir);
                    
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
RobustScaler::RobustScaler(void)
 {
   loaded_scaler = false;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
RobustScaler::RobustScaler(const double &median_[],const double &quantile_[])
 {
   this.median = MatrixExtend::ArrayToVector(median_);
   this.quantile = MatrixExtend::ArrayToVector(quantile_);
   
   loaded_scaler = true;
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
  if (loaded_scaler)
    {
      printf("% This is a loaded scaler | no need to fit to the new data, call another instance of a class",__FUNCTION__);
      return X;
    }

//---

  this.median.Resize(X.Cols());
  this.quantile.Resize(X.Cols());
  
    for (ulong i=0; i<X.Cols(); i++)
     {
       this.median[i] = X.Col(i).Median();
       this.quantile[i] = MathAbs(X.Col(i) - this.median[i]).Median() * 1.4826;  // 1.4826 is a constant for consistency;
     }
     
   if (MQLInfoInteger(MQL_DEBUG))
     Print("Median: ",this.median,"\nQuantile: ",this.quantile);
   
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
       printf("%s Call the fit_transform function fist to fit the scaler or\n the load function to load the pre-fitted scalerbefore attempting to transform the new data",__FUNCTION__);
       return v;
     }
   
   if (X.Size() != this.median.Size())
     {
         printf("%s X of size [%d] doesn't match the same number of features in a given X matrix on the fit_transform function call",__FUNCTION__,this.median.Size());
         return v;
     }
     
    for (ulong i=0; i<X.Size(); i++)
      v[i] = (X[i] - this.median[i]) / (quantile[i] + 1e-10); 
    
    return v;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool RobustScaler::save(string save_dir)
 {
//--- save median
   
   if (!MatrixExtend::write_bin(this.median, save_dir+"\\median.bin"))
     {
       printf("%s Failed to save the Median values for the scaler",__FUNCTION__);
       return false;
     }

//--- save quantile

   if (!MatrixExtend::write_bin(this.quantile, save_dir+"\\quantile.bin"))
     {
       printf("%s Failed to save the Quantile values for the scaler",__FUNCTION__);
       return false;
     }
   
   return true;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
