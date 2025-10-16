//+------------------------------------------------------------------+
//|                                                preprocessing.mqh |
//|                                    Copyright 2022, Fxalgebra.com |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, Fxalgebra.com"
#property link      "https://www.mql5.com/en/users/omegajoctan"

//+------------------------------------------------------------------+
//|               Strings label encoder                              |
//+------------------------------------------------------------------+

class CLabelEncoder
{
   private:
       int m_mapping[];
       
       // Helper function to find index of a string in an array
       int FindStringIndex(const string &array[], const string value)
       {
           for(int i = 0; i < ArraySize(array); i++)
           {
               if(array[i] == value)
                   return i;
           }
           return -1;
       }
       
       // Extract unique values and sort them
       bool GetUniqueSortedClasses(const string &input_[], string &output[])
       {
           // Temporary array to mark duplicates
           string temp[];
           ArrayResize(temp, ArraySize(input_));
           ArrayCopy(temp, input_);
           
           int count = 0;
           
           for(int i = 0; i < ArraySize(temp); i++)
           {
               if(temp[i] == "") continue; // Skip already processed
               
               // Add to output
               ArrayResize(output, count + 1);
               output[count] = temp[i];
               count++;
               
               // Mark all duplicates
               for(int j = i + 1; j < ArraySize(temp); j++)
               {
                   if(temp[j] == temp[i])
                       temp[j] = ""; // Mark as processed
               }
           }
           
           // Sort the unique values
           return BubbleSortStrings(output);
       }
       
       // Bubble sort for strings (same as your original)
       bool BubbleSortStrings(string &arr[])
       {
           int arraySize = ArraySize(arr);
           
           if(arraySize == 0)
           {
               Print(__FUNCTION__, " Failed to Sort | ArraySize = 0");
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
       
       string m_classes[];
       
       CLabelEncoder(void)
        {
        
        }
       
       ~CLabelEncoder(void)
        {
        
        }
        
       bool fit(const string &y[]) // Fit the encoder to the data
       {
           if(ArraySize(y) == 0)
               return false;
               
           // Get unique sorted classes
           if(!GetUniqueSortedClasses(y, m_classes))
               return false;
               
           // Create mapping (not strictly needed but makes transform faster)
           ArrayResize(m_mapping, ArraySize(m_classes));
           for(int i = 0; i < ArraySize(m_classes); i++)
               m_mapping[i] = i;
               
           return true;
       }
       
           
       // Transform a single label to encoded integer
       int transform(const string value)
       {
           if(ArraySize(m_classes) == 0)
           {
               Print("%s error, Encoder not fitted yet", __FUNCTION__);
               return -1;
           }
           
           int idx = FindStringIndex(m_classes, value);
           if(idx == -1)
           {
               Print("Warning: Unknown label '", value, "' found in transform");
               return -1;
           }
           
           return m_mapping[idx];
       }
       
       // Transform labels to encoded integers
       vector transform(const string &y[])
       {
           vector ret(ArraySize(y));
           
           if(ArraySize(m_classes) == 0)
           {
               Print("%s error, Encoder not fitted yet",__FUNCTION__);
               return vector::Zeros(0);
           }
           
           for(int i = 0; i < ArraySize(y); i++)
             ret[i] = transform(y[i]);
           
           return ret;
       }
       
       // Fit and transform in one step
       vector fit_transform(const string &y[])
       {
           if(!fit(y))
           {
               printf("%s failed to fit the transformer",__FUNCTION__);
               return vector::Zeros(0);
           }
           return transform(y);
       }
       
       // Transform encoded integers back to original labels
       string inverse_transform(const int encoded_value)
       {
           if(ArraySize(m_classes) == 0)
           {
               Print("%s error, Encoder not fitted yet",__FUNCTION__);
               return NULL;
           }
           
           if(encoded_value < 0 || encoded_value >= ArraySize(m_classes))
           {
               printf("%s error, encoded value %d out of range",__FUNCTION__,encoded_value);
               return NULL;
           }
           
           return m_classes[encoded_value];
       }
};
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
template<typename T>
vector ArrayToVector(const T &Arr[])
  {
   vector v(ArraySize(Arr));
   
   for (int i=0; i<ArraySize(Arr); i++)
     v[i] = double(Arr[i]);
     
   return (v);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
template<typename T>
bool VectorToArray(const vector<T> &v, T &arr[])
  {
   vector temp = v;
   if (!temp.Swap(arr))
    {
      Print("Failed to Convert vector to Array Err=",GetLastError());
      return false;
    }
   return(true);
  }

bool write_bin(vector &v,string file)
 {
   FileDelete(file);
   int handle = FileOpen(file,FILE_READ|FILE_WRITE|FILE_BIN,",");
   if (handle == INVALID_HANDLE)
    {
      printf("Invalid handle Err=%d",GetLastError());
      DebugBreak();
      return false;
    }
   
   double arr[];
   ArrayResize(arr, (int)v.Size());
   
   for (uint i=0; i<arr.Size(); i++)
    arr[i] = v[i];
   
   FileWriteArray(handle, arr);
   FileClose(handle);
  
  return true;
 }
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
   this.mean = ArrayToVector(mean_);
   this.std = ArrayToVector(std_);
   
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

   if (!write_bin(this.mean, save_dir+"\\mean.bin"))
     {
       printf("%s Failed Save the mean values of the Scaler",__FUNCTION__);
       return false;
     }
   
//--- save std

   if (!write_bin(this.std, save_dir+"\\std.bin"))
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
   this.min = ArrayToVector(min_);
   this.max = ArrayToVector(max_);
   
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

   if (!write_bin(this.min, save_dir+"\\min.bin"))
     {
       printf("%s Failed to save the Min values for the scaler",__FUNCTION__);
       return false;
     }
   
//--- save max
   
   if (!write_bin(this.max, save_dir+"\\max.bin"))
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
   this.median = ArrayToVector(median_);
   this.quantile = ArrayToVector(quantile_);
   
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
   
   if (!write_bin(this.median, save_dir+"\\median.bin"))
     {
       printf("%s Failed to save the Median values for the scaler",__FUNCTION__);
       return false;
     }

//--- save quantile

   if (!write_bin(this.quantile, save_dir+"\\quantile.bin"))
     {
       printf("%s Failed to save the Quantile values for the scaler",__FUNCTION__);
       return false;
     }
   
   return true;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
