//+------------------------------------------------------------------+
//|                                                       pandas.mqh |
//|                                          Copyright 2023, Omegafx |
//|                                           https://www.omegafx.co |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, Omegafx"
#property link      "https://www.omegafx.co"
#property strict
//+------------------------------------------------------------------+
//| defines                                                          |
//+------------------------------------------------------------------+

#ifndef NAN
#define  NAN double("nan")
#endif

#ifndef INF
#define  INF double("inf")
#endif

#ifndef NEG_INF
#define  NEG_INF double("-inf")
#endif

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
             ret[i] = (int)transform(y[i]);
           
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

struct rolling_struct
  {
protected:
   double            CalculateSkewness(const vector &v)
     {
      int n = (int)v.Size();
      if(n < 3)
         return 0; // Skewness is undefined for less than 3 data points

      double mean = v.Mean();
      double std_dev = v.Std();
      double skewness = 0;

      for(int i = 0; i < n; i++)
        {
         double diff = v[i] - mean;
         skewness += MathPow(diff / (std_dev+DBL_EPSILON), 3);
        }

      skewness *= (double)n / (((n - 1) * (n - 2))+DBL_EPSILON);
      return skewness;
     }

   double            CalculateKurtosis(const vector &v)
     {
      int n = (int)v.Size();
      if(n < 4)
         return 0; // Kurtosis is undefined for less than 4 data points

      double mean = v.Mean();
      double std_dev = v.Std();
      double kurtosis = 0;

      for(int i = 0; i < n; i++)
        {
         double diff = v[i] - mean;
         kurtosis += MathPow(diff / (std_dev+DBL_EPSILON), 4);
        }

      kurtosis = kurtosis * (n * (n + 1)) / ((((n - 1) * (n - 2) * (n - 3)) - (3 * MathPow(n - 1, 2))) + DBL_EPSILON) / (((n - 2) * (n - 3)) + DBL_EPSILON);
      return kurtosis;
     }

public:
   matrix            matrix__;

   vector            mean()
     {
      vector res(matrix__.Rows());
      res.Fill(NAN);

      for(ulong i=0; i<res.Size(); i++)
         res[i] = matrix__.Row(i).Mean();

      return res;
     }

   vector            min()
     {
      vector res(matrix__.Rows());
      res.Fill(NAN);

      for(ulong i=0; i<res.Size(); i++)
         res[i] = matrix__.Row(i).Min();

      return res;
     }
     
   vector            max()
     {
      vector res(matrix__.Rows());
      res.Fill(NAN);

      for(ulong i=0; i<res.Size(); i++)
         res[i] = matrix__.Row(i).Max();

      return res;
     }
     
   vector            std()
     {
      vector res(matrix__.Rows());
      res.Fill(NAN);

      for(ulong i=0; i<res.Size(); i++)
         res[i] = matrix__.Row(i).Std();

      return res;
     }

   vector            var()
     {
      vector res(matrix__.Rows());
      res.Fill(NAN);

      for(ulong i=0; i<res.Size(); i++)
         res[i] = matrix__.Row(i).Var();

      return res;
     }

   vector            skew()
     {
      vector res(matrix__.Rows());
      res.Fill(NAN);

      for(ulong i=0; i<res.Size(); i++)
         res[i] = CalculateSkewness(matrix__.Row(i));

      return res;
     }

   vector            kurtosis()
     {
      vector res(matrix__.Rows());
      res.Fill(NAN);

      for(ulong i=0; i<res.Size(); i++)
         res[i] = CalculateKurtosis(matrix__.Row(i));

      return res;
     }

   vector            median()
     {
      vector res(matrix__.Rows());
      res.Fill(NAN);

      for(ulong i=0; i<res.Size(); i++)
         res[i] = matrix__.Row(i).Median();

      return res;
     }
     
   vector            percentile(int value)
     {
      vector res(matrix__.Rows());
      res.Fill(NAN);

      for(ulong i=0; i<res.Size(); i++)
         res[i] = matrix__.Row(i).Percentile(value);

      return res;
     }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CDataFrame
  {
   vector            GetColumn(const string name);

   int               ColNameToIndex(const string name, const string &column_names[])
     {
      int column_index = -1;
      for(uint i=0; i<column_names.Size(); i++)
         if(name == column_names[i])
           {
            column_index = (int)i;
            break;
           }
      return column_index;
     }

   bool              WriteCsv(string csv_name, bool common=false, int digits=5, bool verbosity=false);

   void              CopyFrom(const CDataFrame &other)
     {
      ArrayResize(m_columns, other.m_columns.Size()); //Explicitly set the size
      if(ArrayCopy(m_columns, other.m_columns)<0)
        {
         printf("%s Failed to copy the class, Error = %d",__FUNCTION__,GetLastError());
         return;
        }
      m_values = other.m_values; // Assuming matrix supports deep copy
     }

   int               CDataFrame::CountNaN(const vector &v);
   void              GetCol(const string &data[], string &output[], int col_index, int total_columns);
   template<typename T>
   vector            ArrayToVector(const T &Arr[])
     {
      vector v(ArraySize(Arr));
      
      for (int i=0; i<ArraySize(Arr); i++)
        v[i] = double(Arr[i]);
        
      return (v);
     }
     
public:

   string            m_columns[]; //An array of string values for keeping track of the column names
   matrix            m_values; // A 2D matrix
   CLabelEncoder     m_columns_encoders[];
   vector            shape()
     {
       vector s = {int(m_values.Rows()), int(m_values.Cols())};       return s;
     }

                     CDataFrame();
                     CDataFrame(const string columns, const matrix &values);
                     CDataFrame(const string &columns[], const matrix &values);
                    ~CDataFrame(void);

                     CDataFrame(const CDataFrame &other) { CopyFrom(other); }
   CDataFrame        operator=(const CDataFrame &other) { CopyFrom(other); return GetPointer(this); }

   //--- Data selection and Indexing

   vector            operator[](const string index) {return GetColumn(index); }  //Access a column by its name

   vector            loc(int index, uint axis = 0);
   CDataFrame        iloc(ulong start_row, ulong end_row, ulong start_col, ulong end_col);
   double            at(ulong row, string col_name);
   double            iat(ulong row, ulong col);

   //--- Data exploration methods

   matrix            tail(uint count=5);
   void              info();
   void              describe(void);
   
   //---


   CDataFrame        drop(const string cols);
   void              head(const uint count=5);
   bool              to_csv(const string file_name, const bool common_path=false, bool verbosity=false);

   bool              from_csv(string file_name,string delimiter=",",bool is_common=false, string datetime_columns="", string columns_to_encode="", bool verbosity=false);
   void              insert(string name, const vector &values);

   CDataFrame        dropna();
      
   //--- Timeseries transformation and manipulations
   
   vector            pct_change(const string index);
   vector            pct_change(const vector &v);

   rolling_struct    rolling(const vector &v, const uint window);
   rolling_struct    rolling(const string index, const uint window);
   
   vector            shift(const vector &v, const int shift_index);
   vector            shift(const string col_name, const int shift_index);
   
   vector            diff(const vector &v, int period=1);
   vector            diff(const string col_name, int period=1);

   // Define function pointer type for apply
   typedef double (*ApplyFunction)(double);
   
   // apply function to a specific column
   vector apply_axis1(const string column_name, ApplyFunction func)
    {
      vector col = GetColumn(column_name);
      vector result(col.Size());
      result.Fill(NAN);
   
      for (uint i = 0; i < col.Size(); i++)
         result[i] = func(col[i]);  // apply function
   
      return result;  // Return transformed column
    }
   
   // apply function to a specific column or all elements in the DataFrame
   CDataFrame apply_axis0(ApplyFunction func)
    {
      CDataFrame result = this; // Copy original DataFrame
      // apply function to ALL ELEMENTS (row-wise)
      
      int idx = 0;
      for (uint i = 0; i < m_values.Rows(); i++)
        for (uint j = 0; j < m_values.Cols(); j++)
           result.m_values[i][j] = func(result.m_values[i][j]); // apply function to each element

      return result; // Return transformed values
    }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CDataFrame::CDataFrame()
  {

  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CDataFrame::CDataFrame(const string columns, const matrix &values)
  {
   string columns_names[]; //A temporary array for obtaining column names from a string
   ushort sep = StringGetCharacter(",", 0);
   if(StringSplit(columns, sep, columns_names)<0)
     {
      printf("%s failed to obtain column names",__FUNCTION__);
      return;
     }

   if(columns_names.Size() != values.Cols())  //Check if the given number of column names is equal to the number of columns present in a given matrix
     {
      printf("%s dataframe's columns != columns present in the values matrix",__FUNCTION__);
      return;
     }

   ArrayCopy(m_columns, columns_names); //We assign the columns to the m_columns array
   m_values = values; //We assing the given matrix to the m_values matrix
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CDataFrame::CDataFrame(const string &columns[], const matrix &values)
  {
   if(columns.Size() != values.Cols())  //Check if the given number of column names is equal to the number of columns present in a given matrix
     {
      printf("%s dataframe's columns != columns present in the values matrix",__FUNCTION__);
      return;
     }

   ArrayCopy(m_columns, columns); //We assign the columns to the m_columns array
   m_values = values; //We assing the given matrix to the m_values matrix
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CDataFrame::pct_change(const string index)
  {
   vector col = GetColumn(index);
   return pct_change(col);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CDataFrame::pct_change(const vector &v)
  {
   vector col = v;
   ulong size = col.Size();

   vector results(size);
   results.Fill(NAN);

   for(ulong i=1; i<size; i++)
     {
      double prev_value = col[i - 1];
      double curr_value = col[i];

      // Calculate percentage change and handle division by zero
      if(prev_value != 0.0)
        {
         results[i] = ((curr_value - prev_value) / prev_value) * 100.0;
        }
      else
        {
         results[i] = 0.0; // Handle division by zero case
        }
     }

   return results;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CDataFrame::loc(int index, uint axis=0)
  {
   if(axis == 0)
     {
      vector row = {};

      //--- Convert negative index to positive

      if(index < 0)
         index = (int)m_values.Rows() + index;

      if(index < 0 || index >= (int)m_values.Rows())
        {
         printf("%s Error: Row index out of bounds. Given index: %d", __FUNCTION__, index);
         return row;
        }

      return m_values.Row(index);
     }
   else
      if(axis == 1)
        {
         vector column = {};

         //--- Convert negative index to positive

         if(index < 0)
            index = (int)m_values.Cols() + index;

         //--- Check bounds

         if(index < 0 || index >= (int)m_values.Cols())
           {
            printf("%s Error: Column index out of bounds. Given index: %d", __FUNCTION__, index);
            return column;
           }

         return m_values.Col(index);
        }
      else
         printf("%s Failed, Unknown axis ",__FUNCTION__);

   return vector::Zeros(0);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CDataFrame CDataFrame::iloc(ulong start_row,ulong end_row,ulong start_col,ulong end_col)
  {
   CDataFrame df(m_columns, m_values);

   if(start_row>end_row)
     {
      printf("%s Start row [%d] > end row [%d]",__FUNCTION__,start_row,end_row);
      return df;
     }

   if(start_col>end_col)
     {
      printf("%s start col [%d] > end col [%d]",__FUNCTION__,start_col,end_col);
      return df;
     }

   ulong rows = df.m_values.Rows(), cols = df.m_values.Cols();

   if(start_row>=rows || end_row>=rows)
     {
      printf("%s index out of range start row[%d] end row[%d] df rows[%d]",__FUNCTION__,start_row,end_row,rows);
      return df;
     }

   if(start_col>=cols || end_col>=cols)
     {
      printf("%s index out of range start col[%d] end col[%d] df cols[%d]",__FUNCTION__,start_col,end_col,cols);
      return df;
     }

//---

   df.m_values.Resize(end_row-start_row, end_col-start_col);
   df.m_values.Fill(NAN);

   if(ArrayResize(df.m_columns, int(end_col-start_col))<0)
     {
      printf("%s Failed to resize the columns array, error %d ",__FUNCTION__,GetLastError());
      return df;
     }

   for(ulong col=start_col, new_cols=0; col<end_col; col++, new_cols++)
      df.m_columns[col] = m_columns[new_cols];

//---

   for(ulong row=start_row, new_rows=0; row<end_row; row++, new_rows++)
      for(ulong col=start_col, new_cols=0; col<end_col; col++, new_cols++)
         df.m_values[new_rows][new_cols] = m_values[row][col];

   return df;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CDataFrame::at(ulong row, string col_name)
  {
   ulong col_number = (ulong)ColNameToIndex(col_name, m_columns);
   return m_values[row][col_number];
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CDataFrame::iat(ulong row,ulong col)
  {
   return m_values[row][col];
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CDataFrame::~CDataFrame(void)
  {

  }
//+------------------------------------------------------------------+
//|                                                                  |
//|   It obtains a column using it's name.                           |
//|   It returns a vector of values                                  |
//|                                                                  |
//+------------------------------------------------------------------+
vector CDataFrame::GetColumn(const string name)
  {
   if(m_columns.Size()==0)
     {
      printf("%s Can not get a column in an empty dataframe, add some data to it first",__FUNCTION__);
      return vector::Zeros(0);
     }

   int column_index = ColNameToIndex(name, m_columns);

   if(column_index == -1)
     {
      printf("%s Column '%s' not found in this DataFrame",__FUNCTION__,name);
      return  vector::Zeros(0);
     }

   return m_values.Col(column_index);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CDataFrame CDataFrame::drop(const string cols)
  {
   CDataFrame df;
   
   string column_names[];
   ushort sep = StringGetCharacter(",",0);
   if(StringSplit(cols, sep, column_names) < 0)
     {
      printf("%s Failed to get the columns, ensure they are separated by a comma. Error = %d", __FUNCTION__, GetLastError());
      return df;
     }
   
   int columns_index[];
   uint size = column_names.Size();
   ArrayResize(columns_index, size);

   if(size > m_values.Cols())
     {
      printf("%s failed, The number of columns > columns present in the dataframe", __FUNCTION__);
      return df;
     }

// Fill columns_index with column indices to drop
   for(uint i = 0; i < size; i++)
     {
      columns_index[i] = ColNameToIndex(column_names[i], m_columns);
      if(columns_index[i] == -1)
        {
         printf("%s Column '%s' not found in this DataFrame", __FUNCTION__, column_names[i]);
         //ArrayRemove(column_names, i, 1);
         continue;
        }
     }

   matrix new_data(m_values.Rows(), m_values.Cols() - size);
   string new_columns[];
   ArrayResize(new_columns, (int)m_values.Cols() - size);

// Populate new_data with columns not in columns_index
   for(uint i = 0, count = 0; i < m_values.Cols(); i++)
     {
      bool to_drop = false;
      for(uint j = 0; j < size; j++)
        {
         if(i == columns_index[j])
           {
            to_drop = true;
            break;
           }
        }

      if(!to_drop)
        {
         new_data.Col(m_values.Col(i), count);
         new_columns[count] = m_columns[i];
         count++;
        }
     }

// Replace original data with the updated matrix and columns
   
   df.m_values = new_data;
   ArrayResize(df.m_columns, new_columns.Size());
   ArrayCopy(df.m_columns, new_columns);
   
   return df;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

void CDataFrame::head(const uint count = 5)
{
   uint num_cols = (uint)m_columns.Size();
   uint num_rows = (uint)m_values.Rows();
   
   // Handle case where count is greater than available rows
   uint display_count = (num_rows <= 2 * count) ? num_rows : count;

   uint col_widths[];
   ArrayResize(col_widths, num_cols);

   // Determine max width for each column
   for (uint col = 0; col < num_cols; col++)
   {
      uint max_width = StringLen(m_columns[col]);
      for (uint row = 0; row < display_count && row < num_rows; row++)
      {
         string num_str = StringFormat("%.8f", m_values[row][col]);
         max_width = MathMax(max_width, StringLen(num_str));
      }
      
      col_widths[col] = max_width + 4; // Extra padding
   }

   // Print column headers with an empty index column
   string header = "| Index |";
   for (uint col = 0; col < num_cols; col++)
   {
      header += StringFormat(" %-*s |", col_widths[col], m_columns[col]);
   }
   
   Print(header);

   // Print first `count` rows
   for (uint row = 0; row < display_count; row++)
   {
      string row_str = StringFormat("| %5d |", row); // Index column
      for (uint col = 0; col < num_cols; col++)
      {
         row_str += StringFormat(" %-*.*f |", col_widths[col], 8, m_values[row][col]);
      }
      Print(row_str);
   }

   // Print separator if skipping rows in the middle
   if (num_rows > 2 * count)
   {
      Print("|  ...  |");
   }

   // Print last `count` rows if necessary
   if (num_rows > 2 * count)
   {
      for (uint row = num_rows - display_count; row < num_rows; row++)
      {
         string row_str = StringFormat("| %5d |", row); // Index column
         for (uint col = 0; col < num_cols; col++)
         {
            row_str += StringFormat(" %-*.*f |", col_widths[col], 8, m_values[row][col]);
         }
         Print(row_str);
      }
   }

   // Print DataFrame dimensions at the end
   printf("(%dx%d)", num_rows, num_cols-1);
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CDataFrame::to_csv(string csv_name, bool common=false, bool verbosity=false)
  {
   FileDelete(csv_name);
   int handle = FileOpen(csv_name,FILE_WRITE|FILE_SHARE_WRITE|FILE_CSV|FILE_ANSI|(common?FILE_COMMON:FILE_ANSI),",",CP_UTF8); //open a csv file

   if(handle == INVALID_HANDLE) //Check if the handle is OK
     {
      printf("Invalid %s handle Error %d ",csv_name,GetLastError());
      return (false);
     }

//---

   string concstring;
   vector row = {};
   vector colsinrows = m_values.Row(0);

   if(ArraySize(m_columns) != (int)colsinrows.Size())
     {
      printf("headers=%d and columns=%d from the matrix vary is size ",ArraySize(m_columns),colsinrows.Size());
      DebugBreak();
      return false;
     }

//---

   string header_str = "";
   for(int i=0; i<ArraySize(m_columns); i++)  //We concatenate the header only separating it with a comma delimeter
      header_str += m_columns[i] + (i+1 == colsinrows.Size() ? "" : ",");

   FileWrite(handle,header_str);
   FileSeek(handle,0, SEEK_SET);

   for(ulong i=0; i<m_values.Rows() && !IsStopped(); i++)
     {
      ZeroMemory(concstring);

      row = m_values.Row(i);
      for(ulong j=0, cols =1; j<row.Size() && !IsStopped(); j++, cols++)
        {
         concstring += (string)row[j] + (cols == m_values.Cols() ? "" : ",");
        }

      if(verbosity)  //if verbosity is set to true, we print the information to let the user know the progress, Useful for debugging purposes
         printf("Writing a CSV file... record [%d/%d]",i+1,m_values.Rows());

      FileSeek(handle,0,SEEK_END);
      FileWrite(handle,concstring);
     }

   FileClose(handle);

   return (true);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//|   This function adds a one-dimensional vector of double values   |
//|   to a two-dimensinal matrix known as m_values.                  |
//|                                                                  |
//|   It also adds it's column name to the records                   |
//|                                                                  |
//+------------------------------------------------------------------+
void CDataFrame::insert(string name, const vector &values)
  {
//--- Check if the column exists in the m_columns array if it does exists, instead of creating a new column we modify an existing one

   int col_index = -1;

   for(int i=0; i<(int)m_columns.Size(); i++)
      if(name == m_columns[i])
        {
         col_index = i;
         break;
        }

//---  We check if the dimensiona are Ok

   if(m_values.Rows()==0)
      m_values.Resize(values.Size(), m_values.Cols());

   if(values.Size() > m_values.Rows() && m_values.Rows()>0)  //Check if the new column has a bigger size than the number of rows present in the matrix
     {
      printf("%s new column '%s' size is bigger than the dataframe",__FUNCTION__,name);
      return;
     }

//---

   if(col_index != -1)
     {
      m_values.Col(values, col_index);
      if(MQLInfoInteger(MQL_DEBUG))
         printf("%s column '%s' exists, It will be modified",__FUNCTION__,name);
      return;
     }

//--- If a given vector to be added to the dataframe is smaller than the number of rows present in the matrix, we fill the remaining values with Not a Number (NAN)

   vector temp_vals = vector::Zeros(m_values.Rows());
   temp_vals.Fill(NAN); //to create NAN values when there was a dimensional mismatch

   for(ulong i=0; i<values.Size(); i++)
      temp_vals[i] = values[i];

//---

   m_values.Resize(m_values.Rows(), m_values.Cols()+1); //We resize the m_values matrix to accomodate the new column
   m_values.Col(temp_vals, m_values.Cols()-1);     //We insert the new column after the last column

   ArrayResize(m_columns, m_columns.Size()+1); //We increase the sice of the column names to accomodate the new column name
   m_columns[m_columns.Size()-1] = name;   //we assign the new column to the last place in the array
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CDataFrame CDataFrame::dropna()
  {
   CDataFrame res_df;
   
   matrix res = matrix::Zeros(m_values.Rows(), m_values.Cols()); // Pre-allocate space
   ulong new_rows = 0;

   for(ulong row = 0; row < m_values.Rows(); row++)
     {
      vector v = m_values.Row(row);
      bool dropthisrow = false;

      // Check if the row contains any invalid number
      for(int k = 0; k < (int)v.Size(); k++)
        {
         if(!MathIsValidNumber(v[k]))
           {
            dropthisrow = true;
            break;
           }
        }

      if(dropthisrow)
         continue; // Skip this row

      // Copy valid row to the result matrix
      res.Row(v, new_rows++);
      //printf("%s [%d/%d] new_rows %d",__FUNCTION__,row,m_values.Rows(),new_rows);
     }

// Resize the result matrix to fit the valid rows only
   res.Resize(new_rows, res.Cols());

   PrintFormat("%s completed. Rows dropped: %d/%d", __FUNCTION__, m_values.Rows() - new_rows, m_values.Rows());
   
   res_df.m_values = res;
   ArrayCopy(res_df.m_columns, m_columns);
   
   return res_df;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
rolling_struct CDataFrame::rolling(const vector &v, const uint window)
  {
   rolling_struct roll_res;

   roll_res.matrix__.Resize(v.Size(), window);
   roll_res.matrix__.Fill(NAN);

   for(ulong i = 0; i < v.Size(); i++)
     {
      for(ulong j = 0; j < window; j++)
        {
         // Calculate the index in the vector for the rolling window
         ulong index = i - (window - 1) + j;

         if(index >= 0 && index < v.Size())
            roll_res.matrix__[i][j] = v[index];
        }
     }

   return roll_res;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
rolling_struct CDataFrame::rolling(const string index, const uint window)
  {
   vector v = GetColumn(index);

   return rolling(v, window);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CDataFrame::diff(const vector &v, int period=1)
  {
   vector res(v.Size());
   res.Fill(NAN);

   for(ulong i=period; i<v.Size(); i++)
      res[i] = v[i] - v[i-period]; //Calculate the difference between the current value and the previous one

   return res;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CDataFrame::diff(const string col_name, int period=1)
  {
   vector v = this.GetColumn(col_name);
// Initialize a result vector filled with NAN

   return diff(v, period);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CDataFrame::shift(const vector &v, const int shift_index)
  {
// Initialize a result vector filled with NAN
   vector result(v.Size());
   result.Fill(NAN);

   if(shift_index > 0)
     {
      // Positive shift_index: Move elements forward
      for(ulong i = 0; i < v.Size() - shift_index; i++)
         result[i + shift_index] = v[i];
     }
   else
      if(shift_index < 0)
        {
         // Negative shift_index: Move elements backward
         for(ulong i = -shift_index; i < v.Size(); i++)
            result[i + shift_index] = v[i];
        }
      else
        {
         // Zero shift_index: Return the vector unchanged
         result = v;
        }

   return result;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CDataFrame::shift(const string col_name, const int shift_index)
  {
   vector v = this.GetColumn(col_name);
// Initialize a result vector filled with NAN

   return shift(v, shift_index);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
// Proper column extraction from linear array
void CDataFrame::GetCol(const string &data[], string &output[], int col_index, int total_columns)
{
    int rows = ArraySize(data) / total_columns;
    ArrayResize(output, rows);
    
    for(int i = 0; i < rows; i++)
    {
        output[i] = data[i * total_columns + (col_index - 1)];
    }
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CDataFrame::from_csv(string file_name,string delimiter=",",bool is_common=false, string datetime_columns="",string encode_columns="", bool verbosity=false)
  {
   string Arr[];
   
//--- Optimized array handling

   int CHUNK_SIZE = 1000;
   int capacity = CHUNK_SIZE;
   ArrayResize(Arr, capacity);

//---

   int handle = FileOpen(file_name,FILE_SHARE_READ|FILE_CSV|FILE_ANSI|(is_common?FILE_COMMON:FILE_ANSI),delimiter,CP_UTF8);
   
   int all_size = 0, 
       header_columns = 0;
   
   if(handle == INVALID_HANDLE)
     {
      printf("Invalid %s handle Error %d ",file_name,GetLastError());
      Print(GetLastError()==0?" TIP | File Might be in use Somewhere else or in another Directory":"");
      return false;
     }
   else
     {
      int columns = 0, rows=0;
      while(!FileIsEnding(handle) && !IsStopped())
        {
         string data = FileReadString(handle); 
         
         //---
         
         if(rows ==0)
           {
             header_columns++;
             ArrayResize(m_columns, header_columns);
             
             m_columns[header_columns-1] = data;
           }
         
         columns++;
                  
         if(rows>0)  //Avoid the first column which contains the column's header
          {
            if(all_size >= capacity)
             {
                 capacity += CHUNK_SIZE;
                 ArrayResize(Arr, capacity);
             }
             
             Arr[all_size++] = data;
          }
          
         //---

         if(FileIsLineEnding(handle))
           {            
            if (columns!=header_columns)
             {
                printf("%s there is a mismatch in the number of columns inside '%s'",__FUNCTION__,file_name);
                return false;
             }
             
            columns = 0; //reset columns count
            Comment(StringFormat("Reading %s record [%d] ",file_name,rows++));
           }
        }  
        
      FileClose(handle);
     }
     
//--- Finally resize to exact size:

   ArrayResize(Arr, all_size); 
   Comment(""); //Clear the comment
   
//--- 

   if(all_size % header_columns != 0)
   {
       printf("%s Error data size doesn't match column count",__FUNCTION__);
       return false;
   }
   
   int rows = all_size / header_columns;
   m_values.Resize(rows, header_columns);
   
   string Col[]; //A column in strings format
   vector col_vector = {};
    
//---
   
   int encoder_count=0;
   for (int i=0; i<header_columns; i++)
      {
         string col_name = m_columns[i];
         
         GetCol(Arr, Col, i+1, header_columns); //Get the column in string format
         
         if (StringFind(datetime_columns, col_name)!=-1 && datetime_columns!="") //If the current column is in the list of datetime columns
            {
               printf("%s is a datetime column", col_name);
               col_vector.Resize(Col.Size());
               for (uint k=0; k<Col.Size(); k++)
                  col_vector[k] = (double)StringToTime(Col[k]);
               
               m_values.Col(col_vector, i); //Store the column in a matrix
               continue;
            }
            
         if (StringFind(encode_columns, col_name)!=-1 && encode_columns!="") //If the current column is in the list of columns to encode
            {
               encoder_count++;
               ArrayResize(m_columns_encoders, encoder_count);
               
               if (MQLInfoInteger(MQL_DEBUG))
                  printf("Encoding column: %s",col_name);
                  
               col_vector = m_columns_encoders[encoder_count-1].fit_transform(Col); //Encode the column
               
               m_values.Col(col_vector, i); //Store the column in a matrix
               continue;
            }
             
          col_vector = ArrayToVector(Col);
          m_values.Col(col_vector, i); //Store the column in a matrix
      }
   
   
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
matrix CDataFrame::tail(uint count=5)
  {
   ulong rows = m_values.Rows();
   if(count>=rows)
     {
      printf("%s count[%d] >= number of rows in the df[%d]",__FUNCTION__,count,rows);
      return matrix::Zeros(0,0);
     }

   ulong start = rows-count;
   matrix res = matrix::Zeros(count, m_values.Cols());

   for(ulong i=start, row_count=0; i<rows; i++, row_count++)
      res.Row(m_values.Row(i), row_count);

   return res;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CDataFrame::info(void)
  {
   ulong rows = m_values.Rows(), cols = m_values.Cols();

//--- Calculate the maximum column name length for consistent padding

   uint max_col_name_len = 0;
   for(ulong i = 0; i < cols; i++)
      max_col_name_len = MathMax(max_col_name_len, StringLen(m_columns[i]));

   max_col_name_len += 2; // Add extra padding for readability

//--- Print basic DataFrame info

   printf("<class 'CDataFrame'>");
   printf("RangeIndex: %d entries, 0 to %d", rows, rows - 1);
   printf("Data columns (total %d columns):", cols);

// Print header for column details
   printf(" #   %-*s   Non-Null Count   Dtype", max_col_name_len, "Column");
   printf("---  %-*s   --------------   -----", max_col_name_len, "------");

   for(ulong i = 0; i < cols; i++)  //--- Print each column's info with consistent padding
     {
      int null_count = CountNaN(m_values.Col(i)); //we count all the NAN values
      int non_null_count = (int)rows - null_count;

      string dtype = typename(double); // Since all columns are double
      printf(" %d   %-*s   %d non-null    %s", i, max_col_name_len, m_columns[i], non_null_count, dtype);
     }

//--- Estimate memory usage

   ulong memory_usage = rows * cols * sizeof(double);
   printf("memory usage: %.1f KB", double(memory_usage) / 1024.0);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int CDataFrame::CountNaN(const vector &v)
  {
   int count=0;
   for(ulong i=0; i<v.Size(); i++)
      if(!MathIsValidNumber(v[i]))
         count++;

   return count;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CDataFrame::describe(void)
 {
   uint cols = m_columns.Size();
   if (cols == 0) return; // Handle edge case for empty DataFrame
   
   // Array for dynamic padding
   uint col_widths[];
   ArrayResize(col_widths, cols);
   
   // Calculate maximum width for each column
   for (uint i = 0; i < cols; i++)
   {
      col_widths[i] = StringLen(m_columns[i]); // Start with column name length
      
      // Compare with statistics lengths to determine the maximum width
      col_widths[i] = MathMax(col_widths[i], StringLen(StringFormat("%d", m_values.Col(i).Size())));
      col_widths[i] = MathMax(col_widths[i], StringLen(StringFormat("%.6f", m_values.Col(i).Mean())));
      col_widths[i] = MathMax(col_widths[i], StringLen(StringFormat("%.6f", m_values.Col(i).Std())));
      col_widths[i] = MathMax(col_widths[i], StringLen(StringFormat("%.6f", m_values.Col(i).Min())));
      col_widths[i] = MathMax(col_widths[i], StringLen(StringFormat("%.6f", m_values.Col(i).Percentile(25))));
      col_widths[i] = MathMax(col_widths[i], StringLen(StringFormat("%.6f", m_values.Col(i).Percentile(50))));
      col_widths[i] = MathMax(col_widths[i], StringLen(StringFormat("%.6f", m_values.Col(i).Percentile(75))));
      col_widths[i] = MathMax(col_widths[i], StringLen(StringFormat("%.6f", m_values.Col(i).Max())));
   }
   
   // Print column names with dynamic padding
   string col_names = StringFormat("%-10s", ""); // Leave space for the stat names
   for (uint i = 0; i < cols; i++)
      col_names += StringFormat(" %-*s ", col_widths[i], m_columns[i]);
   Print(col_names);

   // Print statistics rows with dynamic padding
   string count = StringFormat("%-10s", "count");
   string mean = StringFormat("%-10s", "mean");
   string std = StringFormat("%-10s", "std");
   string min = StringFormat("%-10s", "min");
   string __25 = StringFormat("%-10s", "25%");
   string __50 = StringFormat("%-10s", "50%");
   string __75 = StringFormat("%-10s", "75%");
   string max = StringFormat("%-10s", "max");

   for (uint i = 0; i < cols; i++)
   {
      count += StringFormat(" %-*d ", col_widths[i], m_values.Col(i).Size());
      mean += StringFormat(" %-*.6f ", col_widths[i], m_values.Col(i).Mean());
      std += StringFormat(" %-*.6f ", col_widths[i], m_values.Col(i).Std());
      min += StringFormat(" %-*.6f ", col_widths[i], m_values.Col(i).Min());
      __25 += StringFormat(" %-*.6f ", col_widths[i], m_values.Col(i).Percentile(25));
      __50 += StringFormat(" %-*.6f ", col_widths[i], m_values.Col(i).Percentile(50));
      __75 += StringFormat(" %-*.6f ", col_widths[i], m_values.Col(i).Percentile(75));
      max += StringFormat(" %-*.6f ", col_widths[i], m_values.Col(i).Max());
   }

   // Print all rows
   Print(count);
   Print(mean);
   Print(std);
   Print(min);
   Print(__25);
   Print(__50);
   Print(__75);
   Print(max);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
