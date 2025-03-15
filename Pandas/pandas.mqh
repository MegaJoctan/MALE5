//+------------------------------------------------------------------+
//|                                                       pandas.mqh |
//|                                          Copyright 2023, Omegafx |
//|                                           https://www.omegafx.co |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, Omegafx"
#property link      "https://www.omegafx.co"
//+------------------------------------------------------------------+
//| defines                                                          |
//+------------------------------------------------------------------+

#ifndef NaN
#define  NaN double("nan")
#endif

#ifndef inf
#define  inf double("inf")
#endif

#ifndef neg_inf
#define  neg_inf double("-inf")
#endif

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

   vector            Mean()
     {
      vector res(matrix__.Rows());
      res.Fill(NaN);

      for(ulong i=0; i<res.Size(); i++)
         res[i] = matrix__.Row(i).Mean();

      return res;
     }

   vector            Std()
     {
      vector res(matrix__.Rows());
      res.Fill(NaN);

      for(ulong i=0; i<res.Size(); i++)
         res[i] = matrix__.Row(i).Std();

      return res;
     }

   vector            Var()
     {
      vector res(matrix__.Rows());
      res.Fill(NaN);

      for(ulong i=0; i<res.Size(); i++)
         res[i] = matrix__.Row(i).Var();

      return res;
     }

   vector            Skew()
     {
      vector res(matrix__.Rows());
      res.Fill(NaN);

      for(ulong i=0; i<res.Size(); i++)
         res[i] = CalculateSkewness(matrix__.Row(i));

      return res;
     }

   vector            Kurtosis()
     {
      vector res(matrix__.Rows());
      res.Fill(NaN);

      for(ulong i=0; i<res.Size(); i++)
         res[i] = CalculateKurtosis(matrix__.Row(i));

      return res;
     }

   vector            Median()
     {
      vector res(matrix__.Rows());
      res.Fill(NaN);

      for(ulong i=0; i<res.Size(); i++)
         res[i] = matrix__.Row(i).Median();

      return res;
     }
     
   vector            Percentile(int value)
     {
      vector res(matrix__.Rows());
      res.Fill(NaN);

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

public:

   string            m_columns[]; //An array of string values for keeping track of the column names
   matrix            m_values; // A 2D matrix

                     CDataFrame();
                     CDataFrame(const string columns, const matrix &values);
                     CDataFrame(const string &columns[], const matrix &values);
                    ~CDataFrame(void);

                     CDataFrame(const CDataFrame &other) { CopyFrom(other); }
   CDataFrame        operator=(const CDataFrame &other) { CopyFrom(other); return GetPointer(this); }

   //--- Data selection and Indexing

   vector            operator[](const string index) {return GetColumn(index); }  //Access a column by its name

   vector            Loc(int index, uint axis = 0);
   CDataFrame        Iloc(ulong start_row, ulong end_row, ulong start_col, ulong end_col);
   double            At(ulong row, string col_name);
   double            Iat(ulong row, ulong col);

   //--- Data exploration methods

   matrix            Tail(uint count=5);
   void              Info();
   void              Describe(void);
   
   //---


   CDataFrame        Drop(const string cols);
   void              Head(const uint count=5);
   bool              ToCSV(const string file_name, const bool common_path=false, bool verbosity=false);

   bool              FromCSV(string file_name,string delimiter=",",bool is_common=false, bool verbosity=false);
   void              Insert(string name, const vector &values);

   CDataFrame        Dropnan();
      
   //--- Timeseries transformation and manipulations
   
   vector            Pct_change(const string index);
   vector            Pct_change(const vector &v);

   rolling_struct    Rolling(const vector &v, const uint window);
   rolling_struct    Rolling(const string index, const uint window);
   
   vector            Shift(const vector &v, const int shift);
   vector            Shift(const string index, const int shift);
   
   vector            Diff(const vector &v, int period=1);
   vector            Diff(const string index, int period=1);
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
vector CDataFrame::Pct_change(const string index)
  {
   vector col = GetColumn(index);
   return Pct_change(col);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CDataFrame::Pct_change(const vector &v)
  {
   vector col = v;
   ulong size = col.Size();

   vector results(size);
   results.Fill(NaN);

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
vector CDataFrame::Loc(int index, uint axis=0)
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
CDataFrame CDataFrame::Iloc(ulong start_row,ulong end_row,ulong start_col,ulong end_col)
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
   df.m_values.Fill(NaN);

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
double CDataFrame::At(ulong row, string col_name)
  {
   ulong col_number = (ulong)ColNameToIndex(col_name, m_columns);
   return m_values[row][col_number];
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CDataFrame::Iat(ulong row,ulong col)
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
CDataFrame CDataFrame::Drop(const string cols)
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
void CDataFrame::Head(const uint count=5)
  {
// Calculate maximum width needed for each column
   uint num_cols = m_columns.Size();
   uint col_widths[];
   ArrayResize(col_widths, num_cols);

   for(uint col = 0; col < num_cols; col++)  //Determining column width for visualizing a simple table
     {
      uint max_width = StringLen(m_columns[col]);
      for(uint row = 0; row < count && row < m_values.Rows(); row++)
        {
         string num_str = StringFormat("%.8f", m_values[row][col]);
         max_width = MathMax(max_width, StringLen(num_str));
        }
      col_widths[col] = max_width + 4; // Extra padding for readability
     }

// Print column headers with calculated padding
   string header = "";
   for(uint col = 0; col < num_cols; col++)
     {
      header += StringFormat("| %-*s ", col_widths[col], m_columns[col]);
     }
   header += "|";
   Print(header);

// Print rows with padding for each column
   for(uint row = 0; row < count && row < m_values.Rows(); row++)
     {
      string row_str = "";
      for(uint col = 0; col < num_cols; col++)
        {
         row_str += StringFormat("| %-*.*f ", col_widths[col], 8, m_values[row][col]);
        }
      row_str += "|";
      Print(row_str);
     }

// Print dimensions
   printf("(%dx%d)", m_values.Rows(), m_values.Cols());
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CDataFrame::ToCSV(string csv_name, bool common=false, bool verbosity=false)
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
void CDataFrame::Insert(string name, const vector &values)
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

//--- If a given vector to be added to the dataframe is smaller than the number of rows present in the matrix, we fill the remaining values with Not a Number (NaN)

   vector temp_vals = vector::Zeros(m_values.Rows());
   temp_vals.Fill(NaN); //to create NaN values when there was a dimensional mismatch

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
CDataFrame CDataFrame::Dropnan()
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
rolling_struct CDataFrame::Rolling(const vector &v, const uint window)
  {
   rolling_struct roll_res;

   roll_res.matrix__.Resize(v.Size(), window);
   roll_res.matrix__.Fill(NaN);

   for(ulong i = 0; i < v.Size(); i++)
     {
      for(ulong j = 0; j < window; j++)
        {
         // Calculate the index in the vector for the Rolling window
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
rolling_struct CDataFrame::Rolling(const string index, const uint window)
  {
   vector v = GetColumn(index);

   return Rolling(v, window);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CDataFrame::Diff(const vector &v, int period=1)
  {
   vector res(v.Size());
   res.Fill(NaN);

   for(ulong i=period; i<v.Size(); i++)
      res[i] = v[i] - v[i-period]; //Calculate the difference between the current value and the previous one

   return res;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CDataFrame::Diff(const string index, int period=1)
  {
   vector v = this.GetColumn(index);
// Initialize a result vector filled with NaN

   return Diff(v, period);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CDataFrame::Shift(const vector &v, const int shift)
  {
// Initialize a result vector filled with NaN
   vector result(v.Size());
   result.Fill(NaN);

   if(shift > 0)
     {
      // Positive shift: Move elements forward
      for(ulong i = 0; i < v.Size() - shift; i++)
         result[i + shift] = v[i];
     }
   else
      if(shift < 0)
        {
         // Negative shift: Move elements backward
         for(ulong i = -shift; i < v.Size(); i++)
            result[i + shift] = v[i];
        }
      else
        {
         // Zero shift: Return the vector unchanged
         result = v;
        }

   return result;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CDataFrame::Shift(const string index, const int shift)
  {
   vector v = this.GetColumn(index);
// Initialize a result vector filled with NaN

   return Shift(v, shift);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CDataFrame::FromCSV(string file_name,string delimiter=",",bool is_common=false, bool verbosity=false)
  {
   matrix mat_ = {};
   int rows_total=0;
   int handle = FileOpen(file_name,FILE_READ|FILE_CSV|FILE_ANSI|(is_common?FILE_IS_COMMON:FILE_ANSI),delimiter); //Open a csv file

   ResetLastError();
   if(handle == INVALID_HANDLE) //Check if the file handle is ok if not return false
     {
      printf("Invalid %s handle Error %d ",file_name,GetLastError());
      Print(GetLastError()==0?" TIP | File Might be in use Somewhere else or in another Directory":"");

      return false;
     }
   else
     {
      int column = 0, rows=0;

      while(!FileIsEnding(handle))
        {
         string data = FileReadString(handle);
         //---
         if(rows ==0)
           {
            ArrayResize(m_columns,column+1);
            m_columns[column] = data;
           }
         if(rows>0)  //Avoid the first column which contains the column's header
            mat_[rows-1,column] = (double(data)); //add a value to the matrix
         column++;
         //---
         if(FileIsLineEnding(handle)) //At the end of the each line
           {
            rows++;

            mat_.Resize(rows,column); //Resize the matrix to accomodate new values
            column = 0;
           }
        }

      if(verbosity)  //if verbosity is set to true, we print the information to let the user know the progress, Useful for debugging purposes
         printf("Reading a CSV file... record [%d]",rows);

      rows_total = rows;
      FileClose(handle); //Close the file after reading it
     }

   mat_.Resize(rows_total-1,mat_.Cols());
   m_values = mat_;

   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
matrix CDataFrame::Tail(uint count=5)
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
void CDataFrame::Info(void)
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
      int null_count = CountNaN(m_values.Col(i)); //we count all the NaN values
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
void CDataFrame::Describe(void)
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
