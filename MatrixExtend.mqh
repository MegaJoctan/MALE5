//+------------------------------------------------------------------+
//|                                                 matrix_utils.mqh |
//|                                  Copyright 2022, Omega Joctan  . |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, Omega Joctan"
#property link      "https://www.mql5.com/en/users/omegajoctan"

#include <MALE5\preprocessing.mqh>

//+------------------------------------------------------------------+
//|   A class containing additional matrix manipulation functions    |
//+------------------------------------------------------------------+

class MatrixExtend
  {
  
protected:

   template<typename T>
   static T          MathRandom(T mini, T maxi);
   static string     CalcTimeElapsed(double seconds);
   static void       Swap(double &var1, double &var2);
   static string     ConvertTime(double seconds);
   
   template<typename T>
   static void       GetCol(const T &Matrix[], T &Col[], int column, int cols);
   
   static bool       IsNumber(string text);
   static vector     FixColumn(CLabelEncoder &encoder, string &Arr[], double threshold =0.3);


public:
                     MatrixExtend(void);
                    ~MatrixExtend(void);
   
   template<typename T>
   static int Sign(T var)
    {
      if (var<0)
        return -1;
      else if (var==0)
        return 0;
      else
        return 1;
    }
    
//--- File Functions

   template <typename T>
   static bool       WriteCsv(string csv_name, matrix<T> &matrix_, string &header[] ,bool common=false, int digits=5);
   template <typename T>
   static bool       WriteCsv(string csv_name, matrix<T> &matrix_, string header_string="",bool common=false, int digits=5);
   static matrix     ReadCsv(string file_name, string &headers, string delimiter=",",bool common=false);
   static matrix     DBtoMatrix(int db_handle, string table_name,string &column_names[],int total=WHOLE_ARRAY);
   
//--- Manipulations
   
   template<typename T>
   static bool       RemoveCol(matrix<T> &mat, ulong col);
   static void       RemoveMultCols(matrix &mat, int &cols[]);
   static void       RemoveMultCols(matrix &mat, int from, int total=WHOLE_ARRAY);
   static void       RemoveRow(matrix &mat,ulong row);
   static void       VectorRemoveIndex(vector &v, ulong index);  
   
//--- Machine Learning 

   template<typename T>
   static bool       XandYSplitMatrices(const matrix<T> &matrix_, matrix<T> &xmatrix, vector<T> &y_vector,int y_column=-1);
   template <typename T>
   static void       TrainTestSplitMatrices(matrix<T> &matrix_, matrix<T> &x_train, vector<T> &y_train, matrix<T> &x_test, vector<T> &y_test, double train_size=0.7,int random_state=-1);
   static matrix     DesignMatrix(matrix &x_matrix);              
   static matrix     OneHotEncoding(vector &v);    //ONe hot encoding 
   static matrix     Sign(matrix &x);
   static vector     Sign(vector &x);
   
//--- Detection

   static void       Unique(const string &Array[], string &classes_arr[]);
   static vector     Unique(const vector &v);           //Identifies classes available in a vector
   static vector     Unique_count(vector &v);
   
   template<typename T> 
   static vector     Random(T min, T max, int size,int random_state=-1);          //Generates a random vector of type T sized = size
   static matrix     Random(double min, double max, ulong rows, ulong cols, int random_state=-1); 
   
   template<typename T>
   static vector     Search(const vector<T> &v, T value);
   
//--- Transformations

   static matrix     VectorToMatrix(const vector &v, ulong cols=1);
   template<typename T>
   static vector     MatrixToVector(matrix<T> &mat);
   
   template<typename T>
   static vector     ArrayToVector(const T &Arr[]);     
   template<typename T>
   static bool       VectorToArray(const vector<T> &v,T &arr[]);
   
//--- Manipulations

   static vector     concatenate(vector &v1, vector &v2);              //Appends v2 to vector 1
   static matrix     concatenate(matrix &mat1, matrix &mat2, int axis = 0);
   template<typename T>
   static matrix<T>  concatenate(matrix<T> &mat, vector<T> &v, int axis=1);
   
   template<typename T>
   static bool       Copy(const vector<T> &src, vector<T> &dst, ulong src_start,ulong total=WHOLE_ARRAY);
   
   
   template<typename T>
   static void       Reverse(vector<T> &v);
   template<typename T>
   static void       Reverse(matrix<T> &mat);
   
   static matrix     HadamardProduct(matrix &a, matrix &b);
   
   template<typename T>
   static void       Randomize(vector<T> &v, int random_state=-1, bool replace=false);
   template<typename T>
   static void       Randomize(matrix<T> &matrix_,int random_state=-1, bool replace=false);
   
   template<typename T>
   static void       NormalizeDouble_(vector<T> &v, int digits=3);
   template<typename T>
   static void       NormalizeDouble_(matrix<T> &mat, int digits=3);
   
   static int        CopyBufferVector(int handle, int buff_num, int start_pos,int count, vector &v);
   static string     Stringfy(vector &v, int digits = 2);
   static matrix     Zeros(ulong rows, ulong cols) { matrix ret_mat(rows, cols); return(ret_mat.Fill(0.0)); }
   static vector     Zeros(ulong size) { vector ret_v(size); return( ret_v.Fill(0.0)); }
   static matrix     Get(const matrix &mat, ulong start_index, ulong end_index);
   static vector     Get(const vector &v, ulong start_index, ulong end_index);
   template<typename T>
   static vector     Sort(vector<T> &v,ENUM_SORT_MODE sort_mode=SORT_ASCENDING);
   template<typename T>
   static vector     ArgSort(vector<T> &v);

//--- Others
   
   static void       PrintShort(matrix &matrix_,ulong rows=5, int digits=5);

  }; 
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
MatrixExtend::MatrixExtend(void)
  {
    
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
MatrixExtend::~MatrixExtend(void)
  {
  
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
matrix MatrixExtend::VectorToMatrix(const vector &v, ulong cols=1)
  {      
   ulong rows = 0;
   matrix mat = {};
   
   
    if ( v.Size() % cols > 0) //If there is a reminder
      {
        printf("Invalid rows %d and cols %d for this vector size ",rows,v.Size()/cols);
        return mat;
      }
    else
       rows = v.Size()/cols;

//---

   mat.Resize(rows, cols); 

   for(ulong i=0, index =0; i<rows; i++)
      for(ulong j=0; j<cols; j++, index++)
        {
         mat[i][j] = v[index];
        }
   return(mat);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
template<typename T>
vector MatrixExtend::MatrixToVector(matrix<T> &mat)
  {
    vector<T> v = {};
    matrix<T> temp_mat = mat;
    
    if (!temp_mat.Swap(v))
      Print(__FUNCTION__," Failed to turn the matrix[",mat.Rows(),"x",mat.Cols(),"] into a vector");
    
    return(v);
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
template<typename T>
bool MatrixExtend::RemoveCol(matrix<T> &mat, ulong col)
  {
   matrix<T> new_matrix(mat.Rows(),mat.Cols()-1); //Remove the one Column
   if (col > mat.Cols())
     {
       Print(__FUNCTION__," column out of range");
       return false;
     }

   for (ulong i=0, new_col=0; i<mat.Cols(); i++) 
     {
        if (i == col)
          continue;
        else
          {
           new_matrix.Col(mat.Col(i),new_col);
           new_col++;
          }    
     }

   mat.Copy(new_matrix);
   
   return true;
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void MatrixExtend::RemoveMultCols(matrix &mat, int &cols[])
  {
   ulong size = (int)ArraySize(cols);

   if(size > mat.Cols())
     {
      Print(__FUNCTION__," Columns to remove can't be more than the available columns");
      return;
     }


   vector Zeros(mat.Rows());
   Zeros.Fill(0);

   for(ulong i=0; i<size; i++)
      for(ulong j=0; j<mat.Cols(); j++)
        {
         if(cols[i] == j)
            mat.Col(Zeros,j);
        }

//---

   vector column_vector;
   
   while (mat.Cols()-size >= size)
      for(ulong i=0; i<mat.Cols(); i++)
        {
         column_vector = mat.Col(i);
         if(column_vector.Sum()==0)
            if (!RemoveCol(mat,i))
              {
                printf("%s Line %d Failed to remove a column %d from a matrix",__FUNCTION__,__LINE__,i);
                break;
              }
        }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

void MatrixExtend::RemoveMultCols(matrix &mat, int from, int total=WHOLE_ARRAY)
 {
   
   total = total==WHOLE_ARRAY ? (int)mat.Cols()-from : total;
   
   if(total > (int)mat.Cols())
     {
      Print(__FUNCTION__," Columns to remove can't be more than the available columns");
      return;
     }

   vector Zeros(mat.Rows());
   Zeros.Fill(0);

   for (int i=from; i<total+from; i++)
      mat.Col(Zeros, i);
   
//---      
   
   ulong remain_size = mat.Cols()-total;
   
   
   while (mat.Cols() >= remain_size && !IsStopped())
    {
      //printf("cols %d total %d",cols,total);
      
      for(ulong i=0; i<mat.Cols(); i++) //loop the entire matrix searching for columns to remove
         if(mat.Col(i).Sum()==0)
            if (!RemoveCol(mat,i))
              {
                printf("%s Line %s Failed to remove a column %d from a matrix",__FUNCTION__,__LINE__,i);
                break;
              }
    }
 }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void MatrixExtend::RemoveRow(matrix &mat,ulong row)
  {
   matrix new_matrix(mat.Rows()-1,mat.Cols()); //Remove the one Row
 
      for(ulong i=0, new_rows=0; i<mat.Rows(); i++)
        {
         if(i == row)
            continue;
         else
           {
            new_matrix.Row(mat.Row(i),new_rows);
            new_rows++;
           }
        }

   mat.Copy(new_matrix);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void MatrixExtend::VectorRemoveIndex(vector &v, ulong index)
  {
   vector new_v(v.Size()-1);

   for(ulong i=0, count = 0; i<v.Size(); i++)
      if(i != index)
        {
         new_v[count] = v[i];
         count++;
        }
    v.Copy(new_v);
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
template <typename T>
bool MatrixExtend::WriteCsv(string csv_name, matrix<T> &matrix_, string &header[], bool common=false, int digits=5)
  {
   string header_str = "";
   for (int i=0; i<ArraySize(header); i++)
      header_str += header[i] + ((i+1 == ArraySize(header)) ? "" : ",");
      
   return WriteCsv(csv_name, matrix_, header_str, common, digits);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
template <typename T>
bool MatrixExtend::WriteCsv(string csv_name, matrix<T> &matrix_, string header_string="", bool common=false, int digits=5)
  {
   FileDelete(csv_name);
   int handle = FileOpen(csv_name,FILE_WRITE|FILE_CSV|FILE_ANSI|(common?FILE_COMMON:FILE_IS_WRITABLE),",",CP_UTF8);
   
   if (header_string == "" || header_string == NULL)
     for (ulong i=0; i<matrix_.Cols(); i++)
       header_string += "None"+ (i==matrix_.Cols()-1?"":","); 

   if(handle == INVALID_HANDLE)
     {
       printf("Invalid %s handle Error %d ",csv_name,GetLastError());
       return (false);
     }
            
   string concstring;
   vector<T> row = {};
   
   datetime time_start = GetTickCount(), current_time;
   
   string header[];
   
   ushort u_sep;
   u_sep = StringGetCharacter(",",0);
   StringSplit(header_string,u_sep, header);
   
   vector<T> colsinrows = matrix_.Row(0);
   
   if (ArraySize(header) != (int)colsinrows.Size())
      {
         printf("headers=%d and columns=%d from the matrix vary is size ",ArraySize(header),colsinrows.Size());
         return false;
      }

//---

   string header_str = "";
   for (int i=0; i<ArraySize(header); i++)
      header_str += header[i] + (i+1 == colsinrows.Size() ? "" : ",");
   
   FileWrite(handle,header_str);
   
   FileSeek(handle,0,SEEK_SET);
   
   for(ulong i=0; i<matrix_.Rows() && !IsStopped(); i++)
     {
      ZeroMemory(concstring);

      row = matrix_.Row(i);
      for(ulong j=0, cols =1; j<row.Size() && !IsStopped(); j++, cols++)
        {
         current_time = GetTickCount();
         
         Comment("Writting ",csv_name," record [",i+1,"/",matrix_.Rows(),"] Time taken | ",ConvertTime((current_time - time_start) / 1000.0));
         
         concstring += (string)NormalizeDouble(row[j],digits) + (cols == matrix_.Cols() ? "" : ",");
        }

      FileSeek(handle,0,SEEK_END);
      FileWrite(handle,concstring);
     }
        
   FileClose(handle);
   
   return (true);
  }
//+------------------------------------------------------------------+
//| This Function is aimed at Detectin the Strings columns and it    |
//| encodes them, while fixing the missing information in the column |
//+------------------------------------------------------------------+
vector MatrixExtend::FixColumn(CLabelEncoder &encoder, string &Arr[], double threshold =0.3)
 { 
   int size = ArraySize(Arr);
   int str_count =0;
   
   vector ret(size);
   
   for (int i=0; i<size; i++) //Check what percentage of data is strings
      if (!IsNumber(Arr[i]))
        str_count++;

//---

   bool is_strings_col = (str_count>=size*threshold);
     
   if (is_strings_col) //if a column is detected to be a column full of strings
     {
      //Encode it
      return encoder.encode(Arr);;
     }       
     
//---
      
   string value = "";
   int total =0;
   double mean=0;
   
   for (int i=0; i<size; i++) //Detect Missing values | Remove the rows
     { 
       value = Arr[i];
        if (value == "NaN" || value == "-NaN" || value == "!VALUE" ||
           value == "" || value == "NA" || value == "N/A" || value == "null" ||
           value == "Inf" || value == "Infinity" || value == "-Inf" || value == "-Infinity" ||
           value == "#DIV/0!" || value == "#VALUE!") //Check if there are NotANumber values 
          continue;
                
        mean += (double)Arr[i];
        total++;
     }
   
   mean /= total;

//---
   
   for (int i=0; i<size; i++) //Detect Missing values | Remove the rows
     { 
       value = Arr[i];
        if (value == "NaN" || value == "-NaN" || value == "!VALUE" ||
           value == "" || value == "NA" || value == "N/A" || value == "null" ||
           value == "Inf" || value == "Infinity" || value == "-Inf" || value == "-Infinity" ||
           value == "#DIV/0!" || value == "#VALUE!") //Check if there are NotANumber values 
          {
            ret[i] = mean;
            continue;
          }
          
          ret[i] = double(Arr[i]);       
     }
    
   return ret;  
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool MatrixExtend::IsNumber(string text)
{
    int length = StringLen(text);   // Get the length of the string.
    int pointcount = 0;             // Initialize a counter for the number of decimal points.

    // Iterate through each character in the text.
    for (int i = 0; i < length; i++)
    {
        int char1 = StringGetCharacter(text, i);  // Get the ASCII code of the current character.

        // If the character is a decimal point, increment the decimal point counter.
        if (char1 == 46)
            pointcount += 1;

        // If the character is a digit or a decimal point and the number of decimal points is less than 2,
        // continue to the next character; otherwise, return false.
        if (((char1 >= 48 && char1 <= 57) || char1 == 46) && pointcount < 2)
            continue;
        else
            return false;
    }

    // If all characters in the text have been checked without returning false, return true.
    return true;
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
matrix MatrixExtend::ReadCsv(string file_name, string &headers, string delimiter=",",bool common=false)
  {
   CLabelEncoder encoder;
  
   string Arr[];
   int all_size = 0;
   
   int cols_total=0;
   
   int handle = FileOpen(file_name,FILE_SHARE_READ|FILE_CSV|FILE_ANSI|(common?FILE_COMMON:FILE_ANSI),delimiter);
      
   datetime time_start = GetTickCount(), current_time;
   
   if(handle == INVALID_HANDLE)
     {
      printf("Invalid %s handle Error %d ",file_name,GetLastError());
      Print(GetLastError()==0?" TIP | File Might be in use Somewhere else or in another Directory":"");
     }
   else
     {
      int column = 0, rows=0;

      while(!FileIsEnding(handle) && !IsStopped())
        {
         string data = FileReadString(handle); 
         
         //---
         
         if(rows ==0)
           {
            headers += data;
           }
         
         column++;
                  
         if(rows>0)  //Avoid the first column which contains the column's header
          {
            all_size++;
            ArrayResize(Arr,all_size);
            
            Arr[all_size-1] = data;
          }
         //---

         if(FileIsLineEnding(handle))
           {
            cols_total=column;
               
            rows++;               
            column = 0;
               
            current_time = GetTickCount();
            Comment("Reading ",file_name," record = ",rows," Time taken | ",ConvertTime((current_time - time_start) / 1000.0));
           }
        }  
        
      FileClose(handle);
     }
     
   int rows =all_size/cols_total;
   
   Comment("");
      
   matrix mat(rows, cols_total);
   string Col[];
   vector col_vector;
   
   for (int i=0; i<cols_total; i++)
      {
         GetCol(Arr, Col, i+1, cols_total);
         mat.Col(FixColumn(encoder, Col), i);
      }

   return(mat);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
template<typename T>
void MatrixExtend::GetCol(const T &Matrix[], T &Col[], int column, int cols)
 {
   int rows = ArraySize(Matrix)/cols;
   ArrayResize(Col,rows);
   
   int start = 0;
    for (int i=0; i<cols; i++)
     {
      start = i;
      
      if (i != column-1)  continue;
      else
        for (int j=0; j<rows; j++)
          {
            //printf("ColMatrix[%d} Matrix{%d]",j,start);
            Col[j] = Matrix[start];
            
            start += cols;
          }
     }  
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
template<typename T>
vector MatrixExtend::ArrayToVector(const T &Arr[])
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
bool MatrixExtend::VectorToArray(const vector<T> &v, T &arr[])
  {
   vector temp = v;
   if (!temp.Swap(arr))
    {
      Print("Failed to Convert vector to Array Err=",GetLastError());
      return false;
    }
   return(true);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//|                                                                  |
//+------------------------------------------------------------------+
template<typename T>
bool MatrixExtend::XandYSplitMatrices(const matrix<T> &matrix_, matrix<T> &xmatrix, vector<T> &y_vector,int y_column=-1)
  {
   y_column = int( y_column==-1 ? matrix_.Cols()-1 : y_column);
   
   if (matrix_.Rows() == 0 || matrix_.Cols()==0)
     {
       #ifdef DEBUG_MODE
         printf("%s Line %d Cannot split the matrix of size[%dx%d]",__FUNCTION__,__LINE__,matrix_.Rows(),matrix_.Cols());
       #endif 
       
       return false;
     }
   
   y_vector = matrix_.Col(y_column);
   xmatrix.Copy(matrix_);
   
   return RemoveCol(xmatrix, y_column); //Remove the y column
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
template<typename T>
void MatrixExtend::Randomize(vector<T> &v, int random_state=-1, bool replace=false)
 {
   MathSrand(random_state!=-1?random_state:GetTickCount());
     
   int swap_index;
   double temp;
   
   int SIZE = (int)v.Size();
   vector<T> temp_v = v;
   
   for (int i=0; i<SIZE; i++) //Fisher yates algorithm
      {
        if (!replace)
          {
            swap_index = rand() % SIZE;
            
            temp = v[i];
            
            v[i] = v[swap_index];
            v[swap_index] = temp;
          }
        else
          {
            v[i] = temp_v[MathRandom(0, SIZE)];
          }
      }   
 }
//+------------------------------------------------------------------+
//| replace =true parameter allows the same index to be chosen more  |
//| than once, simulating the bootstrapping process.                 |    
//+------------------------------------------------------------------+
template<typename T>
void MatrixExtend::Randomize(matrix<T> &matrix_,int random_state=-1, bool replace=false)
 {
   MathSrand(random_state!=-1?random_state:GetTickCount());
  
   int ROWS=(int)matrix_.Rows(), COL=(int)matrix_.Cols();   
   
   int swap_index;
   vector<T> temp(COL);
   matrix<T> temp_m = matrix_;
   int random = 0;
   
   for (int i=0; i<ROWS; i++)
      {
        if (!replace)
          {
            swap_index = MathRand() % ROWS;
            
            temp = matrix_.Row(i);
                  
            matrix_.Row(matrix_.Row(swap_index),i);
            
            matrix_.Row(temp,swap_index);
          }
        else
          {
            random = MathRandom(1, ROWS);  
            
            temp = temp_m.Row(random-1);                      
            matrix_.Row(temp, i);
          }
      }   

 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
template <typename T>
void MatrixExtend::TrainTestSplitMatrices(matrix<T> &matrix_, matrix<T> &x_train, vector<T> &y_train, matrix<T> &x_test, vector<T> &y_test, double train_size=0.7,int random_state=-1)
  {
   ulong total = matrix_.Rows(), cols = matrix_.Cols();
   
   ulong last_col = cols-1;
   
//--- Random pseudo matrix
   
   Randomize(matrix_,random_state);
   
//---

   int train = (int)MathFloor(total*train_size);
   int test = (int)total-train;
   
   x_train.Resize(train,cols-1);
   x_test.Resize(test,cols-1);
   
   y_train.Resize(train);
   y_test.Resize(test);
   
   int train_count = 0, test_count = 0;
   
   Copy(matrix_.Col(last_col),y_train,0,train);
   Copy(matrix_.Col(last_col),y_test,train);
 
   for(ulong i=0; i<matrix_.Rows(); i++)
     {
      if(i < (ulong)train)
        {
         x_train.Row(matrix_.Row(i),train_count);
         
         train_count++;
        }
      else
        {
         x_test.Row(matrix_.Row(i),test_count);
         test_count++;
        }
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
matrix MatrixExtend::DesignMatrix(matrix &x_matrix)
  {
   matrix out_matrix(x_matrix.Rows(),x_matrix.Cols()+1);

   vector ones(x_matrix.Rows());
   ones.Fill(1);

   out_matrix.Col(ones,0);
   vector new_vector;

   for(ulong i=1; i<out_matrix.Cols(); i++)
     {
      new_vector = x_matrix.Col(i-1);
      out_matrix.Col(new_vector,i);
     }

   return (out_matrix);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
matrix MatrixExtend::OneHotEncoding(vector &v)
 {
   matrix mat = {}; 
   
//---

   vector v_classes = Unique(v);
     
//---

     mat.Resize(v.Size(),v_classes.Size());
     mat.Fill(-100);
     
     for (ulong i=0; i<mat.Rows(); i++)
        for (ulong j=0; j<mat.Cols(); j++)
           {
               if (v[i] == v_classes[j])
                  mat[i][j] = 1;
               else 
                  mat[i][j] = 0;     
           }
   
   return(mat);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void MatrixExtend::Unique(const string &Array[], string &classes_arr[])
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
           }
         else
            continue;
        }
     }
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector MatrixExtend::Unique(const vector &v)
 {
   vector temp_v = v; 
   vector v_classes={v[0]};

   for (ulong i = 0, count=0; i < v.Size(); i++) 
    {
      bool alreadyCounted = false;

      for (ulong j = 0; j < v_classes.Size(); j++) 
       {
         if (temp_v[i] == v_classes[j] && temp_v[i] != -DBL_MAX && i!=0) 
           {
             alreadyCounted = true;
             temp_v[i] = -DBL_MAX;
           }
      }

     if (!alreadyCounted) 
       {
         count++;
         v_classes.Resize(count);
         
         v_classes[count-1] = temp_v[i];
       }
    }
 
   return MatrixExtend::Sort(v_classes); //Sort the unique values in ascending order
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
template<typename T>
T MatrixExtend:: MathRandom(T mini, T maxi)
  {
     double  f  = (MathRand() / 32767.0);
     return (mini + (T)(f * (maxi - mini)));
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
template<typename T> 
vector MatrixExtend::Random(T min, T max,int size,int random_state=-1)
 {
   MathSrand(random_state!=-1?random_state:GetTickCount());
    
   vector v(size);
   
   for (ulong i=0; i<v.Size(); i++)
      v[i] = MathRandom<T>(min,max);
      
   return (v);    
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
matrix MatrixExtend::Random(double min,double max,ulong rows,ulong cols,int random_state=-1)
 {
   MathSrand(random_state!=-1?random_state:GetTickCount());
     
     matrix mat(rows,cols);
     
     for (ulong r=0; r<rows; r++)
       for (ulong c=0; c<cols; c++)
            mat[r][c] = MathRandom<double>(min,max);
     
     return (mat);
 }
//+------------------------------------------------------------------+
//|   Appends vector v1 to the end of vector v2                      |
//+------------------------------------------------------------------+
vector MatrixExtend::concatenate(vector &v1, vector &v2)
 {
   vector v_out = v1; 
   
   v_out.Resize(v1.Size()+v2.Size());
   
   for (ulong i=0; i<v1.Size(); i++)
      v_out[i] = v1[i]; 
   
   for (ulong i=v1.Size(),index =0; i<v_out.Size(); i++, index++)
       v_out[i] = v2[index]; 
   
   return (v_out); 
 }
//+------------------------------------------------------------------+
//|   Appends matrix mat1 to the end of mat2                         |
//+------------------------------------------------------------------+
matrix MatrixExtend::concatenate(matrix &mat1, matrix &mat2, int axis = 0)
 {
     matrix m_out = {};

     if ((axis == 0 && mat1.Cols() != mat2.Cols() && mat1.Cols()>0) || (axis == 1 && mat1.Rows() != mat2.Rows() && mat1.Rows()>0)) 
       {
         Print(__FUNCTION__, "Err | Dimensions mismatch for concatenation");
         return m_out;
       }

     if (axis == 0) {
         m_out.Resize(mat1.Rows() + mat2.Rows(), MathMax(mat1.Cols(), mat2.Cols()));

         for (ulong row = 0; row < mat1.Rows(); row++) {
             for (ulong col = 0; col < m_out.Cols(); col++) {
                 m_out[row][col] = mat1[row][col];
             }
         }

         for (ulong row = 0; row < mat2.Rows(); row++) {
             for (ulong col = 0; col < m_out.Cols(); col++) {
                 m_out[row + mat1.Rows()][col] = mat2[row][col];
             }
         }
     } else if (axis == 1) {
         m_out.Resize(MathMax(mat1.Rows(), mat2.Rows()), mat1.Cols() + mat2.Cols());

         for (ulong row = 0; row < m_out.Rows(); row++) {
             for (ulong col = 0; col < mat1.Cols(); col++) {
                 m_out[row][col] = mat1[row][col];
             }

             for (ulong col = 0; col < mat2.Cols(); col++) {
                 m_out[row][col + mat1.Cols()] = mat2[row][col];
             }
         }
     }
   return m_out;
 }
//+------------------------------------------------------------------+
//|   Concatenates the vector to a matrix, axis =0 along the rows    |
//|   while axis =1 along the colums concatenation
//+------------------------------------------------------------------+
template<typename T>
matrix<T> MatrixExtend::concatenate(matrix<T> &mat, vector<T> &v, int axis=1)
 {
   matrix<T> ret= mat;
     
   ulong new_rows, new_cols;
   
   if (axis == 0) //place it along the rows
    {
      if (mat.Cols() == 0)
        mat.Resize(mat.Rows(), v.Size());
        
      new_rows = mat.Rows()+1; new_cols = mat.Cols();
                 
      if (v.Size() != new_cols)
        {
          Print(__FUNCTION__," Dimensions don't match the vector v needs to have the same size as the number of columns in the original matrix");
          return ret;
        }
      
      ret.Resize(new_rows, new_cols);
      ret.Row(v, new_rows-1);
    }
   else if (axis == 1)
     {
         if (mat.Rows() == 0)
           mat.Resize(v.Size(), mat.Cols());
           
        new_rows = mat.Rows(); new_cols = mat.Cols()+1;
        
        if (v.Size() != new_rows)
          {
            Print(__FUNCTION__," Dimensions don't match the vector v needs to have the same size as the number of rows in the original matrix");
            return ret;
          }
        
        ret.Resize(new_rows, new_cols);
        ret.Col(v, new_cols-1);
     }
   else 
     {
       Print(__FUNCTION__," Axis value Can either be 0 or 1");
       return ret;
     }

//---
   return ret;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
template<typename T>
bool MatrixExtend::Copy(const vector<T> &src, vector<T> &dst,ulong src_start,ulong total=WHOLE_ARRAY)
 {
   if (total == WHOLE_ARRAY)
      total = src.Size()-src_start;
   
   if ( total <= 0 || src.Size() == 0)
    {
       printf("%s Can't copy a vector | Size %d total %d src_start %d ",__FUNCTION__,src.Size(),total,src_start);
       return (false);
    }
   
   dst.Resize(total);
   dst.Fill(0);
   
   for (ulong i=src_start, index =0; i<total+src_start; i++)
      {   
          dst[index] = src[i];         
          index++;
      }
   return (true);
 }
//+------------------------------------------------------------------+
//| Searches for a value in a vector | Returns all the index where   |
//| Such values was located                                          |
//+------------------------------------------------------------------+
template<typename T>
vector MatrixExtend::Search(const vector<T> &v, T value)
 {
   vector<T> v_out ={};
   
   for (ulong i=0, count =0; i<v.Size(); i++)
     { 
      if (value == v[i])
        {
          count++;
          
          v_out.Resize(count);    
          
          v_out[count-1] = (T)i;
        }
     }   
    return v_out;
 }
//+------------------------------------------------------------------+
//| Finds the unique values in a vector and returns a vector of      |
//| the number of values found for each unique value                 |
//+------------------------------------------------------------------+
vector MatrixExtend::Unique_count(vector &v)
 {
  vector classes = MatrixExtend::Unique(v);
  vector keys(classes.Size());
  
   for (ulong i=0; i<classes.Size(); i++)
     keys[i] = (int)Search(v, classes[i]).Size();
    
  return keys;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
template<typename T>
vector MatrixExtend::Sort(vector<T> &v,ENUM_SORT_MODE sort_mode=SORT_ASCENDING)
 {
   T arr[];
   vector temp = v;
   temp.Swap(arr);
   
   if (!ArraySort(arr))
     printf("%s Failed to sort this vector Err=%d",__FUNCTION__,GetLastError());
   
   switch(sort_mode)
     {
      case  SORT_ASCENDING:
        temp = MatrixExtend::ArrayToVector(arr);  
        break;
      case SORT_DESCENDING:
        temp = MatrixExtend::ArrayToVector(arr);  
        MatrixExtend::Reverse(temp);
        break;
      default:
        printf("%s Unknown sort mode");
        break;
     }
   return temp;   
 }
//+------------------------------------------------------------------+
//| Returns the Sorted Argsuments in either ascending order or       |
//|  descending order                                                |
//+------------------------------------------------------------------+
template<typename T>
vector MatrixExtend::ArgSort(vector<T> &v)
 {   
//---

    ulong size = v.Size();
    vector args(size);
    
    // Initialize args array with sequential values
    for (ulong i = 0; i < size; i++)
        args[i] = (int)i;

    // Perform selection sort on args based on array values
    for (ulong i = 0; i < size - 1; i++)
    {
        ulong minIndex = i;
        for (ulong j = i + 1; j < size; j++)
        {
            if (v[(int)args[j]] < v[(int)args[minIndex]])
                minIndex = j;
        }

        // Swap args
        int temp = (int)args[i];
        args[i] = args[minIndex];
        args[minIndex] = temp;
    }
   
  return args;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
template<typename T>
void MatrixExtend::Reverse(vector<T> &v)
 {
  vector<T> v_temp = v;
  
   for (ulong i=0, j=v.Size()-1; i<v.Size(); i++, j--)
        v[i] = v_temp[j];
        
   ZeroMemory(v_temp);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
template<typename T>
void MatrixExtend::Reverse(matrix<T> &mat)
 {
   matrix<T> temp_mat = mat;
   
   for (ulong i=0, j=mat.Rows()-1; i<mat.Rows(); i++, j--)
      mat.Row(mat.Row(j), i); 
 }
//+------------------------------------------------------------------+
//|   Hadamard product --> is a binary operation that takes two      |
//|    matrices of the same dimensions and produces another matrix   |
//|   of the same dimension as the operands. | This operation is     |
//|  widely known as element wise multiplication                     |
//+------------------------------------------------------------------+
matrix MatrixExtend::HadamardProduct(matrix &a,matrix &b)
 {  
  matrix c = {};
  if (a.Rows() != b.Rows() || a.Cols() != b.Cols())
    {
      Print("Cannot calculate Hadamard product | matrix a and b are not having the same size ");
      return c;
    }
//---
         
    return a*b;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+


string MatrixExtend::CalcTimeElapsed(double seconds)
  {
   string time_str = "";

   uint minutes=0, hours=0;

   if(seconds >= 60)
      time_str = StringFormat("%d Minutes and %.3f Seconds ",minutes=(int)round(seconds/60.0), ((int)seconds % 60));
   if(minutes >= 60)
      time_str = StringFormat("%d Hours %d Minutes and %.3f Seconds ",hours=(int)round(minutes/60.0), minutes, ((int)seconds % 60));
   else
      time_str = StringFormat("%.3f Seconds ",seconds);

   return time_str;
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
matrix MatrixExtend::DBtoMatrix(int db_handle, string table_name,string &column_names[],int total=WHOLE_ARRAY)
 {
  matrix matrix_ = {};
  
  
  #ifdef DEBUG_MODE
     Print("---> loading database ");
  #endif 
  
  if(db_handle == INVALID_HANDLE)
     {
      printf("db handle failed, Err = %d",GetLastError());
      DatabaseClose(db_handle);
      return matrix_;
     }

//---

   string sql =  "SELECT * FROM "+table_name;
   int request = DatabasePrepare(db_handle,sql);     
   
   ulong cols = DatabaseColumnsCount(request), rows =0;
   
   ArrayResize(column_names,(int)cols);

//---

   matrix_.Resize(cols,0); 
    
    double time_start = GetMicrosecondCount()/(double)1e6, time_stop=0; //Elapsed time 
    double row_start = 0, row_stop =0;
     
    for (int j=0; DatabaseRead(request) && !IsStopped(); j++)
      {  
        
       row_start = GetMicrosecondCount()/(double)1e6; 
           
        rows = (ulong)j+1;
        matrix_.Resize(cols,rows);
         
         for (ulong k=0; k<cols; k++)
           {
             DatabaseColumnDouble(request,(int)k,matrix_[k][j]);
             
             if (j==0)  DatabaseColumnName(request,(int)k,column_names[k]);
           }
          
         if (total != WHOLE_ARRAY)
            if (j >= total)     break;
            
         #ifdef  DEBUG_MODE
            row_stop =GetMicrosecondCount()/(double)1e6;
            
            printf("Row ----> %d | Elapsed %s",j,CalcTimeElapsed(row_stop-row_start));
         #endif 
      }

//---
   
   DatabaseFinalize(request);
   DatabaseClose(db_handle);
   
   matrix_ = matrix_.Transpose(); //very crucial step
   
   #ifdef DEBUG_MODE
      time_stop = GetMicrosecondCount()/(double)1e6;
     
      printf("---> finished reading DB size=(%dx%d) | Time Elapsed %s",rows,cols,CalcTimeElapsed(time_stop-time_start)); 
     
      ArrayPrint(column_names);
      for (ulong i=0; i<5; i++)     Print(matrix_.Row(i));
   #endif 
   
   
  return matrix_;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
template<typename T>
void MatrixExtend::NormalizeDouble_(vector<T> &v,int digits=3)
 {
   for (ulong i=0; i<v.Size(); i++)
      v[i] = NormalizeDouble(v[i], digits);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
template<typename T>
void MatrixExtend::NormalizeDouble_(matrix<T> &mat,int digits=3)
 {
   for (ulong i=0; i<mat.Rows(); i++)
      for (ulong j=0; j<mat.Cols(); j++)
         mat[i][j] = NormalizeDouble(mat[i][j], digits);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void MatrixExtend::PrintShort(matrix &matrix_, ulong rows=5,int digits=5)
 {
   vector v = {};
    for (ulong i=0; i<rows; i++)
     {
        v = matrix_.Row(i);
        NormalizeDouble_(v, digits);
        
        Print(v); 
     }
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void MatrixExtend::Swap(double &var1,double &var2)
 {
   double temp_1 = var1, temp2=var2;
   
   var1 = temp2;
   var2 = temp_1;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int MatrixExtend::CopyBufferVector(int handle,int buff_num,int start_pos,int count,vector &v)
 {
   double buff_arr[];
   
   int ret = CopyBuffer(handle, buff_num, start_pos, count, buff_arr);
   v = ArrayToVector(buff_arr);
   
   return (ret);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
string MatrixExtend::Stringfy(vector &v, int digits = 2)
 {
   string str = "";
   for (ulong i=0; i<v.Size(); i++)
       str += " "+DoubleToString(v[i], digits) + " ";
   
   return (str);
 }
//+------------------------------------------------------------------+
//| a function to convert the seocnds to Hours and minutes, Useful   |
//| in measuring the time taken for operations that takes a long     |
//| time to complete, Such as reading and writing to a large csv file|
//+------------------------------------------------------------------+
string MatrixExtend::ConvertTime(double seconds)
{
    string time_str = "";
    uint minutes = 0, hours = 0;

    if (seconds >= 60)
    {
        minutes = (uint)(seconds / 60.0) ;
        seconds = fmod(seconds, 1.0) * 60;
        time_str = StringFormat("%d Minutes and %.3f Seconds", minutes, seconds);
    }
    
    if (minutes >= 60)
    {
        hours = (uint)(minutes / 60.0);
        minutes = minutes % 60;
        time_str = StringFormat("%d Hours and %d Minutes", hours, minutes);
    }

    if (time_str == "")
    {
        time_str = StringFormat("%.3f Seconds", seconds);
    }

    return time_str;
}
//+------------------------------------------------------------------+
//|  Obtains a part of the matrix starting from a start_index row to |
//|   end_index row Inclusive                                        |
//+------------------------------------------------------------------+
matrix MatrixExtend::Get(const matrix &mat, ulong start_index, ulong end_index)
 {
  matrix ret_mat(MathAbs(end_index-start_index+1), mat.Cols());
  
  if (start_index >= mat.Rows())
    {
       Print(__FUNCTION__,"Error | start_index (",start_index,") is greater than or Equal to matrix Rows (",mat.Rows(),")");
       return ret_mat;
    }
    
  if (end_index > mat.Rows())
   {
       Print(__FUNCTION__,"Error | end_index (",start_index,") is greater than (",mat.Rows(),")");
       return ret_mat;
   }
  
  if (start_index > end_index)
    {
      Print(__FUNCTION__,"Error | start_index shouldn't be greater than end_index ???");
      return ret_mat;
    }
  
   for (ulong i=start_index, count =0; i<=end_index; i++, count++)
     for (ulong col=0; col<mat.Cols(); col++)
         ret_mat[count][col] = mat[i][col];
       
   return ret_mat;
 }

//+------------------------------------------------------------------+
//|  Obtains a part of the vector starting from a start_index row to |
//|   end_index row Inclusive                                        |
//+------------------------------------------------------------------+

vector MatrixExtend::Get(const vector &v, ulong start_index, ulong end_index)
 {
  vector ret_vec(MathAbs(end_index-start_index+1));
  
  if (start_index >= v.Size())
    {
       Print(__FUNCTION__,"Error | start_index (",start_index,") is greater than or Equal to matrix Rows (",v.Size(),")");
       return ret_vec;
    }
    
  if (end_index > v.Size())
   {
       Print(__FUNCTION__,"Error | end_index (",start_index,") is greater than (",v.Size(),")");
       return ret_vec;
   }
  
  if (start_index > end_index)
    {
      Print(__FUNCTION__,"Error | start_index shouldn't be greater than end_index ???");
      return ret_vec;
    }
  
  for (ulong i=start_index, count=0; i<=end_index; i++, count++)
     ret_vec[count] = v[i];
       
   return ret_vec;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
matrix MatrixExtend::Sign(matrix &x)
 {
   matrix ret_matrix = x;
   
    for (ulong i=0; i<x.Cols(); i++)
     ret_matrix.Col(Sign(x.Col(i)) ,i);
   
   return ret_matrix;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector MatrixExtend::Sign(vector &x)
 {
   vector v(x.Size());
   for (ulong i=0; i<x.Size(); i++)
     v[i] = Sign(x[i]);
     
  return v;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
