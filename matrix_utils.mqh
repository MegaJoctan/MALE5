//+------------------------------------------------------------------+
//|                                                 matrix_utils.mqh |
//|                                  Copyright 2022, Omega Joctan  . |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, Omega Joctan"
#property link      "https://www.mql5.com/en/users/omegajoctan"

#include <MALE5\preprocessing.mqh>

//+------------------------------------------------------------------+
//| defines                                                          |
//+------------------------------------------------------------------+
#ifndef COLS
   #define  COLS 1
#endif 

class CMatrixutils
  {
  CLabelEncoder encoder;
  
private:
   int               CSVOpen(string filename,string delimiter);
   
   double            MathRandom(double mini, double maxi);
   int               MathRandom(int mini, int maxi);
   string            CalcTimeElapsed(double seconds);
   void              Swap(double &var1, double &var2);
   string            ConvertTime(double seconds);
   
public:
                     CMatrixutils(void);
                    ~CMatrixutils(void);

   string            csv_header[];
   
   template <typename T>
   bool              WriteCsv(string csv_name, matrix<T> &matrix_, string &header[] ,bool common=false, int digits=5);
   template <typename T>
   bool              WriteCsv(string csv_name, matrix<T> &matrix_, string header_string,bool common=false, int digits=5);
   matrix            ReadCsv(string file_name,string delimiter=",",bool common=false);
   
   matrix            VectorToMatrix(const vector &v, ulong cols=1);
   vector            MatrixToVector(const matrix &mat);
   
   template<typename T>
   vector            ArrayToVector(const T &Arr[]);     
   template<typename T>
   bool              VectorToArray(const vector &v,T &arr[]);
   template<typename T>
   void              RemoveCol(matrix<T> &mat, ulong col);
   void              RemoveMultCols(matrix &mat, int &cols[]);
   void              RemoveMultCols(matrix &mat, int from, int total=WHOLE_ARRAY);
   void              RemoveRow(matrix &mat,ulong row);
   void              VectorRemoveIndex(vector &v, ulong index);  
   template<typename T>
   void              XandYSplitMatrices(const matrix<T> &matrix_, matrix<T> &xmatrix, vector<T> &y_vector,int y_column=-1);
   template <typename T>
   void              TrainTestSplitMatrices(matrix<T> &matrix_, matrix<T> &x_train, vector<T> &y_train, matrix<T> &x_test, vector<T> &y_test, double train_size=0.7,int random_state=-1);
   matrix            DesignMatrix(matrix &x_matrix);              
   matrix            OneHotEncoding(vector &v);    //ONe hot encoding 
   
   void              Unique(const string &Array[], string &classes_arr[]);
   vector            Unique(vector &v);           //Identifies classes available in a vector
  
   vector            Random(int min, int max, int size,int random_state=-1);          //Generates a random integer vector of a given size
   vector            Random(double min, double max, int size,int random_state=-1);    //Generates a random vector of a given size
   matrix            Random(double min, double max, ulong rows, ulong cols, int random_state=-1);
   
   vector            Append(vector &v1, vector &v2);              //Appends v2 to vector 1
   template<typename T>
   matrix<T>         concatenate(matrix<T> &mat, vector<T> &v, int axis=1);
   matrix            Append(matrix &mat1, matrix &mat2);
   template<typename T>
   bool              Copy(const vector<T> &src, vector<T> &dst, ulong src_start,ulong total=WHOLE_ARRAY);
   
   template<typename T>
   vector            Search(const vector<T> &v, T value);
   
   template<typename T>
   void              Reverse(vector<T> &v);
   template<typename T>
   void              Reverse(matrix<T> &mat);
   matrix            DBtoMatrix(int db_handle, string table_name,string &column_names[],int total=WHOLE_ARRAY);
   
   matrix            HadamardProduct(matrix &a, matrix &b);
   
   template<typename T>
   void              Shuffle(vector<T> &v, int random_state=-1);
   template<typename T>
   void              Shuffle(matrix<T> &matrix_,int random_state=-1);
   void              NormalizeVector(vector<double> &v, int digits=3);
   void              PrintShort(matrix &matrix_,ulong rows=5, int digits=5);
   void              SortAscending(vector &v);
   void              SortDesending(vector &v);
   int               CopyBufferVector(int handle, int buff_num, int start_pos,int count, vector &v);
   string            Stringfy(vector &v, int digits = 2);
   matrix            Zeros(ulong rows, ulong cols) { matrix ret_mat(rows, cols); return(ret_mat.Fill(0.0)); }
   vector            Zeros(ulong size) { vector ret_v(size); return( ret_v.Fill(0.0)); }
   matrix            Get(const matrix &mat, ulong start_index, ulong end_index);
   vector            Get(const vector &v, ulong start_index, ulong end_index);
   vector            Unique_count(vector &v);
   template<typename T>
   vector            Sort(vector<T> &v);
  }; 
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CMatrixutils::CMatrixutils(void)
  {
    
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CMatrixutils::~CMatrixutils(void)
  {
   ZeroMemory(csv_header);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int CMatrixutils::CSVOpen(string filename,string delimiter)
 { 
    
    
    int csv_handle  = FileOpen(filename,FILE_READ|FILE_CSV|FILE_ANSI,delimiter,CP_UTF8); 

    if (csv_handle == INVALID_HANDLE)
      {
         Print(__FUNCTION__," Invalid csv handle err=",GetLastError());
         return(INVALID_HANDLE);
      }
   return (csv_handle);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
matrix CMatrixutils::VectorToMatrix(const vector &v, ulong cols=1)
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
vector CMatrixutils::MatrixToVector(const matrix &mat)
  {
    vector v = {};
    
    if (!v.Assign(mat))
      Print(__FUNCTION__," Failed to turn the matrix to a vector rows ",mat.Rows()," cols ",mat.Cols());
    
    v.Swap(v);
    
    return(v);
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
template<typename T>
void CMatrixutils::RemoveCol(matrix<T> &mat, ulong col)
  {
   matrix<T> new_matrix(mat.Rows(),mat.Cols()-1); //Remove the one Column
   if (col > mat.Cols())
     {
       Print(__FUNCTION__," column out of range");
       return;
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
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CMatrixutils::RemoveMultCols(matrix &mat, int &cols[])
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
            RemoveCol(mat,i);
        }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

void CMatrixutils::RemoveMultCols(matrix &mat, int from, int total=WHOLE_ARRAY)
 {
   
   total = total==WHOLE_ARRAY ? (int)mat.Cols()-from : total;
   
   if(total > (int)mat.Cols())
     {
      Print(__FUNCTION__," Columns to remove can't be more than the available columns");
      return;
     }
   
   
   Print("From ",from," total ",total);

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
            RemoveCol(mat,i);
    }
 }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CMatrixutils::RemoveRow(matrix &mat,ulong row)
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
void CMatrixutils::VectorRemoveIndex(vector &v, ulong index)
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
bool CMatrixutils::WriteCsv(string csv_name, matrix<T> &matrix_, string &header[], bool common=false, int digits=5)
  {
   string header_str = "";
   for (int i=0; i<ArraySize(header); i++)
      header_str += header[i] + (i+1 == ArraySize(header)) ? "" : ",";
      
   return WriteCsv(csv_name, matrix_, header_str, common, digits);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
template <typename T>
bool CMatrixutils::WriteCsv(string csv_name, matrix<T> &matrix_, string header_string, bool common=false, int digits=5)
  {
   FileDelete(csv_name);
   int handle = FileOpen(csv_name,FILE_WRITE|FILE_CSV|FILE_ANSI|(common?FILE_COMMON:FILE_ANSI),",",CP_UTF8);


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
//|                                                                  |
//+------------------------------------------------------------------+
matrix CMatrixutils::ReadCsv(string file_name,string delimiter=",",bool common=false)
  {
   matrix mat_ = {};
   int rows_total=0;
   
   int handle = FileOpen(file_name,FILE_SHARE_READ|FILE_CSV|FILE_ANSI|(common?FILE_COMMON:FILE_ANSI),delimiter);
   
   CLabelEncoder encoder_column[];
   
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
            ArrayResize(csv_header,column+1);
            csv_header[column] = data;
            
            ArrayResize(encoder_column, column+1);
           }
         
         
         if(rows>0)  //Avoid the first column which contains the column's header
            mat_[rows-1,column] = (double(data));
         
         
         column++;

         //---

         if(FileIsLineEnding(handle))
           {
            rows++;

            mat_.Resize(rows,column);
            
            column = 0;
            
            current_time = GetTickCount();
            Comment("Reading ",file_name," record = ",rows," Time taken | ",ConvertTime((current_time - time_start) / 1000.0));
           }
        }

      rows_total = rows;

      FileClose(handle);
     }
   
   Comment("");
   
   mat_.Resize(rows_total-1,mat_.Cols());

   return(mat_);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
template<typename T>
vector CMatrixutils::ArrayToVector(const T &Arr[])
  {
   vector v = {};

   if (!v.Assign(Arr))
     Print("Failed to Convert vector to Array Err=",GetLastError());

   return (v);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
template<typename T>
bool CMatrixutils::VectorToArray(const vector &v, T &arr[])
  {
   ArrayResize(arr,(int)v.Size());

   if(ArraySize(arr) == 0)
      return(false);

   for(ulong i=0; i<v.Size(); i++)
      arr[i] = v[i];

   return(true);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
template<typename T>
void CMatrixutils::XandYSplitMatrices(const matrix<T> &matrix_, matrix<T> &xmatrix, vector<T> &y_vector,int y_column=-1)
  {
   y_column = int( y_column==-1 ? matrix_.Cols()-1 : y_column);

   y_vector = matrix_.Col(y_column);
   xmatrix.Copy(matrix_);

   RemoveCol(xmatrix, y_column); //Remove the y column
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
template<typename T>
void CMatrixutils::Shuffle(vector<T> &v, int random_state=-1)
 {
   if (random_state != -1)
     MathSrand(random_state);
     
   int swap_index;
   double temp;
   
   int SIZE = (int)v.Size();
   
   for (int i=0; i<SIZE; i++)
      {
         swap_index = rand() % SIZE;
         
         temp = v[i];
         
         v[i] = v[swap_index];
         v[swap_index] = temp;
      }   
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
template<typename T>
void CMatrixutils::Shuffle(matrix<T> &matrix_,int random_state=-1)
 {
   if (random_state != -1)
     MathSrand(random_state);
  
   int ROWS=(int)matrix_.Rows(), COL=(int)matrix_.Cols();   
   
   int swap_index;
   vector<T> temp(COL);
   
   for (int i=0; i<ROWS; i++)
      {
         swap_index = rand() % ROWS;
         
         temp = matrix_.Row(i);
               
         matrix_.Row(matrix_.Row(swap_index),i);
         
         matrix_.Row(temp,swap_index);
      }   

 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
template <typename T>
void CMatrixutils::TrainTestSplitMatrices(matrix<T> &matrix_, matrix<T> &x_train, vector<T> &y_train, matrix<T> &x_test, vector<T> &y_test, double train_size=0.7,int random_state=-1)
  {
   ulong total = matrix_.Rows(), cols = matrix_.Cols();
   
   ulong last_col = cols-1;
   
//--- Random pseudo matrix
   
   Shuffle(matrix_,random_state);
   
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
matrix CMatrixutils::DesignMatrix(matrix &x_matrix)
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
matrix CMatrixutils::OneHotEncoding(vector &v)
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
void CMatrixutils::Unique(const string &Array[], string &classes_arr[])
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
            //Print("t vectors vector ",v);
           }
         else
            continue;
        }
     }
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CMatrixutils::Unique(vector &v)
 {
   vector temp_t = v, v_classes = {v[0]};

   for(ulong i=0, count =1; i<v.Size(); i++)  //counting the different neighbors
     {
      for(ulong j=0; j<v.Size(); j++)
        {
         if(v[i] == temp_t[j] && temp_t[j] != -DBL_MAX)
           {
            bool count_ready = false;

            for(ulong n=0; n<v_classes.Size(); n++)
               if(v[i] == v_classes[n])
                    count_ready = true;

            if(!count_ready)
              {
               count++;
               v_classes.Resize(count);

               v_classes[count-1] = v[i];
               //Print("v_classes ",v_classes);

               temp_t[j] = -DBL_MAX; //modify so that it can no more be counted
              }
            else
               break;
            //Print("t vectors vector ",v);
           }
         else
            continue;
        }
     } 
   return this.Sort(v_classes); //Sort the unique values in ascending order
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CMatrixutils:: MathRandom(double mini, double maxi)
  {
     double f   = (MathRand() / 32767.0);
     return mini + (f * (maxi - mini));
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int CMatrixutils:: MathRandom(int mini, int maxi)
  {
     double f   = (MathRand() / 32767.0);
     
     return mini + int(f * (maxi - mini));
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CMatrixutils::Random(double min,double max,int size,int random_state=-1)
 {
  if (random_state != -1)
    MathSrand(random_state);
    
   vector v(size);
   
   for (ulong i=0; i<v.Size(); i++)
      v[i] = MathRandom(min,max);
      
   return (v);    
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CMatrixutils::Random(int min,int max,int size,int random_state=-1)
 {
  if (random_state != -1)
    MathSrand(random_state);
    
   vector v(size);
   
   for (ulong i=0; i<v.Size(); i++) 
      v[i] = MathRandom(min,max); 

   return (v);  
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
matrix CMatrixutils::Random(double min,double max,ulong rows,ulong cols,int random_state=-1)
 {
   if (random_state != -1)
     MathSrand(random_state);
     
     matrix mat(rows,cols);
     
     for (ulong r=0; r<rows; r++)
       for (ulong c=0; c<cols; c++)
            mat[r][c] = MathRandom(min,max);
     
     return (mat);
 }
//+------------------------------------------------------------------+
//|   Appends vector v1 to the end of vector v2                      |
//+------------------------------------------------------------------+
vector CMatrixutils::Append(vector &v1, vector &v2)
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
matrix CMatrixutils::Append(matrix &mat1, matrix &mat2)
 { 
   matrix m_out = mat1;
   
   if ((mat1.Cols()==0 || mat2.Cols()==0)? false : (mat1.Cols() != mat2.Cols()))
     {
       Print(__FUNCTION__,"Err | mat1 and mat2 must have the same number of cols");
       return m_out;
     }
   
   m_out.Resize(mat1.Rows()+mat2.Rows(), MathMax(mat1.Cols(), mat2.Cols()));
   
   
   for (ulong rows=mat1.Rows(), nrows_index=0; rows<m_out.Rows(); rows++, nrows_index++)
     for (ulong col=0; col<m_out.Cols(); col++)
         m_out[rows][col] = mat2[nrows_index][col];  
   
   return m_out;
 }
//+------------------------------------------------------------------+
//|   Concatenates the vector to a matrix, axis =0 along the rows    |
//|   while axis =1 along the colums concatenation
//+------------------------------------------------------------------+
template<typename T>
matrix<T> CMatrixutils::concatenate(matrix<T> &mat, vector<T> &v, int axis=1)
 {
   matrix<T> ret= mat;
     
   ulong new_rows, new_cols;
   
   if (axis == 0) //place it along the rows
    {
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
bool CMatrixutils::Copy(const vector<T> &src, vector<T> &dst,ulong src_start,ulong total=WHOLE_ARRAY)
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
vector CMatrixutils::Search(const vector<T> &v, T value)
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
vector CMatrixutils::Unique_count(vector &v)
 {
  vector classes = this.Unique(v);
  vector keys(classes.Size());
  
   for (ulong i=0; i<classes.Size(); i++)
     keys[i] = (int)Search(v, classes[i]).Size();
    
  return keys;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
template<typename T>
vector CMatrixutils::Sort(vector<T> &v)
 {
   T arr[];
   vector temp = v;
   temp.Swap(arr);
   
   ArraySort(arr);
   
   return this.ArrayToVector(arr);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
template<typename T>
void CMatrixutils::Reverse(vector<T> &v)
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
void CMatrixutils::Reverse(matrix<T> &mat)
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
matrix CMatrixutils::HadamardProduct(matrix &a,matrix &b)
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


string CMatrixutils::CalcTimeElapsed(double seconds)
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
matrix CMatrixutils::DBtoMatrix(int db_handle, string table_name,string &column_names[],int total=WHOLE_ARRAY)
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
void CMatrixutils::NormalizeVector(vector<double> &v,int digits=3)
 {
   for (ulong i=0; i<v.Size(); i++)
      v[i] = NormalizeDouble(v[i],digits); 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CMatrixutils::PrintShort(matrix &matrix_, ulong rows=5,int digits=5)
 {
   vector v = {};
    for (ulong i=0; i<rows; i++)
     {
        v = matrix_.Row(i);
        NormalizeVector(v, digits);
        
        Print(v); 
     }
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CMatrixutils::Swap(double &var1,double &var2)
 {
   double temp_1 = var1, temp2=var2;
   
   var1 = temp2;
   var2 = temp_1;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CMatrixutils::SortAscending(vector &v)
 { 
    ulong n = v.Size();
    for (ulong i = 0; i < n - 1; i++)
      {
        ulong minIndex = i;
        for (ulong j = i + 1; j < n; j++)
          {
            if (v[j] < v[minIndex]) {
                minIndex = j;
           }
      }
      
      Swap(v[i], v[minIndex]);
    }
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CMatrixutils::SortDesending(vector &v)
 {
   SortAscending(v);
   Reverse(v); 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int CMatrixutils::CopyBufferVector(int handle,int buff_num,int start_pos,int count,vector &v)
 {
   double buff_arr[];
   
   int ret = CopyBuffer(handle, buff_num, start_pos, count, buff_arr);
   v = ArrayToVector(buff_arr);
   
   return (ret);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
string CMatrixutils::Stringfy(vector &v, int digits = 2)
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
string CMatrixutils::ConvertTime(double seconds)
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
matrix CMatrixutils::Get(const matrix &mat, ulong start_index, ulong end_index)
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

vector CMatrixutils::Get(const vector &v, ulong start_index, ulong end_index)
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
