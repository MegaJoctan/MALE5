//+------------------------------------------------------------------+
//|                                                 matrix_utils.mqh |
//|                                  Copyright 2022, Omega Joctan  . |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, Omega Joctan"
#property link      "https://www.mql5.com/en/users/omegajoctan"
//+------------------------------------------------------------------+
//| defines                                                          |
//+------------------------------------------------------------------+
#ifndef COLS
   #define  COLS 1
#endif 

class CMatrixutils
  {
  
private:
   void              Classes(const string &Array[], string &classes_arr[]);
   vector            LabelEncoder(const string &Arr[]);
   int               CSVOpen(string filename,string delimiter);
   
   double            MathRandom(double mini, double maxi);
   int               MathRandom(int mini, int maxi);
   string            CalcTimeElapsed(double seconds);
   void              Swap(double &var1, double &var2);
public:
                     CMatrixutils(void);
                    ~CMatrixutils(void);

   string            csv_header[];
   
   bool              WriteCsv(string csv_name, matrix &matrix_, string &header[] , int digits=5);
   matrix            ReadCsv(string file_name,string delimiter=",");
   matrix            ReadCsvEncode(string file_name, string delimiter=",");
   bool              ReadCsvAsStrings(string file_name,string &array[][COLS], string delimiter=",");
   matrix            VectorToMatrix(const vector &v, ulong cols=1);
   vector            MatrixToVector(const matrix &mat);
  
   vector            ArrayToVector(const double &Arr[]);  
   vector            ArrayToVector(const int &Arr[]);
   
   bool              VectorToArray(const vector &v,double &arr[]);
   bool              VectorToArray(const vector &v,int &arr[]);
   
   void              RemoveCol(matrix &mat, ulong col);
   void              RemoveMultCols(matrix &mat, int &cols[]);
   void              RemoveRow(matrix &mat,ulong row);
   void              VectorRemoveIndex(vector &v, ulong index);  
   void              XandYSplitMatrices(const matrix &matrix_,matrix &xmatrix,vector &y_vector,int y_column=-1);
   void              TrainTestSplitMatrices(matrix &matrix_,matrix &x_train,vector &y_train,matrix &x_test, vector &y_test,double train_size=0.7,int random_state=-1);
   matrix            DesignMatrix(matrix &x_matrix);              
   matrix            OneHotEncoding(vector &v);    //ONe hot encoding 
   vector            Classes(vector &v);           //Identifies classes available in a vector
  
   vector            Random(int min, int max, int size,int random_state=-1);          //Generates a random integer vector of a given size
   vector            Random(double min, double max, int size,int random_state=-1);    //Generates a random vector of a given size
   matrix            Random(double min, double max, ulong rows, ulong cols, int random_state=-1);
   
   vector            Append(vector &v1, vector &v2);              //Appends v2 to vector 1
   bool              Copy(const vector &src,vector &dst,ulong src_start,ulong total=WHOLE_ARRAY);
   
   vector            Search(const vector &v, int value);          //Searches a specific integer value in a vector and returns all the index it has been found
   vector            Search(const vector &v,double value);
   
   void              ReverseOrder(vector &v);
   matrix            DBtoMatrix(int db_handle, string table_name,string &column_names[],int total=WHOLE_ARRAY);
   
   matrix            HadamardProduct(matrix &a, matrix &b);
   
   void              Shuffle(vector &v, int random_state=-1);
   void              Shuffle(matrix &matrix_,int random_state=-1);
   void              NormalizeVector(vector<double> &v, int digits=3);
   void              PrintShort(matrix &matrix_,ulong rows=5);
   void              SortAscending(vector &v);
   void              SortDesending(vector &v);
   int               CopyBufferVector(int handle, int buff_num, int start_pos,int count, vector &v);
   string            Stringfy(vector &v, int digits = 2);

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
    ResetLastError();
    
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
void CMatrixutils::RemoveCol(matrix &mat, ulong col)
  {
   matrix new_matrix(mat.Rows(),mat.Cols()-1); //Remove the one Column

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
void CMatrixutils::RemoveMultCols(matrix &mat,int &cols[])
  {
   ulong size = (int)ArraySize(cols);

   if(size > mat.Cols())
     {
      Print(__FUNCTION__," Columns to remove can't be more than the available columns");
      return;
     }


   vector zeros(mat.Rows());
   zeros.Fill(0);

   for(ulong i=0; i<size; i++)
      for(ulong j=0; j<mat.Cols(); j++)
        {
         if(cols[i] == j)
            mat.Col(zeros,j);
        }

//---

   vector column_vector;
   for(ulong A=0; A<mat.Cols(); A++)
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
bool CMatrixutils::WriteCsv(string csv_name, matrix &matrix_, string &header[], int digits=5)
  {
   FileDelete(csv_name);
   int handle = FileOpen(csv_name,FILE_WRITE|FILE_CSV|FILE_ANSI,",",CP_UTF8);

   ResetLastError();

   if(handle == INVALID_HANDLE)
     {
       printf("Invalid %s handle Error %d ",csv_name,GetLastError());
       return (false);
     }
   else
     {       
      string concstring;
      vector row = {};
      
      //FileSeek(handle,0,SEEK_SET);
      
      vector colsinrows = matrix_.Row(0);
      
      if (ArraySize(header) != (int)colsinrows.Size())
         {
            Print("header and columns from the matrix vary is size ");
            return false;
         }

//---

      string header_str = "";
      for (int i=0; i<ArraySize(header); i++)
         header_str += header[i] + (i+1 == colsinrows.Size() ? "" : ",");
      
      FileWrite(handle,header_str);
      
      FileSeek(handle,0,SEEK_SET);
      
      for(ulong i=0; i<matrix_.Rows(); i++)
        {
         ZeroMemory(concstring);

         row = matrix_.Row(i);
         for(ulong j=0, cols =1; j<row.Size(); j++, cols++)
           {
            concstring += (string)NormalizeDouble(row[j],digits) + (cols == matrix_.Cols() ? "" : ",");
           }

         FileSeek(handle,0,SEEK_END);
         FileWrite(handle,concstring);
        }
     }
   FileClose(handle);
   
   return (true);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
matrix CMatrixutils::ReadCsv(string file_name,string delimiter=",")
  {
   matrix mat_ = {};

   int rows_total=0;

   int handle = FileOpen(file_name,FILE_READ|FILE_CSV|FILE_ANSI,delimiter);

   ResetLastError();

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
           }
        }

      rows_total = rows;

      FileClose(handle);
     }

   mat_.Resize(rows_total-1,mat_.Cols());

   return(mat_);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
matrix CMatrixutils::ReadCsvEncode(string file_name,string delimiter=",")
 {
//--- Obtaining the columns

   matrix matrix_={};

   
//--- Loading the entire matrix_ to an Array

   int csv_columns=0,  rows_total=0;
   
   int handle = CSVOpen(file_name,delimiter); 
   
   if (handle != INVALID_HANDLE)
      {
       while (!FileIsEnding(handle) && !IsStopped())
         {
           string data = FileReadString(handle);

           csv_columns++;
//---
           if (FileIsLineEnding(handle)) break;
         } 
      }
      
   FileClose(handle);
   
   if (csv_columns==0)
     {
       Print(__FUNCTION__," Couldn't read the csv file header ");
       return (matrix_);
     }
   
   ArrayResize(csv_header,csv_columns);
   
//---

   string toArr[];

    int counter=0; 
    for (int i=0; i<csv_columns; i++)
      {                    
        if ((handle = CSVOpen(file_name,delimiter)) != INVALID_HANDLE)
         {  
          int column = 0, rows=0;
          while (!FileIsEnding(handle) && !IsStopped())
            {
              string data = FileReadString(handle);

              column++;
   //---      
              if (column==i+1)
                 {                      
                     if (rows>=1 ) //Avoid the first column which contains the column's header
                       {   
                           counter++;
                           
                           ArrayResize(toArr,counter); //array size for all the columns 
                           toArr[counter-1]=data;
                           
                       }
                      else csv_header[column-1]  =  data;
                 }
   //---
              if (FileIsLineEnding(handle))
                {              
                   rows++;
                   column = 0;
                }
            } 
          rows_total += rows-1; 
        }
       FileClose(handle); 
     }

//---
    
    ulong mat_cols = 0,mat_rows = 0;
    
    
    if (ArraySize(toArr) % csv_columns !=0)
     printf("This CSV file named %s has unequal number of columns = %d and rows %d Its size = %d",file_name,csv_columns,ArraySize(toArr) / csv_columns,ArraySize(toArr));
    else 
     {
        mat_cols = (ulong)csv_columns;       
        mat_rows = (ulong)ArraySize(toArr)/mat_cols;
     }
   
   //ArrayPrint(toArr);

//--- Encoding the CSV

     matrix_.Resize(mat_rows,mat_cols);    
      
//---

     string Arr[];
      
     int start =0;
      
     vector v = {};
      
       for (ulong j=0; j<mat_cols; j++)
         {
            ArrayCopy(Arr,toArr,0,start,(int)mat_rows);
          
            v = LabelEncoder(Arr);
               
            matrix_.Col(v, j);
            
            start += (int)mat_rows;
         }
      
   return (matrix_);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CMatrixutils::LabelEncoder(const string &Arr[])
 {
   string classes[];
   
   vector Encoded((ulong)ArraySize(Arr));
   
   Classes(Arr,classes);
   
   //ArrayPrint(classes);
    
    for (ulong A=0; A<classes.Size(); A++)
      for (ulong i=0; i<Encoded.Size(); i++)
        {
           if (classes[A] == Arr[i])
              Encoded[i] = (int)A;
        }
    
   return Encoded;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CMatrixutils::ReadCsvAsStrings(string file_name,string &array[][COLS], string delimiter=",")
 {
   int rows_total=0;

   int handle = FileOpen(file_name,FILE_READ|FILE_CSV|FILE_ANSI,delimiter);

   ResetLastError();

   if(handle == INVALID_HANDLE)
     {
         printf("Invalid %s handle Error %d ",file_name,GetLastError());
         Print(GetLastError()==0?" TIP | File Might be in use Somewhere else or in another Directory":"");
         return false;
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
           }

         if(rows>0)  //Avoid the first column which contains the column's header
            array[rows-1,column] = data;

         column++;

         //---

         if(FileIsLineEnding(handle))
           {
            rows++;
            
            ArrayResize(array,rows);

            column = 0;
           }
        }

      rows_total = rows;

      FileClose(handle);
     }
   
   ArrayResize(array,rows_total-1);
   
   return (true);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CMatrixutils::ArrayToVector(const double &Arr[])
  {
   vector v = {};

   v.Assign(Arr);

   return (v);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

vector CMatrixutils::ArrayToVector(const int &Arr[])
  {
   vector v = {};

   v.Assign(Arr);

   return (v);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CMatrixutils::VectorToArray(const vector &v,double &arr[])
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

bool CMatrixutils::VectorToArray(const vector &v,int &arr[])
  {
   ArrayResize(arr,(int)v.Size());

   if(ArraySize(arr) == 0)
      return(false);

   for(ulong i=0; i<v.Size(); i++)
      arr[i] = (int)v[i];

   return(true);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CMatrixutils::XandYSplitMatrices(const matrix &matrix_,matrix &xmatrix,vector &y_vector,int y_column=-1)
  {
   y_column = int( y_column==-1 ? matrix_.Cols()-1 : y_column);

   y_vector = matrix_.Col(y_column);
   xmatrix.Copy(matrix_);

   RemoveCol(xmatrix, y_column); //Remove the y column
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CMatrixutils::Shuffle(vector &v, int random_state=-1)
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
void CMatrixutils::Shuffle(matrix &matrix_,int random_state=-1)
 {
   if (random_state != -1)
     MathSrand(random_state);
  
   int ROWS=(int)matrix_.Rows(), COL=(int)matrix_.Cols();   
   
   int swap_index;
   vector temp(COL);
   
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
void CMatrixutils::TrainTestSplitMatrices(matrix &matrix_,matrix &x_train,vector &y_train,matrix &x_test, vector &y_test,double train_size=0.7,int random_state=-1)
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

   vector v_classes = Classes(v);
     
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
void CMatrixutils::Classes(const string &Array[],string &classes_arr[])
 {
   string temp_arr[];

   ArrayResize(classes_arr,1);
   ArrayCopy(temp_arr,Array);
   
   classes_arr[0] = Array[0];
   
   for(int i=0, count =1; i<ArraySize(Array); i++)  //counting the different neighbors
     {
      for(int j=0; j<ArraySize(Array); j++)
        {
         if(Array[i] == temp_arr[j] && temp_arr[j] != "nan")
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

               temp_arr[j] = "nan"; //modify so that it can no more be counted
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
vector CMatrixutils::Classes(vector &v)
 {
   vector temp_t = v, v_classes = {v[0]};

   for(ulong i=0, count =1; i<v.Size(); i++)  //counting the different neighbors
     {
      for(ulong j=0; j<v.Size(); j++)
        {
         if(v[i] == temp_t[j] && temp_t[j] != -1000)
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

               temp_t[j] = -1000; //modify so that it can no more be counted
              }
            else
               break;
            //Print("t vectors vector ",v);
           }
         else
            continue;
        }
     } 
   return v_classes;
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
//|                                                                  |
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
//|                                                                  |
//+------------------------------------------------------------------+
bool CMatrixutils::Copy(const vector &src,vector &dst,ulong src_start,ulong total=WHOLE_ARRAY)
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
//|                                                                  |
//+------------------------------------------------------------------+
vector CMatrixutils::Search(const vector &v,int value)
 {
   vector v_out ={};
   
   for (ulong i=0, count =0; i<v.Size(); i++)
      if (value == v[i])
        {
          count++;
          
          v_out.Resize(count);    
          
          v_out[count-1] = (int)i;
        }
    return v_out;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

vector CMatrixutils::Search(const vector &v,double value)
 {
   vector v_out ={};
   
   for (ulong i=0, count =0; i<v.Size(); i++)
      if (value == v[i])
        {
          count++;
          
          v_out.Resize(count);    
          
          v_out[count-1] = (int)i;
        }
    return v_out;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

void CMatrixutils::ReverseOrder(vector &v)
 {
  vector v_temp = v;
  
   for (ulong i=0, j=v.Size()-1; i<v.Size(); i++, j--)
        v[i] = v_temp[j];
        
   ZeroMemory(v_temp);
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
void CMatrixutils::PrintShort(matrix &matrix_, ulong rows=5)
 {
    for (ulong i=0; i<rows; i++)
      Print(matrix_.Row(i)); 
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
   ReverseOrder(v); 
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
//|                                                                  |
//+------------------------------------------------------------------+
