//+------------------------------------------------------------------+
//|                                                 matrix_utils.mqh |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
//+------------------------------------------------------------------+
//| defines                                                          |
//+------------------------------------------------------------------+
class CMatrixutils
  {
public:
                     CMatrixutils(void);
                    ~CMatrixutils(void);

   string            csv_header[];

   void              WriteCsv(string name, matrix &Matrix, int digits=5);
   matrix            ReadCsv(string file_name,string delimiter=",");
   matrix            VectorToMatrix(const vector &v);
   vector            MatrixToVector(const matrix &mat);
   vector            ArrayToVector(const double &Arr[]);
   bool              VectorToArray(const vector &v,double &arr[]);
   void              MatrixRemoveCol(matrix &mat, ulong col);
   void              MatrixRemoveMultCols(matrix &mat, int &cols[]);
   void              MatrixRemoveRow(matrix &mat,ulong row);
   void              VectorRemoveIndex(vector &v, ulong index);
   void              XandYSplitMatrices(const matrix &matrix_,matrix &xmatrix,vector &y_vector,int y_index=-1);
   void              TrainTestSplitMatrices(const matrix &matrix_,matrix &TrainMatrix, matrix &TestMatrix,double train_size = 0.7);
   matrix            DesignMatrix(matrix &x_matrix);
   matrix            OneHotEncoding(vector &v, uint &classes);
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
matrix CMatrixutils::VectorToMatrix(const vector &v)
  {   
   matrix mat = {};
   
   if (!mat.Assign(v))
      Print("Failed to turn the vector to a 1xn matrix");
   
   return(mat);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CMatrixutils::MatrixToVector(const matrix &mat)
  {
    vector v = {};
    
    if (!v.Assign(mat))
      Print("Failed to turn the matrix to a vector");
    
    return(v);
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CMatrixutils::MatrixRemoveCol(matrix &mat, ulong col)
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
void CMatrixutils::MatrixRemoveMultCols(matrix &mat,int &cols[])
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
            MatrixRemoveCol(mat,i);
        }
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CMatrixutils::MatrixRemoveRow(matrix &mat,ulong row)
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
void CMatrixutils::WriteCsv(string name, matrix &Matrix, int digits=5)
  {
   FileDelete(name);
   int handle = FileOpen(name,FILE_WRITE|FILE_CSV|FILE_ANSI,",",CP_UTF8);

   ResetLastError();

   if(handle == INVALID_HANDLE)
      printf("Invalid %s handle Error %d ",name,GetLastError());
   else
     {       
      string concstring;
      vector row;
      FileSeek(handle,0,SEEK_SET);

      for(ulong i=0; i<Matrix.Rows(); i++)
        {
         ZeroMemory(concstring);

         row = Matrix.Row(i);
         for(ulong j=0, cols =1; j<row.Size(); j++, cols++)
           {
            concstring += (string)NormalizeDouble(row[j],digits) + (cols == Matrix.Cols() ? "" : ",");
           }

         //Print(concstring);

         FileSeek(handle,0,SEEK_END);
         FileWrite(handle,concstring);
        }
     }
   FileClose(handle);
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

      while(!FileIsEnding(handle))
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
vector CMatrixutils::ArrayToVector(const double &Arr[])
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
void CMatrixutils::XandYSplitMatrices(const matrix &matrix_,matrix &xmatrix,vector &y_vector,int y_index=-1)
  {
   ulong value = y_index;

   if(y_index == -1)
      value = matrix_.Cols()-1;  //Last column in the matrix

//---

   y_vector = matrix_.Col(value);
   xmatrix.Copy(matrix_);

   MatrixRemoveCol(xmatrix, value); //Remove the y column
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CMatrixutils::TrainTestSplitMatrices(const matrix &matrix_,matrix &TrainMatrix,matrix &TestMatrix,double train_size=0.7)
  {
   ulong total = matrix_.Rows(), cols = matrix_.Cols();

   int train = (int)MathCeil(total*train_size);
   int test = (int)MathFloor(total-train);
   
   TrainMatrix.Resize(train,cols);
   TestMatrix.Resize(test,cols);

   int train_count = 0, test_count = 0;

   for(ulong i=0; i<matrix_.Rows(); i++)
     {
      if(i < (ulong)train)
        {
         TrainMatrix.Row(matrix_.Row(i),train_count);
         train_count++;
        }
      else
        {
         TestMatrix.Row(matrix_.Row(i),test_count);
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
matrix CMatrixutils::OneHotEncoding(vector &v, uint &classes)
 {
   matrix mat = {}; 
   
//---

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

               temp_t[j] = -1000; //modify so that it can no more be counted
              }
            else
               break; 
           }
         else
            continue;
        }
     }
     
//---

     classes = (uint)v_classes.Size();
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
