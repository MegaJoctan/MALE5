//+------------------------------------------------------------------+
//|                                                  matrix test.mq5 |
//|                                    Copyright 2022, Fxalgebra.com |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, Fxalgebra.com"
#property link      "https://www.mql5.com/en/users/omegajoctan"
#property version   "1.00"
#property description "This is a Test EA file for testing matrix_utils.mqh file under the MALE5 repository"

#include <MALE5\matrix_utils.mqh> 

CMatrixutils matrix_utils;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  { 
  
   matrix Matrix;

//--- 
   
   Print("---> Reading a CSV file to Matrix\n");
   Matrix = matrix_utils.ReadCsv("NASDAQ_DATA.csv"); 
 
   ArrayPrint(matrix_utils.csv_header);
   Print(Matrix);

//--- 

   string csv_name = "matrix CSV.csv";
   
   Print("\n---> Writing a Matrix to a CSV File > ",csv_name);
   string write_header[4] = {"S&P500","SMA","RSI","NASDAQ"};
   
   matrix_utils.WriteCsv(csv_name,Matrix,write_header);
   

//---

   matrix mat = {
      {1,2,3},
      {4,5,6},
      {7,8,9}
   };
 
   Print("\nMatrix to play with\n",mat,"\n");
   
//---

   Print("---> Matrix to Vector");
   vector vec = matrix_utils.MatrixToVector(mat);
   Print(vec);
   
//---

   Print("\n---> Vector ",vec," to Matrix");
   mat = matrix_utils.VectorToMatrix(vec); 
   Print(mat);

//--- 

   Print("---> Array to vector");
   double Arr[3] = {1,2,3};
   vec = matrix_utils.ArrayToVector(Arr);
   Print(vec);
   
//---

   Print("---> Vector to Array");
   double new_array[];
   matrix_utils.VectorToArray(vec,new_array);
   ArrayPrint(new_array);

//---
    
   Print("\n---> Col 1 ","Removed from Matrix");
   matrix_utils.RemoveCol(Matrix,1);
   Print("New Matrix\n",Matrix);

//--- 

   Matrix = matrix_utils.ReadCsv("NASDAQ_DATA.csv");
   
   Print("\nRemoving multiple columns");
   int cols[2] = {0,2};
   
   matrix_utils.RemoveMultCols(Matrix,cols);
   Print("Columns at 0,2 index removed New Matrix\n",Matrix);

//---

   Matrix = matrix_utils.ReadCsv("NASDAQ_DATA.csv");
   Print("Removing row 1 from matrix");
    
   matrix_utils.RemoveRow(Matrix,1);
   
   printf("Row %d Removed New Matrix[%d][%d]",1,Matrix.Rows(),Matrix.Cols());
   Print(Matrix); 

//--- 

   vector v= {0,1,2,3,4};
   Print("Vector ",v," remove index 3");
   matrix_utils.VectorRemoveIndex(v,3);
   Print(v);
 

//---
  
   Matrix = matrix_utils.ReadCsv("NASDAQ_DATA.csv");

   
   Print("---> Train / Test Split");
   matrix x_train, x_test;
   vector y_train, y_test;
   
   matrix_utils.TrainTestSplitMatrices(Matrix,x_train,y_train,x_test,y_test);
   Print("Xtrain\n",x_train,"\nx_test\n",x_test);
   Print("y_train\n",y_train,"\ny_test\n",y_test);

//--- 

   Print("\n---> X and Y split matrices");
   
   matrix x;
   vector y;
   
   matrix_utils.XandYSplitMatrices(Matrix,x,y);
   
   Print("independent vars\n",x);
   Print("Target variables\n",y);


//---
   
   Print("---> Design Matrix\n");
   matrix design = matrix_utils.DesignMatrix(x);
   Print(design);


//--- 

   matrix dataset = {
                     {12,28, 1},
                     {11,14, 1},
                     {7,8, 2},
                     {2,4, 3}
                    };
 
   vector y_vector = dataset.Col(2);
   Print("---> One Hot encoded matrix \n",matrix_utils.OneHotEncoding(y_vector));

//---

   Matrix = matrix_utils.ReadCsvEncode("weather dataset.csv");
   ArrayPrint(matrix_utils.csv_header);
   Print(Matrix);
    

   vector v_classes = {1,0,0,0,1,0,1,0};
   
   Print("classes ",matrix_utils.Classes(v_classes));
   
//---
   
   v = matrix_utils.Random(1.0,2.0,10);
   Print("v ",v);

//--- 
   
   vector v1 = {1,2,3}, v2 = {4,5,6};
   
   Print("Vector 1 & 2 ",matrix_utils.Append(v1,v2));
   
//---
   
   vector all = {1,2,3,4,5,6};
   
   matrix_utils.Copy(all,v,3);
   Print("copied vector ",v);
   

   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
    
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
   
  }
//+------------------------------------------------------------------+
