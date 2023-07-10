//+------------------------------------------------------------------+
//|                                             cross_validation.mqh |
//|                                    Copyright 2022, Fxalgebra.com |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, Fxalgebra.com"
#property link      "https://www.mql5.com/en/users/omegajoctan"
//+------------------------------------------------------------------+
//| defines                                                          |
//+------------------------------------------------------------------+
#include <MALE5\Tensors.mqh>

class CCrossValidation_kfold
  {
CTensors  *folds_tensor;
   void XandYSplitMatrices(const matrix &matrix_,matrix &xmatrix,vector &y_vector,int y_column=-1);
   void RemoveCol(matrix &mat, ulong col);
   uint k_folds;
   
public:
                     CCrossValidation_kfold(matrix &data_matrix, uint k_folds=5);
                    ~CCrossValidation_kfold(void);
                    
                    matrix fold(uint index);
                    uint fold_size;
                    
                    matrix fold_x(uint index);
                    vector fold_y(uint index);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CCrossValidation_kfold::CCrossValidation_kfold(matrix &data_matrix, uint k_folds_=5)
 {
   this.k_folds = k_folds_;
   
   folds_tensor = new CTensors(k_folds);
   
   ulong rows = data_matrix.Rows();
   fold_size = (int)MathCeil(rows/k_folds);
   
   matrix temp_tensor(fold_size, data_matrix.Cols());
   
   int start=0;
   for (ulong i=0; i<k_folds; i++)  
     {
       for (ulong j=start, count=0; j<fold_size+start; j++, count++)
          {
            temp_tensor.Row(data_matrix.Row(j), count);
          }
          
       folds_tensor.TensorAdd(temp_tensor, i); //Obtained size=k data matrix
       
       
       start += (int)fold_size;
     }
   
   
   //#ifdef DEBUG_MODE
   //   Print("total ",rows," fold_size ",fold_size," k_folds ",k_folds);
   //   folds_tensor.TensorPrint();
   //#endif 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CCrossValidation_kfold::~CCrossValidation_kfold(void)
 {
   delete(folds_tensor);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
matrix CCrossValidation_kfold::fold(uint index)
 {
   if (index+1 > this.k_folds) 
     {
       matrix ret={};
       Print("k-fold index out of range");
       return (ret);
     }
     
   return folds_tensor.Tensor(index);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
matrix CCrossValidation_kfold::fold_x(uint index)
 {     
   matrix x; vector y;
   matrix fold_matrix = this.fold(index);
   
   this.XandYSplitMatrices(fold_matrix, x, y);
   
   return (x);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CCrossValidation_kfold::fold_y(uint index)
 {   
   matrix x; vector y;
   matrix fold_matrix = this.fold(index);
   
   this.XandYSplitMatrices(fold_matrix, x, y);
   
   return (y);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CCrossValidation_kfold::XandYSplitMatrices(const matrix &matrix_,matrix &xmatrix,vector &y_vector,int y_column=-1)
 {
   y_column = int( y_column==-1 ? matrix_.Cols()-1 : y_column);

   y_vector = matrix_.Col(y_column);
   xmatrix.Copy(matrix_);

   RemoveCol(xmatrix, y_column); //Remove the y column
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CCrossValidation_kfold::RemoveCol(matrix &mat, ulong col)
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
