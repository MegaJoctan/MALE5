//+------------------------------------------------------------------+
//|                                            Linear Regression.mqh |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, Fxalgebra.com"
#property link      "https://www.mql5.com/en/users/omegajoctan" 

//+------------------------------------------------------------------+

#include <MALE5\metrics.mqh>
#include <MALE5\matrix_utils.mqh>
#include <MALE5\preprocessing.mqh>

//+------------------------------------------------------------------+
class CLinearRegression
  {
   private:
   
   CMetrics metrics;
   CMatrixutils matrix_utils;
   CPreprocessing pre_processing;
  
   protected:  
                        matrix XMatrix, YMatrix;
                        vector Y_vector;
                        ulong  m_rows, m_cols;
                        
                        double alpha;
                        uint   iterations;
   
   private:
                        double dx_wrt_bo();
                        vector dx_wrt_b1();
    
   public:
                        matrix Betas;   //Coefficients matrix
                        vector Betas_v; //Coefficients vector
                        
                        CLinearRegression(matrix &Matrix_); //Least squares estimator
                        CLinearRegression(matrix<double> &Matrix_, double Lr, uint iters = 1000); //Lr by Gradient descent
                        CLinearRegression(matrix &Matrix_, vector &coeff_vector);
                        
                       ~CLinearRegression(void);
                        
                        double LRModelPred(const vector &x); 
                        vector LRModelPred(matrix &matrix_, double &accuracy);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CLinearRegression::CLinearRegression(matrix &Matrix_)
 {      
    m_rows = Matrix_.Rows(); 
    
    matrix_utils.XandYSplitMatrices(Matrix_,XMatrix,Y_vector);
    m_cols = XMatrix.Cols();
    
    YMatrix =  matrix_utils.VectorToMatrix(Y_vector);
    
//---

   matrix design = matrix_utils.DesignMatrix(XMatrix);
   
//--- XTX
    
    matrix XT = design.Transpose();
    
    matrix XTX = XT.MatMul(design);
    
    //if (IS_DEBUG_MODE) Print("XTX\n",XTX);
    
//--- Inverse XTX

    matrix InverseXTX = XTX.Inv();
    
    //if (IS_DEBUG_MODE) Print("INverse XTX\n",InverseXTX);

//--- Finding XTY
   
   matrix XTY = XT.MatMul(YMatrix);
   
   //if (IS_DEBUG_MODE) Print("XTY\n",XTY);

//--- Coefficients
   
   Betas = InverseXTX.MatMul(XTY);
   //pre_processing.ReverseMinMaxScaler(Betas);
   
   Betas_v = matrix_utils.MatrixToVector(Betas);
   
   #ifdef DEBUG_MODE 
        Print("Betas\n",Betas);
   #endif 
   
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CLinearRegression::CLinearRegression(matrix<double> &Matrix_,double Lr,uint iters=1000)
 {
    pre_processing.MinMaxScaler(Matrix_);      
    
    m_rows = Matrix_.Rows();
    m_cols = Matrix_.Cols();
    
    matrix_utils.XandYSplitMatrices(Matrix_,XMatrix,Y_vector);
    m_cols = XMatrix.Cols();
  
    YMatrix =  matrix_utils.VectorToMatrix(Y_vector);
    
//---

    alpha = Lr;
    iterations = iters;
    
    //Betas.Resize(1,m_cols);
    
    Betas_v.Resize(m_cols+1);

//---
     #ifdef DEBUG_MODE  
        Print("\nTraining a Linear Regression Model with Gradient Descent\n");
     #endif 
//---
     
     for (ulong i=0; i<iterations; i++)
       {
       
         if (i==0) Betas_v.Fill(0);

//---

         double bo = dx_wrt_bo();
         
         Betas_v[0] = Betas_v[0] - (alpha * bo);
         //printf("----> dx_wrt_bo | Intercept = %.8f | Real Intercept = %.8f",bo,Betas_v[0]);
         
         vector dx = dx_wrt_b1(); 

//---

          for (ulong j=0; j<dx.Size(); j++)
            {
               //Print("out at iterations Betas _v ",Betas_v);
                
                  Betas_v[j+1] = Betas_v[j+1] - (alpha * dx[j]);
                  
                  //printf("k %d | ----> dx_wrt_b%d | Slope = %.8f | Real Slope = %.8f",j,j,dx[j],Betas_v[j+1]); 
            }
         
//---

           #ifdef DEBUG_MODE  
               Betas = matrix_utils.VectorToMatrix(Betas_v);
               double acc =0;
               //Print("Betas ",Betas);
               LRModelPred(Matrix_,acc);
                
               Print("[ ",i," ] Accuracy = ",NormalizeDouble(acc*100,2),"% | COST ---> WRT Intercept | ",NormalizeDouble(bo,5)," | WRT Coeff ",dx);
           #endif  
           
       } 
//---
    Betas = matrix_utils.VectorToMatrix(Betas_v);
//---

    #ifdef DEBUG_MODE 
     Print("Coefficients ",Betas_v);
    #endif 
    
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CLinearRegression::CLinearRegression(matrix &Matrix_, vector &coeff_vector)
 {
   
   Betas_v = coeff_vector;
   Betas = matrix_utils.VectorToMatrix(Betas_v);
   
   m_rows = Matrix_.Rows(); 
    
   matrix_utils.XandYSplitMatrices(Matrix_,XMatrix,Y_vector);
   m_cols = XMatrix.Cols();
    
   YMatrix =  matrix_utils.VectorToMatrix(Y_vector);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CLinearRegression::~CLinearRegression(void)
 {
   ZeroMemory(XMatrix);
   ZeroMemory(YMatrix);
   ZeroMemory(Y_vector);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CLinearRegression::dx_wrt_bo(void)
 {    
   double mx=0, sum=0;
   for (ulong i=0; i<Y_vector.Size(); i++)
      {          
          mx = LRModelPred(XMatrix.Row(i));
          
          sum += (Y_vector[i] - mx);  
      }  
   
   return(-2*sum);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CLinearRegression::dx_wrt_b1(void)
 { 
   vector dx_vector(Betas_v.Size()-1);
   //Print("dx_vector.Size() = ",dx_vector.Size());
   
    double mx=0, sum=0;
   
    for (ulong b=0; b<dx_vector.Size(); b++)  
     {
       ZeroMemory(sum);
       
       for (ulong i=0; i<Y_vector.Size(); i++)
         {             
             //Print("<<<    >>> intercept = ",mx," Betas_v ",Betas_v,"\n");
             
            mx = LRModelPred(XMatrix.Row(i));            

//---

            sum += (Y_vector[i] - mx) * XMatrix[i][b];  
            //PrintFormat("%d xMatrix %.5f",i,XMatrix[i][b]); 
          
            dx_vector[b] = -2*sum;  
        }
    }
      
    return dx_vector;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CLinearRegression::LRModelPred(const vector &x)
 {
   double pred =0; 
   
   double intercept = Betas_v[0];
   
   if (Betas_v.Size() == 0)
      {
         Print(__FUNCTION__,"Err, No coefficients available for LR model\nTrain the model before attempting to use it");
         return(0);
      }
   
    else
      { 
        if (x.Size() != Betas_v.Size()-1)
          Print(__FUNCTION__,"Err, X vars not same size as their coefficients vector ");
        else
          {
            for (ulong i=1; i<Betas_v.Size(); i++) 
               pred += x[i-1] * Betas_v[i];  
               
            pred += intercept; 
          }
      }
      
    return pred;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CLinearRegression::LRModelPred(matrix &matrix_, double &accuracy)
 {
   vector pred(matrix_.Rows()), actual = matrix_.Col(matrix_.Cols()-1);
   
   matrix temp_matrix = matrix_;
   
   matrix_utils.MatrixRemoveCol(temp_matrix,temp_matrix.Cols()-1);
   vector x_vec;
   
   
    for (ulong i=0; i<temp_matrix.Rows(); i++)
      {
         x_vec = temp_matrix.Row(i);
         
         pred[i] = NormalizeDouble(LRModelPred(x_vec),2);
         //printf("Actual %.5f pred %.5f ",actual[i],pred[i]);
      }
   
   accuracy = NormalizeDouble(metrics.r_squared(actual,pred),4);
   
   #ifdef DEBUG_MODE
      printf("R squared %f Adjusted R %f",metrics.r_squared(actual,pred),metrics.adjusted_r(actual,pred));
   #endif 
   
   return pred;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
