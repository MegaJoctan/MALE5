//+------------------------------------------------------------------+
//|                                        Polynomial Regression.mqh |
//|                                    Copyright 2022, Omega Joctan. |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, Omega Joctan."
#property link      "https://www.mql5.com/en/users/omegajoctan"

//+---------------------------------------------------------------------------+
//|   Polynomial/Quadratic curve fitting regression model library             |
//|   Capable of handling as much degree as possible, Thanks to the matrices  |
//|   polynomial regression degrees depends on the independent variables      |
//|   hence one independent variable is a second degree, so on and so forth   |
//+---------------------------------------------------------------------------+

bool debug = false; //set to true to see the maths behind the library

class CPolynomialRegression
  {
   private:
                        ulong  m_degree; //depends on independent vars
                        int    n;   //number of samples in the dataset
                        vector x;
                        vector y;
                        matrix PolyNomialsXMatrix; //x matrix 
                        matrix PolynomialsYMatrix; //y matrix 
                        matrix Betas;
                        double Betas_A[];  //coefficients of the model stored in Array

                        void   Poly_model(vector &Predictions,ulong degree);
                        
   public:
                        CPolynomialRegression(vector& x_vector,vector &y_vector,int degree=2);
                       ~CPolynomialRegression(void);
                        
                        double RSS(vector &Pred);               //sum of squared residuals
                        void   BIC(ulong k, vector &bic,int &best_degree);      //Bayessian information Criterion
                        void   PolynomialRegressionfx(ulong degree, vector &Pred);
                        double r_squared(vector &y,vector &y_predicted); 
                        
                        void   matrixtoArray(matrix &mat, double &Array[]);
                        void   vectortoArray(vector &v, double &Arr[]);
                        //void   MinMaxScaler(vector &v);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CPolynomialRegression::CPolynomialRegression(vector& x_vector,vector &y_vector,int degree=2)
 {
   x.Copy(x_vector);
   y.Copy(y_vector);
   
   n = (int) y.Size(); 
   m_degree =  degree;
   
   if (y.Size() != x.Size()) Print(__FUNCTION__," Number of samples in the Y matrix doesnot, match number of samples in all the x columns");
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CPolynomialRegression::~CPolynomialRegression(void)
 {
   ArrayFree(Betas_A);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CPolynomialRegression::RSS(vector &Pred)
 {
  if (Pred.Size() != y.Size()) Print(__FUNCTION__," Predictions Array and Y matrix doesn't have the same size");
 
   double sum =0;
    for (int i=0; i<(int)y.Size(); i++)
      sum += MathPow(y[i] - Pred[i],2);
      
    return(sum);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//|   BIC is the Function used to find which degree polynomial fits  |
//|   best to a the dataset, k is the number of degrees that the     |
//|   Algorithm will run to try to figure out the best model         |
//|  This is somehow the learning algorithm of polynomial regression |
//+------------------------------------------------------------------+
void  CPolynomialRegression::BIC(ulong k, vector &bic,int &best_degree)
 {
   vector Pred;
   
   bic.Resize(k-2); 
   best_degree = 0;
   
    for (ulong i=2, counter = 0; i<k; i++)
      {         
         PolynomialRegressionfx(i,Pred);         
         bic[counter] = ( n * log(RSS(Pred)) ) + (i * log(n));  
         
         counter++;
      }
      
//--- 

   bool positive = false;
   for (ulong i=0; i<bic.Size(); i++)
      if (bic[i] > 0) { positive = true; break; }
   
  
   double low_bic = DBL_MAX;
   
   if (positive == true) 
    for (ulong i=0; i<bic.Size(); i++) 
     {
      if (bic[i] < low_bic && bic[i] > 0) low_bic = bic[i];
     }
   else  low_bic = bic.Min(); //bic[ best_degree = ArrayMinimum(bic) ];
      
   printf("Best Polynomial Degree(s) is = %d with BIC = %.5f",best_degree = best_degree+2,low_bic);

 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void  CPolynomialRegression::PolynomialRegressionfx(ulong degree,vector &Pred)
 {
    ulong order_size = degree+1;
    PolyNomialsXMatrix.Resize(order_size,order_size);
    
    PolynomialsYMatrix.Resize(order_size,1);
    
    vector c;
    vector x_pow;
    
    for (ulong i=0; i<PolynomialsYMatrix.Rows(); i++)
       for (ulong j=0; j<PolynomialsYMatrix.Cols(); j++) 
         {
            if (i+j == 0)  PolynomialsYMatrix[i][j] = y.Sum(); 
            else         
               {
                   x_pow = MathPow(x,i);       c = y*x_pow;
                   PolynomialsYMatrix[i][j] =  c.Sum(); 
               }
         }
    
    if (debug) Print("Polynomials y vector \n",PolynomialsYMatrix);
    
//---
   
   PolyNomialsXMatrix.Resize(order_size, order_size);
   
   double pow = 0;
   ZeroMemory(x_pow);
   //x_pow.Copy(x);
   
    for (ulong i=0,index = 0; i<PolyNomialsXMatrix.Rows(); i++)
       for (ulong j=0; j<PolyNomialsXMatrix.Cols(); j++, index++)
          {
             pow = (double)i+j;
             if (pow == 0) PolyNomialsXMatrix[i][j] = n;
             else          
                 {
                   x_pow = MathPow(x,pow);
                   
                   PolyNomialsXMatrix[i][j] =  x_pow.Sum();
                 }
          }
    
//---

   if (debug) Print("Polynomial x matrix\n",PolyNomialsXMatrix);
   
//---
 
    PolyNomialsXMatrix = PolyNomialsXMatrix.Inv(); //find the inverse of the matrix
    
    Betas = PolyNomialsXMatrix.MatMul(PolynomialsYMatrix);
    
    if (debug) Print("Betas \n",Betas);
     
    Poly_model(Pred,degree);
     
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CPolynomialRegression::Poly_model(vector &Predictions, ulong degree)
 {
   ulong order_size = degree+1;
   Predictions.Resize(n);
   
   matrixtoArray(Betas,Betas_A); 
   
    for (ulong i=0; i<(ulong)n; i++)
     {
      double sum = 0;
       for (ulong j=0; j<order_size; j++) 
         {
           if (j == 0) sum += Betas_A[j];
           else        sum += Betas_A[j] * MathPow(x[i],j);
         }
       Predictions[i] = sum;
     }
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CPolynomialRegression::r_squared(vector &y_,vector &y_predicted)
 {
   double error=0;
   double numerator =0, denominator=0; 
   
   double mean_y = y_.Mean();
   
//--- 
  
  if (y_predicted.Size()==0)
    Print("The Predicted values Array seems to have no values, Call the main Simple Linear Regression Funtion before any use of this function = ",__FUNCTION__);
  else
    {
      for (ulong i=0; i<y_.Size(); i++)
        {
          numerator += MathPow((y_[i]-y_predicted[i]),2);
          denominator += MathPow((y_[i]-mean_y),2);
        }
      error = 1 - (numerator/denominator);
    }
   return(error);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CPolynomialRegression::matrixtoArray(matrix &mat,double &Array[])
 {
   ArrayFree(Array);
   ArrayResize(Array,int(mat.Rows()*mat.Cols()));

   int index = 0; 
     for (ulong i=0; i<mat.Rows(); i++)
        for (ulong j=0; j<mat.Cols(); j++, index++)
           { 
             Array[index] = mat[i][j];
           }
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CPolynomialRegression::vectortoArray(vector &v, double &Arr[])
 {
   ArrayResize(Arr,(int)v.Size());
   
   for (int i=0; i<(int)v.Size(); i++)
     { Arr[i] = v[i];  }
     
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+