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
#include <MALE5\MatrixExtend.mqh>
#include <MALE5\metrics.mqh>

class CPolynomialRegression
  {
   private:
                        ulong  m_degree; //depends on independent vars
                         
                        matrix Betas;
                        vector Betas_v[];  //coefficients of the model stored in Array

                        
   public:
                        CPolynomialRegression(int degree=2);
                       ~CPolynomialRegression(void);
                        
                        void   BIC(ulong k, vector &bic, int &best_degree);      //Bayessian information Criterion
                        void   fit(matrix &x, vector &y); 
                        
                        double predict(vector &x);
                        vector predict(matrix &x);
                        
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CPolynomialRegression::CPolynomialRegression(int degree=2)
 :m_degree(degree)
 {
 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CPolynomialRegression::~CPolynomialRegression(void)
 {
   ArrayFree(Betas_v);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//|   BIC is the Function used to find which degree polynomial fits  |
//|   best to a the dataset, k is the number of degrees that the     |
//|   Algorithm will run to try to figure out the best model         |
//|  This is somehow the learning algorithm of polynomial regression |
//|                                                                  |
//+------------------------------------------------------------------+
void  CPolynomialRegression::BIC(ulong k, vector &bic, int &best_degree)
 {
   vector Pred;
   
   bic.Resize(k-2); 
   best_degree = 0;
   
    for (ulong i=2, counter = 0; i<k; i++)
      {         
         fit(i, Pred);         
         bic[counter] = ( n * log(metrics.rss(Pred)) ) + (i * log(n));  
         
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
void  CPolynomialRegression::fit(matrix &x, vector &y)
 {
    ulong order_size = degree+1;
    
    matrix PolyNomialsXMatrix(order_size,order_size);
    matrix PolynomialsYMatrix(order_size,1);
    
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
    
    #ifdef DEBUG_MODE 
     Print("Polynomials y vector \n",PolynomialsYMatrix);
    #endif 
    
//---
   
   PolyNomialsXMatrix.Resize(order_size, order_size);
   
   double power = 0;
   ZeroMemory(x_pow);
   
    for (ulong i=0,index = 0; i<PolyNomialsXMatrix.Rows(); i++)
       for (ulong j=0; j<PolyNomialsXMatrix.Cols(); j++, index++)
          {
             power = (double)i+j;
             if (power == 0) PolyNomialsXMatrix[i][j] = n;
             else          
                 {
                   x_pow = MathPow(x,power);
                   
                   PolyNomialsXMatrix[i][j] =  x_pow.Sum();
                 }
          }
    
//---

   #ifdef DEBUG_MODE
     Print("Polynomial x matrix\n",PolyNomialsXMatrix);
   #endif
   
//---
 
    PolyNomialsXMatrix = PolyNomialsXMatrix.Inv(); //find the inverse of the matrix
    
    Betas = PolyNomialsXMatrix.MatMul(PolynomialsYMatrix);
    Betas_v = MatrixExtend::MatrixToVector(Betas);
    
    
    #ifdef DEBUG_MODE 
      Print("Betas \n",Betas);
    #endif 
     
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CPolynomialRegression::predict(matrix &x)
 {
   ulong order_size = m_degree+1;
   vector preds(x.Rows());
   
    for (ulong i=0; i<(ulong)n; i++)
     {
      double sum = 0;
       for (ulong j=0; j<order_size; j++) 
         {
           if (j == 0) sum += Betas_v[j];
           else        sum += Betas_v[j] * MathPow(x[i],j);
         }
         
       preds[i] = sum;
     }
   return preds;
 }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
