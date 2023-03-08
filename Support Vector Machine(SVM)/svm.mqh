//+------------------------------------------------------------------+
//|                                                          svm.mqh |
//|                                    Copyright 2022, Fxalgebra.com |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, Omega Joctan"
#property link      "https://www.mql5.com/en/users/omegajoctan"

#include <MALE5\preprocessing.mqh>

enum kernels
 {
   KERNEL_POLYNOMIAL,
   KERNEL_GAUSSIAN, 
   KERNEL_RADIAL_BASIS_FUNCTION_RBF, 
   KERNEL_LAPLACE_RBF, 
   KERNEL_SIGMOID, 
   KERNEL_ANOVE_RBF
 };

//+------------------------------------------------------------------+
//|  THE SUPPORT VECTOR MACHINE CLASS                                |
//+------------------------------------------------------------------+
class Csvm
  {
   private:
      CPreprocessing    *normalize_x;
      
      matrix            W; 
      ulong             rows;
      ulong             DIM;
      matrix            XMATRIX;
      vector            YVECTOR;
      
                        matrix hyperplane(matrix &weights, matrix &x, double b);
                        int Sign(int var);
                        
                        matrix KernelFunction();
                        double Euclidean_distance(const vector &v1, const vector &v2);
                        
   public:
                        Csvm(matrix &xmatrix, vector &yvector, kernels KERNEL=KERNEL_RADIAL_BASIS_FUNCTION_RBF);
                       ~Csvm(void);
                    
  };

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

Csvm::Csvm(matrix &xmatrix, vector &yvector, kernels KERNEL=KERNEL_RADIAL_BASIS_FUNCTION_RBF)
 {
   rows = xmatrix.Rows();
   DIM = xmatrix.Cols();
   
   if (xmatrix.Rows() != yvector.Size())
      {
         Print("Support vector machine Failed | FATAL | x_matrix rows not same as yvector size");
         return;
      }
   
   W.Resize(1, DIM);
   
   XMATRIX = xmatrix;
   YVECTOR = yvector;
   
   normalize_x = new CPreprocessing(XMATRIX, NORM_STANDARDIZATION);

 }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

Csvm::~Csvm(void)
 {
 
 }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

matrix Csvm::hyperplane(matrix &weights, matrix &x, double b)
 {
   weights = weights.Transpose();
   
   matrix c = (weights * x)  - b;
   
   if ( c.Sum() != 0 )
    {
       Print("Fatal | Can't calculate Hyperplane");
       return (c);
    }
   return (c);    
 }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int Csvm::Sign(int var)
 {
      if (var == 0)
         return 0;
      else if (var < 0)
          return -1; 
      else
          return +1;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
matrix Csvm::KernelFunction()
 {
   matrix KERNEL_MATRIX = {};
   
   double gamma = 0.01;
   
   vector k(2);
   
   for (ulong i=0; i<DIM; i++)
      k[i] = exp(-gamma * Euclidean_distance(XMATRIX.Col(i), YVECTOR));
   
   return (KERNEL_MATRIX); 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double Csvm:: Euclidean_distance(const vector &v1, const vector &v2)
  {
   double dist = 0;

   if(v1.Size() != v2.Size())
      Print(__FUNCTION__, " v1 and v2 not matching in size");
   else
     {
      double c = 0;
      for(ulong i=0; i<v1.Size(); i++)
         c += MathPow(v1[i] - v2[i], 2);

      dist = MathSqrt(c);
     }

   return(dist);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
