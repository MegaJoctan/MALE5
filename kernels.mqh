//+------------------------------------------------------------------+
//|                                                      kernels.mqh |
//|                                     Copyright 2023, Omega Joctan |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, Omega Joctan"
#property link      "https://www.mql5.com/en/users/omegajoctan"
//+------------------------------------------------------------------+
//|  Library containing machine learning kernels                     |
//+------------------------------------------------------------------+
#include <MALE5\linalg.mqh>

enum kernels
 {
   KERNEL_LINEAR,
   KERNEL_POLYNOMIAL,
   KERNEL_RADIAL_BASIS_FUNCTION_RBF, 
   KERNEL_SIGMOID, 
 };
 
 
class __kernels__
 {
  private:
   kernels chosen_kernel;
   CLinAlg linalg;
   
   double alpha;
   double beta;
   int  degree_polynomial;
   double sigma;
   
     
    //+------------------------------------------------------------------+
    //| The linear kernel is the simplest one. It represents the dot     |
    //|  product of the input vectors                                    |
    //+------------------------------------------------------------------+
    matrix LinearKernel(matrix &x1, matrix &x2) 
     {         
       return (x1.MatMul(x2.Transpose()));
     }
    //+------------------------------------------------------------------+
    //|  The polynomial kernel allows for the modeling of polynomial     |
    //|  relationships between data points                               |  
    //+------------------------------------------------------------------+ 
    matrix PolynomialKernel(matrix &x1, matrix &x2, const double lambda=1) 
     {
       return (MathPow(x1.MatMul(x2.Transpose()) + lambda, degree_polynomial));
     }
    //+------------------------------------------------------------------+
    //| Radial Basis Function (RBF) Kernel: The RBF kernel, also known   |
    //| as the Gaussian kernel, is one of the most commonly used kernels.|
    //|  It captures complex, non-linear relationships                   |  
    //+------------------------------------------------------------------+
    matrix RBFKernel(const matrix &x1, const matrix &x2) 
     { 
       matrix norm = linalg.norm(x1,x2);       
       return exp(-1* ((MathPow(norm, 2)) / (2*MathPow(sigma, 2))) );
     }
    //+------------------------------------------------------------------+
    //|   The sigmoid kernel is inspired by the sigmoid function         |
    //+------------------------------------------------------------------+
    matrix SigmoidKernel(matrix &x1, matrix &x2) 
     {
       return (tanh((alpha* x1.MatMul(x2.Transpose())) + beta));
     }   
     
   public:
   
    __kernels__::__kernels__(
                             kernels KERNEL,
                             double alpha_=0.1,
                             double beta_=0.1, 
                             int degree_polynomial_=2, 
                             double sigma_=0.1
                            )
                             :chosen_kernel(KERNEL),
                              alpha(alpha_), 
                              beta(beta_), 
                              degree_polynomial(degree_polynomial_), 
                              sigma(sigma_)
    {
         
    }
    
   __kernels__::~__kernels__(void)
    {
    
    }
   
//--- kernels in matrix form
    
   matrix KernelFunction(matrix &x1, matrix &x2)
    {
      matrix ret = {};
      
      switch(chosen_kernel)
        {
         case  KERNEL_LINEAR:
           ret = this.LinearKernel(x1, x2);     
           break;
         case  KERNEL_POLYNOMIAL:
           ret = this.PolynomialKernel(x1, x2);      
           break;
         case  KERNEL_RADIAL_BASIS_FUNCTION_RBF:
           ret = this.RBFKernel(x1, x2);
           break;
         case  KERNEL_SIGMOID:
           ret = this.SigmoidKernel(x1, x2);
           break;
        }
        
      return ret;
    }
 };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
