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
   
   double alpha;
   double beta;
   int  degree_polynomial;
   double sigma;
   
   //--- Helper function to convert vectors to matrix
   
   matrix VectorToMatrix(vector &v)
    {
      matrix ret_m;
      vector temp_v = v;
      
      temp_v.Swap(ret_m);
      
      return ret_m;
    }
    
    double MatrixToDBL(matrix &mat)
    {   
      if (mat.Rows()>1 || mat.Cols()>1)
       {
         Print(__FUNCTION__," Can't convert matrix to double as this is not a 1x1 matri");
         return 0;
       }
      return mat[0][0];
    }
    //+------------------------------------------------------------------+
    //|              finding Euclidean Distance                          |
    //+------------------------------------------------------------------+
   double Euclidean_distance(const vector &v1, const vector &v2)
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
    //| The linear kernel is the simplest one. It represents the dot     |
    //|  product of the input vectors                                    |
    //+------------------------------------------------------------------+
    double LinearKernel(matrix &x1, matrix &x2) 
     {         
       return MatrixToDBL(x1.MatMul(x2.Transpose()));
     }
    
    double LinearKernel(vector &x1, vector &x2) 
     {         
       return MatrixToDBL(VectorToMatrix(x1).MatMul(VectorToMatrix(x2).Transpose()));
     }
    //+------------------------------------------------------------------+
    //|  The polynomial kernel allows for the modeling of polynomial     |
    //|  relationships between data points                               |  
    //+------------------------------------------------------------------+ 
    double PolynomialKernel(matrix &x1, matrix &x2, const double lambda=1) 
     {
       return MatrixToDBL(MathPow(x1.MatMul(x2.Transpose()) + lambda, degree_polynomial));
     }
    
    double PolynomialKernel(vector &x1, vector &x2, const double lambda=1) 
     {
       return MatrixToDBL(MathPow(VectorToMatrix(x1).MatMul(VectorToMatrix(x2).Transpose()) + lambda, degree_polynomial));
     }
    //+------------------------------------------------------------------+
    //| Radial Basis Function (RBF) Kernel: The RBF kernel, also known   |
    //| as the Gaussian kernel, is one of the most commonly used kernels.|
    //|  It captures complex, non-linear relationships                   |  
    //+------------------------------------------------------------------+
    double RBFKernel(matrix &x1, matrix &x2) 
     { 
       vector v1, v2; x1.Swap(v1); x2.Swap(v2);
        return exp(-1* ((MathPow(Euclidean_distance(v1, v2), 2)) / (2*MathPow(sigma, 2))) );
     }
     
    double RBFKernel(vector &x1, vector &x2) 
     { 
        return exp(-1* ((MathPow(Euclidean_distance(x1, x2), 2)) / (2*MathPow(sigma, 2))) );
     }
    //+------------------------------------------------------------------+
    //|   The sigmoid kernel is inspired by the sigmoid function         |
    //+------------------------------------------------------------------+
    double SigmoidKernel(matrix &x1, matrix &x2) 
     {
       return MatrixToDBL(tanh((alpha* x1.MatMul(x2.Transpose())) + beta));
     }   
         
    double SigmoidKernel(vector &x1, vector &x2) 
     {
       return MatrixToDBL(tanh((alpha* VectorToMatrix(x1).MatMul(VectorToMatrix(x2).Transpose())) + beta));
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
    
   double KernelFunction(matrix &x1, matrix &x2)
    {
      double ret = 0;
      
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

//--- kernels in vector form

   double KernelFunction(vector &x1, vector &x2)
    {
      double ret = 0;
      
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
