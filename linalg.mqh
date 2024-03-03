//+------------------------------------------------------------------+
//|                                                       linalg.mqh |
//|                                     Copyright 2023, Omega Joctan |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, Omega Joctan"
#property link      "https://www.mql5.com/en/users/omegajoctan"

#include "MatrixExtend.mqh"
//+------------------------------------------------------------------+
//|    implementations of standard linear algebra algorithms         |
//+------------------------------------------------------------------+
class LinAlg
  {
public:
                     LinAlg(void);
                    ~LinAlg(void);
                    
                    template<typename T>
                    static matrix<T> dot(matrix<T> &A, matrix<T> &B);
                    
                    template<typename T>
                    static matrix<T> norm(const matrix<T> &A, const matrix<T> &B);
                    
                    template<typename T>
                    static double norm(const vector<T> &v1, const vector<T> &v2);
                    
                    template<typename T>
                    static matrix<T> outer(const matrix<T> &A, const matrix<T> &B);
                    
                    static bool svd(matrix &mat, matrix &U, matrix &V, vector &singular_value);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
LinAlg::LinAlg(void)
 {
 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
LinAlg::~LinAlg(void)
 {
 
 }
//+------------------------------------------------------------------+
//|	Dot product of two matrices | Flexible funciton - numpy like   |
//+------------------------------------------------------------------+
template<typename T>
matrix<T> LinAlg::dot(matrix<T> &A, matrix<T> &B)
 { 
   matrix Z={};
   
   if (A.Cols() == B.Rows()) //Valid Normal matrix multiplication
    {
      Z = A.MatMul(B);
      return Z;
    }
   else
     {
       //---Check for one dimensional matrices | Scalar
       
       if ((A.Rows()==1 && A.Cols()==1))
        {
          Z = B * A[0][0];
          return Z;
        }
        
       if (B.Rows()==1 && B.Cols()==1)
        {
          Z = B[0][0] * A;
          return Z;
        }
        
      //-- Element wise multiplication
      
       if (A.Rows()==B.Rows() && A.Cols()==B.Cols()) 
         {
            Z = A * B;
            return Z;
         }
     }
   
   return Z;
 }
//+------------------------------------------------------------------+
//| Matrix or vector<T> norm. | Finds the equlidean distance of the  |
//| two matrices                                                     |
//+------------------------------------------------------------------+
template<typename T>
matrix<T> LinAlg::norm(const matrix<T> &A, const matrix<T> &B)
 {
   matrix<T> ret = {};
   
    if (B.Cols() != A.Cols())
      {
         Print(__FUNCTION__," Dimensions Error");
         return ret;
      }
   
   if (A.Rows()==1 || B.Rows()==1)
    {
      matrix<T> A_temp = A, B_temp = B;      
      vector<T> A_vector, B_vector;
      
      A_vector.Swap(A_temp);
      B_vector.Swap(B_temp);
      
      ulong size = 0;
      if (A_vector.Size() >= B_vector.Size())
         {
            size = A_vector.Size();
            B_vector.Resize(size);
         }
      else
       {
         size = B_vector.Size();
         A_vector.Resize(size);
       }  
      
      ret.Resize(1,1);
      ret[0][0] = MathSqrt( MathPow(A_vector - B_vector, 2).Sum() ) ; 
      
      return (ret);
    }

   ulong size = A.Rows() > B.Rows() ? A.Rows() : B.Rows();
   vector<T> euc(size);
   
   for (ulong i=0; i<A.Rows(); i++)
      for (ulong j=0; j<B.Rows(); j++)
           euc[i] = MathSqrt( MathPow(A.Row(i) - B.Row(j), 2).Sum() );    
   
   euc.Swap(ret);
   return ret;
 }
//+------------------------------------------------------------------+
//|                Euclidean Distance of two vectors                 |
//+------------------------------------------------------------------+
template<typename T>
double LinAlg::norm(const vector<T> &v1, const vector<T> &v2)
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
template<typename T>
matrix<T> LinAlg::outer(const matrix<T> &A,const matrix<T> &B)
 {
    return A.Outer(B);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool LinAlg::svd(matrix &mat, matrix &U,matrix &V,vector &singular_value)
 {
   return mat.SVD(U,V,singular_value);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

