//+------------------------------------------------------------------+
//|                                                       linalg.mqh |
//|                                     Copyright 2023, Omega Joctan |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, Omega Joctan"
#property link      "https://www.mql5.com/en/users/omegajoctan"
//+------------------------------------------------------------------+
//|    implementations of standard linear algebra algorithms         |
//+------------------------------------------------------------------+
class CLinAlg
  {
public:
                     CLinAlg(void);
                    ~CLinAlg(void);
                    
                    matrix dot(matrix &A, matrix &B);
                    matrix norm(const matrix &A, const matrix &B);
                    matrix outer(const matrix &A, const matrix &B);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CLinAlg::CLinAlg(void)
 {
 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CLinAlg::~CLinAlg(void)
 {
 
 }
//+------------------------------------------------------------------+
//|	Dot product of two matrices.                                   |
//+------------------------------------------------------------------+
matrix CLinAlg::dot(matrix &A,matrix &B)
 {
   return A.MatMul(B);
 }
//+------------------------------------------------------------------+
//| Matrix or vector norm. | Finds the equlidean distance of the     |
//| two matrices                                                     |
//+------------------------------------------------------------------+
matrix CLinAlg::norm(const matrix &A, const matrix &B)
 {
   matrix ret = {};
   
    if (B.Cols() != A.Cols())
      {
         Print(__FUNCTION__," Dimensions Error");
         return ret;
      }
   
   if (A.Rows()==1 || B.Rows()==1)
    {
      matrix A_temp = A, B_temp = B;      
      vector A_vector, B_vector;
      
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
   vector euc(size);
   
   for (ulong i=0; i<A.Rows(); i++)
      for (ulong j=0; j<B.Rows(); j++)
           euc[i] = MathSqrt( MathPow(A.Row(i) - B.Row(j), 2).Sum() );    
   
   euc.Swap(ret);
   return ret;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
matrix CLinAlg::outer(const matrix &A,const matrix &B)
 {
    return A.Outer(B);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

