//+------------------------------------------------------------------+
//|                                                         BaseClustering.mqh |
//|                                     Copyright 2023, Omega Joctan |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, Omega Joctan"
#property link      "https://www.mql5.com/en/users/omegajoctan"

#include <MALE5\MatrixExtend.mqh>
#include <MALE5\linalg.mqh>
#include <MALE5\MqPlotLib\plots.mqh>
//+------------------------------------------------------------------+
//| defines                                                          |
//+------------------------------------------------------------------+
class BaseClustering
  {
public:
                     BaseClustering(void);
                    ~BaseClustering(void);
                    
                    static matrix Get(matrix &X, vector &index, int axis=0);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
matrix BaseClustering::Get(matrix &X, vector &index, int axis=0)
 {
   matrix ret_matrix = {};
   
   ulong row_col=0;
   bool isRows = true;
    switch(axis)
      {
       case  0:
         row_col = X.Rows();
         isRows = true;
         break;
       case 1:
         row_col = X.Cols();
         isRows = false;
         break;
       default:
         printf("%s Invalid axis %d ",__FUNCTION__,axis);
         return ret_matrix;
         break;
      }
//---
   
   ret_matrix.Resize(isRows?index.Size():X.Rows(), isRows?X.Cols(): index.Size());
   
   for (ulong i=0, count=0; i<row_col; i++)
     for (ulong j=0; j<index.Size(); j++)
        {
          if (isRows)
           {
             if (i==index[j])
              {
                if (isRows)
                  ret_matrix.Row(X.Row(i), count);
                else
                  ret_matrix.Col(X.Col(i), count);
                  
                count++;
              }
           }
        }
     
  return ret_matrix; 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+