//+------------------------------------------------------------------+
//|                                       Hierachical Clustering.mqh |
//|                                     Copyright 2023, Omega Joctan |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, Omega Joctan"
#property link      "https://www.mql5.com/en/users/omegajoctan"

#include <MALE5\Tensors.mqh>
#include <MALE5\matrix_utils.mqh>
//+------------------------------------------------------------------+
//| defines                                                          |
//+------------------------------------------------------------------+
class CHierachicalClustering
  {
CTensorsVectors *tensors_v;
CMatrixutils     matrix_utils;

private:
   matrix   EuclideanMatrix;
   matrix   DistanceMatrix;

   ulong    m_rows;
   ulong    m_cols;
   
   vector   Min(matrix &mat);
   uint     HIERACHIES_DIM;
   
public:
                     CHierachicalClustering(matrix &matrix_);
                    ~CHierachicalClustering(void);
                    
                    double Euclidean_distance(vector &v1, vector &v2);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CHierachicalClustering::CHierachicalClustering(matrix &matrix_)
 {
   m_rows = matrix_.Rows();
   m_cols = matrix_.Cols();
   
   if (m_rows <= 1)
      {
         Print("Failed to Initialize Hierachical clustering | The number of rows in a given matrix needs to be more than one for this to work");
         return;
      }
   
//---
   HIERACHIES_DIM = uint(m_rows-1);
   tensors_v = new CTensorsVectors(HIERACHIES_DIM);
//---
   
   vector cluster = {}, temp_cluster;
   vector loc(2);
   
  //HIERACHIES_DIM =1; //Remove this
   EuclideanMatrix.Resize(m_rows, m_rows);
   EuclideanMatrix.Fill(0);
   
   for (ulong i=0; i<m_rows; i++)   
      for (ulong j=0; j<m_rows; j++)
            EuclideanMatrix[i][j] = Euclidean_distance(matrix_.Row(i),matrix_.Row(j));

    vector diag(m_rows);
    diag.Fill(DBL_MAX);
    EuclideanMatrix.Diag(diag);
    
//---

   for (uint d=0; d<HIERACHIES_DIM; d++) 
    {       
       loc = Min(EuclideanMatrix);
       
       #ifdef DEBUG_MODE      
         Print("Euclidean Matrix[",EuclideanMatrix.Rows(),"x",EuclideanMatrix.Cols(),"]\n",EuclideanMatrix,"\nMinDistance ",EuclideanMatrix.Min()," Located ",loc);
       #endif 
      
       temp_cluster = loc;
       cluster = matrix_utils.Append(cluster, temp_cluster);
       tensors_v.TensorAdd(cluster, d);
       
       matrix_utils.RemoveRow(EuclideanMatrix, (int)loc[0]);
       matrix_utils.RemoveCol(EuclideanMatrix, int(loc[0]));
       
       //matrix_utils.RemoveRow(EuclideanMatrix, (int)loc[1]);
       //matrix_utils.RemoveCol(EuclideanMatrix, (int)loc[1]);
    }
   
   Print("HIERACHICAL CLUSTERS");
   tensors_v.TensorPrint();
   
//---
    
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CHierachicalClustering::~CHierachicalClustering(void)
 {
   delete (tensors_v);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

double CHierachicalClustering:: Euclidean_distance(vector &v1, vector &v2)
  {
   double dist = 0;
   ulong size = 0;

//---
   
   if (v1.Size() > v2.Size())
       v2.Resize(v1.Size());
   else 
       v1.Resize(v2.Size());

//---
   
   double c = 0;
   for(ulong i=0; i<v1.Size(); i++)
      c += MathPow(v1[i] - v2[i], 2);

   dist = MathSqrt(c);

   return(dist);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CHierachicalClustering:: Min(matrix &mat)
 {
   vector dir(2);
   
   ulong loc = mat.ArgMin();
   dir[0] = int(loc/mat.Rows()); dir[1]=int(loc % mat.Cols());
   
   return (dir);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
