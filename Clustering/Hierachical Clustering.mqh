//+------------------------------------------------------------------+
//|                                       Hierachical Clustering.mqh |
//|                                     Copyright 2023, Omega Joctan |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, Omega Joctan"
#property link      "https://www.mql5.com/en/users/omegajoctan"

#include <MALE5\matrix_utils.mqh>
#include <MALE5\linalg.mqh>
//+------------------------------------------------------------------+
//| defines                                                          |
//+------------------------------------------------------------------+

enum linkage_enum 
 {
   LINKAGE_SINGLE,
   LINKAGE_COMPLETE,
   LINKAGE_AVG
 };
 
class CAgglomerativeClustering
  {
CMatrixutils matrix_utils;
CLinAlg linalg;

protected:

   linkage_enum linkage;
   vector       labels;
   ulong        n_clusters;
   
   double       compute_linkage(vector &cluster1, vector &cluster2, matrix &distances);
   matrix       ix_(const vector &v1, const vector &v2);
   
public:
                     CAgglomerativeClustering(uint clusters=2, linkage_enum linkage_type=LINKAGE_SINGLE);
                    ~CAgglomerativeClustering(void);
                    
                    void fit(matrix &x);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CAgglomerativeClustering::CAgglomerativeClustering(uint clusters=2, linkage_enum linkage_type=LINKAGE_SINGLE)
 :linkage(linkage_type),
  n_clusters(clusters)
 {
 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CAgglomerativeClustering::~CAgglomerativeClustering(void)
 {
 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CAgglomerativeClustering::compute_linkage(vector &cluster1,vector &cluster2, matrix &distances)
 {
  matrix indices = ix_(cluster1, cluster2);
  vector vec(indices.Rows());
  
  Print("np.ix_ ",indices,"\ndistances:\n",distances);
  
  for (uint i=0; i<vec.Size(); i++)
    {
      printf("index[%dx%d]",(int)indices[i][0], (int)indices[i][1]);
     
      vec[i] = distances[(int)indices[i][0], (int)indices[i][1]];
      Print("distance : ",vec[i]);
    }
    
  double ret = 0;    
  
   switch(linkage)
     {
      case  LINKAGE_SINGLE:
        ret = vec.Min();
        break;
        
      case LINKAGE_COMPLETE:
        ret = vec.Max();
        break;
        
      case LINKAGE_AVG:
        ret = vec.Mean();
        break;
     } 
     
   return ret;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
matrix CAgglomerativeClustering::ix_(const vector &v1, const vector &v2) 
 {
    int size1 = (int)v1.Size();
    int size2 = (int)v2.Size();

    // Ensure that the result array has enough space
    matrix result(size1 * size2, 2);

    // Iterate through combinations and populate the result array
    int index = 0;
    for (int i = 0; i < size1; i++) {
        for (int j = 0; j < size2; j++) {
            result[index][0] = (int)v1[i];
            result[index][1] = (int)v2[j];
            index++;
        }
    }
 return result;
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CAgglomerativeClustering::fit(matrix &x)
 {
  ulong n_samples = x.Rows();

  // Initialize clusters with each data point as a single cluster
  
  matrix clusters(n_samples, 1);
  for (uint i=0; i<n_samples; i++)
    clusters.Row(clusters.Row(i).Fill(i), i);

  // Compute pairwise distances between clusters
  
  matrix distances = matrix_utils.Zeros(n_samples, n_samples);
  
  for (uint i=0; i<n_samples; i++)
      for (uint j=i+1; j<n_samples; j++)
        {
          distances[i, j] = linalg.norm(x.Row(i), x.Row(j));
          distances[j, i] = distances[i, j];
        }
        
  // Main loop: merge clusters until the desired number of clusters is reached
  while (clusters.Rows() > n_clusters)
   {
      Print("clusters\n",clusters);
      
      // Find the indices of the two closest clusters
      double min_distance = DBL_MAX;
      vector merge_indices = {0, 0};

      for (uint i=0; i<clusters.Rows(); i++)
          for (uint j=i + 1; j<clusters.Rows(); j++)
            {
              double distance = this.compute_linkage(clusters.Row(i), clusters.Row(j), distances);
              Print("compute linkage res: ",distance);
              
              if (distance < min_distance)
                {
                  min_distance = distance;
                  merge_indices[0] = i; merge_indices[1] = j;
                  
                  Print("merge indices: ",merge_indices);
                  
                  Print("left");
               }
            }
   }
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
